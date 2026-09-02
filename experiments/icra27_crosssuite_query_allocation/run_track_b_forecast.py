#!/usr/bin/env python3
"""Run the frozen B3 demonstration-reference forecast audit after Track A."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = Path("/home/wjq/research-assets/datasets/HuggingFaceVLA_libero")
ACT_ROOT = Path("/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final")
SMOL_CHECKPOINT = Path("/home/wjq/checkpoints/HuggingFaceVLA_smolvla_libero")
TASKS = {
    "libero_object:task3": "libero_object_task3",
    "libero_spatial:task0": "libero_spatial_task0",
    "libero_goal:task2": "libero_goal_task2",
    "libero_10:task3": "libero_10_task3",
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def seed_for(policy: str, task: str, episode: int, frame: int) -> int:
    key = f"track-b-b3|{policy}|{task}|episode={episode}|frame={frame}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def assert_track_a_finished() -> None:
    manifest = json.loads((ROOT / "track_a_manifest.json").read_text())
    for pid_path in (ROOT / "track_a/pids").glob("*.pid"):
        try:
            os.kill(int(pid_path.read_text().strip()), 0)
        except (ProcessLookupError, ValueError):
            continue
        raise RuntimeError(f"Track A still owns GPUs: {pid_path}")
    missing = [
        cell["cell_id"] for cell in manifest["cells"]
        if not (ROOT / "track_a/markers" / f"{cell['cell_id']}.complete").is_file()
    ]
    failures = list((ROOT / "track_a/markers").glob("*.technical_failed"))
    if missing or failures:
        raise RuntimeError(f"Track A not technically complete: missing={len(missing)}, failed={len(failures)}")


class Runtime:
    def __init__(self, gpu: str):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        self.torch = torch
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        self.PreTrainedConfig = PreTrainedConfig
        self.LeRobotDataset = LeRobotDataset
        self.make_policy = make_policy
        self.make_pre_post_processors = make_pre_post_processors

    def load(self, checkpoint: Path, dataset: Any, expected_type: str):
        cfg = self.PreTrainedConfig.from_pretrained(checkpoint)
        cfg.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        cfg.pretrained_path = checkpoint
        if getattr(cfg, "type", None) != expected_type:
            raise RuntimeError(f"checkpoint type drift: {getattr(cfg, 'type', None)}")
        if expected_type == "act":
            cfg.pretrained_backbone_weights = None
        policy = self.make_policy(cfg=cfg, ds_meta=dataset.meta)
        policy.eval()
        preprocessor, _ = self.make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(cfg.device)}},
        )
        return cfg, policy, preprocessor

    def release(self) -> None:
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def run_task(runtime: Runtime, policy_name: str, task: str, tag: str, episodes: list[int]) -> None:
    slug = f"{policy_name.lower()}-{tag}"
    result_path = ROOT / "track_b/forecast/results" / f"{slug}.npz"
    metadata_path = ROOT / "track_b/forecast/results" / f"{slug}.json"
    marker_path = ROOT / "track_b/forecast/markers" / f"{slug}.complete"
    if result_path.is_file() and metadata_path.is_file() and marker_path.is_file():
        return
    checkpoint = ACT_ROOT / tag / "checkpoints/100000/pretrained_model" if policy_name == "ACT" else SMOL_CHECKPOINT
    normalizer_files = sorted(checkpoint.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
    if len(normalizer_files) != 1:
        raise RuntimeError(f"expected one frozen action normalizer for {checkpoint}")
    with safe_open(normalizer_files[0], framework="numpy") as handle:
        action_mean = np.asarray(handle.get_tensor("action.mean"), dtype=np.float64)
        action_std = np.asarray(handle.get_tensor("action.std"), dtype=np.float64)
    if action_mean.shape != (7,) or action_std.shape != (7,) or np.any(action_std <= 0):
        raise RuntimeError(f"invalid frozen action normalization for {checkpoint}")
    dataset = runtime.LeRobotDataset(
        "HuggingFaceVLA/libero", root=DATASET_ROOT, episodes=episodes,
        delta_timestamps={"action": [k / 10 for k in range(33)]}, download_videos=False,
    )
    expected_type = "act" if policy_name == "ACT" else "smolvla"
    cfg, policy, preprocessor = runtime.load(checkpoint, dataset, expected_type)
    frame_indices = np.asarray(dataset.hf_dataset["frame_index"], dtype=np.int64)
    episode_indices = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    anchors = [i for i, frame in enumerate(frame_indices) if frame % 10 == 0]
    squared_errors: list[np.ndarray] = []
    sign_disagreements: list[np.ndarray] = []
    kept_episode: list[int] = []
    kept_frame: list[int] = []
    try:
        for index in anchors:
            sample = dataset[index]
            pad = np.asarray(sample["action_is_pad"], dtype=bool)
            if pad.shape != (33,) or pad.any():
                continue
            episode, frame = int(episode_indices[index]), int(frame_indices[index])
            batch = {
                key: sample[key].unsqueeze(0)
                for key in ("observation.images.image", "observation.images.image2", "observation.state")
            }
            if policy_name == "SmolVLA":
                batch["task"] = [sample["task"]]
            batch = preprocessor(batch)
            random.seed(seed_for(policy_name, task, episode, frame))
            np.random.seed(seed_for(policy_name, task, episode, frame) & 0xFFFFFFFF)
            runtime.torch.manual_seed(seed_for(policy_name, task, episode, frame))
            if runtime.torch.cuda.is_available():
                runtime.torch.cuda.manual_seed_all(seed_for(policy_name, task, episode, frame))
            policy.reset()
            with runtime.torch.inference_mode():
                predicted = policy.predict_action_chunk(batch)[0, :33].detach().cpu().numpy().astype(np.float64)
            target = sample["action"].detach().cpu().numpy().astype(np.float64)
            if predicted.shape != (33, 7) or target.shape != (33, 7):
                raise RuntimeError(f"forecast shape mismatch for {slug}")
            normalized_target = (target - action_mean) / action_std
            squared_errors.append(np.square(predicted - normalized_target))
            sign_disagreements.append(np.not_equal(np.sign(predicted[:, 6]), np.sign(normalized_target[:, 6])))
            kept_episode.append(episode); kept_frame.append(frame)
    finally:
        policy = preprocessor = cfg = dataset = None
        runtime.release()
    if not squared_errors:
        raise RuntimeError(f"no valid forecast anchors for {slug}")
    atomic_npz(
        result_path,
        squared_error=np.stack(squared_errors),
        gripper_sign_disagreement=np.stack(sign_disagreements),
        episode_index=np.asarray(kept_episode, dtype=np.int64),
        frame_index=np.asarray(kept_frame, dtype=np.int64),
    )
    atomic_json(metadata_path, {
        "status": "COMPLETE", "policy": policy_name, "task": task, "tag": tag,
        "checkpoint": str(checkpoint), "episodes": episodes, "anchor_stride_frames": 10,
        "offsets": list(range(33)), "anchor_count": len(kept_episode),
        "dataset": "HuggingFaceVLA/libero", "dataset_revision": "86958911c0f959db2bbbdb107eb3e17c5f9c798e",
        "provenance_label": "training-demonstration reference analysis; not held-out",
        "seed_rule": "SHA256 first 8 bytes of track-b-b3|policy|task|episode|frame, masked to 63 bits",
        "success_outcomes_loaded": False,
    })
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text("COMPLETE\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    assert_track_a_finished()
    addendum = json.loads((ROOT / "track_b_analysis_addendum.json").read_text())
    if addendum.get("status") != "FROZEN_BEFORE_TRACK_B_PREDICTION_INTERPRETATION":
        raise RuntimeError("analysis addendum is not frozen")
    runtime = Runtime(args.gpu)
    for policy in ("ACT", "SmolVLA"):
        for task, tag in TASKS.items():
            episodes = [int(x) for x in addendum["b3"]["episodes"][task]]
            run_task(runtime, policy, task, tag, episodes)


if __name__ == "__main__":
    main()
