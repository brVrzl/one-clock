"""Resumable one-forward-per-frame RoboTwin frozen-policy cache builder."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np

from .target_builder import GROUP_NAMES, build_refresh_targets


LOGGER = logging.getLogger("robotwin50.cache")


class ContractMismatch(RuntimeError):
    """Raised before inference when a frozen policy cannot consume the dataset."""


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_names(feature: dict[str, Any]) -> list[str] | None:
    names = feature.get("names")
    if isinstance(names, dict):
        names = names.get("motors")
    if isinstance(names, list) and all(isinstance(name, str) for name in names):
        return list(names)
    return None


def audit_checkpoint_contract(dataset_root: Path, checkpoint: Path) -> dict[str, Any]:
    """Check actual pinned JSON contracts without constructing the model."""

    dataset_info = _load_json(dataset_root / "meta" / "info.json")
    config = _load_json(checkpoint / "config.json")
    state_feature = dataset_info["features"]["observation.state"]
    action_feature = dataset_info["features"]["action"]
    state_shape = list(state_feature["shape"])
    action_shape = list(action_feature["shape"])
    dataset_action_names = _feature_names(action_feature)
    policy_state = list(config["input_features"]["observation.state"]["shape"])
    policy_action = list(config["output_features"]["action"]["shape"])
    preprocessor_path = checkpoint / "policy_preprocessor.json"
    preprocessor = _load_json(preprocessor_path) if preprocessor_path.is_file() else {}
    rename_map: dict[str, str] = {}
    for step in preprocessor.get("steps", []):
        if step.get("registry_name") == "rename_observations_processor":
            rename_map = dict(step.get("config", {}).get("rename_map", {}))
            break
    expected_cameras = {
        "observation.images.cam_high": "observation.images.camera1",
        "observation.images.cam_left_wrist": "observation.images.camera2",
        "observation.images.cam_right_wrist": "observation.images.camera3",
    }
    action_schema_path = checkpoint / "action_schema.json"
    action_schema = _load_json(action_schema_path) if action_schema_path.is_file() else {}
    policy_action_names = action_schema.get("action_names", action_schema.get("ordering"))
    if not isinstance(policy_action_names, list) or not all(isinstance(name, str) for name in policy_action_names):
        policy_action_names = None
    report = {
        "dataset_state_shape": state_shape,
        "dataset_action_shape": action_shape,
        "policy_state_shape": policy_state,
        "policy_action_shape": policy_action,
        "policy_chunk_size": int(config.get("chunk_size", -1)),
        "policy_n_action_steps": int(config.get("n_action_steps", -1)),
        "policy_n_obs_steps": int(config.get("n_obs_steps", -1)),
        "policy_num_steps": int(config.get("num_steps", -1)),
        "policy_normalization_mapping": config.get("normalization_mapping"),
        "dataset_action_names": dataset_action_names,
        "policy_action_names": policy_action_names,
        "action_schema_path": str(action_schema_path) if action_schema_path.is_file() else None,
        "action_order_contract_match": policy_action_names == dataset_action_names,
        "camera_rename_map": rename_map,
        "camera_contract_match": rename_map == expected_cameras,
        "config_sha256": _sha256(checkpoint / "config.json"),
    }
    mismatches: list[str] = []
    if state_shape != policy_state:
        mismatches.append(f"state shape dataset={state_shape} policy={policy_state}")
    if action_shape != policy_action:
        mismatches.append(f"action shape dataset={action_shape} policy={policy_action}")
    if dataset_action_names is None:
        mismatches.append("dataset action ordering is absent from meta/info.json")
    elif policy_action_names is None:
        mismatches.append("exact policy action ordering is not declared in checkpoint action_schema.json")
    elif policy_action_names != dataset_action_names:
        mismatches.append("policy action ordering does not match the verified RoboTwin dataset ordering")
    if int(config.get("chunk_size", -1)) != 50:
        mismatches.append("primary cache schema requires the actual policy chunk size to be 50")
    if int(config.get("n_action_steps", -1)) != int(config.get("chunk_size", -2)):
        mismatches.append("cache schema requires full predicted chunk before execution truncation")
    if not report["camera_contract_match"]:
        mismatches.append("pinned camera rename map does not match RoboTwin dataset cameras")
    report["mismatches"] = mismatches
    if mismatches:
        raise ContractMismatch(json.dumps(report, indent=2, sort_keys=True))
    return report


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _write_cache_manifest(path: Path, *, args: argparse.Namespace, contract: dict[str, Any]) -> None:
    _atomic_json(
        path,
        {
            "dataset_repo": args.dataset_repo_id,
            "dataset_revision": args.dataset_revision,
            "checkpoint_revision": args.checkpoint_revision,
            "cache_schema": "robotwin_policy_response_v1",
            "group_schema": "experiments/dynamic_reliability_horizon/robotwin50_dataset/group_schema.json",
            "normalization_mapping": contract.get("policy_normalization_mapping"),
            "contract": contract,
            "future_outputs_used_for": "label construction only",
            "estimator_visible_future_outputs": False,
        },
    )


class FrozenSmolVLARunner:
    """Normal LeRobot inference with a non-invasive prefix observation hook."""

    def __init__(self, checkpoint: Path, lerobot_source: Path, device: str, seed: int) -> None:
        sys.path.insert(0, str(lerobot_source / "src"))
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.torch = torch
        self.device = torch.device(device)
        self.seed = int(seed)
        self.policy = SmolVLAPolicy.from_pretrained(str(checkpoint)).to(self.device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        self.latents: list[np.ndarray] = []
        self._hook = self.policy.model.vlm_with_expert.register_forward_hook(
            self._capture_prefix_latent, with_kwargs=True
        )

    def close(self) -> None:
        self._hook.remove()

    def _capture_prefix_latent(self, _module: Any, _args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        inputs_embeds = kwargs.get("inputs_embeds")
        if not isinstance(inputs_embeds, (list, tuple)) or len(inputs_embeds) != 2 or inputs_embeds[1] is not None:
            return
        try:
            prefix_output = output[0][0]
        except (IndexError, KeyError, TypeError):
            # The latent is optional.  A LeRobot implementation change must
            # not alter frozen-policy actions or make otherwise valid cache
            # generation fail solely because the introspection hook changed.
            return
        if prefix_output is not None:
            self.latents.append(prefix_output.detach().float().mean(dim=1).cpu().numpy()[0])

    def predict(self, frame: dict[str, Any], noise_seed: int) -> np.ndarray:
        generator = self.torch.Generator(device=self.device)
        generator.manual_seed(int(noise_seed))
        noise = self.torch.randn(
            (1, int(self.policy.config.chunk_size), int(self.policy.config.max_action_dim)),
            generator=generator,
            device=self.device,
            dtype=self.torch.float32,
        )
        with self.torch.inference_mode():
            self.policy.reset()
            batch = self.preprocessor(dict(frame))
            action = self.policy.predict_action_chunk(batch, noise=noise)
            action = self.postprocessor(action)
        value = action.detach().cpu().float().numpy()[0]
        expected = (int(self.policy.config.chunk_size), 14)
        if value.shape != expected or not np.isfinite(value).all():
            raise RuntimeError(f"policy returned invalid full chunk shape={value.shape}")
        return value


def _episode_ids(root: Path) -> list[int]:
    import pandas as pd

    paths = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError("no local episode metadata shards")
    values: list[int] = []
    for path in paths:
        values.extend(int(v) for v in pd.read_parquet(path)["episode_index"].tolist())
    return sorted(set(values))


def _read_progress(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    if path.is_file():
        return _load_json(path)
    return {
        "status": "running",
        "tasks_completed": [],
        "episodes_completed": [],
        "frames_inferred": 0,
        "failures": [],
        "throughput_frames_per_second": None,
        "cache_bytes": 0,
        "checkpoint_revision": args.checkpoint_revision,
        "dataset_revision": args.dataset_revision,
        "started_unix": time.time(),
    }


def _write_progress(path: Path, progress: dict[str, Any]) -> None:
    progress["last_update_unix"] = time.time()
    _atomic_json(path, progress)


def _disk_guard(cache_root: Path, minimum_free_bytes: int) -> None:
    usage = shutil.disk_usage(cache_root)
    if usage.free < minimum_free_bytes:
        raise RuntimeError(f"disk pressure guard: only {usage.free} bytes free under {cache_root}")


def process(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.resolve()
    checkpoint = args.checkpoint.resolve()
    cache_root = args.cache_root.resolve() / args.checkpoint_revision
    output_root = cache_root / "policy_outputs"
    label_root = cache_root / "labels"
    manifest_root = cache_root / "manifests"
    for path in (output_root, label_root, manifest_root):
        path.mkdir(parents=True, exist_ok=True)

    contract = audit_checkpoint_contract(dataset_root, checkpoint)
    _atomic_json(manifest_root / "checkpoint_contract.json", contract)
    _write_cache_manifest(manifest_root / "cache_manifest.json", args=args, contract=contract)
    episode_ids = args.episodes if args.episodes else _episode_ids(dataset_root)
    progress_path = args.progress.resolve()
    progress = _read_progress(progress_path, args)
    completed = {int(v) for v in progress.get("episodes_completed", [])}
    failed_path = manifest_root / "failed_shards.json"
    completed_path = manifest_root / "completed_shards.json"
    failures = list(progress.get("failures", []))
    completed_shards = _load_json(completed_path).get("shards", []) if completed_path.is_file() else []
    start = time.perf_counter()
    runner = FrozenSmolVLARunner(checkpoint, args.lerobot_source, args.device, args.seed)
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        for episode_index in episode_ids:
            if int(episode_index) in completed:
                continue
            _disk_guard(cache_root, args.minimum_free_bytes)
            last_error: str | None = None
            for attempt in range(args.retries + 1):
                try:
                    dataset = LeRobotDataset(
                        args.dataset_repo_id,
                        root=dataset_root,
                        episodes=[int(episode_index)],
                        revision=args.dataset_revision,
                        download_videos=False,
                        video_backend="pyav",
                    )
                    chunks: list[np.ndarray] = []
                    frame_ids: list[int] = []
                    task_indices: list[int] = []
                    tasks: list[str] = []
                    noise_seeds: list[int] = []
                    episode_latents_before = len(runner.latents)
                    for row in range(len(dataset)):
                        frame = dataset[row]
                        frame_id = int(frame["index"].item())
                        noise_seed = int(args.seed + int(episode_index) * 100000 + frame_id)
                        chunks.append(runner.predict(frame, noise_seed))
                        frame_ids.append(frame_id)
                        task_indices.append(int(frame["task_index"].item()))
                        tasks.append(str(frame["task"]))
                        noise_seeds.append(noise_seed)
                    latent_slice = runner.latents[episode_latents_before:]
                    latent_dim = len(latent_slice[0]) if latent_slice else 0
                    if latent_slice and len(latent_slice) != len(chunks):
                        LOGGER.warning(
                            "latent hook count=%s differs from frame count=%s; marking z_t unavailable",
                            len(latent_slice),
                            len(chunks),
                        )
                        latent_dim = 0
                    z_t = np.stack(latent_slice).astype(np.float32) if latent_dim else np.empty((len(chunks), 0), dtype=np.float32)
                    action_chunks = np.stack(chunks).astype(np.float32)
                    task_key = f"task_index_{task_indices[0]}"
                    shard = output_root / task_key / f"episode_{int(episode_index):06d}.npz"
                    _atomic_npz(
                        shard,
                        episode_index=np.full(len(chunks), int(episode_index), dtype=np.int64),
                        frame_index=np.asarray(frame_ids, dtype=np.int64),
                        task_index=np.asarray(task_indices, dtype=np.int64),
                        task=np.asarray(tasks, dtype=str),
                        action_chunks=action_chunks,
                        z_t=z_t,
                        noise_seed=np.asarray(noise_seeds, dtype=np.int64),
                    )
                    labels = build_refresh_targets(
                        action_chunks,
                        np.asarray(frame_ids, dtype=np.int64),
                        thresholds={name: args.threshold for name in GROUP_NAMES},
                    )
                    label_shard = label_root / task_key / f"episode_{int(episode_index):06d}.npz"
                    _atomic_npz(label_shard, **labels, episode_index=np.asarray(int(episode_index)), task_index=np.asarray(task_indices[0]))
                    shard_record = {
                        "episode_index": int(episode_index),
                        "task_key": task_key,
                        "frames": len(chunks),
                        "policy_shard": str(shard),
                        "label_shard": str(label_shard),
                        "policy_sha256": _sha256(shard),
                        "label_sha256": _sha256(label_shard),
                        "normalization_mapping": contract.get("policy_normalization_mapping"),
                        "attempt": attempt,
                    }
                    completed_shards.append(shard_record)
                    _atomic_json(completed_path, {"shards": completed_shards})
                    completed.add(int(episode_index))
                    progress["episodes_completed"] = sorted(completed)
                    progress["tasks_completed"] = sorted({str(record["task_key"]) for record in completed_shards})
                    progress["frames_inferred"] = int(progress.get("frames_inferred", 0)) + len(chunks)
                    progress["cache_bytes"] = sum(p.stat().st_size for p in cache_root.rglob("*.npz"))
                    progress["throughput_frames_per_second"] = progress["frames_inferred"] / max(time.perf_counter() - start, 1e-9)
                    _write_progress(progress_path, progress)
                    LOGGER.info("completed episode=%s frames=%s throughput=%.2f", episode_index, len(chunks), progress["throughput_frames_per_second"])
                    break
                except Exception as exc:  # one shard is isolated; failures remain visible
                    last_error = f"attempt={attempt}: {type(exc).__name__}: {exc}"
                    LOGGER.exception("episode %s failed", episode_index)
                    if attempt < args.retries:
                        time.sleep(args.retry_seconds)
            else:
                record = {"episode_index": int(episode_index), "error": last_error}
                failures.append(record)
                progress["failures"] = failures
                _atomic_json(failed_path, {"shards": failures})
                _write_progress(progress_path, progress)
    finally:
        runner.close()
    progress["status"] = "complete" if not failures else "complete_with_failures"
    progress["failures"] = failures
    _write_progress(progress_path, progress)
    return progress


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lerobot-source", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", default="lerobot/robotwin_unified")
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-seconds", type=float, default=5.0)
    parser.add_argument("--minimum-free-bytes", type=int, default=100_000_000_000)
    parser.add_argument("--episodes", type=int, nargs="*")
    parser.add_argument("--log", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(args.log) if args.log else logging.StreamHandler()],
    )
    try:
        result = process(args)
    except ContractMismatch as exc:
        progress = _read_progress(args.progress, args)
        progress["status"] = "blocked_contract"
        progress.setdefault("failures", []).append({"stage": "contract", "reason": str(exc)})
        _write_progress(args.progress, progress)
        LOGGER.error("frozen-policy contract mismatch; no cache inference started: %s", exc)
        raise SystemExit(2)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
