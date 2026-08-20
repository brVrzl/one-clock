#!/usr/bin/env python3
"""Build right-censored temporal-validity targets from frozen LIBERO ACT chunks.

This is a teacher-forced, offline dataset construction step.  It reads stored
LIBERO demonstrations, queries the frozen ACT checkpoint once per sampled
observation, and saves the predicted chunk plus group-wise validity targets.
It never instantiates LIBERO, steps an environment, trains a model, or imports
the executor.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.group_prediction_persistence.audit import (  # noqa: E402
    decode_selected_frames,
    infer_batch,
    load_action_normalization,
    load_episode_arrays,
    load_json_lines,
)
from scripts.run_libero_gate0 import load_policy_and_processors  # noqa: E402


CHUNK_SIZE = 100
SAMPLE_INTERVAL = 25
SEED = 20260820
PHASES = ("early", "middle", "late")
ARM_TRANSLATION_TOLERANCE = 1.0
ARM_ROTATION_TOLERANCE = 1.0
GRIPPER_NORMALIZED_ABSOLUTE_TOLERANCE = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/thor/projects/checkpoints/zeromidnight_act_libero_object"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/home/thor/datasets/libero_object_25_08_23_lerobotv2.1"),
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/temporal_reliability")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def sample_starts(length: int) -> list[int]:
    """Use phase-stratified starts and retain every demonstrable suffix."""

    starts = set(range(0, length, SAMPLE_INTERVAL))
    starts.update({math.ceil(length / 3), math.ceil(2 * length / 3)})
    return sorted(start for start in starts if 0 <= start < length)


def phase_label(frame_index: int, episode_length: int) -> str:
    progress = frame_index / episode_length
    if progress < 1 / 3:
        return "early"
    if progress < 2 / 3:
        return "middle"
    return "late"


def validity_targets(
    predicted: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    action_std: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct pointwise and prefix-survival validity labels.

    The arm label is a conjunction of separately normalized translation and
    rotation criteria; it deliberately does not collapse quantities with
    different physical units into a single six-dimensional norm.  The gripper
    label requires both bounded normalized magnitude error and command-sign
    agreement.
    """

    difference = predicted - target
    with np.errstate(invalid="ignore", divide="ignore"):
        translation_error = np.sqrt(np.mean((difference[:, :, :3] / action_std[:3]) ** 2, axis=2))
        rotation_error = np.sqrt(np.mean((difference[:, :, 3:6] / action_std[3:6]) ** 2, axis=2))
        gripper_error = np.abs(difference[:, :, 6]) / float(action_std[6])
    target_sign = np.where(target[:, :, 6] >= 0.0, 1, -1)
    predicted_sign = np.where(predicted[:, :, 6] >= 0.0, 1, -1)
    gripper_sign_match = predicted_sign == target_sign

    arm_pointwise = (
        (translation_error <= ARM_TRANSLATION_TOLERANCE)
        & (rotation_error <= ARM_ROTATION_TOLERANCE)
        & observed
    )
    gripper_pointwise = (
        (gripper_error <= GRIPPER_NORMALIZED_ABSOLUTE_TOLERANCE)
        & gripper_sign_match
        & observed
    )

    def prefix_survival(valid: np.ndarray) -> np.ndarray:
        # `observed` is a contiguous prefix for every sample.  Marking the
        # censored tail False avoids treating unavailable future labels as
        # failures; `observed` remains the required evaluation mask.
        return np.logical_and.accumulate(valid, axis=1) & observed

    return {
        "arm_translation_error": translation_error.astype(np.float32),
        "arm_rotation_error": rotation_error.astype(np.float32),
        "gripper_normalized_absolute_error": gripper_error.astype(np.float32),
        "gripper_sign_match": gripper_sign_match,
        "arm_pointwise_valid": arm_pointwise,
        "gripper_pointwise_valid": gripper_pointwise,
        "arm_survival_valid": prefix_survival(arm_pointwise),
        "gripper_survival_valid": prefix_survival(gripper_pointwise),
    }


def collect_samples(
    *,
    dataset_root: Path,
    episodes: list[dict[str, Any]],
    tasks: dict[int, str],
    policy: Any,
    policy_preprocessor: Any,
    policy_postprocessor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    predicted_batches: list[np.ndarray] = []
    observed_batches: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    task_points: Counter[str] = Counter()
    task_episodes: Counter[str] = Counter()
    episode_points: Counter[int] = Counter()

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        states = np.stack([item["state"] for item in pending], axis=0)
        predicted_batches.append(
            infer_batch(
                states=states,
                agent_images=[item["agent_image"] for item in pending],
                wrist_images=[item["wrist_image"] for item in pending],
                policy=policy,
                policy_preprocessor=policy_preprocessor,
                policy_postprocessor=policy_postprocessor,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
            )
        )
        observed = np.zeros((len(pending), CHUNK_SIZE), dtype=bool)
        for row, item in enumerate(pending):
            observed[row, : len(item["target_suffix"])] = True
        observed_batches.append(observed)
        metadata.extend(item["metadata"] for item in pending)
        pending = []

    # The target is retained only until labels are materialized, keeping the
    # committed artifact compact and avoiding a duplicate of demonstration data.
    target_batches: list[np.ndarray] = []

    for episode_number, episode in enumerate(episodes, start=1):
        episode_index = int(episode["episode_index"])
        episode_length = int(episode["length"])
        starts = sample_starts(episode_length)
        states, actions = load_episode_arrays(dataset_root, episode_index)
        if states.shape[0] != episode_length or actions.shape[0] != episode_length:
            raise RuntimeError(f"Episode {episode_index} length mismatch")
        task_name = str(episode["tasks"][0])
        task_index = next(index for index, name in tasks.items() if name == task_name)
        agent_path = dataset_root / "videos/chunk-000/observation.images.image" / f"episode_{episode_index:06d}.mp4"
        wrist_path = dataset_root / "videos/chunk-000/observation.images.wrist_image" / f"episode_{episode_index:06d}.mp4"
        agent_frames = decode_selected_frames(agent_path, starts)
        wrist_frames = decode_selected_frames(wrist_path, starts)
        for start in starts:
            suffix = actions[start : start + CHUNK_SIZE]
            pending.append(
                {
                    "state": states[start],
                    "agent_image": agent_frames[start],
                    "wrist_image": wrist_frames[start],
                    "target_suffix": suffix,
                    "metadata": {
                        "episode_index": episode_index,
                        "task_index": task_index,
                        "task_name": task_name,
                        "frame_index": start,
                        "episode_length": episode_length,
                        "progress": start / episode_length,
                        "phase": phase_label(start, episode_length),
                        "observed_offsets": int(len(suffix)),
                    },
                }
            )
            episode_points[episode_index] += 1
            task_points[task_name] += 1
        task_episodes[task_name] += 1
        if len(pending) >= batch_size:
            target = np.full((len(pending), CHUNK_SIZE, 7), np.nan, dtype=np.float32)
            for row, item in enumerate(pending):
                target[row, : len(item["target_suffix"])] = item["target_suffix"]
            target_batches.append(target)
            flush()
        if episode_number == 1 or episode_number % 20 == 0 or episode_number == len(episodes):
            print(f"processed episodes {episode_number}/{len(episodes)}; sampled points {sum(episode_points.values())}", flush=True)
    if pending:
        target = np.full((len(pending), CHUNK_SIZE, 7), np.nan, dtype=np.float32)
        for row, item in enumerate(pending):
            target[row, : len(item["target_suffix"])] = item["target_suffix"]
        target_batches.append(target)
        flush()

    predicted = np.concatenate(predicted_batches, axis=0)
    target = np.concatenate(target_batches, axis=0)
    observed = np.concatenate(observed_batches, axis=0)
    if predicted.shape != target.shape or observed.shape != predicted.shape[:2] or len(metadata) != len(predicted):
        raise RuntimeError("Prediction, target, censoring, and metadata shapes do not match")
    coverage = {
        "episodes_analyzed": len(episodes),
        "observation_points_analyzed": int(predicted.shape[0]),
        "predicted_action_steps": int(np.prod(predicted.shape[:2])),
        "observed_target_pairs": int(observed.sum()),
        "task_distribution_points": dict(sorted(task_points.items())),
        "task_distribution_episodes": dict(sorted(task_episodes.items())),
        "episode_points_min": min(episode_points.values()),
        "episode_points_max": max(episode_points.values()),
        "episode_points_mean": float(np.mean(list(episode_points.values()))),
    }
    return predicted, target, metadata, {"observed": observed, "coverage": coverage}


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    dataset_root = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if not checkpoint.is_dir() or not dataset_root.is_dir():
        raise FileNotFoundError(f"checkpoint={checkpoint}, dataset={dataset_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    tasks_rows = load_json_lines(dataset_root / "meta/tasks.jsonl")
    tasks = {int(row["task_index"]): str(row["task"]) for row in tasks_rows}
    episodes = load_json_lines(dataset_root / "meta/episodes.jsonl")
    if len(episodes) != int(info["total_episodes"]) or len(tasks) != int(info["total_tasks"]):
        raise RuntimeError("Dataset metadata count mismatch")

    import torch

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    runtime_config = {
        "task_suite": "libero_object", "task_id": 0, "obs_type": "pixels_agent_pos",
        "camera_name": "agentview_image,robot0_eye_in_hand_image",
        "camera_name_mapping": {"agentview_image": "image", "robot0_eye_in_hand_image": "wrist_image"},
        "observation_width": 256, "observation_height": 256, "control_freq": 20,
        "init_states": True, "hard_reset": True, "control_mode": "relative", "device": args.device,
    }
    policy, policy_preprocessor, policy_postprocessor, env_preprocessor, env_postprocessor = load_policy_and_processors(runtime_config, checkpoint)
    if policy.config.temporal_ensemble_coeff is not None or int(policy.config.chunk_size) != CHUNK_SIZE:
        raise RuntimeError("Expected a frozen, non-ensembled ACT checkpoint with 100-step chunks")
    action_std = np.asarray(load_action_normalization(checkpoint)["std"], dtype=np.float32)
    if action_std.shape != (7,) or np.any(action_std <= 0):
        raise RuntimeError("Invalid action normalization statistics")

    predicted, target, metadata, extras = collect_samples(
        dataset_root=dataset_root, episodes=episodes, tasks=tasks, policy=policy,
        policy_preprocessor=policy_preprocessor, policy_postprocessor=policy_postprocessor,
        env_preprocessor=env_preprocessor, env_postprocessor=env_postprocessor, batch_size=args.batch_size,
    )
    observed = extras["observed"]
    labels = validity_targets(predicted, target, observed, action_std)
    arrays: dict[str, np.ndarray] = {
        "predicted_actions": predicted,
        "observed_offsets": observed,
        "episode_index": np.asarray([row["episode_index"] for row in metadata], dtype=np.int32),
        "task_index": np.asarray([row["task_index"] for row in metadata], dtype=np.int16),
        "frame_index": np.asarray([row["frame_index"] for row in metadata], dtype=np.int16),
        "episode_length": np.asarray([row["episode_length"] for row in metadata], dtype=np.int16),
        "progress": np.asarray([row["progress"] for row in metadata], dtype=np.float32),
        "phase_code": np.asarray([PHASES.index(row["phase"]) for row in metadata], dtype=np.int8),
    }
    arrays.update(labels)
    np.savez_compressed(output_dir / "reliability_dataset.npz", **arrays)
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    manifest = {
        "purpose": "Offline oracle reliability target; not a ground-truth horizon and not a task-success label.",
        "dataset": {"repo_id": "DorayakiLin/libero_object_25_08_23_lerobotv2.1", "root": str(dataset_root), "revision": "cbf7122bbdbaa0c50517a6a4b2ae663d0e96e51a", "total_episodes": int(info["total_episodes"]), "total_frames": int(info["total_frames"]), "total_tasks": int(info["total_tasks"])},
        "checkpoint": {"root": str(checkpoint), "chunk_size": CHUNK_SIZE, "action_std": action_std.astype(float).tolist(), "frozen": True, "temporal_ensemble_coeff": None},
        "sampling": {"all_episodes": True, "fixed_interval_frames": SAMPLE_INTERVAL, "additional_starts": "ceil(L/3), ceil(2L/3)", "right_censoring": "offsets after episode end are masked, never labelled invalid"},
        "groups": {
            "arm": {"dimensions": "action[0:6]", "pointwise_validity": "translation normalized RMS <= 1.0 AND rotation normalized RMS <= 1.0"},
            "gripper": {"dimensions": "action[6]", "pointwise_validity": "normalized absolute error <= 1.0 AND sign(prediction) == sign(demonstration)"},
        },
        "survival_target": "Y_g(t,k) = product_{j=0..k} V_g(t,j), evaluated only where observed_offsets[t,k] is true.",
        "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()},
        "coverage": extras["coverage"],
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available()), "device_requested": args.device},
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(extras["coverage"], indent=2))
    print(f"wrote {output_dir / 'reliability_dataset.npz'}")


if __name__ == "__main__":
    main()
