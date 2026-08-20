#!/usr/bin/env python3
"""Offline, teacher-forced prediction-persistence audit for the frozen LIBERO ACT policy.

This script deliberately does not instantiate LIBERO, step an environment, or
touch the executor. The v2.1 demonstrations are read directly with PyArrow
and PyAV because the installed LeRobot 0.6.2 reader targets the newer v3.0
dataset format. Policy preprocessing/inference still uses the existing
checkpoint-compatible LeRobot/runner path.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.run_libero_gate0 import load_policy_and_processors  # noqa: E402


CHUNK_SIZE = 100
SAMPLE_INTERVAL = 50
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260820
METRIC_NAMES = (
    "arm_translation_normalized_rms",
    "arm_rotation_normalized_rms",
    "gripper_absolute_error",
    "gripper_absolute_error_normalized",
    "gripper_binary_mismatch_rate",
)
THRESHOLDS = {
    "arm_translation_normalized_rms": 1.0,
    "arm_rotation_normalized_rms": 1.0,
    "gripper_absolute_error_normalized": 1.0,
    "gripper_binary_mismatch_rate": 0.5,
}


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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/group_prediction_persistence",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sample_starts(length: int) -> list[int]:
    """Sample full chunks at fixed intervals, adding the final valid start."""

    max_start = length - CHUNK_SIZE
    if max_start < 0:
        return []
    starts = list(range(0, max_start + 1, SAMPLE_INTERVAL))
    if max_start not in starts:
        starts.append(max_start)
    return sorted(set(starts))


def phase_label(position: int, denominator: int) -> str:
    if denominator <= 0:
        return "early"
    fraction = position / denominator
    if fraction < 1 / 3:
        return "early"
    if fraction < 2 / 3:
        return "middle"
    return "late"


def decode_selected_frames(path: Path, frame_indices: list[int]) -> dict[int, np.ndarray]:
    import av

    wanted = set(int(index) for index in frame_indices)
    if not wanted:
        return {}
    decoded: dict[int, np.ndarray] = {}
    with av.open(str(path), mode="r") as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                decoded[index] = frame.to_ndarray(format="rgb24")
            if index >= max(wanted) and len(decoded) == len(wanted):
                break
    missing = wanted - decoded.keys()
    if missing:
        raise RuntimeError(f"Could not decode frames {sorted(missing)} from {path}")
    return decoded


def axis_angle_to_quaternion(axis_angle: np.ndarray) -> np.ndarray:
    """Inverse of the verified LIBERO processor's quaternion-to-axis-angle map."""

    axis_angle = np.asarray(axis_angle, dtype=np.float32)
    angles = np.linalg.norm(axis_angle, axis=1)
    half_angles = angles / 2.0
    scale = np.empty_like(angles)
    nonzero = angles > 1e-8
    scale[nonzero] = np.sin(half_angles[nonzero]) / angles[nonzero]
    scale[~nonzero] = 0.5
    quaternion = np.zeros((axis_angle.shape[0], 4), dtype=np.float32)
    quaternion[:, :3] = axis_angle * scale[:, None]
    quaternion[:, 3] = np.cos(half_angles)
    return quaternion


def make_raw_observation_batch(
    states: np.ndarray,
    agent_images: list[np.ndarray],
    wrist_images: list[np.ndarray],
) -> dict[str, Any]:
    quaternion = axis_angle_to_quaternion(states[:, 3:6])
    return {
        "pixels": {
            "image": np.stack(agent_images, axis=0),
            "wrist_image": np.stack(wrist_images, axis=0),
        },
        "robot_state": {
            "eef": {
                "pos": states[:, :3],
                "quat": quaternion,
            },
            "gripper": {
                "qpos": states[:, 6:8],
            },
        },
    }


def prepare_policy_batch(
    raw_observation: dict[str, Any],
    expected_states: np.ndarray,
    env_preprocessor: Any,
    policy_preprocessor: Any,
) -> dict[str, Any]:
    """Apply the same env and policy preprocessing as the existing runner.

    The runner helper adds a batch dimension for one live environment. Here
    arrays already have a batch dimension, so the equivalent preprocessing is
    called directly to avoid adding a spurious extra dimension.
    """

    from lerobot.envs.utils import preprocess_observation

    processed = preprocess_observation(raw_observation)
    processed = env_preprocessor(processed)
    recovered_state = processed["observation.state"].detach().cpu().numpy()
    state_error = np.max(np.abs(recovered_state - expected_states))
    if state_error > 2e-4:
        raise RuntimeError(f"LIBERO state reconstruction mismatch: max error {state_error}")
    return policy_preprocessor(processed)


def infer_batch(
    *,
    states: np.ndarray,
    agent_images: list[np.ndarray],
    wrist_images: list[np.ndarray],
    policy: Any,
    policy_preprocessor: Any,
    policy_postprocessor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
) -> np.ndarray:
    import torch

    from lerobot.utils.constants import ACTION

    raw_observation = make_raw_observation_batch(states, agent_images, wrist_images)
    model_observation = prepare_policy_batch(
        raw_observation,
        states,
        env_preprocessor,
        policy_preprocessor,
    )
    with torch.inference_mode():
        normalized_chunk = policy.predict_action_chunk(model_observation)
        action_chunk = policy_postprocessor(normalized_chunk)
    action_chunk = env_postprocessor({ACTION: action_chunk})[ACTION]
    result = action_chunk.detach().cpu().numpy()
    if result.shape != (len(states), CHUNK_SIZE, 7):
        raise RuntimeError(f"Unexpected predicted chunk shape: {result.shape}")
    if not np.isfinite(result).all():
        raise RuntimeError("Frozen policy produced non-finite actions")
    return result.astype(np.float32, copy=False)


def load_action_normalization(checkpoint: Path) -> dict[str, list[float]]:
    from safetensors.torch import load_file

    stats_path = checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    stats = load_file(str(stats_path))
    return {
        "mean": stats["action.mean"].detach().cpu().numpy().astype(float).tolist(),
        "std": stats["action.std"].detach().cpu().numpy().astype(float).tolist(),
        "count": [float(stats["action.count"].item())],
        "source": str(stats_path),
    }


def load_episode_arrays(dataset_root: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq

    path = dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    table = pq.read_table(path, columns=["observation.state", "action", "episode_index"])
    episode_ids = set(int(value) for value in table["episode_index"].to_pylist())
    if episode_ids != {episode_index}:
        raise RuntimeError(f"Unexpected episode IDs in {path}: {episode_ids}")
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if states.shape[1:] != (8,) or actions.shape[1:] != (7,):
        raise RuntimeError(f"Unexpected arrays in {path}: state={states.shape}, action={actions.shape}")
    return states, actions


def collect_predictions(
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
    true_batches: list[np.ndarray] = []
    sample_metadata: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        states = np.stack([item["state"] for item in pending], axis=0)
        agent_images = [item["agent_image"] for item in pending]
        wrist_images = [item["wrist_image"] for item in pending]
        predicted_batches.append(
            infer_batch(
                states=states,
                agent_images=agent_images,
                wrist_images=wrist_images,
                policy=policy,
                policy_preprocessor=policy_preprocessor,
                policy_postprocessor=policy_postprocessor,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
            )
        )
        true_batches.append(np.stack([item["true_actions"] for item in pending], axis=0))
        sample_metadata.extend(item["metadata"] for item in pending)
        pending = []

    episode_point_counts: Counter[int] = Counter()
    task_point_counts: Counter[str] = Counter()
    task_episode_counts: Counter[str] = Counter()
    for episode_number, episode in enumerate(episodes, start=1):
        episode_index = int(episode["episode_index"])
        length = int(episode["length"])
        starts = sample_starts(length)
        if not starts:
            raise RuntimeError(f"Episode {episode_index} has no full {CHUNK_SIZE}-step target")
        states, actions = load_episode_arrays(dataset_root, episode_index)
        if len(states) != length or len(actions) != length:
            raise RuntimeError(f"Episode {episode_index} metadata/data length mismatch")
        task_name = str(episode["tasks"][0])
        task_index = next(index for index, name in tasks.items() if name == task_name)
        agent_path = (
            dataset_root
            / "videos"
            / "chunk-000"
            / "observation.images.image"
            / f"episode_{episode_index:06d}.mp4"
        )
        wrist_path = (
            dataset_root
            / "videos"
            / "chunk-000"
            / "observation.images.wrist_image"
            / f"episode_{episode_index:06d}.mp4"
        )
        agent_frames = decode_selected_frames(agent_path, starts)
        wrist_frames = decode_selected_frames(wrist_path, starts)
        max_start = length - CHUNK_SIZE
        for start in starts:
            pending.append(
                {
                    "state": states[start],
                    "agent_image": agent_frames[start],
                    "wrist_image": wrist_frames[start],
                    "true_actions": actions[start : start + CHUNK_SIZE],
                    "metadata": {
                        "episode_index": episode_index,
                        "task_index": task_index,
                        "task_name": task_name,
                        "frame_index": start,
                        "episode_length": length,
                        "max_valid_start": max_start,
                        "query_window_phase": phase_label(start, max_start),
                        "episode_phase": phase_label(start, length - 1),
                    },
                }
            )
            episode_point_counts[episode_index] += 1
            task_point_counts[task_name] += 1
        task_episode_counts[task_name] += 1
        if len(pending) >= batch_size:
            flush()
        if episode_number == 1 or episode_number % 20 == 0 or episode_number == len(episodes):
            print(
                f"processed episodes {episode_number}/{len(episodes)}; "
                f"sampled points {sum(episode_point_counts.values())}",
                flush=True,
            )
    flush()
    predicted = np.concatenate(predicted_batches, axis=0)
    true = np.concatenate(true_batches, axis=0)
    if len(sample_metadata) != predicted.shape[0]:
        raise RuntimeError("Prediction/sample metadata count mismatch")
    coverage = {
        "episodes_analyzed": len(episodes),
        "observation_points_analyzed": int(predicted.shape[0]),
        "task_distribution_points": dict(sorted(task_point_counts.items())),
        "task_distribution_episodes": dict(sorted(task_episode_counts.items())),
        "episode_points_min": min(episode_point_counts.values()),
        "episode_points_max": max(episode_point_counts.values()),
        "episode_points_mean": float(np.mean(list(episode_point_counts.values()))),
    }
    return predicted, true, sample_metadata, coverage


def error_arrays(predicted: np.ndarray, true: np.ndarray, action_std: np.ndarray) -> dict[str, np.ndarray]:
    difference = predicted - true
    translation = np.sqrt(np.mean((difference[:, :, :3] / action_std[:3]) ** 2, axis=2))
    rotation = np.sqrt(np.mean((difference[:, :, 3:6] / action_std[3:6]) ** 2, axis=2))
    gripper_abs = np.abs(difference[:, :, 6])
    gripper_abs_normalized = gripper_abs / float(action_std[6])
    target_binary = np.where(true[:, :, 6] >= 0.0, 1.0, -1.0)
    predicted_binary = np.where(predicted[:, :, 6] >= 0.0, 1.0, -1.0)
    mismatch = (predicted_binary != target_binary).astype(np.float32)
    return {
        "arm_translation_normalized_rms": translation,
        "arm_rotation_normalized_rms": rotation,
        "gripper_absolute_error": gripper_abs,
        "gripper_absolute_error_normalized": gripper_abs_normalized,
        "gripper_binary_mismatch_rate": mismatch,
    }


def linear_slope(curve: np.ndarray) -> float:
    x = np.arange(len(curve), dtype=np.float64)
    centered_x = x - x.mean()
    centered_y = curve - curve.mean()
    return float(np.sum(centered_x * centered_y) / np.sum(centered_x**2))


def scalar_metrics(curve: np.ndarray, threshold: float | None) -> dict[str, Any]:
    k = np.arange(len(curve), dtype=np.float64)
    auc = float(np.trapezoid(curve, k) / max(1, len(curve) - 1))
    first = float(np.mean(curve[:10]))
    last = float(np.mean(curve[-10:]))
    crossing = None
    if threshold is not None:
        crossed = np.flatnonzero(curve >= threshold)
        crossing = int(crossed[0]) if len(crossed) else None
    return {
        "auc_mean_over_horizon": auc,
        "linear_slope_per_step": linear_slope(curve),
        "last10_minus_first10": last - first,
        "first10_mean": first,
        "last10_mean": last,
        "positive_adjacent_fraction": float(np.mean(np.diff(curve) > 0.0)),
        "first_crossing_threshold_step": crossing,
        "threshold": threshold,
    }


def episode_level_curves(
    error: np.ndarray,
    sample_metadata: list[dict[str, Any]],
    sample_indices: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    episode_to_indices: dict[int, list[int]] = defaultdict(list)
    for index in sample_indices.tolist():
        episode_to_indices[int(sample_metadata[index]["episode_index"])].append(index)
    episode_ids = sorted(episode_to_indices)
    curves = np.stack([error[episode_to_indices[episode_id]].mean(axis=0) for episode_id in episode_ids])
    return curves, episode_ids


def bootstrap_curve_summary(
    episode_curves: np.ndarray,
    curve: np.ndarray,
    threshold: float | None,
    rng: np.random.Generator,
) -> dict[str, Any]:
    n_episodes = episode_curves.shape[0]
    if n_episodes == 0:
        return {"status": "no_samples", "n_episodes": 0}
    if n_episodes == 1:
        bootstrap_curves = episode_curves.copy()
    else:
        draws = rng.integers(0, n_episodes, size=(BOOTSTRAP_DRAWS, n_episodes))
        bootstrap_curves = episode_curves[draws].mean(axis=1)
    low, high = np.percentile(bootstrap_curves, [2.5, 97.5], axis=0)
    point_summary = scalar_metrics(curve, threshold)
    aucs = np.asarray([scalar_metrics(item, threshold)["auc_mean_over_horizon"] for item in bootstrap_curves])
    slopes = np.asarray([linear_slope(item) for item in bootstrap_curves])
    deltas = np.asarray([float(np.mean(item[-10:]) - np.mean(item[:10])) for item in bootstrap_curves])
    point_summary.update(
        {
            "status": "measured",
            "n_episodes": n_episodes,
            "curve": curve.astype(float).tolist(),
            "bootstrap_95ci_low": low.astype(float).tolist(),
            "bootstrap_95ci_high": high.astype(float).tolist(),
            "auc_bootstrap_95ci": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
            "slope_bootstrap_95ci": [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))],
            "delta_bootstrap_95ci": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        }
    )
    return point_summary


def summarize_subset(
    errors: dict[str, np.ndarray],
    sample_metadata: list[dict[str, Any]],
    mask: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    indices = np.flatnonzero(mask)
    result: dict[str, Any] = {
        "n_observation_points": int(len(indices)),
        "n_episodes": int(len({int(sample_metadata[i]["episode_index"]) for i in indices})),
    }
    if len(indices) == 0:
        result["status"] = "no_samples"
        return result
    for metric_name, error in errors.items():
        curve = error[indices].mean(axis=0)
        episode_curves, _ = episode_level_curves(error, sample_metadata, indices)
        result[metric_name] = bootstrap_curve_summary(
            episode_curves,
            curve,
            THRESHOLDS.get(metric_name),
            rng,
        )
    result["status"] = "measured"
    return result


def plot_curve_panel(ax: Any, summary: dict[str, Any], label: str, color: str) -> None:
    if summary.get("status") != "measured":
        return
    k = np.arange(CHUNK_SIZE)
    curve = np.asarray(summary["curve"])
    low = np.asarray(summary["bootstrap_95ci_low"])
    high = np.asarray(summary["bootstrap_95ci_high"])
    ax.plot(k, curve, color=color, label=label, linewidth=2)
    ax.fill_between(k, low, high, color=color, alpha=0.18, linewidth=0)


def make_plots(
    output_dir: Path,
    overall: dict[str, Any],
    task_metrics: dict[str, Any],
    query_phase_metrics: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("arm_translation_normalized_rms", "Arm translation error (dataset-std RMS)"),
        ("arm_rotation_normalized_rms", "Arm rotation error (dataset-std RMS)"),
        ("gripper_absolute_error", "Gripper absolute error (raw action units)"),
        ("gripper_binary_mismatch_rate", "Gripper binary mismatch rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, (metric, title) in zip(axes.flat, panels, strict=True):
        plot_curve_panel(ax, overall[metric], "all tasks", "tab:blue")
        ax.axvline(0, color="0.7", linewidth=0.7)
        ax.set_title(title)
        ax.set_ylabel("error")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("future step k")
    fig.suptitle("Frozen ACT prediction persistence: full 100-step targets\nshaded bands are episode-bootstrap 95% CIs")
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_error_curves.png", dpi=180)
    plt.close(fig)

    task_names = list(task_metrics)
    task_labels = [name.replace("pick up the ", "").replace(" and place it in the basket", "") for name in task_names]
    heat_metrics = [
        ("arm_translation_normalized_rms", "translation AUC"),
        ("arm_rotation_normalized_rms", "rotation AUC"),
        ("gripper_absolute_error_normalized", "gripper AUC (normalized)"),
        ("gripper_binary_mismatch_rate", "gripper mismatch AUC"),
    ]
    matrix = np.asarray(
        [
            [task_metrics[name][metric]["auc_mean_over_horizon"] for metric, _ in heat_metrics]
            for name in task_names
        ]
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_yticks(np.arange(len(task_labels)), task_labels)
    ax.set_xticks(np.arange(len(heat_metrics)), [label for _, label in heat_metrics], rotation=20, ha="right")
    ax.set_title("Task-wise threshold-free error AUC")
    fig.colorbar(image, ax=ax, label="AUC")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", color="white")
    fig.tight_layout()
    fig.savefig(output_dir / "task_error_auc_heatmap.png", dpi=180)
    plt.close(fig)

    phase_colors = {"early": "tab:green", "middle": "tab:orange", "late": "tab:red"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, (metric, title) in zip(axes.flat, panels, strict=True):
        for phase in ("early", "middle", "late"):
            summary = query_phase_metrics[phase].get(metric, {})
            plot_curve_panel(ax, summary, phase, phase_colors[phase])
        ax.set_title(title)
        ax.set_ylabel("error")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("future step k")
    fig.suptitle("Prediction persistence by phase of the admissible full-chunk query window")
    fig.tight_layout()
    fig.savefig(output_dir / "query_window_phase_curves.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    dataset_root = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    tasks_rows = load_json_lines(dataset_root / "meta/tasks.jsonl")
    tasks = {int(row["task_index"]): str(row["task"]) for row in tasks_rows}
    episodes = load_json_lines(dataset_root / "meta/episodes.jsonl")
    if info["total_episodes"] != len(episodes) or info["total_tasks"] != len(tasks):
        raise RuntimeError("Dataset metadata count mismatch")
    if any(int(episode["length"]) < CHUNK_SIZE for episode in episodes):
        raise RuntimeError("A full 100-step target is unavailable in at least one episode")

    import torch

    torch.manual_seed(BOOTSTRAP_SEED)
    np.random.seed(BOOTSTRAP_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(BOOTSTRAP_SEED)
    runtime_config = {
        "task_suite": "libero_object",
        "task_id": 0,
        "obs_type": "pixels_agent_pos",
        "camera_name": "agentview_image,robot0_eye_in_hand_image",
        "camera_name_mapping": {
            "agentview_image": "image",
            "robot0_eye_in_hand_image": "wrist_image",
        },
        "observation_width": 256,
        "observation_height": 256,
        "control_freq": 20,
        "init_states": True,
        "hard_reset": True,
        "control_mode": "relative",
        "device": args.device,
    }
    policy, policy_preprocessor, policy_postprocessor, env_preprocessor, env_postprocessor = (
        load_policy_and_processors(runtime_config, checkpoint)
    )
    if policy.config.temporal_ensemble_coeff is not None:
        raise RuntimeError("Temporal ensembling must be disabled for this audit")
    if int(policy.config.chunk_size) != CHUNK_SIZE:
        raise RuntimeError(f"Checkpoint chunk size is {policy.config.chunk_size}, expected {CHUNK_SIZE}")

    normalization = load_action_normalization(checkpoint)
    action_std = np.asarray(normalization["std"], dtype=np.float32)
    if np.any(action_std[:6] <= 0) or action_std[6] <= 0:
        raise RuntimeError("Invalid checkpoint action standard deviations")

    predicted, true, sample_metadata, coverage = collect_predictions(
        dataset_root=dataset_root,
        episodes=episodes,
        tasks=tasks,
        policy=policy,
        policy_preprocessor=policy_preprocessor,
        policy_postprocessor=policy_postprocessor,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        batch_size=args.batch_size,
    )
    errors = error_arrays(predicted, true, action_std)
    np.savez_compressed(
        output_dir / "predictions.npz",
        predicted_actions=predicted,
        true_actions=true,
        episode_index=np.asarray([row["episode_index"] for row in sample_metadata], dtype=np.int64),
        task_index=np.asarray([row["task_index"] for row in sample_metadata], dtype=np.int64),
        frame_index=np.asarray([row["frame_index"] for row in sample_metadata], dtype=np.int64),
    )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    all_mask = np.ones(len(sample_metadata), dtype=bool)
    overall = summarize_subset(errors, sample_metadata, all_mask, rng)
    task_metrics: dict[str, Any] = {}
    for task_name in tasks.values():
        mask = np.asarray([row["task_name"] == task_name for row in sample_metadata], dtype=bool)
        task_metrics[task_name] = summarize_subset(errors, sample_metadata, mask, rng)
    query_phase_metrics: dict[str, Any] = {}
    episode_phase_metrics: dict[str, Any] = {}
    for phase in ("early", "middle", "late"):
        query_mask = np.asarray([row["query_window_phase"] == phase for row in sample_metadata], dtype=bool)
        episode_mask = np.asarray([row["episode_phase"] == phase for row in sample_metadata], dtype=bool)
        query_phase_metrics[phase] = summarize_subset(errors, sample_metadata, query_mask, rng)
        episode_phase_metrics[phase] = summarize_subset(errors, sample_metadata, episode_mask, rng)

    summary = {
        "protocol": {
            "sampling": {
                "episodes": "all 454 episodes, sorted by episode_index",
                "full_target_only": True,
                "chunk_size": CHUNK_SIZE,
                "fixed_interval_frames": SAMPLE_INTERVAL,
                "rule": "starts=0, interval=50, plus final valid start length-100 if absent",
                "overlap": "at most 50 frames for interval samples; final valid start is included for coverage",
                "random_seed": None,
            },
            "inference": {
                "frozen": True,
                "training": False,
                "temporal_ensemble_coeff": None,
                "policy_query": "predict_action_chunk",
                "predicted_shape": list(predicted.shape),
                "target_shape": list(true.shape),
                "device": args.device,
            },
            "phase_definitions": {
                "query_window_phase": "early/middle/late by frame_index/(episode_length-100), thirds; used for phase curves",
                "episode_phase": "early/middle/late by frame_index/(episode_length-1), thirds; full chunks censor late episode positions",
            },
        },
        "dataset": {
            "repo_id": "DorayakiLin/libero_object_25_08_23_lerobotv2.1",
            "root": str(dataset_root),
            "revision": "cbf7122bbdbaa0c50517a6a4b2ae663d0e96e51a",
            "total_episodes": info["total_episodes"],
            "total_frames": info["total_frames"],
            "total_tasks": info["total_tasks"],
            "fps": info["fps"],
            "action_dim": 7,
            "state_dim": 8,
        },
        "checkpoint": {
            "root": str(checkpoint),
            "chunk_size": int(policy.config.chunk_size),
            "action_dim": int(policy.config.output_features["action"].shape[0]),
            "normalization": normalization,
        },
        "groups": {
            "arm_translation": "action[0:3]",
            "arm_rotation": "action[3:6]",
            "arm": "action[0:6], never combined across translation and rotation",
            "gripper": "action[6]",
        },
        "metrics": {
            "arm_translation_normalized_rms": "sqrt(mean((prediction-target)^2 / action_std^2)) over action[0:3]; units are dataset standard deviations",
            "arm_rotation_normalized_rms": "sqrt(mean((prediction-target)^2 / action_std^2)) over action[3:6]; units are dataset standard deviations",
            "gripper_absolute_error": "absolute prediction-target error in raw action units",
            "gripper_absolute_error_normalized": "gripper absolute error divided by checkpoint dataset action_std[6]",
            "gripper_binary_mismatch_rate": "target is exactly {-1,+1}; prediction is thresholded at 0; mismatch is reported per k",
            "threshold_free": ["AUC mean over k=0..99", "linear slope per step", "last10-minus-first10", "positive adjacent fraction"],
            "predeclared_thresholds": THRESHOLDS,
            "threshold_interpretation": "first k at which the aggregate curve crosses the fixed threshold; not tuned after results",
            "uncertainty": "episode-level bootstrap 95% CIs, 2000 draws, seed 20260820",
        },
        "coverage": coverage,
        "overall": overall,
        "task_metrics": task_metrics,
        "query_window_phase_metrics": query_phase_metrics,
        "episode_phase_metrics": episode_phase_metrics,
        "artifacts": {
            "predictions": str(output_dir / "predictions.npz"),
            "main_plot": str(output_dir / "prediction_error_curves.png"),
            "task_plot": str(output_dir / "task_error_auc_heatmap.png"),
            "phase_plot": str(output_dir / "query_window_phase_curves.png"),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    make_plots(output_dir, overall, task_metrics, query_phase_metrics)
    print(json.dumps(coverage, indent=2))
    print(f"wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
