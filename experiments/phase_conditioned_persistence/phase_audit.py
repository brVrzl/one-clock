#!/usr/bin/env python3
"""Offline phase-conditioned prediction-persistence audit for frozen LIBERO ACT.

This script performs no rollout and does not instantiate or step LIBERO. It
uses the checkpoint-compatible preprocessing/inference helpers from the
Gate-1 analysis, but samples additional late-episode observations so that
episode-progress phases are identifiable. Predictions are compared with the
demonstrated suffix available before the episode ends; phase comparisons use a
predeclared common future-step range to avoid comparing different horizons.
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
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260820
PHASES = ("early", "middle", "late")
METRICS = (
    "arm_translation_normalized_rms",
    "arm_rotation_normalized_rms",
    "gripper_absolute_error_normalized",
    "gripper_binary_mismatch_rate",
)


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
        default=ROOT / "experiments/phase_conditioned_persistence",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def sample_starts(length: int) -> list[int]:
    """Fixed interval plus explicit phase-boundary starts.

    The final frame is intentionally not added: it has only a one-step target
    and would add no useful persistence information. Every start has at least
    one demonstrated future action.
    """

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


def common_horizon(episodes: list[dict[str, Any]]) -> int:
    """Maximum common suffix length for the explicit late boundary samples."""

    return min(int(episode["length"]) - math.ceil(2 * int(episode["length"]) / 3) for episode in episodes)


def infer_samples(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    predicted_batches: list[np.ndarray] = []
    true_batches: list[np.ndarray] = []
    valid_batches: list[np.ndarray] = []
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
        true = np.full((len(pending), CHUNK_SIZE, 7), np.nan, dtype=np.float32)
        valid = np.zeros((len(pending), CHUNK_SIZE), dtype=bool)
        for row, item in enumerate(pending):
            length = min(CHUNK_SIZE, len(item["true_suffix"]))
            true[row, :length] = item["true_suffix"][:length]
            valid[row, :length] = True
        true_batches.append(true)
        valid_batches.append(valid)
        metadata.extend(item["metadata"] for item in pending)
        pending = []

    for episode_number, episode in enumerate(episodes, start=1):
        episode_index = int(episode["episode_index"])
        episode_length = int(episode["length"])
        starts = sample_starts(episode_length)
        states, actions = load_episode_arrays(dataset_root, episode_index)
        if states.shape[0] != episode_length or actions.shape[0] != episode_length:
            raise RuntimeError(f"Episode {episode_index} length mismatch")
        task_name = str(episode["tasks"][0])
        task_index = next(index for index, name in tasks.items() if name == task_name)
        agent_path = dataset_root / "videos" / "chunk-000" / "observation.images.image" / f"episode_{episode_index:06d}.mp4"
        wrist_path = dataset_root / "videos" / "chunk-000" / "observation.images.wrist_image" / f"episode_{episode_index:06d}.mp4"
        agent_frames = decode_selected_frames(agent_path, starts)
        wrist_frames = decode_selected_frames(wrist_path, starts)
        for start in starts:
            pending.append(
                {
                    "state": states[start],
                    "agent_image": agent_frames[start],
                    "wrist_image": wrist_frames[start],
                    "true_suffix": actions[start:],
                    "metadata": {
                        "episode_index": episode_index,
                        "task_index": task_index,
                        "task_name": task_name,
                        "frame_index": start,
                        "episode_length": episode_length,
                        "progress": start / episode_length,
                        "phase": phase_label(start, episode_length),
                    },
                }
            )
            episode_points[episode_index] += 1
            task_points[task_name] += 1
        task_episodes[task_name] += 1
        if len(pending) >= batch_size:
            flush()
        if episode_number == 1 or episode_number % 20 == 0 or episode_number == len(episodes):
            print(
                f"processed episodes {episode_number}/{len(episodes)}; "
                f"sampled points {sum(episode_points.values())}",
                flush=True,
            )
    flush()
    predicted = np.concatenate(predicted_batches, axis=0)
    true = np.concatenate(true_batches, axis=0)
    valid = np.concatenate(valid_batches, axis=0)
    if predicted.shape != true.shape or valid.shape != predicted.shape[:2] or len(metadata) != predicted.shape[0]:
        raise RuntimeError("Prediction, target, validity, and metadata shapes do not match")
    coverage = {
        "episodes_analyzed": len(episodes),
        "observation_points_analyzed": int(predicted.shape[0]),
        "predicted_action_steps": int(predicted.shape[0] * CHUNK_SIZE),
        "task_distribution_points": dict(sorted(task_points.items())),
        "task_distribution_episodes": dict(sorted(task_episodes.items())),
        "episode_points_min": min(episode_points.values()),
        "episode_points_max": max(episode_points.values()),
        "episode_points_mean": float(np.mean(list(episode_points.values()))),
    }
    return predicted, true, valid, metadata, coverage


def error_arrays(predicted: np.ndarray, true: np.ndarray, valid: np.ndarray, action_std: np.ndarray) -> dict[str, np.ndarray]:
    difference = predicted - true
    with np.errstate(invalid="ignore", divide="ignore"):
        translation = np.sqrt(np.mean((difference[:, :, :3] / action_std[:3]) ** 2, axis=2))
        rotation = np.sqrt(np.mean((difference[:, :, 3:6] / action_std[3:6]) ** 2, axis=2))
        gripper = np.abs(difference[:, :, 6]) / float(action_std[6])
    target_binary = np.where(true[:, :, 6] >= 0.0, 1.0, -1.0)
    predicted_binary = np.where(predicted[:, :, 6] >= 0.0, 1.0, -1.0)
    mismatch = (predicted_binary != target_binary).astype(np.float32)
    result = {
        "arm_translation_normalized_rms": translation,
        "arm_rotation_normalized_rms": rotation,
        "gripper_absolute_error_normalized": gripper,
        "gripper_binary_mismatch_rate": mismatch,
    }
    for value in result.values():
        value[~valid] = np.nan
    return result


def curve_slope(curve: np.ndarray) -> float:
    finite = np.isfinite(curve)
    x = np.arange(len(curve), dtype=np.float64)[finite]
    y = curve[finite].astype(np.float64)
    if len(x) < 2:
        return float("nan")
    x -= x.mean()
    y -= y.mean()
    return float(np.sum(x * y) / np.sum(x * x))


def curve_auc(curve: np.ndarray) -> float:
    finite = np.isfinite(curve)
    x = np.arange(len(curve), dtype=np.float64)[finite]
    y = curve[finite].astype(np.float64)
    if len(x) < 2:
        return float("nan")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def curve_scalars(curve: np.ndarray) -> dict[str, float]:
    first = float(np.nanmean(curve[:10]))
    last = float(np.nanmean(curve[-10:]))
    return {
        "auc_mean_over_common_horizon": curve_auc(curve),
        "linear_slope_per_step": curve_slope(curve),
        "last10_minus_first10": last - first,
        "first10_mean": first,
        "last10_mean": last,
    }


def bootstrap_curve(
    episode_curves: np.ndarray,
    draws: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    bootstrap = np.full((draws.shape[0], episode_curves.shape[1]), np.nan, dtype=np.float32)
    for k in range(episode_curves.shape[1]):
        values = episode_curves[:, k]
        bootstrap[:, k] = np.nanmean(values[draws], axis=1)
    return np.nanpercentile(bootstrap, 2.5, axis=0), np.nanpercentile(bootstrap, 97.5, axis=0)


def phase_episode_curves(
    error: np.ndarray,
    metadata: list[dict[str, Any]],
    phase: str,
    common_horizon: int,
) -> tuple[np.ndarray, list[int], np.ndarray]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        if row["phase"] == phase:
            groups[int(row["episode_index"])].append(index)
    episode_ids = sorted(groups)
    curves = np.full((len(episode_ids), common_horizon), np.nan, dtype=np.float32)
    for row, episode_id in enumerate(episode_ids):
        curves[row] = np.nanmean(error[groups[episode_id], :common_horizon], axis=0)
    return curves, episode_ids, np.asarray([len(groups[e]) for e in episode_ids], dtype=np.int64)


def summarize_phase(
    error: np.ndarray,
    metadata: list[dict[str, Any]],
    phase: str,
    common_horizon: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], np.ndarray, list[int]]:
    mask = np.asarray([row["phase"] == phase for row in metadata], dtype=bool)
    indices = np.flatnonzero(mask)
    episode_curves, episode_ids, points_per_episode = phase_episode_curves(error, metadata, phase, common_horizon)
    # Use the same episode-balanced estimator for the point curve and the
    # bootstrap interval; otherwise episodes with more sampled starts would
    # receive extra weight in the displayed curve.
    curve = np.nanmean(episode_curves, axis=0)
    draws = rng.integers(0, len(episode_ids), size=(BOOTSTRAP_DRAWS, len(episode_ids)))
    low, high = bootstrap_curve(episode_curves, draws)
    scalar = curve_scalars(curve)
    bootstrap_auc = np.asarray([curve_auc(np.nanmean(episode_curves[draw], axis=0)) for draw in draws])
    bootstrap_slope = np.asarray([curve_slope(np.nanmean(episode_curves[draw], axis=0)) for draw in draws])
    result = {
        **scalar,
        "status": "measured",
        "n_observation_points": int(len(indices)),
        "n_episodes": int(len(episode_ids)),
        "curve": curve.astype(float).tolist(),
        "bootstrap_95ci_low": low.astype(float).tolist(),
        "bootstrap_95ci_high": high.astype(float).tolist(),
        "valid_observation_points_by_k": np.sum(np.isfinite(error[indices, :common_horizon]), axis=0).astype(int).tolist(),
        "valid_episodes_by_k": np.sum(np.isfinite(episode_curves), axis=0).astype(int).tolist(),
        "auc_bootstrap_95ci": [float(np.percentile(bootstrap_auc, 2.5)), float(np.percentile(bootstrap_auc, 97.5))],
        "slope_bootstrap_95ci": [float(np.percentile(bootstrap_slope, 2.5)), float(np.percentile(bootstrap_slope, 97.5))],
        "points_per_episode_min": int(points_per_episode.min()),
        "points_per_episode_max": int(points_per_episode.max()),
        "points_per_episode_mean": float(points_per_episode.mean()),
    }
    return result, episode_curves, episode_ids


def pair_effect(
    left: np.ndarray,
    right: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float | list[float]]:
    difference = left - right
    finite = np.isfinite(difference)
    difference = difference[finite]
    draws = rng.integers(0, len(difference), size=(BOOTSTRAP_DRAWS, len(difference)))
    means = difference[draws].mean(axis=1)
    mean = float(difference.mean())
    standard_deviation = float(difference.std(ddof=1)) if len(difference) > 1 else float("nan")
    return {
        "mean_auc_difference": mean,
        "bootstrap_95ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "cohen_d_paired": mean / standard_deviation if standard_deviation > 0 else float("nan"),
        "n_episodes": int(len(difference)),
    }


def phase_auc_by_task(
    error: np.ndarray,
    metadata: list[dict[str, Any]],
    common_horizon: int,
) -> dict[str, dict[str, float]]:
    task_phase_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        task_phase_indices[(row["task_name"], row["phase"])].append(index)
    tasks = sorted({row["task_name"] for row in metadata})
    result: dict[str, dict[str, float]] = {}
    for task in tasks:
        result[task] = {}
        for phase in PHASES:
            curve = np.nanmean(error[task_phase_indices[(task, phase)], :common_horizon], axis=0)
            result[task][phase] = curve_auc(curve)
    return result


def make_plots(
    output_dir: Path,
    phase_metrics: dict[str, dict[str, dict[str, Any]]],
    task_effects: dict[str, dict[str, dict[str, float]]],
    phase_effects: dict[str, dict[str, dict[str, Any]]],
    common_horizon: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"early": "tab:green", "middle": "tab:orange", "late": "tab:red"}
    titles = {
        "arm_translation_normalized_rms": "Arm translation (normalized RMS)",
        "arm_rotation_normalized_rms": "Arm rotation (normalized RMS)",
        "gripper_absolute_error_normalized": "Gripper absolute error (normalized)",
        "gripper_binary_mismatch_rate": "Gripper binary mismatch",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, metric in zip(axes.flat, METRICS, strict=True):
        for phase in PHASES:
            summary = phase_metrics[metric][phase]
            x = np.arange(common_horizon)
            curve = np.asarray(summary["curve"])
            low = np.asarray(summary["bootstrap_95ci_low"])
            high = np.asarray(summary["bootstrap_95ci_high"])
            ax.plot(x, curve, color=colors[phase], linewidth=2, label=f"{phase} (n={summary['n_episodes']})")
            ax.fill_between(x, low, high, color=colors[phase], alpha=0.16, linewidth=0)
        ax.axvline(common_horizon - 1, color="0.35", linestyle="--", linewidth=1, label="common limit")
        ax.set_title(titles[metric])
        ax.set_ylabel("error")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("future step k")
    fig.suptitle("Frozen ACT error by normalized episode-progress phase\n95% bands: episode bootstrap")
    fig.tight_layout()
    fig.savefig(output_dir / "phase_error_curves.png", dpi=180)
    plt.close(fig)

    task_names = list(task_effects)
    labels = [name.replace("pick up the ", "").replace(" and place it in the basket", "") for name in task_names]
    matrix = np.asarray(
        [[task_effects[task][metric]["late_minus_early_auc"] for metric in METRICS] for task in task_names],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    limit = float(np.nanmax(np.abs(matrix)))
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xticks(np.arange(len(METRICS)), [titles[m] for m in METRICS], rotation=20, ha="right")
    ax.set_title("Task-wise late minus early common-horizon AUC")
    fig.colorbar(image, ax=ax, label="late - early AUC")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row, column]:+.2f}", ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_dir / "task_phase_effect_heatmap.png", dpi=180)
    plt.close(fig)

    pairs = ("early_vs_middle", "early_vs_late", "middle_vs_late")
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(METRICS))
    width = 0.24
    for offset, pair in zip((-width, 0, width), pairs, strict=True):
        values = np.asarray([phase_effects[metric][pair]["cohen_d_paired"] for metric in METRICS])
        ax.bar(x + offset, values, width=width, label=pair.replace("_", " "))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [titles[m] for m in METRICS], rotation=18, ha="right")
    ax.set_ylabel("paired Cohen's d")
    ax.set_title("Phase effect sizes on common-horizon AUC")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "phase_effect_sizes.png", dpi=180)
    plt.close(fig)


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

    torch.manual_seed(BOOTSTRAP_SEED)
    np.random.seed(BOOTSTRAP_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(BOOTSTRAP_SEED)
    runtime_config = {
        "task_suite": "libero_object",
        "task_id": 0,
        "obs_type": "pixels_agent_pos",
        "camera_name": "agentview_image,robot0_eye_in_hand_image",
        "camera_name_mapping": {"agentview_image": "image", "robot0_eye_in_hand_image": "wrist_image"},
        "observation_width": 256,
        "observation_height": 256,
        "control_freq": 20,
        "init_states": True,
        "hard_reset": True,
        "control_mode": "relative",
        "device": args.device,
    }
    policy, policy_preprocessor, policy_postprocessor, env_preprocessor, env_postprocessor = load_policy_and_processors(
        runtime_config, checkpoint
    )
    if policy.config.temporal_ensemble_coeff is not None:
        raise RuntimeError("Temporal ensembling must be disabled for this audit")
    if int(policy.config.chunk_size) != CHUNK_SIZE:
        raise RuntimeError(f"Checkpoint chunk size is {policy.config.chunk_size}, expected {CHUNK_SIZE}")
    action_std = np.asarray(load_action_normalization(checkpoint)["std"], dtype=np.float32)
    if action_std.shape != (7,) or np.any(action_std <= 0):
        raise RuntimeError("Invalid action normalization statistics")

    predicted, true, valid, metadata, coverage = infer_samples(
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
    errors = error_arrays(predicted, true, valid, action_std)
    horizon = common_horizon(episodes)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    phase_metrics: dict[str, dict[str, dict[str, Any]]] = {metric: {} for metric in METRICS}
    phase_episode_auc: dict[str, dict[str, np.ndarray]] = {metric: {} for metric in METRICS}
    phase_episode_ids: dict[str, list[int]] = {}
    for phase in PHASES:
        phase_episode_ids[phase] = []
    for metric in METRICS:
        for phase in PHASES:
            summary, episode_curves, episode_ids = summarize_phase(errors[metric], metadata, phase, horizon, rng)
            phase_metrics[metric][phase] = summary
            phase_episode_auc[metric][phase] = np.asarray([curve_auc(curve) for curve in episode_curves])
            phase_episode_ids[phase] = episode_ids

    phase_effects: dict[str, dict[str, dict[str, Any]]] = {metric: {} for metric in METRICS}
    for metric in METRICS:
        values = {phase: phase_episode_auc[metric][phase] for phase in PHASES}
        phase_effects[metric]["early_vs_middle"] = pair_effect(values["early"], values["middle"], rng)
        phase_effects[metric]["early_vs_late"] = pair_effect(values["early"], values["late"], rng)
        phase_effects[metric]["middle_vs_late"] = pair_effect(values["middle"], values["late"], rng)

    task_auc: dict[str, dict[str, dict[str, float]]] = {}
    task_phase_values: dict[str, dict[str, dict[str, float]]] = {}
    for metric in METRICS:
        task_phase_values[metric] = phase_auc_by_task(errors[metric], metadata, horizon)
    task_names = sorted(task_phase_values[METRICS[0]])
    task_effects: dict[str, dict[str, dict[str, float]]] = {}
    for task in task_names:
        task_effects[task] = {}
        for metric in METRICS:
            values = task_phase_values[metric][task]
            task_effects[task][metric] = {
                "early_auc": values["early"],
                "middle_auc": values["middle"],
                "late_auc": values["late"],
                "late_minus_early_auc": values["late"] - values["early"],
            }

    summary = {
        "hypothesis": "For the same action group, temporal predictability changes across normalized episode-progress phases.",
        "protocol": {
            "sampling": {
                "episodes": "all 454 episodes in ascending episode_index",
                "fixed_interval_frames": SAMPLE_INTERVAL,
                "additional_starts": "ceil(L/3) and ceil(2L/3) per episode",
                "full_chunk_inference": "always predicts 100 actions",
                "target_comparison": "right-censored demonstrated suffix at episode end",
                "random_seed": None,
            },
            "inference": {
                "frozen": True,
                "training": False,
                "temporal_ensemble_coeff": None,
                "policy_query": "predict_action_chunk",
                "predicted_shape": list(predicted.shape),
                "target_padded_shape": list(true.shape),
                "device": args.device,
                "inference_points": int(predicted.shape[0]),
            },
            "phase_definition": {
                "progress": "frame_index / episode_length",
                "early": "0 <= progress < 1/3",
                "middle": "1/3 <= progress < 2/3",
                "late": "2/3 <= progress < 1",
            },
            "common_comparison": {
                "horizon_steps": horizon,
                "future_steps": f"k=0..{horizon - 1}",
                "reason": "minimum demonstrated suffix at the explicit late boundary across episodes",
            },
            "uncertainty": "episode bootstrap 95% CIs, 2000 draws, seed 20260820",
        },
        "dataset": {
            "repo_id": "DorayakiLin/libero_object_25_08_23_lerobotv2.1",
            "root": str(dataset_root),
            "revision": "cbf7122bbdbaa0c50517a6a4b2ae663d0e96e51a",
            "total_episodes": int(info["total_episodes"]),
            "total_frames": int(info["total_frames"]),
            "total_tasks": int(info["total_tasks"]),
            "action_dim": 7,
            "state_dim": 8,
        },
        "checkpoint": {
            "root": str(checkpoint),
            "chunk_size": int(policy.config.chunk_size),
            "action_dim": int(policy.config.output_features["action"].shape[0]),
            "action_std": action_std.astype(float).tolist(),
        },
        "groups": {"arm_translation": "action[0:3]", "arm_rotation": "action[3:6]", "gripper": "action[6]"},
        "metrics": {
            "arm_translation_normalized_rms": "sqrt(mean((prediction-target)^2 / action_std[0:3]^2))",
            "arm_rotation_normalized_rms": "sqrt(mean((prediction-target)^2 / action_std[3:6]^2))",
            "gripper_absolute_error_normalized": "abs(prediction-target) / action_std[6]",
            "gripper_binary_mismatch_rate": "target and prediction thresholded at zero; target is expected {-1,+1}",
            "effect_size": "paired Cohen's d on per-episode common-horizon AUC differences",
        },
        "coverage": coverage,
        "phase_metrics": phase_metrics,
        "phase_effects": phase_effects,
        "task_effects": task_effects,
        "artifacts": {
            "script": str(output_dir / "phase_audit.py"),
            "phase_curves": str(output_dir / "phase_error_curves.png"),
            "task_heatmap": str(output_dir / "task_phase_effect_heatmap.png"),
            "effect_sizes": str(output_dir / "phase_effect_sizes.png"),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    make_plots(output_dir, phase_metrics, task_effects, phase_effects, horizon)
    print(json.dumps(coverage, indent=2))
    print(f"common comparison horizon: k=0..{horizon - 1}")
    print(f"wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
