#!/usr/bin/env python3
"""Audit aligned ACT chunk errors and the cheapest post-policy corrections."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from one_clock import AffineResidualCalibrator  # noqa: E402


@dataclass
class AlignedSplit:
    prediction: np.ndarray
    target: np.ndarray
    source_state: np.ndarray
    position: np.ndarray
    task: np.ndarray
    episode: np.ndarray
    boundaries: list[tuple[int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def load_split(cache: Path, dataset: Path, split: str) -> AlignedSplit:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    states: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    tasks: list[np.ndarray] = []
    episodes: list[np.ndarray] = []
    boundaries: list[tuple[int, int]] = []
    cursor = 0
    data_root = dataset / "data" / "chunk-000"
    for cache_path in sorted((cache / split).glob("episode_*.npz")):
        with np.load(cache_path, allow_pickle=False) as artifact:
            chunks = artifact["predicted_chunks"].astype(np.float64)
            episode_id = int(artifact["episode_id"])
            task_id = int(artifact["task_id"])
            source_frames = artifact["dataset_frame"].astype(np.int64)
        frame = pd.read_parquet(
            data_root / f"episode_{episode_id:06d}.parquet",
            columns=["action", "observation.state"],
        )
        expert = np.stack(frame["action"].to_numpy()).astype(np.float64)
        observation_state = np.stack(frame["observation.state"].to_numpy()).astype(np.float64)
        for source_index, source_frame in enumerate(source_frames):
            length = min(chunks.shape[1], len(expert) - int(source_frame))
            start, stop = cursor, cursor + length
            boundaries.append((start, stop))
            cursor = stop
            predictions.append(chunks[source_index, :length])
            targets.append(expert[source_frame : source_frame + length])
            states.append(np.repeat(observation_state[source_frame][None], length, axis=0))
            positions.append(np.arange(length, dtype=np.int64))
            tasks.append(np.full(length, task_id, dtype=np.int64))
            episodes.append(np.full(length, episode_id, dtype=np.int64))
    if not predictions:
        raise FileNotFoundError(f"no cache episodes found for split {split!r} under {cache}")
    return AlignedSplit(
        prediction=np.concatenate(predictions),
        target=np.concatenate(targets),
        source_state=np.concatenate(states),
        position=np.concatenate(positions),
        task=np.concatenate(tasks),
        episode=np.concatenate(episodes),
        boundaries=boundaries,
    )


def ema(split: AlignedSplit, alpha: float) -> np.ndarray:
    output = split.prediction.copy()
    for start, stop in split.boundaries:
        for index in range(start + 1, stop):
            output[index] = alpha * split.prediction[index] + (1.0 - alpha) * output[index - 1]
    return output


def previous_prediction_blend(split: AlignedSplit, alpha: float) -> np.ndarray:
    output = split.prediction.copy()
    for start, stop in split.boundaries:
        output[start + 1 : stop] = (
            alpha * split.prediction[start + 1 : stop]
            + (1.0 - alpha) * split.prediction[start : stop - 1]
        )
    return output


def task_position_residual(training: AlignedSplit, evaluation: AlignedSplit) -> np.ndarray:
    task_count = int(max(training.task.max(), evaluation.task.max()) + 1)
    residual = training.target - training.prediction
    task_mean = np.stack([residual[training.task == task].mean(axis=0) for task in range(task_count)])
    table = np.empty((task_count, 10, residual.shape[1]), dtype=np.float64)
    train_bins = np.minimum(training.position // 10, 9)
    for task in range(task_count):
        for position_bin in range(10):
            selected = (training.task == task) & (train_bins == position_bin)
            table[task, position_bin] = residual[selected].mean(axis=0) if selected.any() else task_mean[task]
    evaluation_bins = np.minimum(evaluation.position // 10, 9)
    return evaluation.prediction + table[evaluation.task, evaluation_bins]


def flat_affine_prediction(model: AffineResidualCalibrator, split: AlignedSplit) -> np.ndarray:
    phase = split.position.astype(np.float64) / 99.0
    task = np.eye(model.task_count, dtype=np.float64)[split.task]
    features = np.concatenate(
        (
            split.prediction,
            split.source_state,
            phase[:, None],
            np.square(phase)[:, None],
            np.sin(np.pi * phase)[:, None],
            task,
        ),
        axis=1,
    )
    standardized = (features - model.feature_mean) / model.feature_scale
    design = np.concatenate((standardized, np.ones((len(features), 1))), axis=1)
    return split.prediction + design @ model.weights


def gated_affine_prediction(
    split: AlignedSplit,
    always_repaired: np.ndarray,
    threshold: float,
) -> np.ndarray:
    output = split.prediction.copy()
    residual = always_repaired - split.prediction
    for start, stop in split.boundaries:
        if np.linalg.norm(residual[start:stop]) > threshold:
            output[start:stop] = always_repaired[start:stop]
    return output


def chunk_residual_norms(split: AlignedSplit, repaired: np.ndarray) -> np.ndarray:
    residual = repaired - split.prediction
    return np.asarray(
        [np.linalg.norm(residual[start:stop]) for start, stop in split.boundaries],
        dtype=np.float64,
    )


def episode_metrics(split: AlignedSplit, prediction: np.ndarray) -> pd.DataFrame:
    rows = []
    for episode_id in sorted(np.unique(split.episode)):
        selected = split.episode == episode_id
        error = prediction[selected] - split.target[selected]
        rows.append(
            {
                "episode": int(episode_id),
                "task": int(split.task[selected][0]),
                "aligned_targets": int(selected.sum()),
                "mse": float(np.mean(np.square(error))),
                "mae": float(np.mean(np.abs(error))),
                "arm_mse": float(np.mean(np.square(error[:, :6]))),
                "gripper_mse": float(np.mean(np.square(error[:, 6]))),
                "gripper_sign_accuracy": float(
                    np.mean(np.sign(prediction[selected, 6]) == np.sign(split.target[selected, 6]))
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap(
    raw: np.ndarray,
    method: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    difference = method - raw
    samples = difference[rng.integers(0, len(difference), size=(repetitions, len(difference)))].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "mean_mse_difference": float(difference.mean()),
        "relative_mse_change": float(difference.mean() / raw.mean()),
        "paired_episode_bootstrap_ci95_low": float(low),
        "paired_episode_bootstrap_ci95_high": float(high),
    }


def method_summary(
    split: AlignedSplit,
    prediction: np.ndarray,
    raw_episode: pd.DataFrame,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[dict[str, object], pd.DataFrame]:
    error = prediction - split.target
    episodes = episode_metrics(split, prediction)
    summary: dict[str, object] = {
        "aligned_targets": len(error),
        "episodes": len(episodes),
        "frame_weighted_mse": float(np.mean(np.square(error))),
        "episode_weighted_mse": float(episodes["mse"].mean()),
        "episode_weighted_mse_sd": float(episodes["mse"].std(ddof=1)),
        "frame_weighted_mae": float(np.mean(np.abs(error))),
        "gripper_sign_accuracy": float(
            np.mean(np.sign(prediction[:, 6]) == np.sign(split.target[:, 6]))
        ),
        "per_dimension_mse": np.mean(np.square(error), axis=0).tolist(),
        "mean_correction_norm_per_target": float(
            np.mean(np.linalg.norm(prediction - split.prediction, axis=1))
        ),
    }
    summary.update(
        paired_bootstrap(
            raw_episode["mse"].to_numpy(),
            episodes["mse"].to_numpy(),
            repetitions,
            rng,
        )
    )
    return summary, episodes


def error_structure(split: AlignedSplit) -> dict[str, object]:
    residual = split.target - split.prediction
    centered = residual - residual.mean(axis=0)
    _, singular, _ = np.linalg.svd(centered, full_matrices=False)
    squared_norm = np.sum(np.square(residual), axis=1)
    threshold = np.quantile(squared_norm, 0.9)
    position_bins = np.minimum(split.position // 10, 9)
    lag_products: list[np.ndarray] = []
    lag_left: list[np.ndarray] = []
    lag_right: list[np.ndarray] = []
    for start, stop in split.boundaries:
        chunk_residual = residual[start:stop]
        if len(chunk_residual) > 1:
            lag_left.append(chunk_residual[:-1])
            lag_right.append(chunk_residual[1:])
            lag_products.append(chunk_residual[:-1] * chunk_residual[1:])
    left = np.concatenate(lag_left)
    right = np.concatenate(lag_right)
    lag_correlation = [
        float(np.corrcoef(left[:, dimension], right[:, dimension])[0, 1])
        for dimension in range(residual.shape[1])
    ]
    raw_mse = float(np.mean(np.square(residual)))
    debiased_mse = float(np.mean(np.square(residual - residual.mean(axis=0))))
    return {
        "residual_mean": residual.mean(axis=0).tolist(),
        "residual_std": residual.std(axis=0).tolist(),
        "per_dimension_mse": np.mean(np.square(residual), axis=0).tolist(),
        "constant_bias_explained_mse_fraction": 1.0 - debiased_mse / raw_mse,
        "residual_pca_variance_fraction": (np.square(singular) / np.square(singular).sum()).tolist(),
        "lag1_residual_correlation_by_dimension": lag_correlation,
        "top_10_percent_target_error_share": float(squared_norm[squared_norm >= threshold].sum() / squared_norm.sum()),
        "position_bin_mse": [
            float(np.mean(np.square(residual[position_bins == position_bin])))
            for position_bin in range(10)
        ],
        "gripper_sign_mismatch_rate": float(
            np.mean(np.sign(split.prediction[:, 6]) != np.sign(split.target[:, 6]))
        ),
        "arm_error_share": float(
            np.square(residual[:, :6]).sum() / np.square(residual).sum()
        ),
        "predicted_action_magnitude_mean": float(np.linalg.norm(split.prediction, axis=1).mean()),
        "expert_action_magnitude_mean": float(np.linalg.norm(split.target, axis=1).mean()),
        "predicted_temporal_step_mean": float(
            np.mean([np.linalg.norm(np.diff(split.prediction[start:stop], axis=0), axis=1).mean() for start, stop in split.boundaries if stop - start > 1])
        ),
        "expert_temporal_step_mean": float(
            np.mean([np.linalg.norm(np.diff(split.target[start:stop], axis=0), axis=1).mean() for start, stop in split.boundaries if stop - start > 1])
        ),
    }


def git_state() -> dict[str, str | bool]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    return {
        "git_head": head,
        "git_branch": branch,
        "git_dirty": bool(status),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def make_figures(
    output_dir: Path,
    structure: dict[str, object],
    episode_table: pd.DataFrame,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    kernel_path = Path.home() / ".codex" / "skills" / "figure-style" / "kernel.py"
    spec = importlib.util.spec_from_file_location("figure_style_kernel", kernel_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load figure-style helper from {kernel_path}")
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    kernel.apply_figure_style(sizes=(9, 8, 7))

    labels = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    mean = np.asarray(structure["residual_mean"])
    std = np.asarray(structure["residual_std"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    axes[0].errorbar(range(7), mean, yerr=std, fmt="o", color="#0072B2", capsize=2)
    axes[0].axhline(0.0, color="#777777", linewidth=0.8)
    axes[0].set_xticks(range(7), labels, rotation=30, ha="right")
    axes[0].set_ylabel("Expert minus ACT action")
    axes[0].set_title("Errors are variable rather than constant bias")
    axes[0].margins(x=0.06)
    kernel.panel_letter(axes[0], "a")

    position = np.arange(10) * 10 + 5
    axes[1].plot(position, structure["position_bin_mse"], marker="o", color="#D55E00")
    axes[1].set_xlabel("Predicted chunk position")
    axes[1].set_ylabel("Action MSE")
    axes[1].set_title("Error depends on predicted chunk position")
    axes[1].margins(0.06)
    kernel.panel_letter(axes[1], "b")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"error_structure.{suffix}")
    plt.close(fig)

    order = [
        "frozen_act",
        "clip",
        "previous_blend_0.25",
        "ema_0.25",
        "task_position_residual",
        "affine_residual",
    ]
    readable = ["Frozen ACT", "Clip", "Prev. blend", "EMA", "Task-phase prior", "Affine residual"]
    values = [episode_table.loc[episode_table.method == method, "mse"].to_numpy() for method in order]
    colors = kernel.focal_palette(readable, "Affine residual", "#0072B2", other="grey")
    fig, ax = plt.subplots(figsize=(6.3, 3.0), constrained_layout=True)
    kernel.strip_with_median(ax, readable, values, colors=colors, jitter=0.1)
    ax.set_ylabel("Episode action MSE")
    ax.set_title("Affine residual calibration beats trivial chunk filters")
    ax.tick_params(axis="x", rotation=25)
    ax.margins(x=0.04, y=0.08)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"baseline_comparison.{suffix}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.time()
    training = load_split(args.cache, args.dataset, "validation")
    evaluation = load_split(args.cache, args.dataset, "test")
    task_count = int(max(training.task.max(), evaluation.task.max()) + 1)
    calibrator = AffineResidualCalibrator.fit(
        action=training.prediction,
        state=training.source_state,
        position=training.position,
        task_id=training.task,
        target=training.target,
        chunk_size=100,
        task_count=task_count,
        ridge=args.ridge,
    )
    calibrator.save(args.output_dir / "affine_residual_calibrator.npz")
    validation_affine = flat_affine_prediction(calibrator, training)
    gate_quantiles = {
        str(quantile): float(np.quantile(chunk_residual_norms(training, validation_affine), quantile))
        for quantile in (0.25, 0.5, 0.75)
    }
    evaluation_affine = flat_affine_prediction(calibrator, evaluation)

    validation_low = np.quantile(training.target, 0.005, axis=0)
    validation_high = np.quantile(training.target, 0.995, axis=0)
    methods = {
        "frozen_act": evaluation.prediction,
        "clip": np.clip(evaluation.prediction, -1.0, 1.0),
        "support_quantile_clip": np.clip(evaluation.prediction, validation_low, validation_high),
        "previous_blend_0.25": previous_prediction_blend(evaluation, 0.25),
        "previous_blend_0.5": previous_prediction_blend(evaluation, 0.5),
        "previous_blend_0.75": previous_prediction_blend(evaluation, 0.75),
        "ema_0.25": ema(evaluation, 0.25),
        "ema_0.5": ema(evaluation, 0.5),
        "ema_0.75": ema(evaluation, 0.75),
        "mean_residual": evaluation.prediction
        + (training.target - training.prediction).mean(axis=0),
        "task_position_residual": task_position_residual(training, evaluation),
        "affine_scale_0.25": evaluation.prediction
        + 0.25 * (evaluation_affine - evaluation.prediction),
        "affine_scale_0.5": evaluation.prediction
        + 0.5 * (evaluation_affine - evaluation.prediction),
        "affine_scale_0.75": evaluation.prediction
        + 0.75 * (evaluation_affine - evaluation.prediction),
        "affine_residual": evaluation_affine,
        "affine_gate_q25": gated_affine_prediction(
            evaluation, evaluation_affine, gate_quantiles["0.25"]
        ),
        "affine_gate_q50": gated_affine_prediction(
            evaluation, evaluation_affine, gate_quantiles["0.5"]
        ),
        "affine_gate_q75": gated_affine_prediction(
            evaluation, evaluation_affine, gate_quantiles["0.75"]
        ),
    }
    raw_episode = episode_metrics(evaluation, methods["frozen_act"])
    rng = np.random.default_rng(args.seed)
    summaries: dict[str, object] = {}
    episode_tables = []
    dimension_rows = []
    for name, prediction in methods.items():
        summary, episodes = method_summary(
            evaluation,
            prediction,
            raw_episode,
            args.bootstrap_repetitions,
            rng,
        )
        summaries[name] = summary
        episodes.insert(0, "method", name)
        episode_tables.append(episodes)
        for dimension, mse in enumerate(summary["per_dimension_mse"]):
            dimension_rows.append({"method": name, "dimension": dimension, "mse": mse})
    episode_table = pd.concat(episode_tables, ignore_index=True)
    episode_table.to_csv(args.output_dir / "per_episode_metrics.csv", index=False)
    pd.DataFrame(dimension_rows).to_csv(args.output_dir / "per_dimension_metrics.csv", index=False)

    structure = error_structure(evaluation)
    provenance = {
        **git_state(),
        "cache": str(args.cache.resolve()),
        "dataset": str(args.dataset.resolve()),
        "validation_episodes": int(len(np.unique(training.episode))),
        "test_episodes": int(len(np.unique(evaluation.episode))),
        "validation_aligned_targets": int(len(training.target)),
        "test_aligned_targets": int(len(evaluation.target)),
        "ridge": args.ridge,
        "validation_gate_thresholds": gate_quantiles,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "elapsed_seconds": time.time() - started,
        "evaluation_status": "exploratory_test_inspected_during_method_selection",
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"provenance": provenance, "methods": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "error_structure.json").write_text(
        json.dumps(structure, indent=2) + "\n",
        encoding="utf-8",
    )
    make_figures(args.output_dir, structure, episode_table)
    print(json.dumps({"provenance": provenance, "methods": summaries}, indent=2))


if __name__ == "__main__":
    main()
