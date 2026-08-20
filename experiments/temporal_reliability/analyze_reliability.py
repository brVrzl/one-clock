#!/usr/bin/env python3
"""Analyze empirical group-wise temporal reliability targets without training.

The input is the compact offline target dataset made by construct_dataset.py.
The reported curves are empirical frequencies of a right-censored prefix-valid
event, not forecasts from a learned reliability model.  Consequently this audit
can test target stability and shape, but calibration of an estimator remains a
future held-out evaluation.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
GROUPS = ("arm", "gripper")
PHASES = ("early", "middle", "late")
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260820


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "experiments/temporal_reliability/reliability_dataset.npz")
    parser.add_argument("--metadata", type=Path, default=ROOT / "experiments/temporal_reliability/metadata.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/temporal_reliability/dataset_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/temporal_reliability")
    return parser.parse_args()


def load_metadata(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def common_phase_horizon(metadata: list[dict[str, Any]]) -> int:
    """Use the same predeclared 38-step physical-phase comparison as Gate-2A."""

    lengths = {int(row["episode_index"]): int(row["episode_length"]) for row in metadata}
    return min(length - int(np.ceil(2 * length / 3)) for length in lengths.values())


def episode_balanced_kaplan_meier_components(
    pointwise: np.ndarray,
    observed: np.ndarray,
    episodes: np.ndarray,
    selected: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return episode-balanced discrete survival risk/event components.

    A direct mean of ``Y_g(t,k)`` changes its sample composition as suffixes
    are right-censored and can therefore increase with ``k`` despite each
    individual label being monotone. The Kaplan--Meier product-limit estimate
    avoids that artifact while retaining every observed target. Equal total
    weight is assigned to each episode before aggregation.
    """

    selected_ids = np.unique(episodes[selected])
    risks = np.zeros((len(selected_ids), horizon), dtype=np.float64)
    events = np.zeros((len(selected_ids), horizon), dtype=np.float64)
    for row, episode_id in enumerate(selected_ids):
        rows = np.flatnonzero(selected & (episodes == episode_id))
        weight = 1.0 / len(rows)
        alive = np.ones(len(rows), dtype=bool)
        for k in range(horizon):
            eligible = alive & observed[rows, k]
            risks[row, k] = weight * eligible.sum()
            failed = eligible & ~pointwise[rows, k]
            events[row, k] = weight * failed.sum()
            alive[failed] = False
    return risks, events, selected_ids


def survival_from_components(risks: np.ndarray, events: np.ndarray) -> np.ndarray:
    total_risk = risks.sum(axis=0)
    total_events = events.sum(axis=0)
    if np.any(total_events > total_risk + 1e-10):
        raise RuntimeError("Invalid risk-set accounting for censored survival target")
    curve = np.full(len(total_risk), np.nan, dtype=np.float64)
    current = 1.0
    for k, (risk, event) in enumerate(zip(total_risk, total_events, strict=True)):
        if risk <= 0:
            break
        current *= 1.0 - event / risk
        curve[k] = current
    return curve


def auc(curve: np.ndarray) -> float:
    finite = np.isfinite(curve)
    x = np.flatnonzero(finite)
    if len(x) < 2:
        return float("nan")
    return float(np.trapezoid(curve[finite], x) / (x[-1] - x[0]))


def curve_shape(curve: np.ndarray) -> dict[str, Any]:
    difference = np.diff(curve)
    finite = difference[np.isfinite(difference)]
    return {
        "auc_mean_reliability": auc(curve),
        "reliability_at_k0": float(curve[0]),
        "reliability_at_final_k": float(curve[np.flatnonzero(np.isfinite(curve))[-1]]),
        "mean_absolute_adjacent_change": float(np.mean(np.abs(finite))),
        "max_absolute_adjacent_change": float(np.max(np.abs(finite))),
        "monotonicity_violations_increasing": int(np.sum(finite > 1e-10)),
        "is_nonincreasing": bool(np.all(finite <= 1e-10)),
    }


def summarize_curve(
    survival: np.ndarray,
    pointwise: np.ndarray,
    observed: np.ndarray,
    episodes: np.ndarray,
    selected: np.ndarray,
    horizon: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if not np.array_equal(survival[observed], np.logical_and.accumulate(pointwise, axis=1)[observed]):
        raise RuntimeError("Stored survival labels are not prefix products of pointwise validity")
    risks, events, episode_ids = episode_balanced_kaplan_meier_components(pointwise, observed, episodes, selected, horizon)
    curve = survival_from_components(risks, events)
    draws = rng.integers(0, len(episode_ids), size=(BOOTSTRAP_DRAWS, len(episode_ids)))
    multiplicities = np.eye(len(episode_ids), dtype=np.int16)[draws].sum(axis=1)
    bootstrap_risk = multiplicities @ risks
    bootstrap_events = multiplicities @ events
    factors = np.full(bootstrap_risk.shape, np.nan, dtype=np.float64)
    np.divide(bootstrap_events, bootstrap_risk, out=factors, where=bootstrap_risk > 0)
    bootstrap = np.cumprod(1.0 - factors, axis=1)
    support_horizon = int(np.count_nonzero(np.isfinite(curve)))
    if support_horizon < 2:
        raise RuntimeError("Fewer than two offsets have a non-empty survival risk set")
    bootstrap = bootstrap[:, :support_horizon]
    lower, upper = np.nanpercentile(bootstrap, [2.5, 97.5], axis=0)
    scalar_bootstrap = np.asarray([auc(row) for row in bootstrap])
    counts = np.asarray([(selected & observed[:, k]).sum() for k in range(support_horizon)], dtype=np.int32)
    curve = curve[:support_horizon]
    return {
        **curve_shape(curve),
        "support_horizon_steps": support_horizon,
        "n_episodes": int(len(episode_ids)),
        "n_observation_points": int(selected.sum()),
        "valid_observation_points_by_k": counts.tolist(),
        "curve": curve.astype(float).tolist(),
        "bootstrap_95ci_low": lower.astype(float).tolist(),
        "bootstrap_95ci_high": upper.astype(float).tolist(),
        "auc_bootstrap_95ci": [float(np.percentile(scalar_bootstrap, 2.5)), float(np.percentile(scalar_bootstrap, 97.5))],
    }


def pointwise_shape(
    pointwise: np.ndarray,
    observed: np.ndarray,
    selected: np.ndarray,
    horizon: int,
) -> dict[str, Any]:
    curve = np.asarray(
        [pointwise[selected & observed[:, k], k].mean() if np.any(selected & observed[:, k]) else np.nan for k in range(horizon)],
        dtype=np.float64,
    )
    difference = np.diff(curve)
    finite = difference[np.isfinite(difference)]
    return {
        "curve": curve.astype(float).tolist(),
        "positive_adjacent_differences": int(np.sum(finite > 1e-10)),
        "negative_adjacent_differences": int(np.sum(finite < -1e-10)),
        "is_nonincreasing": bool(np.all(finite <= 1e-10)),
        "mean_absolute_adjacent_change": float(np.mean(np.abs(finite))),
    }


def threshold_sensitivity(data: Any, observed: np.ndarray, episodes: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    """Show that tolerance is part of the proxy definition, not a tuned horizon."""

    result: dict[str, Any] = {}
    for tolerance in (0.5, 1.0, 1.5):
        arm_pointwise = (
            (data["arm_translation_error"] <= tolerance)
            & (data["arm_rotation_error"] <= tolerance)
            & observed
        )
        gripper_pointwise = (
            (data["gripper_normalized_absolute_error"] <= tolerance)
            & data["gripper_sign_match"].astype(bool)
            & observed
        )
        result[str(tolerance)] = {}
        for group, pointwise in (("arm", arm_pointwise), ("gripper", gripper_pointwise)):
            survival = np.logical_and.accumulate(pointwise, axis=1) & observed
            summary = summarize_curve(survival, pointwise, observed, episodes, np.ones(len(episodes), dtype=bool), survival.shape[1], rng)
            result[str(tolerance)][group] = {
                "auc_mean_reliability": summary["auc_mean_reliability"],
                "reliability_at_k0": summary["reliability_at_k0"],
                "reliability_at_final_k": summary["reliability_at_final_k"],
            }
    return result


def make_plots(output_dir: Path, overall: dict[str, Any], phase: dict[str, Any], tasks: dict[str, Any], common_horizon: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"arm": "tab:blue", "gripper": "tab:purple"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, group in zip(axes, GROUPS, strict=True):
        item = overall[group]
        x = np.arange(len(item["curve"]))
        ax.plot(x, item["curve"], color=colors[group], linewidth=2.3, label="prefix-survival reliability")
        ax.fill_between(x, item["bootstrap_95ci_low"], item["bootstrap_95ci_high"], color=colors[group], alpha=0.2)
        ax.plot(x, overall["pointwise"][group]["curve"], color="0.25", linewidth=1.25, linestyle="--", label="pointwise validity")
        ax.set_title(group.capitalize())
        ax.set_xlabel("future offset k")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("empirical validity probability")
    fig.suptitle("Oracle temporal reliability target — all teacher-forced samples")
    fig.tight_layout()
    fig.savefig(output_dir / "overall_reliability_curves.png", dpi=180)
    plt.close(fig)

    phase_colors = {"early": "tab:green", "middle": "tab:orange", "late": "tab:red"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, group in zip(axes, GROUPS, strict=True):
        for phase_name in PHASES:
            item = phase[group][phase_name]
            x = np.arange(len(item["curve"]))
            ax.plot(x, item["curve"], color=phase_colors[phase_name], linewidth=2, label=f"{phase_name} (n={item['n_episodes']})")
            ax.fill_between(x, item["bootstrap_95ci_low"], item["bootstrap_95ci_high"], color=phase_colors[phase_name], alpha=0.13)
        ax.set_title(group.capitalize())
        ax.set_xlabel("future offset k")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("empirical prefix-survival reliability")
    fig.suptitle("Phase-conditioned oracle reliability (common observable range)")
    fig.tight_layout()
    fig.savefig(output_dir / "phase_reliability_curves.png", dpi=180)
    plt.close(fig)

    task_names = list(tasks)
    labels = [name.replace("pick up the ", "").replace(" and place it in the basket", "") for name in task_names]
    matrix = np.asarray([[tasks[name][group]["auc_mean_reliability"] for group in GROUPS] for name in task_names])
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xticks(np.arange(len(GROUPS)), [group.capitalize() for group in GROUPS])
    ax.set_title("Task-wise oracle reliability AUC (per-task available offsets)")
    fig.colorbar(image, ax=ax, label="mean prefix-survival reliability")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", color="white" if matrix[row, column] < .55 else "black")
    fig.tight_layout()
    fig.savefig(output_dir / "task_reliability_auc_heatmap.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.input.is_file() or not args.metadata.is_file() or not args.manifest.is_file():
        raise FileNotFoundError("Run construct_dataset.py first; dataset, metadata, and manifest are required.")
    metadata = load_metadata(args.metadata)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    data = np.load(args.input, allow_pickle=False)
    count = len(metadata)
    if data["predicted_actions"].shape != (count, 100, 7) or data["observed_offsets"].shape != (count, 100):
        raise RuntimeError("Unexpected reliability dataset dimensions")
    observed = data["observed_offsets"].astype(bool)
    episodes = data["episode_index"].astype(np.int32)
    if not np.all(np.diff(observed.astype(np.int8), axis=1) <= 0):
        raise RuntimeError("Censoring masks must be contiguous observed prefixes")
    phase_codes = data["phase_code"].astype(np.int8)
    task_indices = data["task_index"].astype(np.int16)
    task_names = {int(row["task_index"]): str(row["task_name"]) for row in metadata}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    all_rows = np.ones(count, dtype=bool)

    overall: dict[str, Any] = {"pointwise": {}}
    for group in GROUPS:
        overall[group] = summarize_curve(data[f"{group}_survival_valid"].astype(bool), data[f"{group}_pointwise_valid"].astype(bool), observed, episodes, all_rows, 100, rng)
        overall["pointwise"][group] = pointwise_shape(data[f"{group}_pointwise_valid"].astype(bool), observed, all_rows, 100)

    common_horizon = common_phase_horizon(metadata)
    phase: dict[str, dict[str, Any]] = {group: {} for group in GROUPS}
    for group in GROUPS:
        for code, phase_name in enumerate(PHASES):
            selected = phase_codes == code
            phase[group][phase_name] = summarize_curve(data[f"{group}_survival_valid"].astype(bool), data[f"{group}_pointwise_valid"].astype(bool), observed, episodes, selected, common_horizon, rng)

    tasks: dict[str, dict[str, Any]] = {}
    for task_index in sorted(task_names):
        selected = task_indices == task_index
        tasks[task_names[task_index]] = {}
        for group in GROUPS:
            tasks[task_names[task_index]][group] = summarize_curve(data[f"{group}_survival_valid"].astype(bool), data[f"{group}_pointwise_valid"].astype(bool), observed, episodes, selected, 100, rng)

    sample_statistics = {}
    for group in GROUPS:
        pointwise = data[f"{group}_pointwise_valid"].astype(bool)
        survival = data[f"{group}_survival_valid"].astype(bool)
        sample_statistics[group] = {
            "pointwise_valid_fraction_over_observed_pairs": float(pointwise[observed].mean()),
            "survival_valid_fraction_over_observed_pairs": float(survival[observed].mean()),
            "samples_valid_at_k0": int(survival[:, 0].sum()),
            "samples_observed_at_k0": int(observed[:, 0].sum()),
        }

    summary = {
        "purpose": "Stage-0 self-supervised temporal reliability target audit; no reliability network or scheduler was trained or executed.",
        "target_definition": manifest["survival_target"],
        "groups": manifest["groups"],
        "dataset": manifest["dataset"],
        "checkpoint": manifest["checkpoint"],
        "coverage": manifest["coverage"],
        "phase_comparison": {"common_horizon_steps": common_horizon, "future_offsets": f"k=0..{common_horizon - 1}", "reason": "minimum demonstrated suffix after explicit late-phase boundary"},
        "sample_statistics": sample_statistics,
        "overall_reliability": overall,
        "phase_reliability": phase,
        "task_reliability": tasks,
        "threshold_sensitivity": threshold_sensitivity(data, observed, episodes, rng),
        "scientific_checks": {
            "smoothness": "Reported as mean and maximum adjacent change on each empirical prefix-survival curve.",
            "monotonicity": "Individual prefix-survival labels are nonincreasing by construction. Curves use an episode-balanced Kaplan--Meier estimate so right censoring cannot induce artificial increases; pointwise validity is reported separately to test whether raw future validity itself falls with k.",
            "calibration": "not_evaluable_without_a_probability estimator; this audit supplies empirical target frequencies and bootstrap uncertainty only. A future held-out estimator must report calibration slope/intercept, Brier score, ECE, and reliability diagrams.",
            "interpretation": "These are oracle reliability targets under demonstrated-action consistency, not ground-truth horizons, task-success probabilities, or closed-loop safety guarantees.",
        },
        "artifacts": {
            "construction_script": str(output_dir / "construct_dataset.py"),
            "analysis_script": str(output_dir / "analyze_reliability.py"),
            "dataset": str(output_dir / "reliability_dataset.npz"),
            "overall_plot": str(output_dir / "overall_reliability_curves.png"),
            "phase_plot": str(output_dir / "phase_reliability_curves.png"),
            "task_plot": str(output_dir / "task_reliability_auc_heatmap.png"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    make_plots(output_dir, overall, phase, tasks, common_horizon)
    print(json.dumps({"samples": count, "common_phase_horizon": common_horizon, **manifest["coverage"]}, indent=2))
    print(f"wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
