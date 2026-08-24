#!/usr/bin/env python3
"""Analyze multi-scale fragility and predictability from cheap heuristics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


FEATURES = (
    "phase", "object_goal_distance", "eef_object_distance",
    "action_magnitude", "action_velocity", "action_acceleration",
    "gripper_transition",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(0)
    std = x.std(0)
    std[std < 1e-8] = 1.0
    z = (x - mean) / std
    design = np.c_[np.ones(len(z)), z]
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return beta, mean, std


def ridge_predict(beta: np.ndarray, mean: np.ndarray, std: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(x)), (x - mean) / std] @ beta


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - np.sum((y - pred) ** 2) / denom) if denom > 1e-12 else float("nan")


def top_overlap(a: np.ndarray, b: np.ndarray, fraction: float = 0.2) -> float:
    k = max(1, int(np.ceil(len(a) * fraction)))
    ai = set(np.argsort(a)[-k:])
    bi = set(np.argsort(b)[-k:])
    return len(ai & bi) / k


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("experiments/counterfactual_tournament/maniskill_fragility_sweep"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.input
    output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.input / "fragility_branches.csv")
    scales = sorted({float(row["scale_m"]) for row in rows})
    weights = np.asarray([0.2, 0.3, 0.5], dtype=float)
    weights = weights / weights.sum()

    states = {}
    for row in rows:
        key = (row["task"], int(row["episode"]), int(row["timestep"]))
        state = states.setdefault(key, {
            "task": row["task"], "episode": int(row["episode"]), "timestep": int(row["timestep"]),
            "state_id": row["state_id"],
            **{feature: float(row[feature]) for feature in FEATURES},
            "scales": {scale: [] for scale in scales},
        })
        scale = float(row["scale_m"])
        if int(row["branch_valid"]):
            state["scales"][scale].append(1.0 - float(row["branch_success"]))

    state_rows = []
    for state in states.values():
        probabilities = np.asarray([mean_or_nan(state["scales"][scale]) for scale in scales])
        score = float(np.nansum(probabilities * weights))
        first_nonzero = next((scale for scale, probability in zip(scales, probabilities) if probability > 0), None)
        half_margin = next((scale for scale, probability in zip(scales, probabilities) if probability >= 0.5), None)
        state_rows.append({
            key: value for key, value in state.items() if key not in ("scales",)
        } | {
            **{f"p_fail_{i}": float(probabilities[i]) for i in range(len(scales))},
            "fragility_score": score,
            "first_nonzero_scale_m": first_nonzero if first_nonzero is not None else np.nan,
            "robustness_margin_50pct_m": half_margin if half_margin is not None else np.nan,
        })

    with (output / "fragility_state_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(state_rows[0]))
        writer.writeheader()
        writer.writerows(state_rows)

    report = {"scales_m": scales, "score_weights": weights.tolist(), "tasks": {}}
    for task in sorted({state["task"] for state in state_rows}):
        task_states = [state for state in state_rows if state["task"] == task]
        y = np.asarray([state["fragility_score"] for state in task_states])
        task_report = {
            "states": len(task_states),
            "score_mean": float(y.mean()),
            "score_std": float(y.std()),
            "score_min": float(y.min()),
            "score_max": float(y.max()),
            "curve": {
                str(scale): float(np.mean([state[f"p_fail_{i}"] for state in task_states]))
                for i, scale in enumerate(scales)
            },
        }
        correlations = {}
        for feature in FEATURES:
            x = np.asarray([state[feature] for state in task_states])
            correlations[feature] = float(spearmanr(x, y).statistic) if np.std(x) > 1e-12 else float("nan")
        task_report["spearman"] = correlations

        episodes = sorted({state["episode"] for state in task_states})
        fold_predictions = np.full(len(task_states), np.nan)
        x_all = np.asarray([[state[feature] for feature in FEATURES] for state in task_states])
        for held_out in episodes:
            train = np.asarray([i for i, state in enumerate(task_states) if state["episode"] != held_out])
            test = np.asarray([i for i, state in enumerate(task_states) if state["episode"] == held_out])
            beta, mean, std = ridge_fit(x_all[train], y[train])
            fold_predictions[test] = ridge_predict(beta, mean, std, x_all[test])
        task_report["held_out_episode_ridge"] = {
            "r2": r2(y, fold_predictions),
            "spearman": float(spearmanr(y, fold_predictions).statistic),
            "top20_overlap": top_overlap(y, fold_predictions),
        }

        strongest = max(correlations, key=lambda name: abs(correlations[name]) if np.isfinite(correlations[name]) else -1)
        heuristic_values = np.asarray([state[strongest] for state in task_states])
        if correlations[strongest] < 0:
            heuristic_values = -heuristic_values
        task_report["strongest_single_heuristic"] = {
            "feature": strongest,
            "spearman": correlations[strongest],
            "top20_overlap": top_overlap(y, heuristic_values),
        }
        k = max(1, int(np.ceil(len(task_states) * 0.2)))
        top_fragility = set(np.argsort(y)[-k:])
        top_heuristic = set(np.argsort(heuristic_values)[-k:])
        missed = sorted(top_fragility - top_heuristic, key=lambda i: y[i], reverse=True)[:5]
        task_report["high_fragility_missed_by_strongest_heuristic"] = [
            {
                "episode": task_states[i]["episode"],
                "timestep": task_states[i]["timestep"],
                "phase": task_states[i]["phase"],
                "fragility_score": task_states[i]["fragility_score"],
                "heuristic": task_states[i][strongest],
            }
            for i in missed
        ]
        report["tasks"][task] = task_report

        # Per-task scalar score and fragility curves.
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
        phases = np.asarray([state["phase"] for state in task_states])
        bins = np.linspace(0, 1, 11)
        centers = (bins[:-1] + bins[1:]) / 2
        for i, scale in enumerate(scales):
            values = np.asarray([state[f"p_fail_{i}"] for state in task_states])
            means = [values[(phases >= bins[j]) & (phases < bins[j + 1])].mean() if np.any((phases >= bins[j]) & (phases < bins[j + 1])) else np.nan for j in range(10)]
            axes[0].plot(centers, means, marker="o", label=f"{scale*1000:g} mm")
        axes[0].set(xlabel="normalized phase", ylabel="failure probability", title="fragility by phase")
        axes[0].legend(frameon=False, fontsize=8)
        axes[1].hist(y, bins=min(12, max(4, len(y) // 8)), color="#3b6fb6", alpha=0.85)
        axes[1].set(xlabel="Counterfactual Fragility Score", ylabel="states", title="score distribution")
        axes[2].scatter(np.asarray([state["object_goal_distance"] for state in task_states]), y, s=14, alpha=0.8)
        axes[2].set(xlabel="object-goal distance", ylabel="fragility score", title=f"r={correlations['object_goal_distance']:.2f}")
        fig.tight_layout()
        fig.savefig(output / f"{task.replace('-v1', '').lower()}_fragility_analysis.png", dpi=160)
        plt.close(fig)

    (output / "fragility_analysis.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(json.dumps(report, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
