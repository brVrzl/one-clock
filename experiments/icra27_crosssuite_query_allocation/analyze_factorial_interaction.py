#!/usr/bin/env python3
"""Post-hoc robustness analysis for the frozen 140-block ACT factorial."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


METHODS = ("FRESH", "FO20", "REVERSE20", "FULL_OLD20")
ALIASES = {
    "FRESH": "A0G0",
    "FO20": "A0G20",
    "REVERSE20": "A20G0",
    "FULL_OLD20": "A20G20",
}
BOOTSTRAP_DRAWS = 20_000
BLOCK_SEED = 27_501
CLUSTER_SEED = 27_502


def load_historical_module(repo: Path):
    path = repo / "experiments/cross_suite_confirmation/analyze_confirmation.py"
    spec = importlib.util.spec_from_file_location("cross_suite_confirmation_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def logit(p: np.ndarray | float) -> np.ndarray | float:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.asarray(p) / (1.0 - np.asarray(p)))


def interaction_rd(cell_means: np.ndarray) -> np.ndarray:
    # Canonical order: A0G0, A0G20, A20G0, A20G20.
    return cell_means[..., 3] - cell_means[..., 2] - cell_means[..., 1] + cell_means[..., 0]


def interaction_log_odds(cell_means: np.ndarray) -> np.ndarray:
    return logit(cell_means[..., 3]) - logit(cell_means[..., 2]) - logit(cell_means[..., 1]) + logit(cell_means[..., 0])


def percentile_ci(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "interaction_robustness")
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    protocol_path = repo / "experiments/cross_suite_confirmation/protocol.json"
    results_root = repo / "experiments/cross_suite_confirmation/results"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    historical = load_historical_module(repo)
    outcomes, _, validation = historical.load_results(protocol, results_root)
    tasks = [task for task in protocol["cohort"]["tasks"] if task["role"] == "primary_unseen_to_executor_development"]
    if len(tasks) != 10:
        raise AssertionError(f"expected 10 frozen primary tasks, found {len(tasks)}")

    blocks = [(task["suite"], int(task["task_id"]), state) for task in tasks for state in range(14)]
    y = np.asarray([[outcomes[block + (method,)] for method in METHODS] for block in blocks], dtype=float)
    if y.shape != (140, 4):
        raise AssertionError(y.shape)
    means = y.mean(axis=0)
    expected_counts = [77, 83, 38, 66]
    if y.sum(axis=0).astype(int).tolist() != expected_counts:
        raise AssertionError(f"frozen-count regression mismatch: {y.sum(axis=0).tolist()}")

    task_arrays = []
    task_rows = []
    for task in tasks:
        mask = np.asarray([suite == task["suite"] and task_id == int(task["task_id"]) for suite, task_id, _ in blocks])
        task_y = y[mask]
        task_arrays.append(task_y)
        task_means = task_y.mean(axis=0)
        row = {
            "task": f"{task['suite']}:task{int(task['task_id'])}",
            "blocks": int(task_y.shape[0]),
            "rd_interaction": float(interaction_rd(task_means)),
        }
        for i, method in enumerate(METHODS):
            row[f"{ALIASES[method]}_successes"] = int(task_y[:, i].sum())
            row[f"{ALIASES[method]}_rate"] = float(task_means[i])
        task_rows.append(row)
    task_effects = np.asarray([row["rd_interaction"] for row in task_rows])

    rng = np.random.default_rng(BLOCK_SEED)
    block_indices = rng.integers(0, len(y), size=(BOOTSTRAP_DRAWS, len(y)))
    block_means = y[block_indices].mean(axis=1)
    block_rd = interaction_rd(block_means)
    block_log_odds = interaction_log_odds(block_means)

    rng = np.random.default_rng(CLUSTER_SEED)
    cluster_indices = rng.integers(0, len(tasks), size=(BOOTSTRAP_DRAWS, len(tasks)))
    cluster_means = np.empty((BOOTSTRAP_DRAWS, 4), dtype=float)
    for draw, indices in enumerate(cluster_indices):
        cluster_means[draw] = np.concatenate([task_arrays[i] for i in indices], axis=0).mean(axis=0)
    cluster_rd = interaction_rd(cluster_means)
    cluster_log_odds = interaction_log_odds(cluster_means)

    sign_flip_means = np.asarray([
        np.mean(task_effects * np.asarray(signs, dtype=float))
        for signs in itertools.product((-1.0, 1.0), repeat=len(task_effects))
    ])
    observed_rd = float(interaction_rd(means))
    sign_flip_p = float(np.mean(np.abs(sign_flip_means) >= abs(observed_rd) - 1e-15))

    task_se = float(np.std(task_effects, ddof=1) / np.sqrt(len(task_effects)))
    tcrit = float(student_t.ppf(0.975, df=len(task_effects) - 1))
    rd_t_ci = [observed_rd - tcrit * task_se, observed_rd + tcrit * task_se]

    observed_log_odds = float(interaction_log_odds(means))
    loo_log_odds = []
    for i in range(len(tasks)):
        kept = np.concatenate([task_arrays[j] for j in range(len(tasks)) if j != i], axis=0)
        loo_log_odds.append(float(interaction_log_odds(kept.mean(axis=0))))
    loo_log_odds_arr = np.asarray(loo_log_odds)
    loo_bar = float(loo_log_odds_arr.mean())
    jackknife_se = float(np.sqrt((len(tasks) - 1) / len(tasks) * np.sum((loo_log_odds_arr - loo_bar) ** 2)))
    log_odds_jackknife_t_ci = [observed_log_odds - tcrit * jackknife_se, observed_log_odds + tcrit * jackknife_se]

    result = {
        "analysis_role": "POST_HOC_SUPPORTING_INTERACTION",
        "source_protocol": str(protocol_path),
        "source_results_root": str(results_root),
        "validation": validation,
        "task_clusters": len(tasks),
        "blocks": len(y),
        "canonical_cell_order": [ALIASES[m] for m in METHODS],
        "canonical_rd_formula": "p(A20G20) - p(A20G0) - p(A0G20) + p(A0G0)",
        "canonical_log_odds_formula": "logit(p(A20G20)) - logit(p(A20G0)) - logit(p(A0G20)) + logit(p(A0G0))",
        "cell_successes": {ALIASES[m]: int(y[:, i].sum()) for i, m in enumerate(METHODS)},
        "cell_rates": {ALIASES[m]: float(means[i]) for i, m in enumerate(METHODS)},
        "risk_difference": {
            "estimate": observed_rd,
            "percentage_points": 100.0 * observed_rd,
            "paired_block_bootstrap_ci": percentile_ci(block_rd),
            "task_cluster_bootstrap_ci": percentile_ci(cluster_rd),
            "task_t_interval": [float(x) for x in rd_t_ci],
            "task_sign_flip_test": {
                "null": "task-level interaction effects are sign-exchangeable around zero",
                "enumerations": int(len(sign_flip_means)),
                "two_sided_p": sign_flip_p,
            },
        },
        "log_odds": {
            "estimate": observed_log_odds,
            "interaction_odds_ratio": float(np.exp(observed_log_odds)),
            "paired_block_bootstrap_ci": percentile_ci(block_log_odds),
            "paired_block_bootstrap_invalid_draws": int(np.count_nonzero(~np.isfinite(block_log_odds))),
            "task_cluster_bootstrap_ci": percentile_ci(cluster_log_odds),
            "task_cluster_bootstrap_invalid_draws": int(np.count_nonzero(~np.isfinite(cluster_log_odds))),
            "delete_one_task_jackknife_t_interval": [float(x) for x in log_odds_jackknife_t_ci],
            "delete_one_task_estimates": loo_log_odds,
        },
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "block_seed": BLOCK_SEED, "task_cluster_seed": CLUSTER_SEED},
        "task_rows": task_rows,
        "interpretation_guard": "Post-hoc scale-sensitivity analysis; do not promote to confirmatory evidence or use it for additive component attribution.",
    }
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    fieldnames = list(task_rows[0])
    with (output / "per_task.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(task_rows)

    lines = [
        "# Frozen 140-block factorial interaction robustness",
        "",
        "Status: **POST_HOC_SUPPORTING_INTERACTION**. The earlier 126-block Object preregistration does not transfer to this analysis.",
        "",
        "Canonical signed formula:",
        "",
        "`I_RD = p(A20G20) - p(A20G0) - p(A0G20) + p(A0G0)`",
        "",
        "The log-odds sensitivity uses the identical orientation. An unsigned value must be called the interaction magnitude.",
        "",
        "| Scale | Estimate | Paired-block bootstrap 95% CI | Task-cluster bootstrap 95% CI | Small-cluster sensitivity |",
        "|---|---:|---:|---:|---:|",
        f"| Risk difference | {100*observed_rd:.2f} pp | [{100*percentile_ci(block_rd)[0]:.2f}, {100*percentile_ci(block_rd)[1]:.2f}] pp | [{100*percentile_ci(cluster_rd)[0]:.2f}, {100*percentile_ci(cluster_rd)[1]:.2f}] pp | task-t CI [{100*rd_t_ci[0]:.2f}, {100*rd_t_ci[1]:.2f}] pp; exact task sign-flip p={sign_flip_p:.6g} |",
        f"| Log odds | {observed_log_odds:.4f} (interaction OR {np.exp(observed_log_odds):.3f}) | [{percentile_ci(block_log_odds)[0]:.4f}, {percentile_ci(block_log_odds)[1]:.4f}] | [{percentile_ci(cluster_log_odds)[0]:.4f}, {percentile_ci(cluster_log_odds)[1]:.4f}] | delete-one-task jackknife-t CI [{log_odds_jackknife_t_ci[0]:.4f}, {log_odds_jackknife_t_ci[1]:.4f}] |",
        "",
        "The exact sign-flip calculation enumerates all 1,024 sign assignments of the ten task-level risk-difference interactions and assumes task effects are sign-exchangeable under the null. The bootstrap intervals are descriptive sensitivity analyses; inference is not based solely on percentile cluster-bootstrap exclusion of zero.",
        "",
        "These interactions quantify departure from additivity on a chosen scale. They do not identify a unique arm or gripper contribution, and they do not justify path-dependent contribution percentages.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
