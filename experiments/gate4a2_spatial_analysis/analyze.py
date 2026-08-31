"""Reanalyse the completed preregistered Spatial ACT rollout from logs only."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


METHODS = (
    "A_NEWEST",
    "C_ASYMMETRIC_FO20",
    "B_FULL_OLD20",
    "D_AGE_EXP_B003",
    "E_COGACT_A03",
)
DISPLAY = {
    "A_NEWEST": "Fresh",
    "C_ASYMMETRIC_FO20": "FO20",
    "B_FULL_OLD20": "FullOld20",
    "D_AGE_EXP_B003": "AGE_EXP_B003",
    "E_COGACT_A03": "COGACT_A03",
}
TASKS = list(range(10))
STATES = [1, 13, 15, 19, 21, 24, 31, 37, 40, 47]
BOOTSTRAP_DRAWS = 20_000


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def paired_ci(differences: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(BOOTSTRAP_DRAWS, len(differences)))
    draws = differences[indices].mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def cluster_ci(task_differences: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(task_differences), size=(BOOTSTRAP_DRAWS, len(task_differences)))
    draws = task_differences[indices].mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def comparison(
    outcomes: dict[tuple[int, int, str], int],
    first: str,
    second: str,
    paired_seed: int,
    cluster_seed: int,
) -> dict[str, Any]:
    keys = [(task, state) for task in TASKS for state in STATES]
    first_values = np.asarray([outcomes[(t, s, first)] for t, s in keys], dtype=np.int8)
    second_values = np.asarray([outcomes[(t, s, second)] for t, s in keys], dtype=np.int8)
    differences = first_values.astype(float) - second_values.astype(float)
    task_differences = np.asarray([
        differences[[task == task_id for task_id, _ in keys]].mean() for task in TASKS
    ])
    first_only = int(np.count_nonzero((first_values == 1) & (second_values == 0)))
    second_only = int(np.count_nonzero((first_values == 0) & (second_values == 1)))
    discordant = first_only + second_only
    p_value = float(binomtest(first_only, discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "first_method": first,
        "second_method": second,
        "first_display": DISPLAY[first],
        "second_display": DISPLAY[second],
        "blocks": len(keys),
        "first_successes": int(first_values.sum()),
        "second_successes": int(second_values.sum()),
        "first_success_rate": float(first_values.mean()),
        "second_success_rate": float(second_values.mean()),
        "first_only_wins": first_only,
        "second_only_wins": second_only,
        "net_wins": first_only - second_only,
        "discordant_blocks": discordant,
        "success_delta": float(differences.mean()),
        "success_delta_percentage_points": float(100 * differences.mean()),
        "exact_two_sided_mcnemar_p": p_value,
        "paired_bootstrap_draws": BOOTSTRAP_DRAWS,
        "paired_bootstrap_seed": paired_seed,
        "paired_bootstrap_ci": paired_ci(differences, paired_seed),
        "task_cluster_bootstrap_draws": BOOTSTRAP_DRAWS,
        "task_cluster_bootstrap_seed": cluster_seed,
        "task_cluster_bootstrap_ci": cluster_ci(task_differences, cluster_seed),
        "task_ids": TASKS,
        "task_differences": task_differences.tolist(),
        "leave_one_task_out": [
            float(np.delete(task_differences, i).mean()) for i in range(len(TASKS))
        ],
    }


def collect(manifest_path: Path, repo_root: Path) -> tuple[dict, dict[tuple[int, int, str], int]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete") or int(manifest.get("completed_episodes", -1)) != 500:
        raise RuntimeError("Spatial manifest is not the completed 500-episode rollout")
    outcomes: dict[tuple[int, int, str], int] = {}
    for episode in manifest["episodes"]:
        task = int(episode["task_id"])
        state = int(episode["state_id"])
        method = str(episode["method"])
        if task not in TASKS or state not in STATES or method not in METHODS:
            raise RuntimeError(f"unexpected Spatial manifest cell: {task}, {state}, {method}")
        key = (task, state, method)
        if key in outcomes:
            raise RuntimeError(f"duplicate Spatial cell: {key}")
        log_path = Path(str(episode["log_path"]))
        if not log_path.is_absolute():
            log_path = repo_root / log_path
        if not log_path.is_file():
            raise FileNotFoundError(log_path)
        with gzip.open(log_path, "rt", encoding="utf-8") as handle:
            log = json.load(handle)
        summary = log.get("summary", {})
        if bool(summary.get("success")) != bool(episode["success"]):
            raise RuntimeError(f"manifest/log success mismatch: {key}")
        outcomes[key] = int(bool(episode["success"]))
    expected = {(task, state, method) for task in TASKS for state in STATES for method in METHODS}
    if set(outcomes) != expected:
        raise RuntimeError(f"Spatial coverage is not exactly 10 x 10 x 5: {len(outcomes)}")
    return manifest, outcomes


def per_task(outcomes: dict[tuple[int, int, str], int]) -> list[dict[str, Any]]:
    rows = []
    for task in TASKS:
        counts = {method: int(sum(outcomes[(task, state, method)] for state in STATES)) for method in METHODS}
        rows.append({
            "task_id": task,
            "blocks": len(STATES),
            **{f"{DISPLAY[method]}_successes": counts[method] for method in METHODS},
            **{f"{DISPLAY[method]}_success_rate": counts[method] / len(STATES) for method in METHODS},
            "FO20_minus_Fresh": counts["C_ASYMMETRIC_FO20"] - counts["A_NEWEST"],
            "FO20_minus_FullOld20": counts["C_ASYMMETRIC_FO20"] - counts["B_FULL_OLD20"],
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("research/audit_outputs/gate4a2_spatial_rollout_manifest.json"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/gate4a2_spatial_analysis"))
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    manifest, outcomes = collect(args.manifest, repo_root)
    table = []
    for method in METHODS:
        values = [outcomes[(task, state, method)] for task in TASKS for state in STATES]
        table.append({
            "method": method,
            "display_name": DISPLAY[method],
            "successes": int(sum(values)),
            "blocks": len(values),
            "success_rate": float(np.mean(values)),
        })
    comparisons = {
        "FO20_VS_FRESH": comparison(outcomes, "C_ASYMMETRIC_FO20", "A_NEWEST", 20260831, 20260832),
        "FO20_VS_FULL_OLD20": comparison(outcomes, "C_ASYMMETRIC_FO20", "B_FULL_OLD20", 20260833, 20260834),
    }
    task_rows = per_task(outcomes)
    loto = {
        label: row["leave_one_task_out"] for label, row in comparisons.items()
    }
    analysis = {
        "schema_version": 1,
        "analysis_type": "zero_rollout_log_reanalysis",
        "manifest": str(args.manifest.resolve()),
        "preregistered_completed_before_final_confirmation": True,
        "scope": {
            "suite": "LIBERO Spatial",
            "task_ids": TASKS,
            "state_ids": STATES,
            "blocks_per_method": 100,
            "episodes": 500,
        },
        "conditions": list(METHODS),
        "aggregate": table,
        "comparisons": comparisons,
        "per_task": task_rows,
        "leave_one_task_out": loto,
        "limitations": [
            "This dataset was preregistered and completed before the final cross-suite confirmation.",
            "It contains no Reverse20 and therefore cannot establish the full arm-versus-gripper factorial asymmetry.",
            "It contains no hard-h16 practical baseline.",
            "Its suite-level checkpoint differs from the final confirmation checkpoint family, so absolute success rates are not compared across experiments.",
        ],
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seeds": {"paired": [20260831, 20260833], "task_cluster": [20260832, 20260834]}},
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "analysis.json", analysis)
    write_csv(args.output_root / "per_task.csv", task_rows)
    lines = [
        "# Gate-4A2 Spatial ACT reanalysis",
        "",
        "## Results",
        "",
        "This is a zero-rollout reanalysis of existing logs from the preregistered, completed LIBERO Spatial rollout. The dataset was preregistered and completed before the final cross-suite confirmation.",
        "",
        "| Method | Success | Success % |",
        "|---|---:|---:|",
    ]
    for row in table:
        lines.append(f"| {row['display_name']} | {row['successes']}/{row['blocks']} | {100*row['success_rate']:.1f}% |")
    lines += ["", "## Paired contrasts", "", "| Contrast | First-only | Second-only | Net | Delta (pp) | Exact two-sided McNemar p | Paired 95% CI | Task-cluster 95% CI |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, row in comparisons.items():
        lines.append(f"| {row['first_display']} vs {row['second_display']} | {row['first_only_wins']} | {row['second_only_wins']} | {row['net_wins']} | {row['success_delta_percentage_points']:.2f} | {row['exact_two_sided_mcnemar_p']:.6g} | [{row['paired_bootstrap_ci'][0]:.3f}, {row['paired_bootstrap_ci'][1]:.3f}] | [{row['task_cluster_bootstrap_ci'][0]:.3f}, {row['task_cluster_bootstrap_ci'][1]:.3f}] |")
    lines += ["", "## Per-task success", "", "| Task | Fresh | FO20 | FullOld20 | AGE_EXP_B003 | COGACT_A03 | FO20−Fresh | FO20−FullOld20 |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in task_rows:
        lines.append(f"| {row['task_id']} | {row['Fresh_successes']}/10 | {row['FO20_successes']}/10 | {row['FullOld20_successes']}/10 | {row['AGE_EXP_B003_successes']}/10 | {row['COGACT_A03_successes']}/10 | {row['FO20_minus_Fresh']} | {row['FO20_minus_FullOld20']} |")
    lines += ["", "## Leave-one-task-out", "", "| Omitted task | FO20−Fresh | FO20−FullOld20 |", "|---:|---:|---:|"]
    for i, task in enumerate(TASKS):
        lines.append(f"| {task} | {loto['FO20_VS_FRESH'][i]:.4f} | {loto['FO20_VS_FULL_OLD20'][i]:.4f} |")
    lines += ["", "## Scope limitations", "", "This dataset was preregistered and completed before the final cross-suite confirmation.", "", "It contains no Reverse20 and therefore cannot establish the full arm-vs-gripper factorial asymmetry.", "", "It contains no hard-h16 practical baseline.", "", "Its suite-level checkpoint differs from the final confirmation checkpoint family, so absolute success rates are not compared across experiments.", ""]
    (args.output_root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"analysis": str((args.output_root / 'analysis.json').resolve()), "episodes": 500}, indent=2))


if __name__ == "__main__":
    main()
