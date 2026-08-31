"""Zero-rollout decomposition of the existing Object executor outcomes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
FACTORIAL = ROOT / "experiments/group_delay_factorial_act20"
ASYM = ROOT / "experiments/asymmetric_chunk_reuse_dev"
TASKS = list(range(1, 10))
STATES = [20, 21, 22, 23, 27, 31, 34, 35, 38, 39, 44, 45, 47, 48]
METHODS = ("FRESH", "C2_H16_ARM_FRESH_GRIP", "HARD_H16", "C1_PREVIOUS_CHUNK_GRIP")
DISPLAY = {
    "FRESH": "Fresh",
    "C2_H16_ARM_FRESH_GRIP": "C2 H16Arm+FreshGrip",
    "HARD_H16": "hard h16",
    "C1_PREVIOUS_CHUNK_GRIP": "C1 PreviousChunkGrip",
}
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


def compare(outcomes: dict[tuple[int, int, str], int], first: str, second: str, paired_seed: int, cluster_seed: int) -> dict[str, Any]:
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
    return {
        "first_method": first,
        "second_method": second,
        "first_display": DISPLAY[first],
        "second_display": DISPLAY[second],
        "blocks": len(keys),
        "first_successes": int(first_values.sum()),
        "second_successes": int(second_values.sum()),
        "first_only_wins": first_only,
        "second_only_wins": second_only,
        "net_wins": first_only - second_only,
        "discordant_blocks": discordant,
        "success_delta": float(differences.mean()),
        "success_delta_percentage_points": float(100 * differences.mean()),
        "exact_two_sided_mcnemar_p": float(binomtest(first_only, discordant, 0.5).pvalue) if discordant else 1.0,
        "paired_bootstrap_draws": BOOTSTRAP_DRAWS,
        "paired_bootstrap_seed": paired_seed,
        "paired_bootstrap_ci": paired_ci(differences, paired_seed),
        "task_cluster_bootstrap_draws": BOOTSTRAP_DRAWS,
        "task_cluster_bootstrap_seed": cluster_seed,
        "task_cluster_bootstrap_ci": cluster_ci(task_differences, cluster_seed),
        "task_ids": TASKS,
        "task_differences": task_differences.tolist(),
        "leave_one_task_out": [float(np.delete(task_differences, i).mean()) for i in range(len(TASKS))],
    }


def compatibility(factorial: dict[str, Any], asym: dict[str, Any], factorial_root: Path, asym_root: Path) -> dict[str, Any]:
    f_cohort, a_cohort = factorial["cohort"], asym["cohort"]
    f_run, a_run = factorial["runtime"], asym["runtime"]
    checks = {
        "tasks": f_cohort["primary_task_ids"] == a_cohort["primary_task_ids"] == TASKS,
        "state_ids": f_cohort["state_ids"] == a_cohort["state_ids"] == STATES,
        "environment_seed_rule": f_cohort["environment_seed_rule"] == a_cohort["environment_seed_rule"],
        "full_per_task_seed_list": f_cohort["environment_seeds_by_task"] == a_cohort["environment_seeds_by_task"],
        "act_checkpoint": f_run["checkpoint"] == a_run["checkpoint"],
        "checkpoint_chunk_size": f_run["policy_checkpoint_chunk_size"] == a_run["policy_checkpoint_chunk_size"] == 100,
        "observation_preprocessing": all(f_run[key] == a_run[key] for key in ("obs_type", "camera_name", "camera_name_mapping", "observation_width", "observation_height")),
        "control_mode": f_run["control_mode"] == a_run["control_mode"],
        "control_frequency_hz": f_run["control_frequency_hz"] == a_run["control_frequency_hz"],
        "max_episode_steps": f_run["max_episode_steps"] == a_run["max_episode_steps"] == 280,
        "success_criterion": f_run["success_criterion"] == a_run["success_criterion"],
        "fresh_environment_reset_protocol": all(f_run[key] == a_run[key] for key in ("hard_reset", "n_envs", "use_async_envs")),
        "policy_deterministic_inference_settings": all(f_run[key] == a_run[key] for key in ("policy_rng_seed", "policy_temporal_ensemble", "action_smoothing")),
        "source_executor": asym.get("baseline_reuse_verification", {}).get("source_file_path") == "experiments/group_delay_factorial_act20/temporal_reuse.py",
        "existing_results_are_fresh_env": True,
    }
    details = {
        "factorial_protocol": str((factorial_root / "protocol.json").resolve()),
        "asym_protocol": str((asym_root / "protocol.json").resolve()),
        "factorial_commit": "7ab52cbc6360ae8436cfe5a04f8d200130d3f7a4",
        "asym_commit": "4cf1cbf97411e0cd7face0974c26adc1b25de37d",
        "checks": checks,
        "compatible": all(checks.values()),
    }
    return details


def collect() -> tuple[dict[tuple[int, int, str], int], dict[str, Any]]:
    outcomes: dict[tuple[int, int, str], int] = {}
    for task in TASKS:
        factorial = json.loads((FACTORIAL / "results" / f"task_{task:02d}.json").read_text())
        asym = json.loads((ASYM / "results" / f"task_{task:02d}.json").read_text())
        for episode in factorial["episodes"]["FRESH"] + factorial["episodes"]["HARD_H16"]:
            method = episode["method"]
            key = (task, int(episode["requested_initial_state_id"]), method)
            if key in outcomes:
                raise RuntimeError(f"duplicate outcome cell: {key}")
            outcomes[key] = int(bool(episode["success"]))
            if episode["environment_seed"] != 330000 + 100 * task + key[1] or not episode["fresh_environment_instance"]:
                raise RuntimeError(f"factorial outcome identity mismatch: {key}")
        for episode in asym["episodes"]["C1_PREVIOUS_CHUNK_GRIP"] + asym["episodes"]["C2_H16_ARM_FRESH_GRIP"]:
            source_method = episode["method"]
            method = source_method
            key = (task, int(episode["requested_initial_state_id"]), method)
            if key in outcomes:
                raise RuntimeError(f"duplicate outcome cell: {key}")
            outcomes[key] = int(bool(episode["success"]))
            if episode["environment_seed"] != 330000 + 100 * task + key[1] or not episode["fresh_environment_instance"]:
                raise RuntimeError(f"asymmetric outcome identity mismatch: {key}")
    expected = {(task, state, method) for task in TASKS for state in STATES for method in METHODS}
    if set(outcomes) != expected:
        raise RuntimeError(f"Object decomposition coverage mismatch: {len(outcomes)}")
    factorial_protocol = json.loads((FACTORIAL / "protocol.json").read_text())
    asym_protocol = json.loads((ASYM / "protocol.json").read_text())
    return outcomes, compatibility(factorial_protocol, asym_protocol, FACTORIAL, ASYM)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("experiments/object_executor_decomposition"))
    args = parser.parse_args()
    outcomes, compat = collect()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not compat["compatible"]:
        analysis = {"schema_version": 1, "compatible": False, "compatibility": compat, "analysis_skipped": True}
        write_json(args.output_root / "analysis.json", analysis)
        (args.output_root / "report.md").write_text("# Object executor decomposition\n\nCompatibility was insufficient; analysis was skipped.\n\n```json\n" + json.dumps(compat, indent=2) + "\n```\n", encoding="utf-8")
        print(json.dumps(analysis, indent=2))
        return
    comparisons = {
        "C2_VS_FRESH": compare(outcomes, "C2_H16_ARM_FRESH_GRIP", "FRESH", 20260841, 20260842),
        "HARD_H16_VS_FRESH": compare(outcomes, "HARD_H16", "FRESH", 20260843, 20260844),
        "C1_VS_FRESH": compare(outcomes, "C1_PREVIOUS_CHUNK_GRIP", "FRESH", 20260845, 20260846),
        "HARD_H16_VS_C2": compare(outcomes, "HARD_H16", "C2_H16_ARM_FRESH_GRIP", 20260847, 20260848),
    }
    task_rows = []
    for task in TASKS:
        counts = {method: int(sum(outcomes[(task, state, method)] for state in STATES)) for method in METHODS}
        task_rows.append({
            "task_id": task,
            "blocks": len(STATES),
            **{f"{method}_successes": counts[method] for method in METHODS},
            "C2_minus_FRESH": counts["C2_H16_ARM_FRESH_GRIP"] - counts["FRESH"],
            "HARD_H16_minus_FRESH": counts["HARD_H16"] - counts["FRESH"],
            "C1_minus_FRESH": counts["C1_PREVIOUS_CHUNK_GRIP"] - counts["FRESH"],
        })
    decomposition = {
        "identity": "HARD_H16 - FRESH = (HARD_H16 - C2_H16_ARM_FRESH_GRIP) + (C2_H16_ARM_FRESH_GRIP - FRESH)",
        "hard_minus_fresh": comparisons["HARD_H16_VS_FRESH"]["success_delta"],
        "hard_minus_c2": comparisons["HARD_H16_VS_C2"]["success_delta"],
        "c2_minus_fresh": comparisons["C2_VS_FRESH"]["success_delta"],
        "identity_holds_exactly": bool(np.isclose(
            comparisons["HARD_H16_VS_FRESH"]["success_delta"],
            comparisons["HARD_H16_VS_C2"]["success_delta"] + comparisons["C2_VS_FRESH"]["success_delta"],
            atol=0.0,
        )),
    }
    analysis = {
        "schema_version": 1,
        "analysis_type": "zero_rollout_executor_decomposition",
        "compatible": True,
        "compatibility": compat,
        "scope": {"tasks": TASKS, "states": STATES, "blocks": 126, "source": "existing per-state outcomes only"},
        "comparisons": comparisons,
        "decomposition": decomposition,
        "per_task": task_rows,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS},
    }
    write_json(args.output_root / "analysis.json", analysis)
    with (args.output_root / "per_task.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(task_rows)
    lines = [
        "# Object executor decomposition",
        "",
        "This is a zero-rollout analysis of the exact existing Object development outcomes. The two protocols match on the listed compatibility checks, so the 126 paired task-state blocks are combined.",
        "",
        "## Paired contrasts",
        "",
        "| Contrast | First successes | Second successes | First-only | Second-only | Net | Delta (pp) | Exact McNemar p | Paired 95% CI | Cluster 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("C2_VS_FRESH", "HARD_H16_VS_FRESH", "C1_VS_FRESH"):
        row = comparisons[label]
        lines.append(f"| {row['first_display']} vs {row['second_display']} | {row['first_successes']} | {row['second_successes']} | {row['first_only_wins']} | {row['second_only_wins']} | {row['net_wins']} | {row['success_delta_percentage_points']:.2f} | {row['exact_two_sided_mcnemar_p']:.6g} | [{row['paired_bootstrap_ci'][0]:.3f}, {row['paired_bootstrap_ci'][1]:.3f}] | [{row['task_cluster_bootstrap_ci'][0]:.3f}, {row['task_cluster_bootstrap_ci'][1]:.3f}] |")
    lines += ["", "## Mean-effect decomposition", "", f"HARD_H16 − FRESH = {decomposition['hard_minus_fresh']:.6f} = (HARD_H16 − C2) {decomposition['hard_minus_c2']:+.6f} + (C2 − FRESH) {decomposition['c2_minus_fresh']:+.6f}.", "", "This is an arithmetic decomposition of paired mean success differences, not a mediation or percentage attribution analysis.", "", "## Per-task counts and leave-one-task-out deltas", "", "| Task | Fresh | C2 | hard h16 | C1 | C2−Fresh | hard−Fresh | C1−Fresh |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in task_rows:
        lines.append(f"| {row['task_id']} | {row['FRESH_successes']}/14 | {row['C2_H16_ARM_FRESH_GRIP_successes']}/14 | {row['HARD_H16_successes']}/14 | {row['C1_PREVIOUS_CHUNK_GRIP_successes']}/14 | {row['C2_minus_FRESH']} | {row['HARD_H16_minus_FRESH']} | {row['C1_minus_FRESH']} |")
    lines += ["", "| Omitted task | C2−Fresh | hard−Fresh | C1−Fresh |", "|---:|---:|---:|---:|"]
    for i, task in enumerate(TASKS):
        lines.append(f"| {task} | {comparisons['C2_VS_FRESH']['leave_one_task_out'][i]:.4f} | {comparisons['HARD_H16_VS_FRESH']['leave_one_task_out'][i]:.4f} | {comparisons['C1_VS_FRESH']['leave_one_task_out'][i]:.4f} |")
    lines += ["", "## Compatibility checks", "", "```json", json.dumps(compat, indent=2), "```", "", "Interpretation is conditional on these paired-result comparisons. No new episodes were run.", ""]
    (args.output_root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"analysis": str((args.output_root / 'analysis.json').resolve()), "compatible": True}, indent=2))


if __name__ == "__main__":
    main()
