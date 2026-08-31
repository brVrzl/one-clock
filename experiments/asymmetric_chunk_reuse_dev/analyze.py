"""Analyze only the frozen C1/C2 outcomes against the reused hard-h16 baseline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
OLD_ROOT = REPO_ROOT / "experiments" / "group_delay_factorial_act20"
sys.path.insert(0, str(ROOT))

from asymmetric_executor import C1, C2, H16, METHODS  # noqa: E402
from validate_shard import validate_shard  # noqa: E402


HARD = "HARD_H16"
ALL_METHODS = (C2, HARD, C1)
COMPARISONS = (
    ("C1_VS_HARD_H16", C1, HARD),
    ("C1_VS_C2", C1, C2),
    ("HARD_H16_VS_C2", HARD, C2),
)
BOOTSTRAP_DRAWS = 20_000


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def paired_bootstrap(differences: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(BOOTSTRAP_DRAWS, len(differences)))
    draws = differences[indices].mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def cluster_bootstrap(task_differences: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(task_differences), size=(BOOTSTRAP_DRAWS, len(task_differences)))
    draws = task_differences[indices].mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def comparison_statistics(
    outcomes: dict[tuple[int, int, str], int],
    tasks: list[int],
    states: list[int],
    first: str,
    second: str,
    paired_seed: int,
    cluster_seed: int,
) -> dict[str, Any]:
    keys = [(task, state) for task in tasks for state in states]
    first_values = np.asarray([outcomes[(task, state, first)] for task, state in keys], dtype=np.int8)
    second_values = np.asarray([outcomes[(task, state, second)] for task, state in keys], dtype=np.int8)
    differences = first_values.astype(np.float64) - second_values.astype(np.float64)
    task_differences = np.asarray([
        differences[[task == key_task for key_task, _ in keys]].mean() for task in tasks
    ])
    first_only = int(np.count_nonzero((first_values == 1) & (second_values == 0)))
    second_only = int(np.count_nonzero((first_values == 0) & (second_values == 1)))
    discordant = first_only + second_only
    p_value = float(binomtest(first_only, discordant, 0.5).pvalue) if discordant else 1.0
    paired_ci = paired_bootstrap(differences, paired_seed)
    cluster_ci = cluster_bootstrap(task_differences, cluster_seed)
    leave_one_out = [float(np.delete(task_differences, index).mean()) for index in range(len(tasks))]
    stable_positive = bool(paired_ci[0] > 0 and cluster_ci[0] > 0 and min(leave_one_out) > 0)
    stable_negative = bool(paired_ci[1] < 0 and cluster_ci[1] < 0 and max(leave_one_out) < 0)
    return {
        "first_method": first,
        "second_method": second,
        "blocks": len(keys),
        "first_successes": int(first_values.sum()),
        "second_successes": int(second_values.sum()),
        "first_success_rate": float(first_values.mean()),
        "second_success_rate": float(second_values.mean()),
        "success_delta": float(differences.mean()),
        "success_delta_percentage_points": float(100 * differences.mean()),
        "first_only_wins": first_only,
        "second_only_wins": second_only,
        "net_wins": first_only - second_only,
        "discordant_blocks": discordant,
        "exact_two_sided_mcnemar_p": p_value,
        "paired_bootstrap_draws": BOOTSTRAP_DRAWS,
        "paired_bootstrap_seed": paired_seed,
        "paired_bootstrap_ci": paired_ci,
        "task_cluster_bootstrap_draws": BOOTSTRAP_DRAWS,
        "task_cluster_bootstrap_seed": cluster_seed,
        "task_cluster_bootstrap_ci": cluster_ci,
        "task_ids": tasks,
        "task_differences": task_differences.tolist(),
        "positive_task_differences": int(np.count_nonzero(task_differences > 0)),
        "leave_one_task_out": leave_one_out,
        "stable_positive": stable_positive,
        "stable_negative": stable_negative,
    }


def episode_record(episode: dict[str, Any], method: str, task_id: int, state_id: int) -> dict[str, Any]:
    rows = episode["step_log"]
    records = []
    for index, row in enumerate(rows):
        if method == HARD:
            action = np.asarray(row["action"], dtype=np.float64)
            age = index % H16
        else:
            action = np.asarray(row["executed_action_7d"], dtype=np.float64)
            age = int(row["grip_source_q"]) * 0 + int(row["grip_offset"])  # same-target age
        records.append({"t": index, "action": action, "gripper_age": age})
    return {
        "task_id": int(task_id),
        "state_id": int(state_id),
        "method": method,
        "success": bool(episode["success"]),
        "environment_seed": int(episode["environment_seed"]),
        "environment_steps": int(episode["environment_steps"]),
        "policy_queries": int(episode["policy_queries"]),
        "query_rate": float(episode["query_rate"]),
        "wall_clock_seconds": None if method == HARD else float(episode["wall_clock_seconds"]),
        "mean_policy_call_latency_seconds": None if method == HARD else float(episode["mean_policy_call_latency_seconds"]),
        "steps": records,
    }


def collect_new(protocol: dict[str, Any], result_root: Path) -> dict[tuple[int, int, str], dict[str, Any]]:
    tasks = [int(x) for x in protocol["cohort"]["primary_task_ids"]]
    states = [int(x) for x in protocol["cohort"]["state_ids"]]
    episodes: dict[tuple[int, int, str], dict[str, Any]] = {}
    for task_id in tasks:
        result_path = result_root / "results" / f"task_{task_id:02d}.json"
        validate_shard(result_path, ROOT / "protocol.json")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for method in METHODS:
            for episode in result["episodes"][method]:
                state_id = int(episode["requested_initial_state_id"])
                key = (task_id, state_id, method)
                if key in episodes:
                    raise RuntimeError(f"duplicate new outcome cell: {key}")
                episodes[key] = episode_record(episode, method, task_id, state_id)
    expected = {(task, state, method) for task in tasks for state in states for method in METHODS}
    if set(episodes) != expected:
        raise RuntimeError("new outcome coverage is not exactly 9 x 14 x 2")
    return episodes


def collect_hard(protocol: dict[str, Any]) -> dict[tuple[int, int, str], dict[str, Any]]:
    tasks = [int(x) for x in protocol["cohort"]["primary_task_ids"]]
    states = [int(x) for x in protocol["cohort"]["state_ids"]]
    episodes: dict[tuple[int, int, str], dict[str, Any]] = {}
    for task_id in tasks:
        path = OLD_ROOT / "results" / f"task_{task_id:02d}.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["methods"] != ["FRESH", "FO20", "REVERSE20", "FULL_OLD20", HARD]:
            raise RuntimeError("inherited hard-h16 result method identity drifted")
        expected_seeds = protocol["cohort"]["environment_seeds_by_task"][str(task_id)]
        for episode, state_id, seed in zip(result["episodes"][HARD], states, expected_seeds, strict=True):
            if int(episode["environment_seed"]) != int(seed) or episode["fresh_environment_instance"] is not True:
                raise RuntimeError("inherited hard-h16 seed/environment identity mismatch")
            rows = episode["step_log"]
            query_steps = [int(x) for x in episode["query_steps"]]
            if query_steps != list(range(0, len(rows), H16)):
                raise RuntimeError("inherited hard-h16 query schedule mismatch")
            for t, row in enumerate(rows):
                q = int(row["arm_source_query_q"])
                offset = int(row["arm_chunk_offset"])
                grip_q = int(row["gripper_source_query_q"])
                grip_offset = int(row["gripper_chunk_offset"])
                if q % H16 != 0 or q + offset != t or grip_q != q or grip_offset != offset:
                    raise RuntimeError("inherited hard-h16 source semantics mismatch")
            key = (task_id, int(state_id), HARD)
            episodes[key] = episode_record(episode, HARD, task_id, int(state_id))
    expected = {(task, state, HARD) for task in tasks for state in states}
    if set(episodes) != expected:
        raise RuntimeError("inherited hard-h16 coverage is not exactly 9 x 14")
    return episodes


def descriptive(records: list[dict[str, Any]], method: str, outcome: bool) -> dict[str, Any]:
    selected = [record for record in records if record["success"] is outcome]
    all_steps = [step for record in selected for step in record["steps"]]
    ages = [int(step["gripper_age"]) for step in all_steps]
    actions = np.asarray([step["action"] for step in all_steps], dtype=np.float64) if all_steps else np.empty((0, 7))
    gripper_values = actions[:, 6] if len(actions) else np.asarray([], dtype=np.float64)
    first_positive = []
    negative_recurred = []
    sign_flips = 0
    for record in selected:
        values = np.asarray([step["action"][6] for step in record["steps"]], dtype=np.float64)
        positives = np.flatnonzero(values > 0)
        first = int(positives[0]) if len(positives) else None
        first_positive.append(first)
        negative_recurred.append(bool(first is not None and np.any(values[first + 1 :] < 0)))
        if len(values) > 1:
            sign_flips += int(np.count_nonzero(values[:-1] * values[1:] < 0))
    positive_first = [value for value in first_positive if value is not None]
    histogram = {str(age): int(ages.count(age)) for age in range(32)}
    return {
        "method": method,
        "outcome": "success" if outcome else "failure",
        "episodes": len(selected),
        "steps": len(all_steps),
        "gripper_fraction_abs_below_0_5": float(np.mean(np.abs(gripper_values) < 0.5)) if len(gripper_values) else None,
        "gripper_sign_flips_per_100_steps": float(100 * sign_flips / len(all_steps)) if all_steps else None,
        "first_positive_step_values": first_positive,
        "first_positive_step_mean_conditional": float(np.mean(positive_first)) if positive_first else None,
        "first_positive_step_median_conditional": float(np.median(positive_first)) if positive_first else None,
        "first_positive_step_missing_episodes": int(len(first_positive) - len(positive_first)),
        "gripper_fraction_positive": float(np.mean(gripper_values > 0)) if len(gripper_values) else None,
        "negative_recurred_after_first_positive_fraction": float(np.mean(negative_recurred)) if negative_recurred else None,
        "negative_recurred_denominator_episodes": int(len(negative_recurred)),
        "arm_translation_norm_mean": float(np.mean(np.linalg.norm(actions[:, 0:3], axis=1))) if len(actions) else None,
        "arm_rotation_norm_mean": float(np.mean(np.linalg.norm(actions[:, 3:6], axis=1))) if len(actions) else None,
        "gripper_source_age_histogram": histogram,
        "gripper_source_age_min": int(min(ages)) if ages else None,
        "gripper_source_age_max": int(max(ages)) if ages else None,
    }


def method_table(episodes: dict[tuple[int, int, str], dict[str, Any]], tasks: list[int], states: list[int]) -> list[dict[str, Any]]:
    rows = []
    labels = {
        C2: ("current h16 chunk", "fresh q=t"),
        HARD: ("current h16 chunk", "current h16 chunk"),
        C1: ("current h16 chunk", "previous h16 chunk"),
    }
    for method in ALL_METHODS:
        selected = [episodes[(task, state, method)] for task in tasks for state in states]
        steps = sum(record["environment_steps"] for record in selected)
        queries = sum(record["policy_queries"] for record in selected)
        ages = [step["gripper_age"] for record in selected for step in record["steps"]]
        wall = [record["wall_clock_seconds"] for record in selected if record["wall_clock_seconds"] is not None]
        latency = [record["mean_policy_call_latency_seconds"] for record in selected if record["mean_policy_call_latency_seconds"] is not None]
        rows.append({
            "method": method,
            "arm_source": labels[method][0],
            "grip_source": labels[method][1],
            "successes": int(sum(record["success"] for record in selected)),
            "success_rate": float(np.mean([record["success"] for record in selected])),
            "policy_queries": queries,
            "environment_steps": steps,
            "query_rate": queries / steps,
            "mean_gripper_age": float(np.mean(ages)),
            "min_gripper_age": int(min(ages)),
            "max_gripper_age": int(max(ages)),
            "mean_wall_clock_seconds_per_episode": float(np.mean(wall)) if wall else None,
            "mean_policy_call_latency_seconds": float(np.mean(latency)) if latency else None,
        })
    return rows


def per_task_rows(outcomes: dict[tuple[int, int, str], int], tasks: list[int], states: list[int]) -> list[dict[str, Any]]:
    rows = []
    for task_id in tasks:
        counts = {method: int(sum(outcomes[(task_id, state, method)] for state in states)) for method in ALL_METHODS}
        rows.append({
            "task_id": task_id,
            "blocks": len(states),
            "C2_successes": counts[C2],
            "HARD_H16_successes": counts[HARD],
            "C1_successes": counts[C1],
            "C1_minus_HARD_H16": counts[C1] - counts[HARD],
            "C1_minus_C2": counts[C1] - counts[C2],
            "HARD_H16_minus_C2": counts[HARD] - counts[C2],
        })
    return rows


def write_per_task(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def decision(comparisons: dict[str, dict[str, Any]]) -> tuple[str, str]:
    c1h = comparisons["C1_VS_HARD_H16"]
    c12 = comparisons["C1_VS_C2"]
    if (
        c1h["first_successes"] > c1h["second_successes"]
        and c12["first_successes"] > c12["second_successes"]
        and c1h["net_wins"] > 0
        and sum(delta > 0 for delta in c1h["leave_one_task_out"]) >= 7
    ):
        return "ASYM_REUSE_STRONG", "C1 exceeded hard h16 and C2, had positive paired net wins versus hard h16, and at least 7/9 leave-one-task-out C1-minus-hard deltas were positive."
    if (
        c1h["first_successes"] > c1h["second_successes"]
        and c12["first_successes"] > c12["second_successes"]
        and c1h["net_wins"] > 0
        and sum(delta > 0 for delta in c1h["leave_one_task_out"]) < 7
    ):
        return "ASYM_REUSE_PROMISING", "C1 exceeded hard h16 and C2 with positive paired net wins versus hard h16, but fewer than 7/9 leave-one-task-out C1-minus-hard deltas were positive."
    if c1h["first_successes"] <= c1h["second_successes"] and c12["first_successes"] - c12["second_successes"] >= 5:
        return "ASYM_REUSE_MECHANISM_ONLY", "C1 did not exceed hard h16 but exceeded C2 by at least 5/126 successes."
    return "ASYM_REUSE_NULL", "The frozen strong, promising, and mechanism-only criteria were not met."


def report_text(analysis: dict[str, Any]) -> str:
    lines = [
        "# Asymmetric Temporal Reuse development gate",
        "",
        "## Results",
        "",
        "These are development results on the exposed Object tasks 1-9 cohort, used only to decide whether to freeze the executor. The inferential result for the paper will come from the subsequent frozen cross-suite / unseen confirmation run, not from this table.",
        "",
        f"Decision branch reached: **{analysis['decision']}**. No paper title is chosen in this run.",
        "",
        "### C1/C2/hard-h16 table",
        "",
        "| Method | Arm source | Grip source | Success /126 | Success % | Observed query rate | Observed mean grip age | Age range |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["primary_table"]:
        lines.append(f"| {row['method']} | {row['arm_source']} | {row['grip_source']} | {row['successes']}/126 | {100*row['success_rate']:.1f}% | {row['query_rate']:.5f} | {row['mean_gripper_age']:.3f} | {row['min_gripper_age']}–{row['max_gripper_age']} |")
    lines += [
        "",
        "C1 has the same structural h16 query schedule as hard h16. C2 queries densely for fresh gripper values, but its arm always comes from the scheduled h16 chunk.",
        "",
        "### Timing (secondary)",
        "",
        "| Method | Total policy queries | Total environment steps | Mean wall-clock s/episode | Mean policy-call latency (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in analysis["primary_table"]:
        wall = "n/a (reused baseline)" if row["mean_wall_clock_seconds_per_episode"] is None else f"{row['mean_wall_clock_seconds_per_episode']:.3f}"
        latency = "n/a (reused baseline)" if row["mean_policy_call_latency_seconds"] is None else f"{row['mean_policy_call_latency_seconds']:.6f}"
        lines.append(f"| {row['method']} | {row['policy_queries']} | {row['environment_steps']} | {wall} | {latency} |")
    lines += ["", "### Primary and secondary paired contrasts", "", "| Contrast | First-only | Second-only | Net | Success delta (pp) | Exact two-sided McNemar p | Paired 95% CI | Task-cluster 95% CI |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, _, _ in COMPARISONS:
        row = analysis["comparisons"][label]
        lines.append(f"| {label}: {row['first_method']} vs {row['second_method']} | {row['first_only_wins']} | {row['second_only_wins']} | {row['net_wins']} | {row['success_delta_percentage_points']:.2f} | {row['exact_two_sided_mcnemar_p']:.6g} | [{row['paired_bootstrap_ci'][0]:.3f}, {row['paired_bootstrap_ci'][1]:.3f}] | [{row['task_cluster_bootstrap_ci'][0]:.3f}, {row['task_cluster_bootstrap_ci'][1]:.3f}] |")
    lines += ["", "### Per-task success counts", "", "| Task | C2 | hard h16 | C1 | C1−hard | C1−C2 | hard−C2 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in analysis["per_task"]:
        lines.append(f"| {row['task_id']} | {row['C2_successes']}/14 | {row['HARD_H16_successes']}/14 | {row['C1_successes']}/14 | {row['C1_minus_HARD_H16']} | {row['C1_minus_C2']} | {row['HARD_H16_minus_C2']} |")
    lines += ["", "### Leave-one-task-out stability", "", "| Omitted task | C1−hard h16 | C1−C2 | hard h16−C2 |", "|---:|---:|---:|---:|"]
    for index, task_id in enumerate(analysis["tasks"]):
        lines.append(f"| {task_id} | {analysis['leave_one_task_out']['C1_minus_HARD_H16'][index]:.4f} | {analysis['leave_one_task_out']['C1_minus_C2'][index]:.4f} | {analysis['leave_one_task_out']['HARD_H16_minus_C2'][index]:.4f} |")
    lines += ["", "### Outcome-stratified descriptive step-log analyses", "", "These are descriptive only and are not used to choose a lag, offset bound, horizon, or source-selection rule.", ""]
    for method in ALL_METHODS:
        lines += [f"#### {method}", "", "| Outcome | Episodes | Steps | |a[6]|<0.5 | Sign flips/100 steps | First a[6]>0 step (mean; median; no-positive) | a[6]>0 fraction | Negative recurrence after first positive | Mean ||a[0:3]|| | Mean ||a[3:6]|| |", "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
        for outcome in (True, False):
            row = analysis["descriptive"][method]["success" if outcome else "failure"]
            first = f"{row['first_positive_step_mean_conditional']}; {row['first_positive_step_median_conditional']}; {row['first_positive_step_missing_episodes']}"
            lines.append(f"| {row['outcome']} | {row['episodes']} | {row['steps']} | {row['gripper_fraction_abs_below_0_5'] if row['gripper_fraction_abs_below_0_5'] is not None else 'n/a'} | {row['gripper_sign_flips_per_100_steps'] if row['gripper_sign_flips_per_100_steps'] is not None else 'n/a'} | {first} | {row['gripper_fraction_positive'] if row['gripper_fraction_positive'] is not None else 'n/a'} | {row['negative_recurred_after_first_positive_fraction'] if row['negative_recurred_after_first_positive_fraction'] is not None else 'n/a'} | {row['arm_translation_norm_mean'] if row['arm_translation_norm_mean'] is not None else 'n/a'} | {row['arm_rotation_norm_mean'] if row['arm_rotation_norm_mean'] is not None else 'n/a'} |")
        lines.append("")
        lines.append("Gripper source-age histograms (age: step count, all bins 0–31):")
        for outcome in ("success", "failure"):
            histogram = analysis["descriptive"][method][outcome]["gripper_source_age_histogram"]
            lines.append(f"- {outcome}: " + ", ".join(f"{age}:{count}" for age, count in histogram.items() if count))
        lines.append("")
    lines += [
        "### Pre-registered paper artifact mapping",
        "",
        "The `{C2, hard h16, C1} × {success/126, observed query rate, observed mean gripper age}` table is the three-point gripper-age series at fixed h16 arm semantics.",
        "",
        "- If `C1 > hard h16`: this table is the method table; working title recorded for later consideration: `Asymmetric Temporal Reuse for Action-Chunked Robot Policies`.",
        "- If `C1 ~= hard h16` and `C1 > C2`: this table is the executor-decomposition result; working title recorded for later consideration: `Component-Dependent Effects of Delayed Prediction in Action-Chunked Robot Policies`.",
        "- If `C1 <= hard h16` and `C1 ~= C2`: this table is a negative/scope result and the paper centres on the repaired group-delay factorial.",
        "",
        f"Reached branch recorded for this run: `{analysis['artifact_branch']}`. The title remains undecided.",
        "",
        "Method development stops after this gate. Any subsequent run requires explicit approval and is limited to the single frozen cross-suite / unseen confirmation.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    tasks = [int(x) for x in protocol["cohort"]["primary_task_ids"]]
    states = [int(x) for x in protocol["cohort"]["state_ids"]]
    new_episodes = collect_new(protocol, args.output_root)
    hard_episodes = collect_hard(protocol)
    episodes = {**new_episodes, **hard_episodes}
    outcomes = {(task, state, method): int(record["success"]) for (task, state, method), record in episodes.items()}
    comparisons: dict[str, dict[str, Any]] = {}
    for index, (label, first, second) in enumerate(COMPARISONS):
        comparisons[label] = comparison_statistics(outcomes, tasks, states, first, second, 20260922 + 2 * index, 20260923 + 2 * index)
    decision_label, decision_reason = decision(comparisons)
    per_task = per_task_rows(outcomes, tasks, states)
    primary_table = method_table(episodes, tasks, states)
    descriptive_results: dict[str, dict[str, Any]] = {}
    for method in ALL_METHODS:
        selected = [episodes[(task, state, method)] for task in tasks for state in states]
        descriptive_results[method] = {
            "success": descriptive(selected, method, True),
            "failure": descriptive(selected, method, False),
        }
    leave_one_out = {
        "C1_minus_HARD_H16": comparisons["C1_VS_HARD_H16"]["leave_one_task_out"],
        "C1_minus_C2": comparisons["C1_VS_C2"]["leave_one_task_out"],
        "HARD_H16_minus_C2": comparisons["HARD_H16_VS_C2"]["leave_one_task_out"],
    }
    if decision_label == "ASYM_REUSE_STRONG":
        artifact_branch = "C1 > hard h16: method table"
    elif decision_label in ("ASYM_REUSE_PROMISING",):
        artifact_branch = "C1 > hard h16: method table, pending the single frozen confirmation"
    elif decision_label == "ASYM_REUSE_MECHANISM_ONLY":
        artifact_branch = "C1 <= hard h16 and C1 > C2: executor-decomposition result"
    else:
        artifact_branch = "C1 <= hard h16 and no >=5/126 C1-over-C2 effect: negative/scope result"
    analysis = {
        "schema_version": 1,
        "decision": decision_label,
        "decision_reason": decision_reason,
        "primary_scope": "LIBERO Object tasks 1-9, states 20,21,22,23,27,31,34,35,38,39,44,45,47,48; development only",
        "tasks": tasks,
        "states": states,
        "primary_blocks": 126,
        "new_conditions": list(METHODS),
        "new_total_episodes": 252,
        "historical_hard_h16_reused": True,
        "historical_hard_h16_source_commit": protocol["historical_reference"]["source_commit"],
        "primary_table": primary_table,
        "comparisons": comparisons,
        "per_task": per_task,
        "leave_one_task_out": leave_one_out,
        "descriptive": descriptive_results,
        "artifact_branch": artifact_branch,
        "source": "new C1/C2 outcomes plus exact compatible reused hard-h16 baseline; no other new condition",
    }
    write_json(args.output_root / "analysis.json", analysis)
    write_per_task(args.output_root / "per_task.csv", per_task)
    shard_root = args.output_root / "condition_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    for method in ALL_METHODS:
        rows = []
        for task in tasks:
            for state in states:
                record = episodes[(task, state, method)]
                rows.append({
                    "task_id": task,
                    "state_id": state,
                    "environment_seed": record["environment_seed"],
                    "success": record["success"],
                    "environment_steps": record["environment_steps"],
                    "policy_queries": record["policy_queries"],
                    "query_rate": record["query_rate"],
                    "mean_gripper_age": float(np.mean([step["gripper_age"] for step in record["steps"]])),
                })
        write_json(shard_root / f"{method}.json", {"schema_version": 1, "method": method, "scope": "development tasks 1-9 only", "episodes": rows})
    (args.output_root / "report.md").write_text(report_text(analysis), encoding="utf-8")
    print(json.dumps({"decision": decision_label, "analysis": str((args.output_root / 'analysis.json').resolve())}))


if __name__ == "__main__":
    main()
