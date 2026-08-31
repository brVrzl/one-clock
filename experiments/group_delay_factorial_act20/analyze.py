"""Analyze the frozen repaired ACT20 five-condition factorial."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from temporal_reuse import METHODS
from validate_shard import validate_shard


ROOT = Path(__file__).resolve().parent
COMPARISONS = (
    ("P1_REPLICATION", "FO20", "FRESH"),
    ("P2_GROUP_ASYMMETRY", "FO20", "REVERSE20"),
    ("P3_GROUP_STRUCTURE", "FO20", "FULL_OLD20"),
    ("P4_PRACTICAL_BASELINE", "FO20", "HARD_H16"),
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
    leave_one_out = [
        float(np.delete(task_differences, index).mean()) for index in range(len(tasks))
    ]
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
        "leave_one_task_out": leave_one_out,
        "stable_positive": stable_positive,
        "stable_negative": stable_negative,
    }


def collect(protocol: dict[str, Any], result_root: Path) -> tuple[dict, dict[tuple[int, int, str], int]]:
    tasks = [int(x) for x in protocol["cohort"]["primary_task_ids"]]
    states = [int(x) for x in protocol["cohort"]["state_ids"]]
    outcomes: dict[tuple[int, int, str], int] = {}
    all_episodes: dict[tuple[int, int, str], dict] = {}
    for task_id in tasks:
        result_path = result_root / "results" / f"task_{task_id:02d}.json"
        validate_shard(result_path, ROOT / "protocol.json")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for method in METHODS:
            for episode in result["episodes"][method]:
                key = (task_id, int(episode["requested_initial_state_id"]), method)
                if key in outcomes:
                    raise RuntimeError(f"duplicate primary outcome cell: {key}")
                outcomes[key] = int(bool(episode["success"]))
                all_episodes[key] = episode
    expected = {(task, state, method) for task in tasks for state in states for method in METHODS}
    if set(outcomes) != expected:
        raise RuntimeError("primary outcome coverage is not exactly 9 x 14 x 5")
    return all_episodes, outcomes


def decision(comparisons: dict[str, dict[str, Any]]) -> tuple[str, str]:
    p1 = comparisons["P1_REPLICATION"]
    p2 = comparisons["P2_GROUP_ASYMMETRY"]
    p3 = comparisons["P3_GROUP_STRUCTURE"]
    p4 = comparisons["P4_PRACTICAL_BASELINE"]
    if not p1["stable_positive"]:
        return "GROUP_DELAY_NOT_REPLICATED", "FO20 did not meet the predeclared stable-positive replication rule versus FRESH."
    reverse_fresh = comparisons["REVERSE20_VS_FRESH"]
    if not p2["stable_positive"] and not p2["stable_negative"] and reverse_fresh["stable_positive"]:
        return "GROUP_DELAY_SCALAR_ONLY", "FO20 and REVERSE20 were not clearly different, while both improved over FRESH."
    if p2["stable_positive"] and p3["stable_positive"]:
        if p4["stable_negative"]:
            return "GROUP_DELAY_STRUCTURE_STRONG", "FO20 clearly beat the dense fixed-source controls but was clearly below HARD_H16."
        return "GROUP_DELAY_METHOD_STRONG", "FO20 clearly beat FRESH, REVERSE20, and FULL_OLD20 and was not clearly below HARD_H16."
    return "GROUP_DELAY_NOT_REPLICATED", "FO20 replicated versus FRESH but did not establish the full predeclared group-structure contrast."


def make_per_task(outcomes: dict[tuple[int, int, str], int], tasks: list[int], states: list[int]) -> list[dict[str, Any]]:
    rows = []
    for task_id in tasks:
        counts = {method: int(sum(outcomes[(task_id, state, method)] for state in states)) for method in METHODS}
        interaction = counts["FO20"] - counts["FRESH"] - counts["FULL_OLD20"] + counts["REVERSE20"]
        rows.append({
            "task_id": task_id,
            "blocks": len(states),
            **{f"{method}_successes": counts[method] for method in METHODS},
            **{f"{method}_success_rate": counts[method] / len(states) for method in METHODS},
            "interaction_count": interaction,
            "interaction_rate": interaction / len(states),
        })
    return rows


def write_per_task(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_condition_shards(
    output_root: Path,
    all_episodes: dict[tuple[int, int, str], dict],
    tasks: list[int],
    states: list[int],
) -> None:
    shard_root = output_root / "condition_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        records = []
        for task_id in tasks:
            for state_id in states:
                episode = all_episodes[(task_id, state_id, method)]
                records.append({
                    "task_id": task_id,
                    "state_id": state_id,
                    "environment_seed": int(episode["environment_seed"]),
                    "success": bool(episode["success"]),
                    "environment_steps": int(episode["environment_steps"]),
                    "policy_queries": int(episode["policy_queries"]),
                    "query_rate": float(episode["query_rate"]),
                })
        write_json(
            shard_root / f"{method}.json",
            {"schema_version": 1, "method": method, "scope": "primary tasks 1-9 only", "episodes": records},
        )


def method_table(all_episodes: dict[tuple[int, int, str], dict], tasks: list[int], states: list[int]) -> list[dict[str, Any]]:
    descriptions = {
        "FRESH": ("0", "0", "1.0"),
        "FO20": ("0", "20", "1.0"),
        "REVERSE20": ("20", "0", "1.0"),
        "FULL_OLD20": ("20", "20", "1.0"),
        "HARD_H16": ("joint h16", "joint h16", "observed"),
    }
    rows = []
    for method in METHODS:
        episodes = [all_episodes[(task, state, method)] for task in tasks for state in states]
        queries = sum(int(e["policy_queries"]) for e in episodes)
        steps = sum(int(e["environment_steps"]) for e in episodes)
        rows.append({
            "method": method,
            "d_arm": descriptions[method][0],
            "d_grip": descriptions[method][1],
            "successes": int(sum(bool(e["success"]) for e in episodes)),
            "success_rate": float(np.mean([bool(e["success"]) for e in episodes])),
            "policy_queries": queries,
            "environment_steps": steps,
            "query_rate": queries / steps,
            "configured_query_rate": descriptions[method][2],
        })
    return rows


def report_text(analysis: dict[str, Any]) -> str:
    lines = [
        "# Repaired ACT group-delay factorial",
        "",
        f"Decision: **{analysis['decision']}**",
        "",
        "The primary aggregate contains only the new repaired outcomes for Object tasks 1–9, 14 states per task, and 126 paired blocks per method. Historical Gate-3C outcomes are context only and are not spliced into this table.",
        "",
        "## Primary table",
        "",
        "| Method | d_arm | d_grip | Success /126 | Success % | Query rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["primary_table"]:
        lines.append(f"| {row['method']} | {row['d_arm']} | {row['d_grip']} | {row['successes']}/126 | {100*row['success_rate']:.1f}% | {row['query_rate']:.5f} |")
    lines += ["", "## Primary contrasts", "", "| Contrast | First-only | Second-only | Net | Exact McNemar p | Delta (pp) | Paired 95% CI | Cluster 95% CI |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label in ("P1_REPLICATION", "P2_GROUP_ASYMMETRY", "P3_GROUP_STRUCTURE", "P4_PRACTICAL_BASELINE"):
        row = analysis["comparisons"][label]
        lines.append(f"| {label}: {row['first_method']} vs {row['second_method']} | {row['first_only_wins']} | {row['second_only_wins']} | {row['net_wins']} | {row['exact_two_sided_mcnemar_p']:.6g} | {row['success_delta_percentage_points']:.1f} | [{row['paired_bootstrap_ci'][0]:.3f}, {row['paired_bootstrap_ci'][1]:.3f}] | [{row['task_cluster_bootstrap_ci'][0]:.3f}, {row['task_cluster_bootstrap_ci'][1]:.3f}] |")
    lines += ["", "## Per-task primary results", "", "| Task | Fresh | FO20 | Reverse20 | FullOld20 | hard h16 | Interaction I |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in analysis["per_task"]:
        lines.append(f"| {row['task_id']} | {row['FRESH_successes']}/14 | {row['FO20_successes']}/14 | {row['REVERSE20_successes']}/14 | {row['FULL_OLD20_successes']}/14 | {row['HARD_H16_successes']}/14 | {row['interaction_count']} ({row['interaction_rate']:.3f}) |")
    interaction = analysis["interaction"]
    lines += ["", "## Descriptive 2×2 interaction", "", f"I = FO20 − FRESH − FULL_OLD20 + REVERSE20 = {interaction['count']} successes, or {interaction['rate']:.3f} per primary block.", "", "## Leave-one-task-out deltas", "", "| Omitted task | FO20−Fresh | FO20−Reverse20 | FO20−hard h16 |", "|---:|---:|---:|---:|"]
    loto = analysis["leave_one_task_out_stability"]
    for index, task_id in enumerate(analysis["tasks"]):
        lines.append(f"| {task_id} | {loto['FO20_minus_FRESH'][index]:.3f} | {loto['FO20_minus_REVERSE20'][index]:.3f} | {loto['FO20_minus_HARD_H16'][index]:.3f} |")
    lines += ["", "## Protocol interpretation", "", "The four dense conditions query ACT at every controller step. HARD_H16 queries only at q=0,16,32,… and executes A_q[t−q] from the newest query; its query rate is reported from observed policy queries divided by environment steps. No task0 or newly claimed held-out task is included in the primary aggregate.", "", f"{analysis['decision_reason']}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    all_episodes, outcomes = collect(protocol, args.output_root)
    tasks = [int(x) for x in protocol["cohort"]["primary_task_ids"]]
    states = [int(x) for x in protocol["cohort"]["state_ids"]]

    comparisons: dict[str, dict[str, Any]] = {}
    for index, (label, first, second) in enumerate(COMPARISONS):
        comparisons[label] = comparison_statistics(
            outcomes, tasks, states, first, second, 20260920 + index, 20260940 + index
        )
    comparisons["REVERSE20_VS_FRESH"] = comparison_statistics(
        outcomes, tasks, states, "REVERSE20", "FRESH", 20260930, 20260950
    )
    decision_label, reason = decision(comparisons)
    per_task = make_per_task(outcomes, tasks, states)
    primary_table = method_table(all_episodes, tasks, states)
    interaction_count = sum(row["interaction_count"] for row in per_task)
    leave_one_out = {
        "FO20_minus_FRESH": comparisons["P1_REPLICATION"]["leave_one_task_out"],
        "FO20_minus_REVERSE20": comparisons["P2_GROUP_ASYMMETRY"]["leave_one_task_out"],
        "FO20_minus_HARD_H16": comparisons["P4_PRACTICAL_BASELINE"]["leave_one_task_out"],
    }
    analysis = {
        "schema_version": 1,
        "decision": decision_label,
        "decision_reason": reason,
        "primary_scope": "LIBERO Object tasks 1-9, states 20,21,22,23,27,31,34,35,38,39,44,45,47,48",
        "tasks": tasks,
        "states": states,
        "primary_blocks": 126,
        "primary_episodes_per_method": 126,
        "primary_total_episodes": 630,
        "historical_context": {"FO20": "80/126", "Fresh": "53/126"},
        "primary_table": primary_table,
        "comparisons": comparisons,
        "per_task": per_task,
        "interaction": {"formula": "FO20 - FRESH - FULL_OLD20 + REVERSE20", "count": interaction_count, "rate": interaction_count / 126},
        "leave_one_task_out_stability": leave_one_out,
        "source": "new repaired outcomes only; no historical outcome splicing",
    }
    write_json(args.output_root / "analysis.json", analysis)
    write_per_task(args.output_root / "per_task.csv", per_task)
    write_condition_shards(args.output_root, all_episodes, tasks, states)
    (args.output_root / "report.md").write_text(report_text(analysis), encoding="utf-8")
    print(json.dumps({"decision": decision_label, "analysis": str((args.output_root / 'analysis.json').resolve())}))


if __name__ == "__main__":
    main()
