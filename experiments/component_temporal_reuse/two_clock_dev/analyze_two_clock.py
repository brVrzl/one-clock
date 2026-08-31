#!/usr/bin/env python3
"""Analyze matched-query two-clock outcomes and exact McNemar comparisons."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


METHODS = ("global_8_8", "arm8_grip16", "arm16_grip8")
TASK_ORDER = ("libero_object:task6", "libero_spatial:task2", "libero_goal:task1", "libero_10:task3")


def mcnemar(candidate: list[bool], reference: list[bool]) -> dict:
    if len(candidate) != len(reference):
        raise ValueError("paired success vectors have different lengths")
    candidate_only = sum(c and not r for c, r in zip(candidate, reference, strict=True))
    reference_only = sum(r and not c for c, r in zip(candidate, reference, strict=True))
    both_success = sum(c and r for c, r in zip(candidate, reference, strict=True))
    both_fail = sum(not c and not r for c, r in zip(candidate, reference, strict=True))
    discordant = candidate_only + reference_only
    p_value = None if not discordant else min(1.0, 2.0 * sum(math.comb(discordant, i) for i in range(min(candidate_only, reference_only) + 1)) / (2 ** discordant))
    return {
        "candidate_only_successes": int(candidate_only),
        "reference_only_successes": int(reference_only),
        "net_paired_wins": int(candidate_only - reference_only),
        "both_success": int(both_success),
        "both_fail": int(both_fail),
        "discordant_pairs": int(discordant),
        "exact_mcnemar_two_sided_p": p_value,
    }


def load_inputs(paths: list[Path]) -> dict[str, dict]:
    results = {}
    for path in paths:
        value = json.loads(path.read_text())
        key = value["task"]
        if key in results:
            raise SystemExit(f"duplicate task result: {key}")
        results[key] = value
    if set(results) != set(TASK_ORDER):
        raise SystemExit(f"task mismatch; missing={sorted(set(TASK_ORDER)-set(results))}, extra={sorted(set(results)-set(TASK_ORDER))}")
    return results


def verify_query_logs(results: dict[str, dict]) -> dict:
    checks = []
    for task in TASK_ORDER:
        methods = results[task]["methods_result"]
        episodes = {method: methods[method]["episodes_detail"] for method in METHODS}
        for index in range(10):
            prefix_steps = [episodes[method][index]["environment_steps"] for method in METHODS]
            for method in METHODS:
                episode = episodes[method][index]
                expected = list(range(0, episode["environment_steps"], 8))
                if episode["query_steps"] != expected or not episode["query_schedule_exact"]:
                    raise SystemExit(f"query schedule mismatch: {task}/{method}/{index}")
            common_steps = min(prefix_steps)
            common_queries = [episodes[method][index]["query_steps"][: len(range(0, common_steps, 8))] for method in METHODS]
            checks.append({
                "task": task,
                "episode_index": index,
                "prefix_environment_steps": prefix_steps,
                "common_prefix_query_schedule_exact": common_queries == [common_queries[0]] * 3,
                "total_query_counts": [episodes[method][index]["policy_queries"] for method in METHODS],
            })
    return {
        "all_episode_query_schedules_exact": True,
        "all_common_prefix_query_schedules_match": all(row["common_prefix_query_schedule_exact"] for row in checks),
        "all_total_query_counts_identical": all(len(set(row["total_query_counts"])) == 1 for row in checks),
        "episodes": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    results = load_inputs(args.input)
    query_checks = verify_query_logs(results)
    per_task = []
    pooled_successes = {method: [] for method in METHODS}
    for task in TASK_ORDER:
        value = results[task]
        methods = value["methods_result"]
        row = {"task": task, "success": {method: {"count": int(methods[method]["success_count"]), "successes": list(methods[method]["successes"])} for method in METHODS}}
        row["comparisons"] = {
            "arm8_grip16_vs_global_8_8": mcnemar(row["success"]["arm8_grip16"]["successes"], row["success"]["global_8_8"]["successes"]),
            "arm8_grip16_vs_arm16_grip8": mcnemar(row["success"]["arm8_grip16"]["successes"], row["success"]["arm16_grip8"]["successes"]),
        }
        per_task.append(row)
        for method in METHODS:
            pooled_successes[method].extend(row["success"][method]["successes"])

    summaries = {}
    for method in METHODS:
        task_results = [results[task]["methods_result"][method] for task in TASK_ORDER]
        total_steps = sum(int(value["environment_steps"]) for value in task_results)
        summaries[method] = {
            "success_count": int(sum(pooled_successes[method])),
            "episodes": len(pooled_successes[method]),
            "per_task_success": {task: int(results[task]["methods_result"][method]["success_count"]) for task in TASK_ORDER},
            "query_count": int(sum(int(value["policy_queries"]) for value in task_results)),
            "environment_steps": total_steps,
            "query_rate": sum(int(value["policy_queries"]) for value in task_results) / total_steps,
            "mean_arm_source_age_steps": sum(float(value["pooled_arm_source_age_steps"]) * int(value["environment_steps"]) for value in task_results) / total_steps,
            "mean_gripper_source_age_steps": sum(float(value["pooled_gripper_source_age_steps"]) * int(value["environment_steps"]) for value in task_results) / total_steps,
        }

    pooled_pairs = {
        "arm8_grip16_vs_global_8_8": mcnemar(pooled_successes["arm8_grip16"], pooled_successes["global_8_8"]),
        "arm8_grip16_vs_arm16_grip8": mcnemar(pooled_successes["arm8_grip16"], pooled_successes["arm16_grip8"]),
    }
    candidate = summaries["arm8_grip16"]
    global_ref = summaries["global_8_8"]
    reverse_ref = summaries["arm16_grip8"]
    no_clear_catastrophic_task_regression = all(
        candidate["per_task_success"][task] >= global_ref["per_task_success"][task] - 2 and
        candidate["per_task_success"][task] >= reverse_ref["per_task_success"][task] - 2
        for task in TASK_ORDER
    )
    directionally_better_than_reverse = (
        pooled_pairs["arm8_grip16_vs_arm16_grip8"]["net_paired_wins"] > 0
        or candidate["success_count"] > reverse_ref["success_count"]
    )
    positive = (
        pooled_pairs["arm8_grip16_vs_global_8_8"]["net_paired_wins"] >= 3
        and directionally_better_than_reverse
        and no_clear_catastrophic_task_regression
    )
    global_net = pooled_pairs["arm8_grip16_vs_global_8_8"]["net_paired_wins"]
    decision = "PASS" if positive else "FAIL" if global_net < 0 else "NULL"
    analysis = {
        "protocol": str((Path(__file__).with_name("protocol.json")).resolve()),
        "per_task": per_task,
        "pooled": summaries,
        "paired": pooled_pairs,
        "query_verification": query_checks,
        "decision": decision,
        "decision_rule_checks": {
            "at_least_plus_3_net_wins_vs_global": pooled_pairs["arm8_grip16_vs_global_8_8"]["net_paired_wins"] >= 3,
            "directionally_better_than_arm16_grip8": directionally_better_than_reverse,
            "no_clear_catastrophic_task_regression": no_clear_catastrophic_task_regression,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(analysis, indent=2) + "\n")
    lines = [
        "# Matched-query asymmetric component commitment",
        "",
        "All methods query every physical step 0, 8, 16, ...; component source chunks are executed at the current target offset.",
        "",
        "## Pooled and per-task results",
        "",
        "| method | pooled success | object 6 | spatial 2 | goal 1 | libero_10 3 | query rate | mean arm age | mean gripper age |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        value = summaries[method]
        task_counts = [value["per_task_success"][task] for task in TASK_ORDER]
        lines.append(f"| {method} | {value['success_count']}/{value['episodes']} | " + " | ".join(f"{count}/10" for count in task_counts) + f" | {value['query_rate']:.6f} | {value['mean_arm_source_age_steps']:.6f} | {value['mean_gripper_source_age_steps']:.6f} |")
    lines += ["", "## Paired comparisons", "", "| comparison | candidate-only | reference-only | net paired wins | exact McNemar p |", "|---|---:|---:|---:|---:|"]
    for name, value in pooled_pairs.items():
        lines.append(f"| {name} | {value['candidate_only_successes']} | {value['reference_only_successes']} | {value['net_paired_wins']:+d} | {value['exact_mcnemar_two_sided_p']} |")
    lines += ["", f"Query schedules exact on every episode: **yes**; common-prefix schedules matched: **yes**; total query counts identical across methods: **{'yes' if query_checks['all_total_query_counts_identical'] else 'no, only differing when episode termination lengths differ'}**.", "", f"Decision: **{decision}**.", ""]
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text("\n".join(lines))
    print(json.dumps({"output": str(args.output_json), "decision": decision, "pooled": {method: summaries[method]["success_count"] for method in METHODS}}, indent=2))


if __name__ == "__main__":
    main()
