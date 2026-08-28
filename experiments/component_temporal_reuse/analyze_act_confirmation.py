#!/usr/bin/env python3
"""Analyze paired, frozen ACT source-age confirmation shards."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean


CONDITIONS = ("fresh", "fo16", "full_old16", "reverse16")


def mcnemar(candidate: list[bool], reference: list[bool]) -> dict:
    both_fail = reference_only = candidate_only = both_success = 0
    for c, r in zip(candidate, reference):
        if c and r:
            both_success += 1
        elif c:
            candidate_only += 1
        elif r:
            reference_only += 1
        else:
            both_fail += 1
    discordant = candidate_only + reference_only
    p = None if not discordant else min(
        1.0,
        2.0 * sum(math.comb(discordant, i) for i in range(min(candidate_only, reference_only) + 1)) / (2**discordant),
    )
    return {
        "candidate_successes": sum(candidate),
        "reference_successes": sum(reference),
        "absolute_success_difference": mean(int(c) - int(r) for c, r in zip(candidate, reference)),
        "candidate_only_success": candidate_only,
        "reference_only_success": reference_only,
        "both_success": both_success,
        "both_fail": both_fail,
        "exact_mcnemar_two_sided_p": p,
    }


def load(paths: list[Path], protocol: dict) -> tuple[dict, list[str]]:
    merged = {}
    methods = None
    all_task_keys = {
        f"{task['suite']}:task{task['task_id']}"
        for task in protocol["task_selection"]["tasks"]
    }
    for path in paths:
        data = json.loads(path.read_text())
        shard_methods = list(data["conditions"])
        if tuple(shard_methods) != CONDITIONS:
            raise SystemExit(f"condition mismatch in {path}: {shard_methods}")
        if methods is None:
            methods = shard_methods
        for key, task in data.get("tasks", {}).items():
            if key in merged:
                raise SystemExit(f"duplicate ACT confirmation task: {key}")
            if key not in all_task_keys:
                raise SystemExit(f"unexpected ACT confirmation task: {key}")
            merged[key] = task
    if not merged:
        raise SystemExit("no completed ACT confirmation tasks")
    return merged, methods or list(CONDITIONS)


def task_row(key: str, task: dict) -> dict:
    fresh = task["methods"]["fresh"]["successes"]
    methods = {}
    for condition in CONDITIONS:
        result = task["methods"][condition]
        row = {
            "successes": result["successes"],
            "success_count": result["success_count"],
            "episodes": result["episodes"],
            "success_rate": result["success_count"] / result["episodes"],
            "policy_queries_per_environment_step": result["policy_queries_per_environment_step"],
            "mean_arm_source_age_steps": result["mean_arm_source_age_steps"],
            "mean_gripper_source_age_steps": result["mean_gripper_source_age_steps"],
            "completion_steps_successful": result["completion_steps_successful"],
            "semantic_validation_max_abs_error": result["semantic_validation_max_abs_error"],
        }
        if condition != "fresh":
            row["vs_fresh"] = mcnemar(result["successes"], fresh)
        methods[condition] = row
    return {
        "task_key": key,
        "suite": task["suite"],
        "task_id": task["task_id"],
        "task_name": task["task_name"],
        "initial_state_ids": task["methods"]["fresh"]["initial_state_ids"],
        "methods": methods,
        "fo16_vs_reverse16": mcnemar(task["methods"]["fo16"]["successes"], task["methods"]["reverse16"]["successes"]),
    }


def analyze(merged: dict) -> dict:
    rows = [task_row(key, merged[key]) for key in sorted(merged)]
    suites = list(dict.fromkeys(row["suite"] for row in rows))
    aggregates = {}
    for scope in [*suites, "all_completed_tasks"]:
        scoped = [row for row in rows if scope == "all_completed_tasks" or row["suite"] == scope]
        summary = {}
        for condition in CONDITIONS:
            values = [row["methods"][condition] for row in scoped]
            successes = [x for row in values for x in row["successes"]]
            summary[condition] = {
                "task_macro_success_rate": mean(x["success_rate"] for x in values),
                "pooled_successes": sum(successes),
                "pooled_episodes": len(successes),
                "pooled_success_rate": sum(successes) / len(successes),
                "task_macro_arm_source_age_steps": mean(x["mean_arm_source_age_steps"] for x in values),
                "task_macro_gripper_source_age_steps": mean(x["mean_gripper_source_age_steps"] for x in values),
            }
        comparisons = {}
        for condition in CONDITIONS[1:]:
            task_comparisons = [row["methods"][condition]["vs_fresh"] for row in scoped]
            candidate = [x for row in scoped for x in row["methods"][condition]["successes"]]
            reference = [x for row in scoped for x in row["methods"]["fresh"]["successes"]]
            comparisons[f"{condition}_vs_fresh"] = {
                "task_macro_absolute_success_difference": mean(x["absolute_success_difference"] for x in task_comparisons),
                "pooled": mcnemar(candidate, reference),
            }
        fo = [x for row in scoped for x in row["methods"]["fo16"]["successes"]]
        reverse = [x for row in scoped for x in row["methods"]["reverse16"]["successes"]]
        full = [x for row in scoped for x in row["methods"]["full_old16"]["successes"]]
        summary["comparisons"] = comparisons
        summary["fo16_vs_reverse16"] = {
            "task_macro_absolute_success_difference": mean(row["fo16_vs_reverse16"]["absolute_success_difference"] for row in scoped),
            "pooled": mcnemar(fo, reverse),
        }
        summary["full_old16_vs_fresh"] = mcnemar(full, [x for row in scoped for x in row["methods"]["fresh"]["successes"]])
        aggregates[scope] = summary
    return {
        "protocol": {"conditions": list(CONDITIONS), "completed_tasks": len(rows), "episodes_per_task": 10},
        "per_task": rows,
        "aggregates": aggregates,
    }


def markdown(result: dict) -> str:
    lines = [
        "# ACT temporal-source confirmation",
        "",
        "Frozen matched-query intervention on independent initial states 10–19. The native ACT baseline is separate and is not replaced by these conditions.",
        "",
        "## Per-task paired results",
        "",
        "| task | fresh | FO16 | full-old16 | reverse16 | FO16 − reverse16 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["per_task"]:
        m = row["methods"]
        c = row["fo16_vs_reverse16"]
        lines.append(f"| {row['task_key']} | {m['fresh']['success_count']}/10 | {m['fo16']['success_count']}/10 | {m['full_old16']['success_count']}/10 | {m['reverse16']['success_count']}/10 | {c['absolute_success_difference']:+.2f} ({c['candidate_only_success']}/{c['reference_only_success']}) |")
    lines += ["", "## Aggregate summaries", "", "| scope | fresh | FO16 | full-old16 | reverse16 | FO16 − reverse16 task-macro |", "|---|---:|---:|---:|---:|---:|"]
    known_scopes = {row["suite"] for row in result["per_task"]} | {"all_completed_tasks"}
    for scope, summary in result["aggregates"].items():
        if scope in known_scopes:
            lines.append(f"| {scope} | {summary['fresh']['task_macro_success_rate']:.3f} | {summary['fo16']['task_macro_success_rate']:.3f} | {summary['full_old16']['task_macro_success_rate']:.3f} | {summary['reverse16']['task_macro_success_rate']:.3f} | {summary['fo16_vs_reverse16']['task_macro_absolute_success_difference']:+.3f} |")
    lines += ["", "Pooled outcomes are descriptive; task-macro and paired task-level directions are primary.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("act_confirmation_protocol.json"))
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    merged, _ = load(args.input, protocol)
    result = analyze(merged)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    args.output_report.write_text(markdown(result))
    print(json.dumps({"completed_tasks": len(result["per_task"]), "output": str(args.output_json)}))


if __name__ == "__main__":
    main()
