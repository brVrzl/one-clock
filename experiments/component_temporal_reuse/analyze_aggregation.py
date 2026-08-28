#!/usr/bin/env python3
"""Merge and analyze fixed temporal aggregation shards."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean


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
        "absolute_success_difference": mean([int(c) - int(r) for c, r in zip(candidate, reference)]),
        "paired_contingency": {
            "both_fail": both_fail,
            "reference_only_success": reference_only,
            "candidate_only_success": candidate_only,
            "both_success": both_success,
        },
        "exact_mcnemar_two_sided_p": p,
    }


def load_and_merge(paths: list[Path], protocol: dict) -> tuple[dict, list[str]]:
    merged: dict[str, dict] = {}
    methods: list[str] | None = None
    for path in paths:
        data = json.loads(path.read_text())
        shard_methods = list(data["methods"])
        if methods is None:
            methods = shard_methods
        elif shard_methods != methods:
            raise SystemExit(f"method mismatch in {path}")
        for task_key, task in data.get("tasks", {}).items():
            if task_key in merged:
                raise SystemExit(f"duplicate aggregation task in {path}: {task_key}")
            merged[task_key] = task
    expected = {f"{task['suite']}:task{int(task['task_id'])}" for task in protocol["tasks"]}
    if set(merged) != expected:
        raise SystemExit(f"aggregation task mismatch; missing={sorted(expected-set(merged))}, extra={sorted(set(merged)-expected)}")
    if methods is None:
        raise SystemExit("no aggregation shards")
    return merged, methods


def analyze(merged: dict[str, dict], methods: list[str], protocol: dict) -> dict:
    task_spec = {f"{task['suite']}:task{int(task['task_id'])}": task for task in protocol["tasks"]}
    per_task = []
    for task_key, task in merged.items():
        row = {"task_key": task_key, "suite": task_spec[task_key]["suite"], "task_id": task_spec[task_key]["task_id"], "task_name": task["task_name"], "methods": {}}
        fresh = task["methods"]["fresh"]["successes"]
        for method in methods:
            result = task["methods"][method]
            values = {
                "successes": result["successes"],
                "success_count": result["success_count"],
                "episodes": result["episodes"],
                "success_rate": result["success_rate"],
                "policy_queries_per_environment_step": result["policy_queries_per_environment_step"],
                "mean_arm_source_age_steps": result["mean_arm_source_age_steps"],
                "mean_gripper_source_age_steps": result["mean_gripper_source_age_steps"],
                "successful_episode_completion_steps_mean": mean(result["completion_steps_successful"]) if result["completion_steps_successful"] else None,
            }
            if method != "fresh":
                values["vs_fresh"] = mcnemar(result["successes"], fresh)
            row["methods"][method] = values
        row["component_vs_official_act"] = mcnemar(
            task["methods"]["component_arm_fresh_gripper_act"]["successes"],
            task["methods"]["official_act_m001"]["successes"],
        )
        per_task.append(row)

    suites = list(dict.fromkeys(task["suite"] for task in protocol["tasks"]))
    aggregates = {}
    for scope in [*suites, "all_tasks"]:
        rows = [row for row in per_task if scope == "all_tasks" or row["suite"] == scope]
        summary = {}
        for method in methods:
            metrics = [row["methods"][method] for row in rows]
            successes = [value for row in rows for value in row["methods"][method]["successes"]]
            summary[method] = {
                "task_macro_success_rate": mean(value["success_rate"] for value in metrics),
                "pooled_successes": sum(successes),
                "pooled_episodes": len(successes),
                "pooled_success_rate": sum(successes) / len(successes),
                "task_macro_queries_per_environment_step": mean(value["policy_queries_per_environment_step"] for value in metrics),
                "task_macro_mean_arm_source_age_steps": mean(value["mean_arm_source_age_steps"] for value in metrics),
                "task_macro_mean_gripper_source_age_steps": mean(value["mean_gripper_source_age_steps"] for value in metrics),
            }
        vs_fresh = {}
        for method in methods:
            if method == "fresh":
                continue
            comparisons = [row["methods"][method]["vs_fresh"] for row in rows]
            candidate = [value for row in rows for value in row["methods"][method]["successes"]]
            reference = [value for row in rows for value in row["methods"]["fresh"]["successes"]]
            vs_fresh[method] = {
                "task_macro_absolute_success_difference": mean(value["absolute_success_difference"] for value in comparisons),
                "pooled": mcnemar(candidate, reference),
            }
        component = [row["component_vs_official_act"] for row in rows]
        candidate = [value for row in rows for value in row["methods"]["component_arm_fresh_gripper_act"]["successes"]]
        reference = [value for row in rows for value in row["methods"]["official_act_m001"]["successes"]]
        aggregates[scope] = {
            "condition_metrics": summary,
            "vs_fresh": vs_fresh,
            "component_vs_official_act": {"task_macro_absolute_success_difference": mean(value["absolute_success_difference"] for value in component), "pooled": mcnemar(candidate, reference)},
        }
    return {
        "analysis_status": "complete_aggregation_followup",
        "protocol": {"tasks": len(per_task), "episodes_per_task": int(protocol["environment"]["episodes_per_task"]), "methods": methods},
        "per_task": sorted(per_task, key=lambda row: row["task_key"]),
        "aggregates": aggregates,
    }


def report(analysis: dict, output: Path) -> None:
    methods = analysis["protocol"]["methods"]
    lines = ["# Temporal aggregation follow-up", "", "Fixed coefficients, frozen eight-task cohort, paired seeds 1000–1009. Task-macro values are primary; pooled values are descriptive.", "", "## Per-task success", ""]
    headers = ["task", *methods, "component vs ACT"]
    lines += ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in analysis["per_task"]:
        cells = [row["task_key"]]
        for method in methods:
            cells.append(f"{row['methods'][method]['success_count']}/{row['methods'][method]['episodes']}")
        c = row["component_vs_official_act"]
        cells.append(f"{c['candidate_successes']}-{c['reference_successes']} ({c['absolute_success_difference']:+.2f})")
        lines.append("| " + " | ".join(cells) + " |")
    for scope, summary in analysis["aggregates"].items():
        lines += ["", f"## {scope}", "", "| method | task macro | pooled | queries/step | arm age | gripper age |", "|---|---:|---:|---:|---:|---:|"]
        for method, values in summary["condition_metrics"].items():
            lines.append(f"| {method} | {values['task_macro_success_rate']:.3f} | {values['pooled_successes']}/{values['pooled_episodes']} ({values['pooled_success_rate']:.1%}) | {values['task_macro_queries_per_environment_step']:.3f} | {values['task_macro_mean_arm_source_age_steps']:.2f} | {values['task_macro_mean_gripper_source_age_steps']:.2f} |")
        lines += ["", "### Paired comparisons vs fresh", "", "| method | task-macro delta | pooled delta | candidate-only/reference-only | p |", "|---|---:|---:|---:|---:|"]
        for method, values in summary["vs_fresh"].items():
            p = values["pooled"]
            cont = p["paired_contingency"]
            lines.append(f"| {method} | {values['task_macro_absolute_success_difference']:+.3f} | {p['absolute_success_difference']:+.3f} | {cont['candidate_only_success']}/{cont['reference_only_success']} | {p['exact_mcnemar_two_sided_p'] if p['exact_mcnemar_two_sided_p'] is not None else 'n/a'} |")
        p = summary["component_vs_official_act"]["pooled"]
        cont = p["paired_contingency"]
        lines += ["", f"Component-aware minus official full-action ACT: task-macro {summary['component_vs_official_act']['task_macro_absolute_success_difference']:+.3f}; pooled {p['absolute_success_difference']:+.3f}; candidate-only/reference-only {cont['candidate_only_success']}/{cont['reference_only_success']}; exact McNemar p={p['exact_mcnemar_two_sided_p'] if p['exact_mcnemar_two_sided_p'] is not None else 'n/a'}."]
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    merged, methods = load_and_merge(args.input, protocol)
    analysis = analyze(merged, methods, protocol)
    args.output_json.write_text(json.dumps(analysis, indent=2) + "\n")
    report(analysis, args.output_report)
    print(json.dumps({"status": analysis["analysis_status"], "tasks": len(merged), "methods": methods}, indent=2))


if __name__ == "__main__":
    main()
