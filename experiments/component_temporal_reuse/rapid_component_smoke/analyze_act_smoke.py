#!/usr/bin/env python3
"""Analyze paired rapid ACT component-wise aggregation smoke outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean


VARIANTS = (
    "groupwise_similarity",
    "groupwise_similarity_age",
    "groupwise_similarity_age_gripper_vote",
)
REFERENCE = "cogact_shared_full_action"


def mcnemar(candidate: list[bool], reference: list[bool]) -> dict:
    if len(candidate) != len(reference):
        raise ValueError("paired success vectors have different lengths")
    both_fail = reference_only = candidate_only = both_success = 0
    for candidate_success, reference_success in zip(candidate, reference):
        if candidate_success and reference_success:
            both_success += 1
        elif candidate_success:
            candidate_only += 1
        elif reference_success:
            reference_only += 1
        else:
            both_fail += 1
    discordant = candidate_only + reference_only
    p_value = None if not discordant else min(
        1.0,
        2.0
        * sum(math.comb(discordant, i) for i in range(min(candidate_only, reference_only) + 1))
        / (2**discordant),
    )
    return {
        "candidate_successes": int(sum(candidate)),
        "reference_successes": int(sum(reference)),
        "absolute_success_difference": mean(
            [int(candidate_success) - int(reference_success) for candidate_success, reference_success in zip(candidate, reference)]
        ),
        "candidate_only_success": candidate_only,
        "reference_only_success": reference_only,
        "both_success": both_success,
        "both_fail": both_fail,
        "exact_mcnemar_two_sided_p": p_value,
    }


def load_inputs(paths: list[Path], protocol: dict) -> dict[str, dict]:
    merged = {}
    for path in paths:
        result = json.loads(path.read_text())
        task_key = result["task"]
        if task_key in merged:
            raise SystemExit(f"duplicate task result: {task_key}")
        merged[task_key] = result
    expected = {
        f"{task['suite']}:task{int(task['task_id'])}" for task in protocol["tasks"]
    }
    if set(merged) != expected:
        raise SystemExit(
            f"task result mismatch; missing={sorted(expected - set(merged))}, extra={sorted(set(merged) - expected)}"
        )
    return merged


def analyze(merged: dict[str, dict], protocol: dict) -> dict:
    task_rows = []
    for task_key in sorted(merged):
        result = merged[task_key]
        methods = result["methods_result"]
        row = {
            "task_key": task_key,
            "task_name": result["task_name"],
            "methods": {
                method: {
                    "success_count": int(methods[method]["success_count"]),
                    "episodes": int(methods[method]["episodes"]),
                    "success_rate": float(methods[method]["success_rate"]),
                    "successes": list(methods[method]["successes"]),
                }
                for method in result["methods"]
            },
        }
        row["paired_vs_cogact"] = {
            method: mcnemar(
                list(methods[method]["successes"]),
                list(methods[REFERENCE]["successes"]),
            )
            for method in VARIANTS
        }
        task_rows.append(row)

    pooled = {}
    task_macro = {}
    for method in task_rows[0]["methods"]:
        candidate = [success for row in task_rows for success in row["methods"][method]["successes"]]
        task_macro[method] = mean(row["methods"][method]["success_rate"] for row in task_rows)
        pooled[method] = {
            "successes": int(sum(candidate)),
            "episodes": len(candidate),
            "success_rate": sum(candidate) / len(candidate),
        }
    paired = {
        method: mcnemar(
            [success for row in task_rows for success in row["methods"][method]["successes"]],
            [success for row in task_rows for success in row["methods"][REFERENCE]["successes"]],
        )
        for method in VARIANTS
    }
    variant_decisions = {}
    for method in VARIANTS:
        goal_row = next(row for row in task_rows if row["task_key"] == "libero_goal:task1")
        goal_candidate = goal_row["methods"][method]
        goal_reference = goal_row["methods"][REFERENCE]
        goal_paired = goal_row["paired_vs_cogact"][method]
        task_macro_delta = task_macro[method] - task_macro[REFERENCE]
        variant_decisions[method] = {
            "task_macro_delta_vs_cogact": task_macro_delta,
            "pooled_delta_vs_cogact": paired[method]["absolute_success_difference"],
            "meaningful_margin_met": task_macro_delta >= 0.20,
            "goal_candidate_successes": goal_candidate["success_count"],
            "goal_cogact_successes": goal_reference["success_count"],
            "goal_catastrophic_failure": (
                goal_candidate["success_count"] == 0
                or (
                    goal_paired["candidate_only_success"] == 0
                    and goal_paired["reference_only_success"] == goal_reference["episodes"]
                )
            ),
        }
    eligible = [
        method
        for method, decision in variant_decisions.items()
        if decision["meaningful_margin_met"] and not decision["goal_catastrophic_failure"]
    ]
    leading = (
        max(eligible, key=lambda method: (variant_decisions[method]["task_macro_delta_vs_cogact"], paired[method]["absolute_success_difference"]))
        if eligible
        else None
    )
    return {
        "analysis_status": "complete_rapid_act_smoke",
        "protocol": {
            "path": str(protocol.get("protocol_version")),
            "task_count": len(task_rows),
            "episodes_per_task": protocol["environment"]["episodes_per_task"],
            "methods": protocol["methods"],
            "policy_rng_seed": protocol["policy"]["policy_rng_seed"],
            "initial_state_ids": protocol["environment"]["initial_state_ids"],
            "environment_seeds": protocol["environment"]["seeds"],
        },
        "per_task": task_rows,
        "task_macro_success_rate": task_macro,
        "pooled_success": pooled,
        "paired_vs_cogact": paired,
        "variant_decisions": variant_decisions,
        "leading_variant": leading,
        "recommendation": (
            f"Freeze {leading} for the next evaluation." if leading else "No ACT variant met the predeclared smoke criterion; run the same smoke on the existing SmolVLA development tasks."
        ),
    }


def write_report(analysis: dict, output: Path) -> None:
    methods = analysis["protocol"]["methods"]
    lines = [
        "# Rapid ACT component-wise aggregation smoke",
        "",
        "New paired initial states and fixed policy RNG seed; task-macro values are primary and pooled values are descriptive.",
        "",
        "## Per-task success",
        "",
        "| task | " + " | ".join(methods) + " |",
        "|" + "---|" * (len(methods) + 1),
    ]
    for row in analysis["per_task"]:
        cells = [row["task_key"]]
        cells.extend(
            f"{row['methods'][method]['success_count']}/{row['methods'][method]['episodes']}"
            for method in methods
        )
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Paired variants versus CogACT shared full-action aggregation",
        "",
        "| variant | task-macro delta | pooled delta | candidate-only/reference-only | exact McNemar p | Goal |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in VARIANTS:
        decision = analysis["variant_decisions"][method]
        paired = analysis["paired_vs_cogact"][method]
        goal = f"{decision['goal_candidate_successes']}/{decision['goal_cogact_successes']}"
        lines.append(
            f"| {method} | {decision['task_macro_delta_vs_cogact']:+.3f} | {decision['pooled_delta_vs_cogact']:+.3f} | "
            f"{paired['candidate_only_success']}/{paired['reference_only_success']} | {paired['exact_mcnemar_two_sided_p']} | {goal} |"
        )
    lines += [
        "",
        f"Leading variant under the predeclared rule: `{analysis['leading_variant'] or 'none'}`.",
        "",
        analysis["recommendation"],
    ]
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    analysis = analyze(load_inputs(args.input, protocol), protocol)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(analysis, indent=2) + "\n")
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    write_report(analysis, args.output_report)
    print(json.dumps({"status": analysis["analysis_status"], "leading_variant": analysis["leading_variant"]}, indent=2))


if __name__ == "__main__":
    main()
