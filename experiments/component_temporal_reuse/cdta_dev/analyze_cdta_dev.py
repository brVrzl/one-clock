#!/usr/bin/env python3
"""Validate and summarize the frozen 200-episode CDTA ACT development panel."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
METHODS = [
    "fresh",
    "official_act_m001",
    "cogact_full_alpha01",
    "matched_shared_a16_alpha03_beta003",
    "cdta_a16_alpha03_beta003",
]
CDTA = METHODS[-1]
MATCHED = METHODS[-2]


def discordance(candidate: list[bool], reference: list[bool]) -> dict[str, int]:
    candidate_only = sum(a and not b for a, b in zip(candidate, reference, strict=True))
    reference_only = sum(b and not a for a, b in zip(candidate, reference, strict=True))
    return {
        "candidate_only": candidate_only,
        "reference_only": reference_only,
        "net_wins": candidate_only - reference_only,
    }


def main() -> None:
    protocol = json.loads((ROOT / "protocol.json").read_text())
    expected_ids = protocol["environment"]["initial_state_ids"]
    expected_tasks = {f"{x['suite']}:task{x['task_id']}" for x in protocol["tasks"]}
    task_rows: dict[str, dict] = {}

    for path in sorted(RESULTS.glob("*.json")):
        payload = json.loads(path.read_text())
        task = payload["task"]
        if task not in expected_tasks or not payload.get("finished_at"):
            raise RuntimeError(f"incomplete or unexpected result: {path}")
        if payload["methods"] != METHODS or set(payload["methods_result"]) != set(METHODS):
            raise RuntimeError(f"method drift in {path}")
        row = {"success_counts": {}, "mean_source_ages": {}}
        for method in METHODS:
            result = payload["methods_result"][method]
            if result["requested_initial_state_ids"] != expected_ids:
                raise RuntimeError(f"requested state drift for {task}/{method}")
            if result["actual_initial_state_ids"] != expected_ids:
                raise RuntimeError(f"actual state drift for {task}/{method}")
            if len(result["successes"]) != 10 or result["episodes"] != 10:
                raise RuntimeError(f"episode-count drift for {task}/{method}")
            row["success_counts"][method] = result["success_count"]
            row["mean_source_ages"][method] = {
                "arm": result["mean_arm_source_age_steps"],
                "gripper": result["mean_gripper_source_age_steps"],
            }
        cdta = payload["methods_result"][CDTA]["successes"]
        row["cdta_vs_matched"] = discordance(cdta, payload["methods_result"][MATCHED]["successes"])
        row["cdta_vs_fresh"] = discordance(cdta, payload["methods_result"]["fresh"]["successes"])
        task_rows[task] = row

    if set(task_rows) != expected_tasks:
        raise RuntimeError(f"missing tasks: {sorted(expected_tasks - set(task_rows))}")

    pooled = {method: sum(row["success_counts"][method] for row in task_rows.values()) for method in METHODS}
    matched_pair = {
        key: sum(row["cdta_vs_matched"][key] for row in task_rows.values())
        for key in ("candidate_only", "reference_only", "net_wins")
    }
    fresh_pair = {
        key: sum(row["cdta_vs_fresh"][key] for row in task_rows.values())
        for key in ("candidate_only", "reference_only", "net_wins")
    }
    nonworse_tasks = sum(
        row["success_counts"][CDTA] >= row["success_counts"][MATCHED]
        for row in task_rows.values()
    )
    max_task_fresh_net_loss = max(
        row["cdta_vs_fresh"]["reference_only"] - row["cdta_vs_fresh"]["candidate_only"]
        for row in task_rows.values()
    )
    gates = {
        "matched_net_wins_at_least_3": matched_pair["net_wins"] >= 3,
        "nonworse_on_at_least_3_of_4_tasks": nonworse_tasks >= 3,
        "fresh_pooled_deficit_at_most_2": pooled["fresh"] - pooled[CDTA] <= 2,
        "no_task_fresh_net_loss_reaches_2": max_task_fresh_net_loss < 2,
    }
    gates["advance_to_blind"] = all(gates.values())
    analysis = {
        "protocol": str((ROOT / "protocol.json").resolve()),
        "validated_task_state_pairing": True,
        "episodes": 200,
        "task_rows": task_rows,
        "pooled_success_counts": pooled,
        "cdta_vs_matched_paired": matched_pair,
        "cdta_vs_fresh_paired": fresh_pair,
        "nonworse_tasks_vs_matched": nonworse_tasks,
        "gates": gates,
        "decision": "advance_to_blind" if gates["advance_to_blind"] else "stop_cdta_blind_panel",
    }
    (ROOT / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")

    header = "| task | Fresh | ACT ensemble | CogACT-style | matched shared | CDTA-16 | CDTA vs matched | CDTA vs Fresh |"
    lines = [
        "# CDTA-16 ACT development result",
        "",
        "Frozen 4-task, 10-state, 5-method panel. All methods used explicit initial-state IDs 10--19 and identical environment seeds.",
        "",
        header,
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, row in sorted(task_rows.items()):
        c = row["success_counts"]
        vm = row["cdta_vs_matched"]
        vf = row["cdta_vs_fresh"]
        lines.append(
            f"| {task} | {c['fresh']}/10 | {c['official_act_m001']}/10 | "
            f"{c['cogact_full_alpha01']}/10 | {c[MATCHED]}/10 | {c[CDTA]}/10 | "
            f"{vm['candidate_only']}/{vm['reference_only']} (net {vm['net_wins']:+d}) | "
            f"{vf['candidate_only']}/{vf['reference_only']} (net {vf['net_wins']:+d}) |"
        )
    lines.extend([
        "",
        "## Pooled gate",
        "",
        f"Successes: Fresh {pooled['fresh']}/40; ACT ensemble {pooled['official_act_m001']}/40; "
        f"CogACT-style {pooled['cogact_full_alpha01']}/40; matched shared {pooled[MATCHED]}/40; CDTA-16 {pooled[CDTA]}/40.",
        "",
        f"CDTA versus matched shared: {matched_pair['candidate_only']}/{matched_pair['reference_only']} discordant pairs "
        f"(net {matched_pair['net_wins']:+d}). CDTA versus Fresh: {fresh_pair['candidate_only']}/{fresh_pair['reference_only']} "
        f"(net {fresh_pair['net_wins']:+d}).",
        "",
        f"Advance decision: **{str(gates['advance_to_blind']).upper()}**. The component-decoupling primary gate failed because "
        f"the paired net advantage over the matched shared control was {matched_pair['net_wins']:+d}, below the frozen +3 threshold. "
        "The Fresh safeguards and task-direction safeguard passed. The predeclared 800-episode CDTA blind panel should not start.",
        "",
        "This development result supports the age/window control, which nearly matched Fresh, but it does not establish a closed-loop benefit from component decoupling.",
    ])
    (ROOT / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"analysis": str(ROOT / "analysis.json"), "report": str(ROOT / "report.md"), "decision": analysis["decision"]}, indent=2))


if __name__ == "__main__":
    main()
