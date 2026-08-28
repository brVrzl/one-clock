#!/usr/bin/env python3
"""Analyze the frozen fixed-horizon blind panel after all task shards finish."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METHODS = ("fresh_h1", "fixed_h8", "fixed_h16", "native_h100")


def paired(candidate: list[bool], reference: list[bool]) -> dict[str, int]:
    candidate_only = sum(a and not b for a, b in zip(candidate, reference, strict=True))
    reference_only = sum(b and not a for a, b in zip(candidate, reference, strict=True))
    return {
        "candidate_only": candidate_only,
        "reference_only": reference_only,
        "net": candidate_only - reference_only,
    }


def main() -> None:
    protocol = json.loads((ROOT / "protocol.json").read_text())
    expected_tasks = {f"{task['suite']}:task{task['task_id']}" for task in protocol["tasks"]}
    result_paths = sorted((ROOT / "results").glob("libero_*task*.json"))
    if len(result_paths) != len(expected_tasks):
        raise RuntimeError(f"expected {len(expected_tasks)} task results, found {len(result_paths)}")

    pooled = {
        method: {"successes": [], "queries": 0, "steps": 0, "source_ages": []}
        for method in METHODS
    }
    tasks: list[dict] = []
    observed_tasks: set[str] = set()

    for path in result_paths:
        result = json.loads(path.read_text())
        task = result["task"]
        observed_tasks.add(task)
        if result.get("live_smoke"):
            raise RuntimeError(f"live-smoke file entered confirmatory analysis: {path}")
        row = {"task": task, "methods": {}}
        for method in METHODS:
            method_result = result["methods_result"][method]
            successes = [bool(value) for value in method_result["successes"]]
            if len(successes) != 10:
                raise RuntimeError(f"{task}/{method}: expected 10 episodes, found {len(successes)}")
            actual_ids = [episode["actual_initial_state_id"] for episode in method_result["episodes_detail"]]
            if actual_ids != protocol["environment"]["initial_state_ids"]:
                raise RuntimeError(f"{task}/{method}: initial-state IDs do not match the frozen protocol")
            pooled[method]["successes"].extend(successes)
            pooled[method]["queries"] += int(method_result["query_count"])
            pooled[method]["steps"] += int(method_result["environment_steps"])
            for episode in method_result["episodes_detail"]:
                pooled[method]["source_ages"].extend(episode["executed_source_age_steps"])
            row["methods"][method] = {
                "success_count": sum(successes),
                "successes": successes,
                "query_rate": method_result["query_rate"],
            }
        h16 = row["methods"]["fixed_h16"]["successes"]
        row["paired_h16_vs_h100"] = paired(h16, row["methods"]["native_h100"]["successes"])
        row["paired_h16_vs_fresh"] = paired(h16, row["methods"]["fresh_h1"]["successes"])
        row["paired_h16_vs_h8"] = paired(h16, row["methods"]["fixed_h8"]["successes"])
        tasks.append(row)

    if observed_tasks != expected_tasks:
        raise RuntimeError(f"task mismatch: expected {sorted(expected_tasks)}, observed {sorted(observed_tasks)}")

    pooled_summary = {}
    for method, values in pooled.items():
        pooled_summary[method] = {
            "success_count": sum(values["successes"]),
            "episodes": len(values["successes"]),
            "query_count": values["queries"],
            "environment_steps": values["steps"],
            "query_rate": values["queries"] / values["steps"],
            "mean_executed_source_age_steps": sum(values["source_ages"]) / len(values["source_ages"]),
        }

    h16_successes = pooled["fixed_h16"]["successes"]
    pairwise = {
        "h16_vs_h100": paired(h16_successes, pooled["native_h100"]["successes"]),
        "h16_vs_fresh": paired(h16_successes, pooled["fresh_h1"]["successes"]),
        "h16_vs_h8": paired(h16_successes, pooled["fixed_h8"]["successes"]),
    }
    taskwise_fresh_loss_ok = all(row["paired_h16_vs_fresh"]["net"] > -2 for row in tasks)
    h16_nonworse_tasks = sum(
        row["methods"]["fixed_h16"]["success_count"] >= row["methods"]["fixed_h8"]["success_count"]
        for row in tasks
    )
    gates = {
        "h16_vs_h100_paired_net_at_least_8": pairwise["h16_vs_h100"]["net"] >= 8,
        "h16_at_least_fresh_minus_4": (
            pooled_summary["fixed_h16"]["success_count"]
            >= pooled_summary["fresh_h1"]["success_count"] - 4
        ),
        "h16_at_least_h8_minus_2": (
            pooled_summary["fixed_h16"]["success_count"]
            >= pooled_summary["fixed_h8"]["success_count"] - 2
        ),
        "taskwise_fresh_paired_net_loss_below_2": taskwise_fresh_loss_ok,
        "h16_nonworse_than_h8_on_at_least_6_tasks": h16_nonworse_tasks >= 6,
        "h16_query_rate_at_most_0.075": pooled_summary["fixed_h16"]["query_rate"] <= 0.075,
    }
    analysis = {
        "protocol": str((ROOT / "protocol.json").resolve()),
        "tasks": tasks,
        "pooled": pooled_summary,
        "pairwise": pairwise,
        "h16_nonworse_than_h8_task_count": h16_nonworse_tasks,
        "gates": gates,
        "advance_to_smolvla": all(gates.values()),
    }
    (ROOT / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")

    rows = []
    for row in tasks:
        values = row["methods"]
        rows.append(
            f"| {row['task']} | {values['fresh_h1']['success_count']}/10 | "
            f"{values['fixed_h8']['success_count']}/10 | {values['fixed_h16']['success_count']}/10 | "
            f"{values['native_h100']['success_count']}/10 | "
            f"{row['paired_h16_vs_h100']['candidate_only']}/{row['paired_h16_vs_h100']['reference_only']} |"
        )
    p = pooled_summary
    gate_rows = "\n".join(
        f"| {name} | {'yes' if passed else '**no**'} |" for name, passed in gates.items()
    )
    report = f"""# Fixed-horizon ACT blind result

Frozen eight-task, 320-episode confirmatory panel on initial-state IDs 20--29. These tasks were recorded before the custom method results and received no intervention tuning.

| task | Fresh h1 | fixed h8 | fixed h16 | native h100 | h16-only / h100-only |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}
| **pooled** | **{p['fresh_h1']['success_count']}/80** | **{p['fixed_h8']['success_count']}/80** | **{p['fixed_h16']['success_count']}/80** | **{p['native_h100']['success_count']}/80** | **{pairwise['h16_vs_h100']['candidate_only']}/{pairwise['h16_vs_h100']['reference_only']} (net {pairwise['h16_vs_h100']['net']:+d})** |

## Query efficiency

| method | successes | pooled query rate | mean executed source age |
|---|---:|---:|---:|
| Fresh h1 | {p['fresh_h1']['success_count']}/80 | {p['fresh_h1']['query_rate']:.4f} | {p['fresh_h1']['mean_executed_source_age_steps']:.3f} |
| fixed h8 | {p['fixed_h8']['success_count']}/80 | {p['fixed_h8']['query_rate']:.4f} | {p['fixed_h8']['mean_executed_source_age_steps']:.3f} |
| fixed h16 | {p['fixed_h16']['success_count']}/80 | {p['fixed_h16']['query_rate']:.4f} | {p['fixed_h16']['mean_executed_source_age_steps']:.3f} |
| native h100 | {p['native_h100']['success_count']}/80 | {p['native_h100']['query_rate']:.4f} | {p['native_h100']['mean_executed_source_age_steps']:.3f} |

## Frozen gate

| criterion | pass |
|---|:---:|
{gate_rows}

Advance to SmolVLA cross-policy confirmation: **{'YES' if analysis['advance_to_smolvla'] else 'NO'}**.
"""
    (ROOT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
