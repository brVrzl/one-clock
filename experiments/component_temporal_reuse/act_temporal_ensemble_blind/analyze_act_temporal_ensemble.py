#!/usr/bin/env python3
"""Analyze canonical ACT temporal-ensemble blind task outputs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    expected = {f"{task['suite']}:task{int(task['task_id'])}" for task in protocol["tasks"]}
    tasks = {}
    for path in args.input:
        payload = json.loads(path.read_text())
        for key, value in payload["tasks"].items():
            if key in tasks:
                raise SystemExit(f"duplicate task result: {key}")
            tasks[key] = value
    if set(tasks) != expected:
        raise SystemExit(f"task result mismatch; missing={sorted(expected-set(tasks))}, extra={sorted(set(tasks)-expected)}")

    rows = []
    by_suite = defaultdict(lambda: {"successes": 0, "episodes": 0, "queries": 0, "steps": 0})
    pooled = {"successes": 0, "episodes": 0, "queries": 0, "steps": 0}
    for key in sorted(tasks):
        result = tasks[key]
        episodes = result["episodes_detail"]
        if len(episodes) != 10 or result["query_every_environment_step"] is not True:
            raise SystemExit(f"invalid episode/query records for {key}")
        row = {"task": key, "success_count": int(result["success_count"]), "episodes": 10, "query_rate": float(result["query_rate"]), "query_every_environment_step": True}
        rows.append(row)
        suite = result["suite"]
        for target in (by_suite[suite], pooled):
            target["successes"] += int(result["success_count"])
            target["episodes"] += 10
            target["queries"] += int(result["policy_queries"])
            target["steps"] += int(result["environment_steps"])
    suites = {}
    for suite, value in sorted(by_suite.items()):
        suites[suite] = {**value, "success_rate": value["successes"] / value["episodes"], "query_rate": value["queries"] / value["steps"]}
    pooled_summary = {**pooled, "success_rate": pooled["successes"] / pooled["episodes"], "query_rate": pooled["queries"] / pooled["steps"]}
    analysis = {
        "protocol": str(args.protocol.resolve()),
        "implementation": "LeRobot 0.6.2 ACTPolicy.select_action -> ACTTemporalEnsembler.update",
        "temporal_ensemble_coefficient": float(protocol["policy"]["temporal_ensemble_coefficient"]),
        "query_every_environment_step": True,
        "per_task": rows,
        "per_suite": suites,
        "pooled": pooled_summary,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(analysis, indent=2) + "\n")
    lines = [
        "# Canonical ACT temporal-ensemble blind baseline",
        "",
        f"LeRobot 0.6.2 `ACTPolicy.select_action` with `ACTTemporalEnsembler`, coefficient `{analysis['temporal_ensemble_coefficient']}`, effective `n_action_steps=1`.",
        "",
        "## Per-task success",
        "",
        "| task | success | query rate |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['task']} | {row['success_count']}/{row['episodes']} | {row['query_rate']:.6f} |")
    lines += ["", "## Per-suite pooled success", "", "| suite | success | query rate |", "|---|---:|---:|"]
    for suite, value in suites.items():
        lines.append(f"| {suite} | {value['successes']}/{value['episodes']} | {value['query_rate']:.6f} |")
    lines += ["", f"Pooled: **{pooled_summary['successes']}/{pooled_summary['episodes']}**; query rate **{pooled_summary['query_rate']:.6f}**; query every environment step: **yes**.", ""]
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text("\n".join(lines))
    print(json.dumps({"output": str(args.output_json), "success": f"{pooled_summary['successes']}/{pooled_summary['episodes']}", "query_rate": pooled_summary["query_rate"]}, indent=2))


if __name__ == "__main__":
    main()
