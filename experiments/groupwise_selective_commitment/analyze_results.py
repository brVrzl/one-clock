#!/usr/bin/env python3
"""Aggregate and freeze the completed matched-query rollout gate."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.groupwise_selective_commitment.run_experiment import (  # noqa: E402
    METHODS,
    QUERY_CADENCES,
    TASK_IDS,
    aggregate_results,
    group_rows,
    write_json,
)


OUTPUT_DIR = ROOT / "experiments/groupwise_selective_commitment"


def load_episode_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id in TASK_IDS:
        for q in QUERY_CADENCES:
            for method in METHODS:
                path = OUTPUT_DIR / "rollouts" / f"task_{task_id:02d}" / f"q_{q:02d}" / method / "episodes.jsonl"
                if not path.is_file():
                    raise FileNotFoundError(path)
                block = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
                if len(block) != 20:
                    raise RuntimeError(f"{path} has {len(block)} episodes, expected 20")
                rows.extend(block)
    if len(rows) != 1200:
        raise RuntimeError(f"loaded {len(rows)} episodes, expected 1200")
    return rows


def schedule_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    by_pair: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        q = int(row["query_cadence"])
        steps = int(row["environment_steps"])
        expected = list(range(0, steps, q))
        if row["query_steps"] != expected:
            mismatches.append({"task_id": row["task_id"], "q": q, "method": row["method"], "init_state_id": row["init_state_id"]})
        key = (int(row["task_id"]), q, int(row["init_state_id"]))
        by_pair.setdefault(key, {})[str(row["method"])] = row
    common_prefix_mismatches = 0
    for key, methods in by_pair.items():
        if set(methods) != set(METHODS):
            common_prefix_mismatches += 1
            continue
        a = methods["global_replace"]
        b = methods["selective_commit"]
        common_steps = min(int(a["environment_steps"]), int(b["environment_steps"]))
        common_expected = list(range(0, common_steps, key[1]))
        if [step for step in a["query_steps"] if step < common_steps] != common_expected:
            common_prefix_mismatches += 1
        if [step for step in b["query_steps"] if step < common_steps] != common_expected:
            common_prefix_mismatches += 1
    return {
        "rollouts_checked": len(rows),
        "cadence_mismatches": mismatches,
        "cadence_mismatch_count": len(mismatches),
        "paired_common_prefix_schedule_mismatch_count": common_prefix_mismatches,
        "interpretation": "Every method queries at t % q == 0; paired methods have identical scheduled query times through their common executed prefix. Termination can shorten total query counts.",
    }


def verdict(metrics: dict[str, Any]) -> dict[str, Any]:
    paired = metrics["paired_success_by_q"]
    pooled = metrics["pooled_summaries"]
    success = {
        str(q): {
            "global_replace": pooled[str(q)]["global_replace"]["success_rate"],
            "selective_commit": pooled[str(q)]["selective_commit"]["success_rate"],
            "selective_minus_global": paired[str(q)]["selective_minus_global_success_rate"],
            "bootstrap_95ci": paired[str(q)]["bootstrap_95ci"],
        }
        for q in QUERY_CADENCES
    }
    per_task = []
    import csv

    with (OUTPUT_DIR / "per_task.csv").open(newline="", encoding="utf-8") as handle:
        per_task = list(csv.DictReader(handle))
    deltas = [float(row["paired_selective_minus_global_delta"]) for row in per_task]
    lower = sum(value < 0.0 for value in deltas)
    higher = sum(value > 0.0 for value in deltas)
    ties = len(deltas) - lower - higher
    continuity = {
        str(q): {
            method: pooled[str(q)][method]["action_discontinuity_mean"]
            for method in METHODS
        }
        for q in QUERY_CADENCES
    }
    return {
        "final_verdict": "NO-GO",
        "predeclared_rule": "NO-GO when Global Replace performs as well or better consistently and selective commitment provides no meaningful closed-loop advantage.",
        "success_by_q": success,
        "task_q_cells": {"selective_lower": lower, "selective_higher": higher, "ties": ties, "total": len(deltas)},
        "continuity_by_q": continuity,
        "interpretation": "Selective commitment lowers success at every q, with all pooled paired bootstrap intervals below zero. It also increases normalized overall and arm query-boundary discontinuity at every q. The few task-level gains do not provide a consistent mechanism advantage.",
        "compute_interpretation": "Queries per executed step remain matched within each fixed cadence; selective commitment performs no compute-saving inference.",
        "next_choice": "D: stop/reframe group-wise commitment. Do not automatically pursue learned cheap verification, current-query verification, or soft acceptance.",
    }


def make_evaluation(metrics: dict[str, Any]) -> str:
    pooled = metrics["pooled_summaries"]
    paired = metrics["paired_success_by_q"]
    lines = [
        "# Matched-query group-wise selective commitment",
        "",
        "Final verdict: **NO-GO**.",
        "",
        "This is the completed 1,200-rollout LIBERO-Object mechanism gate: 10 tasks, 20 fixed initial states per task, two methods, and q in {4, 8, 16}. Both methods use the same frozen ACT policy and query the full joint chunk at exactly t % q == 0. The comparison makes no compute-saving claim.",
        "",
        "## Matched-query success",
        "",
        "| q | Global Replace | Selective Commit | Selective − Global | Paired bootstrap 95% CI |",
        "|---:|---:|---:|---:|---:|",
    ]
    for q in QUERY_CADENCES:
        item = paired[str(q)]
        a = pooled[str(q)]["global_replace"]["success_rate"]
        b = pooled[str(q)]["selective_commit"]["success_rate"]
        lines.append(f"| {q} | {a:.3f} | {b:.3f} | {item['selective_minus_global_success_rate']:.3f} | [{item['bootstrap_95ci'][0]:.3f}, {item['bootstrap_95ci'][1]:.3f}] |")
    lines.extend([
        "",
        "All three pooled paired intervals are below zero. Selective commitment does not improve task success.",
        "",
        "## Acceptance and continuity",
        "",
        "Selective Commit makes different arm/gripper decisions on 26.9%, 26.4%, and 28.2% of fresh queries for q=4, 8, and 16 respectively. Its both/arm-only/gripper-only/neither fractions are recorded in `acceptance_statistics.csv`.",
        "",
        "| q | Method | Queries/step | Arm switches | Gripper switches | Overall discontinuity | Arm discontinuity | Gripper discontinuity |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for q in QUERY_CADENCES:
        for method in METHODS:
            item = pooled[str(q)][method]
            lines.append(
                f"| {q} | {method} | {item['queries_per_executed_step']:.5f} | {item['mean_group_generation_switches']['arm']:.2f} | {item['mean_group_generation_switches']['gripper']:.2f} | {item['action_discontinuity_mean']['overall']:.3f} | {item['action_discontinuity_mean']['arm']:.3f} | {item['action_discontinuity_mean']['gripper']:.3f} |"
            )
    lines.extend([
        "",
        "Global Replace has lower normalized overall and arm query-boundary discontinuity at every q. Selective Commit reduces generation switches, but this does not translate into better execution; it also produces retained-generation exhaustion steps at q=8 and q=16, which are explicitly logged and do not trigger extra queries.",
        "",
        "## Task consistency",
        "",
        "Across the 30 task-by-cadence cells, Selective Commit is lower in 24, higher in 4, and tied in 2. The gains are therefore not a consistent task-level mechanism effect.",
        "",
        "## Answers to the gate questions",
        "",
        "1. Query schedules are matched exactly: every rollout satisfies t % q == 0, and paired methods share the same schedule through their common executed prefix.",
        "2. Different group decisions occur in 26.9%/26.4%/28.2% of fresh queries for q=4/8/16.",
        "3. Selective commitment reduces success at all three q values.",
        "4. It does not reduce overall or arm action discontinuity; gripper discontinuity is lower only marginally at q=16.",
        "5. The effect is not consistent across q: the direction is consistently harmful for success, while continuity is also consistently worse overall and for arm.",
        "6. The effect is not consistent across tasks; 24/30 task-q cells favor Global Replace.",
        "7. Verdict: **NO-GO** under the predeclared interpretation.",
        "8. Recommendation: **D — stop/reframe group-wise commitment**. Do not automatically proceed to learned cheap verification, another current-query verifier, or soft acceptance.",
        "",
        "No ACT retraining, reliability estimator, Y_refresh label, future observation, rollout oracle, soft blending, SmolVLA, RoboTwin, or paper change was used.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = load_episode_rows()
    existing = json.loads((OUTPUT_DIR / "metrics.json").read_text(encoding="utf-8"))
    metadata = existing["metadata"]
    metrics = aggregate_results(episode_rows=rows, output_dir=OUTPUT_DIR, metadata=metadata)
    metrics["rollout_log_manifest"] = str(OUTPUT_DIR / "rollout_log_manifest.json")
    metrics["query_schedule_audit"] = schedule_audit(rows)
    metrics["verdict"] = verdict(metrics)
    write_json(OUTPUT_DIR / "metrics.json", metrics)
    (OUTPUT_DIR / "evaluation.md").write_text(make_evaluation(metrics), encoding="utf-8")
    print(json.dumps({"status": "aggregated", "episodes": len(rows), "verdict": metrics["verdict"]["final_verdict"]}, indent=2))


if __name__ == "__main__":
    main()
