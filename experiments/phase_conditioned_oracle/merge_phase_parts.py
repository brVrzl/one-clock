#!/usr/bin/env python3
"""Merge task/state-partitioned Gate-2B oracle aggregates and select maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_conditioned_oracle.phase_oracle import (  # noqa: E402
    build_initial_configs,
    choose_best_phase_configs,
    combined_config,
)


PHASES = ("early", "middle", "late")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def combine_task_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) == 1:
        return rows[0]
    task_ids = {int(row["task_id"]) for row in rows}
    if len(task_ids) != 1:
        raise ValueError(f"cannot merge different tasks: {task_ids}")
    state_ids = []
    success_vector = []
    episodes = 0
    successes = 0
    environment_steps = 0
    policy_queries = 0
    phase_steps = {phase: 0 for phase in PHASES}
    phase_queries = {phase: 0 for phase in PHASES}
    phase_reached = {phase: 0 for phase in PHASES}
    first = rows[0]
    for row in sorted(rows, key=lambda value: min(value["state_ids"])):
        state_ids.extend(int(value) for value in row["state_ids"])
        success_vector.extend(bool(value) for value in row["success_vector"])
        episodes += int(row["episodes"])
        successes += int(row["successes"])
        environment_steps += int(row["environment_steps"])
        policy_queries += int(row["policy_queries"])
        for phase in PHASES:
            source = row["phase_summary"][phase]
            phase_steps[phase] += int(source["phase_environment_steps"])
            phase_queries[phase] += int(source["phase_policy_queries"])
            phase_reached[phase] += int(source["episodes_reaching_phase"])
    if len(state_ids) != len(set(state_ids)):
        raise ValueError(f"duplicate states while merging task {first['task_id']}")
    order = sorted(range(len(state_ids)), key=lambda index: state_ids[index])
    state_ids = [state_ids[index] for index in order]
    success_vector = [success_vector[index] for index in order]
    phase_summary = {}
    for phase in PHASES:
        phase_summary[phase] = {
            "episodes_reaching_phase": phase_reached[phase],
            "phase_environment_steps": phase_steps[phase],
            "phase_policy_queries": phase_queries[phase],
            "phase_query_rate": phase_queries[phase] / phase_steps[phase] if phase_steps[phase] else None,
            "mean_phase_steps_per_episode": phase_steps[phase] / episodes,
            "mean_phase_queries_per_episode": phase_queries[phase] / episodes,
        }
    return {
        "task_id": int(first["task_id"]),
        "task_name": first["task_name"],
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "success_rate_ci95": first["success_rate_ci95"] if len(rows) == 1 else None,
        "success_vector": success_vector,
        "state_ids": state_ids,
        "environment_steps": environment_steps,
        "policy_queries": policy_queries,
        "policy_query_rate": policy_queries / environment_steps,
        "mean_environment_steps": environment_steps / episodes,
        "mean_policy_queries": policy_queries / episodes,
        "phase_summary": phase_summary,
    }


def main() -> None:
    args = parse_args()
    merged: dict[str, dict[str, dict[str, Any]]] = {}
    for part in args.parts:
        cache_path = part / "config_results.json"
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        for config_name, task_rows in cache.items():
            destination = merged.setdefault(config_name, {})
            for task_id, row in task_rows.items():
                if task_id in destination:
                    destination[task_id] = combine_task_rows([destination[task_id], row])
                else:
                    destination[task_id] = row
    initial_configs = build_initial_configs()
    expected_names = {config["name"] for config in initial_configs}
    if set(merged) != expected_names:
        missing = sorted(expected_names - set(merged))
        extra = sorted(set(merged) - expected_names)
        raise ValueError(f"configuration coverage mismatch; missing={missing[:5]}, extra={extra[:5]}")
    for name, task_rows in merged.items():
        if set(task_rows) != {str(task_id) for task_id in range(10)}:
            raise ValueError(f"{name} does not cover tasks 0..9")
        if int(task_rows["0"]["episodes"]) != 50 or any(int(task_rows[str(task_id)]["episodes"]) != 20 for task_id in range(1, 10)):
            raise ValueError(f"{name} does not preserve 50/20 state coverage")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config_results.json").write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    selected_global, selected_group = choose_best_phase_configs(
        configs=initial_configs,
        cache=merged,
        task_ids=list(range(10)),
    )
    selected = [
        combined_config("phase_oracle_global_combined", "global", selected_global),
        combined_config("phase_oracle_group_combined", "group", selected_group),
    ]
    (args.output_dir / "selected_configs.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"merged_configs": len(merged), "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
