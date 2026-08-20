#!/usr/bin/env python3
"""Build the final Gate-2B report data from merged rollout aggregates."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_conditioned_oracle.phase_oracle import (  # noqa: E402
    BASE_GLOBAL_HORIZON,
    BASE_GROUP_HORIZONS,
    BOOTSTRAP_SEED,
    HORIZONS,
    PHASES,
    paired_task_bootstrap,
    summarize_config,
    static_task_results,
    choose_best_phase_configs,
    load_static_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-results", type=Path, required=True)
    parser.add_argument("--combined-results", type=Path, required=True)
    parser.add_argument("--selected-configs", type=Path, required=True)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def initial_configs() -> list[dict[str, Any]]:
    from experiments.phase_conditioned_oracle.phase_oracle import build_initial_configs

    return build_initial_configs()


def main() -> None:
    args = parse_args()
    configs = initial_configs()
    initial_cache = json.loads(args.initial_results.read_text(encoding="utf-8"))
    selected_configs = json.loads(args.selected_configs.read_text(encoding="utf-8"))
    combined_cache = json.loads(args.combined_results.read_text(encoding="utf-8"))
    task_ids = list(range(10))
    baseline = load_static_baseline(args.baseline_json, task_ids)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summarized = {
        config["name"]: summarize_config(config["name"], config, initial_cache[config["name"]], rng)
        for config in configs
    }
    selected_global, selected_group = choose_best_phase_configs(
        configs=configs, cache=initial_cache, task_ids=task_ids
    )
    selected_by_name = {config["name"]: config for config in selected_configs}
    combined_summaries = {
        config["name"]: summarize_config(
            config["name"], config, combined_cache[config["name"]], rng
        )
        for config in selected_configs
    }
    static_group = static_task_results(baseline, task_ids, group=True)
    static_global = static_task_results(baseline, task_ids, group=False)
    combined_global = combined_summaries["phase_oracle_global_combined"]
    combined_group = combined_summaries["phase_oracle_group_combined"]
    summary = {
        "status": "complete",
        "starting_commit": "ba20d60adf8d5f03f1b1d3615266f81b788805c7",
        "checkpoint": "/home/thor/projects/checkpoints/zeromidnight_act_libero_object",
        "dataset": "/home/thor/datasets/libero_object_25_08_23_lerobotv2.1",
        "task_coverage": {
            "task_ids": task_ids,
            "task_count": 10,
            "episodes_per_task": {str(task_id): len(baseline[str(task_id)]["state_ids"]) for task_id in task_ids},
            "total_episodes": 230,
            "seed_rule": "seed = 1000 + init_state_id",
        },
        "protocol": {
            "phase_definition": "environment_step / env._max_episode_steps; early < 1/3, middle < 2/3, late otherwise",
            "phase_decision_timing": "phase horizon applies when a group commitment expires and its next chunk is queried",
            "phase_transition": "no forced query at phase boundary; existing commitment continues",
            "global_horizons": list(HORIZONS),
            "group_horizon_grid": [[arm, gripper] for arm, gripper in itertools.product(HORIZONS, repeat=2)],
            "baseline_outside_target_phase": {"global": BASE_GLOBAL_HORIZON, "group": BASE_GROUP_HORIZONS},
            "oracle_name": "phase-conditioned oracle horizon",
            "training": False,
            "videos": False,
        },
        "action_groups": {"arm": "action[0:6]", "gripper": "action[6]"},
        "metrics": {
            "success_rate": "macro mean of per-task success rates; pooled rate also reported",
            "environment_steps": "sum and macro mean over runtime rollouts",
            "policy_queries": "sum and macro mean of frozen ACT full-chunk calls",
            "query_rate": "policy queries / environment steps",
            "confidence_intervals": "task-level bootstrap 95% CI, 20,000 draws, seed 20260819; per-task Wilson intervals",
        },
        "phase_global_table": {
            phase: {
                "selected": selected_global[phase],
                "candidate_summaries": [
                    summarized[name]
                    for name in sorted(summarized)
                    if summarized[name]["kind"] == "global_phase_candidate"
                    and summarized[name]["target_phase"] == phase
                ],
            }
            for phase in PHASES
        },
        "phase_group_table": {
            phase: {
                "selected": selected_group[phase],
                "candidate_summaries": [
                    summarized[name]
                    for name in sorted(summarized)
                    if summarized[name]["kind"] == "group_phase_candidate"
                    and summarized[name]["target_phase"] == phase
                ],
            }
            for phase in PHASES
        },
        "selected_configs": selected_configs,
        "combined_oracles": {"global": combined_global, "group": combined_group},
        "static_baselines": {
            "global_h16": {
                "task_results": static_global,
                "macro_success_rate": float(np.mean([row["success_rate"] for row in static_global.values()])),
            },
            "group_arm4_grip16": {
                "task_results": static_group,
                "macro_success_rate": float(np.mean([row["success_rate"] for row in static_group.values()])),
            },
        },
        "comparisons": {
            "phase_oracle_global_vs_static_global": paired_task_bootstrap(
                combined_global["task_results"], static_global, task_ids
            ),
            "phase_oracle_group_vs_static_group": paired_task_bootstrap(
                combined_group["task_results"], static_group, task_ids
            ),
        },
        "runtime": {"python": platform.python_version()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
