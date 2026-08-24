#!/usr/bin/env python3
"""Generate the frozen Gate-3B task-state and treatment-order schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gate3b_composition import METHODS, SOURCE_AGE_TICKS


STARTING_SHA = "eb4f6bfeb40a9d1444d3fb1d17c841601ca29a76"
STATE_SELECTION_SEED = 20260827
METHOD_ORDER_SEED = 20260828
EPISODE_SEED_BASE = 320000
GATE3A2_STATES = (0, 7, 11, 13, 25, 30, 36, 41, 42, 43)


def build_schedule() -> dict[str, object]:
    available_states = [state for state in range(20, 50) if state not in GATE3A2_STATES]
    sampled_states = np.random.default_rng(STATE_SELECTION_SEED).choice(
        available_states, size=10, replace=False
    )
    selected_states = sorted(int(state) for state in sampled_states)
    order_rng = np.random.default_rng(METHOD_ORDER_SEED)
    runs: list[dict[str, object]] = []
    blocks: list[dict[str, object]] = []
    run_index = 0
    for task_id in range(10):
        for state_id in selected_states:
            order = [METHODS[index] for index in order_rng.permutation(len(METHODS))]
            episode_seed = EPISODE_SEED_BASE + 100 * task_id + state_id
            blocks.append(
                {
                    "task_id": task_id,
                    "state_id": state_id,
                    "episode_seed": episode_seed,
                    "method_order": order,
                }
            )
            for within_block_order, method in enumerate(order):
                runs.append(
                    {
                        "run_index": run_index,
                        "task_id": task_id,
                        "state_id": state_id,
                        "episode_seed": episode_seed,
                        "within_block_order": within_block_order,
                        "method": method,
                    }
                )
                run_index += 1
    return {
        "schema_version": 1,
        "starting_scientific_commit": STARTING_SHA,
        "state_selection": {
            "candidate_state_ids": available_states,
            "excluded_gate3a2_state_ids": list(GATE3A2_STATES),
            "sample_without_replacement": 10,
            "seed": STATE_SELECTION_SEED,
            "draw_order_before_sort": [int(state) for state in sampled_states],
            "selected_state_ids": selected_states,
            "note": "IDs sorted after sampling only for deterministic traversal readability.",
        },
        "method_order_seed": METHOD_ORDER_SEED,
        "episode_seed_rule": "320000 + 100 * task_id + state_id",
        "source_age_ticks": SOURCE_AGE_TICKS,
        "source_age_seconds": 1.0,
        "methods": list(METHODS),
        "task_ids": list(range(10)),
        "blocks": blocks,
        "runs": runs,
        "planned_blocks": len(blocks),
        "planned_episodes": len(runs),
    }


def pending_runs(
    schedule: dict[str, object],
    completed_run_indices: set[int],
    *,
    task_id: int | None = None,
    max_new_runs: int | None = None,
) -> list[dict[str, object]]:
    runs = [
        run
        for run in schedule["runs"]
        if int(run["run_index"]) not in completed_run_indices
        and (task_id is None or int(run["task_id"]) == task_id)
    ]
    return runs if max_new_runs is None else runs[:max_new_runs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_schedule(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
