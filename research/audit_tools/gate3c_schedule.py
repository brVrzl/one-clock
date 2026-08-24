#!/usr/bin/env python3
"""Generate the frozen Gate-3C randomized complete-block schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gate3c_temporal_reuse import METHODS, SOURCE_AGE_TICKS


SCIENTIFIC_PARENT = "2817411a4210b8611dc8dae5d32ec99fc6b94cf3"
STATE_SELECTION_SEED = 20260830
METHOD_ORDER_SEED = 20260831
EPISODE_SEED_BASE = 330000
SELECTED_STATES = (20, 21, 22, 23, 27, 31, 34, 35, 38, 39, 44, 45, 47, 48)
GATE3A2_STATES = (0, 7, 11, 13, 25, 30, 36, 41, 42, 43)
GATE3B_STATES = (24, 26, 28, 29, 32, 33, 37, 40, 46, 49)


def build_schedule() -> dict[str, object]:
    order_rng = np.random.default_rng(METHOD_ORDER_SEED)
    runs: list[dict[str, object]] = []
    blocks: list[dict[str, object]] = []
    for task_id in range(10):
        for state_id in SELECTED_STATES:
            order = [METHODS[index] for index in order_rng.permutation(len(METHODS))]
            episode_seed = EPISODE_SEED_BASE + 100 * task_id + state_id
            block = {
                "task_id": task_id,
                "state_id": state_id,
                "episode_seed": episode_seed,
                "method_order": order,
                "confirmatory_scope": "secondary_task0" if task_id == 0 else "primary_tasks1_to_9",
            }
            blocks.append(block)
            for within_block_order, method in enumerate(order):
                runs.append(
                    {
                        "run_index": len(runs),
                        "task_id": task_id,
                        "state_id": state_id,
                        "episode_seed": episode_seed,
                        "within_block_order": within_block_order,
                        "method": method,
                    }
                )
    return {
        "schema_version": 1,
        "scientific_parent": SCIENTIFIC_PARENT,
        "state_selection": {
            "identity_fields_only_audit": True,
            "common_genuinely_unused_task1_to_9_state_ids": list(SELECTED_STATES),
            "selected_state_ids": list(SELECTED_STATES),
            "selection_rule": "all common unused IDs because count is between 10 and 15",
            "selection_seed_reserved_but_not_used": STATE_SELECTION_SEED,
            "gate3a2_state_ids": list(GATE3A2_STATES),
            "gate3b_state_ids": list(GATE3B_STATES),
            "task0_note": "Task 0 historically used all 50 states and is secondary only.",
        },
        "method_order_seed": METHOD_ORDER_SEED,
        "episode_seed_rule": "330000 + 100 * task_id + state_id",
        "source_age_ticks": SOURCE_AGE_TICKS,
        "source_age_seconds": 1.0,
        "methods": list(METHODS),
        "primary_task_ids": list(range(1, 10)),
        "secondary_task_ids": [0],
        "task_ids": list(range(10)),
        "blocks": blocks,
        "runs": runs,
        "planned_blocks": len(blocks),
        "planned_episodes": len(runs),
    }


def pending_runs(
    schedule: dict[str, object], completed_run_indices: set[int], max_new_runs: int | None = None
) -> list[dict[str, object]]:
    runs = [
        run for run in schedule["runs"] if int(run["run_index"]) not in completed_run_indices
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
