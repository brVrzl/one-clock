#!/usr/bin/env python3
"""Generate the frozen Gate-4A2 Spatial randomized complete-block schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gate3c_temporal_reuse import METHODS, SOURCE_AGE_TICKS


SCIENTIFIC_PARENT = "36bebdace1ffbd8d36bacc061feb146cd55f894a"
STATE_SELECTION_SEED = 20260825
METHOD_ORDER_SEED = 20260826
EPISODE_SEED_BASE = 340000
COMMON_VALID_STATES = tuple(range(50))
STATE_SELECTION_RNG_RESULT = (40, 15, 13, 47, 37, 24, 19, 1, 31, 21)
SELECTED_STATES = tuple(sorted(STATE_SELECTION_RNG_RESULT))


def build_schedule() -> dict[str, object]:
    order_rng = np.random.default_rng(METHOD_ORDER_SEED)
    runs: list[dict[str, object]] = []
    blocks: list[dict[str, object]] = []
    for task_id in range(10):
        for state_id in SELECTED_STATES:
            order = [METHODS[index] for index in order_rng.permutation(len(METHODS))]
            episode_seed = EPISODE_SEED_BASE + 100 * task_id + state_id
            blocks.append(
                {
                    "task_id": task_id,
                    "state_id": state_id,
                    "episode_seed": episode_seed,
                    "method_order": order,
                    "confirmatory_scope": "primary",
                }
            )
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
            "outcome_blind": True,
            "common_valid_state_ids": list(COMMON_VALID_STATES),
            "selection_seed": STATE_SELECTION_SEED,
            "rng_result_unsorted": list(STATE_SELECTION_RNG_RESULT),
            "selected_state_ids_sorted": list(SELECTED_STATES),
            "selection_rule": "numpy.random.default_rng(20260825).choice(common_ids, 10, replace=False)",
        },
        "method_order_seed": METHOD_ORDER_SEED,
        "episode_seed_rule": "340000 + 100 * task_id + state_id",
        "source_age_ticks": SOURCE_AGE_TICKS,
        "control_frequency_hz": 20.0,
        "source_age_seconds": 1.0,
        "methods": list(METHODS),
        "primary_task_ids": list(range(10)),
        "task_ids": list(range(10)),
        "blocks": blocks,
        "runs": runs,
        "planned_blocks": len(blocks),
        "planned_episodes": len(runs),
    }


def pending_runs(
    schedule: dict[str, object], completed_run_indices: set[int]
) -> list[dict[str, object]]:
    return [
        run for run in schedule["runs"] if int(run["run_index"]) not in completed_run_indices
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_schedule(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
