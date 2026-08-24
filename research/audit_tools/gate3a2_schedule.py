#!/usr/bin/env python3
"""Generate the frozen Gate-3A2 task-state and treatment-order schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


STATE_SELECTION_SEED = 20260824
METHOD_ORDER_SEED = 20260825
EPISODE_SEED_BASE = 310000
METHODS = (
    "newest",
    "exact_act_m001",
    "cogact_a03",
    "newest_age_exp_b003",
)


def build_schedule() -> dict[str, object]:
    selected_states = sorted(
        np.random.default_rng(STATE_SELECTION_SEED).choice(50, size=10, replace=False).tolist()
    )
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
        "registration_parent": "1ce9bf0eb1443abb7452086ac85a7c4ed0ea5752",
        "state_selection": {
            "available_state_ids": list(range(50)),
            "sample_without_replacement": 10,
            "seed": STATE_SELECTION_SEED,
            "selected_state_ids": selected_states,
            "note": "IDs sorted after sampling only for execution readability.",
        },
        "method_order_seed": METHOD_ORDER_SEED,
        "episode_seed_rule": "310000 + 100 * task_id + state_id",
        "methods": list(METHODS),
        "task_ids": list(range(10)),
        "blocks": blocks,
        "runs": runs,
        "planned_blocks": len(blocks),
        "planned_episodes": len(runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_schedule(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
