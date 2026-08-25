#!/usr/bin/env python3
"""Build the frozen 600-cell RoboTwin exploratory schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


TASKS = (
    "beat_block_hammer",
    "click_alarmclock",
    "dump_bin_bigbin",
    "handover_block",
    "open_laptop",
)
METHODS = (
    "NATIVE_ACT",
    "NEWEST",
    "FULL_OLD_1S",
    "FO_1S",
    "GRIPPER_HOLD",
    "GRIPPER_EMA_1S",
)
RANDOMIZATION_SEED = 20270825
PENDING_ARTIFACT = "PENDING_ARTIFACT_COMPLETION"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def checkpoint_contract(manifest: dict[str, Any], task: str) -> dict[str, Any]:
    checkpoint = manifest["checkpoints"][task]
    if task == "open_laptop" and checkpoint["checkpoint_sha256"] == PENDING_ARTIFACT:
        return {
            "checkpoint_path": checkpoint["checkpoint_path"],
            "checkpoint_sha256": PENDING_ARTIFACT,
            "checkpoint_identity": checkpoint["prospective_checkpoint_identity"],
            "config_sha256": checkpoint["config_sha256"],
            "training_seed": checkpoint["training_seed"],
            "final_epoch": checkpoint["prospective_final_epoch"],
            "artifact_state": PENDING_ARTIFACT,
        }
    if checkpoint["status"] != "COMPLETE":
        raise RuntimeError(f"{task} checkpoint is not complete")
    path = Path(checkpoint["checkpoint_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = file_sha256(path)
    if digest != checkpoint["checkpoint_sha256"]:
        raise RuntimeError(f"{task} checkpoint hash differs from canonical manifest")
    return {
        "checkpoint_path": str(path),
        "checkpoint_sha256": digest,
        "config_sha256": checkpoint["config_sha256"],
        "training_seed": checkpoint["training_seed"],
        "final_epoch": checkpoint["final_epoch"],
        "artifact_state": "COMPLETE",
    }


def balanced_method_orders(task_index: int, block_count: int) -> list[list[str]]:
    """Randomized cyclic orders, balanced by position in complete six-block groups."""

    rng = random.Random(RANDOMIZATION_SEED + task_index)
    orders = []
    while len(orders) < block_count:
        base = list(METHODS)
        rng.shuffle(base)
        shifts = list(range(len(METHODS)))
        rng.shuffle(shifts)
        for shift in shifts:
            orders.append(base[shift:] + base[:shift])
            if len(orders) == block_count:
                break
    return orders


def build_schedule(
    eligible: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    checkpoints = {task: checkpoint_contract(manifest, task) for task in TASKS}
    cells = []
    global_order = 0
    for task_index, task in enumerate(TASKS):
        seeds = eligible["tasks"][task]["eligible_seeds"]
        if len(seeds) != 20 or len(set(seeds)) != 20:
            raise RuntimeError(f"{task} does not have 20 unique eligible seeds")
        method_orders = balanced_method_orders(task_index, len(seeds))
        for eligible_seed_index, (seed, method_order) in enumerate(
            zip(seeds, method_orders)
        ):
            for within_block_order, method in enumerate(method_order):
                identity = {
                    "task": task,
                    "eligible_seed_index": eligible_seed_index,
                    "robotwin_seed": seed,
                    "method": method,
                    "checkpoint_sha256": checkpoints[task]["checkpoint_sha256"],
                    "config_sha256": checkpoints[task]["config_sha256"],
                }
                cell_id = (
                    f"{task}__eligible-{eligible_seed_index:02d}__seed-{seed}"
                    f"__method-{method}__ckpt-{identity['checkpoint_sha256']}"
                    f"__config-{identity['config_sha256']}"
                )
                cells.append(
                    {
                        "cell_id": cell_id,
                        **identity,
                        "task_index": task_index,
                        "within_seed_block_order": within_block_order,
                        "within_task_run_order": eligible_seed_index * len(METHODS)
                        + within_block_order,
                        "global_schedule_order": global_order,
                    }
                )
                global_order += 1

    cell_ids = [cell["cell_id"] for cell in cells]
    if len(cells) != 600 or len(set(cell_ids)) != 600:
        raise RuntimeError("schedule must contain exactly 600 unique cells")
    for task in TASKS:
        task_cells = [cell for cell in cells if cell["task"] == task]
        if len(task_cells) != 120:
            raise RuntimeError(f"{task} does not have 120 cells")
        for seed_index in range(20):
            methods = {
                cell["method"]
                for cell in task_cells
                if cell["eligible_seed_index"] == seed_index
            }
            if methods != set(METHODS):
                raise RuntimeError(f"{task} seed block {seed_index} is incomplete")

    return {
        "schema_version": 1,
        "study": "RoboTwin one-clock sealed exploratory pilot",
        "design": "randomized complete block; block = task x eligible seed",
        "task_order": list(TASKS),
        "methods": list(METHODS),
        "eligible_seeds_source": "robotwin_exploratory_eligible_seeds.json",
        "eligible_seed_rule": (
            "first 20 official-expert-eligible seeds in ascending order from 100000, "
            "independently per task"
        ),
        "randomization_seed": RANDOMIZATION_SEED,
        "randomization_algorithm": (
            "Within each task, randomly permute the six methods and randomized cyclic "
            "rotations in groups of six seed blocks; execute each seed block in its "
            "frozen method order."
        ),
        "checkpoint_contracts": checkpoints,
        "method_contract": {
            "physical_source_age_seconds": 1.0,
            "chunk_length": 50,
            "same_current_decision_target": "old candidate = chunk_q[t-q]",
            "q_star_tie_break": "more recent q",
            "no_age_error_tolerance_or_fallback": True,
            "no_history_warmup": "NEWEST",
            "gripper_hold_initialization": "fresh grippers at decision 0",
            "gripper_ema_tau_seconds": 1.0,
            "gripper_ema_initialization": "fresh grippers at decision 0",
        },
        "failure_policy": {
            "valid_policy_failure": "retain as a completed cell; never replace seed",
            "infrastructure_failure": (
                "rerun only the exact same cell, up to two retries after the initial attempt"
            ),
            "persistent_infrastructure_failure": "TECHNICAL_INVALIDATION",
            "provenance_assertion_failure": (
                "halt pilot and declare TECHNICAL_INVALIDATION; no automatic retry"
            ),
            "method_specific_seed_replacement": False,
        },
        "outcome_sealing": (
            "Per-cell outcomes are written separately from technical status and provenance; "
            "no success aggregation or inspection before all 600 cells complete."
        ),
        "cell_count": len(cells),
        "cells_sha256": canonical_sha256(cells),
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eligible-seeds", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    eligible = json.loads(args.eligible_seeds.read_text())
    manifest = json.loads(args.manifest.read_text())
    schedule = build_schedule(eligible, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(schedule, indent=2) + "\n")
    print(f"cells={schedule['cell_count']} sha256={schedule['cells_sha256']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
