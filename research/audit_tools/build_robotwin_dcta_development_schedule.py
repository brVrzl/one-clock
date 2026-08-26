#!/usr/bin/env python3
"""Freeze the paired 300-cell RoboTwin DCTA method-development schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


TASKS = (
    "beat_block_hammer",
    "click_alarmclock",
    "dump_bin_bigbin",
    "handover_block",
    "open_laptop",
)
METHODS = ("NATIVE_ACT", "SHARED_DYNAMIC_AGG", "DCTA")


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eligible-seeds", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shared-gate", type=Path, required=True)
    parser.add_argument("--dcta-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    eligible = json.loads(args.eligible_seeds.read_text())
    manifest = json.loads(args.manifest.read_text())
    generator = np.random.default_rng(20270827)
    checkpoints = {}
    cells = []
    run_order = 0
    for task in TASKS:
        entry = manifest["checkpoints"][task]
        checkpoints[task] = {
            "path": entry["checkpoint_path"],
            "sha256": entry["checkpoint_sha256"],
        }
        seeds = eligible["tasks"][task]["eligible_seeds"]
        if len(seeds) != 20:
            raise RuntimeError(f"expected 20 eligible seeds for {task}")
        for eligible_index, seed in enumerate(seeds):
            methods = list(METHODS)
            generator.shuffle(methods)
            for method in methods:
                cells.append(
                    {
                        "cell_id": f"{task}__eligible-{eligible_index:02d}__seed-{seed}__method-{method}",
                        "task": task,
                        "eligible_seed_index": eligible_index,
                        "robotwin_seed": seed,
                        "method": method,
                        "run_order": run_order,
                    }
                )
                run_order += 1
    if len(cells) != 300 or len({cell["cell_id"] for cell in cells}) != 300:
        raise RuntimeError("DCTA development schedule is not 300 unique cells")
    output = {
        "study": "RoboTwin DCTA method-development exploratory rollout",
        "tasks": list(TASKS),
        "methods": list(METHODS),
        "episodes_per_task_method": 20,
        "seed_source": str(args.eligible_seeds),
        "checkpoints": checkpoints,
        "gates": {
            "SHARED_DYNAMIC_AGG": {"path": str(args.shared_gate), "sha256": sha256(args.shared_gate)},
            "DCTA": {"path": str(args.dcta_gate), "sha256": sha256(args.dcta_gate)},
        },
        "cells": cells,
    }
    canonical = json.dumps(cells, sort_keys=True, separators=(",", ":")).encode()
    output["cells_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(output["cells_sha256"])


if __name__ == "__main__":
    main()
