#!/usr/bin/env python3
"""Materialize the already-frozen, outcome-blind H_temp dev values.

This script is intentionally a file/provenance check, not a new score
definition.  It reads only the completed offline audit artifact and never
opens a rollout, intervention, or success file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OFFLINE = ROOT.parent / "group_temporal_memory_offline"
DEFAULT_SOURCE = OFFLINE / "h_temp_frozen.json"
DEFAULT_OUTPUT = ROOT / "h_temp_development_frozen.json"
DEV_TASKS = (
    "libero_object:task3",
    "libero_spatial:task0",
    "libero_goal:task2",
    "libero_10:task3",
)


def freeze(source: Path, output: Path) -> dict:
    data = json.loads(source.read_text())
    if data.get("status") != "frozen_before_closed_loop_comparison":
        raise RuntimeError("source H_temp artifact is not frozen before closed-loop comparison")
    if not data.get("definition", {}).get("frozen_before_closed_loop"):
        raise RuntimeError("source H_temp definition is not marked frozen before closed-loop comparison")
    values = {row["task_key"]: float(row["H_temp"]) for row in data.get("task_values", [])}
    missing = [task for task in DEV_TASKS if task not in values]
    if missing:
        raise RuntimeError(f"frozen H_temp artifact is missing development tasks: {missing}")
    frozen = {
        "status": "frozen_before_group_memory_outcomes",
        "source": str(source.resolve()),
        "source_status": data["status"],
        "definition": data["definition"],
        "task_values": [{"task_key": task, "H_temp": values[task]} for task in DEV_TASKS],
        "outcome_blind": True,
        "outcomes_loaded": False,
        "executor_use": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(frozen, indent=2) + "\n")
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frozen = freeze(args.source, args.output)
    print(json.dumps({"status": frozen["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()

