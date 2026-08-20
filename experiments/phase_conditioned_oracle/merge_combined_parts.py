#!/usr/bin/env python3
"""Merge partitioned rollouts for the two selected phase-oracle maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_conditioned_oracle.merge_phase_parts import combine_task_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", nargs="+", type=Path, required=True)
    parser.add_argument("--configs-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = json.loads(args.configs_json.read_text(encoding="utf-8"))
    expected_names = {str(config["name"]) for config in expected}
    merged: dict[str, dict[str, dict[str, Any]]] = {}
    for part in args.parts:
        cache = json.loads((part / "config_results.json").read_text(encoding="utf-8"))
        if set(cache) != expected_names:
            raise ValueError(f"{part} has unexpected combined configurations: {sorted(cache)}")
        for config_name, task_rows in cache.items():
            destination = merged.setdefault(config_name, {})
            for task_id, row in task_rows.items():
                if task_id in destination:
                    destination[task_id] = combine_task_rows([destination[task_id], row])
                else:
                    destination[task_id] = row
    expected_tasks = {str(task_id) for task_id in range(10)}
    for name, task_rows in merged.items():
        if set(task_rows) != expected_tasks:
            raise ValueError(f"{name} does not cover tasks 0..9")
        if int(task_rows["0"]["episodes"]) != 50 or any(
            int(task_rows[str(task_id)]["episodes"]) != 20 for task_id in range(1, 10)
        ):
            raise ValueError(f"{name} does not preserve 50/20 state coverage")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"configs": sorted(merged), "tasks": 10, "episodes": 230}, indent=2))


if __name__ == "__main__":
    main()
