#!/usr/bin/env python3
"""Deterministically merge disjoint component-reuse result shards."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", default=[])
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    merged = json.loads(args.canonical.read_text())
    sources = [args.canonical, *args.shard]
    for shard_path in args.shard:
        shard = json.loads(shard_path.read_text())
        for task_key, task_result in shard.get("tasks", {}).items():
            tasks = merged.setdefault("tasks", {})
            if task_key not in tasks:
                tasks[task_key] = task_result
                continue
            existing = tasks[task_key]
            overlap = set(existing.get("conditions", {})) & set(task_result.get("conditions", {}))
            if overlap:
                raise SystemExit(f"duplicate task-condition blocks in {shard_path}: {task_key} {sorted(overlap)}")
            for field, value in task_result.items():
                if field != "conditions":
                    existing.setdefault(field, value)
            existing.setdefault("conditions", {}).update(task_result.get("conditions", {}))

    task_keys = [f"{task['suite']}:task{task['task_id']}" for task in protocol["tasks"]]
    condition_names = [condition["name"] for condition in protocol["conditions"]]
    expected = {(task_key, condition) for task_key in task_keys for condition in condition_names}
    found = {(task_key, condition) for task_key, task in merged.get("tasks", {}).items() for condition in task.get("conditions", {})}
    if found - expected:
        raise SystemExit(f"unexpected task-condition blocks: {sorted(found - expected)}")
    if expected - found:
        raise SystemExit(f"missing task-condition blocks: {sorted(expected - found)}")
    if len(found) != 80:
        raise SystemExit(f"expected exactly 80 unique blocks, found {len(found)}")
    merged["merged_from"] = [str(path.resolve()) for path in sources]
    merged["merged_at"] = time.time()
    atomic_json(args.canonical, merged)
    print(json.dumps({"canonical": str(args.canonical), "unique_blocks": len(found), "sources": [str(path) for path in sources]}, indent=2))


if __name__ == "__main__":
    main()
