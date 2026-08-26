#!/usr/bin/env python3
"""Summarize technical completion without opening sealed outcome files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text())
    root = args.result_root / schedule["cells_sha256"] / "cells"
    states = Counter()
    complete_by_task = Counter()
    complete_by_method = Counter()
    reruns = 0
    provenance_failures = 0
    for cell in schedule["cells"]:
        cell_key = hashlib.sha256(cell["cell_id"].encode()).hexdigest()
        status_path = root / cell_key / "technical_status.json"
        if not status_path.exists():
            states["NOT_STARTED"] += 1
            continue
        status = json.loads(status_path.read_text())
        state = status.get("state", "UNKNOWN")
        states[state] += 1
        if state == "COMPLETE":
            complete_by_task[cell["task"]] += 1
            complete_by_method[cell["method"]] += 1
            reruns += int(status.get("attempt", 1) > 1)
        if state == "PROVENANCE_FAILURE":
            provenance_failures += 1
    summary = {
        "planned_cells": schedule["cell_count"],
        "technical_states": dict(states),
        "complete_by_task": dict(complete_by_task),
        "complete_by_method": dict(complete_by_method),
        "cells_completed_after_retry": reruns,
        "provenance_failures": provenance_failures,
        "sealed_outcomes_opened": False,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
