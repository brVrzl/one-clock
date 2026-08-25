#!/usr/bin/env python3
"""Run the preregistered RoboTwin exploratory analysis after 600/600 completion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


FO = "FO_1S"
COMPARATORS = (
    "NEWEST",
    "NATIVE_ACT",
    "FULL_OLD_1S",
    "GRIPPER_HOLD",
    "GRIPPER_EMA_1S",
)
SIMPLE_CONTROLS = ("FULL_OLD_1S", "GRIPPER_HOLD", "GRIPPER_EMA_1S")
BOOTSTRAP_DRAWS = 10_000
ANALYSIS_SEED = 20270826


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def cell_key(cell_id: str) -> str:
    return hashlib.sha256(cell_id.encode()).hexdigest()


def task_cluster_interval(task_deltas: dict[str, float], task_order: list[str]) -> list[float]:
    values = np.asarray([task_deltas[task] for task in task_order], dtype=np.float64)
    rng = np.random.default_rng(ANALYSIS_SEED)
    sampled = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    draws = values[sampled].mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    schedule = json.loads(args.schedule.read_text())
    root = args.result_root / schedule["cells_sha256"]
    cells_root = root / "cells"
    outcomes_root = root / "sealed_outcomes"
    if len(schedule["cells"]) != 600:
        raise RuntimeError("schedule is not 600 cells")

    statuses = []
    outcomes: dict[str, bool] = {}
    for cell in schedule["cells"]:
        key = cell_key(cell["cell_id"])
        status_path = cells_root / key / "technical_status.json"
        outcome_path = outcomes_root / f"{key}.json"
        if not status_path.is_file() or not outcome_path.is_file():
            raise RuntimeError(f"incomplete scheduled cell {key}")
        status = json.loads(status_path.read_text())
        if status.get("state") != "COMPLETE":
            raise RuntimeError(f"non-complete scheduled cell {key}: {status.get('state')}")
        outcome = json.loads(outcome_path.read_text())
        if outcome.get("cell_id") != cell["cell_id"]:
            raise RuntimeError(f"outcome identity mismatch {key}")
        statuses.append(status)
        outcomes[cell["cell_id"]] = bool(outcome["success"])

    task_order = schedule["task_order"]
    methods = schedule["methods"]
    observed: dict[tuple[str, int, str], int] = {}
    for cell in schedule["cells"]:
        observed[(cell["task"], cell["eligible_seed_index"], cell["method"])] = int(
            outcomes[cell["cell_id"]]
        )

    task_method = []
    pooled = {}
    for method in methods:
        pooled_success = 0
        for task in task_order:
            count = sum(observed[(task, seed_index, method)] for seed_index in range(20))
            pooled_success += count
            task_method.append(
                {"task": task, "method": method, "success": count, "n": 20, "rate": count / 20}
            )
        pooled[method] = {"success": pooled_success, "n": 100, "rate": pooled_success / 100}

    contrasts = {}
    for comparator in COMPARATORS:
        task_deltas = {}
        wins = losses = ties = 0
        for task in task_order:
            differences = []
            for seed_index in range(20):
                fo = observed[(task, seed_index, FO)]
                other = observed[(task, seed_index, comparator)]
                difference = fo - other
                differences.append(difference)
                wins += difference == 1
                losses += difference == -1
                ties += difference == 0
            task_deltas[task] = float(np.mean(differences))
        pooled_difference = float(np.mean(list(task_deltas.values())))
        contrasts[comparator] = {
            "contrast": f"{FO} - {comparator}",
            "pooled_paired_difference": pooled_difference,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "task_deltas": task_deltas,
            "task_cluster_bootstrap_95_interval": task_cluster_interval(task_deltas, task_order),
            "leave_one_task_out": {
                omitted: float(np.mean([delta for task, delta in task_deltas.items() if task != omitted]))
                for omitted in task_order
            },
        }

    primary = contrasts["NEWEST"]
    simple_control_positive = all(
        contrasts[comparator]["pooled_paired_difference"] > 0 for comparator in SIMPLE_CONTROLS
    )
    mechanism_criteria = (
        primary["pooled_paired_difference"] > 0
        and primary["task_cluster_bootstrap_95_interval"][0] > 0
        and sum(delta > 0 for delta in primary["task_deltas"].values()) >= 3
        and simple_control_positive
    )
    native_difference = contrasts["NATIVE_ACT"]["pooled_paired_difference"]
    if mechanism_criteria and native_difference >= -0.05:
        classification = "STRONG_SIGNAL"
    elif mechanism_criteria:
        classification = "MECHANISM_SIGNAL_ONLY"
    else:
        classification = "NO_SIGNAL"

    result = {
        "study": schedule["study"],
        "schedule_hash": schedule["cells_sha256"],
        "analysis_seed": ANALYSIS_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "completed_cells": 600,
        "technical_reruns": sum(int(status.get("attempt", 1)) > 1 for status in statuses),
        "task_method": task_method,
        "pooled": pooled,
        "contrasts": contrasts,
        "classification": classification,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output_json, result)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("task", "method", "success", "n", "rate"))
        writer.writeheader()
        writer.writerows(task_method)

    lines = [
        "# RoboTwin sealed exploratory analysis",
        "",
        f"Classification: **{classification}**",
        "",
        "## Success by task and method",
        "",
        "| Task | Method | Success | Rate |",
        "|---|---|---:|---:|",
    ]
    for row in task_method:
        lines.append(
            f"| `{row['task']}` | `{row['method']}` | {row['success']}/20 | {row['rate']:.1%} |"
        )
    lines.extend(["", "## Paired contrasts", ""])
    for comparator in COMPARATORS:
        contrast = contrasts[comparator]
        interval = contrast["task_cluster_bootstrap_95_interval"]
        lines.append(
            f"- `{contrast['contrast']}`: {contrast['pooled_paired_difference']:+.3f}; "
            f"wins/losses/ties {contrast['wins']}/{contrast['losses']}/{contrast['ties']}; "
            f"task-cluster 95% interval [{interval[0]:+.3f}, {interval[1]:+.3f}]."
        )
    lines.extend(
        [
            "",
            "This is the preregistered exploratory analysis; no confirmatory p-value is claimed.",
            "",
        ]
    )
    args.output_report.write_text("\n".join(lines))
    print(json.dumps({"classification": classification, "output": str(args.output_json)}))


if __name__ == "__main__":
    main()
