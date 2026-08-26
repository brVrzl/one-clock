#!/usr/bin/env python3
"""Frozen paired analysis for the 300-cell RoboTwin DCTA development run."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METHODS = ("NATIVE_ACT", "SHARED_DYNAMIC_AGG", "DCTA")
PRIMARY_CONTRASTS = (
    ("DCTA", "NATIVE_ACT"),
    ("DCTA", "SHARED_DYNAMIC_AGG"),
    ("SHARED_DYNAMIC_AGG", "NATIVE_ACT"),
)
GROUPS = ("left_arm", "left_gripper", "right_arm", "right_gripper")


def paired_contrast(
    outcomes: dict[tuple[str, int, str], int],
    tasks: list[str],
    treatment: str,
    comparator: str,
    generator: np.random.Generator,
) -> dict[str, Any]:
    task_deltas = {}
    wins = losses = ties = 0
    for task in tasks:
        differences = []
        seeds = sorted({seed for candidate_task, seed, _ in outcomes if candidate_task == task})
        for seed in seeds:
            difference = outcomes[(task, seed, treatment)] - outcomes[(task, seed, comparator)]
            differences.append(difference)
            wins += difference > 0
            losses += difference < 0
            ties += difference == 0
        task_deltas[task] = float(np.mean(differences))
    draws = np.asarray(
        [np.mean([task_deltas[tasks[index]] for index in generator.integers(0, len(tasks), len(tasks))]) for _ in range(10000)]
    )
    return {
        "contrast": f"{treatment} - {comparator}",
        "pooled_paired_difference": float(np.mean(list(task_deltas.values()))),
        "wins": int(wins),
        "losses": int(losses),
        "ties": int(ties),
        "task_deltas": task_deltas,
        "task_cluster_bootstrap_95_interval": np.quantile(draws, [0.025, 0.975]).tolist(),
        "leave_one_task_out": {
            omitted: float(np.mean([value for task, value in task_deltas.items() if task != omitted]))
            for omitted in tasks
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text())
    root = args.result_root / schedule["cells_sha256"]
    technical_paths = list((root / "technical").glob("*.json"))
    outcome_paths = list((root / "sealed_outcomes").glob("*.json"))
    technical = [json.loads(path.read_text()) for path in technical_paths]
    if len(technical) != 300 or len(outcome_paths) != 300 or any(item.get("state") != "COMPLETE" for item in technical):
        raise RuntimeError("DCTA development matrix is not technically complete")
    cells = {cell["cell_id"]: cell for cell in schedule["cells"]}
    raw_outcomes = [json.loads(path.read_text()) for path in outcome_paths]
    if {item["cell_id"] for item in raw_outcomes} != set(cells):
        raise RuntimeError("DCTA outcomes do not match the frozen schedule")
    outcomes = {
        (cells[item["cell_id"]]["task"], cells[item["cell_id"]]["robotwin_seed"], cells[item["cell_id"]]["method"]): int(item["success"])
        for item in raw_outcomes
    }
    tasks = schedule["tasks"]
    table = []
    for task in tasks:
        for method in METHODS:
            values = [value for (candidate_task, _, candidate_method), value in outcomes.items() if candidate_task == task and candidate_method == method]
            table.append({"task": task, "method": method, "success": int(sum(values)), "n": len(values), "rate": float(np.mean(values))})
    pooled = {
        method: {
            "success": int(sum(row["success"] for row in table if row["method"] == method)),
            "n": 100,
        }
        for method in METHODS
    }
    for value in pooled.values():
        value["rate"] = value["success"] / value["n"]
    generator = np.random.default_rng(20270828)
    contrasts = {
        f"{treatment}_vs_{comparator}": paired_contrast(
            outcomes, tasks, treatment, comparator, generator
        )
        for treatment, comparator in PRIMARY_CONTRASTS
    }
    age_values = {method: {group: [] for group in GROUPS} for method in METHODS[1:]}
    for item in technical:
        if item["method"] == "NATIVE_ACT":
            continue
        with gzip.open(item["provenance_path"], "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                for group in GROUPS:
                    age_values[item["method"]][group].append(record["effective_source_age_seconds"][group])
    temporal_ages = {
        method: {
            group: {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
            }
            for group, values in groups.items()
        }
        for method, groups in age_values.items()
    }
    result = {
        "study": schedule["study"],
        "cells_sha256": schedule["cells_sha256"],
        "completed_cells": 300,
        "task_method": table,
        "pooled": pooled,
        "contrasts": contrasts,
        "rollout_effective_source_age_seconds": temporal_ages,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# RoboTwin DCTA method-development result", "", "## Success", "", "| Task | Method | Success | Rate |", "|---|---|---:|---:|"]
    for row in table:
        lines.append(f"| `{row['task']}` | `{row['method']}` | {row['success']}/{row['n']} | {100 * row['rate']:.1f}% |")
    lines.extend(["", "## Paired contrasts", ""])
    for contrast in contrasts.values():
        low, high = contrast["task_cluster_bootstrap_95_interval"]
        lines.append(
            f"- {contrast['contrast']}: {contrast['pooled_paired_difference']:+.3f}; "
            f"W/L/T {contrast['wins']}/{contrast['losses']}/{contrast['ties']}; "
            f"task-cluster 95% interval [{low:+.3f}, {high:+.3f}]."
        )
    args.report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"pooled": pooled, "contrasts": contrasts}, indent=2))


if __name__ == "__main__":
    main()
