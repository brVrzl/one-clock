#!/usr/bin/env python3
"""Freeze trajectory splits and paired rollout order before outcome inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


SUITES = {
    "spatial": {
        "revision": "d86c0b94922572b3b657e1d1a3d01f0952ddeb46",
        "dataset_to_benchmark": [6, 4, 5, 7, 0, 3, 8, 1, 2, 9],
    },
    "object": {
        "revision": "e1e080d7df1d0a359dff5c86c222e047549f447f",
        "dataset_to_benchmark": [9, 4, 1, 3, 0, 7, 2, 6, 5, 8],
    },
    "goal": {
        "revision": "91a97115558b5b611200a432d9c82e4f30991b60",
        "dataset_to_benchmark": [8, 9, 3, 6, 2, 5, 7, 1, 4, 0],
    },
}
METHODS = ["standard_act", "shared_dynamic", "dcta"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation fraction must lie strictly between zero and one")

    split_manifest = {
        "seed": args.seed,
        "unit": "demonstration trajectory",
        "stratification": "dataset task_index",
        "validation_fraction": args.validation_fraction,
        "suites": {},
    }
    for suite_index, (suite, suite_info) in enumerate(SUITES.items()):
        dataset_root = args.datasets_dir / f"libero_{suite}_image"
        episodes_path = dataset_root / "meta/episodes/chunk-000/file-000.parquet"
        table = pq.read_table(
            episodes_path,
            columns=["episode_index", "tasks", "stats/task_index/min"],
        ).to_pydict()
        episodes_by_task: dict[int, list[int]] = {task_index: [] for task_index in range(10)}
        language_by_task: dict[int, str] = {}
        for episode_index, tasks, task_index_value in zip(
            table["episode_index"], table["tasks"], table["stats/task_index/min"]
        ):
            task_index = int(task_index_value[0])
            episodes_by_task[task_index].append(int(episode_index))
            language_by_task[task_index] = str(tasks[0])

        suite_rng = np.random.default_rng(args.seed + suite_index)
        task_splits = []
        for dataset_task_id in range(10):
            episodes = np.asarray(sorted(episodes_by_task[dataset_task_id]), dtype=np.int64)
            shuffled = suite_rng.permutation(episodes)
            validation_count = max(1, int(round(len(episodes) * args.validation_fraction)))
            validation = sorted(int(x) for x in shuffled[:validation_count])
            training = sorted(int(x) for x in shuffled[validation_count:])
            task_splits.append(
                {
                    "dataset_task_id": dataset_task_id,
                    "benchmark_task_id": suite_info["dataset_to_benchmark"][dataset_task_id],
                    "language": language_by_task[dataset_task_id],
                    "training_episode_ids": training,
                    "validation_episode_ids": validation,
                }
            )
        split_manifest["suites"][suite] = {
            "dataset_revision": suite_info["revision"],
            "tasks": task_splits,
        }

    rollout_rng = np.random.default_rng(args.seed)
    rollout_schedule = {
        "seed": args.seed,
        "unit": "official LIBERO reset state",
        "block": "suite x benchmark_task_id x trial_id",
        "methods": METHODS,
        "blocks": [],
    }
    for suite in SUITES:
        for benchmark_task_id in range(10):
            for trial_id in range(10):
                rollout_schedule["blocks"].append(
                    {
                        "suite": suite,
                        "benchmark_task_id": benchmark_task_id,
                        "trial_id": trial_id,
                        "method_order": rollout_rng.permutation(METHODS).tolist(),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "demo_splits.json").write_text(
        json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "rollout_schedule.json").write_text(
        json.dumps(rollout_schedule, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
