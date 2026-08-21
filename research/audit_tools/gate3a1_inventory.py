#!/usr/bin/env python3
"""Inventory the existing episode split for the proposed dense Gate-3A1.

Reads only NPZ metadata and Parquet row counts; it performs no ACT inference and
does not modify historical artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DATASET = Path("/home/thor/datasets/libero_object_25_08_23_lerobotv2.1")
CACHE = ROOT / "experiments/temporal_reliability/reliability_dataset.npz"
BUNDLE = ROOT / "experiments/dynamic_reliability_horizon/artifact_handoff/minimal_y_refresh_training_bundle.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "research/audit_outputs/gate3a1_inventory.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    cache = np.load(args.cache, allow_pickle=False)
    bundle = np.load(args.bundle, allow_pickle=False)
    episode_task: dict[int, int] = {}
    for episode, task in zip(cache["episode_index"], cache["task_index"], strict=True):
        episode_task.setdefault(int(episode), int(task))
        if episode_task[int(episode)] != int(task):
            raise RuntimeError("one episode maps to multiple tasks")
    episode_split: dict[int, int] = {}
    for episode, split in zip(
        bundle["episode_index"], bundle["split_membership"], strict=True
    ):
        episode_split.setdefault(int(episode), int(split))
        if episode_split[int(episode)] != int(split):
            raise RuntimeError("split is not episode-level")
    if set(episode_task) != set(episode_split):
        raise RuntimeError("cache and split bundle have different episode sets")

    rows: dict[int, int] = {}
    for episode in sorted(episode_task):
        path = args.dataset / "data/chunk-000" / f"episode_{episode:06d}.parquet"
        rows[episode] = int(pq.read_metadata(path).num_rows)

    split_names = {0: "train", 1: "validation", 2: "test"}
    splits = {}
    for split, name in split_names.items():
        episodes = [episode for episode, value in episode_split.items() if value == split]
        task_episodes = Counter(episode_task[episode] for episode in episodes)
        task_steps: dict[int, int] = defaultdict(int)
        for episode in episodes:
            task_steps[episode_task[episode]] += rows[episode]
        splits[name] = {
            "split_membership_code": split,
            "episodes": len(episodes),
            "dataset_steps": int(sum(rows[episode] for episode in episodes)),
            "episodes_per_task": {str(key): value for key, value in sorted(task_episodes.items())},
            "dataset_steps_per_task": {str(key): value for key, value in sorted(task_steps.items())},
            "episode_ids": sorted(episodes),
        }

    output = {
        "audit_script": str(Path(__file__).relative_to(ROOT)),
        "scope": "Read-only Gate-3A1 inventory; one proposed dense ACT query per eligible dataset step.",
        "provenance": {
            "cache": {"path": str(args.cache.resolve()), "sha256": sha256(args.cache)},
            "split_bundle": {"path": str(args.bundle.resolve()), "sha256": sha256(args.bundle)},
            "dataset": str(args.dataset.resolve()),
        },
        "dataset_frequency_hz": 10.0,
        "all_episodes": len(episode_task),
        "splits": splits,
        "proposed_validation_plus_test": {
            "episodes": splits["validation"]["episodes"] + splits["test"]["episodes"],
            "dataset_steps_and_dense_act_calls": splits["validation"]["dataset_steps"]
            + splits["test"]["dataset_steps"],
            "note": "Count assumes one ACT query at every demonstration step and excludes retries/resume verification calls.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output["proposed_validation_plus_test"], indent=2))


if __name__ == "__main__":
    main()
