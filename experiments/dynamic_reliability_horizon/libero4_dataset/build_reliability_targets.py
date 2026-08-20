#!/usr/bin/env python3
"""Materialize raw Y_refresh distances and censored labels from cache shards."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from dataset_common import DATASET_REVISION, K_MAX, atomic_write_json, sha256_file
from policy_cache import atomic_save_npz, valid_cache_shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--policy-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action-std", type=float, nargs=7, required=True)
    return parser.parse_args()


def prefix(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    return np.logical_and.accumulate(values, axis=1) & observed


def build_labels(old: np.ndarray, fresh: np.ndarray, observed: np.ndarray, action_std: np.ndarray) -> dict[str, np.ndarray]:
    difference = old.astype(np.float32) - fresh.astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        translation = np.sqrt(np.mean((difference[:, :, :3] / action_std[:3]) ** 2, axis=2))
        rotation = np.sqrt(np.mean((difference[:, :, 3:6] / action_std[3:6]) ** 2, axis=2))
        gripper = np.abs(difference[:, :, 6]) / float(action_std[6])
    arm_distance = np.maximum(translation, rotation)
    sign_match = np.where(old[:, :, 6] >= 0.0, 1, -1) == np.where(fresh[:, :, 6] >= 0.0, 1, -1)
    arm_valid = (arm_distance <= 1.0) & observed
    gripper_valid = (gripper <= 1.0) & sign_match & observed
    return {
        "raw_group_distances": np.stack([arm_distance, gripper], axis=2).astype(np.float32),
        "arm_translation_normalized_rms": translation.astype(np.float32),
        "arm_rotation_normalized_rms": rotation.astype(np.float32),
        "gripper_normalized_absolute_error": gripper.astype(np.float32),
        "gripper_sign_match": sign_match.astype(np.bool_),
        "pointwise_valid": np.stack([arm_valid, gripper_valid], axis=2),
        "Y_refresh": np.stack([prefix(arm_valid, observed), prefix(gripper_valid, observed)], axis=2),
        "label_observed": np.repeat(observed[:, :, None], 2, axis=2),
    }


def main() -> int:
    args = parse_args()
    corpus_dir = args.corpus_dir.resolve()
    cache_dir = args.policy_cache.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(corpus_dir / "source_window_index.npz", allow_pickle=False) as windows:
        future_frame_ids = np.asarray(windows["future_frame_ids"], dtype=np.int64)
        canonical_observed = np.asarray(windows["observed"], dtype=np.bool_)
    if future_frame_ids.shape != (273465, K_MAX) or canonical_observed.shape != future_frame_ids.shape:
        raise RuntimeError("Canonical source-window index has an unexpected shape")
    shard_paths = sorted(cache_dir.glob("shard-*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No cache shards under {cache_dir}")
    first_actions = np.full((len(future_frame_ids), 7), np.nan, dtype=np.float32)
    cached_ids: set[int] = set()
    for path in shard_paths:
        if not valid_cache_shard(path):
            continue
        with np.load(path, allow_pickle=False) as data:
            ids = np.asarray(data["frame_id"], dtype=np.int64)
            chunks = np.asarray(data["source_chunks"], dtype=np.float32)
            first_actions[ids] = chunks[:, 0]
            cached_ids.update(int(value) for value in ids)
    if len(cached_ids) != len(future_frame_ids) or not np.all(np.isfinite(first_actions)):
        raise RuntimeError(f"Need a complete cache for target materialization: {len(cached_ids)}/{len(future_frame_ids)} frames")
    action_std = np.asarray(args.action_std, dtype=np.float32)
    if np.any(~np.isfinite(action_std)) or np.any(action_std <= 0):
        raise ValueError("action_std must be finite and positive")
    completed = 0
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as data:
            ids = np.asarray(data["frame_id"], dtype=np.int64)
            old_chunks = np.asarray(data["source_chunks"], dtype=np.float32)
            metadata = {key: np.asarray(data[key]) for key in ("episode_id", "frame_index", "task_id", "split")}
        length = old_chunks.shape[1]
        observed = canonical_observed[ids].copy()
        observed[:, length:] = False
        future_ids = future_frame_ids[ids].copy()
        future_ids[~observed] = -1
        fresh = np.full((len(ids), K_MAX, 7), np.nan, dtype=np.float32)
        for row in range(len(ids)):
            valid = observed[row]
            if np.any(valid):
                fresh[row, valid] = first_actions[future_ids[row, valid]]
        old = np.full_like(fresh, np.nan)
        old[:, :length] = old_chunks
        labels = build_labels(old, fresh, observed, action_std)
        labels.update(metadata)
        labels["source_frame_id"] = ids
        labels["offsets"] = np.arange(1, K_MAX + 1, dtype=np.int16)
        target_path = output_dir / shard_path.name
        atomic_save_npz(target_path, labels)
        completed += len(ids)
    manifest = {
        "purpose": "Right-censored frozen-policy replanning consistency labels",
        "dataset_revision": DATASET_REVISION,
        "policy_cache": str(cache_dir),
        "output_dir": str(output_dir),
        "contract": {
            "old": "cached A_t[k]",
            "fresh": "cached A_{t+k}[0]",
            "future_information": "label-side only",
            "forbidden_estimator_inputs": ["future observations", "future actions", "episode length", "normalized progress", "phase", "terminal/future metadata"],
        },
        "groups": {
            "arm": {"indices": [0, 1, 2, 3, 4, 5], "raw_distance": "max(translation normalized RMS, rotation normalized RMS)", "threshold": 1.0},
            "gripper": {"indices": [6], "raw_distance": "normalized absolute error", "threshold": 1.0, "additional_criterion": "command-sign agreement"},
        },
        "arrays": {
            "raw_group_distances": "[N,100,2] float32; never thresholded",
            "Y_refresh": "[N,100,2] bool prefix survival",
            "label_observed": "[N,100,2] bool; right-censored offsets are false",
            "fresh": "not stored; reconstructed from policy cache first actions",
        },
        "action_std": action_std.tolist(),
        "frames_materialized": completed,
        "source_window_sha256": sha256_file(corpus_dir / "source_window_index.npz"),
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"frames": completed, "output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
