"""Build offline RoboTwin ``Y_refresh`` labels from one episode cache shard.

The policy-response shard is the only input containing frozen-policy outputs.
This module deliberately has no model or observation loading path: future
outputs are used only to construct label-side distances and prefix survival.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np


GROUP_NAMES = ("left_arm", "left_gripper", "right_arm", "right_gripper")
GROUP_INDICES = {
    "left_arm": np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64),
    "left_gripper": np.asarray([6], dtype=np.int64),
    "right_arm": np.asarray([7, 8, 9, 10, 11, 12], dtype=np.int64),
    "right_gripper": np.asarray([13], dtype=np.int64),
}


def build_refresh_targets(
    action_chunks: np.ndarray,
    frame_indices: np.ndarray,
    *,
    thresholds: Mapping[str, float],
    chunk_size: int = 50,
) -> dict[str, np.ndarray]:
    """Return raw distances, instantaneous validity, prefix labels, and censoring.

    ``action_chunks`` contains one complete chunk per source frame in one
    episode.  A future frame is looked up by its recorded frame index rather
    than by an implicit array offset, so a missing/corrupt frame is censored.
    """

    chunks = np.asarray(action_chunks, dtype=np.float32)
    frames = np.asarray(frame_indices, dtype=np.int64)
    if chunks.ndim != 3 or chunks.shape[2] != 14:
        raise ValueError("action_chunks must have shape [N, 50, 14]")
    if chunks.shape[1] < chunk_size:
        raise ValueError("cached chunks are shorter than the configured chunk size")
    if frames.shape != (chunks.shape[0],) or len(set(frames.tolist())) != len(frames):
        raise ValueError("frame_indices must be unique and match the cache rows")
    if any(name not in thresholds or float(thresholds[name]) < 0 for name in GROUP_NAMES):
        raise ValueError("a non-negative threshold is required for every verified group")

    n = len(frames)
    horizon = int(chunk_size)
    raw = np.full((n, len(GROUP_NAMES), horizon), np.nan, dtype=np.float32)
    valid = np.zeros((n, len(GROUP_NAMES), horizon), dtype=np.uint8)
    censor = np.zeros_like(valid)
    row_by_frame = {int(frame): row for row, frame in enumerate(frames.tolist())}

    for source_row, source_frame in enumerate(frames.tolist()):
        for offset in range(horizon):
            future_row = row_by_frame.get(int(source_frame) + offset)
            if future_row is None:
                continue
            censor[source_row, :, offset] = 1
            for group_row, group_name in enumerate(GROUP_NAMES):
                indices = GROUP_INDICES[group_name]
                distance = float(
                    np.max(
                        np.abs(
                            chunks[source_row, offset, indices]
                            - chunks[future_row, 0, indices]
                        )
                    )
                )
                raw[source_row, group_row, offset] = distance
                valid[source_row, group_row, offset] = int(
                    distance <= float(thresholds[group_name])
                )

    prefix = np.zeros_like(valid)
    for source_row in range(n):
        for group_row in range(len(GROUP_NAMES)):
            observed = censor[source_row, group_row].astype(bool)
            running = 1
            for offset in range(horizon):
                if not observed[offset]:
                    running = 0
                else:
                    running *= int(valid[source_row, group_row, offset])
                    prefix[source_row, group_row, offset] = running

    return {
        "raw_distances": raw,
        "validity": valid,
        "y_refresh": prefix,
        "censor_mask": censor,
        "offset_k": np.arange(horizon, dtype=np.int64),
        "group_names": np.asarray(GROUP_NAMES, dtype=str),
    }


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    # Passing a path ending in ``.partial`` makes NumPy append ``.npz`` and
    # breaks the atomic replace below.  A file handle preserves the exact
    # temporary pathname.
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def build_from_shard(
    shard: Path,
    output: Path,
    *,
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    with np.load(shard, allow_pickle=False) as data:
        rows = int(data["action_chunks"].shape[0])
        targets = build_refresh_targets(
            data["action_chunks"], data["frame_index"], thresholds=thresholds
        )
        episode_index = int(np.unique(data["episode_index"])[0])
        task_index = int(np.unique(data["task_index"])[0])
    _atomic_savez(output, **targets, episode_index=np.asarray(episode_index), task_index=np.asarray(task_index))
    return {
        "status": "complete",
        "episode_index": episode_index,
        "task_index": task_index,
        "rows": rows,
        "observed_windows": int(targets["censor_mask"].sum()),
        "output": str(output),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", action="append", required=True, metavar="GROUP=VALUE")
    return parser


def _parse_thresholds(values: list[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for value in values:
        name, separator, threshold = value.partition("=")
        if not separator or name not in GROUP_NAMES:
            raise ValueError(f"threshold must be GROUP=VALUE for {GROUP_NAMES}: {value!r}")
        parsed[name] = float(threshold)
    return parsed


def main() -> None:
    args = _parser().parse_args()
    result = build_from_shard(args.shard, args.output, thresholds=_parse_thresholds(args.threshold))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
