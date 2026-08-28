#!/usr/bin/env python3
"""Build the frozen Phase-0 PPPR offline feature table.

This command reads only the completed Fresh query caches and writes an
outcome-blind feature table.  It never imports a simulator or policy and never
trains a predictor.  Run from the repository root as:

    /home/wjq/workspace/venvs/libero_act/bin/python \
      research/overnight_pppr_20260828/build_phase0.py

The default invocation is idempotent after a complete output exists.  A
partial output is never silently overwritten; pass ``--force`` deliberately
when replacing it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pppr_metrics import (
    ACTION_DIM,
    METRIC_NAMES,
    action_at,
    event_score_from_fresh_chunk,
    fit_arm_scales,
    flatten_row,
    pair_feature,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_PROTOCOL = HERE / "phase0_protocol.json"
DEFAULT_ROLLOUT_PROTOCOL = REPO_ROOT / "experiments/component_temporal_reuse/protocol.json"
DEFAULT_PILOT = REPO_ROOT / "experiments/component_temporal_reuse/pilot_results.json"
DEFAULT_CACHE_ROOT = REPO_ROOT / "experiments/component_temporal_reuse/query_cache"
DEFAULT_OUTPUT = HERE / "phase0_features.npz"
DEFAULT_METADATA = HERE / "phase0_feature_metadata.json"
DEFAULT_MARKER = HERE / "phase0_features.complete"

DEVELOPMENT_KEYS = (
    "libero_object:task3",
    "libero_spatial:task0",
    "libero_goal:task2",
    "libero_10:task3",
)
HELD_OUT_KEYS = (
    "libero_object:task5",
    "libero_spatial:task4",
    "libero_goal:task5",
    "libero_10:task5",
)
SPLIT_BY_KEY = {key: "development" for key in DEVELOPMENT_KEYS} | {
    key: "held_out" for key in HELD_OUT_KEYS
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".npz.tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        # Passing an open handle prevents NumPy from appending another .npz
        # suffix to our temporary path.
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_cache(path: Path) -> list[np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        names = sorted(archive.files, key=lambda name: int(name.split("_")[-1]))
        if names != [f"episode_{i}" for i in range(len(names))]:
            raise ValueError(f"{path} has noncanonical episode keys: {archive.files}")
        return [np.asarray(archive[name], dtype=np.float32) for name in names]


def _task_cache_path(cache_root: Path, suite: str, task_id: int) -> Path:
    return cache_root / f"{suite}_task{int(task_id)}_fresh.npz"


def _validate_task_split(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = list(protocol.get("tasks", []))
    keys = [f"{task['suite']}:task{int(task['task_id'])}" for task in tasks]
    expected = list(DEVELOPMENT_KEYS + HELD_OUT_KEYS)
    if set(keys) != set(expected):
        raise ValueError("protocol task set does not match the frozen Phase-0 split")
    return tasks


def _empty_row(
    *,
    task_index: int,
    task_key: str,
    split: str,
    episode: int,
    t: int,
    age: int,
    radius: int,
    window_size: int,
    window_start_offset: int,
    chunks: np.ndarray,
    scales: np.ndarray,
) -> dict[str, object]:
    """Represent a masked boundary candidate with NaN metric values."""

    u = int(t + age)
    window = np.arange(u + window_start_offset, u + window_start_offset + window_size, dtype=np.int64)
    family = np.arange(u, u + radius + 1, dtype=np.int64)
    current_action = np.full(ACTION_DIM, np.nan, dtype=np.float32)
    event = {
        "nearest_gripper_transition_offset": -1,
        "gripper_transition_proximity": np.nan,
        "arm_change": np.nan,
        "arm_curvature": np.nan,
        "event_score": np.nan,
    }
    if 0 <= t < chunks.shape[0]:
        current_action = np.asarray(chunks[t, 0], dtype=np.float32).copy()
        event = event_score_from_fresh_chunk(chunks[t], scales)
    row: dict[str, object] = {
        "task_index": task_index,
        "task_key": task_key,
        "split": split,
        "episode": episode,
        "old_query_t": t,
        "future_query_u": u,
        "age_steps": age,
        "valid": False,
        "window_targets": window,
        "future_family_queries": family,
        "current_action": current_action,
        **{f"event_{field}": value for field, value in event.items()},
    }
    for group in ("raw_ppr", "old_to_consensus", "future_dispersion", "pppr"):
        for metric in METRIC_NAMES:
            row[f"{group}_{metric}"] = np.nan
    return row


def _row_arrays(rows: list[dict[str, object]]) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("no candidate rows were generated")

    def col(name: str) -> list[object]:
        return [row[name] for row in rows]

    arrays: dict[str, np.ndarray] = {
        "task_index": np.asarray(col("task_index"), dtype=np.int16),
        "task_key": np.asarray(col("task_key"), dtype="U32"),
        "split": np.asarray(col("split"), dtype="U12"),
        "episode": np.asarray(col("episode"), dtype=np.int16),
        "old_query_t": np.asarray(col("old_query_t"), dtype=np.int32),
        "future_query_u": np.asarray(col("future_query_u"), dtype=np.int32),
        "age_steps": np.asarray(col("age_steps"), dtype=np.int16),
        "valid": np.asarray(col("valid"), dtype=bool),
        "window_targets": np.asarray(
            [int(row["future_query_u"]) + np.arange(2, 6, dtype=np.int32) for row in rows], dtype=np.int32
        ),
        "future_family_queries": np.asarray(
            [int(row["future_query_u"]) + np.arange(0, 3, dtype=np.int32) for row in rows], dtype=np.int32
        ),
        "current_action": np.stack(col("current_action")).astype(np.float32),
    }
    for group in ("raw_ppr", "old_to_consensus", "future_dispersion", "pppr"):
        for metric in METRIC_NAMES:
            arrays[f"{group}_{metric}"] = np.asarray(col(f"{group}_{metric}"), dtype=np.float32)
    event_fields = (
        "nearest_gripper_transition_offset",
        "gripper_transition_proximity",
        "arm_change",
        "arm_curvature",
        "event_score",
    )
    for field in event_fields:
        dtype = np.int16 if field.endswith("offset") else np.float32
        arrays[f"event_{field}"] = np.asarray([row[f"event_{field}"] for row in rows], dtype=dtype)
    return arrays


def _current_chunk_arrays(current_chunks: list[dict[str, object]]) -> dict[str, np.ndarray]:
    return {
        "current_chunk_task_index": np.asarray([x["task_index"] for x in current_chunks], dtype=np.int16),
        "current_chunk_task_key": np.asarray([x["task_key"] for x in current_chunks], dtype="U32"),
        "current_chunk_split": np.asarray([x["split"] for x in current_chunks], dtype="U12"),
        "current_chunk_episode": np.asarray([x["episode"] for x in current_chunks], dtype=np.int16),
        "current_chunk_query_t": np.asarray([x["query_t"] for x in current_chunks], dtype=np.int32),
        "current_chunks": np.stack([x["chunk"] for x in current_chunks]).astype(np.float32),
    }


def build(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    rollout_protocol_path: Path = DEFAULT_ROLLOUT_PROTOCOL,
    pilot_path: Path = DEFAULT_PILOT,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    output_path: Path = DEFAULT_OUTPUT,
    metadata_path: Path = DEFAULT_METADATA,
    marker_path: Path = DEFAULT_MARKER,
) -> dict[str, Any]:
    started = time.monotonic()
    protocol = json.loads(protocol_path.read_text())
    rollout_protocol = json.loads(rollout_protocol_path.read_text())
    pilot = json.loads(pilot_path.read_text())
    tasks = _validate_task_split(rollout_protocol)
    frozen_split = protocol.get("split", {})
    if tuple(frozen_split.get("development", [])) != DEVELOPMENT_KEYS:
        raise ValueError("phase0 development split changed")
    if tuple(frozen_split.get("heldout_offline", [])) != HELD_OUT_KEYS:
        raise ValueError("phase0 held-out split changed")
    conditions = rollout_protocol.get("conditions", [])
    age_steps = tuple(int(x) for x in protocol.get("label", {}).get("forecast_offsets", []))
    if age_steps != tuple(range(1, 17)):
        raise ValueError(f"frozen feature-offset schedule changed: {age_steps}")
    radius = int(protocol.get("label", {}).get("persistence_depth", 2))
    window_size = int(protocol.get("label", {}).get("local_prefix_width", 4))
    window_start_offset = 2
    if (radius, window_size, window_start_offset) != (2, 4, 2):
        raise ValueError("frozen PPPR geometry changed")
    if not conditions:
        raise ValueError("protocol has no condition schedule")

    cache_data: dict[str, list[np.ndarray]] = {}
    cache_meta: dict[str, Any] = {}
    for task in tasks:
        key = f"{task['suite']}:task{int(task['task_id'])}"
        path = _task_cache_path(cache_root, str(task["suite"]), int(task["task_id"]))
        episodes = _load_cache(path)
        cache_data[key] = episodes
        cache_meta[key] = {
            "path": str(path.resolve()),
            "file": path.name,
            "episodes": len(episodes),
            "episode_shapes": [list(np.asarray(ep).shape) for ep in episodes],
            "action_dim": ACTION_DIM,
            "postprocessed_actions_only": True,
            "proprio_available": False,
        }
        pilot_task = pilot.get("tasks", {}).get(key)
        if pilot_task is None or "fresh" not in pilot_task.get("conditions", {}):
            raise ValueError(f"pilot_results has no Fresh record for {key}")
        if len(episodes) != int(pilot_task["conditions"]["fresh"]["episodes"]):
            raise ValueError(f"cache/pilot episode count mismatch for {key}")

    development_episodes = [ep for key in DEVELOPMENT_KEYS for ep in cache_data[key]]
    scale_fit = fit_arm_scales(development_episodes)
    scales = np.asarray(scale_fit.scales, dtype=np.float64)

    rows: list[dict[str, object]] = []
    current_chunks: list[dict[str, object]] = []
    per_task_counts: dict[str, Any] = {}
    task_keys = []
    for task_index, task in enumerate(tasks):
        key = f"{task['suite']}:task{int(task['task_id'])}"
        split = SPLIT_BY_KEY[key]
        task_keys.append(key)
        task_rows_start = len(rows)
        valid_by_age = {str(age): 0 for age in age_steps}
        invalid_by_age = {str(age): 0 for age in age_steps}
        for episode_index, raw_episode in enumerate(cache_data[key]):
            chunks = np.asarray(raw_episode, dtype=np.float64)
            for t in range(chunks.shape[0]):
                current_chunks.append(
                    {
                        "task_index": task_index,
                        "task_key": key,
                        "split": split,
                        "episode": episode_index,
                        "query_t": t,
                        "chunk": chunks[t].copy(),
                    }
                )
                for age in age_steps:
                    row = pair_feature(
                        chunks,
                        old_query=t,
                        age_steps=age,
                        scales=scales,
                        radius=radius,
                        window_size=window_size,
                        window_start_offset=window_start_offset,
                    )
                    if row is None:
                        flat = _empty_row(
                            task_index=task_index,
                            task_key=key,
                            split=split,
                            episode=episode_index,
                            t=t,
                            age=age,
                            radius=radius,
                            window_size=window_size,
                            window_start_offset=window_start_offset,
                            chunks=chunks,
                            scales=scales,
                        )
                        invalid_by_age[str(age)] += 1
                    else:
                        flat = flatten_row(row)
                        flat.update(
                            {
                                "task_index": task_index,
                                "task_key": key,
                                "split": split,
                                "episode": episode_index,
                                "valid": True,
                            }
                        )
                        valid_by_age[str(age)] += 1
                    rows.append(flat)
        per_task_counts[key] = {
            "split": split,
            "episodes": len(cache_data[key]),
            "candidate_rows": len(rows) - task_rows_start,
            "valid_rows_by_age": valid_by_age,
            "masked_rows_by_age": invalid_by_age,
            "current_chunk_rows": sum(len(ep) for ep in cache_data[key]),
        }

    feature_arrays = _row_arrays(rows)
    feature_arrays.update(_current_chunk_arrays(current_chunks))
    feature_arrays["metric_order"] = np.asarray(METRIC_NAMES, dtype="U12")
    feature_arrays["action_dim_order"] = np.asarray(["p0", "p1", "p2", "r0", "r1", "r2", "grip"], dtype="U8")
    atomic_npz(output_path, feature_arrays)
    # Read back the committed archive before publishing metadata/marker.
    with np.load(output_path, allow_pickle=False) as readback:
        if "valid" not in readback.files or readback["valid"].shape != feature_arrays["valid"].shape:
            raise RuntimeError("feature-table NPZ readback failed")

    valid_mask = feature_arrays["valid"]
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "generator": "research/overnight_pppr_20260828/build_phase0.py",
        "generator_command": " ".join([sys.executable, *sys.argv]),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": float(time.monotonic() - started),
        "protocol_path": str(protocol_path.resolve()),
        "rollout_protocol_path": str(rollout_protocol_path.resolve()),
        "pilot_results_path": str(pilot_path.resolve()),
        "cache_root": str(cache_root.resolve()),
        "output_path": str(output_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "completion_marker": str(marker_path.resolve()),
        "frozen_split": {
            "development": list(DEVELOPMENT_KEYS),
            "held_out": list(HELD_OUT_KEYS),
        },
        "task_order": task_keys,
        "source_age_steps": list(age_steps),
        "source_age_seconds_at_30hz": [float(x) / 30.0 for x in age_steps],
        "future_family": {"radius_r": radius, "queries": "u,u+1,u+2", "earlier_queries_forbidden": True},
        "window": {"M": window_size, "targets": "{u+2,u+3,u+4,u+5}"},
        "alignment": {
            "action_at": "a[v|q] = cache[q, v-q, :]",
            "old_action": "a[v|t]",
            "raw_reference": "a[v|u]",
            "episode_target_boundary": "not applied; predicted v may extend beyond final executed step",
            "future_query_boundary": "u+r < episode_query_count",
            "chunk_boundary": "0 <= v-q < chunk_horizon for every required q",
        },
        "action_layout": {
            "postprocessed_7d": True,
            "position_indices": [0, 1, 2],
            "rotation_indices": [3, 4, 5],
            "gripper_index": 6,
            "zero_gripper_sign": "0 (np.sign; no open/nonnegative remapping)",
        },
        "scale_fit": scale_fit.as_dict(),
        "metric_formulas": {
            "dp": "||((a_p-b_p)/scale_p)||_2 / sqrt(3)",
            "dr": "||((a_r-b_r)/scale_r)||_2 / sqrt(3)",
            "d_arm_unbounded": "0.5*(dp+dr)",
            "arm": "d_arm_unbounded/(1+d_arm_unbounded)",
            "grip": "1 if sign(a_grip)!=sign(b_grip), else 0",
            "joint": "0.5*arm+0.5*grip",
            "future_consensus": "per-dimension arm median; majority gripper sign",
            "raw_ppr": "median over W of d(a[v|t], a[v|u])",
            "old_to_consensus": "R(v)=d(a[v|t], C_v), C_v=future consensus",
            "future_dispersion": "C(v)=median_{q=u..u+2} d(a[v|q], C_v)",
            "pppr": "median over W of max(R(v)-C(v), 0)",
            "metric_column_order": list(METRIC_NAMES),
        },
        "event_score": {
            "source": "Fresh current chunk only, cache[t,:,:]",
            "formula": "0.5*gripper_transition_proximity + 0.25*arm_change + 0.25*arm_curvature",
            "transition": "first sign change in current Fresh chunk after offset 0; no transition=0 proximity",
            "arm_change": "clip(median normalized chunk-local first difference, 0, 1)",
            "arm_curvature": "clip(median normalized chunk-local second difference, 0, 1)",
            "tuning": "none; weights and clipping are fixed before outcome inspection",
        },
        "cache_inventory": cache_meta,
        "cache_limitation": {
            "available": ["postprocessed action chunks"],
            "not_available": ["proprioceptive state", "images", "rewards", "termination state"],
            "retained_for_later": "current_chunks array is included in phase0_features.npz; source caches are untouched",
            "training": "none",
        },
        "counts": {
            "tasks": len(tasks),
            "episodes": int(sum(len(x) for x in cache_data.values())),
            "candidate_rows": int(len(rows)),
            "valid_rows": int(valid_mask.sum()),
            "masked_rows": int((~valid_mask).sum()),
            "current_chunk_rows": int(len(current_chunks)),
            "valid_rows_by_age": {str(age): int(np.sum(valid_mask & (feature_arrays["age_steps"] == age))) for age in age_steps},
            "masked_rows_by_age": {str(age): int(np.sum((~valid_mask) & (feature_arrays["age_steps"] == age))) for age in age_steps},
            "per_task": per_task_counts,
        },
        "condition_schedule_from_protocol": conditions,
    }
    atomic_json(metadata_path, metadata)
    marker_payload = {
        "status": "complete",
        "generated_at_utc": metadata["generated_at_utc"],
        "feature_table": str(output_path.resolve()),
        "metadata": str(metadata_path.resolve()),
        "valid_rows": metadata["counts"]["valid_rows"],
        "candidate_rows": metadata["counts"]["candidate_rows"],
    }
    atomic_json(marker_path, marker_payload)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--rollout-protocol", type=Path, default=DEFAULT_ROLLOUT_PROTOCOL)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--marker", type=Path, default=DEFAULT_MARKER)
    parser.add_argument("--force", action="store_true", help="replace existing outputs deliberately")
    args = parser.parse_args()

    outputs = (args.output, args.metadata, args.marker)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        if len(existing) == len(outputs):
            print(json.dumps({"status": "already_complete", "marker": str(args.marker.resolve())}, indent=2))
            return
        raise SystemExit(
            "partial Phase-0 output exists; refusing to overwrite. Remove only the partial files or pass --force."
        )
    metadata = build(
        protocol_path=args.protocol,
        rollout_protocol_path=args.rollout_protocol,
        pilot_path=args.pilot,
        cache_root=args.cache_root,
        output_path=args.output,
        metadata_path=args.metadata,
        marker_path=args.marker,
    )
    print(
        json.dumps(
            {
                "status": "generated",
                "output": str(args.output.resolve()),
                "metadata": str(args.metadata.resolve()),
                "marker": str(args.marker.resolve()),
                "candidate_rows": metadata["counts"]["candidate_rows"],
                "valid_rows": metadata["counts"]["valid_rows"],
                "masked_rows": metadata["counts"]["masked_rows"],
                "runtime_seconds": metadata["runtime_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
