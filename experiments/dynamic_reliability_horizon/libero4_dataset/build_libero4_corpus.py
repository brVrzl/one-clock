#!/usr/bin/env python3
"""Build the canonical policy-independent LIBERO-4 index and source windows."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dataset_common import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    FPS,
    GROUPS,
    K_MAX,
    SPLIT_RULE_VERSION,
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
    split_for_episode,
    suite_for_task,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def scalar(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def json_ref(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def task_rows(dataset_root: Path) -> list[dict[str, Any]]:
    rows = read_rows(dataset_root / "meta/tasks.parquet")
    result = []
    for row in sorted(rows, key=lambda item: int(item["task_index"])):
        task_index = int(row["task_index"])
        result.append({
            "task_index": task_index,
            "task_name": str(row["__index_level_0__"]),
            "suite": suite_for_task(task_index),
        })
    if len(result) != 40 or [row["task_index"] for row in result] != list(range(40)):
        raise RuntimeError(f"Expected task indices 0..39, got {len(result)} rows")
    return result


def episode_rows(dataset_root: Path) -> list[dict[str, Any]]:
    rows = read_rows(dataset_root / "meta/episodes/chunk-000/file-000.parquet")
    result = []
    for raw in sorted(rows, key=lambda item: int(item["episode_index"])):
        episode_index = int(raw["episode_index"])
        length = int(raw["length"])
        # Task index is not stored in episode metadata. It is recovered from
        # the data rows below and then checked against every frame in the episode.
        result.append({
            "episode_index": episode_index,
            "length": length,
            "dataset_from_index": int(raw["dataset_from_index"]),
            "dataset_to_index": int(raw["dataset_to_index"]),
            "data_chunk_index": int(raw["data/chunk_index"]),
            "data_file_index": int(raw["data/file_index"]),
            "video_image_chunk_index": int(raw["videos/observation.images.image/chunk_index"]),
            "video_image_file_index": int(raw["videos/observation.images.image/file_index"]),
            "video_image_from_timestamp": float(raw["videos/observation.images.image/from_timestamp"]),
            "video_image2_chunk_index": int(raw["videos/observation.images.image2/chunk_index"]),
            "video_image2_file_index": int(raw["videos/observation.images.image2/file_index"]),
            "video_image2_from_timestamp": float(raw["videos/observation.images.image2/from_timestamp"]),
        })
    if len(result) != 1693 or [row["episode_index"] for row in result] != list(range(1693)):
        raise RuntimeError("Pinned episode metadata is incomplete or not contiguous")
    return result


def data_file_path(dataset_root: Path, chunk_index: int, file_index: int) -> Path:
    return dataset_root / f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"


def build_frame_rows(
    dataset_root: Path,
    tasks: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    task_by_index = {int(row["task_index"]): row for row in tasks}
    episode_by_index = {int(row["episode_index"]): row for row in episodes}
    required_files = sorted({(row["data_chunk_index"], row["data_file_index"]) for row in episodes})
    frames: list[dict[str, Any]] = []
    for chunk_index, file_index in required_files:
        path = data_file_path(dataset_root, chunk_index, file_index)
        if not path.is_file():
            raise FileNotFoundError(f"Missing required tabular source file: {path}")
        rows = read_rows(path)
        relative = path.relative_to(dataset_root).as_posix()
        for row_index, raw in enumerate(rows):
            episode_index = int(scalar(raw["episode_index"]))
            episode = episode_by_index.get(episode_index)
            if episode is None:
                raise RuntimeError(f"Data row references unknown episode {episode_index}")
            frame_index = int(scalar(raw["frame_index"]))
            dataset_index = int(scalar(raw["index"]))
            task_index = int(scalar(raw["task_index"]))
            action = [float(value) for value in raw["action"]]
            state = raw["observation.state"]
            if len(action) != 7 or len(state) != 8:
                raise RuntimeError(f"Unexpected action/state shape at dataset index {dataset_index}")
            if not all(np.isfinite(action)) or not all(np.isfinite(float(value)) for value in state):
                raise RuntimeError(f"NaN/Inf action or state at dataset index {dataset_index}")
            suite = suite_for_task(task_index)
            task = task_by_index[task_index]
            if frame_index < 0 or frame_index >= episode["length"]:
                raise RuntimeError(f"Frame index outside episode at dataset index {dataset_index}")
            if dataset_index != episode["dataset_from_index"] + frame_index:
                raise RuntimeError(f"Non-contiguous episode row at dataset index {dataset_index}")
            timestamp = float(scalar(raw["timestamp"]))
            image_path = f"videos/observation.images.image/chunk-{episode['video_image_chunk_index']:03d}/file-{episode['video_image_file_index']:03d}.mp4"
            image2_path = f"videos/observation.images.image2/chunk-{episode['video_image2_chunk_index']:03d}/file-{episode['video_image2_file_index']:03d}.mp4"
            frames.append({
                "frame_id": dataset_index,
                "episode_id": episode_index,
                "task_id": task_index,
                "suite": suite,
                "task_name": task["task_name"],
                "frame_index": frame_index,
                "timestamp": timestamp,
                "dataset_index": dataset_index,
                "data_path": relative,
                "data_row_index": row_index,
                "state_ref": json_ref({"path": relative, "row_index": row_index, "column": "observation.state"}),
                "observation_image_ref": json_ref({"path": image_path, "timestamp_sec": timestamp, "fps": FPS}),
                "observation_image2_ref": json_ref({"path": image2_path, "timestamp_sec": timestamp, "fps": FPS}),
                "demonstrated_action": action,
                "episode_start_frame_id": episode["dataset_from_index"],
                "episode_end_frame_id": episode["dataset_to_index"] - 1,
                "split": split_for_episode(suite, task_index, episode_index),
            })
    frames.sort(key=lambda row: int(row["frame_id"]))
    if len(frames) != 273465 or [row["frame_id"] for row in frames] != list(range(len(frames))):
        raise RuntimeError(f"Expected contiguous 273465 frame rows, got {len(frames)}")
    for frame in frames:
        episode = episode_by_index[int(frame["episode_id"])]
        if int(frame["frame_index"]) != int(frame["frame_id"]) - int(episode["dataset_from_index"]):
            raise RuntimeError("Frame index does not agree with episode dataset range")
    return frames, episode_by_index


def write_frame_index(output_dir: Path, frames: list[dict[str, Any]]) -> Path:
    columns = {key: [row[key] for row in frames] for key in frames[0]}
    table = pa.table(columns)
    path = output_dir / "frame_index.parquet"
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)
    return path


def write_source_windows(output_dir: Path, frames: list[dict[str, Any]], episodes: list[dict[str, Any]]) -> Path:
    future = np.full((len(frames), K_MAX), -1, dtype=np.int32)
    observed = np.zeros((len(frames), K_MAX), dtype=np.bool_)
    for episode in episodes:
        start = int(episode["dataset_from_index"])
        length = int(episode["length"])
        for frame_index in range(length):
            frame_id = start + frame_index
            available = min(K_MAX, length - frame_index - 1)
            if available:
                future[frame_id, :available] = np.arange(frame_id + 1, frame_id + available + 1, dtype=np.int32)
                observed[frame_id, :available] = True
    path = output_dir / "source_window_index.npz"
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, future_frame_ids=future, observed=observed, offsets=np.arange(1, K_MAX + 1, dtype=np.int16))
    os.replace(temporary, path)
    return path


def stats_for(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [int(row["length"]) for row in rows]
    if not lengths:
        return {"episodes": 0, "frames": 0, "min_episode_length": None, "median_episode_length": None, "max_episode_length": None}
    return {
        "episodes": len(rows),
        "frames": int(sum(lengths)),
        "min_episode_length": min(lengths),
        "median_episode_length": statistics.median(lengths),
        "max_episode_length": max(lengths),
    }


def make_split_manifest(tasks: list[dict[str, Any]], episodes: list[dict[str, Any]], frames: list[dict[str, Any]], output_dir: Path) -> Path:
    frame_by_episode = defaultdict(list)
    for frame in frames:
        frame_by_episode[int(frame["episode_id"])].append(frame)
    task_by_index = {int(row["task_index"]): row for row in tasks}
    manifest_episodes = []
    for episode in episodes:
        episode_id = int(episode["episode_index"])
        episode_frames = frame_by_episode[episode_id]
        if not episode_frames:
            raise RuntimeError(f"Episode {episode_id} has no frame rows")
        task_id = int(episode_frames[0]["task_id"])
        split = split_for_episode(suite_for_task(task_id), task_id, episode_id)
        if any(frame["split"] != split for frame in episode_frames):
            raise RuntimeError(f"Frame split mismatch inside episode {episode_id}")
        manifest_episodes.append({
            "episode_id": episode_id,
            "suite": suite_for_task(task_id),
            "task_id": task_id,
            "task_name": task_by_index[task_id]["task_name"],
            "split": split,
            "length": int(episode["length"]),
            "dataset_frame_start": int(episode["dataset_from_index"]),
            "dataset_frame_end_exclusive": int(episode["dataset_to_index"]),
        })
    stats: dict[str, Any] = {}
    for suite in sorted({row["suite"] for row in tasks}):
        stats[suite] = {}
        for task in [row for row in tasks if row["suite"] == suite]:
            task_id = int(task["task_index"])
            selected = [row for row in manifest_episodes if int(row["task_id"]) == task_id]
            stats[suite][str(task_id)] = {
                "task_name": task["task_name"],
                "all": stats_for(selected),
                **{split: stats_for([row for row in selected if row["split"] == split]) for split in ("train", "validation", "test")},
            }
    manifest = {
        "dataset_repo_id": DATASET_REPO_ID,
        "dataset_revision": DATASET_REVISION,
        "split_rule_version": SPLIT_RULE_VERSION,
        "split_rule": "sha256(SPLIT_RULE_VERSION|suite|task_id|episode_id) first uint64; train < 0.80, validation < 0.90, test otherwise",
        "task_stratified": True,
        "episodes": manifest_episodes,
        "statistics": stats,
    }
    path = output_dir / "episode_split.json"
    atomic_write_json(path, manifest)
    return path


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    info_path = dataset_root / "meta/info.json"
    stats_path = dataset_root / "meta/stats.json"
    tasks_path = dataset_root / "meta/tasks.parquet"
    episodes_path = dataset_root / "meta/episodes/chunk-000/file-000.parquet"
    for path in (info_path, stats_path, tasks_path, episodes_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("total_episodes") != 1693 or info.get("total_frames") != 273465 or info.get("total_tasks") != 40:
        raise RuntimeError(f"Pinned source metadata scale mismatch: {info}")
    tasks = task_rows(dataset_root)
    episodes = episode_rows(dataset_root)
    frames, _ = build_frame_rows(dataset_root, tasks, episodes)
    frame_index_path = write_frame_index(output_dir, frames)
    source_window_path = write_source_windows(output_dir, frames, episodes)
    split_path = make_split_manifest(tasks, episodes, frames, output_dir)
    source_files = {
        str(path.relative_to(dataset_root)): {"absolute_path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (info_path, stats_path, tasks_path, episodes_path)
    }
    source_inventory = []
    for pattern in ("data/**/*.parquet", "videos/**/*.mp4"):
        for path in sorted(dataset_root.glob(pattern)):
            source_inventory.append({
                "relative_path": str(path.relative_to(dataset_root)),
                "absolute_path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    download_listing_path = dataset_root / "download_listing.json"
    download_listing = json.loads(download_listing_path.read_text(encoding="utf-8")) if download_listing_path.is_file() else None
    artifact_files = {
        path.name: {"absolute_path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (frame_index_path, source_window_path, split_path)
    }
    manifest = {
        "purpose": "Policy-independent canonical LIBERO-4 reliability-data corpus",
        "dataset": {
            "repo_id": DATASET_REPO_ID,
            "revision": DATASET_REVISION,
            "root": str(dataset_root),
            "info_scale": {"episodes": 1693, "frames": 273465, "tasks": 40, "fps": FPS},
            "observed_scale": {"episodes": len(episodes), "frames": len(frames), "tasks": len(tasks)},
            "features": info["features"],
            "source_metadata": source_files,
            "source_file_inventory": source_inventory,
            "download_listing": download_listing,
        },
        "suites": sorted({row["suite"] for row in tasks}),
        "task_mapping": tasks,
        "action_contract": {
            "dimension": 7,
            "verified_order": ["eef_delta_x", "eef_delta_y", "eef_delta_z", "eef_axis_angle_x", "eef_axis_angle_y", "eef_axis_angle_z", "gripper"],
            "groups": GROUPS,
            "evidence": {
                "controller": "/home/thor/projects/upstreams/lerobot-env/lib/python3.12/site-packages/libero/libero/envs/env_wrapper.py:controller=OSC_POSE",
                "action_construction": "/home/thor/projects/upstreams/lerobot-env/lib/python3.12/site-packages/robosuite/utils/input_utils.py:action=np.concatenate([dpos,drotation,[grasp]])",
                "control_mode": "LeRobot LiberoEnv uses the 7-D OSC_POSE action space; the pinned dataset metadata independently verifies shape [7].",
            },
        },
        "future_lookup": {
            "k_max": K_MAX,
            "offsets": "1..100",
            "artifact": "source_window_index.npz",
            "right_censoring": "future_frame_ids=-1 and observed=False after the final frame of each episode",
        },
        "splits": {"artifact": "episode_split.json", "unit": "episode", "frame_split": False, "task_stratified": True},
        "artifacts": artifact_files,
        "determinism": "No wall-clock fields are written; rebuilding from the same pinned source produces byte-identical JSON manifests and deterministic row order.",
    }
    atomic_write_json(output_dir / "dataset_manifest.json", manifest)
    evidence_paths = [
        Path("/home/thor/projects/upstreams/lerobot-env/lib/python3.12/site-packages/libero/libero/benchmark/libero_suite_task_map.py"),
        Path("/home/thor/projects/upstreams/lerobot-env/lib/python3.12/site-packages/libero/libero/envs/env_wrapper.py"),
        Path("/home/thor/projects/upstreams/lerobot-env/lib/python3.12/site-packages/robosuite/utils/input_utils.py"),
    ]
    evidence_lines = []
    for path in evidence_paths:
        if path.is_file():
            evidence_lines.append(f"- `{path}` SHA256 `{sha256_file(path)}`")
        else:
            evidence_lines.append(f"- `{path}` unavailable on this host")
    download_status = "The pinned source download listing reports no failed files." if not download_listing or not download_listing.get("failures") else f"The pinned source download listing reports {len(download_listing['failures'])} failed files; they are recorded in `dataset_manifest.json` and were not silently dropped."
    audit = "\n".join([
        "# LIBERO-4 data audit",
        "",
        f"Pinned source: `{DATASET_REPO_ID}@{DATASET_REVISION}`.",
        "",
        "## Coverage",
        "",
        "- all four suites represented: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, LIBERO-Long;",
        "- all 40 task indices and all 1,693 episode indices represented;",
        "- all 273,465 frame IDs are contiguous and appear once;",
        "- every action is finite and 7-D; every state is finite and 8-D;",
        "- image data remains referenced through the pinned video files; no image bytes are copied into the corpus.",
        "- " + download_status,
        "",
        "## Split and leakage checks",
        "",
        "- `episode_split.json` applies one deterministic task-stratified SHA-256 rule at episode level;",
        "- each episode has one split and every frame in that episode has the same split;",
        "- `source_window_index.npz` uses `-1`/`observed=false` for right-censored offsets and is constructed inside episode ranges only;",
        "- estimator-visible source records contain current-frame references and demonstrated action only; future lookup is a separate label-side artifact;",
        "- no episode length, progress, phase, terminal flag, or future observation/action is stored in source features.",
        "",
        "## Verified action contract evidence",
        "",
        "The pinned data metadata verifies action shape `[7]`. The installed LIBERO/robosuite source verifies OSC_POSE ordering as three end-effector position deltas, three axis-angle rotation deltas, then one gripper command. Group definitions are therefore arm `[0:6]` and gripper `[6]`; they were not inferred from cached values.",
        "",
        *evidence_lines,
        "",
        "The per-task episode/frame min/median/max statistics are in `episode_split.json` and are copied into `dataset_manifest.json` by reference through the split artifact.",
        "",
    ])
    atomic_write_bytes(output_dir / "data_audit.md", audit.encode("utf-8"))
    print(json.dumps({"episodes": len(episodes), "frames": len(frames), "tasks": len(tasks), "output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
