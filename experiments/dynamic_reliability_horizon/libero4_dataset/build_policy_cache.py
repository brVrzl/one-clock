#!/usr/bin/env python3
"""Build deterministic, resumable per-frame policy-response cache shards."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from dataset_common import DATASET_REVISION, K_MAX, atomic_write_json
from policy_cache import atomic_save_npz, base_manifest, load_adapter, update_cache_manifest, valid_cache_shard


_DATA_TABLE_CACHE: dict[str, Any] = {}
_VIDEO_CONTAINER_CACHE: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter", required=True, help="MODULE:FACTORY returning a frozen PolicyAdapter")
    parser.add_argument("--episodes-per-shard", type=int, default=20)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--max-episode-attempts", type=int, default=3)
    return parser.parse_args()


def read_rows(corpus_dir: Path) -> list[dict[str, Any]]:
    return pq.read_table(corpus_dir / "frame_index.parquet").to_pylist()


def decode_video_frame(path: Path, timestamp_sec: float) -> np.ndarray:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required for image-backed policy cache generation") from exc
    key = str(path)
    container = _VIDEO_CONTAINER_CACHE.get(key)
    if container is None:
        container = av.open(key)
        _VIDEO_CONTAINER_CACHE[key] = container
    stream = container.streams.video[0]
    seek_value = int(max(0.0, timestamp_sec) / float(stream.time_base))
    container.seek(seek_value, stream=stream, any_frame=False, backward=True)
    selected = None
    for frame in container.decode(stream):
        selected = frame
        if frame.time is None or frame.time >= timestamp_sec:
            break
    if selected is None:
        raise RuntimeError(f"No video frame at timestamp {timestamp_sec}: {path}")
    return selected.to_ndarray(format="rgb24")


def source_observation(dataset_root: Path, frame: dict[str, Any]) -> dict[str, Any]:
    data_path = dataset_root / str(frame["data_path"])
    table_key = str(data_path)
    table = _DATA_TABLE_CACHE.get(table_key)
    if table is None:
        table = pq.read_table(data_path, columns=["observation.state"])
        _DATA_TABLE_CACHE[table_key] = table
    state = np.asarray(table.column("observation.state")[int(frame["data_row_index"])].as_py(), dtype=np.float32)
    image_ref = json.loads(str(frame["observation_image_ref"]))
    image2_ref = json.loads(str(frame["observation_image2_ref"]))
    return {
        "images": {
            "image": decode_video_frame(dataset_root / image_ref["path"], float(image_ref["timestamp_sec"])),
            "image2": decode_video_frame(dataset_root / image2_ref["path"], float(image2_ref["timestamp_sec"])),
        },
        "state": state,
        "frame_id": int(frame["frame_id"]),
        "episode_id": int(frame["episode_id"]),
        "task_id": int(frame["task_id"]),
        "task_name": str(frame["task_name"]),
    }


def expected_shards(episodes: list[int], episodes_per_shard: int) -> list[list[int]]:
    return [episodes[start : start + episodes_per_shard] for start in range(0, len(episodes), episodes_per_shard)]


def main() -> int:
    args = parse_args()
    if args.episodes_per_shard < 1:
        raise ValueError("episodes-per-shard must be positive")
    if args.max_episode_attempts < 1:
        raise ValueError("max-episode-attempts must be positive")
    corpus_dir = args.corpus_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(corpus_dir)
    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[int(row["episode_id"])].append(row)
    episode_ids = sorted(by_episode)
    if args.limit_episodes is not None:
        episode_ids = episode_ids[: max(0, args.limit_episodes)]
    shards = expected_shards(episode_ids, args.episodes_per_shard)
    adapter = load_adapter(args.adapter)
    if not episode_ids:
        raise RuntimeError("No episodes selected for policy cache")
    chunk_length = int(adapter.metadata.get("chunk_length", 0))
    if not 1 <= chunk_length <= K_MAX:
        raise RuntimeError("Adapter metadata must declare chunk_length in 1..100")
    manifest = base_manifest(adapter, dataset_root, output_dir, chunk_length)
    manifest["episode_shards"] = [
        {"shard_id": index, "episode_ids": shard, "path": f"shard-{index:04d}.npz"}
        for index, shard in enumerate(shards)
    ]
    progress_path = output_dir / "progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    else:
        progress = {"dataset_revision": DATASET_REVISION, "completed_shards": [], "failed_shards": [], "attempts": {}, "inference_calls": 0, "started_unix": int(time.time()), "throughput_calls_per_sec": None}
    progress["inference_calls"] = int(progress.get("inference_calls", 0))
    progress.setdefault("completed_shards", [])
    progress.setdefault("failed_shards", [])
    progress.setdefault("attempts", {})
    progress.setdefault("episode_attempts", {})
    progress.setdefault("completed_episode_ids", {})
    for shard_id, episode_list in enumerate(shards):
        shard_path = output_dir / f"shard-{shard_id:04d}.npz"
        partial_path = output_dir / f"shard-{shard_id:04d}.partial"
        expected_count = sum(len(by_episode[episode]) for episode in episode_list)
        if str(shard_id) in {str(item) for item in progress.get("completed_shards", [])} and valid_cache_shard(shard_path, expected_count):
            continue
        if valid_cache_shard(shard_path, expected_count):
            if shard_id not in progress["completed_shards"]:
                progress["completed_shards"].append(shard_id)
            update_cache_manifest(output_dir, manifest, progress)
            continue
        progress["attempts"][str(shard_id)] = int(progress.get("attempts", {}).get(str(shard_id), 0)) + 1

        # A partial file is deliberately extensionless so target generation's
        # shard-*.npz glob never treats it as a complete policy cache shard.
        frame_ids: list[int] = []
        episode_values: list[int] = []
        frame_indices: list[int] = []
        task_values: list[int] = []
        split_values: list[str] = []
        source_chunks: list[np.ndarray] = []
        latent_values: list[np.ndarray] = []
        latent_enabled: bool | None = None
        if partial_path.is_file():
            try:
                with np.load(partial_path, allow_pickle=False) as data:
                    partial_episode = np.asarray(data["episode_id"], dtype=np.int32)
                    partial_chunks = np.asarray(data["source_chunks"], dtype=np.float32)
                    if partial_chunks.ndim != 3 or partial_chunks.shape[1:] != (chunk_length, 7):
                        raise RuntimeError("partial shard has an unexpected source_chunks shape")
                    for episode_id in set(int(value) for value in partial_episode):
                        if int(np.sum(partial_episode == episode_id)) != len(by_episode[episode_id]):
                            raise RuntimeError(f"partial shard contains incomplete episode {episode_id}")
                    frame_ids = np.asarray(data["frame_id"], dtype=np.int64).tolist()
                    episode_values = partial_episode.tolist()
                    frame_indices = np.asarray(data["frame_index"], dtype=np.int32).tolist()
                    task_values = np.asarray(data["task_id"], dtype=np.int16).tolist()
                    split_values = np.asarray(data["split"]).astype(str).tolist()
                    source_chunks = [row.copy() for row in partial_chunks]
                    if "source_latents" in data.files:
                        partial_latents = np.asarray(data["source_latents"], dtype=np.float32)
                        if partial_latents.shape[0] != len(source_chunks) or not np.all(np.isfinite(partial_latents)):
                            raise RuntimeError("partial shard has invalid source_latents")
                        latent_values = [row.copy() for row in partial_latents]
                        latent_enabled = True
                    else:
                        latent_enabled = False
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                raise RuntimeError(f"cannot resume partial shard {partial_path}: {exc}") from exc

        completed_episode_ids = {int(value) for value in episode_values}
        progress["completed_episode_ids"][str(shard_id)] = sorted(completed_episode_ids)
        shard_start = time.monotonic()
        failed_episodes: list[dict[str, Any]] = []

        def save_partial() -> None:
            arrays: dict[str, np.ndarray] = {
                "frame_id": np.asarray(frame_ids, dtype=np.int64),
                "episode_id": np.asarray(episode_values, dtype=np.int32),
                "frame_index": np.asarray(frame_indices, dtype=np.int32),
                "task_id": np.asarray(task_values, dtype=np.int16),
                "split": np.asarray(split_values),
                "source_chunks": np.stack(source_chunks).astype(np.float32),
            }
            if latent_enabled:
                arrays["source_latents"] = np.stack(latent_values).astype(np.float32)
                manifest["latent_shape"] = list(arrays["source_latents"].shape[1:])
            atomic_save_npz(partial_path, arrays)

        for episode_id in episode_list:
            if episode_id in completed_episode_ids:
                continue
            attempt_key = str(episode_id)
            prior_attempts = int(progress["episode_attempts"].get(attempt_key, 0))
            if prior_attempts >= args.max_episode_attempts:
                failed_episodes.append({"episode_id": episode_id, "error": "max episode attempts reached", "attempts": prior_attempts})
                continue
            progress["episode_attempts"][attempt_key] = prior_attempts + 1
            episode_chunks: list[np.ndarray] = []
            episode_latents: list[np.ndarray] = []
            try:
                for frame in by_episode[episode_id]:
                    response = adapter.infer(source_observation(dataset_root, frame))
                    chunk = np.asarray(response["actions"], dtype=np.float32)
                    if chunk.shape != (chunk_length, 7) or not np.all(np.isfinite(chunk)):
                        raise RuntimeError(f"Invalid adapter response for frame {frame['frame_id']}: {chunk.shape}")
                    episode_chunks.append(chunk)
                    if response.get("latent") is not None:
                        episode_latents.append(np.asarray(response["latent"], dtype=np.float32))
                    progress["inference_calls"] = int(progress.get("inference_calls", 0)) + 1
                if bool(episode_latents) != bool(latent_enabled):
                    if latent_enabled is False and episode_latents:
                        raise RuntimeError("adapter returned latents after earlier responses had none")
                    if latent_enabled is True and not episode_latents:
                        raise RuntimeError("adapter omitted latents after earlier responses returned them")
                if episode_latents and len(episode_latents) != len(episode_chunks):
                    raise RuntimeError("adapter returned latents for only some frames in an episode")
                if latent_enabled is None:
                    latent_enabled = bool(episode_latents)
                for frame, chunk in zip(by_episode[episode_id], episode_chunks):
                    frame_ids.append(int(frame["frame_id"]))
                    episode_values.append(int(frame["episode_id"]))
                    frame_indices.append(int(frame["frame_index"]))
                    task_values.append(int(frame["task_id"]))
                    split_values.append(str(frame["split"]))
                    source_chunks.append(chunk)
                latent_values.extend(episode_latents)
                completed_episode_ids.add(episode_id)
                progress["completed_episode_ids"][str(shard_id)] = sorted(completed_episode_ids)
                save_partial()
                update_cache_manifest(output_dir, manifest, progress)
            except Exception as exc:  # one bad episode must not stop the shard
                failed_episodes.append({"episode_id": episode_id, "error": repr(exc), "attempts": progress["episode_attempts"][attempt_key]})
                print(f"FAILED episode {episode_id} in shard {shard_id}: {exc}", flush=True)

        missing_episodes = [episode_id for episode_id in episode_list if episode_id not in completed_episode_ids]
        if not missing_episodes:
            arrays = {
                "frame_id": np.asarray(frame_ids, dtype=np.int64),
                "episode_id": np.asarray(episode_values, dtype=np.int32),
                "frame_index": np.asarray(frame_indices, dtype=np.int32),
                "task_id": np.asarray(task_values, dtype=np.int16),
                "split": np.asarray(split_values),
                "source_chunks": np.stack(source_chunks).astype(np.float32),
            }
            if latent_enabled:
                arrays["source_latents"] = np.stack(latent_values).astype(np.float32)
                manifest["latent_shape"] = list(arrays["source_latents"].shape[1:])
            atomic_save_npz(shard_path, arrays)
            progress["completed_shards"] = sorted(set(progress.get("completed_shards", [])) | {shard_id})
            progress["failed_shards"] = [item for item in progress.get("failed_shards", []) if int(item.get("shard_id", -1)) != shard_id]
            elapsed = max(time.monotonic() - shard_start, 1e-6)
            progress["throughput_calls_per_sec"] = float(expected_count / elapsed)
        else:
            failure = {"shard_id": shard_id, "episode_ids": missing_episodes, "errors": failed_episodes}
            progress["failed_shards"] = [item for item in progress.get("failed_shards", []) if int(item.get("shard_id", -1)) != shard_id]
            progress["failed_shards"].append(failure)
            print(f"FAILED shard {shard_id}; remaining episodes={missing_episodes}", flush=True)
        update_cache_manifest(output_dir, manifest, progress)
        print(f"shard={shard_id + 1}/{len(shards)} completed={len(progress['completed_shards'])} failed={len(progress['failed_shards'])}", flush=True)
    manifest["complete"] = len(progress["completed_shards"]) == len(shards)
    update_cache_manifest(output_dir, manifest, progress)
    return 0 if manifest["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
