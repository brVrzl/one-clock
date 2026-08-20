"""Policy-response cache contract and atomic shard utilities.

Adapters implement one method:

    infer(observation) -> {"actions": float32[K, 7], "latent": optional array}

The observation is current-frame-only.  This module never supplies future
frames to an adapter and never stores future-derived fields in source cache
rows.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from dataset_common import DATASET_REVISION, K_MAX, atomic_write_json, sha256_file


class PolicyAdapter(Protocol):
    policy_id: str
    metadata: dict[str, Any]

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Run exactly one frozen-policy inference for one current frame."""


def load_adapter(spec: str) -> PolicyAdapter:
    module_name, separator, attribute = spec.partition(":")
    if not separator:
        raise ValueError("Adapter must be MODULE:FACTORY")
    factory = getattr(importlib.import_module(module_name), attribute)
    adapter = factory()
    if not hasattr(adapter, "infer") or not hasattr(adapter, "policy_id"):
        raise TypeError("Adapter must expose policy_id and infer(observation)")
    return adapter


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def valid_cache_shard(path: Path, expected_frame_count: int | None = None) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"frame_id", "episode_id", "frame_index", "task_id", "source_chunks"}
            if not required.issubset(data.files):
                return False
            frame_ids = np.asarray(data["frame_id"])
            chunks = np.asarray(data["source_chunks"])
            if frame_ids.ndim != 1 or chunks.ndim != 3 or chunks.shape[0] != frame_ids.shape[0] or chunks.shape[2] != 7:
                return False
            if expected_frame_count is not None and len(frame_ids) != expected_frame_count:
                return False
            return bool(np.all(np.isfinite(chunks)))
    except (OSError, ValueError, KeyError):
        return False


def update_cache_manifest(output_dir: Path, manifest: dict[str, Any], progress: dict[str, Any]) -> None:
    manifest["progress"] = {
        "completed_shards": len(progress["completed_shards"]),
        "failed_shards": len(progress["failed_shards"]),
        "inference_calls": int(progress["inference_calls"]),
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    atomic_write_json(output_dir / "progress.json", progress)


def artifact_record(path: Path) -> dict[str, Any]:
    return {"absolute_path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def base_manifest(adapter: PolicyAdapter, dataset_root: Path, output_dir: Path, chunk_length: int) -> dict[str, Any]:
    if chunk_length < 1 or chunk_length > K_MAX:
        raise ValueError(f"chunk length must be in 1..{K_MAX}")
    return {
        "purpose": "Frozen policy-response cache; one inference per unique current frame",
        "dataset_revision": DATASET_REVISION,
        "dataset_root": str(dataset_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "policy_id": str(adapter.policy_id),
        "policy_metadata": dict(adapter.metadata),
        "chunk_length": int(chunk_length),
        "action_shape": [int(chunk_length), 7],
        "source_only_contract": {
            "stored_fields": ["frame_id", "episode_id", "frame_index", "task_id", "split", "source_chunks", "optional source_latent"],
            "grouping_only_fields": ["frame_id", "episode_id", "frame_index", "task_id", "split"],
            "forbidden_estimator_inputs": ["future_observations", "future_actions", "episode_length", "normalized_progress", "phase", "terminal_metadata", "grouping_only fields"],
        },
        "rebuild_complexity": "one adapter.infer call per unique frame_id; labels are built from cached chunks and fresh first actions",
    }
