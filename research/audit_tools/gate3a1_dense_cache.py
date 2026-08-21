#!/usr/bin/env python3
"""Build the preregistered dense, teacher-forced Gate-3A1 ACT cache.

The script is read-only with respect to the dataset, checkpoint, and historical
experiment artifacts. It writes one atomic NPZ per episode under the new
Gate-3A1 cache root and can safely resume completed episodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CHECKPOINT = Path("/home/thor/projects/checkpoints/zeromidnight_act_libero_object")
DATASET = Path("/home/thor/datasets/libero_object_25_08_23_lerobotv2.1")
INVENTORY = ROOT / "research/audit_outputs/gate3a1_inventory.json"
CACHE_ROOT = ROOT / "experiments/gate3a1_dense_temporal_cache"
COMPACT_MANIFEST = ROOT / "research/audit_outputs/gate3a1_dense_cache_manifest.json"
LEROBOT_ROOT = Path("/home/thor/projects/embodied_lab/third_party/lerobot")
REGISTRATION_COMMIT = "d163f5a76a46c9368adbb8c2f56f09e248b3a81c"
LEROBOT_COMMIT = "f66e5128ecb2456e8c54a63d15404fa59c16aebc"
DATASET_TREE_SHA256 = "2c7b87d23936dcd9d511c77234907f99e2da8ac4d23b68bb7b23af9b71297608"
DATASET_REVISION = "cbf7122bbdbaa0c50517a6a4b2ae663d0e96e51a"
SEED = 20260821
DATASET_HZ = 10.0
CHUNK_SIZE = 100
ACTION_DIM = 7
STATE_TOLERANCE = 2e-4
EXPECTED_HASHES = {
    "model.safetensors": "340071d7497238669459d93517eb3f8690862ad6fdf14207966759dfe6da9410",
    "config.json": "a76eebed357b3cbed8745c3d0f18c1335ecdd5449fcc498257676c9cbd27453d",
    "policy_preprocessor.json": "e7e3815a9e23eabe88e3dc5697cbccf8c59e61b59cf916d947dd673123426450",
    "policy_postprocessor.json": "c27cf6f42b42352f9b8f9c40da155fd4459e0ee9b85b9f23072941eb52b3ffb5",
    "policy_preprocessor_step_3_normalizer_processor.safetensors": (
        "3cb90679b116d22c960772f75e567c32b51778df2ca065cc4784bd6cd593e941"
    ),
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors": (
        "3cb90679b116d22c960772f75e567c32b51778df2ca065cc4784bd6cd593e941"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test", "all"), default="validation")
    parser.add_argument("--episode", type=int, action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--compact-manifest", type=Path, default=COMPACT_MANIFEST)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def quarantine(path: Path) -> Path:
    index = 0
    while True:
        suffix = ".corrupt" if index == 0 else f".corrupt.{index}"
        destination = path.with_name(path.name + suffix)
        if not destination.exists():
            path.replace(destination)
            return destination
        index += 1


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify_static_provenance(checkpoint: Path, dataset: Path) -> dict[str, Any]:
    if git_head(LEROBOT_ROOT) != LEROBOT_COMMIT:
        raise RuntimeError("Pinned LeRobot checkout is not at the audited commit")
    observed_hashes = {name: sha256(checkpoint / name) for name in EXPECTED_HASHES}
    if observed_hashes != EXPECTED_HASHES:
        raise RuntimeError(f"Checkpoint provenance mismatch: {observed_hashes}")
    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    expected_info = {
        "codebase_version": "v2.1",
        "total_episodes": 454,
        "total_frames": 66984,
        "total_tasks": 10,
        "fps": 10,
    }
    for key, expected in expected_info.items():
        if info.get(key) != expected:
            raise RuntimeError(f"Dataset metadata mismatch for {key}: {info.get(key)} != {expected}")
    return {
        "checkpoint_root": str(checkpoint.resolve()),
        "checkpoint_files_sha256": observed_hashes,
        "dataset_root": str(dataset.resolve()),
        "dataset_content_tree_sha256": DATASET_TREE_SHA256,
        "dataset_revision": DATASET_REVISION,
        "dataset_frequency_hz": DATASET_HZ,
        "lerobot_root": str(LEROBOT_ROOT),
        "lerobot_commit": LEROBOT_COMMIT,
        "registration_commit": REGISTRATION_COMMIT,
        "generator_git_head": git_head(ROOT),
    }


def axis_angle_to_quaternion(axis_angle: np.ndarray) -> np.ndarray:
    axis_angle = np.asarray(axis_angle, dtype=np.float32)
    angles = np.linalg.norm(axis_angle, axis=1)
    half_angles = angles / 2.0
    scale = np.empty_like(angles)
    nonzero = angles > 1e-8
    scale[nonzero] = np.sin(half_angles[nonzero]) / angles[nonzero]
    scale[~nonzero] = 0.5
    quaternion = np.zeros((axis_angle.shape[0], 4), dtype=np.float32)
    quaternion[:, :3] = axis_angle * scale[:, None]
    quaternion[:, 3] = np.cos(half_angles)
    return quaternion


def make_raw_observation_batch(
    states: np.ndarray, agent_images: np.ndarray, wrist_images: np.ndarray
) -> dict[str, Any]:
    quaternion = axis_angle_to_quaternion(states[:, 3:6])
    return {
        "pixels": {"image": agent_images, "wrist_image": wrist_images},
        "robot_state": {
            "eef": {"pos": states[:, :3], "quat": quaternion},
            "gripper": {"qpos": states[:, 6:8]},
        },
    }


def decode_all_frames(path: Path, expected: int) -> np.ndarray:
    import av

    frames: list[np.ndarray] = []
    with av.open(str(path), mode="r") as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if len(frames) != expected:
        raise RuntimeError(f"Decoded {len(frames)} frames from {path}; expected {expected}")
    result = np.stack(frames, axis=0)
    if result.shape != (expected, 256, 256, 3) or result.dtype != np.uint8:
        raise RuntimeError(f"Unexpected decoded video array for {path}: {result.shape} {result.dtype}")
    return result


def load_episode(dataset: Path, episode_id: int) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq

    path = dataset / "data/chunk-000" / f"episode_{episode_id:06d}.parquet"
    columns = [
        "observation.state",
        "frame_index",
        "episode_index",
        "task_index",
        "index",
        "timestamp",
    ]
    table = pq.read_table(path, columns=columns)
    result = {
        "state": np.asarray(table["observation.state"].to_pylist(), dtype=np.float32),
        "frame_index": np.asarray(table["frame_index"].to_pylist(), dtype=np.int64),
        "episode_index": np.asarray(table["episode_index"].to_pylist(), dtype=np.int64),
        "task_index": np.asarray(table["task_index"].to_pylist(), dtype=np.int64),
        "dataset_index": np.asarray(table["index"].to_pylist(), dtype=np.int64),
        "timestamp": np.asarray(table["timestamp"].to_pylist(), dtype=np.float64),
    }
    length = len(result["frame_index"])
    if result["state"].shape != (length, 8):
        raise RuntimeError(f"Unexpected state shape for episode {episode_id}: {result['state'].shape}")
    if not np.array_equal(result["frame_index"], np.arange(length)):
        raise RuntimeError(f"Episode {episode_id} frame indices are not a dense local sequence")
    if set(result["episode_index"].tolist()) != {episode_id}:
        raise RuntimeError(f"Episode {episode_id} parquet contains another episode ID")
    expected_time = result["frame_index"] / DATASET_HZ
    if not np.allclose(result["timestamp"], expected_time, atol=1e-5, rtol=0.0):
        raise RuntimeError(f"Episode {episode_id} timestamps do not match the 10 Hz frame clock")
    return result


def prepare_policy_batch(
    raw_observation: dict[str, Any],
    expected_states: np.ndarray,
    env_preprocessor: Any,
    policy_preprocessor: Any,
) -> dict[str, Any]:
    from lerobot.envs.utils import preprocess_observation

    processed = preprocess_observation(raw_observation)
    processed = env_preprocessor(processed)
    recovered = processed["observation.state"].detach().cpu().numpy()
    error = float(np.max(np.abs(recovered - expected_states)))
    if error > STATE_TOLERANCE:
        raise RuntimeError(f"LIBERO state reconstruction mismatch: max error {error}")
    return policy_preprocessor(processed)


def infer_episode(
    *,
    states: np.ndarray,
    agent_images: np.ndarray,
    wrist_images: np.ndarray,
    batch_size: int,
    policy: Any,
    policy_preprocessor: Any,
    policy_postprocessor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
) -> tuple[np.ndarray, int]:
    import torch
    from lerobot.utils.constants import ACTION

    outputs: list[np.ndarray] = []
    forward_batches = 0
    for start in range(0, len(states), batch_size):
        stop = min(start + batch_size, len(states))
        raw = make_raw_observation_batch(states[start:stop], agent_images[start:stop], wrist_images[start:stop])
        model_observation = prepare_policy_batch(
            raw, states[start:stop], env_preprocessor, policy_preprocessor
        )
        with torch.inference_mode():
            normalized = policy.predict_action_chunk(model_observation)
            chunk = policy_postprocessor(normalized)
        chunk = env_postprocessor({ACTION: chunk})[ACTION]
        array = chunk.detach().cpu().numpy().astype(np.float32, copy=False)
        expected_shape = (stop - start, CHUNK_SIZE, ACTION_DIM)
        if array.shape != expected_shape or not np.isfinite(array).all():
            raise RuntimeError(f"Invalid ACT output: shape={array.shape}, expected={expected_shape}")
        outputs.append(array)
        forward_batches += 1
    return np.concatenate(outputs, axis=0), forward_batches


def episode_path(cache_root: Path, split: str, episode_id: int) -> Path:
    return cache_root / split / f"episode_{episode_id:06d}.npz"


def validate_episode_file(
    path: Path,
    *,
    episode_id: int,
    task_id: int,
    split: str,
    expected_frames: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as cache:
        required = {
            "predicted_chunks",
            "episode_id",
            "task_id",
            "split",
            "dataset_frame",
            "dataset_index",
            "source_time_dataset_steps",
            "source_time_seconds",
            "chunk_length",
            "action_dim",
            "dataset_frequency_hz",
            "provenance_json",
        }
        if not required.issubset(cache.files):
            raise RuntimeError(f"Cache file lacks arrays: {sorted(required - set(cache.files))}")
        chunks = cache["predicted_chunks"]
        frames = cache["dataset_frame"]
        if chunks.shape != (expected_frames, CHUNK_SIZE, ACTION_DIM):
            raise RuntimeError(f"Wrong cache shape {chunks.shape}")
        if chunks.dtype != np.float32 or not np.isfinite(chunks).all():
            raise RuntimeError("Predicted chunks are non-finite or not float32")
        if not np.array_equal(frames, np.arange(expected_frames)) or len(np.unique(frames)) != expected_frames:
            raise RuntimeError("Missing or duplicate source frame")
        scalar_checks = {
            "episode_id": episode_id,
            "task_id": task_id,
            "split": split,
            "chunk_length": CHUNK_SIZE,
            "action_dim": ACTION_DIM,
            "dataset_frequency_hz": DATASET_HZ,
        }
        for name, expected in scalar_checks.items():
            observed = cache[name].item()
            if observed != expected:
                raise RuntimeError(f"{name} mismatch: {observed!r} != {expected!r}")
        expected_steps = np.arange(expected_frames, dtype=np.int64)
        if not np.array_equal(cache["source_time_dataset_steps"], expected_steps):
            raise RuntimeError("Source-time step mismatch")
        if not np.allclose(cache["source_time_seconds"], expected_steps / DATASET_HZ):
            raise RuntimeError("Source-time second mismatch")
        file_provenance = json.loads(str(cache["provenance_json"].item()))
        for key in (
            "checkpoint_files_sha256",
            "dataset_content_tree_sha256",
            "dataset_revision",
            "lerobot_commit",
            "registration_commit",
        ):
            if file_provenance.get(key) != provenance.get(key):
                raise RuntimeError(f"Provenance mismatch for {key}")
    return {
        "episode_id": episode_id,
        "task_id": task_id,
        "split": split,
        "expected_frames": expected_frames,
        "completed_frames": expected_frames,
        "cache_file": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "checkpoint_sha256": EXPECTED_HASHES["model.safetensors"],
        "status": "complete",
    }


def manifest_payload(
    *,
    cache_root: Path,
    provenance: dict[str, Any],
    entries: list[dict[str, Any]],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    complete = [entry for entry in entries if entry.get("status") == "complete"]
    by_split: dict[str, dict[str, int]] = {}
    for split in ("validation", "test"):
        current = [entry for entry in complete if entry["split"] == split]
        by_split[split] = {
            "complete_episodes": len(current),
            "completed_source_queries": int(sum(entry["completed_frames"] for entry in current)),
        }
    all_complete = len(complete) == 82 and sum(entry["completed_frames"] for entry in complete) == 12294
    return {
        "schema_version": 1,
        "scope": "Gate-3A1 dense teacher-forced ACT prediction cache; local arrays excluded from Git.",
        "cache_root": str(cache_root.resolve()),
        "provenance": provenance,
        "runtime": runtime,
        "expected": {"episodes": 82, "source_act_queries": 12294, "chunk_length": 100, "action_dim": 7},
        "completion": {"by_split": by_split, "all_complete": all_complete},
        "entries": sorted(entries, key=lambda item: (item["split"], item["episode_id"])),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    checkpoint = args.checkpoint.resolve()
    dataset = args.dataset.resolve()
    cache_root = args.cache_root.resolve()
    provenance = verify_static_provenance(checkpoint, dataset)

    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    episodes_meta = {int(row["episode_index"]): row for row in load_json_lines(dataset / "meta/episodes.jsonl")}
    split_ids = {
        split: [int(value) for value in inventory["splits"][split]["episode_ids"]]
        for split in ("validation", "test")
    }
    episode_split = {episode: split for split, ids in split_ids.items() for episode in ids}
    selected_splits = ("validation", "test") if args.split == "all" else (args.split,)
    selected_ids = [episode for split in selected_splits for episode in split_ids[split]]
    if args.episode:
        requested = set(args.episode)
        unknown = requested - set(selected_ids)
        if unknown:
            raise ValueError(f"Requested episodes are outside split selection: {sorted(unknown)}")
        selected_ids = [episode for episode in selected_ids if episode in requested]

    import torch
    import pandas
    import scipy
    import pyarrow
    import av
    import lerobot

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    from scripts.run_libero_gate0 import load_policy_and_processors

    runtime_config = {
        "task_suite": "libero_object",
        "task_id": 0,
        "obs_type": "pixels_agent_pos",
        "camera_name": "agentview_image,robot0_eye_in_hand_image",
        "camera_name_mapping": {
            "agentview_image": "image",
            "robot0_eye_in_hand_image": "wrist_image",
        },
        "observation_width": 256,
        "observation_height": 256,
        "control_freq": 20,
        "init_states": True,
        "hard_reset": True,
        "control_mode": "relative",
        "device": args.device,
    }
    policy, policy_preprocessor, policy_postprocessor, env_preprocessor, env_postprocessor = (
        load_policy_and_processors(runtime_config, checkpoint)
    )
    policy.eval()
    if policy.config.temporal_ensemble_coeff is not None:
        raise RuntimeError("Dense source cache requires policy-side temporal ensemble disabled")
    if int(policy.config.chunk_size) != CHUNK_SIZE or int(policy.config.output_features["action"].shape[0]) != ACTION_DIM:
        raise RuntimeError("Checkpoint chunk/action contract mismatch")

    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_requested": args.device,
        "device_observed": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "lerobot": getattr(lerobot, "__version__", "unknown"),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "pyarrow": pyarrow.__version__,
        "pyav": av.__version__,
        "seed": SEED,
        "batch_size": args.batch_size,
        "inference_mode": True,
        "policy_eval": True,
        "amp": False,
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }

    working_manifest_path = cache_root / "manifest.json"
    existing_entries: dict[tuple[str, int], dict[str, Any]] = {}
    if working_manifest_path.exists():
        old = json.loads(working_manifest_path.read_text(encoding="utf-8"))
        existing_entries = {
            (str(entry["split"]), int(entry["episode_id"])): entry
            for entry in old.get("entries", [])
            if entry.get("status") == "complete"
        }

    new_forward_batches = 0
    new_source_queries = 0
    for position, episode_id in enumerate(selected_ids, start=1):
        split = episode_split[episode_id]
        metadata = episodes_meta[episode_id]
        expected_frames = int(metadata["length"])
        parquet = load_episode(dataset, episode_id)
        task_values = set(int(value) for value in parquet["task_index"])
        if len(task_values) != 1:
            raise RuntimeError(f"Episode {episode_id} has multiple task IDs")
        task_id = next(iter(task_values))
        path = episode_path(cache_root, split, episode_id)
        if path.exists():
            try:
                entry = validate_episode_file(
                    path,
                    episode_id=episode_id,
                    task_id=task_id,
                    split=split,
                    expected_frames=expected_frames,
                    provenance=provenance,
                )
                existing_entries[(split, episode_id)] = entry
                print(f"[{position}/{len(selected_ids)}] verified existing {split} episode {episode_id}", flush=True)
                continue
            except Exception as error:
                destination = quarantine(path)
                print(f"quarantined invalid cache {path} -> {destination}: {error}", flush=True)

        agent_path = dataset / "videos/chunk-000/observation.images.image" / f"episode_{episode_id:06d}.mp4"
        wrist_path = dataset / "videos/chunk-000/observation.images.wrist_image" / f"episode_{episode_id:06d}.mp4"
        agent_images = decode_all_frames(agent_path, expected_frames)
        wrist_images = decode_all_frames(wrist_path, expected_frames)
        chunks, forward_batches = infer_episode(
            states=parquet["state"],
            agent_images=agent_images,
            wrist_images=wrist_images,
            batch_size=args.batch_size,
            policy=policy,
            policy_preprocessor=policy_preprocessor,
            policy_postprocessor=policy_postprocessor,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
        )
        file_provenance = dict(provenance)
        file_provenance["runtime"] = runtime
        arrays = {
            "predicted_chunks": chunks,
            "episode_id": np.asarray(episode_id, dtype=np.int64),
            "task_id": np.asarray(task_id, dtype=np.int16),
            "split": np.asarray(split),
            "dataset_frame": parquet["frame_index"].astype(np.int64),
            "dataset_index": parquet["dataset_index"].astype(np.int64),
            "source_time_dataset_steps": parquet["frame_index"].astype(np.int64),
            "source_time_seconds": (parquet["frame_index"] / DATASET_HZ).astype(np.float64),
            "chunk_length": np.asarray(CHUNK_SIZE, dtype=np.int16),
            "action_dim": np.asarray(ACTION_DIM, dtype=np.int8),
            "dataset_frequency_hz": np.asarray(DATASET_HZ, dtype=np.float64),
            "provenance_json": np.asarray(json.dumps(file_provenance, sort_keys=True)),
        }
        atomic_npz(path, arrays)
        entry = validate_episode_file(
            path,
            episode_id=episode_id,
            task_id=task_id,
            split=split,
            expected_frames=expected_frames,
            provenance=provenance,
        )
        existing_entries[(split, episode_id)] = entry
        new_forward_batches += forward_batches
        new_source_queries += expected_frames
        payload = manifest_payload(
            cache_root=cache_root,
            provenance=provenance,
            entries=list(existing_entries.values()),
            runtime={**runtime, "new_forward_batches_this_invocation": new_forward_batches, "new_source_queries_this_invocation": new_source_queries},
        )
        atomic_json(working_manifest_path, payload)
        atomic_json(args.compact_manifest, payload)
        print(
            f"[{position}/{len(selected_ids)}] cached {split} episode {episode_id}: "
            f"{expected_frames} sources, sha256={entry['sha256'][:12]}",
            flush=True,
        )

    payload = manifest_payload(
        cache_root=cache_root,
        provenance=provenance,
        entries=list(existing_entries.values()),
        runtime={**runtime, "new_forward_batches_this_invocation": new_forward_batches, "new_source_queries_this_invocation": new_source_queries},
    )
    atomic_json(working_manifest_path, payload)
    atomic_json(args.compact_manifest, payload)
    print(json.dumps({"completion": payload["completion"], "new_source_queries": new_source_queries}, indent=2))


if __name__ == "__main__":
    main()
