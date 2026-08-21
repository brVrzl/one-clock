#!/usr/bin/env python3
"""Materialize exact causal state and frozen-ACT encoder context features.

This command consumes the existing portable Y_refresh bundle and its aligned
source metadata.  It never rebuilds targets or resamples source windows.  The
only model call is the frozen ACT inference call used by the original target
construction, with a forward hook on ``policy.model.encoder`` to copy the
source-conditioned context token.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BUNDLE_PATH = ROOT / "experiments/dynamic_reliability_horizon/artifact_handoff/minimal_y_refresh_training_bundle.npz"
SPLIT_PATH = ROOT / "experiments/dynamic_reliability_horizon/artifact_handoff/episode_split_manifest.json"
METADATA_PATH = ROOT / "experiments/temporal_reliability/metadata.jsonl"
TARGET_CACHE_PATH = ROOT / "experiments/temporal_reliability_target_comparison/target_comparison.npz"
DATASET_ROOT = Path("/home/thor/datasets/libero_object_25_08_23_lerobotv2.1")
CHECKPOINT_ROOT = Path("/home/thor/projects/checkpoints/zeromidnight_act_libero_object")
OUTPUT_DIR = ROOT / "experiments/dynamic_reliability_horizon/source_context_ablation"
FEATURE_PATH = OUTPUT_DIR / "feature_bundle.npz"
EXPECTED_BUNDLE_SHA256 = "45a37a57fc03a3850b5c87e88604d66b16886d306e5ee09aa322f52c7e6c50b4"
GROUP_NAMES = ("arm", "gripper")
CHUNK_SIZE = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def load_metadata() -> list[dict[str, Any]]:
    return [json.loads(line) for line in METADATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_bundle() -> dict[str, np.ndarray]:
    digest = sha256(BUNDLE_PATH)
    if digest != EXPECTED_BUNDLE_SHA256:
        raise RuntimeError(f"Y_refresh bundle checksum mismatch: {digest}")
    with np.load(BUNDLE_PATH, allow_pickle=False) as arrays:
        expected = {
            "source_chunk_actions",
            "group_ids",
            "offsets",
            "y_refresh",
            "label_observed",
            "episode_index",
            "split_membership",
        }
        if set(arrays.files) != expected:
            raise RuntimeError(f"unexpected handoff arrays: {arrays.files}")
        result = {name: np.asarray(arrays[name]).copy() for name in arrays.files}
    if result["source_chunk_actions"].shape != (3740, 100, 7):
        raise RuntimeError("source chunk shape is not (3740, 100, 7)")
    if result["y_refresh"].shape != (3740, 2, 99):
        raise RuntimeError("Y_refresh shape is not (3740, 2, 99)")
    if result["label_observed"].shape != (3740, 2, 99):
        raise RuntimeError("label censor mask shape is not (3740, 2, 99)")
    if not np.array_equal(result["group_ids"], np.asarray([0, 1], dtype=np.int8)):
        raise RuntimeError("unexpected group IDs")
    if not np.array_equal(result["offsets"], np.arange(1, 100, dtype=np.int16)):
        raise RuntimeError("bundle offsets are not exactly k=1..99")
    return result


def validate_keys_and_split(
    bundle: dict[str, np.ndarray], metadata: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if len(metadata) != 3740:
        raise RuntimeError(f"expected 3740 metadata rows, got {len(metadata)}")
    episode_ids = np.asarray([int(row["episode_index"]) for row in metadata], dtype=np.int32)
    source_steps = np.asarray([int(row["frame_index"]) for row in metadata], dtype=np.int32)
    if not np.array_equal(episode_ids, bundle["episode_index"]):
        raise RuntimeError("metadata episode IDs are not aligned to the portable bundle")
    keys = list(zip(episode_ids.tolist(), source_steps.tolist(), strict=True))
    if len(set(keys)) != len(keys):
        duplicates = sorted(key for key in set(keys) if keys.count(key) > 1)
        raise RuntimeError(f"duplicate source keys: {duplicates[:5]}")

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    split_by_episode: dict[int, int] = {}
    split_names = ("train", "validation", "test")
    for code, name in enumerate(split_names):
        for episode in split["episodes_by_split"][name]:
            episode = int(episode)
            if episode in split_by_episode:
                raise RuntimeError(f"episode appears in multiple splits: {episode}")
            split_by_episode[episode] = code
    expected_split = np.asarray([split_by_episode.get(int(ep), -1) for ep in episode_ids], dtype=np.int8)
    if np.any(expected_split < 0) or not np.array_equal(expected_split, bundle["split_membership"]):
        raise RuntimeError("episode-level split membership is missing or inconsistent")
    split_sets = {
        name: set(episode_ids[bundle["split_membership"] == code].tolist())
        for code, name in enumerate(split_names)
    }
    if set.union(*split_sets.values()) != set(episode_ids.tolist()):
        raise RuntimeError("not every source episode is assigned to a split")
    if any(split_sets[left] & split_sets[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise RuntimeError("episode leakage across train/validation/test")

    # Optional read-only provenance check: the historical label-side cache is
    # not required for this experiment and is absent on this checkout. When it
    # is available, reconcile it without constructing or resampling Y_refresh.
    target_cache_audit: dict[str, Any]
    if not TARGET_CACHE_PATH.exists():
        target_cache_audit = {
            "available": False,
            "path": str(TARGET_CACHE_PATH.resolve()),
            "missing_explicitly_reported": True,
            "effect": "not required; canonical portable bundle remains the frozen label source",
        }
    else:
        with np.load(TARGET_CACHE_PATH, allow_pickle=False) as target_cache:
            if not np.array_equal(target_cache["episode_index"].astype(np.int32), episode_ids):
                raise RuntimeError("target cache episode IDs are not aligned to metadata")
            if not np.array_equal(target_cache["frame_index"].astype(np.int32), source_steps):
                raise RuntimeError("target cache source steps are not aligned to metadata")
            if not np.array_equal(target_cache["old_predicted_actions"].astype(np.float32), bundle["source_chunk_actions"]):
                raise RuntimeError("portable source chunks differ from the original target cache")
            target_y = np.stack(
                [target_cache["arm_refresh_survival"][:, 1:], target_cache["gripper_refresh_survival"][:, 1:]],
                axis=1,
            ).astype(bool)
            if not np.array_equal(target_y, bundle["y_refresh"]):
                raise RuntimeError("portable Y_refresh differs from the original target cache")
        target_cache_audit = {
            "available": True,
            "path": str(TARGET_CACHE_PATH.resolve()),
            "sha256": sha256(TARGET_CACHE_PATH),
            "reconciled_without_regeneration": True,
        }

    audit = {
        "expected_sources": 3740,
        "unique_source_keys": len(set(keys)),
        "duplicate_source_keys": 0,
        "source_key": "(episode_index, frame_index) == (episode_id, source_step)",
        "split_episode_counts": {name: len(values) for name, values in split_sets.items()},
        "split_source_counts": {
            name: int(np.sum(bundle["split_membership"] == code))
            for code, name in enumerate(split_names)
        },
        "train_validation_test_episode_intersections": 0,
        "target_cache_sha256": sha256(TARGET_CACHE_PATH) if TARGET_CACHE_PATH.exists() else None,
        "metadata_sha256": sha256(METADATA_PATH),
        "target_cache_reconciliation": target_cache_audit,
    }
    return episode_ids, source_steps, audit


def recover_source_states(
    episode_ids: np.ndarray, source_steps: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read only observation.state at each exact source key from parquet."""
    import pyarrow.parquet as pq

    required: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row_index, (episode, step) in enumerate(zip(episode_ids, source_steps, strict=True)):
        required[int(episode)].append((int(step), row_index))
    states = np.full((len(episode_ids), 8), np.nan, dtype=np.float32)
    matched = 0
    episode_lengths: dict[int, int] = {}
    for episode, requests in sorted(required.items()):
        path = DATASET_ROOT / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
        if not path.exists():
            raise RuntimeError(f"missing source parquet for episode {episode}: {path}")
        table = pq.read_table(path, columns=["observation.state", "episode_index"])
        table_episode_ids = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        if set(table_episode_ids.tolist()) != {episode}:
            raise RuntimeError(f"unexpected episode IDs in {path}")
        episode_states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        if episode_states.shape != (len(table), 8):
            raise RuntimeError(f"episode {episode} state shape is {episode_states.shape}, expected ({len(table)}, 8)")
        episode_lengths[episode] = int(len(episode_states))
        for step, row_index in requests:
            if step < 0 or step >= len(episode_states):
                raise RuntimeError(f"source key {(episode, step)} is outside episode data")
            if np.isfinite(states[row_index]).all():
                raise RuntimeError(f"ambiguous state match for {(episode, step)}")
            states[row_index] = episode_states[step]
            matched += 1
    if matched != len(states) or not np.isfinite(states).all():
        missing = np.flatnonzero(~np.isfinite(states).all(axis=1)).tolist()
        raise RuntimeError(f"missing source states: {missing[:10]}")
    return states, {
        "matches": matched,
        "missing_matches": 0,
        "ambiguous_matches": 0,
        "shape": list(states.shape),
        "dtype": str(states.dtype),
        "episode_lengths_read_for_lookup_only": episode_lengths,
        "semantics": [
            "observation.state[0:3]: current EEF Cartesian position",
            "observation.state[3:6]: current EEF axis-angle orientation",
            "observation.state[6:8]: current two-finger gripper qpos",
        ],
        "future_fields_read": [],
    }


def decode_selected_frames(path: Path, frame_indices: list[int]) -> dict[int, np.ndarray]:
    import av

    wanted = set(int(index) for index in frame_indices)
    decoded: dict[int, np.ndarray] = {}
    with av.open(str(path), mode="r") as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                decoded[index] = frame.to_ndarray(format="rgb24")
            if wanted and index >= max(wanted) and len(decoded) == len(wanted):
                break
    missing = wanted - decoded.keys()
    if missing:
        raise RuntimeError(f"could not decode frames {sorted(missing)} from {path}")
    return decoded


def load_act_runtime() -> tuple[Any, Any, Any, Any, Any]:
    from scripts.run_libero_gate0 import load_policy_and_processors

    config = {
        "task_suite": "libero_object",
        "task_id": 0,
        "obs_type": "pixels_agent_pos",
        "camera_name": "agentview_image,robot0_eye_in_hand_image",
        "camera_name_mapping": {"agentview_image": "image", "robot0_eye_in_hand_image": "wrist_image"},
        "observation_width": 256,
        "observation_height": 256,
        "control_freq": 20,
        "init_states": True,
        "hard_reset": True,
        "control_mode": "relative",
        "device": "cuda",
    }
    return load_policy_and_processors(config, CHECKPOINT_ROOT)


def infer_batch(
    states: np.ndarray,
    agent_images: list[np.ndarray],
    wrist_images: list[np.ndarray],
    policy: Any,
    policy_preprocessor: Any,
    policy_postprocessor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    *,
    extract_latent: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    import torch

    from experiments.group_prediction_persistence.audit import make_raw_observation_batch, prepare_policy_batch
    from lerobot.utils.constants import ACTION

    raw = make_raw_observation_batch(states, agent_images, wrist_images)
    batch = prepare_policy_batch(raw, states, env_preprocessor, policy_preprocessor)
    captured: dict[str, Any] = {}
    hook = None
    if extract_latent:
        def capture_encoder_context(_module: Any, _inputs: Any, output: Any) -> None:
            if not hasattr(output, "shape") or output.ndim != 3:
                raise RuntimeError(f"unexpected ACT encoder output: {type(output)}")
            captured["z"] = output[0].detach()

        hook = policy.model.encoder.register_forward_hook(capture_encoder_context)
    try:
        with torch.inference_mode():
            normalized_chunk = policy.predict_action_chunk(batch)
            action_chunk = policy_postprocessor(normalized_chunk)
            action_chunk = env_postprocessor({ACTION: action_chunk})[ACTION]
    finally:
        if hook is not None:
            hook.remove()
    actions = action_chunk.detach().cpu().numpy().astype(np.float32, copy=False)
    if actions.shape != (len(states), 100, 7) or not np.isfinite(actions).all():
        raise RuntimeError(f"unexpected frozen ACT action output: {actions.shape}")
    if not extract_latent:
        return actions, None
    if "z" not in captured:
        raise RuntimeError("ACT encoder hook did not capture a representation")
    z = captured["z"].detach().cpu().numpy().astype(np.float32, copy=False)
    if z.shape != (len(states), 512) or not np.isfinite(z).all():
        raise RuntimeError(f"unexpected ACT encoder context shape: {z.shape}")
    return actions, z


def source_records(
    episode_ids: np.ndarray, source_steps: np.ndarray
) -> dict[int, list[tuple[int, int]]]:
    by_episode: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row_index, (episode, step) in enumerate(zip(episode_ids, source_steps, strict=True)):
        by_episode[int(episode)].append((int(step), row_index))
    return {episode: sorted(rows) for episode, rows in by_episode.items()}


def extract_features(
    bundle: dict[str, np.ndarray],
    episode_ids: np.ndarray,
    source_steps: np.ndarray,
    states: np.ndarray,
    *,
    batch_size: int,
    invariance_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    policy, policy_pre, policy_post, env_pre, env_post = load_act_runtime()
    records_by_episode = source_records(episode_ids, source_steps)
    latent = np.full((len(states), 512), np.nan, dtype=np.float32)
    extracted_chunks = np.full_like(bundle["source_chunk_actions"], np.nan)

    def make_records(rows: list[tuple[int, int]]) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[int]]:
        selected_states = np.asarray([states[row_index] for _step, row_index in rows], dtype=np.float32)
        episode = int(episode_ids[rows[0][1]])
        video_root = DATASET_ROOT / "videos" / "chunk-000"
        agent_path = video_root / "observation.images.image" / f"episode_{episode:06d}.mp4"
        wrist_path = video_root / "observation.images.wrist_image" / f"episode_{episode:06d}.mp4"
        frame_indices = [step for step, _row_index in rows]
        agent = decode_selected_frames(agent_path, frame_indices)
        wrist = decode_selected_frames(wrist_path, frame_indices)
        return selected_states, [agent[step] for step in frame_indices], [wrist[step] for step in frame_indices], [row for _step, row in rows]

    first_rows: list[tuple[int, int]] = []
    for rows in records_by_episode.values():
        first_rows.extend(rows)
        if len(first_rows) >= invariance_count:
            first_rows = first_rows[:invariance_count]
            break
    baseline_states, baseline_agent, baseline_wrist, _ = make_records(first_rows)
    baseline_actions, _ = infer_batch(
        baseline_states, baseline_agent, baseline_wrist, policy, policy_pre, policy_post, env_pre, env_post, extract_latent=False
    )
    invariant_actions, invariant_z = infer_batch(
        baseline_states, baseline_agent, baseline_wrist, policy, policy_pre, policy_post, env_pre, env_post, extract_latent=True
    )
    if invariant_z is None:
        raise RuntimeError("invariance extraction did not return z")
    invariance_delta = np.abs(baseline_actions.astype(np.float64) - invariant_actions.astype(np.float64))
    invariance = {
        "subset_rows": len(first_rows),
        "max_abs_action_delta": float(invariance_delta.max(initial=0.0)),
        "mean_abs_action_delta": float(invariance_delta.mean()),
        "allclose_atol_1e-6_rtol_1e-6": bool(np.allclose(baseline_actions, invariant_actions, atol=1e-6, rtol=1e-6)),
    }
    if not invariance["allclose_atol_1e-6_rtol_1e-6"]:
        raise RuntimeError(f"latent extraction changed ACT output: {invariance}")

    pending_states: list[np.ndarray] = []
    pending_agent: list[np.ndarray] = []
    pending_wrist: list[np.ndarray] = []
    pending_rows: list[int] = []

    def flush() -> None:
        if not pending_rows:
            return
        batch_states = np.stack(pending_states, axis=0).astype(np.float32, copy=False)
        actions, z = infer_batch(
            batch_states, pending_agent, pending_wrist, policy, policy_pre, policy_post, env_pre, env_post, extract_latent=True
        )
        if z is None:
            raise RuntimeError("latent extraction returned no z")
        extracted_chunks[pending_rows] = actions
        latent[pending_rows] = z
        pending_states.clear()
        pending_agent.clear()
        pending_wrist.clear()
        pending_rows.clear()

    for episode, rows in records_by_episode.items():
        selected_states, agent, wrist, row_indices = make_records(rows)
        for state, image, wrist_image, row_index in zip(selected_states, agent, wrist, row_indices, strict=True):
            pending_states.append(state)
            pending_agent.append(image)
            pending_wrist.append(wrist_image)
            pending_rows.append(row_index)
            if len(pending_rows) >= batch_size:
                flush()
        print(f"materialized episode {episode}; source rows {len(rows)}", flush=True)
    flush()

    if not np.isfinite(latent).all() or not np.isfinite(extracted_chunks).all():
        raise RuntimeError("feature extraction left missing or non-finite rows")
    chunk_delta = np.abs(extracted_chunks.astype(np.float64) - bundle["source_chunk_actions"].astype(np.float64))
    parity = {
        "all_rows": int(len(states)),
        "max_abs_source_chunk_delta": float(chunk_delta.max(initial=0.0)),
        "mean_abs_source_chunk_delta": float(chunk_delta.mean()),
        "allclose_atol_1e-5_rtol_1e-5": bool(np.allclose(extracted_chunks, bundle["source_chunk_actions"], atol=1e-5, rtol=1e-5)),
        "within_replay_atol_1e-2": bool(np.allclose(extracted_chunks, bundle["source_chunk_actions"], atol=1e-2, rtol=1e-5)),
        "interpretation": "The canonical source chunk remains the bundle array. The re-query uses the same checkpoint/runtime path; this replay check records small GPU numerical drift rather than replacing the locked chunk.",
    }
    if not parity["within_replay_atol_1e-2"]:
        raise RuntimeError(f"recomputed frozen ACT chunks do not match the bundle: {parity}")
    return latent, {
        "latent": {
            "definition": "z_t = policy.model.encoder output token 0 after the final ACT source-conditioned encoder layer; inference latent input is the model's all-zero latent token",
            "module": "lerobot.policies.act.modeling_act.ACT.model.encoder",
            "hook": "forward hook captures encoder_out[0, batch, :] before ACTDecoder",
            "shape": list(latent.shape),
            "per_source_shape": [512],
            "dtype": str(latent.dtype),
            "includes": ["current normalized observation.state", "current agent-view image", "current wrist image", "frozen ACT encoder parameters"],
            "exists_before_action_decoding": True,
            "demonstration_action_input": False,
            "future_input": False,
            "extraction_overhead": "one forward hook, one 512-value detach/host copy per source; no additional network forward and no changed decoder path",
        },
        "invariance": invariance,
        "source_chunk_parity": parity,
        "runtime": {
            "python": platform.python_version(),
            "torch_device": str(next(policy.parameters()).device),
            "torch": __import__("torch").__version__,
            "lerobot": __import__("lerobot").__version__,
        },
    }


def write_artifacts(
    bundle: dict[str, np.ndarray],
    episode_ids: np.ndarray,
    source_steps: np.ndarray,
    states: np.ndarray,
    latent: np.ndarray,
    key_audit: dict[str, Any],
    state_audit: dict[str, Any],
    extraction_audit: dict[str, Any],
) -> None:
    if FEATURE_PATH.exists():
        raise RuntimeError(f"refusing to overwrite immutable feature artifact: {FEATURE_PATH}")
    arrays = {
        "source_chunk_actions": bundle["source_chunk_actions"].astype(np.float32, copy=False),
        "source_state": states.astype(np.float32, copy=False),
        "source_policy_latent": latent.astype(np.float32, copy=False),
        "episode_id": episode_ids.astype(np.int32, copy=False),
        "source_step": source_steps.astype(np.int32, copy=False),
        "group_ids": bundle["group_ids"].astype(np.int8, copy=False),
        "offsets": bundle["offsets"].astype(np.int16, copy=False),
        "y_refresh": bundle["y_refresh"].astype(bool, copy=False),
        "label_observed": bundle["label_observed"].astype(bool, copy=False),
        "split_membership": bundle["split_membership"].astype(np.int8, copy=False),
    }
    np.savez_compressed(FEATURE_PATH, **arrays)
    manifest = {
        "schema_version": 1,
        "status": "materialized_and_verified",
        "feature_artifact": {
            "path": str(FEATURE_PATH.resolve()),
            "sha256": sha256(FEATURE_PATH),
            "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()},
            "estimator_source_inputs": ["source_chunk_actions", "source_state", "source_policy_latent"],
            "source_key_arrays_not_model_features": ["episode_id", "source_step"],
            "group_context_array": "group_ids",
            "label_and_split_references": ["y_refresh", "label_observed", "split_membership", "offsets"],
        },
        "portable_bundle": {
            "path": str(BUNDLE_PATH.resolve()),
            "sha256": sha256(BUNDLE_PATH),
            "expected_sha256": EXPECTED_BUNDLE_SHA256,
            "not_regenerated": True,
            "source_windows": 3740,
        },
        "source_key_audit": key_audit,
        "state_audit": state_audit,
        "act_checkpoint": {
            "path": str(CHECKPOINT_ROOT.resolve()),
            "config_sha256": sha256(CHECKPOINT_ROOT / "config.json"),
            "model_sha256": sha256(CHECKPOINT_ROOT / "model.safetensors"),
            "architecture": "ACT, chunk_size=100, n_action_steps=100, temporal_ensemble_coeff=null",
        },
        "dataset": {
            "path": str(DATASET_ROOT.resolve()),
            "revision": "cbf7122bbdbaa0c50517a6a4b2ae663d0e96e51a",
            "info_sha256": sha256(DATASET_ROOT / "meta/info.json"),
            "state_semantics": state_audit["semantics"],
        },
        "extraction": extraction_audit,
        "provenance": {
            "git_head_at_materialization": git_head(),
            "script": str(Path(__file__).resolve()),
            "future_observations_read": False,
            "future_actions_read": False,
            "episode_length_as_feature": False,
            "progress_as_feature": False,
            "phase_as_feature": False,
            "success_as_feature": False,
            "rollout": False,
        },
    }
    (OUTPUT_DIR / "feature_manifest.json").write_text(json.dumps(jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance = f"""# Source-context feature provenance

Status: materialized and verified from the exact 3,740-row portable cohort.

## Cohort lock

- Portable bundle SHA256: `{EXPECTED_BUNDLE_SHA256}`.
- Source key: `(episode_id, source_step) = (episode_index, frame_index)`.
- Duplicate source keys: `{key_audit['duplicate_source_keys']}`.
- Ambiguous state matches: `{state_audit['ambiguous_matches']}`.
- Missing state matches: `{state_audit['missing_matches']}`.
- Train/validation/test episode leakage: `{key_audit['train_validation_test_episode_intersections']}`.
- `Y_refresh` was read from the existing bundle and was not regenerated. The historical target-side cache is absent in this checkout and that absence is recorded explicitly in `feature_manifest.json`.

## Exact causal state

`source_state` is the original LIBERO `observation.state` row at source time
`t`, stored as float32 shape `(8,)`: EEF Cartesian position (indices `0:3`),
EEF axis-angle orientation (indices `3:6`), and the two-finger gripper qpos
(indices `6:8`). No future row, episode-length value, normalized progress,
phase, success, or action is included in the feature vector.

## ACT representation candidates considered before fitting

1. Per-camera ResNet-18 `layer4` feature map: causal and reusable, but camera-local and not the fused policy context.
2. Final fused ACT encoder output: causal, shared with action prediction, and available before decoding. **Selected.**
3. Final ACT decoder token sequence: causal and action-proximal, but decoder-output features are more tightly coupled to the action head and carry a larger sequence.

The primary representation is therefore:

`z_t = policy.model.encoder(batch)[0, :, :]`, where the first token is the
source-conditioned ACT latent/context token after the final encoder layer.
For each source it is shape `(512,)`, float32. The encoder input contains the
current normalized `observation.state`, current agent-view and wrist images,
and ACT's all-zero inference latent token. It is extracted by a forward hook
on `policy.model.encoder` immediately before `policy.model.decoder`; the hook
does not alter the returned action chunk. The training-time VAE encoder was
not selected because its input includes demonstration actions.

## Verification

- Frozen ACT action chunks recomputed from the exact source state/images match
  the bundle within the recorded tolerance in `feature_manifest.json`.
- On a deterministic subset, enabling the latent hook changes the postprocessed
  frozen ACT chunk by the recorded maximum absolute delta and passes the
  `1e-6` allclose check.
- Feature array SHA256, shapes, dtypes, checkpoint checksums, and dataset
  provenance are recorded in `feature_manifest.json`.

The feature artifact is immutable; the materializer refuses to overwrite it.
"""
    (OUTPUT_DIR / "feature_provenance.md").write_text(provenance, encoding="utf-8")


def run(batch_size: int, invariance_count: int) -> None:
    if not DATASET_ROOT.is_dir() or not CHECKPOINT_ROOT.is_dir():
        raise FileNotFoundError("exact external dataset/checkpoint provenance is unavailable")
    bundle = load_bundle()
    metadata = load_metadata()
    episode_ids, source_steps, key_audit = validate_keys_and_split(bundle, metadata)
    states, state_audit = recover_source_states(episode_ids, source_steps)
    latent, extraction_audit = extract_features(
        bundle, episode_ids, source_steps, states, batch_size=batch_size, invariance_count=invariance_count
    )
    write_artifacts(bundle, episode_ids, source_steps, states, latent, key_audit, state_audit, extraction_audit)
    print(json.dumps({"feature_bundle": str(FEATURE_PATH), "sha256": sha256(FEATURE_PATH), "source_windows": len(states)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--invariance-count", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1 or args.invariance_count < 1:
        raise ValueError("batch sizes must be positive")
    run(args.batch_size, args.invariance_count)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
