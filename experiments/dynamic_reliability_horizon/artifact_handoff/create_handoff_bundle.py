#!/usr/bin/env python3
"""Build the immutable, estimator-safe Y_refresh handoff bundle.

The source target-comparison cache is intentionally not copied into the
portable bundle: it contains future actions and refreshed policy outputs that
are label/evaluation-side artifacts. Only source-time predicted chunks,
labels, masks, group IDs, episode IDs, and split membership are exported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HANDOFF = Path(__file__).resolve().parent
TARGET_CACHE = ROOT / "experiments/temporal_reliability_target_comparison/target_comparison.npz"
TARGET_SCRIPT = ROOT / "experiments/temporal_reliability_target_comparison/compare_targets.py"
TARGET_REPORT = ROOT / "experiments/temporal_reliability_target_comparison/target_comparison_report.md"
METADATA = ROOT / "experiments/temporal_reliability/metadata.jsonl"
DATASET_MANIFEST = ROOT / "experiments/temporal_reliability/dataset_manifest.json"
CONSTRUCT_SCRIPT = ROOT / "experiments/temporal_reliability/construct_dataset.py"
ORACLE_SCRIPT = ROOT / "experiments/dynamic_reliability_horizon/analyze_oracle_and_pace.py"
SPLIT_PATH = HANDOFF / "episode_split_manifest.json"
BUNDLE_PATH = HANDOFF / "minimal_y_refresh_training_bundle.npz"
GROUPS = ("arm", "gripper")
CHUNK_SIZE = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def git_commit_for_path(path: Path) -> str | None:
    relative = path.relative_to(ROOT).as_posix()
    try:
        values = git("log", "--all", "--format=%H", "--", relative).splitlines()
    except subprocess.CalledProcessError:
        return None
    return values[0] if values else None


def relative_or_absolute(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def file_record(
    path: Path,
    *,
    role: str,
    estimator_visible: bool,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": relative_or_absolute(path),
        "absolute_path_at_generation": str(path.resolve()),
        "role": role,
        "estimator_visible": estimator_visible,
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256(path) if path.exists() else None,
        "git_commit": git_commit_for_path(path) if path.is_relative_to(ROOT) and path.exists() else None,
    }
    if schema is not None:
        record["schema"] = schema
    return record


def load_metadata() -> list[dict[str, Any]]:
    return [json.loads(line) for line in METADATA.read_text().splitlines() if line.strip()]


def build_split(episodes: np.ndarray, tasks: np.ndarray) -> dict[str, Any]:
    episode_to_task: dict[int, int] = {}
    for episode, task in zip(episodes.tolist(), tasks.tolist(), strict=True):
        episode_to_task.setdefault(int(episode), int(task))
        if episode_to_task[int(episode)] != int(task):
            raise RuntimeError("episode appears with more than one task")
    by_split: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    # Stable, task-stratified, episode-level assignment: positions 0-7 train,
    # 8 validation, and 9 test within each task's sorted episode IDs.
    for task in sorted(set(episode_to_task.values())):
        task_episodes = sorted(ep for ep, value in episode_to_task.items() if value == task)
        for position, episode in enumerate(task_episodes):
            remainder = position % 10
            split = "train" if remainder < 8 else "validation" if remainder == 8 else "test"
            by_split[split].append(episode)
    for values in by_split.values():
        values.sort()
    return {
        "schema_version": 1,
        "created_by": "create_handoff_bundle.py",
        "source_target_cache": relative_or_absolute(TARGET_CACHE),
        "assignment": "within each task, sort unique episode IDs; position modulo 10: 0-7 train, 8 validation, 9 test",
        "episode_level": True,
        "frame_level_split": False,
        "episodes_total": int(len(episode_to_task)),
        "row_counts": {
            split: int(np.sum(np.isin(episodes, values)))
            for split, values in by_split.items()
        },
        "episodes_by_split": by_split,
        "task_episode_counts": {
            str(task): int(sum(value == task for value in episode_to_task.values()))
            for task in sorted(set(episode_to_task.values()))
        },
    }


def build_bundle() -> tuple[dict[str, Any], dict[str, Any]]:
    if not TARGET_CACHE.exists() or not METADATA.exists():
        raise FileNotFoundError("target cache and aligned metadata are required")
    data = np.load(TARGET_CACHE, allow_pickle=False)
    metadata = load_metadata()
    n = len(metadata)
    episodes = data["episode_index"].astype(np.int32)
    frames = data["frame_index"].astype(np.int32)
    if episodes.shape != (n,) or frames.shape != (n,):
        raise RuntimeError("metadata and target cache row counts differ")
    expected_episodes = np.asarray([int(row["episode_index"]) for row in metadata], dtype=np.int32)
    expected_frames = np.asarray([int(row["frame_index"]) for row in metadata], dtype=np.int32)
    if not np.array_equal(episodes, expected_episodes) or not np.array_equal(frames, expected_frames):
        raise RuntimeError("metadata is not row-aligned with target cache")
    tasks = np.asarray([int(row["task_index"]) for row in metadata], dtype=np.int32)
    observed = data["observed_offsets"].astype(bool)
    source_chunk = data["old_predicted_actions"].astype(np.float32)
    if observed.shape != (n, CHUNK_SIZE) or source_chunk.shape != (n, CHUNK_SIZE, 7):
        raise RuntimeError("unexpected observed mask or source chunk shape")
    y_refresh = np.stack(
        [data[f"{group}_refresh_survival"].astype(bool)[:, 1:] for group in GROUPS],
        axis=1,
    )
    label_observed = np.broadcast_to(observed[:, 1:][:, None, :], y_refresh.shape).copy()
    split = build_split(episodes, tasks)
    split_by_episode = {
        int(episode): split_name
        for split_name, values in split["episodes_by_split"].items()
        for episode in values
    }
    split_code = {"train": 0, "validation": 1, "test": 2}
    split_membership = np.asarray(
        [split_code[split_by_episode[int(episode)]] for episode in episodes], dtype=np.int8
    )
    np.savez_compressed(
        BUNDLE_PATH,
        source_chunk_actions=source_chunk,
        group_ids=np.asarray([0, 1], dtype=np.int8),
        offsets=np.arange(1, CHUNK_SIZE, dtype=np.int16),
        y_refresh=y_refresh,
        label_observed=label_observed,
        episode_index=episodes,
        split_membership=split_membership,
    )
    SPLIT_PATH.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
    schema = {
        "source_chunk_actions": {"shape": list(source_chunk.shape), "dtype": str(source_chunk.dtype), "meaning": "frozen ACT chunk predicted at source time; causal input candidate"},
        "group_ids": {"shape": [2], "dtype": "int8", "mapping": {"0": "arm", "1": "gripper"}},
        "offsets": {"shape": [CHUNK_SIZE - 1], "dtype": "int16", "values": "1..99; k=0 intentionally excluded"},
        "y_refresh": {"shape": list(y_refresh.shape), "dtype": "bool", "axes": ["row", "group_id", "offset index"], "meaning": "prefix survival Y_refresh_g(k), k=1..99"},
        "label_observed": {"shape": list(label_observed.shape), "dtype": "bool", "meaning": "true iff the demonstrated future frame at offset k exists; false means right-censored/unavailable"},
        "episode_index": {"shape": list(episodes.shape), "dtype": "int32", "meaning": "grouping/split key, not an estimator feature"},
        "split_membership": {"shape": list(split_membership.shape), "dtype": "int8", "mapping": {"0": "train", "1": "validation", "2": "test"}, "meaning": "episode-level membership, not an estimator feature"},
    }
    return schema, split


def dependency_record(path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    record = {
        "absolute_path": str(path.resolve()),
        "identity": identity,
        "exists": path.exists(),
    }
    if path.exists() and path.is_file():
        record.update({"size_bytes": path.stat().st_size, "sha256": sha256(path)})
    return record


def verification_command(bundle_sha: str, split_sha: str) -> str:
    return (
        'HANDOFF_DIR=/path/to/experiments/dynamic_reliability_horizon/artifact_handoff; '
        f'sha256sum "$HANDOFF_DIR/minimal_y_refresh_training_bundle.npz" "$HANDOFF_DIR/episode_split_manifest.json"; '
        f'python3 -c "import hashlib, pathlib, sys; expected={{\"minimal_y_refresh_training_bundle.npz\":\"{bundle_sha}\", \"episode_split_manifest.json\":\"{split_sha}\"}}; '
        '[(lambda p: (_ for _ in ()).throw(SystemExit(f\"checksum mismatch: {p}\")) if hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()!=v else None)(str(pathlib.Path(sys.argv[1])/p)) for p,v in expected.items()]" "$HANDOFF_DIR"'
    )


def build_manifest(schema: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
    data = np.load(TARGET_CACHE, allow_pickle=False)
    metadata = load_metadata()
    episodes = data["episode_index"].astype(np.int32)
    tasks = np.asarray([int(row["task_index"]) for row in metadata], dtype=np.int32)
    task_names = sorted({str(row["task_name"]) for row in metadata})
    records = [
        file_record(BUNDLE_PATH, role="portable estimator handoff bundle", estimator_visible=True, schema=schema),
        file_record(SPLIT_PATH, role="episode-level split manifest", estimator_visible=False, schema={"keys": list(split.keys())}),
        file_record(TARGET_CACHE, role="cached frozen-policy re-query labels and evaluation cache; label-side only", estimator_visible=False, schema={
            "arrays": {name: {"shape": list(data[name].shape), "dtype": str(data[name].dtype)} for name in data.files},
            "rows": int(len(episodes)),
            "episodes": sorted(set(int(value) for value in episodes.tolist())),
            "task_names": task_names,
        }),
        file_record(TARGET_SCRIPT, role="Y_refresh construction/re-query code; working-tree artifact if git_commit is null", estimator_visible=False),
        file_record(TARGET_REPORT, role="target comparison report", estimator_visible=False),
        file_record(METADATA, role="aligned source-window metadata; label/evaluation provenance only", estimator_visible=False, schema={"rows": len(metadata), "fields": sorted(metadata[0])}),
        file_record(CONSTRUCT_SCRIPT, role="original Y_demo validity construction and threshold definition", estimator_visible=False),
        file_record(DATASET_MANIFEST, role="frozen checkpoint/action normalization provenance", estimator_visible=False),
        file_record(ORACLE_SCRIPT, role="oracle horizon derivation and sensitivity analysis", estimator_visible=False),
    ]
    checkpoint_root = Path("/home/thor/projects/checkpoints/zeromidnight_act_libero_object")
    dataset_root = Path("/home/thor/datasets/libero_object_25_08_23_lerobotv2.1")
    external = {
        "dataset": dependency_record(dataset_root, {
            "hf_repo_id": "DorayakiLin/libero_object_25_08_23_lerobotv2.1",
            "local_hf_revision": "cbf7122bbdbaa0c50517a6a4b2ae663d0e96e51a",
            "episodes": 454, "frames": 66984, "tasks": 10, "fps": 10,
            "action_shape": [7], "required_for_bundle_training": False,
            "required_for_rebuilding_Y_refresh": True,
        }),
        "frozen_act_checkpoint": dependency_record(checkpoint_root, {
            "repo_id": "zeromidnight/act_libero_object", "checkpoint_type": "ACT",
            "chunk_size": 100, "n_action_steps": 100, "action_shape": [7],
            "config_sha256": sha256(checkpoint_root / "config.json"),
            "model_sha256": sha256(checkpoint_root / "model.safetensors"),
            "required_for_bundle_training": False,
            "required_for_rebuilding_Y_refresh": True,
        }),
    }
    bundle_sha = sha256(BUNDLE_PATH)
    split_sha = sha256(SPLIT_PATH)
    return {
        "schema_version": 1,
        "purpose": "portable offline Y_refresh reliability-estimator handoff; no training or rollout performed here",
        "source_repository_sha": git("rev-parse", "HEAD"),
        "source_repository_dirty_at_generation": bool(git("status", "--porcelain")),
        "missing_requested_contract_commit": {
            "sha": "928ffba", "available_in_this_clone": False,
            "verification": "git rev-list --all and unreachable-object inspection found no 928ffba object",
            "consequence": "bundle is the minimal known causal source-chunk contract, not certified equivalent to the unavailable estimator contract",
        },
        "required_code_commits": sorted({record["git_commit"] for record in records if record["git_commit"]}),
        "provenance_records": records,
        "episode_rows": {
            "rows": int(len(episodes)),
            "episodes": sorted(set(int(value) for value in episodes.tolist())),
            "task_indices": sorted(set(int(value) for value in tasks.tolist())),
            "episode_level_split_manifest": relative_or_absolute(SPLIT_PATH),
        },
        "groups": {
            "0": {"name": "arm", "action_dimensions": "0:6", "meaning": "relative Cartesian position plus axis-angle action channels"},
            "1": {"name": "gripper", "action_dimensions": "6:7", "meaning": "gripper action channel"},
        },
        "offset_and_horizon_convention": {
            "K": 100, "offsets_in_raw_cache": "k=0..99", "offsets_in_bundle": "k=1..99",
            "k0_is_evidence": False,
            "action_count_mapping": "h = k + 1; h*=max{h>=1: Y_refresh(h-1) remains true}",
            "max_action_count": 100,
            "censoring": "observed_offsets is a contiguous prefix; labels beyond the demonstrated episode suffix are unavailable/right-censored; label_observed is the k>=1 mask",
        },
        "y_refresh_definition": {
            "source": "frozen ACT chunk at source time t versus same frozen policy's first action at demonstrated future observation t+k",
            "prefix_survival": "logical AND of pointwise validity from offset 0 through k, evaluated only when observed",
            "arm": "normalized translation RMS <= 1.0 AND normalized rotation RMS <= 1.0",
            "gripper": "normalized absolute error <= 1.0 AND sign(old action) == sign(refreshed action)",
            "normalization": "checkpoint action standard deviations from dataset_manifest.json",
            "teacher_forced": True, "rollout_success_supervision": False,
        },
        "feature_contract_audit": {
            "estimator_visible_bundle_fields": ["source_chunk_actions", "group_ids", "offsets", "y_refresh", "label_observed", "episode_index", "split_membership"],
            "source_time_causal_feature_materialized": False,
            "future_observations_in_bundle": False, "future_demonstration_actions_in_bundle": False,
            "episode_length_in_bundle": False, "normalized_progress_in_bundle": False,
            "phase_code_in_bundle": False, "terminal_metadata_in_bundle": False,
            "contract_condition": "If the 5080 estimator requires current images/state or frozen-ACT latent features beyond source_chunk_actions, this bundle is insufficient and those exact causal inputs must be transferred/materialized separately; do not substitute future data or silently redesign the estimator.",
        },
        "external_dependencies": external,
        "verification": {
            "bundle_sha256": bundle_sha, "split_manifest_sha256": split_sha,
            "command": verification_command(bundle_sha, split_sha),
        },
    }


def build_readme(manifest: dict[str, Any]) -> str:
    bundle = next(item for item in manifest["provenance_records"] if item["path"].endswith("minimal_y_refresh_training_bundle.npz"))
    split = next(item for item in manifest["provenance_records"] if item["path"].endswith("episode_split_manifest.json"))
    return f"""# Portable Y_refresh handoff

Generated from repository commit `{manifest['source_repository_sha']}`. This is
an offline data handoff only. No reliability network was trained, no rollout
was run, and executor semantics and paper claims were not changed.

## What to transfer

Transfer these two files together:

* `minimal_y_refresh_training_bundle.npz` — SHA256 `{bundle['sha256']}` ({bundle['size_bytes']} bytes)
* `episode_split_manifest.json` — SHA256 `{split['sha256']}` ({split['size_bytes']} bytes)

The bundle contains only the known source-time predicted ACT chunk, group IDs,
candidate offsets `k=1..99`, `Y_refresh` prefix labels, label-observation masks,
episode IDs, and episode-level split membership. It contains no future
observations, future demonstration actions, episode length, progress, phase,
or terminal metadata. Do not pass `episode_index` or `split_membership` as model
features; they are grouping/split fields.

Array schema and all provenance checksums are in `handoff_manifest.json`.

## Contract limitation

The requested estimator contract commit `928ffba` is not present in this
clone, and no materialized source-time image/state/ACT-latent tensor exists in
the cached re-query artifact. Therefore this is the minimal known causal
source-chunk handoff, not a certification that the unavailable contract is
fully satisfied. Training can proceed from this bundle only if the 5080
implementation consumes the source chunk (plus group/offset context) as its
causal feature input. If it requires current observation images/state or a
frozen-ACT latent, transfer those exact causal inputs separately and record
their checksums; do not use future data or silently redesign the estimator.

## External provenance

The original cache was built from the local dataset
`/home/thor/datasets/libero_object_25_08_23_lerobotv2.1` (Hugging Face
`DorayakiLin/libero_object_25_08_23_lerobotv2.1`, local revision
`cbf7122bbdbaa0c50517a6a4b2ae663d0e96e51a`) and frozen ACT checkpoint
`/home/thor/projects/checkpoints/zeromidnight_act_libero_object`. These large
external files are not committed. They are required to reproduce the cached
Y_refresh labels, but not for bundle-only training when the contract condition
above holds. The manifest records checkpoint config/model checksums without
copying credentials.

## Verification on 5080

After transferring the directory, replace `/path/to` in the following exact
command with its destination root:

```bash
{manifest['verification']['command']}
```

The expected result is two matching SHA256 lines and no `checksum mismatch`
exception. Then load the NPZ with `allow_pickle=False` and assert every shape
and dtype in the manifest before any training job is started.

## Label semantics

`Y_refresh` is teacher-forced frozen-policy replanning consistency: a source
chunk is compared with the same policy queried at a demonstrated future
observation. It is not rollout success supervision. `k=0` is omitted from the
portable labels as a trivial identity check. Censored offsets are represented
by `label_observed=False` and must not be treated as negative labels.
"""


def main() -> None:
    args = parse_args()
    HANDOFF.mkdir(parents=True, exist_ok=True)
    schema, split = build_bundle()
    if args.bundle_only:
        print(json.dumps({"bundle": str(BUNDLE_PATH), "split": str(SPLIT_PATH)}, indent=2))
        return
    manifest = build_manifest(schema, split)
    (HANDOFF / "handoff_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (HANDOFF / "handoff_readme.md").write_text(build_readme(manifest))
    print(json.dumps({"manifest": str(HANDOFF / "handoff_manifest.json"), "bundle_sha256": manifest["verification"]["bundle_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
