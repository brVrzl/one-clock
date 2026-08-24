#!/usr/bin/env python3
"""Post-result integrity validation for Gate-4A2 Spatial rollouts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from gate3a2_temporal_aggregation import temporal_weights  # noqa: E402
from gate3c_temporal_reuse import METHODS, SOURCE_AGE_TICKS, compose_fixed_action  # noqa: E402


MANIFEST = ROOT / "research/audit_outputs/gate4a2_spatial_rollout_manifest.json"
SCHEDULE = ROOT / "research/audit_outputs/gate4a2_spatial_schedule.json"
OUTPUT = ROOT / "research/audit_outputs/gate4a2_spatial_rollout_validation.json"
ROLLOUT_ROOT = ROOT / "experiments/gate4a2_spatial_act_generalization"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--schedule", type=Path, default=SCHEDULE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
    if not manifest["complete"] or manifest["completed_episodes"] != 500:
        raise RuntimeError("rollout manifest is incomplete")
    provenance = manifest["provenance"]
    if provenance["policy_temporal_ensemble_active"] or provenance["action_smoothing_active"]:
        raise RuntimeError("forbidden policy ensemble or smoothing was active")
    if provenance["schedule_sha256"] != sha256(args.schedule):
        raise RuntimeError("manifest schedule identity differs from the frozen schedule")
    if provenance["temporal_executor_source_sha256"] != sha256(
        TOOLS / "gate3c_temporal_reuse.py"
    ):
        raise RuntimeError("Gate-3C temporal executor source identity changed")
    if provenance["scalar_weight_source_sha256"] != sha256(
        TOOLS / "gate3a2_temporal_aggregation.py"
    ):
        raise RuntimeError("Gate-3C scalar weighting source identity changed")
    expected_runs = {
        (run["task_id"], run["state_id"], run["method"]): run for run in schedule["runs"]
    }
    if len(expected_runs) != 500:
        raise RuntimeError("frozen schedule does not contain 500 unique cells")

    prefixes: dict[tuple[int, int], dict[str, list[np.ndarray]]] = {}
    hash_lines: list[str] = []
    total_steps = 0
    total_bytes = 0
    max_fixed_formula_error = 0.0
    max_fixed_prefix_difference = 0.0
    max_age_weight_error = 0.0
    scalar_weight_records = 0
    fixed_formula_records = 0
    observed_paths: set[Path] = set()
    for episode in manifest["episodes"]:
        key = (episode["task_id"], episode["state_id"], episode["method"])
        if key not in expected_runs:
            raise RuntimeError(f"unexpected rollout cell: {key}")
        expected = expected_runs[key]
        for field in ("run_index", "episode_seed", "within_block_order"):
            if episode[field] != expected[field]:
                raise RuntimeError(f"schedule mismatch for {key}: {field}")
        if int(episode["initial_state_id"]) != int(episode["state_id"]):
            raise RuntimeError(f"initial-state ID mismatch: {key}")
        expected_state_hash = provenance["official_initial_states"][
            "selected_state_vector_sha256"
        ][f"{episode['task_id']}:{episode['state_id']}"]
        if episode["initial_state_vector_sha256"] != expected_state_hash:
            raise RuntimeError(f"initial-state vector mismatch: {key}")
        path = Path(episode["log_path"])
        observed_paths.add(path.resolve())
        observed_hash = sha256(path)
        if observed_hash != episode["log_sha256"]:
            raise RuntimeError(f"log hash mismatch: {path}")
        relative = path.relative_to(ROLLOUT_ROOT)
        hash_lines.append(f"{observed_hash}  {relative.as_posix()}\n")
        total_bytes += path.stat().st_size
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload["run"] != expected:
            raise RuntimeError(f"payload schedule identity mismatch: {path}")
        records = payload["steps"]
        if len(records) != episode["steps"] or episode["steps"] != episode["policy_queries"]:
            raise RuntimeError(f"step/query mismatch: {path}")
        if len(records) < SOURCE_AGE_TICKS:
            raise RuntimeError(f"episode ended before t=20: {path}")
        total_steps += len(records)
        method = str(episode["method"])
        if method in METHODS[:3]:
            prefixes.setdefault((episode["task_id"], episode["state_id"]), {})[method] = [
                np.asarray(record["action"], dtype=np.float64) for record in records[:20]
            ]
        for step, record in enumerate(records):
            if int(record["step"]) != step or int(record["fresh_source_step"]) != step:
                raise RuntimeError(f"step/fresh identity mismatch: {path}:{step}")
            action = np.asarray(record["action"], dtype=np.float64)
            fresh = np.asarray(record["fresh_action"], dtype=np.float64)
            if (
                action.shape != (7,)
                or fresh.shape != (7,)
                or not np.isfinite(action).all()
                or not np.isfinite(fresh).all()
            ):
                raise RuntimeError(f"invalid action: {path}:{step}")
            expected_ages = np.arange(min(step, 99), -1, -1, dtype=np.int64)
            ages = np.asarray(record["candidate_ages"], dtype=np.int64)
            if not np.array_equal(ages, expected_ages):
                raise RuntimeError(f"temporal cache age order mismatch: {path}:{step}")
            if step < SOURCE_AGE_TICKS:
                if record["old_action"] is not None or record["old_source_step"] is not None:
                    raise RuntimeError(f"old source appeared before t=20: {path}:{step}")
            else:
                if int(record["old_source_step"]) != step - SOURCE_AGE_TICKS:
                    raise RuntimeError(f"old source is not q=t-20: {path}:{step}")
                if int(record["old_chunk_offset"]) != SOURCE_AGE_TICKS:
                    raise RuntimeError(f"old source offset is not 20: {path}:{step}")
            if method in METHODS[:3]:
                old = (
                    None
                    if record["old_action"] is None
                    else np.asarray(record["old_action"], dtype=np.float64)
                )
                expected_action = compose_fixed_action(method, fresh, old)
                max_fixed_formula_error = max(
                    max_fixed_formula_error, float(np.max(np.abs(action - expected_action)))
                )
                fixed_formula_records += 1
            else:
                weights = np.asarray(record["scalar_weights"], dtype=np.float64)
                if weights.shape != ages.shape or not np.isfinite(weights).all():
                    raise RuntimeError(f"invalid scalar weight vector: {path}:{step}")
                if not np.isclose(weights.sum(), 1.0) or np.any(weights < 0):
                    raise RuntimeError(f"unnormalized scalar weight vector: {path}:{step}")
                recorded_age = float(record["arm_effective_age_ticks"])
                if not np.isclose(recorded_age, float(weights @ ages)):
                    raise RuntimeError(f"effective age mismatch: {path}:{step}")
                if recorded_age != float(record["gripper_effective_age_ticks"]):
                    raise RuntimeError(f"scalar baseline group ages differ: {path}:{step}")
                if method == "D_AGE_EXP_B003":
                    dummy = np.zeros((len(ages), 7), dtype=np.float64)
                    expected_weights = temporal_weights(
                        "newest_age_exp_b003", dummy, ages.astype(np.float64)
                    )
                    max_age_weight_error = max(
                        max_age_weight_error, float(np.max(np.abs(weights - expected_weights)))
                    )
                scalar_weight_records += 1

    for block, method_prefixes in prefixes.items():
        if set(method_prefixes) != set(METHODS[:3]):
            raise RuntimeError(f"missing fixed-source prefix at block {block}")
        reference = method_prefixes["A_NEWEST"]
        for method in METHODS[1:3]:
            for left, right in zip(reference, method_prefixes[method], strict=True):
                max_fixed_prefix_difference = max(
                    max_fixed_prefix_difference, float(np.max(np.abs(left - right)))
                )
    if (
        max_fixed_formula_error != 0.0
        or max_fixed_prefix_difference != 0.0
        or max_age_weight_error > 1e-12
    ):
        raise RuntimeError("registered temporal formula or first-20 prefix validation failed")
    observed_cells = {
        (episode["task_id"], episode["state_id"], episode["method"])
        for episode in manifest["episodes"]
    }
    if observed_cells != set(expected_runs):
        raise RuntimeError("rollout coverage differs from frozen schedule")
    files_on_disk = {path.resolve() for path in ROLLOUT_ROOT.rglob("*.json.gz")}
    if files_on_disk != observed_paths:
        raise RuntimeError("extra or missing official episode files indicate retry/exclusion drift")
    position_counts: dict[str, dict[str, int]] = {}
    for method in METHODS:
        counts = Counter(
            int(run["within_block_order"]) for run in schedule["runs"] if run["method"] == method
        )
        position_counts[method] = {str(position): counts[position] for position in range(5)}
    content_tree = hashlib.sha256("".join(sorted(hash_lines)).encode()).hexdigest()
    output: dict[str, Any] = {
        "scope": "Post-result technical integrity; gate statistics come from the frozen analyzer.",
        "manifest_sha256": sha256(args.manifest),
        "schedule_sha256": sha256(args.schedule),
        "complete_unique_task_state_method_cells": 500,
        "task_state_blocks": len(prefixes),
        "total_step_records": total_steps,
        "total_policy_queries": manifest["valid_policy_queries"],
        "all_queries_equal_environment_steps": total_steps
        == manifest["valid_policy_queries"]
        == manifest["valid_environment_steps"],
        "all_initial_state_ids_and_vectors_match_frozen_schedule": True,
        "policy_temporal_ensemble_active": False,
        "action_smoothing_active": False,
        "source_age_ticks": SOURCE_AGE_TICKS,
        "all_old_sources_equal_t_minus_20": True,
        "all_old_chunk_offsets_equal_20": True,
        "all_fixed_source_actions_match_registered_formulas": True,
        "maximum_fixed_source_formula_error": max_fixed_formula_error,
        "all_age_exponential_weights_match_beta_003": True,
        "maximum_age_exponential_weight_error": max_age_weight_error,
        "cogact_uses_frozen_gate3c_source_identity": True,
        "temporal_executor_source_sha256": provenance["temporal_executor_source_sha256"],
        "scalar_weight_source_sha256": provenance["scalar_weight_source_sha256"],
        "all_actions_shape_7_and_finite": True,
        "first_20_actions_identical_across_A_B_C_in_all_blocks": True,
        "maximum_first_20_A_B_C_difference": max_fixed_prefix_difference,
        "scalar_baselines_use_normalized_shared_full_action_weights": True,
        "no_excluded_or_retried_official_episode_files": True,
        "fixed_formula_step_records": fixed_formula_records,
        "scalar_weight_step_records": scalar_weight_records,
        "method_position_counts": position_counts,
        "local_rollout_root": str(ROLLOUT_ROOT),
        "compressed_episode_files": 500,
        "compressed_episode_file_bytes": total_bytes,
        "episode_file_content_tree_sha256": content_tree,
        "content_tree_definition": "SHA256 of concatenated sorted '<file_sha256>  <relative_path>\\n' lines.",
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
