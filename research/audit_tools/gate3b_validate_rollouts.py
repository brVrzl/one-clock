#!/usr/bin/env python3
"""Post-result integrity checks for Gate-3B composition rollouts."""

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate3b_composition import METHODS, SOURCE_AGE_TICKS, compose_action  # noqa: E402


MANIFEST = ROOT / "research/audit_outputs/gate3b_rollout_manifest.json"
SCHEDULE = ROOT / "research/audit_outputs/gate3b_run_schedule.json"
OUTPUT = ROOT / "research/audit_outputs/gate3b_rollout_validation.json"
ROLLOUT_ROOT = ROOT / "experiments/gate3b_cross_generation_composition"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_sources(method: str, step: int) -> tuple[int, int]:
    if step < SOURCE_AGE_TICKS or method == "FF":
        return step, step
    old_source = step - SOURCE_AGE_TICKS
    if method == "OO":
        return old_source, old_source
    if method == "FO":
        return step, old_source
    if method == "OF":
        return old_source, step
    raise ValueError(method)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--schedule", type=Path, default=SCHEDULE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
    if not manifest["complete"] or manifest["completed_episodes"] != 400:
        raise RuntimeError("rollout manifest is incomplete")
    provenance = manifest["provenance"]
    if provenance["policy_temporal_ensemble_active"]:
        raise RuntimeError("policy temporal ensemble was active")
    if provenance["composition_temporal_ensemble_active"]:
        raise RuntimeError("composition temporal ensemble was active")
    if provenance["action_smoothing_active"]:
        raise RuntimeError("action smoothing was active")

    expected_runs = {
        (run["task_id"], run["state_id"], run["method"]): run for run in schedule["runs"]
    }
    first_twenty: dict[tuple[int, int], list[np.ndarray]] = {}
    per_file_hash_lines: list[str] = []
    max_formula_error = 0.0
    max_first_twenty_difference = 0.0
    total_step_records = 0
    total_log_bytes = 0
    active_steps = 0

    for episode in manifest["episodes"]:
        key = (episode["task_id"], episode["state_id"], episode["method"])
        if key not in expected_runs:
            raise RuntimeError(f"unexpected rollout cell: {key}")
        expected = expected_runs[key]
        for field in ("run_index", "episode_seed", "within_block_order"):
            if episode[field] != expected[field]:
                raise RuntimeError(f"schedule mismatch for {key}: {field}")
        path = Path(episode["log_path"])
        observed_hash = sha256(path)
        if observed_hash != episode["log_sha256"]:
            raise RuntimeError(f"log hash mismatch: {path}")
        relative = path.relative_to(ROLLOUT_ROOT)
        per_file_hash_lines.append(f"{observed_hash}  {relative.as_posix()}\n")
        total_log_bytes += path.stat().st_size
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload["steps"]
        if len(records) != episode["steps"] or episode["steps"] != episode["policy_queries"]:
            raise RuntimeError(f"step/query mismatch: {path}")
        if len(records) < SOURCE_AGE_TICKS:
            raise RuntimeError(f"episode ended before the common 20-step prefix: {path}")
        total_step_records += len(records)
        prefix = [np.asarray(record["action"], dtype=np.float64) for record in records[:20]]
        block = (int(episode["task_id"]), int(episode["state_id"]))
        if block in first_twenty:
            for reference, action in zip(first_twenty[block], prefix, strict=True):
                max_first_twenty_difference = max(
                    max_first_twenty_difference, float(np.max(np.abs(reference - action)))
                )
        else:
            first_twenty[block] = prefix

        for step, record in enumerate(records):
            if int(record["step"]) != step:
                raise RuntimeError(f"nonsequential step at {path}:{step}")
            action = np.asarray(record["action"], dtype=np.float64)
            fresh = np.asarray(record["fresh_action"], dtype=np.float64)
            if action.shape != (7,) or fresh.shape != (7,) or not np.isfinite(action).all():
                raise RuntimeError(f"invalid action at {path}:{step}")
            if int(record["fresh_source_step"]) != step or int(record["fresh_chunk_offset"]) != 0:
                raise RuntimeError(f"fresh-source contract mismatch at {path}:{step}")
            arm_source, gripper_source = expected_sources(str(episode["method"]), step)
            if int(record["arm_source_step"]) != arm_source:
                raise RuntimeError(f"arm-source mismatch at {path}:{step}")
            if int(record["gripper_source_step"]) != gripper_source:
                raise RuntimeError(f"gripper-source mismatch at {path}:{step}")
            if step < SOURCE_AGE_TICKS:
                if record["old_action"] is not None or record["old_source_step"] is not None:
                    raise RuntimeError(f"old source appeared before t=20 at {path}:{step}")
                expected_action = fresh
            else:
                active_steps += 1
                if int(record["old_source_step"]) != step - SOURCE_AGE_TICKS:
                    raise RuntimeError(f"old-source identity mismatch at {path}:{step}")
                if int(record["old_chunk_offset"]) != SOURCE_AGE_TICKS:
                    raise RuntimeError(f"old-source offset mismatch at {path}:{step}")
                old = np.asarray(record["old_action"], dtype=np.float64)
                if old.shape != (7,) or not np.isfinite(old).all():
                    raise RuntimeError(f"invalid old action at {path}:{step}")
                expected_action = compose_action(str(episode["method"]), fresh, old)
            max_formula_error = max(
                max_formula_error, float(np.max(np.abs(action - expected_action)))
            )

    observed_cells = {
        (episode["task_id"], episode["state_id"], episode["method"])
        for episode in manifest["episodes"]
    }
    if set(expected_runs) != observed_cells:
        raise RuntimeError("rollout coverage differs from the frozen schedule")
    if max_formula_error != 0.0:
        raise RuntimeError(f"executed action formula error: {max_formula_error}")
    if max_first_twenty_difference != 0.0:
        raise RuntimeError(f"first-20 common-prefix mismatch: {max_first_twenty_difference}")
    content_tree = hashlib.sha256("".join(sorted(per_file_hash_lines)).encode()).hexdigest()
    method_position_counts: dict[str, dict[str, int]] = {}
    for method in METHODS:
        counts = Counter(
            int(run["within_block_order"]) for run in schedule["runs"] if run["method"] == method
        )
        method_position_counts[method] = {str(position): counts[position] for position in range(4)}

    output: dict[str, Any] = {
        "scope": "Post-result technical integrity checks; gate statistics remain those in the frozen analyzer.",
        "manifest_sha256": sha256(args.manifest),
        "schedule_sha256": sha256(args.schedule),
        "complete_unique_task_state_method_cells": 400,
        "task_state_blocks": len(first_twenty),
        "total_step_records": total_step_records,
        "total_policy_queries": manifest["valid_policy_queries"],
        "all_queries_equal_environment_steps": total_step_records
        == manifest["valid_policy_queries"]
        == manifest["valid_environment_steps"],
        "policy_temporal_ensemble_active": False,
        "composition_temporal_ensemble_active": False,
        "action_smoothing_active": False,
        "source_age_ticks": SOURCE_AGE_TICKS,
        "source_age_seconds": 1.0,
        "all_old_sources_equal_t_minus_20": True,
        "all_old_chunk_offsets_equal_20": True,
        "all_executed_actions_match_registered_formulas": True,
        "maximum_executed_formula_error": max_formula_error,
        "all_actions_shape_7_and_finite": True,
        "first_20_actions_identical_across_methods_in_all_blocks": True,
        "maximum_first_20_action_difference_within_block": max_first_twenty_difference,
        "intervention_step_records": active_steps,
        "method_position_counts": method_position_counts,
        "local_rollout_root": str(ROLLOUT_ROOT),
        "compressed_episode_files": 400,
        "compressed_episode_file_bytes": total_log_bytes,
        "episode_file_content_tree_sha256": content_tree,
        "content_tree_definition": "SHA256 of concatenated sorted '<file_sha256>  <relative_path>\\n' lines.",
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
