#!/usr/bin/env python3
"""Post-result integrity checks for Gate-3A2; does not recompute gate decisions."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "research/audit_outputs/gate3a2_rollout_manifest.json"
SCHEDULE = ROOT / "research/audit_outputs/gate3a2_run_schedule.json"
OUTPUT = ROOT / "research/audit_outputs/gate3a2_rollout_validation.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def registered_mean_age(method: str, candidate_count: int) -> float | None:
    ages = np.arange(candidate_count - 1, -1, -1, dtype=np.float64)
    if method == "newest":
        return 0.0
    if method == "exact_act_m001":
        weights = np.exp(-0.01 * np.arange(candidate_count, dtype=np.float64))
    elif method == "newest_age_exp_b003":
        weights = np.exp(-0.03 * ages)
    else:
        return None
    weights /= weights.sum()
    return float(weights @ ages)


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

    expected_runs = {
        (run["task_id"], run["state_id"], run["method"]): run for run in schedule["runs"]
    }
    first_actions: dict[tuple[int, int], np.ndarray] = {}
    per_file_hash_lines: list[str] = []
    max_first_action_difference = 0.0
    max_registered_age_error = 0.0
    total_step_records = 0
    total_log_bytes = 0

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
        relative = path.relative_to(Path("/home/thor/projects/one-clock/experiments/gate3a2_temporal_aggregation"))
        per_file_hash_lines.append(f"{observed_hash}  {relative.as_posix()}\n")
        total_log_bytes += path.stat().st_size
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload["steps"]
        if len(records) != episode["steps"] or episode["steps"] != episode["policy_queries"]:
            raise RuntimeError(f"step/query mismatch: {path}")
        total_step_records += len(records)
        for step, record in enumerate(records):
            expected_count = min(step + 1, 100)
            if record["step"] != step or record["candidate_count"] != expected_count:
                raise RuntimeError(f"candidate coverage mismatch at {path}:{step}")
            action = np.asarray(record["action"], dtype=np.float64)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise RuntimeError(f"invalid action at {path}:{step}")
            expected_age = registered_mean_age(episode["method"], expected_count)
            if expected_age is not None:
                max_registered_age_error = max(
                    max_registered_age_error,
                    abs(expected_age - float(record["mean_effective_age_ticks"])),
                )
        block = (episode["task_id"], episode["state_id"])
        first_action = np.asarray(records[0]["action"], dtype=np.float64)
        if block in first_actions:
            max_first_action_difference = max(
                max_first_action_difference, float(np.max(np.abs(first_actions[block] - first_action)))
            )
        else:
            first_actions[block] = first_action

    if set(expected_runs) != {
        (episode["task_id"], episode["state_id"], episode["method"])
        for episode in manifest["episodes"]
    }:
        raise RuntimeError("rollout coverage differs from the frozen schedule")
    content_tree = hashlib.sha256("".join(sorted(per_file_hash_lines)).encode()).hexdigest()
    method_position_counts: dict[str, dict[str, int]] = {}
    for method in schedule["methods"]:
        counts = Counter(
            int(run["within_block_order"]) for run in schedule["runs"] if run["method"] == method
        )
        method_position_counts[method] = {str(position): counts[position] for position in range(4)}

    output: dict[str, Any] = {
        "scope": "Post-result technical integrity checks; gate statistics remain those in the frozen analyzer.",
        "manifest_sha256": sha256(args.manifest),
        "schedule_sha256": sha256(args.schedule),
        "complete_unique_task_state_method_cells": 400,
        "task_state_blocks": len(first_actions),
        "total_step_records": total_step_records,
        "total_policy_queries": manifest["valid_policy_queries"],
        "all_queries_equal_environment_steps": total_step_records
        == manifest["valid_policy_queries"]
        == manifest["valid_environment_steps"],
        "all_candidate_counts_match_dense_window": True,
        "all_actions_shape_7_and_finite": True,
        "maximum_registered_fixed_weight_age_error_ticks": max_registered_age_error,
        "maximum_first_action_difference_within_block": max_first_action_difference,
        "first_actions_identical_across_methods_in_all_blocks": max_first_action_difference == 0.0,
        "method_position_counts": method_position_counts,
        "local_rollout_root": "/home/thor/projects/one-clock/experiments/gate3a2_temporal_aggregation",
        "compressed_episode_files": 400,
        "compressed_episode_file_bytes": total_log_bytes,
        "episode_file_content_tree_sha256": content_tree,
        "content_tree_definition": "SHA256 of concatenated sorted '<file_sha256>  <relative_path>\\n' lines.",
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
