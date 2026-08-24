#!/usr/bin/env python3
"""Analyze the frozen Gate-3B paired 2x2 composition outcomes."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "research/audit_outputs/gate3b_rollout_manifest.json"
DEFAULT_SUMMARY = ROOT / "research/audit_outputs/gate3b_success_summary.json"
DEFAULT_PER_TASK = ROOT / "research/audit_outputs/gate3b_per_task.csv"
DEFAULT_CONTRAST = ROOT / "research/audit_outputs/gate3b_composition_contrast.csv"
BOOTSTRAP_DRAWS = 20_000
PAIRED_BOOTSTRAP_SEED = 20260829
TASK_BOOTSTRAP_SEED = 20261829
METHODS = ("FF", "OO", "FO", "OF")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--per-task", type=Path, default=DEFAULT_PER_TASK)
    parser.add_argument("--contrast", type=Path, default=DEFAULT_CONTRAST)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("complete") is not True or int(manifest.get("completed_episodes", -1)) != 400:
        raise RuntimeError("Gate-3B analysis requires all 400 valid episodes")
    episodes = manifest["episodes"]
    seen: set[tuple[int, int, str]] = set()
    for episode in episodes:
        key = (int(episode["task_id"]), int(episode["state_id"]), str(episode["method"]))
        if key in seen:
            raise RuntimeError(f"duplicate task-state-method episode: {key}")
        seen.add(key)
        if int(episode["steps"]) != int(episode["policy_queries"]):
            raise RuntimeError(f"query cadence mismatch: {key}")
        path = Path(episode["log_path"])
        if not path.is_file() or sha256(path) != episode["log_sha256"]:
            raise RuntimeError(f"missing or hash-invalid local rollout log: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if int(payload["summary"]["steps"]) != len(payload["steps"]):
            raise RuntimeError(f"local rollout log step mismatch: {path}")
    if len(seen) != 400:
        raise RuntimeError("Gate-3B manifest does not contain 400 unique cells")
    return episodes


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def descriptive_pairwise(
    first: str,
    second: str,
    outcomes: dict[tuple[int, int, str], int],
    keys: list[tuple[int, int]],
) -> dict[str, Any]:
    first_values = np.asarray([outcomes[(*key, first)] for key in keys], dtype=np.int8)
    second_values = np.asarray([outcomes[(*key, second)] for key in keys], dtype=np.int8)
    delta = first_values.astype(np.float64) - second_values.astype(np.float64)
    task_differences = [float(delta[[task == key_task for key_task, _ in keys]].mean()) for task in range(10)]
    return {
        "comparison": f"{first}_minus_{second}",
        "first_method": first,
        "second_method": second,
        "first_successes": int(first_values.sum()),
        "second_successes": int(second_values.sum()),
        "first_success_rate": float(first_values.mean()),
        "second_success_rate": float(second_values.mean()),
        "first_minus_second_success_difference": float(delta.mean()),
        "first_only_successes": int(np.count_nonzero((first_values == 1) & (second_values == 0))),
        "second_only_successes": int(np.count_nonzero((first_values == 0) & (second_values == 1))),
        "per_task_differences": task_differences,
        "status": "descriptive secondary comparison; not the gate estimand",
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    episodes = validate_manifest(manifest)
    methods = {str(episode["method"]) for episode in episodes}
    states = sorted({int(episode["state_id"]) for episode in episodes})
    if methods != set(METHODS) or len(states) != 10:
        raise RuntimeError("manifest method/state coverage differs from preregistration")
    outcomes = {
        (int(episode["task_id"]), int(episode["state_id"]), str(episode["method"])): int(
            bool(episode["success"])
        )
        for episode in episodes
    }
    keys = [(task, state) for task in range(10) for state in states]
    cell_values = {
        method: np.asarray([outcomes[(*key, method)] for key in keys], dtype=np.float64)
        for method in METHODS
    }
    block_contrast = 0.5 * (cell_values["FF"] + cell_values["OO"]) - 0.5 * (
        cell_values["FO"] + cell_values["OF"]
    )
    point = float(block_contrast.mean())
    interaction = 2.0 * point
    paired_rng = np.random.default_rng(PAIRED_BOOTSTRAP_SEED)
    paired_indices = paired_rng.integers(
        0, len(block_contrast), size=(BOOTSTRAP_DRAWS, len(block_contrast))
    )
    paired_ci = percentile_ci(block_contrast[paired_indices].mean(axis=1))
    task_contrasts = np.asarray(
        [block_contrast[[task == key_task for key_task, _ in keys]].mean() for task in range(10)]
    )
    task_rng = np.random.default_rng(TASK_BOOTSTRAP_SEED)
    sampled_tasks = task_rng.integers(0, 10, size=(BOOTSTRAP_DRAWS, 10))
    task_ci = percentile_ci(task_contrasts[sampled_tasks].mean(axis=1))
    leave_one_task_out = np.asarray(
        [np.delete(task_contrasts, task_id).mean() for task_id in range(10)]
    )

    confirmed = bool(
        point > 0
        and paired_ci[0] > 0
        and task_ci[0] > 0
        and leave_one_task_out.min() > 0
    )
    contradicted = bool(
        point < 0
        and paired_ci[1] < 0
        and task_ci[1] < 0
        and leave_one_task_out.max() < 0
    )
    suggestive = bool(
        point > 0
        and leave_one_task_out.min() > 0
        and not confirmed
    )
    if confirmed:
        decision = "COMPOSITION-HARM-CONFIRMED"
    elif contradicted:
        decision = "COMPOSITION-HARM-CONTRADICTED"
    elif suggestive:
        decision = "COMPOSITION-HARM-SUGGESTIVE"
    else:
        decision = "COMPOSITION-HARM-NULL"

    per_task_rows: list[dict[str, Any]] = []
    for task_id in range(10):
        task_episodes = [episode for episode in episodes if int(episode["task_id"]) == task_id]
        for method in METHODS:
            subset = [episode for episode in task_episodes if str(episode["method"]) == method]
            successes = sum(int(bool(episode["success"])) for episode in subset)
            per_task_rows.append(
                {
                    "task_id": task_id,
                    "method": method,
                    "episodes": len(subset),
                    "successes": successes,
                    "success_rate": successes / len(subset),
                    "mean_steps": float(np.mean([episode["steps"] for episode in subset])),
                    "mean_policy_queries": float(
                        np.mean([episode["policy_queries"] for episode in subset])
                    ),
                    "task_coherence_contrast": float(task_contrasts[task_id]),
                }
            )
    write_csv(args.per_task, per_task_rows)

    contrast_rows = [
        {
            "scope": "overall",
            "task_id": "",
            "blocks": 100,
            "coherence_contrast": point,
            "interaction": interaction,
            "paired_bootstrap_ci_low": paired_ci[0],
            "paired_bootstrap_ci_high": paired_ci[1],
            "task_cluster_bootstrap_ci_low": task_ci[0],
            "task_cluster_bootstrap_ci_high": task_ci[1],
        }
    ]
    for task_id, task_contrast in enumerate(task_contrasts):
        contrast_rows.append(
            {
                "scope": "task",
                "task_id": task_id,
                "blocks": 10,
                "coherence_contrast": float(task_contrast),
                "interaction": float(2.0 * task_contrast),
                "paired_bootstrap_ci_low": "",
                "paired_bootstrap_ci_high": "",
                "task_cluster_bootstrap_ci_low": "",
                "task_cluster_bootstrap_ci_high": "",
            }
        )
    for omitted_task, loto in enumerate(leave_one_task_out):
        contrast_rows.append(
            {
                "scope": "leave_one_task_out",
                "task_id": omitted_task,
                "blocks": 90,
                "coherence_contrast": float(loto),
                "interaction": float(2.0 * loto),
                "paired_bootstrap_ci_low": "",
                "paired_bootstrap_ci_high": "",
                "task_cluster_bootstrap_ci_low": "",
                "task_cluster_bootstrap_ci_high": "",
            }
        )
    write_csv(args.contrast, contrast_rows)

    method_summary: dict[str, Any] = {}
    diagnostic_fields = (
        "mean_episode_steps",
        "mean_arm_source_age_ticks",
        "mean_gripper_source_age_ticks",
        "mean_translation_action_delta_l2",
        "mean_rotation_action_delta_radians",
        "mean_raw_action_acceleration_l2",
        "mean_raw_action_jerk_l2",
        "mean_gripper_transitions",
        "mean_fresh_old_arm_l2",
        "mean_fresh_old_translation_l2",
        "mean_fresh_old_rotation_radians",
        "mean_fresh_old_gripper_sign_disagreement",
        "mean_distance_to_fresh_joint_source_action",
        "mean_distance_to_old_joint_source_action",
        "mean_distance_to_nearest_jointly_predicted_source_action",
    )
    for method in METHODS:
        subset = [episode for episode in episodes if str(episode["method"]) == method]
        steps = int(sum(int(episode["steps"]) for episode in subset))
        queries = int(sum(int(episode["policy_queries"]) for episode in subset))
        method_values = {
            "mean_episode_steps": float(np.mean([episode["steps"] for episode in subset])),
            "mean_arm_source_age_ticks": float(
                np.mean([episode["mean_arm_source_age_ticks"] for episode in subset])
            ),
            "mean_gripper_source_age_ticks": float(
                np.mean([episode["mean_gripper_source_age_ticks"] for episode in subset])
            ),
            "mean_gripper_transitions": float(
                np.mean([episode["gripper_transitions"] for episode in subset])
            ),
        }
        for field in diagnostic_fields:
            if field not in method_values:
                method_values[field] = float(np.mean([episode[field] for episode in subset]))
        successes = int(sum(int(bool(episode["success"])) for episode in subset))
        method_summary[method] = {
            "episodes": len(subset),
            "successes": successes,
            "success_rate": successes / len(subset),
            "environment_steps": steps,
            "policy_queries": queries,
            "policy_queries_per_surviving_step": queries / steps,
            **method_values,
        }

    pairwise = [
        descriptive_pairwise(first, second, outcomes, keys)
        for first, second in combinations(METHODS, 2)
    ]
    query_cadence_valid = all(
        int(episode["policy_queries"]) == int(episode["steps"])
        and float(episode["policy_queries_per_surviving_step"]) == 1.0
        for episode in episodes
    )
    write_json(
        args.summary,
        {
            "schema_version": 1,
            "gate_decision": decision,
            "primary_outcome": "binary LIBERO task success",
            "primary_estimand": "0.5*(success_FF + success_OO) - 0.5*(success_FO + success_OF)",
            "inference_unit": "paired task-state block; task-cluster sensitivity",
            "planned_and_completed_episodes": 400,
            "task_state_blocks": 100,
            "query_cadence_valid": query_cadence_valid,
            "manifest_sha256": sha256(args.manifest),
            "manifest_provenance": manifest["provenance"],
            "method_summary": method_summary,
            "coherence_contrast": {
                "overall_mean": point,
                "standard_2x2_interaction": interaction,
                "per_task": task_contrasts.tolist(),
                "paired_state_bootstrap_draws": BOOTSTRAP_DRAWS,
                "paired_state_bootstrap_seed": PAIRED_BOOTSTRAP_SEED,
                "paired_state_bootstrap_ci": list(paired_ci),
                "task_cluster_bootstrap_draws": BOOTSTRAP_DRAWS,
                "task_cluster_bootstrap_seed": TASK_BOOTSTRAP_SEED,
                "task_cluster_bootstrap_ci": list(task_ci),
                "leave_one_task_out": leave_one_task_out.tolist(),
            },
            "descriptive_pairwise_comparisons": pairwise,
            "decision_rule_source": "research/gate3b_cross_generation_preregistered_protocol.md",
        },
    )
    print(f"Gate-3B decision: {decision}")


if __name__ == "__main__":
    main()
