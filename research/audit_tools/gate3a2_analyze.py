#!/usr/bin/env python3
"""Analyze the frozen Gate-3A2 paired closed-loop outcomes."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "research/audit_outputs/gate3a2_rollout_manifest.json"
DEFAULT_SUMMARY = ROOT / "research/audit_outputs/gate3a2_success_summary.json"
DEFAULT_PER_TASK = ROOT / "research/audit_outputs/gate3a2_per_task.csv"
DEFAULT_PAIRWISE = ROOT / "research/audit_outputs/gate3a2_pairwise_comparisons.csv"
BOOTSTRAP_DRAWS = 20_000
COMPARISONS = (
    ("D_minus_A", "newest", "newest_age_exp_b003"),
    ("D_minus_B", "exact_act_m001", "newest_age_exp_b003"),
    ("D_minus_C", "cogact_a03", "newest_age_exp_b003"),
    ("C_minus_B", "exact_act_m001", "cogact_a03"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--per-task", type=Path, default=DEFAULT_PER_TASK)
    parser.add_argument("--pairwise", type=Path, default=DEFAULT_PAIRWISE)
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
        raise RuntimeError("Gate-3A2 analysis requires all 400 valid episodes")
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
        raise RuntimeError("Gate-3A2 manifest does not contain 400 unique cells")
    return episodes


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def comparison_row(
    label: str,
    left_method: str,
    right_method: str,
    outcomes: dict[tuple[int, int, str], int],
    states: list[int],
    comparison_index: int,
) -> dict[str, Any]:
    keys = [(task, state) for task in range(10) for state in states]
    left = np.asarray([outcomes[(task, state, left_method)] for task, state in keys], dtype=np.int8)
    right = np.asarray([outcomes[(task, state, right_method)] for task, state in keys], dtype=np.int8)
    delta = right.astype(np.float64) - left.astype(np.float64)

    paired_rng = np.random.default_rng(20260826 + comparison_index)
    paired_indices = paired_rng.integers(0, len(delta), size=(BOOTSTRAP_DRAWS, len(delta)))
    paired_ci = percentile_ci(delta[paired_indices].mean(axis=1))

    task_delta = np.asarray([delta[np.asarray([task == key_task for key_task, _ in keys])].mean() for task in range(10)])
    task_rng = np.random.default_rng(20261826 + comparison_index)
    sampled_tasks = task_rng.integers(0, 10, size=(BOOTSTRAP_DRAWS, 10))
    task_ci = percentile_ci(task_delta[sampled_tasks].mean(axis=1))
    leave_one_task_out = np.asarray([np.delete(task_delta, task).mean() for task in range(10)])

    right_only = int(np.count_nonzero((right == 1) & (left == 0)))
    left_only = int(np.count_nonzero((right == 0) & (left == 1)))
    discordant = right_only + left_only
    mcnemar_p = float(binomtest(min(right_only, left_only), discordant, 0.5).pvalue) if discordant else 1.0
    point = float(delta.mean())
    stable_positive = bool(paired_ci[0] > 0 and task_ci[0] > 0 and leave_one_task_out.min() > 0)
    stable_negative = bool(paired_ci[1] < 0 and task_ci[1] < 0 and leave_one_task_out.max() < 0)
    return {
        "comparison": label,
        "left_method": left_method,
        "right_method": right_method,
        "right_minus_left_success_difference": point,
        "left_successes": int(left.sum()),
        "right_successes": int(right.sum()),
        "right_only_successes": right_only,
        "left_only_successes": left_only,
        "paired_state_bootstrap_ci_low": paired_ci[0],
        "paired_state_bootstrap_ci_high": paired_ci[1],
        "task_cluster_bootstrap_ci_low": task_ci[0],
        "task_cluster_bootstrap_ci_high": task_ci[1],
        "mcnemar_exact_two_sided_p": mcnemar_p,
        "positive_task_differences": int(np.count_nonzero(task_delta > 0)),
        "zero_task_differences": int(np.count_nonzero(task_delta == 0)),
        "negative_task_differences": int(np.count_nonzero(task_delta < 0)),
        "per_task_differences": json.dumps(task_delta.tolist(), separators=(",", ":")),
        "leave_one_task_out_min": float(leave_one_task_out.min()),
        "leave_one_task_out_max": float(leave_one_task_out.max()),
        "stable_positive": stable_positive,
        "stable_negative": stable_negative,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    episodes = validate_manifest(manifest)
    methods = sorted({str(episode["method"]) for episode in episodes})
    states = sorted({int(episode["state_id"]) for episode in episodes})
    if len(methods) != 4 or len(states) != 10:
        raise RuntimeError("manifest method/state coverage differs from preregistration")
    outcomes = {
        (int(episode["task_id"]), int(episode["state_id"]), str(episode["method"])): int(
            bool(episode["success"])
        )
        for episode in episodes
    }

    per_task_rows: list[dict[str, Any]] = []
    for task_id in range(10):
        for method in methods:
            subset = [
                episode
                for episode in episodes
                if int(episode["task_id"]) == task_id and episode["method"] == method
            ]
            successes = sum(int(bool(episode["success"])) for episode in subset)
            per_task_rows.append(
                {
                    "task_id": task_id,
                    "method": method,
                    "episodes": len(subset),
                    "successes": successes,
                    "success_rate": successes / len(subset),
                    "mean_steps": float(np.mean([episode["steps"] for episode in subset])),
                    "mean_policy_queries": float(np.mean([episode["policy_queries"] for episode in subset])),
                    "mean_effective_source_age_ticks": float(
                        np.mean([episode["mean_effective_source_age_ticks"] for episode in subset])
                    ),
                    "mean_effective_source_age_seconds": float(
                        np.mean([episode["mean_effective_source_age_seconds"] for episode in subset])
                    ),
                }
            )
    write_csv(args.per_task, per_task_rows)

    pairwise_rows = [
        comparison_row(label, left, right, outcomes, states, index)
        for index, (label, left, right) in enumerate(COMPARISONS)
    ]
    write_csv(args.pairwise, pairwise_rows)

    method_summary: dict[str, Any] = {}
    for method in methods:
        subset = [episode for episode in episodes if episode["method"] == method]
        successes = sum(int(bool(episode["success"])) for episode in subset)
        method_summary[method] = {
            "episodes": len(subset),
            "successes": successes,
            "success_rate": successes / len(subset),
            "environment_steps": int(sum(episode["steps"] for episode in subset)),
            "policy_queries": int(sum(episode["policy_queries"] for episode in subset)),
            "policy_queries_per_surviving_step": float(
                sum(episode["policy_queries"] for episode in subset) / sum(episode["steps"] for episode in subset)
            ),
            "mean_episode_steps": float(np.mean([episode["steps"] for episode in subset])),
            "mean_policy_query_seconds_per_episode": float(
                np.mean([episode["policy_query_seconds"] for episode in subset])
            ),
            "mean_effective_source_age_ticks": float(
                np.mean([episode["mean_effective_source_age_ticks"] for episode in subset])
            ),
            "mean_effective_source_age_seconds": float(
                np.mean([episode["mean_effective_source_age_seconds"] for episode in subset])
            ),
            "mean_translation_action_delta_l2": float(
                np.mean([episode["mean_translation_action_delta_l2"] for episode in subset])
            ),
            "mean_rotation_action_delta_radians": float(
                np.mean([episode["mean_rotation_action_delta_radians"] for episode in subset])
            ),
            "mean_gripper_transitions": float(np.mean([episode["gripper_transitions"] for episode in subset])),
            "mean_raw_action_acceleration_l2": float(
                np.mean([episode["mean_raw_action_acceleration_l2"] for episode in subset])
            ),
            "mean_raw_action_jerk_l2": float(
                np.mean([episode["mean_raw_action_jerk_l2"] for episode in subset])
            ),
        }

    comparisons_by_label = {row["comparison"]: row for row in pairwise_rows}
    d_minus_a = comparisons_by_label["D_minus_A"]
    d_minus_b = comparisons_by_label["D_minus_B"]
    d_minus_c = comparisons_by_label["D_minus_C"]
    query_cadence_valid = all(
        episode["policy_queries"] == episode["steps"] and episode["policy_queries_per_surviving_step"] == 1.0
        for episode in episodes
    )
    if d_minus_b["stable_negative"] or d_minus_a["stable_negative"]:
        decision = "CONTROL-LINK-NEGATIVE"
    elif (
        d_minus_b["stable_positive"]
        and d_minus_a["right_minus_left_success_difference"] > 0
        and query_cadence_valid
    ):
        decision = "STRONG-CONTROL-LINK" if d_minus_c["stable_positive"] else "CONTROL-LINK-POSITIVE"
    else:
        decision = "CONTROL-LINK-NULL"

    write_json(
        args.summary,
        {
            "schema_version": 1,
            "gate_decision": decision,
            "primary_outcome": "binary LIBERO task success",
            "inference_unit": "paired task-state block; task-cluster sensitivity",
            "planned_and_completed_episodes": 400,
            "task_state_blocks": 100,
            "query_cadence_valid": query_cadence_valid,
            "manifest_sha256": sha256(args.manifest),
            "manifest_provenance": manifest["provenance"],
            "method_summary": method_summary,
            "pairwise_comparisons": pairwise_rows,
            "decision_rule_source": "research/gate3a2_preregistered_protocol.md",
        },
    )
    print(f"Gate-3A2 decision: {decision}")


if __name__ == "__main__":
    main()
