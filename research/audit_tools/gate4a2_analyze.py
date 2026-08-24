#!/usr/bin/env python3
"""Analyze the frozen Gate-4A2 Spatial paired outcomes."""

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
MANIFEST = ROOT / "research/audit_outputs/gate4a2_spatial_rollout_manifest.json"
SUMMARY = ROOT / "research/audit_outputs/gate4a2_spatial_success_summary.json"
PAIRWISE = ROOT / "research/audit_outputs/gate4a2_spatial_pairwise.csv"
PER_TASK = ROOT / "research/audit_outputs/gate4a2_spatial_per_task.csv"
LOTO = ROOT / "research/audit_outputs/gate4a2_spatial_leave_one_task_out.csv"
METHODS = (
    "A_NEWEST",
    "B_FULL_OLD20",
    "C_ASYMMETRIC_FO20",
    "D_AGE_EXP_B003",
    "E_COGACT_A03",
)
COMPARISONS = (
    ("C_MINUS_A", "C_ASYMMETRIC_FO20", "A_NEWEST", 20260911, 20261911),
    ("C_MINUS_B", "C_ASYMMETRIC_FO20", "B_FULL_OLD20", 20260912, 20261912),
    ("C_MINUS_D", "C_ASYMMETRIC_FO20", "D_AGE_EXP_B003", 20260913, 20261913),
    ("C_MINUS_E", "C_ASYMMETRIC_FO20", "E_COGACT_A03", 20260914, 20261914),
)
BOOTSTRAP_DRAWS = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--pairwise", type=Path, default=PAIRWISE)
    parser.add_argument("--per-task", type=Path, default=PER_TASK)
    parser.add_argument("--loto", type=Path, default=LOTO)
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


def percentile_ci(draws: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("complete") is not True or int(manifest.get("completed_episodes", -1)) != 500:
        raise RuntimeError("Gate-4A2 analysis requires all 500 episodes")
    episodes = manifest["episodes"]
    seen: set[tuple[int, int, str]] = set()
    for episode in episodes:
        key = (int(episode["task_id"]), int(episode["state_id"]), str(episode["method"]))
        if key in seen:
            raise RuntimeError(f"duplicate cell: {key}")
        seen.add(key)
        if int(episode["steps"]) != int(episode["policy_queries"]):
            raise RuntimeError(f"query cadence mismatch: {key}")
        path = Path(episode["log_path"])
        if not path.is_file() or sha256(path) != episode["log_sha256"]:
            raise RuntimeError(f"missing or hash-invalid local log: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if int(payload["summary"]["steps"]) != len(payload["steps"]):
            raise RuntimeError(f"local log step mismatch: {path}")
    if len(seen) != 500:
        raise RuntimeError("Gate-4A2 manifest does not contain 500 unique cells")
    return episodes


def comparison_statistics(
    outcomes: dict[tuple[int, int, str], int],
    states: list[int],
    first: str,
    second: str,
    paired_seed: int,
    cluster_seed: int,
) -> dict[str, Any]:
    tasks = list(range(10))
    keys = [(task, state) for task in tasks for state in states]
    first_values = np.asarray([outcomes[(*key, first)] for key in keys], dtype=np.int8)
    second_values = np.asarray([outcomes[(*key, second)] for key in keys], dtype=np.int8)
    differences = first_values.astype(np.float64) - second_values.astype(np.float64)
    task_differences = np.asarray(
        [differences[[task == key_task for key_task, _ in keys]].mean() for task in tasks]
    )
    paired_rng = np.random.default_rng(paired_seed)
    paired_index = paired_rng.integers(0, len(keys), size=(BOOTSTRAP_DRAWS, len(keys)))
    paired_ci = percentile_ci(differences[paired_index].mean(axis=1))
    cluster_rng = np.random.default_rng(cluster_seed)
    cluster_index = cluster_rng.integers(0, len(tasks), size=(BOOTSTRAP_DRAWS, len(tasks)))
    cluster_ci = percentile_ci(task_differences[cluster_index].mean(axis=1))
    loto = np.asarray(
        [np.delete(task_differences, index).mean() for index in range(len(task_differences))]
    )
    first_only = int(np.count_nonzero((first_values == 1) & (second_values == 0)))
    second_only = int(np.count_nonzero((first_values == 0) & (second_values == 1)))
    discordant = first_only + second_only
    mcnemar_p = float(binomtest(first_only, discordant, 0.5).pvalue) if discordant else 1.0
    stable_positive = bool(paired_ci[0] > 0 and cluster_ci[0] > 0 and loto.min() > 0)
    stable_negative = bool(paired_ci[1] < 0 and cluster_ci[1] < 0 and loto.max() < 0)
    return {
        "tasks": tasks,
        "blocks": len(keys),
        "first_method": first,
        "second_method": second,
        "first_successes": int(first_values.sum()),
        "second_successes": int(second_values.sum()),
        "first_success_rate": float(first_values.mean()),
        "second_success_rate": float(second_values.mean()),
        "paired_block_difference": float(differences.mean()),
        "first_only_successes": first_only,
        "second_only_successes": second_only,
        "discordant_blocks": discordant,
        "exact_two_sided_mcnemar_binomial_p": mcnemar_p,
        "paired_bootstrap_draws": BOOTSTRAP_DRAWS,
        "paired_bootstrap_seed": paired_seed,
        "paired_bootstrap_ci": list(paired_ci),
        "task_cluster_bootstrap_draws": BOOTSTRAP_DRAWS,
        "task_cluster_bootstrap_seed": cluster_seed,
        "task_ids_for_task_differences": tasks,
        "task_differences": task_differences.tolist(),
        "leave_one_task_out": loto.tolist(),
        "stable_positive": stable_positive,
        "stable_negative": stable_negative,
    }


def frozen_decision(comparisons: dict[str, dict[str, Any]]) -> str:
    if comparisons["C_MINUS_A"]["stable_negative"] or comparisons["C_MINUS_B"][
        "stable_negative"
    ]:
        return "SPATIAL-GENERALIZATION-CONTRADICTED"
    if all(value["stable_positive"] for value in comparisons.values()):
        return "SPATIAL-GENERALIZATION-STRONG"
    if comparisons["C_MINUS_A"]["stable_positive"] and comparisons["C_MINUS_B"][
        "stable_positive"
    ]:
        return "SPATIAL-GENERALIZATION-DIRECTIONAL"
    if all(value["paired_block_difference"] > 0 for value in comparisons.values()):
        return "SPATIAL-GENERALIZATION-SUGGESTIVE"
    return "SPATIAL-GENERALIZATION-NULL"


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    episodes = validate_manifest(manifest)
    states = sorted({int(episode["state_id"]) for episode in episodes})
    methods = {str(episode["method"]) for episode in episodes}
    if len(states) != 10 or methods != set(METHODS):
        raise RuntimeError("method/state coverage differs from preregistration")
    outcomes = {
        (int(episode["task_id"]), int(episode["state_id"]), str(episode["method"])): int(
            bool(episode["success"])
        )
        for episode in episodes
    }

    comparisons: dict[str, Any] = {}
    pairwise_rows: list[dict[str, Any]] = []
    for label, first, second, paired_seed, cluster_seed in COMPARISONS:
        stats = comparison_statistics(
            outcomes, states, first, second, paired_seed, cluster_seed
        )
        comparisons[label] = stats
        pairwise_rows.append(
            {
                "contrast": label,
                "first_method": first,
                "second_method": second,
                "blocks": stats["blocks"],
                "first_successes": stats["first_successes"],
                "second_successes": stats["second_successes"],
                "first_success_rate": stats["first_success_rate"],
                "second_success_rate": stats["second_success_rate"],
                "paired_difference": stats["paired_block_difference"],
                "paired_ci_low": stats["paired_bootstrap_ci"][0],
                "paired_ci_high": stats["paired_bootstrap_ci"][1],
                "task_cluster_ci_low": stats["task_cluster_bootstrap_ci"][0],
                "task_cluster_ci_high": stats["task_cluster_bootstrap_ci"][1],
                "first_only_successes": stats["first_only_successes"],
                "second_only_successes": stats["second_only_successes"],
                "mcnemar_exact_p": stats["exact_two_sided_mcnemar_binomial_p"],
                "stable_positive": stats["stable_positive"],
                "stable_negative": stats["stable_negative"],
                "all_loto_positive": min(stats["leave_one_task_out"]) > 0,
                "all_loto_negative": max(stats["leave_one_task_out"]) < 0,
            }
        )
    write_csv(args.pairwise, pairwise_rows)

    per_task_rows: list[dict[str, Any]] = []
    for task_id in range(10):
        row: dict[str, Any] = {"task_id": task_id, "states": len(states)}
        for method in METHODS:
            values = [outcomes[(task_id, state, method)] for state in states]
            row[f"{method}_successes"] = int(sum(values))
            row[f"{method}_success_rate"] = float(np.mean(values))
        for label, first, second, _, _ in COMPARISONS:
            row[f"{label}_difference"] = float(
                np.mean([outcomes[(task_id, state, first)] - outcomes[(task_id, state, second)] for state in states])
            )
        per_task_rows.append(row)
    write_csv(args.per_task, per_task_rows)

    loto_rows: list[dict[str, Any]] = []
    for task_id in range(10):
        row = {"omitted_task_id": task_id}
        for label, *_ in COMPARISONS:
            row[f"{label}_estimate"] = comparisons[label]["leave_one_task_out"][task_id]
            row[f"{label}_positive"] = row[f"{label}_estimate"] > 0
            row[f"{label}_negative"] = row[f"{label}_estimate"] < 0
        loto_rows.append(row)
    write_csv(args.loto, loto_rows)

    diagnostic_fields = (
        "mean_arm_effective_age_ticks",
        "mean_gripper_effective_age_ticks",
        "mean_fresh_old_gripper_sign_disagreement",
        "mean_fresh_old_translation_l2",
        "mean_fresh_old_rotation_radians",
        "mean_translation_action_delta_l2",
        "mean_rotation_action_delta_radians",
        "mean_raw_action_acceleration_l2",
        "mean_raw_action_jerk_l2",
    )
    method_summary: dict[str, Any] = {}
    for method in METHODS:
        subset = [episode for episode in episodes if str(episode["method"]) == method]
        method_summary[method] = {
            "episodes": len(subset),
            "successes": int(sum(bool(episode["success"]) for episode in subset)),
            "success_rate": float(np.mean([bool(episode["success"]) for episode in subset])),
            "environment_steps": int(sum(episode["steps"] for episode in subset)),
            "policy_queries": int(sum(episode["policy_queries"] for episode in subset)),
            "secondary_diagnostics": {
                **{
                    field: float(np.mean([episode[field] for episode in subset]))
                    for field in diagnostic_fields
                },
                "mean_episode_steps": float(np.mean([episode["steps"] for episode in subset])),
                "mean_gripper_transitions": float(
                    np.mean([episode["gripper_transitions"] for episode in subset])
                ),
            },
        }
    decision = frozen_decision(comparisons)
    write_json(
        args.summary,
        {
            "schema_version": 1,
            "gate_decision": decision,
            "training_provenance_category": "MULTI-SUITE",
            "primary_outcome": "binary LIBERO Spatial task success",
            "primary_scope": "all ten LIBERO Spatial tasks on ten frozen common official state IDs",
            "inference_unit": "paired task-state block; whole-task cluster bootstrap",
            "planned_and_completed_episodes": 500,
            "task_state_blocks": 100,
            "query_cadence_valid": all(
                int(episode["steps"]) == int(episode["policy_queries"]) for episode in episodes
            ),
            "manifest_sha256": sha256(args.manifest),
            "manifest_provenance": manifest["provenance"],
            "method_summary": method_summary,
            "primary_comparisons": comparisons,
            "decision_rule_source": "research/gate4a2_spatial_generalization_protocol.md",
        },
    )
    print(f"Gate-4A2 decision: {decision}")


if __name__ == "__main__":
    main()
