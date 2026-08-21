#!/usr/bin/env python3
"""Zero-trust recomputation of saved Gate-2B phase-conditioned rollouts.

This script does not run LIBERO or ACT.  It validates and recomputes only what
is identifiable from the saved per-task success vectors and accounting totals.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/phase_conditioned_oracle"
PHASES = ("early", "middle", "late")
HORIZONS = (1, 2, 4, 8, 16)
TASK_IDS = tuple(range(10))
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260821


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "research/audit_outputs")
    return parser.parse_args()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_name(name: str) -> dict[str, Any]:
    parts = name.split("_")
    phase = parts[1]
    if "_global_h" in name:
        horizon = int(name.rsplit("h", 1)[1])
        return {"phase": phase, "strategy": "global", "arm": horizon, "gripper": horizon}
    arm = int(name.split("arm", 1)[1].split("_", 1)[0])
    gripper = int(name.rsplit("grip", 1)[1])
    return {"phase": phase, "strategy": "group", "arm": arm, "gripper": gripper}


def summarize(name: str, task_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    meta = parse_name(name)
    task_rates = np.asarray([float(task_results[str(task_id)]["success_rate"]) for task_id in TASK_IDS])
    successes = sum(int(task_results[str(task_id)]["successes"]) for task_id in TASK_IDS)
    episodes = sum(int(task_results[str(task_id)]["episodes"]) for task_id in TASK_IDS)
    query_rates = np.asarray([float(task_results[str(task_id)]["policy_query_rate"]) for task_id in TASK_IDS])
    return {
        "name": name,
        **meta,
        "macro_success_rate": float(task_rates.mean()),
        "pooled_success_rate": successes / episodes,
        "successes": successes,
        "episodes": episodes,
        "macro_query_rate": float(query_rates.mean()),
        "policy_queries": sum(int(task_results[str(task_id)]["policy_queries"]) for task_id in TASK_IDS),
        "environment_steps": sum(int(task_results[str(task_id)]["environment_steps"]) for task_id in TASK_IDS),
        "task_rates": task_rates.tolist(),
        "phase_exposure": {
            phase: {
                "episodes_reaching_phase": sum(
                    int(task_results[str(task_id)]["phase_summary"][phase]["episodes_reaching_phase"])
                    for task_id in TASK_IDS
                ),
                "environment_steps": sum(
                    int(task_results[str(task_id)]["phase_summary"][phase]["phase_environment_steps"])
                    for task_id in TASK_IDS
                ),
                "policy_queries": sum(
                    int(task_results[str(task_id)]["phase_summary"][phase]["phase_policy_queries"])
                    for task_id in TASK_IDS
                ),
            }
            for phase in PHASES
        },
    }


def task_bootstrap(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    draws = values[indices].mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def select(rows: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    subset = [row for row in rows if row["strategy"] == strategy]
    return sorted(
        subset,
        key=lambda row: (
            -row["macro_success_rate"],
            row["macro_query_rate"],
            row["arm"],
            row["gripper"],
        ),
    )[0]


def leave_one_task_out(rows: list[dict[str, Any]], strategy: str) -> list[dict[str, Any]]:
    subset = [row for row in rows if row["strategy"] == strategy]
    results = []
    for held_out in TASK_IDS:
        train_ids = [task_id for task_id in TASK_IDS if task_id != held_out]
        selected = sorted(
            subset,
            key=lambda row: (
                -float(np.mean(np.asarray(row["task_rates"])[train_ids])),
                row["macro_query_rate"],
                row["arm"],
                row["gripper"],
            ),
        )[0]
        results.append(
            {
                "held_out_task": held_out,
                "selected_name": selected["name"],
                "held_out_success_rate": selected["task_rates"][held_out],
            }
        )
    return results


def stratified_state_split(
    names: list[str],
    cache: dict[str, dict[str, dict[str, Any]]],
    summaries: dict[str, dict[str, Any]],
    *,
    repeats: int = 1000,
) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    selected_counts: Counter[str] = Counter()
    heldout_rates: list[float] = []
    for _ in range(repeats):
        train_masks: dict[int, np.ndarray] = {}
        for task_id in TASK_IDS:
            count = int(cache[names[0]][str(task_id)]["episodes"])
            order = rng.permutation(count)
            mask = np.zeros(count, dtype=bool)
            mask[order[: count // 2]] = True
            train_masks[task_id] = mask

        train_scores: dict[str, float] = {}
        for name in names:
            task_scores = []
            for task_id in TASK_IDS:
                vector = np.asarray(cache[name][str(task_id)]["success_vector"], dtype=np.float64)
                task_scores.append(float(vector[train_masks[task_id]].mean()))
            train_scores[name] = float(np.mean(task_scores))
        selected_name = sorted(
            names,
            key=lambda name: (
                -train_scores[name],
                summaries[name]["macro_query_rate"],
                summaries[name]["arm"],
                summaries[name]["gripper"],
            ),
        )[0]
        selected_counts[selected_name] += 1
        heldout_task_scores = []
        for task_id in TASK_IDS:
            vector = np.asarray(cache[selected_name][str(task_id)]["success_vector"], dtype=np.float64)
            heldout_task_scores.append(float(vector[~train_masks[task_id]].mean()))
        heldout_rates.append(float(np.mean(heldout_task_scores)))
    return {
        "repeats": repeats,
        "selection_frequency": dict(selected_counts.most_common()),
        "mean_heldout_macro_success": float(np.mean(heldout_rates)),
        "heldout_macro_success_ci95_across_splits": [
            float(np.percentile(heldout_rates, 2.5)),
            float(np.percentile(heldout_rates, 97.5)),
        ],
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = json.loads((SOURCE / "config_results.json").read_text(encoding="utf-8"))
    combined = json.loads((SOURCE / "combined_results.json").read_text(encoding="utf-8"))
    selected_configs = json.loads((SOURCE / "selected_configs.json").read_text(encoding="utf-8"))
    issues: list[str] = []

    expected_names = {
        f"phase_{phase}_global_h{horizon}" for phase in PHASES for horizon in HORIZONS
    } | {
        f"phase_{phase}_group_arm{arm}_grip{gripper}"
        for phase in PHASES for arm in HORIZONS for gripper in HORIZONS
    }
    if set(cache) != expected_names:
        issues.append("candidate_name_set_mismatch")
    expected_state_ids = {0: list(range(50)), **{task_id: list(range(20)) for task_id in range(1, 10)}}
    for name, task_results in cache.items():
        if set(task_results) != {str(task_id) for task_id in TASK_IDS}:
            issues.append(f"{name}_task_coverage")
        for task_id in TASK_IDS:
            row = task_results[str(task_id)]
            vector = row["success_vector"]
            if row["state_ids"] != expected_state_ids[task_id]:
                issues.append(f"{name}_task{task_id}_state_ids")
            if len(vector) != len(expected_state_ids[task_id]):
                issues.append(f"{name}_task{task_id}_vector_length")
            if sum(bool(value) for value in vector) != int(row["successes"]):
                issues.append(f"{name}_task{task_id}_success_count")
            if not np.isclose(np.mean(vector), float(row["success_rate"])):
                issues.append(f"{name}_task{task_id}_success_rate")

    summaries = {name: summarize(name, task_results) for name, task_results in cache.items()}
    curve_rows: list[dict[str, Any]] = []
    phase_results: dict[str, Any] = {}
    for phase_index, phase in enumerate(PHASES):
        rows = [summary for summary in summaries.values() if summary["phase"] == phase]
        phase_results[phase] = {}
        for strategy in ("global", "group"):
            strategy_rows = [row for row in rows if row["strategy"] == strategy]
            chosen = select(rows, strategy)
            chosen_rates = np.asarray(chosen["task_rates"])
            for index, row in enumerate(strategy_rows):
                row_rates = np.asarray(row["task_rates"])
                difference = row_rates - chosen_rates
                point_ci = task_bootstrap(row_rates, BOOTSTRAP_SEED + phase_index * 1000 + index)
                diff_ci = task_bootstrap(difference, BOOTSTRAP_SEED + 5000 + phase_index * 1000 + index)
                curve_rows.append(
                    {
                        "phase": phase,
                        "strategy": strategy,
                        "name": row["name"],
                        "arm_horizon": row["arm"],
                        "gripper_horizon": row["gripper"],
                        "episodes": row["episodes"],
                        "successes": row["successes"],
                        "macro_success_rate": row["macro_success_rate"],
                        "macro_success_ci95_low": point_ci[0],
                        "macro_success_ci95_high": point_ci[1],
                        "pooled_success_rate": row["pooled_success_rate"],
                        "macro_query_rate": row["macro_query_rate"],
                        "difference_vs_selected": float(difference.mean()),
                        "difference_vs_selected_ci95_low": diff_ci[0],
                        "difference_vs_selected_ci95_high": diff_ci[1],
                        "statistically_tied_with_selected_descriptive": bool(diff_ci[0] <= 0.0 <= diff_ci[1]),
                        **{f"task_{task_id}_success_rate": row["task_rates"][task_id] for task_id in TASK_IDS},
                    }
                )
            phase_results[phase][strategy] = {
                "selected": chosen,
                "selected_point_ci95": task_bootstrap(chosen_rates, BOOTSTRAP_SEED + 9000 + phase_index),
                "candidates_descriptively_tied_with_selected": [
                    curve["name"]
                    for curve in curve_rows
                    if curve["phase"] == phase
                    and curve["strategy"] == strategy
                    and curve["statistically_tied_with_selected_descriptive"]
                ],
                "leave_one_task_out": leave_one_task_out(rows, strategy),
                "state_split": stratified_state_split(
                    [row["name"] for row in strategy_rows], cache, summaries
                ),
            }

    candidate_queries = sum(row["policy_queries"] for row in summaries.values())
    candidate_steps = sum(row["environment_steps"] for row in summaries.values())
    combined_queries = sum(
        int(task["policy_queries"])
        for config in combined.values()
        for task in config.values()
    )
    combined_steps = sum(
        int(task["environment_steps"])
        for config in combined.values()
        for task in config.values()
    )
    combined_rollouts = sum(
        int(task["episodes"])
        for config in combined.values()
        for task in config.values()
    )

    recomputed_selected_maps = {
        "global": {
            phase: {
                "arm": phase_results[phase]["global"]["selected"]["arm"],
                "gripper": phase_results[phase]["global"]["selected"]["gripper"],
            }
            for phase in PHASES
        },
        "group": {
            phase: {
                "arm": phase_results[phase]["group"]["selected"]["arm"],
                "gripper": phase_results[phase]["group"]["selected"]["gripper"],
            }
            for phase in PHASES
        },
    }
    saved_maps = {config["strategy"]: config["phase_map"] for config in selected_configs}
    if recomputed_selected_maps != saved_maps:
        issues.append("selected_map_mismatch")

    combined_summaries: dict[str, Any] = {}
    for name, task_results in combined.items():
        task_rates = np.asarray([float(task_results[str(task_id)]["success_rate"]) for task_id in TASK_IDS])
        combined_summaries[name] = {
            "macro_success_rate": float(task_rates.mean()),
            "pooled_success_rate": sum(int(task_results[str(task_id)]["successes"]) for task_id in TASK_IDS)
            / sum(int(task_results[str(task_id)]["episodes"]) for task_id in TASK_IDS),
            "task_rates": task_rates.tolist(),
            "successes": sum(int(task_results[str(task_id)]["successes"]) for task_id in TASK_IDS),
            "episodes": sum(int(task_results[str(task_id)]["episodes"]) for task_id in TASK_IDS),
            "policy_queries": sum(int(task_results[str(task_id)]["policy_queries"]) for task_id in TASK_IDS),
            "environment_steps": sum(int(task_results[str(task_id)]["environment_steps"]) for task_id in TASK_IDS),
        }

    static_comparisons: dict[str, Any] = {}
    static_audit_path = args.output_dir / "rollout_evidence_recomputed.json"
    if static_audit_path.is_file():
        static = json.loads(static_audit_path.read_text(encoding="utf-8"))["static_gate"]["task_tables"]
        for combined_name, static_name in (
            ("phase_oracle_global_combined", "group_arm16_grip16"),
            ("phase_oracle_group_combined", "group_arm4_grip16"),
        ):
            differences = []
            for task_id in TASK_IDS:
                combined_vector = np.asarray(combined[combined_name][str(task_id)]["success_vector"], dtype=np.float64)
                static_vector = np.asarray(static[str(task_id)][static_name]["success_vector"], dtype=np.float64)
                differences.append(float(np.mean(combined_vector - static_vector)))
            values = np.asarray(differences)
            static_comparisons[combined_name] = {
                "static_control": static_name,
                "task_differences": differences,
                "macro_difference": float(values.mean()),
                "task_cluster_bootstrap_ci95": task_bootstrap(values, BOOTSTRAP_SEED + 12_000),
            }
    else:
        issues.append("static_audit_output_missing_for_combined_comparison")

    write_csv(args.output_dir / "gate2b_candidate_support_curves.csv", curve_rows)
    dump(
        args.output_dir / "gate2b_recomputed.json",
        {
            "audit_script": str(Path(__file__).relative_to(ROOT)),
            "issues": sorted(set(issues)),
            "definition": {
                "candidate_map": "One target phase varies over 5 global or 25 arm/gripper horizons; other phases remain at global 16 or group (4,16).",
                "candidate_count": len(cache),
                "formula": "3 phases * (5 global + 25 group) = 90",
                "unique_init_state_conditions_per_map": 230,
                "candidate_rollouts": sum(row["episodes"] for row in summaries.values()),
                "combined_map_count": len(combined),
                "combined_rollouts": combined_rollouts,
                "all_rollouts": sum(row["episodes"] for row in summaries.values()) + combined_rollouts,
                "phase_boundaries": {
                    "denominator": 280,
                    "early": "steps 0..93",
                    "middle": "steps 94..186",
                    "late": "steps 187..279",
                    "warning": "These are fixed fractions of the time limit, not thirds of each realized rollout duration or semantic progress.",
                },
            },
            "accounting": {
                "candidate_policy_queries": candidate_queries,
                "combined_policy_queries": combined_queries,
                "all_policy_queries": candidate_queries + combined_queries,
                "candidate_environment_steps": candidate_steps,
                "combined_environment_steps": combined_steps,
                "all_environment_steps": candidate_steps + combined_steps,
                "per_call_validation_limit": "Only aggregate query counts are saved; duplicate calls, invalid outputs, and cache reuse cannot be independently checked per call.",
            },
            "selected_maps_recomputed": recomputed_selected_maps,
            "phase_results": phase_results,
            "combined_maps": combined_summaries,
            "combined_vs_static_controls": static_comparisons,
            "saved_data_limitations": [
                "No per-step Gate-2B action, prediction, phase, or query log is saved.",
                "Quartile, continuous-time, semantic-progress, and no-phase resegmentation cannot be recomputed from the saved aggregate vectors.",
                "No thresholded support metric is used here; the outcome is rollout success. Threshold sensitivity is therefore not applicable.",
                "Candidate-grid sensitivity outside {1,2,4,8,16} is not identifiable without new rollouts.",
                "Maps were selected and evaluated on the same 230 init-state conditions; point estimates are selection-biased.",
            ],
        },
    )
    print(json.dumps({
        "issues": sorted(set(issues)),
        "candidates": len(cache),
        "candidate_queries": candidate_queries,
        "combined_queries": combined_queries,
        "all_queries": candidate_queries + combined_queries,
        "selected_maps": recomputed_selected_maps,
        "output": str(args.output_dir / "gate2b_recomputed.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
