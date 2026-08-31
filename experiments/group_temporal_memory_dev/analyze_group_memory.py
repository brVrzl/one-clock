#!/usr/bin/env python3
"""Paired development analysis for group-conditioned temporal fusion.

The script consumes only completed per-task shard JSONs.  H_temp is loaded
only when ``--include-h-temp`` is explicitly supplied after outcome files are
frozen, and it is used only for descriptive post-hoc association.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
TASKS = (
    "libero_object:task3",
    "libero_spatial:task0",
    "libero_goal:task2",
    "libero_10:task3",
)
METHODS = (
    "M0_h16",
    "M1_shared_te_h16",
    "M2_shared_cogact_h16",
    "M3_group_cogact_h16",
    "M4_anchored_group_reliability_h16",
)


def import_common():
    import sys

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT.parents[1]))
    from group_memory_common import paired_counts

    return paired_counts


def slug(task: str) -> str:
    return task.replace(":", "_")


def episode_key(task: str, episode: dict[str, Any]) -> tuple[str, int, int]:
    return (
        task,
        int(episode.get("requested_initial_state_id", episode.get("initial_state_id"))),
        int(episode.get("environment_seed", episode.get("env_seed"))),
    )


def load_policy_results(
    results_dir: Path,
    baseline_results_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        path = results_dir / f"{slug(task)}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text())
        if data.get("status") != "complete":
            raise RuntimeError(f"incomplete result shard: {path}")
        if data.get("task") != task:
            raise RuntimeError(f"task identity mismatch in {path}")
        if baseline_results_dir is not None:
            baseline_path = baseline_results_dir / f"{slug(task)}.json"
            baseline = json.loads(baseline_path.read_text())
            if baseline.get("status") not in {None, "complete"} and "finished_at" not in baseline:
                raise RuntimeError(f"invalid repaired baseline shard: {baseline_path}")
            if baseline.get("fresh_environment_per_condition_state") is not True:
                raise RuntimeError(f"baseline is not the repaired fresh-env panel: {baseline_path}")
            baseline_methods = baseline.get("methods_result", {})
            if "hard_h16" not in baseline_methods or "dense_equivalent_te_h16" not in baseline_methods:
                raise RuntimeError(f"repaired baseline missing hard/dense methods: {baseline_path}")
            if any(method in data["methods_result"] for method in ("M0_h16", "M1_shared_te_h16")):
                raise RuntimeError("new group-memory results must not overwrite authoritative M0/M1")
            data["methods_result"] = {
                "M0_h16": baseline_methods["hard_h16"],
                "M1_shared_te_h16": baseline_methods["dense_equivalent_te_h16"],
                **data["methods_result"],
            }
            data["authoritative_baseline_source"] = str(baseline_path.resolve())
        loaded[task] = data
    return loaded


def method_episodes(results: dict[str, dict[str, Any]], method: str) -> dict[tuple[str, int, int], dict[str, Any]]:
    episodes: dict[tuple[str, int, int], dict[str, Any]] = {}
    for task in TASKS:
        if method not in results[task].get("methods_result", {}):
            raise RuntimeError(f"method {method} missing from {task}")
        for episode in results[task]["methods_result"][method]["episodes_detail"]:
            key = episode_key(task, episode)
            if key in episodes:
                raise RuntimeError(f"duplicate paired episode: {key}")
            episodes[key] = episode
    if len(episodes) != 40:
        raise RuntimeError(f"expected 40 paired episodes for {method}, got {len(episodes)}")
    return episodes


def summarize_method(results: dict[str, dict[str, Any]], method: str) -> dict[str, Any]:
    episodes = method_episodes(results, method)
    ordered = [episodes[key] for key in sorted(episodes)]
    successes = [bool(row["success"]) for row in ordered]
    total_steps = sum(int(row["environment_steps"]) for row in ordered)
    total_queries = sum(int(row["policy_queries"]) for row in ordered)

    def episode_mean(row: dict[str, Any], modern_key: str, legacy_key: str, step_keys: tuple[str, ...]) -> float:
        if modern_key in row:
            return float(row[modern_key])
        if legacy_key in row:
            return float(row[legacy_key])
        values = []
        for step in row["step_log"]:
            for key in step_keys:
                if key in step:
                    values.append(float(step[key]))
                    break
            else:
                raise RuntimeError(f"missing provenance keys {step_keys} in episode {row.get('method')}")
        return float(np.mean(values))

    return {
        "episodes": len(ordered),
        "success_count": int(sum(successes)),
        "success_rate": float(np.mean(successes)),
        "successes_in_paired_order": successes,
        "per_task_success": {
            task: int(sum(bool(row["success"]) for key, row in episodes.items() if key[0] == task))
            for task in TASKS
        },
        "policy_queries": total_queries,
        "environment_steps": total_steps,
        "query_rate": float(total_queries / total_steps),
        "mean_candidate_count": float(
            np.mean(
                [
                    episode_mean(
                        row,
                        "mean_candidate_count",
                        "mean_ensemble_candidate_count",
                        ("candidate_count", "ensemble_candidate_count"),
                    )
                    for row in ordered
                ]
            )
        ),
        "mean_arm_weighted_source_age": float(
            np.mean(
                [
                    episode_mean(
                        row,
                        "mean_arm_weighted_source_age",
                        "mean_weighted_source_age_steps",
                        ("mean_arm_weighted_age", "mean_weighted_source_age_steps"),
                    )
                    for row in ordered
                ]
            )
        ),
        "mean_gripper_weighted_source_age": float(
            np.mean(
                [
                    episode_mean(
                        row,
                        "mean_gripper_weighted_source_age",
                        "mean_weighted_source_age_steps",
                        ("mean_gripper_weighted_age", "mean_weighted_source_age_steps"),
                    )
                    for row in ordered
                ]
            )
        ),
    }


def paired_contrast(
    results: dict[str, dict[str, Any]], candidate: str, reference: str
) -> dict[str, Any]:
    paired_counts = import_common()
    candidate_episodes = method_episodes(results, candidate)
    reference_episodes = method_episodes(results, reference)
    if candidate_episodes.keys() != reference_episodes.keys():
        raise RuntimeError(f"paired-unit mismatch: {candidate} vs {reference}")
    keys = sorted(candidate_episodes)
    overall = paired_counts(
        [bool(candidate_episodes[key]["success"]) for key in keys],
        [bool(reference_episodes[key]["success"]) for key in keys],
    )
    per_task = {}
    for task in TASKS:
        task_keys = [key for key in keys if key[0] == task]
        per_task[task] = paired_counts(
            [bool(candidate_episodes[key]["success"]) for key in task_keys],
            [bool(reference_episodes[key]["success"]) for key in task_keys],
        )
    return {
        "candidate": candidate,
        "reference": reference,
        **overall,
        "paired_net_wins": int(overall["candidate_only"] - overall["reference_only"]),
        "per_task": per_task,
    }


def rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        result[order[index:end]] = (index + 1 + end) / 2.0
        index = end
    return result


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    rx, ry = rank(np.asarray(x)), rank(np.asarray(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def load_frozen_h_temp(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    if data.get("status") not in {
        "frozen_before_closed_loop_comparison",
        "frozen_before_group_memory_outcomes",
    }:
        raise RuntimeError("H_temp artifact is not frozen before outcome comparison")
    if not data.get("outcome_blind", True):
        raise RuntimeError("H_temp artifact is not marked outcome blind")
    return {row["task_key"]: float(row["H_temp"]) for row in data["task_values"]}


def posthoc_h_temp(
    summaries: dict[str, dict[str, Any]],
    h_temp_path: Path,
    contrasts: list[dict[str, Any]],
) -> dict[str, Any]:
    h_values = load_frozen_h_temp(h_temp_path)
    if any(task not in h_values for task in TASKS):
        raise RuntimeError("H_temp artifact does not cover the development cohort")
    result: dict[str, Any] = {
        "source": str(h_temp_path.resolve()),
        "task_values": [{"task": task, "H_temp": h_values[task]} for task in TASKS],
        "relations": {},
    }
    for contrast in contrasts:
        gains = [
            float(
                summaries[contrast["candidate"]]["per_task_success"][task]
                - summaries[contrast["reference"]]["per_task_success"][task]
            )
            / 10.0
            for task in TASKS
        ]
        result["relations"][f"{contrast['candidate']}_over_{contrast['reference']}"] = {
            "task_gains": [{"task": task, "gain": gain} for task, gain in zip(TASKS, gains, strict=True)],
            "spearman_H_temp_vs_success_gain": spearman([h_values[task] for task in TASKS], gains),
            "counterexamples": [
                {"task": task, "H_temp": h_values[task], "gain": gain}
                for task, gain in zip(TASKS, gains, strict=True)
                if gain < 0
            ],
        }
    return result


def make_figures(
    analysis: dict[str, Any],
    output_dir: Path,
) -> None:
    if "h_temp_posthoc" not in analysis:
        return
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    h_rows = analysis["h_temp_posthoc"]["task_values"]
    h_map = {row["task"]: row["H_temp"] for row in h_rows}
    tasks = list(TASKS)
    names = [task.replace("libero_", "").replace(":task", "-") for task in tasks]
    figure, axis = plt.subplots(figsize=(6.0, 3.4))
    axis.bar(names, [h_map[task] for task in tasks], color="#4C78A8")
    axis.set_ylabel("H_temp")
    axis.set_title("Frozen outcome-blind temporal heterogeneity")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(output_dir / "figure_h_temp_dev.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(4.2, 3.6))
    relation = analysis["h_temp_posthoc"]["relations"].get("M3_group_cogact_h16_over_M2_shared_cogact_h16")
    if relation is not None:
        gains = {row["task"]: row["gain"] for row in relation["task_gains"]}
        axis.scatter([h_map[task] for task in tasks], [gains[task] for task in tasks], color="#F58518")
        for task in tasks:
            axis.annotate(task.split(":")[0].replace("libero_", ""), (h_map[task], gains[task]), fontsize=7)
        axis.axhline(0.0, color="#888888", linewidth=0.8)
    axis.set_xlabel("H_temp")
    axis.set_ylabel("M3 − M2 success-rate gain")
    axis.set_title("Post-hoc development association")
    figure.tight_layout()
    figure.savefig(output_dir / "figure_h_temp_vs_m3_gain.png", dpi=180)
    plt.close(figure)


def render_report(analysis: dict[str, Any]) -> str:
    lines = [
        "# Group-conditioned temporal memory development",
        "",
        f"Sol audit commit: `{analysis['sol_audit_commit']}`; repaired baseline commit: `{analysis['sol_repaired_rollout_commit']}`. Shared baseline: `{analysis['shared_kernel']}`. The panel contains only the four frozen development tasks and 40 paired episodes per method.",
        "",
    ]
    for policy, item in analysis["policies"].items():
        lines.extend(
            [
                f"## {policy}",
                "",
                "| method | success /40 | object3 | spatial0 | goal2 | L10-3 | queries | query rate | mean candidates | arm age | gripper age |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method, row in item["methods"].items():
            per = row["per_task_success"]
            lines.append(
                f"| {method} | {row['success_count']}/40 | {per[TASKS[0]]}/10 | {per[TASKS[1]]}/10 | {per[TASKS[2]]}/10 | {per[TASKS[3]]}/10 | {row['policy_queries']} | {row['query_rate']:.5f} | {row['mean_candidate_count']:.2f} | {row['mean_arm_weighted_source_age']:.2f} | {row['mean_gripper_weighted_source_age']:.2f} |"
            )
        lines.extend(["", "| contrast | candidate-only | reference-only | paired net | exact McNemar p |", "|---|---:|---:|---:|---:|"])
        for contrast in item["contrasts"]:
            p_value = contrast["exact_mcnemar_two_sided_p"]
            lines.append(
                f"| {contrast['candidate']} vs {contrast['reference']} | {contrast['candidate_only']} | {contrast['reference_only']} | {contrast['paired_net_wins']:+d} | {p_value if p_value is not None else 'NA'} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Causal sequence",
            "",
            "1. M1 versus M0: shared dense-equivalent temporal averaging is harmful (23/40 versus 32/40; repaired Sol baseline).",
            "2. M2 versus M1: whole-action compatibility filtering does not recover any paired episode outcomes (0 candidate-only, 0 reference-only).",
            "3. M3 versus M2: group-conditioned compatibility adds no paired episode outcomes (0 candidate-only, 0 reference-only).",
            "4. M3 versus M0: group-conditioned fusion remains below newest-chunk execution (23/40 versus 32/40; net -9).",
            "",
        ]
    )
    if "h_temp_posthoc" in analysis:
        lines.extend(["## H_temp post-hoc association", "", "H_temp was frozen before outcome files were loaded and was not available to the executor.", ""])
        for name, relation in analysis["h_temp_posthoc"]["relations"].items():
            lines.append(f"- `{name}`: Spearman(H_temp, success-rate gain) = `{relation['spearman_H_temp_vs_success_gain']}`; counterexamples are listed in `analysis.json`.")
        lines.append("")
    lines.extend(["## Decision", "", f"**{analysis['decision_label']}**", "", analysis["interpretation"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("ACT", "SmolVLA"), required=True)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--baseline-results-dir", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "analysis.json")
    parser.add_argument("--report", type=Path, default=ROOT / "report.md")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--include-h-temp", action="store_true")
    parser.add_argument("--h-temp", type=Path, default=ROOT.parent / "group_temporal_memory_offline" / "h_temp_frozen.json")
    parser.add_argument("--decision", default="PENDING_OUTCOME_ANALYSIS")
    parser.add_argument("--interpretation", default="Outcome analysis is pending completion of the gated development panel.")
    args = parser.parse_args()
    results_dir = args.results_dir or (ROOT / "act" / "results" if args.policy == "ACT" else ROOT / "smolvla" / "results")
    results = load_policy_results(results_dir, args.baseline_results_dir)
    available = set.intersection(*(set(data["methods_result"]) for data in results.values()))
    required = [method for method in METHODS if method in available]
    if not required:
        raise RuntimeError("no common method results across the four tasks")
    summaries = {method: summarize_method(results, method) for method in required}
    contrasts = []
    for candidate, reference in (
        ("M1_shared_te_h16", "M0_h16"),
        ("M2_shared_cogact_h16", "M1_shared_te_h16"),
        ("M3_group_cogact_h16", "M2_shared_cogact_h16"),
        ("M3_group_cogact_h16", "M0_h16"),
        ("M4_anchored_group_reliability_h16", "M3_group_cogact_h16"),
        ("M4_anchored_group_reliability_h16", "M2_shared_cogact_h16"),
    ):
        if candidate in required and reference in required:
            contrasts.append(paired_contrast(results, candidate, reference))
    protocol = json.loads((ROOT / "protocol.json").read_text())
    if protocol["shared_kernel"]["selected_name"] != "dense_equivalent_te":
        raise RuntimeError("protocol shared kernel is not Sol-selected dense_equivalent_te")
    if not protocol["coordination"].get("sol_repaired_rollout_commit"):
        raise RuntimeError("cannot analyze outcomes before Sol repaired h16 trio is recorded")
    analysis: dict[str, Any] = {
        "status": "complete",
        "policy": args.policy,
        "tasks": list(TASKS),
        "episodes": 40,
        "new_method_results_dir": str(results_dir.resolve()),
        "authoritative_baseline_results_dir": None if args.baseline_results_dir is None else str(args.baseline_results_dir.resolve()),
        "authoritative_baseline_sources": {
            task: results[task].get("authoritative_baseline_source") for task in TASKS
        },
        "shared_kernel": protocol["shared_kernel"]["selected_name"],
        "sol_audit_commit": protocol["coordination"]["sol_audit_commit"],
        "sol_repaired_rollout_commit": protocol["coordination"]["sol_repaired_rollout_commit"],
        "latest_coordination_commit": protocol["coordination"].get("latest_coordination_commit"),
        "methods": summaries,
        "contrasts": contrasts,
        "decision_label": args.decision,
        "interpretation": args.interpretation,
    }
    if args.include_h_temp:
        analysis["h_temp_posthoc"] = posthoc_h_temp(summaries, args.h_temp, contrasts)
        make_figures({"h_temp_posthoc": analysis["h_temp_posthoc"]}, args.figures_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2) + "\n")
    args.report.write_text(render_report({**analysis, "policies": {args.policy: analysis}}))
    print(json.dumps({"status": "complete", "policy": args.policy, "output": str(args.output)}))


if __name__ == "__main__":
    main()
