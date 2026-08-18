#!/usr/bin/env python3
"""Aggregate and analyze the paired 50-state LIBERO static-horizon sweep."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any

import numpy as np

from analyze_libero_static_grid import HORIZONS, format_cell, public_entry, save_heatmap


STATE_IDS = tuple(range(50))
OLD_STATE_IDS = tuple(range(20))
EXTENSION_STATE_IDS = tuple(range(20, 50))
Z95 = 1.959963984540054


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-runs-root",
        type=Path,
        default=root / "experiments/runs/libero_static_grid_20",
    )
    parser.add_argument(
        "--extension-runs-root",
        type=Path,
        default=root / "experiments/runs/libero_static_grid_50_extension",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=root / "experiments/libero_static_grid_50.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=root / "experiments/libero_static_grid_50.md",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=root / "experiments/figures",
    )
    return parser.parse_args()


def configuration_key(metadata: dict[str, Any]) -> tuple[str, int, int]:
    strategy = str(metadata["strategy"])
    if strategy == "global_fixed":
        horizon = int(metadata["global_horizon"])
        return strategy, horizon, horizon
    return (
        strategy,
        int(metadata["group_horizons"]["arm"]),
        int(metadata["group_horizons"]["gripper"]),
    )


def load_segments(root: Path, allowed_states: set[int]) -> dict[tuple[str, int, int], list[dict[str, Any]]]:
    segments: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for run_dir in sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "metadata.json").is_file()
    ):
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        episodes = [
            json.loads(line)
            for line in (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        steps = [
            json.loads(line)
            for line in (run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        ids = [int(episode["init_state_id"]) for episode in episodes]
        if not set(ids).issubset(allowed_states):
            raise ValueError(f"{run_dir} contains states outside the requested source range")
        if len(ids) != len(set(ids)):
            raise ValueError(f"{run_dir} contains duplicate init states")
        if any(int(episode["seed"]) != 1000 + int(episode["init_state_id"]) for episode in episodes):
            raise ValueError(f"{run_dir} does not use the established seed mapping")
        if tuple(metadata.get("observed_chunk_shape", ())) != (100, 7):
            raise ValueError(f"{run_dir} does not record the verified ACT chunk shape")
        step_by_state: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in steps:
            step_by_state[int(record["init_state_id"])].append(record)
        for episode in episodes:
            state_id = int(episode["init_state_id"])
            if state_id not in step_by_state:
                raise ValueError(f"{run_dir} has no step trace for init state {state_id}")
            episode_copy = dict(episode)
            episode_copy["steps"] = step_by_state[state_id]
            episode_copy["source_run_dir"] = str(run_dir)
            episode_copy["source_metadata"] = metadata
            segments[configuration_key(metadata)].append(episode_copy)
    return segments


def validate_and_combine(
    old_segments: dict[tuple[str, int, int], list[dict[str, Any]]],
    extension_segments: dict[tuple[str, int, int], list[dict[str, Any]]],
) -> tuple[dict[tuple[str, int, int], list[dict[str, Any]]], dict[str, Any]]:
    expected_keys = {
        ("global_fixed", horizon, horizon) for horizon in HORIZONS
    } | {
        ("groupwise_fixed", arm, gripper)
        for arm in HORIZONS
        for gripper in HORIZONS
    }
    if set(old_segments) != expected_keys or set(extension_segments) != expected_keys:
        raise ValueError("old and extension sources must each cover exactly the 30 configurations")
    combined: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    issues: list[str] = []
    for key in sorted(expected_keys):
        old = old_segments[key]
        extension = extension_segments[key]
        old_ids = sorted(int(episode["init_state_id"]) for episode in old)
        extension_ids = sorted(int(episode["init_state_id"]) for episode in extension)
        if old_ids != list(OLD_STATE_IDS):
            issues.append(f"{key}: old source IDs are {old_ids}")
        if extension_ids != list(EXTENSION_STATE_IDS):
            issues.append(f"{key}: extension source IDs are {extension_ids}")
        records = sorted(old + extension, key=lambda episode: int(episode["init_state_id"]))
        if [int(episode["init_state_id"]) for episode in records] != list(STATE_IDS):
            issues.append(f"{key}: combined IDs are not 0..49")
        combined[key] = records
    if issues:
        raise ValueError("source coverage validation failed: " + "; ".join(issues))
    return combined, {
        "old_state_ids": list(OLD_STATE_IDS),
        "extension_state_ids": list(EXTENSION_STATE_IDS),
        "final_state_ids": list(STATE_IDS),
        "old_segments": sum(len(values) for values in old_segments.values()),
        "extension_segments": sum(len(values) for values in extension_segments.values()),
        "old_episodes": sum(len(values) for values in old_segments.values()),
        "extension_episodes": sum(len(values) for values in extension_segments.values()),
        "final_episodes": sum(len(values) for values in combined.values()),
    }


def wilson_interval(successes: int, episodes: int) -> list[float]:
    proportion = successes / episodes
    denominator = 1.0 + Z95**2 / episodes
    center = (proportion + Z95**2 / (2.0 * episodes)) / denominator
    half_width = (
        Z95
        * math.sqrt(
            proportion * (1.0 - proportion) / episodes
            + Z95**2 / (4.0 * episodes**2)
        )
        / denominator
    )
    return [center - half_width, center + half_width]


def summarize_episodes(
    episodes: list[dict[str, Any]],
    key: tuple[str, int, int],
    *,
    name: str,
) -> dict[str, Any]:
    strategy, arm_horizon, gripper_horizon = key
    successful = [episode for episode in episodes if bool(episode["success"])]
    environment_steps = [int(episode["environment_steps"]) for episode in episodes]
    policy_queries = [int(episode["policy_queries"]) for episode in episodes]
    query_rates = [float(episode["policy_query_rate"]) for episode in episodes]
    success_steps = [int(episode["environment_steps"]) for episode in successful]
    success_vector = [bool(episode["success"]) for episode in episodes]
    return {
        "name": name,
        "strategy": strategy,
        "arm_horizon": arm_horizon,
        "gripper_horizon": gripper_horizon,
        "episodes": len(episodes),
        "successes": len(successful),
        "success_rate": len(successful) / len(episodes),
        "success_rate_ci95": wilson_interval(len(successful), len(episodes)),
        "environment_steps": sum(environment_steps),
        "mean_environment_steps": float(np.mean(environment_steps)),
        "median_environment_steps": float(statistics.median(environment_steps)),
        "mean_success_steps": float(np.mean(success_steps)) if success_steps else None,
        "median_success_steps": float(statistics.median(success_steps)) if success_steps else None,
        "policy_queries": sum(policy_queries),
        "mean_policy_queries": float(np.mean(policy_queries)),
        "mean_policy_query_rate": float(np.mean(query_rates)),
        "policy_query_rate": sum(policy_queries) / sum(environment_steps),
        "mean_source_age_arm": float(np.mean([float(e["mean_source_age_arm"]) for e in episodes])),
        "mean_source_age_gripper": float(np.mean([float(e["mean_source_age_gripper"]) for e in episodes])),
        "successful_init_state_ids": [int(e["init_state_id"]) for e in successful],
        "failed_init_state_ids": [int(e["init_state_id"]) for e in episodes if not bool(e["success"])],
        "success_vector": success_vector,
        "episodes_data": episodes,
    }


def paired_counts(
    a: dict[str, Any],
    b: dict[str, Any],
    state_ids: tuple[int, ...] = STATE_IDS,
) -> dict[str, int]:
    a_by_state = {int(e["init_state_id"]): bool(e["success"]) for e in a["episodes_data"]}
    b_by_state = {int(e["init_state_id"]): bool(e["success"]) for e in b["episodes_data"]}
    counts = {"both_succeed": 0, "a_only_succeeds": 0, "b_only_succeeds": 0, "both_fail": 0}
    for state_id in state_ids:
        a_success = a_by_state[state_id]
        b_success = b_by_state[state_id]
        if a_success and b_success:
            counts["both_succeed"] += 1
        elif a_success:
            counts["a_only_succeeds"] += 1
        elif b_success:
            counts["b_only_succeeds"] += 1
        else:
            counts["both_fail"] += 1
    return counts


def exact_mcnemar_p(counts: dict[str, int]) -> float:
    discordant_a = counts["a_only_succeeds"]
    discordant_b = counts["b_only_succeeds"]
    total = discordant_a + discordant_b
    if total == 0:
        return 1.0
    lower_tail = sum(math.comb(total, k) for k in range(min(discordant_a, discordant_b) + 1)) / 2**total
    return min(1.0, 2.0 * lower_tail)


def paired_comparison(
    a: dict[str, Any],
    b: dict[str, Any],
    state_ids: tuple[int, ...] = STATE_IDS,
) -> dict[str, Any]:
    counts = paired_counts(a, b, state_ids)
    return {
        "a": a["name"],
        "b": b["name"],
        "a_success_rate": a["success_rate"],
        "b_success_rate": b["success_rate"],
        "a_policy_query_rate": a["policy_query_rate"],
        "b_policy_query_rate": b["policy_query_rate"],
        "counts": counts,
        "discordant_pairs": counts["a_only_succeeds"] + counts["b_only_succeeds"],
        "exact_two_sided_mcnemar_p": exact_mcnemar_p(counts),
        "b_minus_a_success_difference": b["success_rate"] - a["success_rate"],
    }


def diagonal_checks(entries: dict[tuple[str, int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ("success", "environment_steps", "policy_queries", "policy_query_rate")
    result = []
    for horizon in HORIZONS:
        global_entry = entries[("global_fixed", horizon, horizon)]
        group_entry = entries[("groupwise_fixed", horizon, horizon)]
        global_by_state = {int(e["init_state_id"]): e for e in global_entry["episodes_data"]}
        group_by_state = {int(e["init_state_id"]): e for e in group_entry["episodes_data"]}
        differences = {
            field: [
                state_id
                for state_id in STATE_IDS
                if global_by_state[state_id][field] != group_by_state[state_id][field]
            ]
            for field in fields
        }
        result.append({
            "horizon": horizon,
            "global": global_entry["name"],
            "groupwise": group_entry["name"],
            "all_equal": not any(differences.values()),
            "differences_by_field": differences,
        })
    return result


def pareto_analysis(entries: list[dict[str, Any]]) -> dict[str, Any]:
    dominated_by: dict[str, list[str]] = {}
    for candidate in entries:
        better = []
        for other in entries:
            if other is candidate:
                continue
            lower_budget = other["mean_policy_query_rate"] <= candidate["mean_policy_query_rate"]
            higher_success = other["success_rate"] >= candidate["success_rate"]
            strict = (
                other["mean_policy_query_rate"] < candidate["mean_policy_query_rate"]
                or other["success_rate"] > candidate["success_rate"]
            )
            if lower_budget and higher_success and strict:
                better.append(other["name"])
        if better:
            dominated_by[candidate["name"]] = sorted(better)
    frontier = [entry for entry in entries if entry["name"] not in dominated_by]
    return {
        "x": "mean_policy_query_rate",
        "y": "success_rate",
        "frontier": [public_entry(entry) for entry in frontier],
        "dominated": dominated_by,
    }


def trace_metrics(steps: list[dict[str, Any]]) -> dict[str, float]:
    actions = np.asarray([record["action"] for record in sorted(steps, key=lambda r: int(r["environment_step"]))], dtype=float)
    gripper = actions[:, 6]
    arm = actions[:, :6]
    gripper_delta = np.abs(np.diff(gripper))
    arm_delta = np.linalg.norm(np.diff(arm, axis=0), axis=1)
    nonzero_signs = np.sign(gripper)[np.sign(gripper) != 0]
    sign_changes = int(np.sum(nonzero_signs[1:] != nonzero_signs[:-1])) if len(nonzero_signs) > 1 else 0
    negative = gripper[:-1] < 0
    positive = gripper[:-1] > 0
    next_negative = gripper[1:] < 0
    next_positive = gripper[1:] > 0
    zero_crossings = int(np.sum((negative & ~next_negative) | (positive & ~next_positive)))
    return {
        "environment_steps": float(len(actions)),
        "gripper_total_variation": float(np.sum(gripper_delta)),
        "gripper_mean_abs_step_change": float(np.mean(gripper_delta)) if len(gripper_delta) else 0.0,
        "gripper_sign_changes_ignoring_zero_hold": float(sign_changes),
        "gripper_zero_threshold_crossings": float(zero_crossings),
        "gripper_positive_fraction": float(np.mean(gripper > 0)),
        "gripper_negative_fraction": float(np.mean(gripper < 0)),
        "gripper_zero_fraction": float(np.mean(gripper == 0)),
        "arm_mean_l2_step_change": float(np.mean(arm_delta)) if len(arm_delta) else 0.0,
        "arm_total_l2_variation": float(np.sum(arm_delta)),
    }


def aggregate_trace_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [trace_metrics(episode["steps"]) for episode in episodes]
    successful = [metric for metric, episode in zip(metrics, episodes) if bool(episode["success"])]
    failed = [metric for metric, episode in zip(metrics, episodes) if not bool(episode["success"])]

    def aggregate(values: list[dict[str, float]]) -> dict[str, float | None]:
        if not values:
            return {key: None for key in metrics[0]}
        return {key: float(np.mean([value[key] for value in values])) for key in values[0]}

    return {
        "all": aggregate(metrics),
        "successful": aggregate(successful),
        "failed": aggregate(failed),
    }


def make_scatter(entries: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    markers = {
        "global": ("global_fixed", "o"),
        "diagonal group-wise": ("diagonal", "s"),
        "off-diagonal group-wise": ("off_diagonal", "^"),
    }
    figure, axis = plt.subplots(figsize=(7, 5))
    for label, (kind, marker) in markers.items():
        selected = []
        for entry in entries:
            if kind == "global" and entry["strategy"] == "global_fixed":
                selected.append(entry)
            elif kind == "diagonal" and entry["strategy"] == "groupwise_fixed" and entry["arm_horizon"] == entry["gripper_horizon"]:
                selected.append(entry)
            elif kind == "off_diagonal" and entry["strategy"] == "groupwise_fixed" and entry["arm_horizon"] != entry["gripper_horizon"]:
                selected.append(entry)
        axis.scatter(
            [entry["mean_policy_query_rate"] for entry in selected],
            [entry["success_rate"] for entry in selected],
            marker=marker,
            label=label,
            alpha=0.8,
        )
    for entry in entries:
        axis.annotate(entry["name"].replace("group_arm", "(").replace("_grip", ",").replace("global_h", "G"), (entry["mean_policy_query_rate"], entry["success_rate"]), fontsize=6, xytext=(2, 2), textcoords="offset points")
    axis.set_xlabel("mean policy query rate")
    axis.set_ylabel("success rate")
    axis.set_title("LIBERO static success vs policy-query rate (50 states)")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def matrix(entries: dict[tuple[str, int, int], dict[str, Any]], field: str) -> list[list[Any]]:
    return [
        [entries[("groupwise_fixed", arm, gripper)][field] for gripper in HORIZONS]
        for arm in HORIZONS
    ]


def markdown_matrix(title: str, values: list[list[Any]], precision: int) -> list[str]:
    lines = [f"### {title}", "", "| arm \\ gripper | " + " | ".join(str(h) for h in HORIZONS) + " |", "|---:|" + "---:|" * len(HORIZONS)]
    for arm, row in zip(HORIZONS, values):
        lines.append("| " + str(arm) + " | " + " | ".join(format_cell(value, precision) for value in row) + " |")
    return lines


def write_markdown(
    path: Path,
    artifact: dict[str, Any],
    entries: list[dict[str, Any]],
    matrices: dict[str, list[list[Any]]],
) -> None:
    globals_only = sorted((entry for entry in entries if entry["strategy"] == "global_fixed"), key=lambda e: e["arm_horizon"])
    lines = [
        "# LIBERO static horizon landscape — 50 paired states",
        "",
        "This combines the existing states 0–19 with the controlled extension states 20–49. It is a diagnostic execution result, not a statistical claim.",
        "",
        f"States: `{artifact['sources']['final_state_ids']}`; total episodes: **{artifact['sources']['final_episodes']}**; pairing valid: **{artifact['pairing_valid']}**.",
        "",
        "## Global fixed",
        "",
        "| Global horizon | Successes | Success rate (95% Wilson CI) | Mean success steps | Query rate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for entry in globals_only:
        ci = entry["success_rate_ci95"]
        lines.append(f"| {entry['arm_horizon']} | {entry['successes']} | {entry['success_rate']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {format_cell(entry['mean_success_steps'], 2)} | {entry['policy_query_rate']:.3f} |")
    lines.extend(["", *markdown_matrix("Group-wise success rate", matrices["success_rate"], 3), "", *markdown_matrix("Group-wise mean successful completion steps", matrices["mean_success_steps"], 2), "", *markdown_matrix("Group-wise mean policy query rate", matrices["mean_policy_query_rate"], 3), ""])
    lines.extend(["## Best configurations", ""])
    for label in ("best_global", "best_groupwise", "best_off_diagonal"):
        lines.append(f"- **{label}:** " + ", ".join(entry["name"] for entry in artifact[label]))
    lines.extend(["", "## Diagonal controls", ""])
    for check in artifact["diagonal_checks"]:
        lines.append(f"- `{check['global']} vs {check['groupwise']}`: all equal = **{check['all_equal']}**; differences `{check['differences_by_field']}`")
    lines.extend(["", "## Key paired comparisons", ""])
    for label, comparison in artifact["key_paired_comparisons"].items():
        lines.append(f"- **{label}:** `{comparison['a']}` vs `{comparison['b']}`, counts `{comparison['counts']}`, exact p={comparison['exact_two_sided_mcnemar_p']:.4f}, difference b−a={comparison['b_minus_a_success_difference']:.3f}")
    lines.extend(["", "## Directionality", "", "| Pair | A success | B success | A query rate | B query rate | A-only | B-only | Both fail | Exact p |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for comparison in artifact["directionality"]:
        counts = comparison["counts"]
        lines.append(f"| {comparison['a']} vs {comparison['b']} | {comparison['a_success_rate']:.3f} | {comparison['b_success_rate']:.3f} | {comparison['a_policy_query_rate']:.3f} | {comparison['b_policy_query_rate']:.3f} | {counts['a_only_succeeds']} | {counts['b_only_succeeds']} | {counts['both_fail']} | {comparison['exact_two_sided_mcnemar_p']:.4f} |")
    lines.extend(["", "## Pareto frontier", "", "- " + "\n- ".join(entry["name"] for entry in artifact["pareto"]["frontier"])])
    lines.extend(["", "## 20-state vs 50-state comparison", "", f"- Best global: `{artifact['comparison_20_vs_50']['old_best_global']}` → `{artifact['comparison_20_vs_50']['final_best_global']}`.", f"- Best off-diagonal: `{artifact['comparison_20_vs_50']['old_best_off_diagonal']}` → `{artifact['comparison_20_vs_50']['final_best_off_diagonal']}`.", f"- Pareto frontier: `{artifact['comparison_20_vs_50']['old_pareto_frontier']}` → `{artifact['comparison_20_vs_50']['final_pareto_frontier']}`."])
    lines.extend(["", "## Exploratory trace comparisons", "", "LIBERO/robosuite PandaGripper source semantics were verified: gripper command −1 means open, +1 means closed, and zero produces no sign change/holds the current gripper action. Trace metrics are post-hoc execution-pattern diagnostics only.", "", "| Comparison | Side | Gripper TV | Mean gripper Δ | Gripper sign changes | Arm mean L2 Δ | Arm total L2 |", "|---|---|---:|---:|---:|---:|---:|"])
    for comparison_name, comparison in artifact["trace_comparisons"].items():
        for side in ("a", "b"):
            trace = comparison[f"{side}_trace"]["all"]
            lines.append(f"| {comparison_name} | {side} `{comparison[side]}` | {trace['gripper_total_variation']:.3f} | {trace['gripper_mean_abs_step_change']:.3f} | {trace['gripper_sign_changes_ignoring_zero_hold']:.2f} | {trace['arm_mean_l2_step_change']:.3f} | {trace['arm_total_l2_variation']:.3f} |")
    lines.extend(["", "The JSON artifact contains per-configuration Wilson intervals, exact success vectors, paired diagnostics, Pareto dominance, and full trace aggregates."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    old_segments = load_segments(args.old_runs_root, set(OLD_STATE_IDS))
    extension_segments = load_segments(args.extension_runs_root, set(EXTENSION_STATE_IDS))
    combined, source_info = validate_and_combine(old_segments, extension_segments)
    old_entries_by_key = {
        key: summarize_episodes(
            sorted(episodes, key=lambda episode: int(episode["init_state_id"])),
            key,
            name=(f"global_h{key[1]}" if key[0] == "global_fixed" else f"group_arm{key[1]}_grip{key[2]}"),
        )
        for key, episodes in old_segments.items()
    }
    entries_by_key = {
        key: summarize_episodes(episodes, key, name=(f"global_h{key[1]}" if key[0] == "global_fixed" else f"group_arm{key[1]}_grip{key[2]}"))
        for key, episodes in combined.items()
    }
    entries = list(entries_by_key.values())
    global_entries = [entry for entry in entries if entry["strategy"] == "global_fixed"]
    group_entries = [entry for entry in entries if entry["strategy"] == "groupwise_fixed"]
    off_diagonal_entries = [entry for entry in group_entries if entry["arm_horizon"] != entry["gripper_horizon"]]
    best_global = [entry for entry in global_entries if entry["success_rate"] == max(e["success_rate"] for e in global_entries)]
    best_groupwise = [entry for entry in group_entries if entry["success_rate"] == max(e["success_rate"] for e in group_entries)]
    best_off_diagonal = [entry for entry in off_diagonal_entries if entry["success_rate"] == max(e["success_rate"] for e in off_diagonal_entries)]

    best_global_entry = best_global[0]
    best_off_entry = best_off_diagonal[0]
    closest_global = min(global_entries, key=lambda entry: abs(entry["mean_policy_query_rate"] - best_off_entry["mean_policy_query_rate"]))
    key_comparisons = {
        "best_global_vs_best_off_diagonal": paired_comparison(best_global_entry, best_off_entry),
        "closest_query_rate_global_vs_best_off_diagonal": paired_comparison(closest_global, best_off_entry),
    }
    budget_matched = []
    for group_entry in sorted(off_diagonal_entries, key=lambda entry: (entry["arm_horizon"], entry["gripper_horizon"])):
        matched = min(global_entries, key=lambda entry: abs(entry["mean_policy_query_rate"] - group_entry["mean_policy_query_rate"]))
        comparison = paired_comparison(matched, group_entry)
        comparison["relative_query_rate_difference"] = abs(matched["mean_policy_query_rate"] - group_entry["mean_policy_query_rate"]) / matched["mean_policy_query_rate"]
        budget_matched.append(comparison)

    direction_pairs = ((1, 2), (1, 4), (1, 8), (1, 16), (2, 4), (2, 8), (2, 16), (4, 8), (4, 16), (8, 16))
    directionality = []
    for left, right in direction_pairs:
        a = entries_by_key[("groupwise_fixed", left, right)]
        b = entries_by_key[("groupwise_fixed", right, left)]
        directionality.append(paired_comparison(a, b))

    old_globals = [entry for entry in old_entries_by_key.values() if entry["strategy"] == "global_fixed"]
    old_groups = [entry for entry in old_entries_by_key.values() if entry["strategy"] == "groupwise_fixed"]
    old_off_diagonal = [entry for entry in old_groups if entry["arm_horizon"] != entry["gripper_horizon"]]
    old_best_global = max(old_globals, key=lambda entry: entry["success_rate"])
    old_best_off_diagonal = max(old_off_diagonal, key=lambda entry: entry["success_rate"])
    old_closest_global = min(old_globals, key=lambda entry: abs(entry["mean_policy_query_rate"] - old_best_off_diagonal["mean_policy_query_rate"]))
    old_directionality = []
    for left, right in direction_pairs:
        old_directionality.append(
            paired_comparison(
                old_entries_by_key[("groupwise_fixed", left, right)],
                old_entries_by_key[("groupwise_fixed", right, left)],
                OLD_STATE_IDS,
            )
        )
    old_matrices = {
        "success_rate": matrix(old_entries_by_key, "success_rate"),
        "mean_success_steps": matrix(old_entries_by_key, "mean_success_steps"),
        "mean_policy_query_rate": matrix(old_entries_by_key, "mean_policy_query_rate"),
    }
    comparison_20_vs_50 = {
        "old_best_global": old_best_global["name"],
        "final_best_global": [entry["name"] for entry in best_global],
        "old_best_off_diagonal": old_best_off_diagonal["name"],
        "final_best_off_diagonal": [entry["name"] for entry in best_off_diagonal],
        "old_best_global_success_rate": old_best_global["success_rate"],
        "final_best_global_success_rate": [entry["success_rate"] for entry in best_global],
        "old_best_off_diagonal_success_rate": old_best_off_diagonal["success_rate"],
        "final_best_off_diagonal_success_rate": [entry["success_rate"] for entry in best_off_diagonal],
        "old_budget_matched_best": paired_comparison(old_closest_global, old_best_off_diagonal, OLD_STATE_IDS),
        "final_budget_matched_best": key_comparisons["closest_query_rate_global_vs_best_off_diagonal"],
        "old_pareto_frontier": [entry["name"] for entry in pareto_analysis(old_globals + old_groups)["frontier"]],
        "final_pareto_frontier": [entry["name"] for entry in pareto_analysis(entries)["frontier"]],
        "old_success_matrix": old_matrices["success_rate"],
        "final_success_matrix": None,
        "directionality": [
            {
                "a": old_result["a"],
                "b": old_result["b"],
                "old_success_rates": [old_result["a_success_rate"], old_result["b_success_rate"]],
                "final_success_rates": [new_result["a_success_rate"], new_result["b_success_rate"]],
                "old_exact_p": old_result["exact_two_sided_mcnemar_p"],
                "final_exact_p": new_result["exact_two_sided_mcnemar_p"],
            }
            for old_result, new_result in zip(old_directionality, directionality)
        ],
    }

    diagonal = diagonal_checks(entries_by_key)
    matrices = {
        "success_rate": matrix(entries_by_key, "success_rate"),
        "mean_success_steps": matrix(entries_by_key, "mean_success_steps"),
        "mean_policy_query_rate": matrix(entries_by_key, "mean_policy_query_rate"),
    }
    comparison_20_vs_50["final_success_matrix"] = matrices["success_rate"]
    trace_aggregates = {
        entry["name"]: aggregate_trace_metrics(entry["episodes_data"])
        for entry in entries
    }
    trace_episode_metrics = {
        entry["name"]: [
            {
                "init_state_id": int(episode["init_state_id"]),
                "success": bool(episode["success"]),
                **trace_metrics(episode["steps"]),
            }
            for episode in entry["episodes_data"]
        ]
        for entry in entries
    }
    trace_comparisons = {
        "best_global_vs_best_off_diagonal": {
            "a": best_global_entry["name"],
            "b": best_off_entry["name"],
            "a_trace": trace_aggregates[best_global_entry["name"]],
            "b_trace": trace_aggregates[best_off_entry["name"]],
        },
    }
    reversed_key = ("groupwise_fixed", best_off_entry["gripper_horizon"], best_off_entry["arm_horizon"])
    if reversed_key in entries_by_key:
        reverse = entries_by_key[reversed_key]
        trace_comparisons["best_off_diagonal_vs_reversed"] = {
            "a": best_off_entry["name"],
            "b": reverse["name"],
            "a_trace": trace_aggregates[best_off_entry["name"]],
            "b_trace": trace_aggregates[reverse["name"]],
            "paired_success": paired_comparison(best_off_entry, reverse),
        }
    trace_comparisons["closest_query_rate_global_vs_best_off_diagonal"] = {
        "a": closest_global["name"],
        "b": best_off_entry["name"],
        "a_trace": trace_aggregates[closest_global["name"]],
        "b_trace": trace_aggregates[best_off_entry["name"]],
        "paired_success": key_comparisons["closest_query_rate_global_vs_best_off_diagonal"],
    }

    artifact: dict[str, Any] = {
        "horizons": list(HORIZONS),
        "sources": source_info,
        "pairing_valid": True,
        "configurations": sorted([public_entry(entry) for entry in entries], key=lambda entry: (entry["strategy"], entry["arm_horizon"], entry["gripper_horizon"])),
        "best_global": [public_entry(entry) for entry in best_global],
        "best_groupwise": [public_entry(entry) for entry in best_groupwise],
        "best_off_diagonal": [public_entry(entry) for entry in best_off_diagonal],
        "diagonal_checks": diagonal,
        "matrices": matrices,
        "key_paired_comparisons": key_comparisons,
        "budget_matched_comparisons": budget_matched,
        "pareto": pareto_analysis(entries),
        "directionality": directionality,
        "comparison_20_vs_50": comparison_20_vs_50,
        "trace_semantics": {
            "source": "robosuite.models.grippers.panda_gripper.PandaGripper.format_action",
            "negative_command": "open",
            "positive_command": "closed",
            "zero_command": "holds current gripper action because sign(0)=0",
        },
        "trace_aggregates": trace_aggregates,
        "trace_episode_metrics": trace_episode_metrics,
        "trace_comparisons": trace_comparisons,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_markdown, artifact, entries, matrices)
    save_heatmap(matrices["success_rate"], "LIBERO static group-wise success rate (50 states)", "success rate", args.figure_dir / "libero_static_success_heatmap_50.png", cmap="viridis", vmin=0, vmax=1)
    save_heatmap(matrices["mean_success_steps"], "LIBERO static group-wise mean successful steps (50 states)", "mean successful steps", args.figure_dir / "libero_static_success_steps_heatmap_50.png", cmap="magma")
    save_heatmap(matrices["mean_policy_query_rate"], "LIBERO static group-wise mean query rate (50 states)", "policy queries / environment step", args.figure_dir / "libero_static_query_rate_heatmap_50.png", cmap="plasma")
    make_scatter(entries, args.figure_dir / "libero_static_success_vs_query_50.png")
    print(json.dumps({"old_episodes": source_info["old_episodes"], "extension_episodes": source_info["extension_episodes"], "final_episodes": source_info["final_episodes"], "best_global": [entry["name"] for entry in best_global], "best_off_diagonal": [entry["name"] for entry in best_off_diagonal], "output_json": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    main()
