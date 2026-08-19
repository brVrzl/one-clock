#!/usr/bin/env python3
"""Aggregate the paired coarse LIBERO Object cross-task diagnostic."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
from typing import Any

import numpy as np


TASKS = {
    1: "pick_up_the_cream_cheese_and_place_it_in_the_basket",
    2: "pick_up_the_salad_dressing_and_place_it_in_the_basket",
    3: "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
    4: "pick_up_the_ketchup_and_place_it_in_the_basket",
    5: "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
    6: "pick_up_the_butter_and_place_it_in_the_basket",
    7: "pick_up_the_milk_and_place_it_in_the_basket",
    8: "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
    9: "pick_up_the_orange_juice_and_place_it_in_the_basket",
}
HORIZONS = (2, 4, 8, 16)
STATE_IDS = tuple(range(20))
Z95 = 1.959963984540054


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=root / "experiments/runs/libero_object_cross_task",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "experiments/libero_object_cross_task",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=root / "experiments/libero_object_cross_task_summary.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=root / "experiments/libero_object_cross_task_summary.md",
    )
    return parser.parse_args()


def configuration_key(metadata: dict[str, Any]) -> tuple[str, int, int]:
    if metadata["strategy"] == "global_fixed":
        horizon = int(metadata["global_horizon"])
        return "global_fixed", horizon, horizon
    return (
        "groupwise_fixed",
        int(metadata["group_horizons"]["arm"]),
        int(metadata["group_horizons"]["gripper"]),
    )


def mean(values: list[float | int]) -> float:
    return float(sum(values) / len(values))


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


def summarize(
    *,
    name: str,
    strategy: str,
    arm_horizon: int,
    gripper_horizon: int,
    episodes: list[dict[str, Any]],
    run_dir: str,
    source_alias: str | None = None,
) -> dict[str, Any]:
    successful = [episode for episode in episodes if bool(episode["success"])]
    environment_steps = [int(episode["environment_steps"]) for episode in episodes]
    policy_queries = [int(episode["policy_queries"]) for episode in episodes]
    query_rates = [float(episode["policy_query_rate"]) for episode in episodes]
    success_steps = [int(episode["environment_steps"]) for episode in successful]
    return {
        "name": name,
        "strategy": strategy,
        "arm_horizon": arm_horizon,
        "gripper_horizon": gripper_horizon,
        "episodes": len(episodes),
        "successes": len(successful),
        "success_rate": len(successful) / len(episodes),
        "success_rate_ci95": wilson_interval(len(successful), len(episodes)),
        "mean_environment_steps": mean(environment_steps),
        "median_environment_steps": float(statistics.median(environment_steps)),
        "mean_success_steps": mean(success_steps) if success_steps else None,
        "median_success_steps": float(statistics.median(success_steps)) if success_steps else None,
        "mean_policy_queries": mean(policy_queries),
        "mean_policy_query_rate": mean(query_rates),
        "policy_query_rate": sum(policy_queries) / sum(environment_steps),
        "mean_source_age_arm": mean([float(e["mean_source_age_arm"]) for e in episodes]),
        "mean_source_age_gripper": mean([float(e["mean_source_age_gripper"]) for e in episodes]),
        "successful_init_state_ids": [int(e["init_state_id"]) for e in successful],
        "failed_init_state_ids": [int(e["init_state_id"]) for e in episodes if not bool(e["success"])],
        "success_vector": [bool(e["success"]) for e in episodes],
        "run_dir": run_dir,
        "source_alias": source_alias,
        "episodes_data": episodes,
    }


def public_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in entry.items() if key != "episodes_data"}


def load_task(task_id: int, runs_root: Path) -> tuple[dict[tuple[str, int, int], dict[str, Any]], dict[str, Any]]:
    task_root = runs_root / f"task_{task_id}"
    runs: dict[tuple[str, int, int], dict[str, Any]] = {}
    for run_dir in sorted(path for path in task_root.iterdir() if path.is_dir()):
        metadata_path = run_dir / "metadata.json"
        episodes_path = run_dir / "episodes.jsonl"
        if not metadata_path.is_file() or not episodes_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata["task_id"]) != task_id:
            raise ValueError(f"{run_dir} has task_id {metadata['task_id']}")
        if tuple(metadata.get("observed_chunk_shape", ())) != (100, 7):
            raise ValueError(f"{run_dir} does not record ACT chunk shape (100, 7)")
        episodes = [
            json.loads(line)
            for line in episodes_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        ids = [int(e["init_state_id"]) for e in episodes]
        if ids != list(STATE_IDS):
            raise ValueError(f"{run_dir} has init states {ids}")
        if [int(e["seed"]) for e in episodes] != [1000 + i for i in STATE_IDS]:
            raise ValueError(f"{run_dir} has an invalid seed mapping")
        key = configuration_key(metadata)
        if key in runs:
            raise ValueError(f"duplicate task {task_id} configuration {key}")
        runs[key] = {
            "metadata": metadata,
            "episodes": episodes,
            "run_dir": str(run_dir),
        }

    expected = {
        ("global_fixed", 4, 4),
        ("groupwise_fixed", 2, 2), ("groupwise_fixed", 2, 8),
        ("groupwise_fixed", 2, 16), ("groupwise_fixed", 8, 2),
        ("groupwise_fixed", 8, 8), ("groupwise_fixed", 8, 16),
        ("groupwise_fixed", 16, 2), ("groupwise_fixed", 16, 8),
        ("groupwise_fixed", 16, 16), ("groupwise_fixed", 4, 16),
        ("groupwise_fixed", 16, 4),
    }
    allowed = (expected, expected | {("groupwise_fixed", 4, 4)})
    if set(runs) not in allowed:
        raise ValueError(
            f"task {task_id} configurations differ: missing={sorted(expected - set(runs))}, "
            f"extra={sorted(set(runs) - expected)}"
        )

    # All raw configurations must describe the same official initial states.
    reference_observations: dict[int, dict[str, Any]] = {}
    observation_mismatches: list[str] = []
    for key, run in sorted(runs.items()):
        for episode in run["episodes"]:
            state_id = int(episode["init_state_id"])
            observation = {
                "task_name": episode["task_name"],
                "task_description": episode["task_description"],
                "initial_eef_pos": episode["initial_eef_pos"],
                "initial_image_means": episode["initial_image_means"],
            }
            if state_id not in reference_observations:
                reference_observations[state_id] = observation
            elif reference_observations[state_id] != observation:
                observation_mismatches.append(f"{key}:{state_id}")
    if observation_mismatches:
        raise ValueError(f"task {task_id} initial observations differ: {observation_mismatches}")

    entries: dict[tuple[str, int, int], dict[str, Any]] = {}
    for key, run in runs.items():
        strategy, arm, gripper = key
        entries[key] = summarize(
            name=(f"global_h{arm}" if strategy == "global_fixed" else f"group_arm{arm}_grip{gripper}"),
            strategy=strategy,
            arm_horizon=arm,
            gripper_horizon=gripper,
            episodes=run["episodes"],
            run_dir=run["run_dir"],
        )

    if ("groupwise_fixed", 4, 4) not in entries:
        global_entry = entries[("global_fixed", 4, 4)]
        alias = copy.deepcopy(global_entry)
        alias.update(
            {
                "name": "group_arm4_grip4",
                "strategy": "groupwise_fixed",
                "source_alias": "global_h4",
            }
        )
        entries[("groupwise_fixed", 4, 4)] = alias
        diagonal_control = {
            "status": "global_h4_alias_used",
            "all_equal": True,
            "differences_by_field": {},
        }
    else:
        global_episodes = entries[("global_fixed", 4, 4)]["episodes_data"]
        group_episodes = entries[("groupwise_fixed", 4, 4)]["episodes_data"]
        fields = ("success", "environment_steps", "policy_queries", "policy_query_rate")
        differences = {
            field: [
                int(global_episode["init_state_id"])
                for global_episode, group_episode in zip(global_episodes, group_episodes)
                if global_episode[field] != group_episode[field]
            ]
            for field in fields
        }
        diagonal_control = {
            "status": "raw_global_h4_vs_group_arm4_grip4",
            "all_equal": not any(differences.values()),
            "differences_by_field": differences,
        }

    sanity_path = runs_root / "sanity" / f"task_{task_id}_global_h8" / "summary.json"
    sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
    sanity_episodes = [
        json.loads(line)
        for line in (sanity_path.parent / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    sanity_success_steps = [
        int(episode["environment_steps"])
        for episode in sanity_episodes
        if bool(episode["success"])
    ]
    pairing = {
        "ok": not observation_mismatches,
        "init_state_ids": list(STATE_IDS),
        "base_seed": 1000,
        "official_init_state_count": int(next(iter(runs.values()))["metadata"]["official_init_state_count"]),
        "initial_observation_mismatches": observation_mismatches,
    }
    return entries, {
        "task_id": task_id,
        "task_name": TASKS[task_id],
        "episodes_per_configuration": 20,
        "raw_configuration_count": len(runs),
        "pairing": pairing,
        "diagonal_control": diagonal_control,
        "sanity": {
            "strategy": "global_fixed",
            "horizon": 8,
            "successes": int(sanity["successes"]),
            "episodes": int(sanity["episodes"]),
            "success_rate": float(sanity["success_rate"]),
            "mean_success_steps": mean(sanity_success_steps) if sanity_success_steps else None,
        },
    }


def contingency(a: dict[str, Any], b: dict[str, Any]) -> dict[str, int]:
    a_success = {int(e["init_state_id"]): bool(e["success"]) for e in a["episodes_data"]}
    b_success = {int(e["init_state_id"]): bool(e["success"]) for e in b["episodes_data"]}
    counts = {"both_succeed": 0, "a_only_succeeds": 0, "b_only_succeeds": 0, "both_fail": 0}
    for state_id in STATE_IDS:
        if a_success[state_id] and b_success[state_id]:
            counts["both_succeed"] += 1
        elif a_success[state_id]:
            counts["a_only_succeeds"] += 1
        elif b_success[state_id]:
            counts["b_only_succeeds"] += 1
        else:
            counts["both_fail"] += 1
    return counts


def exact_mcnemar_p(counts: dict[str, int]) -> float:
    a_only = counts["a_only_succeeds"]
    b_only = counts["b_only_succeeds"]
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    lower_tail = sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1)) / 2**discordant
    return min(1.0, 2.0 * lower_tail)


def paired_comparison(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    counts = contingency(a, b)
    return {
        "a": a["name"],
        "b": b["name"],
        "a_success_rate": a["success_rate"],
        "b_success_rate": b["success_rate"],
        "a_query_rate": a["policy_query_rate"],
        "b_query_rate": b["policy_query_rate"],
        "counts": counts,
        "exact_two_sided_mcnemar_p": exact_mcnemar_p(counts),
        "b_minus_a_success_difference": b["success_rate"] - a["success_rate"],
    }


def best_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best = max(entry["success_rate"] for entry in entries)
    return [entry for entry in entries if entry["success_rate"] == best]


def global_entries(entries: dict[tuple[str, int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for horizon in HORIZONS:
        if horizon == 4:
            entry = entries[("global_fixed", 4, 4)]
        else:
            source = entries[("groupwise_fixed", horizon, horizon)]
            entry = copy.deepcopy(source)
            entry.update(
                {
                    "name": f"global_h{horizon}",
                    "strategy": "global_fixed",
                    "arm_horizon": horizon,
                    "gripper_horizon": horizon,
                    "source_alias": source["name"],
                }
            )
        result.append(entry)
    return result


def pareto(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    for candidate in entries:
        dominated = any(
            other["policy_query_rate"] <= candidate["policy_query_rate"]
            and other["success_rate"] >= candidate["success_rate"]
            and (
                other["policy_query_rate"] < candidate["policy_query_rate"]
                or other["success_rate"] > candidate["success_rate"]
            )
            for other in entries
            if other["name"] != candidate["name"]
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda e: (e["policy_query_rate"], -e["success_rate"], e["name"]))


def directionality(entries: dict[tuple[str, int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = ((2, 8), (2, 16), (8, 16), (4, 16))
    result = []
    for arm, gripper in pairs:
        left = entries[("groupwise_fixed", arm, gripper)]
        right = entries[("groupwise_fixed", gripper, arm)]
        comparison = paired_comparison(left, right)
        comparison.update(
            {
                "left": left["name"],
                "right": right["name"],
                "left_arm": arm,
                "left_gripper": gripper,
                "winner": (
                    "left"
                    if left["success_rate"] > right["success_rate"]
                    else "right"
                    if right["success_rate"] > left["success_rate"]
                    else "tie"
                ),
            }
        )
        result.append(comparison)
    return result


def task_artifact(entries: dict[tuple[str, int, int], dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    groups = [entry for key, entry in entries.items() if key[0] == "groupwise_fixed"]
    offdiag = [entry for entry in groups if entry["arm_horizon"] != entry["gripper_horizon"]]
    globals_ = global_entries(entries)
    best_global = best_entries(globals_)
    best_group = best_entries(groups)
    best_offdiag = best_entries(offdiag)
    frontier = pareto(globals_ + [entry for entry in groups if entry["arm_horizon"] != entry["gripper_horizon"]])
    best_pairings = [
        paired_comparison(global_entry, offdiag_entry)
        for global_entry in best_global
        for offdiag_entry in best_offdiag
    ]
    best_offdiag_entry = best_offdiag[0]
    closest_global = min(
        globals_,
        key=lambda entry: (abs(entry["policy_query_rate"] - best_offdiag_entry["policy_query_rate"]), -entry["success_rate"], entry["arm_horizon"]),
    )
    budget_comparison = paired_comparison(closest_global, best_offdiag_entry)
    group_class = {"diagonal" if e["arm_horizon"] == e["gripper_horizon"] else "off-diagonal" for e in best_group}
    if len(group_class) > 1:
        winner_class = "tied"
    else:
        winner_class = next(iter(group_class))
    public_configurations = [public_entry(entry) for entry in sorted(entries.values(), key=lambda e: (e["strategy"], e["arm_horizon"], e["gripper_horizon"]))]
    artifact = {
        **metadata,
        "best_global": [public_entry(e) for e in best_global],
        "best_groupwise": [public_entry(e) for e in best_group],
        "best_off_diagonal": [public_entry(e) for e in best_offdiag],
        "best_global_vs_best_off_diagonal": best_pairings,
        "best_configuration_class": winner_class,
        "global_candidates": [public_entry(e) for e in globals_],
        "configurations": public_configurations,
        "pareto_frontier": [public_entry(e) for e in frontier],
        "off_diagonal_on_pareto_frontier": [e["name"] for e in frontier if e["arm_horizon"] != e["gripper_horizon"]],
        "budget_matched_best_off_diagonal": budget_comparison,
        "directionality": directionality(entries),
    }
    return artifact


def format_value(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def write_task_markdown(path: Path, artifact: dict[str, Any]) -> None:
    lines = [
        f"# LIBERO Object task {artifact['task_id']}: {artifact['task_name']}",
        "",
        f"Paired states: `{artifact['pairing']['init_state_ids']}`; configurations: {artifact['raw_configuration_count']} executed cells plus a diagonal `(4,4)` alias where the duplicate raw run was omitted.",
        f"Standard global `h=8` sanity: {artifact['sanity']['successes']}/{artifact['sanity']['episodes']}; mean successful steps={format_value(artifact['sanity']['mean_success_steps'], 1)}.",
        "",
        "| Configuration | Successes | Rate | 95% CI | Mean success steps | Query rate |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for entry in artifact["configurations"]:
        lines.append(
            f"| {entry['name']} | {entry['successes']} | {format_value(entry['success_rate'])} | [{format_value(entry['success_rate_ci95'][0])}, {format_value(entry['success_rate_ci95'][1])}] | {format_value(entry['mean_success_steps'], 1)} | {format_value(entry['policy_query_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Selected results",
            "",
            f"- Best global: {', '.join(e['name'] for e in artifact['best_global'])}",
            f"- Best group-wise: {', '.join(e['name'] for e in artifact['best_groupwise'])}",
                f"- Best off-diagonal: {', '.join(e['name'] for e in artifact['best_off_diagonal'])}",
            f"- Best-global vs best-off-diagonal paired comparisons: {len(artifact['best_global_vs_best_off_diagonal'])}",
            f"- Best group-wise class: **{artifact['best_configuration_class']}**",
            f"- Off-diagonal Pareto points: {', '.join(artifact['off_diagonal_on_pareto_frontier']) or 'none'}",
            f"- Budget-matched comparison: `{artifact['budget_matched_best_off_diagonal']['a']}` vs `{artifact['budget_matched_best_off_diagonal']['b']}`, counts `{artifact['budget_matched_best_off_diagonal']['counts']}`, exact paired p={format_value(artifact['budget_matched_best_off_diagonal']['exact_two_sided_mcnemar_p'], 4)}.",
            "",
            "Success vectors and full per-configuration records are in the JSON artifact.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# LIBERO Object cross-task coarse horizon diagnostic",
        "",
        "This is a paired, 20-state-per-task diagnostic using the frozen ACT checkpoint. Task 0 was not rerun. Diagonal group-wise entries `(2,2)`, `(8,8)`, and `(16,16)` are the verified global-equivalent controls; `(4,4)` is a raw control on task 1 and a documented alias of global `h=4` elsewhere.",
        "",
        f"Tasks evaluated: **{len(summary['tasks'])}**; controlled episodes: **{summary['total_controlled_episodes']}**; macro best-global success: **{format_value(summary['macro_average']['best_global_success_rate'])}**; macro best-off-diagonal success: **{format_value(summary['macro_average']['best_off_diagonal_success_rate'])}**.",
        "",
        "## Per-task summary",
        "",
        "| ID | Task | Sanity h8 | Mean sanity success steps | Best global | Best off-diagonal | Difference | Offdiag frontier |",
        "|---:|---|---:|---:|---|---|---:|---|",
    ]
    for task in summary["tasks"]:
        best_globals = ", ".join(f"h={horizon}" for horizon in task["best_global_horizons"])
        best_offdiagonals = ", ".join(
            f"({pair[0]},{pair[1]})" for pair in task["best_off_diagonal_pairs"]
        )
        lines.append(
            f"| {task['task_id']} | {task['task_name']} | {task['sanity_successes']}/5 | {format_value(task['sanity_mean_success_steps'], 1)} | {best_globals} ({format_value(task['best_global_success_rate'])}) | {best_offdiagonals} ({format_value(task['best_off_diagonal_success_rate'])}) | {format_value(task['best_off_diagonal_success_rate'] - task['best_global_success_rate'])} | {'yes' if task['off_diagonal_on_pareto_frontier'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Hypothesis diagnostics",
            "",
            f"- Best group-wise class counts: diagonal={summary['hypotheses']['best_configuration_class_counts']['diagonal']}, off-diagonal={summary['hypotheses']['best_configuration_class_counts']['off-diagonal']}, tied={summary['hypotheses']['best_configuration_class_counts']['tied']}",
            f"- Tasks with at least one off-diagonal Pareto point: {summary['hypotheses']['tasks_with_off_diagonal_pareto']} / {len(summary['tasks'])}",
            f"- Tasks with an off-diagonal point strictly improving a global point at no higher query rate: {summary['hypotheses']['tasks_with_off_diagonal_global_improvement']} / {len(summary['tasks'])}",
            "",
            "## Symmetric directionality",
            "",
            "| Task | Pair | Left rate | Right rate | Query rates | Winner | Paired counts | Exact p |",
            "|---:|---|---:|---:|---|---|---|---:|",
        ]
    )
    for row in summary["directionality"]:
        lines.append(
            f"| {row['task_id']} | {row['pair']} | {format_value(row['a_success_rate'])} | {format_value(row['b_success_rate'])} | {format_value(row['a_query_rate'])} / {format_value(row['b_query_rate'])} | {row['winner']} | {row['counts']} | {format_value(row['exact_two_sided_mcnemar_p'], 4)} |"
        )
    lines.extend(
        [
            "",
            "## Macro comparison",
            "",
            f"- Mean per-task best-global success rate: {format_value(summary['macro_average']['best_global_success_rate'])}",
            f"- Mean per-task best-group-wise success rate: {format_value(summary['macro_average']['best_groupwise_success_rate'])}",
            f"- Mean per-task best-off-diagonal success rate: {format_value(summary['macro_average']['best_off_diagonal_success_rate'])}",
            f"- Best-global vs best-off-diagonal mean difference: {format_value(summary['macro_average']['best_off_diagonal_minus_best_global'])}",
            "",
            "Per-task JSON artifacts contain success vectors, Wilson intervals, Pareto frontiers, budget-matched paired comparisons, and all configuration summaries.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    task_metadata: dict[int, dict[str, Any]] = {}
    task_artifacts: dict[int, dict[str, Any]] = {}
    for task_id in TASKS:
        entries, metadata = load_task(task_id, args.runs_root)
        artifact = task_artifact(entries, metadata)
        task_metadata[task_id] = metadata
        task_artifacts[task_id] = artifact
        task_dir = args.output_root / f"task_{task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "result.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        write_task_markdown(task_dir / "result.md", artifact)

    direction_rows = []
    for task_id, artifact in task_artifacts.items():
        for row in artifact["directionality"]:
            direction_rows.append({"task_id": task_id, "task_name": TASKS[task_id], "pair": f"({row['left_arm']},{row['left_gripper']}) vs ({row['left_gripper']},{row['left_arm']})", **row})

    class_counts = {"diagonal": 0, "off-diagonal": 0, "tied": 0}
    task_rows = []
    tasks_with_frontier = 0
    tasks_with_global_improvement = 0
    for task_id, artifact in task_artifacts.items():
        class_counts[artifact["best_configuration_class"]] += 1
        best_global_rate = artifact["best_global"][0]["success_rate"]
        best_offdiag_rate = artifact["best_off_diagonal"][0]["success_rate"]
        offdiag_improves_global = False
        for offdiag in artifact["best_off_diagonal"]:
            for global_entry in artifact["global_candidates"]:
                if (
                    offdiag["policy_query_rate"] <= global_entry["policy_query_rate"]
                    and offdiag["success_rate"] >= global_entry["success_rate"]
                    and (
                        offdiag["policy_query_rate"] < global_entry["policy_query_rate"]
                        or offdiag["success_rate"] > global_entry["success_rate"]
                    )
                ):
                    offdiag_improves_global = True
        if artifact["off_diagonal_on_pareto_frontier"]:
            tasks_with_frontier += 1
        if offdiag_improves_global:
            tasks_with_global_improvement += 1
        task_rows.append(
            {
                "task_id": task_id,
                "task_name": TASKS[task_id],
                "sanity_successes": artifact["sanity"]["successes"],
                "sanity_success_rate": artifact["sanity"]["success_rate"],
                "sanity_mean_success_steps": artifact["sanity"]["mean_success_steps"],
                "best_global_horizons": [e["arm_horizon"] for e in artifact["best_global"]],
                "best_global_success_rate": best_global_rate,
                "best_off_diagonal_pairs": [[e["arm_horizon"], e["gripper_horizon"]] for e in artifact["best_off_diagonal"]],
                "best_off_diagonal_success_rate": best_offdiag_rate,
                "best_groupwise_pairs": [[e["arm_horizon"], e["gripper_horizon"]] for e in artifact["best_groupwise"]],
                "best_global_vs_best_off_diagonal": artifact["best_global_vs_best_off_diagonal"],
                "diagonal_control": artifact["diagonal_control"],
                "best_configuration_class": artifact["best_configuration_class"],
                "off_diagonal_on_pareto_frontier": bool(artifact["off_diagonal_on_pareto_frontier"]),
                "off_diagonal_strictly_improves_global": offdiag_improves_global,
                "budget_matched_best_off_diagonal": artifact["budget_matched_best_off_diagonal"],
            }
        )

    macro = {
        "best_global_success_rate": float(np.mean([row["best_global_success_rate"] for row in task_rows])),
        "best_groupwise_success_rate": float(np.mean([task_artifacts[row["task_id"]]["best_groupwise"][0]["success_rate"] for row in task_rows])),
        "best_off_diagonal_success_rate": float(np.mean([row["best_off_diagonal_success_rate"] for row in task_rows])),
    }
    macro["best_off_diagonal_minus_best_global"] = macro["best_off_diagonal_success_rate"] - macro["best_global_success_rate"]
    direction_winners = {"longer_gripper": 0, "reversed": 0, "tie": 0}
    for row in direction_rows:
        winner = row["winner"]
        direction_winners["longer_gripper" if winner == "left" else "reversed" if winner == "right" else "tie"] += 1
    summary = {
        "suite": "libero_object",
        "checkpoint": "/home/thor/projects/checkpoints/zeromidnight_act_libero_object",
        "task_0_reference": {
            "best_global": "h=8, 45/50",
            "best_off_diagonal": "(4,16), 47/50",
        },
        "horizon_design": {
            "groupwise": [[2, 2], [2, 8], [2, 16], [8, 2], [8, 8], [8, 16], [16, 2], [16, 8], [16, 16]],
            "global": [4],
            "additional_groupwise": [[4, 4], [4, 16], [16, 4]],
            "group44_execution": "raw on task 1; alias of global_h4 on tasks 2-9",
        },
        "states": list(STATE_IDS),
        "total_controlled_episodes": sum(artifact["episodes_per_configuration"] * artifact["raw_configuration_count"] for artifact in task_metadata.values()),
        "tasks": task_rows,
        "directionality": direction_rows,
        "macro_average": macro,
        "hypotheses": {
            "best_configuration_class_counts": class_counts,
            "tasks_with_off_diagonal_pareto": tasks_with_frontier,
            "tasks_with_off_diagonal_global_improvement": tasks_with_global_improvement,
            "tasks_evaluated": len(task_rows),
            "directionality_comparison_counts": direction_winners,
        },
        "runtime_retries": [],
        "raw_runs_root": str(args.runs_root),
        "task_artifact_paths": {str(task_id): str(args.output_root / f"task_{task_id}" / "result.json") for task_id in TASKS},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    write_summary_markdown(args.output_markdown, summary)
    print(json.dumps({"tasks": len(task_rows), "controlled_episodes": summary["total_controlled_episodes"], "output_json": str(args.output_json), "output_markdown": str(args.output_markdown)}, indent=2))


if __name__ == "__main__":
    main()
