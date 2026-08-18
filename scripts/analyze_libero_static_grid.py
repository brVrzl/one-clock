#!/usr/bin/env python3
"""Aggregate the paired LIBERO static-horizon Gate-0 sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np


HORIZONS = (1, 2, 4, 8, 16)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=root / "experiments/runs/libero_static_grid_20",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=root / "experiments/libero_static_grid_20.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=root / "experiments/libero_static_grid_20.md",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=root / "experiments/figures",
    )
    return parser.parse_args()


def load_runs(runs_root: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    runs: dict[tuple[str, int, int], dict[str, Any]] = {}
    for run_dir in sorted(
        path
        for path in runs_root.iterdir()
        if path.is_dir() and (path / "metadata.json").is_file()
    ):
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        episodes = [
            json.loads(line)
            for line in (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        strategy = str(metadata["strategy"])
        if strategy == "global_fixed":
            horizon = int(metadata["global_horizon"])
            arm_horizon = horizon
            gripper_horizon = horizon
        else:
            arm_horizon = int(metadata["group_horizons"]["arm"])
            gripper_horizon = int(metadata["group_horizons"]["gripper"])
        key = (strategy, arm_horizon, gripper_horizon)
        if key in runs:
            raise ValueError(f"duplicate configuration for {key}")
        runs[key] = {
            "name": run_dir.name,
            "run_dir": str(run_dir),
            "metadata": metadata,
            "episodes": episodes,
        }
    return runs


def mean(values: list[float | int]) -> float:
    return float(sum(values) / len(values))


def summarize(run: dict[str, Any], key: tuple[str, int, int]) -> dict[str, Any]:
    strategy, arm_horizon, gripper_horizon = key
    episodes = run["episodes"]
    successful = [episode for episode in episodes if bool(episode["success"])]
    success_steps = [int(episode["environment_steps"]) for episode in successful]
    environment_steps = [int(episode["environment_steps"]) for episode in episodes]
    policy_queries = [int(episode["policy_queries"]) for episode in episodes]
    query_rates = [float(episode["policy_query_rate"]) for episode in episodes]
    arm_ages = [float(episode["mean_source_age_arm"]) for episode in episodes]
    gripper_ages = [float(episode["mean_source_age_gripper"]) for episode in episodes]
    successful_ids = [int(episode["init_state_id"]) for episode in successful]
    failed_ids = [
        int(episode["init_state_id"])
        for episode in episodes
        if not bool(episode["success"])
    ]
    return {
        "name": run["name"],
        "run_dir": run["run_dir"],
        "strategy": strategy,
        "arm_horizon": arm_horizon,
        "gripper_horizon": gripper_horizon,
        "episodes": len(episodes),
        "successes": len(successful),
        "success_rate": len(successful) / len(episodes),
        "environment_steps": sum(environment_steps),
        "policy_queries": sum(policy_queries),
        "policy_query_rate": sum(policy_queries) / sum(environment_steps),
        "mean_environment_steps": mean(environment_steps),
        "median_environment_steps": float(statistics.median(environment_steps)),
        "mean_success_steps": mean(success_steps) if success_steps else None,
        "median_success_steps": float(statistics.median(success_steps)) if success_steps else None,
        "mean_policy_queries": mean(policy_queries),
        "mean_policy_query_rate": mean(query_rates),
        "mean_source_age_arm": mean(arm_ages),
        "mean_source_age_gripper": mean(gripper_ages),
        "successful_init_state_ids": successful_ids,
        "failed_init_state_ids": failed_ids,
        "episodes_data": episodes,
    }


def validate_pairing(runs: dict[tuple[str, int, int], dict[str, Any]]) -> dict[str, Any]:
    issues: list[str] = []
    init_state_sets: list[list[int]] = []
    base_seeds: list[int] = []
    initial_observations: dict[int, dict[str, Any]] = {}
    observation_mismatches: list[dict[str, Any]] = []
    for key, run in sorted(runs.items()):
        metadata = run["metadata"]
        episodes = run["episodes"]
        ids = [int(episode["init_state_id"]) for episode in episodes]
        init_state_sets.append(ids)
        base_seed = int(metadata["base_seed"])
        base_seeds.append(base_seed)
        expected_ids = list(range(int(metadata["init_state_start"]), int(metadata["init_state_start"]) + len(episodes)))
        if ids != expected_ids:
            issues.append(f"{run['name']}: init_state_ids={ids}, expected={expected_ids}")
        seeds = [int(episode["seed"]) for episode in episodes]
        expected_seeds = [base_seed + init_state_id for init_state_id in ids]
        if seeds != expected_seeds:
            issues.append(f"{run['name']}: seeds do not match base_seed + init_state_id")
        for episode in episodes:
            init_state_id = int(episode["init_state_id"])
            observation = {
                "task_name": episode["task_name"],
                "task_description": episode["task_description"],
                "initial_eef_pos": episode["initial_eef_pos"],
                "initial_image_means": episode["initial_image_means"],
            }
            if init_state_id not in initial_observations:
                initial_observations[init_state_id] = observation
            elif observation != initial_observations[init_state_id]:
                observation_mismatches.append(
                    {"configuration": run["name"], "init_state_id": init_state_id}
                )
    if len(set(tuple(ids) for ids in init_state_sets)) != 1:
        issues.append("configurations do not share the same ordered init-state IDs")
    if len(set(base_seeds)) != 1:
        issues.append("configurations do not share one base seed")
    if observation_mismatches:
        issues.append("initial observation provenance differs across configurations")
    return {
        "ok": not issues,
        "official_init_state_count": int(next(iter(runs.values()))["metadata"]["official_init_state_count"]),
        "init_state_ids": init_state_sets[0] if init_state_sets else [],
        "base_seed": base_seeds[0] if base_seeds else None,
        "initial_observation_mismatches": observation_mismatches,
        "issues": issues,
    }


def config_name(strategy: str, arm_horizon: int, gripper_horizon: int) -> str:
    if strategy == "global_fixed":
        return f"global_h{arm_horizon}"
    return f"group_arm{arm_horizon}_grip{gripper_horizon}"


def contingency(global_entry: dict[str, Any], group_entry: dict[str, Any]) -> dict[str, int]:
    global_by_id = {int(episode["init_state_id"]): bool(episode["success"]) for episode in global_entry["episodes_data"]}
    group_by_id = {int(episode["init_state_id"]): bool(episode["success"]) for episode in group_entry["episodes_data"]}
    counts = {"both_succeed": 0, "groupwise_only_succeeds": 0, "global_only_succeeds": 0, "both_fail": 0}
    for init_state_id in sorted(global_by_id):
        global_success = global_by_id[init_state_id]
        group_success = group_by_id[init_state_id]
        if global_success and group_success:
            counts["both_succeed"] += 1
        elif group_success:
            counts["groupwise_only_succeeds"] += 1
        elif global_success:
            counts["global_only_succeeds"] += 1
        else:
            counts["both_fail"] += 1
    return counts


def diagonal_checks(entries: dict[tuple[str, int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    fields = ("success", "environment_steps", "policy_queries", "policy_query_rate")
    for horizon in HORIZONS:
        global_entry = entries[("global_fixed", horizon, horizon)]
        group_entry = entries[("groupwise_fixed", horizon, horizon)]
        differences = {
            field: [
                int(global_episode["init_state_id"])
                for global_episode, group_episode in zip(
                    global_entry["episodes_data"], group_entry["episodes_data"]
                )
                if global_episode[field] != group_episode[field]
            ]
            for field in fields
        }
        checks.append(
            {
                "horizon": horizon,
                "global": global_entry["name"],
                "groupwise": group_entry["name"],
                "all_equal": not any(differences.values()),
                "differences_by_field": differences,
            }
        )
    return checks


def best_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_rate = max(entry["success_rate"] for entry in entries)
    return [entry for entry in entries if entry["success_rate"] == best_rate]


def public_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in entry.items() if key != "episodes_data"}


def matrix(entries: dict[tuple[str, int, int], dict[str, Any]], field: str) -> list[list[float | None]]:
    return [
        [entries[("groupwise_fixed", arm, gripper)][field] for gripper in HORIZONS]
        for arm in HORIZONS
    ]


def format_cell(value: Any, precision: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{precision}f}"


def markdown_matrix(title: str, values: list[list[Any]], precision: int) -> str:
    lines = [f"### {title}", "", "| arm \\ gripper | " + " | ".join(str(horizon) for horizon in HORIZONS) + " |", "|---:|" + "---:|" * len(HORIZONS)]
    for arm, row in zip(HORIZONS, values):
        lines.append("| " + str(arm) + " | " + " | ".join(format_cell(value, precision) for value in row) + " |")
    return "\n".join(lines)


def save_heatmap(
    values: list[list[float | None]],
    title: str,
    colorbar_label: str,
    output_path: Path,
    *,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    import matplotlib.pyplot as plt

    array = np.asarray(
        [[np.nan if value is None else float(value) for value in row] for row in values],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xticks(range(len(HORIZONS)), labels=[str(horizon) for horizon in HORIZONS])
    axis.set_yticks(range(len(HORIZONS)), labels=[str(horizon) for horizon in HORIZONS])
    axis.set_xlabel("gripper horizon")
    axis.set_ylabel("arm horizon")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=colorbar_label)
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                "—" if value is None else format_cell(value),
                ha="center",
                va="center",
                color="black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def write_markdown(
    output_path: Path,
    artifact: dict[str, Any],
    entries: list[dict[str, Any]],
    matrices: dict[str, list[list[Any]]],
) -> None:
    globals_only = [entry for entry in entries if entry["strategy"] == "global_fixed"]
    lines = [
        "# LIBERO static horizon landscape",
        "",
        "This is the complete paired static sweep on the frozen LIBERO ACT checkpoint. It is a diagnostic execution result, not a statistical claim.",
        "",
        f"Pairing valid: **{artifact['pairing']['ok']}**; official states available: **{artifact['pairing']['official_init_state_count']}**; states used: `{artifact['pairing']['init_state_ids']}`; total episodes: **{artifact['total_episodes']}**.",
        "",
        "## Global fixed",
        "",
        "| Global horizon | Successes | Success rate | Mean success steps | Query rate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for entry in sorted(globals_only, key=lambda item: item["arm_horizon"]):
        lines.append(
            f"| {entry['arm_horizon']} | {entry['successes']} | {format_cell(entry['success_rate'])} | {format_cell(entry['mean_success_steps'])} | {format_cell(entry['policy_query_rate'])} |"
        )
    lines.extend(
        [
            "",
            markdown_matrix("Group-wise success rate", matrices["success_rate"], 3),
            "",
            markdown_matrix("Group-wise mean successful completion steps", matrices["mean_success_steps"], 2),
            "",
            markdown_matrix("Group-wise mean policy query rate", matrices["mean_policy_query_rate"], 3),
            "",
            "## Best configurations",
            "",
        ]
    )
    for label in ("best_global", "best_groupwise", "best_off_diagonal"):
        lines.append(f"- **{label}:** " + ", ".join(entry["name"] for entry in artifact[label]))
    lines.extend(["", "## Diagonal controls", ""])
    for check in artifact["diagonal_checks"]:
        lines.append(
            f"- `{check['global']} vs {check['groupwise']}`: all compared per-state fields equal = **{check['all_equal']}**. Differences: `{check['differences_by_field']}`"
        )
    lines.extend(["", "## Best-global vs best-off-diagonal paired contingencies", ""])
    for comparison in artifact["paired_best"]:
        lines.append(
            f"- `{comparison['global']}` vs `{comparison['groupwise']}`: `{comparison['counts']}`"
        )
    lines.extend(["", "## Budget-controlled best off-diagonal diagnostic", ""])
    for comparison in artifact["budget_controlled_best_off_diagonal"]:
        lines.append(
            f"- `{comparison['global']}` ({comparison['global_successes']}/20, query rate {comparison['global_policy_query_rate']:.3f}) vs `{comparison['groupwise']}` ({comparison['groupwise_successes']}/20, query rate {comparison['groupwise_policy_query_rate']:.3f}): `{comparison['counts']}`"
        )
    lines.extend(["", "## Directional paired diagnostics", ""])
    for comparison in artifact["directional_pairs"]:
        lines.append(
            f"- `{comparison['left']}` ({comparison['left_successes']}/20) vs `{comparison['right']}` ({comparison['right_successes']}/20): `{comparison['counts']}`"
        )
    lines.extend(["", "## Configuration details", ""])
    lines.append("The JSON artifact contains environment-step means/medians, successful-step means/medians, query budgets, source ages, and success/failure state IDs for every cell.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs = load_runs(args.runs_root)
    expected_keys = {
        ("global_fixed", horizon, horizon) for horizon in HORIZONS
    } | {
        ("groupwise_fixed", arm, gripper)
        for arm in HORIZONS
        for gripper in HORIZONS
    }
    if set(runs) != expected_keys:
        missing = sorted(expected_keys - set(runs))
        extra = sorted(set(runs) - expected_keys)
        raise ValueError(f"sweep configurations do not match 5 + 25 design; missing={missing}, extra={extra}")

    summaries_by_key = {
        key: summarize(run, key)
        for key, run in runs.items()
    }
    entries = list(summaries_by_key.values())
    pairing = validate_pairing(runs)
    total_episodes = sum(entry["episodes"] for entry in entries)
    group_entries = [entry for entry in entries if entry["strategy"] == "groupwise_fixed"]
    global_entries = [entry for entry in entries if entry["strategy"] == "global_fixed"]
    off_diagonal_entries = [entry for entry in group_entries if entry["arm_horizon"] != entry["gripper_horizon"]]
    best_global = best_entries(global_entries)
    best_groupwise = best_entries(group_entries)
    best_off_diagonal = best_entries(off_diagonal_entries)
    artifact: dict[str, Any] = {
        "horizons": list(HORIZONS),
        "total_configurations": len(entries),
        "total_episodes": total_episodes,
        "pairing": pairing,
        "configurations": sorted(
            [{key: value for key, value in entry.items() if key != "episodes_data"} for entry in entries],
            key=lambda entry: (entry["strategy"], entry["arm_horizon"], entry["gripper_horizon"]),
        ),
        "best_global": [public_entry(entry) for entry in best_global],
        "best_groupwise": [public_entry(entry) for entry in best_groupwise],
        "best_off_diagonal": [public_entry(entry) for entry in best_off_diagonal],
        "diagonal_checks": diagonal_checks(summaries_by_key),
        "paired_best": [],
        "budget_controlled_best_off_diagonal": [],
        "directional_pairs": [],
    }
    for global_entry in best_global:
        for group_entry in best_off_diagonal:
            artifact["paired_best"].append(
                {
                    "global": global_entry["name"],
                    "groupwise": group_entry["name"],
                    "counts": contingency(global_entry, group_entry),
                }
            )
    for group_entry in best_off_diagonal:
        matched_horizon = min(group_entry["arm_horizon"], group_entry["gripper_horizon"])
        global_entry = summaries_by_key[("global_fixed", matched_horizon, matched_horizon)]
        artifact["budget_controlled_best_off_diagonal"].append(
            {
                "global": global_entry["name"],
                "groupwise": group_entry["name"],
                "global_successes": global_entry["successes"],
                "groupwise_successes": group_entry["successes"],
                "global_policy_query_rate": global_entry["policy_query_rate"],
                "groupwise_policy_query_rate": group_entry["policy_query_rate"],
                "counts": contingency(global_entry, group_entry),
            }
        )
    for arm, gripper, other_arm, other_gripper in ((8, 2, 2, 8), (16, 2, 2, 16), (8, 4, 4, 8)):
        left = summaries_by_key[("groupwise_fixed", arm, gripper)]
        right = summaries_by_key[("groupwise_fixed", other_arm, other_gripper)]
        artifact["directional_pairs"].append(
                {
                    "left": left["name"],
                    "right": right["name"],
                    "left_successes": left["successes"],
                    "right_successes": right["successes"],
                    "left_policy_query_rate": left["policy_query_rate"],
                    "right_policy_query_rate": right["policy_query_rate"],
                    "counts": contingency(left, right),
                }
        )

    matrices = {
        "success_rate": matrix(summaries_by_key, "success_rate"),
        "mean_success_steps": matrix(summaries_by_key, "mean_success_steps"),
        "mean_policy_query_rate": matrix(summaries_by_key, "mean_policy_query_rate"),
    }
    artifact["matrices"] = matrices
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_markdown, artifact, entries, matrices)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    save_heatmap(
        matrices["success_rate"],
        "LIBERO static group-wise success rate",
        "success rate",
        args.figure_dir / "libero_static_success_heatmap.png",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    save_heatmap(
        matrices["mean_success_steps"],
        "LIBERO static group-wise mean successful steps",
        "mean successful steps",
        args.figure_dir / "libero_static_success_steps_heatmap.png",
        cmap="magma",
    )
    save_heatmap(
        matrices["mean_policy_query_rate"],
        "LIBERO static group-wise mean policy query rate",
        "policy queries / environment step",
        args.figure_dir / "libero_static_query_rate_heatmap.png",
        cmap="plasma",
    )
    print(json.dumps({"total_episodes": total_episodes, "pairing_ok": pairing["ok"], "output_json": str(args.output_json), "output_markdown": str(args.output_markdown)}, indent=2))


if __name__ == "__main__":
    main()
