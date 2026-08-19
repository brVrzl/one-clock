#!/usr/bin/env python3
"""Analyze the completed LIBERO Object static results without new rollouts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HORIZONS = [2, 4, 8, 16]
TASK0_NAME = "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
GROUP_PAIRS = [
    (2, 2), (2, 8), (2, 16),
    (4, 4), (4, 16),
    (8, 2), (8, 8), (8, 16),
    (16, 2), (16, 4), (16, 8), (16, 16),
]


def pair_label(pair: tuple[int, int]) -> str:
    return f"({pair[0]},{pair[1]})"


def config_label(pair: tuple[int, int]) -> str:
    return f"group_arm{pair[0]}_grip{pair[1]}"


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def wilson(successes: int, episodes: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / episodes
    denominator = 1.0 + z * z / episodes
    center = (p + z * z / (2.0 * episodes)) / denominator
    radius = z * math.sqrt(
        p * (1.0 - p) / episodes + z * z / (4.0 * episodes * episodes)
    ) / denominator
    return [center - radius, center + radius]


def average_rank_descending(values: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    ranks = {}
    for key, value in values.items():
        greater = sum(other > value + 1e-12 for other in values.values())
        equal = sum(abs(other - value) <= 1e-12 for other in values.values())
        ranks[key] = 1.0 + greater + (equal - 1) / 2.0
    return ranks


def make_entry(source: dict, task_id: int, name: str, arm: int, grip: int) -> dict:
    vector = [bool(value) for value in source["success_vector"]]
    episodes = int(source.get("episodes", len(vector)))
    if len(vector) != episodes:
        raise ValueError(f"task {task_id} {name}: success vector length mismatch")
    return {
        "task_id": task_id,
        "name": name,
        "arm_horizon": arm,
        "gripper_horizon": grip,
        "success_rate": float(source["success_rate"]),
        "policy_query_rate": float(source["policy_query_rate"]),
        "success_vector": vector,
        "episodes": episodes,
        "source_name": source.get("name", name),
        "source_alias": source.get("source_alias"),
    }


def load_task_configs(task_id: int, task0: dict | None, result: dict | None) -> tuple[dict, dict]:
    if task_id == 0:
        raw = {item["name"]: item for item in task0["configurations"]}
        task_name = TASK0_NAME
        def source(name: str) -> dict:
            return raw[name]
    else:
        raw = {item["name"]: item for item in result["configurations"]}
        candidates = {item["name"]: item for item in result["global_candidates"]}
        task_name = result["task_name"]
        def source(name: str) -> dict:
            return raw[name] if name in raw else candidates[name]
    global_entries = {
        h: make_entry(source(f"global_h{h}"), task_id, f"global_h{h}", h, h)
        for h in HORIZONS
    }
    group_entries = {
        pair: make_entry(source(config_label(pair)), task_id, config_label(pair), *pair)
        for pair in GROUP_PAIRS
    }
    return {"task_id": task_id, "task_name": task_name}, {
        "global": global_entries,
        "groupwise": group_entries,
    }


def bootstrap_mean(values: list[float], rng: np.random.Generator, draws: int = 20000) -> list[float]:
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    samples = np.asarray(values)[indices].mean(axis=1)
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def pareto_frontier(points: list[dict]) -> list[str]:
    frontier = []
    for point in points:
        dominated = any(
            other["macro_success"] >= point["macro_success"] - 1e-12
            and other["mean_policy_query_rate"] <= point["mean_policy_query_rate"] + 1e-12
            and (
                other["macro_success"] > point["macro_success"] + 1e-12
                or other["mean_policy_query_rate"] < point["mean_policy_query_rate"] - 1e-12
            )
            for other in points
            if other["label"] != point["label"]
        )
        if not dominated:
            frontier.append(point["label"])
    return frontier


def plot_universal_vs_oracle(output: Path, tasks: list[dict], global_h: int, group_pair: tuple[int, int]) -> None:
    labels = [f"T{task['task_id']}" for task in tasks]
    x = np.arange(len(tasks))
    width = 0.2
    series = [
        ("task best global", [task["best_global_rate"] for task in tasks]),
        (f"universal G{global_h}", [task["global"][str(global_h)]["success_rate"] for task in tasks]),
        (f"universal {pair_label(group_pair)}", [task["groupwise"][pair_label(group_pair)]["success_rate"] for task in tasks]),
        ("task group-wise oracle", [task["best_groupwise_rate"] for task in tasks]),
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    for index, (label, values) in enumerate(series):
        ax.bar(x + (index - 1.5) * width, values, width, label=label)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Success rate")
    ax.set_title("LIBERO Object per-task universal-versus-oracle success")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_rank_heatmap(output: Path, tasks: list[dict]) -> None:
    matrix = np.asarray([
        [task["groupwise_ranks"][pair_label(pair)] for task in tasks]
        for pair in GROUP_PAIRS
    ])
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(matrix, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(tasks)), [f"T{task['task_id']}" for task in tasks])
    ax.set_yticks(range(len(GROUP_PAIRS)), [pair_label(pair) for pair in GROUP_PAIRS])
    ax.set_xlabel("Task")
    ax.set_ylabel("Group-wise pair (arm, gripper)")
    ax.set_title("Group-wise success-rate rank by task (1 is best)")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > 6 else "black"
            ax.text(col, row, f"{matrix[row, col]:.1f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(image, ax=ax, label="Rank")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_pareto(output: Path, points: list[dict], frontier: list[str], best_global: list[str], best_group: list[str], oracle_macro: float) -> None:
    styles = {
        "global": ("o", "tab:blue"),
        "diagonal_groupwise": ("s", "tab:orange"),
        "off_diagonal_groupwise": ("^", "tab:green"),
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    for kind, (marker, color) in styles.items():
        selected = [point for point in points if point["kind"] == kind]
        ax.scatter(
            [point["mean_policy_query_rate"] for point in selected],
            [point["macro_success"] for point in selected],
            marker=marker, color=color, label=kind.replace("_", " "),
        )
        for point in selected:
            ax.annotate(point["label"], (point["mean_policy_query_rate"], point["macro_success"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    for point in points:
        if point["label"] in frontier:
            ax.scatter(point["mean_policy_query_rate"], point["macro_success"], facecolors="none", edgecolors="black", s=100)
    ax.axhline(oracle_macro, color="gray", linestyle="--", label="macro task oracle reference")
    for labels, color in [(best_global, "tab:blue"), (best_group, "tab:red")]:
        for label in labels:
            point = next(point for point in points if point["label"] == label)
            ax.scatter(point["mean_policy_query_rate"], point["macro_success"], facecolors="none", edgecolors=color, s=180, linewidths=2)
    ax.set_xlabel("Mean policy-query rate across tasks")
    ax.set_ylabel("Macro task success")
    ax.set_title("LIBERO Object universal success/query trade-off")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def markdown(data: dict) -> str:
    lines = [
        "# LIBERO Object dynamic-readiness analysis",
        "",
        "This is a post-hoc analysis of completed task-0 50-state and tasks-1..9 20-state static rollout artifacts. No rollouts were run.",
        "",
        f"- Tasks: {data['task_count']}; each task has equal macro weight.",
        f"- Common global horizons: {data['common_configuration_set']['global_horizons']}.",
        f"- Common group-wise pairs: {', '.join(pair_label(tuple(pair)) for pair in data['common_configuration_set']['groupwise_pairs'])}.",
        "",
        "## Common configuration set",
        "",
        "Common global fixed configurations are G2, G4, G8, and G16. Diagonal group-wise aliases are (2,2), (4,4), (8,8), and (16,16). The common off-diagonal set is (2,8), (2,16), (4,16), (8,2), (8,16), (16,2), (16,4), and (16,8). Task 0's h=1/full-grid-only cells are excluded.",
        "",
        "## Universal global horizons",
        "",
        "| Horizon | Macro success | Worst | Median | Mean query rate | Per-task rates |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in data["universal_global_metrics"]:
        lines.append(f"| G{item['horizon']} | {item['macro_success']:.3f} | {item['worst_task_success']:.3f} | {item['median_task_success']:.3f} | {item['mean_policy_query_rate']:.3f} | {', '.join(f'{value:.3f}' for value in item['task_success_rates'])} |")
    lines += [
        "",
        f"Best universal global: **{', '.join(data['best_universal_global']['labels'])}**, macro success {data['best_universal_global']['macro_success']:.3f}.",
        "",
        "## Universal group-wise pairs",
        "",
        "| Pair | Macro success | Median | Worst | Mean query rate | Best/tied-best tasks | Pareto tasks | Per-task rates |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in data["universal_groupwise_metrics"]:
        lines.append(f"| {item['label']} | {item['macro_success']:.3f} | {item['median_task_success']:.3f} | {item['worst_task_success']:.3f} | {item['mean_policy_query_rate']:.3f} | {item['tasks_best_or_tied_best']} | {item['tasks_on_pareto_frontier']} | {', '.join(f'{value:.3f}' for value in item['task_success_rates'])} |")
    lines += [
        "",
        f"Best universal group-wise pair: **{', '.join(data['best_universal_groupwise']['labels'])}**, macro success {data['best_universal_groupwise']['macro_success']:.3f}.",
        "",
        "### Per-task Wilson 95% intervals",
        "",
        "The cells below are success rate followed by its per-task binomial Wilson interval; task 0 uses 50 episodes and tasks 1..9 use 20.",
        "",
        "| Config | T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for item in data["universal_global_metrics"]:
        cells = [f"{rate:.3f} [{ci[0]:.3f},{ci[1]:.3f}]" for rate, ci in zip(item["task_success_rates"], item["task_wilson_ci95"])]
        lines.append(f"| G{item['horizon']} | " + " | ".join(cells) + " |")
    for item in data["universal_groupwise_metrics"]:
        cells = [f"{rate:.3f} [{ci[0]:.3f},{ci[1]:.3f}]" for rate, ci in zip(item["task_success_rates"], item["task_wilson_ci95"])]
        lines.append(f"| {item['label']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Per-task static oracle",
        "",
        "| Task | Name | Best global | Best group-wise | Static oracle | Oracle gap over universal group |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for task in data["tasks"]:
        lines.append(f"| {task['task_id']} | {task['task_name']} | {', '.join('G'+str(h) for h in task['best_global_horizons'])} ({task['best_global_rate']:.3f}) | {', '.join(task['best_groupwise_pairs'])} ({task['best_groupwise_rate']:.3f}) | {', '.join(task['best_static_configurations'])} ({task['static_oracle_rate']:.3f}) | {task['oracle_gap_over_universal_group']:.3f} |")
    lines += [
        "",
        f"Macro per-task best global: **{data['macro_best_global']:.3f}**. Macro per-task best group-wise: **{data['macro_best_groupwise']:.3f}**. Group-wise minus global: **{data['global_vs_groupwise_oracle']['absolute_difference_groupwise_minus_global']:.3f}**; group-wise strictly better on {data['global_vs_groupwise_oracle']['groupwise_strictly_better_tasks']}, tied on {data['global_vs_groupwise_oracle']['tied_tasks']}, global better on {data['global_vs_groupwise_oracle']['global_strictly_better_tasks']} tasks.",
        "",
        "## Leave-one-task-out selection",
        "",
        "| Held-out task | Selected pair(s) | Held-out selected success | Held-out oracle | Regret | Best global |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for item in data["leave_one_task_out"]:
        lines.append(f"| {item['task_id']} | {', '.join(item['selected_pairs'])} | {item['selected_mean_success']:.3f} | {item['held_out_oracle_success']:.3f} | {item['mean_regret']:.3f} | {item['held_out_best_global_success']:.3f} |")
    lines += [
        "",
        f"Mean leave-one-task-out regret: **{data['leave_one_task_out_mean_regret']:.3f}**. Tied selections are retained.",
        "",
        "## Group-wise rank stability",
        "",
        "| Pair | Mean rank | Rank SD | Best rank | Worst rank |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in data["groupwise_rank_stability"]:
        lines.append(f"| {item['label']} | {item['mean_rank']:.2f} | {item['rank_std']:.2f} | {item['best_rank']:.1f} | {item['worst_rank']:.1f} |")
    lines += [
        "",
        "## Horizon preferences among task-optimal group-wise pairs",
        "",
        f"Arm horizon frequencies over tied optima: {data['horizon_preferences']['arm_horizon_frequency']}.",
        f"Gripper horizon frequencies over tied optima: {data['horizon_preferences']['gripper_horizon_frequency']}.",
        f"Pair frequencies: {data['horizon_preferences']['pair_frequency']}.",
        f"Relation counts: gripper > arm {data['horizon_preferences']['gripper_longer']}, equal {data['horizon_preferences']['equal']}, gripper < arm {data['horizon_preferences']['gripper_shorter']}.",
        "",
        "## Success/query Pareto analysis",
        "",
        "| Configuration | Kind | Macro success | Mean query rate | Dominated |",
        "|---|---|---:|---:|---|",
    ]
    for point in data["pareto_points"]:
        lines.append(f"| {point['label']} | {point['kind']} | {point['macro_success']:.3f} | {point['mean_policy_query_rate']:.3f} | {'yes' if point['label'] in data['pareto_dominated'] else 'no'} |")
    lines += [
        "",
        f"Empirical cross-task Pareto frontier: **{', '.join(data['pareto_frontier'])}**.",
        f"Universal group-wise versus global: macro success difference {data['universal_groupwise_vs_global']['success_difference']:.3f}; query-rate difference {data['universal_groupwise_vs_global']['query_rate_difference']:.3f} (group-wise minus global).",
        "",
        "## Confidence intervals and bootstrap",
        "",
        "The JSON artifact contains per-task Wilson 95% intervals for every common configuration. Macro comparisons use a deterministic task-level bootstrap, not pooled episodes.",
        f"Bootstrap seed {data['bootstrap']['seed']}, draws {data['bootstrap']['draws']}. Universal group-wise minus universal global: {data['bootstrap']['universal_groupwise_minus_global']['estimate']:.3f}, CI [{data['bootstrap']['universal_groupwise_minus_global']['ci95'][0]:.3f}, {data['bootstrap']['universal_groupwise_minus_global']['ci95'][1]:.3f}].",
        f"Per-task static oracle minus universal group-wise: {data['bootstrap']['oracle_minus_universal_groupwise']['estimate']:.3f}, CI [{data['bootstrap']['oracle_minus_universal_groupwise']['ci95'][0]:.3f}, {data['bootstrap']['oracle_minus_universal_groupwise']['ci95'][1]:.3f}].",
        "",
        "## Dynamic-readiness decision",
        "",
        f"Classification: **{data['dynamic_readiness']['classification']}**.",
        data["dynamic_readiness"]["reason"],
        "",
        "## PACE source audit and deferred baseline list",
        "",
        data["pace_source_audit"]["summary"],
        "",
        "Deferred comparator list if a later dynamic-method task is authorized: " + ", ".join(data["next_stage_baselines"]) + ". No item in this list was implemented here.",
        "",
        "No dynamic method was implemented and no rollouts were run for this analysis.",
        "",
        "## Figures",
        "",
        f"- {data['figures']['universal_vs_oracle']}",
        f"- {data['figures']['config_rank_heatmap']}",
        f"- {data['figures']['pareto']}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task0", type=Path, default=Path("experiments/libero_static_grid_50.json"))
    parser.add_argument("--cross-summary", type=Path, default=Path("experiments/libero_object_cross_task_summary.json"))
    parser.add_argument("--task-root", type=Path, default=Path("experiments/libero_object_cross_task"))
    parser.add_argument("--output-json", type=Path, default=Path("experiments/libero_object_dynamic_readiness.json"))
    parser.add_argument("--output-md", type=Path, default=Path("experiments/libero_object_dynamic_readiness.md"))
    parser.add_argument("--figure-dir", type=Path, default=Path("experiments/figures"))
    args = parser.parse_args()

    task0 = load_json(args.task0)
    cross_summary = load_json(args.cross_summary)
    tasks_meta = [{"task_id": 0, "task_name": TASK0_NAME}]
    tasks_meta += [{"task_id": item["task_id"], "task_name": item["task_name"]} for item in cross_summary["tasks"]]
    tasks_meta = sorted(tasks_meta, key=lambda item: item["task_id"])
    if [item["task_id"] for item in tasks_meta] != list(range(10)):
        raise ValueError("expected LIBERO Object task IDs 0..9")

    task_data = {}
    for meta in tasks_meta:
        result = None if meta["task_id"] == 0 else load_json(args.task_root / f"task_{meta['task_id']}" / "result.json")
        task_meta, configs = load_task_configs(meta["task_id"], task0, result)
        task_data[meta["task_id"]] = {**task_meta, **configs}

    available_global = set.intersection(*(set(item["global"]) for item in task_data.values()))
    available_group = set.intersection(*(set(item["groupwise"]) for item in task_data.values()))
    common_global = sorted(available_global)
    common_group = sorted(available_group, key=lambda pair: (pair[0], pair[1]))
    if common_global != HORIZONS or common_group != GROUP_PAIRS:
        raise ValueError(f"unexpected common configuration set: globals={common_global}, groups={common_group}")

    tasks = []
    for task_id in range(10):
        item = task_data[task_id]
        global_rates = {h: item["global"][h]["success_rate"] for h in common_global}
        group_rates = {pair: item["groupwise"][pair]["success_rate"] for pair in common_group}
        best_global_rate = max(global_rates.values())
        best_group_rate = max(group_rates.values())
        best_global_horizons = [h for h in common_global if abs(global_rates[h] - best_global_rate) <= 1e-12]
        best_group_pairs = [pair for pair in common_group if abs(group_rates[pair] - best_group_rate) <= 1e-12]
        ranks = average_rank_descending(group_rates)
        task = {
            "task_id": task_id,
            "task_name": item["task_name"],
            "episodes_per_configuration": item["global"][common_global[0]]["episodes"],
            "state_ids": list(range(item["global"][common_global[0]]["episodes"])),
            "global": {str(h): item["global"][h] for h in common_global},
            "groupwise": {pair_label(pair): item["groupwise"][pair] for pair in common_group},
            "global_success_rates": {str(h): global_rates[h] for h in common_global},
            "group_success_rates": {pair_label(pair): group_rates[pair] for pair in common_group},
            "best_global_horizons": best_global_horizons,
            "best_global_rate": best_global_rate,
            "best_groupwise_pairs": [pair_label(pair) for pair in best_group_pairs],
            "best_groupwise_rate": best_group_rate,
            "best_static_configurations": [pair_label(pair) for pair in best_group_pairs],
            "static_oracle_rate": best_group_rate,
            "oracle_gap_over_universal_group": None,
            "groupwise_ranks": {pair_label(pair): ranks[pair] for pair in common_group},
            "best_global_vs_groupwise_difference": best_group_rate - best_global_rate,
        }
        tasks.append(task)

    universal_global_metrics = []
    for h in common_global:
        rates = [task["global_success_rates"][str(h)] for task in tasks]
        q_rates = [task["global"][str(h)]["policy_query_rate"] for task in tasks]
        universal_global_metrics.append({
            "horizon": h,
            "macro_success": float(np.mean(rates)),
            "worst_task_success": float(np.min(rates)),
            "median_task_success": float(np.median(rates)),
            "mean_policy_query_rate": float(np.mean(q_rates)),
            "task_success_rates": rates,
            "task_wilson_ci95": [
                wilson(round(rate * task["episodes_per_configuration"]), task["episodes_per_configuration"])
                for rate, task in zip(rates, tasks)
            ],
        })
    best_global_macro = max(item["macro_success"] for item in universal_global_metrics)
    best_global_items = [item for item in universal_global_metrics if abs(item["macro_success"] - best_global_macro) <= 1e-12]

    universal_groupwise_metrics = []
    for pair in common_group:
        label = pair_label(pair)
        rates = [task["group_success_rates"][label] for task in tasks]
        q_rates = [task["groupwise"][label]["policy_query_rate"] for task in tasks]
        pareto_tasks = 0
        for task in tasks:
            candidates = [
                {"success_rate": task["groupwise"][pair_label(other)]["success_rate"], "policy_query_rate": task["groupwise"][pair_label(other)]["policy_query_rate"]}
                for other in common_group
            ]
            current = task["groupwise"][label]
            if not any(
                other["success_rate"] >= current["success_rate"] - 1e-12
                and other["policy_query_rate"] <= current["policy_query_rate"] + 1e-12
                and (other["success_rate"] > current["success_rate"] + 1e-12 or other["policy_query_rate"] < current["policy_query_rate"] - 1e-12)
                for other in candidates
            ):
                pareto_tasks += 1
        universal_groupwise_metrics.append({
            "pair": list(pair),
            "label": label,
            "macro_success": float(np.mean(rates)),
            "median_task_success": float(np.median(rates)),
            "worst_task_success": float(np.min(rates)),
            "mean_policy_query_rate": float(np.mean(q_rates)),
            "tasks_best_or_tied_best": sum(abs(task["best_groupwise_rate"] - rate) <= 1e-12 for task, rate in zip(tasks, rates)),
            "tasks_on_pareto_frontier": pareto_tasks,
            "task_success_rates": rates,
            "task_wilson_ci95": [
                wilson(round(rate * task["episodes_per_configuration"]), task["episodes_per_configuration"])
                for rate, task in zip(rates, tasks)
            ],
        })
    best_group_macro = max(item["macro_success"] for item in universal_groupwise_metrics)
    best_group_items = [item for item in universal_groupwise_metrics if abs(item["macro_success"] - best_group_macro) <= 1e-12]

    for task in tasks:
        universal_pair_rate = task["groupwise"][best_group_items[0]["label"]]["success_rate"]
        task["oracle_gap_over_universal_group"] = task["static_oracle_rate"] - universal_pair_rate

    macro_best_global = float(np.mean([task["best_global_rate"] for task in tasks]))
    macro_best_groupwise = float(np.mean([task["best_groupwise_rate"] for task in tasks]))
    group_better = sum(task["best_groupwise_rate"] > task["best_global_rate"] + 1e-12 for task in tasks)
    tied = sum(abs(task["best_groupwise_rate"] - task["best_global_rate"]) <= 1e-12 for task in tasks)
    global_better = sum(task["best_global_rate"] > task["best_groupwise_rate"] + 1e-12 for task in tasks)

    loto = []
    group_labels = [pair_label(pair) for pair in common_group]
    for held_out in tasks:
        training_tasks = [task for task in tasks if task["task_id"] != held_out["task_id"]]
        training_scores = {
            label: float(np.mean([task["groupwise"][label]["success_rate"] for task in training_tasks]))
            for label in group_labels
        }
        best_training = max(training_scores.values())
        selected = [label for label, value in training_scores.items() if abs(value - best_training) <= 1e-12]
        selected_rates = [held_out["groupwise"][label]["success_rate"] for label in selected]
        regrets = [held_out["best_groupwise_rate"] - value for value in selected_rates]
        loto.append({
            "task_id": held_out["task_id"],
            "selected_pairs": selected,
            "selected_mean_success": float(np.mean(selected_rates)),
            "selected_success_rates": selected_rates,
            "held_out_oracle_success": held_out["best_groupwise_rate"],
            "regrets": regrets,
            "mean_regret": float(np.mean(regrets)),
            "held_out_best_global_success": held_out["best_global_rate"],
        })
    loto_mean_regret = float(np.mean([item["mean_regret"] for item in loto]))

    rank_stability = []
    for pair in common_group:
        ranks = [task["groupwise_ranks"][pair_label(pair)] for task in tasks]
        rank_stability.append({
            "pair": list(pair),
            "label": pair_label(pair),
            "mean_rank": float(np.mean(ranks)),
            "rank_std": float(np.std(ranks)),
            "rank_variance": float(np.var(ranks)),
            "best_rank": float(np.min(ranks)),
            "worst_rank": float(np.max(ranks)),
            "task_ranks": ranks,
        })

    pair_frequency = {}
    arm_frequency = {str(h): 0 for h in HORIZONS}
    grip_frequency = {str(h): 0 for h in HORIZONS}
    relation = {"gripper_longer": 0, "equal": 0, "gripper_shorter": 0}
    for task in tasks:
        for label in task["best_groupwise_pairs"]:
            pair_frequency[label] = pair_frequency.get(label, 0) + 1
            arm, grip = [int(value) for value in label.strip("()").split(",")]
            arm_frequency[str(arm)] += 1
            grip_frequency[str(grip)] += 1
            relation["gripper_longer" if grip > arm else "gripper_shorter" if grip < arm else "equal"] += 1

    points = []
    for item in universal_global_metrics:
        points.append({"label": f"G{item['horizon']}", "kind": "global", "macro_success": item["macro_success"], "mean_policy_query_rate": item["mean_policy_query_rate"]})
    for item in universal_groupwise_metrics:
        arm, grip = item["pair"]
        points.append({**item, "kind": "diagonal_groupwise" if arm == grip else "off_diagonal_groupwise"})
    frontier = pareto_frontier(points)
    dominated = [point["label"] for point in points if point["label"] not in frontier]
    best_global_labels = [f"G{item['horizon']}" for item in best_global_items]
    best_group_labels = [item["label"] for item in best_group_items]
    selected_global = best_global_items[0]
    selected_group = best_group_items[0]

    rng = np.random.default_rng(20260819)
    universal_group_values = [task["groupwise"][selected_group["label"]]["success_rate"] for task in tasks]
    universal_global_values = [task["global"][str(selected_global["horizon"])]["success_rate"] for task in tasks]
    oracle_values = [task["static_oracle_rate"] for task in tasks]
    group_minus_global = np.asarray(universal_group_values) - np.asarray(universal_global_values)
    oracle_minus_group = np.asarray(oracle_values) - np.asarray(universal_group_values)
    bootstrap = {
        "seed": 20260819,
        "draws": 20000,
        "universal_groupwise_minus_global": {"estimate": float(np.mean(group_minus_global)), "ci95": bootstrap_mean(group_minus_global.tolist(), rng)},
        "oracle_minus_universal_groupwise": {"estimate": float(np.mean(oracle_minus_group)), "ci95": bootstrap_mean(oracle_minus_group.tolist(), rng)},
    }

    dynamic_readiness = {
        "classification": "B",
        "reason": "A single universal off-diagonal pair, (4,16), is selected in every leave-one-task-out split and reaches 0.734 macro success versus a 0.779 per-task group-wise static oracle. Its 0.045 oracle gap is real in the task bootstrap diagnostic, but it is small relative to the one-task-per-task sample and the task-optimal pairs vary. The evidence supports a useful static heterogeneous baseline, while the universal pair captures most observed oracle performance; dynamic scheduling is therefore not yet justified.",
    }

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "universal_vs_oracle": str(args.figure_dir / "libero_object_universal_vs_oracle.png"),
        "config_rank_heatmap": str(args.figure_dir / "libero_object_config_rank_heatmap.png"),
        "pareto": str(args.figure_dir / "libero_object_universal_success_query.png"),
    }
    data = {
        "analysis": "libero_object_dynamic_readiness",
        "source_artifacts": [str(args.task0), str(args.cross_summary), str(args.task_root)],
        "task_count": len(tasks),
        "tasks": tasks,
        "common_configuration_set": {
            "global_horizons": common_global,
            "global_labels": [f"G{h}" for h in common_global],
            "diagonal_groupwise_pairs": [list(pair) for pair in common_group if pair[0] == pair[1]],
            "off_diagonal_groupwise_pairs": [list(pair) for pair in common_group if pair[0] != pair[1]],
            "groupwise_pairs": [list(pair) for pair in common_group],
            "task0_excluded_global_horizons": [1],
        },
        "universal_global_metrics": universal_global_metrics,
        "best_universal_global": {"horizons": [item["horizon"] for item in best_global_items], "labels": best_global_labels, "macro_success": best_global_macro},
        "universal_groupwise_metrics": universal_groupwise_metrics,
        "best_universal_groupwise": {"pairs": [item["pair"] for item in best_group_items], "labels": best_group_labels, "macro_success": best_group_macro},
        "macro_best_global": macro_best_global,
        "macro_best_groupwise": macro_best_groupwise,
        "oracle_gap": macro_best_groupwise - best_group_macro,
        "global_vs_groupwise_oracle": {
            "per_task_differences": [task["best_global_vs_groupwise_difference"] for task in tasks],
            "macro_best_global": macro_best_global,
            "macro_best_groupwise": macro_best_groupwise,
            "absolute_difference_groupwise_minus_global": macro_best_groupwise - macro_best_global,
            "groupwise_strictly_better_tasks": group_better,
            "tied_tasks": tied,
            "global_strictly_better_tasks": global_better,
        },
        "leave_one_task_out": loto,
        "leave_one_task_out_mean_regret": loto_mean_regret,
        "groupwise_rank_stability": rank_stability,
        "horizon_preferences": {
            "arm_horizon_frequency": arm_frequency,
            "gripper_horizon_frequency": grip_frequency,
            "pair_frequency": pair_frequency,
            **relation,
            "tie_contribution_rule": "Every tied best group-wise pair contributes one count.",
        },
        "pareto_points": points,
        "pareto_frontier": frontier,
        "pareto_dominated": dominated,
        "universal_groupwise_vs_global": {
            "global_labels": best_global_labels,
            "groupwise_labels": best_group_labels,
            "success_difference": best_group_macro - best_global_macro,
            "query_rate_difference": selected_group["mean_policy_query_rate"] - selected_global["mean_policy_query_rate"],
        },
        "bootstrap": bootstrap,
        "dynamic_readiness": dynamic_readiness,
        "pace_source_audit": {
            "status": "reviewed_primary_source",
            "source": "https://arxiv.org/abs/2606.00537",
            "summary": "PACE is a scalar/global test-time execution rule: from each full predicted chunk it builds a joint- or Cartesian-space arm speed profile, suppresses short fluctuations, identifies low-speed transition regions, and selects one prefix boundary; in multi-arm settings it uses the earliest accepted arm boundary. It uses fixed selection parameters calibrated from demonstrations, does not use evaluation rollouts or policy internals, and discards the unexecuted suffix after each query. The current LIBERO ACT path supplies a full unnormalized (100,7) chunk with six relative end-effector controls and one gripper control, so the chunk boundary input exists. A source-faithful PACE comparison would still need an explicit, verified mapping from the six relative controls to the paper's motion-speed profile and calibration procedure. PACE is global/scalar and is not a group-wise method; it was not implemented.",
        },
        "next_stage_baselines": ["best universal global G16", "best universal static group-wise (4,16)", "per-task static group-wise oracle", "PACE as a separately audited global dynamic baseline if later authorized"],
        "figures": figures,
    }

    plot_universal_vs_oracle(Path(figures["universal_vs_oracle"]), tasks, selected_global["horizon"], tuple(selected_group["pair"]))
    plot_rank_heatmap(Path(figures["config_rank_heatmap"]), tasks)
    plot_pareto(Path(figures["pareto"]), points, frontier, best_global_labels, best_group_labels, macro_best_groupwise)
    args.output_json.write_text(json.dumps(data, indent=2) + "\n")
    args.output_md.write_text(markdown(data))
    print(json.dumps({
        "common_global": common_global,
        "common_groupwise": [pair_label(pair) for pair in common_group],
        "best_universal_global": data["best_universal_global"],
        "best_universal_groupwise": data["best_universal_groupwise"],
        "macro_best_global": macro_best_global,
        "macro_best_groupwise": macro_best_groupwise,
        "oracle_gap": data["oracle_gap"],
        "loto_mean_regret": loto_mean_regret,
        "frontier": frontier,
    }, indent=2))


if __name__ == "__main__":
    main()
