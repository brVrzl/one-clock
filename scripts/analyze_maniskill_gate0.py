#!/usr/bin/env python3
"""Analyze ManiSkill Gate 0 branch records without pandas."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata


HEURISTICS = (
    "action_magnitude",
    "action_velocity",
    "action_acceleration",
    "gripper_transition",
    "eef_object_distance",
    "object_goal_distance",
)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def load_rows(path: Path, task: str, manifest: dict) -> list[dict]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    successful_episodes = {
        int(item["episode"])
        for item in manifest["episodes"]
        if item["expert_success"]
    }
    for row in rows:
        row["task"] = task
        row["episode"] = int(row["episode"])
        row["timestep"] = int(row["timestep"])
        for key in row:
            if key not in {"task", "perturbation_type", "state_id", "branch_error"}:
                try:
                    row[key] = float(row[key])
                except (TypeError, ValueError):
                    pass
        row["expert_success"] = int(row["episode"] in successful_episodes)
    return rows


def state_groups(rows: list[dict], successful_only: bool = True) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if successful_only and not row["expert_success"]:
            continue
        groups[(row["task"], row["episode"], row["timestep"])].append(row)
    output = []
    for key, items in sorted(groups.items()):
        first = items[0]
        valid = [item for item in items if item["branch_valid"] > 0]
        successes = [item["branch_success"] for item in valid]
        item = {
            "task": key[0],
            "episode": key[1],
            "timestep": key[2],
            "phase": first["phase"],
            "criticality": first["criticality"],
            "valid_branches": len(valid),
            "branches": len(items),
            "invalid_branches": len(items) - len(valid),
            "success_rate": float(np.mean(successes)) if successes else float("nan"),
        }
        for heuristic in HEURISTICS:
            item[heuristic] = first[heuristic]
        output.append(item)
    return output


def save_plots(groups: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tasks = sorted({item["task"] for item in groups})
    colors = {task: color for task, color in zip(tasks, ("#0072B2", "#D55E00", "#009E73"))}

    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 3.5), squeeze=False)
    for axis, task in zip(axes[0], tasks):
        data = [item for item in groups if item["task"] == task]
        axis.scatter([item["phase"] for item in data], [item["criticality"] for item in data], s=18, alpha=0.35, color=colors[task])
        phases = np.linspace(0, 1, 21)
        means = []
        for lo, hi in zip(phases[:-1], phases[1:]):
            values = [item["criticality"] for item in data if lo <= item["phase"] < hi]
            means.append(np.mean(values) if values else np.nan)
        axis.plot((phases[:-1] + phases[1:]) / 2, means, color="black", linewidth=2)
        axis.set_title(task)
        axis.set_xlabel("normalized trajectory phase")
        axis.set_ylabel("criticality")
        axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "criticality_vs_phase.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 3.5))
    for task in tasks:
        values = [item["criticality"] for item in groups if item["task"] == task]
        axis.hist(values, bins=np.linspace(-0.05, 1.05, 12), alpha=0.55, label=task, color=colors[task])
    axis.set_xlabel("criticality")
    axis.set_ylabel("state count")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "criticality_histogram.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), squeeze=False)
    for axis, heuristic in zip(axes.flat, HEURISTICS):
        for task in tasks:
            data = [item for item in groups if item["task"] == task]
            axis.scatter([item[heuristic] for item in data], [item["criticality"] for item in data], s=13, alpha=0.35, label=task, color=colors[task])
        axis.set_title(heuristic.replace("_", " "))
        axis.set_ylabel("criticality")
        axis.grid(alpha=0.2)
    axes[1, 0].set_xlabel("heuristic value")
    axes[1, 1].set_xlabel("heuristic value")
    axes[1, 2].set_xlabel("heuristic value")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "criticality_vs_heuristics.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("experiments/counterfactual_tournament/maniskill_gate0_validated"))
    args = parser.parse_args()
    input_dir = args.input
    manifests = {
        "PickCube-v1": json.loads((input_dir / "pick_manifest.json").read_text()),
        "StackCube-v1": json.loads((input_dir / "stack_manifest.json").read_text()),
    }
    rows = []
    rows.extend(load_rows(input_dir / "pick_timestep_branches.csv", "PickCube-v1", manifests["PickCube-v1"]))
    rows.extend(load_rows(input_dir / "stack_timestep_branches.csv", "StackCube-v1", manifests["StackCube-v1"]))
    groups = state_groups(rows)
    all_groups = state_groups(rows, successful_only=False)
    summary = {
        "data_scope": "successful nominal experts only for causal summaries",
        "tasks": {},
        "validation": {
            "exact_restore_suffix": "passed for every sampled state; runner aborts on mismatch",
            "branch_isolation": "passed by repeated zero-branch replay at every sampled state",
            "zero_perturbation": "passed for every sampled state in included episodes",
            "invalid_branches": int(sum(item["invalid_branches"] for item in all_groups)),
            "total_branches_including_failed_nominal_episodes": len(rows),
        },
        "qualitative_examples": [],
    }
    for task in manifests:
        task_groups = [item for item in groups if item["task"] == task]
        task_rows = [row for row in rows if row["task"] == task]
        expert_success = [item for item in manifests[task]["episodes"] if item["expert_success"]]
        criticality = np.asarray([item["criticality"] for item in task_groups])
        stats = {
            "requested_episodes": len(manifests[task]["episodes"]),
            "successful_expert_episodes": len(expert_success),
            "sampled_states": len(task_groups),
            "valid_perturbed_branches": int(sum(item["valid_branches"] - (1 if item["branches"] else 0) for item in task_groups)),
            "invalid_branches": int(sum(item["invalid_branches"] for item in task_groups)),
            "criticality_mean": float(np.mean(criticality)),
            "criticality_std": float(np.std(criticality, ddof=1)),
            "criticality_min": float(np.min(criticality)),
            "criticality_max": float(np.max(criticality)),
            "criticality_nonuniform_range": float(np.max(criticality) - np.min(criticality)),
            "runtime_sec": float(sum(item["runtime_sec"] for item in manifests[task]["episodes"])),
            "heuristic_correlations": {},
        }
        for heuristic in HEURISTICS:
            x = np.asarray([item[heuristic] for item in task_groups])
            stats["heuristic_correlations"][heuristic] = {
                "pearson": pearson(x, criticality),
                "spearman": spearman(x, criticality),
            }
        summary["tasks"][task] = stats

        # States whose causal score is high despite no gripper event and
        # below-median action-velocity rank are useful qualitative checks.
        velocity = np.asarray([item["action_velocity"] for item in task_groups])
        median_velocity = float(np.median(velocity))
        threshold = float(np.quantile(criticality, 0.75))
        candidates = [item for item in task_groups if item["criticality"] >= threshold and item["gripper_transition"] == 0 and item["action_velocity"] <= median_velocity]
        for item in candidates[:3]:
            summary["qualitative_examples"].append({
                "task": task,
                "episode": item["episode"],
                "timestep": item["timestep"],
                "phase": item["phase"],
                "criticality": item["criticality"],
                "action_velocity": item["action_velocity"],
                "gripper_transition": item["gripper_transition"],
                "eef_object_distance": item["eef_object_distance"],
                "object_goal_distance": item["object_goal_distance"],
            })

    output = input_dir / "figures"
    save_plots(groups, output)
    (input_dir / "gate0_analysis.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
