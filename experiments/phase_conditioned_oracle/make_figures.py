#!/usr/bin/env python3
"""Create Gate-2B figures from summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PHASES = ("early", "middle", "late")
HORIZONS = (1, 2, 4, 8, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def phase_global_figure(summary: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for axis, phase in zip(axes, PHASES):
        entries = summary["phase_global_table"][phase]["candidate_summaries"]
        entries = sorted(entries, key=lambda row: int(row["arm_horizon"]))
        x = [int(row["arm_horizon"]) for row in entries]
        y = [float(row["macro_success_rate"]) for row in entries]
        low = [float(row["macro_success_rate_bootstrap_ci95"][0]) for row in entries]
        high = [float(row["macro_success_rate_bootstrap_ci95"][1]) for row in entries]
        axis.errorbar(x, y, yerr=[np.asarray(y) - low, np.asarray(high) - y], marker="o", capsize=3)
        selected = summary["phase_global_table"][phase]["selected"]
        axis.axvline(int(selected["arm_horizon"]), color="tab:red", linestyle="--", alpha=0.7)
        axis.set_title(phase)
        axis.set_xlabel("Target-phase global horizon")
        axis.set_xticks(HORIZONS)
        axis.set_xscale("symlog", linthresh=1)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Macro task success rate")
    fig.suptitle("Gate-2B: phase-conditioned global horizon candidates")
    fig.tight_layout()
    fig.savefig(output_dir / "phase_global_success.png", dpi=180)
    plt.close(fig)


def phase_group_figure(summary: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    image = None
    for axis, phase in zip(axes, PHASES):
        entries = summary["phase_group_table"][phase]["candidate_summaries"]
        values = np.full((len(HORIZONS), len(HORIZONS)), np.nan)
        for row in entries:
            arm = HORIZONS.index(int(row["arm_horizon"]))
            gripper = HORIZONS.index(int(row["gripper_horizon"]))
            values[arm, gripper] = float(row["macro_success_rate"])
        image = axis.imshow(values, vmin=0.0, vmax=1.0, cmap="viridis", origin="upper")
        selected = summary["phase_group_table"][phase]["selected"]
        selected_arm = HORIZONS.index(int(selected["arm_horizon"]))
        selected_gripper = HORIZONS.index(int(selected["gripper_horizon"]))
        axis.scatter([selected_gripper], [selected_arm], facecolors="none", edgecolors="red", s=150, linewidths=2)
        for arm_index in range(len(HORIZONS)):
            for grip_index in range(len(HORIZONS)):
                axis.text(grip_index, arm_index, f"{values[arm_index, grip_index]:.2f}", ha="center", va="center", color="white" if values[arm_index, grip_index] < 0.55 else "black", fontsize=8)
        axis.set_title(phase)
        axis.set_xticks(range(len(HORIZONS)), HORIZONS)
        axis.set_yticks(range(len(HORIZONS)), HORIZONS)
        axis.set_xlabel("Gripper horizon")
        axis.set_ylabel("Arm horizon")
    if image is not None:
        fig.colorbar(image, ax=axes, label="Macro task success rate", shrink=0.85)
    fig.suptitle("Gate-2B: phase-conditioned group horizon candidates")
    fig.savefig(output_dir / "phase_group_success_heatmaps.png", dpi=180)
    plt.close(fig)


def comparison_figure(summary: dict, output_dir: Path) -> None:
    static_global = summary["static_baselines"]["global_h16"]
    static_group = summary["static_baselines"]["group_arm4_grip16"]
    global_oracle = summary["combined_oracles"]["global"]
    group_oracle = summary["combined_oracles"]["group"]
    labels = ["static global\nh=16", "phase oracle\nglobal", "static group\n(4,16)", "phase oracle\ngroup"]
    success = [
        float(static_global["macro_success_rate"]),
        float(global_oracle["macro_success_rate"]),
        float(static_group["macro_success_rate"]),
        float(group_oracle["macro_success_rate"]),
    ]
    static_global_query = float(np.mean([row["policy_query_rate"] for row in static_global["task_results"].values()]))
    static_group_query = float(np.mean([row["policy_query_rate"] for row in static_group["task_results"].values()]))
    query = [static_global_query, float(global_oracle["macro_query_rate"]), static_group_query, float(group_oracle["macro_query_rate"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(labels, success, color=["0.55", "tab:blue", "0.55", "tab:orange"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Macro task success rate")
    axes[0].set_title("Static controls vs phase-conditioned oracle")
    axes[0].tick_params(axis="x", labelrotation=20)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, query, color=["0.55", "tab:blue", "0.55", "tab:orange"])
    axes[1].set_ylabel("Macro policy query rate")
    axes[1].set_title("Query-rate accounting")
    axes[1].tick_params(axis="x", labelrotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "phase_oracle_vs_static.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase_global_figure(summary, args.output_dir)
    phase_group_figure(summary, args.output_dir)
    comparison_figure(summary, args.output_dir)
    print(json.dumps({"figures": [
        "phase_global_success.png",
        "phase_group_success_heatmaps.png",
        "phase_oracle_vs_static.png",
    ]}, indent=2))


if __name__ == "__main__":
    main()
