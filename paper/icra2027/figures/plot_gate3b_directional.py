#!/usr/bin/env python3
"""Validate and render the post-hoc Gate-3B directional figure interface."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "gate3b_directional_figure_interface.json"


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def load_and_validate(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("expected schema_version 1")
    if data.get("status") != "final_validated_gate3b_posthoc":
        raise ValueError("interface is not a final validated Gate-3B post-hoc result")

    matrix = data["matrix"]
    expected_conditions = [["FF", "FO"], ["OF", "OO"]]
    if matrix.get("condition_labels") != expected_conditions:
        raise ValueError("matrix condition layout must be [[FF, FO], [OF, OO]]")
    for name in ("successes", "trials", "rates"):
        values = matrix.get(name)
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError(f"matrix.{name} must be 2 by 2")
        if any(not isinstance(row, list) or len(row) != 2 for row in values):
            raise ValueError(f"matrix.{name} must be 2 by 2")
    for row in range(2):
        for column in range(2):
            successes = finite(matrix["successes"][row][column], "successes")
            trials = finite(matrix["trials"][row][column], "trials")
            rate = finite(matrix["rates"][row][column], "rates")
            if trials <= 0 or not math.isclose(successes / trials, rate, abs_tol=1e-12):
                raise ValueError("success counts and rates disagree")

    effects = data["posthoc_main_effects"]
    if set(effects) != {"fresh_arm", "old_gripper"}:
        raise ValueError("expected fresh_arm and old_gripper main effects")
    for name, effect in effects.items():
        estimate = finite(effect["estimate"], f"{name}.estimate")
        for interval_name in ("paired_state_ci95", "task_cluster_ci95"):
            bounds = effect[interval_name]
            if len(bounds) != 2 or not bounds[0] <= estimate <= bounds[1]:
                raise ValueError(f"{name}.{interval_name} must contain estimate")

    tasks = data["task_directional_comparisons"]
    if len(tasks) != 10 or {row["task_id"] for row in tasks} != set(range(10)):
        raise ValueError("task comparisons must contain tasks 0 through 9")
    return data


def render(data: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    blue = "#4477AA"
    orange = "#EE7733"
    gray = "#666666"

    fig, axes = plt.subplots(
        1, 3, figsize=(7.05, 2.35),
        gridspec_kw={"width_ratios": [1.0, 1.1, 1.35], "wspace": 0.52},
    )

    matrix = np.asarray(data["matrix"]["rates"], dtype=float)
    ax = axes[0]
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=0.7)
    labels = data["matrix"]["condition_labels"]
    successes = data["matrix"]["successes"]
    for row in range(2):
        for column in range(2):
            ax.text(column, row, f"{labels[row][column]}\n{successes[row][column]}/100",
                    ha="center", va="center", color="black", fontweight="bold")
    ax.set_xticks([0, 1], ["fresh", "old20"])
    ax.set_yticks([0, 1], ["fresh", "old20"])
    ax.set_xlabel("Gripper source")
    ax.set_ylabel("Arm source")
    ax.set_title("(a) Gate-3B success", loc="left")
    image.colorbar = None

    ax = axes[1]
    effect_order = ("fresh_arm", "old_gripper")
    labels_effect = ("fresh arm", "old gripper")
    y = np.asarray([1.0, 0.0])
    for index, name in enumerate(effect_order):
        effect = data["posthoc_main_effects"][name]
        estimate = effect["estimate"]
        paired = effect["paired_state_ci95"]
        task = effect["task_cluster_ci95"]
        ax.errorbar(estimate, y[index] + 0.08,
                    xerr=[[estimate - paired[0]], [paired[1] - estimate]],
                    fmt="o", color=blue, capsize=2.5, label="paired" if index == 0 else None)
        ax.errorbar(estimate, y[index] - 0.08,
                    xerr=[[estimate - task[0]], [task[1] - estimate]],
                    fmt="s", color=orange, capsize=2.5, label="task cluster" if index == 0 else None)
    ax.axvline(0, color=gray, linewidth=0.8, linestyle="--")
    ax.set_yticks(y, labels_effect)
    ax.set_xlabel("Success difference")
    ax.set_title("(b) Marginal effects", loc="left")
    ax.text(0.02, 1.02, "post-hoc", transform=ax.transAxes, color=gray, fontsize=7)
    ax.legend(frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.24))

    ax = axes[2]
    tasks = sorted(data["task_directional_comparisons"], key=lambda row: row["task_id"])
    x = np.arange(10)
    width = 0.36
    ax.bar(x - width / 2, [row["FO_minus_FF"] for row in tasks], width,
           color=blue, label="FO - FF")
    ax.bar(x + width / 2, [row["FO_minus_OO"] for row in tasks], width,
           color=orange, label="FO - OO")
    ax.axhline(0, color=gray, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xlabel("LIBERO Object task")
    ax.set_ylabel("Success difference")
    ax.set_title("(c) Task consistency", loc="left")
    ax.text(0.02, 1.02, "post-hoc", transform=ax.transAxes, color=gray, fontsize=7)
    ax.legend(frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.24))

    fig.suptitle(
        "Preregistered coherence unresolved; directional effects are exploratory",
        x=0.52, y=1.01, fontsize=9,
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.84, bottom=0.31)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=HERE / "gate3b_directional.pdf")
    parser.add_argument("--check-interface", action="store_true")
    args = parser.parse_args()
    data = load_and_validate(args.input)
    if args.check_interface:
        print("valid Gate-3B directional interface")
        return
    render(data, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
