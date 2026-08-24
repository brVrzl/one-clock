#!/usr/bin/env python3
"""Render reserved ICRA Figure 4 only from complete validated Gate-3B data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "gate3b_figure4_interface.json"
CONDITIONS = ("FF", "OO", "FO", "OF")
READY_STATUS = "final_validated_gate3b"


def load_interface(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("schema_version") != 1:
        raise ValueError("expected Gate-3B figure schema_version 1")
    if data.get("condition_order") != list(CONDITIONS):
        raise ValueError("condition_order must be exactly FF, OO, FO, OF")
    return data


def placeholder_paths(value: Any, prefix: str = "root") -> list[str]:
    paths: list[str] = []
    if isinstance(value, str) and value.startswith("<GATE3B_"):
        paths.append(prefix)
    elif isinstance(value, dict):
        for key, child in value.items():
            paths.extend(placeholder_paths(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(placeholder_paths(child, f"{prefix}[{index}]"))
    return paths


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def validate_ready(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("status") != READY_STATUS:
        raise ValueError(
            f"status must be {READY_STATUS!r}; current status is "
            f"{data.get('status')!r}"
        )
    if not data.get("source_report"):
        raise ValueError("source_report must identify the final validated report")
    pending = placeholder_paths(data)
    if pending:
        raise ValueError("unresolved Gate-3B placeholders: " + ", ".join(pending))

    rates = data.get("success_rates")
    if not isinstance(rates, dict) or set(rates) != set(CONDITIONS):
        raise ValueError("success_rates must contain exactly FF, OO, FO, and OF")
    numeric_rates = {
        condition: finite_number(rates[condition], f"success_rates.{condition}")
        for condition in CONDITIONS
    }
    if any(rate < 0.0 or rate > 1.0 for rate in numeric_rates.values()):
        raise ValueError("success rates must lie in [0, 1]")

    coherence = finite_number(data.get("primary_coherence"), "primary_coherence")
    intervals = data.get("confidence_intervals")
    if not isinstance(intervals, dict) or set(intervals) != {
        "paired_state",
        "task_cluster",
    }:
        raise ValueError(
            "confidence_intervals must contain paired_state and task_cluster"
        )
    numeric_intervals: dict[str, tuple[float, float]] = {}
    for name in ("paired_state", "task_cluster"):
        bounds = intervals[name]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"confidence_intervals.{name} must be [lower, upper]")
        lower = finite_number(bounds[0], f"confidence_intervals.{name}[0]")
        upper = finite_number(bounds[1], f"confidence_intervals.{name}[1]")
        if lower > coherence or coherence > upper:
            raise ValueError(f"{name} interval must contain primary_coherence")
        numeric_intervals[name] = (lower, upper)

    task_rows = data.get("task_coherence")
    if not isinstance(task_rows, list) or len(task_rows) != 10:
        raise ValueError("task_coherence must contain exactly ten task rows")
    seen: set[int] = set()
    numeric_tasks: list[tuple[int, float]] = []
    for row in task_rows:
        if not isinstance(row, dict) or set(row) != {"task_id", "coherence"}:
            raise ValueError("each task row must contain task_id and coherence")
        task_id = row["task_id"]
        if isinstance(task_id, bool) or not isinstance(task_id, int):
            raise ValueError("task_id must be an integer")
        seen.add(task_id)
        numeric_tasks.append(
            (task_id, finite_number(row["coherence"], f"task {task_id} coherence"))
        )
    if seen != set(range(10)):
        raise ValueError("task_coherence must contain task IDs 0 through 9 once each")
    numeric_tasks.sort()

    return {
        "rates": numeric_rates,
        "coherence": coherence,
        "intervals": numeric_intervals,
        "tasks": numeric_tasks,
        "source_report": data["source_report"],
    }


def render(ready: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    coherent_color = "#4C78A8"
    mixed_color = "#F58518"
    colors = [coherent_color, coherent_color, mixed_color, mixed_color]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.05, 2.35),
        gridspec_kw={"width_ratios": [1.05, 1.0, 1.25], "wspace": 0.48},
    )

    ax = axes[0]
    rates = [ready["rates"][condition] for condition in CONDITIONS]
    ax.bar(CONDITIONS, rates, color=colors, width=0.72)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Task success")
    ax.set_title("(a) Four conditions", loc="left")
    ax.text(0.25, -0.25, "source-coherent", transform=ax.transAxes,
            ha="center", va="top", color=coherent_color)
    ax.text(0.75, -0.25, "mixed", transform=ax.transAxes,
            ha="center", va="top", color=mixed_color)

    ax = axes[1]
    names = ("paired state", "task cluster")
    y = np.array([1.0, 0.0])
    point = ready["coherence"]
    for index, name in enumerate(("paired_state", "task_cluster")):
        lower, upper = ready["intervals"][name]
        ax.errorbar(
            point,
            y[index],
            xerr=[[point - lower], [upper - point]],
            fmt="o",
            color="#2F4B7C",
            capsize=3,
        )
    ax.axvline(0.0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_yticks(y, names)
    ax.set_xlabel(r"$C_{coherence}$")
    ax.set_title("(b) Primary contrast", loc="left")

    ax = axes[2]
    task_ids = [task_id for task_id, _ in ready["tasks"]]
    task_values = [value for _, value in ready["tasks"]]
    task_colors = [coherent_color if value >= 0 else mixed_color for value in task_values]
    ax.bar(task_ids, task_values, color=task_colors, width=0.72)
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.set_xticks(task_ids)
    ax.set_xlabel("LIBERO Object task")
    ax.set_ylabel(r"Task $C_{coherence}$")
    ax.set_title("(c) Task sensitivity", loc="left")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.87, bottom=0.25)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=HERE / "gate3b_figure4.pdf")
    parser.add_argument(
        "--check-interface",
        action="store_true",
        help="check the waiting interface without attempting to render",
    )
    args = parser.parse_args()

    data = load_interface(args.input)
    pending = placeholder_paths(data)
    if args.check_interface:
        print(f"schema_version={data['schema_version']}")
        print(f"status={data.get('status')}")
        print("pending=" + (", ".join(pending) if pending else "none"))
        return

    ready = validate_ready(data)
    render(ready, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
