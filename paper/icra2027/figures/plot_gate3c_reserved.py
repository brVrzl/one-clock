#!/usr/bin/env python3
"""Guard and render reserved Gate-3C Figure 4 after final validation only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "gate3c_figure4_interface.json"
CONDITIONS = ("FO20", "FF", "OO20", "age_exp", "CogACT")


def placeholder_paths(value: Any, prefix: str = "root") -> list[str]:
    paths: list[str] = []
    if isinstance(value, str) and value.startswith("<GATE3C_"):
        paths.append(prefix)
    elif isinstance(value, dict):
        for key, child in value.items():
            paths.extend(placeholder_paths(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(placeholder_paths(child, f"{prefix}[{index}]"))
    return paths


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("expected schema_version 1")
    if data.get("condition_order") != list(CONDITIONS):
        raise ValueError("unexpected Gate-3C condition order")
    return data


def validate_ready(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("status") != "final_validated_gate3c":
        raise ValueError("status must be final_validated_gate3c")
    pending = placeholder_paths(data)
    if pending:
        raise ValueError("unresolved Gate-3C placeholders: " + ", ".join(pending))
    if not data.get("source_report"):
        raise ValueError("source_report must identify the final validated report")

    rates = data.get("success_rates")
    if not isinstance(rates, dict) or set(rates) != set(CONDITIONS):
        raise ValueError("success_rates must contain exactly the frozen conditions")
    numeric_rates = {name: finite(rates[name], name) for name in CONDITIONS}
    if any(rate < 0 or rate > 1 for rate in numeric_rates.values()):
        raise ValueError("success rates must lie in [0, 1]")

    contrasts = data.get("primary_contrasts")
    if not isinstance(contrasts, dict) or not contrasts:
        raise ValueError("primary_contrasts must be a nonempty numeric mapping")
    intervals = data.get("confidence_intervals")
    if not isinstance(intervals, dict) or set(intervals) != set(contrasts):
        raise ValueError("confidence_intervals must match primary_contrasts")
    numeric_contrasts: dict[str, tuple[float, tuple[float, float]]] = {}
    for name, value in contrasts.items():
        estimate = finite(value, f"primary_contrasts.{name}")
        bounds = intervals[name]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"confidence_intervals.{name} must be [lower, upper]")
        lower = finite(bounds[0], f"{name}.lower")
        upper = finite(bounds[1], f"{name}.upper")
        if not lower <= estimate <= upper:
            raise ValueError(f"confidence interval for {name} must contain estimate")
        numeric_contrasts[name] = (estimate, (lower, upper))

    tasks = data.get("task_results")
    if not isinstance(tasks, list) or len(tasks) != 10:
        raise ValueError("task_results must contain ten complete task rows")
    if {row.get("task_id") for row in tasks} != set(range(10)):
        raise ValueError("task_results must contain task IDs 0 through 9")
    for row in tasks:
        if not isinstance(row.get("contrasts"), dict):
            raise ValueError("each task row requires a numeric contrasts mapping")
        for name, value in row["contrasts"].items():
            finite(value, f"task {row['task_id']} contrast {name}")
    return {"rates": numeric_rates, "contrasts": numeric_contrasts, "tasks": tasks}


def render(ready: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42}
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.3),
                             gridspec_kw={"wspace": 0.5})
    colors = ["#228833", "#4477AA", "#AA3377", "#CCBB44", "#66CCEE"]

    ax = axes[0]
    ax.bar(CONDITIONS, [ready["rates"][name] for name in CONDITIONS], color=colors)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylabel("Task success")
    ax.set_title("(a) Untouched-state success", loc="left")

    ax = axes[1]
    names = list(ready["contrasts"])
    y = np.arange(len(names))[::-1]
    for row, name in enumerate(names):
        estimate, (lower, upper) = ready["contrasts"][name]
        ax.errorbar(estimate, y[row], xerr=[[estimate - lower], [upper - estimate]],
                    fmt="o", color="#332288", capsize=3)
    ax.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_yticks(y, names)
    ax.set_xlabel("Success difference")
    ax.set_title("(b) Frozen contrasts", loc="left")

    ax = axes[2]
    first_contrast = names[0]
    tasks = sorted(ready["tasks"], key=lambda row: row["task_id"])
    values = [row["contrasts"][first_contrast] for row in tasks]
    ax.bar(range(10), values, color="#228833")
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xticks(range(10))
    ax.set_xlabel("LIBERO Object task")
    ax.set_ylabel(first_contrast)
    ax.set_title("(c) Task effects", loc="left")

    fig.subplots_adjust(left=0.08, right=0.995, top=0.88, bottom=0.27)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=HERE / "gate3c_figure4.pdf")
    parser.add_argument("--check-interface", action="store_true")
    args = parser.parse_args()
    data = load(args.input)
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
