#!/usr/bin/env python3
"""Secondary association of frozen H_temp with existing intervention outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rank_correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(spearmanr(x, y).statistic)


def load_frozen_h(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    rows = data.get("task_values", [])
    result = {str(row["task_key"]): float(row["H_temp"]) for row in rows}
    if len(result) != 8:
        raise ValueError(f"expected eight frozen H_temp values, got {len(result)}")
    return result


def build_relation(protocol: dict[str, Any], frozen_h_path: Path, outcome_path: Path) -> dict[str, Any]:
    frozen_h = load_frozen_h(frozen_h_path)
    outcomes = json.loads(outcome_path.read_text())
    outcome_rows = {str(row["task_key"]): row for row in outcomes.get("per_task", [])}
    task_keys = list(protocol["task_split"]["task_order"])
    if set(outcome_rows) != set(task_keys):
        raise ValueError("closed-loop outcome file does not contain exactly the frozen eight-task cohort")

    rows: list[dict[str, Any]] = []
    delay_summaries: dict[str, Any] = {}
    for delay in (4, 8, 16):
        asymmetries = []
        full_old_deltas = []
        for task_key in task_keys:
            conditions = outcome_rows[task_key]["conditions"]
            fo = float(conditions[f"fo{delay}"]["success_rate"])
            reverse = float(conditions[f"reverse{delay}"]["success_rate"])
            full_old = float(conditions[f"full_old{delay}"]["success_rate"])
            fresh = float(conditions["fresh"]["success_rate"])
            asymmetry = fo - reverse
            full_old_delta = full_old - fresh
            asymmetries.append(asymmetry)
            full_old_deltas.append(full_old_delta)
            rows.append(
                {
                    "task_key": task_key,
                    "split": "development" if task_key in protocol["task_split"]["development"] else "held_out",
                    "H_temp": frozen_h[task_key],
                    "delay_steps": delay,
                    "fresh_success_rate": fresh,
                    "fo_success_rate": fo,
                    "reverse_success_rate": reverse,
                    "full_old_success_rate": full_old,
                    "FO_minus_Reverse": asymmetry,
                    "abs_FO_minus_Reverse": abs(asymmetry),
                    "FullOld_minus_Fresh": full_old_delta,
                }
            )
        h_values = [frozen_h[key] for key in task_keys]
        delay_summaries[str(delay)] = {
            "task_count": len(task_keys),
            "mean_FO_success_rate": float(np.mean([row["fo_success_rate"] for row in rows if row["delay_steps"] == delay])),
            "mean_Reverse_success_rate": float(np.mean([row["reverse_success_rate"] for row in rows if row["delay_steps"] == delay])),
            "mean_FullOld_success_rate": float(np.mean([row["full_old_success_rate"] for row in rows if row["delay_steps"] == delay])),
            "mean_FO_minus_Reverse": float(np.mean(asymmetries)),
            "mean_abs_FO_minus_Reverse": float(np.mean(np.abs(asymmetries))),
            "positive_tasks": int(sum(value > 0 for value in asymmetries)),
            "negative_tasks": int(sum(value < 0 for value in asymmetries)),
            "zero_tasks": int(sum(value == 0 for value in asymmetries)),
            "spearman_H_temp_vs_FO_minus_Reverse": rank_correlation(h_values, asymmetries),
            "spearman_H_temp_vs_abs_FO_minus_Reverse": rank_correlation(h_values, [abs(value) for value in asymmetries]),
            "spearman_H_temp_vs_FullOld_minus_Fresh": rank_correlation(h_values, full_old_deltas),
        }

    mean_rows = []
    for task_key in task_keys:
        task_rows = [row for row in rows if row["task_key"] == task_key]
        mean_rows.append(
            {
                "task_key": task_key,
                "split": task_rows[0]["split"],
                "H_temp": task_rows[0]["H_temp"],
                "mean_FO_minus_Reverse": float(np.mean([row["FO_minus_Reverse"] for row in task_rows])),
                "mean_abs_FO_minus_Reverse": float(np.mean([row["abs_FO_minus_Reverse"] for row in task_rows])),
                "mean_FullOld_minus_Fresh": float(np.mean([row["FullOld_minus_Fresh"] for row in task_rows])),
            }
        )
    h_values = [row["H_temp"] for row in mean_rows]
    signed = [row["mean_FO_minus_Reverse"] for row in mean_rows]
    absolute = [row["mean_abs_FO_minus_Reverse"] for row in mean_rows]
    relation = {
        "status": "complete_secondary_after_frozen_H_temp",
        "frozen_H_temp_path": str(frozen_h_path.resolve()),
        "outcome_path": str(outcome_path.resolve()),
        "outcomes_used_only_after_freeze": True,
        "task_rows_by_delay": rows,
        "task_mean_summary": mean_rows,
        "delay_summaries": delay_summaries,
        "across_delays_task_mean": {
            "spearman_H_temp_vs_mean_FO_minus_Reverse": rank_correlation(h_values, signed),
            "spearman_H_temp_vs_mean_abs_FO_minus_Reverse": rank_correlation(h_values, absolute),
            "qualitative_note": "With eight tasks, correlations are descriptive; signed A can change direction across tasks, so magnitude is also reported.",
        },
        "obvious_counterexamples": [
            {
                "task_key": row["task_key"],
                "H_temp": row["H_temp"],
                "mean_FO_minus_Reverse": row["mean_FO_minus_Reverse"],
            }
            for row in sorted(mean_rows, key=lambda item: item["H_temp"], reverse=True)
            if row["mean_FO_minus_Reverse"] < 0
        ],
        "full_old_reported_separately": True,
        "interpretation_limits": [
            "Success outcomes are not part of H_temp and cannot retroactively change its definition.",
            "The existing intervention results are paired by task/initial state in the source analysis, but stochastic policy sampling was not keyed by physical step.",
            "Eight task-level points are too few for inferential claims; no p-value is used for the decision.",
        ],
    }
    return relation


def make_figure(relation: dict[str, Any], figures_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style_path = Path("/home/wjq/.codex/skills/figure-style")
    sys.path.insert(0, str(style_path))
    from kernel import apply_figure_style  # type: ignore

    apply_figure_style(frame="open", sizes=(8, 7, 6), grid=False)
    colors = {"development": "#2166ac", "held_out": "#e08214"}
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.1), sharex=True, sharey=True)
    for ax, delay in zip(axes, (4, 8, 16)):
        points = [row for row in relation["task_rows_by_delay"] if row["delay_steps"] == delay]
        for split in ("development", "held_out"):
            selected = [row for row in points if row["split"] == split]
            ax.scatter(
                [row["H_temp"] for row in selected],
                [row["FO_minus_Reverse"] for row in selected],
                s=26,
                color=colors[split],
                label=split.replace("_", " "),
            )
        ax.axhline(0, color="#888888", lw=0.7)
        ax.set_title(f"d={delay}")
        ax.set_xlabel("H_temp")
        ax.set_ylim(-0.75, 0.75)
        ax.margins(x=0.12)
    axes[0].set_ylabel("FO − Reverse success-rate difference")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Secondary outcome association after H_temp was frozen", x=0.04, ha="left", y=1.03)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(figures_dir / "figure_C_h_temp_vs_fo_reverse.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--frozen-h", type=Path, default=ROOT / "h_temp_frozen.json")
    parser.add_argument("--outcomes", type=Path, default=ROOT / "../component_temporal_reuse/final_analysis/analysis.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    outcome_path = args.outcomes.resolve()
    relation = build_relation(protocol, args.frozen_h.resolve(), outcome_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "closed_loop_relation.json", relation)
    write_csv(args.output_dir / "closed_loop_asymmetry.csv", relation["task_rows_by_delay"])
    make_figure(relation, args.output_dir / "figures")
    print(json.dumps({"status": relation["status"], "task_rows": len(relation["task_rows_by_delay"])}, indent=2))


if __name__ == "__main__":
    main()
