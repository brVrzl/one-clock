#!/usr/bin/env python3
"""Render every frozen supplement condition and contrast without selection."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def fmt_ci(values: list[float]) -> str:
    return f"[{values[0]:+.2f}, {values[1]:+.2f}]"


def main() -> None:
    analysis = json.loads((ROOT / "analysis.json").read_text(encoding="utf-8"))
    if analysis.get("status") != "COMPLETE":
        raise RuntimeError("canonical supplement analysis is incomplete")

    summaries = analysis["condition_summaries"]
    contrasts = analysis["contrasts"]
    summary = {(row["family"], row["method"]): row for row in summaries}
    contrast = {(row["family"], row["contrast"]): row for row in contrasts}

    expected = {
        "r1a": {"A0_G0", *(f"A{d}_G0" for d in (2, 4, 8, 12, 16, 20, 32)), *(f"A0_G{d}" for d in (2, 4, 8, 12, 16, 20, 32))},
        "r1b": {"A0_G0", "T20_R0_G0", "T0_R20_G0"},
        "r1c": {"C00", "C10", "C01", "C11"},
        "r1d": {"A0_G0", "A0_G20", "A20_G0", "A20_G20"},
    }
    for family, methods in expected.items():
        available = {method for fam, method in summary if fam == family}
        if not methods.issubset(available):
            raise RuntimeError(f"{family} condition coverage mismatch")

    enriched = []
    for row in contrasts:
        by_suite: dict[str, list[float]] = defaultdict(list)
        for task, value in row["per_task_delta_percentage_points"].items():
            by_suite[task.split(":task", 1)[0]].append(float(value))
        per_suite = {suite: sum(values) / len(values) for suite, values in sorted(by_suite.items())}
        leave_one_suite_out = {
            suite: sum(v for other, v in per_suite.items() if other != suite) / (len(per_suite) - 1)
            for suite in per_suite
        } if len(per_suite) > 1 else {}
        enriched.append({**row, "per_suite_delta_percentage_points": per_suite,
                         "leave_one_suite_out_percentage_points": leave_one_suite_out})

    output = {
        "status": "COMPLETE",
        "source": "canonical analyze_supplement.py outputs",
        "selection": "NONE_ALL_FROZEN_CONDITIONS_AND_CONTRASTS",
        "condition_summaries": summaries,
        "contrasts": enriched,
        "r1c_risk_difference_interaction": analysis["r1c_risk_difference_interaction"],
        "scientific_retries": analysis["scientific_retries"],
    }
    (ROOT / "canonical_report.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    with (ROOT / "canonical_per_task_effects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["family", "contrast", "task", "delta_percentage_points"], lineterminator="\n")
        writer.writeheader()
        for row in enriched:
            for task, value in sorted(row["per_task_delta_percentage_points"].items()):
                writer.writerow({"family": row["family"], "contrast": row["contrast"], "task": task, "delta_percentage_points": value})

    lines = [
        "# Canonical reviewer-supplement analysis", "",
        "Status: `COMPLETE`; all frozen conditions and contrasts are reported without selection.", "",
    ]
    for family in ("r1a", "r1b", "r1c", "r1d"):
        lines += [f"## {family.upper()} conditions", "", "| Method | Success/N | Rate |", "|---|---:|---:|"]
        for method in sorted(expected[family]):
            row = summary[(family, method)]
            lines.append(f"| `{method}` | {row['successes']}/{row['N']} | {100 * row['success_rate']:.2f}% |")
        lines += ["", f"## {family.upper()} frozen contrasts", "",
                  "| Contrast | Delta (pp) | Discordance | Exact McNemar p | Paired 95% CI (pp) | Task-cluster 95% CI (pp) |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in (item for item in enriched if item["family"] == family):
            lines.append(
                f"| `{row['contrast']}` | {row['delta_percentage_points']:+.2f} | "
                f"{row['first_only']}/{row['second_only']} | {row['exact_two_sided_mcnemar_p']:.6g} | "
                f"{fmt_ci(row['paired_bootstrap_ci_percentage_points'])} | "
                f"{fmt_ci(row['task_cluster_bootstrap_ci_percentage_points'])} |"
            )
        lines += [""]
    lines += [
        "## R1C frozen interaction", "",
        "The frozen risk-difference interaction is `C11 - C10 - C01 + C00`.", "",
        f"Observed interaction: `{100 * analysis['r1c_risk_difference_interaction']:+.2f} pp`.", "",
        "Complete per-task, per-suite, leave-one-task-out, and leave-one-suite-out values are in `canonical_report.json`; tidy per-task values are in `canonical_per_task_effects.csv`.", "",
    ]
    (ROOT / "CANONICAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "conditions": len(summaries), "contrasts": len(contrasts)}, indent=2))


if __name__ == "__main__":
    main()
