#!/usr/bin/env python3
"""Apply the frozen pre-unblinding k/20 physical-axis correction to B3 outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "track_b/forecast/analysis"


def main() -> None:
    summary_path = ANALYSIS / "summary.json"
    csv_path = ANALYSIS / "forecast_metrics.csv"
    report_path = ANALYSIS / "report.md"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["status"] != "COMPLETE" or summary["offsets"] != list(range(33)):
        raise RuntimeError("B3 canonical output identity mismatch")
    if summary.get("offset_seconds") not in (
        [k / 10 for k in range(33)],
        [k / 20 for k in range(33)],
    ):
        raise RuntimeError("unexpected pre-correction B3 physical axis")

    summary["dataset_declared_fps"] = 10
    summary["physical_target_rate_hz"] = 20
    summary["offset_seconds"] = [k / 20 for k in range(33)]
    summary["physical_time_axis_correction"] = {
        "status": "CORRECTED_BEFORE_SCIENTIFIC_UNBLINDING",
        "authority": "TEMPORAL_CONTRACT_AUDIT_SUPERSEDING_20260903.md",
        "mapping": "physical offset = k / 20 seconds",
        "scientific_metrics_recomputed": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or {int(row["offset"]) for row in rows} != set(range(33)):
        raise RuntimeError("B3 canonical CSV offset coverage mismatch")
    fields = list(rows[0])
    for row in rows:
        row["offset_seconds"] = str(int(row["offset"]) / 20)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    report = report_path.read_text(encoding="utf-8")
    old = "Chunk offset k is an exact 10 Hz dataset-frame target at k/10 seconds. No interpolation, resampling, or repetition is used."
    new = "Chunk offset k is an exact stored-row target at k/20 physical seconds. The dataset-declared 10 Hz timestamps relabel the retained 20 Hz sequence; no interpolation, resampling, or repetition is used."
    if old in report:
        report = report.replace(old, new)
    elif new not in report:
        raise RuntimeError("unexpected B3 report time-axis wording")
    report_path.write_text(report, encoding="utf-8")

    marker = ROOT / "orchestration/B3_PHYSICAL_AXIS_CORRECTED"
    marker.write_text("CORRECTED_BEFORE_SCIENTIFIC_UNBLINDING k_over_20 metrics_unchanged\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "offsets": 33, "scientific_metrics_recomputed": False}, indent=2))


if __name__ == "__main__":
    main()
