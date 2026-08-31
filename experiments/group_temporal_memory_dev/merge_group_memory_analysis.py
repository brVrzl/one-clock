#!/usr/bin/env python3
"""Merge separately frozen ACT and SmolVLA analyses into deliverable files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_group_memory import render_report


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--act", type=Path, default=ROOT / "analysis_act.json")
    parser.add_argument("--smolvla", type=Path, default=ROOT / "analysis_smolvla.json")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis.json")
    parser.add_argument("--report", type=Path, default=ROOT / "report.md")
    parser.add_argument("--decision", default="PENDING_OUTCOME_ANALYSIS")
    parser.add_argument("--interpretation", default="Supervisor interpretation pending completion of the development gate.")
    args = parser.parse_args()
    act = json.loads(args.act.read_text())
    smolvla = json.loads(args.smolvla.read_text())
    if act.get("shared_kernel") != "dense_equivalent_te" or smolvla.get("shared_kernel") != "dense_equivalent_te":
        raise RuntimeError("cannot merge analyses with a non-Sol-selected shared kernel")
    if act.get("sol_repaired_rollout_commit") != smolvla.get("sol_repaired_rollout_commit"):
        raise RuntimeError("ACT and SmolVLA analyses use different repaired-trio commits")
    combined = {
        "status": "complete",
        "shared_kernel": "dense_equivalent_te",
        "sol_audit_commit": act["sol_audit_commit"],
        "sol_repaired_rollout_commit": act["sol_repaired_rollout_commit"],
        "policies": {"ACT": act, "SmolVLA": smolvla},
        "decision_label": args.decision,
        "interpretation": args.interpretation,
    }
    args.output.write_text(json.dumps(combined, indent=2) + "\n")
    args.report.write_text(render_report(combined))
    print(json.dumps({"status": "complete", "output": str(args.output), "report": str(args.report)}))


if __name__ == "__main__":
    main()

