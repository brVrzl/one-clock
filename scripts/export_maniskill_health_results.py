#!/usr/bin/env python3
"""Export ACT TensorBoard health curves to a reviewable CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("results/maniskill_act_health_curve.csv"))
    args = parser.parse_args()
    rows = []
    for run in sorted(args.run_root.glob("*uniformact*")):
        if not run.is_dir() or not list(run.glob("events.out.tfevents.*")):
            continue
        acc = EventAccumulator(str(run))
        acc.Reload()
        once = {event.step: event.value for event in acc.Scalars("eval/success_once")}
        end = {event.step: event.value for event in acc.Scalars("eval/success_at_end")}
        for step in sorted(set(once) | set(end)):
            rows.append({"run": run.name, "step": step, "success_once": once.get(step, ""), "success_at_end": end.get(step, "")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "step", "success_once", "success_at_end"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
