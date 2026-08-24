#!/usr/bin/env python3
"""Analyze the paired closed-loop gripper timing sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


SHIFTS = (-8, -4, 0, 4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials == 0:
        return [float("nan"), float("nan")]
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(
        rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return [center - radius, center + radius]


def load_runs(input_dir: Path) -> dict[int, dict[int, dict[int, dict[str, object]]]]:
    tasks: dict[int, dict[int, dict[int, dict[str, object]]]] = {}
    for metadata_path in sorted(input_dir.glob("task*_shift_*/metadata.json")):
        run_dir = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        task_id = int(metadata["task_id"])
        shift = int(metadata["gripper_shift_steps"])
        if shift not in SHIFTS:
            raise ValueError(f"unexpected shift {shift} in {run_dir}")
        episodes = {}
        for line in (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            episodes[int(record["init_state_id"])] = record
        if len(episodes) != int(metadata["episodes"]):
            raise ValueError(f"incomplete episodes in {run_dir}")
        tasks.setdefault(task_id, {})[shift] = episodes
    return tasks


def validate_pairing(task_id: int, runs: dict[int, dict[int, dict[str, object]]]) -> list[int]:
    if set(runs) != set(SHIFTS):
        raise ValueError(f"task {task_id} has shifts {sorted(runs)}, expected {list(SHIFTS)}")
    state_sets = {shift: set(episodes) for shift, episodes in runs.items()}
    baseline_states = state_sets[0]
    if any(states != baseline_states for states in state_sets.values()):
        raise ValueError(f"task {task_id} has unpaired initial states")
    for state in baseline_states:
        seeds = {int(runs[shift][state]["seed"]) for shift in SHIFTS}
        if len(seeds) != 1:
            raise ValueError(f"task {task_id}, state {state} has seeds {seeds}")
    return sorted(baseline_states)


def main() -> None:
    args = parse_args()
    tasks = load_runs(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_state_rows: list[dict[str, object]] = []
    task_summaries: list[dict[str, object]] = []
    for task_id in sorted(tasks):
        runs = tasks[task_id]
        states = validate_pairing(task_id, runs)
        baseline_successes = sum(bool(runs[0][state]["success"]) for state in states)
        baseline_failures = len(states) - baseline_successes
        oracle_successes = 0
        recovered_failures = 0
        fixed = {}
        for shift in SHIFTS:
            successes = sum(bool(runs[shift][state]["success"]) for state in states)
            broken = sum(
                bool(runs[0][state]["success"]) and not bool(runs[shift][state]["success"])
                for state in states
            )
            rescued = sum(
                not bool(runs[0][state]["success"]) and bool(runs[shift][state]["success"])
                for state in states
            )
            fixed[str(shift)] = {
                "successes": successes,
                "trials": len(states),
                "success_rate": successes / len(states),
                "wilson_ci95": wilson(successes, len(states)),
                "baseline_failures_rescued": rescued,
                "baseline_successes_broken": broken,
                "baseline_success_break_fraction": (
                    broken / baseline_successes if baseline_successes else None
                ),
            }
        for state in states:
            baseline = bool(runs[0][state]["success"])
            candidate_success = any(bool(runs[shift][state]["success"]) for shift in SHIFTS)
            rescue = not baseline and candidate_success
            oracle_successes += int(candidate_success)
            recovered_failures += int(rescue)
            per_state_rows.append(
                {
                    "task_id": task_id,
                    "init_state_id": state,
                    "seed": int(runs[0][state]["seed"]),
                    "shift_-8": int(bool(runs[-8][state]["success"])),
                    "shift_-4": int(bool(runs[-4][state]["success"])),
                    "shift_0_baseline": int(baseline),
                    "shift_+4": int(bool(runs[4][state]["success"])),
                    "shift_+8": int(bool(runs[8][state]["success"])),
                    "any_candidate_success": int(candidate_success),
                    "baseline_failure_rescued": int(rescue),
                }
            )
        task_summaries.append(
            {
                "task_id": task_id,
                "task_name": runs[0][states[0]]["task_name"],
                "paired_initial_states": len(states),
                "baseline_successes": baseline_successes,
                "baseline_success_rate": baseline_successes / len(states),
                "fixed_shifts": fixed,
                "timing_oracle_successes": oracle_successes,
                "timing_oracle_success_rate": oracle_successes / len(states),
                "timing_oracle_wilson_ci95": wilson(oracle_successes, len(states)),
                "baseline_failures": baseline_failures,
                "baseline_failures_recoverable": recovered_failures,
                "recoverable_baseline_failure_fraction": (
                    recovered_failures / baseline_failures if baseline_failures else None
                ),
                "oracle_absolute_headroom": (oracle_successes - baseline_successes) / len(states),
            }
        )

    total_states = sum(int(row["paired_initial_states"]) for row in task_summaries)
    total_baseline = sum(int(row["baseline_successes"]) for row in task_summaries)
    total_oracle = sum(int(row["timing_oracle_successes"]) for row in task_summaries)
    total_failures = total_states - total_baseline
    total_recovered = sum(int(row["baseline_failures_recoverable"]) for row in task_summaries)
    aggregate = {
        "paired_initial_states": total_states,
        "baseline_successes": total_baseline,
        "baseline_success_rate": total_baseline / total_states,
        "timing_oracle_successes": total_oracle,
        "timing_oracle_success_rate": total_oracle / total_states,
        "oracle_absolute_headroom": (total_oracle - total_baseline) / total_states,
        "baseline_failures": total_failures,
        "baseline_failures_recoverable": total_recovered,
        "recoverable_baseline_failure_fraction": total_recovered / total_failures,
    }
    output = {
        "status": "exploratory paired causal upper-bound diagnostic",
        "shift_convention": "negative advances and positive delays the nominal gripper sequence; arm dimensions 0:6 unchanged",
        "candidate_shifts_control_steps": list(SHIFTS),
        "tasks": task_summaries,
        "aggregate": aggregate,
        "all_completed_runs_included": True,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "per_state.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_state_rows[0]))
        writer.writeheader()
        writer.writerows(per_state_rows)

    lines = [
        "# EventAlign closed-loop timing sweep",
        "",
        "`1` denotes task success. Negative shifts advance and positive shifts delay the nominal gripper sequence.",
        "",
        "| Task | State | Seed | -8 | -4 | 0 baseline | +4 | +8 | Any success | Failure rescued |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_state_rows:
        lines.append(
            f"| {row['task_id']} | {row['init_state_id']} | {row['seed']} | "
            f"{row['shift_-8']} | {row['shift_-4']} | {row['shift_0_baseline']} | "
            f"{row['shift_+4']} | {row['shift_+8']} | {row['any_candidate_success']} | "
            f"{row['baseline_failure_rescued']} |"
        )
    lines.extend(["", "## Summary", ""])
    for row in task_summaries:
        lines.append(
            f"- Task {row['task_id']}: baseline {row['baseline_successes']}/{row['paired_initial_states']}; "
            f"oracle {row['timing_oracle_successes']}/{row['paired_initial_states']}; "
            f"recoverable failures {row['baseline_failures_recoverable']}/{row['baseline_failures']}."
        )
    lines.append(
        f"- Aggregate: baseline {total_baseline}/{total_states}; oracle {total_oracle}/{total_states}; "
        f"recoverable failures {total_recovered}/{total_failures}."
    )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
