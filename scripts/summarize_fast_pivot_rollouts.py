#!/usr/bin/env python3
"""Aggregate exploratory rollout artifacts without discarding failed episodes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def wilson(successes: int, episodes: int, z: float = 1.959963984540054) -> list[float]:
    rate = successes / episodes
    denominator = 1.0 + z * z / episodes
    center = (rate + z * z / (2.0 * episodes)) / denominator
    radius = z * math.sqrt(
        rate * (1.0 - rate) / episodes + z * z / (4.0 * episodes * episodes)
    ) / denominator
    return [center - radius, center + radius]


def main() -> None:
    args = parse_args()
    rows = []
    for summary_path in sorted(args.input_dir.glob("*/summary.json")):
        run_dir = summary_path.parent
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        episodes = [
            json.loads(line)
            for line in (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        successes = int(summary["successes"])
        count = int(summary["episodes"])
        # The output directory is created immediately before rollout logging; the
        # summary is created after all episodes and env.close(). This excludes model setup.
        birth_seconds = float(
            subprocess.run(
                ["stat", "-c", "%W", str(run_dir)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        rollout_wall_seconds = summary_path.stat().st_mtime - birth_seconds
        rows.append(
            {
                "run": run_dir.name,
                "task_id": int(metadata["task_id"]),
                "task_name": metadata["task_name"],
                "method": metadata["post_policy"],
                "correction_scale": metadata.get("correction_scale"),
                "gate_threshold": metadata.get("gate_threshold"),
                "episodes": count,
                "successes": successes,
                "success_rate": successes / count,
                "wilson_ci95": wilson(successes, count),
                "episode_outcomes": [bool(row["success"]) for row in episodes],
                "environment_steps": int(summary["environment_steps"]),
                "policy_queries": int(summary["policy_queries"]),
                "gate_activation_rate": summary.get("gate_activation_rate"),
                "mean_chunk_correction_norm": summary.get("mean_chunk_correction_norm"),
                "rollout_wall_seconds": rollout_wall_seconds,
                "runtime_measurement": "summary mtime minus output-directory birth time; excludes model/env setup",
            }
        )
    result = {
        "status": "exploratory_screen; task set chosen from historical performance before this screen",
        "all_completed_runs_included": True,
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
