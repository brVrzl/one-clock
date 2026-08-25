#!/usr/bin/env python3
"""Run the official ManiSkill ACT trainer on original demos only.

This wrapper keeps the ACT command explicit and records the learning-curve
configuration.  The trainer itself is the official ManiSkill ACT source
snapshot documented in research/maniskill_act_provenance.md.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--act-root", type=Path, default=Path("/tmp/maniskill_official_act"))
    parser.add_argument("--python", type=Path, default=Path("/home/wjq/workspace/venvs/maniskill_act/bin/python"))
    parser.add_argument("--task", default="PickCube-v1", choices=["PickCube-v1", "StackCube-v1"])
    parser.add_argument("--demo-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/maniskill_health_curve"))
    parser.add_argument("--total-iters", type=int, default=30000)
    parser.add_argument("--eval-freq", type=int, default=2000)
    parser.add_argument("--num-eval-episodes", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-demos", type=int, default=10)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-temporal-agg", action="store_true")
    args = parser.parse_args()

    run_name = args.run_name or f"{args.task.lower().replace('-v1', '')}_uniformact_original_{args.total_iters}"
    run_dir = Path("runs") / run_name
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.python), str(args.act_root / "train.py"),
        "--exp-name", run_name,
        "--env-id", args.task,
        "--demo-path", str(args.demo_path),
        "--control-mode", "pd_ee_pose",
        "--sim-backend", "physx_cpu",
        "--num-demos", str(args.num_demos),
        "--max-episode-steps", str(args.max_episode_steps),
        "--total-iters", str(args.total_iters),
        "--batch-size", "32",
        "--num-queries", "30",
        "--num-eval-envs", "1",
        "--num-eval-episodes", str(args.num_eval_episodes),
        "--eval-freq", str(args.eval_freq),
        "--log-freq", str(args.eval_freq),
        "--save-freq", str(args.eval_freq),
        "--seed", str(args.seed),
        "--no-capture-video", "--no-track",
    ]
    if args.no_temporal_agg:
        command.append("--no-temporal-agg")
    metadata = {
        "task": args.task,
        "dataset": "original expert demonstrations only",
        "demonstrations": args.num_demos,
        "control_mode": "pd_ee_pose",
        "sim_backend": "physx_cpu",
        "action_chunk": 30,
        "batch_size": 32,
        "total_iters": args.total_iters,
        "eval_freq": args.eval_freq,
        "num_eval_episodes": args.num_eval_episodes,
        "seed": args.seed,
        "command": command,
        "started_at": time.time(),
    }
    (run_dir / "command.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (run_dir / "stdout.log").open("w") as log:
        result = subprocess.run(command, cwd=Path.cwd(), env={"PYTHONPATH": str(args.act_root), **os.environ}, stdout=log, stderr=subprocess.STDOUT)
    metadata["returncode"] = result.returncode
    metadata["finished_at"] = time.time()
    (run_dir / "command.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
