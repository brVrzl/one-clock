#!/usr/bin/env python3
"""Cache additional successful ManiSkill scripted expert trajectories.

This is only a data-volume health-gate control.  It does not alter the
counterfactual fork data or introduce a research method.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_maniskill_counterfactual_gate0 import execute_expert, make_env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["PickCube-v1", "StackCube-v1"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--hold-steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = make_env(args.task)
    successes = 0
    try:
        prefix = args.task.replace("-v1", "").lower()
        for episode in range(args.episodes):
            trajectory = execute_expert(env, args.task, args.seed + episode, args.hold_steps)
            successes += int(trajectory["success"])
            torch.save(
                {
                    "task": args.task,
                    "episode": episode,
                    "seed": trajectory["seed"],
                    "actions": np.asarray(trajectory["actions"], dtype=np.float32),
                    "states": trajectory["states"],
                    "success": trajectory["success"],
                },
                args.output / f"{prefix}_episode_{episode:03d}.pt",
            )
    finally:
        env.close()
    print({"task": args.task, "episodes": args.episodes, "successful": successes, "output": str(args.output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
