#!/usr/bin/env python3
"""Convert cached ManiSkill state/action trajectories to official ACT H5.

The cached Gate-0 trajectories contain exact simulator states and the
expert action suffix.  This converter replays no actions: it restores each
state and asks the official task for its state observation, then writes the
H5 layout consumed by ManiSkill's ACT baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

import gymnasium as gym
import mani_skill.envs  # noqa: F401


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--task", required=True, choices=["PickCube-v1", "StackCube-v1"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-mode", default="pd_ee_pose")
    args = parser.parse_args()

    prefix = args.task.replace("-v1", "").lower()
    paths = sorted(args.input.glob(f"{prefix}_episode_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no .pt trajectories under {args.input}")

    env = gym.make(
        args.task,
        obs_mode="state",
        control_mode=args.control_mode,
        render_mode=None,
        sim_backend="physx_cpu",
        max_episode_steps=None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    episode_records = []
    try:
        with h5py.File(args.output, "w") as handle:
            for traj_idx, path in enumerate(paths):
                record = torch.load(path, map_location="cpu", weights_only=False)
                states = record["states"]
                actions = np.asarray(record["actions"], dtype=np.float32)
                if len(states) != len(actions) + 1:
                    raise ValueError(f"{path}: expected len(states)=len(actions)+1")

                env.reset(seed=int(record.get("seed", traj_idx)))
                observations = []
                for state in states:
                    env.unwrapped.set_state_dict(state)
                    obs = env.unwrapped.get_obs()
                    obs = obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor) else np.asarray(obs)
                    observations.append(np.asarray(obs[0], dtype=np.float32))
                observations = np.stack(observations, axis=0)

                group = handle.create_group(f"traj_{traj_idx}")
                group.create_dataset("obs", data=observations, compression="lzf")
                group.create_dataset("actions", data=actions, compression="lzf")
                group.create_dataset("success", data=np.asarray([int(record["success"])], dtype=np.int8))
                episode_records.append(
                    {
                        "traj_idx": traj_idx,
                        "source": str(path),
                        "seed": int(record.get("seed", traj_idx)),
                        "obs_shape": list(observations.shape),
                        "action_shape": list(actions.shape),
                        "success": bool(record["success"]),
                    }
                )
    finally:
        env.close()

    metadata = {
        "env_info": {
            "env_id": args.task,
            "env_kwargs": {
                "control_mode": args.control_mode,
                "obs_mode": "state",
                "sim_backend": "physx_cpu",
            },
        },
        "episodes": episode_records,
        "source": "ManiSkill Gate-0 cached exact states and expert actions",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "episodes": episode_records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
