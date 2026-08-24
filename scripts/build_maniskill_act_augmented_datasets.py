#!/usr/bin/env python3
"""Build equal-budget ACT H5 datasets from successful cached branches."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

from scripts.run_maniskill_counterfactual_gate0 import make_env, perturbations


METHODS = ("random", "late", "goal_distance", "motion", "fragility")


def copy_group(source: h5py.Group, target: h5py.Group) -> None:
    for key, value in source.items():
        source.copy(value, target, name=key)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["PickCube-v1", "StackCube-v1"])
    parser.add_argument("--base-h5", type=Path, required=True)
    parser.add_argument("--branch-csv", type=Path, default=Path("experiments/counterfactual_tournament/maniskill_fragility_sweep/fragility_branches.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("experiments/counterfactual_tournament/maniskill_gate0_final"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/counterfactual_tournament/act_data/augmented"))
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    score_path = args.branch_csv.parent / "fragility_state_scores.csv"
    scores = {}
    if score_path.exists():
        with score_path.open() as handle:
            for row in csv.DictReader(handle):
                scores[(int(row["episode"]), int(row["timestep"]))] = float(row["fragility_score"])

    all_rows = []
    with args.branch_csv.open() as handle:
        for row in csv.DictReader(handle):
            if row["task"] != args.task or abs(float(row["scale_m"]) - 0.003) > 1e-9:
                continue
            row["episode"] = int(row["episode"])
            row["timestep"] = int(row["timestep"])
            row["phase"] = float(row["phase"])
            for key in ("object_goal_distance", "action_magnitude"):
                row[key] = float(row[key])
            row["branch_success"] = int(row["branch_success"])
            row["branch_valid"] = int(row["branch_valid"])
            row["fragility_score"] = scores[(row["episode"], row["timestep"])]
            all_rows.append(row)

    by_state = defaultdict(list)
    for row in all_rows:
        by_state[(row["episode"], row["timestep"])].append(row)
    eligible = {
        key: rows for key, rows in by_state.items()
        if any(row["branch_valid"] and row["branch_success"] for row in rows)
    }
    if len(eligible) < args.budget:
        raise RuntimeError(f"only {len(eligible)} eligible states, budget={args.budget}")

    trajectories = {}
    prefix = args.task.replace("-v1", "").lower()
    for episode in sorted({key[0] for key in eligible}):
        trajectories[episode] = torch.load(
            args.raw_dir / f"{prefix}_episode_{episode:03d}.pt",
            map_location="cpu", weights_only=False,
        )

    state_keys = list(eligible)
    rng = np.random.default_rng(args.seed)
    orders = {
        "random": [state_keys[i] for i in rng.permutation(len(state_keys))],
        "late": sorted(state_keys, key=lambda key: max(float(row["phase"]) for row in eligible[key]), reverse=True),
        "goal_distance": sorted(state_keys, key=lambda key: max(float(row["object_goal_distance"]) for row in eligible[key]), reverse=True),
        "motion": sorted(state_keys, key=lambda key: max(float(row["action_magnitude"]) for row in eligible[key]), reverse=True),
        "fragility": sorted(state_keys, key=lambda key: max(float(row["fragility_score"]) for row in eligible[key]), reverse=True),
    }

    env = make_env(args.task)
    try:
        env.reset(seed=0)
        for method in METHODS:
            selected = []
            attempted = 0
            for key in orders[method]:
                attempted += 1
                candidates = [row for row in eligible[key] if row["branch_valid"] and row["branch_success"]]
                if not candidates:
                    continue
                # One branch per selected state keeps the budget focused on
                # state coverage rather than repeated perturbation directions.
                selected.append(candidates[0])
                if len(selected) == args.budget:
                    break

            output = args.output_dir / f"{prefix}.{method}aug.state.pd_ee_pose.cpu.h5"
            metadata = {
                "env_info": {"env_id": args.task, "env_kwargs": {"control_mode": "pd_ee_pose", "obs_mode": "state", "sim_backend": "physx_cpu"}},
                "base_demonstrations": 10,
                "augmentation_method": method,
                "added_branch_budget": args.budget,
                "selected_states": [{"episode": row["episode"], "timestep": row["timestep"], "state_id": row["state_id"]} for row in selected],
                "branch_attempts_until_budget": attempted,
            }
            with h5py.File(args.base_h5, "r") as source, h5py.File(output, "w") as target:
                for key in sorted(source.keys(), key=lambda name: int(name.split("_")[-1])):
                    source.copy(key, target, name=key)
                next_idx = len(source)
                for branch_idx, row in enumerate(selected):
                    trajectory = trajectories[row["episode"]]
                    state = trajectory["states"][row["timestep"]]
                    perturb = next(item for item in perturbations(state, args.task, 0.003) if item[0] == row["perturbation_type"])
                    branch_state = perturb[2]
                    suffix = np.asarray(trajectory["actions"][row["timestep"]:], dtype=np.float32)
                    env.unwrapped.set_state_dict(branch_state)
                    observations = []
                    obs = env.unwrapped.get_obs()
                    observations.append(obs.detach().cpu().numpy()[0].astype(np.float32))
                    for action in suffix:
                        env.step(action.copy())
                        obs = env.unwrapped.get_obs()
                        observations.append(obs.detach().cpu().numpy()[0].astype(np.float32))
                    group = target.create_group(f"traj_{next_idx + branch_idx}")
                    group.create_dataset("obs", data=np.stack(observations), compression="lzf")
                    group.create_dataset("actions", data=suffix, compression="lzf")
                    group.create_dataset("success", data=np.asarray([1], dtype=np.int8))
            output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(json.dumps({"method": method, "output": str(output), "added": len(selected), "branch_attempts": attempted}))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
