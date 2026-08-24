#!/usr/bin/env python3
"""Measure multi-magnitude counterfactual fragility on cached Gate-0 demos."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from scripts.run_maniskill_counterfactual_gate0 import (
    TASKS,
    clone_state,
    make_env,
    perturbations,
    run_suffix,
    state_hash,
    tensor_to_numpy,
)


def load_heuristics(path: Path) -> dict[tuple[int, int], dict[str, float]]:
    result = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            key = (int(row["episode"]), int(row["timestep"]))
            if key not in result:
                result[key] = {
                    key_name: float(row[key_name])
                    for key_name in (
                        "phase", "action_magnitude", "action_velocity",
                        "action_acceleration", "gripper_transition",
                        "eef_object_distance", "object_goal_distance",
                    )
                }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("experiments/counterfactual_tournament/maniskill_gate0_final"))
    parser.add_argument("--output", type=Path, default=Path("experiments/counterfactual_tournament/maniskill_fragility_sweep"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--branches-per-state", type=int, default=6)
    parser.add_argument("--scales", nargs=3, type=float, default=[0.0015, 0.003, 0.006], metavar=("SMALL", "MEDIUM", "LARGE"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows = []
    manifest = {
        "backend": "ManiSkill 3.0.1",
        "sim_backend": "physx_cpu",
        "control_mode": "pd_ee_pose",
        "source": "cached successful Gate-0 demonstrations; original expert suffix",
        "scales_m": args.scales,
        "branches_per_state": args.branches_per_state,
        "sample_every": args.sample_every,
        "episodes": [],
    }

    for task in ("PickCube-v1", "StackCube-v1"):
        csv_name = "pick_timestep_branches.csv" if task.startswith("Pick") else "stack_timestep_branches.csv"
        heuristics = load_heuristics(args.input / csv_name)
        env = make_env(task)
        try:
            # set_state_dict restores physical state, while Gymnasium still
            # requires one reset to arm the wrapper before the first step.
            env.reset(seed=0)
            for episode in range(args.episodes):
                prefix = task.replace("-v1", "").lower()
                path = args.input / f"{prefix}_episode_{episode:03d}.pt"
                trajectory = torch.load(path, map_location="cpu", weights_only=False)
                actions = np.asarray(trajectory["actions"], dtype=np.float32)
                states = trajectory["states"]
                sampled = list(range(0, max(1, len(actions)), args.sample_every))
                sampled = sampled[: max(1, math.ceil(len(actions) / args.sample_every))]
                state_count = 0
                branch_count = 0
                invalid_count = 0
                for timestep in sampled:
                    state = states[timestep]
                    original_success, original_valid, original_error = run_suffix(env, state, actions[timestep:])
                    zero_again, zero_valid, zero_error = run_suffix(env, state, actions[timestep:])
                    if not (original_valid and zero_valid and original_success == zero_again):
                        raise RuntimeError(f"restore/suffix validation failed {task} ep={episode} t={timestep}")
                    h = heuristics[(episode, timestep)]
                    state_id = state_hash(state)
                    state_count += 1
                    for scale_index, magnitude in enumerate(args.scales):
                        specs = perturbations(state, task, magnitude)[: args.branches_per_state]
                        valid_successes = []
                        for perturb_type, delta, branch_state in specs:
                            success, valid, error = run_suffix(env, branch_state, actions[timestep:])
                            branch_count += 1
                            invalid_count += int(not valid)
                            if valid:
                                valid_successes.append(int(success))
                            rows.append({
                                "task": task,
                                "episode": episode,
                                "timestep": timestep,
                                "phase": h["phase"],
                                "state_id": state_id,
                                "scale_index": scale_index,
                                "scale_m": magnitude,
                                "perturbation_type": perturb_type,
                                "perturbation_magnitude": float(np.linalg.norm(delta)),
                                "perturbation_dx": float(delta[0]),
                                "perturbation_dy": float(delta[1]),
                                "perturbation_dz": float(delta[2]),
                                "branch_success": int(success),
                                "branch_valid": int(valid),
                                "branch_error": error or "",
                                "original_suffix_success": int(original_success),
                                "expert_success": int(trajectory["success"]),
                                "criticality_at_scale": np.nan,
                                **{key: h[key] for key in h if key != "phase"},
                            })
                        criticality = float(1.0 - np.mean(valid_successes)) if valid_successes else float("nan")
                        for row in rows[-len(specs):]:
                            row["criticality_at_scale"] = criticality
                manifest["episodes"].append({
                    "task": task,
                    "episode": episode,
                    "states": state_count,
                    "branches": branch_count,
                    "invalid_branches": invalid_count,
                    "source": str(path),
                })
        finally:
            env.close()

    if rows:
        with (args.output / "fragility_branches.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    manifest["rows"] = len(rows)
    manifest["runtime_sec"] = time.time() - started
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
