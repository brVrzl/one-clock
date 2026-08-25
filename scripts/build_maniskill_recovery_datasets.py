#!/usr/bin/env python3
"""Build matched ACT datasets from successful corrective trajectories.

All selectors consume the same pre-generated recovery pool and add exactly
the same number of successful branches.  The common pool attempt count is
reported separately from selected-branch count so simulator cost cannot be
hidden inside the data budget.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch


METHODS = ("random", "goal", "motion", "fragility")


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in (None, "") else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["PickCube-v1", "StackCube-v1"])
    parser.add_argument("--base-h5", type=Path, required=True)
    parser.add_argument("--branch-dir", type=Path, default=Path("experiments/counterfactual_tournament/maniskill_recovery_branches"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/counterfactual_tournament/act_data/recovery"))
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with (args.branch_dir / "recovery_branches.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["task"] != args.task or int(row["full_success"]) != 1:
                continue
            if int(row["bridge_reduces_position_error"]) != 1 or int(row["teleport_violation"]) != 0:
                continue
            if not row.get("branch_file"):
                continue
            row["episode"] = int(row["episode"])
            row["timestep"] = int(row["timestep"])
            row["phase"] = float(row["phase"])
            for key in ("initial_position_error_m", "final_position_error_m"):
                row[key] = float(row[key])
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no eligible successful corrective branches for {args.task}")

    # One branch per source state makes the comparison about state allocation.
    by_state: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_state[(row["episode"], row["timestep"])].append(row)
    state_rows = {key: sorted(value, key=lambda row: (int(row["k"]), row["perturbation_type"])) for key, value in by_state.items()}
    if len(state_rows) < args.budget:
        raise RuntimeError(f"only {len(state_rows)} eligible source states, budget={args.budget}")

    score_path = Path("experiments/counterfactual_tournament/maniskill_fragility_sweep/fragility_state_scores.csv")
    scores: dict[tuple[str, int, int], dict[str, float]] = {}
    if score_path.exists():
        with score_path.open() as handle:
            for row in csv.DictReader(handle):
                scores[(row["task"], int(row["episode"]), int(row["timestep"]))] = {
                    key: as_float(row, key)
                    for key in ("fragility_score", "object_goal_distance", "action_magnitude")
                }

    def score(key: tuple[int, int], metric: str) -> float:
        row = state_rows[key][0]
        values = scores.get((args.task, key[0], key[1]), {})
        if metric == "phase":
            return float(row["phase"])
        return values.get(metric, float("nan"))

    state_keys = sorted(state_rows)
    rng = np.random.default_rng(args.seed)
    orders = {
        "random": [state_keys[i] for i in rng.permutation(len(state_keys))],
        "goal": sorted(state_keys, key=lambda key: score(key, "object_goal_distance"), reverse=True),
        "motion": sorted(state_keys, key=lambda key: score(key, "action_magnitude"), reverse=True),
        "fragility": sorted(state_keys, key=lambda key: score(key, "fragility_score"), reverse=True),
    }

    prefix = args.task.replace("-v1", "").lower()
    manifest = {
        "task": args.task,
        "base_h5": str(args.base_h5),
        "branch_dir": str(args.branch_dir),
        "methods": METHODS,
        "requested_successful_branch_budget": args.budget,
        "eligible_successful_branches": len(rows),
        "eligible_source_states": len(state_rows),
        "common_pool_attempts": sum(1 for row in csv.DictReader((args.branch_dir / "recovery_branches.csv").open()) if row["task"] == args.task),
        "seed": args.seed,
        "datasets": [],
    }
    for method in METHODS:
        selected = []
        states_considered = 0
        for key in orders[method]:
            states_considered += 1
            selected.append(state_rows[key][0])
            if len(selected) == args.budget:
                break
        if len(selected) != args.budget:
            raise RuntimeError(f"{method}: selected {len(selected)} branches, expected {args.budget}")

        output = args.output_dir / f"{prefix}.{method}recover.state.pd_ee_pose.cpu.h5"
        metadata = {
            "env_info": {
                "env_id": args.task,
                "env_kwargs": {"control_mode": "pd_ee_pose", "obs_mode": "state", "sim_backend": "physx_cpu"},
            },
            "base_demonstrations": 0,
            "augmentation_method": method,
            "successful_added_branches": len(selected),
            "common_pool_attempts": manifest["common_pool_attempts"],
            "selector_states_considered": states_considered,
            "bridge_generator": "bounded direct EEF correction to action[t+k], then suffix from t+k",
            "selected": [
                {key: row[key] for key in ("episode", "timestep", "k", "perturbation_type", "branch_file")}
                for row in selected
            ],
        }
        with h5py.File(args.base_h5, "r") as source, h5py.File(output, "w") as target:
            metadata["base_demonstrations"] = len(source)
            for key in sorted(source.keys(), key=lambda name: int(name.split("_")[-1])):
                source.copy(key, target, name=key)
            next_idx = len(source)
            for offset, row in enumerate(selected):
                branch = torch.load(row["branch_file"], map_location="cpu", weights_only=False)
                group = target.create_group(f"traj_{next_idx + offset}")
                group.create_dataset("obs", data=np.asarray(branch["observations"], dtype=np.float32), compression="lzf")
                group.create_dataset("actions", data=np.asarray(branch["actions"], dtype=np.float32), compression="lzf")
                group.create_dataset("success", data=np.asarray([1], dtype=np.int8))
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        manifest["datasets"].append({"method": method, "output": str(output), "added": len(selected), "states_considered": states_considered})
        print(json.dumps(manifest["datasets"][-1]))

    (args.output_dir / f"{prefix}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
