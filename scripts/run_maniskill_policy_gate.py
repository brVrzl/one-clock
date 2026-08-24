#!/usr/bin/env python3
"""Cheap matched UniformBC/CriticalBC/ContrastBC state-policy screen."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import mani_skill.envs  # noqa: F401
import gymnasium as gym


TASK_ACTOR = {"PickCube-v1": "cube", "StackCube-v1": "cubeA"}


def clone_state(value):
    if isinstance(value, dict):
        return {key: clone_state(item) for key, item in value.items()}
    return value.detach().clone() if isinstance(value, torch.Tensor) else value


def flatten_state(state):
    chunks = []
    for group in sorted(state):
        for name in sorted(state[group]):
            chunks.append(state[group][name].detach().cpu().numpy().reshape(-1))
    return np.concatenate(chunks).astype(np.float32)


def perturb_state(state, task, row):
    branch = clone_state(state)
    ptype = row["perturbation_type"]
    if ptype.startswith("eef_joint"):
        name = next(iter(branch["articulations"]))
        branch["articulations"][name][0, 1] += float(row["perturbation_dy"])
    else:
        actor = branch["actors"][TASK_ACTOR[task]]
        actor[0, :3] += torch.tensor(
            [float(row["perturbation_dx"]), float(row["perturbation_dy"]), float(row["perturbation_dz"])],
            dtype=actor.dtype,
        )
    return branch


class Policy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def load_task_data(root: Path, task: str):
    stem = "pickcube" if task == "PickCube-v1" else "stackcube"
    manifest_name = "pick_manifest.json" if task == "PickCube-v1" else "stack_manifest.json"
    manifest = json.loads((root / manifest_name).read_text())
    episodes = []
    for item in manifest["episodes"]:
        path = root / f"{stem}_episode_{int(item['episode']):03d}.pt"
        episodes.append(torch.load(path, map_location="cpu", weights_only=False))
    branch_name = "pick_timestep_branches.csv" if task == "PickCube-v1" else "stack_timestep_branches.csv"
    branch_path = root / branch_name
    with branch_path.open() as handle:
        branch_rows = list(csv.DictReader(handle))
    by_state = defaultdict(list)
    for row in branch_rows:
        key = (int(row["episode"]), int(row["timestep"]))
        row["criticality"] = float(row["criticality"])
        row["branch_success"] = int(row["branch_success"])
        row["branch_valid"] = int(row["branch_valid"])
        by_state[key].append(row)

    positives = []
    negatives = []
    for episode in episodes:
        for timestep, state in enumerate(episode["states"][:-1]):
            if timestep >= len(episode["actions"]):
                continue
            rows = by_state[(int(episode["episode"]), timestep)]
            c = float(rows[0]["criticality"]) if rows else 0.0
            positives.append((flatten_state(state), np.asarray(episode["actions"][timestep], dtype=np.float32), c))
            for row in rows:
                if row["perturbation_type"] != "zero" and row["branch_valid"] and not row["branch_success"] and c > 0:
                    negatives.append((flatten_state(perturb_state(state, task, row)), np.asarray(episode["actions"][timestep], dtype=np.float32), c))
    return positives, negatives


def fit_model(positives, negatives, method, steps, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    x = torch.tensor(np.stack([item[0] for item in positives]))
    y = torch.tensor(np.stack([item[1] for item in positives]))
    weights = torch.tensor([0.1 + item[2] for item in positives], dtype=torch.float32)
    mean_x, std_x = x.mean(0), x.std(0).clamp_min(1e-4)
    mean_y, std_y = y.mean(0), y.std(0).clamp_min(1e-4)
    xn, yn = (x - mean_x) / std_x, (y - mean_y) / std_y
    model = Policy(x.shape[1], y.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    neg_x = neg_y = neg_w = None
    if negatives:
        neg_x = (torch.tensor(np.stack([item[0] for item in negatives])) - mean_x) / std_x
        neg_y = (torch.tensor(np.stack([item[1] for item in negatives])) - mean_y) / std_y
        neg_w = torch.tensor([item[2] for item in negatives], dtype=torch.float32)
    for step in range(steps):
        pred = model(xn)
        per_item = ((pred - yn) ** 2).mean(1)
        if method == "UniformBC":
            loss = per_item.mean()
        elif method == "CriticalBC":
            loss = (per_item * weights).sum() / weights.sum()
        else:
            loss = per_item.mean()
            if neg_x is not None:
                neg_pred = model(neg_x)
                neg_error = ((neg_pred - neg_y) ** 2).mean(1)
                # Failed perturbed states should not receive the same action
                # as the paired expert state; this is the cheapest margin proxy.
                margin = torch.relu(per_item.mean() - neg_error + 0.25)
                loss = loss + 0.5 * (margin * (0.1 + neg_w)).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model, mean_x, std_x, mean_y, std_y


def evaluate(task, model, mean_x, std_x, mean_y, std_y, episodes, seed0):
    outcomes = []
    for offset in range(episodes):
        env = gym.make(task, obs_mode="state", control_mode="pd_ee_pose", render_mode=None, sim_backend="physx_cpu", max_episode_steps=None)
        env.reset(seed=seed0 + offset)
        success = False
        steps = 0
        try:
            for steps in range(60):
                state = flatten_state(env.unwrapped.get_state_dict())
                x = (torch.tensor(state)[None] - mean_x) / std_x
                action = (model(x)[0] * std_y + mean_y).detach().numpy().astype(np.float32)
                action = np.clip(action, env.action_space.low, env.action_space.high)
                _, _, terminated, truncated, _ = env.step(action)
                success = bool(env.unwrapped.evaluate()["success"][0].item())
                if success or bool(terminated[0]) or bool(truncated[0]):
                    break
        finally:
            env.close()
        outcomes.append({"seed": seed0 + offset, "success": int(success), "steps": steps + 1})
    return outcomes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("experiments/counterfactual_tournament/maniskill_gate0_final"))
    parser.add_argument("--output", type=Path, default=Path("experiments/counterfactual_tournament/maniskill_policy_gate"))
    parser.add_argument("--train-steps", type=int, default=1500)
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "train_steps": args.train_steps, "eval_episodes": args.eval_episodes, "tasks": {}}
    for task in TASK_ACTOR:
        positives, negatives = load_task_data(args.data, task)
        task_summary = {"positive_samples": len(positives), "failed_perturbed_samples": len(negatives), "methods": {}}
        for method in ("UniformBC", "CriticalBC", "ContrastBC"):
            model, mean_x, std_x, mean_y, std_y = fit_model(positives, negatives, method, args.train_steps, 0)
            checkpoint = args.output / f"{task.replace('-v1','').lower()}_{method}.pt"
            torch.save({"model": model.state_dict(), "mean_x": mean_x, "std_x": std_x, "mean_y": mean_y, "std_y": std_y}, checkpoint)
            outcomes = evaluate(task, model, mean_x, std_x, mean_y, std_y, args.eval_episodes, 100)
            (args.output / f"{task.replace('-v1','').lower()}_{method}_episodes.jsonl").write_text("\n".join(json.dumps(row) for row in outcomes) + "\n")
            task_summary["methods"][method] = {"checkpoint": str(checkpoint), "successes": int(sum(row["success"] for row in outcomes)), "success_rate": float(np.mean([row["success"] for row in outcomes])), "outcomes": outcomes}
        summary["tasks"][task] = task_summary
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
