#!/usr/bin/env python3
"""Fit task-specific arm/gripper temporal kernels from training demos only."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.optimize import minimize

from research.audit_tools.audit_robotwin_act_recorded_sequence import CAMERAS, build_policy, infer_chunk
from research.audit_tools.robotwin_temporal_reuse import postprocess_action


ARM = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPPER = np.asarray([6, 13])
LAGS = 50
TRAIN_EPISODES = tuple(range(40))
HELDOUT_EPISODES = tuple(range(40, 50))


def fit_simplex(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    gram = (x.T @ x) / len(x)
    cross = (x.T @ y) / len(x)

    def objective(weights: np.ndarray) -> float:
        return float(weights @ gram @ weights - 2 * cross @ weights)

    def gradient(weights: np.ndarray) -> np.ndarray:
        return 2 * (gram @ weights - cross)

    result = minimize(
        objective,
        np.full(LAGS, 1 / LAGS),
        jac=gradient,
        bounds=[(0.0, 1.0)] * LAGS,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0, "jac": lambda _: np.ones(LAGS)},
        method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"kernel optimization failed: {result.message}")
    weights = np.maximum(result.x, 0)
    return weights / weights.sum()


def compose(candidates: np.ndarray, arm_kernel: np.ndarray, gripper_kernel: np.ndarray) -> np.ndarray:
    action = np.empty(14, dtype=np.float32)
    action[ARM] = arm_kernel @ candidates[:, ARM]
    action[GRIPPER] = gripper_kernel @ candidates[:, GRIPPER]
    return action


def episode_design(policy, stats: dict, episode_path: Path, device: torch.device):
    with h5py.File(episode_path, "r") as episode:
        length = len(episode["action"])
        actions = np.asarray(episode["action"])
        chunks = []
        for step in range(length):
            qpos = np.asarray(episode["observations/qpos"][step])
            images = np.stack(
                [
                    np.moveaxis(np.asarray(episode[f"observations/images/{camera}"][step]), -1, 0) / 255.0
                    for camera in CAMERAS
                ]
            )
            normalized = infer_chunk(policy, qpos, images, stats, device)
            chunks.append(postprocess_action(normalized, stats["action_mean"], stats["action_std"]))
    arm_x, arm_y, grip_x, grip_y, all_x, all_y = [], [], [], [], [], []
    first_candidates = None
    for target in range(LAGS - 1, length):
        candidates = np.stack([chunks[target - lag][lag] for lag in range(LAGS)])
        first_candidates = candidates if first_candidates is None else first_candidates
        arm_x.append(candidates[:, ARM].T)
        arm_y.append(actions[target, ARM])
        grip_x.append(candidates[:, GRIPPER].T)
        grip_y.append(actions[target, GRIPPER])
        all_x.append(candidates.T)
        all_y.append(actions[target])
    return (
        np.concatenate(arm_x), np.concatenate(arm_y),
        np.concatenate(grip_x), np.concatenate(grip_y),
        np.concatenate(all_x), np.concatenate(all_y), first_candidates,
    )


def collect(policy, stats: dict, data_root: Path, episodes: tuple[int, ...], device: torch.device):
    grouped = [[] for _ in range(6)]
    canary = None
    for episode_id in episodes:
        values = episode_design(policy, stats, data_root / f"episode_{episode_id}.hdf5", device)
        for index in range(6):
            grouped[index].append(values[index])
        canary = values[6] if canary is None else canary
    return tuple(np.concatenate(items) for items in grouped), canary


def mse(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    return float(np.mean((x @ weights - y) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--act-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)
    policy = build_policy(args.act_root, args.checkpoint_dir, device)
    with (args.checkpoint_dir / "dataset_stats.pkl").open("rb") as handle:
        stats = pickle.load(handle)
    train, _ = collect(policy, stats, args.data_root, TRAIN_EPISODES, device)
    heldout, canary_candidates = collect(policy, stats, args.data_root, HELDOUT_EPISODES, device)
    train_arm_x, train_arm_y, train_grip_x, train_grip_y, train_all_x, train_all_y = train
    test_arm_x, test_arm_y, test_grip_x, test_grip_y, test_all_x, test_all_y = heldout
    arm_kernel = fit_simplex(train_arm_x, train_arm_y)
    gripper_kernel = fit_simplex(train_grip_x, train_grip_y)
    shared_kernel = fit_simplex(train_all_x, train_all_y)
    composed = compose(canary_candidates, arm_kernel, gripper_kernel)
    expected = np.empty(14, dtype=np.float32)
    expected[ARM] = arm_kernel @ canary_candidates[:, ARM]
    expected[GRIPPER] = gripper_kernel @ canary_candidates[:, GRIPPER]
    output = {
        "task": args.task,
        "fit_source": "official training demonstrations only",
        "train_episodes": list(TRAIN_EPISODES),
        "heldout_episodes": list(HELDOUT_EPISODES),
        "lag_count": LAGS,
        "arm_indices": ARM.tolist(),
        "gripper_indices": GRIPPER.tolist(),
        "arm_kernel": arm_kernel.tolist(),
        "gripper_kernel": gripper_kernel.tolist(),
        "shared_kernel": shared_kernel.tolist(),
        "heldout_reconstruction_mse": {
            "arm_component_kernel": mse(test_arm_x, test_arm_y, arm_kernel),
            "arm_shared_kernel": mse(test_arm_x, test_arm_y, shared_kernel),
            "arm_newest": mse(test_arm_x, test_arm_y, np.eye(1, LAGS, 0).ravel()),
            "gripper_component_kernel": mse(test_grip_x, test_grip_y, gripper_kernel),
            "gripper_shared_kernel": mse(test_grip_x, test_grip_y, shared_kernel),
            "gripper_newest": mse(test_grip_x, test_grip_y, np.eye(1, LAGS, 0).ravel()),
        },
        "recorded_sequence_provenance_canary": {
            "max_absolute_composition_difference": float(np.max(np.abs(composed - expected))),
            "passed": bool(np.array_equal(composed, expected)),
        },
        "rollout_success_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
