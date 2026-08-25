#!/usr/bin/env python3
"""Generate short, bounded corrective bridges from ManiSkill fork states.

The original successful-suffix augmentation is intentionally not used here:
an object or robot-state perturbation can change the correct action.  This
script instead moves the EEF toward a future expert action waypoint using
bounded absolute ``pd_ee_pose`` commands, then replays the suffix from that
future index.  Every attempted branch receives an explicit validity,
teleport, error-reduction, and final-success label.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import sapien
import torch
from scipy.spatial.transform import Rotation

import mani_skill.envs  # noqa: F401

# Allow direct ``python scripts/<file>.py`` execution from the repository
# root, matching the commands recorded in the experiment manifests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_maniskill_counterfactual_gate0 import (
    TASKS,
    clone_state,
    make_env,
    perturbations,
    reset_clock,
    state_hash,
    tensor_to_numpy,
)


def pose_from_action(base_env: Any, action: np.ndarray) -> sapien.Pose:
    """Convert an absolute pd_ee_pose action to a world-frame EEF pose."""
    robot_pose = base_env.agent.robot.pose
    robot_pose = robot_pose if isinstance(robot_pose, sapien.Pose) else robot_pose.sp
    q = Rotation.from_euler("xyz", np.asarray(action[3:6], dtype=np.float64)).as_quat()
    target_in_base = sapien.Pose(
        p=np.asarray(action[:3], dtype=np.float64),
        q=np.asarray([q[3], q[0], q[1], q[2]], dtype=np.float64),
    )
    return robot_pose * target_in_base


def current_eef_pose(base_env: Any) -> sapien.Pose:
    robot_pose = base_env.agent.robot.pose
    tcp_pose = base_env.agent.tcp.pose
    robot_pose = robot_pose if isinstance(robot_pose, sapien.Pose) else robot_pose.sp
    tcp_pose = tcp_pose if isinstance(tcp_pose, sapien.Pose) else tcp_pose.sp
    return robot_pose.inv() * tcp_pose


def action_from_pose(base_env: Any, pose_in_base: sapien.Pose, gripper: float) -> np.ndarray:
    q = np.asarray(pose_in_base.q, dtype=np.float64)
    euler = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")
    return np.asarray(np.r_[pose_in_base.p, euler, gripper], dtype=np.float32)


def pose_errors(current: sapien.Pose, target: sapien.Pose) -> tuple[float, float]:
    delta = target * current.inv()
    position = float(np.linalg.norm(np.asarray(delta.p, dtype=np.float64)))
    q = np.asarray(delta.q, dtype=np.float64)
    rotation = Rotation.from_quat([q[1], q[2], q[3], q[0]])
    return position, float(np.linalg.norm(rotation.as_rotvec()))


def bounded_pose_step(
    base_env: Any,
    target: sapien.Pose,
    gripper: float,
    max_translation: float,
    max_rotation: float,
) -> tuple[np.ndarray, float, float]:
    """Return one non-teleporting absolute pose command toward ``target``."""
    current = current_eef_pose(base_env)
    delta = target * current.inv()
    dp = np.asarray(delta.p, dtype=np.float64)
    dp_norm = float(np.linalg.norm(dp))
    if dp_norm > max_translation:
        dp = dp * (max_translation / dp_norm)
    dq = np.asarray(delta.q, dtype=np.float64)
    drot = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]]).as_rotvec()
    drot_norm = float(np.linalg.norm(drot))
    if drot_norm > max_rotation:
        drot = drot * (max_rotation / drot_norm)
    q = Rotation.from_rotvec(drot).as_quat()
    step = sapien.Pose(
        p=np.asarray(current.p, dtype=np.float64) + dp,
        q=np.asarray([q[3], q[0], q[1], q[2]], dtype=np.float64),
    )
    return action_from_pose(base_env, step, gripper), float(np.linalg.norm(dp)), float(np.linalg.norm(drot))


def obs_numpy(base_env: Any) -> np.ndarray:
    obs = base_env.get_obs()
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()
    return np.asarray(obs[0], dtype=np.float32)


def evaluate_branch(
    env: Any,
    trajectory: dict[str, Any],
    timestep: int,
    perturbation_type: str,
    branch_state: dict[str, Any],
    k: int,
    max_translation: float,
    max_rotation: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run one bridge candidate and return its row and successful trajectory."""
    base_env = env.unwrapped
    actions = np.asarray(trajectory["actions"], dtype=np.float32)
    target_idx = min(timestep + k, len(actions) - 1)
    target = pose_from_action(base_env, actions[target_idx])
    row: dict[str, Any] = {
        "timestep": timestep,
        "target_timestep": target_idx,
        "k": k,
        "perturbation_type": perturbation_type,
        "state_id": state_hash(trajectory["states"][timestep]),
        "branch_state_id": state_hash(branch_state),
        "branch_valid": 0,
        "teleport_violation": 0,
        "bridge_reduces_position_error": 0,
        "bridge_reduces_orientation_error": 0,
        "full_success": 0,
        "error": "",
    }
    try:
        base_env.set_state_dict(clone_state(branch_state))
        reset_clock(base_env)
        before = current_eef_pose(base_env)
        before_pos, before_rot = pose_errors(before, target)
        bridge_actions: list[np.ndarray] = []
        observations: list[np.ndarray] = [obs_numpy(base_env)]
        commanded_translation: list[float] = []
        commanded_rotation: list[float] = []
        for _ in range(k):
            command, dp_norm, drot_norm = bounded_pose_step(
                base_env, target, float(actions[target_idx, -1]), max_translation, max_rotation
            )
            commanded_translation.append(dp_norm)
            commanded_rotation.append(drot_norm)
            if dp_norm > max_translation + 1e-6 or drot_norm > max_rotation + 1e-6:
                row["teleport_violation"] = 1
            bridge_actions.append(command)
            _, _, terminated, truncated, _ = env.step(command)
            observations.append(obs_numpy(base_env))
            if bool(tensor_to_numpy(terminated)[0]) or bool(tensor_to_numpy(truncated)[0]):
                break
        after = current_eef_pose(base_env)
        after_pos, after_rot = pose_errors(after, target)
        row.update(
            {
                "branch_valid": 1,
                "initial_position_error_m": before_pos,
                "final_position_error_m": after_pos,
                "initial_orientation_error_rad": before_rot,
                "final_orientation_error_rad": after_rot,
                "max_command_translation_m": max(commanded_translation, default=0.0),
                "max_command_rotation_rad": max(commanded_rotation, default=0.0),
                "bridge_reduces_position_error": int(after_pos < before_pos - 1e-5),
                "bridge_reduces_orientation_error": int(after_rot < before_rot - 1e-5),
            }
        )
        if len(bridge_actions) != k:
            row["error"] = "terminated_during_bridge"
            return row, None
        suffix = [np.asarray(a, dtype=np.float32).copy() for a in actions[target_idx:]]
        for action in suffix:
            _, _, terminated, truncated, _ = env.step(action)
            observations.append(obs_numpy(base_env))
            if bool(tensor_to_numpy(terminated)[0]) or bool(tensor_to_numpy(truncated)[0]):
                break
        success = bool(tensor_to_numpy(base_env.evaluate()["success"])[0])
        row["full_success"] = int(success)
        if success and row["teleport_violation"] == 0 and row["bridge_reduces_position_error"]:
            branch = {
                "observations": np.stack(observations),
                "actions": np.stack(bridge_actions + suffix).astype(np.float32),
                "success": True,
                "metadata": dict(row),
            }
            return row, branch
        return row, None
    except Exception as exc:  # invalid/interpenetrating branches are separate
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["PickCube-v1", "StackCube-v1"], choices=["PickCube-v1", "StackCube-v1"])
    parser.add_argument("--input-dir", type=Path, default=Path("experiments/counterfactual_tournament/maniskill_gate0_final"))
    parser.add_argument("--output", type=Path, default=Path("experiments/counterfactual_tournament/maniskill_recovery_branches"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--magnitude", type=float, default=0.003)
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--max-translation", type=float, default=0.02)
    parser.add_argument("--max-rotation", type=float, default=0.12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "backend": "ManiSkill 3.0.1",
        "control_mode": "pd_ee_pose",
        "bridge": "bounded direct EEF pose correction to action[t+k], then original suffix from t+k",
        "tasks": args.tasks,
        "episodes_requested": args.episodes,
        "sample_every": args.sample_every,
        "perturbation_magnitude_m": args.magnitude,
        "k_values": args.k_values,
        "max_translation_m": args.max_translation,
        "max_rotation_rad": args.max_rotation,
        "episodes": [],
    }
    for task in args.tasks:
        env = make_env(task)
        try:
            prefix = task.replace("-v1", "").lower()
            for episode in range(args.episodes):
                path = args.input_dir / f"{prefix}_episode_{episode:03d}.pt"
                if not path.exists():
                    raise FileNotFoundError(path)
                trajectory = torch.load(path, map_location="cpu", weights_only=False)
                actions = np.asarray(trajectory["actions"], dtype=np.float32)
                # The wrapper must be reset once before direct state restores;
                # set_state_dict restores simulator state but does not mark
                # Gymnasium's reset-before-step contract as satisfied.
                env.reset(seed=int(trajectory.get("seed", episode)))
                sampled = list(range(0, len(actions), args.sample_every))
                episode_rows: list[dict[str, Any]] = []
                episode_successes = 0
                episode_started = time.time()
                for timestep in sampled:
                    state = trajectory["states"][timestep]
                    for perturb_type, _, branch_state in perturbations(state, task, args.magnitude):
                        for k in args.k_values:
                            row, branch = evaluate_branch(
                                env, trajectory, timestep, perturb_type, branch_state, k,
                                args.max_translation, args.max_rotation,
                            )
                            row.update({"task": task, "episode": episode, "phase": timestep / max(1, len(actions) - 1)})
                            episode_rows.append(row)
                            if branch is not None:
                                episode_successes += 1
                                branch_id = f"{prefix}_episode_{episode:03d}_t{timestep:04d}_{perturb_type}_k{k}_{episode_successes:04d}"
                                torch.save(branch, args.output / f"{branch_id}.pt")
                                row["branch_file"] = str(args.output / f"{branch_id}.pt")
                rows.extend(episode_rows)
                manifest["episodes"].append(
                    {
                        "task": task,
                        "episode": episode,
                        "action_count": int(len(actions)),
                        "sampled_states": len(sampled),
                        "attempts": len(episode_rows),
                        "valid": int(sum(r["branch_valid"] for r in episode_rows)),
                        "invalid": int(sum(not r["branch_valid"] for r in episode_rows)),
                        "teleport_violations": int(sum(r["teleport_violation"] for r in episode_rows)),
                        "full_successes": int(sum(r["full_success"] for r in episode_rows)),
                        "eligible_successful_bridges": episode_successes,
                        "runtime_sec": time.time() - episode_started,
                    }
                )
        finally:
            env.close()
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with (args.output / "recovery_branches.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    manifest["runtime_sec"] = time.time() - started
    manifest["attempts"] = len(rows)
    manifest["valid"] = int(sum(r["branch_valid"] for r in rows))
    manifest["invalid"] = int(sum(not r["branch_valid"] for r in rows))
    manifest["teleport_violations"] = int(sum(r["teleport_violation"] for r in rows))
    manifest["full_successes"] = int(sum(r["full_success"] for r in rows))
    manifest["eligible_successful_bridges"] = int(sum(bool(r.get("branch_file")) for r in rows))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
