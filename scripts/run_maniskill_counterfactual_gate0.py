#!/usr/bin/env python3
"""ManiSkill Gate 0: local state perturbations followed by expert suffixes.

This backend deliberately keeps the causal fork separate from the RoboTwin
scaffold.  Expert trajectories are generated with ManiSkill's official task
implementations and bundled Panda IK controller.  The official mplib motion
planner is probed separately; on the current host its native Planner
constructor segfaults, so this script records the fallback source explicitly.

The raw output is intended to be inspectable, not a benchmark implementation:
each episode has actions, full state snapshots, and a branch-level CSV.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import sapien
import torch
from scipy.spatial.transform import Rotation

import mani_skill.envs  # noqa: F401: registers official ManiSkill tasks
import gymnasium as gym
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)


TASKS = ("PickCube-v1", "StackCube-v1", "PegInsertionSide-v1")
TASK_ACTOR = {
    "PickCube-v1": "cube",
    "StackCube-v1": "cubeA",
    "PegInsertionSide-v1": "peg",
}


def tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def clone_state(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clone_state(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return copy.deepcopy(value)


def state_hash(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()

    def visit(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], f"{prefix}{key}/")
            return
        array = tensor_to_numpy(value)
        digest.update(prefix.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(repr(array.shape).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())

    visit(state)
    return digest.hexdigest()


def state_get_actor(state: dict[str, Any], actor_name: str) -> torch.Tensor:
    return state["actors"][actor_name]


def make_env(task: str):
    # max_episode_steps=None prevents the wrapper clock from truncating a
    # suffix after a state restore.  The physical task still supplies success.
    return gym.make(
        task,
        obs_mode="state",
        control_mode="pd_ee_pose",
        render_mode=None,
        sim_backend="physx_cpu",
        max_episode_steps=None,
    )


def as_sapien_pose(pose: Any) -> sapien.Pose:
    if isinstance(pose, sapien.Pose):
        return pose
    if hasattr(pose, "sp"):
        return pose.sp
    # Batched ManiSkill poses expose p/q tensors; Gate 0 uses one env.
    p = tensor_to_numpy(pose.p)
    q = tensor_to_numpy(pose.q)
    if p.ndim == 2:
        p = p[0]
    if q.ndim == 2:
        q = q[0]
    return sapien.Pose(p=p, q=q)


def pose_action(base_env: Any, pose: Any, gripper: float) -> np.ndarray:
    target = as_sapien_pose(pose)
    robot_pose = as_sapien_pose(base_env.agent.robot.pose)
    target_in_base = robot_pose.inv() * target
    q = tensor_to_numpy(target_in_base.q)
    euler = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")
    return np.asarray(
        np.r_[tensor_to_numpy(target_in_base.p), euler, gripper], dtype=np.float32
    )


def task_waypoints(base_env: Any, task: str) -> list[tuple[str, np.ndarray, float]]:
    """Return a short deterministic expert action sequence.

    The action-space and task geometry are official ManiSkill APIs.  The
    bundled IK controller avoids the unavailable native mplib constructor.
    """
    if task == "PickCube-v1":
        actor = base_env.cube
        obb = get_actor_obb(actor)
        approaching = np.array([0.0, 0.0, -1.0])
        closing = tensor_to_numpy(
            base_env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1]
        )
        grasp = compute_grasp_info_by_obb(
            obb, approaching=approaching, target_closing=closing, depth=0.025
        )
        grasp_pose = base_env.agent.build_grasp_pose(
            approaching, grasp["closing"], actor.pose.sp.p
        )
        goal = tensor_to_numpy(base_env.goal_site.pose.p)[0]
        return [
            ("reach", pose_action(base_env, grasp_pose * sapien.Pose([0, 0, -0.07]), 1), 1),
            ("grasp", pose_action(base_env, grasp_pose, 1), 1),
            ("close", pose_action(base_env, grasp_pose, -1), -1),
            ("lift", pose_action(base_env, sapien.Pose([0, 0, 0.18]) * grasp_pose, -1), -1),
            ("place", pose_action(base_env, sapien.Pose(goal, grasp_pose.q), -1), -1),
        ]

    if task == "StackCube-v1":
        actor = base_env.cubeA
        obb = get_actor_obb(actor)
        approaching = np.array([0.0, 0.0, -1.0])
        closing = tensor_to_numpy(
            base_env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1]
        )
        grasp = compute_grasp_info_by_obb(
            obb, approaching=approaching, target_closing=closing, depth=0.025
        )
        grasp_pose = base_env.agent.build_grasp_pose(
            approaching, grasp["closing"], actor.pose.sp.p
        )
        cube_b = tensor_to_numpy(base_env.cubeB.pose.p)[0]
        stack_center = cube_b + np.array([0.0, 0.0, 0.04])
        return [
            ("reach", pose_action(base_env, grasp_pose * sapien.Pose([0, 0, -0.07]), 1), 1),
            ("grasp", pose_action(base_env, grasp_pose, 1), 1),
            ("close", pose_action(base_env, grasp_pose, -1), -1),
            ("lift", pose_action(base_env, sapien.Pose([0, 0, 0.14]) * grasp_pose, -1), -1),
            ("align", pose_action(base_env, sapien.Pose(stack_center + [0, 0, 0.08], grasp_pose.q), -1), -1),
            ("lower", pose_action(base_env, sapien.Pose(stack_center, grasp_pose.q), -1), -1),
            ("release", pose_action(base_env, sapien.Pose(stack_center, grasp_pose.q), 1), 1),
            ("retreat", pose_action(base_env, sapien.Pose(stack_center + [0, 0, 0.08], grasp_pose.q), 1), 1),
        ]

    if task == "PegInsertionSide-v1":
        actor = base_env.peg
        obb = get_actor_obb(actor)
        approaching = np.array([0.0, 0.0, -1.0])
        closing = tensor_to_numpy(
            base_env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1]
        )
        grasp = compute_grasp_info_by_obb(
            obb, approaching=approaching, target_closing=closing, depth=0.025
        )
        grasp_pose = base_env.agent.build_grasp_pose(
            approaching, grasp["closing"], actor.pose.sp.p
        )
        grasp_pose = grasp_pose * sapien.Pose(
            [-max(0.05, float(tensor_to_numpy(base_env.peg_half_sizes)[0, 0] / 2 + 0.01)), 0, 0]
        )
        peg_init_pose = actor.pose.sp
        insert_pose = as_sapien_pose(base_env.goal_pose) * peg_init_pose.inv() * grasp_pose
        offset = sapien.Pose([-0.01 - float(tensor_to_numpy(base_env.peg_half_sizes)[0, 0]), 0, 0])
        pre_insert = insert_pose * offset
        actions = [
            ("reach", pose_action(base_env, grasp_pose * sapien.Pose([0, 0, -0.05]), 1), 1),
            ("grasp", pose_action(base_env, grasp_pose, 1), 1),
            ("close", pose_action(base_env, grasp_pose, -1), -1),
            ("pre_insert", pose_action(base_env, pre_insert, -1), -1),
        ]
        for i in range(3):
            delta = as_sapien_pose(base_env.goal_pose) * offset * actor.pose.sp.inv()
            pre_insert = delta * pre_insert
            actions.append((f"refine_{i}", pose_action(base_env, pre_insert, -1), -1))
        actions.append(("insert", pose_action(base_env, insert_pose * sapien.Pose([0.05, 0, 0]), -1), -1))
        return actions

    raise KeyError(task)


def execute_expert(env: Any, task: str, seed: int, hold_steps: int) -> dict[str, Any]:
    env.reset(seed=seed)
    base_env = env.unwrapped
    actions: list[np.ndarray] = []
    states: list[dict[str, Any]] = [clone_state(base_env.get_state_dict())]
    eef_positions: list[np.ndarray] = [tensor_to_numpy(base_env.agent.tcp_pose.p)[0]]
    actor_name = TASK_ACTOR[task]
    actor_positions: list[np.ndarray] = [tensor_to_numpy(getattr(base_env, actor_name).pose.p)[0]]
    goal_positions: list[np.ndarray | None] = []
    if task == "PickCube-v1":
        goal_positions.append(tensor_to_numpy(base_env.goal_site.pose.p)[0])
    elif task == "StackCube-v1":
        goal_positions.append(tensor_to_numpy(base_env.cubeB.pose.p)[0] + [0, 0, 0.04])
    else:
        goal_positions.append(tensor_to_numpy(base_env.box_hole_pose.p)[0])

    for name, action, gripper in task_waypoints(base_env, task):
        for _ in range(hold_steps):
            actions.append(np.asarray(action, dtype=np.float32).copy())
            _, _, terminated, truncated, info = env.step(actions[-1])
            states.append(clone_state(base_env.get_state_dict()))
            eef_positions.append(tensor_to_numpy(base_env.agent.tcp_pose.p)[0])
            actor_positions.append(tensor_to_numpy(getattr(base_env, actor_name).pose.p)[0])
            if bool(tensor_to_numpy(info["success"])[0]):
                # Preserve the first successful suffix endpoint.  For Stack
                # and Peg, a few additional steps are needed to settle.
                if task == "PickCube-v1":
                    break
        if task == "PickCube-v1" and len(actions) and bool(tensor_to_numpy(info["success"])[0]):
            break
        if bool(tensor_to_numpy(terminated)[0]) or bool(tensor_to_numpy(truncated)[0]):
            break

    success = bool(tensor_to_numpy(base_env.evaluate()["success"])[0])
    return {
        "seed": seed,
        "actions": actions,
        "states": states,
        "eef_positions": eef_positions,
        "actor_positions": actor_positions,
        "goal_positions": goal_positions,
        "success": success,
        "final_info": {key: tensor_to_numpy(value).tolist() for key, value in info.items()},
    }


def perturbations(state: dict[str, Any], task: str, magnitude: float) -> list[tuple[str, np.ndarray, dict[str, Any]]]:
    actor_name = TASK_ACTOR[task]
    directions = [
        ("object_pos_x+", np.array([magnitude, 0, 0])),
        ("object_pos_x-", np.array([-magnitude, 0, 0])),
        ("object_pos_y+", np.array([0, magnitude, 0])),
        ("object_pos_y-", np.array([0, -magnitude, 0])),
    ]
    output = []
    for name, delta in directions:
        branch = clone_state(state)
        actor = state_get_actor(branch, actor_name)
        actor[0, :3] += torch.as_tensor(delta, dtype=actor.dtype, device=actor.device)
        output.append((name, delta, branch))
    # Small robot-state perturbations are a simulator-native proxy for local
    # EEF offsets.  They preserve the exact state-save/restore protocol while
    # avoiding a second controller or planner in the causal gate.
    for sign in (1.0, -1.0):
        branch = clone_state(state)
        articulation_name = next(iter(branch["articulations"]))
        qpos = branch["articulations"][articulation_name]
        qpos[0, 1] += sign * (magnitude / 0.003) * 0.02
        output.append((f"eef_joint_1{'+' if sign > 0 else '-'}", np.array([0.0, sign * (magnitude / 0.003) * 0.02, 0.0]), branch))
    return output


def reset_clock(base_env: Any) -> None:
    # TimeLimitWrapper derives truncation from this clock; it is not part of
    # get_state_dict and must be reset for a suffix that starts at arbitrary t.
    if hasattr(base_env, "_elapsed_steps"):
        base_env._elapsed_steps.zero_()


def run_suffix(env: Any, state: dict[str, Any], actions: Iterable[np.ndarray]) -> tuple[bool, bool, str | None]:
    base_env = env.unwrapped
    try:
        base_env.set_state_dict(clone_state(state))
        reset_clock(base_env)
        for action in actions:
            _, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32).copy())
            if bool(tensor_to_numpy(terminated)[0]) or bool(tensor_to_numpy(truncated)[0]):
                break
        success = bool(tensor_to_numpy(base_env.evaluate()["success"])[0])
        return success, True, None
    except Exception as exc:  # invalid/interpenetrating branches are separate
        return False, False, f"{type(exc).__name__}: {exc}"


def record_branch_rows(task: str, episode: int, trajectory: dict[str, Any], env: Any, sample_every: int, magnitude: float, branches_per_state: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_env = env.unwrapped
    states = trajectory["states"]
    actions = trajectory["actions"]
    actor_name = TASK_ACTOR[task]
    sampled = list(range(0, max(1, len(actions)), sample_every))
    sampled = sampled[: max(1, math.ceil(len(actions) / sample_every))]
    for timestep in sampled:
        state = states[timestep]
        original_success, valid, error = run_suffix(env, state, actions[timestep:])
        # Repeat the zero branch to test isolation and deterministic restore.
        zero_again, valid_again, error_again = run_suffix(env, state, actions[timestep:])
        if not (valid and valid_again and original_success == zero_again):
            raise RuntimeError(f"restore/suffix validation failed at {task} ep={episode} t={timestep}")

        state_actor = tensor_to_numpy(state_get_actor(state, actor_name))[0, :3]
        eef = trajectory["eef_positions"][timestep]
        goal = trajectory["goal_positions"][0]
        prev_action = actions[timestep - 1] if timestep > 0 else np.zeros_like(actions[timestep])
        prev_prev = actions[timestep - 2] if timestep > 1 else prev_action
        action_mag = float(np.linalg.norm(actions[timestep]))
        action_velocity = float(np.linalg.norm(actions[timestep] - prev_action))
        action_acceleration = float(np.linalg.norm(actions[timestep] - 2 * prev_action + prev_prev))
        gripper_transition = float(abs(actions[timestep][-1] - prev_action[-1]))
        eef_object_distance = float(np.linalg.norm(eef - state_actor))
        object_goal_distance = float(np.linalg.norm(state_actor - goal))

        branch_specs = [("zero", np.zeros(3), state)]
        branch_specs.extend((name, delta, branch) for name, delta, branch in perturbations(state, task, magnitude))
        branch_specs = branch_specs[: 1 + branches_per_state]
        outcomes = []
        for perturb_type, delta, branch_state in branch_specs:
            success, branch_valid, branch_error = run_suffix(env, branch_state, actions[timestep:])
            outcomes.append((success, branch_valid, branch_error))
            rows.append(
                {
                    "task": task,
                    "episode": episode,
                    "timestep": timestep,
                    "phase": timestep / max(1, len(actions) - 1),
                    "state_id": state_hash(state),
                    "perturbation_type": perturb_type,
                    "perturbation_magnitude": float(np.linalg.norm(delta)),
                    "perturbation_dx": float(delta[0]),
                    "perturbation_dy": float(delta[1]),
                    "perturbation_dz": float(delta[2]),
                    "branch_success": int(success),
                    "branch_valid": int(branch_valid),
                    "branch_error": branch_error or "",
                    "original_suffix_success": int(original_success),
                    "expert_success": int(trajectory["success"]),
                    "criticality": np.nan,
                    "action_magnitude": action_mag,
                    "action_velocity": action_velocity,
                    "action_acceleration": action_acceleration,
                    "gripper_transition": gripper_transition,
                    "eef_object_distance": eef_object_distance,
                    "object_goal_distance": object_goal_distance,
                }
            )
        # The zero branch is a restore/isolation control, not one of the N
        # perturbations in c_t = 1 - successes / N.
        valid_outcomes = [success for success, is_valid, _ in outcomes[1:] if is_valid]
        criticality = float(1.0 - np.mean(valid_outcomes)) if valid_outcomes else float("nan")
        for row in rows[-len(outcomes) :]:
            row["criticality"] = criticality
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--hold-steps", type=int, default=6)
    parser.add_argument("--magnitude", type=float, default=0.003)
    parser.add_argument("--branches-per-state", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("experiments/counterfactual_tournament/maniskill_gate0"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    all_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "backend": "ManiSkill 3.0.1",
        "sim_backend": "physx_cpu",
        "control_mode": "pd_ee_pose",
        "expert_source": "official ManiSkill task + bundled IK controller scripted waypoints",
        "official_mplib_status": "unavailable: mplib.Planner constructor segfaults on this host",
        "tasks": args.tasks,
        "episodes_requested": args.episodes,
        "sample_every": args.sample_every,
        "branches_per_state": args.branches_per_state,
        "magnitude_m": args.magnitude,
        "episodes": [],
    }
    for task in args.tasks:
        env = make_env(task)
        try:
            for episode in range(args.episodes):
                episode_started = time.time()
                trajectory = execute_expert(env, task, args.seed + episode, args.hold_steps)
                if not trajectory["success"]:
                    print(f"WARNING expert did not succeed: {task} episode={episode}", file=sys.stderr)
                raw_path = args.output / f"{task.replace('-v1', '').lower()}_episode_{episode:03d}.pt"
                torch.save(
                    {
                        "task": task,
                        "episode": episode,
                        "seed": trajectory["seed"],
                        "actions": np.asarray(trajectory["actions"], dtype=np.float32),
                        "states": trajectory["states"],
                        "success": trajectory["success"],
                    },
                    raw_path,
                )
                rows = record_branch_rows(task, episode, trajectory, env, args.sample_every, args.magnitude, args.branches_per_state)
                all_rows.extend(rows)
                manifest["episodes"].append(
                    {
                        "task": task,
                        "episode": episode,
                        "seed": trajectory["seed"],
                        "expert_success": trajectory["success"],
                        "action_count": len(trajectory["actions"]),
                        "sampled_states": len({row["timestep"] for row in rows}),
                        "branch_count": len(rows),
                        "raw_trajectory": str(raw_path),
                        "runtime_sec": time.time() - episode_started,
                    }
                )
        finally:
            env.close()

    if all_rows:
        csv_path = args.output / "timestep_branches.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    manifest["runtime_sec"] = time.time() - started
    manifest["rows"] = len(all_rows)
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
