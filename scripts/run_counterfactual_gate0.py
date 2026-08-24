#!/usr/bin/env python3
"""Run the first RoboTwin counterfactual-criticality screening gate.

This runner records one state before each top-level scripted ``move`` call.
Each branch restores the SAPIEN PhysX snapshot, applies one small perturbation,
and replays the recorded expert action suffix. The fork score is downstream
expert-continuation failure, not a policy success claim.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import types
import traceback
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "counterfactual_tournament"))
from fork_engine import CounterfactualFork  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, default=Path("/home/wjq/workspace/upstreams/RoboTwin"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--planner", default="mplib_RRT", choices=("mplib_RRT", "mplib_screw", "none"))
    parser.add_argument("--fork-every", type=int, default=5)
    parser.add_argument("--max-forks", type=int, default=0)
    parser.add_argument("--expert-only", action="store_true")
    return parser.parse_args()


def git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def add_upstream_paths(root: Path) -> None:
    for path in (root, root / "scripts", root / "description" / "utils", root / "XPolicyLab"):
        if str(path.resolve()) not in sys.path:
            sys.path.insert(0, str(path.resolve()))


def install_mplib_only_planner(root: Path) -> None:
    """Avoid an optional Curobo JIT import on the documented MPLIB path.

    The local RoboTwin checkout imports Curobo before selecting the planner.
    Curobo's optional CUDA extension is unavailable on this host, while MPLIB
    is installed and is the selected scripted-expert planner. Load only the
    upstream MPLIB class into the expected module name, without editing the
    user-dirty upstream checkout.
    """

    source_path = root / "envs" / "robot" / "planner.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if "curobo" not in ast.unparse(node):
                selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "MplibPlanner":
            selected.append(node)
    module = types.ModuleType("envs.robot.planner")
    module.__file__ = str(source_path)
    module.__package__ = "envs.robot"
    module.__dict__["__name__"] = "envs.robot.planner"
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), module.__dict__)
    module.CuroboPlanner = type("CuroboPlanner", (), {})

    # The upstream MPLIB adapter retries failed RRT plans ten times. For the
    # first gate, a failed scripted expert is discarded and a successful seed
    # is retained, so the retry budget is capped to two to keep seed screening
    # bounded.
    original_plan_pose = module.MplibPlanner.plan_pose

    def bounded_plan_pose(self, *args, **kwargs):
        kwargs["try_times"] = min(int(kwargs.get("try_times", 2)), 2)
        return original_plan_pose(self, *args, **kwargs)

    module.MplibPlanner.plan_pose = bounded_plan_pose
    sys.modules["envs.robot.planner"] = module


def qpos_snapshot(task_env: Any) -> dict[str, np.ndarray]:
    return {
        "left": np.asarray(task_env.robot.left_entity.get_qpos(), dtype=np.float32).copy(),
        "right": np.asarray(task_env.robot.right_entity.get_qpos(), dtype=np.float32).copy(),
        "left_vel": np.asarray(task_env.robot.left_entity.get_qvel(), dtype=np.float32).copy(),
        "right_vel": np.asarray(task_env.robot.right_entity.get_qvel(), dtype=np.float32).copy(),
    }


def make_perturbation(task_env: Any, name: str) -> None:
    if name.startswith("robot_joint_0"):
        sign = 1.0 if name.endswith("plus") else -1.0
        arm = task_env.robot.left_entity if task_env.arm_tag == "left" else task_env.robot.right_entity
        qpos = np.asarray(arm.get_qpos(), dtype=np.float64).copy()
        qpos[0] += sign * 0.03
        arm.set_qpos(qpos)
        arm.set_qvel(np.zeros_like(qpos))
        return
    if name.startswith("object_xy"):
        sign = 1.0 if name.endswith("plus") else -1.0
        actor = getattr(task_env, "can", None)
        if actor is None:
            actors = [a for a in task_env.scene.get_all_actors() if a.get_name() not in {"table", "wall", "ground"}]
            if not actors:
                raise RuntimeError("task has no perturbable object actor")
            actor = actors[0]
        pose = actor.get_pose()
        pose.p[0] += sign * 0.015
        actor.set_pose(pose)
        return
    if name.startswith("action_pose_x"):
        # This marker is consumed by the runner's action-copy perturbation.
        return
    raise ValueError(f"unknown perturbation {name}")


def perturb_action_suffix(actions: tuple[Any, ...], name: str) -> tuple[Any, ...]:
    if not name.startswith("action_pose_x"):
        return actions
    sign = 1.0 if name.endswith("plus") else -1.0
    result = clone_move_args(actions)
    for item in result:
        if item is None:
            continue
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], list):
            for action in item[1]:
                if getattr(action, "action", None) == "move":
                    action.target_pose[0] += sign * 0.02
                    return result
    raise RuntimeError("action perturbation found no move action")


def clone_move_args(move_args: tuple[Any, ...]) -> tuple[Any, ...]:
    """Clone scripted Action records without copying the singleton ArmTag."""

    result = []
    for group in move_args:
        if group is None:
            result.append(None)
            continue
        arm_tag, group_actions = group
        cloned_actions = []
        for action in group_actions:
            cloned = copy.copy(action)
            if getattr(action, "target_pose", None) is not None:
                cloned.target_pose = list(action.target_pose)
            cloned.args = copy.deepcopy(getattr(action, "args", {}))
            cloned_actions.append(cloned)
        result.append((arm_tag, cloned_actions))
    return tuple(result)


def score_heuristics(records: list[dict[str, Any]]) -> None:
    previous = None
    previous_velocity = None
    for row in records:
        q = row["qpos"]
        if previous is None:
            row["velocity_heuristic"] = 0.0
            row["acceleration_heuristic"] = 0.0
        else:
            velocity = float(np.linalg.norm(q["left"] - previous["left"]) + np.linalg.norm(q["right"] - previous["right"]))
            row["velocity_heuristic"] = velocity
            row["acceleration_heuristic"] = 0.0 if previous_velocity is None else abs(velocity - previous_velocity)
            previous_velocity = velocity
        row["gripper_event_heuristic"] = float(q["left"][-1] != previous["left"][-1] if previous is not None else False) + float(q["right"][-1] != previous["right"][-1] if previous is not None else False)
        previous = q


def run_episode(args: argparse.Namespace) -> dict[str, Any]:
    add_upstream_paths(args.robotwin_root)
    os.chdir(args.robotwin_root)
    if args.planner.startswith("mplib"):
        install_mplib_only_planner(args.robotwin_root)
    import eval_policy_xpolicylab as official

    task_env = official.class_decorator(args.task)
    usr_args = {
        "task_name": args.task,
        "task_config": args.task_config,
        "policy_name": "ACT",
        "ckpt_setting": "counterfactual_gate0",
        "action_type": "joint",
        "seed": args.seed,
    }
    task_args, _ = official.load_task_args(usr_args)
    task_args.update({
        "eval_mode": True,
        "render_freq": 0,
        "eval_video_log": False,
        "safe_qpos": True,
    })
    if args.planner != "none":
        for key in ("left_embodiment_config", "right_embodiment_config"):
            if key in task_args:
                task_args[key]["planner"] = args.planner

    records: list[dict[str, Any]] = []
    actions: list[tuple[Any, ...]] = []
    original_move = task_env.move

    def recording_move(*move_args: Any, **move_kwargs: Any) -> Any:
        records.append({
            "segment": len(records),
            "state": task_env.scene.get_physx_system().pack(),
            "qpos": qpos_snapshot(task_env),
        })
        actions.append(clone_move_args(move_args))
        return original_move(*move_args, **move_kwargs)

    task_env.setup_demo(now_ep_num=0, seed=args.seed, is_test=True, **task_args)
    task_env.move = recording_move
    start = time.monotonic()
    expert_info = task_env.play_once()
    expert_success = bool(task_env.plan_success and task_env.check_success())
    elapsed = time.monotonic() - start
    if not expert_success:
        raise RuntimeError(f"expert scripted continuation failed for {args.task} seed {args.seed}")

    score_heuristics(records)
    if args.expert_only:
        return {
            "task": args.task,
            "seed": args.seed,
            "expert_success": expert_success,
            "expert_segments": len(records),
            "fork_segments": 0,
            "forks": 0,
            "elapsed_s": elapsed,
            "expert_info": expert_info,
            "records": records,
            "fork_rows": [],
        }
    selected = list(range(0, len(records), max(1, args.fork_every)))
    if args.max_forks:
        selected = selected[: args.max_forks]
    perturbations = [
        "action_pose_x_plus", "action_pose_x_minus",
        "robot_joint_0_plus", "robot_joint_0_minus",
        "object_xy_plus", "object_xy_minus",
    ]

    def snapshot() -> bytes:
        return task_env.scene.get_physx_system().pack()

    def restore(state: bytes) -> None:
        task_env.scene.get_physx_system().unpack(state)
        task_env.plan_success = True
        task_env.eval_success = False
        task_env.scene.update_render()

    def perturb(name: str) -> None:
        make_perturbation(task_env, name)

    def continue_from(index: int) -> tuple[bool, int]:
        steps = 0
        for j in range(index, len(actions)):
            move_args = actions[j]
            if j == index:
                move_args = perturb_action_suffix(move_args, current_perturbation[0])
            original_move(*move_args)
            steps += 1
            if not task_env.plan_success:
                break
        return bool(task_env.plan_success and task_env.check_success()), steps

    current_perturbation = [""]
    fork_rows: list[dict[str, Any]] = []
    for index in selected:
        base_state = records[index]["state"]

        def perturb_with_action(name: str) -> None:
            current_perturbation[0] = name
            perturb(name)

        def continue_with_action(j: int) -> tuple[bool, int]:
            return continue_from(j)

        fork = CounterfactualFork(
            snapshot=lambda: base_state,
            restore=restore,
            perturb=perturb_with_action,
            continue_from=continue_with_action,
        )
        outcomes = fork.evaluate(index, perturbations)
        for outcome in outcomes:
            fork_rows.append({
                "segment": index,
                "perturbation": outcome.perturbation,
                "success": outcome.success,
                "continuation_segments": outcome.continuation_steps,
                "error": outcome.error,
            })
        records[index]["criticality"] = float(1.0 - np.mean([row["success"] for row in fork_rows if row["segment"] == index]))
        records[index]["fork_count"] = len(outcomes)

    return {
        "task": args.task,
        "seed": args.seed,
        "expert_success": expert_success,
        "expert_segments": len(records),
        "fork_segments": len(selected),
        "forks": len(fork_rows),
        "elapsed_s": elapsed,
        "expert_info": expert_info,
        "records": records,
        "fork_rows": fork_rows,
    }


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "task": args.task,
        "seed": args.seed,
        "task_config": args.task_config,
        "planner": args.planner,
        "fork_every": args.fork_every,
        "project_git_sha": git_commit(ROOT),
        "robotwin_git_sha": git_commit(args.robotwin_root),
        "xpolicylab_git_sha": git_commit(args.robotwin_root / "XPolicyLab"),
        "python": sys.executable,
        "status": "running",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        result = run_episode(args)
        records = result.pop("records")
        fork_rows = result.pop("fork_rows")
        with (args.output_dir / "fork_states.pkl").open("wb") as stream:
            pickle.dump([row["state"] for row in records], stream, protocol=pickle.HIGHEST_PROTOCOL)
        with (args.output_dir / "segments.jsonl").open("w", encoding="utf-8") as stream:
            for row in records:
                row = dict(row)
                row.pop("state", None)
                row["qpos"] = {key: value.tolist() for key, value in row["qpos"].items()}
                stream.write(json.dumps(row) + "\n")
        with (args.output_dir / "forks.jsonl").open("w", encoding="utf-8") as stream:
            for row in fork_rows:
                stream.write(json.dumps(row) + "\n")
        result.update(metadata)
        result["status"] = "complete"
        (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        metadata["status"] = "complete"
    except Exception as exc:
        metadata.update({
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
