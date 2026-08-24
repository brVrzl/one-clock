#!/usr/bin/env python3
"""Bounded, single-seed RoboTwin ACT baseline and stall diagnosis.

This driver deliberately keeps the benchmark and upstream ACT code unchanged.
It only adds phase timing around reset, observation/rendering, ACT inference,
and ``take_action`` so an incomplete rollout can be attributed to its phase.
Each invocation runs exactly one seed in a fresh process; the shell caller can
apply a hard timeout without allowing a stalled SAPIEN call to affect another
paired trial.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/gate0_place_can_basket.yaml")
    parser.add_argument("--mode", choices=("official", "global8", "group416"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--planner",
        choices=("mplib_RRT", "mplib_screw", "none"),
        default="mplib_RRT",
        help="Planner override. 'none' preserves the official (Curobo) default.",
    )
    return parser.parse_args()


def add_upstream_paths(robotwin_root: Path) -> None:
    for path in (
        robotwin_root,
        robotwin_root / "scripts",
        robotwin_root / "description" / "utils",
        robotwin_root / "XPolicyLab",
    ):
        value = str(path.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)


def git_commit(path: Path) -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"expected a mapping in {path}")
    return config


def load_act_model(robotwin_root: Path, config: dict[str, Any], checkpoint: Path, *, temporal_agg: bool):
    deploy_path = robotwin_root / "XPolicyLab/policy/ACT/deploy.yml"
    deploy_config = yaml.safe_load(deploy_path.read_text(encoding="utf-8"))
    if not isinstance(deploy_config, dict):
        raise ValueError(f"expected a mapping in {deploy_path}")
    action_dim = sum(len(indices) for indices in config["action_groups"].values())
    deploy_config.update(
        {
            "ckpt_dir": str(checkpoint.resolve()),
            "ckpt_name": str(checkpoint.resolve()),
            "env_cfg_type": config["env_cfg_type"],
            "action_type": config["action_type"],
            "action_dim": action_dim,
            "chunk_size": int(config["chunk_size"]),
            "temporal_agg": temporal_agg,
        }
    )
    os.environ["ACT_ACTION_DIM"] = str(action_dim)
    from XPolicyLab.policy.ACT.model import Model

    return Model(deploy_config)


def full_chunk(act_model: Any) -> np.ndarray:
    """Read one complete ACT chunk without adding a second policy implementation."""

    act_model.t = 0
    act_model.get_action()
    chunk = act_model.all_actions[0].detach().cpu().numpy()
    result = np.asarray(act_model.post_process(chunk))
    if result.ndim != 2:
        raise ValueError(f"ACT full chunk has shape {result.shape}, expected rank 2")
    return result


def build_executor(config: dict[str, Any], mode: str):
    from one_clock import ActionGroup, FixedChunkExecutor

    action_dim = sum(len(indices) for indices in config["action_groups"].values())
    if mode == "global8":
        groups = tuple(
            ActionGroup(name, tuple(indices), 8)
            for name, indices in config["action_groups"].items()
        )
        return FixedChunkExecutor.global_fixed(
            action_dim=action_dim,
            chunk_size=int(config["chunk_size"]),
            horizon=8,
            groups=groups,
        )
    groups = tuple(
        ActionGroup(
            name,
            tuple(indices),
            4 if "arm" in name else 16,
        )
        for name, indices in config["action_groups"].items()
    )
    return FixedChunkExecutor.groupwise_fixed(
        action_dim=action_dim,
        chunk_size=int(config["chunk_size"]),
        groups=groups,
    )


class Timing:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = {}

    def add(self, name: str, elapsed: float) -> None:
        self.values.setdefault(name, []).append(float(elapsed))

    def summary(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "count": len(values),
                "total_s": float(sum(values)),
                "mean_s": float(sum(values) / len(values)),
                "max_s": float(max(values)),
            }
            for name, values in self.values.items()
        }


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    add_upstream_paths(args.robotwin_root)
    # RoboTwin imports several asset paths relative to its checkout.
    os.chdir(args.robotwin_root)
    import eval_policy_xpolicylab as official

    timings = Timing()
    result: dict[str, Any] = {
        "task": config["task_name"],
        "task_config": config.get("task_config", "demo_clean"),
        "mode": args.mode,
        "seed": int(args.seed),
        "planner_override": None if args.planner == "none" else args.planner,
        "status": "initializing",
        "success": False,
        "environment_steps": 0,
        "policy_queries": 0,
        "last_phase": "startup",
        "timings": timings.summary(),
        "robotwin_commit": git_commit(args.robotwin_root),
        "xpolicylab_commit": git_commit(args.robotwin_root / "XPolicyLab"),
        "checkpoint": str(args.checkpoint.resolve()),
    }
    task_env = None

    def phase(name: str) -> None:
        result["last_phase"] = name
        print(json.dumps({"event": "phase", "phase": name}), flush=True)

    def timed(name: str, callback):
        phase(name)
        start = time.monotonic()
        value = callback()
        timings.add(name, time.monotonic() - start)
        result["timings"] = timings.summary()
        return value

    try:
        temporal_agg = args.mode == "official"
        phase("model_init")
        model_start = time.monotonic()
        model = load_act_model(args.robotwin_root, config, args.checkpoint, temporal_agg=temporal_agg)
        timings.add("model_init", time.monotonic() - model_start)

        usr_args = {
            "task_name": str(config["task_name"]),
            "task_config": str(config.get("task_config", "demo_clean")),
            "policy_name": "ACT",
            "ckpt_setting": str(args.checkpoint),
            "action_type": str(config.get("action_type", "joint")),
            "seed": int(args.seed),
        }
        task_args, _ = official.load_task_args(usr_args)
        task_args.update(
            {
                "eval_mode": True,
                "render_freq": 0,
                "eval_video_log": False,
                # This is the already-audited headless qpos safety fallback,
                # held constant across all three calibration modes.
                "safe_qpos": True,
            }
        )
        if args.planner != "none":
            for key in ("left_embodiment_config", "right_embodiment_config"):
                if key in task_args:
                    task_args[key]["planner"] = args.planner

        task_env = official.class_decorator(str(config["task_name"]))
        episode_info = {"info": {}}
        phase("setup_demo")
        setup_start = time.monotonic()
        task_env.setup_demo(now_ep_num=0, seed=int(args.seed), is_test=True, **task_args)
        timings.add("setup_demo", time.monotonic() - setup_start)
        result["timings"] = timings.summary()
        result["step_limit"] = int(task_env.step_lim)
        result["renderer"] = {
            "shader": "rt",
            "headless": os.environ.get("EGL_PLATFORM", "unset"),
            "render_freq": 0,
            "rt_samples_per_pixel": 1,
            "rt_path_depth": 1,
            "rt_denoiser": "none",
        }

        instruction = official.build_instruction(
            task_args,
            episode_info,
            config.get("instruction_type", "seen"),
            1,
        )
        task_env.set_instruction(instruction=instruction)
        model.reset()
        executor = None if args.mode == "official" else build_executor(config, args.mode)
        if executor is not None:
            executor.reset()

        def observe() -> dict[str, Any]:
            return task_env.get_obs()

        def encode(observation: dict[str, Any]) -> dict[str, Any]:
            return official.robotwin_obs_to_xpolicylab(
                observation,
                instruction=task_env.get_instruction(),
                env_idx=0,
                frequency=int(config.get("frequency", 30)),
                task_env=task_env,
            )

        def query_one(observation: dict[str, Any]) -> np.ndarray:
            model.update_obs(encode(observation))
            result["policy_queries"] = int(result["policy_queries"]) + 1
            start = time.monotonic()
            if args.mode == "official":
                actions = model.get_action()
                if not actions:
                    raise RuntimeError("ACT returned an empty action list")
                flat, action_type = official.xpolicylab_action_to_robotwin(
                    actions[0],
                    action_type=str(config.get("action_type", "joint")),
                    current_observation=observation,
                )
                if action_type != "qpos":
                    raise ValueError(f"ACT joint checkpoint converted to {action_type}, expected qpos")
                action = flat
            else:
                action = full_chunk(model.model)
                expected = (int(config["chunk_size"]), 14)
                if action.shape != expected:
                    raise ValueError(f"ACT full chunk has shape {action.shape}, expected {expected}")
            timings.add("policy_inference", time.monotonic() - start)
            if np.asarray(action).ndim != 1 and args.mode == "official":
                raise ValueError(f"official ACT action has shape {np.asarray(action).shape}")
            if args.mode == "official" and np.asarray(action).shape != (14,):
                raise ValueError(f"official ACT action has shape {np.asarray(action).shape}, expected (14,)")
            if not np.isfinite(np.asarray(action)).all():
                raise ValueError("ACT action contains non-finite values")
            return np.asarray(action, dtype=np.float32)

        while not official.is_episode_end(task_env):
            if args.mode == "official":
                observation = timed("get_obs", observe)
                action = query_one(observation)
                timed("take_action", lambda: task_env.take_action(action, action_type="qpos"))
            else:
                def query_chunk() -> np.ndarray:
                    observation = timed("get_obs", observe)
                    # query_one increments the policy count and returns one
                    # action only in official mode; custom mode returns chunk.
                    return query_one(observation)

                decision = executor.step(query_chunk)
                timed("take_action", lambda: task_env.take_action(decision.action, action_type="qpos"))
            result["environment_steps"] = int(result["environment_steps"]) + 1
            if int(result["environment_steps"]) > int(task_env.step_lim) + 1:
                raise RuntimeError("episode exceeded RoboTwin step limit")

        phase("terminal_check")
        terminal_start = time.monotonic()
        result["success"] = bool(task_env.eval_success or task_env.check_success())
        timings.add("terminal_check", time.monotonic() - terminal_start)
        result["status"] = "complete"
        result["failure_reason"] = None if result["success"] else "task_not_successful_at_step_limit"
    except Exception as exc:
        result["status"] = "error"
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=12)
    finally:
        result["timings"] = timings.summary()
        if task_env is not None:
            phase("close_env")
            close_start = time.monotonic()
            try:
                task_env.close_env()
            except Exception as exc:
                result["close_error"] = f"{type(exc).__name__}: {exc}"
            timings.add("close_env", time.monotonic() - close_start)
        result["timings"] = timings.summary()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    # RoboTwin requires chdir into its checkout; pin a relative artifact path
    # before that transition so results remain in the calling repository.
    args.output = args.output.resolve()
    result = run_one(args)
    print(json.dumps(result, indent=2), flush=True)
    if result.get("status") != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
