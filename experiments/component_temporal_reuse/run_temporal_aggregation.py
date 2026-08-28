#!/usr/bin/env python3
"""Evaluate fixed same-target temporal aggregation operators on the frozen tasks.

The policy is queried exactly once per environment step.  Each operator then
combines the overlapping predictions for that step's physical target time.
This is separate from the frozen source-reuse pilot and writes its own result
file.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from run_component_reuse import atomic_json, infer_chunk, write_progress
from temporal_operators import (
    aggregate_components,
    aggregate_full_action,
    act_temporal_weights,
    cogact_cosine_weights,
    exponential_age_weights,
    one_hot_age,
    same_target_candidates,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = ROOT / "protocol.json"
DEFAULT_CHECKPOINT = Path("/home/wjq/checkpoints/HuggingFaceVLA_smolvla_libero")
METHODS = (
    "fresh",
    "official_act_m001",
    "physical_exp_beta003",
    "cogact_alpha03",
    "component_arm_fresh_gripper_act",
)


def compose_action(method: str, candidates) -> tuple[np.ndarray, float, float]:
    """Return action and effective arm/gripper source ages for one target time."""

    actions = candidates.actions
    ages = candidates.ages
    if method == "fresh":
        return actions[-1].copy(), 0.0, 0.0
    if method == "official_act_m001":
        weights = act_temporal_weights(len(actions), coefficient=0.01)
        action = aggregate_full_action(actions, weights)
        age = float(weights @ ages)
        return action, age, age
    if method == "physical_exp_beta003":
        weights = exponential_age_weights(ages, beta=0.03)
        action = aggregate_full_action(actions, weights)
        age = float(weights @ ages)
        return action, age, age
    if method == "cogact_alpha03":
        weights = cogact_cosine_weights(actions, alpha=0.3)
        action = aggregate_full_action(actions, weights)
        age = float(weights @ ages)
        return action, age, age
    if method == "component_arm_fresh_gripper_act":
        fresh = one_hot_age(ages, 0)
        gripper = act_temporal_weights(len(actions), coefficient=0.01)
        action = aggregate_components(actions, {"arm": fresh, "gripper": gripper})
        return action, 0.0, float(gripper @ ages)
    raise ValueError(f"unknown aggregation method: {method}")


def semantic_smoke() -> None:
    """Check target alignment and all fixed operator compositions without MuJoCo."""

    chunks = [np.asarray([[[100.0 * source + offset + d for d in range(7)] for offset in range(6)]]) for source in range(5)]
    candidates = same_target_candidates(chunks, target_step=4)
    if candidates.source_steps.tolist() != [0, 1, 2, 3, 4]:
        raise SystemExit(f"unexpected target sources: {candidates.source_steps.tolist()}")
    if candidates.ages.tolist() != [4, 3, 2, 1, 0]:
        raise SystemExit(f"unexpected target ages: {candidates.ages.tolist()}")
    for method in METHODS:
        action, _, _ = compose_action(method, candidates)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise SystemExit(f"invalid smoke action for {method}: {action.shape}")
    fresh, _, _ = compose_action("fresh", candidates)
    np.testing.assert_array_equal(fresh, candidates.actions[-1])
    act, _, _ = compose_action("official_act_m001", candidates)
    np.testing.assert_allclose(act, aggregate_full_action(candidates.actions, act_temporal_weights(5, 0.01)))
    component, arm_age, gripper_age = compose_action("component_arm_fresh_gripper_act", candidates)
    np.testing.assert_array_equal(component[:6], candidates.actions[-1, :6])
    if arm_age != 0.0 or not (0.0 < gripper_age < 4.0):
        raise SystemExit("component-aware smoke age assignment is invalid")
    print(json.dumps({"status": "semantic_smoke_pass", "methods": list(METHODS)}))


def rollout_episode(*, env, policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor, seed: int, method: str) -> dict:
    policy.reset()
    observation, _ = env.reset(seed=[int(seed)])
    max_steps = int(env.call("_max_episode_steps")[0])
    query_history: list[np.ndarray] = []
    arm_ages: list[float] = []
    gripper_ages: list[float] = []
    success = False
    completion_step = None
    done = False
    for step in range(max_steps):
        fresh_chunk = infer_chunk(
            observation, env, policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor
        )
        query_history.append(fresh_chunk.copy())
        candidates = same_target_candidates(query_history, target_step=step)
        action, arm_age, gripper_age = compose_action(method, candidates)
        arm_ages.append(arm_age)
        gripper_ages.append(gripper_age)
        observation, _, terminated, truncated, info = env.step(action[None, :].astype(np.float32))
        terminated = bool(np.asarray(terminated).reshape(-1)[0])
        truncated = bool(np.asarray(truncated).reshape(-1)[0])
        if "final_info" in info and isinstance(info["final_info"], dict):
            final_success = info["final_info"].get("is_success")
            if final_success is not None:
                success = bool(np.asarray(final_success).reshape(-1)[0])
        done = terminated or truncated
        if done:
            completion_step = step + 1 if success else None
            break
    return {
        "seed": int(seed),
        "success": success,
        "completion_steps": completion_step,
        "environment_steps": step + 1,
        "policy_queries": step + 1,
        "mean_arm_source_age_steps": float(np.mean(arm_ages)),
        "mean_gripper_source_age_steps": float(np.mean(gripper_ages)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--task", action="append")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--semantic-smoke", action="store_true")
    args = parser.parse_args()
    if args.semantic_smoke:
        semantic_smoke()
        return
    if args.output is None:
        raise SystemExit("--output is required unless --semantic-smoke is used")

    protocol = json.loads(args.protocol.read_text())
    tasks = protocol["tasks"]
    if args.task:
        wanted = set(args.task)
        tasks = [task for task in tasks if f"{task['suite']}:{task['task_id']}" in wanted]
    methods = [method for method in args.methods.split(",") if method]
    if not tasks or not methods or any(method not in METHODS for method in methods):
        raise SystemExit("invalid task or method selection")

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    checkpoint = args.checkpoint.resolve()
    policy_config = PreTrainedConfig.from_pretrained(checkpoint)
    policy_config.device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    policy_config.pretrained_path = checkpoint
    if int(policy_config.n_action_steps) != 1:
        raise RuntimeError(f"expected native SmolVLA n_action_steps=1, got {policy_config.n_action_steps}")
    output = {
        "protocol": str(args.protocol.resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_revision": protocol["checkpoint_revision"],
        "chunk_size": int(policy_config.chunk_size),
        "n_action_steps": int(policy_config.n_action_steps),
        "methods": methods,
        "one_policy_query_per_environment_step": True,
        "tasks": {},
        "started_at": time.time(),
    }
    progress = {"pid": os.getpid(), "started_at": output["started_at"], "completed_tasks": 0, "completed_episodes": 0}
    write_progress(args.progress_file, progress)

    for task in tasks:
        suite = task["suite"]
        task_id = int(task["task_id"])
        env_config = LiberoEnv(
            task=suite,
            task_ids=[task_id],
            fps=int(protocol["environment"]["fps"]),
            obs_type=protocol["environment"]["obs_type"],
            camera_name=protocol["environment"]["camera_name"],
            init_states=bool(protocol["environment"]["init_states"]),
            observation_width=int(protocol["environment"]["observation_width"]),
            observation_height=int(protocol["environment"]["observation_height"]),
            control_mode=protocol["environment"]["control_mode"],
        )
        policy = make_policy(cfg=policy_config, env_cfg=env_config)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_config, pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(policy_config.device)}},
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_config, policy_cfg=policy_config)
        task_methods = {}
        for method in methods:
            episodes = []
            for seed in protocol["environment"]["seeds"]:
                envs = make_env(env_config, n_envs=1, use_async_envs=False)
                env = envs[suite][task_id]
                episodes.append(rollout_episode(
                    env=env, policy=policy, env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor, preprocessor=preprocessor,
                    postprocessor=postprocessor, seed=int(seed), method=method,
                ))
                env.close()
                progress["completed_episodes"] += 1
                progress["current_task"] = f"{suite}:task{task_id}"
                progress["current_method"] = method
                write_progress(args.progress_file, progress)
            successes = [bool(ep["success"]) for ep in episodes]
            task_methods[method] = {
                "successes": successes,
                "success_count": sum(successes),
                "episodes": len(episodes),
                "success_rate": sum(successes) / len(episodes),
                "policy_queries": sum(ep["policy_queries"] for ep in episodes),
                "environment_steps": sum(ep["environment_steps"] for ep in episodes),
                "policy_queries_per_environment_step": sum(ep["policy_queries"] for ep in episodes) / sum(ep["environment_steps"] for ep in episodes),
                "mean_arm_source_age_steps": float(np.mean([ep["mean_arm_source_age_steps"] for ep in episodes])),
                "mean_gripper_source_age_steps": float(np.mean([ep["mean_gripper_source_age_steps"] for ep in episodes])),
                "completion_steps_successful": [ep["completion_steps"] for ep in episodes if ep["success"]],
                "episodes_detail": episodes,
            }
        output["tasks"][f"{suite}:task{task_id}"] = {"task_name": task["task_name"], "methods": task_methods}
        progress["completed_tasks"] += 1
        atomic_json(args.output, output)
        del policy
        if __import__("torch").cuda.is_available():
            __import__("torch").cuda.empty_cache()
    output["finished_at"] = time.time()
    progress["finished_at"] = output["finished_at"]
    write_progress(args.progress_file, progress)
    atomic_json(args.output, output)
    print(json.dumps({"output": str(args.output), "tasks": len(tasks), "methods": methods}, indent=2))


if __name__ == "__main__":
    main()
