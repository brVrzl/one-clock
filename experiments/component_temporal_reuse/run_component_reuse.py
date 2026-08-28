#!/usr/bin/env python3
"""Run frozen SmolVLA component-wise temporal-source interventions."""

from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = ROOT / "protocol.json"
DEFAULT_CHECKPOINT = Path("/home/wjq/checkpoints/HuggingFaceVLA_smolvla_libero")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_progress(path: Path | None, value: object) -> None:
    if path is not None:
        atomic_json(path, value)


def condition_action(
    *,
    condition: dict,
    step: int,
    fresh_chunk: np.ndarray,
    query_history: list[np.ndarray],
    chunk_size: int,
) -> tuple[np.ndarray, dict]:
    action = fresh_chunk[:, 0, :].copy()
    assignments = {}
    for group, indices, requested_age in (
        ("arm", slice(0, 6), int(condition["arm_source_age"])),
        ("gripper", slice(6, 7), int(condition["gripper_source_age"])),
    ):
        if step >= requested_age:
            source_step = step - requested_age
            chunk_index = requested_age
            source_action = query_history[source_step][:, chunk_index, :]
            actual_age = requested_age
        else:
            source_step = step
            chunk_index = 0
            source_action = fresh_chunk[:, 0, :]
            actual_age = 0
        action[:, indices] = source_action[:, indices]
        assignments[group] = {
            "source_query_step": source_step,
            "requested_source_age_steps": requested_age,
            "actual_source_age_steps": actual_age,
            "source_chunk_index": chunk_index,
            "target_step": step,
            "target_time_seconds": step / 30.0,
        }
        if requested_age >= chunk_size:
            raise RuntimeError(f"requested age {requested_age} exceeds chunk size {chunk_size}")
    return action, assignments


def infer_chunk(observation, env, policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor):
    from lerobot.envs.utils import add_envs_task, preprocess_observation
    from lerobot.utils.constants import ACTION

    observation = preprocess_observation(observation)
    observation = add_envs_task(env, observation)
    observation = env_preprocessor(observation)
    observation = preprocessor(observation)
    with torch.inference_mode():
        chunk = postprocessor(policy.predict_action_chunk(observation))
    chunk = env_postprocessor({ACTION: chunk})[ACTION]
    result = chunk.detach().cpu().numpy().astype(np.float32, copy=False)
    if result.ndim != 3 or result.shape[0] != 1 or result.shape[2] != 7:
        raise RuntimeError(f"unexpected action chunk shape: {result.shape}")
    return result


def rollout_episode(
    *,
    env,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    seed: int,
    condition: dict,
    chunk_size: int,
    keep_query_cache: bool,
) -> tuple[dict, np.ndarray | None]:
    policy.reset()
    observation, _ = env.reset(seed=[int(seed)])
    max_steps = int(env.call("_max_episode_steps")[0])
    query_history: list[np.ndarray] = []
    source_events = []
    validation = {
        "fresh_semantics_max_abs_error": 0.0,
        "full_old_max_abs_error": 0.0,
        "fresh_arm_old_gripper_max_abs_error": 0.0,
        "old_arm_fresh_gripper_max_abs_error": 0.0,
        "old_gripper_vs_previous_applied_non_equal_count": 0,
        "old_gripper_vs_previous_applied_max_abs_error": 0.0,
    }
    applied_previous = None
    success = False
    completion_step = None
    done = False
    for step in range(max_steps):
        fresh_chunk = infer_chunk(
            observation, env, policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor
        )
        query_history.append(fresh_chunk.copy())
        action, assignments = condition_action(
            condition=condition,
            step=step,
            fresh_chunk=fresh_chunk,
            query_history=query_history,
            chunk_size=chunk_size,
        )
        if condition["name"] == "fresh":
            validation["fresh_semantics_max_abs_error"] = max(
                validation["fresh_semantics_max_abs_error"],
                float(np.max(np.abs(action - fresh_chunk[:, 0, :]))),
            )
        if step >= max(int(condition["arm_source_age"]), int(condition["gripper_source_age"])):
            if condition["arm_source_age"] == condition["gripper_source_age"]:
                source = query_history[step - int(condition["arm_source_age"])][:, int(condition["arm_source_age"]), :]
                validation["full_old_max_abs_error"] = max(
                    validation["full_old_max_abs_error"], float(np.max(np.abs(action - source)))
                )
            if int(condition["arm_source_age"]) == 0 and int(condition["gripper_source_age"]) > 0:
                expected = np.concatenate(
                    [fresh_chunk[:, 0, :6], query_history[step - int(condition["gripper_source_age"])][:, int(condition["gripper_source_age"]), 6:7]],
                    axis=1,
                )
                validation["fresh_arm_old_gripper_max_abs_error"] = max(
                    validation["fresh_arm_old_gripper_max_abs_error"], float(np.max(np.abs(action - expected)))
                )
            if int(condition["arm_source_age"]) > 0 and int(condition["gripper_source_age"]) == 0:
                expected = np.concatenate(
                    [query_history[step - int(condition["arm_source_age"])][:, int(condition["arm_source_age"]), :6], fresh_chunk[:, 0, 6:7]],
                    axis=1,
                )
                validation["old_arm_fresh_gripper_max_abs_error"] = max(
                    validation["old_arm_fresh_gripper_max_abs_error"], float(np.max(np.abs(action - expected)))
                )
            if int(condition["gripper_source_age"]) > 0 and applied_previous is not None:
                old_gripper = action[:, 6]
                difference = np.abs(old_gripper - applied_previous[:, 6])
                validation["old_gripper_vs_previous_applied_max_abs_error"] = max(
                    validation["old_gripper_vs_previous_applied_max_abs_error"], float(difference.max())
                )
                validation["old_gripper_vs_previous_applied_non_equal_count"] += int(np.sum(difference > 1e-6))
        source_events.append(
            {
                "environment_step": step,
                "condition": condition["name"],
                "arm": assignments["arm"],
                "gripper": assignments["gripper"],
            }
        )
        observation, reward, terminated, truncated, info = env.step(action)
        applied_previous = action.copy()
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
    if not done:
        completion_step = None
    record = {
        "seed": seed,
        "success": success,
        "completion_steps": completion_step,
        "environment_steps": step + 1,
        "policy_queries": step + 1,
        "mean_arm_source_age_steps": float(np.mean([e["arm"]["actual_source_age_steps"] for e in source_events])),
        "mean_gripper_source_age_steps": float(np.mean([e["gripper"]["actual_source_age_steps"] for e in source_events])),
        "source_events": source_events,
        "validation": validation,
    }
    return record, (np.concatenate(query_history, axis=0) if keep_query_cache else None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "query_cache")
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--progress-file", type=Path, help="optional per-worker progress sidecar")
    parser.add_argument("--task", action="append", help="suite:task_id; repeatable subset for validation")
    parser.add_argument("--conditions", help="comma-separated condition subset for validation")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    tasks = protocol["tasks"]
    if args.task:
        wanted = set(args.task)
        tasks = [task for task in tasks if f"{task['suite']}:{task['task_id']}" in wanted]
    conditions = protocol["conditions"]
    if args.conditions:
        wanted = set(args.conditions.split(","))
        conditions = [condition for condition in conditions if condition["name"] in wanted]
    if not tasks or not conditions:
        raise SystemExit("empty task or condition selection")

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    checkpoint = args.checkpoint.resolve()
    policy_config = PreTrainedConfig.from_pretrained(checkpoint)
    policy_config.device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_config.pretrained_path = checkpoint
    chunk_size = int(policy_config.chunk_size)
    if int(policy_config.n_action_steps) != 1:
        raise RuntimeError(f"native SmolVLA checkpoint n_action_steps unexpectedly changed: {policy_config.n_action_steps}")
    args.cache_root.mkdir(parents=True, exist_ok=True)
    output = {
        "protocol": str(args.protocol.resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_revision": protocol["checkpoint_revision"],
        "chunk_size": chunk_size,
        "n_action_steps": int(policy_config.n_action_steps),
        "conditions": conditions,
        "tasks": {},
        "started_at": time.time(),
    }
    progress = {
        "pid": os.getpid(),
        "started_at": output["started_at"],
        "completed_blocks": 0,
        "completed_episodes": 0,
        "environment_steps": 0,
        "current_task": None,
        "current_condition": None,
    }
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
            policy_cfg=policy_config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(policy_config.device)}},
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_config, policy_cfg=policy_config
        )
        task_result = {
            "task_name": task["task_name"],
            "native_baseline_successes": task["native_baseline_successes"],
            "conditions": {},
        }
        for condition in conditions:
            progress["current_task"] = f"{suite}:task{task_id}"
            progress["current_condition"] = condition["name"]
            write_progress(args.progress_file, progress)
            envs = make_env(env_config, n_envs=1, use_async_envs=False)
            env = envs[suite][task_id]
            episodes = []
            cache_arrays = []
            for seed in protocol["environment"]["seeds"]:
                episode, cache = rollout_episode(
                    env=env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    seed=int(seed),
                    condition=condition,
                    chunk_size=chunk_size,
                    keep_query_cache=condition["name"] == "fresh",
                )
                episodes.append(episode)
                progress["completed_episodes"] += 1
                progress["environment_steps"] += episode["environment_steps"]
                write_progress(args.progress_file, progress)
                if cache is not None:
                    cache_arrays.append(cache)
            env.close()
            successes = [bool(episode["success"]) for episode in episodes]
            ages_arm = [episode["mean_arm_source_age_steps"] for episode in episodes]
            ages_gripper = [episode["mean_gripper_source_age_steps"] for episode in episodes]
            validations = episodes[0]["validation"]
            for episode in episodes[1:]:
                for key, value in episode["validation"].items():
                    if key.endswith("non_equal_count"):
                        validations[key] += value
                    else:
                        validations[key] = max(validations[key], value)
            task_result["conditions"][condition["name"]] = {
                "successes": successes,
                "success_count": sum(successes),
                "episodes": len(episodes),
                "success_rate": sum(successes) / len(episodes),
                "policy_queries": sum(episode["policy_queries"] for episode in episodes),
                "environment_steps": sum(episode["environment_steps"] for episode in episodes),
                "policy_queries_per_environment_step": sum(episode["policy_queries"] for episode in episodes)
                / sum(episode["environment_steps"] for episode in episodes),
                "mean_arm_source_age_steps": float(np.mean(ages_arm)),
                "mean_gripper_source_age_steps": float(np.mean(ages_gripper)),
                "completion_steps_successful": [episode["completion_steps"] for episode in episodes if episode["success"]],
                "validation": validations,
                "episodes_detail": episodes,
            }
            if cache_arrays:
                cache_path = args.cache_root / f"{suite}_task{task_id}_fresh.npz"
                np.savez_compressed(cache_path, **{f"episode_{i}": array for i, array in enumerate(cache_arrays)})
                task_result["fresh_query_cache"] = str(cache_path.resolve())
            output["tasks"][f"{suite}:task{task_id}"] = task_result
            progress["completed_blocks"] += 1
            write_progress(args.progress_file, progress)
            atomic_json(args.output, output)
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output["finished_at"] = time.time()
    progress["finished_at"] = output["finished_at"]
    write_progress(args.progress_file, progress)
    atomic_json(args.output, output)
    print(json.dumps({
        "output": str(args.output),
        "tasks": len(tasks),
        "conditions": len(conditions),
    }, indent=2))


if __name__ == "__main__":
    main()
