#!/usr/bin/env python3
"""Frozen matched-query ACT source-age confirmation.

This is a research intervention, not the native ACT baseline.  The checkpoint
is queried once at every environment step and the selected same-target action
components are composed after the native LeRobot processors.  Ten vector-env
workers are assigned explicit LIBERO initial-state IDs 10--19, so every
condition uses the same independent paired cohort.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from run_component_reuse import atomic_json, write_progress
from temporal_operators import same_target_candidates


ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = ROOT / "act_confirmation_protocol.json"


def infer_chunk_batch(observation, env, policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor):
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
    if result.ndim != 3 or result.shape[2] != 7:
        raise RuntimeError(f"unexpected ACT action chunk shape: {result.shape}")
    return result


def compose(condition: str, candidates) -> tuple[np.ndarray, int, int]:
    actions = candidates.actions
    ages = candidates.ages
    if condition == "fresh":
        return actions[-1].copy(), 0, 0
    if condition == "fo16":
        arm = actions[-1, :6]
        gripper = actions[np.flatnonzero(ages == 16)[0], 6:7] if 16 in ages else actions[-1, 6:7]
        actual = 16 if 16 in ages else 0
        return np.concatenate([arm, gripper]), 0, actual
    if condition == "full_old16":
        if 16 not in ages:
            return actions[-1].copy(), 0, 0
        action = actions[np.flatnonzero(ages == 16)[0]].copy()
        return action, 16, 16
    if condition == "reverse16":
        if 16 not in ages:
            return actions[-1].copy(), 0, 0
        old = actions[np.flatnonzero(ages == 16)[0]]
        return np.concatenate([old[:6], actions[-1, 6:7]]), 16, 0
    raise ValueError(f"unknown confirmation condition: {condition}")


def semantic_smoke() -> None:
    history = [np.full((20, 7), float(source * 100 + offset)) for source in range(17) for offset in [0]]
    candidates = same_target_candidates(history, target_step=16)
    if candidates.source_steps.tolist() != list(range(17)) or candidates.ages.tolist() != list(range(16, -1, -1)):
        raise SystemExit("ACT same-target candidate alignment failed")
    fresh, _, _ = compose("fresh", candidates)
    fo, arm_age, grip_age = compose("fo16", candidates)
    full, full_arm_age, full_grip_age = compose("full_old16", candidates)
    reverse, reverse_arm_age, reverse_grip_age = compose("reverse16", candidates)
    np.testing.assert_array_equal(fresh, candidates.actions[-1])
    np.testing.assert_array_equal(fo[:6], candidates.actions[-1, :6])
    np.testing.assert_array_equal(fo[6:], candidates.actions[0, 6:7])
    np.testing.assert_array_equal(full, candidates.actions[0])
    np.testing.assert_array_equal(reverse[:6], candidates.actions[0, :6])
    np.testing.assert_array_equal(reverse[6:], candidates.actions[-1, 6:7])
    if (arm_age, grip_age) != (0, 16) or (full_arm_age, full_grip_age) != (16, 16) or (reverse_arm_age, reverse_grip_age) != (16, 0):
        raise SystemExit("ACT source-age assignment failed")
    if np.array_equal(fo[6:], fresh[6:]):
        raise SystemExit("FO16 unexpectedly became a gripper hold")
    print(json.dumps({"status": "act_confirmation_semantic_smoke_pass", "conditions": ["fresh", "fo16", "full_old16", "reverse16"]}))


def extract_success(info, index: int):
    final_info = info.get("final_info") if isinstance(info, dict) else None
    if not isinstance(final_info, dict) or "is_success" not in final_info:
        return None
    value = np.asarray(final_info["is_success"])
    if value.ndim == 0:
        return bool(value.item())
    return bool(value.reshape(-1)[index])


def rollout_condition(*, env, policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor, seeds, state_ids, condition: str, max_steps: int) -> dict:
    policy.reset()
    observation, _ = env.reset(seed=list(map(int, seeds)))
    batch_size = len(state_ids)
    query_history: list[np.ndarray] = []
    successes: list[bool | None] = [None] * batch_size
    done = np.zeros(batch_size, dtype=bool)
    completion_steps: list[int | None] = [None] * batch_size
    per_env_events: list[list[dict]] = [[] for _ in range(batch_size)]
    arm_ages: list[list[int]] = [[] for _ in range(batch_size)]
    gripper_ages: list[list[int]] = [[] for _ in range(batch_size)]
    fresh_errors = {"fresh": 0.0, "full_old16": 0.0, "fo16": 0.0, "reverse16": 0.0}
    for step in range(max_steps):
        fresh_chunk = infer_chunk_batch(observation, env, policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor)
        query_history.append(fresh_chunk.copy())
        actions = []
        for index in range(batch_size):
            candidates = same_target_candidates([chunk[index] for chunk in query_history], target_step=step)
            action, arm_age, gripper_age = compose(condition, candidates)
            if condition == "fresh":
                fresh_errors[condition] = max(fresh_errors[condition], float(np.max(np.abs(action - candidates.actions[-1]))))
            if step >= 16 and condition == "full_old16":
                expected = candidates.actions[np.flatnonzero(candidates.ages == 16)[0]]
                fresh_errors[condition] = max(fresh_errors[condition], float(np.max(np.abs(action - expected))))
            if step >= 16 and condition == "fo16":
                old = candidates.actions[np.flatnonzero(candidates.ages == 16)[0]]
                expected = np.concatenate([candidates.actions[-1, :6], old[6:7]])
                fresh_errors[condition] = max(fresh_errors[condition], float(np.max(np.abs(action - expected))))
            if step >= 16 and condition == "reverse16":
                old = candidates.actions[np.flatnonzero(candidates.ages == 16)[0]]
                expected = np.concatenate([old[:6], candidates.actions[-1, 6:7]])
                fresh_errors[condition] = max(fresh_errors[condition], float(np.max(np.abs(action - expected))))
            actions.append(action)
            if not done[index]:
                arm_ages[index].append(arm_age)
                gripper_ages[index].append(gripper_age)
                per_env_events[index].append({
                    "environment_step": step,
                    "target_step": step,
                    "arm_source_query_step": step - arm_age,
                    "gripper_source_query_step": step - gripper_age,
                    "arm_source_age_steps": arm_age,
                    "gripper_source_age_steps": gripper_age,
                    "arm_target_chunk_index": arm_age,
                    "gripper_target_chunk_index": gripper_age,
                })
        previous_done = done.copy()
        observation, reward, terminated, truncated, info = env.step(np.asarray(actions, dtype=np.float32))
        terminated = np.asarray(terminated, dtype=bool).reshape(-1)
        truncated = np.asarray(truncated, dtype=bool).reshape(-1)
        just_done = (~done) & (terminated | truncated)
        for index in np.flatnonzero(just_done):
            success = extract_success(info, int(index))
            if success is None:
                reward_array = np.asarray(reward).reshape(-1)
                success = bool(reward_array[index] > 0) if len(reward_array) > index else False
            successes[index] = bool(success)
            completion_steps[index] = step + 1 if success else None
        done |= terminated | truncated
        if done.all():
            break
    for index in range(batch_size):
        if successes[index] is None:
            successes[index] = False
    return {
        "initial_state_ids": [int(x) for x in state_ids],
        "seeds": [int(x) for x in seeds],
        "condition": condition,
        "successes": [bool(x) for x in successes],
        "success_count": int(sum(bool(x) for x in successes)),
        "episodes": batch_size,
        "environment_steps": [len(events) for events in per_env_events],
        "policy_queries": [len(events) for events in per_env_events],
        "policy_queries_per_environment_step": 1.0,
        "mean_arm_source_age_steps": float(np.mean([np.mean(x) for x in arm_ages])),
        "mean_gripper_source_age_steps": float(np.mean([np.mean(x) for x in gripper_ages])),
        "completion_steps_successful": [int(x) for x in completion_steps if x is not None],
        "source_events": per_env_events,
        "semantic_validation_max_abs_error": fresh_errors[condition],
        "max_steps_reached": len(query_history) >= max_steps and not done.all(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--task", action="append", help="suite:task_id")
    parser.add_argument("--semantic-smoke", action="store_true")
    args = parser.parse_args()

    if args.semantic_smoke:
        semantic_smoke()
        return
    if not args.task or args.checkpoint is None or args.output is None:
        raise SystemExit("--checkpoint, --output, and at least one --task are required unless --semantic-smoke is used")

    protocol = json.loads(args.protocol.read_text())
    wanted = set(args.task)
    tasks = [task for task in protocol["task_selection"]["tasks"] if f"{task['suite']}:{task['task_id']}" in wanted]
    if not tasks or len(tasks) != len(wanted):
        raise SystemExit(f"requested tasks are absent from frozen protocol: {sorted(wanted)}")
    conditions = protocol["policy"]["confirmation_conditions"]
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    checkpoint = args.checkpoint.resolve()
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.pretrained_path = checkpoint
    if int(config.chunk_size) < 16:
        raise RuntimeError(f"ACT chunk_size {config.chunk_size} cannot support frozen age 16")
    output = {
        "protocol": str(args.protocol.resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_chunk_size": int(config.chunk_size),
        "checkpoint_n_action_steps": int(config.n_action_steps),
        "checkpoint_temporal_ensemble_coeff": config.temporal_ensemble_coeff,
        "intervention_is_not_native_baseline": True,
        "conditions": conditions,
        "tasks": {},
        "started_at": time.time(),
    }
    progress = {"pid": os.getpid(), "started_at": output["started_at"], "completed_tasks": 0, "completed_blocks": 0}
    write_progress(args.progress_file, progress)
    seeds = protocol["environment"]["seeds"]
    state_ids = protocol["environment"]["initial_state_ids"]
    for task in tasks:
        suite = task["suite"]
        task_id = int(task["task_id"])
        env_config = LiberoEnv(
            task=suite,
            task_ids=[task_id],
            fps=int(protocol["environment"]["fps"]),
            obs_type=protocol["environment"]["obs_type"],
            camera_name=protocol["environment"]["camera_name"],
            init_states=True,
            observation_width=int(protocol["environment"]["observation_width"]),
            observation_height=int(protocol["environment"]["observation_height"]),
            control_mode=protocol["environment"]["control_mode"],
        )
        envs = make_env(env_config, n_envs=len(state_ids), use_async_envs=False)
        env = envs[suite][task_id]
        for index, state_id in enumerate(state_ids):
            env.envs[index].init_state_id = int(state_id)
        policy = make_policy(cfg=config, env_cfg=env_config)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(config.device)}},
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_config, policy_cfg=config)
        max_steps = int(env.call("_max_episode_steps")[0])
        task_result = {"task_name": task["task_name"], "task_id": task_id, "suite": suite, "methods": {}}
        for condition in conditions:
            result = rollout_condition(
                env=env,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                seeds=seeds,
                state_ids=state_ids,
                condition=condition,
                max_steps=max_steps,
            )
            task_result["methods"][condition] = result
            progress["completed_blocks"] += 1
            progress["current_task"] = f"{suite}:task{task_id}"
            progress["current_condition"] = condition
            write_progress(args.progress_file, progress)
        output["tasks"][f"{suite}:task{task_id}"] = task_result
        progress["completed_tasks"] += 1
        atomic_json(args.output, output)
        env.close()
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output["finished_at"] = time.time()
    progress["finished_at"] = output["finished_at"]
    write_progress(args.progress_file, progress)
    atomic_json(args.output, output)
    print(json.dumps({"output": str(args.output), "tasks": len(tasks), "conditions": conditions}, indent=2))


if __name__ == "__main__":
    main()
