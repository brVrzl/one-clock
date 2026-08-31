#!/usr/bin/env python3
"""Run matched-query global and asymmetric ACT component commitments."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TEMPORAL_ROOT = ROOT.parent
REPO_ROOT = TEMPORAL_ROOT.parent.parent
sys.path.insert(0, str(TEMPORAL_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_component_reuse import atomic_json, write_progress  # noqa: E402
from one_clock import ActionGroup, FixedChunkExecutor  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "protocol.json"
METHODS = ("global_8_8", "arm8_grip16", "arm16_grip8")
ARM = tuple(range(6))
GRIPPER = (6,)


def make_executor(method: str, chunk_size: int) -> FixedChunkExecutor:
    if method == "global_8_8":
        return FixedChunkExecutor.global_fixed(
            action_dim=7,
            chunk_size=chunk_size,
            horizon=8,
            groups=(ActionGroup("arm", ARM, 8), ActionGroup("gripper", GRIPPER, 8)),
        )
    if method == "arm8_grip16":
        return FixedChunkExecutor.groupwise_fixed(
            action_dim=7,
            chunk_size=chunk_size,
            groups=(ActionGroup("arm", ARM, 8), ActionGroup("gripper", GRIPPER, 16)),
        )
    if method == "arm16_grip8":
        return FixedChunkExecutor.groupwise_fixed(
            action_dim=7,
            chunk_size=chunk_size,
            groups=(ActionGroup("arm", ARM, 16), ActionGroup("gripper", GRIPPER, 8)),
        )
    raise ValueError(f"unknown two-clock method: {method}")


def reset_policy_rng(torch, seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def extract_success(info, reward) -> bool:
    final_info = info.get("final_info") if isinstance(info, dict) else None
    if isinstance(final_info, dict) and "is_success" in final_info:
        value = np.asarray(final_info["is_success"])
        return bool(value.reshape(-1)[0])
    values = np.asarray(reward).reshape(-1)
    return bool(len(values) and values[0] > 0)


def infer_chunk_current(
    observation,
    env,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    torch,
):
    """Use the current LeRobot 0.6.2 LIBERO observation pipeline."""

    from lerobot.envs.utils import preprocess_observation
    from lerobot.utils.constants import ACTION

    batch = preprocess_observation(observation)
    batch = env_preprocessor(batch)
    batch = preprocessor(batch)
    with torch.inference_mode():
        chunk = postprocessor(policy.predict_action_chunk(batch))
        chunk = env_postprocessor({ACTION: chunk})[ACTION]
    result = chunk.detach().cpu().numpy().astype(np.float32, copy=False)
    if result.shape != (1, int(policy.config.chunk_size), 7):
        raise RuntimeError(f"unexpected postprocessed ACT chunk shape: {result.shape}")
    return result


def rollout_episode(
    *,
    env,
    policy,
    processors,
    torch,
    method: str,
    state_id: int,
    seed: int,
    policy_rng_seed: int,
    chunk_size: int,
    max_steps: int,
) -> dict:
    env.envs[0].init_state_id = int(state_id)
    actual_state_id = int(env.envs[0].init_state_id)
    if actual_state_id != int(state_id):
        raise RuntimeError(f"initial-state assignment mismatch: requested={state_id}, actual={actual_state_id}")
    reset_policy_rng(torch, policy_rng_seed)
    policy.reset()
    observation, _ = env.reset(seed=[int(seed)])
    env_preprocessor, env_postprocessor, preprocessor, postprocessor = processors
    executor = make_executor(method, chunk_size)
    query_steps: list[int] = []
    decisions = []
    success = False
    done = False
    completion_step = None

    for step in range(int(max_steps)):
        def query_policy():
            chunk = infer_chunk_current(
                observation,
                env,
                policy,
                env_preprocessor,
                env_postprocessor,
                preprocessor,
                postprocessor,
                torch,
            )
            if chunk.shape != (1, chunk_size, 7):
                raise RuntimeError(f"unexpected ACT chunk shape: {chunk.shape}; expected (1,{chunk_size},7)")
            return chunk[0]

        decision = executor.step(query_policy)
        if decision.policy_query:
            query_steps.append(step)
        source_query_steps = {
            group: query_steps[int(chunk_id)]
            for group, chunk_id in decision.source_chunk_ids.items()
        }
        record = decision.as_log_record()
        record["source_query_steps"] = source_query_steps
        decisions.append(record)
        observation, reward, terminated, truncated, info = env.step(decision.action[None].astype(np.float32))
        terminated = bool(np.asarray(terminated).reshape(-1)[0])
        truncated = bool(np.asarray(truncated).reshape(-1)[0])
        done = terminated or truncated
        if done:
            success = extract_success(info, reward)
            completion_step = step + 1 if success else None
            break

    if not done:
        completion_step = None
    arm_ages = [int(record["source_ages"]["arm"]) for record in decisions]
    gripper_ages = [int(record["source_ages"]["gripper"]) for record in decisions]
    return {
        "seed": int(seed),
        "requested_initial_state_id": int(state_id),
        "actual_initial_state_id": actual_state_id,
        "method": method,
        "success": bool(success),
        "completion_steps": completion_step,
        "environment_steps": len(decisions),
        "policy_queries": len(query_steps),
        "query_count": len(query_steps),
        "query_rate": len(query_steps) / float(len(decisions)),
        "query_steps": query_steps,
        "expected_query_steps_for_prefix": list(range(0, len(decisions), 8)),
        "query_schedule_exact": query_steps == list(range(0, len(decisions), 8)),
        "mean_arm_source_age_steps": float(np.mean(arm_ages)),
        "mean_gripper_source_age_steps": float(np.mean(gripper_ages)),
        "source_events": decisions,
    }


def semantic_smoke() -> None:
    """CPU-only test helper; the full assertions live in test_two_clock.py."""

    from test_two_clock import run_trace

    traces = {method: run_trace(method, steps=33) for method in METHODS}
    query_steps = [trace["query_steps"] for trace in traces.values()]
    assert query_steps == [query_steps[0]] * 3
    print(json.dumps({"status": "two_clock_cpu_semantic_smoke_pass", "methods": list(METHODS), "query_steps": query_steps[0]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--task", required=False, help="suite:task_id")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress", "--progress-file", dest="progress_file", type=Path)
    parser.add_argument("--semantic-smoke", action="store_true")
    args = parser.parse_args()
    if args.semantic_smoke:
        semantic_smoke()
        return
    if args.task is None or args.output is None:
        raise SystemExit("--task and --output are required unless --semantic-smoke is used")

    protocol = json.loads(args.protocol.read_text())
    task_map = {f"{task['suite']}:task{int(task['task_id'])}": task for task in protocol["tasks"]}
    if args.task not in task_map:
        raise SystemExit(f"task is absent from frozen protocol: {args.task}")
    task = task_map[args.task]
    checkpoint = Path(task["checkpoint"]).resolve()
    if not (checkpoint / "config.json").is_file() or not (checkpoint / "model.safetensors").is_file():
        raise SystemExit(f"ACT checkpoint is missing required files: {checkpoint}")

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.pretrained_path = checkpoint
    if getattr(config, "type", None) != "act":
        raise RuntimeError(f"expected ACT checkpoint, got {getattr(config, 'type', None)!r}")
    chunk_size = int(config.chunk_size)
    if chunk_size < int(protocol["policy"]["chunk_size_minimum"]):
        raise RuntimeError(f"ACT chunk_size {chunk_size} cannot expose commitment offset 16")

    env_config = LiberoEnv(
        task=task["suite"],
        task_ids=[int(task["task_id"])],
        fps=int(protocol["environment"]["fps"]),
        obs_type=protocol["environment"]["obs_type"],
        camera_name=protocol["environment"]["camera_name"],
        init_states=True,
        observation_width=int(protocol["environment"]["observation_width"]),
        observation_height=int(protocol["environment"]["observation_height"]),
        control_mode=protocol["environment"]["control_mode"],
    )
    env = make_env(env_config, n_envs=1, use_async_envs=False)[task["suite"]][int(task["task_id"])]
    policy = make_policy(cfg=config, env_cfg=env_config)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(config.device)}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_config, policy_cfg=config)
    processors = (env_preprocessor, env_postprocessor, preprocessor, postprocessor)
    max_steps = int(np.asarray(env.call("_max_episode_steps")).reshape(-1)[0])
    state_ids = [int(value) for value in protocol["environment"]["initial_state_ids"]]
    seeds = [int(value) for value in protocol["environment"]["seeds"]]
    started = time.time()
    output = {
        "protocol": str(args.protocol.resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_training_step": 100000,
        "checkpoint_chunk_size": chunk_size,
        "checkpoint_n_action_steps": int(config.n_action_steps),
        "task": args.task,
        "task_name": task["task_name"],
        "methods": list(METHODS),
        "policy_query_period_steps": 8,
        "one_synchronous_environment": True,
        "paired_initial_state_ids": state_ids,
        "paired_environment_seeds": seeds,
        "policy_rng_seed": int(protocol["policy"]["policy_rng_seed"]),
        "methods_result": {},
        "started_at": started,
    }
    progress = {"pid": os.getpid(), "started_at": started, "completed_methods": 0, "completed_episodes": 0}
    write_progress(args.progress_file, progress)
    for method in METHODS:
        episodes = []
        for state_id, seed in zip(state_ids, seeds):
            episodes.append(
                rollout_episode(
                    env=env,
                    policy=policy,
                    processors=processors,
                    torch=torch,
                    method=method,
                    state_id=state_id,
                    seed=seed,
                    policy_rng_seed=int(protocol["policy"]["policy_rng_seed"]),
                    chunk_size=chunk_size,
                    max_steps=max_steps,
                )
            )
            progress["completed_episodes"] += 1
            progress["current_method"] = method
            progress["current_state_id"] = state_id
            write_progress(args.progress_file, progress)
        successes = [bool(episode["success"]) for episode in episodes]
        total_steps = sum(episode["environment_steps"] for episode in episodes)
        total_queries = sum(episode["policy_queries"] for episode in episodes)
        output["methods_result"][method] = {
            "successes": successes,
            "success_count": int(sum(successes)),
            "episodes": len(episodes),
            "policy_queries": total_queries,
            "environment_steps": total_steps,
            "query_rate": total_queries / total_steps,
            "mean_arm_source_age_steps": float(np.mean([episode["mean_arm_source_age_steps"] for episode in episodes])),
            "mean_gripper_source_age_steps": float(np.mean([episode["mean_gripper_source_age_steps"] for episode in episodes])),
            "pooled_arm_source_age_steps": float(sum(sum(record["source_ages"]["arm"] for record in episode["source_events"]) for episode in episodes) / total_steps),
            "pooled_gripper_source_age_steps": float(sum(sum(record["source_ages"]["gripper"] for record in episode["source_events"]) for episode in episodes) / total_steps),
            "episodes_detail": episodes,
        }
        progress["completed_methods"] += 1
        atomic_json(args.output, output)
    output["finished_at"] = time.time()
    progress["finished_at"] = output["finished_at"]
    write_progress(args.progress_file, progress)
    atomic_json(args.output, output)
    env.close()
    print(json.dumps({"output": str(args.output), "task": args.task, "episodes": len(METHODS) * len(state_ids)}, indent=2))


if __name__ == "__main__":
    main()
