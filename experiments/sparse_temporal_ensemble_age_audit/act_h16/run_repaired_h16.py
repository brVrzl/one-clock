#!/usr/bin/env python3
"""Repaired ACT h16 trio with fresh paired environment construction.

Every method/state episode receives a newly constructed LIBERO environment.
Python and NumPy are seeded with the frozen environment seed before
construction, and ACT RNG is reset identically before the episode reset.  This
prevents the task-10 fixture-placement drift found by the pairing audit.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ACT_ROOT = Path(__file__).resolve().parent
AUDIT_ROOT = ACT_ROOT.parent
REPO_ROOT = AUDIT_ROOT.parents[1]
HISTORICAL_ROOT = REPO_ROOT / "experiments" / "sparse_temporal_ensemble_dev"
HISTORICAL_ACT_ROOT = HISTORICAL_ROOT / "act"
sys.path.insert(0, str(AUDIT_ROOT))
sys.path.insert(0, str(HISTORICAL_ROOT))
sys.path.insert(0, str(HISTORICAL_ACT_ROOT))

from dense_equivalent_executor import DenseEquivalentSparseExecutor  # noqa: E402
from run_act_sparse_te import (  # noqa: E402
    atomic_json,
    extract_success,
    infer_chunk,
    reset_policy_rng,
    summarize_method,
    write_progress,
)
from sparse_executor import SparseExecutor  # noqa: E402


METHODS = ("hard_h16", "candidate_index_te_h16", "dense_equivalent_te_h16")
DEFAULT_PROTOCOL = AUDIT_ROOT / "protocol.json"


def flatten_low_dimensional(value: Any, prefix: str = "") -> dict[str, list | int | float | bool]:
    """Return JSON-safe non-image arrays from a nested raw observation."""

    result: dict[str, list | int | float | bool] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            if "pixel" in child.lower() or "image" in child.lower():
                continue
            result.update(flatten_low_dimensional(value[key], child))
        return result
    if isinstance(value, np.ndarray):
        result[prefix] = value.tolist()
    elif isinstance(value, (bool, int, float, np.number)):
        result[prefix] = value.item() if isinstance(value, np.generic) else value
    return result


def make_executor(method: str):
    if method == "hard_h16":
        return SparseExecutor(
            cadence=16,
            prediction_horizon=100,
            mode="hard",
            coefficient=0.01,
            action_dim=7,
        )
    if method == "candidate_index_te_h16":
        return SparseExecutor(
            cadence=16,
            prediction_horizon=100,
            mode="sparse_te",
            coefficient=0.01,
            action_dim=7,
        )
    if method == "dense_equivalent_te_h16":
        return DenseEquivalentSparseExecutor(
            cadence=16,
            prediction_horizon=100,
            mode="dense_equivalent_te",
            coefficient=0.01,
            action_dim=7,
        )
    raise ValueError(f"unknown method: {method}")


def rollout_episode(
    *,
    env,
    policy,
    processors,
    torch,
    task_key: str,
    method: str,
    state_id: int,
    env_seed: int,
    policy_rng_seed: int,
    max_steps: int,
) -> dict:
    """Run one repaired h16 episode with full executor provenance."""

    env.envs[0].init_state_id = int(state_id)
    selected_state_id = int(env.envs[0].init_state_id)
    if selected_state_id != int(state_id):
        raise RuntimeError(
            f"initial-state assignment mismatch: requested={state_id}, selected={selected_state_id}"
        )
    random.seed(int(env_seed))
    np.random.seed(int(env_seed))
    reset_policy_rng(torch, int(policy_rng_seed))
    policy.reset()
    observation, _ = env.reset(seed=[int(env_seed)])
    initial_sim_state = np.asarray(env.envs[0]._env.get_sim_state()).copy()
    initial_model_body_pos = np.asarray(env.envs[0]._env.sim.model.body_pos).copy()
    initial_low_dimensional_observation = flatten_low_dimensional(observation)
    env_preprocessor, env_postprocessor, preprocessor, postprocessor = processors

    executor = make_executor(method)
    step_log: list[dict] = []
    query_log: list[dict] = []
    success = False
    completion_step: int | None = None
    done = False

    for target_step in range(int(max_steps)):
        query_latency = None
        if executor.should_query(target_step):
            query_started = time.perf_counter()

            def query() -> np.ndarray:
                return infer_chunk(
                    observation,
                    env,
                    policy,
                    env_preprocessor,
                    env_postprocessor,
                    preprocessor,
                    postprocessor,
                    torch,
                )

            result = executor.step(target_step, query)
            query_latency = time.perf_counter() - query_started
            query_log.append(
                {
                    "query_physical_step_q": int(target_step),
                    "latency_seconds": float(query_latency),
                }
            )
        else:
            result = executor.step(
                target_step,
                lambda: (_ for _ in ()).throw(RuntimeError("query_fn called off schedule")),
            )

        action = result.action.astype(np.float32, copy=False)
        observation, reward, terminated, truncated, info = env.step(action[None])
        terminated = bool(np.asarray(terminated).reshape(-1)[0])
        truncated = bool(np.asarray(truncated).reshape(-1)[0])
        done = terminated or truncated
        if done:
            success = extract_success(info, reward)
            completion_step = target_step + 1 if success else None

        step_log.append(
            {
                "task": task_key,
                "episode_initial_state_id": int(state_id),
                "environment_seed": int(env_seed),
                "method": method,
                "physical_target_t": int(target_step),
                "latest_query_q": int(result.latest_query_step),
                "policy_queried_at_t": bool(result.queried),
                "ensemble_source_query_ids": result.candidates.source_query_steps.astype(int).tolist(),
                "candidate_offsets_t_minus_q": result.candidates.offsets.astype(int).tolist(),
                "ensemble_candidate_count": int(result.candidate_count),
                "normalized_ensemble_weights": result.weights.astype(float).tolist(),
                "mean_weighted_source_age_steps": float(result.weighted_source_age),
                "query_latency_seconds": None if query_latency is None else float(query_latency),
                "chosen_executed_action_7d": action.astype(float).tolist(),
                "success_termination": bool(success) if done else None,
            }
        )
        if done:
            break

    environment_steps = len(step_log)
    query_count = len(query_log)
    return {
        "task": task_key,
        "method": method,
        "environment_seed": int(env_seed),
        "requested_initial_state_id": int(state_id),
        "selected_initial_state_id_before_reset": selected_state_id,
        "policy_rng_seed": int(policy_rng_seed),
        "fresh_environment_instance": True,
        "environment_construction_seed": int(env_seed),
        "initial_sim_state": initial_sim_state.astype(float).tolist(),
        "initial_model_body_pos": initial_model_body_pos.astype(float).tolist(),
        "initial_low_dimensional_observation": initial_low_dimensional_observation,
        "cadence_h": 16,
        "prediction_horizon": 100,
        "temporal_ensemble_coefficient": 0.01,
        "success": bool(success),
        "completion_steps": completion_step,
        "environment_steps": environment_steps,
        "policy_queries": query_count,
        "query_count": query_count,
        "query_rate": query_count / float(environment_steps),
        "query_steps": [entry["query_physical_step_q"] for entry in query_log],
        "query_latency_seconds": [entry["latency_seconds"] for entry in query_log],
        "mean_query_latency_seconds": float(
            np.mean([entry["latency_seconds"] for entry in query_log])
        )
        if query_log
        else None,
        "mean_ensemble_candidate_count": float(
            np.mean([entry["ensemble_candidate_count"] for entry in step_log])
        ),
        "mean_weighted_source_age_steps": float(
            np.mean([entry["mean_weighted_source_age_steps"] for entry in step_log])
        ),
        "step_log": step_log,
        "query_log": query_log,
    }


def task_map(protocol: dict) -> dict[str, dict]:
    return {
        f"{task['suite']}:task{int(task['task_id'])}": task
        for task in protocol["tasks"]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--task")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--live-smoke", action="store_true")
    args = parser.parse_args()
    if args.task is None or args.output is None:
        raise SystemExit("--task and --output are required")

    protocol = json.loads(args.protocol.read_text())
    panel = protocol["repaired_h16_panel"]
    if panel["methods"] != list(METHODS):
        raise RuntimeError("protocol method list differs from the repaired h16 trio")
    state_ids = [int(value) for value in panel["initial_state_ids"]]
    seeds = [int(value) for value in panel["environment_seeds"]]
    if state_ids != list(range(10, 20)) or seeds != list(range(2000, 2010)):
        raise RuntimeError("repaired h16 panel drifted from states 10..19 / seeds 2000..2009")
    if not panel["fresh_environment_per_condition_state"]:
        raise RuntimeError("fresh environment construction is mandatory")
    tasks = task_map(protocol)
    if args.task not in tasks:
        raise SystemExit(f"task is absent from frozen protocol: {args.task}")
    task = tasks[args.task]
    checkpoint = Path(task["act_checkpoint"]).resolve()
    if not (checkpoint / "config.json").is_file() or not (checkpoint / "model.safetensors").is_file():
        raise SystemExit(f"ACT checkpoint is missing required files: {checkpoint}")

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
    policy_cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_cfg.pretrained_path = checkpoint
    if getattr(policy_cfg, "type", None) != "act" or int(policy_cfg.chunk_size) != 100:
        raise RuntimeError("repaired panel requires the frozen ACT H_pred=100 checkpoint")
    environment = protocol["environment"]
    env_config = LiberoEnv(
        task=task["suite"],
        task_ids=[int(task["task_id"])],
        fps=int(environment["fps"]),
        obs_type=environment["obs_type"],
        camera_name=environment["camera_name"],
        init_states=True,
        observation_width=int(environment["observation_width"]),
        observation_height=int(environment["observation_height"]),
        control_mode=environment["control_mode"],
    )
    policy = make_policy(cfg=policy_cfg, env_cfg=env_config)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(policy_cfg.device)}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_config, policy_cfg=policy_cfg
    )
    processors = (env_preprocessor, env_postprocessor, preprocessor, postprocessor)

    selected_states = state_ids[:1] if args.live_smoke else state_ids
    selected_seeds = seeds[:1] if args.live_smoke else seeds
    started = time.time()
    output = {
        "protocol": str(args.protocol.resolve()),
        "implementation": "LeRobot 0.4.4 ACT predict_action_chunk with fresh environment per condition/state",
        "runtime": {
            "python_executable": sys.executable,
            "lerobot": "0.4.4",
            "torch": str(torch.__version__),
            "mujoco": "3.3.1",
            "cuda_visible_devices": str(args.gpu),
        },
        "task": args.task,
        "task_name": task["task_name"],
        "checkpoint": str(checkpoint),
        "prediction_horizon": 100,
        "temporal_ensemble_coefficient": 0.01,
        "fresh_environment_per_condition_state": True,
        "methods": list(METHODS),
        "live_smoke": bool(args.live_smoke),
        "started_at": started,
        "methods_result": {},
    }
    progress = {
        "pid": os.getpid(),
        "started_at": started,
        "task": args.task,
        "completed_methods": 0,
        "completed_episodes": 0,
        "current_method": None,
        "current_state_id": None,
    }
    write_progress(args.progress, progress)

    for method in METHODS:
        episodes = []
        for state_id, env_seed in zip(selected_states, selected_seeds):
            progress.update({"current_method": method, "current_state_id": int(state_id)})
            write_progress(args.progress, progress)
            random.seed(int(env_seed))
            np.random.seed(int(env_seed))
            reset_policy_rng(torch, int(protocol["pairing_audit"]["policy_rng_seed"]))
            env = make_env(env_config, n_envs=1, use_async_envs=False)[task["suite"]][int(task["task_id"])]
            try:
                max_steps = int(np.asarray(env.call("_max_episode_steps")).reshape(-1)[0])
                episode = rollout_episode(
                    env=env,
                    policy=policy,
                    processors=processors,
                    torch=torch,
                    task_key=args.task,
                    method=method,
                    state_id=state_id,
                    env_seed=env_seed,
                    policy_rng_seed=int(protocol["pairing_audit"]["policy_rng_seed"]),
                    max_steps=max_steps,
                )
            finally:
                env.close()
            episodes.append(episode)
            progress["completed_episodes"] += 1
            write_progress(args.progress, progress)
        output["methods_result"][method] = summarize_method(episodes)
        progress["completed_methods"] += 1
        write_progress(args.progress, progress)
        atomic_json(args.output, output)

    output["finished_at"] = time.time()
    progress["finished_at"] = output["finished_at"]
    write_progress(args.progress, progress)
    atomic_json(args.output, output)
    print(json.dumps({"output": str(args.output), "task": args.task, "episodes": len(METHODS) * len(selected_states)}))


if __name__ == "__main__":
    main()
