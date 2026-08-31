#!/usr/bin/env python3
"""Prepare/run the SmolVLA group-memory shards after the ACT gate."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
SPARSE_ROOT = REPO_ROOT / "experiments" / "sparse_temporal_ensemble_dev"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SPARSE_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from group_memory_common import RUNNABLE_METHODS, compose_method, smolvla_query_seed  # noqa: E402
from sparse_executor import SparseExecutor  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "protocol.json"
PAIRING_PREFIX_LENGTH = 16


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    coordination = protocol["coordination"]
    if coordination["sol_decision"] not in coordination["accepted_pairing_decisions"]:
        raise RuntimeError("BLOCKED_BY_PAIRING_AUDIT: Sol decision is not recorded as accepted")
    if not protocol["shared_kernel"]["status"].startswith("selected_by_sol"):
        raise RuntimeError("BLOCKED_BY_SOL_KERNEL_DECISION: shared kernel is not selected")
    if not coordination["sol_audit_commit"]:
        raise RuntimeError("BLOCKED_BY_PAIRING_AUDIT: Sol audit commit is not recorded")
    if not coordination.get("sol_repaired_rollout_commit"):
        raise RuntimeError("BLOCKED_BY_SOL_REPAIRED_TRIO: repaired h16 trio commit is not recorded")
    return protocol


def task_map(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        f"{task['suite']}:task{int(task['task_id'])}": task
        for task in protocol["cohort_task_specs"]
    }


def validate_values(protocol: dict[str, Any]) -> tuple[list[int], list[int], str]:
    states = [int(value) for value in protocol["cohort"]["states"]]
    seeds = [int(value) for value in protocol["cohort"]["environment_seeds"]]
    if states != list(range(10, 20)) or seeds != list(range(2000, 2010)):
        raise RuntimeError("SmolVLA development cohort drifted from states 10..19 / seeds 2000..2009")
    if int(protocol["policy"]["query_cadence_h"]) != 16 or int(protocol["policy"]["smolvla_prediction_horizon"]) != 50:
        raise RuntimeError("SmolVLA cadence or prediction horizon drifted")
    kernel = str(protocol["shared_kernel"]["selected_name"])
    if kernel != "dense_equivalent_te" or kernel not in protocol["shared_kernel"]["allowed_names"]:
        raise RuntimeError(f"unknown shared kernel {kernel!r}")
    return states, seeds, kernel


def reset_query_generators(torch: Any, seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def infer_chunk(observation, env, policy, processors, torch) -> np.ndarray:
    from lerobot.envs.utils import add_envs_task, preprocess_observation
    from lerobot.utils.constants import ACTION

    env_preprocessor, env_postprocessor, preprocessor, postprocessor = processors
    batch = preprocess_observation(copy.deepcopy(observation))
    batch = add_envs_task(env, batch)
    batch = env_preprocessor(batch)
    batch = preprocessor(batch)
    with torch.inference_mode():
        chunk = postprocessor(policy.predict_action_chunk(batch))
        chunk = env_postprocessor({ACTION: chunk})[ACTION]
    result = chunk.detach().cpu().numpy().astype(np.float32, copy=False)
    if result.ndim != 3 or result.shape[0] != 1 or result.shape[1] < 50 or result.shape[2] != 7:
        raise RuntimeError(f"unexpected postprocessed SmolVLA chunk shape: {result.shape}")
    return result[0].copy()


def extract_success(info: Any, reward: Any) -> bool:
    final_info = info.get("final_info") if isinstance(info, dict) else None
    if isinstance(final_info, dict) and "is_success" in final_info:
        return bool(np.asarray(final_info["is_success"]).reshape(-1)[0])
    values = np.asarray(reward).reshape(-1)
    return bool(len(values) and values[0] > 0)


def flatten_numeric(value: Any, prefix: str = "") -> dict[str, np.ndarray]:
    if isinstance(value, dict):
        result: dict[str, np.ndarray] = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_numeric(value[key], child))
        return result
    if isinstance(value, (list, tuple)):
        array = np.asarray(value)
        if array.dtype != object:
            return {prefix: array.copy()}
        result = {}
        for index, item in enumerate(value):
            result.update(flatten_numeric(item, f"{prefix}[{index}]"))
        return result
    try:
        array = np.asarray(value)
    except Exception:
        return {}
    if array.dtype == object:
        return {}
    return {prefix: array.copy()}


def compare_array_maps(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> dict[str, Any]:
    keys_equal = set(first) == set(second)
    exact = keys_equal
    maximum = 0.0
    for key in sorted(set(first) | set(second)):
        if key not in first or key not in second:
            exact = False
            continue
        a = np.asarray(first[key])
        b = np.asarray(second[key])
        same = a.shape == b.shape and np.array_equal(a, b)
        exact = exact and same
        if a.shape == b.shape and np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number) and a.size:
            maximum = max(maximum, float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
    return {"keys_equal": keys_equal, "exact": bool(exact), "max_absolute_difference": maximum}


def get_sim_state(env) -> np.ndarray:
    return np.asarray(env.envs[0]._env.get_sim_state()).copy()


def run_episode(
    *,
    env,
    policy,
    processors,
    torch,
    task_key: str,
    method: str,
    state_id: int,
    env_seed: int,
    protocol: dict[str, Any],
    max_steps: int,
    capture_prefix: bool = False,
) -> dict[str, Any]:
    env.envs[0].init_state_id = int(state_id)
    actual_state_id = int(env.envs[0].init_state_id)
    if actual_state_id != int(state_id):
        raise RuntimeError("initial-state assignment mismatch")
    random.seed(int(env_seed))
    np.random.seed(int(env_seed))
    policy.reset()
    observation, _ = env.reset(seed=[int(env_seed)])
    initial_observation = flatten_numeric(copy.deepcopy(observation)) if capture_prefix else None
    initial_sim_state = get_sim_state(env) if capture_prefix else None
    executor = SparseExecutor(cadence=16, prediction_horizon=50, mode="hard", coefficient=0.01, action_dim=7)
    step_log: list[dict[str, Any]] = []
    query_log: list[dict[str, Any]] = []
    prefix_actions: list[np.ndarray] = []
    prefix_sim_states: list[np.ndarray] = []
    prefix_observations: list[dict[str, np.ndarray]] = []
    initial_chunk = None
    success = False
    completion_step: int | None = None
    done = False
    for target_step in range(int(max_steps)):
        query_seed_key = None
        query_seed_value = None
        query_latency = None
        if executor.should_query(target_step):
            query_seed_key, query_seed_value = smolvla_query_seed(task_key, state_id, env_seed, target_step)
            reset_query_generators(torch, query_seed_value)
            started = time.perf_counter()

            def query() -> np.ndarray:
                return infer_chunk(observation, env, policy, processors, torch)

            result = executor.step(target_step, query)
            query_latency = time.perf_counter() - started
            query_log.append(
                {
                    "query_physical_step_q": target_step,
                    "query_seed_key": query_seed_key,
                    "query_rng_seed": query_seed_value,
                    "latency_seconds": query_latency,
                }
            )
            if capture_prefix and target_step == 0:
                initial_chunk = result.candidates.actions[-1].copy()
        else:
            result = executor.step(
                target_step,
                lambda: (_ for _ in ()).throw(RuntimeError("query function called off q=0,16,32 schedule")),
            )
        action, diagnostics = compose_method(
            method,
            result.candidates,
            kernel_name=str(protocol["shared_kernel"]["selected_name"]),
            coefficient=float(protocol["shared_kernel"]["coefficient"]),
            alpha=float(protocol["methods"]["M2_shared_cogact_h16"]["alpha"]),
        )
        action = action.astype(np.float32, copy=False)
        if capture_prefix and target_step < PAIRING_PREFIX_LENGTH:
            prefix_actions.append(action.copy())
        observation, reward, terminated, truncated, info = env.step(action[None])
        if capture_prefix and target_step < PAIRING_PREFIX_LENGTH:
            prefix_sim_states.append(get_sim_state(env))
            prefix_observations.append(flatten_numeric(copy.deepcopy(observation)))
        terminated = bool(np.asarray(terminated).reshape(-1)[0])
        truncated = bool(np.asarray(truncated).reshape(-1)[0])
        done = terminated or truncated
        if done:
            success = extract_success(info, reward)
            completion_step = target_step + 1 if success else None
        step_log.append(
            {
                "physical_target_t": target_step,
                "latest_query_q": int(result.latest_query_step),
                "policy_queried_at_t": bool(result.queried),
                "candidate_source_queries": result.candidates.source_query_steps.astype(int).tolist(),
                "candidate_ages": result.candidates.ages.astype(int).tolist(),
                "candidate_count": int(result.candidate_count),
                "shared_weights": diagnostics.get("shared_weights", diagnostics.get("base_weights", np.asarray([]))).astype(float).tolist(),
                "arm_weights": diagnostics["arm_weights"].astype(float).tolist(),
                "gripper_weights": diagnostics["gripper_weights"].astype(float).tolist(),
                "mean_arm_weighted_age": float(diagnostics["arm_weights"] @ result.candidates.ages),
                "mean_gripper_weighted_age": float(diagnostics["gripper_weights"] @ result.candidates.ages),
                "query_seed_key": query_seed_key,
                "query_rng_seed": query_seed_value,
                "query_latency_seconds": query_latency,
                "chosen_action_7d": action.astype(float).tolist(),
            }
        )
        if done:
            break
    if capture_prefix and len(prefix_actions) != PAIRING_PREFIX_LENGTH:
        raise RuntimeError("episode terminated before SmolVLA common prefix ended")
    record: dict[str, Any] = {
        "task": task_key,
        "method": method,
        "environment_seed": int(env_seed),
        "requested_initial_state_id": int(state_id),
        "actual_initial_state_id": actual_state_id,
        "cadence_h": 16,
        "prediction_horizon": 50,
        "success": bool(success),
        "completion_steps": completion_step,
        "environment_steps": len(step_log),
        "policy_queries": len(query_log),
        "query_rate": len(query_log) / float(len(step_log)),
        "query_steps": [row["query_physical_step_q"] for row in query_log],
        "query_seed_keys": [row["query_seed_key"] for row in query_log],
        "query_rng_seeds": [row["query_rng_seed"] for row in query_log],
        "mean_candidate_count": float(np.mean([row["candidate_count"] for row in step_log])),
        "mean_arm_weighted_source_age": float(np.mean([row["mean_arm_weighted_age"] for row in step_log])),
        "mean_gripper_weighted_source_age": float(np.mean([row["mean_gripper_weighted_age"] for row in step_log])),
        "step_log": step_log,
        "query_log": query_log,
    }
    if capture_prefix:
        record["pairing_trace"] = {
            "initial_observation": initial_observation,
            "initial_sim_state": initial_sim_state,
            "initial_chunk": initial_chunk,
            "prefix_actions": np.stack(prefix_actions),
            "prefix_sim_states": np.stack(prefix_sim_states),
            "prefix_observations": prefix_observations,
        }
    return record


def compare_prefix(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    first = reference["pairing_trace"]
    second = candidate["pairing_trace"]
    initial_observation = compare_array_maps(first["initial_observation"], second["initial_observation"])
    simulator_state = compare_array_maps(
        {"initial": first["initial_sim_state"], "prefix": first["prefix_sim_states"]},
        {"initial": second["initial_sim_state"], "prefix": second["prefix_sim_states"]},
    )
    chunk_equal = np.array_equal(first["initial_chunk"], second["initial_chunk"])
    actions_equal = np.array_equal(first["prefix_actions"], second["prefix_actions"])
    post_observations = [
        compare_array_maps(a, b)
        for a, b in zip(first["prefix_observations"], second["prefix_observations"], strict=True)
    ]
    maximum = max(
        initial_observation["max_absolute_difference"],
        simulator_state["max_absolute_difference"],
        float(np.max(np.abs(first["initial_chunk"] - second["initial_chunk"]))),
        float(np.max(np.abs(first["prefix_actions"] - second["prefix_actions"]))),
        *(row["max_absolute_difference"] for row in post_observations),
    )
    return {
        "reference_method": reference["method"],
        "candidate_method": candidate["method"],
        "initial_observation_exact": initial_observation["exact"],
        "initial_simulator_state_and_prefix_exact": simulator_state["exact"],
        "initial_chunk_exact": bool(chunk_equal),
        "common_prefix_actions_exact": bool(actions_equal),
        "post_action_observations_exact": all(row["exact"] for row in post_observations),
        "max_absolute_difference": maximum,
        "passed": bool(
            initial_observation["exact"]
            and simulator_state["exact"]
            and chunk_equal
            and actions_equal
            and all(row["exact"] for row in post_observations)
        ),
    }


def build_runtime(protocol: dict[str, Any], task_key: str, gpu: str):
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    checkpoint = Path(protocol["policy"]["smolvla_checkpoint"]).resolve()
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.pretrained_path = checkpoint
    if int(cfg.chunk_size) != 50 or int(cfg.n_action_steps) != 1:
        raise RuntimeError("SmolVLA checkpoint/config does not match H_pred=50, n_action_steps=1")
    task = task_map(protocol)[task_key]
    env_config = LiberoEnv(
        task=task["suite"],
        task_ids=[int(task["task_id"])],
        fps=30,
        obs_type="pixels_agent_pos",
        camera_name="agentview_image,robot0_eye_in_hand_image",
        init_states=True,
        observation_width=256,
        observation_height=256,
        control_mode="relative",
    )
    def make_fresh_env(environment_seed: int):
        random.seed(int(environment_seed))
        np.random.seed(int(environment_seed))
        return make_env(env_config, n_envs=1, use_async_envs=False)[task["suite"]][int(task["task_id"])]
    policy = make_policy(cfg=cfg, env_cfg=env_config)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(cfg.device)}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_config, policy_cfg=cfg)
    prototype = make_fresh_env(0)
    try:
        max_steps = int(np.asarray(prototype.call("_max_episode_steps")).reshape(-1)[0])
    finally:
        prototype.close()
    return make_fresh_env, policy, (env_preprocessor, env_postprocessor, preprocessor, postprocessor), torch, max_steps


def summarize_method(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [bool(row["success"]) for row in episodes]
    steps = sum(int(row["environment_steps"]) for row in episodes)
    queries = sum(int(row["policy_queries"]) for row in episodes)
    return {
        "successes": successes,
        "success_count": int(sum(successes)),
        "episodes": len(episodes),
        "success_rate": float(np.mean(successes)),
        "policy_queries": queries,
        "environment_steps": steps,
        "query_rate": queries / float(steps),
        "mean_candidate_count": float(np.mean([row["mean_candidate_count"] for row in episodes])),
        "mean_arm_weighted_source_age": float(np.mean([row["mean_arm_weighted_source_age"] for row in episodes])),
        "mean_gripper_weighted_source_age": float(np.mean([row["mean_gripper_weighted_source_age"] for row in episodes])),
        "episodes_detail": episodes,
    }


def semantic_smoke() -> None:
    key, seed = smolvla_query_seed("libero_object:task3", 10, 2000, 16)
    assert key == "smolvla|libero_object:task3|state=10|env_seed=2000|q=16"
    assert seed == smolvla_query_seed("libero_object:task3", 10, 2000, 16)[1]
    assert seed != smolvla_query_seed("libero_object:task3", 10, 2000, 8)[1]
    print(json.dumps({"status": "smolvla_group_memory_cpu_semantic_smoke_pass", "key": key, "seed": seed}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--task")
    parser.add_argument("--methods", default="M2_shared_cogact_h16,M3_group_cogact_h16")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--pairing-smoke", action="store_true")
    parser.add_argument("--pairing-audit", type=Path)
    parser.add_argument("--semantic-smoke", action="store_true")
    args = parser.parse_args()
    if args.semantic_smoke:
        semantic_smoke()
        return
    protocol = load_protocol(args.protocol)
    if args.task not in task_map(protocol):
        raise SystemExit(f"task must be one of {sorted(task_map(protocol))}")
    methods = [value for value in args.methods.split(",") if value]
    if not methods or any(value not in RUNNABLE_METHODS for value in methods):
        raise SystemExit(f"methods must be a nonempty subset of {RUNNABLE_METHODS}")
    if args.pairing_smoke:
        if args.output is None:
            raise SystemExit("--output is required for --pairing-smoke")
        states, seeds, _ = validate_values(protocol)
        make_fresh_env, policy, processors, torch, _ = build_runtime(protocol, args.task, args.gpu)
        traces = {}
        for method in methods:
            env = make_fresh_env(seeds[0])
            try:
                traces[method] = run_episode(
                    env=env,
                    policy=policy,
                    processors=processors,
                    torch=torch,
                    task_key=args.task,
                    method=method,
                    state_id=states[0],
                    env_seed=seeds[0],
                    protocol=protocol,
                    max_steps=PAIRING_PREFIX_LENGTH,
                    capture_prefix=True,
                )
            finally:
                env.close()
        reference = traces[methods[0]]
        comparisons = [compare_prefix(reference, traces[method]) for method in methods[1:]]
        result = {
            "status": "smolvla_paired_common_prefix_pass" if all(row["passed"] for row in comparisons) else "smolvla_paired_common_prefix_fail",
            "task": args.task,
            "state_id": states[0],
            "environment_seed": seeds[0],
            "methods": methods,
            "reference_method": methods[0],
            "comparisons": comparisons,
            "passed": all(row["passed"] for row in comparisons),
        }
        atomic_json(args.output, result)
        if not result["passed"]:
            raise RuntimeError("SmolVLA strict pairing smoke failed")
        return
    if args.output is None or args.pairing_audit is None:
        raise SystemExit("full rollout requires --output and a passed --pairing-audit artifact")
    pairing = json.loads(args.pairing_audit.read_text())
    if pairing.get("passed") is not True:
        raise RuntimeError("BLOCKED_BY_PAIRING_AUDIT: pairing artifact is not a pass")
    states, seeds, kernel = validate_values(protocol)
    make_fresh_env, policy, processors, torch, max_steps = build_runtime(protocol, args.task, args.gpu)
    output: dict[str, Any] = {
        "status": "running",
        "policy": "SmolVLA",
        "protocol": str(args.protocol.resolve()),
        "sol_audit_commit": protocol["coordination"]["sol_audit_commit"],
        "sol_decision": protocol["coordination"]["sol_decision"],
        "shared_kernel": kernel,
        "task": args.task,
        "methods": methods,
        "pairing_audit": str(args.pairing_audit.resolve()),
        "started_at": time.time(),
        "methods_result": {},
    }
    progress = {"pid": os.getpid(), "task": args.task, "completed_methods": 0, "completed_episodes": 0}
    try:
        for method in methods:
            episodes = []
            for state_id, env_seed in zip(states, seeds, strict=True):
                env = make_fresh_env(env_seed)
                try:
                    episodes.append(
                        run_episode(
                            env=env,
                            policy=policy,
                            processors=processors,
                            torch=torch,
                            task_key=args.task,
                            method=method,
                            state_id=state_id,
                            env_seed=env_seed,
                            protocol=protocol,
                            max_steps=max_steps,
                        )
                    )
                finally:
                    env.close()
                progress["completed_episodes"] += 1
                write_path = args.progress_file
                if write_path is not None:
                    atomic_json(write_path, progress)
            output["methods_result"][method] = summarize_method(episodes)
            progress["completed_methods"] += 1
            atomic_json(args.output, output)
        output["status"] = "complete"
        output["finished_at"] = time.time()
        atomic_json(args.output, output)
    finally:
        env.close()
    print(json.dumps({"status": output["status"], "task": args.task, "methods": methods}, indent=2))


if __name__ == "__main__":
    main()
