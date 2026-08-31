"""Run only C1 and C2 for the frozen asymmetric-reuse development gate."""

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


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
OLD_ROOT = REPO_ROOT / "experiments" / "group_delay_factorial_act20"
sys.path.insert(0, str(EXPERIMENT_ROOT))
sys.path.insert(0, str(OLD_ROOT))

from asymmetric_executor import ACTION_DIM, C1, C2, CHUNK_LENGTH, METHODS, make_executor  # noqa: E402
from run_factorial import (  # noqa: E402
    construct_env,
    extract_success,
    load_task_runtime,
    query_act_chunk,
    reset_policy_rng,
    sim_state_snapshot,
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_outcome_rollout":
        raise RuntimeError("protocol is not frozen before outcomes")
    cohort = protocol["cohort"]
    if cohort["primary_task_ids"] != list(range(1, 10)):
        raise RuntimeError("primary cohort drifted from Object tasks 1-9")
    if len(cohort["state_ids"]) != 14 or cohort["primary_paired_blocks"] != 126:
        raise RuntimeError("primary cohort block count is not 126")
    if protocol["rollout"]["new_total_episodes"] != 252:
        raise RuntimeError("new rollout is not exactly 252 episodes")
    if [condition["name"] for condition in protocol["conditions"]] != list(METHODS):
        raise RuntimeError("protocol conditions differ from C1/C2")
    if protocol["baseline_reuse_verification"]["hard_h16_rerun_required"] is not False:
        raise RuntimeError("hard h16 baseline was not verified before freezing")
    return protocol


def run_episode(
    runtime: dict[str, Any], method: str, state_id: int, environment_seed: int
) -> dict[str, Any]:
    import torch

    episode_started = time.perf_counter()
    env = construct_env(runtime, environment_seed)
    try:
        env.envs[0].init_state_id = int(state_id)
        if int(env.envs[0].init_state_id) != int(state_id):
            raise RuntimeError("initial-state assignment mismatch")
        random.seed(int(environment_seed))
        np.random.seed(int(environment_seed))
        reset_policy_rng(torch, runtime["policy_rng_seed"])
        runtime["policy"].reset()
        observation, _ = env.reset(seed=[int(environment_seed)])
        initial_sim_state, initial_body_pos = sim_state_snapshot(env)
        initial_image_means = {
            key: float(np.asarray(value).mean())
            for key, value in observation["pixels"].items()
        }
        executor = make_executor(method)
        step_log: list[dict[str, Any]] = []
        query_steps: list[int] = []
        scheduled_query_steps: list[int] = []
        policy_call_latencies: list[float] = []
        success = False
        completion_step: int | None = None
        last_info: Any = {"is_success": False}
        last_reward: Any = 0.0
        last_done = False

        for target_t in range(int(runtime["max_steps"])):
            query_started = time.perf_counter()
            result = executor.step(
                target_t,
                lambda: query_act_chunk(observation, env, runtime)[0],
            )
            query_latency = time.perf_counter() - query_started if result.queried else None
            if result.queried:
                query_steps.append(target_t)
                policy_call_latencies.append(float(query_latency))
            if result.scheduled_query_q is not None:
                scheduled_query_steps.append(int(result.scheduled_query_q))
            if method == C1:
                if result.grip_source_q % 16 != 0:
                    raise RuntimeError("C1 gripper source q is not an h16 schedule point")
                if not executor.schedule.has_chunk(result.grip_source_q):
                    raise RuntimeError("C1 gripper source chunk was not present in the cache")
                if result.grip_source_q + result.grip_offset != target_t:
                    raise RuntimeError("C1 gripper source violated q+k=t")
                if target_t >= 16 and not 16 <= result.grip_offset <= 31:
                    raise RuntimeError("C1 gripper offset is outside the previous-chunk range")
                if result.grip_offset >= 100:
                    raise RuntimeError("C1 gripper offset exceeds ACT chunk length")
            action = result.action.astype(np.float32, copy=False)
            observation, reward, terminated, truncated, info = env.step(action[None])
            terminated = bool(np.asarray(terminated).reshape(-1)[0])
            truncated = bool(np.asarray(truncated).reshape(-1)[0])
            done = terminated or truncated
            if done:
                success = extract_success(info, reward)
                completion_step = target_t + 1 if success else None
            last_info, last_reward, last_done = info, reward, done
            step_log.append(
                {
                    "t": int(result.target_t),
                    "queried": bool(result.queried),
                    "query_q": result.query_q,
                    "arm_source_q": int(result.arm_source_q),
                    "arm_offset": int(result.arm_offset),
                    "grip_source_q": int(result.grip_source_q),
                    "grip_offset": int(result.grip_offset),
                    "executed_action_7d": result.action.astype(float).tolist(),
                    "gripper_source_chunk_cached": bool(result.gripper_from_cached_chunk),
                    "physical_target_t": int(target_t),
                    "policy_queried_at_t": bool(result.queried),
                    "query_physical_step_q": result.query_q,
                    "scheduled_query_q": result.scheduled_query_q,
                    "fresh_query_q": result.fresh_query_q,
                    "scheduled_source_q": int(result.scheduled_source_q),
                    "scheduled_chunk_offset": int(result.scheduled_offset),
                    "arm_source_query_q": int(result.arm_source_q),
                    "arm_chunk_offset": int(result.arm_offset),
                    "gripper_source_query_q": int(result.grip_source_q),
                    "gripper_chunk_offset": int(result.grip_offset),
                    "arm_source_age": int(result.arm_age),
                    "gripper_source_age": int(result.grip_age),
                    "action": result.action.astype(float).tolist(),
                    "scheduled_action": result.scheduled_action.astype(float).tolist(),
                    "fresh_action": None if result.fresh_action is None else result.fresh_action.astype(float).tolist(),
                    "previous_action": None if result.previous_action is None else result.previous_action.astype(float).tolist(),
                    "query_latency_seconds": None if query_latency is None else float(query_latency),
                    "success_termination": bool(success) if done else None,
                    "terminated": terminated,
                    "truncated": truncated,
                }
            )
            if done:
                break

        if not step_log:
            raise RuntimeError("episode executed no controller steps")
        steps = len(step_log)
        expected_scheduled_queries = list(range(0, steps, 16))
        if scheduled_query_steps != expected_scheduled_queries:
            raise RuntimeError("scheduled h16 query schedule drifted")
        if method == C1:
            expected_policy_queries = (steps + 15) // 16
            if query_steps != expected_scheduled_queries or len(query_steps) != expected_policy_queries:
                raise RuntimeError("C1 compute-parity assertion failed")
            gripper_cache_ok = all(
                bool(row["grip_source_q"] % 16 == 0)
                and bool(row["grip_source_q"] in executor.schedule.chunks)
                and int(row["grip_source_q"] + row["grip_offset"]) == int(row["t"])
                and (int(row["t"]) < 16 or 16 <= int(row["grip_offset"]) <= 31)
                and int(row["grip_offset"]) < 100
                for row in step_log
            )
            compute_parity = {
                "valid": bool(gripper_cache_ok),
                "expected_policy_queries_ceil_T_over_16": expected_policy_queries,
                "observed_policy_queries": len(query_steps),
                "query_steps_exact": query_steps == expected_scheduled_queries,
                "gripper_sources_h16_aligned": True,
                "gripper_source_chunks_were_cached": bool(gripper_cache_ok),
                "no_gripper_behalf_query": True,
                "same_target_q_plus_k_equals_t": True,
                "post_t16_gripper_offset_16_to_31": all(int(row["t"]) < 16 or 16 <= int(row["grip_offset"]) <= 31 for row in step_log),
                "gripper_offset_below_chunk_length": all(int(row["grip_offset"]) < 100 for row in step_log),
            }
            if not compute_parity["valid"]:
                raise RuntimeError("C1 compute-parity assertion failed")
        else:
            expected_policy_queries = steps
            if query_steps != list(range(steps)):
                raise RuntimeError("C2 dense query schedule drifted")
            arm_sources = {int(row["arm_source_q"]) for row in step_log}
            compute_parity = {
                "valid": True,
                "expected_policy_queries": expected_policy_queries,
                "observed_policy_queries": len(query_steps),
                "dense_query_steps_exact": query_steps == list(range(steps)),
                "scheduled_arm_source_h16_aligned": all(int(row["arm_source_q"]) % 16 == 0 for row in step_log),
                "distinct_arm_source_chunks": len(arm_sources),
                "expected_distinct_arm_source_chunks_ceil_T_over_16": (steps + 15) // 16,
                "arm_source_chunk_count_exact": len(arm_sources) == (steps + 15) // 16,
                "same_target_q_plus_k_equals_t": all(int(row["grip_source_q"]) + int(row["grip_offset"]) == int(row["t"]) for row in step_log),
            }
            if not compute_parity["scheduled_arm_source_h16_aligned"] or not compute_parity["arm_source_chunk_count_exact"]:
                raise RuntimeError("C2 arm-source schedule assertion failed")
        grip_ages = [int(row["gripper_source_age"]) for row in step_log]
        distinct_arm_sources = sorted({int(row["arm_source_q"]) for row in step_log})
        return {
            "task_id": int(runtime["task_id"]),
            "task_name": runtime["task_name"],
            "method": method,
            "requested_initial_state_id": int(state_id),
            "environment_seed": int(environment_seed),
            "environment_construction_seed": int(environment_seed),
            "policy_rng_seed": int(runtime["policy_rng_seed"]),
            "fresh_environment_instance": True,
            "max_episode_steps": int(runtime["max_steps"]),
            "success": bool(success),
            "completion_step": completion_step,
            "environment_steps": steps,
            "policy_queries": len(query_steps),
            "query_rate": len(query_steps) / steps,
            "query_steps": query_steps,
            "scheduled_query_steps": scheduled_query_steps,
            "compute_parity_assertions": compute_parity,
            "distinct_arm_source_chunks": len(distinct_arm_sources),
            "arm_source_chunk_queries": distinct_arm_sources,
            "wall_clock_seconds": float(time.perf_counter() - episode_started),
            "mean_policy_call_latency_seconds": float(np.mean(policy_call_latencies)),
            "policy_call_count_for_latency": len(policy_call_latencies),
            "mean_gripper_source_age": float(np.mean(grip_ages)),
            "min_gripper_source_age": int(min(grip_ages)),
            "max_gripper_source_age": int(max(grip_ages)),
            "initial_image_means": initial_image_means,
            "initial_sim_state": initial_sim_state.astype(float).tolist(),
            "initial_model_body_pos": initial_body_pos.astype(float).tolist(),
            "step_log": step_log,
            "terminal_info_success": bool(last_info.get("is_success", False)) if isinstance(last_info, dict) and last_done else bool(success),
            "terminal_reward": float(np.asarray(last_reward).reshape(-1)[0]) if last_done else None,
        }
    finally:
        env.close()


def task_result_skeleton(runtime: dict[str, Any], protocol_path: Path) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    return {
        "schema_version": 1,
        "protocol": str(protocol_path.resolve()),
        "task_id": int(runtime["task_id"]),
        "task_name": runtime["task_name"],
        "methods": list(METHODS),
        "state_ids": [int(x) for x in protocol["cohort"]["state_ids"]],
        "episodes": {method: [] for method in METHODS},
        "finished": False,
    }


def run_task(
    protocol: dict[str, Any], protocol_path: Path, task_id: int, gpu: str,
    output_root: Path, progress_path: Path | None,
) -> None:
    output_path = output_root / "results" / f"task_{int(task_id):02d}.json"
    existing: dict[str, Any] | None = None
    if output_path.is_file():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("task_id") != int(task_id) or existing.get("methods") != list(METHODS):
            raise RuntimeError(f"existing task result identity mismatch: {output_path}")
    runtime = load_task_runtime(protocol, task_id, gpu)
    result = existing or task_result_skeleton(runtime, protocol_path)
    result.setdefault("episodes", {method: [] for method in METHODS})
    existing_keys = {
        (str(method), int(episode["requested_initial_state_id"]))
        for method, episodes in result["episodes"].items()
        for episode in episodes
    }
    state_ids = [int(x) for x in protocol["cohort"]["state_ids"]]
    seeds_by_task = protocol["cohort"]["environment_seeds_by_task"][str(task_id)]
    if len(seeds_by_task) != len(state_ids):
        raise RuntimeError("frozen task seed list does not match frozen state list")
    progress = {
        "pid": os.getpid(),
        "task_id": int(task_id),
        "gpu": str(gpu),
        "completed_episodes": sum(len(episodes) for episodes in result["episodes"].values()),
        "current_method": None,
        "current_state_id": None,
    }
    if progress_path is not None:
        atomic_json(progress_path, progress)
    for method in METHODS:
        for state_id, environment_seed in zip(state_ids, seeds_by_task, strict=True):
            key = (method, int(state_id))
            if key in existing_keys:
                continue
            progress.update({"current_method": method, "current_state_id": int(state_id)})
            if progress_path is not None:
                atomic_json(progress_path, progress)
            episode = run_episode(runtime, method, int(state_id), int(environment_seed))
            result["episodes"][method].append(episode)
            result["episodes"][method].sort(key=lambda row: int(row["requested_initial_state_id"]))
            existing_keys.add(key)
            progress["completed_episodes"] += 1
            atomic_json(output_path, result)
            if progress_path is not None:
                atomic_json(progress_path, progress)
            print(
                f"task={task_id} method={method} state={state_id} success={episode['success']} "
                f"steps={episode['environment_steps']} queries={episode['policy_queries']}", flush=True
            )
    result["finished"] = True
    result["finished_at"] = time.time()
    atomic_json(output_path, result)
    progress["finished"] = True
    progress["finished_at"] = result["finished_at"]
    if progress_path is not None:
        atomic_json(progress_path, progress)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=EXPERIMENT_ROOT / "protocol.json")
    parser.add_argument("--tasks", required=True, help="comma-separated Object task IDs")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--progress", type=Path)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    task_ids = [int(value) for value in args.tasks.split(",") if value.strip()]
    if not task_ids or any(task_id not in protocol["cohort"]["primary_task_ids"] for task_id in task_ids):
        raise SystemExit("tasks must be a non-empty subset of frozen primary Object tasks 1-9")
    for task_id in task_ids:
        run_task(protocol, args.protocol, task_id, args.gpu, args.output_root, args.progress)


if __name__ == "__main__":
    main()
