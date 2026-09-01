"""Run only coherent h32 and true arm16/grip32 on the frozen Object cohort."""

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


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
FACTORIAL_ROOT = REPO_ROOT / "experiments" / "group_delay_factorial_act20"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FACTORIAL_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from fixed_clock_executor import (  # noqa: E402
    ACTION_DIM,
    CHUNK_LENGTH,
    H16,
    H32,
    H32_COHERENT,
    METHODS,
    TWO_CLOCK,
    make_executor,
)
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
    if cohort["state_ids"] != [20, 21, 22, 23, 27, 31, 34, 35, 38, 39, 44, 45, 47, 48]:
        raise RuntimeError("state cohort drifted from the authoritative asymmetric protocol")
    if cohort["primary_paired_blocks"] != 126 or protocol["rollout"]["new_total_episodes"] != 252:
        raise RuntimeError("rollout is not exactly two conditions over 126 blocks")
    if [condition["name"] for condition in protocol["conditions"]] != list(METHODS):
        raise RuntimeError("protocol conditions differ from the two implemented methods")
    if protocol["exact_result_audit"]["exact_h32_found"] or protocol["exact_result_audit"]["exact_two_clock_found"]:
        raise RuntimeError("protocol says a new condition should have been reused")
    runtime = protocol["runtime"]
    if runtime["policy_checkpoint_chunk_size"] != CHUNK_LENGTH:
        raise RuntimeError("checkpoint chunk size is not 100")
    if runtime["policy_temporal_ensemble"] is not False or runtime["action_smoothing"] is not False:
        raise RuntimeError("temporal ensemble and action smoothing must remain disabled")
    if protocol["action_groups"] != {"arm": list(range(6)), "gripper": [6]}:
        raise RuntimeError("action groups drifted from the verified 7-D contract")
    return protocol


def environment_seed(task_id: int, state_id: int) -> int:
    return 330000 + 100 * int(task_id) + int(state_id)


def run_episode(runtime: dict[str, Any], method: str, state_id: int, seed: int) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    env = construct_env(runtime, seed)
    try:
        env.envs[0].init_state_id = int(state_id)
        if int(env.envs[0].init_state_id) != int(state_id):
            raise RuntimeError("initial-state assignment mismatch")
        random.seed(seed)
        np.random.seed(seed)
        reset_policy_rng(torch, runtime["policy_rng_seed"])
        runtime["policy"].reset()
        observation, _ = env.reset(seed=[seed])
        initial_sim_state, initial_body_pos = sim_state_snapshot(env)
        initial_image_means = {key: float(np.asarray(value).mean()) for key, value in observation["pixels"].items()}
        executor = make_executor(method)
        query_steps: list[int] = []
        policy_call_latencies: list[float] = []
        step_log: list[dict[str, Any]] = []
        success = False
        completion_step: int | None = None
        last_info: Any = {"is_success": False}
        last_reward: Any = 0.0
        last_done = False

        for t in range(int(runtime["max_steps"])):
            query_count_before = len(query_steps)

            def query() -> np.ndarray:
                query_started = time.perf_counter()
                chunk = query_act_chunk(observation, env, runtime)[0]
                policy_call_latencies.append(float(time.perf_counter() - query_started))
                return chunk

            decision = executor.step(query)
            if decision.policy_query:
                query_steps.append(t)
            if len(query_steps) - query_count_before != int(decision.policy_query):
                raise RuntimeError("executor query flag disagrees with policy call count")
            source_q = {group: query_steps[chunk_id] for group, chunk_id in decision.source_chunk_ids.items()}
            arm_q, grip_q = source_q["arm"], source_q["gripper"]
            arm_offset = int(decision.source_positions["arm"])
            grip_offset = int(decision.source_positions["gripper"])
            if arm_q + arm_offset != t or grip_q + grip_offset != t:
                raise RuntimeError("fixed-clock source violated q+k=t")
            if decision.source_ages != {"arm": t - arm_q, "gripper": t - grip_q}:
                raise RuntimeError("logged source age disagrees with physical source time")
            if method == H32_COHERENT:
                expected_q = H32 * (t // H32)
                if (arm_q, grip_q) != (expected_q, expected_q):
                    raise RuntimeError("coherent h32 source schedule drifted")
            else:
                if arm_q != H16 * (t // H16) or grip_q != H32 * (t // H32):
                    raise RuntimeError("independent arm16/grip32 source schedule drifted")
                expected_refresh = ("arm", "gripper") if t % H32 == 0 else (("arm",) if t % H16 == 0 else ())
                if decision.refreshed_groups != expected_refresh:
                    raise RuntimeError("two-clock group refresh boundary drifted")

            action = decision.action.astype(np.float32, copy=False)
            observation, reward, terminated, truncated, info = env.step(action[None])
            terminated = bool(np.asarray(terminated).reshape(-1)[0])
            truncated = bool(np.asarray(truncated).reshape(-1)[0])
            done = terminated or truncated
            if done:
                success = extract_success(info, reward)
                completion_step = t + 1 if success else None
            last_info, last_reward, last_done = info, reward, done
            step_log.append(
                {
                    "t": t,
                    "policy_queried_at_t": bool(decision.policy_query),
                    "query_physical_step_q": t if decision.policy_query else None,
                    "refreshed_groups": list(decision.refreshed_groups),
                    "arm_source_query_q": arm_q,
                    "arm_chunk_offset": arm_offset,
                    "gripper_source_query_q": grip_q,
                    "gripper_chunk_offset": grip_offset,
                    "arm_source_age": t - arm_q,
                    "gripper_source_age": t - grip_q,
                    "action": decision.action.astype(float).tolist(),
                    "terminated": terminated,
                    "truncated": truncated,
                    "success_termination": bool(success) if done else None,
                }
            )
            if done:
                break

        steps = len(step_log)
        period = H32 if method == H32_COHERENT else H16
        expected_queries = list(range(0, steps, period))
        if query_steps != expected_queries:
            raise RuntimeError("policy query schedule drifted")
        return {
            "task_id": int(runtime["task_id"]),
            "task_name": runtime["task_name"],
            "method": method,
            "requested_initial_state_id": int(state_id),
            "environment_seed": int(seed),
            "environment_construction_seed": int(seed),
            "policy_rng_seed": int(runtime["policy_rng_seed"]),
            "fresh_environment_instance": True,
            "max_episode_steps": int(runtime["max_steps"]),
            "success": bool(success),
            "completion_step": completion_step,
            "environment_steps": steps,
            "policy_queries": len(query_steps),
            "query_rate": len(query_steps) / steps,
            "query_steps": query_steps,
            "wall_clock_seconds": float(time.perf_counter() - started),
            "mean_policy_call_latency_seconds": float(np.mean(policy_call_latencies)),
            "policy_call_count_for_latency": len(policy_call_latencies),
            "initial_image_means": initial_image_means,
            "initial_sim_state": initial_sim_state.astype(float).tolist(),
            "initial_model_body_pos": initial_body_pos.astype(float).tolist(),
            "step_log": step_log,
            "terminal_info_success": bool(last_info.get("is_success", False)) if isinstance(last_info, dict) and last_done else bool(success),
            "terminal_reward": float(np.asarray(last_reward).reshape(-1)[0]) if last_done else None,
        }
    finally:
        env.close()


def run_task(protocol: dict[str, Any], protocol_path: Path, task_id: int, gpu: str, output_root: Path, progress_path: Path | None) -> None:
    output_path = output_root / "results" / f"task_{task_id:02d}.json"
    runtime = load_task_runtime(protocol, task_id, gpu)
    if output_path.is_file():
        result = json.loads(output_path.read_text(encoding="utf-8"))
        if result.get("task_id") != task_id or result.get("methods") != list(METHODS):
            raise RuntimeError(f"existing task result identity mismatch: {output_path}")
    else:
        result = {
            "schema_version": 1,
            "protocol": str(protocol_path.resolve()),
            "task_id": task_id,
            "task_name": runtime["task_name"],
            "methods": list(METHODS),
            "state_ids": [int(value) for value in protocol["cohort"]["state_ids"]],
            "episodes": {method: [] for method in METHODS},
            "finished": False,
        }
    existing = {
        (method, int(episode["requested_initial_state_id"]))
        for method, episodes in result["episodes"].items()
        for episode in episodes
    }
    progress = {
        "pid": os.getpid(),
        "task_id": task_id,
        "gpu": str(gpu),
        "completed_episodes": len(existing),
        "current_method": None,
        "current_state_id": None,
    }
    if progress_path is not None:
        atomic_json(progress_path, progress)
    for method in METHODS:
        for state_id in protocol["cohort"]["state_ids"]:
            key = (method, int(state_id))
            if key in existing:
                continue
            progress.update({"current_method": method, "current_state_id": int(state_id)})
            if progress_path is not None:
                atomic_json(progress_path, progress)
            episode = run_episode(runtime, method, int(state_id), environment_seed(task_id, int(state_id)))
            result["episodes"][method].append(episode)
            result["episodes"][method].sort(key=lambda row: int(row["requested_initial_state_id"]))
            existing.add(key)
            progress["completed_episodes"] = len(existing)
            atomic_json(output_path, result)
            if progress_path is not None:
                atomic_json(progress_path, progress)
            print(
                f"task={task_id} method={method} state={state_id} success={episode['success']} "
                f"steps={episode['environment_steps']} queries={episode['policy_queries']}",
                flush=True,
            )
    result["finished"] = True
    result["finished_at"] = time.time()
    atomic_json(output_path, result)
    progress.update({"finished": True, "finished_at": result["finished_at"]})
    if progress_path is not None:
        atomic_json(progress_path, progress)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--tasks", required=True, help="comma-separated Object task IDs")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--progress", type=Path)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    task_ids = [int(value) for value in args.tasks.split(",") if value.strip()]
    if not task_ids or any(task_id not in protocol["cohort"]["primary_task_ids"] for task_id in task_ids):
        raise SystemExit("tasks must be a non-empty subset of frozen Object tasks 1-9")
    for task_id in task_ids:
        run_task(protocol, args.protocol, task_id, args.gpu, args.output_root, args.progress)


if __name__ == "__main__":
    main()
