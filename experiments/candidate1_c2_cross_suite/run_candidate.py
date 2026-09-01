"""Run frozen Candidate-1 validation or C2 task shards."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
ASYMMETRIC_ROOT = REPO_ROOT / "experiments" / "asymmetric_chunk_reuse_dev"
CROSS_SUITE_ROOT = REPO_ROOT / "experiments" / "cross_suite_confirmation"
sys.path.insert(0, str(ASYMMETRIC_ROOT))
sys.path.insert(0, str(CROSS_SUITE_ROOT))

from asymmetric_executor import C2, CHUNK_LENGTH, H16ArmFreshGripExecutor  # noqa: E402
import run_confirmation as confirmation  # noqa: E402


RUNNER_VERSION = "candidate1_c2_cross_suite_v1"
EXPECTED_TASKS = {
    "libero_goal": [4, 6, 7, 8, 9],
    "libero_10": [0, 2, 4, 6, 7],
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_protocol(path: Path, *, require_frozen: bool = False) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    allowed_status = {"frozen_before_outcome_rollout"} if require_frozen else {
        "pre_outcome_draft",
        "frozen_before_outcome_rollout",
    }
    if protocol.get("status") not in allowed_status:
        raise RuntimeError(f"unexpected Candidate-1 protocol status: {protocol.get('status')!r}")
    if protocol.get("authorized_condition") != C2:
        raise RuntimeError("Candidate-1 condition is not the validated C2 executor")
    if protocol["cohort"]["primary_tasks"] != EXPECTED_TASKS:
        raise RuntimeError("Candidate-1 task cohort drifted")
    if protocol["cohort"]["state_ids"] != list(range(14)):
        raise RuntimeError("Candidate-1 state cohort drifted")
    if int(protocol["cohort"]["paired_blocks"]) != 140:
        raise RuntimeError("Candidate-1 block count is not 140")
    if int(protocol["rollout"]["new_c2_experimental_episodes"]) != 140:
        raise RuntimeError("Candidate-1 C2 workload is not 140 episodes")
    if protocol["runtime"]["policy_temporal_ensemble"] is not False:
        raise RuntimeError("ACT temporal ensemble must remain disabled")
    if protocol["runtime"]["action_smoothing"] is not False:
        raise RuntimeError("action smoothing must remain disabled")
    validate_task_seeds(protocol)
    return protocol


def validate_task_seeds(protocol: dict[str, Any]) -> None:
    suite_indices = protocol["cohort"]["suite_index"]
    states = protocol["cohort"]["state_ids"]
    for task in protocol["cohort"]["tasks"]:
        expected = [
            340000 + 1000 * int(suite_indices[task["suite"]]) + 100 * int(task["task_id"]) + state
            for state in states
        ]
        if task["environment_seeds"] != expected:
            raise RuntimeError(f"seed list drifted for {task_label(task)}")


def task_label(task: dict[str, Any]) -> str:
    return f"{task['suite']}:task{int(task['task_id'])}"


def task_map(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {task_label(task): task for task in protocol["cohort"]["tasks"]}


def validation_results(protocol: dict[str, Any]) -> dict[str, Any]:
    results = protocol["pre_outcome_validation"].get("results")
    return {} if results is None else dict(results)


def record_validation(protocol_path: Path, name: str, result: object) -> None:
    protocol = load_protocol(protocol_path)
    if protocol["status"] != "pre_outcome_draft":
        raise RuntimeError("pre-outcome validation records cannot modify a frozen protocol")
    results = validation_results(protocol)
    results[name] = result
    protocol["pre_outcome_validation"]["results"] = results
    atomic_json(protocol_path, protocol)


def environment_metadata(torch: Any) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("lerobot", "libero", "robosuite", "mujoco"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "lerobot": packages["lerobot"],
        "libero": packages["libero"],
        "robosuite": packages["robosuite"],
        "mujoco": packages["mujoco"],
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_names_visible": gpu_names,
    }


def run_checkpoint_preflight(protocol_path: Path, gpu: str) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    rows: list[dict[str, Any]] = []
    for task in protocol["cohort"]["tasks"]:
        checkpoint = Path(task["checkpoint"])
        exists = checkpoint.is_dir() and (checkpoint / "config.json").is_file() and (
            checkpoint / "model.safetensors"
        ).is_file()
        if not exists:
            raise FileNotFoundError(f"missing configured checkpoint: {checkpoint}")
        runtime = confirmation.build_task_runtime(task, gpu)
        policy_config = runtime["policy"].config
        row = {
            "task": task_label(task),
            "checkpoint": str(checkpoint),
            "path_exists": True,
            "loaded_as_act": getattr(policy_config, "type", None) == "act",
            "chunk_size": int(policy_config.chunk_size),
            "action_dim": int(policy_config.action_feature.shape[0]),
            "temporal_ensemble_disabled": policy_config.temporal_ensemble_coeff is None,
            "expected_preprocessing_constructed": all(
                runtime[name] is not None
                for name in ("preprocessor", "postprocessor", "env_preprocessor", "env_postprocessor")
            ),
        }
        if row != {
            **row,
            "loaded_as_act": True,
            "chunk_size": CHUNK_LENGTH,
            "action_dim": 7,
            "temporal_ensemble_disabled": True,
            "expected_preprocessing_constructed": True,
        }:
            raise RuntimeError(f"checkpoint contract mismatch: {row}")
        rows.append(row)
        torch = runtime["torch"]
        del runtime
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    import torch

    result = {
        "status": "PASS",
        "checkpoints_checked": len(rows),
        "action_smoothing_disabled_by_runner": protocol["runtime"]["action_smoothing"] is False,
        "rows": rows,
        "environment_metadata": environment_metadata(torch),
    }
    record_validation(protocol_path, "checkpoint_preflight", result)
    return result


def frozen_reference_episode(task: dict[str, Any], state_id: int, method: str) -> dict[str, Any]:
    path = CROSS_SUITE_ROOT / "results" / f"{task['suite']}_task{int(task['task_id'])}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        episode
        for episode in data["episodes"][method]
        if int(episode["requested_initial_state_id"]) == int(state_id)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one frozen {method} reference for {task_label(task)} state {state_id}")
    return matches[0]


def compare_fresh_episode(
    task: dict[str, Any], state_id: int, seed: int, current: dict[str, Any]
) -> dict[str, Any]:
    frozen = frozen_reference_episode(task, state_id, "FRESH")
    frozen_cameras = frozen["initial_image_means"]
    current_cameras = current["initial_image_means"]
    camera_keys_match = set(frozen_cameras) == set(current_cameras)
    camera_differences = {
        key: float(current_cameras[key] - frozen_cameras[key])
        for key in sorted(set(frozen_cameras) & set(current_cameras))
    }
    cameras_consistent = camera_keys_match and all(
        np.isclose(current_cameras[key], frozen_cameras[key], rtol=0.0, atol=1e-12)
        for key in frozen_cameras
    )
    return {
        "task": task_label(task),
        "state_id": int(state_id),
        "seed": int(seed),
        "frozen_success": bool(frozen["success"]),
        "rerun_success": bool(current["success"]),
        "binary_success_exact": bool(current["success"]) == bool(frozen["success"]),
        "frozen_completion_step": frozen["completion_step"],
        "rerun_completion_step": current["completion_step"],
        "completion_step_exact": current["completion_step"] == frozen["completion_step"],
        "frozen_environment_steps": int(frozen["environment_steps"]),
        "rerun_environment_steps": int(current["environment_steps"]),
        "environment_steps_exact": int(current["environment_steps"]) == int(frozen["environment_steps"]),
        "frozen_initial_camera_means": frozen_cameras,
        "rerun_initial_camera_means": current_cameras,
        "initial_camera_mean_differences": camera_differences,
        "initial_camera_statistics_consistent": bool(cameras_consistent),
        "excluded_from_candidate_inference": True,
    }


def run_fresh_reproducibility_control(protocol_path: Path, gpu: str) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    tasks = task_map(protocol)
    planned = protocol["pre_outcome_validation"]["fresh_reproducibility_control"]["blocks"]
    stored = validation_results(protocol).get("fresh_reproducibility_control", {})
    records = list(stored.get("records", []))
    existing = {(row["task"], int(row["state_id"])) for row in records}
    for row in records:
        if not row["binary_success_exact"]:
            raise RuntimeError("stored Fresh reproducibility success mismatch requires a hard stop")

    current_runtime_label: str | None = None
    runtime: dict[str, Any] | None = None
    for block in planned:
        label = f"{block['suite']}:task{int(block['task_id'])}"
        key = (label, int(block["state_id"]))
        if key in existing:
            continue
        if label != current_runtime_label:
            if runtime is not None:
                torch = runtime["torch"]
                del runtime
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            runtime = confirmation.build_task_runtime(tasks[label], gpu)
            current_runtime_label = label
        assert runtime is not None
        episode = confirmation.run_episode(
            runtime,
            "FRESH",
            int(block["state_id"]),
            int(block["seed"]),
            int(protocol["runtime"]["policy_rng_seed"]),
        )
        record = compare_fresh_episode(
            tasks[label], int(block["state_id"]), int(block["seed"]), episode
        )
        records.append(record)
        existing.add(key)
        partial = {
            "status": "RUNNING" if len(records) < len(planned) else "PASS",
            "records": records,
            "reference_outcomes_replaced": False,
            "entered_candidate_inference": False,
            "increased_experimental_n": False,
        }
        record_validation(protocol_path, "fresh_reproducibility_control", partial)
        if not record["binary_success_exact"]:
            raise RuntimeError(
                f"Fresh reproducibility success mismatch at {label} state {block['state_id']}"
            )
    if runtime is not None:
        torch = runtime["torch"]
        del runtime
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if len(records) != 6:
        raise RuntimeError(f"Fresh reproducibility control has {len(records)} records, expected 6")
    result = {
        "status": "PASS",
        "records": records,
        "all_binary_success_exact": all(row["binary_success_exact"] for row in records),
        "all_completion_steps_exact": all(row["completion_step_exact"] for row in records),
        "all_environment_steps_exact": all(row["environment_steps_exact"] for row in records),
        "all_initial_camera_statistics_consistent": all(
            row["initial_camera_statistics_consistent"] for row in records
        ),
        "reference_outcomes_replaced": False,
        "entered_candidate_inference": False,
        "increased_experimental_n": False,
    }
    record_validation(protocol_path, "fresh_reproducibility_control", result)
    return result


def assert_c2_step(result: Any, target_t: int) -> None:
    expected_q = 16 * (target_t // 16)
    expected_k = target_t - expected_q
    if result.arm_source_q != expected_q or result.arm_offset != expected_k:
        raise RuntimeError("C2 arm source/index differs from scheduled HARD_H16 semantics")
    if result.grip_source_q != target_t or result.grip_offset != 0:
        raise RuntimeError("C2 gripper is not the offset-0 component of the dense current query")
    if result.arm_source_q + result.arm_offset != target_t:
        raise RuntimeError("C2 arm violated q+k=t")
    if result.grip_source_q + result.grip_offset != target_t:
        raise RuntimeError("C2 gripper violated q+k=t")
    if not result.policy_queried or result.query_q != target_t:
        raise RuntimeError("C2 did not perform one whole-policy query at the current step")


def reset_episode(runtime: dict[str, Any], state_id: int, seed: int, policy_rng_seed: int):
    torch = runtime["torch"]
    env = confirmation.make_fresh_env(runtime, seed)
    env.envs[0].init_state_id = int(state_id)
    if int(env.envs[0].init_state_id) != int(state_id):
        env.close()
        raise RuntimeError("initial-state assignment mismatch")
    random.seed(int(seed))
    np.random.seed(int(seed))
    confirmation.reset_policy_rng(torch, policy_rng_seed)
    runtime["policy"].reset()
    observation, _ = env.reset(seed=[int(seed)])
    return env, observation


def run_c2_episode(
    runtime: dict[str, Any], state_id: int, seed: int, policy_rng_seed: int
) -> dict[str, Any]:
    task = runtime["task"]
    env, observation = reset_episode(runtime, state_id, seed, policy_rng_seed)
    started = time.perf_counter()
    try:
        initial_image_means = {
            key: float(np.asarray(value).mean()) for key, value in observation["pixels"].items()
        }
        executor = H16ArmFreshGripExecutor()
        step_log: list[dict[str, Any]] = []
        query_latencies: list[float] = []
        success = False
        completion_step: int | None = None
        last_info: Any = {"is_success": False}
        last_reward: Any = 0.0
        last_done = False
        for target_t in range(int(task["max_episode_steps"])):
            query_started = time.perf_counter()
            result = executor.step(
                target_t, lambda: confirmation.query_act_chunk(observation, env, runtime)[0]
            )
            query_latencies.append(time.perf_counter() - query_started)
            assert_c2_step(result, target_t)
            observation, reward, terminated, truncated, info = env.step(
                result.action.astype(np.float32, copy=False)[None]
            )
            terminated = bool(np.asarray(terminated).reshape(-1)[0])
            truncated = bool(np.asarray(truncated).reshape(-1)[0])
            done = terminated or truncated
            if done:
                success = confirmation.extract_success(info, reward)
                completion_step = target_t + 1 if success else None
            last_info, last_reward, last_done = info, reward, done
            step_log.append(
                {
                    "physical_target_t": int(target_t),
                    "policy_queried_at_t": True,
                    "query_physical_step_q": int(result.query_q),
                    "scheduled_query_q": result.scheduled_query_q,
                    "arm_source_query_q": int(result.arm_source_q),
                    "arm_chunk_offset": int(result.arm_offset),
                    "gripper_source_query_q": int(result.grip_source_q),
                    "gripper_chunk_offset": int(result.grip_offset),
                    "arm_source_age": int(result.arm_age),
                    "gripper_source_age": int(result.grip_age),
                    "action": result.action.astype(float).tolist(),
                    "terminated": terminated,
                    "truncated": truncated,
                }
            )
            if done:
                break
        steps = len(step_log)
        if steps == 0 or [row["query_physical_step_q"] for row in step_log] != list(range(steps)):
            raise RuntimeError("C2 total policy query schedule drifted")
        return {
            "runner_version": RUNNER_VERSION,
            "suite": task["suite"],
            "task_id": int(task["task_id"]),
            "task_name": task["task_name"],
            "method": C2,
            "requested_initial_state_id": int(state_id),
            "environment_seed": int(seed),
            "environment_construction_seed": int(seed),
            "policy_rng_seed": int(policy_rng_seed),
            "fresh_environment_instance": True,
            "max_episode_steps": int(task["max_episode_steps"]),
            "success": bool(success),
            "completion_step": completion_step,
            "environment_steps": steps,
            "policy_queries": steps,
            "query_rate": 1.0,
            "query_steps": list(range(steps)),
            "mean_arm_source_age": float(np.mean([row["arm_source_age"] for row in step_log])),
            "min_arm_source_age": int(min(row["arm_source_age"] for row in step_log)),
            "max_arm_source_age": int(max(row["arm_source_age"] for row in step_log)),
            "mean_gripper_source_age": 0.0,
            "min_gripper_source_age": 0,
            "max_gripper_source_age": 0,
            "wall_clock_seconds": float(time.perf_counter() - started),
            "mean_policy_call_latency_seconds": float(np.mean(query_latencies)),
            "initial_image_means": initial_image_means,
            "step_log": step_log,
            "terminal_info_success": bool(last_info.get("is_success", False))
            if isinstance(last_info, dict) and last_done
            else bool(success),
            "terminal_reward": float(np.asarray(last_reward).reshape(-1)[0]) if last_done else None,
        }
    finally:
        env.close()


def run_two_step_smoke(protocol_path: Path, gpu: str) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    tasks = task_map(protocol)
    smoke_blocks = [
        (tasks["libero_goal:task4"], 0, 342400),
        (tasks["libero_10:task0"], 0, 343000),
    ]
    records: list[dict[str, Any]] = []
    for task, state_id, seed in smoke_blocks:
        runtime = confirmation.build_task_runtime(task, gpu)
        env, observation = reset_episode(
            runtime, state_id, seed, int(protocol["runtime"]["policy_rng_seed"])
        )
        try:
            executor = H16ArmFreshGripExecutor()
            steps: list[dict[str, Any]] = []
            for target_t in range(2):
                result = executor.step(
                    target_t, lambda: confirmation.query_act_chunk(observation, env, runtime)[0]
                )
                assert_c2_step(result, target_t)
                observation, _, _, _, _ = env.step(result.action.astype(np.float32)[None])
                steps.append(
                    {
                        "t": target_t,
                        "arm_source_q": int(result.arm_source_q),
                        "arm_offset": int(result.arm_offset),
                        "gripper_source_q": int(result.grip_source_q),
                        "gripper_offset": int(result.grip_offset),
                        "policy_queried": bool(result.policy_queried),
                    }
                )
            records.append(
                {
                    "task": task_label(task),
                    "state_id": state_id,
                    "seed": seed,
                    "environment_steps": 2,
                    "steps": steps,
                    "full_episode_success_inspected_or_persisted": False,
                }
            )
        finally:
            env.close()
        torch = runtime["torch"]
        del runtime
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = {
        "status": "PASS",
        "records": records,
        "total_environment_steps": 4,
        "full_episode_success_inspected_or_persisted": False,
    }
    record_validation(protocol_path, "two_step_non_outcome_smoke", result)
    return result


def result_skeleton(protocol_path: Path, task: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "runner_version": RUNNER_VERSION,
        "protocol": str(protocol_path.resolve()),
        "suite": task["suite"],
        "task_id": int(task["task_id"]),
        "task_name": task["task_name"],
        "checkpoint": task["checkpoint"],
        "methods": [C2],
        "state_ids": list(range(14)),
        "episodes": {C2: []},
        "finished": False,
    }


def run_task(
    protocol: dict[str, Any], protocol_path: Path, task: dict[str, Any], gpu: str, output_root: Path
) -> None:
    output_path = output_root / "results" / f"{task['suite']}_task{int(task['task_id'])}.json"
    existing = json.loads(output_path.read_text(encoding="utf-8")) if output_path.is_file() else None
    if existing is not None and (
        existing.get("runner_version") != RUNNER_VERSION
        or existing.get("methods") != [C2]
        or existing.get("protocol") != str(protocol_path.resolve())
        or existing.get("suite") != task["suite"]
        or int(existing.get("task_id", -1)) != int(task["task_id"])
    ):
        raise RuntimeError(f"existing result identity/version mismatch: {output_path}")
    result = existing or result_skeleton(protocol_path, task)
    episodes = result["episodes"][C2]
    states_present = [int(episode["requested_initial_state_id"]) for episode in episodes]
    if len(states_present) != len(set(states_present)):
        raise RuntimeError(f"duplicate persisted C2 task-state key in {output_path}")
    runtime = confirmation.build_task_runtime(task, gpu)
    for state_id, seed in zip(
        protocol["cohort"]["state_ids"], task["environment_seeds"], strict=True
    ):
        if int(state_id) in states_present:
            continue
        episode = run_c2_episode(
            runtime, int(state_id), int(seed), int(protocol["runtime"]["policy_rng_seed"])
        )
        episodes.append(episode)
        episodes.sort(key=lambda row: int(row["requested_initial_state_id"]))
        states_present.append(int(state_id))
        atomic_json(output_path, result)
        print(
            f"task={task_label(task)} state={state_id} success={episode['success']} "
            f"steps={episode['environment_steps']} queries={episode['policy_queries']}",
            flush=True,
        )
    result["finished"] = len(episodes) == 14
    result["finished_at"] = time.time() if result["finished"] else None
    atomic_json(output_path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--gpu", default="0")

    fresh = subparsers.add_parser("fresh-control")
    fresh.add_argument("--gpu", default="0")

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--gpu", default="0")

    rollout = subparsers.add_parser("run")
    rollout.add_argument("--tasks", required=True, help="comma-separated frozen suite:task entries")
    rollout.add_argument("--gpu", default="0")
    rollout.add_argument("--output-root", type=Path, default=ROOT)

    args = parser.parse_args()
    if args.command == "preflight":
        result = run_checkpoint_preflight(args.protocol, args.gpu)
    elif args.command == "fresh-control":
        result = run_fresh_reproducibility_control(args.protocol, args.gpu)
    elif args.command == "smoke":
        result = run_two_step_smoke(args.protocol, args.gpu)
    else:
        protocol = load_protocol(args.protocol, require_frozen=True)
        tasks = task_map(protocol)
        requested = [value.strip() for value in args.tasks.split(",") if value.strip()]
        if not requested or any(label not in tasks for label in requested):
            raise SystemExit("every requested task must belong to the frozen Candidate-1 cohort")
        for label in requested:
            run_task(protocol, args.protocol, tasks[label], args.gpu, args.output_root)
        return
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
