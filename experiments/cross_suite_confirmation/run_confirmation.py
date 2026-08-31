"""Run the frozen five-condition Branch K confirmation in resumable task shards."""

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
TEMPORAL_ROOT = REPO_ROOT / "experiments" / "group_delay_factorial_act20"
sys.path.insert(0, str(TEMPORAL_ROOT))

from temporal_reuse import CHUNK_LENGTH, METHODS, make_executor  # noqa: E402


RUNNER_VERSION = "branch_k_confirmation_v1"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def reset_policy_rng(torch: Any, seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_outcome_rollout":
        raise RuntimeError("confirmation protocol is not frozen before outcome rollout")
    if [row["name"] for row in protocol["conditions"]] != list(METHODS):
        raise RuntimeError("confirmation conditions differ from frozen temporal_reuse methods")
    if int(protocol["cohort"]["total_episodes"]) != 910:
        raise RuntimeError("confirmation workload is not the frozen 910 episodes")
    if protocol["checkpoint_preflight"]["status"] != "passed":
        raise RuntimeError("checkpoint preflight did not pass")
    return protocol


def build_task_runtime(task: dict[str, Any], gpu: str) -> dict[str, Any]:
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    checkpoint = Path(task["checkpoint"]).resolve()
    if not (checkpoint / "config.json").is_file() or not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"missing ACT checkpoint: {checkpoint}")
    policy_config = PreTrainedConfig.from_pretrained(checkpoint)
    policy_config.device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_config.pretrained_path = checkpoint
    policy_config.pretrained_backbone_weights = None
    if getattr(policy_config, "type", None) != "act":
        raise RuntimeError(f"expected ACT checkpoint, got {getattr(policy_config, 'type', None)!r}")
    if int(policy_config.chunk_size) != CHUNK_LENGTH or policy_config.temporal_ensemble_coeff is not None:
        raise RuntimeError("confirmation checkpoint violates frozen ACT chunk/ensemble settings")
    if int(policy_config.action_feature.shape[0]) != 7:
        raise RuntimeError("confirmation checkpoint action dimension is not 7")
    env_config = LiberoEnv(
        task=str(task["suite"]),
        task_ids=[int(task["task_id"])],
        fps=20,
        episode_length=int(task["max_episode_steps"]),
        obs_type="pixels_agent_pos",
        camera_name="agentview_image,robot0_eye_in_hand_image",
        camera_name_mapping={"agentview_image": "image", "robot0_eye_in_hand_image": "wrist_image"},
        init_states=True,
        observation_width=256,
        observation_height=256,
        control_mode="relative",
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
    return {
        "torch": torch,
        "task": task,
        "env_config": env_config,
        "policy": policy,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "env_preprocessor": env_preprocessor,
        "env_postprocessor": env_postprocessor,
    }


def make_fresh_env(runtime: dict[str, Any], environment_seed: int):
    from lerobot.envs.factory import make_env

    random.seed(int(environment_seed))
    np.random.seed(int(environment_seed))
    task = runtime["task"]
    return make_env(runtime["env_config"], n_envs=1, use_async_envs=False)[task["suite"]][int(task["task_id"])]


def prepare_processed_input(observation: dict[str, Any], env: Any, runtime: dict[str, Any]) -> Any:
    from lerobot.envs.utils import add_envs_task, preprocess_observation

    batch = preprocess_observation(observation)
    batch = add_envs_task(env, batch)
    batch = runtime["env_preprocessor"](batch)
    return runtime["preprocessor"](batch)


def query_act_chunk(observation: dict[str, Any], env: Any, runtime: dict[str, Any]) -> tuple[np.ndarray, Any]:
    import torch
    from lerobot.utils.constants import ACTION

    processed = prepare_processed_input(observation, env, runtime)
    with torch.inference_mode():
        chunk = runtime["postprocessor"](runtime["policy"].predict_action_chunk(processed))
        chunk = runtime["env_postprocessor"]({ACTION: chunk})[ACTION]
    result = chunk.detach().cpu().numpy()
    if result.shape != (1, CHUNK_LENGTH, 7):
        raise RuntimeError(f"unexpected ACT chunk shape: {result.shape}")
    return result[0].astype(np.float64, copy=False), processed


def processed_equal(left: Any, right: Any) -> bool:
    import torch

    if isinstance(left, dict) and isinstance(right, dict):
        return list(sorted(left)) == list(sorted(right)) and all(processed_equal(left[k], right[k]) for k in left)
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    return np.array_equal(np.asarray(left), np.asarray(right))


def extract_success(info: Any, reward: Any) -> bool:
    final_info = info.get("final_info") if isinstance(info, dict) else None
    if isinstance(final_info, dict) and "is_success" in final_info:
        return bool(np.asarray(final_info["is_success"]).reshape(-1)[0])
    values = np.asarray(reward).reshape(-1)
    return bool(len(values) and values[0] > 0)


def run_episode(runtime: dict[str, Any], method: str, state_id: int, seed: int, policy_rng_seed: int) -> dict[str, Any]:
    import torch

    task = runtime["task"]
    env = make_fresh_env(runtime, seed)
    started = time.perf_counter()
    try:
        env.envs[0].init_state_id = int(state_id)
        if int(env.envs[0].init_state_id) != int(state_id):
            raise RuntimeError("initial-state assignment mismatch")
        random.seed(int(seed))
        np.random.seed(int(seed))
        reset_policy_rng(torch, policy_rng_seed)
        runtime["policy"].reset()
        observation, _ = env.reset(seed=[int(seed)])
        initial_image_means = {key: float(np.asarray(value).mean()) for key, value in observation["pixels"].items()}
        executor = make_executor(method)
        step_log: list[dict[str, Any]] = []
        query_steps: list[int] = []
        query_latencies: list[float] = []
        success = False
        completion_step: int | None = None
        last_info: Any = {"is_success": False}
        last_reward: Any = 0.0
        last_done = False
        max_steps = int(task["max_episode_steps"])
        for target_t in range(max_steps):
            started_query = time.perf_counter()
            result = executor.step(target_t, lambda: query_act_chunk(observation, env, runtime)[0])
            if result.queried:
                query_steps.append(target_t)
                query_latencies.append(time.perf_counter() - started_query)
            action = result.action.astype(np.float32, copy=False)
            observation, reward, terminated, truncated, info = env.step(action[None])
            terminated = bool(np.asarray(terminated).reshape(-1)[0])
            truncated = bool(np.asarray(truncated).reshape(-1)[0])
            done = terminated or truncated
            if done:
                success = extract_success(info, reward)
                completion_step = target_t + 1 if success else None
            last_info, last_reward, last_done = info, reward, done
            step_log.append({
                "physical_target_t": int(target_t),
                "policy_queried_at_t": bool(result.queried),
                "query_physical_step_q": result.query_q,
                "arm_source_query_q": int(result.arm_source_q),
                "arm_chunk_offset": int(result.arm_offset),
                "gripper_source_query_q": int(result.grip_source_q),
                "gripper_chunk_offset": int(result.grip_offset),
                "arm_source_age": int(result.arm_age),
                "gripper_source_age": int(result.grip_age),
                "action": result.action.astype(float).tolist(),
                "fresh_action": None if result.fresh_action is None else result.fresh_action.astype(float).tolist(),
                "old_action": None if result.old_action is None else result.old_action.astype(float).tolist(),
                "query_latency_seconds": (time.perf_counter() - started_query) if result.queried else None,
                "success_termination": bool(success) if done else None,
                "terminated": terminated,
                "truncated": truncated,
            })
            if done:
                break
        if not step_log:
            raise RuntimeError("episode executed no environment steps")
        elapsed = time.perf_counter() - started
        return {
            "runner_version": RUNNER_VERSION,
            "suite": task["suite"],
            "task_id": int(task["task_id"]),
            "task_name": task["task_name"],
            "method": method,
            "requested_initial_state_id": int(state_id),
            "environment_seed": int(seed),
            "environment_construction_seed": int(seed),
            "policy_rng_seed": int(policy_rng_seed),
            "fresh_environment_instance": True,
            "max_episode_steps": max_steps,
            "success": bool(success),
            "completion_step": completion_step,
            "environment_steps": len(step_log),
            "policy_queries": len(query_steps),
            "query_rate": len(query_steps) / len(step_log),
            "query_steps": query_steps,
            "mean_gripper_source_age": float(np.mean([row["gripper_source_age"] for row in step_log])),
            "observed_gripper_source_age_histogram": {str(age): int(sum(row["gripper_source_age"] == age for row in step_log)) for age in range(100)},
            "wall_clock_seconds": float(elapsed),
            "mean_policy_call_latency_seconds": float(np.mean(query_latencies)) if query_latencies else None,
            "initial_image_means": initial_image_means,
            "step_log": step_log,
            "terminal_info_success": bool(last_info.get("is_success", False)) if isinstance(last_info, dict) and last_done else bool(success),
            "terminal_reward": float(np.asarray(last_reward).reshape(-1)[0]) if last_done else None,
        }
    finally:
        env.close()


def task_skeleton(protocol_path: Path, task: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "runner_version": RUNNER_VERSION,
        "protocol": str(protocol_path.resolve()),
        "suite": task["suite"],
        "task_id": int(task["task_id"]),
        "task_name": task["task_name"],
        "checkpoint": task["checkpoint"],
        "methods": list(METHODS),
        "state_ids": [int(x) for x in range(14)],
        "episodes": {method: [] for method in METHODS},
        "finished": False,
    }


def run_task(protocol: dict[str, Any], protocol_path: Path, task: dict[str, Any], gpu: str, output_root: Path, progress_path: Path | None) -> None:
    name = f"{task['suite']}_task{int(task['task_id'])}"
    output_path = output_root / "results" / f"{name}.json"
    existing = json.loads(output_path.read_text(encoding="utf-8")) if output_path.is_file() else None
    if existing is not None:
        if existing.get("runner_version") != RUNNER_VERSION or existing.get("methods") != list(METHODS) or existing.get("protocol") != str(protocol_path.resolve()):
            raise RuntimeError(f"existing result identity/version mismatch: {output_path}")
    runtime = build_task_runtime(task, gpu)
    result = existing or task_skeleton(protocol_path, task)
    existing_keys = {(method, int(episode["requested_initial_state_id"])) for method, episodes in result["episodes"].items() for episode in episodes}
    seeds = [int(x) for x in task["environment_seeds"]]
    states = [int(x) for x in protocol["cohort"]["state_ids"]]
    if len(seeds) != len(states):
        raise RuntimeError("frozen seed/state count mismatch")
    progress = {"pid": os.getpid(), "runner_version": RUNNER_VERSION, "gpu": str(gpu), "task": name, "completed_episodes": len(existing_keys), "current_method": None, "current_state_id": None}
    if progress_path:
        atomic_json(progress_path, progress)
    for method in METHODS:
        for state_id, seed in zip(states, seeds, strict=True):
            key = (method, state_id)
            if key in existing_keys:
                continue
            progress.update({"current_method": method, "current_state_id": state_id})
            if progress_path:
                atomic_json(progress_path, progress)
            episode = run_episode(runtime, method, state_id, seed, int(protocol["runtime"]["policy_rng_seed"]))
            result["episodes"][method].append(episode)
            result["episodes"][method].sort(key=lambda row: int(row["requested_initial_state_id"]))
            existing_keys.add(key)
            progress["completed_episodes"] += 1
            atomic_json(output_path, result)
            if progress_path:
                atomic_json(progress_path, progress)
            print(f"task={name} method={method} state={state_id} success={episode['success']} steps={episode['environment_steps']}", flush=True)
    result["finished"] = True
    result["finished_at"] = time.time()
    atomic_json(output_path, result)
    progress["finished"] = True
    progress["finished_at"] = result["finished_at"]
    if progress_path:
        atomic_json(progress_path, progress)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--tasks", required=True, help="comma-separated suite:task entries")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--progress", type=Path)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    task_map = {f"{task['suite']}:task{int(task['task_id'])}": task for task in protocol["cohort"]["tasks"]}
    task_keys = [value.strip() for value in args.tasks.split(",") if value.strip()]
    if not task_keys or any(value not in task_map for value in task_keys):
        raise SystemExit("every --tasks entry must be in the frozen confirmation cohort")
    for key in task_keys:
        run_task(protocol, args.protocol, task_map[key], args.gpu, args.output_root, args.progress)


if __name__ == "__main__":
    main()
