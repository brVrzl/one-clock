"""Run the repaired five-condition ACT factorial in resumable task shards."""

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
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXPERIMENT_ROOT))

from temporal_reuse import (  # noqa: E402
    ACTION_DIM,
    CHUNK_LENGTH,
    METHODS,
    make_executor,
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_progress(path: Path | None, value: object) -> None:
    if path is not None:
        atomic_json(path, value)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_outcome_rollout":
        raise RuntimeError("protocol is not frozen before outcomes")
    cohort = protocol["cohort"]
    if cohort["primary_task_ids"] != list(range(1, 10)):
        raise RuntimeError("primary cohort drifted from Object tasks 1-9")
    if len(cohort["state_ids"]) != 14 or cohort["primary_paired_blocks"] != 126:
        raise RuntimeError("primary cohort block count is not 126")
    if protocol["rollout"]["primary_episodes"] != 630:
        raise RuntimeError("primary episode count is not 630")
    if [condition["name"] for condition in protocol["conditions"]] != list(METHODS):
        raise RuntimeError("protocol conditions differ from the five implemented methods")
    return protocol


def reset_policy_rng(torch: Any, seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_task_runtime(protocol: dict[str, Any], task_id: int, gpu: str) -> dict[str, Any]:
    runtime = protocol["runtime"]
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    import torch
    from libero.libero import benchmark
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    checkpoint = Path(runtime["checkpoint"]).resolve()
    if not (checkpoint / "config.json").is_file() or not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"ACT checkpoint is missing required files: {checkpoint}")
    task_suite = str("libero_object")
    task_id = int(task_id)
    task = benchmark.get_benchmark_dict()[task_suite]().get_task(task_id)
    env_config = LiberoEnv(
        task=task_suite,
        task_ids=[task_id],
        fps=int(runtime["control_frequency_hz"]),
        obs_type=str(runtime["obs_type"]),
        camera_name=str(runtime["camera_name"]),
        camera_name_mapping=dict(runtime["camera_name_mapping"]),
        init_states=bool(runtime["init_states"]),
        observation_width=int(runtime["observation_width"]),
        observation_height=int(runtime["observation_height"]),
        control_mode=str(runtime["control_mode"]),
    )
    # The frozen ACT checkpoint uses the historical wrist feature name.  The
    # installed LiberoEnv default calls the same camera image2; only the
    # policy-facing feature map is corrected here, not the camera or pixels.
    env_config.features_map["pixels/robot0_eye_in_hand_image"] = "observation.images.wrist_image"
    policy_config = PreTrainedConfig.from_pretrained(checkpoint)
    policy_config.device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_config.pretrained_path = checkpoint
    policy_config.pretrained_backbone_weights = None
    if getattr(policy_config, "type", None) != "act":
        raise RuntimeError(f"expected ACT checkpoint, got {getattr(policy_config, 'type', None)!r}")
    if int(policy_config.chunk_size) != CHUNK_LENGTH:
        raise RuntimeError("ACT checkpoint chunk size differs from frozen H_pred=100")
    if policy_config.temporal_ensemble_coeff is not None:
        raise RuntimeError("policy-internal temporal ensembling must be disabled")
    if int(policy_config.action_feature.shape[0]) != ACTION_DIM:
        raise RuntimeError("ACT action dimension differs from frozen 7-D contract")
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
        "env_config": env_config,
        "policy": policy,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "env_preprocessor": env_preprocessor,
        "env_postprocessor": env_postprocessor,
        "task_id": task_id,
        "task_name": task.name,
        "suite": task_suite,
        "policy_rng_seed": int(runtime["policy_rng_seed"]),
        "max_steps": int(runtime["max_episode_steps"]),
    }


def construct_env(runtime: dict[str, Any], environment_seed: int):
    from lerobot.envs.factory import make_env

    random.seed(int(environment_seed))
    np.random.seed(int(environment_seed))
    return make_env(
        runtime["env_config"],
        n_envs=1,
        use_async_envs=False,
    )[runtime["suite"]][int(runtime["task_id"])]


def prepare_processed_input(observation: dict[str, Any], env: Any, runtime: dict[str, Any]) -> Any:
    from lerobot.envs.utils import add_envs_task, preprocess_observation

    batch = preprocess_observation(observation)
    batch = add_envs_task(env, batch)
    batch = runtime["env_preprocessor"](batch)
    # LeRobot 0.4.4's Libero environment preprocessor retains image2 in the
    # emitted batch even when the policy-facing feature map names that camera
    # wrist_image.  Rename the existing tensor without changing its pixels.
    if "observation.images.image2" in batch and "observation.images.wrist_image" not in batch:
        batch["observation.images.wrist_image"] = batch.pop("observation.images.image2")
    return runtime["preprocessor"](batch)


def freeze_value(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: freeze_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [freeze_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(freeze_value(item) for item in value)
    return value


def values_equal(left: Any, right: Any) -> bool:
    import torch

    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and list(sorted(left)) == list(sorted(right))
            and all(values_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, type(right))
            and len(left) == len(right)
            and all(values_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def query_act_chunk(observation: dict[str, Any], env: Any, runtime: dict[str, Any]) -> tuple[np.ndarray, Any]:
    import torch
    from lerobot.utils.constants import ACTION

    processed = prepare_processed_input(observation, env, runtime)
    with torch.inference_mode():
        chunk = runtime["postprocessor"](runtime["policy"].predict_action_chunk(processed))
        chunk = runtime["env_postprocessor"]({ACTION: chunk})[ACTION]
    result = chunk.detach().cpu().numpy()
    if result.shape != (1, CHUNK_LENGTH, ACTION_DIM):
        raise RuntimeError(f"unexpected postprocessed ACT chunk shape: {result.shape}")
    return result[0].astype(np.float64, copy=False), processed


def sim_state_snapshot(env: Any) -> tuple[np.ndarray, np.ndarray]:
    inner = env.envs[0]._env
    return (
        np.asarray(inner.get_sim_state()).copy(),
        np.asarray(inner.sim.model.body_pos).copy(),
    )


def extract_success(info: Any, reward: Any) -> bool:
    final_info = info.get("final_info") if isinstance(info, dict) else None
    if isinstance(final_info, dict) and "is_success" in final_info:
        return bool(np.asarray(final_info["is_success"]).reshape(-1)[0])
    values = np.asarray(reward).reshape(-1)
    return bool(len(values) and values[0] > 0)


def run_episode(
    runtime: dict[str, Any], method: str, state_id: int, environment_seed: int
) -> dict[str, Any]:
    import torch

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
            "environment_steps": len(step_log),
            "policy_queries": len(query_steps),
            "query_rate": len(query_steps) / len(step_log),
            "query_steps": query_steps,
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
    return {
        "schema_version": 1,
        "protocol": str(protocol_path.resolve()),
        "task_id": int(runtime["task_id"]),
        "task_name": runtime["task_name"],
        "methods": list(METHODS),
        "state_ids": [int(x) for x in load_protocol(protocol_path)["cohort"]["state_ids"]],
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
    write_progress(progress_path, progress)
    for method in METHODS:
        for state_id, environment_seed in zip(state_ids, seeds_by_task, strict=True):
            key = (method, int(state_id))
            if key in existing_keys:
                continue
            progress.update({"current_method": method, "current_state_id": int(state_id)})
            write_progress(progress_path, progress)
            episode = run_episode(runtime, method, int(state_id), int(environment_seed))
            result["episodes"][method].append(episode)
            result["episodes"][method].sort(key=lambda row: int(row["requested_initial_state_id"]))
            existing_keys.add(key)
            progress["completed_episodes"] += 1
            atomic_json(output_path, result)
            write_progress(progress_path, progress)
            print(
                f"task={task_id} method={method} state={state_id} success={episode['success']} "
                f"steps={episode['environment_steps']}", flush=True
            )
    result["finished"] = True
    result["finished_at"] = time.time()
    atomic_json(output_path, result)
    progress["finished"] = True
    progress["finished_at"] = result["finished_at"]
    write_progress(progress_path, progress)


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
