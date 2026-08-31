#!/usr/bin/env python3
"""ACT hard-sparse versus sparse temporal-ensemble development runner.

Each invocation owns one task and evaluates all four methods on the same ten
state/seed pairs.  ACT is queried only at the requested sparse cadence.  The
policy and environment processing path is the validated LeRobot 0.4.4 path;
the shared CPU executor receives the resulting postprocessed 100-step chunks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


ACT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ACT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from sparse_executor import SparseExecutor  # noqa: E402


METHODS = ("hard_h8", "sparse_te_h8", "hard_h16", "sparse_te_h16")
CADENCE_BY_METHOD = {
    "hard_h8": 8,
    "sparse_te_h8": 8,
    "hard_h16": 16,
    "sparse_te_h16": 16,
}
MODE_BY_METHOD = {
    "hard_h8": "hard",
    "sparse_te_h8": "sparse_te",
    "hard_h16": "hard",
    "sparse_te_h16": "sparse_te",
}
DEFAULT_PROTOCOL = EXPERIMENT_ROOT / "protocol.json"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_progress(path: Path | None, value: object) -> None:
    if path is not None:
        atomic_json(path, value)


def reset_policy_rng(torch, seed: int) -> None:
    """Retain the validated deterministic ACT reset convention."""

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


def infer_chunk(
    observation,
    env,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    torch,
) -> np.ndarray:
    """Run the established ACT observation/action postprocessing path."""

    from lerobot.envs.utils import add_envs_task, preprocess_observation
    from lerobot.utils.constants import ACTION

    batch = preprocess_observation(observation)
    batch = add_envs_task(env, batch)
    batch = env_preprocessor(batch)
    batch = preprocessor(batch)
    with torch.inference_mode():
        chunk = postprocessor(policy.predict_action_chunk(batch))
        chunk = env_postprocessor({ACTION: chunk})[ACTION]
    result = chunk.detach().cpu().numpy().astype(np.float32, copy=False)
    if result.ndim != 3 or result.shape[0] != 1 or result.shape[1] < 100 or result.shape[2] != 7:
        raise RuntimeError(f"unexpected postprocessed ACT chunk shape: {result.shape}; expected (1,100,7)")
    return result[0]


def _task_map(protocol: dict) -> dict[str, dict]:
    return {
        f"{task['suite']}:task{int(task['task_id'])}": task
        for task in protocol["tasks"]
    }


def _protocol_values(protocol: dict) -> tuple[list[int], list[int], int, int, float]:
    environment = protocol["environment_pairing"]
    policy = protocol["policies"]["act"]
    state_ids = [int(value) for value in environment["initial_state_ids"]]
    seeds = [int(value) for value in environment["seeds"]]
    if state_ids != list(range(10, 20)) or seeds != list(range(2000, 2010)):
        raise RuntimeError("ACT protocol drifted from frozen development states 10..19 / seeds 2000..2009")
    if len(state_ids) != len(seeds) or len(state_ids) != int(environment["episodes_per_task_method"]):
        raise RuntimeError("state/seed count does not match frozen episodes_per_task_method")
    horizon = int(policy["prediction_horizon"])
    coefficient = float(protocol["temporal_ensemble"]["coefficient"])
    if horizon != 100:
        raise RuntimeError(f"ACT prediction horizon must be 100, got {horizon}")
    if coefficient != 0.01:
        raise RuntimeError(f"canonical ACT coefficient must remain 0.01, got {coefficient}")
    return state_ids, seeds, int(environment["fps"]), horizon, coefficient


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
    prediction_horizon: int,
    coefficient: float,
    max_steps: int,
) -> dict:
    """Run one state/method and retain complete step-level provenance."""

    env.envs[0].init_state_id = int(state_id)
    actual_state_id = int(env.envs[0].init_state_id)
    if actual_state_id != int(state_id):
        raise RuntimeError(
            f"initial-state assignment mismatch: requested={state_id}, actual={actual_state_id}"
        )
    reset_policy_rng(torch, policy_rng_seed)
    policy.reset()
    observation, _ = env.reset(seed=[int(env_seed)])
    env_preprocessor, env_postprocessor, preprocessor, postprocessor = processors

    cadence = CADENCE_BY_METHOD[method]
    executor = SparseExecutor(
        cadence=cadence,
        prediction_horizon=prediction_horizon,
        mode=MODE_BY_METHOD[method],
        coefficient=coefficient,
        action_dim=7,
    )
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
        "actual_initial_state_id": actual_state_id,
        "policy_rng_seed": int(policy_rng_seed),
        "cadence_h": cadence,
        "prediction_horizon": int(prediction_horizon),
        "temporal_ensemble_coefficient": float(coefficient),
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


def summarize_method(episodes: list[dict]) -> dict:
    successes = [bool(episode["success"]) for episode in episodes]
    total_queries = sum(int(episode["query_count"]) for episode in episodes)
    total_steps = sum(int(episode["environment_steps"]) for episode in episodes)
    latencies = [
        latency
        for episode in episodes
        for latency in episode["query_latency_seconds"]
    ]
    return {
        "successes": successes,
        "success_count": int(sum(successes)),
        "episodes": len(episodes),
        "policy_queries": total_queries,
        "query_count": total_queries,
        "environment_steps": total_steps,
        "query_rate": total_queries / float(total_steps),
        "mean_query_latency_seconds": float(np.mean(latencies)) if latencies else None,
        "mean_ensemble_candidate_count": float(
            np.mean([episode["mean_ensemble_candidate_count"] for episode in episodes])
        ),
        "mean_weighted_source_age_steps": float(
            np.mean([episode["mean_weighted_source_age_steps"] for episode in episodes])
        ),
        "completion_steps_successful": [
            int(episode["completion_steps"])
            for episode in episodes
            if episode["completion_steps"] is not None
        ],
        "mean_completion_steps_successful": float(
            np.mean(
                [episode["completion_steps"] for episode in episodes if episode["completion_steps"] is not None]
            )
        )
        if any(episode["completion_steps"] is not None for episode in episodes)
        else None,
        "episodes_detail": episodes,
    }


def semantic_smoke() -> None:
    """Small CPU check that the ACT runner uses the shared sparse semantics."""

    def fake_chunk(source: int, horizon: int = 100) -> np.ndarray:
        return np.asarray(
            [[1000.0 * source + 10.0 * offset + d for d in range(7)] for offset in range(horizon)]
        )

    for method, cadence in CADENCE_BY_METHOD.items():
        executor = SparseExecutor(
            cadence=cadence,
            prediction_horizon=100,
            mode=MODE_BY_METHOD[method],
            coefficient=0.01,
        )
        for target in range(cadence + 1):
            result = executor.step(target, lambda target=target: fake_chunk(target))
        assert executor.query_steps == [0, cadence]
        assert result.candidates.offsets.tolist() == [cadence, 0]
        assert result.candidate_count == 2
        assert result.action.shape == (7,)
    print(json.dumps({"status": "act_sparse_te_runner_semantic_smoke_pass", "methods": list(METHODS)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--task", help="suite:task_id")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress", "--progress-file", dest="progress_file", type=Path)
    parser.add_argument("--live-smoke", action="store_true", help="one state per method")
    parser.add_argument("--semantic-smoke", action="store_true")
    args = parser.parse_args()
    if args.semantic_smoke:
        semantic_smoke()
        return
    if args.task is None or args.output is None:
        raise SystemExit("--task and --output are required unless --semantic-smoke is used")

    protocol = json.loads(args.protocol.read_text())
    tasks = _task_map(protocol)
    if args.task not in tasks:
        raise SystemExit(f"task is absent from frozen protocol: {args.task}")
    state_ids, seeds, fps, prediction_horizon, coefficient = _protocol_values(protocol)
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
    if getattr(policy_cfg, "type", None) != "act":
        raise RuntimeError(f"expected ACT checkpoint, got {getattr(policy_cfg, 'type', None)!r}")
    if int(policy_cfg.chunk_size) != prediction_horizon:
        raise RuntimeError(
            f"checkpoint ACT chunk_size {policy_cfg.chunk_size} differs from protocol H_pred={prediction_horizon}"
        )

    env_config = LiberoEnv(
        task=task["suite"],
        task_ids=[int(task["task_id"])],
        fps=fps,
        obs_type=protocol["environment_pairing"]["obs_type"],
        camera_name=protocol["environment_pairing"]["camera_name"],
        init_states=bool(protocol["environment_pairing"]["init_states"]),
        observation_width=int(protocol["policies"]["act"]["observation_width"]),
        observation_height=int(protocol["policies"]["act"]["observation_height"]),
        control_mode=protocol["environment_pairing"]["control_mode"],
    )
    env = make_env(
        env_config,
        n_envs=int(protocol["environment_pairing"]["n_envs"]),
        use_async_envs=bool(protocol["environment_pairing"]["use_async_envs"]),
    )[task["suite"]][int(task["task_id"])]
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
    max_steps = int(np.asarray(env.call("_max_episode_steps")).reshape(-1)[0])
    selected_states = state_ids[:1] if args.live_smoke else state_ids
    selected_seeds = seeds[:1] if args.live_smoke else seeds
    started = time.time()
    output = {
        "protocol": str(args.protocol.resolve()),
        "implementation": "LeRobot 0.4.4 ACTPolicy.predict_action_chunk + native policy/environment postprocessors + shared SparseExecutor",
        "runtime": {
            "python_executable": "/home/wjq/workspace/venvs/libero_act/bin/python",
            "lerobot": "0.4.4",
            "torch": "2.7.1+cu128",
            "mujoco": "3.3.1",
            "cuda_visible_devices": str(args.gpu),
        },
        "task": args.task,
        "task_name": task["task_name"],
        "checkpoint": str(checkpoint),
        "prediction_horizon": prediction_horizon,
        "temporal_ensemble_coefficient": coefficient,
        "methods": list(METHODS),
        "live_smoke": bool(args.live_smoke),
        "n_envs": 1,
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
    write_progress(args.progress_file, progress)

    for method in METHODS:
        episodes = []
        for state_id, env_seed in zip(selected_states, selected_seeds):
            progress.update({"current_method": method, "current_state_id": int(state_id)})
            write_progress(args.progress_file, progress)
            episode = rollout_episode(
                env=env,
                policy=policy,
                processors=processors,
                torch=torch,
                task_key=args.task,
                method=method,
                state_id=int(state_id),
                env_seed=int(env_seed),
                policy_rng_seed=int(protocol["policies"]["act"]["policy_rng_seed"]),
                prediction_horizon=prediction_horizon,
                coefficient=coefficient,
                max_steps=max_steps,
            )
            episodes.append(episode)
            progress["completed_episodes"] += 1
            write_progress(args.progress_file, progress)
        output["methods_result"][method] = summarize_method(episodes)
        progress["completed_methods"] += 1
        write_progress(args.progress_file, progress)
        atomic_json(args.output, output)

    finished = time.time()
    output["finished_at"] = finished
    progress["finished_at"] = finished
    write_progress(args.progress_file, progress)
    atomic_json(args.output, output)
    env.close()
    print(json.dumps({"output": str(args.output), "task": args.task, "episodes": len(METHODS) * len(selected_states)}, indent=2))


if __name__ == "__main__":
    main()

