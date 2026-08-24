#!/usr/bin/env python3
"""Paired LIBERO evaluation for wall-clock ACT versus StateTrack."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np

ONE_CLOCK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ONE_CLOCK_ROOT / "src"))
sys.path.insert(0, str(ONE_CLOCK_ROOT / "scripts"))

from one_clock import IdentityPostPolicy, StateTrackChunk  # noqa: E402
from run_libero_gate0 import (  # noqa: E402
    batch_robot_state,
    feature_summary,
    git_commit,
    load_config,
    load_policy_and_processors,
    prepare_policy_observation,
    query_full_act_chunk,
    set_episode_seed,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ONE_CLOCK_ROOT / "configs/gate0_libero_object.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--mode", choices=("wall_clock", "state_track"), required=True)
    parser.add_argument("--lookahead", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--init-state-start", type=int, default=0)
    parser.add_argument("--hold-period", type=int, default=0,
                        help="Every Nth control tick repeats the previous action (0 disables stress test).")
    return parser.parse_args()


def run_episode(
    *, env: Any, policy: Any, processors: tuple[Any, Any, Any, Any], task_id: int,
    mode: str, lookahead: int, horizon: int, episode: int, init_state_id: int,
    seed: int, hold_period: int,
) -> dict[str, Any]:
    _, policy_preprocessor, policy_postprocessor, env_preprocessor, env_postprocessor = processors
    set_episode_seed(seed)
    env.init_state_id = init_state_id
    observation, _ = env.reset(seed=seed)
    policy.reset()
    identity = IdentityPostPolicy()
    tracker = StateTrackChunk(lookahead=lookahead, active_horizon=horizon) if mode == "state_track" else None
    chunk: np.ndarray | None = None
    chunk_age = horizon
    records: list[dict[str, Any]] = []
    previous_action: np.ndarray | None = None
    query_latencies: list[float] = []
    for environment_step in range(env._max_episode_steps):
        queried = chunk is None or chunk_age >= horizon
        if queried:
            result, latency = query_full_act_chunk(
                observation=observation,
                policy=policy,
                policy_preprocessor=policy_preprocessor,
                policy_postprocessor=policy_postprocessor,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                post_policy=identity,
                task_id=task_id,
            )
            chunk = np.clip(np.asarray(result.action_chunk, dtype=np.float64), -1.0, 1.0)
            chunk_age = 0
            query_latencies.append(latency)
            if tracker is not None:
                tracker.start_chunk(observation, chunk)
        assert chunk is not None
        if tracker is None:
            selected_index = min(chunk_age, chunk.shape[0] - 1)
            action = chunk[selected_index].copy()
            progress_record = {
                "progress_index": selected_index,
                "selected_index": selected_index,
                "nearest_index": selected_index,
                "tracking_error": 0.0,
                "repeated": False,
                "skipped": 0,
            }
        else:
            action, diagnostics = tracker.select(observation)
            progress_record = diagnostics.as_log_record()
        if hold_period and (environment_step + 1) % hold_period == 0 and previous_action is not None:
            executed_action = previous_action.copy()
            held = True
        else:
            executed_action = action.copy()
            held = False
        observation, _, terminated, truncated, info = env.step(executed_action.astype(np.float32))
        record = {
            "environment_step": environment_step,
            "policy_query": queried,
            "chunk_age": chunk_age,
            "action": action.tolist(),
            "executed_action": executed_action.tolist(),
            "execution_hold": held,
            "is_success": bool(info["is_success"]),
            **progress_record,
        }
        records.append(record)
        previous_action = executed_action
        chunk_age += 1
        if terminated or truncated:
            break
    success = bool(info["is_success"])
    progress_values = [float(record["tracking_error"]) for record in records]
    return {
        "episode": episode,
        "init_state_id": init_state_id,
        "seed": seed,
        "mode": mode,
        "lookahead": lookahead if mode == "state_track" else 0,
        "hold_period": hold_period,
        "success": success,
        "environment_steps": len(records),
        "policy_queries": sum(int(record["policy_query"]) for record in records),
        "repeated_action_count": sum(int(record["repeated"]) for record in records),
        "skipped_action_count": sum(int(record["skipped"]) for record in records),
        "execution_hold_count": sum(int(record["execution_hold"]) for record in records),
        "mean_tracking_error": float(np.mean(progress_values)) if progress_values else 0.0,
        "max_tracking_error": float(np.max(progress_values)) if progress_values else 0.0,
        "mean_progress_index": float(np.mean([record["progress_index"] for record in records])) if records else 0.0,
        "max_progress_index": int(max(record["progress_index"] for record in records)) if records else -1,
        "records": records,
        "mean_policy_query_latency_seconds": float(np.mean(query_latencies)) if query_latencies else 0.0,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.episodes < 1 or args.init_state_start < 0:
        raise ValueError("episodes must be positive and init-state-start must be non-negative")
    if args.hold_period < 0:
        raise ValueError("hold-period cannot be negative")
    checkpoint = args.checkpoint.resolve()
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv

    task_suite_name = str(config["task_suite"])
    suite = benchmark.get_benchmark_dict()[task_suite_name]()
    task_id = int(args.task_id)
    task = suite.get_task(task_id)
    runtime_config = dict(config)
    runtime_config.update({"task_id": task_id, "task_name": task.name})
    policy, policy_preprocessor, policy_postprocessor, env_preprocessor, env_postprocessor = (
        load_policy_and_processors(runtime_config, checkpoint)
    )
    processors = (policy, policy_preprocessor, policy_postprocessor, env_preprocessor, env_postprocessor)
    horizon = 8
    if int(policy.config.chunk_size) < horizon:
        raise ValueError("ACT chunk is shorter than the configured 8-step query horizon")
    env = LiberoEnv(
        task_suite=suite, task_id=task_id, task_suite_name=task_suite_name,
        obs_type=str(config["obs_type"]), camera_name=str(config["camera_name"]),
        camera_name_mapping=dict(config["camera_name_mapping"]),
        observation_width=int(config["observation_width"]), observation_height=int(config["observation_height"]),
        control_freq=int(config.get("control_freq", 20)), init_states=bool(config["init_states"]),
        hard_reset=bool(config["hard_reset"]), control_mode=str(config["control_mode"]),
    )
    init_state_ids = list(range(args.init_state_start, args.init_state_start + args.episodes))
    if init_state_ids[-1] >= len(env._init_states):
        raise ValueError(f"requested initial state {init_state_ids[-1]} but task has {len(env._init_states)}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    episode_records = [
        run_episode(
            env=env, policy=policy, processors=processors, task_id=task_id,
            mode=args.mode, lookahead=args.lookahead, horizon=horizon,
            episode=episode, init_state_id=init_state_id, seed=int(config["seed"]) + init_state_id,
            hold_period=args.hold_period,
        )
        for episode, init_state_id in enumerate(init_state_ids)
    ]
    (output_dir / "episodes.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in episode_records) + "\n",
        encoding="utf-8",
    )
    successes = sum(int(record["success"]) for record in episode_records)
    summary = {
        "mode": args.mode, "lookahead": args.lookahead if args.mode == "state_track" else 0,
        "hold_period": args.hold_period, "task_id": task_id, "task_name": task.name,
        "task_description": task.language, "episodes": len(episode_records), "successes": successes,
        "success_rate": successes / len(episode_records),
        "mean_environment_steps": float(np.mean([row["environment_steps"] for row in episode_records])),
        "mean_repeated_action_count": float(np.mean([row["repeated_action_count"] for row in episode_records])),
        "mean_skipped_action_count": float(np.mean([row["skipped_action_count"] for row in episode_records])),
        "mean_tracking_error": float(np.mean([row["mean_tracking_error"] for row in episode_records])),
        "max_tracking_error": float(max(row["max_tracking_error"] for row in episode_records)),
        "mean_progress_index": float(np.mean([row["mean_progress_index"] for row in episode_records])),
        "mean_policy_query_latency_seconds": float(np.mean([row["mean_policy_query_latency_seconds"] for row in episode_records])),
        "action_semantics": "LIBERO OSC_POSE relative EE delta: first 6 normalized [-1,1] mapped to +/-0.05m,+/-0.5rad; dim6 gripper",
        "control_mode": config["control_mode"], "chunk_size": int(policy.config.chunk_size), "active_horizon": horizon,
        "checkpoint": str(checkpoint), "checkpoint_model_sha256": sha256_file(checkpoint / "model.safetensors"),
        "project_commit_at_launch": git_commit(ONE_CLOCK_ROOT),
        "lerobot_commit": git_commit(Path(__import__("lerobot").__file__).resolve().parents[2]),
        "python": platform.python_version(), "episodes_init_state_ids": init_state_ids,
        "gpu": __import__("torch").cuda.get_device_name(0) if __import__("torch").cuda.is_available() else None,
        "lerobot_version": importlib.metadata.version("lerobot"), "libero_version": importlib.metadata.version("hf-libero"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

