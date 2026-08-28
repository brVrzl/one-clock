#!/usr/bin/env python3
"""Run the frozen ACT fixed source-age cap blind confirmatory panel."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TEMPORAL_ROOT = ROOT.parent
sys.path.insert(0, str(TEMPORAL_ROOT))
from temporal_operators import same_target_candidates  # noqa: E402


METHODS = ("fresh_h1", "fixed_h8", "fixed_h16", "native_h100")
HORIZONS = {"fresh_h1": 1, "fixed_h8": 8, "fixed_h16": 16, "native_h100": 100}
DEFAULT_PROTOCOL = ROOT / "protocol.json"


@dataclass(frozen=True)
class QueryRecord:
    query_step: int
    chunk: np.ndarray


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_progress(path: Path | None, value: object) -> None:
    if path is not None:
        atomic_json(path, value)


def _candidate_records(records: list[QueryRecord], target_step: int) -> list[tuple[int, np.ndarray, int]]:
    """Return available (age, action, query_step) rows oldest query to newest."""

    available: list[tuple[int, np.ndarray, int]] = []
    for record in records:
        offset = int(target_step) - int(record.query_step)
        if 0 <= offset < len(record.chunk):
            available.append((offset, np.asarray(record.chunk[offset], dtype=np.float64), record.query_step))
    return available


def scheduler_step(method: str, target_step: int, records: list[QueryRecord], query_fn):
    """Return (action, queried, reasons, diagnostics, executed_source_age)."""

    if method not in METHODS:
        raise ValueError(f"unknown dynamic-horizon method: {method}")
    horizon = HORIZONS[method]
    should_query = target_step % horizon == 0
    reasons = [f"fixed_period_h{horizon}"] if should_query else []
    diagnostics = {"candidate_count": len(_candidate_records(records, target_step)), "horizon": horizon}
    if should_query:
        chunk = np.asarray(query_fn(), dtype=np.float64)
        if chunk.ndim == 3 and chunk.shape[0] == 1:
            chunk = chunk[0]
        if chunk.ndim != 2 or chunk.shape[1] != 7:
            raise ValueError(f"query_fn must return (H,7), got {chunk.shape}")
        records.append(QueryRecord(int(target_step), chunk.copy()))
        diagnostics = {**diagnostics, "query_record_added": True}
    candidates = _candidate_records(records, target_step)
    if not candidates:
        raise RuntimeError(f"no executable same-target candidate at step {target_step}")
    age, action, source_query_step = candidates[-1]
    return action.copy(), bool(should_query), reasons, diagnostics, int(age), int(source_query_step)


def semantic_smoke() -> None:
    """CPU-only sparse indexing and frozen trigger checks."""

    chunks = [np.arange(120 * 7, dtype=np.float64).reshape(120, 7) + 100.0 * step for step in (0, 16, 32)]
    records = [QueryRecord(0, chunks[0]), QueryRecord(16, chunks[1]), QueryRecord(32, chunks[2])]
    candidates = _candidate_records(records, 32)
    assert [row[0] for row in candidates] == [32, 16, 0]
    assert [row[2] for row in candidates] == [0, 16, 32]
    for method, period in HORIZONS.items():
        records_for_method: list[QueryRecord] = []
        action, queried, reasons, _, age, source = scheduler_step(method, 0, records_for_method, lambda: chunks[0])
        assert queried and age == 0 and source == 0
        action, queried, _, _, age, source = scheduler_step(method, 1, records_for_method, lambda: chunks[1])
        if period == 1:
            assert queried and age == 0 and source == 1
        else:
            assert not queried and age == 1 and source == 0
            np.testing.assert_array_equal(action, chunks[0][1])
    print(json.dumps({"status": "fixed_horizon_blind_cpu_semantic_smoke_pass", "methods": list(METHODS)}))


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


def rollout_episode(*, env, policy, infer_chunk, processors, torch, method, seed, state_id, policy_rng_seed, max_steps):
    for index, actual in enumerate([state_id]):
        env.envs[index].init_state_id = int(actual)
    requested_state_id = int(state_id)
    actual_state_id = int(env.envs[0].init_state_id)
    if actual_state_id != requested_state_id:
        raise RuntimeError(f"initial state assignment mismatch: requested={requested_state_id}, actual={actual_state_id}")
    reset_policy_rng(torch, policy_rng_seed)
    policy.reset()
    observation, _ = env.reset(seed=[int(seed)])
    env_preprocessor, env_postprocessor, preprocessor, postprocessor = processors
    records: list[QueryRecord] = []
    executed_ages: list[int] = []
    executed_sources: list[int] = []
    query_steps: list[int] = []
    trigger_counts = {"fixed_period_h1": 0, "fixed_period_h8": 0, "fixed_period_h16": 0, "fixed_period_h100": 0}
    success = False
    completion_step = None
    done = False
    for step in range(int(max_steps)):
        def query():
            return infer_chunk(observation, env, policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor)

        action, queried, reasons, diagnostics, age, source_query_step = scheduler_step(method, step, records, query)
        if queried:
            query_steps.append(step)
        for reason in reasons:
            trigger_counts[reason] = trigger_counts.get(reason, 0) + 1
        executed_ages.append(age)
        executed_sources.append(source_query_step)
        observation, reward, terminated, truncated, info = env.step(action[None].astype(np.float32))
        terminated = bool(np.asarray(terminated).reshape(-1)[0])
        truncated = bool(np.asarray(truncated).reshape(-1)[0])
        done = terminated or truncated
        if done:
            success = extract_success(info, reward)
            completion_step = step + 1 if success else None
            break
    if not done:
        completion_step = None
    return {
        "seed": int(seed), "requested_initial_state_id": requested_state_id, "actual_initial_state_id": actual_state_id,
        "method": method, "success": bool(success), "completion_steps": completion_step,
        "environment_steps": step + 1, "policy_queries": len(query_steps), "query_count": len(query_steps),
        "query_rate": len(query_steps) / float(step + 1), "query_steps": query_steps,
        "trigger_counts": trigger_counts, "executed_source_age_steps": executed_ages,
        "executed_source_query_steps": executed_sources,
        "mean_executed_source_age_steps": float(np.mean(executed_ages)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--task", required=False, help="suite:task_id")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress", "--progress-file", dest="progress_file", type=Path)
    parser.add_argument("--live-smoke", action="store_true", help="run one state per method instead of full 20-episode task shard")
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
    checkpoint = (args.checkpoint or Path(task["checkpoint"])).resolve()
    if not (checkpoint / "config.json").is_file() or not (checkpoint / "model.safetensors").is_file():
        raise SystemExit(f"ACT checkpoint is missing required files: {checkpoint}")
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from run_component_reuse import infer_chunk

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.pretrained_path = checkpoint
    if getattr(config, "type", None) != "act":
        raise RuntimeError(f"expected ACT checkpoint, got {getattr(config, 'type', None)!r}")
    state_ids = [int(x) for x in protocol["environment"]["initial_state_ids"]]
    seeds = [int(x) for x in protocol["environment"]["seeds"]]
    env_config = LiberoEnv(task=task["suite"], task_ids=[int(task["task_id"])], fps=int(protocol["environment"]["fps"]), obs_type=protocol["environment"]["obs_type"], camera_name=protocol["environment"]["camera_name"], init_states=True, observation_width=int(protocol["environment"]["observation_width"]), observation_height=int(protocol["environment"]["observation_height"]), control_mode=protocol["environment"]["control_mode"])
    envs = make_env(env_config, n_envs=1, use_async_envs=False)
    env = envs[task["suite"]][int(task["task_id"])]
    policy = make_policy(cfg=config, env_cfg=env_config)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=config, pretrained_path=str(checkpoint), preprocessor_overrides={"device_processor": {"device": str(config.device)}})
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_config, policy_cfg=config)
    processors = (env_preprocessor, env_postprocessor, preprocessor, postprocessor)
    max_steps = int(np.asarray(env.call("_max_episode_steps")).reshape(-1)[0])
    selected_states = state_ids[:1] if args.live_smoke else state_ids
    selected_seeds = seeds[:1] if args.live_smoke else seeds
    selected_methods = METHODS
    started = time.time()
    output = {"protocol": str(args.protocol.resolve()), "checkpoint": str(checkpoint), "task": args.task, "task_name": task["task_name"], "methods": list(selected_methods), "live_smoke": bool(args.live_smoke), "n_envs": 1, "one_policy_query_only_on_scheduler_query": True, "started_at": started, "methods_result": {}}
    progress = {"pid": os.getpid(), "started_at": started, "completed_methods": 0, "completed_episodes": 0}
    write_progress(args.progress_file, progress)
    for method in selected_methods:
        episodes = []
        for state_id, seed in zip(selected_states, selected_seeds):
            episodes.append(rollout_episode(env=env, policy=policy, infer_chunk=infer_chunk, processors=processors, torch=torch, method=method, seed=seed, state_id=state_id, policy_rng_seed=int(protocol["policy"]["policy_rng_seed"]), max_steps=max_steps))
            progress.update({"completed_episodes": progress["completed_episodes"] + 1, "current_method": method, "current_state_id": state_id})
            write_progress(args.progress_file, progress)
        successes = [episode["success"] for episode in episodes]
        output["methods_result"][method] = {
            "successes": successes,
            "success_count": int(sum(successes)),
            "episodes": len(episodes),
            "query_count": sum(episode["query_count"] for episode in episodes),
            "environment_steps": sum(episode["environment_steps"] for episode in episodes),
            "query_rate": sum(episode["query_count"] for episode in episodes)
            / sum(episode["environment_steps"] for episode in episodes),
            "trigger_counts": {
                key: sum(episode["trigger_counts"].get(key, 0) for episode in episodes)
                for key in {reason for episode in episodes for reason in episode["trigger_counts"]}
            },
            "mean_executed_source_age_steps": float(
                np.mean([episode["mean_executed_source_age_steps"] for episode in episodes])
            ),
            "episodes_detail": episodes,
        }
        progress["completed_methods"] += 1
        write_progress(args.progress_file, progress)
        atomic_json(args.output, output)
    output["finished_at"] = time.time()
    progress["finished_at"] = output["finished_at"]
    write_progress(args.progress_file, progress)
    atomic_json(args.output, output)
    env.close()
    print(json.dumps({"output": str(args.output), "task": args.task, "episodes": len(selected_methods) * len(selected_states)}, indent=2))


if __name__ == "__main__":
    main()
