#!/usr/bin/env python3
"""Run the preregistered, resumable Gate-3A2 LIBERO rollouts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path("/home/thor/projects/embodied_lab/third_party/lerobot/src")))

from gate3a2_temporal_aggregation import DenseTemporalAggregator, METHODS  # noqa: E402
from scripts.run_libero_gate0 import (  # noqa: E402
    load_config,
    load_policy_and_processors,
    query_full_act_chunk,
    set_episode_seed,
)


CHECKPOINT = Path("/home/thor/projects/checkpoints/zeromidnight_act_libero_object")
CONFIG = ROOT / "configs/gate0_libero_object.yaml"
SCHEDULE = ROOT / "research/audit_outputs/gate3a2_run_schedule.json"
OUTPUT_ROOT = ROOT / "experiments/gate3a2_temporal_aggregation"
MANIFEST = ROOT / "research/audit_outputs/gate3a2_rollout_manifest.json"
LEROBOT_ROOT = Path("/home/thor/projects/embodied_lab/third_party/lerobot")
EXPECTED_MODEL_SHA256 = "340071d7497238669459d93517eb3f8690862ad6fdf14207966759dfe6da9410"
EXPECTED_CONFIG_SHA256 = "a76eebed357b3cbed8745c3d0f18c1335ecdd5449fcc498257676c9cbd27453d"
EXPECTED_LEROBOT_COMMIT = "f66e5128ecb2456e8c54a63d15404fa59c16aebc"
CHUNK_LENGTH = 100
ACTION_DIM = 7
CONTROL_HZ = 20.0
MAX_STEPS = 280


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, default=SCHEDULE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--max-new-runs", type=int)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--validate-runtime",
        action="store_true",
        help="Load the frozen task-0 policy/environment contract without resetting or rolling out.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def provenance(checkpoint: Path, config_path: Path, schedule_path: Path) -> dict[str, Any]:
    model_path = checkpoint / "model.safetensors"
    policy_config_path = checkpoint / "config.json"
    observed_model_hash = sha256(model_path)
    observed_config_hash = sha256(policy_config_path)
    observed_lerobot_commit = git_commit(LEROBOT_ROOT)
    if observed_model_hash != EXPECTED_MODEL_SHA256:
        raise RuntimeError(f"checkpoint hash mismatch: {observed_model_hash}")
    if observed_config_hash != EXPECTED_CONFIG_SHA256:
        raise RuntimeError(f"checkpoint config hash mismatch: {observed_config_hash}")
    if observed_lerobot_commit != EXPECTED_LEROBOT_COMMIT:
        raise RuntimeError(f"pinned LeRobot commit mismatch: {observed_lerobot_commit}")
    if subprocess.run(
        ["git", "-C", str(LEROBOT_ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout:
        raise RuntimeError("pinned LeRobot checkout is dirty")
    return {
        "checkpoint_directory": str(checkpoint.resolve()),
        "model_sha256": observed_model_hash,
        "config_sha256": observed_config_hash,
        "policy_preprocessor_sha256": sha256(checkpoint / "policy_preprocessor.json"),
        "policy_postprocessor_sha256": sha256(checkpoint / "policy_postprocessor.json"),
        "normalizer_sha256": sha256(checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"),
        "project_git_commit": git_commit(ROOT),
        "lerobot_git_commit": observed_lerobot_commit,
        "schedule_path": str(schedule_path.resolve()),
        "schedule_sha256": sha256(schedule_path),
        "rollout_config_path": str(config_path.resolve()),
        "rollout_config_sha256": sha256(config_path),
        "control_frequency_hz": CONTROL_HZ,
        "controller_tick_seconds": 1.0 / CONTROL_HZ,
        "chunk_length": CHUNK_LENGTH,
        "action_dim": ACTION_DIM,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in ("lerobot", "torch", "numpy", "scipy", "robosuite", "gymnasium")
        },
    }


def make_env_and_policy(
    config: dict[str, Any], checkpoint: Path, task_id: int
) -> tuple[Any, Any, Any, Any, Any, Any, str, int]:
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv, get_task_init_states

    suite_name = str(config["task_suite"])
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    available_states = len(get_task_init_states(suite, task_id))
    runtime_config = dict(config)
    runtime_config["task_id"] = task_id
    runtime_config["task_name"] = task.name
    components = load_policy_and_processors(runtime_config, checkpoint)
    policy, policy_preprocessor, policy_postprocessor, env_preprocessor, env_postprocessor = components
    policy.eval()
    if policy.config.temporal_ensemble_coeff is not None:
        raise RuntimeError("Gate-3A2 requires policy-internal temporal ensembling to be disabled")
    if int(policy.config.chunk_size) != CHUNK_LENGTH or int(policy.config.action_feature.shape[0]) != ACTION_DIM:
        raise RuntimeError("checkpoint ACT chunk/action shape does not match the preregistration")
    env = LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=suite_name,
        obs_type=str(config["obs_type"]),
        camera_name=str(config["camera_name"]),
        camera_name_mapping=dict(config["camera_name_mapping"]),
        observation_width=int(config["observation_width"]),
        observation_height=int(config["observation_height"]),
        control_freq=int(config["control_freq"]),
        init_states=bool(config["init_states"]),
        hard_reset=bool(config["hard_reset"]),
        control_mode=str(config["control_mode"]),
    )
    if env.control_freq != CONTROL_HZ or int(env._max_episode_steps) != MAX_STEPS:
        raise RuntimeError("runtime LIBERO frequency or episode limit differs from preregistration")
    return (*components, env, task.name, available_states)


def rotation_step_distance(previous: np.ndarray, current: np.ndarray) -> float:
    relative = Rotation.from_rotvec(previous).inv() * Rotation.from_rotvec(current)
    return float(relative.magnitude())


def summarize_action_sequence(records: list[dict[str, Any]]) -> dict[str, Any]:
    actions = np.asarray([record["action"] for record in records], dtype=np.float64)
    effective_ages = np.asarray([record["mean_effective_age_ticks"] for record in records])
    summary: dict[str, Any] = {
        "mean_effective_source_age_ticks": float(effective_ages.mean()),
        "mean_effective_source_age_seconds": float(effective_ages.mean() / CONTROL_HZ),
        "max_candidate_count": int(max(record["candidate_count"] for record in records)),
        "gripper_transitions": int(np.count_nonzero(np.signbit(actions[1:, 6]) != np.signbit(actions[:-1, 6])))
        if len(actions) > 1
        else 0,
    }
    if len(actions) > 1:
        summary["mean_translation_action_delta_l2"] = float(
            np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=1).mean()
        )
        summary["mean_rotation_action_delta_radians"] = float(
            np.mean(
                [rotation_step_distance(actions[i - 1, 3:6], actions[i, 3:6]) for i in range(1, len(actions))]
            )
        )
    else:
        summary["mean_translation_action_delta_l2"] = 0.0
        summary["mean_rotation_action_delta_radians"] = 0.0
    summary["mean_raw_action_acceleration_l2"] = (
        float(np.linalg.norm(np.diff(actions, n=2, axis=0), axis=1).mean()) if len(actions) > 2 else 0.0
    )
    summary["mean_raw_action_jerk_l2"] = (
        float(np.linalg.norm(np.diff(actions, n=3, axis=0), axis=1).mean()) if len(actions) > 3 else 0.0
    )
    return summary


def run_episode(
    *,
    env: Any,
    policy: Any,
    policy_preprocessor: Any,
    policy_postprocessor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    run: dict[str, Any],
) -> dict[str, Any]:
    import torch

    seed = int(run["episode_seed"])
    state_id = int(run["state_id"])
    method = str(run["method"])
    set_episode_seed(seed)
    env.init_state_id = state_id
    observation, _ = env.reset(seed=seed)
    policy.reset()
    aggregator = DenseTemporalAggregator(method, chunk_length=CHUNK_LENGTH, action_dim=ACTION_DIM)
    records: list[dict[str, Any]] = []
    query_seconds = 0.0
    episode_start = time.perf_counter()
    info: dict[str, Any] = {"is_success": False}

    for step in range(MAX_STEPS):
        query_start = time.perf_counter()
        chunk = query_full_act_chunk(
            observation=observation,
            policy=policy,
            policy_preprocessor=policy_preprocessor,
            policy_postprocessor=policy_postprocessor,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        query_elapsed = time.perf_counter() - query_start
        query_seconds += query_elapsed
        aggregated = aggregator.update(step, chunk)
        observation, reward, terminated, truncated, info = env.step(aggregated.action)
        records.append(
            {
                "step": step,
                "action": aggregated.action.astype(float).tolist(),
                "candidate_count": aggregated.candidate_count,
                "oldest_age_ticks": int(aggregated.candidate_ages[0]),
                "oldest_weight": float(aggregated.weights[0]),
                "newest_weight": float(aggregated.weights[-1]),
                "mean_effective_age_ticks": aggregated.mean_effective_age,
                "mean_effective_age_seconds": aggregated.mean_effective_age / CONTROL_HZ,
                "query_seconds": query_elapsed,
                "reward": float(reward),
                "is_success": bool(info["is_success"]),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        if terminated or truncated:
            break

    episode_seconds = time.perf_counter() - episode_start
    steps = len(records)
    if steps == 0:
        raise RuntimeError("episode executed no actions")
    summary = {
        "success": bool(info["is_success"]),
        "failure_category": "success" if bool(info["is_success"]) else "time_limit",
        "steps": steps,
        "policy_queries": steps,
        "policy_queries_per_surviving_step": 1.0,
        "policy_query_seconds": query_seconds,
        "episode_wall_seconds": episode_seconds,
        **summarize_action_sequence(records),
    }
    return {"run": run, "summary": summary, "steps": records}


def episode_path(output_root: Path, run: dict[str, Any]) -> Path:
    return (
        output_root
        / "episodes"
        / f"task_{int(run['task_id']):02d}"
        / f"state_{int(run['state_id']):02d}"
        / f"{run['method']}.json.gz"
    )


def write_episode(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    os.replace(temporary, path)
    return sha256(path)


def read_valid_episode(path: Path, run: dict[str, Any], provenance_data: dict[str, Any]) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1 or payload.get("run") != run:
        raise RuntimeError(f"existing episode identity/schema mismatch: {path}")
    saved_provenance = payload.get("provenance", {})
    for field in ("model_sha256", "config_sha256", "lerobot_git_commit", "schedule_sha256"):
        if saved_provenance.get(field) != provenance_data[field]:
            raise RuntimeError(f"existing episode provenance mismatch for {field}: {path}")
    summary = payload.get("summary", {})
    if int(summary.get("steps", -1)) != len(payload.get("steps", [])):
        raise RuntimeError(f"existing episode step count mismatch: {path}")
    if int(summary.get("policy_queries", -1)) != int(summary.get("steps", -2)):
        raise RuntimeError(f"existing episode violates one query per step: {path}")
    return payload


def manifest_entry(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    run = payload["run"]
    summary = payload["summary"]
    return {
        **run,
        "status": "complete",
        "success": bool(summary["success"]),
        "failure_category": summary["failure_category"],
        "steps": int(summary["steps"]),
        "policy_queries": int(summary["policy_queries"]),
        "policy_queries_per_surviving_step": float(summary["policy_queries_per_surviving_step"]),
        "policy_query_seconds": float(summary["policy_query_seconds"]),
        "episode_wall_seconds": float(summary["episode_wall_seconds"]),
        "mean_effective_source_age_ticks": float(summary["mean_effective_source_age_ticks"]),
        "mean_effective_source_age_seconds": float(summary["mean_effective_source_age_seconds"]),
        "mean_translation_action_delta_l2": float(summary["mean_translation_action_delta_l2"]),
        "mean_rotation_action_delta_radians": float(summary["mean_rotation_action_delta_radians"]),
        "gripper_transitions": int(summary["gripper_transitions"]),
        "mean_raw_action_acceleration_l2": float(summary["mean_raw_action_acceleration_l2"]),
        "mean_raw_action_jerk_l2": float(summary["mean_raw_action_jerk_l2"]),
        "log_path": str(path.resolve()),
        "log_bytes": path.stat().st_size,
        "log_sha256": sha256(path),
    }


def write_manifest(
    path: Path,
    *,
    schedule: dict[str, Any],
    provenance_data: dict[str, Any],
    entries: dict[int, dict[str, Any]],
) -> None:
    ordered = [entries[index] for index in sorted(entries)]
    atomic_json(
        path,
        {
            "schema_version": 1,
            "scope": "Gate-3A2 preregistered closed-loop temporal aggregation audit",
            "planned_episodes": int(schedule["planned_episodes"]),
            "completed_episodes": len(ordered),
            "complete": len(ordered) == int(schedule["planned_episodes"]),
            "valid_policy_queries": int(sum(entry["policy_queries"] for entry in ordered)),
            "valid_environment_steps": int(sum(entry["steps"] for entry in ordered)),
            "provenance": provenance_data,
            "episodes": ordered,
        },
    )


def configure_determinism() -> None:
    import torch

    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def main() -> None:
    args = parse_args()
    schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
    if schedule.get("planned_episodes") != 400 or tuple(schedule.get("methods", [])) != METHODS:
        raise RuntimeError("schedule does not match the preregistered 400-episode method set")
    config = load_config(args.config)
    if int(config.get("control_freq", -1)) != int(CONTROL_HZ):
        raise RuntimeError("rollout config is not the preregistered 20 Hz configuration")
    provenance_data = provenance(args.checkpoint, args.config, args.schedule)
    configure_determinism()
    args.output_root.mkdir(parents=True, exist_ok=True)

    entries: dict[int, dict[str, Any]] = {}
    for run in schedule["runs"]:
        path = episode_path(args.output_root, run)
        if path.exists():
            payload = read_valid_episode(path, run, provenance_data)
            entries[int(run["run_index"])] = manifest_entry(path, payload)
    write_manifest(
        args.manifest,
        schedule=schedule,
        provenance_data=provenance_data,
        entries=entries,
    )
    if args.validate_runtime:
        runtime = make_env_and_policy(config, args.checkpoint, 0)
        runtime[5].close()
        if int(runtime[7]) != 50:
            raise RuntimeError(f"task 0 has {runtime[7]} official states, expected 50")
        print("validated frozen policy and task-0 environment contract without executing an episode")
    if args.verify_only:
        print(f"verified {len(entries)}/{schedule['planned_episodes']} completed episodes")
        return

    pending = [
        run
        for run in schedule["runs"]
        if int(run["run_index"]) not in entries
        and (args.task_id is None or int(run["task_id"]) == args.task_id)
    ]
    if args.max_new_runs is not None:
        pending = pending[: args.max_new_runs]

    current_task: int | None = None
    runtime: tuple[Any, ...] | None = None
    completed_new = 0
    try:
        for run in pending:
            task_id = int(run["task_id"])
            if task_id != current_task:
                if runtime is not None:
                    runtime[5].close()
                runtime = make_env_and_policy(config, args.checkpoint, task_id)
                current_task = task_id
                if int(runtime[7]) != 50:
                    raise RuntimeError(f"task {task_id} has {runtime[7]} official states, expected 50")
            assert runtime is not None
            policy, policy_pre, policy_post, env_pre, env_post, env, _, _ = runtime
            payload = run_episode(
                env=env,
                policy=policy,
                policy_preprocessor=policy_pre,
                policy_postprocessor=policy_post,
                env_preprocessor=env_pre,
                env_postprocessor=env_post,
                run=run,
            )
            output_path = episode_path(args.output_root, run)
            payload = {"schema_version": 1, "provenance": provenance_data, **payload}
            write_episode(output_path, payload)
            entries[int(run["run_index"])] = manifest_entry(output_path, payload)
            write_manifest(
                args.manifest,
                schedule=schedule,
                provenance_data=provenance_data,
                entries=entries,
            )
            completed_new += 1
            print(
                f"[{len(entries):03d}/{schedule['planned_episodes']}] task={task_id} "
                f"state={run['state_id']} method={run['method']} "
                f"success={payload['summary']['success']} steps={payload['summary']['steps']}",
                flush=True,
            )
    finally:
        if runtime is not None:
            runtime[5].close()
    print(f"completed {completed_new} new episodes; total valid={len(entries)}")


if __name__ == "__main__":
    main()
