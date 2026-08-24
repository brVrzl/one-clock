#!/usr/bin/env python3
"""Run the execution-only Gate-0 experiment through current LeRobot LIBERO."""

from __future__ import annotations

import argparse
import copy
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
import yaml


ONE_CLOCK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ONE_CLOCK_ROOT / "src"))

from one_clock import (  # noqa: E402
    ActionGroup,
    AffineResidualCalibrator,
    ExponentialChunkSmoother,
    FixedChunkExecutor,
    IdentityPostPolicy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ONE_CLOCK_ROOT / "configs/gate0_libero_object.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--task-name", type=str)
    parser.add_argument(
        "--strategy",
        choices=("global_fixed", "groupwise_fixed"),
        required=True,
    )
    parser.add_argument("--horizon", type=int, help="Global horizon for global_fixed.")
    parser.add_argument(
        "--group-horizons",
        type=str,
        help="Comma-separated group=horizon values for groupwise_fixed.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, help="Number of paired episodes to run.")
    parser.add_argument(
        "--init-state-start",
        type=int,
        default=0,
        help="Official LIBERO initial-state ID for the first episode.",
    )
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-path", type=Path)
    parser.add_argument(
        "--post-policy",
        choices=("identity", "ema", "affine"),
        default="identity",
    )
    parser.add_argument("--calibrator", type=Path)
    parser.add_argument("--smoothing-alpha", type=float, default=0.25)
    parser.add_argument("--correction-scale", type=float, default=1.0)
    parser.add_argument("--gate-threshold", type=float)
    parser.add_argument(
        "--correction-dimensions",
        type=str,
        help="Comma-separated action dimensions repaired by the affine model.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"LIBERO Gate-0 config must be a mapping: {path}")
    return config


def parse_group_horizons(raw: str | None, defaults: dict[str, int]) -> dict[str, int]:
    if raw is None:
        return {name: int(value) for name, value in defaults.items()}
    result: dict[str, int] = {}
    for item in raw.split(","):
        name, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"group horizon must be name=value: {item!r}")
        result[name.strip()] = int(value)
    return result


def build_groups(config: dict[str, Any], horizons: dict[str, int]) -> tuple[ActionGroup, ...]:
    raw_groups = config["action_groups"]
    if set(raw_groups) != set(horizons):
        raise ValueError("action_groups and selected group horizons must have identical names")
    return tuple(
        ActionGroup(name, tuple(int(index) for index in raw_groups[name]), int(horizons[name]))
        for name in raw_groups
    )


def build_executor(
    config: dict[str, Any],
    strategy: str,
    horizon: int | None,
    group_horizons: dict[str, int],
    chunk_size: int,
) -> FixedChunkExecutor:
    action_dim = sum(len(indices) for indices in config["action_groups"].values())
    if strategy == "global_fixed":
        if horizon is None:
            raise ValueError("--horizon is required for global_fixed")
        valid_horizons = {int(value) for value in config.get("global_horizons", [])}
        if valid_horizons and horizon not in valid_horizons:
            raise ValueError(f"global horizon must be one of {sorted(valid_horizons)}")
        groups = build_groups(config, {name: horizon for name in config["action_groups"]})
        return FixedChunkExecutor.global_fixed(
            action_dim=action_dim,
            chunk_size=chunk_size,
            horizon=horizon,
            groups=groups,
        )
    groups = build_groups(config, group_horizons)
    return FixedChunkExecutor.groupwise_fixed(
        action_dim=action_dim,
        chunk_size=chunk_size,
        groups=groups,
    )


def git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_summary(features: dict[str, Any] | None) -> dict[str, Any] | None:
    if features is None:
        return None
    return {
        name: {"type": str(feature.type), "shape": list(feature.shape)}
        for name, feature in features.items()
    }


def batch_robot_state(observation: dict[str, Any]) -> dict[str, Any]:
    """Match the batch shape produced by LeRobot's vector LIBERO environment."""

    result = copy.copy(observation)
    result["robot_state"] = {
        group: {
            name: np.expand_dims(value, axis=0)
            if isinstance(value, np.ndarray) and value.ndim in (1, 2)
            else value
            for name, value in values.items()
        }
        for group, values in observation["robot_state"].items()
    }
    return result


def prepare_policy_observation(
    observation: dict[str, Any],
    env_preprocessor: Any,
    policy_preprocessor: Any,
) -> tuple[dict[str, Any], np.ndarray]:
    from lerobot.envs.utils import preprocess_observation

    batched = batch_robot_state(observation)
    environment_observation = preprocess_observation(batched)
    environment_observation = env_preprocessor(environment_observation)
    state = np.asarray(environment_observation["observation.state"])[0]
    return policy_preprocessor(environment_observation), state


def load_policy_and_processors(config: dict[str, Any], checkpoint: Path) -> tuple[Any, Any, Any, Any, Any]:
    from lerobot.configs import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies import make_policy, make_pre_post_processors

    camera_mapping = dict(config["camera_name_mapping"])
    env_config = LiberoEnvConfig(
        task=str(config["task_suite"]),
        task_ids=[int(config["task_id"])],
        obs_type=str(config["obs_type"]),
        camera_name=str(config["camera_name"]),
        camera_name_mapping=camera_mapping,
        observation_width=int(config["observation_width"]),
        observation_height=int(config["observation_height"]),
        fps=int(config.get("control_freq", 20)),
        init_states=bool(config["init_states"]),
        hard_reset=bool(config["hard_reset"]),
        control_mode=str(config["control_mode"]),
    )
    policy_config = PreTrainedConfig.from_pretrained(checkpoint)
    policy_config.device = str(config.get("device", "cuda"))
    policy_config.pretrained_path = checkpoint
    if policy_config.temporal_ensemble_coeff is not None:
        raise ValueError("Gate-0 requires ACT temporal aggregation to be disabled")

    # The checkpoint contains its complete ResNet backbone. Avoid re-downloading
    # an optional torchvision initialization file before loading those weights.
    policy_config.pretrained_backbone_weights = None
    policy = make_policy(policy_config, env_cfg=env_config)
    policy_preprocessor, policy_postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(policy_config.device)}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_config,
        policy_cfg=policy_config,
    )
    return policy, policy_preprocessor, policy_postprocessor, env_preprocessor, env_postprocessor


def query_full_act_chunk(
    *,
    observation: dict[str, Any],
    policy: Any,
    policy_preprocessor: Any,
    policy_postprocessor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    post_policy: Any,
    task_id: int,
) -> Any:
    import torch

    from lerobot.utils.constants import ACTION

    model_observation, state = prepare_policy_observation(
        observation,
        env_preprocessor,
        policy_preprocessor,
    )
    with torch.inference_mode():
        normalized_chunk = policy.predict_action_chunk(model_observation)
        action = policy_postprocessor(normalized_chunk)
    action = env_postprocessor({ACTION: action})[ACTION]
    chunk = action[0].detach().cpu().numpy()
    post_policy_start = time.perf_counter()
    result = post_policy(state=state, action_chunk=chunk, task_id=task_id)
    return result, time.perf_counter() - post_policy_start


def summarize_run(
    episode_records: list[list[dict[str, object]]],
    successes: int,
    configured_horizons: dict[str, int],
) -> dict[str, object]:
    episodes = len(episode_records)
    environment_steps = sum(len(records) for records in episode_records)
    policy_queries = sum(
        int(record["policy_query"])
        for records in episode_records
        for record in records
    )
    source_age_totals: dict[str, int] = {}
    source_age_counts: dict[str, int] = {}
    for records in episode_records:
        for record in records:
            for group, age in record["source_ages"].items():
                source_age_totals[group] = source_age_totals.get(group, 0) + int(age)
                source_age_counts[group] = source_age_counts.get(group, 0) + 1

    summary: dict[str, object] = {
        "episodes": episodes,
        "success": successes == episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "environment_steps": environment_steps,
        "policy_queries": policy_queries,
        "policy_queries_per_episode": policy_queries / episodes,
        "policy_query_rate": policy_queries / environment_steps,
        "configured_horizons": dict(configured_horizons),
    }
    if source_age_totals:
        summary["mean_source_age_by_group"] = {
            group: source_age_totals[group] / source_age_counts[group]
            for group in source_age_totals
        }
    corrections = [
        record["post_policy"]
        for records in episode_records
        for record in records
        if "post_policy" in record
    ]
    if corrections:
        summary.update(
            {
                "corrected_policy_queries": len(corrections),
                "gate_activation_rate": sum(int(row["gate_active"]) for row in corrections)
                / len(corrections),
                "mean_chunk_correction_norm": sum(float(row["correction_norm"]) for row in corrections)
                / len(corrections),
                "mean_chunk_correction_max_abs": sum(
                    float(row["correction_max_abs"]) for row in corrections
                )
                / len(corrections),
                "mean_post_policy_latency_seconds": sum(
                    float(row["latency_seconds"]) for row in corrections
                )
                / len(corrections),
            }
        )
    return summary


def build_post_policy(args: argparse.Namespace) -> Any:
    if args.post_policy == "identity":
        return IdentityPostPolicy()
    if args.post_policy == "ema":
        return ExponentialChunkSmoother(args.smoothing_alpha)
    if args.calibrator is None:
        raise ValueError("--calibrator is required for --post-policy affine")
    correction_dimensions = (
        tuple(int(value) for value in args.correction_dimensions.split(","))
        if args.correction_dimensions
        else None
    )
    return AffineResidualCalibrator.load(
        args.calibrator,
        correction_scale=args.correction_scale,
        gate_threshold=args.gate_threshold,
        correction_dimensions=correction_dimensions,
    )


def make_episode_record(
    *,
    episode: int,
    init_state_id: int,
    seed: int,
    strategy: str,
    configured_horizons: dict[str, int],
    success: bool,
    records: list[dict[str, object]],
    task_name: str,
    task_description: str,
    initial_eef_pos: np.ndarray,
    initial_image_means: dict[str, float],
) -> dict[str, object]:
    environment_steps = len(records)
    policy_queries = sum(int(record["policy_query"]) for record in records)
    return {
        "episode": episode,
        "init_state_id": init_state_id,
        "seed": seed,
        "strategy": strategy,
        "arm_horizon": configured_horizons["arm"],
        "gripper_horizon": configured_horizons["gripper"],
        "success": bool(success),
        "environment_steps": environment_steps,
        "policy_queries": policy_queries,
        "policy_query_rate": policy_queries / environment_steps,
        "mean_source_age_arm": sum(
            int(record["source_ages"]["arm"]) for record in records
        )
        / environment_steps,
        "mean_source_age_gripper": sum(
            int(record["source_ages"]["gripper"]) for record in records
        )
        / environment_steps,
        "task_name": task_name,
        "task_description": task_description,
        "initial_eef_pos": initial_eef_pos.tolist(),
        "initial_image_means": dict(initial_image_means),
    }


def set_episode_seed(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_video_frame(video_writer: Any, observation: dict[str, Any]) -> None:
    agentview = np.asarray(observation["pixels"]["image"])
    wristview = np.asarray(observation["pixels"]["wrist_image"])
    video_writer.append_data(np.concatenate((agentview, wristview), axis=1))


def run_episode(
    *,
    env: Any,
    policy: Any,
    policy_preprocessor: Any,
    policy_postprocessor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    executor: FixedChunkExecutor,
    post_policy: Any,
    task_id: int,
    episode: int,
    init_state_id: int,
    seed: int,
    video_writer: Any | None,
) -> tuple[bool, list[dict[str, object]], tuple[int, ...], dict[str, object]]:
    set_episode_seed(seed)
    env.init_state_id = init_state_id
    observation, _ = env.reset(seed=seed)
    policy.reset()
    executor.reset()
    records: list[dict[str, object]] = []
    observed_chunk_shapes: list[tuple[int, ...]] = []
    initial_eef_pos = np.asarray(observation["robot_state"]["eef"]["pos"])
    initial_image_means = {
        name: float(np.asarray(image).mean())
        for name, image in observation["pixels"].items()
    }
    for _ in range(env._max_episode_steps):
        if video_writer is not None:
            append_video_frame(video_writer, observation)

        def query() -> np.ndarray:
            result, post_policy_latency = query_full_act_chunk(
                observation=observation,
                policy=policy,
                policy_preprocessor=policy_preprocessor,
                policy_postprocessor=policy_postprocessor,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                post_policy=post_policy,
                task_id=task_id,
            )
            chunk = np.clip(result.action_chunk, -1.0, 1.0)
            observed_chunk_shapes.append(tuple(chunk.shape))
            post_policy_record = result.as_log_record()
            post_policy_record["latency_seconds"] = post_policy_latency
            post_policy_records.append(post_policy_record)
            return chunk

        post_policy_records: list[dict[str, float | bool]] = []
        decision = executor.step(query)
        observation, _, terminated, truncated, info = env.step(decision.action.astype(np.float32))
        record = decision.as_log_record()
        if post_policy_records:
            record["post_policy"] = post_policy_records[0]
        record["is_success"] = bool(info["is_success"])
        records.append(record)
        if terminated or truncated:
            break
    if video_writer is not None:
        append_video_frame(video_writer, observation)
    success = bool(info["is_success"])
    episode_record = make_episode_record(
        episode=episode,
        init_state_id=init_state_id,
        seed=seed,
        strategy=executor.strategy,
        configured_horizons=executor._configured_horizons,
        success=success,
        records=records,
        task_name=env.task,
        task_description=env.task_description,
        initial_eef_pos=initial_eef_pos,
        initial_image_means=initial_image_means,
    )
    return success, records, tuple(observed_chunk_shapes), episode_record


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    episodes = int(config["episodes"] if args.episodes is None else args.episodes)
    init_state_start = int(args.init_state_start)
    if episodes < 1 or init_state_start < 0:
        raise ValueError("episodes must be positive and init-state-start must be non-negative")
    if args.video_path is not None and not args.record_video:
        raise ValueError("--video-path requires --record-video")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"ACT checkpoint directory does not exist: {checkpoint}")

    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv

    task_suite_name = str(config["task_suite"])
    task_id = int(config["task_id"] if args.task_id is None else args.task_id)
    suite = benchmark.get_benchmark_dict()[task_suite_name]()
    task = suite.get_task(task_id)
    if args.task_name is not None and args.task_name != task.name:
        raise ValueError(f"--task-name does not match LIBERO task: {task.name}")
    if args.task_id is None and str(config["task_name"]) != task.name:
        raise ValueError(f"config task_name does not match LIBERO task: {task.name}")

    runtime_config = dict(config)
    runtime_config["task_id"] = task_id
    runtime_config["task_name"] = task.name
    policy, policy_preprocessor, policy_postprocessor, env_preprocessor, env_postprocessor = (
        load_policy_and_processors(runtime_config, checkpoint)
    )
    chunk_size = int(policy.config.chunk_size)
    action_dim = int(policy.config.output_features["action"].shape[0])
    configured_chunk_size = int(config["chunk_size"])
    if chunk_size != configured_chunk_size:
        raise ValueError(f"config chunk_size {configured_chunk_size} != ACT chunk_size {chunk_size}")
    if action_dim != 7:
        raise ValueError(f"LIBERO Gate-0 requires a 7-D ACT action, got {action_dim}")

    group_horizons = parse_group_horizons(
        args.group_horizons,
        {name: int(value) for name, value in config["groupwise_horizons"].items()},
    )
    configured_horizons = (
        {name: int(args.horizon) for name in config["action_groups"]}
        if args.strategy == "global_fixed"
        else group_horizons
    )
    executor = build_executor(
        config,
        args.strategy,
        args.horizon,
        group_horizons,
        chunk_size,
    )
    post_policy = build_post_policy(args)
    env = LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=task_suite_name,
        obs_type=str(config["obs_type"]),
        camera_name=str(config["camera_name"]),
        camera_name_mapping=dict(config["camera_name_mapping"]),
        observation_width=int(config["observation_width"]),
        observation_height=int(config["observation_height"]),
        control_freq=int(config.get("control_freq", 20)),
        init_states=bool(config["init_states"]),
        hard_reset=bool(config["hard_reset"]),
        control_mode=str(config["control_mode"]),
    )
    official_init_state_count = len(env._init_states)
    init_state_ids = list(range(init_state_start, init_state_start + episodes))
    if init_state_ids[-1] >= official_init_state_count:
        raise ValueError(
            f"requested init state {init_state_ids[-1]} but task has "
            f"{official_init_state_count} official initial states"
        )

    output_dir = args.output_dir
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    video_path = None
    video_writer = None
    if args.record_video:
        import imageio.v2 as imageio

        video_path = (args.video_path or output_dir / "rollout.mp4").resolve()
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(
            video_path,
            fps=int(config.get("control_freq", 20)),
            codec="libx264",
        )
    import torch

    lerobot_root = Path(__import__("lerobot").__file__).resolve().parents[2]
    metadata = {
        "strategy": args.strategy,
        "global_horizon": args.horizon,
        "group_horizons": configured_horizons,
        "action_groups": config["action_groups"],
        "task_suite": task_suite_name,
        "task_id": task_id,
        "task_name": task.name,
        "task_description": task.language,
        "control_mode": config["control_mode"],
        "control_frequency": int(config.get("control_freq", 20)),
        "chunk_size": chunk_size,
        "action_dim": action_dim,
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": sha256_file(checkpoint / "model.safetensors"),
        "project_commit_at_launch": git_commit(ONE_CLOCK_ROOT),
        "lerobot_root": str(lerobot_root),
        "lerobot_commit": git_commit(lerobot_root),
        "lerobot_version": importlib.metadata.version("lerobot"),
        "libero_version": importlib.metadata.version("hf-libero"),
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "policy_temporal_ensemble_coeff": policy.config.temporal_ensemble_coeff,
        "policy_input_features": feature_summary(policy.config.input_features),
        "policy_output_features": feature_summary(policy.config.output_features),
        "episodes": episodes,
        "init_state_start": init_state_start,
        "init_state_ids": init_state_ids,
        "official_init_state_count": official_init_state_count,
        "base_seed": int(config["seed"]),
        "video_path": str(video_path) if video_path is not None else None,
        "post_policy": args.post_policy,
        "calibrator": str(args.calibrator.resolve()) if args.calibrator is not None else None,
        "smoothing_alpha": args.smoothing_alpha if args.post_policy == "ema" else None,
        "correction_scale": args.correction_scale if args.post_policy == "affine" else None,
        "gate_threshold": args.gate_threshold if args.post_policy == "affine" else None,
        "correction_dimensions": (
            args.correction_dimensions if args.post_policy == "affine" else None
        ),
        "action_safety_projection": "clip[-1,1] after post-policy module",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    successes = 0
    episode_records: list[list[dict[str, object]]] = []
    with (
        (output_dir / "steps.jsonl").open("w", encoding="utf-8") as log_file,
        (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as episode_log,
    ):
        for episode_index, init_state_id in enumerate(init_state_ids):
            success, records, chunk_shapes, episode_record = run_episode(
                env=env,
                policy=policy,
                policy_preprocessor=policy_preprocessor,
                policy_postprocessor=policy_postprocessor,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                executor=executor,
                post_policy=post_policy,
                task_id=task_id,
                episode=episode_index,
                init_state_id=init_state_id,
                seed=int(config["seed"]) + init_state_id,
                video_writer=video_writer,
            )
            successes += int(success)
            episode_records.append(records)
            episode_log.write(json.dumps(episode_record) + "\n")
            for record in records:
                record["episode"] = episode_index
                record["init_state_id"] = init_state_id
                record["seed"] = int(config["seed"]) + init_state_id
                log_file.write(json.dumps(record) + "\n")
            if chunk_shapes and "observed_chunk_shape" not in metadata:
                metadata["observed_chunk_shape"] = list(chunk_shapes[0])

    if video_writer is not None:
        video_writer.close()
    env.close()
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    summary = summarize_run(
        episode_records,
        successes,
        configured_horizons,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
