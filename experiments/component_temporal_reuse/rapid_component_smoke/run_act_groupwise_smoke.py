#!/usr/bin/env python3
"""Run the rapid paired ACT smoke for the three minimal group-wise methods.

Each invocation evaluates one task-specific frozen 100k ACT checkpoint.  Every
method gets the same initial-state IDs and environment seeds, and the same
PyTorch policy RNG stream is reset before each paired episode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from temporal_operators import (
    aggregate_components,
    aggregate_full_action,
    act_temporal_weights,
    cogact_cosine_weights,
    groupwise_similarity_weights,
    same_target_candidates,
    weighted_gripper_vote,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "act_protocol.json"
METHODS = (
    "fresh",
    "official_act_temporal_ensemble",
    "cogact_shared_full_action",
    "component_arm_fresh_gripper_act",
    "groupwise_similarity",
    "groupwise_similarity_age",
    "groupwise_similarity_age_gripper_vote",
)


def compose_action(method: str, candidates) -> tuple[np.ndarray, float, float, dict]:
    """Compose one action and return effective arm/gripper ages plus diagnostics."""

    actions = candidates.actions
    ages = candidates.ages
    if method == "fresh":
        return actions[-1].copy(), 0.0, 0.0, {}
    if method == "official_act_temporal_ensemble":
        weights = act_temporal_weights(len(actions), coefficient=0.01)
        action = aggregate_full_action(actions, weights)
        age = float(weights @ ages)
        return action, age, age, {}
    if method == "cogact_shared_full_action":
        weights = cogact_cosine_weights(actions, alpha=0.3)
        action = aggregate_full_action(actions, weights)
        age = float(weights @ ages)
        return action, age, age, {}
    if method == "component_arm_fresh_gripper_act":
        arm_weights = np.zeros(len(actions), dtype=np.float64)
        arm_weights[-1] = 1.0
        gripper_weights = act_temporal_weights(len(actions), coefficient=0.01)
        action = aggregate_components(actions, {"arm": arm_weights, "gripper": gripper_weights})
        return action, 0.0, float(gripper_weights @ ages), {}
    if method == "groupwise_similarity":
        weights = groupwise_similarity_weights(actions, ages, alpha=0.3, beta=0.0)
        action = aggregate_components(actions, weights)
        return action, float(weights["arm"] @ ages), float(weights["gripper"] @ ages), {}
    if method == "groupwise_similarity_age":
        weights = groupwise_similarity_weights(actions, ages, alpha=0.3, beta=0.03)
        action = aggregate_components(actions, weights)
        return action, float(weights["arm"] @ ages), float(weights["gripper"] @ ages), {}
    if method == "groupwise_similarity_age_gripper_vote":
        weights = groupwise_similarity_weights(actions, ages, alpha=0.3, beta=0.03)
        gripper, representative, winning_sign, support = weighted_gripper_vote(
            actions, weights["gripper"]
        )
        gripper_one_hot = np.zeros(len(actions), dtype=np.float64)
        gripper_one_hot[representative] = 1.0
        action = aggregate_components(actions, {"arm": weights["arm"], "gripper": gripper_one_hot})
        return action, float(weights["arm"] @ ages), float(ages[representative]), {
            "gripper_winning_sign": winning_sign,
            "gripper_representative_index": representative,
            "gripper_support": support,
            "gripper_value": float(gripper[0]),
        }
    raise ValueError(f"unknown method: {method}")


def reset_policy_rng(torch, seed: int) -> None:
    """Reset the ACT latent-sampling stream on CPU and all visible CUDA devices."""

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


def rollout_episode(
    *,
    env,
    policy,
    infer_chunk,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    torch,
    seed: int,
    initial_state_id: int,
    method: str,
    policy_rng_seed: int,
) -> dict:
    """Run one episode for one method with an explicitly paired RNG reset."""

    # The vector wrapper can advance its internal reset stride during setup.
    # Assign the predeclared state explicitly before every reset so paired
    # methods use exactly the same LIBERO initial state.
    env.envs[0].init_state_id = int(initial_state_id)
    actual_initial_state_id = int(env.envs[0].init_state_id)
    if actual_initial_state_id != int(initial_state_id):
        raise RuntimeError(
            f"expected initial_state_id {initial_state_id}, got {actual_initial_state_id}"
        )
    reset_policy_rng(torch, policy_rng_seed)
    policy.reset()
    observation, _ = env.reset(seed=[int(seed)])
    max_steps = int(env.call("_max_episode_steps")[0])
    query_history: list[np.ndarray] = []
    arm_ages: list[float] = []
    gripper_ages: list[float] = []
    vote_open_steps = 0
    vote_close_steps = 0
    vote_tie_steps = 0
    success = False
    completion_step = None
    done = False
    for step in range(max_steps):
        fresh_chunk = infer_chunk(
            observation,
            env,
            policy,
            env_preprocessor,
            env_postprocessor,
            preprocessor,
            postprocessor,
        )
        query_history.append(fresh_chunk.copy())
        candidates = same_target_candidates(query_history, target_step=step)
        action, arm_age, gripper_age, details = compose_action(method, candidates)
        arm_ages.append(arm_age)
        gripper_ages.append(gripper_age)
        if method == "groupwise_similarity_age_gripper_vote":
            support = details["gripper_support"]
            if support["open"] > support["close"]:
                vote_open_steps += 1
            elif support["close"] > support["open"]:
                vote_close_steps += 1
            else:
                vote_tie_steps += 1
        observation, reward, terminated, truncated, info = env.step(action[None, :].astype(np.float32))
        terminated = bool(np.asarray(terminated).reshape(-1)[0])
        truncated = bool(np.asarray(truncated).reshape(-1)[0])
        done = terminated or truncated
        if done:
            success = extract_success(info, reward)
            completion_step = step + 1 if success else None
            break
    return {
        "seed": int(seed),
        "initial_state_id": int(initial_state_id),
        "policy_rng_seed": int(policy_rng_seed),
        "success": bool(success),
        "completion_steps": completion_step,
        "environment_steps": step + 1,
        "policy_queries": step + 1,
        "mean_arm_source_age_steps": float(np.mean(arm_ages)),
        "mean_gripper_source_age_steps": float(np.mean(gripper_ages)),
        "gripper_vote_open_steps": vote_open_steps,
        "gripper_vote_close_steps": vote_close_steps,
        "gripper_vote_tie_steps": vote_tie_steps,
    }


def semantic_smoke() -> None:
    """Exercise candidate alignment, independent weights, and gripper voting."""

    chunks = [
        np.asarray([[[100.0 * source + offset + d for d in range(7)] for offset in range(8)]])
        for source in range(5)
    ]
    candidates = same_target_candidates(chunks, target_step=4)
    if candidates.ages.tolist() != [4, 3, 2, 1, 0]:
        raise SystemExit(f"unexpected target ages: {candidates.ages.tolist()}")
    for method in METHODS:
        action, _, _, _ = compose_action(method, candidates)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise SystemExit(f"invalid smoke action for {method}: {action}")
    weights = groupwise_similarity_weights(candidates.actions, candidates.ages, alpha=0.3, beta=0.03)
    for name, values in weights.items():
        np.testing.assert_allclose(values.sum(), 1.0)
        if np.any(values < 0):
            raise SystemExit(f"invalid {name} weights")

    conflicting = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
        ]
    )
    gripper, representative, sign, support = weighted_gripper_vote(
        conflicting, np.asarray([0.2, 0.3, 0.5])
    )
    np.testing.assert_array_equal(gripper, np.asarray([0.8]))
    if (representative, sign) != (2, 1) or support["open"] <= support["close"]:
        raise SystemExit("gripper vote smoke failed")
    print(json.dumps({"status": "rapid_groupwise_semantic_smoke_pass", "methods": list(METHODS)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--task", required=False, help="suite:task_id")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--methods", default=None, help="comma-separated subset")
    parser.add_argument("--semantic-smoke", action="store_true")
    args = parser.parse_args()
    if args.semantic_smoke:
        semantic_smoke()
        return
    if args.task is None or args.output is None:
        raise SystemExit("--task and --output are required unless --semantic-smoke is used")

    protocol = json.loads(args.protocol.read_text())
    tasks = {
        f"{task['suite']}:task{int(task['task_id'])}": task for task in protocol["tasks"]
    }
    if args.task not in tasks:
        raise SystemExit(f"task is absent from frozen protocol: {args.task}")
    task = tasks[args.task]
    methods = [method for method in (args.methods or ",".join(METHODS)).split(",") if method]
    if not methods or any(method not in METHODS for method in methods):
        raise SystemExit("invalid method selection")
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.resolve()
    else:
        checkpoint = Path(task["checkpoint"]).resolve()
    if not (checkpoint / "config.json").is_file() or not (checkpoint / "model.safetensors").is_file():
        raise SystemExit(f"completed checkpoint is missing required files: {checkpoint}")

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from run_component_reuse import atomic_json, infer_chunk, write_progress

    policy_config = PreTrainedConfig.from_pretrained(checkpoint)
    policy_config.device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_config.pretrained_path = checkpoint
    if getattr(policy_config, "type", None) != "act":
        raise RuntimeError(f"expected ACT checkpoint, got policy type {getattr(policy_config, 'type', None)!r}")
    output = {
        "protocol": str(args.protocol.resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_training_step": 100000,
        "checkpoint_chunk_size": int(policy_config.chunk_size),
        "checkpoint_n_action_steps": int(policy_config.n_action_steps),
        "checkpoint_temporal_ensemble_coeff": policy_config.temporal_ensemble_coeff,
        "task": args.task,
        "task_name": task["task_name"],
        "methods": methods,
        "one_policy_query_per_environment_step": True,
        "paired_initial_state_ids": protocol["environment"]["initial_state_ids"],
        "paired_environment_seeds": protocol["environment"]["seeds"],
        "policy_rng_seed": protocol["policy"]["policy_rng_seed"],
        "policy_rng_reset_before_each_method_episode": True,
        "started_at": time.time(),
        "methods_result": {},
    }
    progress = {
        "pid": os.getpid(),
        "started_at": output["started_at"],
        "completed_methods": 0,
        "completed_episodes": 0,
    }
    write_progress(args.progress_file, progress)

    env_config = LiberoEnv(
        task=task["suite"],
        task_ids=[int(task["task_id"])],
        fps=int(protocol["environment"]["fps"]),
        obs_type=protocol["environment"]["obs_type"],
        camera_name=protocol["environment"]["camera_name"],
        init_states=True,
        observation_width=int(protocol["environment"]["observation_width"]),
        observation_height=int(protocol["environment"]["observation_height"]),
        control_mode=protocol["environment"]["control_mode"],
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

    initial_state_ids = [int(value) for value in protocol["environment"]["initial_state_ids"]]
    seeds = [int(value) for value in protocol["environment"]["seeds"]]
    for method in methods:
        envs = make_env(env_config, n_envs=1, use_async_envs=False)
        env = envs[task["suite"]][int(task["task_id"])]
        env.envs[0].init_state_id = initial_state_ids[0]
        episodes = []
        for initial_state_id, seed in zip(initial_state_ids, seeds):
            episode = rollout_episode(
                env=env,
                policy=policy,
                infer_chunk=infer_chunk,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                torch=torch,
                seed=seed,
                initial_state_id=initial_state_id,
                method=method,
                policy_rng_seed=int(protocol["policy"]["policy_rng_seed"]),
            )
            episodes.append(episode)
            progress["completed_episodes"] += 1
            progress["current_method"] = method
            write_progress(args.progress_file, progress)
        env.close()
        successes = [bool(episode["success"]) for episode in episodes]
        output["methods_result"][method] = {
            "successes": successes,
            "success_count": int(sum(successes)),
            "episodes": len(episodes),
            "success_rate": sum(successes) / len(episodes),
            "policy_queries": sum(episode["policy_queries"] for episode in episodes),
            "environment_steps": sum(episode["environment_steps"] for episode in episodes),
            "policy_queries_per_environment_step": sum(episode["policy_queries"] for episode in episodes)
            / sum(episode["environment_steps"] for episode in episodes),
            "mean_arm_source_age_steps": float(
                np.mean([episode["mean_arm_source_age_steps"] for episode in episodes])
            ),
            "mean_gripper_source_age_steps": float(
                np.mean([episode["mean_gripper_source_age_steps"] for episode in episodes])
            ),
            "completion_steps_successful": [
                episode["completion_steps"] for episode in episodes if episode["success"]
            ],
            "episodes_detail": episodes,
        }
        progress["completed_methods"] += 1
        atomic_json(args.output, output)

    output["finished_at"] = time.time()
    progress["finished_at"] = output["finished_at"]
    write_progress(args.progress_file, progress)
    atomic_json(args.output, output)
    print(json.dumps({"output": str(args.output), "task": args.task, "methods": methods}, indent=2))


if __name__ == "__main__":
    main()
