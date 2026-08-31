#!/usr/bin/env python3
"""Run the canonical LeRobot ACT temporal ensemble on the frozen blind panel."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TEMPORAL_ROOT = ROOT.parent
sys.path.insert(0, str(TEMPORAL_ROOT))

from run_component_reuse import atomic_json, write_progress  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "protocol.json"


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


def infer_chunk_current(
    observation,
    env,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    torch,
):
    """Use the validated LeRobot 0.4.4 LIBERO observation pipeline."""

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
    if result.shape != (1, int(policy.config.chunk_size), 7):
        raise RuntimeError(f"unexpected postprocessed ACT chunk shape: {result.shape}")
    return result


def official_action(
    observation,
    env,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    torch,
):
    """Call the installed LeRobot policy's actual temporal-ensemble path."""

    from lerobot.envs.utils import add_envs_task, preprocess_observation
    from lerobot.utils.constants import ACTION

    batch = preprocess_observation(observation)
    batch = add_envs_task(env, batch)
    batch = env_preprocessor(batch)
    batch = preprocessor(batch)
    with torch.inference_mode():
        # ACTPolicy.select_action invokes ACTTemporalEnsembler.update when
        # temporal_ensemble_coeff is enabled in the effective config.
        action = policy.select_action(batch)
        action = postprocessor(action)
        action = env_postprocessor({ACTION: action})[ACTION]
    result = action.detach().cpu().numpy().astype(np.float32, copy=False)
    if result.shape != (1, 7):
        raise RuntimeError(f"unexpected postprocessed official ACT action shape: {result.shape}")
    return result


def rollout_episode(
    *,
    env,
    policy,
    processors,
    torch,
    state_id: int,
    seed: int,
    policy_rng_seed: int,
    max_steps: int,
) -> dict:
    env.envs[0].init_state_id = int(state_id)
    actual_state_id = int(env.envs[0].init_state_id)
    if actual_state_id != int(state_id):
        raise RuntimeError(f"initial-state assignment mismatch: requested={state_id}, actual={actual_state_id}")
    reset_policy_rng(torch, policy_rng_seed)
    policy.reset()
    observation, _ = env.reset(seed=[int(seed)])
    env_preprocessor, env_postprocessor, preprocessor, postprocessor = processors
    query_steps: list[int] = []
    success = False
    done = False
    completion_step = None
    for step in range(int(max_steps)):
        action = official_action(
            observation,
            env,
            policy,
            env_preprocessor,
            env_postprocessor,
            preprocessor,
            postprocessor,
            torch,
        )
        query_steps.append(step)
        observation, reward, terminated, truncated, info = env.step(action)
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
        "seed": int(seed),
        "requested_initial_state_id": int(state_id),
        "actual_initial_state_id": actual_state_id,
        "success": bool(success),
        "completion_steps": completion_step,
        "environment_steps": step + 1,
        "policy_queries": len(query_steps),
        "query_count": len(query_steps),
        "query_rate": len(query_steps) / float(step + 1),
        "query_steps": query_steps,
        "query_every_environment_step": query_steps == list(range(step + 1)),
    }


def semantic_smoke() -> None:
    """Verify the installed class uses the documented coefficient direction."""

    import torch
    from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

    ensembler = ACTTemporalEnsembler(temporal_ensemble_coeff=0.01, chunk_size=4)
    chunks = torch.tensor(
        [
            [[0.0], [1.0], [2.0], [3.0]],
            [[10.0], [11.0], [12.0], [13.0]],
        ]
    )
    first = ensembler.update(chunks[:1])
    second = ensembler.update(chunks[1:])
    assert first.shape == (1, 1) and float(first[0, 0]) == 0.0
    # At target step one, the oldest prediction is chunk_0[1] and the newest
    # is chunk_1[0]. Positive 0.01 gives the oldest a slightly larger weight.
    expected = (np.exp(-0.01 * 0.0) * 1.0 + np.exp(-0.01 * 1.0) * 10.0) / (1.0 + np.exp(-0.01))
    np.testing.assert_allclose(float(second[0, 0]), expected, rtol=1e-6, atol=1e-6)
    print(json.dumps({"status": "act_temporal_ensemble_official_semantic_smoke_pass", "coefficient": 0.01, "queries_every_step": True}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--task", action="append", help="suite:task_id; repeatable")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress", "--progress-file", dest="progress_file", type=Path)
    parser.add_argument("--semantic-smoke", action="store_true")
    args = parser.parse_args()
    if args.semantic_smoke:
        semantic_smoke()
        return
    if not args.task or args.output is None:
        raise SystemExit("--task and --output are required unless --semantic-smoke is used")

    protocol = json.loads(args.protocol.read_text())
    task_map = {f"{task['suite']}:task{int(task['task_id'])}": task for task in protocol["tasks"]}
    wanted = list(dict.fromkeys(args.task))
    if any(key not in task_map for key in wanted):
        raise SystemExit(f"task is absent from frozen protocol: {[key for key in wanted if key not in task_map]}")

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    state_ids = [int(value) for value in protocol["environment"]["initial_state_ids"]]
    seeds = [int(value) for value in protocol["environment"]["seeds"]]
    coefficient = float(protocol["policy"]["temporal_ensemble_coefficient"])
    started = time.time()
    output = {
        "protocol": str(args.protocol.resolve()),
        "implementation": "LeRobot 0.4.4 ACTPolicy.select_action -> ACTTemporalEnsembler.update",
        "runtime": {
            "python_executable": "/home/wjq/workspace/venvs/libero_act/bin/python",
            "lerobot": "0.4.4",
            "torch": "2.7.1+cu128",
            "mujoco": "3.3.1",
        },
        "effective_temporal_ensemble_coefficient": coefficient,
        "effective_n_action_steps": 1,
        "checkpoint_training_step": 100000,
        "tasks": {},
        "started_at": started,
    }
    progress = {"pid": os.getpid(), "started_at": started, "completed_tasks": 0, "completed_episodes": 0}
    write_progress(args.progress_file, progress)

    for task_key in wanted:
        task = task_map[task_key]
        checkpoint = Path(task["checkpoint"]).resolve()
        if not (checkpoint / "config.json").is_file() or not (checkpoint / "model.safetensors").is_file():
            raise SystemExit(f"ACT checkpoint is missing required files: {checkpoint}")
        config = PreTrainedConfig.from_pretrained(checkpoint)
        checkpoint_n_action_steps = int(config.n_action_steps)
        checkpoint_coeff = config.temporal_ensemble_coeff
        config.temporal_ensemble_coeff = coefficient
        config.n_action_steps = 1
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
        config.pretrained_path = checkpoint
        if getattr(config, "type", None) != "act":
            raise RuntimeError(f"expected ACT checkpoint, got {getattr(config, 'type', None)!r}")
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
        env = make_env(env_config, n_envs=1, use_async_envs=False)[task["suite"]][int(task["task_id"])]
        policy = make_policy(cfg=config, env_cfg=env_config)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(config.device)}},
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_config, policy_cfg=config)
        processors = (env_preprocessor, env_postprocessor, preprocessor, postprocessor)
        max_steps = int(np.asarray(env.call("_max_episode_steps")).reshape(-1)[0])
        episodes = []
        for state_id, seed in zip(state_ids, seeds):
            episodes.append(
                rollout_episode(
                    env=env,
                    policy=policy,
                    processors=processors,
                    torch=torch,
                    state_id=state_id,
                    seed=seed,
                    policy_rng_seed=int(protocol["policy"]["policy_rng_seed"]),
                    max_steps=max_steps,
                )
            )
            progress["completed_episodes"] += 1
            progress["current_task"] = task_key
            write_progress(args.progress_file, progress)
        successes = [bool(episode["success"]) for episode in episodes]
        output["tasks"][task_key] = {
            "suite": task["suite"],
            "task_id": int(task["task_id"]),
            "task_name": task["task_name"],
            "checkpoint": str(checkpoint),
            "checkpoint_temporal_ensemble_coeff": checkpoint_coeff,
            "checkpoint_n_action_steps": checkpoint_n_action_steps,
            "successes": successes,
            "success_count": int(sum(successes)),
            "episodes": len(episodes),
            "policy_queries": int(sum(episode["policy_queries"] for episode in episodes)),
            "environment_steps": int(sum(episode["environment_steps"] for episode in episodes)),
            "query_rate": sum(episode["policy_queries"] for episode in episodes) / sum(episode["environment_steps"] for episode in episodes),
            "query_every_environment_step": all(episode["query_every_environment_step"] for episode in episodes),
            "episodes_detail": episodes,
        }
        progress["completed_tasks"] += 1
        atomic_json(args.output, output)
        env.close()
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output["finished_at"] = time.time()
    progress["finished_at"] = output["finished_at"]
    write_progress(args.progress_file, progress)
    atomic_json(args.output, output)
    print(json.dumps({"output": str(args.output), "tasks": len(wanted), "episodes": len(wanted) * len(state_ids)}, indent=2))


if __name__ == "__main__":
    main()
