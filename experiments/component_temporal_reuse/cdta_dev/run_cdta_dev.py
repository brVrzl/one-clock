#!/usr/bin/env python3
"""Run the frozen 200-episode ACT CDTA-16 development panel.

One invocation evaluates one protocol task with one ten-worker vector
environment.  Every method is reset on the same ten explicit LIBERO initial
state IDs and seeds before rollout.  The policy is queried once per vector
environment step and same-target candidates are extracted after native
postprocessing.
"""

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

from temporal_operators import (  # noqa: E402
    act_temporal_weights,
    aggregate_components,
    aggregate_full_action,
    cogact_cosine_weights,
    same_target_candidates,
)


METHODS = (
    "fresh",
    "official_act_m001",
    "cogact_full_alpha01",
    "matched_shared_a16_alpha03_beta003",
    "cdta_a16_alpha03_beta003",
)
DEFAULT_PROTOCOL = ROOT / "protocol.json"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def write_progress(path: Path | None, value: object) -> None:
    if path is not None:
        atomic_json(path, value)


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    shifted = scores - np.max(scores)
    values = np.exp(shifted)
    return values / values.sum()


def _cosine_to_newest(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    newest = values[-1]
    denominator = np.linalg.norm(values, axis=1) * np.linalg.norm(newest) + 1e-7
    return (values @ newest) / denominator


def _recent(candidates, max_age: int = 16) -> tuple[np.ndarray, np.ndarray]:
    keep = np.asarray(candidates.ages) <= int(max_age)
    if not np.any(keep) or not np.any(np.asarray(candidates.ages)[keep] == 0):
        raise RuntimeError("same-target candidate window lost the newest source")
    return candidates.actions[keep], np.asarray(candidates.ages, dtype=np.float64)[keep]


def compose_action(method: str, candidates) -> tuple[np.ndarray, float, float, dict]:
    """Compose one postprocessed same-target candidate set."""

    actions = candidates.actions
    ages = np.asarray(candidates.ages, dtype=np.float64)
    if method == "fresh":
        return actions[-1].copy(), 0.0, 0.0, {"candidate_count": len(actions)}
    if method == "official_act_m001":
        weights = act_temporal_weights(len(actions), coefficient=0.01)
        action = aggregate_full_action(actions, weights)
        age = float(weights @ ages)
        return action, age, age, {"candidate_count": len(actions)}
    if method == "cogact_full_alpha01":
        weights = cogact_cosine_weights(actions, alpha=0.1)
        action = aggregate_full_action(actions, weights)
        age = float(weights @ ages)
        return action, age, age, {"candidate_count": len(actions)}

    recent_actions, recent_ages = _recent(candidates)
    if method == "matched_shared_a16_alpha03_beta003":
        scores = 0.3 * _cosine_to_newest(recent_actions) - 0.03 * recent_ages
        weights = _softmax(scores)
        action = aggregate_full_action(recent_actions, weights)
        age = float(weights @ recent_ages)
        return action, age, age, {"candidate_count": len(recent_actions), "max_age": 16}
    if method == "cdta_a16_alpha03_beta003":
        arm_scores = 0.3 * _cosine_to_newest(recent_actions[:, :6]) - 0.03 * recent_ages
        gripper_sign_agreement = np.sign(recent_actions[:, 6]) * np.sign(recent_actions[-1, 6])
        gripper_scores = 0.3 * gripper_sign_agreement - 0.03 * recent_ages
        arm_weights = _softmax(arm_scores)
        gripper_weights = _softmax(gripper_scores)
        action = aggregate_components(
            recent_actions, {"arm": arm_weights, "gripper": gripper_weights}
        )
        return (
            action,
            float(arm_weights @ recent_ages),
            float(gripper_weights @ recent_ages),
            {
                "candidate_count": len(recent_actions),
                "max_age": 16,
                "gripper_sign_agreement": gripper_sign_agreement.tolist(),
            },
        )
    raise ValueError(f"unknown CDTA development method: {method}")


def semantic_smoke() -> None:
    """CPU-only checks for target alignment and the frozen five methods."""

    chunks = [
        np.asarray([[100.0 * source + offset + d for d in range(7)] for offset in range(20)])
        for source in range(18)
    ]
    candidates = same_target_candidates(chunks, target_step=17)
    assert candidates.ages.tolist() == list(range(17, -1, -1))
    for method in METHODS:
        action, arm_age, gripper_age, details = compose_action(method, candidates)
        assert action.shape == (7,) and np.isfinite(action).all()
        assert 0.0 <= arm_age <= 16.0 or method in {"fresh", "official_act_m001", "cogact_full_alpha01"}
        assert 0.0 <= gripper_age <= 16.0 or method in {"fresh", "official_act_m001", "cogact_full_alpha01"}
        if method in {"matched_shared_a16_alpha03_beta003", "cdta_a16_alpha03_beta003"}:
            assert details["candidate_count"] == 17

    fresh, *_ = compose_action("fresh", candidates)
    np.testing.assert_array_equal(fresh, candidates.actions[-1])
    shared, *_ = compose_action("matched_shared_a16_alpha03_beta003", candidates)
    np.testing.assert_allclose(shared, compose_action("matched_shared_a16_alpha03_beta003", candidates)[0])

    # Gripper score must use sign agreement, not a continuous cosine or a vote.
    gripper_case = np.asarray(
        [[1.0, 0, 0, 0, 0, 0, -1.0], [2.0, 0, 0, 0, 0, 0, 1.0], [3.0, 0, 0, 0, 0, 0, 1.0]],
        dtype=np.float64,
    )
    fake = type("Candidates", (), {"actions": gripper_case, "ages": np.asarray([2, 1, 0])})()
    cdta, _, gripper_age, details = compose_action("cdta_a16_alpha03_beta003", fake)
    assert details["gripper_sign_agreement"] == [-1.0, 1.0, 1.0]
    assert 0.0 < gripper_age < 1.0
    assert cdta[6] > 0.0
    print(json.dumps({"status": "cdta_dev_cpu_semantic_smoke_pass", "methods": list(METHODS)}))


def reset_policy_rng(torch, seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def infer_chunk_batch(
    observation, env, policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor, torch
) -> np.ndarray:
    """Run the native ACT chunk path while preserving the vector batch."""

    from lerobot.envs.utils import add_envs_task, preprocess_observation
    from lerobot.utils.constants import ACTION

    observation = preprocess_observation(observation)
    observation = add_envs_task(env, observation)
    observation = env_preprocessor(observation)
    observation = preprocessor(observation)
    with torch.inference_mode():
        chunk = postprocessor(policy.predict_action_chunk(observation))
    chunk = env_postprocessor({ACTION: chunk})[ACTION]
    result = chunk.detach().cpu().numpy().astype(np.float32, copy=False)
    if result.ndim != 3 or result.shape[0] != 10 or result.shape[2] != 7:
        raise RuntimeError(f"unexpected batched ACT action chunk shape: {result.shape}; expected (10,H,7)")
    return result


def _success_from_info(info, index: int, reward) -> bool:
    final_info = info.get("final_info") if isinstance(info, dict) else None
    if isinstance(final_info, dict) and "is_success" in final_info:
        value = np.asarray(final_info["is_success"])
        return bool(value.item()) if value.ndim == 0 else bool(value.reshape(-1)[index])
    rewards = np.asarray(reward).reshape(-1)
    return bool(len(rewards) > index and rewards[index] > 0)


def rollout_method(*, env, policy, processors, torch, seeds, state_ids, method, policy_rng_seed, max_steps):
    """Run one paired method over all ten vector workers."""

    for index, requested in enumerate(state_ids):
        env.envs[index].init_state_id = int(requested)
    actual_ids = [int(env.envs[index].init_state_id) for index in range(len(state_ids))]
    if actual_ids != [int(x) for x in state_ids]:
        raise RuntimeError(f"initial state assignment mismatch: requested={state_ids}, actual={actual_ids}")

    reset_policy_rng(torch, policy_rng_seed)
    policy.reset()
    observation, _ = env.reset(seed=[int(x) for x in seeds])
    env_preprocessor, env_postprocessor, preprocessor, postprocessor = processors
    history: list[np.ndarray] = []
    done = np.zeros(len(state_ids), dtype=bool)
    successes: list[bool | None] = [None] * len(state_ids)
    completion_steps: list[int | None] = [None] * len(state_ids)
    arm_ages: list[list[float]] = [[] for _ in state_ids]
    gripper_ages: list[list[float]] = [[] for _ in state_ids]
    candidate_counts: list[list[int]] = [[] for _ in state_ids]
    for step in range(int(max_steps)):
        fresh_batch = infer_chunk_batch(
            observation, env, policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor, torch
        )
        history.append(fresh_batch.copy())
        actions = []
        for index in range(len(state_ids)):
            candidates = same_target_candidates([chunk[index] for chunk in history], target_step=step)
            action, arm_age, gripper_age, details = compose_action(method, candidates)
            actions.append(action if not done[index] else candidates.actions[-1])
            if not done[index]:
                arm_ages[index].append(arm_age)
                gripper_ages[index].append(gripper_age)
                candidate_counts[index].append(int(details["candidate_count"]))
        observation, reward, terminated, truncated, info = env.step(
            np.asarray(actions, dtype=np.float32)
        )
        terminated = np.asarray(terminated, dtype=bool).reshape(-1)
        truncated = np.asarray(truncated, dtype=bool).reshape(-1)
        just_done = (~done) & (terminated | truncated)
        for index in np.flatnonzero(just_done):
            successes[index] = _success_from_info(info, int(index), reward)
            completion_steps[index] = step + 1 if successes[index] else None
        done |= terminated | truncated
        if done.all():
            break
    for index in range(len(state_ids)):
        if successes[index] is None:
            successes[index] = False
    return {
        "method": method,
        "requested_initial_state_ids": [int(x) for x in state_ids],
        "actual_initial_state_ids": actual_ids,
        "seeds": [int(x) for x in seeds],
        "successes": [bool(x) for x in successes],
        "success_count": int(sum(bool(x) for x in successes)),
        "episodes": len(state_ids),
        "environment_steps": [len(values) for values in arm_ages],
        "policy_queries": [len(values) for values in arm_ages],
        "policy_queries_per_environment_step": 1.0,
        "mean_arm_source_age_steps": float(np.mean([np.mean(x) for x in arm_ages])),
        "mean_gripper_source_age_steps": float(np.mean([np.mean(x) for x in gripper_ages])),
        "mean_candidate_count": float(np.mean([np.mean(x) for x in candidate_counts])),
        "completion_steps_successful": [int(x) for x in completion_steps if x is not None],
        "environment_steps_global": step + 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--task", required=False, help="suite:task_id")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress", "--progress-file", dest="progress_file", type=Path)
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
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.pretrained_path = checkpoint
    if getattr(config, "type", None) != "act":
        raise RuntimeError(f"expected ACT checkpoint, got {getattr(config, 'type', None)!r}")
    if int(config.chunk_size) <= 16:
        raise RuntimeError(f"ACT chunk_size {config.chunk_size} cannot expose age 16")

    state_ids = [int(x) for x in protocol["environment"]["initial_state_ids"]]
    seeds = [int(x) for x in protocol["environment"]["seeds"]]
    if state_ids != list(range(10, 20)) or seeds != list(range(2000, 2010)):
        raise RuntimeError("protocol state IDs or seeds drifted from the frozen CDTA dev panel")
    env_config = LiberoEnv(
        task=task["suite"], task_ids=[int(task["task_id"])],
        fps=int(protocol["environment"]["fps"]), obs_type=protocol["environment"]["obs_type"],
        camera_name=protocol["environment"]["camera_name"], init_states=True,
        observation_width=int(protocol["environment"]["observation_width"]),
        observation_height=int(protocol["environment"]["observation_height"]),
        control_mode=protocol["environment"]["control_mode"],
    )
    envs = make_env(env_config, n_envs=10, use_async_envs=False)
    env = envs[task["suite"]][int(task["task_id"])]
    policy = make_policy(cfg=config, env_cfg=env_config)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(config.device)}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_config, policy_cfg=config)
    processors = (env_preprocessor, env_postprocessor, preprocessor, postprocessor)
    max_steps = int(np.max(np.asarray(env.call("_max_episode_steps")).reshape(-1)))
    started = time.time()
    output = {
        "protocol": str(args.protocol.resolve()), "checkpoint": str(checkpoint),
        "checkpoint_chunk_size": int(config.chunk_size), "checkpoint_n_action_steps": int(config.n_action_steps),
        "task": args.task, "task_name": task["task_name"], "methods": list(METHODS),
        "one_policy_query_per_environment_step": True, "n_envs": 10,
        "started_at": started, "methods_result": {},
    }
    progress = {"pid": os.getpid(), "started_at": started, "completed_methods": 0, "completed_episodes": 0}
    write_progress(args.progress_file, progress)
    for method in METHODS:
        result = rollout_method(
            env=env, policy=policy, processors=processors, torch=torch,
            seeds=seeds, state_ids=state_ids, method=method,
            policy_rng_seed=int(protocol["policy"]["policy_rng_seed"]), max_steps=max_steps,
        )
        output["methods_result"][method] = result
        progress.update({"completed_methods": progress["completed_methods"] + 1,
                         "completed_episodes": progress["completed_episodes"] + 10,
                         "current_method": method})
        write_progress(args.progress_file, progress)
        atomic_json(args.output, output)
    output["finished_at"] = time.time()
    progress["finished_at"] = output["finished_at"]
    write_progress(args.progress_file, progress)
    atomic_json(args.output, output)
    env.close()
    print(json.dumps({"output": str(args.output), "task": args.task, "episodes": 50}, indent=2))


if __name__ == "__main__":
    main()
