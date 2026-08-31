#!/usr/bin/env python3
"""SmolVLA sparse-query hard versus temporal-ensemble development runner.

This runner deliberately keeps SmolVLA's native action processing path and
changes only the executor.  The policy is queried at ``q = 0, h, 2h, ...``;
hard execution selects the newest valid chunk and sparse temporal ensembling
combines all valid same-target predictions with the frozen canonical
oldest-to-newest ``exp(-0.01*i)`` weights.

SmolVLA flow-matching inference is stochastic.  Every query resets the torch
CPU and CUDA generators to a seed derived from the frozen canonical key
``smolvla|{suite}:task{task_id}|state={state_id}|env_seed={env_seed}|q={q}``.
The key excludes method and cadence, so hard and TE paths use the same flow
noise at a shared physical query time.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np


SMOL_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SMOL_ROOT.parent
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
DEFAULT_CHECKPOINT = Path("/home/wjq/checkpoints/HuggingFaceVLA_smolvla_libero")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_progress(path: Path | None, value: object) -> None:
    if path is not None:
        atomic_json(path, value)


def canonical_query_key(task_key: str, state_id: int, env_seed: int, q: int) -> str:
    """Return the frozen key; method and cadence are intentionally absent."""

    return (
        f"smolvla|{task_key}|state={int(state_id)}|env_seed={int(env_seed)}|q={int(q)}"
    )


def query_seed_from_key(key: str) -> int:
    """Map the canonical UTF-8 key to the frozen nonnegative 63-bit seed."""

    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def query_seed(task_key: str, state_id: int, env_seed: int, q: int) -> tuple[str, int]:
    key = canonical_query_key(task_key, state_id, env_seed, q)
    return key, query_seed_from_key(key)


def reset_query_generators(torch, seed: int) -> None:
    """Reset CPU and all CUDA generators immediately before each policy call."""

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
    """Run the validated LeRobot SmolVLA postprocessed action path."""

    from lerobot.envs.utils import add_envs_task, preprocess_observation
    from lerobot.utils.constants import ACTION

    # The native preprocessing functions are allowed to mutate their input.
    # The caller therefore passes a per-query observation copy when it needs
    # to compare two identical-observation flow samples.
    batch = preprocess_observation(observation)
    batch = add_envs_task(env, batch)
    batch = env_preprocessor(batch)
    batch = preprocessor(batch)
    with torch.inference_mode():
        # Do not pass a hand-made noise tensor here.  The validated policy
        # samples its native noise, while reset_query_generators supplies the
        # frozen paired per-query RNG semantics.
        chunk = postprocessor(policy.predict_action_chunk(batch))
        chunk = env_postprocessor({ACTION: chunk})[ACTION]
    result = chunk.detach().cpu().numpy().astype(np.float32, copy=False)
    if result.ndim != 3 or result.shape[0] != 1 or result.shape[1] < 50 or result.shape[2] != 7:
        raise RuntimeError(
            f"unexpected postprocessed SmolVLA chunk shape: {result.shape}; expected (1,50,7)"
        )
    return result[0]


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
        queried_seed_key = None
        queried_seed_value = None
        if executor.should_query(target_step):
            queried_seed_key, queried_seed_value = query_seed(
                task_key, state_id, env_seed, target_step
            )
            reset_query_generators(torch, queried_seed_value)
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
                    "query_seed_key": queried_seed_key,
                    "query_rng_seed": int(queried_seed_value),
                    "latency_seconds": float(query_latency),
                }
            )
        else:
            result = executor.step(
                target_step,
                lambda: (_ for _ in ()).throw(RuntimeError("query_fn called off schedule")),
            )

        action = result.action.astype(np.float32, copy=False)
        observation, reward, terminated, truncated, info = env.step(action[None, :])
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
                "query_seed_key": queried_seed_key,
                "query_rng_seed": None if queried_seed_value is None else int(queried_seed_value),
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
        "query_rng_seeds": [entry["query_rng_seed"] for entry in query_log],
        "query_seed_keys": [entry["query_seed_key"] for entry in query_log],
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
                [
                    episode["completion_steps"]
                    for episode in episodes
                    if episode["completion_steps"] is not None
                ]
            )
        )
        if any(episode["completion_steps"] is not None for episode in episodes)
        else None,
        "episodes_detail": episodes,
    }


def semantic_smoke() -> None:
    """Run the Smol-specific CPU semantics with the real shared executor."""

    def fake_chunk(source: int, horizon: int = 50) -> np.ndarray:
        return np.asarray(
            [
                [1000.0 * source + 10.0 * offset + d for d in range(7)]
                for offset in range(horizon)
            ],
            dtype=np.float64,
        )

    for method, cadence in CADENCE_BY_METHOD.items():
        executor = SparseExecutor(
            cadence=cadence,
            prediction_horizon=50,
            mode=MODE_BY_METHOD[method],
            coefficient=0.01,
            action_dim=7,
        )
        calls: list[int] = []
        for target in range(0, 80):
            result = executor.step(target, lambda target=target: calls.append(target) or fake_chunk(target))
            assert result.candidates.source_query_steps[-1] == max(executor.query_steps)
        assert calls == list(range(0, 80, cadence))

    # First segment has one source, and the first overlap is exactly the
    # canonical oldest-to-newest two-source combination at t=h.
    hard = SparseExecutor(cadence=8, prediction_horizon=50, mode="hard")
    te = SparseExecutor(cadence=8, prediction_horizon=50, mode="sparse_te")
    for target in range(8):
        hard_result = hard.step(target, lambda: fake_chunk(0))
        te_result = te.step(target, lambda: fake_chunk(0))
        np.testing.assert_allclose(hard_result.action, te_result.action, atol=1e-12, rtol=0)
    overlap_hard = hard.step(8, lambda: fake_chunk(8))
    overlap_te = te.step(8, lambda: fake_chunk(8))
    np.testing.assert_allclose(overlap_hard.candidates.actions[-1], fake_chunk(8)[0])
    assert overlap_te.candidates.source_query_steps.tolist() == [0, 8]
    assert overlap_te.candidates.offsets.tolist() == [8, 0]
    weights = np.exp(-0.01 * np.arange(2, dtype=np.float64))
    weights /= weights.sum()
    np.testing.assert_allclose(overlap_te.weights, weights, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        overlap_te.action, weights[0] * fake_chunk(0)[8] + weights[1] * fake_chunk(8)[0], atol=1e-12, rtol=0
    )

    # At t=16 all three sources must identify the same physical target.
    assert overlap_te.target_step == 8
    later = SparseExecutor(cadence=8, prediction_horizon=50, mode="sparse_te")
    for target in range(17):
        later_result = later.step(target, lambda target=target: fake_chunk(target))
    assert later_result.candidates.source_query_steps.tolist() == [0, 8, 16]
    assert later_result.candidates.offsets.tolist() == [16, 8, 0]
    assert all(
        int(source) + int(offset) == 16
        for source, offset in zip(
            later_result.candidates.source_query_steps, later_result.candidates.offsets
        )
    )

    # q=0 expires exactly at t=50.  The shorter Smol horizon yields the
    # expected maxima 7 (h8) and 4 (h16).
    for cadence, expected_max in ((8, 7), (16, 4)):
        executor = SparseExecutor(cadence=cadence, prediction_horizon=50, mode="sparse_te")
        counts = [
            executor.step(target, lambda target=target: fake_chunk(target)).candidate_count
            for target in range(80)
        ]
        assert max(counts) == expected_max
        assert 0 in executor.same_target_candidates(49).source_query_steps
        assert 0 not in executor.same_target_candidates(50).source_query_steps

    # Hard and TE have the exact same scheduler query times.  No TE operation
    # can call the function off schedule because SparseExecutor guards it.
    hard = SparseExecutor(cadence=16, prediction_horizon=50, mode="hard")
    te = SparseExecutor(cadence=16, prediction_horizon=50, mode="sparse_te")
    for target in range(80):
        hard.step(target, lambda target=target: fake_chunk(target))
        te.step(target, lambda target=target: fake_chunk(target))
    assert hard.query_steps == te.query_steps == list(range(0, 80, 16))
    print(json.dumps({"status": "smolvla_sparse_te_cpu_semantic_smoke_pass"}))


def paired_rng_smoke() -> None:
    """Direct numerical smoke for paired per-query stochastic sampling."""

    # This mirrors the policy's native torch.normal path without loading a
    # checkpoint, so it is a fast deterministic test of the seed contract.
    import torch

    key, seed = query_seed("libero_object:task3", 10, 2000, 16)

    def sample() -> np.ndarray:
        reset_query_generators(torch, seed)
        return torch.normal(0.0, 1.0, size=(1, 50, 32), dtype=torch.float32).numpy()

    hard_raw = sample()
    te_raw = sample()
    np.testing.assert_array_equal(hard_raw, te_raw)
    assert key == "smolvla|libero_object:task3|state=10|env_seed=2000|q=16"
    assert query_seed("libero_object:task3", 10, 2000, 16) == (key, seed)
    assert query_seed("libero_object:task3", 10, 2000, 8)[1] != seed
    print(
        json.dumps(
            {
                "status": "smolvla_paired_query_rng_cpu_smoke_pass",
                "key": key,
                "seed": seed,
                "raw_chunk_max_abs_error": float(np.max(np.abs(hard_raw - te_raw))),
            }
        )
    )


def extract_protocol_values(protocol: dict) -> tuple[list[int], list[int], int, int, float]:
    environment = protocol["environment_pairing"]
    policy = protocol["policies"]["smolvla"]
    state_ids = [int(value) for value in environment["initial_state_ids"]]
    seeds = [int(value) for value in environment["seeds"]]
    if state_ids != list(range(10, 20)) or seeds != list(range(2000, 2010)):
        raise RuntimeError("frozen SmolVLA development states/seeds drifted")
    if len(state_ids) != len(seeds) or len(state_ids) != int(environment["episodes_per_task_method"]):
        raise RuntimeError("state/seed count does not match frozen episodes_per_task_method")
    horizon = int(policy["prediction_horizon"])
    coefficient = float(protocol["temporal_ensemble"]["coefficient"])
    if horizon != 50:
        raise RuntimeError(f"SmolVLA prediction horizon must be 50, got {horizon}")
    if coefficient != 0.01:
        raise RuntimeError(f"canonical coefficient must remain 0.01, got {coefficient}")
    expected_rule = "int.from_bytes(sha256(key.encode('utf-8')).digest()[:8], 'big') & ((1 << 63) - 1)"
    configured_rule = protocol["policies"]["smolvla"]["query_rng"]["seed_rule"]
    if configured_rule != expected_rule:
        raise RuntimeError("frozen SmolVLA query seed rule drifted")
    if protocol["policies"]["smolvla"]["query_rng"]["method_and_horizon_excluded"] is not True:
        raise RuntimeError("SmolVLA pairing must exclude method and horizon")
    return state_ids, seeds, int(environment["fps"]), horizon, coefficient


def paired_flow_smoke(
    *,
    protocol: dict,
    checkpoint: Path,
    gpu: str,
    task_key: str,
) -> dict:
    """Check real SmolVLA raw chunks on identical observations and query key."""

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch

    # Import policy factory before config so LeRobot registers SmolVLA under
    # the validated 0.4.4 policy registry before loading its config.
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors

    task_map = {f"{task['suite']}:task{int(task['task_id'])}": task for task in protocol["tasks"]}
    task = task_map[task_key]
    suite = task["suite"]
    task_id = int(task["task_id"])
    env_config = LiberoEnv(
        task=suite,
        task_ids=[task_id],
        fps=int(protocol["environment_pairing"]["fps"]),
        obs_type=protocol["environment_pairing"]["obs_type"],
        camera_name=protocol["environment_pairing"]["camera_name"],
        init_states=True,
        observation_width=int(protocol["policies"]["smolvla"]["observation_width"]),
        observation_height=int(protocol["policies"]["smolvla"]["observation_height"]),
        control_mode=protocol["environment_pairing"]["control_mode"],
    )
    envs = make_env(env_config, n_envs=1, use_async_envs=False)
    env = envs[suite][task_id]
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.pretrained_path = checkpoint
    if int(cfg.chunk_size) != 50 or int(cfg.n_action_steps) != 1:
        raise RuntimeError(f"unexpected SmolVLA config chunk={cfg.chunk_size}, n_action_steps={cfg.n_action_steps}")
    policy = make_policy(cfg=cfg, env_cfg=env_config)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(cfg.device)}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_config, policy_cfg=cfg
    )
    env.envs[0].init_state_id = 10
    observation, _ = env.reset(seed=[2000])
    key, seed = query_seed(task_key, 10, 2000, 0)
    policy.reset()
    reset_query_generators(torch, seed)
    first = infer_chunk(
        copy.deepcopy(observation), env, policy, env_preprocessor, env_postprocessor,
        preprocessor, postprocessor, torch
    )
    policy.reset()
    reset_query_generators(torch, seed)
    second = infer_chunk(
        copy.deepcopy(observation), env, policy, env_preprocessor, env_postprocessor,
        preprocessor, postprocessor, torch
    )
    max_abs_error = float(np.max(np.abs(first - second)))
    env.close()
    if not np.array_equal(first, second):
        raise RuntimeError(f"paired SmolVLA raw chunks differ: max_abs_error={max_abs_error}")
    return {
        "status": "smolvla_paired_flow_real_smoke_pass",
        "task": task_key,
        "state_id": 10,
        "environment_seed": 2000,
        "query_physical_step_q": 0,
        "query_seed_key": key,
        "query_rng_seed": seed,
        "raw_chunk_shape": list(first.shape),
        "raw_chunk_max_abs_error": max_abs_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--task", help="suite:task_id; one task per shard")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--live-smoke", action="store_true", help="one task/state, all four methods")
    parser.add_argument("--paired-flow-smoke", action="store_true", help="real identical-observation flow smoke")
    parser.add_argument("--smoke-output", type=Path, help="optional JSON artifact for a smoke result")
    parser.add_argument("--semantic-smoke", action="store_true")
    parser.add_argument("--paired-rng-smoke", action="store_true")
    args = parser.parse_args()
    if args.semantic_smoke:
        semantic_smoke()
        return
    if args.paired_rng_smoke:
        paired_rng_smoke()
        return

    protocol = json.loads(args.protocol.read_text())
    state_ids, seeds, fps, horizon, coefficient = extract_protocol_values(protocol)
    task_map = {f"{task['suite']}:task{int(task['task_id'])}": task for task in protocol["tasks"]}
    task_key = args.task or "libero_object:task3"
    if task_key not in task_map:
        raise SystemExit(f"task is absent from frozen protocol: {task_key}")
    if args.paired_flow_smoke:
        result = paired_flow_smoke(
            protocol=protocol,
            checkpoint=args.checkpoint.resolve(),
            gpu=args.gpu,
            task_key=task_key,
        )
        if args.smoke_output is not None:
            atomic_json(args.smoke_output, result)
        print(json.dumps(result, indent=2))
        return
    if args.output is None:
        raise SystemExit("--output is required unless a smoke mode is used")

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    # Import policy factory before config to preserve the validated 0.4.4
    # registry initialization used by the existing SmolVLA runner.
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors

    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "config.json").is_file():
        raise SystemExit(f"SmolVLA checkpoint config is missing: {checkpoint}")
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.pretrained_path = checkpoint
    if int(cfg.chunk_size) != horizon or int(cfg.n_action_steps) != 1:
        raise RuntimeError(
            f"unexpected native SmolVLA config chunk={cfg.chunk_size}, n_action_steps={cfg.n_action_steps}"
        )
    task = task_map[task_key]
    suite = task["suite"]
    task_id = int(task["task_id"])
    env_config = LiberoEnv(
        task=suite,
        task_ids=[task_id],
        fps=fps,
        obs_type=protocol["environment_pairing"]["obs_type"],
        camera_name=protocol["environment_pairing"]["camera_name"],
        init_states=True,
        observation_width=int(protocol["policies"]["smolvla"]["observation_width"]),
        observation_height=int(protocol["policies"]["smolvla"]["observation_height"]),
        control_mode=protocol["environment_pairing"]["control_mode"],
    )
    envs = make_env(env_config, n_envs=1, use_async_envs=False)
    env = envs[suite][task_id]
    policy = make_policy(cfg=cfg, env_cfg=env_config)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(cfg.device)}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_config, policy_cfg=cfg
    )
    processors = (env_preprocessor, env_postprocessor, preprocessor, postprocessor)
    max_steps = int(np.asarray(env.call("_max_episode_steps")).reshape(-1)[0])
    selected_states = state_ids[:1] if args.live_smoke else state_ids
    selected_seeds = seeds[:1] if args.live_smoke else seeds
    started = time.time()
    output = {
        "policy": "SmolVLA",
        "protocol": str(args.protocol.resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_revision": protocol["policies"]["smolvla"]["checkpoint_revision"],
        "task": task_key,
        "task_name": task["task_name"],
        "methods": list(METHODS),
        "prediction_horizon": horizon,
        "query_rng_rule": protocol["policies"]["smolvla"]["query_rng"],
        "smolvla_query_seed_key_format": "smolvla|{suite}:task{task_id}|state={state_id}|env_seed={env_seed}|q={q}",
        "one_policy_query_only_on_scheduler_query": True,
        "live_smoke": bool(args.live_smoke),
        "gpu": str(args.gpu),
        "started_at": started,
        "methods_result": {},
    }
    progress = {
        "pid": os.getpid(),
        "started_at": started,
        "task": task_key,
        "completed_methods": 0,
        "completed_episodes": 0,
    }
    write_progress(args.progress_file, progress)

    for method in METHODS:
        episodes = []
        progress["current_method"] = method
        write_progress(args.progress_file, progress)
        for state_id, env_seed in zip(selected_states, selected_seeds):
            episode = rollout_episode(
                env=env,
                policy=policy,
                processors=processors,
                torch=torch,
                task_key=task_key,
                method=method,
                state_id=int(state_id),
                env_seed=int(env_seed),
                prediction_horizon=horizon,
                coefficient=coefficient,
                max_steps=max_steps,
            )
            episodes.append(episode)
            progress["completed_episodes"] += 1
            progress["current_state_id"] = int(state_id)
            write_progress(args.progress_file, progress)
        output["methods_result"][method] = summarize_method(episodes)
        progress["completed_methods"] += 1
        write_progress(args.progress_file, progress)
        atomic_json(args.output, output)

    output["finished_at"] = time.time()
    progress["finished_at"] = output["finished_at"]
    write_progress(args.progress_file, progress)
    atomic_json(args.output, output)
    env.close()
    print(json.dumps({"output": str(args.output), "task": task_key, "episodes": len(METHODS) * len(selected_states)}, indent=2))


if __name__ == "__main__":
    main()
