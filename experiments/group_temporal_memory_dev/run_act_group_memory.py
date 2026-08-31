#!/usr/bin/env python3
"""Run one ACT group-memory development shard after Sol's pairing clearance.

This runner evaluates one task and one method subset over the frozen ten
state/seed pairs.  It reuses the validated LeRobot ACT preprocessing path and
the repository's sparse candidate scheduler.  The default execution is
blocked until the local protocol records Sol's explicit pairing decision and
selected shared kernel.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
SPARSE_ROOT = REPO_ROOT / "experiments" / "sparse_temporal_ensemble_dev"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SPARSE_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from group_memory_common import RUNNABLE_METHODS, compose_method  # noqa: E402
from sparse_executor import SparseExecutor  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "protocol.json"
PAIRING_PREFIX_LENGTH = 16


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_progress(path: Path | None, value: object) -> None:
    if path is not None:
        atomic_json(path, value)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol["coordination"]["sol_decision"] not in protocol["coordination"]["accepted_pairing_decisions"]:
        raise RuntimeError("BLOCKED_BY_PAIRING_AUDIT: Sol decision is not recorded as accepted")
    if not protocol["shared_kernel"]["status"].startswith("selected_by_sol"):
        raise RuntimeError("BLOCKED_BY_SOL_KERNEL_DECISION: shared kernel is not selected")
    if not protocol["shared_kernel"]["selected_name"]:
        raise RuntimeError("BLOCKED_BY_SOL_KERNEL_DECISION: selected kernel name is empty")
    if not protocol["coordination"]["sol_audit_commit"]:
        raise RuntimeError("BLOCKED_BY_PAIRING_AUDIT: Sol audit commit is not recorded")
    if not protocol["coordination"].get("sol_repaired_rollout_commit"):
        raise RuntimeError("BLOCKED_BY_SOL_REPAIRED_TRIO: repaired h16 trio commit is not recorded")
    return protocol


def validate_protocol_values(protocol: dict[str, Any]) -> tuple[list[int], list[int], int, int, str]:
    states = [int(value) for value in protocol["cohort"]["states"]]
    seeds = [int(value) for value in protocol["cohort"]["environment_seeds"]]
    if states != list(range(10, 20)) or seeds != list(range(2000, 2010)):
        raise RuntimeError("development cohort drifted from frozen states 10..19 / seeds 2000..2009")
    policy = protocol["policy"]
    if int(policy["query_cadence_h"]) != 16 or int(policy["act_prediction_horizon"]) != 100:
        raise RuntimeError("ACT query cadence or prediction horizon drifted")
    kernel = str(protocol["shared_kernel"]["selected_name"])
    if kernel != "dense_equivalent_te" or kernel not in protocol["shared_kernel"]["allowed_names"]:
        raise RuntimeError(f"unknown Sol-selected shared kernel {kernel!r}")
    return states, seeds, 30, 100, kernel


def task_map(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        f"{task['suite']}:task{int(task['task_id'])}": task
        for task in protocol["cohort_task_specs"]
    }


def infer_chunk(observation, env, policy, processors, torch) -> np.ndarray:
    from lerobot.envs.utils import add_envs_task, preprocess_observation
    from lerobot.utils.constants import ACTION

    env_preprocessor, env_postprocessor, preprocessor, postprocessor = processors
    batch = preprocess_observation(copy.deepcopy(observation))
    batch = add_envs_task(env, batch)
    batch = env_preprocessor(batch)
    batch = preprocessor(batch)
    with torch.inference_mode():
        chunk = postprocessor(policy.predict_action_chunk(batch))
        chunk = env_postprocessor({ACTION: chunk})[ACTION]
    result = chunk.detach().cpu().numpy().astype(np.float32, copy=False)
    if result.ndim != 3 or result.shape[0] != 1 or result.shape[1] < 100 or result.shape[2] != 7:
        raise RuntimeError(f"unexpected postprocessed ACT chunk shape: {result.shape}")
    return result[0].copy()


def reset_act_rng(torch: Any, policy_seed: int, environment_seed: int) -> None:
    # Sol's repaired protocol uses the environment seed for Python/NumPy
    # reset state and the frozen ACT seed for torch policy sampling.
    random.seed(int(environment_seed))
    np.random.seed(int(environment_seed))
    torch.manual_seed(int(policy_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(policy_seed))


def extract_success(info: Any, reward: Any) -> bool:
    final_info = info.get("final_info") if isinstance(info, dict) else None
    if isinstance(final_info, dict) and "is_success" in final_info:
        return bool(np.asarray(final_info["is_success"]).reshape(-1)[0])
    values = np.asarray(reward).reshape(-1)
    return bool(len(values) and values[0] > 0)


def flatten_numeric(value: Any, prefix: str = "") -> dict[str, np.ndarray]:
    if isinstance(value, dict):
        result: dict[str, np.ndarray] = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_numeric(value[key], child))
        return result
    if isinstance(value, (list, tuple)):
        array = np.asarray(value)
        if array.dtype != object:
            return {prefix: array.copy()}
        result = {}
        for index, item in enumerate(value):
            result.update(flatten_numeric(item, f"{prefix}[{index}]"))
        return result
    try:
        array = np.asarray(value)
    except Exception:
        return {}
    if array.dtype == object:
        return {}
    return {prefix: array.copy()}


def compare_array_maps(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> dict[str, Any]:
    keys_equal = set(first) == set(second)
    exact = keys_equal
    maximum = 0.0
    for key in sorted(set(first) | set(second)):
        if key not in first or key not in second:
            exact = False
            continue
        a = np.asarray(first[key])
        b = np.asarray(second[key])
        same = a.shape == b.shape and np.array_equal(a, b)
        exact = exact and same
        if a.shape == b.shape and np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number) and a.size:
            maximum = max(maximum, float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
    return {"keys_equal": keys_equal, "exact": bool(exact), "max_absolute_difference": maximum}


def get_sim_state(env) -> np.ndarray:
    return np.asarray(env.envs[0]._env.get_sim_state()).copy()


def run_episode(
    *,
    env,
    policy,
    processors,
    torch,
    task_key: str,
    method: str,
    state_id: int,
    env_seed: int,
    protocol: dict[str, Any],
    max_steps: int,
    capture_pairing_prefix: bool = False,
) -> dict[str, Any]:
    env.envs[0].init_state_id = int(state_id)
    actual_state_id = int(env.envs[0].init_state_id)
    if actual_state_id != int(state_id):
        raise RuntimeError(f"initial-state assignment mismatch: {state_id} vs {actual_state_id}")
    reset_act_rng(
        torch,
        int(protocol["policy"]["policy_rng"]["act_seed"]),
        int(env_seed),
    )
    policy.reset()
    observation, _ = env.reset(seed=[int(env_seed)])
    initial_observation = flatten_numeric(copy.deepcopy(observation)) if capture_pairing_prefix else None
    initial_sim_state = get_sim_state(env) if capture_pairing_prefix else None
    processors = tuple(processors)
    executor = SparseExecutor(
        cadence=16,
        prediction_horizon=100,
        mode="hard",
        coefficient=0.01,
        action_dim=7,
    )
    step_log: list[dict[str, Any]] = []
    query_log: list[dict[str, Any]] = []
    prefix_actions: list[np.ndarray] = []
    prefix_sim_states: list[np.ndarray] = []
    prefix_observations: list[dict[str, np.ndarray]] = []
    initial_chunk = None
    success = False
    completion_step: int | None = None
    done = False

    for target_step in range(int(max_steps)):
        query_latency = None
        if executor.should_query(target_step):
            started = time.perf_counter()

            def query() -> np.ndarray:
                return infer_chunk(observation, env, policy, processors, torch)

            result = executor.step(target_step, query)
            query_latency = time.perf_counter() - started
            query_log.append({"query_physical_step_q": target_step, "latency_seconds": query_latency})
            if target_step == 0 and capture_pairing_prefix:
                initial_chunk = result.candidates.actions[-1].copy()
        else:
            result = executor.step(
                target_step,
                lambda: (_ for _ in ()).throw(RuntimeError("query function called off q=0,16,32 schedule")),
            )

        action, diagnostics = compose_method(
            method,
            result.candidates,
            kernel_name=str(protocol["shared_kernel"]["selected_name"]),
            coefficient=float(protocol["shared_kernel"]["coefficient"]),
            alpha=float(protocol["methods"]["M2_shared_cogact_h16"]["alpha"]),
        )
        action = action.astype(np.float32, copy=False)
        if capture_pairing_prefix and target_step < PAIRING_PREFIX_LENGTH:
            prefix_actions.append(action.copy())
        observation, reward, terminated, truncated, info = env.step(action[None])
        if capture_pairing_prefix and target_step < PAIRING_PREFIX_LENGTH:
            prefix_sim_states.append(get_sim_state(env))
            prefix_observations.append(flatten_numeric(copy.deepcopy(observation)))
        terminated = bool(np.asarray(terminated).reshape(-1)[0])
        truncated = bool(np.asarray(truncated).reshape(-1)[0])
        done = terminated or truncated
        if done:
            success = extract_success(info, reward)
            completion_step = target_step + 1 if success else None
        step_log.append(
            {
                "physical_target_t": target_step,
                "latest_query_q": int(result.latest_query_step),
                "policy_queried_at_t": bool(result.queried),
                "candidate_source_queries": result.candidates.source_query_steps.astype(int).tolist(),
                "candidate_ages": result.candidates.ages.astype(int).tolist(),
                "candidate_count": int(result.candidate_count),
                "shared_weights": diagnostics.get("shared_weights", diagnostics.get("base_weights", np.asarray([]))).astype(float).tolist(),
                "arm_weights": diagnostics["arm_weights"].astype(float).tolist(),
                "gripper_weights": diagnostics["gripper_weights"].astype(float).tolist(),
                "mean_arm_weighted_age": float(diagnostics["arm_weights"] @ result.candidates.ages),
                "mean_gripper_weighted_age": float(diagnostics["gripper_weights"] @ result.candidates.ages),
                "query_latency_seconds": query_latency,
                "chosen_action_7d": action.astype(float).tolist(),
            }
        )
        if done:
            break

    if capture_pairing_prefix and len(prefix_actions) != PAIRING_PREFIX_LENGTH:
        raise RuntimeError(f"{task_key} state {state_id} terminated before t=15 pairing prefix")
    result_record: dict[str, Any] = {
        "task": task_key,
        "method": method,
        "environment_seed": int(env_seed),
        "requested_initial_state_id": int(state_id),
        "actual_initial_state_id": actual_state_id,
        "policy_rng_seed": int(protocol["policy"]["policy_rng"]["act_seed"]),
        "cadence_h": 16,
        "prediction_horizon": 100,
        "success": bool(success),
        "completion_steps": completion_step,
        "environment_steps": len(step_log),
        "policy_queries": len(query_log),
        "query_rate": len(query_log) / float(len(step_log)),
        "query_steps": [int(entry["query_physical_step_q"]) for entry in query_log],
        "mean_candidate_count": float(np.mean([row["candidate_count"] for row in step_log])),
        "mean_arm_weighted_source_age": float(np.mean([row["mean_arm_weighted_age"] for row in step_log])),
        "mean_gripper_weighted_source_age": float(np.mean([row["mean_gripper_weighted_age"] for row in step_log])),
        "step_log": step_log,
        "query_log": query_log,
    }
    if capture_pairing_prefix:
        result_record["pairing_trace"] = {
            "initial_observation": initial_observation,
            "initial_sim_state": initial_sim_state,
            "initial_chunk": initial_chunk,
            "prefix_actions": np.stack(prefix_actions),
            "prefix_sim_states": np.stack(prefix_sim_states),
            "prefix_observations": prefix_observations,
        }
    return result_record


def trace_summary(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    first = reference["pairing_trace"]
    second = candidate["pairing_trace"]
    initial_obs = compare_array_maps(first["initial_observation"], second["initial_observation"])
    post_obs = [
        compare_array_maps(a, b)
        for a, b in zip(first["prefix_observations"], second["prefix_observations"], strict=True)
    ]
    sim = compare_array_maps(
        {"initial": first["initial_sim_state"], "prefix": first["prefix_sim_states"]},
        {"initial": second["initial_sim_state"], "prefix": second["prefix_sim_states"]},
    )
    chunk = compare_array_maps({"chunk": first["initial_chunk"]}, {"chunk": second["initial_chunk"]})
    actions = compare_array_maps({"prefix": first["prefix_actions"]}, {"prefix": second["prefix_actions"]})
    exact = initial_obs["exact"] and sim["exact"] and chunk["exact"] and actions["exact"] and all(row["exact"] for row in post_obs)
    maximum = max(
        initial_obs["max_absolute_difference"],
        sim["max_absolute_difference"],
        chunk["max_absolute_difference"],
        actions["max_absolute_difference"],
        *(row["max_absolute_difference"] for row in post_obs),
    )
    return {
        "candidate_method": candidate["method"],
        "reference_method": reference["method"],
        "initial_observation": initial_obs,
        "initial_and_prefix_simulator_states": sim,
        "initial_predicted_chunk": chunk,
        "common_prefix_actions_t0_t15": actions,
        "post_action_observations_exact": all(row["exact"] for row in post_obs),
        "post_action_observations_max_absolute_difference": max(
            (row["max_absolute_difference"] for row in post_obs), default=0.0
        ),
        "passed": bool(exact),
        "max_absolute_difference": maximum,
    }


def build_runtime(protocol: dict[str, Any], task_key: str, gpu: str):
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    task = task_map(protocol)[task_key]
    checkpoint = Path(task["act_checkpoint"]).resolve()
    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
    policy_cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_cfg.pretrained_path = checkpoint
    if getattr(policy_cfg, "type", None) != "act" or int(policy_cfg.chunk_size) != 100:
        raise RuntimeError("ACT checkpoint/config does not match the frozen H_pred=100 protocol")
    env_config = LiberoEnv(
        task=task["suite"],
        task_ids=[int(task["task_id"])],
        fps=30,
        obs_type="pixels_agent_pos",
        camera_name="agentview_image,robot0_eye_in_hand_image",
        init_states=True,
        observation_width=256,
        observation_height=256,
        control_mode="relative",
    )
    def make_fresh_env(environment_seed: int):
        # Static LIBERO fixture placement is sampled during construction and
        # is not fully represented by the flattened MuJoCo dynamic state.
        # Construct a fresh, identically seeded environment per condition and
        # state, exactly as required by Sol's repaired pairing protocol.
        random.seed(int(environment_seed))
        np.random.seed(int(environment_seed))
        return make_env(env_config, n_envs=1, use_async_envs=False)[task["suite"]][int(task["task_id"])]

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
    prototype = make_fresh_env(0)
    try:
        max_steps = int(np.asarray(prototype.call("_max_episode_steps")).reshape(-1)[0])
    finally:
        prototype.close()
    return make_fresh_env, policy, (env_preprocessor, env_postprocessor, preprocessor, postprocessor), torch, max_steps


def run_pairing_smoke(protocol: dict[str, Any], task_key: str, methods: list[str], gpu: str, output: Path) -> None:
    states, seeds, _, _, _ = validate_protocol_values(protocol)
    make_fresh_env, policy, processors, torch, _ = build_runtime(protocol, task_key, gpu)
    traces: dict[str, dict[str, Any]] = {}
    for method in methods:
        env = make_fresh_env(seeds[0])
        try:
            traces[method] = run_episode(
                env=env,
                policy=policy,
                processors=processors,
                torch=torch,
                task_key=task_key,
                method=method,
                state_id=states[0],
                env_seed=seeds[0],
                protocol=protocol,
                max_steps=PAIRING_PREFIX_LENGTH,
                capture_pairing_prefix=True,
            )
        finally:
            env.close()
    reference = traces[methods[0]]
    comparisons = [trace_summary(reference, traces[method]) for method in methods[1:]]
    passed = all(row["passed"] for row in comparisons)
    result = {
        "status": "paired_common_prefix_pass" if passed else "paired_common_prefix_fail",
        "task": task_key,
        "state_id": states[0],
        "environment_seed": seeds[0],
        "methods": methods,
        "reference_method": methods[0],
        "comparisons": comparisons,
        "passed": passed,
    }
    atomic_json(output, result)
    if not passed:
        raise RuntimeError("strict pairing smoke failed; full rollout is not permitted")


def summarize_method(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [bool(row["success"]) for row in episodes]
    steps = sum(int(row["environment_steps"]) for row in episodes)
    queries = sum(int(row["policy_queries"]) for row in episodes)
    return {
        "successes": successes,
        "success_count": int(sum(successes)),
        "episodes": len(episodes),
        "success_rate": float(np.mean(successes)),
        "policy_queries": queries,
        "environment_steps": steps,
        "query_rate": queries / float(steps),
        "mean_candidate_count": float(np.mean([row["mean_candidate_count"] for row in episodes])),
        "mean_arm_weighted_source_age": float(np.mean([row["mean_arm_weighted_source_age"] for row in episodes])),
        "mean_gripper_weighted_source_age": float(np.mean([row["mean_gripper_weighted_source_age"] for row in episodes])),
        "completion_steps_successful": [row["completion_steps"] for row in episodes if row["completion_steps"] is not None],
        "episodes_detail": episodes,
    }


def semantic_smoke() -> None:
    """CPU-only test of scheduler, candidate alignment, and method outputs."""

    def fake_chunk(source: int) -> np.ndarray:
        return np.asarray(
            [[1000.0 * source + 10.0 * offset + dim for dim in range(7)] for offset in range(100)],
            dtype=np.float64,
        )

    protocol = {
        "shared_kernel": {"selected_name": "physical_age_te", "coefficient": 0.01},
        "methods": {"M2_shared_cogact_h16": {"alpha": 0.3}},
    }
    for method in RUNNABLE_METHODS:
        executor = SparseExecutor(cadence=16, prediction_horizon=100, mode="hard", coefficient=0.01)
        calls: list[int] = []
        for target in range(33):
            result = executor.step(target, lambda target=target: calls.append(target) or fake_chunk(target))
            action, diagnostics = compose_method(
                method,
                result.candidates,
                kernel_name="physical_age_te",
                coefficient=0.01,
                alpha=0.3,
            )
            assert action.shape == (7,) and np.isfinite(action).all()
            assert np.isclose(diagnostics["arm_weights"].sum(), 1.0)
            assert np.isclose(diagnostics["gripper_weights"].sum(), 1.0)
        assert calls == [0, 16, 32]
    print(json.dumps({"status": "act_group_memory_cpu_semantic_smoke_pass", "methods": list(RUNNABLE_METHODS)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--task")
    parser.add_argument("--methods", default=",".join(RUNNABLE_METHODS))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--pairing-smoke", action="store_true")
    parser.add_argument("--pairing-audit", type=Path)
    parser.add_argument("--semantic-smoke", action="store_true")
    args = parser.parse_args()
    if args.semantic_smoke:
        semantic_smoke()
        return
    protocol = load_protocol(args.protocol)
    if args.task not in task_map(protocol):
        raise SystemExit(f"task must be one of {sorted(task_map(protocol))}")
    methods = [value for value in args.methods.split(",") if value]
    if not methods or any(value not in RUNNABLE_METHODS for value in methods):
        raise SystemExit(f"methods must be a nonempty subset of {RUNNABLE_METHODS}")
    if args.pairing_smoke:
        if args.output is None:
            raise SystemExit("--output is required for --pairing-smoke")
        run_pairing_smoke(protocol, args.task, methods, args.gpu, args.output)
        return
    if args.output is None or args.pairing_audit is None:
        raise SystemExit("full rollout requires --output and a passed --pairing-audit artifact")
    pairing = json.loads(args.pairing_audit.read_text())
    if pairing.get("status") != "paired_common_prefix_pass" or pairing.get("passed") is not True:
        raise RuntimeError("BLOCKED_BY_PAIRING_AUDIT: pairing artifact is not a pass")
    make_fresh_env, policy, processors, torch, max_steps = build_runtime(protocol, args.task, args.gpu)
    states, seeds, _, _, kernel = validate_protocol_values(protocol)
    started = time.time()
    output = {
        "status": "running",
        "policy": "ACT",
        "protocol": str(args.protocol.resolve()),
        "sol_audit_commit": protocol["coordination"]["sol_audit_commit"],
        "sol_decision": protocol["coordination"]["sol_decision"],
        "shared_kernel": kernel,
        "task": args.task,
        "methods": methods,
        "pairing_audit": str(args.pairing_audit.resolve()),
        "started_at": started,
        "methods_result": {},
    }
    progress = {"pid": os.getpid(), "task": args.task, "completed_methods": 0, "completed_episodes": 0}
    write_progress(args.progress_file, progress)
    try:
        for method in methods:
            episodes = []
            progress["current_method"] = method
            write_progress(args.progress_file, progress)
            for state_id, env_seed in zip(states, seeds, strict=True):
                env = make_fresh_env(env_seed)
                try:
                    episode = run_episode(
                        env=env,
                        policy=policy,
                        processors=processors,
                        torch=torch,
                        task_key=args.task,
                        method=method,
                        state_id=state_id,
                        env_seed=env_seed,
                        protocol=protocol,
                        max_steps=max_steps,
                    )
                finally:
                    env.close()
                episodes.append(episode)
                progress["completed_episodes"] += 1
                write_progress(args.progress_file, progress)
            output["methods_result"][method] = summarize_method(episodes)
            progress["completed_methods"] += 1
            atomic_json(args.output, output)
        output["status"] = "complete"
        output["finished_at"] = time.time()
        progress["finished_at"] = output["finished_at"]
        write_progress(args.progress_file, progress)
        atomic_json(args.output, output)
    finally:
        # Every episode owns and closes its fresh environment above.
        pass
    print(json.dumps({"status": output["status"], "task": args.task, "methods": methods}, indent=2))


if __name__ == "__main__":
    main()
