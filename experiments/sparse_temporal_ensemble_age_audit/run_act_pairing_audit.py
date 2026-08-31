#!/usr/bin/env python3
"""Strict ACT initialization and common-prefix pairing audit.

This deliberately runs two separate resets for each frozen state/seed pair.
It stops at t=15, before the first h16 re-query, so hard execution and the
historical candidate-index ensemble each have exactly one cached chunk.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
SPARSE_ROOT = REPO_ROOT / "experiments" / "sparse_temporal_ensemble_dev"
sys.path.insert(0, str(SPARSE_ROOT))
sys.path.insert(0, str(ROOT))

from dense_equivalent_executor import DenseEquivalentSparseExecutor  # noqa: E402
from sparse_executor import SparseExecutor  # noqa: E402


def to_numpy(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value.copy()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, (bool, int, float, np.number)):
        return np.asarray(value)
    return None


def flatten_values(value: Any, prefix: str = "") -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    literals: dict[str, Any] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            child_arrays, child_literals = flatten_values(value[key], child)
            arrays.update(child_arrays)
            literals.update(child_literals)
        return arrays, literals
    if isinstance(value, (list, tuple)):
        array = to_numpy(value)
        if array is not None and array.dtype != object:
            arrays[prefix] = array
        else:
            for index, item in enumerate(value):
                child_arrays, child_literals = flatten_values(item, f"{prefix}[{index}]")
                arrays.update(child_arrays)
                literals.update(child_literals)
        return arrays, literals
    array = to_numpy(value)
    if array is not None:
        arrays[prefix] = array
    elif isinstance(value, (str, type(None))):
        literals[prefix] = value
    else:
        literals[prefix] = repr(value)
    return arrays, literals


def compare_arrays(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> dict:
    keys_equal = set(first) == set(second)
    rows = {}
    all_exact = keys_equal
    maximum = 0.0
    for key in sorted(set(first) | set(second)):
        if key not in first or key not in second:
            rows[key] = {"present_in_both": False}
            all_exact = False
            continue
        a = np.asarray(first[key])
        b = np.asarray(second[key])
        same_shape = a.shape == b.shape
        exact = bool(same_shape and np.array_equal(a, b))
        if same_shape and np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
            max_abs = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
            maximum = max(maximum, max_abs)
        else:
            max_abs = None
        rows[key] = {
            "present_in_both": True,
            "shape": list(a.shape),
            "dtype_hard": str(a.dtype),
            "dtype_te": str(b.dtype),
            "exact_array_equality": exact,
            "max_absolute_difference": max_abs,
        }
        all_exact = all_exact and exact
    return {
        "keys_equal": keys_equal,
        "all_arrays_exact": bool(all_exact),
        "max_absolute_difference": float(maximum),
        "arrays": rows,
    }


def subset_by_key(arrays: dict[str, np.ndarray], words: tuple[str, ...], invert: bool = False) -> dict[str, np.ndarray]:
    selected = {}
    for key, value in arrays.items():
        match = any(word in key.lower() for word in words)
        if match != invert:
            selected[key] = value
    return selected


def make_processed_input(observation, env, processors):
    from lerobot.envs.utils import add_envs_task, preprocess_observation

    env_preprocessor, _, preprocessor, _ = processors
    batch = preprocess_observation(copy.deepcopy(observation))
    batch = add_envs_task(env, batch)
    task_augmented = copy.deepcopy(batch)
    batch = env_preprocessor(batch)
    after_env = copy.deepcopy(batch)
    batch = preprocessor(batch)
    final_batch = copy.deepcopy(batch)
    return task_augmented, after_env, final_batch


def predict_from_processed(batch, policy, processors, torch) -> np.ndarray:
    from lerobot.utils.constants import ACTION

    _, env_postprocessor, _, postprocessor = processors
    with torch.inference_mode():
        chunk = postprocessor(policy.predict_action_chunk(batch))
        chunk = env_postprocessor({ACTION: chunk})[ACTION]
    result = chunk.detach().cpu().numpy().astype(np.float32, copy=False)
    if result.shape != (1, 100, 7):
        raise RuntimeError(f"unexpected postprocessed ACT chunk shape: {result.shape}")
    return result[0].copy()


def capture_condition(*, env, policy, processors, torch, condition: str, state_id: int, env_seed: int) -> dict:
    env.envs[0].init_state_id = int(state_id)
    selected_state_id = int(env.envs[0].init_state_id)
    random.seed(int(env_seed))
    np.random.seed(int(env_seed))
    torch.manual_seed(424242)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(424242)
    policy.reset()
    observation, _ = env.reset(seed=[int(env_seed)])
    post_reset_counter = int(env.envs[0].init_state_id)
    sim_state = np.asarray(env.envs[0]._env.get_sim_state()).copy()

    raw_arrays, raw_literals = flatten_values(observation)
    task_augmented, after_env, processed = make_processed_input(observation, env, processors)
    task_arrays, task_literals = flatten_values(task_augmented)
    env_processed_arrays, env_processed_literals = flatten_values(after_env)
    processed_arrays, processed_literals = flatten_values(processed)
    chunk = predict_from_processed(processed, policy, processors, torch)

    if condition == "dense_equivalent_te_h16":
        executor = DenseEquivalentSparseExecutor(
            cadence=16,
            prediction_horizon=100,
            mode="dense_equivalent_te",
            coefficient=0.01,
            action_dim=7,
        )
    else:
        mode = "hard" if condition == "hard_h16" else "sparse_te"
        executor = SparseExecutor(
            cadence=16,
            prediction_horizon=100,
            mode=mode,
            coefficient=0.01,
            action_dim=7,
        )
    actions = []
    post_step_sim_states = []
    post_step_observations = []
    for target in range(16):
        if target == 0:
            result = executor.step(target, lambda: chunk)
        else:
            result = executor.step(
                target,
                lambda: (_ for _ in ()).throw(RuntimeError("unexpected query before t=16")),
            )
        if result.candidate_count != 1:
            raise RuntimeError(f"{condition} t={target} has {result.candidate_count} candidates, expected one")
        action = result.action.astype(np.float32, copy=False)
        actions.append(action.copy())
        observation, _, terminated, truncated, _ = env.step(action[None])
        if bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]):
            raise RuntimeError(f"{condition} terminated before the h16 common prefix ended at t={target}")
        post_step_sim_states.append(np.asarray(env.envs[0]._env.get_sim_state()).copy())
        arrays, _ = flatten_values(observation)
        post_step_observations.append(arrays)

    return {
        "condition": condition,
        "requested_state_id": int(state_id),
        "selected_state_id_before_reset": selected_state_id,
        "state_counter_after_reset": post_reset_counter,
        "environment_seed": int(env_seed),
        "policy_rng_seed": 424242,
        "sim_state": sim_state,
        "raw_arrays": raw_arrays,
        "raw_literals": raw_literals,
        "task_augmented_arrays": task_arrays,
        "task_augmented_literals": task_literals,
        "env_processed_arrays": env_processed_arrays,
        "env_processed_literals": env_processed_literals,
        "processed_arrays": processed_arrays,
        "processed_literals": processed_literals,
        "chunk": chunk,
        "actions": np.stack(actions),
        "post_step_sim_states": np.stack(post_step_sim_states),
        "post_step_observations": post_step_observations,
        "query_steps": list(executor.query_steps),
    }


def compare_conditions(hard: dict, te: dict) -> dict:
    raw_images_hard = subset_by_key(hard["raw_arrays"], ("pixel", "image"))
    raw_images_te = subset_by_key(te["raw_arrays"], ("pixel", "image"))
    raw_low_hard = subset_by_key(hard["raw_arrays"], ("pixel", "image"), invert=True)
    raw_low_te = subset_by_key(te["raw_arrays"], ("pixel", "image"), invert=True)
    processed_images_hard = subset_by_key(hard["processed_arrays"], ("pixel", "image"))
    processed_images_te = subset_by_key(te["processed_arrays"], ("pixel", "image"))
    processed_state_hard = subset_by_key(hard["processed_arrays"], ("state", "proprio", "robot"))
    processed_state_te = subset_by_key(te["processed_arrays"], ("state", "proprio", "robot"))
    processed_task_hard = subset_by_key(hard["processed_arrays"], ("task", "language", "token"))
    processed_task_te = subset_by_key(te["processed_arrays"], ("task", "language", "token"))

    sim = compare_arrays({"sim_state": hard["sim_state"]}, {"sim_state": te["sim_state"]})
    raw_low = compare_arrays(raw_low_hard, raw_low_te)
    raw_images = compare_arrays(raw_images_hard, raw_images_te)
    processed_all = compare_arrays(hard["processed_arrays"], te["processed_arrays"])
    processed_state = compare_arrays(processed_state_hard, processed_state_te)
    processed_images = compare_arrays(processed_images_hard, processed_images_te)
    processed_task = compare_arrays(processed_task_hard, processed_task_te)
    chunk = compare_arrays({"A0": hard["chunk"]}, {"A0": te["chunk"]})
    actions = compare_arrays({"actions_t0_t15": hard["actions"]}, {"actions_t0_t15": te["actions"]})
    post_sim = compare_arrays(
        {"post_step_sim_states": hard["post_step_sim_states"]},
        {"post_step_sim_states": te["post_step_sim_states"]},
    )
    post_observation_max = 0.0
    post_observation_exact = True
    for hard_obs, te_obs in zip(hard["post_step_observations"], te["post_step_observations"]):
        comparison = compare_arrays(hard_obs, te_obs)
        post_observation_exact = post_observation_exact and comparison["all_arrays_exact"]
        post_observation_max = max(post_observation_max, comparison["max_absolute_difference"])

    task_literals_equal = (
        hard["task_augmented_literals"] == te["task_augmented_literals"]
        and hard["env_processed_literals"] == te["env_processed_literals"]
        and hard["processed_literals"] == te["processed_literals"]
    )
    identity = {
        "requested_state_id_equal": hard["requested_state_id"] == te["requested_state_id"],
        "selected_state_id_equal": hard["selected_state_id_before_reset"] == te["selected_state_id_before_reset"],
        "environment_seed_equal": hard["environment_seed"] == te["environment_seed"],
        "policy_rng_seed_equal": hard["policy_rng_seed"] == te["policy_rng_seed"],
        "query_steps_hard": hard["query_steps"],
        "query_steps_te": te["query_steps"],
    }
    passed = all(
        [
            *[bool(value) for key, value in identity.items() if key.endswith("_equal")],
            hard["query_steps"] == te["query_steps"] == [0],
            sim["all_arrays_exact"],
            raw_low["all_arrays_exact"],
            raw_images["all_arrays_exact"],
            processed_all["all_arrays_exact"],
            task_literals_equal,
            chunk["max_absolute_difference"] <= 1e-6,
            actions["max_absolute_difference"] <= 1e-6,
            post_sim["all_arrays_exact"],
            post_observation_exact,
        ]
    )
    return {
        "passed": bool(passed),
        "identity": identity,
        "B1_initial_simulator_state": sim,
        "B1_initial_low_dimensional_observation": raw_low,
        "B2_initial_camera_observations": raw_images,
        "B3_processed_policy_input_all_numeric": processed_all,
        "B3_processed_state_proprio": processed_state,
        "B3_processed_images": processed_images,
        "B3_processed_task_numeric": processed_task,
        "B3_task_conditioning_literals_equal": task_literals_equal,
        "B3_task_conditioning_literals_hard": hard["task_augmented_literals"],
        "B3_task_conditioning_literals_te": te["task_augmented_literals"],
        "B4_initial_predicted_chunk": chunk,
        "B5_common_prefix_actions": actions,
        "B5_post_action_simulator_states": post_sim,
        "B5_post_action_observations_exact": bool(post_observation_exact),
        "B5_post_action_observations_max_absolute_difference": float(post_observation_max),
    }


def save_capture(path: Path, capture: dict) -> None:
    arrays = {
        "initial_sim_state": capture["sim_state"],
        "initial_chunk_A0": capture["chunk"],
        "actions_t0_t15": capture["actions"],
        "post_step_sim_states": capture["post_step_sim_states"],
    }
    for name, value in capture["raw_arrays"].items():
        arrays[f"raw__{name}"] = value
    for name, value in capture["processed_arrays"].items():
        arrays[f"processed__{name}"] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", type=Path, default=ROOT / "pairing_audit_task10_fresh_env_trio" / "summary.json")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    audit = protocol["pairing_audit"]
    task = audit["task"]
    checkpoint = Path(task["checkpoint"]).resolve()
    state_ids = [int(value) for value in audit["initial_state_ids"]]
    seeds = [int(value) for value in audit["environment_seeds"]]
    if len(state_ids) != 3 or len(seeds) != 3 or len(state_ids) != len(seeds):
        raise RuntimeError("pairing audit requires exactly three frozen state/seed pairs")
    if audit["common_prefix_targets"] != list(range(16)):
        raise RuntimeError("pairing audit must stop before the first h16 re-query")

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
    if getattr(policy_cfg, "type", None) != "act" or int(policy_cfg.chunk_size) != 100:
        raise RuntimeError("pairing audit requires the frozen ACT H_pred=100 checkpoint")
    environment = protocol["environment"]
    env_config = LiberoEnv(
        task=task["suite"],
        task_ids=[int(task["task_id"])],
        fps=int(environment["fps"]),
        obs_type=environment["obs_type"],
        camera_name=environment["camera_name"],
        init_states=True,
        observation_width=int(environment["observation_width"]),
        observation_height=int(environment["observation_height"]),
        control_mode=environment["control_mode"],
    )
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

    output = {
        "protocol": str(args.protocol.resolve()),
        "runtime": {
            "python_executable": sys.executable,
            "lerobot": "0.4.4",
            "torch": str(torch.__version__),
            "cuda_visible_devices": str(args.gpu),
        },
        "task": f"{task['suite']}:task{int(task['task_id'])}",
        "checkpoint": str(checkpoint),
        "states": [],
    }
    for state_id, env_seed in zip(state_ids, seeds):
        captures = {}
        for condition in audit["conditions"]:
            random.seed(int(env_seed))
            np.random.seed(int(env_seed))
            env = make_env(env_config, n_envs=1, use_async_envs=False)[task["suite"]][int(task["task_id"])]
            try:
                capture = capture_condition(
                    env=env,
                    policy=policy,
                    processors=processors,
                    torch=torch,
                    condition=condition,
                    state_id=state_id,
                    env_seed=env_seed,
                )
                captures[condition] = capture
                save_capture(
                    args.output.parent / f"state{state_id}_{condition}.npz",
                    capture,
                )
            finally:
                env.close()
        comparisons = {
            method: compare_conditions(captures["hard_h16"], captures[method])
            for method in audit["conditions"][1:]
        }
        output["states"].append(
            {
                "state_id": state_id,
                "environment_seed": env_seed,
                "passed": all(comparison["passed"] for comparison in comparisons.values()),
                "comparisons_to_hard": comparisons,
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")

    output["passed"] = all(state["passed"] for state in output["states"])
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"passed": output["passed"], "output": str(args.output), "states": state_ids}))
    if not output["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
