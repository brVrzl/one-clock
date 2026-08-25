#!/usr/bin/env python3
"""Outcome-blind layered determinism diagnostics for Gate-4A2 prefixes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("__GL_THREADED_OPTIMIZATIONS", "0")
os.environ.setdefault("__GL_YIELD", "NOTHING")

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path("/home/wjq/workspace/one-clock")
LEROBOT_ROOT = Path("/home/wjq/workspace/upstreams/lerobot")
CHECKPOINT = Path("/home/wjq/checkpoints/ishandotsh_act_libero_spatial_test")
CONFIG = ROOT / "configs/gate4a2_libero_spatial.yaml"
HISTORICAL_TRACE_ROOT = (
    SOURCE_ROOT / "experiments/gate4a2_spatial_act_generalization/episodes"
)
CANARY_OUTPUT = ROOT / "research/audit_outputs/gate4a2_determinism_canary.json"
SELECTED_STATES = (1, 13, 15, 19, 21, 24, 31, 37, 40, 47)
METHODS_ABC = ("A_NEWEST", "B_FULL_OLD20", "C_ASYMMETRIC_FO20")
OUTCOME_KEYS = frozenset(
    {
        "success",
        "is_success",
        "reward",
        "failure_category",
        "terminated",
        "truncated",
    }
)

sys.path[:0] = [
    str(ROOT),
    str(ROOT / "src"),
    str(ROOT / "research/audit_tools"),
    str(LEROBOT_ROOT / "src"),
]

from gate4a2_rollout import configure_determinism, make_env_and_policy  # noqa: E402
from scripts.run_libero_gate0 import (  # noqa: E402
    load_config,
    prepare_policy_observation,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def array(value: Any) -> np.ndarray:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def array_fingerprint(value: Any) -> str:
    item = np.ascontiguousarray(array(value))
    digest = hashlib.sha256()
    digest.update(str(item.dtype).encode())
    digest.update(json.dumps(list(item.shape), separators=(",", ":")).encode())
    digest.update(item.tobytes())
    return digest.hexdigest()


def tree_arrays(value: Any, prefix: str = "") -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(tree_arrays(value[key], path))
    elif isinstance(value, (list, tuple)) and value and not np.isscalar(value[0]):
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]"
            result.update(tree_arrays(item, path))
    else:
        result[prefix] = np.ascontiguousarray(array(value)).copy()
    return result


def tree_fingerprint(value: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for path in sorted(value):
        digest.update(path.encode())
        digest.update(array_fingerprint(value[path]).encode())
    return digest.hexdigest()


def first_index(left: np.ndarray, right: np.ndarray) -> list[int] | None:
    if left.shape != right.shape:
        return None
    differing = np.argwhere(left != right)
    return None if not len(differing) else [int(index) for index in differing[0]]


def compare_trees(repeats: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    normalized = []
    for repeat in repeats:
        flattened: dict[str, np.ndarray] = {}
        for path, value in repeat.items():
            if isinstance(value, np.ndarray):
                flattened[path] = value
            else:
                flattened.update(tree_arrays(value, path))
        normalized.append(flattened)
    paths = sorted(set.union(*(set(repeat) for repeat in normalized)))
    result: dict[str, Any] = {
        "exact_equal": True,
        "max_abs_difference": 0.0,
        "first_differing_path": None,
        "first_differing_index": None,
    }
    for path in paths:
        if any(path not in repeat for repeat in normalized):
            result["exact_equal"] = False
            if result["first_differing_path"] is None:
                result["first_differing_path"] = path
            continue
        reference = normalized[0][path]
        for other_repeat in normalized[1:]:
            other = other_repeat[path]
            if reference.shape != other.shape or reference.dtype != other.dtype:
                result["exact_equal"] = False
                if result["first_differing_path"] is None:
                    result["first_differing_path"] = path
                continue
            if reference.dtype.kind in "biufc" and reference.size:
                difference = float(
                    np.max(np.abs(reference.astype(np.float64) - other.astype(np.float64)))
                )
                result["max_abs_difference"] = max(
                    float(result["max_abs_difference"]), difference
                )
            if not np.array_equal(reference, other):
                result["exact_equal"] = False
                if result["first_differing_path"] is None:
                    result["first_differing_path"] = path
                    result["first_differing_index"] = first_index(reference, other)
    return result


def seed_all(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_fingerprints() -> dict[str, str]:
    import torch

    result = {
        "python": sha256_bytes(pickle.dumps(random.getstate())),
        "numpy": sha256_bytes(pickle.dumps(np.random.get_state())),
        "torch_cpu": array_fingerprint(torch.random.get_rng_state()),
    }
    if torch.cuda.is_available():
        for index, state in enumerate(torch.cuda.get_rng_state_all()):
            result[f"torch_cuda_{index}"] = array_fingerprint(state)
    return result


def drop_outcome_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Drop outcome-bearing keys while JSON objects are being constructed."""

    return {key: value for key, value in pairs if key not in OUTCOME_KEYS}


def assert_outcome_sealed(value: Any) -> None:
    if isinstance(value, dict):
        forbidden = OUTCOME_KEYS.intersection(value)
        if forbidden:
            raise RuntimeError(f"sealed trace reader returned outcome keys: {sorted(forbidden)}")
        for child in value.values():
            assert_outcome_sealed(child)
    elif isinstance(value, list):
        for child in value:
            assert_outcome_sealed(child)


def read_historical_prefix(path: Path) -> dict[str, Any]:
    """Return only identity and t=0..19 action fields, never outcome fields."""

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=drop_outcome_pairs)
    assert_outcome_sealed(payload)
    run = payload["run"]
    steps = [
        {
            "step": int(record["step"]),
            "fresh_action": np.asarray(record["fresh_action"], dtype=np.float64),
            "executed_action": np.asarray(record["action"], dtype=np.float64),
        }
        for record in payload["steps"][:20]
    ]
    state_hash = payload["summary"]["initial_state_vector_sha256"]
    result = {
        "task_id": int(run["task_id"]),
        "state_id": int(run["state_id"]),
        "method": str(run["method"]),
        "episode_seed": int(run["episode_seed"]),
        "initial_state_vector_sha256": state_hash,
        "steps": steps,
    }
    assert_outcome_sealed(result)
    return result


def historical_prefix_audit() -> dict[str, Any]:
    blocks = [(4, state) for state in SELECTED_STATES]
    blocks += [(3, 1), (3, 13), (0, 1), (5, 1)]
    rows = []
    for task_id, state_id in blocks:
        traces = {
            method: read_historical_prefix(
                HISTORICAL_TRACE_ROOT
                / f"task_{task_id:02d}"
                / f"state_{state_id:02d}"
                / f"{method}.json.gz"
            )
            for method in METHODS_ABC
        }
        state_hashes = {trace["initial_state_vector_sha256"] for trace in traces.values()}
        seeds = {trace["episode_seed"] for trace in traces.values()}
        if len(state_hashes) != 1 or len(seeds) != 1:
            raise RuntimeError("historical L0 identity differs within a paired block")
        first_step = None
        max_fresh = 0.0
        max_executed = 0.0
        for step in range(20):
            fresh = [traces[method]["steps"][step]["fresh_action"] for method in METHODS_ABC]
            executed = [
                traces[method]["steps"][step]["executed_action"] for method in METHODS_ABC
            ]
            step_fresh = max(
                float(np.max(np.abs(fresh[left] - fresh[right])))
                for left, right in ((0, 1), (0, 2), (1, 2))
            )
            step_executed = max(
                float(np.max(np.abs(executed[left] - executed[right])))
                for left, right in ((0, 1), (0, 2), (1, 2))
            )
            max_fresh = max(max_fresh, step_fresh)
            max_executed = max(max_executed, step_executed)
            if first_step is None and (step_fresh != 0.0 or step_executed != 0.0):
                first_step = step
        rows.append(
            {
                "task_id": task_id,
                "state_id": state_id,
                "control": not (task_id == 4 or (task_id == 3 and state_id == 1)),
                "first_divergent_step": first_step,
                "earliest_stored_divergent_layer": None if first_step is None else "L5",
                "L0_exact": True,
                "L1_stored": False,
                "L2_stored": False,
                "L3_stored": False,
                "L4_stored": False,
                "L5_complete_chunk_stored": False,
                "L5_fresh_action_stored": True,
                "L6_executed_action_stored": True,
                "max_raw_state_difference": None,
                "max_image_difference": None,
                "max_processed_input_difference": None,
                "max_raw_chunk_difference": None,
                "max_postprocessed_fresh_action_difference": max_fresh,
                "max_executed_action_difference": max_executed,
            }
        )
    return {
        "reader": "JSON object_pairs_hook drops all outcome-bearing keys before return",
        "forbidden_keys": sorted(OUTCOME_KEYS),
        "layers_stored": {
            "L0": "state ID and registered vector SHA256",
            "L1": False,
            "L2": False,
            "L3": False,
            "L4": False,
            "L5": "fresh action only, not the complete postprocessed chunk",
            "L6": "executed action",
        },
        "rows": rows,
    }


def make_fresh_env(config: dict[str, Any], task_id: int) -> Any:
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv

    suite_name = str(config["task_suite"])
    suite = benchmark.get_benchmark_dict()[suite_name]()
    return LiberoEnv(
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


def step_without_outcome(env: Any, action: np.ndarray) -> dict[str, Any]:
    """Advance controller/physics and observations without reward or success code."""

    inner = env._env.env
    inner.timestep += 1
    policy_step = True
    for _ in range(int(inner.control_timestep / inner.model_timestep)):
        inner.sim.forward()
        inner._pre_action(action, policy_step)
        inner.sim.step()
        inner._update_observables()
        policy_step = False
    inner.cur_time += inner.control_timestep
    raw_observation = inner._get_observations()
    return env._format_raw_obs(raw_observation)


def reset_without_outcome(env: Any, state_id: int, seed: int) -> dict[str, Any]:
    """Hard reset and restore an official state without invoking success/reward."""

    from lerobot.envs.libero import get_libero_dummy_action

    env._ensure_env()
    if os.environ.get("GATE4A2_GL_FINISH") == "1":
        render_context = env._env.sim._render_context_offscreen
        if not getattr(render_context, "_gate4a2_gl_finish_wrapped", False):
            from OpenGL.GL import glFinish

            original_render = render_context.render

            def render_then_finish(*args: Any, **kwargs: Any) -> Any:
                result = original_render(*args, **kwargs)
                glFinish()
                return result

            render_context.render = render_then_finish
            render_context._gate4a2_gl_finish_wrapped = True
    env._env.seed(seed)
    env._env.reset()
    init_state = env._init_states[state_id % len(env._init_states)]
    env._env.set_state(init_state)
    env._env.env.sim.forward()
    env._env._post_process()
    env._env._update_observables(force=True)
    raw_observation = env._env.env._get_observations()
    observation = env._format_raw_obs(raw_observation)
    for _ in range(env.num_steps_wait):
        observation = step_without_outcome(
            env, np.asarray(get_libero_dummy_action(), dtype=np.float32)
        )
    for robot in env._env.robots:
        robot.controller.use_delta = env.control_mode == "relative"
    return observation


def simulator_tree(env: Any) -> dict[str, np.ndarray]:
    import torch

    sim = env._env.sim
    result = {"sim_state": np.asarray(sim.get_state().flatten()).copy()}
    for owner_name, owner in (("data", sim.data), ("model", sim.model)):
        for name in dir(owner):
            if name.startswith("_"):
                continue
            try:
                value = getattr(owner, name)
            except Exception:
                continue
            if isinstance(value, (np.ndarray, np.number, int, float, bool)):
                result[f"{owner_name}.{name}"] = np.ascontiguousarray(
                    np.asarray(value)
                ).copy()
    controller = env._env.robots[0].controller
    for name, value in vars(controller).items():
        if isinstance(value, (np.ndarray, np.number, int, float, bool, torch.Tensor)):
            result[f"controller.{name}"] = np.ascontiguousarray(array(value)).copy()
        elif isinstance(value, Mapping):
            result.update(tree_arrays(value, f"controller.{name}"))
    return result


def simulator_core_tree(env: Any) -> dict[str, np.ndarray]:
    """Persistent simulator state, excluding MuJoCo's uninitialized scratch buffers."""

    sim = env._env.sim
    return {
        "sim_state": np.asarray(sim.get_state().flatten()).copy(),
        "data.qpos": np.asarray(sim.data.qpos).copy(),
        "data.qvel": np.asarray(sim.data.qvel).copy(),
        "data.act": np.asarray(sim.data.act).copy(),
        "data.ctrl": np.asarray(sim.data.ctrl).copy(),
        "data.mocap_pos": np.asarray(sim.data.mocap_pos).copy(),
        "data.mocap_quat": np.asarray(sim.data.mocap_quat).copy(),
        "model.body_pos": np.asarray(sim.model.body_pos).copy(),
        "model.body_quat": np.asarray(sim.model.body_quat).copy(),
    }


def query_layers(
    observation: dict[str, Any],
    policy: Any,
    policy_preprocessor: Any,
    policy_postprocessor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    import torch
    from lerobot.utils.constants import ACTION

    processed = prepare_policy_observation(
        observation, env_preprocessor, policy_preprocessor
    )
    with torch.inference_mode():
        raw_chunk = policy.predict_action_chunk(processed)
        postprocessed = policy_postprocessor(raw_chunk)
    postprocessed = env_postprocessor({ACTION: postprocessed})[ACTION]
    fresh = postprocessed[0, 0].detach().cpu().numpy().astype(np.float32, copy=True)
    policy_inputs = {
        key: value for key, value in processed.items() if str(key).startswith("observation.")
    }
    layers = {
        "L2": tree_arrays(observation),
        "L3": tree_arrays(policy_inputs),
        "L4": {"raw_chunk": np.ascontiguousarray(array(raw_chunk)).copy()},
        "L5": {"postprocessed_chunk": np.ascontiguousarray(array(postprocessed)).copy()},
        "L6": {"fresh_action": fresh.copy()},
    }
    return layers, fresh


def diagnostic_prefix(
    *,
    config: dict[str, Any],
    policy_components: tuple[Any, ...],
    task_id: int,
    state_id: int,
    seed: int,
    steps: int = 20,
    corrected_fresh_environment: bool,
    reused_env: Any | None = None,
    full_simulator_inventory: bool = True,
) -> list[dict[str, dict[str, np.ndarray]]]:
    policy, policy_pre, policy_post, env_pre, env_post = policy_components[:5]
    seed_all(seed)
    env = make_fresh_env(config, task_id) if corrected_fresh_environment else reused_env
    if env is None:
        raise ValueError("original-order diagnostic requires a reused environment")
    try:
        observation = reset_without_outcome(env, state_id, seed)
        policy.reset()
        records = []
        for _ in range(steps):
            layers, fresh = query_layers(
                observation,
                policy,
                policy_pre,
                policy_post,
                env_pre,
                env_post,
            )
            records.append(
                {
                    "L1": simulator_tree(env)
                    if full_simulator_inventory
                    else simulator_core_tree(env),
                    **layers,
                }
            )
            observation = step_without_outcome(env, fresh)
        return records
    finally:
        if corrected_fresh_environment:
            env.close()


def compare_layered_repeats(
    repeats: list[list[dict[str, dict[str, np.ndarray]]]],
) -> dict[str, Any]:
    layers = ("L1", "L2", "L3", "L4", "L5", "L6")
    result: dict[str, Any] = {}
    for layer in layers:
        first_step = None
        first_difference = None
        maximum = 0.0
        for step in range(len(repeats[0])):
            comparison = compare_trees([repeat[step][layer] for repeat in repeats])
            maximum = max(maximum, float(comparison["max_abs_difference"]))
            if first_step is None and not comparison["exact_equal"]:
                first_step = step
                first_difference = comparison
        result[layer] = {
            "exact_equal_all_steps": first_step is None,
            "first_divergent_step": first_step,
            "maximum_absolute_difference": maximum,
            "first_difference": first_difference,
        }
    divergence = [
        (result[layer]["first_divergent_step"], index, layer)
        for index, layer in enumerate(layers)
        if result[layer]["first_divergent_step"] is not None
    ]
    first = min(divergence, default=None)
    result["first_divergent_step"] = None if first is None else first[0]
    result["earliest_divergent_layer"] = None if first is None else first[2]
    return result


def exact_input_policy_audit(config: dict[str, Any]) -> dict[str, Any]:
    import torch
    from lerobot.utils.constants import ACTION

    seed = 340001
    seed_all(seed)
    runtime = make_env_and_policy(config, CHECKPOINT, 0)
    policy, policy_pre, policy_post, env_pre, env_post = runtime[:5]
    env = runtime[5]
    observation = reset_without_outcome(env, 1, seed)
    policy.reset()

    preprocess_fingerprints = []
    preprocess_rng_before = rng_fingerprints()
    processed_values = []
    for _ in range(20):
        processed = prepare_policy_observation(observation, env_pre, policy_pre)
        processed_values.append(processed)
        preprocess_fingerprints.append(tree_fingerprint(tree_arrays(processed)))
    preprocess_rng_after = rng_fingerprints()

    processed = processed_values[0]
    raw_chunks = []
    inference_rng_before = rng_fingerprints()
    with torch.inference_mode():
        for _ in range(50):
            raw_chunks.append(policy.predict_action_chunk(processed).detach().clone())
    inference_rng_after = rng_fingerprints()

    post_fingerprints = []
    post_rng_before = rng_fingerprints()
    with torch.inference_mode():
        for raw_chunk in raw_chunks:
            post = policy_post(raw_chunk.detach().clone())
            post = env_post({ACTION: post})[ACTION]
            post_fingerprints.append(array_fingerprint(post))
    post_rng_after = rng_fingerprints()

    reset_raw_fingerprints = []
    for _ in range(20):
        policy.reset()
        with torch.inference_mode():
            reset_raw_fingerprints.append(
                array_fingerprint(policy.predict_action_chunk(processed))
            )

    seed_all(seed)
    second = make_env_and_policy(config, CHECKPOINT, 0)
    second_policy = second[0]
    second_policy.reset()
    with torch.inference_mode():
        second_raw = second_policy.predict_action_chunk(processed)
    second[5].close()
    env.close()

    dropout_modules = [module for module in policy.modules() if isinstance(module, torch.nn.Dropout)]
    processor_steps = {
        "policy_preprocessor": [type(step).__name__ for step in policy_pre.steps],
        "policy_postprocessor": [type(step).__name__ for step in policy_post.steps],
        "environment_preprocessor": [type(step).__name__ for step in env_pre.steps],
        "environment_postprocessor": [type(step).__name__ for step in env_post.steps],
    }
    raw_fingerprints = [array_fingerprint(chunk) for chunk in raw_chunks]
    return {
        "same_immutable_input": True,
        "preprocessing_repeats": 20,
        "preprocessing_unique_fingerprints": len(set(preprocess_fingerprints)),
        "preprocessing_exact_deterministic": len(set(preprocess_fingerprints)) == 1,
        "inference_repeats": 50,
        "raw_chunk_unique_fingerprints": len(set(raw_fingerprints)),
        "raw_chunk_exact_deterministic": len(set(raw_fingerprints)) == 1,
        "postprocessing_repeats": 50,
        "postprocessed_unique_fingerprints": len(set(post_fingerprints)),
        "postprocessing_exact_deterministic": len(set(post_fingerprints)) == 1,
        "policy_reset_repeats": 20,
        "policy_reset_raw_unique_fingerprints": len(set(reset_raw_fingerprints)),
        "policy_reset_exact_deterministic": len(set(reset_raw_fingerprints)) == 1,
        "new_policy_object_matches": array_fingerprint(second_raw) == raw_fingerprints[0],
        "policy_training": bool(policy.training),
        "dropout_module_count": len(dropout_modules),
        "dropout_modules_in_training_mode": sum(bool(module.training) for module in dropout_modules),
        "processor_steps": processor_steps,
        "random_image_augmentation_present": False,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "preprocessing_rng_unchanged": preprocess_rng_before == preprocess_rng_after,
        "inference_rng_unchanged": inference_rng_before == inference_rng_after,
        "postprocessing_rng_unchanged": post_rng_before == post_rng_after,
    }


def process_isolated_prefix(
    config: dict[str, Any], task_id: int, state_id: int, output: Path
) -> None:
    """Emit fingerprints for one <=20-step, outcome-free prefix in this process."""

    configure_determinism()
    seed = 340000 + 100 * task_id + state_id
    seed_all(seed)
    runtime = make_env_and_policy(config, CHECKPOINT, task_id)
    try:
        records = diagnostic_prefix(
            config=config,
            policy_components=runtime[:5],
            task_id=task_id,
            state_id=state_id,
            seed=seed,
            corrected_fresh_environment=True,
            full_simulator_inventory=False,
        )
    finally:
        runtime[5].close()
    value = {
        "task_success_accessed": False,
        "task_id": task_id,
        "state_id": state_id,
        "episode_seed": seed,
        "steps": len(records),
        "layer_fingerprints": {
            layer: [tree_fingerprint(record[layer]) for record in records]
            for layer in ("L1", "L2", "L3", "L4", "L5", "L6")
        },
        "processed_path_fingerprints": [
            {path: array_fingerprint(value) for path, value in record["L3"].items()}
            for record in records
        ],
    }
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def task4_reset_accumulator_audit(config: dict[str, Any], output: Path) -> None:
    """Reproduce task-4 hard-reset accumulation without reward/success access."""

    seed = 340401
    seed_all(seed)
    env = make_fresh_env(config, 4)
    rows = []
    try:
        official_hash = array_fingerprint(env._init_states[1])
        for repeat_id in range(3):
            seed_all(seed)
            observation = reset_without_outcome(env, 1, seed)
            inner = env._env.env
            cabinet_id = inner.sim.model.body_name2id("wooden_cabinet_1_main")
            rows.append(
                {
                    "repeat_id": repeat_id,
                    "object_property_initializer_count": len(
                        inner.object_property_initializers
                    ),
                    "object_property_initializer_types": [
                        type(value).__name__
                        for value in inner.object_property_initializers
                    ],
                    "official_state_vector_sha256": official_hash,
                    "sim_state_sha256": array_fingerprint(
                        inner.sim.get_state().flatten()
                    ),
                    "cabinet_body_id": int(cabinet_id),
                    "cabinet_body_pos": np.asarray(
                        inner.sim.model.body_pos[cabinet_id]
                    ).tolist(),
                    "cabinet_body_quat": np.asarray(
                        inner.sim.model.body_quat[cabinet_id]
                    ).tolist(),
                    "image_sha256": {
                        name: array_fingerprint(image)
                        for name, image in observation["pixels"].items()
                    },
                }
            )
    finally:
        env.close()
    value = {
        "task_success_accessed": False,
        "task_id": 4,
        "state_id": 1,
        "episode_seed": seed,
        "same_official_state_vector": len(
            {row["official_state_vector_sha256"] for row in rows}
        )
        == 1,
        "rows": rows,
    }
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canary(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Run every repeat in a new process to isolate EGL and LIBERO mutable state."""

    block_rows = []
    with tempfile.TemporaryDirectory(prefix="gate4a2_canary_") as temporary:
        temporary_root = Path(temporary)
        for task_id in range(10):
            for state_id in SELECTED_STATES:
                seed = 340000 + 100 * task_id + state_id
                repeats = []
                for repeat_id in range(3):
                    repeat_path = (
                        temporary_root
                        / f"task_{task_id:02d}_state_{state_id:02d}_repeat_{repeat_id}.json"
                    )
                    subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "prefix-worker",
                            "--task-id",
                            str(task_id),
                            "--state-id",
                            str(state_id),
                            "--output",
                            str(repeat_path),
                        ],
                        cwd=ROOT,
                        check=True,
                        stdout=subprocess.DEVNULL,
                    )
                    repeat = json.loads(repeat_path.read_text(encoding="utf-8"))
                    if repeat.get("task_success_accessed") is not False:
                        raise RuntimeError("prefix worker did not attest outcome sealing")
                    repeats.append(repeat)
                initial_observation = [repeat["layer_fingerprints"]["L2"][0] for repeat in repeats]
                processed_sequence = [
                    sha256_bytes("".join(repeat["layer_fingerprints"]["L3"]).encode())
                    for repeat in repeats
                ]
                action_sequence = [
                    sha256_bytes("".join(repeat["layer_fingerprints"]["L6"]).encode())
                    for repeat in repeats
                ]
                layer_first_steps: dict[str, int | None] = {}
                for layer in ("L1", "L2", "L3", "L4", "L5", "L6"):
                    layer_first_steps[layer] = next(
                        (
                            step
                            for step in range(20)
                            if len(
                                {
                                    repeat["layer_fingerprints"][layer][step]
                                    for repeat in repeats
                                }
                            )
                            != 1
                        ),
                        None,
                    )
                first_divergence = min(
                    (
                        (step, index, layer)
                        for index, layer in enumerate(("L1", "L2", "L3", "L4", "L5", "L6"))
                        if (step := layer_first_steps[layer]) is not None
                    ),
                    default=None,
                )
                passed = (
                    len(set(initial_observation)) == 1
                    and len(set(processed_sequence)) == 1
                    and len(set(action_sequence)) == 1
                    and first_divergence is None
                )
                block_rows.append(
                    {
                        "task_id": task_id,
                        "state_id": state_id,
                        "episode_seed": seed,
                        "repeats": 3,
                        "prefix_steps": 20,
                        "pass": passed,
                        "initial_observation_fingerprint": initial_observation[0]
                        if passed
                        else initial_observation,
                        "processed_input_sequence_fingerprint": processed_sequence[0]
                        if passed
                        else processed_sequence,
                        "fresh_action_sequence_fingerprint": action_sequence[0]
                        if passed
                        else action_sequence,
                        "earliest_divergent_layer": None
                        if first_divergence is None
                        else first_divergence[2],
                        "first_divergent_step": None
                        if first_divergence is None
                        else first_divergence[0],
                    }
                )
                print(
                    f"[{len(block_rows):03d}/100] task={task_id} state={state_id} "
                    f"pass={passed}",
                    flush=True,
                )
    passed_blocks = sum(bool(row["pass"]) for row in block_rows)
    result = {
        "schema_version": 1,
        "scope": "Outcome-blind fresh-action determinism canary; no episode exceeds 20 steps.",
        "historical_gate4a2_traces_reused": False,
        "task_success_accessed": False,
        "checkpoint_revision": "8f04de1472975d62db214238b2fc07e78bde2474",
        "model_sha256": "912f41808962d80ca9084435aa01eccccdd97b7eae3a841c9f4ac71caaf9f8b0",
        "dataset_revision": "38927e939de5d2bfd40effcf27d16710aea6f864",
        "selected_states": list(SELECTED_STATES),
        "technical_correction": (
            "Run every diagnostic/official episode in a fresh process. Before importing/"
            "constructing the rendering stack set "
            "__GL_THREADED_OPTIMIZATIONS=0 and __GL_YIELD=NOTHING; seed Python, "
            "NumPy, Torch, and CUDA before constructing a fresh LIBERO environment "
            "for every episode; never reuse the mutable environment across episodes."
        ),
        "render_driver_settings": {
            "__GL_THREADED_OPTIMIZATIONS": os.environ.get("__GL_THREADED_OPTIMIZATIONS"),
            "__GL_YIELD": os.environ.get("__GL_YIELD"),
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
        },
        "blocks": 100,
        "repeats_per_block": 3,
        "prefix_steps_per_repeat": 20,
        "diagnostic_controller_steps": 6000,
        "passed_blocks": passed_blocks,
        "failed_blocks": 100 - passed_blocks,
        "all_100_blocks_exact": passed_blocks == 100,
        "numerical_tolerance_used": False,
        "block_results": block_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def audit(output: Path) -> None:
    configure_determinism()
    config = load_config(CONFIG)
    historical = historical_prefix_audit()
    exact_input = exact_input_policy_audit(config)
    cases = ((4, 1, 340401), (4, 37, 340437), (3, 1, 340301), (0, 1, 340001))
    original_rows = []
    corrected_rows = []
    for task_id, state_id, seed in cases:
        seed_all(seed)
        runtime = make_env_and_policy(config, CHECKPOINT, task_id)
        reused_env = runtime[5]
        original = [
            diagnostic_prefix(
                config=config,
                policy_components=runtime[:5],
                task_id=task_id,
                state_id=state_id,
                seed=seed,
                corrected_fresh_environment=False,
                reused_env=reused_env,
            )
            for _ in range(3)
        ]
        original_rows.append(
            {
                "task_id": task_id,
                "state_id": state_id,
                "comparison": compare_layered_repeats(original),
            }
        )
        corrected = [
            diagnostic_prefix(
                config=config,
                policy_components=runtime[:5],
                task_id=task_id,
                state_id=state_id,
                seed=seed,
                corrected_fresh_environment=True,
            )
            for _ in range(3)
        ]
        corrected_rows.append(
            {
                "task_id": task_id,
                "state_id": state_id,
                "comparison": compare_layered_repeats(corrected),
            }
        )
        reused_env.close()
    value = {
        "schema_version": 1,
        "task_success_accessed": False,
        "historical": historical,
        "exact_input_policy": exact_input,
        "original_reused_environment_prefixes": original_rows,
        "corrected_fresh_environment_prefixes": corrected_rows,
    }
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--output", type=Path, required=True)
    canary_parser = subparsers.add_parser("canary")
    canary_parser.add_argument("--output", type=Path, default=CANARY_OUTPUT)
    worker_parser = subparsers.add_parser("prefix-worker")
    worker_parser.add_argument("--task-id", type=int, required=True)
    worker_parser.add_argument("--state-id", type=int, required=True)
    worker_parser.add_argument("--output", type=Path, required=True)
    task4_parser = subparsers.add_parser("task4-reset")
    task4_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        audit(args.output)
    elif args.command == "canary":
        configure_determinism()
        result = canary(load_config(CONFIG), args.output)
        print(
            f"determinism canary: {result['passed_blocks']}/{result['blocks']} blocks exact",
            flush=True,
        )
    elif args.command == "prefix-worker":
        process_isolated_prefix(
            load_config(CONFIG), args.task_id, args.state_id, args.output
        )
    else:
        configure_determinism()
        task4_reset_accumulator_audit(load_config(CONFIG), args.output)


if __name__ == "__main__":
    main()
