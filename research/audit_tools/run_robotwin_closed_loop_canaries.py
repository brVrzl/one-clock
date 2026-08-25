#!/usr/bin/env python3
"""Outcome-free RoboTwin decision-target-aligned temporal canaries."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from research.audit_tools.robotwin_temporal_reuse import (
    ACTION_GROUPS,
    NOMINAL_SOURCE_AGE_TICKS,
    RoboTwinTemporalExecutor,
    native_act_aggregate,
    postprocess_action,
)


PHYSICS_HZ = 250.0


class CountingScene:
    """Transparent scene proxy that counts the official physics-step calls."""

    def __init__(self, scene: Any) -> None:
        self._scene = scene
        self.physics_steps = 0

    def step(self) -> Any:
        result = self._scene.step()
        self.physics_steps += 1
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._scene, name)


def digest(value: Any) -> str:
    hasher = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, dict):
            hasher.update(b"dict")
            for key in sorted(item, key=str):
                hasher.update(str(key).encode())
                update(item[key])
        elif isinstance(item, (list, tuple)):
            hasher.update(f"seq:{len(item)}".encode())
            for child in item:
                update(child)
        elif isinstance(item, np.ndarray) or torch.is_tensor(item):
            array = item.detach().cpu().numpy() if torch.is_tensor(item) else np.asarray(item)
            contiguous = np.ascontiguousarray(array)
            hasher.update(str(contiguous.dtype).encode())
            hasher.update(str(contiguous.shape).encode())
            hasher.update(contiguous.view(np.uint8))
        elif isinstance(item, (float, int, bool, np.number)):
            update(np.asarray(item))
        elif item is None:
            hasher.update(b"None")
        else:
            hasher.update(repr(item).encode())

    update(value)
    return hasher.hexdigest()


def numeric_vector(value: Any) -> np.ndarray:
    parts = []

    def collect(item: Any) -> None:
        if isinstance(item, dict):
            for key in sorted(item, key=str):
                collect(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect(child)
        elif isinstance(item, np.ndarray) or torch.is_tensor(item):
            array = item.detach().cpu().numpy() if torch.is_tensor(item) else np.asarray(item)
            if np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_:
                parts.append(array.astype(np.float64, copy=False).ravel())
        elif isinstance(item, (float, int, bool, np.number)):
            parts.append(np.asarray([item], dtype=np.float64))

    collect(value)
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)


def sim_state(task_env: Any) -> dict[str, Any]:
    robot = task_env.robot
    state: dict[str, Any] = {
        "left_joint_state": np.asarray(robot.get_left_arm_jointState()),
        "right_joint_state": np.asarray(robot.get_right_arm_jointState()),
        "left_gripper": float(robot.get_left_gripper_val()),
        "right_gripper": float(robot.get_right_gripper_val()),
        "actors": [],
        "articulations": [],
    }
    scene = task_env.scene
    actors = sorted(
        enumerate(scene.get_all_actors()), key=lambda item: (item[1].get_name(), item[0])
    )
    for _, actor in actors:
        actor_state = {"name": actor.get_name()}
        pose = actor.get_pose()
        actor_state["pose"] = np.concatenate([np.asarray(pose.p), np.asarray(pose.q)])
        for name in ("get_linear_velocity", "get_angular_velocity"):
            method = getattr(actor, name, None)
            if callable(method):
                actor_state[name] = np.asarray(method())
        state["actors"].append(actor_state)
    articulations = sorted(
        enumerate(scene.get_all_articulations()),
        key=lambda item: (item[1].get_name(), item[0]),
    )
    for _, articulation in articulations:
        articulation_state = {
            "name": articulation.get_name(),
            "qpos": np.asarray(articulation.get_qpos()),
            "qvel": np.asarray(articulation.get_qvel()),
        }
        pose = articulation.get_root_pose()
        articulation_state["root_pose"] = np.concatenate(
            [np.asarray(pose.p), np.asarray(pose.q)]
        )
        state["articulations"].append(articulation_state)
    return state


def infer_full_chunk(model: Any) -> np.ndarray:
    act = model.model
    observation = act.obs_cache
    qpos = torch.from_numpy(act.pre_process(np.asarray(observation["qpos"]))).float()
    qpos = qpos.to(act.device).unsqueeze(0)
    images = np.stack([observation[name] for name in act.camera_names], axis=0)
    image_tensor = torch.from_numpy(images).float().to(act.device).unsqueeze(0)
    with torch.no_grad():
        chunk = act.policy(qpos, image_tensor)[0]
    return chunk.detach().cpu().numpy()


def setup_environment(evaluator: Any, task_args: dict[str, Any], seed: int):
    task_env = evaluator.class_decorator(task_args["task_name"])
    args = dict(task_args)
    args["eval_mode"] = True
    args["render_freq"] = 0
    args["eval_video_log"] = False
    task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
    task_env.set_instruction(instruction=task_args["task_name"])
    task_env.scene = CountingScene(task_env.scene)
    return task_env


def close_environment(task_env: Any) -> None:
    try:
        task_env.close_env(clear_cache=True)
    finally:
        del task_env
        gc.collect()
        torch.cuda.empty_cache()


def run_method(
    evaluator: Any,
    model: Any,
    task_args: dict[str, Any],
    *,
    method: str,
    seed: int,
    decisions: int,
) -> dict[str, Any]:
    model.reset()
    executor = RoboTwinTemporalExecutor(method)
    task_env = setup_environment(evaluator, task_args, seed)
    reset_state = sim_state(task_env)
    records = []
    source_times: dict[int, float] = {}
    chunks: dict[int, np.ndarray] = {}
    try:
        for decision in range(decisions):
            sim_time_before = task_env.scene.physics_steps / PHYSICS_HZ
            source_times[decision] = sim_time_before
            raw_observation = task_env.get_obs()
            xpolicy_observation = evaluator.robotwin_obs_to_xpolicylab(
                raw_observation,
                instruction=task_args["task_name"],
                env_idx=0,
                frequency=30,
                task_env=task_env,
            )
            encoded_observation = model.encode_obs(
                xpolicy_observation, "joint", model.robot_action_dim_info
            )
            model.model.update_obs(encoded_observation)
            normalized_chunk = infer_full_chunk(model)
            chunks[decision] = normalized_chunk
            temporal = executor.update(decision, normalized_chunk)
            fresh_action = postprocess_action(
                temporal.fresh_action,
                model.model.stats["action_mean"],
                model.model.stats["action_std"],
            )
            old_action = (
                None
                if temporal.old_action is None
                else postprocess_action(
                    temporal.old_action,
                    model.model.stats["action_mean"],
                    model.model.stats["action_std"],
                )
            )
            composed_action = postprocess_action(
                temporal.action,
                model.model.stats["action_mean"],
                model.model.stats["action_std"],
            ).astype(np.float32)
            if not np.isfinite(composed_action).all():
                raise RuntimeError("non-finite composed action")
            physics_before = task_env.scene.physics_steps
            task_env.take_action(composed_action, action_type="qpos")
            physics_after = task_env.scene.physics_steps
            simulator_state_after = sim_state(task_env)
            physical_ages = {
                group: sim_time_before - source_times[source]
                for group, source in temporal.group_source_steps.items()
            }
            record = temporal.as_log_record()
            record["sim_time_before_execution"] = sim_time_before
            record["sim_time_after_execution"] = physics_after / PHYSICS_HZ
            record["internal_physics_steps"] = physics_after - physics_before
            record["source_age_simulator_seconds_per_group"] = physical_ages
            record["fresh_action"] = fresh_action.tolist()
            record["old_action"] = None if old_action is None else old_action.tolist()
            record["executed_composed_action"] = composed_action.tolist()
            record["fingerprints"] = {
                "raw_observation": digest(raw_observation),
                "processed_policy_input": digest(encoded_observation),
                "full_act_chunk": digest(normalized_chunk),
                "postprocessed_action": digest(composed_action),
                "simulator_state_after": digest(simulator_state_after),
            }
            record["numeric_layers"] = {
                "raw_observation": numeric_vector(raw_observation),
                "processed_policy_input": numeric_vector(encoded_observation),
                "full_act_chunk": numeric_vector(normalized_chunk),
                "postprocessed_action": numeric_vector(composed_action),
                "simulator_state_after": numeric_vector(simulator_state_after),
            }
            records.append(record)
    finally:
        close_environment(task_env)

    native_at_last = postprocess_action(
        native_act_aggregate(chunks, decisions - 1),
        model.model.stats["action_mean"],
        model.model.stats["action_std"],
    )
    return {
        "method": method,
        "seed": seed,
        "reset_state_sha256": digest(reset_state),
        "reset_state_numeric": numeric_vector(reset_state),
        "records": records,
        "native_act_reference_at_last_decision": native_at_last.tolist(),
    }


def compare_runs(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    result = {
        "reset_state_exact_equal": first["reset_state_sha256"] == second["reset_state_sha256"],
        "reset_state_max_abs_difference": None,
        "earliest_divergent_tick": None,
        "earliest_divergent_layer": None,
        "layer_max_abs_difference": {},
    }
    reset_first = first["reset_state_numeric"]
    reset_second = second["reset_state_numeric"]
    if reset_first.shape == reset_second.shape:
        result["reset_state_max_abs_difference"] = float(
            np.max(np.abs(reset_first - reset_second), initial=0.0)
        )
    layers = (
        "raw_observation",
        "processed_policy_input",
        "full_act_chunk",
        "postprocessed_action",
        "simulator_state_after",
    )
    for tick, (left, right) in enumerate(zip(first["records"], second["records"])):
        for layer in layers:
            a = left["numeric_layers"][layer]
            b = right["numeric_layers"][layer]
            difference = (
                float(np.max(np.abs(a - b), initial=0.0))
                if a.shape == b.shape
                else None
            )
            result["layer_max_abs_difference"].setdefault(layer, []).append(difference)
            equal = left["fingerprints"][layer] == right["fingerprints"][layer]
            if not equal and result["earliest_divergent_tick"] is None:
                result["earliest_divergent_tick"] = tick
                result["earliest_divergent_layer"] = layer
    result["exact_equal_all_layers"] = (
        result["reset_state_exact_equal"]
        and result["earliest_divergent_tick"] is None
    )
    result["max_abs_difference_by_layer"] = {
        layer: (
            None
            if any(value is None for value in values)
            else max(values, default=0.0)
        )
        for layer, values in result.pop("layer_max_abs_difference").items()
    }
    return result


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(child) for key, child in value.items() if key != "numeric_layers"}
    if isinstance(value, list):
        return [json_ready(child) for child in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=100000)
    args = parser.parse_args()

    sys.path.insert(0, str(args.robotwin_root / "scripts"))
    sys.path.insert(0, str(args.robotwin_root))
    import eval_policy_xpolicylab as evaluator
    from XPolicyLab.policy.ACT.model import Model

    os.environ["ACT_ACTION_DIM"] = "14"
    deploy = yaml.safe_load(
        (args.robotwin_root / "XPolicyLab/policy/ACT/deploy.yml").read_text()
    )
    deploy.update(
        {
            "ckpt_dir": str(args.checkpoint_dir),
            "ckpt_name": args.checkpoint_dir.name,
            "bench_name": "RoboTwin",
            "task_name": "beat_block_hammer",
            "env_cfg_type": "aloha_agilex",
            "action_type": "joint",
            "action_dim": 14,
            "device": "cuda:0",
        }
    )
    model = Model(deploy)
    task_usr_args = {
        "task_name": "beat_block_hammer",
        "task_config": "demo_clean",
        "ckpt_setting": args.checkpoint_dir.name,
        "policy_name": "ACT",
    }
    task_args, _ = evaluator.load_task_args(task_usr_args)
    task_args["eval_instruction"] = "seen"

    newest_first = run_method(
        evaluator, model, task_args, method="NEWEST", seed=args.seed, decisions=20
    )
    newest_second = run_method(
        evaluator, model, task_args, method="NEWEST", seed=args.seed, decisions=20
    )
    determinism = compare_runs(newest_first, newest_second)
    full_old = run_method(
        evaluator, model, task_args, method="FULL_OLD_17", seed=args.seed, decisions=18
    )
    fo = run_method(
        evaluator, model, task_args, method="FO_17", seed=args.seed, decisions=18
    )

    output = {
        "scope": "Outcome-free closed-loop determinism and temporal provenance canaries",
        "task": "beat_block_hammer",
        "seed": args.seed,
        "checkpoint": str(args.checkpoint_dir / "policy_last.ckpt"),
        "action_groups": {name: list(indices) for name, indices in ACTION_GROUPS.items()},
        "physics_hz": PHYSICS_HZ,
        "nominal_demo_decision_hz": PHYSICS_HZ / 15.0,
        "source_age_ticks": NOMINAL_SOURCE_AGE_TICKS,
        "source_age_nominal_demo_seconds": NOMINAL_SOURCE_AGE_TICKS / (PHYSICS_HZ / 15.0),
        "determinism": determinism,
        "newest_first": newest_first,
        "newest_second": newest_second,
        "full_old_17": full_old,
        "fo_17": fo,
        "scientific_outcomes_inspected": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_ready(output), indent=2) + "\n")
    print(json.dumps(determinism, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
