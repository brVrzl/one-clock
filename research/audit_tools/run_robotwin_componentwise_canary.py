#!/usr/bin/env python3
"""Outcome-free 20-decision closed-loop canary for learned component kernels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from research.audit_tools.fit_robotwin_componentwise_kernel import ARM, GRIPPER, compose
from research.audit_tools.robotwin_temporal_reuse import postprocess_action
from research.audit_tools.run_robotwin_closed_loop_canaries import PHYSICS_HZ, infer_full_chunk
from research.audit_tools.run_robotwin_exploratory_task import close_environment, load_model, setup_environment


def available_prefix(kernel: np.ndarray, count: int) -> np.ndarray:
    prefix = kernel[:count].copy()
    if prefix.sum() <= 0:
        prefix[:] = 0
        prefix[0] = 1
        return prefix
    return prefix / prefix.sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    kernel = json.loads(args.kernel.read_text())
    arm_kernel = np.asarray(kernel["arm_kernel"])
    gripper_kernel = np.asarray(kernel["gripper_kernel"])
    task = kernel["task"]
    sys.path.insert(0, str(args.robotwin_root / "scripts"))
    sys.path.insert(0, str(args.robotwin_root))
    import eval_policy_xpolicylab as evaluator

    model = load_model(args.robotwin_root, args.checkpoint_dir, task)
    task_args, _ = evaluator.load_task_args(
        {"task_name": task, "task_config": "demo_clean", "ckpt_setting": args.checkpoint_dir.name, "policy_name": "ACT"}
    )
    task_args["eval_instruction"] = "seen"
    cell = {"task": task, "eligible_seed_index": 0, "robotwin_seed": 100000}
    environment = setup_environment(evaluator, task_args, cell)
    model.reset()
    history: dict[int, np.ndarray] = {}
    records = []
    try:
        for decision in range(20):
            query_time = environment.scene.physics_steps / PHYSICS_HZ
            observation = environment.get_obs()
            converted = evaluator.robotwin_obs_to_xpolicylab(
                observation, instruction=task, env_idx=0, frequency=30, task_env=environment
            )
            encoded = model.encode_obs(converted, "joint", model.robot_action_dim_info)
            model.model.update_obs(encoded)
            normalized = infer_full_chunk(model)
            raw_chunk = postprocess_action(
                normalized,
                model.model.stats["action_mean"],
                model.model.stats["action_std"],
            )
            history[decision] = raw_chunk
            available = min(decision + 1, 50)
            candidates = np.stack([history[decision - lag][lag] for lag in range(available)])
            arm_weights = available_prefix(arm_kernel, available)
            grip_weights = available_prefix(gripper_kernel, available)
            executed = compose(candidates, arm_weights, grip_weights)
            expected_arm = arm_weights @ candidates[:, ARM]
            expected_grip = grip_weights @ candidates[:, GRIPPER]
            if not np.array_equal(executed[ARM], expected_arm) or not np.array_equal(executed[GRIPPER], expected_grip):
                raise RuntimeError("component-wise composition provenance failure")
            before = environment.scene.physics_steps
            environment.take_action(executed, action_type="qpos")
            after = environment.scene.physics_steps
            records.append(
                {
                    "decision": decision,
                    "simulator_query_timestamp_seconds": query_time,
                    "source_decision_lags": list(range(available)),
                    "arm_weights": arm_weights.tolist(),
                    "gripper_weights": grip_weights.tolist(),
                    "executed_action": executed.tolist(),
                    "internal_physics_steps": after - before,
                }
            )
    finally:
        close_environment(environment)
    output = {
        "scope": "outcome-free closed-loop component-wise aggregation provenance canary",
        "task": task,
        "decision_count": len(records),
        "records": records,
        "provenance_passed": len(records) == 20,
        "task_success_inspected": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
