#!/usr/bin/env python3
"""Outcome-blind RoboTwin native-ACT query-time calibration."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from research.audit_tools.robotwin_temporal_reuse import (
    CHUNK_LENGTH,
    PHYSICAL_SOURCE_AGE_SECONDS,
    select_physical_age_source,
)
from research.audit_tools.run_robotwin_closed_loop_canaries import CountingScene


PHYSICS_HZ = 250.0


class RecordingPlanner:
    """Transparent planner proxy recording the official TOPP return values."""

    def __init__(self, planner: Any, arm: str) -> None:
        self._planner = planner
        self._arm = arm
        self.calls: list[dict[str, Any]] = []

    def TOPP(self, *args: Any, **kwargs: Any) -> Any:
        try:
            result = self._planner.TOPP(*args, **kwargs)
        except Exception as error:
            self.calls.append(
                {"arm": self._arm, "status": "ERROR", "error_type": type(error).__name__}
            )
            raise
        duration = float(result[4])
        positions = np.asarray(result[1])
        self.calls.append(
            {
                "arm": self._arm,
                "status": "PASS",
                "duration_seconds": duration,
                "trajectory_steps": int(positions.shape[0]),
            }
        )
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._planner, name)


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "q25": None,
            "q75": None,
            "iqr": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    q25, q75 = np.quantile(array, [0.25, 0.75])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "standard_deviation": float(array.std(ddof=0)),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def close_environment(task_env: Any) -> None:
    try:
        task_env.close_env(clear_cache=True)
    finally:
        del task_env
        gc.collect()
        torch.cuda.empty_cache()


def expert_seed_is_valid(evaluator: Any, task_args: dict[str, Any], seed: int) -> bool:
    """Apply the official expert-reset screen without retaining its outcome."""

    task_env = evaluator.class_decorator(task_args["task_name"])
    args = dict(task_args)
    args["eval_mode"] = True
    args["render_freq"] = 0
    args["eval_video_log"] = False
    try:
        task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
        task_env.play_once()
        return bool(task_env.plan_success and task_env.check_success())
    except Exception:
        return False
    finally:
        close_environment(task_env)


def setup_rollout(evaluator: Any, task_args: dict[str, Any], seed: int) -> Any:
    task_env = evaluator.class_decorator(task_args["task_name"])
    args = dict(task_args)
    args["eval_mode"] = True
    args["render_freq"] = 0
    args["eval_video_log"] = False
    task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
    task_env.set_instruction(instruction=task_args["task_name"])
    task_env.scene = CountingScene(task_env.scene)
    task_env.robot.left_mplib_planner = RecordingPlanner(
        task_env.robot.left_mplib_planner, "left"
    )
    task_env.robot.right_mplib_planner = RecordingPlanner(
        task_env.robot.right_mplib_planner, "right"
    )
    return task_env


def run_native_timing_rollout(
    evaluator: Any,
    model: Any,
    task_args: dict[str, Any],
    *,
    seed: int,
    decisions: int,
) -> dict[str, Any]:
    model.reset()
    task_env = setup_rollout(evaluator, task_args, seed)
    query_times: dict[int, float] = {}
    records = []
    previous_topp: list[dict[str, Any]] | None = None
    previous_internal_steps: int | None = None
    try:
        for decision in range(decisions):
            query_time = task_env.scene.physics_steps / PHYSICS_HZ
            query_times[decision] = query_time
            selection = select_physical_age_source(
                query_times,
                decision,
                target_age_seconds=PHYSICAL_SOURCE_AGE_SECONDS,
                chunk_length=CHUNK_LENGTH,
            )
            observation = task_env.get_obs()
            xpolicy_observation = evaluator.robotwin_obs_to_xpolicylab(
                observation,
                instruction=task_args["task_name"],
                env_idx=0,
                frequency=30,
                task_env=task_env,
            )
            model.update_obs(xpolicy_observation)
            native_action_dict = model.get_action()[0]
            flat_action, action_type = evaluator.xpolicylab_action_to_robotwin(
                native_action_dict,
                action_type="joint",
                current_observation=observation,
            )
            if np.asarray(flat_action).shape != (14,):
                raise RuntimeError("native ACT did not produce one 14-D action")

            left_recorder = task_env.robot.left_mplib_planner
            right_recorder = task_env.robot.right_mplib_planner
            left_start = len(left_recorder.calls)
            right_start = len(right_recorder.calls)
            physics_before = task_env.scene.physics_steps
            task_env.take_action(flat_action, action_type=action_type)
            physics_after = task_env.scene.physics_steps
            current_topp = (
                left_recorder.calls[left_start:] + right_recorder.calls[right_start:]
            )

            record = {
                "decision_index": decision,
                "simulator_query_time_seconds": query_time,
                "prior_query_timestamps_seconds": [
                    query_times[index] for index in range(decision)
                ],
                "previous_action_topp": previous_topp,
                "previous_action_internal_physics_steps": previous_internal_steps,
                "previous_action_internal_simulator_duration_seconds": (
                    None
                    if previous_internal_steps is None
                    else previous_internal_steps / PHYSICS_HZ
                ),
                "selected_old_source_query": (
                    None if selection is None else selection.old_source_step
                ),
                "selected_lag_ticks": (
                    None if selection is None else selection.chunk_offset
                ),
                "realized_source_age_seconds": (
                    None if selection is None else selection.realized_source_age_seconds
                ),
                "absolute_error_from_1s_seconds": (
                    None if selection is None else selection.absolute_age_error_seconds
                ),
                "current_action_topp": current_topp,
                "current_action_internal_physics_steps": physics_after - physics_before,
                "current_action_internal_simulator_duration_seconds": (
                    physics_after - physics_before
                )
                / PHYSICS_HZ,
            }
            records.append(record)
            previous_topp = current_topp
            previous_internal_steps = physics_after - physics_before

            # The official environment refuses further actions after terminal state;
            # stop without retaining or reporting that outcome.
            if task_env.eval_success or task_env.take_action_cnt >= task_env.step_lim:
                break
    finally:
        close_environment(task_env)
    return {"seed": seed, "query_count": len(records), "records": records}


def aggregate(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    inter_decision = []
    lags = []
    ages = []
    errors = []
    for rollout in rollouts:
        records = rollout["records"]
        timestamps = [record["simulator_query_time_seconds"] for record in records]
        inter_decision.extend(np.diff(timestamps).tolist())
        for record in records:
            if record["selected_lag_ticks"] is None:
                continue
            lags.append(int(record["selected_lag_ticks"]))
            ages.append(float(record["realized_source_age_seconds"]))
            errors.append(float(record["absolute_error_from_1s_seconds"]))

    lag_counts = Counter(lags)
    lag_total = sum(lag_counts.values())
    lag_distribution = {
        str(lag): {
            "count": count,
            "percentage": 100.0 * count / lag_total,
        }
        for lag, count in sorted(lag_counts.items())
    }
    return {
        "inter_decision_physical_duration_seconds": numeric_summary(inter_decision),
        "selected_lag_ticks": numeric_summary([float(value) for value in lags]),
        "selected_lag_distribution": lag_distribution,
        "realized_source_age_seconds": numeric_summary(ages),
        "absolute_error_from_1s_seconds": {
            **numeric_summary(errors),
            "percentage_at_or_below_0_5s": (
                None
                if not errors
                else 100.0 * float(np.mean(np.asarray(errors) <= 0.5))
            ),
        },
        "warmup_rule": "NEWEST only when no valid past query exists (decision 0)",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--valid-seeds", type=int, default=10)
    parser.add_argument("--decisions", type=int, default=30)
    parser.add_argument("--max-seed-attempts", type=int, default=50)
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
            "temporal_agg": True,
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

    rollouts = []
    rejected_seed_count = 0
    for seed in range(args.seed_start, args.seed_start + args.max_seed_attempts):
        if len(rollouts) >= args.valid_seeds:
            break
        if not expert_seed_is_valid(evaluator, task_args, seed):
            rejected_seed_count += 1
            print(f"seed {seed}: reset rejected by official expert screen", flush=True)
            continue
        print(f"seed {seed}: timing rollout {len(rollouts) + 1}/{args.valid_seeds}", flush=True)
        rollouts.append(
            run_native_timing_rollout(
                evaluator,
                model,
                task_args,
                seed=seed,
                decisions=args.decisions,
            )
        )

    if len(rollouts) != args.valid_seeds:
        raise RuntimeError(
            f"obtained {len(rollouts)} valid seeds, expected {args.valid_seeds}"
        )
    output = {
        "scope": "Outcome-blind native-ACT physical query-time calibration",
        "task": "beat_block_hammer",
        "checkpoint": str(args.checkpoint_dir / "policy_last.ckpt"),
        "deployment": "NATIVE_ACT unchanged official temporal aggregation",
        "target_source_age_seconds": PHYSICAL_SOURCE_AGE_SECONDS,
        "chunk_length": CHUNK_LENGTH,
        "physics_hz": PHYSICS_HZ,
        "valid_seed_count": len(rollouts),
        "rejected_seed_count": rejected_seed_count,
        "maximum_decisions_per_rollout": args.decisions,
        "rollouts": rollouts,
        "aggregate": aggregate(rollouts),
        "temporal_method_success_inspected_or_recorded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["aggregate"], indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
