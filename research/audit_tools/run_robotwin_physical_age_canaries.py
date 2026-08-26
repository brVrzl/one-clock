#!/usr/bin/env python3
"""Outcome-blind RoboTwin FO_1S same-current-decision-target canaries."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from research.audit_tools.robotwin_temporal_reuse import (
    ACTION_GROUPS,
    ARM_GROUPS,
    GRIPPER_GROUPS,
    PHYSICAL_AGE_METHODS,
    PHYSICAL_SOURCE_AGE_SECONDS,
    RoboTwinPhysicalAgeExecutor,
    native_act_aggregate,
    postprocess_action,
    select_physical_age_source,
)
from research.audit_tools.run_robotwin_closed_loop_canaries import (
    PHYSICS_HZ,
    close_environment,
    compare_runs,
    digest,
    infer_full_chunk,
    json_ready,
    numeric_vector,
    setup_environment,
    sim_state,
)


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
    executor = RoboTwinPhysicalAgeExecutor(method)
    task_env = setup_environment(evaluator, task_args, seed)
    reset_state = sim_state(task_env)
    records = []
    source_times: dict[int, float] = {}
    chunks: dict[int, np.ndarray] = {}
    previous_executed_action: np.ndarray | None = None
    try:
        for decision in range(decisions):
            query_time = task_env.scene.physics_steps / PHYSICS_HZ
            source_times[decision] = query_time
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
            temporal = executor.update(
                decision,
                normalized_chunk,
                query_time_seconds=query_time,
            )

            expected_selection = select_physical_age_source(source_times, decision)
            if expected_selection is None:
                if temporal.old_action is not None:
                    raise AssertionError("warmup unexpectedly used an old action")
            else:
                if temporal.old_source_step != expected_selection.old_source_step:
                    raise AssertionError("executor selected the wrong physical-time source")
                if temporal.old_chunk_offset != expected_selection.chunk_offset:
                    raise AssertionError(
                        "executor selected the wrong current-decision-target offset"
                    )
                expected_old = chunks[expected_selection.old_source_step][
                    expected_selection.chunk_offset
                ]
                np.testing.assert_array_equal(temporal.old_action, expected_old)
                if expected_selection.chunk_offset != 0:
                    wrong_old_first = chunks[expected_selection.old_source_step][0]
                    if np.array_equal(temporal.old_action, wrong_old_first):
                        raise AssertionError("old_chunk[0] was not discriminated")

            if method == "NEWEST" or temporal.old_action is None:
                np.testing.assert_array_equal(temporal.action, temporal.fresh_action)
            elif method == "FULL_OLD_1S":
                np.testing.assert_array_equal(temporal.action, temporal.old_action)
            elif method == "FO_1S":
                for group_name in ARM_GROUPS:
                    indices = list(ACTION_GROUPS[group_name])
                    np.testing.assert_array_equal(
                        temporal.action[indices], temporal.fresh_action[indices]
                    )
                for group_name in GRIPPER_GROUPS:
                    indices = list(ACTION_GROUPS[group_name])
                    np.testing.assert_array_equal(
                        temporal.action[indices], temporal.old_action[indices]
                    )

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
            physics_before = task_env.scene.physics_steps
            task_env.take_action(composed_action, action_type="qpos")
            physics_after = task_env.scene.physics_steps
            simulator_state_after = sim_state(task_env)

            record = temporal.as_log_record()
            record["prior_query_timestamps_seconds"] = [
                source_times[index] for index in range(decision)
            ]
            record["sim_time_before_execution"] = query_time
            record["sim_time_after_execution"] = physics_after / PHYSICS_HZ
            record["internal_physics_steps"] = physics_after - physics_before
            record["fresh_action"] = fresh_action.tolist()
            record["old_action"] = None if old_action is None else old_action.tolist()
            record["executed_composed_action"] = composed_action.tolist()
            record["gripper_hold_used"] = False
            record["previous_executed_grippers"] = (
                None
                if previous_executed_action is None
                else previous_executed_action[[6, 13]].tolist()
            )
            record["mechanical_provenance_pass"] = True
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
            previous_executed_action = composed_action.copy()

            # Stop if the official environment will refuse another action, without
            # retaining or reporting the task outcome.
            if task_env.eval_success or task_env.take_action_cnt >= task_env.step_lim:
                break
    finally:
        close_environment(task_env)

    native_at_last = postprocess_action(
        native_act_aggregate(chunks, len(records) - 1),
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


def method_age_summary(run: dict[str, Any]) -> dict[str, Any]:
    selected = [
        record
        for record in run["records"]
        if record["candidate_old_source_age_seconds"] is not None
    ]
    ages = np.asarray(
        [record["candidate_old_source_age_seconds"] for record in selected],
        dtype=np.float64,
    )
    errors = np.asarray(
        [record["candidate_old_absolute_age_error_seconds"] for record in selected],
        dtype=np.float64,
    )
    lags = [int(record["old_chunk_offset"]) for record in selected]
    return {
        "selected_steps": len(selected),
        "lag_counts": {
            str(lag): lags.count(lag) for lag in sorted(set(lags))
        },
        "candidate_old_source_age_mean_seconds": float(ages.mean()),
        "candidate_old_source_age_median_seconds": float(np.median(ages)),
        "candidate_old_source_age_min_seconds": float(ages.min()),
        "candidate_old_source_age_max_seconds": float(ages.max()),
        "candidate_old_absolute_error_mean_seconds": float(errors.mean()),
        "candidate_old_absolute_error_max_seconds": float(errors.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--decisions", type=int, default=20)
    args = parser.parse_args()
    if args.decisions > 20:
        raise ValueError("physical-age canary is limited to 20 decisions")

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
        evaluator,
        model,
        task_args,
        method="NEWEST",
        seed=args.seed,
        decisions=args.decisions,
    )
    newest_second = run_method(
        evaluator,
        model,
        task_args,
        method="NEWEST",
        seed=args.seed,
        decisions=args.decisions,
    )
    determinism = compare_runs(newest_first, newest_second)
    full_old = run_method(
        evaluator,
        model,
        task_args,
        method="FULL_OLD_1S",
        seed=args.seed,
        decisions=args.decisions,
    )
    fo = run_method(
        evaluator,
        model,
        task_args,
        method="FO_1S",
        seed=args.seed,
        decisions=args.decisions,
    )
    runs = {
        "newest_first": newest_first,
        "newest_second": newest_second,
        "full_old_1s": full_old,
        "fo_1s": fo,
    }
    output = {
        "scope": "Outcome-blind physical-age closed-loop provenance canaries",
        "task": "beat_block_hammer",
        "seed": args.seed,
        "checkpoint": str(args.checkpoint_dir / "policy_last.ckpt"),
        "methods": list(PHYSICAL_AGE_METHODS),
        "target_source_age_seconds": PHYSICAL_SOURCE_AGE_SECONDS,
        "action_groups": {name: list(indices) for name, indices in ACTION_GROUPS.items()},
        "warmup_rule": "NEWEST only when no valid past query exists (decision 0)",
        "source_rule": (
            "argmin over q<t, t-q<50 of abs((T_t-T_q)-1.0s); "
            "ties choose the more recent q"
        ),
        "same_current_decision_target_rule": (
            "old candidate is chunk_from_query_q[t-q]"
        ),
        "executed_source_age_contract": {
            "NEWEST": "all channels execute source age 0",
            "FULL_OLD_1S": "all channels execute the q* candidate age",
            "FO_1S": "arms execute source age 0; grippers execute the q* candidate age",
        },
        "determinism": determinism,
        "q_star_candidate_age_summaries": {
            name: method_age_summary(run) for name, run in runs.items()
        },
        **runs,
        "mechanical_provenance_pass": all(
            record["mechanical_provenance_pass"]
            for run in runs.values()
            for record in run["records"]
        ),
        "gripper_hold_used": False,
        "temporal_method_success_inspected_or_recorded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_ready(output), indent=2) + "\n")
    print(
        json.dumps(
            {
                "determinism": determinism,
                "candidate_ages": output["q_star_candidate_age_summaries"],
            },
            indent=2,
        )
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
