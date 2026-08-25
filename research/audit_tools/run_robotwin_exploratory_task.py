#!/usr/bin/env python3
"""Run one task's sealed RoboTwin exploratory cells in preregistered order."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from research.audit_tools.robotwin_temporal_reuse import (
    ACTION_GROUPS,
    ARM_GROUPS,
    GRIPPER_GROUPS,
    GRIPPER_CONTROL_METHODS,
    RoboTwinGripperControlExecutor,
    RoboTwinPhysicalAgeExecutor,
    postprocess_action,
)
from research.audit_tools.run_robotwin_closed_loop_canaries import (
    CountingScene,
    PHYSICS_HZ,
    infer_full_chunk,
)


MAX_ATTEMPTS = 3
OLD_SOURCE_METHODS = ("FULL_OLD_1S", "FO_1S")
PENDING_ARTIFACT = "PENDING_ARTIFACT_COMPLETION"


class ProvenanceFailure(RuntimeError):
    pass


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: Any, *, sealed: bool = False) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)
    if sealed:
        path.chmod(0o600)


def require_exact(actual: np.ndarray, expected: np.ndarray, message: str) -> None:
    if not np.array_equal(actual, expected):
        raise ProvenanceFailure(message)


def setup_environment(evaluator: Any, task_args: dict[str, Any], cell: dict[str, Any]):
    task_env = evaluator.class_decorator(cell["task"])
    args = dict(task_args)
    args["eval_mode"] = True
    args["render_freq"] = 0
    args["eval_video_log"] = False
    try:
        task_env.setup_demo(
            now_ep_num=cell["eligible_seed_index"],
            seed=cell["robotwin_seed"],
            is_test=True,
            **args,
        )
        task_env.set_instruction(instruction=cell["task"])
        task_env.scene = CountingScene(task_env.scene)
        return task_env
    except Exception:
        try:
            task_env.close_env(clear_cache=True)
        except Exception:
            pass
        raise


def close_environment(task_env: Any) -> None:
    try:
        task_env.close_env(clear_cache=True)
    finally:
        del task_env
        torch.cuda.empty_cache()


def group_contract(
    method: str,
    decision: int,
    old_source: int | None,
    old_offset: int | None,
) -> tuple[dict[str, Any], dict[str, int]]:
    chunk_source: dict[str, Any] = {}
    chunk_offset: dict[str, int] = {}
    for group in ACTION_GROUPS:
        if method == "FULL_OLD_1S" and old_source is not None:
            chunk_source[group] = old_source
            chunk_offset[group] = old_offset
        elif method == "FO_1S" and old_source is not None and group in GRIPPER_GROUPS:
            chunk_source[group] = old_source
            chunk_offset[group] = old_offset
        elif method == "GRIPPER_HOLD" and group in GRIPPER_GROUPS and decision > 0:
            chunk_source[group] = "PREVIOUS_EXECUTED_GRIPPER"
            chunk_offset[group] = 0
        elif method == "GRIPPER_EMA_1S" and group in GRIPPER_GROUPS and decision > 0:
            chunk_source[group] = "PHYSICAL_TIME_EMA_STATE"
            chunk_offset[group] = 0
        else:
            chunk_source[group] = decision
            chunk_offset[group] = 0
    return chunk_source, chunk_offset


def experimental_action(
    method: str,
    model: Any,
    decision: int,
    query_time: float,
    normalized_chunk: np.ndarray,
    physical_executor: RoboTwinPhysicalAgeExecutor | None,
    gripper_executor: RoboTwinGripperControlExecutor | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    mean = model.model.stats["action_mean"]
    std = model.model.stats["action_std"]
    fresh = postprocess_action(normalized_chunk[0], mean, std).astype(np.float32)
    old = None
    old_source = None
    old_offset = None
    candidate_age = None
    candidate_error = None
    hold_or_ema = None

    if method == "NEWEST":
        executed = fresh.copy()
    elif method in OLD_SOURCE_METHODS:
        temporal = physical_executor.update(
            decision,
            normalized_chunk,
            query_time_seconds=query_time,
        )
        old_source = temporal.old_source_step
        old_offset = temporal.old_chunk_offset
        candidate_age = temporal.realized_source_age_seconds
        candidate_error = temporal.absolute_age_error_seconds
        old = (
            None
            if temporal.old_action is None
            else postprocess_action(temporal.old_action, mean, std).astype(np.float32)
        )
        executed = postprocess_action(temporal.action, mean, std).astype(np.float32)
        if old is None:
            require_exact(executed, fresh, "physical-age warmup is not NEWEST")
        elif method == "FULL_OLD_1S":
            require_exact(executed, old, "FULL_OLD_1S did not execute old q*[k]")
        elif method == "FO_1S":
            for group in ARM_GROUPS:
                indices = list(ACTION_GROUPS[group])
                require_exact(
                    executed[indices], fresh[indices], "FO_1S arm is not fresh"
                )
            for group in GRIPPER_GROUPS:
                indices = list(ACTION_GROUPS[group])
                require_exact(
                    executed[indices], old[indices], "FO_1S gripper is not q*[k]"
                )
    elif method in GRIPPER_CONTROL_METHODS:
        controlled = gripper_executor.update(
            decision,
            fresh,
            query_time_seconds=query_time,
        )
        executed = controlled.action
        hold_or_ema = {
            "previous_executed_grippers": (
                None
                if controlled.previous_executed_grippers is None
                else controlled.previous_executed_grippers.tolist()
            ),
            "executed_grippers": controlled.executed_grippers.tolist(),
            "ema_alpha": controlled.ema_alpha,
            "ema_tau_seconds": (
                1.0 if method == "GRIPPER_EMA_1S" else None
            ),
        }
    else:
        raise ProvenanceFailure(f"unsupported experimental method {method}")

    if executed.shape != (14,) or not np.isfinite(executed).all():
        raise ProvenanceFailure("executed action is not one finite 14-D vector")
    chunk_source, chunk_offset = group_contract(
        method, decision, old_source, old_offset
    )
    provenance = {
        "decision_index": decision,
        "simulator_query_timestamp_seconds": query_time,
        "method": method,
        "fresh_query_id": decision,
        "selected_q_star": old_source,
        "k_t_minus_q": old_offset,
        "candidate_old_source_age_seconds": candidate_age,
        "candidate_old_absolute_error_seconds": candidate_error,
        "chunk_source_per_group": chunk_source,
        "chunk_offset_per_group": chunk_offset,
        "fresh_14d_candidate": fresh.tolist(),
        "old_14d_candidate": None if old is None else old.tolist(),
        "executed_14d_action": executed.tolist(),
        "gripper_control_state": hold_or_ema,
        "executed_source_age_seconds_per_group": {
            group: (
                candidate_age
                if candidate_age is not None
                and (
                    method == "FULL_OLD_1S"
                    or (method == "FO_1S" and group in GRIPPER_GROUPS)
                )
                else 0.0
            )
            for group in ACTION_GROUPS
        },
        "same_current_decision_target_asserted": method in OLD_SOURCE_METHODS,
    }
    return executed, provenance


def run_cell(
    evaluator: Any,
    model: Any,
    task_args: dict[str, Any],
    cell: dict[str, Any],
    attempt_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    method = cell["method"]
    model.reset()
    physical_executor = (
        RoboTwinPhysicalAgeExecutor(method) if method in OLD_SOURCE_METHODS else None
    )
    gripper_executor = (
        RoboTwinGripperControlExecutor(method)
        if method in GRIPPER_CONTROL_METHODS
        else None
    )
    task_env = setup_environment(evaluator, task_args, cell)
    provenance_path = attempt_dir / "provenance.jsonl.gz"
    decision_count = 0
    try:
        with gzip.open(provenance_path, "wt", encoding="utf-8") as provenance_file:
            while not (task_env.eval_success or task_env.take_action_cnt >= task_env.step_lim):
                decision = decision_count
                query_time = task_env.scene.physics_steps / PHYSICS_HZ
                observation = task_env.get_obs()
                xpolicy_observation = evaluator.robotwin_obs_to_xpolicylab(
                    observation,
                    instruction=cell["task"],
                    env_idx=0,
                    frequency=30,
                    task_env=task_env,
                )

                if method == "NATIVE_ACT":
                    if not model.model.temporal_agg or model.model.query_frequency != 1:
                        raise ProvenanceFailure("official native ACT aggregation is disabled")
                    model.update_obs(xpolicy_observation)
                    native_t_before = int(model.model.t)
                    action_dict = model.get_action()[0]
                    executed, action_type = evaluator.xpolicylab_action_to_robotwin(
                        action_dict,
                        action_type="joint",
                        current_observation=observation,
                    )
                    executed = np.asarray(executed, dtype=np.float32)
                    provenance = {
                        "decision_index": decision,
                        "simulator_query_timestamp_seconds": query_time,
                        "method": method,
                        "native_temporal_aggregation_enabled": True,
                        "native_query_frequency_decisions": 1,
                        "native_model_t_before": native_t_before,
                        "native_model_t_after": int(model.model.t),
                        "official_action_postprocessing": True,
                        "executed_14d_action": executed.tolist(),
                    }
                else:
                    encoded = model.encode_obs(
                        xpolicy_observation,
                        "joint",
                        model.robot_action_dim_info,
                    )
                    model.model.update_obs(encoded)
                    normalized_chunk = infer_full_chunk(model)
                    executed, provenance = experimental_action(
                        method,
                        model,
                        decision,
                        query_time,
                        normalized_chunk,
                        physical_executor,
                        gripper_executor,
                    )
                    action_type = "qpos"

                if executed.shape != (14,) or not np.isfinite(executed).all():
                    raise ProvenanceFailure("action shape/finite assertion failed")
                physics_before = task_env.scene.physics_steps
                task_env.take_action(executed, action_type=action_type)
                physics_after = task_env.scene.physics_steps
                provenance["internal_physics_steps"] = physics_after - physics_before
                provenance["simulator_time_after_execution_seconds"] = (
                    physics_after / PHYSICS_HZ
                )
                provenance_file.write(json.dumps(provenance, separators=(",", ":")) + "\n")
                provenance_file.flush()
                decision_count += 1
    finally:
        success = bool(task_env.eval_success)
        horizon_reached = task_env.take_action_cnt >= task_env.step_lim
        close_environment(task_env)

    outcome = {
        "cell_id": cell["cell_id"],
        "cell_key": hashlib.sha256(cell["cell_id"].encode()).hexdigest(),
        "success": success,
    }
    technical_status = {
        "cell_id": cell["cell_id"],
        "cell_key": outcome["cell_key"],
        "task": cell["task"],
        "eligible_seed_index": cell["eligible_seed_index"],
        "robotwin_seed": cell["robotwin_seed"],
        "method": method,
        "checkpoint_sha256": cell["checkpoint_sha256"],
        "config_sha256": cell["config_sha256"],
        "decision_count": decision_count,
        "episode_completion": bool(success or horizon_reached),
        "provenance_path": str(provenance_path),
        "provenance_assertions_passed": True,
        "outcome_stored_separately": True,
    }
    return outcome, technical_status


def load_model(robotwin_root: Path, checkpoint_dir: Path, task: str) -> Any:
    from XPolicyLab.policy.ACT.model import Model

    deploy = yaml.safe_load(
        (robotwin_root / "XPolicyLab/policy/ACT/deploy.yml").read_text()
    )
    deploy.update(
        {
            "ckpt_dir": str(checkpoint_dir),
            "ckpt_name": checkpoint_dir.name,
            "bench_name": "RoboTwin",
            "task_name": task,
            "env_cfg_type": "aloha_agilex",
            "action_type": "joint",
            "action_dim": 14,
            "device": "cuda:0",
            "temporal_agg": True,
        }
    )
    return Model(deploy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()

    schedule = json.loads(args.schedule.read_text())
    if schedule["cells_sha256"] != canonical_sha256(schedule["cells"]):
        raise RuntimeError("schedule cell hash mismatch")
    task_cells = sorted(
        [cell for cell in schedule["cells"] if cell["task"] == args.task],
        key=lambda cell: cell["within_task_run_order"],
    )
    if len(task_cells) != 120:
        raise RuntimeError(f"expected 120 cells for {args.task}")
    checkpoint_contract = schedule["checkpoint_contracts"][args.task]
    checkpoint_path = Path(checkpoint_contract["checkpoint_path"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    resolved_checkpoint_sha256 = file_sha256(checkpoint_path)
    expected_checkpoint_sha256 = checkpoint_contract["checkpoint_sha256"]
    if expected_checkpoint_sha256 == PENDING_ARTIFACT:
        if args.task != "open_laptop":
            raise RuntimeError("pending checkpoint identity is only valid for open_laptop")
    elif resolved_checkpoint_sha256 != expected_checkpoint_sha256:
        raise RuntimeError("checkpoint no longer matches preregistered hash")

    sys.path.insert(0, str(args.robotwin_root / "scripts"))
    sys.path.insert(0, str(args.robotwin_root))
    import eval_policy_xpolicylab as evaluator

    os.environ["ACT_ACTION_DIM"] = "14"
    model = load_model(args.robotwin_root, checkpoint_path.parent, args.task)
    task_usr_args = {
        "task_name": args.task,
        "task_config": "demo_clean",
        "ckpt_setting": checkpoint_path.parent.name,
        "policy_name": "ACT",
    }
    task_args, _ = evaluator.load_task_args(task_usr_args)
    task_args["eval_instruction"] = "seen"

    result_root = args.result_root / schedule["cells_sha256"]
    cells_root = result_root / "cells"
    outcomes_root = result_root / "sealed_outcomes"
    cells_root.mkdir(parents=True, exist_ok=True)
    outcomes_root.mkdir(parents=True, exist_ok=True)
    for cell in task_cells:
        cell_key = hashlib.sha256(cell["cell_id"].encode()).hexdigest()
        cell_root = cells_root / cell_key
        cell_root.mkdir(exist_ok=True)
        status_path = cell_root / "technical_status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get("state") == "COMPLETE":
                continue
            if status.get("state") == "PROVENANCE_FAILURE":
                raise ProvenanceFailure(f"pilot already halted at {cell_key}")

        completed = False
        for attempt in range(1, MAX_ATTEMPTS + 1):
            attempt_dir = cell_root / f"attempt-{attempt}"
            if attempt_dir.exists():
                continue
            attempt_dir.mkdir()
            try:
                outcome, technical = run_cell(
                    evaluator,
                    model,
                    task_args,
                    cell,
                    attempt_dir,
                )
            except ProvenanceFailure as error:
                failure = {
                    "state": "PROVENANCE_FAILURE",
                    "attempt": attempt,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                write_json(attempt_dir / "technical_failure.json", failure)
                write_json(status_path, {"cell_key": cell_key, **failure})
                raise
            except Exception as error:
                failure = {
                    "state": "INFRASTRUCTURE_FAILURE",
                    "attempt": attempt,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                write_json(attempt_dir / "technical_failure.json", failure)
                write_json(status_path, {"cell_key": cell_key, **failure})
                print(
                    f"technical retry {attempt}/{MAX_ATTEMPTS} cell={cell_key} "
                    f"error={type(error).__name__}",
                    flush=True,
                )
                continue

            outcome_path = outcomes_root / f"{cell_key}.json"
            write_json(outcome_path, outcome, sealed=True)
            technical["resolved_checkpoint_sha256"] = resolved_checkpoint_sha256
            write_json(
                status_path,
                {
                    "state": "COMPLETE",
                    "attempt": attempt,
                    **technical,
                    "sealed_outcome_path": str(outcome_path),
                },
            )
            print(f"technical complete cell={cell_key}", flush=True)
            completed = True
            break
        if not completed:
            write_json(
                status_path,
                {
                    "state": "PERSISTENT_INFRASTRUCTURE_FAILURE",
                    "cell_key": cell_key,
                    "attempts": MAX_ATTEMPTS,
                },
            )
            raise RuntimeError(f"persistent infrastructure failure at {cell_key}")

    print(f"task technical complete: {args.task} cells={len(task_cells)}", flush=True)


if __name__ == "__main__":
    main()
