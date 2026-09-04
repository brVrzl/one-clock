#!/usr/bin/env python3
"""Run the frozen Phase-1 LIBERO-10 five-condition queue."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
TRACK_A_ROOT = REPO_ROOT / "experiments" / "icra27_crosssuite_query_allocation"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(ROOT))

from phase1_conditions import CONDITION_ORDER, CONDITIONS, make_fixed_executor  # noqa: E402

# Import the validated evaluator after our condition module. run_track_a adds its
# own directory before importing its frozen Track-A condition definitions.
sys.path.insert(0, str(TRACK_A_ROOT))
from run_track_a import Runtime  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def effective_protocol() -> dict[str, Any]:
    protocol = load_json(ROOT / "protocol.json")
    amendment = load_json(ROOT / "amendment_20260904T104520+0800.json")
    interpretation_addendum = load_json(ROOT / "interpretation_addendum_20260904T105029+0800.json")
    design = amendment["effective_design"]
    statistics = amendment["statistics"]
    statistics["paired_bootstrap_seeds"][interpretation_addendum["interpretation_only_contrast"]] = interpretation_addendum["paired_bootstrap_seed"]
    statistics["task_cluster_bootstrap_seeds"][interpretation_addendum["interpretation_only_contrast"]] = interpretation_addendum["task_cluster_bootstrap_seed"]
    protocol.update(
        {
            "status": amendment["status"],
            "amendment": amendment["timestamp"],
            "suite": design["suite"],
            "task_ids": design["task_ids"],
            "states_per_task": design["states_per_task"],
            "paired_blocks": design["paired_blocks"],
            "conditions_per_block": design["conditions_per_block"],
            "scientific_episodes": design["scientific_episodes"],
            "state_ids_by_task": {str(task_id): list(design["state_ids"]) for task_id in design["task_ids"]},
            "condition_order": design["condition_order"],
            "conditions": design["conditions"],
            "contrasts": amendment["effective_contrasts"],
            "statistics": statistics,
            "decision_rule": amendment["effective_decision_rule"],
            "interpretation_addendum": interpretation_addendum,
        }
    )
    return protocol


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_marker(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def frozen_commit() -> str:
    path = ROOT / "PREREGISTRATION_COMMIT"
    sha = path.read_text(encoding="utf-8").strip()
    if len(sha) != 40:
        raise RuntimeError("PREREGISTRATION_COMMIT is not a full commit ID")
    subprocess.run(["git", "merge-base", "--is-ancestor", sha, "HEAD"], cwd=REPO_ROOT, check=True)
    return sha


def checkpoints() -> dict[int, str]:
    inventory = load_json(TRACK_A_ROOT / "checkpoint_inventory.json")
    result: dict[int, str] = {}
    for policy in inventory["policies"]:
        if policy["suite"] == "libero_10" and policy["load_smoke"]["succeeds"]:
            result[int(policy["task_id"])] = policy["exact_local_path"]
    if sorted(result) != list(range(10)):
        raise RuntimeError("validated checkpoint inventory does not contain all ten LIBERO-10 tasks")
    return result


def build_cells(protocol: dict[str, Any], preregistration_commit: str) -> list[dict[str, Any]]:
    checkpoint_by_task = checkpoints()
    cells = []
    for task_id in protocol["task_ids"]:
        task_id = int(task_id)
        for state_id in protocol["state_ids_by_task"][str(task_id)]:
            state_id = int(state_id)
            block_id = f"libero_10-task{task_id:02d}-state{state_id:02d}"
            for method in CONDITION_ORDER:
                condition = CONDITIONS[method]
                cells.append(
                    {
                        "cell_id": f"{block_id}-{method}",
                        "block_id": block_id,
                        "suite": "libero_10",
                        "task_id": task_id,
                        "state_id": state_id,
                        "environment_seed": 390000 + 100 * task_id + state_id,
                        "policy_seed": int(protocol["execution"]["policy_seed"]),
                        "method": method,
                        "strategy": condition.strategy,
                        "arm_horizon": condition.arm_horizon,
                        "gripper_horizon": condition.gripper_horizon,
                        "checkpoint": checkpoint_by_task[task_id],
                        "control_frequency_hz": int(protocol["execution"]["control_frequency_hz"]),
                        "max_episode_steps": None,
                        "preregistration_commit": preregistration_commit,
                    }
                )
    expected = int(protocol["scientific_episodes"])
    if len(cells) != expected:
        raise RuntimeError(f"constructed {len(cells)} cells, expected {expected}")
    return cells


def result_path(cell: dict[str, Any]) -> Path:
    return ROOT / "results" / f"{cell['cell_id']}.json"


def marker_path(cell: dict[str, Any], status: str = "complete") -> Path:
    return ROOT / "markers" / f"{cell['cell_id']}.{status}"


def attempt_path(cell: dict[str, Any]) -> Path:
    return ROOT / "attempts" / f"{cell['cell_id']}.json"


def validate_result(cell: dict[str, Any], path: Path) -> dict[str, Any]:
    value = load_json(path)
    for key in (
        "cell_id",
        "block_id",
        "suite",
        "task_id",
        "state_id",
        "environment_seed",
        "policy_seed",
        "method",
        "strategy",
        "arm_horizon",
        "gripper_horizon",
        "checkpoint",
        "preregistration_commit",
    ):
        if value.get(key) != cell.get(key):
            raise ValueError(f"{key} mismatch for {cell['cell_id']}")
    if value.get("status") != "COMPLETE":
        raise ValueError("result status is not COMPLETE")
    steps = int(value["environment_steps"])
    queries = int(value["policy_queries"])
    if not 1 <= steps <= int(value["resolved_max_episode_steps"]):
        raise ValueError("invalid environment step count")
    if queries != int(value["model_forward_count"]):
        raise ValueError("policy-call count differs from model-forward count")
    expected_queries = list(range(0, steps, min(cell["arm_horizon"], cell["gripper_horizon"])))
    if value["query_steps"] != expected_queries or queries != len(expected_queries):
        raise ValueError("actual fixed-condition policy-query schedule mismatch")
    if len(value["executed_actions"]) != steps or len(value["source_ages"]) != steps:
        raise ValueError("step-level result log is incomplete")
    if value.get("temporal_ensemble_coeff") is not None:
        raise ValueError("temporal ensemble was not disabled")
    if not value.get("fresh_environment_per_condition"):
        raise ValueError("fresh-environment contract is absent")
    if int(value["action_dim"]) != 7 or int(value["chunk_size"]) < 32:
        raise ValueError("action dimension or chunk length drifted")
    return value


def is_complete(cell: dict[str, Any]) -> bool:
    path = result_path(cell)
    if not path.is_file() or not marker_path(cell).is_file():
        return False
    try:
        validate_result(cell, path)
    except Exception:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=3)
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.num_workers:
        raise ValueError("worker index is outside worker count")

    protocol = effective_protocol()
    if protocol["status"] != "FROZEN_PROSPECTIVE_AMENDMENT_BEFORE_PHASE1_HELD_OUT_OUTCOMES":
        raise RuntimeError("effective amended protocol is not frozen")
    if tuple(protocol["condition_order"]) != CONDITION_ORDER:
        raise RuntimeError("condition order drifted")
    preregistration_commit = frozen_commit()
    cells = build_cells(protocol, preregistration_commit)
    assigned_tasks = [task_id for task_id in protocol["task_ids"] if int(task_id) % args.num_workers == args.worker_index]
    assigned = [cell for cell in cells if int(cell["task_id"]) in assigned_tasks]
    progress_path = ROOT / "progress" / f"worker_{args.worker_index}.json"
    runtime = Runtime(args.gpu)
    current_task: int | None = None
    try:
        for cell in assigned:
            task_id = int(cell["task_id"])
            if current_task is not None and task_id != current_task:
                runtime.drop_policy()
            current_task = task_id
            if is_complete(cell) or marker_path(cell, "technical_failed").is_file():
                continue
            apath = attempt_path(cell)
            attempts = load_json(apath).get("attempts", []) if apath.is_file() else []
            while len(attempts) < int(protocol["execution"]["maximum_attempts_per_cell"]) and not is_complete(cell):
                atomic_json(
                    progress_path,
                    {
                        "pid": os.getpid(),
                        "gpu": args.gpu,
                        "worker_index": args.worker_index,
                        "cell_id": cell["cell_id"],
                        "attempt": len(attempts) + 1,
                        "state": "RUNNING",
                    },
                )
                try:
                    method = str(cell["method"])
                    result = runtime.run(
                        cell,
                        executor_override=lambda chunk_size, method=method: make_fixed_executor(method, chunk_size),
                    )
                    atomic_json(result_path(cell), result)
                    validate_result(cell, result_path(cell))
                    write_marker(marker_path(cell), "COMPLETE")
                except Exception as exc:
                    attempts.append(
                        {
                            "attempt": len(attempts) + 1,
                            "time": time.time(),
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    atomic_json(apath, {"cell_id": cell["cell_id"], "attempts": attempts})
                    runtime.drop_policy()
            if not is_complete(cell):
                write_marker(marker_path(cell, "technical_failed"), "TECHNICAL_FAILED")
    finally:
        runtime.drop_policy()
        gc.collect()
    atomic_json(
        progress_path,
        {
            "pid": os.getpid(),
            "gpu": args.gpu,
            "worker_index": args.worker_index,
            "state": "SHARD_COMPLETE",
            "assigned_tasks": assigned_tasks,
            "assigned_cells": len(assigned),
            "completed_cells": sum(is_complete(cell) for cell in assigned),
            "finished_at": time.time(),
        },
    )


if __name__ == "__main__":
    main()
