#!/usr/bin/env python3
"""Bounded pre-rollout semantics and live evaluator smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
TRACK_A_ROOT = REPO_ROOT / "experiments" / "icra27_crosssuite_query_allocation"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(ROOT))

from phase1_conditions import ACTION_DIM, ARM, GRIPPER, CONDITION_ORDER, CONDITIONS, make_fixed_executor  # noqa: E402
from run_phase1 import atomic_json, checkpoints, effective_protocol, frozen_commit  # noqa: E402


def tagged_chunk(query_index: int, chunk_size: int = 100) -> np.ndarray:
    chunk = np.empty((chunk_size, ACTION_DIM), dtype=np.float64)
    for offset in range(chunk_size):
        for dim in range(ACTION_DIM):
            chunk[offset, dim] = 100000 * query_index + 100 * offset + dim
    return chunk


def static_checks() -> dict[str, Any]:
    if ARM != tuple(range(6)) or GRIPPER != (6,):
        raise AssertionError("action groups are not arm=0:6 and gripper=6")
    traces: dict[str, Any] = {}
    decisions_by_method = {}
    for method in CONDITION_ORDER:
        executor = make_fixed_executor(method, 100)
        query_steps = []
        decisions = []
        for t in range(65):
            def query(t: int = t) -> np.ndarray:
                query_steps.append(t)
                return tagged_chunk(t)

            decisions.append(executor.step(query))
        expected = list(range(0, 65, min(CONDITIONS[method].arm_horizon, CONDITIONS[method].gripper_horizon)))
        if query_steps != expected:
            raise AssertionError(f"{method} static query schedule mismatch")
        traces[method] = {"query_steps": query_steps}
        decisions_by_method[method] = decisions

    if traces["H8"]["query_steps"] != traces["ARM8_GRIP32"]["query_steps"]:
        raise AssertionError("H8 and ARM8_GRIP32 static full-policy schedules differ")
    for method in ("ARM8_GRIP16", "ZOH8_GRIP16"):
        if traces["H8"]["query_steps"] != traces[method]["query_steps"]:
            raise AssertionError(f"H8 and {method} static full-policy schedules differ")
    a8_t8 = decisions_by_method["ARM8_GRIP32"][8]
    h8_t8 = decisions_by_method["H8"][8]
    if a8_t8.refreshed_groups != ("arm",) or a8_t8.source_ages != {"arm": 0, "gripper": 8}:
        raise AssertionError("ARM8_GRIP32 t=8 component refresh semantics are wrong")
    if not np.array_equal(a8_t8.action[:6], tagged_chunk(8)[0, :6]):
        raise AssertionError("ARM8_GRIP32 arm does not use the new query at t=8")
    if a8_t8.action[6] != tagged_chunk(0)[8, 6]:
        raise AssertionError("ARM8_GRIP32 gripper does not preserve the q=0 chunk at t=8")
    if not np.array_equal(h8_t8.action, tagged_chunk(8)[0]):
        raise AssertionError("H8 does not coherently refresh at t=8")
    a8g16_t8 = decisions_by_method["ARM8_GRIP16"][8]
    if a8g16_t8.source_ages != {"arm": 0, "gripper": 8} or a8g16_t8.action[6] != tagged_chunk(0)[8, 6]:
        raise AssertionError("ARM8_GRIP16 retained-chunk progression is wrong at t=8")
    zoh_t8 = decisions_by_method["ZOH8_GRIP16"][8]
    zoh_t15 = decisions_by_method["ZOH8_GRIP16"][15]
    zoh_t16 = decisions_by_method["ZOH8_GRIP16"][16]
    if zoh_t8.action[6] != tagged_chunk(0)[0, 6] or zoh_t15.action[6] != tagged_chunk(0)[0, 6]:
        raise AssertionError("ZOH8_GRIP16 did not hold gripper index 0 for 16 steps")
    if zoh_t16.action[6] != tagged_chunk(16)[0, 6] or zoh_t16.source_positions["gripper"] != 0:
        raise AssertionError("ZOH8_GRIP16 did not refresh from fresh gripper index 0 at t=16")

    protocol = effective_protocol()
    for task_id in protocol["task_ids"]:
        selected = set(protocol["state_ids_by_task"][str(task_id)])
        if selected != set(range(15, 50)):
            raise AssertionError(f"task {task_id} does not use the full amended state range 15-49")

    checkpoint_chunk_sizes = {}
    for task_id, checkpoint in checkpoints().items():
        cfg = json.loads((Path(checkpoint) / "config.json").read_text(encoding="utf-8"))
        chunk_size = int(cfg["chunk_size"])
        if chunk_size < 32 or cfg.get("temporal_ensemble_coeff") is not None:
            raise AssertionError(f"task {task_id} checkpoint violates chunk/TE contract")
        checkpoint_chunk_sizes[str(task_id)] = chunk_size
    prior_smoke = json.loads((TRACK_A_ROOT / "phase0_smoke" / "libero_10.json").read_text(encoding="utf-8"))
    if prior_smoke["status"] != "PASS" or prior_smoke["scientific_outcomes_used"]:
        raise AssertionError("validated Track-A LIBERO-10 smoke contract is unavailable")
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    official_counts = {str(task_id): len(suite.get_task_init_states(task_id)) for task_id in range(10)}
    if set(official_counts.values()) != {50}:
        raise AssertionError(f"official state count is not 50 for all tasks: {official_counts}")
    return {
        "status": "PASS",
        "action_groups": {"arm": list(ARM), "gripper": list(GRIPPER)},
        "static_traces": traces,
        "checkpoint_chunk_sizes": checkpoint_chunk_sizes,
        "held_out_overlap_with_track_a_0_14": 0,
        "official_state_bounds": [0, 49],
        "official_state_count_by_task": official_counts,
        "validated_track_a_smoke_status": prior_smoke["status"],
    }


def live_checks() -> dict[str, Any]:
    # State 0 is already exposed and excluded from the scientific cohort.
    sys.path.insert(0, str(TRACK_A_ROOT))
    from run_track_a import Runtime

    checkpoint = checkpoints()[0]
    base = {
        "block_id": "technical-smoke-libero_10-task00-state00",
        "suite": "libero_10",
        "task_id": 0,
        "state_id": 0,
        "environment_seed": 390000,
        "policy_seed": 424242,
        "checkpoint": checkpoint,
        "control_frequency_hz": 10,
        "max_episode_steps": 20,
        "preregistration_commit": frozen_commit(),
    }
    runtime = Runtime("0")

    def run(method: str, suffix: str) -> dict[str, Any]:
        condition = CONDITIONS[method]
        cell = {
            **base,
            "cell_id": f"technical-smoke-{method}-{suffix}",
            "method": method,
            "strategy": condition.strategy,
            "arm_horizon": condition.arm_horizon,
            "gripper_horizon": condition.gripper_horizon,
        }
        return runtime.run(cell, executor_override=lambda chunk_size: make_fixed_executor(method, chunk_size))

    try:
        h8_first = run("H8", "repeat1")
        h8_second = run("H8", "repeat2")
        a8g32 = run("ARM8_GRIP32", "schedule")
        a8g16 = run("ARM8_GRIP16", "schedule")
        zoh8g16 = run("ZOH8_GRIP16", "schedule")
    finally:
        runtime.drop_policy()

    expected = [0, 8, 16]
    for result in (h8_first, h8_second, a8g32, a8g16, zoh8g16):
        if result["environment_steps"] != 20 or result["query_steps"] != expected:
            raise AssertionError("live actual query schedule mismatch")
        if result["temporal_ensemble_coeff"] is not None or not result["fresh_environment_per_condition"]:
            raise AssertionError("live TE/fresh-environment contract mismatch")
    if any(h8_first["initial_sim_state"] != result["initial_sim_state"] for result in (h8_second, a8g32, a8g16, zoh8g16)):
        raise AssertionError("paired reset did not reproduce the exact initial simulator state")
    if h8_first["executed_actions"] != h8_second["executed_actions"]:
        raise AssertionError("repeated H8 action trajectory is not deterministic")
    if any(h8_first["query_steps"] != result["query_steps"] for result in (a8g32, a8g16, zoh8g16)):
        raise AssertionError("actual arm8-condition query schedules differ")
    return {
        "status": "PASS",
        "technical_cell": {"suite": "libero_10", "task_id": 0, "state_id": 0, "max_episode_steps": 20},
        "scientific_cohort_member": False,
        "success_outcomes_recorded": False,
        "fresh_environment_instances": 5,
        "h8_repeat_exact_action_match": True,
        "initial_sim_state_exact_match_across_runs": True,
        "h8_actual_query_steps": h8_first["query_steps"],
        "arm8_grip32_actual_query_steps": a8g32["query_steps"],
        "arm8_grip16_actual_query_steps": a8g16["query_steps"],
        "zoh8_grip16_actual_query_steps": zoh8g16["query_steps"],
        "actual_schedule_equivalence": True,
        "temporal_ensemble_disabled": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    output = {"static": static_checks()}
    if not args.static_only:
        output["live"] = live_checks()
    output["status"] = "PASS"
    if args.static_only:
        print(json.dumps(output, indent=2))
    else:
        atomic_json(ROOT / "smoke_test.json", output)
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
