#!/usr/bin/env python3
"""Real exposed-state identity, reload/RNG, and runtime smoke for one suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from conditions import ACTION_DIM, ARM, GRIPPER
from run_track_a import ROOT, Runtime, atomic_json
from one_clock import ActionGroup, FixedChunkExecutor


SPECS = {
    "libero_spatial": (0, 10, 2000),
    "libero_goal": (2, 10, 2000),
    "libero_10": (3, 10, 2000),
}


def group_diagonal(horizon: int):
    def factory(chunk_size: int):
        groups = (ActionGroup("arm", ARM, horizon), ActionGroup("gripper", GRIPPER, horizon))
        return FixedChunkExecutor.groupwise_fixed(action_dim=ACTION_DIM, chunk_size=chunk_size, groups=groups)
    return factory


def equal_trajectory(first: dict[str, Any], second: dict[str, Any]) -> dict[str, bool]:
    checks = {
        "executed_actions": np.array_equal(np.asarray(first["executed_actions"]), np.asarray(second["executed_actions"])),
        "initial_sim_state": first["initial_sim_state"] == second["initial_sim_state"],
        "episode_length": first["environment_steps"] == second["environment_steps"],
        "terminal_success": first["success"] == second["success"],
        "query_steps": first["query_steps"] == second["query_steps"],
        "source_ages": first["source_ages"] == second["source_ages"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"trajectory identity failed: {checks}")
    return checks


def cell(suite: str, task_id: int, state_id: int, seed: int, method: str, checkpoint: str) -> dict[str, Any]:
    horizons = {"H16": (16,16), "H4": (4,4), "ARM4_GRIP32": (4,32), "H2": (2,2), "ARM2_GRIP16": (2,16), "TE_DENSE": (None,None)}
    arm, grip = horizons[method]
    return {
        "cell_id": f"smoke-{suite}-task{task_id}-state{state_id}-{method}", "block_id": f"smoke-{suite}-task{task_id}-state{state_id}",
        "suite": suite, "task_id": task_id, "state_id": state_id, "environment_seed": seed,
        "policy_seed": 424242, "method": method, "strategy": "technical_smoke",
        "arm_horizon": arm, "gripper_horizon": grip, "checkpoint": checkpoint,
        "preregistration_commit": "PHASE0_TECHNICAL_SMOKE", "control_frequency_hz": 10,
        "max_episode_steps": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=tuple(SPECS), required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    suite = args.suite
    task_id, state_id, seed = SPECS[suite]
    checkpoint = f"/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/{suite}_task{task_id}/checkpoints/100000/pretrained_model"
    runtime = Runtime(args.gpu)
    results: dict[str, dict[str, Any]] = {}
    try:
        h4a = runtime.run(cell(suite, task_id, state_id, seed, "H4", checkpoint))
        runtime.drop_policy()
        h4b = runtime.run(cell(suite, task_id, state_id, seed, "H4", checkpoint))
        reload_checks = equal_trajectory(h4a, h4b)
        h4_group = runtime.run(cell(suite, task_id, state_id, seed, "H4", checkpoint), executor_override=group_diagonal(4))
        h4_checks = equal_trajectory(h4a, h4_group)
        h2 = runtime.run(cell(suite, task_id, state_id, seed, "H2", checkpoint))
        h2_group = runtime.run(cell(suite, task_id, state_id, seed, "H2", checkpoint), executor_override=group_diagonal(2))
        h2_checks = equal_trajectory(h2, h2_group)
        results["H4"] = h4a
        results["H2"] = h2
        for method in ("H16", "ARM4_GRIP32", "ARM2_GRIP16", "TE_DENSE"):
            results[method] = runtime.run(cell(suite, task_id, state_id, seed, method, checkpoint))
    finally:
        runtime.drop_policy()
    timing = {method: {"steps": row["environment_steps"], "resolved_max_episode_steps": row["resolved_max_episode_steps"], "wall_clock_seconds": row["wall_clock_seconds"], "seconds_per_env_step": row["wall_clock_seconds"] / row["environment_steps"], "policy_queries": row["policy_queries"], "query_rate": row["query_rate"]} for method, row in results.items()}
    output = {
        "status": "PASS", "suite": suite, "task_id": task_id, "state_id": state_id,
        "cell_role": "technical exposed state; excluded from Track A confirmation",
        "H4_equals_group_arm4_grip4": h4_checks,
        "H2_equals_group_arm2_grip2": h2_checks,
        "repeated_load_unload_and_rng_isolation": reload_checks,
        "timing": timing,
        "scientific_outcomes_used": False,
    }
    atomic_json(ROOT / "phase0_smoke" / f"{suite}.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
