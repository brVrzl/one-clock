#!/usr/bin/env python3
"""Build the frozen, outcome-independent overnight queue manifest."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ACT_OBJECT = "/home/wjq/checkpoints/zeromidnight_act_libero_object"
SMOLVLA = "/home/wjq/checkpoints/HuggingFaceVLA_smolvla_libero"
TASK_ACT_ROOT = Path(
    "/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final"
)
SUITE_INDEX = {"libero_spatial": 0, "libero_object": 1, "libero_goal": 2, "libero_10": 3}
SMOL_SUITES = tuple(SUITE_INDEX)
POSTHOC_TASKS = {
    "libero_goal": (4, 6, 7, 8, 9),
    "libero_10": (0, 2, 4, 6, 7),
}


def cell(*, phase: str, policy: str, suite: str, task: int, state: int, seed: int,
         method: str, arm: int, grip: int, checkpoint: str, max_steps: int | None,
         exposure: str) -> dict:
    cid = f"{phase}__{suite}_task{task:02d}_state{state:02d}__{method}"
    return {
        "cell_id": cid,
        "phase": phase,
        "policy": policy,
        "suite": suite,
        "task_id": task,
        "state_id": state,
        "environment_seed": seed,
        "policy_seed_rule": "ACT=424242 per episode; SmolVLA=SHA256(policy,suite,task,state,environment_seed,query_step)",
        "method": method,
        "strategy": "global_fixed" if arm == grip else "groupwise_fixed",
        "arm_horizon": arm,
        "gripper_horizon": grip,
        "checkpoint": checkpoint,
        "evaluator": "LeRobot LiberoEnv 0.4.4, synchronous n_envs=1, native policy/environment processors",
        "max_episode_steps": max_steps,
        "control_mode": "relative",
        "control_frequency_hz": 20 if phase.startswith("act_object") else 30,
        "success_criterion": "terminal info.is_success; positive terminal reward fallback",
        "temporal_aggregation": False,
        "smoothing": False,
        "action_groups": {"arm": [0, 1, 2, 3, 4, 5], "gripper": [6]},
        "exposure": exposure,
    }


def main() -> None:
    cells: list[dict] = []
    dev_states = (20, 21, 22, 23, 27, 31, 34, 35, 38, 39, 44, 45, 47, 48)
    for task in range(1, 10):
        for state in dev_states:
            cells.append(cell(
                phase="act_object_h8_126", policy="ACT", suite="libero_object",
                task=task, state=state, seed=330000 + 100 * task + state,
                method="COHERENT_H8", arm=8, grip=8, checkpoint=ACT_OBJECT,
                max_steps=280, exposure="OUTCOME_EXPOSED",
            ))
    for suite, tasks in POSTHOC_TASKS.items():
        for task in tasks:
            checkpoint = str(TASK_ACT_ROOT / f"{suite}_task{task}" / "checkpoints/100000/pretrained_model")
            for state in range(14):
                cells.append(cell(
                    phase="act_posthoc_h8_140", policy="ACT", suite=suite,
                    task=task, state=state,
                    seed=340000 + 1000 * SUITE_INDEX[suite] + 100 * task + state,
                    method="COHERENT_H8", arm=8, grip=8, checkpoint=checkpoint,
                    max_steps=300 if suite == "libero_goal" else 520,
                    exposure="OUTCOME_EXPOSED_POST_HOC",
                ))
    for task in range(1, 10):
        for state in range(20):
            cells.append(cell(
                phase="act_arm4_grip32_180", policy="ACT", suite="libero_object",
                task=task, state=state, seed=1000 + state,
                method="ARM4_GRIP32", arm=4, grip=32, checkpoint=ACT_OBJECT,
                max_steps=280, exposure="OUTCOME_EXPOSED",
            ))
    for suite in SMOL_SUITES:
        for task in range(10):
            for state in range(4):
                seed = 360000 + 1000 * SUITE_INDEX[suite] + 100 * task + state
                for method, arm, grip in (
                    ("SMOLVLA_COHERENT_H8", 8, 8),
                    ("SMOLVLA_ARM8_GRIP16", 8, 16),
                ):
                    cells.append(cell(
                        phase="smolvla_primary", policy="SmolVLA", suite=suite,
                        task=task, state=state, seed=seed, method=method,
                        arm=arm, grip=grip, checkpoint=SMOLVLA, max_steps=None,
                        exposure="PREDECLARED_PRIMARY",
                    ))
    for suite in SMOL_SUITES:
        for task in range(10):
            for state in range(4):
                cells.append(cell(
                    phase="smolvla_capacity_h16", policy="SmolVLA", suite=suite,
                    task=task, state=state,
                    seed=360000 + 1000 * SUITE_INDEX[suite] + 100 * task + state,
                    method="SMOLVLA_COHERENT_H16", arm=16, grip=16,
                    checkpoint=SMOLVLA, max_steps=None,
                    exposure="PREDECLARED_CAPACITY_AFTER_PRIMARY_COMPLETION",
                ))
    expected = {
        "act_object_h8_126": 126,
        "act_posthoc_h8_140": 140,
        "act_arm4_grip32_180": 180,
        "smolvla_primary": 320,
        "smolvla_capacity_h16": 160,
    }
    observed = {phase: sum(c["phase"] == phase for c in cells) for phase in expected}
    assert observed == expected, (observed, expected)
    manifest = {
        "schema_version": 1,
        "frozen_from_commit": "7ea83e1c0bea4367cc722a3d7b72ac0ca827e009",
        "created_before_new_scientific_outcomes": True,
        "scientific_queue_is_outcome_independent": True,
        "retry_policy": {
            "maximum_attempts": 3,
            "allowed": ["crash", "CUDA error", "OOM", "simulator exception", "incomplete/corrupt result"],
            "forbidden": ["low success", "zero reward", "unusual trajectory", "surprising result", "method losing"],
        },
        "capacity_trigger": "launch iff all 320 smolvla_primary cells are COMPLETE or TECHNICAL_FAILED; do not inspect success",
        "stop_after": "smolvla_capacity_h16",
        "expected_counts": expected,
        "cells": cells,
    }
    (ROOT / "queue_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(ROOT / "queue_manifest.json"), "counts": observed, "total": len(cells)}))


if __name__ == "__main__":
    main()
