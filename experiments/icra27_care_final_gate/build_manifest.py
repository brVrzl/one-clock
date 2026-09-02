#!/usr/bin/env python3
"""Build the frozen final-gate and cross-policy robustness manifest."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ACT_OBJECT = "/home/wjq/checkpoints/zeromidnight_act_libero_object"
SMOLVLA = "/home/wjq/checkpoints/HuggingFaceVLA_smolvla_libero"
SUITE_INDEX = {"libero_spatial": 0, "libero_object": 1, "libero_goal": 2, "libero_10": 3}
GATE_METHODS = ("M0_HARD16", "M2_GRIPPER_EVENT", "FIXED_H13", "SHUFFLED_TRIGGER")
SMOL_METHODS = ("ARM4_GRIP4", "ARM4_GRIP32")
HELD_OUT_STATES = {
    1: (30, 32, 33, 36, 37, 40, 41, 42, 43, 46, 49),
    2: (24, 25, 26, 28, 29, 30, 32, 33, 36, 37, 40, 41, 42, 43, 46, 49),
    3: (24, 25, 26, 28, 29, 30, 32, 33, 36, 37, 40, 41, 42, 43, 46, 49),
    4: (30, 32, 33, 36, 37, 40, 41, 42, 43, 46, 49),
    5: (24, 25, 26, 28, 29, 30, 32, 33, 36, 37, 40, 41, 42, 43, 46, 49),
    # States 25, 26, 28, and 29 are removed because raw outcomes exist at commit 38046a9.
    6: (24, 30, 32, 33, 36, 37, 40, 41, 42, 43, 46, 49),
    7: (24, 25, 26, 28, 29, 30, 32, 33, 36, 37, 40, 41, 42, 43, 46, 49),
    8: (24, 25, 26, 28, 29, 30, 32, 33, 36, 37, 40, 41, 42, 43, 46, 49),
    9: (24, 25, 26, 28, 29, 30, 32, 33, 36, 37, 40, 41, 42, 43, 46, 49),
}


def cell(
    *,
    phase: str,
    policy: str,
    suite: str,
    task: int,
    state: int,
    seed: int,
    method: str,
    checkpoint: str,
    max_steps: int | None,
    control_hz: int,
) -> dict:
    cell_id = f"{phase}__{suite}_task{task:02d}_state{state:02d}__{method}"
    value = {
        "cell_id": cell_id,
        "block_id": f"{phase}__{suite}_task{task:02d}_state{state:02d}",
        "phase": phase,
        "policy": policy,
        "suite": suite,
        "task_id": task,
        "state_id": state,
        "environment_seed": seed,
        "method": method,
        "checkpoint": checkpoint,
        "max_episode_steps": max_steps,
        "control_frequency_hz": control_hz,
        "control_mode": "relative",
        "fresh_environment_per_condition_block": True,
        "action_dim": 7,
        "temporal_aggregation": False,
        "smoothing": False,
    }
    if phase == "gate_m":
        value.update({
            "policy_seed": 424242,
            "execution_semantics": "coherent newest chunk; execute row t-q until frozen horizon",
        })
    else:
        arm, grip = (4, 4) if method == "ARM4_GRIP4" else (4, 32)
        value.update({
            "policy_seed_rule": "SHA256(smolvla,suite,task,state,environment_seed,physical_query_step)",
            "strategy": "groupwise_fixed",
            "arm_horizon": arm,
            "gripper_horizon": grip,
            "scope": "CROSS_POLICY_ROBUSTNESS",
        })
    return value


def main() -> None:
    cells = []
    for task, states in HELD_OUT_STATES.items():
        for state in states:
            for method in GATE_METHODS:
                cells.append(cell(
                    phase="gate_m", policy="ACT", suite="libero_object",
                    task=task, state=state, seed=330000 + 100 * task + state,
                    method=method, checkpoint=ACT_OBJECT, max_steps=280, control_hz=20,
                ))
    for suite in SUITE_INDEX:
        for task in range(10):
            for state in range(4):
                for method in SMOL_METHODS:
                    cells.append(cell(
                        phase="smolvla_robustness", policy="SmolVLA", suite=suite,
                        task=task, state=state,
                        seed=360000 + 1000 * SUITE_INDEX[suite] + 100 * task + state,
                        method=method, checkpoint=SMOLVLA, max_steps=None, control_hz=30,
                    ))
    held_out_blocks = sum(len(states) for states in HELD_OUT_STATES.values())
    expected = {"gate_m": 4 * held_out_blocks, "smolvla_robustness": 320}
    observed = {phase: sum(c["phase"] == phase for c in cells) for phase in expected}
    assert held_out_blocks == 130
    assert observed == expected
    manifest = {
        "schema_version": 1,
        "created_before_gate_m_outcomes": True,
        "scientific_queue_is_outcome_independent": True,
        "held_out_blocks": held_out_blocks,
        "expected_counts": expected,
        "gate_condition_order_within_block": list(GATE_METHODS),
        "smolvla_condition_order_within_block": list(SMOL_METHODS),
        "static_sharding_unit": "task/state block; every block's conditions remain on one worker",
        "smolvla_barrier": "launch after every Gate M cell is COMPLETE or TECHNICAL_FAILED without reading success",
        "retry_policy": {
            "maximum_attempts": 3,
            "maximum_retries_after_initial_attempt": 2,
            "allowed": ["CUDA crash", "OOM", "simulator exception", "incomplete/corrupted output"],
            "forbidden": ["task failure", "low reward", "unexpected trajectory", "surprising method result"],
        },
        "stop_after": "smolvla_robustness",
        "cells": cells,
    }
    path = ROOT / "queue_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(path), "counts": observed, "blocks": held_out_blocks}))


if __name__ == "__main__":
    main()

