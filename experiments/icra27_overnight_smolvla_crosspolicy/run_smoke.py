#!/usr/bin/env python3
"""Run the predeclared SmolVLA technical smoke gate; no scientific cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_queue import ROOT, Runtime, atomic_json


CHECKPOINT = "/home/wjq/checkpoints/HuggingFaceVLA_smolvla_libero"
SUITE_INDEX = {"libero_spatial": 0, "libero_object": 1, "libero_goal": 2, "libero_10": 3}


def cell(suite: str, task: int, state: int, method: str, strategy: str) -> dict:
    return {
        "cell_id": f"smoke__{suite}_task{task}_state{state}__{method}",
        "phase": "technical_smoke", "policy": "SmolVLA", "suite": suite,
        "task_id": task, "state_id": state,
        "environment_seed": 370000 + 1000 * SUITE_INDEX[suite] + 100 * task + state,
        "method": method, "strategy": strategy, "arm_horizon": 8,
        "gripper_horizon": 8, "checkpoint": CHECKPOINT,
        "max_episode_steps": None, "control_frequency_hz": 30,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    runtime = Runtime(args.gpu)
    diagonal = []
    for state in (10, 11):
        global_cell = cell("libero_object", 3, state, "global_h8", "global_fixed")
        group_cell = cell("libero_object", 3, state, "group_arm8_grip8", "groupwise_fixed")
        left = runtime.run(global_cell)
        right = runtime.run(group_cell)
        fields = ("executed_actions", "success", "environment_steps", "policy_queries", "query_steps")
        equal = {field: left[field] == right[field] for field in fields}
        if not all(equal.values()):
            raise RuntimeError(f"diagonal smoke mismatch at state {state}: {equal}")
        diagonal.append({"state_id": state, "equal": equal, "global": left, "groupwise": right})
    suite_results = []
    memory = []
    for suite in SUITE_INDEX:
        before = int(runtime.torch.cuda.memory_allocated()) if runtime.torch.cuda.is_available() else 0
        value = runtime.run(cell(suite, 0, 0, "SMOLVLA_COHERENT_H8", "global_fixed"))
        after = int(runtime.torch.cuda.memory_allocated()) if runtime.torch.cuda.is_available() else 0
        suite_results.append(value)
        memory.append({"suite": suite, "before_bytes": before, "after_bytes": after, "delta_bytes": after - before})
    allocated = [m["after_bytes"] for m in memory]
    growth = max(allocated) - min(allocated) if allocated else 0
    result = {
        "status": "PASS", "checkpoint": CHECKPOINT,
        "checkpoint_requirements": {"chunk_size": 50, "n_action_steps": 1,
            "action_dim": 7, "arm_indices": [0, 1, 2, 3, 4, 5], "gripper_index": 6,
            "temporal_aggregation": False, "smoothing": False},
        "diagonal_control": diagonal, "suite_smoke": suite_results,
        "cuda_memory": memory, "allocated_growth_bytes": growth,
        "no_gpu_memory_growth": growth == 0,
        "video_disabled": True,
    }
    atomic_json(ROOT / "smoke" / "smolvla_smoke.json", result)
    runtime.drop_policy()
    print(json.dumps({"status": "PASS", "result": str(ROOT / "smoke/smolvla_smoke.json"),
        "wall_seconds": [r["wall_clock_seconds"] for r in suite_results],
        "steps": [r["environment_steps"] for r in suite_results],
        "forwards": [r["model_forward_count"] for r in suite_results],
        "memory_growth_bytes": growth}, indent=2))


if __name__ == "__main__":
    main()
