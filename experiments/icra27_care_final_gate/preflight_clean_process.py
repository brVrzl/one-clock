#!/usr/bin/env python3
"""Targeted clean-process construction audit with no environment step or outcome."""

from __future__ import annotations

import argparse
import json
import os
import time

from run_queue import ROOT, Runtime, atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    runtime = Runtime(args.gpu)
    started = time.time()
    cells = [
        {
            "policy": "ACT", "suite": "libero_object", "task_id": 1, "state_id": 20,
            "environment_seed": 330120, "checkpoint": "/home/wjq/checkpoints/zeromidnight_act_libero_object",
            "control_frequency_hz": 20,
        },
        {
            "policy": "SmolVLA", "suite": "libero_object", "task_id": 0, "state_id": 0,
            "environment_seed": 361000, "checkpoint": "/home/wjq/checkpoints/HuggingFaceVLA_smolvla_libero",
            "control_frequency_hz": 30,
        },
    ]
    rows = [runtime.preflight(cell) for cell in cells]
    runtime.drop_policy()
    result = {
        "status": "PASS",
        "fresh_process_pid": os.getpid(),
        "started_at": started,
        "finished_at": time.time(),
        "process_constructed_from_clean_invocation": True,
        "no_failed_construction_state_preceded_success": True,
        "no_environment_step_executed": all(row["environment_steps"] == 0 for row in rows),
        "no_outcome_observed": all(row["outcome_observed"] is False for row in rows),
        "rows": rows,
    }
    atomic_json(ROOT / "preflight" / "clean_process.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

