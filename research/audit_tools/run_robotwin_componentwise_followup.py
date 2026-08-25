#!/usr/bin/env python3
"""Run the authorized demonstration-only component-wise kernel follow-up."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO = Path("/home/wjq/workspace/one-clock")
ASSETS = Path("/home/wjq/research-assets/robotwin")
ROBOTWIN = ASSETS / "RoboTwin"
ACT_ROOT = ROBOTWIN / "XPolicyLab/policy/ACT"
VENV = ASSETS / "robotwin2_overnight"
CUDA = ASSETS / "cuda-12.8-local-root/usr/local/cuda-12.8"
CUROBO = ASSETS / "curobo-v0.7.8-py310-cu128-sm120-overlay"
FIT = REPO / "research/audit_tools/fit_robotwin_componentwise_kernel.py"
CANARY = REPO / "research/audit_tools/run_robotwin_componentwise_canary.py"
OUTPUT_ROOT = REPO / "research/audit_outputs/robotwin_componentwise_temporal_aggregation"
STATUS = REPO / "research/audit_outputs/robotwin_exploratory_followup_status.json"
PREREG = REPO / "research/robotwin_componentwise_temporal_aggregation_preregistration.md"
SUMMARY = OUTPUT_ROOT / "summary.json"
TASKS = (
    "beat_block_hammer",
    "click_alarmclock",
    "dump_bin_bigbin",
    "handover_block",
    "open_laptop",
)


def environment(gpu: int) -> dict[str, str]:
    result = os.environ.copy()
    result.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "CUDA_HOME": str(CUDA),
            "PATH": f"{VENV / 'bin'}:{CUDA / 'bin'}:{result.get('PATH', '')}",
            "LD_LIBRARY_PATH": f"{CUDA / 'lib64'}:{result.get('LD_LIBRARY_PATH', '')}",
            "PYTHONPATH": f"{REPO}:{CUROBO}:{ROBOTWIN}",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return result


def checkpoint_dir(task: str) -> Path:
    return ACT_ROOT / "checkpoints" / f"demo_clean-{task}-aloha_agilex-joint-0"


def data_root(task: str) -> Path:
    return ACT_ROOT / "processed_data/demo_clean" / task / "aloha_agilex-joint"


def run_wave(assignments: tuple[tuple[str, int], ...]) -> None:
    processes = []
    for task, gpu in assignments:
        output = OUTPUT_ROOT / f"{task}.json"
        log = ASSETS / "coordination" / f"componentwise_fit_{task}_gpu{gpu}.log"
        handle = log.open("a")
        process = subprocess.Popen(
            [
                str(VENV / "bin/python"), str(FIT), "--task", task,
                "--act-root", str(ACT_ROOT), "--checkpoint-dir", str(checkpoint_dir(task)),
                "--data-root", str(data_root(task)), "--output", str(output), "--device", "cuda:0",
            ],
            cwd=REPO,
            env=environment(gpu),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        handle.close()
        processes.append((task, process))
    for task, process in processes:
        if process.wait() != 0:
            raise RuntimeError(f"component-wise kernel fit failed for {task}")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_wave((("beat_block_hammer", 0), ("click_alarmclock", 1), ("dump_bin_bigbin", 2)))
    run_wave((("handover_block", 0), ("open_laptop", 1)))
    results = [json.loads((OUTPUT_ROOT / f"{task}.json").read_text()) for task in TASKS]
    if not all(item["recorded_sequence_provenance_canary"]["passed"] for item in results):
        raise RuntimeError("recorded-sequence component composition canary failed")

    canary_output = OUTPUT_ROOT / "closed_loop_canary.json"
    canary_log = ASSETS / "coordination/componentwise_closed_loop_canary_gpu0.log"
    with canary_log.open("a") as handle:
        subprocess.run(
            [
                str(VENV / "bin/python"), str(CANARY), "--robotwin-root", str(ROBOTWIN),
                "--checkpoint-dir", str(checkpoint_dir("beat_block_hammer")),
                "--kernel", str(OUTPUT_ROOT / "beat_block_hammer.json"),
                "--output", str(canary_output),
            ],
            cwd=ROBOTWIN,
            env=environment(0),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    canary = json.loads(canary_output.read_text())
    if not canary["provenance_passed"] or canary["decision_count"] != 20:
        raise RuntimeError("closed-loop component-wise provenance canary failed")

    summary = {
        "method": "Component-wise Temporal Aggregation",
        "fit_source": "official training demonstrations only",
        "train_episodes_per_task": list(range(40)),
        "heldout_episodes_per_task": list(range(40, 50)),
        "task_results": {
            item["task"]: {
                "heldout_reconstruction_mse": item["heldout_reconstruction_mse"],
                "arm_kernel": item["arm_kernel"],
                "gripper_kernel": item["gripper_kernel"],
            }
            for item in results
        },
        "closed_loop_provenance_canary": "PASS",
        "rollout_success_used_for_fitting": False,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    PREREG.write_text(
        "# Component-wise Temporal Aggregation follow-up preregistration draft\n\n"
        "This follow-up was authorized only after a positive frozen RoboTwin exploratory gate. "
        "For each task, separate convex 50-lag arm and gripper kernels are fit on official "
        "training demonstrations 0–39 and evaluated offline on held-out demonstrations 40–49. "
        "No rollout-success outcome enters fitting or kernel selection. Early-decision execution "
        "renormalizes the available prefix of each frozen kernel. The next closed-loop evaluation "
        "must freeze tasks, seeds, comparators, and checkpoint identities before outcome collection.\n"
    )
    status = json.loads(STATUS.read_text())
    status["status"] = "COMPONENT_WISE_KERNELS_AND_CANARY_COMPLETE"
    status["summary"] = str(SUMMARY.relative_to(REPO))
    status["followup_preregistration"] = str(PREREG.relative_to(REPO))
    STATUS.write_text(json.dumps(status, indent=2) + "\n")
    subprocess.run(
        ["git", "add", str(OUTPUT_ROOT.relative_to(REPO)), str(PREREG.relative_to(REPO)), str(STATUS.relative_to(REPO))],
        cwd=REPO,
        check=True,
    )
    subprocess.run(["git", "commit", "-m", "experiment: prepare component-wise temporal aggregation"], cwd=REPO, check=True)


if __name__ == "__main__":
    main()
