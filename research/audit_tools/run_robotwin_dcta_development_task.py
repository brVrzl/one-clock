#!/usr/bin/env python3
"""Run one task's frozen DCTA development cells with separated outcomes."""

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

from research.audit_tools.robotwin_dcta import (
    DCTAExecutor,
    DynamicTemporalGate,
    GROUP_NAMES,
    effective_source_age,
)
from research.audit_tools.robotwin_temporal_reuse import postprocess_action
from research.audit_tools.run_robotwin_closed_loop_canaries import PHYSICS_HZ, infer_full_chunk
from research.audit_tools.run_robotwin_exploratory_task import (
    close_environment,
    load_model,
    setup_environment,
)


def load_gate(path: Path, device: torch.device) -> DynamicTemporalGate:
    saved = torch.load(path, map_location=device)
    gate = DynamicTemporalGate().to(device)
    gate.load_state_dict(saved["state_dict"])
    gate.eval()
    return gate


def resolve_schedule_path(schedule_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    # The frozen schedule records repository-relative gate paths.
    return schedule_path.resolve().parents[2] / path


def write_json(path: Path, value: Any, *, sealed: bool = False) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)
    if sealed:
        path.chmod(0o600)


def run_cell(
    evaluator: Any,
    model: Any,
    task_args: dict[str, Any],
    cell: dict[str, Any],
    gate: DynamicTemporalGate | None,
    provenance_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    method = cell["method"]
    model.reset()
    executor = None if method == "NATIVE_ACT" else DCTAExecutor(gate, method, model.model.device)
    environment = setup_environment(evaluator, task_args, cell)
    captured: list[torch.Tensor] = []

    def capture(_module, inputs) -> None:
        captured.append(inputs[0].detach())

    hook = model.model.policy.model.action_head.register_forward_pre_hook(capture)
    decisions = 0
    try:
        with gzip.open(provenance_path, "wt", encoding="utf-8") as provenance_file:
            while not (environment.eval_success or environment.take_action_cnt >= environment.step_lim):
                query_time = environment.scene.physics_steps / PHYSICS_HZ
                observation = environment.get_obs()
                converted = evaluator.robotwin_obs_to_xpolicylab(
                    observation, instruction=cell["task"], env_idx=0, frequency=30, task_env=environment
                )
                if method == "NATIVE_ACT":
                    captured.clear()
                    model.update_obs(converted)
                    action_dict = model.get_action()[0]
                    executed, action_type = evaluator.xpolicylab_action_to_robotwin(
                        action_dict, action_type="joint", current_observation=observation
                    )
                    executed = np.asarray(executed, dtype=np.float32)
                    provenance = {
                        "decision": decisions,
                        "query_time_seconds": query_time,
                        "method": method,
                        "native_temporal_aggregation_enabled": bool(model.model.temporal_agg),
                        "native_candidate_count": min(decisions + 1, 50),
                    }
                else:
                    encoded = model.encode_obs(converted, "joint", model.robot_action_dim_info)
                    normalized_qpos = model.model.pre_process(np.asarray(encoded["qpos"]))
                    model.model.update_obs(encoded)
                    captured.clear()
                    normalized_chunk = infer_full_chunk(model)
                    if len(captured) != 1 or captured[0].shape != (1, 50, 512):
                        raise RuntimeError("ACT context hook failed")
                    step = executor.update(
                        decisions,
                        query_time,
                        normalized_chunk,
                        normalized_qpos,
                        captured[0][0, 0].cpu().numpy(),
                    )
                    executed = postprocess_action(
                        step.action, model.model.stats["action_mean"], model.model.stats["action_std"]
                    ).astype(np.float32)
                    action_type = "qpos"
                    provenance = {
                        "decision": decisions,
                        "query_time_seconds": query_time,
                        "method": method,
                        "candidate_sources": list(step.candidate_sources),
                        "candidate_offsets": list(step.candidate_offsets),
                        "source_ages_seconds": list(step.source_ages_seconds),
                        "weights": {
                            name: step.weights[group].tolist()
                            for group, name in enumerate(GROUP_NAMES)
                        },
                        "effective_source_age_seconds": {
                            name: float(effective_source_age(step.weights[group], step.source_ages_seconds))
                            for group, name in enumerate(GROUP_NAMES)
                        },
                    }
                if executed.shape != (14,) or not np.isfinite(executed).all():
                    raise RuntimeError("invalid DCTA rollout action")
                before = environment.scene.physics_steps
                environment.take_action(executed, action_type=action_type)
                provenance["internal_physics_steps"] = environment.scene.physics_steps - before
                provenance_file.write(json.dumps(provenance, separators=(",", ":")) + "\n")
                provenance_file.flush()
                decisions += 1
        success = bool(environment.eval_success)
        completed = bool(success or environment.take_action_cnt >= environment.step_lim)
    finally:
        hook.remove()
        close_environment(environment)
    outcome = {"cell_id": cell["cell_id"], "success": success}
    technical = {
        "state": "COMPLETE",
        "cell_id": cell["cell_id"],
        "task": cell["task"],
        "method": method,
        "robotwin_seed": cell["robotwin_seed"],
        "decision_count": decisions,
        "episode_completion": completed,
        "provenance_path": str(provenance_path),
    }
    return outcome, technical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text())
    cells = sorted(
        [cell for cell in schedule["cells"] if cell["task"] == args.task],
        key=lambda cell: cell["run_order"],
    )
    if len(cells) != 60:
        raise RuntimeError(f"expected 60 cells for {args.task}")
    sys.path.insert(0, str(args.robotwin_root / "scripts"))
    sys.path.insert(0, str(args.robotwin_root))
    import eval_policy_xpolicylab as evaluator

    os.environ["ACT_ACTION_DIM"] = "14"
    checkpoint = Path(schedule["checkpoints"][args.task]["path"])
    model = load_model(args.robotwin_root, checkpoint.parent, args.task)
    gates = {
        "SHARED_DYNAMIC_AGG": load_gate(
            resolve_schedule_path(
                args.schedule, schedule["gates"]["SHARED_DYNAMIC_AGG"]["path"]
            ),
            model.model.device,
        ),
        "DCTA": load_gate(
            resolve_schedule_path(args.schedule, schedule["gates"]["DCTA"]["path"]),
            model.model.device,
        ),
    }
    task_args, _ = evaluator.load_task_args(
        {"task_name": args.task, "task_config": "demo_clean", "ckpt_setting": checkpoint.parent.name, "policy_name": "ACT"}
    )
    task_args["eval_instruction"] = "seen"
    root = args.result_root / schedule["cells_sha256"]
    status_root = root / "technical"
    outcome_root = root / "sealed_outcomes"
    status_root.mkdir(parents=True, exist_ok=True)
    outcome_root.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        key = hashlib.sha256(cell["cell_id"].encode()).hexdigest()
        status_path = status_root / f"{key}.json"
        if status_path.exists() and json.loads(status_path.read_text()).get("state") == "COMPLETE":
            continue
        attempt = 1
        provenance_path = status_root / f"{key}.provenance.jsonl.gz"
        try:
            outcome, technical = run_cell(
                evaluator, model, task_args, cell, gates.get(cell["method"]), provenance_path
            )
        except Exception as error:
            write_json(
                status_path,
                {
                    "state": "TECHNICAL_FAILURE",
                    "cell_id": cell["cell_id"],
                    "attempt": attempt,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
        write_json(outcome_root / f"{key}.json", outcome, sealed=True)
        write_json(status_path, technical)
        print(f"technical complete cell={key}", flush=True)
    print(f"task technical complete: {args.task} cells=60", flush=True)


if __name__ == "__main__":
    main()
