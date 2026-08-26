#!/usr/bin/env python3
"""Outcome-free closed-loop provenance canary for NATIVE, shared, and DCTA."""

from __future__ import annotations

import argparse
import json
import os
import sys
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


def run(
    evaluator: Any,
    robotwin_root: Path,
    checkpoint_dir: Path,
    task: str,
    seed: int,
    decisions: int,
    mode: str,
    gate: DynamicTemporalGate,
) -> dict[str, Any]:
    model = load_model(robotwin_root, checkpoint_dir, task)
    task_args, _ = evaluator.load_task_args(
        {"task_name": task, "task_config": "demo_clean", "ckpt_setting": checkpoint_dir.name, "policy_name": "ACT"}
    )
    task_args["eval_instruction"] = "seen"
    environment = setup_environment(
        evaluator,
        task_args,
        {"task": task, "eligible_seed_index": 0, "robotwin_seed": seed},
    )
    captured: list[torch.Tensor] = []

    def capture(_module, inputs) -> None:
        captured.append(inputs[0].detach())

    hook = model.model.policy.model.action_head.register_forward_pre_hook(capture)
    executor = DCTAExecutor(gate, mode, model.model.device)
    records = []
    model.reset()
    try:
        for decision in range(decisions):
            query_time = environment.scene.physics_steps / PHYSICS_HZ
            observation = environment.get_obs()
            converted = evaluator.robotwin_obs_to_xpolicylab(
                observation, instruction=task, env_idx=0, frequency=30, task_env=environment
            )
            encoded = model.encode_obs(converted, "joint", model.robot_action_dim_info)
            normalized_qpos = model.model.pre_process(np.asarray(encoded["qpos"]))
            model.model.update_obs(encoded)
            captured.clear()
            normalized_chunk = infer_full_chunk(model)
            if len(captured) != 1 or captured[0].shape != (1, 50, 512):
                raise RuntimeError("ACT context hook failed during DCTA canary")
            context = captured[0][0, 0].cpu().numpy()
            step = executor.update(
                decision,
                query_time,
                normalized_chunk,
                normalized_qpos,
                context,
            )
            if step.candidate_offsets != tuple(decision - source for source in step.candidate_sources):
                raise RuntimeError("same-decision-target candidate offset failure")
            if not np.allclose(step.weights.sum(axis=1), 1.0, rtol=0, atol=1e-6):
                raise RuntimeError("temporal weights do not sum to one")
            if mode == "SHARED_DYNAMIC_AGG" and not all(
                np.array_equal(step.weights[0], step.weights[group]) for group in range(1, 4)
            ):
                raise RuntimeError("shared dynamic weights differ across groups")
            executed = postprocess_action(
                step.action, model.model.stats["action_mean"], model.model.stats["action_std"]
            ).astype(np.float32)
            before = environment.scene.physics_steps
            environment.take_action(executed, action_type="qpos")
            after = environment.scene.physics_steps
            records.append(
                {
                    "decision": decision,
                    "query_time_seconds": query_time,
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
                    "executed_action": executed.tolist(),
                    "internal_physics_steps": after - before,
                }
            )
    finally:
        hook.remove()
        close_environment(environment)
        del model
        torch.cuda.empty_cache()
    return {"mode": mode, "decision_count": len(records), "records": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--shared-gate", type=Path, required=True)
    parser.add_argument("--dcta-gate", type=Path, required=True)
    parser.add_argument("--task", default="beat_block_hammer")
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--decisions", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.robotwin_root / "scripts"))
    sys.path.insert(0, str(args.robotwin_root))
    import eval_policy_xpolicylab as evaluator

    os.environ["ACT_ACTION_DIM"] = "14"
    device = torch.device("cuda:0")
    zero_gate = DynamicTemporalGate().to(device).eval()
    shared_gate = load_gate(args.shared_gate, device)
    dcta_gate = load_gate(args.dcta_gate, device)
    runs = {
        mode: run(
            evaluator,
            args.robotwin_root,
            args.checkpoint_dir,
            args.task,
            args.seed,
            args.decisions,
            mode,
            gate,
        )
        for mode, gate in (
            ("NATIVE_ACT", zero_gate),
            ("SHARED_DYNAMIC_AGG", shared_gate),
            ("DCTA", dcta_gate),
        )
    }
    result = {
        "scope": "outcome-free DCTA closed-loop provenance canary",
        "task": args.task,
        "seed": args.seed,
        "runs": runs,
        "provenance_passed": all(run_result["decision_count"] > 0 for run_result in runs.values()),
        "task_success_recorded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({mode: value["decision_count"] for mode, value in runs.items()}, indent=2))


if __name__ == "__main__":
    main()
