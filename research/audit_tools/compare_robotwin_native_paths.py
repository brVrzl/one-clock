#!/usr/bin/env python3
"""Short outcome-free trace comparison of official and pilot NATIVE_ACT paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.audit_tools.run_robotwin_exploratory_task import (
    close_environment,
    load_model,
    setup_environment,
)


def digest(value: Any) -> str:
    hasher = hashlib.sha256()
    if isinstance(value, dict):
        for key in sorted(value):
            hasher.update(str(key).encode())
            hasher.update(digest(value[key]).encode())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        hasher.update(str(array.dtype).encode())
        hasher.update(str(array.shape).encode())
        hasher.update(array.view(np.uint8))
    elif torch.is_tensor(value):
        return digest(value.detach().cpu().numpy())
    elif isinstance(value, (list, tuple)):
        for item in value:
            hasher.update(digest(item).encode())
    else:
        hasher.update(repr(value).encode())
    return hasher.hexdigest()


def trace(
    evaluator: Any,
    robotwin_root: Path,
    checkpoint_dir: Path,
    task: str,
    seed: int,
    path_name: str,
    decisions: int,
) -> list[dict[str, Any]]:
    model = load_model(robotwin_root, checkpoint_dir, task)
    task_args, _ = evaluator.load_task_args(
        {
            "task_name": task,
            "task_config": "demo_clean",
            "ckpt_setting": checkpoint_dir.name,
            "policy_name": "ACT",
        }
    )
    task_args["eval_instruction"] = "seen"
    environment = setup_environment(
        evaluator,
        task_args,
        {"task": task, "eligible_seed_index": 0, "robotwin_seed": seed},
    )
    model.reset()
    records = []
    try:
        for decision in range(decisions):
            observation = environment.get_obs()
            converted = evaluator.robotwin_obs_to_xpolicylab(
                observation,
                instruction=task,
                env_idx=0,
                frequency=30,
                task_env=environment,
            )
            encoded = model.encode_obs(
                converted, "joint", model.robot_action_dim_info
            )
            normalized_qpos = model.model.pre_process(np.asarray(encoded["qpos"]))
            model.update_obs(converted)
            if path_name == "official":
                action_chunk = evaluator.normalize_action_chunk(model.get_action())
                action_dict = action_chunk[0]
            elif path_name == "pilot":
                action_dict = model.get_action()[0]
            else:
                raise ValueError(path_name)
            full_chunk = model.model.all_actions.detach().cpu().numpy()[0]
            populated = model.model.all_time_actions[:, decision]
            populated = populated[torch.all(populated != 0, axis=1)]
            weights = np.exp(-0.01 * np.arange(len(populated)))
            weights /= weights.sum()
            denormalized = model.model.post_process(
                (populated * torch.as_tensor(weights, device=populated.device)[:, None])
                .sum(dim=0, keepdim=True)
                .cpu()
                .numpy()
            )[0]
            action, action_type = evaluator.xpolicylab_action_to_robotwin(
                action_dict,
                action_type="joint",
                current_observation=observation,
            )
            action = np.asarray(action, dtype=np.float32)
            records.append(
                {
                    "decision": decision,
                    "raw_observation": digest(observation),
                    "processed_observation": digest(converted),
                    "normalized_qpos": digest(normalized_qpos),
                    "full_chunk": digest(full_chunk),
                    "candidate_count": len(populated),
                    "aggregation_weights": weights.tolist(),
                    "denormalized_action": denormalized.tolist(),
                    "sent_action": action.tolist(),
                    "action_type": action_type,
                }
            )
            environment.take_action(action, action_type=action_type)
    finally:
        close_environment(environment)
        del model
        torch.cuda.empty_cache()
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--task", default="handover_block")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--decisions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.robotwin_root / "scripts"))
    sys.path.insert(0, str(args.robotwin_root))
    import eval_policy_xpolicylab as evaluator

    os.environ["ACT_ACTION_DIM"] = "14"
    comparisons = []
    for seed in args.seeds:
        official = trace(
            evaluator,
            args.robotwin_root,
            args.checkpoint_dir,
            args.task,
            seed,
            "official",
            args.decisions,
        )
        pilot = trace(
            evaluator,
            args.robotwin_root,
            args.checkpoint_dir,
            args.task,
            seed,
            "pilot",
            args.decisions,
        )
        comparisons.append(
            {
                "seed": seed,
                "decisions": len(official),
                "exact_equal": official == pilot,
                "earliest_divergence": next(
                    (
                        {"decision": index, "official": left, "pilot": right}
                        for index, (left, right) in enumerate(zip(official, pilot))
                        if left != right
                    ),
                    None,
                ),
            }
        )
    result = {
        "scope": "outcome-free pinned official-versus-pilot NATIVE_ACT trace",
        "task": args.task,
        "checkpoint": str(args.checkpoint_dir / "policy_last.ckpt"),
        "comparisons": comparisons,
        "all_exact_equal": all(item["exact_equal"] for item in comparisons),
        "success_criterion": "both paths use task_env.eval_success",
        "task_success_recorded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
