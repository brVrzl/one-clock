#!/usr/bin/env python3
"""Freeze expert-screened RoboTwin seeds before exploratory policy rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


TASKS = (
    "beat_block_hammer",
    "click_alarmclock",
    "dump_bin_bigbin",
    "handover_block",
    "open_laptop",
)


def safe_close(task_env: Any, *, clear_cache: bool = False) -> None:
    try:
        task_env.close_env(clear_cache=clear_cache)
    except Exception:
        pass


def screen_task(
    evaluator: Any,
    unstable_error: type[Exception],
    task: str,
    *,
    seed_start: int,
    eligible_count: int,
    max_candidates: int,
) -> dict[str, Any]:
    task_usr_args = {
        "task_name": task,
        "task_config": "demo_clean",
        "ckpt_setting": "eligibility_screen_no_policy",
        "policy_name": "ACT",
    }
    task_args, _ = evaluator.load_task_args(task_usr_args)
    task_args["eval_instruction"] = "seen"
    task_args["eval_mode"] = True
    task_args["render_freq"] = 0
    task_args["eval_video_log"] = False
    task_env = evaluator.class_decorator(task)
    eligible = []
    rejected = []
    candidates = []
    try:
        for seed in range(seed_start, seed_start + max_candidates):
            if len(eligible) == eligible_count:
                break
            episode_index = len(eligible)
            try:
                task_env.setup_demo(
                    now_ep_num=episode_index,
                    seed=seed,
                    is_test=True,
                    **task_args,
                )
                task_env.play_once()
                safe_close(task_env)
            except unstable_error:
                safe_close(task_env)
                result = {
                    "candidate_seed": seed,
                    "eligible": False,
                    "rejection_reason": "UNSTABLE",
                }
                rejected.append(result)
                candidates.append(result)
                print(f"{task} seed {seed}: UNSTABLE", flush=True)
                continue
            except Exception as error:
                safe_close(task_env, clear_cache=True)
                result = {
                    "candidate_seed": seed,
                    "eligible": False,
                    "rejection_reason": f"EXPERT_ERROR:{type(error).__name__}",
                }
                rejected.append(result)
                candidates.append(result)
                print(f"{task} seed {seed}: {result['rejection_reason']}", flush=True)
                continue

            if not (task_env.plan_success and task_env.check_success()):
                result = {
                    "candidate_seed": seed,
                    "eligible": False,
                    "rejection_reason": "EXPERT_FAILED",
                }
                rejected.append(result)
                candidates.append(result)
                print(f"{task} seed {seed}: EXPERT_FAILED", flush=True)
                continue

            result = {
                "candidate_seed": seed,
                "eligible": True,
                "eligible_seed_index": len(eligible),
                "rejection_reason": None,
            }
            eligible.append(seed)
            candidates.append(result)
            print(
                f"{task} seed {seed}: eligible {len(eligible)}/{eligible_count}",
                flush=True,
            )
    finally:
        safe_close(task_env, clear_cache=True)

    if len(eligible) != eligible_count:
        raise RuntimeError(
            f"{task}: obtained {len(eligible)} eligible seeds from {max_candidates} candidates"
        )
    return {
        "task": task,
        "ordering": "ascending official RoboTwin seed from seed_start",
        "seed_start": seed_start,
        "eligible_seed_count": eligible_count,
        "eligible_seeds": eligible,
        "rejected_seeds": rejected,
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--eligible-per-task", type=int, default=20)
    parser.add_argument("--max-candidates-per-task", type=int, default=200)
    args = parser.parse_args()

    sys.path.insert(0, str(args.robotwin_root / "scripts"))
    sys.path.insert(0, str(args.robotwin_root))
    import eval_policy_xpolicylab as evaluator
    from envs.utils.create_actor import UnStableError

    tasks = {}
    for task in TASKS:
        tasks[task] = screen_task(
            evaluator,
            UnStableError,
            task,
            seed_start=args.seed_start,
            eligible_count=args.eligible_per_task,
            max_candidates=args.max_candidates_per_task,
        )
    output = {
        "schema_version": 1,
        "purpose": "Outcome-independent official expert eligibility screen",
        "robotwin_seed_convention": (
            "ascending seeds beginning at 100000, matching evaluator seed argument 0"
        ),
        "selection_rule": (
            "first 20 expert-eligible seeds in ascending candidate-seed order, "
            "independently per frozen task"
        ),
        "task_order": list(TASKS),
        "tasks": tasks,
        "temporal_policy_rollouts_executed": False,
        "temporal_method_outcomes_inspected": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
