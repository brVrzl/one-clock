#!/usr/bin/env python3
"""Analyze the complete repaired ACT h16 trio as paired binary outcomes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "act_h16" / "results"
OUTPUT = ROOT / "analysis.json"
METHODS = ("hard_h16", "candidate_index_te_h16", "dense_equivalent_te_h16")
TASKS = (
    "libero_object:task3",
    "libero_spatial:task0",
    "libero_goal:task2",
    "libero_10:task3",
)


def slug(task: str) -> str:
    return task.replace(":task", "_task")


def exact_mcnemar_p(first_only: int, reference_only: int) -> float:
    discordant = int(first_only) + int(reference_only)
    if discordant == 0:
        return 1.0
    tail = min(int(first_only), int(reference_only))
    probability = sum(math.comb(discordant, index) for index in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def load_results() -> dict[str, dict]:
    results = {}
    for task in TASKS:
        path = RESULT_ROOT / f"{slug(task)}.json"
        marker = ROOT / "act_h16" / "markers" / f"{slug(task)}.complete"
        if not path.is_file() or not marker.is_file():
            raise RuntimeError(f"missing validated repaired shard: {task}")
        data = json.loads(path.read_text())
        if data["task"] != task or list(data["methods_result"]) != list(METHODS):
            raise RuntimeError(f"unexpected task/method metadata in {path}")
        results[task] = data
    return results


def paired_rows(results: dict[str, dict]) -> list[dict]:
    rows = []
    for task in TASKS:
        by_method = {}
        for method in METHODS:
            episodes = results[task]["methods_result"][method]["episodes_detail"]
            by_method[method] = {
                (int(episode["requested_initial_state_id"]), int(episode["environment_seed"])): episode
                for episode in episodes
            }
        pairs = sorted(by_method["hard_h16"])
        if pairs != [(state, 1990 + state) for state in range(10, 20)]:
            raise RuntimeError(f"unexpected repaired pairing for {task}: {pairs}")
        if any(sorted(by_method[method]) != pairs for method in METHODS):
            raise RuntimeError(f"method pairing mismatch for {task}")
        for state_id, env_seed in pairs:
            rows.append(
                {
                    "task": task,
                    "state_id": state_id,
                    "environment_seed": env_seed,
                    **{
                        method: bool(by_method[method][(state_id, env_seed)]["success"])
                        for method in METHODS
                    },
                }
            )
    return rows


def summarize_method(results: dict[str, dict], method: str) -> dict:
    episodes = [
        episode
        for task in TASKS
        for episode in results[task]["methods_result"][method]["episodes_detail"]
    ]
    steps = [row for episode in episodes for row in episode["step_log"]]
    latencies = [value for episode in episodes for value in episode["query_latency_seconds"]]
    total_queries = sum(int(episode["query_count"]) for episode in episodes)
    total_environment_steps = sum(int(episode["environment_steps"]) for episode in episodes)
    completion = [int(episode["completion_steps"]) for episode in episodes if episode["completion_steps"] is not None]
    per_task = {
        task: int(results[task]["methods_result"][method]["success_count"])
        for task in TASKS
    }
    return {
        "success_count": int(sum(bool(episode["success"]) for episode in episodes)),
        "episodes": len(episodes),
        "per_task_success_count": per_task,
        "total_policy_queries": total_queries,
        "total_environment_steps": total_environment_steps,
        "query_rate": total_queries / float(total_environment_steps),
        "mean_query_count_per_episode": float(np.mean([episode["query_count"] for episode in episodes])),
        "mean_ensemble_candidate_count": float(
            np.mean([row["ensemble_candidate_count"] for row in steps])
        ),
        "mean_weighted_source_age_steps": float(
            np.mean([row["mean_weighted_source_age_steps"] for row in steps])
        ),
        "mean_completion_steps_successful": float(np.mean(completion)) if completion else None,
        "mean_query_latency_seconds": float(np.mean(latencies)) if latencies else None,
    }


def compare(rows: list[dict], first: str, reference: str) -> dict:
    first_only_rows = [row for row in rows if row[first] and not row[reference]]
    reference_only_rows = [row for row in rows if row[reference] and not row[first]]
    return {
        "first": first,
        "reference": reference,
        "first_only_wins": len(first_only_rows),
        "reference_only_wins": len(reference_only_rows),
        "paired_net_wins": len(first_only_rows) - len(reference_only_rows),
        "discordant_pairs": len(first_only_rows) + len(reference_only_rows),
        "exact_mcnemar_p_two_sided": exact_mcnemar_p(
            len(first_only_rows), len(reference_only_rows)
        ),
        "first_only_cases": [
            {"task": row["task"], "state_id": row["state_id"], "environment_seed": row["environment_seed"]}
            for row in first_only_rows
        ],
        "reference_only_cases": [
            {"task": row["task"], "state_id": row["state_id"], "environment_seed": row["environment_seed"]}
            for row in reference_only_rows
        ],
    }


def main() -> None:
    results = load_results()
    rows = paired_rows(results)
    output = {
        "experiment": "repaired ACT h16 trio with fresh environment per condition/state",
        "cohort": {
            "tasks": list(TASKS),
            "state_ids": list(range(10, 20)),
            "environment_seeds": list(range(2000, 2010)),
            "paired_episodes": len(rows),
        },
        "methods": {method: summarize_method(results, method) for method in METHODS},
        "paired_comparisons": {
            "candidate_index_vs_hard": compare(rows, "candidate_index_te_h16", "hard_h16"),
            "dense_equivalent_vs_hard": compare(rows, "dense_equivalent_te_h16", "hard_h16"),
            "dense_equivalent_vs_candidate_index": compare(
                rows, "dense_equivalent_te_h16", "candidate_index_te_h16"
            ),
        },
        "paired_state_outcomes": rows,
    }
    hard = output["methods"]["hard_h16"]
    candidate_index = output["methods"]["candidate_index_te_h16"]
    dense_equivalent = output["methods"]["dense_equivalent_te_h16"]
    dense_vs_hard = output["paired_comparisons"]["dense_equivalent_vs_hard"]
    candidate_vs_hard = output["paired_comparisons"]["candidate_index_vs_hard"]
    dense_task_deltas = {
        task: dense_equivalent["per_task_success_count"][task]
        - hard["per_task_success_count"][task]
        for task in TASKS
    }
    if (
        dense_equivalent["success_count"] < hard["success_count"]
        and dense_vs_hard["paired_net_wins"] <= -5
        and all(delta <= 0 for delta in dense_task_deltas.values())
    ):
        decision = "DENSE_EQ_TE_HARMFUL"
    else:
        raise RuntimeError("completed result requires a scientific classification outside the predeclared clear-harm case")
    candidate_task_deltas = {
        task: candidate_index["per_task_success_count"][task]
        - hard["per_task_success_count"][task]
        for task in TASKS
    }
    output["decision"] = {
        "label": decision,
        "basis": {
            "dense_equivalent_minus_hard_successes": dense_equivalent["success_count"]
            - hard["success_count"],
            "dense_equivalent_vs_hard_paired_net_wins": dense_vs_hard["paired_net_wins"],
            "dense_equivalent_taskwise_deltas": dense_task_deltas,
        },
        "repaired_candidate_index_te_remains_harmful": bool(
            candidate_index["success_count"] < hard["success_count"]
            and candidate_vs_hard["paired_net_wins"] <= -5
            and all(delta <= 0 for delta in candidate_task_deltas.values())
        ),
        "candidate_index_taskwise_deltas": candidate_task_deltas,
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "paired_episodes": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
