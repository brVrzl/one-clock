#!/usr/bin/env python3
"""Compute exploratory component-wise fresh-versus-retained disagreements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pilot = json.loads(args.pilot.read_text())
    protocol = json.loads(args.protocol.read_text())
    ages = protocol["source_ages_steps"]
    task_rows = {}
    observations = {"arm": [], "gripper": []}
    outcome_rows = {"arm": [], "gripper": []}

    for task_key, task in pilot["tasks"].items():
        cache_path = task.get("fresh_query_cache")
        if not cache_path or not Path(cache_path).exists():
            continue
        cache = np.load(cache_path)
        fresh = task["conditions"].get("fresh")
        if fresh is None:
            continue
        fresh_successes = fresh["successes"]
        per_age = {}
        for age in ages:
            age = int(age)
            group_stats = {}
            for group, indices in (("arm", slice(0, 6)), ("gripper", slice(6, 7))):
                episode_means = []
                episode_stds = []
                sampled_means = []
                for episode_index in range(len(fresh_successes)):
                    chunks = np.asarray(cache[f"episode_{episode_index}"], dtype=np.float32)
                    target_steps = np.arange(age, len(chunks))
                    fresh_actions = chunks[target_steps, 0, indices]
                    retained_actions = chunks[target_steps - age, age, indices]
                    distances = np.linalg.norm(fresh_actions - retained_actions, axis=-1)
                    episode_means.append(float(np.mean(distances)))
                    episode_stds.append(float(np.std(distances)))
                    if episode_index == 0:
                        sample_steps = np.arange(max(age, 16), len(chunks))[:10]
                        sampled = chunks[sample_steps, 0, indices] - chunks[sample_steps - age, age, indices]
                        sampled_means.append(float(np.mean(np.linalg.norm(sampled, axis=-1))))
                    observations[group].extend(float(value) for value in distances)

                group_stats[group] = {
                    "episode_mean_l2": episode_means,
                    "episode_std_l2": episode_stds,
                    "mean_l2": float(np.mean(episode_means)),
                    "std_l2_across_episodes": float(np.std(episode_means)),
                    "episode_0_first_10_states_mean_l2": sampled_means[0] if sampled_means else None,
                    "target_step_definition": f"t={max(age, 16)}..min(steps-1,{max(age, 16) + 9}), episode 0",
                }

                condition_names = {
                    "arm": [f"full_old{age}", f"reverse{age}"],
                    "gripper": [f"fo{age}", f"full_old{age}"],
                }[group]
                for condition_name in condition_names:
                    condition = task["conditions"].get(condition_name)
                    if condition is None:
                        continue
                    for episode_index, success in enumerate(condition["successes"]):
                        changed = int(bool(success) != bool(fresh_successes[episode_index]))
                        outcome_rows[group].append(
                            {
                                "task": task_key,
                                "age_steps": age,
                                "condition": condition_name,
                                "episode": episode_index,
                                "disagreement_l2": episode_means[episode_index],
                                "fresh_success": bool(fresh_successes[episode_index]),
                                "intervention_success": bool(success),
                                "outcome_changed": changed,
                            }
                        )
            per_age[str(age)] = group_stats
        task_rows[task_key] = per_age

    aggregate = {}
    for group in ("arm", "gripper"):
        rows = outcome_rows[group]
        aggregate[group] = {
            "n_condition_episode_pairs": len(rows),
            "n_outcome_changes": sum(row["outcome_changed"] for row in rows),
            "correlation_disagreement_with_outcome_change": correlation(
                [row["disagreement_l2"] for row in rows], [row["outcome_changed"] for row in rows]
            ),
            "correlation_disagreement_with_intervention_success": correlation(
                [row["disagreement_l2"] for row in rows], [float(row["intervention_success"]) for row in rows]
            ),
            "rows": rows,
        }

    result = {
        "pilot": str(args.pilot.resolve()),
        "definition": "L2 between current fresh query chunk[0] and the retained source query chunk[age] for the same target step t",
        "source_ages_steps": [int(age) for age in ages],
        "source_ages_seconds_at_30hz": [float(age) / 30.0 for age in ages],
        "task_results": task_rows,
        "aggregate_outcome_association": aggregate,
        "interpretation_limits": [
            "This is exploratory and uses two paired episodes per task.",
            "Outcome change means intervention success differs from fresh success on the paired seed.",
            "The rollout log has no independent unsafe-event label; this analysis therefore tests outcome-change association, not safety causality.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "tasks": len(task_rows)}, indent=2))


if __name__ == "__main__":
    main()
