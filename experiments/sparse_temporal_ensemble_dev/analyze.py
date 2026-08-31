#!/usr/bin/env python3
"""Paired analysis for the ACT and SmolVLA sparse-TE development panels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
METHODS = ("hard_h8", "sparse_te_h8", "hard_h16", "sparse_te_h16")
TASKS = ("libero_object:task3", "libero_spatial:task0", "libero_goal:task2", "libero_10:task3")
SLUGS = tuple(task.replace(":", "_") for task in TASKS)


def exact_mcnemar(candidate_only: int, reference_only: int) -> float:
    discordant = int(candidate_only) + int(reference_only)
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(candidate_only, reference_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def load_policy(policy: str) -> dict[str, dict]:
    results = {}
    for task, slug in zip(TASKS, SLUGS):
        path = ROOT / policy / "results" / f"{slug}.json"
        data = json.loads(path.read_text())
        if data.get("task") != task or "finished_at" not in data:
            raise RuntimeError(f"invalid or incomplete {policy} task result: {path}")
        results[task] = data
    return results


def episode_key(task: str, episode: dict) -> tuple[str, int, int]:
    return (
        task,
        int(episode.get("requested_initial_state_id", episode.get("initial_state_id"))),
        int(episode.get("environment_seed", episode.get("env_seed"))),
    )


def method_episodes(results: dict[str, dict], method: str) -> dict[tuple[str, int, int], dict]:
    episodes = {}
    for task in TASKS:
        for episode in results[task]["methods_result"][method]["episodes_detail"]:
            key = episode_key(task, episode)
            if key in episodes:
                raise RuntimeError(f"duplicate paired unit: {key}")
            episodes[key] = episode
    if len(episodes) != 40:
        raise RuntimeError(f"expected 40 episodes for {method}, got {len(episodes)}")
    return episodes


def summarize_method(results: dict[str, dict], method: str) -> dict:
    episodes = method_episodes(results, method)
    ordered = list(episodes.values())
    successes = [bool(episode["success"]) for episode in ordered]
    step_rows = [row for episode in ordered for row in episode["step_log"]]
    latencies = [
        float(value)
        for episode in ordered
        for value in episode.get("query_latency_seconds", [])
    ]
    completion = [int(episode["completion_steps"]) for episode in ordered if episode["completion_steps"] is not None]
    total_queries = sum(int(episode["query_count"]) for episode in ordered)
    total_steps = sum(int(episode["environment_steps"]) for episode in ordered)
    return {
        "success_count": int(sum(successes)),
        "episodes": len(ordered),
        "per_task_success": {
            task: int(
                sum(
                    bool(episode["success"])
                    for key, episode in episodes.items()
                    if key[0] == task
                )
            )
            for task in TASKS
        },
        "policy_queries": total_queries,
        "environment_steps": total_steps,
        "query_rate": total_queries / total_steps,
        "mean_query_count": total_queries / len(ordered),
        "mean_ensemble_candidate_count": float(
            np.mean([row["ensemble_candidate_count"] for row in step_rows])
        ),
        "mean_weighted_source_age_steps": float(
            np.mean([row["mean_weighted_source_age_steps"] for row in step_rows])
        ),
        "successful_completion_count": len(completion),
        "mean_completion_steps_successful": float(np.mean(completion)) if completion else None,
        "mean_query_latency_seconds": float(np.mean(latencies)) if latencies else None,
    }


def paired_contrast(results: dict[str, dict], candidate: str, reference: str) -> dict:
    candidate_episodes = method_episodes(results, candidate)
    reference_episodes = method_episodes(results, reference)
    if candidate_episodes.keys() != reference_episodes.keys():
        raise RuntimeError(f"paired-unit mismatch: {candidate} vs {reference}")
    candidate_only = 0
    reference_only = 0
    both_success = 0
    both_fail = 0
    per_task = {}
    for key in candidate_episodes:
        c = bool(candidate_episodes[key]["success"])
        r = bool(reference_episodes[key]["success"])
        if c and not r:
            candidate_only += 1
        elif r and not c:
            reference_only += 1
        elif c:
            both_success += 1
        else:
            both_fail += 1
    for task in TASKS:
        keys = [key for key in candidate_episodes if key[0] == task]
        co = sum(bool(candidate_episodes[key]["success"]) and not bool(reference_episodes[key]["success"]) for key in keys)
        ro = sum(bool(reference_episodes[key]["success"]) and not bool(candidate_episodes[key]["success"]) for key in keys)
        per_task[task] = {"candidate_only": co, "reference_only": ro, "net_wins": co - ro}
    return {
        "candidate": candidate,
        "reference": reference,
        "candidate_only": candidate_only,
        "reference_only": reference_only,
        "both_success": both_success,
        "both_fail": both_fail,
        "paired_net_wins": candidate_only - reference_only,
        "exact_mcnemar_p_two_sided": exact_mcnemar(candidate_only, reference_only),
        "per_task": per_task,
    }


def fmt(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def render_report(analysis: dict) -> str:
    lines = [
        "# Sparse temporal ensemble development experiment",
        "",
        "At the same fixed sparse-query cadence, this experiment compares executing the newest chunk with canonical oldest-to-newest temporal ensembling over all still-valid same-target predictions. Policies are analyzed separately on four exposed development tasks; no blind tasks are included.",
        "",
    ]
    for policy, title in (("act", "ACT"), ("smolvla", "SmolVLA")):
        item = analysis["policies"][policy]
        lines.extend([f"## {title}", "", "| method | success | object3 | spatial0 | goal2 | libero10-3 | queries | query rate | queries/ep | mean candidates | weighted age | completion steps | latency/query (s) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for method in METHODS:
            row = item["methods"][method]
            per = row["per_task_success"]
            lines.append(
                f"| {method} | {row['success_count']}/40 | {per[TASKS[0]]}/10 | {per[TASKS[1]]}/10 | {per[TASKS[2]]}/10 | {per[TASKS[3]]}/10 | {row['policy_queries']} | {row['query_rate']:.5f} | {row['mean_query_count']:.2f} | {row['mean_ensemble_candidate_count']:.2f} | {row['mean_weighted_source_age_steps']:.2f} | {fmt(row['mean_completion_steps_successful'], 1)} | {fmt(row['mean_query_latency_seconds'], 4)} |"
            )
        lines.extend(["", "| contrast | candidate-only | reference-only | net wins | task nets (obj/sp/goal/10) | exact McNemar p |", "|---|---:|---:|---:|---:|---:|"])
        for contrast in item["contrasts"]:
            task_nets = "/".join(f"{contrast['per_task'][task]['net_wins']:+d}" for task in TASKS)
            lines.append(
                f"| {contrast['candidate']} vs {contrast['reference']} | {contrast['candidate_only']} | {contrast['reference_only']} | {contrast['paired_net_wins']:+d} | {task_nets} | {contrast['exact_mcnemar_p_two_sided']:.6f} |"
            )
        if policy == "smolvla":
            verification = item.get("paired_rng_verification", {})
            lines.extend(
                [
                    "",
                    f"Paired SmolVLA query RNG verification: **{verification.get('status', 'missing')}**. "
                    f"The real postprocessed raw chunks had shape `{verification.get('raw_chunk_shape')}` and "
                    f"maximum absolute difference `{verification.get('raw_chunk_max_abs_error')}` for key "
                    f"`{verification.get('query_seed_key', 'missing')}`.",
                ]
            )
        lines.append("")

    lines.extend([
        "## Cross-policy summary",
        "",
        "| policy | hard h8 | h8+TE | hard h16 | h16+TE |",
        "|---|---:|---:|---:|---:|",
    ])
    for policy, title in (("act", "ACT"), ("smolvla", "SmolVLA")):
        methods = analysis["policies"][policy]["methods"]
        cells = [f"{methods[method]['success_count']}/40 ({methods[method]['query_rate']:.5f})" for method in METHODS]
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "Query rates are shown in parentheses. Differences in query rate within a cadence arise only from different episode completion lengths; the scheduled policy-query times are identical over every common trajectory prefix.",
        "",
        "## Decision",
        "",
        f"**{analysis['decision_label']}**",
        "",
        analysis["interpretation"],
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decision",
        choices=("PENDING_HUMAN_REVIEW", "SPARSE_TE_STRONG", "SPARSE_TE_POLICY_DEPENDENT", "SPARSE_TE_NULL", "SPARSE_TE_HARMFUL"),
        default="PENDING_HUMAN_REVIEW",
    )
    parser.add_argument("--interpretation", default="Supervisor interpretation pending completion of both policy panels.")
    args = parser.parse_args()

    analysis = {"policies": {}, "decision_label": args.decision, "interpretation": args.interpretation}
    for policy in ("act", "smolvla"):
        results = load_policy(policy)
        item = {
            "methods": {method: summarize_method(results, method) for method in METHODS},
            "contrasts": [
                paired_contrast(results, "sparse_te_h8", "hard_h8"),
                paired_contrast(results, "sparse_te_h16", "hard_h16"),
                paired_contrast(results, "sparse_te_h16", "sparse_te_h8"),
            ],
        }
        if policy == "smolvla":
            smoke_path = ROOT / "smolvla" / "paired_rng_smoke.json"
            item["paired_rng_verification"] = json.loads(smoke_path.read_text())
        analysis["policies"][policy] = item

    (ROOT / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    (ROOT / "report.md").write_text(render_report(analysis))
    print(json.dumps({"analysis": str(ROOT / "analysis.json"), "report": str(ROOT / "report.md"), "decision": args.decision}))


if __name__ == "__main__":
    main()
