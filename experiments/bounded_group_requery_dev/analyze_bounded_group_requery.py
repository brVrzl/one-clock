#!/usr/bin/env python3
"""Analyze the fixed development panel at episode/task level."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
TASKS = (
    "libero_object:task3",
    "libero_spatial:task0",
    "libero_goal:task2",
    "libero_10:task3",
)
METHODS = ("M0_hard16", "M1_arm_phase", "M2_gripper_event", "M3_group_event_joint")
ADAPTIVE = METHODS[1:]
TASK_LABELS = {
    "libero_object:task3": "object3",
    "libero_spatial:task0": "spatial0",
    "libero_goal:task2": "goal2",
    "libero_10:task3": "L10-3",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def task_slug(task: str) -> str:
    return task.replace(":", "_")


def load_episodes(method: str, task: str) -> list[dict[str, Any]]:
    if method == "M0_hard16":
        path = REPO_ROOT / "experiments" / "sparse_temporal_ensemble_age_audit" / "act_h16" / "results" / f"{task_slug(task)}.json"
        value = load_json(path)["methods_result"]["hard_h16"]
    else:
        path = ROOT / "act" / "results" / method / f"{task_slug(task)}.json"
        value = load_json(path)["methods_result"][method]
    episodes = list(value["episodes_detail"])
    if len(episodes) != 10:
        raise ValueError(f"{method} {task} has {len(episodes)} episodes, expected 10")
    return episodes


REPO_ROOT = ROOT.parents[1]


def exact_mcnemar(candidate: list[bool], reference: list[bool]) -> dict[str, Any]:
    candidate_only = sum(int(c and not r) for c, r in zip(candidate, reference, strict=True))
    reference_only = sum(int(r and not c) for c, r in zip(candidate, reference, strict=True))
    discordant = candidate_only + reference_only
    p_value = 1.0
    if discordant:
        p_value = min(
            1.0,
            2.0
            * sum(math.comb(discordant, index) for index in range(min(candidate_only, reference_only) + 1))
            / (2**discordant),
        )
    return {
        "candidate_only": candidate_only,
        "reference_only": reference_only,
        "paired_net_wins": candidate_only - reference_only,
        "exact_mcnemar_two_sided_p": p_value,
        "candidate_successes": sum(candidate),
        "reference_successes": sum(reference),
    }


def episode_key(episode: dict[str, Any]) -> tuple[int, int]:
    return int(episode["requested_initial_state_id"]), int(episode["environment_seed"])


def paired_vectors(candidate: list[dict[str, Any]], reference: list[dict[str, Any]]) -> tuple[list[bool], list[bool]]:
    c = {episode_key(row): row for row in candidate}
    r = {episode_key(row): row for row in reference}
    if set(c) != set(r):
        raise ValueError("paired episode keys differ")
    keys = sorted(c)
    return [bool(c[key]["success"]) for key in keys], [bool(r[key]["success"]) for key in keys]


def summary_for_episodes(method: str, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [bool(row["success"]) for row in episodes]
    steps = sum(int(row["environment_steps"]) for row in episodes)
    queries = sum(int(row["policy_queries"]) for row in episodes)
    if method == "M0_hard16":
        planned = [16] * queries
        arm_nomination = grip_nomination = both = nearby = 0
    else:
        planned = [int(h) for row in episodes for h in row["planned_horizons"]]
        noninitial = [entry for row in episodes for entry in row["query_log"][1:]]
        arm_active = method in {"M1_arm_phase", "M3_group_event_joint"}
        grip_active = method in {"M2_gripper_event", "M3_group_event_joint"}
        arm_nomination = sum(int(arm_active and row["h_arm"] < 16) for row in noninitial)
        grip_nomination = sum(int(grip_active and row["h_grip"] < 16) for row in noninitial)
        both = sum(int(arm_active and grip_active and row["both_nominated"]) for row in noninitial)
        nearby = sum(int(arm_active and grip_active and row["both_nearby"]) for row in noninitial)
    successful_completion = [int(row["completion_steps"]) for row in episodes if row["completion_steps"] is not None]
    return {
        "success_count": int(sum(successes)),
        "episodes": len(episodes),
        "success_rate": float(np.mean(successes)),
        "policy_queries": queries,
        "environment_steps": steps,
        "query_rate": queries / float(steps),
        "mean_planned_horizon": float(np.mean(planned)),
        "median_planned_horizon": float(np.median(planned)),
        "horizon_histogram": {str(h): int(planned.count(h)) for h in range(4, 17)},
        "arm_nomination_count": arm_nomination,
        "gripper_nomination_count": grip_nomination,
        "noninitial_query_count": max(0, queries - len(episodes)),
        "arm_nomination_fraction": arm_nomination / max(1, queries - len(episodes)),
        "gripper_nomination_fraction": grip_nomination / max(1, queries - len(episodes)),
        "both_nomination_count": both,
        "both_nearby_count": nearby,
        "both_nearby_fraction_of_both": nearby / both if both else None,
        "mean_completion_steps_successful": float(np.mean(successful_completion)) if successful_completion else None,
        "successes": successes,
    }


def first_dynamic_event(episode: dict[str, Any]) -> dict[str, Any] | None:
    events = episode.get("query_log", [])[1:]
    if not events:
        return None
    event = events[0]
    return {
        "query_step": int(event["query_physical_step_q"]),
        "h_arm": int(event["h_arm"]),
        "h_grip": int(event["h_grip"]),
        "h_exec": int(event["h_exec"]),
        "trigger_reason": event["trigger_reason"],
        "arm_triggered": bool(event["arm_triggered"]),
        "gripper_triggered": bool(event["gripper_triggered"]),
        "both_nominated": bool(event["both_nominated"]),
        "both_nearby": bool(event["both_nearby"]),
    }


def action_row(episode: dict[str, Any], index: int) -> np.ndarray:
    row = episode["step_log"][index]
    for key in ("chosen_executed_action_7d", "chosen_action_7d"):
        if key in row:
            return np.asarray(row[key], dtype=np.float64)
    raise KeyError("action field not found")


def first_divergence(m0: dict[str, Any], m3: dict[str, Any]) -> dict[str, Any] | None:
    limit = min(len(m0["step_log"]), len(m3["step_log"]))
    for index in range(limit):
        a = action_row(m0, index)
        b = action_row(m3, index)
        if not np.allclose(a, b, atol=1e-6, rtol=0.0):
            row = m3["step_log"][index]
            query_event = next(
                (
                    event
                    for event in m3.get("query_log", [])
                    if int(event["query_physical_step_q"]) == int(row["query_physical_step_q"])
                ),
                None,
            )
            return {
                "physical_step": int(index),
                "m3_query_step": int(row["query_physical_step_q"]),
                "m3_chunk_offset": int(row["chunk_offset"]),
                "max_action_difference": float(np.max(np.abs(a - b))),
                "query_event": query_event,
            }
    return None


def transition_records(all_episodes: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, list[dict[str, Any]]]:
    m0_by_task = {task: {episode_key(row): row for row in all_episodes["M0_hard16"][task]} for task in TASKS}
    m3_by_task = {task: {episode_key(row): row for row in all_episodes["M3_group_event_joint"][task]} for task in TASKS}
    wins: list[dict[str, Any]] = []
    losses: list[dict[str, Any]] = []
    for task in TASKS:
        for key in sorted(m0_by_task[task]):
            m0 = m0_by_task[task][key]
            m3 = m3_by_task[task][key]
            if bool(m0["success"]) == bool(m3["success"]):
                continue
            record = {
                "task": task,
                "state_id": key[0],
                "environment_seed": key[1],
                "m0_success": bool(m0["success"]),
                "m3_success": bool(m3["success"]),
                "first_dynamic_requery": first_dynamic_event(m3),
                "first_action_divergence": first_divergence(m0, m3),
            }
            (wins if record["m3_success"] else losses).append(record)
    return {"m0_to_m3_wins": wins, "m0_to_m3_losses": losses}


def rankdata(values: list[float]) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0 + 1.0
        index = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(set(y)) < 2 or len(set(x)) < 2:
        return None
    a = rankdata(x)
    b = rankdata(y)
    return float(np.corrcoef(a, b)[0, 1])


def load_h_temp() -> dict[str, float]:
    path = REPO_ROOT / "experiments" / "group_temporal_memory_dev" / "h_temp_development_frozen.json"
    return {row["task_key"]: float(row["H_temp"]) for row in load_json(path)["task_values"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-h-temp", action="store_true")
    parser.add_argument("--decision", required=True)
    parser.add_argument("--interpretation", required=True)
    args = parser.parse_args()

    all_episodes: dict[str, dict[str, list[dict[str, Any]]]] = {method: {} for method in METHODS}
    for method in METHODS:
        for task in TASKS:
            all_episodes[method][task] = load_episodes(method, task)

    summaries = {method: summary_for_episodes(method, [row for task in TASKS for row in all_episodes[method][task]]) for method in METHODS}
    task_summaries = {
        task: {method: summary_for_episodes(method, all_episodes[method][task]) for method in METHODS}
        for task in TASKS
    }
    contrasts = {}
    comparison_pairs = (
        ("M1_arm_phase", "M0_hard16"),
        ("M2_gripper_event", "M0_hard16"),
        ("M3_group_event_joint", "M0_hard16"),
        ("M3_group_event_joint", "M1_arm_phase"),
        ("M3_group_event_joint", "M2_gripper_event"),
    )
    for candidate, reference in comparison_pairs:
        pooled_c = []
        pooled_r = []
        per_task = {}
        for task in TASKS:
            c, r = paired_vectors(all_episodes[candidate][task], all_episodes[reference][task])
            pooled_c.extend(c)
            pooled_r.extend(r)
            per_task[TASK_LABELS[task]] = exact_mcnemar(c, r)
        contrasts[f"{candidate}_vs_{reference}"] = {**exact_mcnemar(pooled_c, pooled_r), "per_task": per_task}

    h_temp_posthoc = None
    if args.include_h_temp:
        htemp = load_h_temp()
        h_temp_posthoc = {"source": str((REPO_ROOT / "experiments" / "group_temporal_memory_dev" / "h_temp_development_frozen.json").resolve()), "task_values": htemp, "methods": {}}
        for method in ADAPTIVE:
            frequency = [task_summaries[task][method]["arm_nomination_fraction"] for task in TASKS]
            grip_frequency = [task_summaries[task][method]["gripper_nomination_fraction"] for task in TASKS]
            gain = [task_summaries[task][method]["success_rate"] - task_summaries[task]["M0_hard16"]["success_rate"] for task in TASKS]
            h_values = [htemp[task] for task in TASKS]
            h_temp_posthoc["methods"][method] = {
                "h_temp_vs_arm_nomination_spearman": spearman(h_values, frequency),
                "h_temp_vs_gripper_nomination_spearman": spearman(h_values, grip_frequency),
                "h_temp_vs_success_gain_spearman": spearman(h_values, gain),
                "task_rows": [
                    {"task": TASK_LABELS[task], "H_temp": htemp[task], "arm_nomination_fraction": frequency[i], "gripper_nomination_fraction": grip_frequency[i], "success_gain_vs_M0": gain[i]}
                    for i, task in enumerate(TASKS)
                ],
            }

    transitions = transition_records(all_episodes)
    analysis = {
        "status": "complete",
        "policy": "ACT",
        "tasks": list(TASKS),
        "episodes_per_method": 40,
        "adaptive_new_episodes": 120,
        "baseline_source": str((REPO_ROOT / "experiments" / "sparse_temporal_ensemble_age_audit" / "act_h16" / "results").resolve()),
        "baseline_commit": "b0b2a6d18ccc9da9ded0057d9f512ad8b535dac0",
        "summaries": summaries,
        "task_summaries": task_summaries,
        "contrasts": contrasts,
        "m0_m3_transition_records": transitions,
        "h_temp_posthoc": h_temp_posthoc,
        "decision_label": args.decision,
        "interpretation": args.interpretation,
    }
    (ROOT / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    write_report(analysis)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(analysis: dict[str, Any]) -> None:
    lines = [
        "# Bounded group-triggered joint re-query development",
        "",
        "## Protocol",
        "",
        "This ACT-only development panel used four exposed tasks, states 10--19, and environment seeds 2000--2009. Every method/state condition used a fresh identically seeded environment. M0 is the repaired authoritative hard16 result; M1--M3 are 120 new episodes. All methods query one new ACT chunk and execute only that newest chunk for a bounded joint horizon. No historical action averaging, temporal ensemble, CogACT aggregation, independent group action source, learned predictor, or H_temp control was used.",
        "",
        "The arm rule uses the earliest local minimum in normalized six-dimensional arm speed with threshold 0.5. The gripper rule uses the earliest open/close intent transition, with nonnegative commands treated as open and negative commands as close. M3 takes the minimum of the two proposed horizons.",
        "",
        "## ACT results",
        "",
        "| Method | Success /40 | Object | Spatial | Goal | L10 | Query rate | Mean horizon | Median horizon | Mean successful completion |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"M0_hard16": "M0 hard16", "M1_arm_phase": "M1 arm phase", "M2_gripper_event": "M2 grip event", "M3_group_event_joint": "M3 combined"}
    for method in METHODS:
        s = analysis["summaries"][method]
        task_counts = [analysis["task_summaries"][task][method]["success_count"] for task in TASKS]
        lines.append(
            f"| {labels[method]} | {s['success_count']}/40 | {task_counts[0]} | {task_counts[1]} | {task_counts[2]} | {task_counts[3]} | {s['query_rate']:.5f} | {s['mean_planned_horizon']:.2f} | {s['median_planned_horizon']:.1f} | {fmt(s['mean_completion_steps_successful'], 1)} |"
        )
    lines += [
        "",
        "### Horizon and trigger statistics",
        "",
        "Fractions below use noninitial queries as the denominator. Both-nomination proximity is defined a priori as boundaries within one action step.",
        "",
        "| Method | Total queries | Env steps | Arm nominations | Grip nominations | Both nominations | Both nearby (count/fraction) | Horizon histogram 4..16 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ADAPTIVE:
        s = analysis["summaries"][method]
        hist = ", ".join(f"{h}:{s['horizon_histogram'][str(h)]}" for h in range(4, 17))
        nearby = "0" if not s["both_nomination_count"] else f"{s['both_nearby_count']}/{s['both_nomination_count']} ({s['both_nearby_fraction_of_both']:.3f})"
        lines.append(
            f"| {labels[method]} | {s['policy_queries']} | {s['environment_steps']} | {s['arm_nomination_fraction']:.3f} | {s['gripper_nomination_fraction']:.3f} | {s['both_nomination_count']} | {nearby} | {hist} |"
        )
    m0 = analysis["summaries"]["M0_hard16"]
    m0_hist = ", ".join(f"{h}:{m0['horizon_histogram'][str(h)]}" for h in range(4, 17) if m0["horizon_histogram"][str(h)])
    lines.append(f"| M0 hard16 | {m0['policy_queries']} | {m0['environment_steps']} | n/a | n/a | n/a | n/a | {m0_hist} |")
    lines += [
        "",
        "M0's exact query count and environment-step denominator are retained in `analysis.json`; its repaired baseline query rate is approximately 0.065.",
        "McNemar probabilities are exact paired descriptive values for this 40-episode development cohort, not confirmatory significance claims.",
        "",
        "## Paired comparisons",
        "",
        "| Contrast | Candidate-only | Reference-only | Paired net | Exact McNemar p |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, value in analysis["contrasts"].items():
        lines.append(f"| {name} | {value['candidate_only']} | {value['reference_only']} | {value['paired_net_wins']:+d} | {value['exact_mcnemar_two_sided_p']:.6g} |")
    lines += [
        "",
        "## M0 to M3 transition mechanism",
        "",
        "The lists below contain every paired outcome transition between M0 and M3. The first dynamic re-query is logged from M3's newly predicted chunk; the first action divergence is compared only as a diagnostic, not as an additional inferential sample.",
        "",
        "| Outcome transition | Task/state | First re-query (q, h_arm, h_grip, h_exec, reason) | First action divergence |",
        "|---|---|---|---|",
    ]
    for transition, records in (("M0 success → M3 failure", analysis["m0_m3_transition_records"]["m0_to_m3_losses"]), ("M0 failure → M3 success", analysis["m0_m3_transition_records"]["m0_to_m3_wins"])):
        for row in records:
            event = row["first_dynamic_requery"]
            event_text = "none" if event is None else f"{event['query_step']}, {event['h_arm']}, {event['h_grip']}, {event['h_exec']}, {event['trigger_reason']}"
            divergence = row["first_action_divergence"]
            if divergence is None:
                divergence_text = "none"
            else:
                event = divergence.get("query_event") or {}
                divergence_text = f"t={divergence['physical_step']} (q={divergence['m3_query_step']}, h={event.get('h_exec', 'n/a')}, {event.get('trigger_reason', 'n/a')}, Δ={divergence['max_action_difference']:.3g})"
            lines.append(f"| {transition} | {TASK_LABELS[row['task']]} / {row['state_id']} | {event_text} | {divergence_text} |")
    if not analysis["m0_m3_transition_records"]["m0_to_m3_losses"] and not analysis["m0_m3_transition_records"]["m0_to_m3_wins"]:
        lines.append("| none | none | none | none |")
    lines += [
        "",
        "## H_temp post-hoc analysis",
        "",
        "H_temp was loaded only after adaptive outcomes were frozen and was never read by the executor. It is descriptive only.",
    ]
    if analysis["h_temp_posthoc"] is None:
        lines.append("No H_temp analysis was requested.")
    else:
        for method, value in analysis["h_temp_posthoc"]["methods"].items():
            lines.append(f"- {method}: H_temp versus arm-trigger frequency Spearman={value['h_temp_vs_arm_nomination_spearman']}; versus gripper-trigger frequency Spearman={value['h_temp_vs_gripper_nomination_spearman']}; versus success gain Spearman={value['h_temp_vs_success_gain_spearman']}.")
    lines += [
        "",
        "## SmolVLA",
        "",
        "SmolVLA was not launched. Because ACT selected SINGLE_TRIGGER_BETTER, a minimal M2-only confirmation is prepared in `protocol.json` and remains unrun; any execution requires separate approval and the same method-independent keyed flow-sampling protocol.",
        "",
        "## Decision",
        "",
        f"**{analysis['decision_label']}**",
        "",
        analysis["interpretation"],
        "",
        "See `protocol.json`, `analysis.json`, and the per-method result shards for the frozen definitions and complete episode-level logs.",
        "",
    ]
    (ROOT / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
