#!/usr/bin/env python3
"""Run the bounded group-triggered joint-requery ACT development panel.

Each query produces one new ACT chunk.  The executor runs only that chunk for
the selected bounded horizon and then discards it at the next query.  The
validated ACT runtime and preprocessing helpers are imported from the prior
experiment without modifying that experiment's artifacts.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
VALIDATED_ACT_ROOT = REPO_ROOT / "experiments" / "group_temporal_memory_dev"
sys.path.insert(0, str(VALIDATED_ACT_ROOT))

from requery_policy import (  # noqa: E402
    MAX_HORIZON,
    action_from_newest_chunk,
    choose_horizon,
)
from run_act_group_memory import (  # noqa: E402
    build_runtime,
    compare_array_maps,
    extract_success,
    flatten_numeric,
    get_sim_state,
    infer_chunk,
    reset_act_rng,
    task_map,
)


DEFAULT_PROTOCOL = ROOT / "protocol.json"
METHODS = ("M0_hard16", "M1_arm_phase", "M2_gripper_event", "M3_group_event_joint")
ADAPTIVE_METHODS = METHODS[1:]
PAIRING_MAX_STEPS = 16


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_progress(path: Path | None, value: object) -> None:
    if path is not None:
        atomic_json(path, value)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol.get("status") not in {
        "frozen_before_adaptive_outcomes",
        "act_rollout_in_progress",
        "act_complete",
        "act_complete_single_trigger_better",
    }:
        raise RuntimeError(f"protocol is not in a runnable development state: {protocol.get('status')!r}")
    if protocol["cohort"]["states"] != list(range(10, 20)):
        raise RuntimeError("state cohort drifted from 10..19")
    if protocol["cohort"]["environment_seeds"] != list(range(2000, 2010)):
        raise RuntimeError("environment seed cohort drifted from 2000..2009")
    policy = protocol["policy"]
    if int(policy["prediction_horizon"]) != 100 or int(policy["policy_rng_seed"]) != 424242:
        raise RuntimeError("ACT horizon or frozen policy seed drifted")
    if protocol["pairing"]["fresh_environment_per_method_state"] is not True:
        raise RuntimeError("fresh-environment pairing is not enabled")
    return protocol


def task_key_list(protocol: dict[str, Any]) -> list[str]:
    return list(protocol["cohort"]["tasks"])


def episode_seed_pairs(protocol: dict[str, Any]) -> list[tuple[int, int]]:
    return list(zip(protocol["cohort"]["states"], protocol["cohort"]["environment_seeds"], strict=True))


def summarize_trigger_logs(method: str, query_log: list[dict[str, Any]]) -> dict[str, Any]:
    noninitial = query_log[1:]
    arm_active = method in {"M1_arm_phase", "M3_group_event_joint"}
    gripper_active = method in {"M2_gripper_event", "M3_group_event_joint"}
    arm_nominated = sum(int(arm_active and row["h_arm"] < MAX_HORIZON) for row in noninitial)
    gripper_nominated = sum(int(gripper_active and row["h_grip"] < MAX_HORIZON) for row in noninitial)
    both_nominated = sum(int(arm_active and gripper_active and row["both_nominated"]) for row in noninitial)
    both_nearby = sum(int(arm_active and gripper_active and row["both_nearby"]) for row in noninitial)
    arm_selected = sum(
        int(row["h_arm"] < MAX_HORIZON and row["h_arm"] == row["h_exec"])
        for row in noninitial
    )
    gripper_selected = sum(
        int(row["h_grip"] < MAX_HORIZON and row["h_grip"] == row["h_exec"])
        for row in noninitial
    )
    denominator = len(noninitial)
    return {
        "method": method,
        "noninitial_query_count": denominator,
        "arm_nomination_count": arm_nominated,
        "gripper_nomination_count": gripper_nominated,
        "arm_nomination_fraction": arm_nominated / denominator if denominator else 0.0,
        "gripper_nomination_fraction": gripper_nominated / denominator if denominator else 0.0,
        "arm_selected_trigger_count": arm_selected,
        "gripper_selected_trigger_count": gripper_selected,
        "both_nomination_count": both_nominated,
        "both_nearby_count": both_nearby,
        "both_nearby_fraction_of_both": both_nearby / both_nominated if both_nominated else None,
        "trigger_reason_counts": dict(Counter(str(row["trigger_reason"]) for row in query_log)),
        "horizon_histogram": {
            str(horizon): sum(int(row["h_exec"] == horizon) for row in query_log)
            for horizon in range(4, 17)
        },
    }


def run_episode(
    *,
    env,
    policy,
    processors,
    torch,
    task_key: str,
    method: str,
    state_id: int,
    env_seed: int,
    protocol: dict[str, Any],
    max_steps: int,
    capture_pairing: bool = False,
) -> dict[str, Any]:
    env.envs[0].init_state_id = int(state_id)
    actual_state_id = int(env.envs[0].init_state_id)
    if actual_state_id != int(state_id):
        raise RuntimeError(f"initial state assignment mismatch: {state_id} versus {actual_state_id}")
    reset_act_rng(torch, int(protocol["policy"]["policy_rng_seed"]), int(env_seed))
    policy.reset()
    observation, _ = env.reset(seed=[int(env_seed)])
    initial_observation = flatten_numeric(copy.deepcopy(observation)) if capture_pairing else None
    initial_sim_state = get_sim_state(env) if capture_pairing else None
    initial_processed: dict[str, np.ndarray] | None = {} if capture_pairing else None

    step_log: list[dict[str, Any]] = []
    query_log: list[dict[str, Any]] = []
    prefix_actions: list[np.ndarray] = []
    prefix_sim_states: list[np.ndarray] = []
    prefix_observations: list[dict[str, np.ndarray]] = []
    prefix_source_queries: list[int] = []
    prefix_chunk_offsets: list[int] = []
    initial_chunk: np.ndarray | None = None
    success = False
    completion_step: int | None = None
    done = False
    target_step = 0

    while target_step < int(max_steps):
        query_step = int(target_step)
        chunk = infer_chunk(
            observation,
            env,
            policy,
            processors,
            torch,
            processed_capture=initial_processed if capture_pairing and query_step == 0 else None,
        )
        if capture_pairing and query_step == 0:
            initial_chunk = chunk.copy()
        proposed_horizon, proposal = choose_horizon(method, chunk)
        trigger_reason = "initial_query" if query_step == 0 else str(proposal["trigger_reason"])
        query_log.append(
            {
                "query_physical_step_q": query_step,
                "h_arm": int(proposal["h_arm"]),
                "h_grip": int(proposal["h_grip"]),
                "h_exec": int(proposed_horizon),
                "trigger_reason": trigger_reason,
                "proposal_trigger_reason": str(proposal["trigger_reason"]),
                "arm_triggered": bool(proposal["arm_triggered"]),
                "gripper_triggered": bool(proposal["gripper_triggered"]),
                "both_nominated": bool(proposal["both_nominated"]),
                "both_nearby": bool(proposal["both_nearby"]),
                "arm_boundary_candidates": [int(x) for x in proposal.get("arm_boundary_candidates", [])],
                "gripper_event_candidates": [int(x) for x in proposal.get("gripper_event_candidates", [])],
                "arm_trigger_offset": proposal.get("arm_trigger_offset"),
                "gripper_trigger_offset": proposal.get("gripper_trigger_offset"),
            }
        )

        actual_interval = min(int(proposed_horizon), int(max_steps) - query_step)
        for chunk_offset in range(actual_interval):
            physical_step = query_step + chunk_offset
            action, source_offset = action_from_newest_chunk(chunk, query_step, physical_step)
            if source_offset != chunk_offset:
                raise AssertionError("newest chunk source offset drifted")
            action = action.astype(np.float32, copy=False)
            if capture_pairing:
                prefix_actions.append(action.copy())
                prefix_source_queries.append(query_step)
                prefix_chunk_offsets.append(source_offset)
            observation, reward, terminated, truncated, info = env.step(action[None])
            if capture_pairing:
                prefix_sim_states.append(get_sim_state(env))
                prefix_observations.append(flatten_numeric(copy.deepcopy(observation)))
            terminated = bool(np.asarray(terminated).reshape(-1)[0])
            truncated = bool(np.asarray(truncated).reshape(-1)[0])
            done = terminated or truncated
            if done:
                success = extract_success(info, reward)
                completion_step = physical_step + 1 if success else None
            step_log.append(
                {
                    "physical_target_t": physical_step,
                    "query_physical_step_q": query_step,
                    "chunk_offset": source_offset,
                    "action_source_query_q": query_step,
                    "chosen_executed_action_7d": action.astype(float).tolist(),
                    "success_termination": bool(success) if done else None,
                }
            )
            if done:
                break
        if done:
            break
        next_query = query_step + actual_interval
        if next_query >= int(max_steps):
            target_step = next_query
            break
        expected_next_query = query_step + int(proposed_horizon)
        if next_query != expected_next_query or len(step_log) != expected_next_query:
            raise AssertionError(
                f"dynamic schedule mismatch: after q={query_step}, expected next q={expected_next_query}, "
                f"executed steps={len(step_log)}"
            )
        target_step = expected_next_query

    result: dict[str, Any] = {
        "task": task_key,
        "method": method,
        "environment_seed": int(env_seed),
        "requested_initial_state_id": int(state_id),
        "actual_initial_state_id": actual_state_id,
        "policy_rng_seed": int(protocol["policy"]["policy_rng_seed"]),
        "prediction_horizon": int(protocol["policy"]["prediction_horizon"]),
        "min_h_exec": 4,
        "max_h_exec": 16,
        "success": bool(success),
        "completion_steps": completion_step,
        "environment_steps": len(step_log),
        "policy_queries": len(query_log),
        "query_rate": len(query_log) / float(len(step_log)) if step_log else 0.0,
        "query_steps": [int(row["query_physical_step_q"]) for row in query_log],
        "planned_horizons": [int(row["h_exec"]) for row in query_log],
        "actual_intervals": [
            int(min(row["h_exec"], len(step_log) - row["query_physical_step_q"]))
            for row in query_log
        ],
        "mean_planned_horizon": float(np.mean([row["h_exec"] for row in query_log])),
        "median_planned_horizon": float(np.median([row["h_exec"] for row in query_log])),
        "mean_completion_steps": float(completion_step) if completion_step is not None else None,
        "trigger_summary": summarize_trigger_logs(method, query_log),
        "query_log": query_log,
        "step_log": step_log,
    }
    if capture_pairing:
        result["pairing_trace"] = {
            "initial_observation": initial_observation,
            "initial_processed_input": initial_processed,
            "initial_sim_state": initial_sim_state,
            "initial_chunk": initial_chunk,
            "prefix_actions": prefix_actions,
            "prefix_source_queries": prefix_source_queries,
            "prefix_chunk_offsets": prefix_chunk_offsets,
            "prefix_sim_states": prefix_sim_states,
            "prefix_observations": prefix_observations,
        }
    return result


def _first_requery(trace: dict[str, Any]) -> int:
    steps = [int(x) for x in trace["query_steps"]]
    return steps[1] if len(steps) > 1 else PAIRING_MAX_STEPS


def run_pairing_smoke(protocol: dict[str, Any], task_key: str, gpu: str, output: Path) -> None:
    states = [int(x) for x in protocol["cohort"]["states"]]
    seeds = [int(x) for x in protocol["cohort"]["environment_seeds"]]
    make_fresh_env, policy, processors, torch, _ = build_runtime(protocol, task_key, gpu)
    traces: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        env = make_fresh_env(seeds[0])
        try:
            traces[method] = run_episode(
                env=env,
                policy=policy,
                processors=processors,
                torch=torch,
                task_key=task_key,
                method=method,
                state_id=states[0],
                env_seed=seeds[0],
                protocol=protocol,
                max_steps=PAIRING_MAX_STEPS,
                capture_pairing=True,
            )
        finally:
            env.close()

    reference = traces[METHODS[0]]["pairing_trace"]
    first_requeries = {method: _first_requery(traces[method]) for method in METHODS}
    common_prefix_length = min(first_requeries.values())
    initial_observation = {}
    initial_processed = {}
    initial_sim = {}
    initial_chunks = {}
    comparisons = []
    for method in METHODS[1:]:
        candidate = traces[method]["pairing_trace"]
        initial_observation[method] = compare_array_maps(reference["initial_observation"], candidate["initial_observation"])
        initial_processed[method] = compare_array_maps(reference["initial_processed_input"], candidate["initial_processed_input"])
        initial_sim[method] = compare_array_maps(
            {"state": reference["initial_sim_state"]}, {"state": candidate["initial_sim_state"]}
        )
        initial_chunks[method] = compare_array_maps(
            {"chunk": reference["initial_chunk"]}, {"chunk": candidate["initial_chunk"]}
        )
        action_first = np.asarray(reference["prefix_actions"][:common_prefix_length])
        action_second = np.asarray(candidate["prefix_actions"][:common_prefix_length])
        sim_first = np.asarray(reference["prefix_sim_states"][:common_prefix_length])
        sim_second = np.asarray(candidate["prefix_sim_states"][:common_prefix_length])
        post_first = reference["prefix_observations"][:common_prefix_length]
        post_second = candidate["prefix_observations"][:common_prefix_length]
        actions = compare_array_maps({"prefix": action_first}, {"prefix": action_second})
        sims = compare_array_maps({"prefix": sim_first}, {"prefix": sim_second})
        posts = [compare_array_maps(a, b) for a, b in zip(post_first, post_second, strict=True)]
        passed = (
            initial_observation[method]["exact"]
            and initial_processed[method]["exact"]
            and initial_sim[method]["exact"]
            and actions["exact"]
            and sims["exact"]
            and all(row["exact"] for row in posts)
        )
        comparisons.append(
            {
                "candidate_method": method,
                "common_prefix_length_before_earliest_requery": common_prefix_length,
                "initial_observation": initial_observation[method],
                "initial_processed_input": initial_processed[method],
                "initial_sim_state": initial_sim[method],
            "initial_raw_chunk": initial_chunks[method],
                "common_prefix_actions": actions,
                "common_prefix_sim_states": sims,
                "common_prefix_post_observations_exact": all(row["exact"] for row in posts),
                "passed": bool(passed),
            }
        )
    passed = all(row["passed"] for row in comparisons)
    result = {
        "status": "paired_dynamic_common_prefix_pass" if passed else "paired_dynamic_common_prefix_fail",
        "task": task_key,
        "state_id": states[0],
        "environment_seed": seeds[0],
        "methods": list(METHODS),
        "fresh_environment_per_method": True,
        "first_requery_steps": first_requeries,
        "common_prefix_length_before_earliest_requery": common_prefix_length,
        "method_query_logs": {
            method: {
                "query_steps": traces[method]["query_steps"],
                "query_log": traces[method]["query_log"],
                "prefix_source_queries": traces[method]["pairing_trace"]["prefix_source_queries"],
                "prefix_chunk_offsets": traces[method]["pairing_trace"]["prefix_chunk_offsets"],
            }
            for method in METHODS
        },
        "comparisons": comparisons,
        "passed": passed,
    }
    atomic_json(output, result)
    if not passed:
        raise RuntimeError("strict dynamic common-prefix pairing smoke failed")


def summarize_method(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [bool(row["success"]) for row in episodes]
    environment_steps = sum(int(row["environment_steps"]) for row in episodes)
    queries = sum(int(row["policy_queries"]) for row in episodes)
    horizons = [horizon for row in episodes for horizon in row["planned_horizons"]]
    noninitial = [row for episode in episodes for row in episode["query_log"][1:]]
    arm_nominated = sum(int(row["h_arm"] < MAX_HORIZON) for row in noninitial)
    grip_nominated = sum(int(row["h_grip"] < MAX_HORIZON) for row in noninitial)
    both_nominated = sum(int(row["both_nominated"]) for row in noninitial)
    both_nearby = sum(int(row["both_nearby"]) for row in noninitial)
    return {
        "successes": successes,
        "success_count": int(sum(successes)),
        "episodes": len(episodes),
        "success_rate": float(np.mean(successes)),
        "policy_queries": queries,
        "environment_steps": environment_steps,
        "query_rate": queries / float(environment_steps),
        "mean_planned_horizon": float(np.mean(horizons)),
        "median_planned_horizon": float(np.median(horizons)),
        "horizon_histogram": {str(horizon): horizons.count(horizon) for horizon in range(4, 17)},
        "arm_nomination_count": arm_nominated,
        "gripper_nomination_count": grip_nominated,
        "arm_nomination_fraction": arm_nominated / len(noninitial) if noninitial else 0.0,
        "gripper_nomination_fraction": grip_nominated / len(noninitial) if noninitial else 0.0,
        "both_nomination_count": both_nominated,
        "both_nearby_count": both_nearby,
        "both_nearby_fraction_of_both": both_nearby / both_nominated if both_nominated else None,
        "mean_completion_steps": float(np.mean([row["completion_steps"] for row in episodes if row["completion_steps"] is not None]))
        if any(row["completion_steps"] is not None for row in episodes)
        else None,
        "episodes_detail": episodes,
    }


def run_task(protocol: dict[str, Any], task_key: str, methods: list[str], gpu: str, output: Path, progress_path: Path | None) -> None:
    if any(method not in METHODS for method in methods):
        raise ValueError(f"methods must be a subset of {METHODS}")
    make_fresh_env, policy, processors, torch, max_steps = build_runtime(protocol, task_key, gpu)
    started = time.time()
    output_value: dict[str, Any] = {
        "status": "running",
        "policy": "ACT",
        "protocol": str((ROOT / "protocol.json").resolve()),
        "task": task_key,
        "methods": methods,
        "pairing_protocol": "fresh environment per method/state; dynamic prefix equality only before earliest re-query",
        "started_at": started,
        "methods_result": {},
    }
    progress = {"pid": os.getpid(), "task": task_key, "completed_methods": 0, "completed_episodes": 0}
    write_progress(progress_path, progress)
    try:
        for method in methods:
            episodes: list[dict[str, Any]] = []
            progress["current_method"] = method
            write_progress(progress_path, progress)
            for state_id, env_seed in episode_seed_pairs(protocol):
                env = make_fresh_env(int(env_seed))
                try:
                    episode = run_episode(
                        env=env,
                        policy=policy,
                        processors=processors,
                        torch=torch,
                        task_key=task_key,
                        method=method,
                        state_id=int(state_id),
                        env_seed=int(env_seed),
                        protocol=protocol,
                        max_steps=max_steps,
                    )
                finally:
                    env.close()
                episodes.append(episode)
                progress["completed_episodes"] += 1
                write_progress(progress_path, progress)
            output_value["methods_result"][method] = summarize_method(episodes)
            progress["completed_methods"] += 1
            atomic_json(output, output_value)
        output_value["status"] = "complete"
        output_value["finished_at"] = time.time()
        atomic_json(output, output_value)
        write_progress(progress_path, {**progress, "finished_at": output_value["finished_at"]})
    finally:
        policy.reset()


def semantic_smoke() -> None:
    """CPU-only dynamic schedule and newest-chunk-source checks."""

    stationary = np.zeros((100, 7), dtype=np.float64)
    assert choose_horizon("M0_hard16", stationary)[0] == 16
    assert choose_horizon("M1_arm_phase", stationary)[0] == 4
    no_events = np.zeros((100, 7), dtype=np.float64)
    assert choose_horizon("M2_gripper_event", no_events)[0] == 16
    transition = no_events.copy()
    transition[7:, 6] = -1.0
    h_grip, grip_diag = choose_horizon("M2_gripper_event", transition)
    assert h_grip == 7 and grip_diag["gripper_trigger_offset"] == 7
    arm_trigger = np.zeros((100, 7), dtype=np.float64)
    arm_trigger[1:5, 0] = np.asarray([4.0, 8.0, 8.0, 12.0])
    assert 4 <= choose_horizon("M1_arm_phase", arm_trigger)[0] <= 16
    combined, combined_diag = choose_horizon("M3_group_event_joint", transition)
    assert combined == min(16, combined_diag["h_arm"], combined_diag["h_grip"])
    assert 4 <= combined <= 16
    chunk = np.arange(100 * 7, dtype=np.float64).reshape(100, 7)
    action, offset = action_from_newest_chunk(chunk, 16, 20)
    assert offset == 4
    np.testing.assert_array_equal(action, chunk[4])
    print(json.dumps({"status": "bounded_group_requery_cpu_semantic_smoke_pass", "methods": list(METHODS)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--task")
    parser.add_argument("--methods", default=",".join(ADAPTIVE_METHODS))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--pairing-smoke", action="store_true")
    parser.add_argument("--semantic-smoke", action="store_true")
    args = parser.parse_args()
    if args.semantic_smoke:
        semantic_smoke()
        return
    protocol = load_protocol(args.protocol)
    if args.task not in task_key_list(protocol):
        raise SystemExit(f"task must be one of {task_key_list(protocol)}")
    methods = [value for value in args.methods.split(",") if value]
    if args.pairing_smoke:
        if args.output is None:
            raise SystemExit("--output is required for --pairing-smoke")
        run_pairing_smoke(protocol, args.task, args.gpu, args.output)
        return
    if args.output is None:
        raise SystemExit("--output is required for a rollout")
    run_task(protocol, args.task, methods, args.gpu, args.output, args.progress_file)


if __name__ == "__main__":
    main()
