#!/usr/bin/env python3
"""Audit non-Object ACT task/state exposure from the frozen remote-ref snapshot."""

from __future__ import annotations

import gzip
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SUITES = ("libero_spatial", "libero_goal", "libero_10")
STATE_KEYS = ("state_id", "initial_state_id", "requested_initial_state_id", "init_state_id")
STATE_LIST_KEYS = (
    "state_ids", "states", "initial_state_ids", "requested_initial_state_ids",
    "paired_initial_state_ids",
)
METHOD_KEYS = ("method", "condition", "strategy", "name")
OUTCOME_KEYS = {
    "success", "successes", "success_count", "success_rate", "is_success",
    "episode_success", "reward", "outcome", "terminal_reward", "methods_result",
    "environment_steps", "completion_step",
}
OUTCOME_PATH = re.compile(r"(^|/)(results?|analysis|condition_shards?|rollouts?|episodes?)(/|\.|_)", re.I)
TASK_PATH = re.compile(r"(libero_(?:spatial|goal|10))[_:/-]*task[_-]?0*([0-9]+)", re.I)
EPISODE_PATH = re.compile(
    r"gate4a2_spatial_act_generalization/episodes/task_0*([0-9]+)/state_0*([0-9]+)/([^/]+)\.json\.gz$",
    re.I,
)


def git(*args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=text)


def normalize_method(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    token = re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")
    aliases = {
        "H4": "H4", "COHERENT_H4": "H4", "GLOBAL_H4": "H4",
        "ARM4_GRIP4": "H4", "GROUP_ARM4_GRIP4": "H4",
        "H2": "H2", "COHERENT_H2": "H2", "GLOBAL_H2": "H2",
        "ARM2_GRIP2": "H2", "GROUP_ARM2_GRIP2": "H2",
        "ARM4_GRIP32": "ARM4_GRIP32", "GROUP_ARM4_GRIP32": "ARM4_GRIP32",
        "ARM2_GRIP16": "ARM2_GRIP16", "GROUP_ARM2_GRIP16": "ARM2_GRIP16",
    }
    return aliases.get(token)


def infer_policy(path: str, value: dict[str, Any] | None = None) -> str:
    candidates = [] if value is None else [value.get("policy"), value.get("policy_name"), value.get("model")]
    joined = " ".join(str(item) for item in candidates if item is not None).lower() + " " + path.lower()
    if "smolvla" in joined:
        return "SmolVLA"
    return "ACT"


def infer_task(path: str, value: dict[str, Any], inherited: tuple[str, int] | None) -> tuple[str, int] | None:
    suite = value.get("suite") or value.get("task_group")
    task_id = value.get("task_id")
    if suite in SUITES and isinstance(task_id, int):
        return str(suite), int(task_id)
    for key in ("task", "task_key", "cell_id", "block_id"):
        item = value.get(key)
        if isinstance(item, str):
            match = TASK_PATH.search(item)
            if match:
                return match.group(1).lower(), int(match.group(2))
    match = TASK_PATH.search(path)
    if match:
        return match.group(1).lower(), int(match.group(2))
    return inherited


def infer_method(value: dict[str, Any], inherited: str | None) -> str | None:
    for key in METHOD_KEYS:
        method = value.get(key)
        if isinstance(method, str):
            return method
    return inherited


def structured_events(
    value: Any,
    path: str,
    task: tuple[str, int] | None = None,
    method: str | None = None,
    policy: str | None = None,
) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        task = infer_task(path, value, task)
        method = infer_method(value, method)
        policy = infer_policy(path, value) if policy is None or any(k in value for k in ("policy", "policy_name", "model")) else policy
        has_outcome = any(key in value for key in OUTCOME_KEYS)
        states: set[int] = set()
        if has_outcome:
            for key in STATE_KEYS:
                state = value.get(key)
                if isinstance(state, int) and 0 <= state <= 49:
                    states.add(int(state))
            for key in STATE_LIST_KEYS:
                sequence = value.get(key)
                if isinstance(sequence, list) and all(isinstance(item, int) for item in sequence):
                    states.update(int(item) for item in sequence if 0 <= item <= 49)
        if task is not None and task[0] in SUITES:
            for state in states:
                yield {
                    "suite": task[0], "task_id": task[1], "state_id": state,
                    "policy": policy or infer_policy(path), "method": method,
                }
        for key, child in value.items():
            child_method = method
            if isinstance(child, (dict, list)) and isinstance(key, str):
                if key not in {"metrics", "episodes", "results", "methods_result", "per_task"}:
                    child_method = key
            yield from structured_events(child, path, task, child_method, policy)
    elif isinstance(value, list):
        for child in value:
            yield from structured_events(child, path, task, method, policy)


def provenance_for_blob(blob: str, path: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    log = str(git("log", "--remotes=origin", "--all", "--format=%H", "--find-object=" + blob, "--", path)).splitlines()
    commit = log[-1] if log else None
    refs = []
    if commit:
        refs = str(git("branch", "-r", "--contains", commit, "--format=%(refname:short)")).splitlines()
    snapshot_refs = {row["ref"] for row in snapshot["remote_refs"]}
    refs = sorted(ref.strip() for ref in refs if ref.strip() in snapshot_refs)
    return {"commit": commit, "branch_refs": refs, "artifact": path, "blob": blob}


def select_states(executor_exposed: set[tuple[str, int, int]], suite: str, task: int) -> list[int]:
    primary = [state for state in range(20, 50) if (suite, task, state) not in executor_exposed]
    selected = primary[:15]
    if len(selected) < 15:
        fallback = [state for state in range(19, -1, -1) if (suite, task, state) not in executor_exposed]
        selected.extend(fallback[: 15 - len(selected)])
    return selected


def main() -> None:
    snapshot = json.loads((ROOT / "remote_ref_snapshot.json").read_text(encoding="utf-8"))
    current = {
        row.split()[0]: row.split()[1]
        for row in str(git("for-each-ref", "--format=%(refname:short) %(objectname)", "refs/remotes/origin")).splitlines()
    }
    frozen = {row["ref"]: row["sha"] for row in snapshot["remote_refs"]}
    if current != frozen:
        raise RuntimeError("remote refs differ from the frozen pre-selection snapshot")

    objects = str(git("rev-list", "--objects", *[row["sha"] for row in snapshot["remote_refs"]])).splitlines()
    seen: set[tuple[str, str]] = set()
    raw_events: list[dict[str, Any]] = []
    parsed = 0
    for line in objects:
        oid_path = line.split(" ", 1)
        if len(oid_path) != 2:
            continue
        oid, path = oid_path
        key = (oid, path)
        if key in seen or not OUTCOME_PATH.search(path):
            continue
        seen.add(key)
        episode_match = EPISODE_PATH.search(path)
        if episode_match:
            raw_events.append({
                "suite": "libero_spatial", "task_id": int(episode_match.group(1)),
                "state_id": int(episode_match.group(2)), "policy": "ACT",
                "method": episode_match.group(3), "path": path, "blob": oid,
            })
            continue
        if not path.lower().endswith((".json", ".jsonl", ".json.gz", ".jsonl.gz")):
            continue
        if str(git("cat-file", "-t", oid)).strip() != "blob":
            continue
        raw = git("cat-file", "-p", oid, text=False)
        try:
            if path.lower().endswith(".gz"):
                raw = gzip.decompress(raw)
            text = raw.decode("utf-8")
            value = (
                [json.loads(row) for row in text.splitlines() if row.strip()]
                if ".jsonl" in path.lower()
                else json.loads(text)
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        parsed += 1
        for event in structured_events(value, path):
            event.update({"path": path, "blob": oid})
            raw_events.append(event)

    # Deduplicate repeated aggregate/nested matches without losing distinct methods or artifacts.
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for event in raw_events:
        key = (
            event["suite"], event["task_id"], event["state_id"], event["policy"],
            event.get("method"), event["path"], event["blob"],
        )
        unique[key] = event
    raw_events = list(unique.values())

    by_cell: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for event in raw_events:
        if event["policy"] == "ACT":
            by_cell[(event["suite"], int(event["task_id"]), int(event["state_id"]))].append(event)

    provenance_cache: dict[tuple[str, str], dict[str, Any]] = {}
    candidate_rows = []
    excluded_records = []
    executor_exposed: set[tuple[str, int, int]] = set()
    exact_anywhere = []
    for suite in SUITES:
        for task in range(10):
            for state in range(50):
                key = (suite, task, state)
                events = by_cell.get(key, [])
                exact_events = [event for event in events if normalize_method(event.get("method")) is not None]
                other_events = [event for event in events if normalize_method(event.get("method")) is None]
                baseline_exposed = True  # Existing eval10 exists for every audited task policy.
                other_exposed = bool(other_events)
                exact_exposed = bool(exact_events)
                if events:
                    executor_exposed.add(key)
                row = {
                    "suite": suite,
                    "task_id": task,
                    "state_id": state,
                    "BASELINE_EXPOSED": baseline_exposed,
                    "BASELINE_EXPOSURE_GRANULARITY": "task-policy level; existing standard eval10, state identity not used for exclusion",
                    "OTHER_EXECUTOR_EXPOSED": other_exposed,
                    "QUERY_ALLOCATION_CONDITION_EXPOSED": exact_exposed,
                    "TRACK_A_CELL_PROSPECTIVE": not exact_exposed,
                    "conservative_executor_outcome_exposed": bool(events),
                }
                candidate_rows.append(row)
                if exact_events:
                    exact_anywhere.append({"suite": suite, "task_id": task, "state_id": state})

    selected_by_task = {}
    for suite in SUITES:
        for task in range(10):
            selected = select_states(executor_exposed, suite, task)
            selected_by_task[f"{suite}:task{task}"] = selected
            if len(selected) < 15:
                # The rule requires retaining the task with all eligible states.
                pass
            for state in range(50):
                if state in selected:
                    continue
                events = by_cell.get((suite, task, state), [])
                if not events:
                    continue
                evidence = []
                for event in events:
                    cache_key = (event["blob"], event["path"])
                    if cache_key not in provenance_cache:
                        provenance_cache[cache_key] = provenance_for_blob(event["blob"], event["path"], snapshot)
                    prov = provenance_cache[cache_key]
                    evidence.append({
                        "experiment": event["path"].split("/")[1] if event["path"].startswith("experiments/") else event["path"].split("/")[0],
                        "policy": event["policy"],
                        "method": event.get("method"),
                        **prov,
                    })
                excluded_records.append({
                    "suite": suite, "task_id": task, "state_id": state,
                    "reason": "prior executor-variant outcome",
                    "evidence": sorted(evidence, key=lambda item: (item["artifact"], str(item.get("method")))),
                })

    selected_keys = {
        (suite_task.split(":task")[0], int(suite_task.split(":task")[1]), state)
        for suite_task, states in selected_by_task.items() for state in states
    }
    selected_rows = [row for row in candidate_rows if (row["suite"], row["task_id"], row["state_id"]) in selected_keys]
    if any(row["conservative_executor_outcome_exposed"] for row in selected_rows):
        raise RuntimeError("deterministic cohort contains a prior executor-outcome-exposed cell")
    if any(not row["TRACK_A_CELL_PROSPECTIVE"] for row in selected_rows):
        raise RuntimeError("deterministic cohort contains an exact-condition-exposed ACT cell")

    cohort = {
        "schema_version": 1,
        "remote_ref_snapshot": "remote_ref_snapshot.json",
        "state_selection_rule": [
            "For each non-Object task policy consider state IDs 20..49.",
            "Remove states with any prior executor-variant outcome.",
            "Take the lowest 15 remaining state IDs.",
            "If fewer than 15 remain, inspect 0..19 and take the highest remaining executor-outcome-unexposed IDs until reaching 15.",
            "If fewer than 15 total eligible states exist across 0..49, use all eligible states and do not exclude the task.",
            "Never use success rate, task difficulty, or previous executor effect to choose or replace a state.",
        ],
        "selected_states_by_task": selected_by_task,
        "task_count": len(selected_by_task),
        "selected_block_count": len(selected_rows),
        "all_selected_track_a_cell_prospective": all(row["TRACK_A_CELL_PROSPECTIVE"] for row in selected_rows),
        "all_selected_conservatively_executor_outcome_unexposed": all(not row["conservative_executor_outcome_exposed"] for row in selected_rows),
        "selected_cell_exposure_labels": selected_rows,
    }
    (ROOT / "confirmation_cohort.json").write_text(json.dumps(cohort, indent=2) + "\n", encoding="utf-8")
    audit = {
        "schema_version": 1,
        "scope": "all objects reachable from every ref in remote_ref_snapshot.json",
        "parsed_outcome_json_or_jsonl_blobs": parsed,
        "candidate_cell_count": len(candidate_rows),
        "candidate_cell_exposure_labels": candidate_rows,
        "exact_query_allocation_condition_exposure_on_non_object_task_specific_act": sorted(
            exact_anywhere, key=lambda row: (row["suite"], row["task_id"], row["state_id"])
        ),
        "excluded_state_records": sorted(excluded_records, key=lambda row: (row["suite"], row["task_id"], row["state_id"])),
        "selected_block_count": len(selected_rows),
    }
    (ROOT / "exposure_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    counts = defaultdict(int)
    for row in candidate_rows:
        counts["baseline"] += int(row["BASELINE_EXPOSED"])
        counts["other"] += int(row["OTHER_EXECUTOR_EXPOSED"])
        counts["exact"] += int(row["QUERY_ALLOCATION_CONDITION_EXPOSED"])
    lines = [
        "# Track-A exposure audit",
        "",
        f"The audit used the frozen 28-ref snapshot and parsed {parsed} unique outcome JSON/JSONL artifacts plus path-identified compressed episode artifacts. No success magnitude entered state selection.",
        "",
        "Exposure is recorded per task-specific ACT task/state cell with four distinct fields: `BASELINE_EXPOSED`, `OTHER_EXECUTOR_EXPOSED`, `QUERY_ALLOCATION_CONDITION_EXPOSED`, and `TRACK_A_CELL_PROSPECTIVE`. Standard baseline exposure is recorded at task-policy granularity because every policy has an existing eval10; it does not exclude a state.",
        "",
        f"Across the 1,500 candidate non-Object ACT cells, {counts['other']} have another executor outcome and {counts['exact']} have an exact H4/H2/ARM4_GRIP32/ARM2_GRIP16-family outcome. Cross-policy SmolVLA outcomes are not treated as exposure of the task-specific ACT policy cell.",
        "",
        f"The deterministic rule selected **{len(selected_rows)} blocks across {len(selected_by_task)} task policies**. Every selected cell is `TRACK_A_CELL_PROSPECTIVE=true` and conservatively free of any prior ACT executor-variant outcome.",
        "",
        "The preregistered scientific wording is: **query-allocation conditions frozen from Object development were prospectively evaluated on non-Object task-state cells selected without reference to their query-allocation outcomes.** The policies and suites are not described as unseen or globally executor-unexposed.",
        "",
        "## Deterministic selected states",
        "",
    ]
    for key, states in selected_by_task.items():
        lines.append(f"- `{key}`: {','.join(map(str, states))}")
    lines += [
        "",
        "## Exclusion provenance",
        "",
        "Every conservatively excluded state with an outcome has experiment, remote ref(s), introducing commit, artifact path, and blob recorded in `exposure_audit.json`. States not chosen merely because they followed the first 15 eligible IDs are not exposure exclusions.",
        "",
    ]
    (ROOT / "exposure_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "candidate_cells": len(candidate_rows),
        "other_executor_exposed": counts["other"],
        "exact_condition_exposed": counts["exact"],
        "selected_blocks": len(selected_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
