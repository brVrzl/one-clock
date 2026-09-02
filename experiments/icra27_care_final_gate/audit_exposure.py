#!/usr/bin/env python3
"""Audit proposed Gate M cells against all objects reachable from origin refs."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PROPOSED = {
    1: {30, 32, 33, 36, 37, 40, 41, 42, 43, 46, 49},
    4: {30, 32, 33, 36, 37, 40, 41, 42, 43, 46, 49},
}
for task in (2, 3, 5, 6, 7, 8, 9):
    PROPOSED[task] = {24, 25, 26, 28, 29, 30, 32, 33, 36, 37, 40, 41, 42, 43, 46, 49}

OUTCOME_PATH = re.compile(r"(^|/)(results?|analysis|condition_shards?|rollouts?|episodes?)(/|\.|_)", re.I)
TASK_PATH = re.compile(r"libero_object[_:/-]*task[_-]?0*([0-9]+)", re.I)
STATE_KEYS = ("state_id", "initial_state_id", "requested_initial_state_id")
STATE_LIST_KEYS = ("paired_initial_state_ids", "initial_state_ids", "states", "state_ids", "requested_initial_state_ids")
OUTCOME_KEYS = {"success", "successes", "success_count", "success_rate", "is_success", "episode_success", "reward", "outcome", "methods_result"}


def git(*args: str, text: bool = True):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=text)


def task_context(value: dict, path: str, inherited: int | None) -> int | None:
    for key in ("task", "task_key"):
        item = value.get(key)
        if isinstance(item, str):
            match = re.search(r"libero_object\s*[:/_-]\s*task\s*([0-9]+)", item, re.I)
            if match:
                return int(match.group(1))
    if value.get("suite") == "libero_object" and isinstance(value.get("task_id"), int):
        return int(value["task_id"])
    match = TASK_PATH.search(path)
    return int(match.group(1)) if match else inherited


def structured_matches(value: Any, path: str, task: int | None = None) -> set[tuple[int, int, str]]:
    matches: set[tuple[int, int, str]] = set()
    if isinstance(value, dict):
        task = task_context(value, path, task)
        if task in PROPOSED and any(key in value for key in OUTCOME_KEYS):
            for key in STATE_KEYS:
                state = value.get(key)
                if isinstance(state, int) and state in PROPOSED[task]:
                    matches.add((task, state, key))
            for key in STATE_LIST_KEYS:
                states = value.get(key)
                if isinstance(states, list) and all(isinstance(state, int) for state in states):
                    matches.update((task, state, key) for state in states if state in PROPOSED[task])
        for child in value.values():
            matches.update(structured_matches(child, path, task))
    elif isinstance(value, list):
        for child in value:
            matches.update(structured_matches(child, path, task))
    return matches


def main() -> None:
    remote_refs = []
    for line in git("for-each-ref", "--format=%(refname:short) %(objectname)", "refs/remotes/origin").splitlines():
        ref, sha = line.split()
        remote_refs.append({"ref": ref, "sha": sha})
    commits = git("rev-list", "--remotes=origin").splitlines()
    objects = git("rev-list", "--objects", "--remotes=origin").splitlines()
    seen: set[str] = set()
    parsed = 0
    evidence = []
    for line in objects:
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        oid, path = parts
        if oid in seen or not path.lower().endswith((".json", ".jsonl")) or not OUTCOME_PATH.search(path):
            continue
        seen.add(oid)
        if git("cat-file", "-t", oid).strip() != "blob":
            continue
        raw = git("cat-file", "-p", oid, text=False)
        try:
            text = raw.decode("utf-8")
            value = (
                [json.loads(row) for row in text.splitlines() if row.strip()]
                if path.lower().endswith(".jsonl")
                else json.loads(text)
            )
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        parsed += 1
        for task, state, field in structured_matches(value, path):
            evidence.append({"task_id": task, "state_id": state, "path": path, "blob": oid, "matched_field": field})
    unique = {}
    for item in evidence:
        unique[(item["task_id"], item["state_id"], item["path"], item["blob"])] = item
    evidence = sorted(unique.values(), key=lambda item: (item["task_id"], item["state_id"], item["path"]))
    exposed = {(item["task_id"], item["state_id"]) for item in evidence}
    expected_exposed = {(6, 25), (6, 26), (6, 28), (6, 29)}
    if exposed != expected_exposed:
        raise RuntimeError(f"unexpected proposed-cell exposure set: {sorted(exposed)}")
    final = {str(task): sorted(states - {state for found_task, state in exposed if found_task == task}) for task, states in sorted(PROPOSED.items())}
    final_count = sum(len(states) for states in final.values())
    if final_count != 130:
        raise RuntimeError(f"expected 130 final blocks, got {final_count}")
    result = {
        "audit_scope": "all Git objects reachable from every fetched refs/remotes/origin/* tip",
        "remote_refs": remote_refs,
        "remote_ref_count": len(remote_refs),
        "reachable_commit_count": len(commits),
        "reachable_object_count": len(objects),
        "unique_outcome_json_or_jsonl_blobs_parsed": parsed,
        "latest_authoritative_inventory": {
            "commit": "eb4e29a62b28010c714d961f42449ae33bbe2312",
            "path": "experiments/icra27_overnight_smolvla_crosspolicy/exposure_inventory.md",
        },
        "proposed_blocks": sum(len(states) for states in PROPOSED.values()),
        "additional_raw_outcome_exposure": evidence,
        "exposed_cells_removed": [{"task_id": task, "state_id": state} for task, state in sorted(exposed)],
        "exposure_source_commit": "38046a961cd796b30b554c9de407d64aa82518cf",
        "exposure_source_blob": "018c26fbb8ada57dc459bfaa3b91403f87a99ca5",
        "final_states_by_task": final,
        "final_clean_block_count": final_count,
        "minimum_required": 120,
        "proceed": final_count >= 120,
        "protocol_only_exposure_note": "No structured protocol-only candidate cell was used as an exclusion; the four removals each have recorded rollout success outcomes.",
        "conclusion": "No outcome record was found for any of the remaining 130 cells before preregistration.",
    }
    (ROOT / "exposure_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Gate M remote exposure audit",
        "",
        f"The audit inspected {len(remote_refs)} fetched `origin/*` refs, {len(commits)} reachable commits, {len(objects)} reachable objects, and parsed {parsed} unique outcome JSON/JSONL blobs.",
        "",
        "The previously proposed 134 blocks contained four additional raw outcome exposures, all in `experiments/component_temporal_reuse/rapid_component_smoke/results_libero_object_task6.json` at commit `38046a961cd796b30b554c9de407d64aa82518cf` (blob `018c26fbb8ada57dc459bfaa3b91403f87a99ca5`): task 6 states 25, 26, 28, and 29.",
        "",
        "Those four blocks and only those blocks were removed. No replacement states were added. The frozen Gate M cohort contains **130 paired blocks**, above the minimum of 120.",
        "",
        "No outcome record was found for any remaining cell before preregistration. Protocol-only exposure was not used as an automatic exclusion.",
        "",
        "Exact final states by task:",
        "",
    ]
    for task, states in final.items():
        lines.append(f"- Object task {task}: {','.join(str(state) for state in states)}")
    (ROOT / "exposure_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"proposed": 134, "removed": sorted(exposed), "final": final_count, "parsed_outcome_blobs": parsed}))


if __name__ == "__main__":
    main()

