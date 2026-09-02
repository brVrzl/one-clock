#!/usr/bin/env python3
"""Fail closed on any drift in the frozen Phase-0 artifacts."""

from __future__ import annotations

import json
import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "src"))

from conditions import CONDITION_ORDER, CONDITIONS


def read(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completed-track-a", action="store_true")
    args = parser.parse_args()
    inventory, cohort = read("checkpoint_inventory.json"), read("confirmation_cohort.json")
    a, b = read("track_a_manifest.json"), read("track_b_manifest.json")
    assert inventory["summary"]["technically_valid_non_object"] == 30
    assert a["task_count"] == 30 and a["states_per_task"] in (10, 15)
    assert a["paired_block_count"] == 30 * a["states_per_task"]
    assert a["scientific_episode_count"] == a["paired_block_count"] * 6
    assert tuple(a["condition_order"]) == CONDITION_ORDER
    assert list(a["conditions"]) == list(CONDITION_ORDER)
    assert a["outcomes_inspected_before_freeze"] is False
    blocks = defaultdict(list)
    for cell in a["cells"]:
        blocks[cell["block_id"]].append(cell)
    assert len(blocks) == a["paired_block_count"]
    for block_cells in blocks.values():
        assert tuple(cell["method"] for cell in block_cells) == CONDITION_ORDER
        invariant = {(c["suite"], c["task_id"], c["state_id"], c["environment_seed"], c["policy_seed"], c["checkpoint"]) for c in block_cells}
        assert len(invariant) == 1
        for cell in block_cells:
            condition = CONDITIONS[cell["method"]]
            assert cell["arm_horizon"] == condition.arm_horizon
            assert cell["gripper_horizon"] == condition.gripper_horizon
    selected = defaultdict(list)
    for block_cells in blocks.values():
        first = block_cells[0]
        selected[f"{first['suite']}:task{first['task_id']}"] .append(first["state_id"])
    for key, states in selected.items():
        assert states == cohort["selected_states_by_task"][key][:a["states_per_task"]]
    assert b["total_episodes"] == 80 and b["cells_per_policy"] == 40
    assert len(b["cells"]) == 80
    assert set(b["panel"]) == {"libero_object:task3", "libero_spatial:task0", "libero_goal:task2", "libero_10:task3"}
    assert b["scope"] == "mechanism-only logging on already outcome-exposed development cells; outcomes are not used for method selection"
    track_b_canary = json.loads((ROOT / "track_b/canary.json").read_text())
    assert track_b_canary["status"] == "PASS"
    assert {row["policy"] for row in track_b_canary["comparisons"]} == {"ACT", "SmolVLA"}
    assert all(all(row["checks"].values()) for row in track_b_canary["comparisons"])
    for suite in ("libero_spatial", "libero_goal", "libero_10"):
        smoke = json.loads((ROOT / "phase0_smoke" / f"{suite}.json").read_text())
        assert smoke["status"] == "PASS" and not smoke["scientific_outcomes_used"]
        assert all(smoke["H4_equals_group_arm4_grip4"].values())
        assert all(smoke["H2_equals_group_arm2_grip2"].values())
        assert all(smoke["repeated_load_unload_and_rng_isolation"].values())
    assert read("te_dense_audit.json")["status"] == "PASS"
    assert read("historical_operator_audit.json")["historical_sign_vote_exists"] is True
    assert read("normalization_buffer_audit.json")["status"] == "PASS"
    assert len(read("gripper_activity_moderator.json")["tasks"]) == 30
    manuscript_diff = subprocess.check_output(
        ["git", "diff", "--name-only", "92ed6b281b5c85ff526c2e84ce47b684e86c9d7a", "--", "paper", "CLAIMS.md"],
        cwd=REPO, text=True,
    ).strip()
    assert not manuscript_diff, manuscript_diff
    outcome_paths = [ROOT / "track_a/results", ROOT / "track_a/markers", ROOT / "track_a/attempts"]
    scientific_files = [path for directory in outcome_paths if directory.exists() for path in directory.iterdir() if path.is_file()]
    if args.completed_track_a:
        complete = list((ROOT / "track_a/markers").glob("*.complete"))
        failed = list((ROOT / "track_a/markers").glob("*.technical_failed"))
        assert len(complete) == len(a["cells"]) and not failed
    else:
        assert not scientific_files, f"Track-A scientific artifacts exist before freeze: {scientific_files[:5]}"
    print(json.dumps({"status": "PASS", "tasks": a["task_count"], "blocks": len(blocks), "episodes": len(a["cells"]), "track_b_episodes": len(b["cells"]), "track_a_outcomes_present": bool(scientific_files), "mode": "completed" if args.completed_track_a else "pre_outcome"}, indent=2))


if __name__ == "__main__":
    main()
