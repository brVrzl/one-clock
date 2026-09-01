from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = EXPERIMENT_ROOT / "protocol.json"


def find_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(find_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(find_keys(item) for item in value)) if value else set()
    return set()


def test_frozen_candidate1_protocol_contract() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["status"] == "frozen_before_outcome_rollout"
    created = datetime.fromisoformat(protocol["created_utc"].replace("Z", "+00:00"))
    assert created.tzinfo == timezone.utc
    assert protocol["authorized_condition"] == "C2_H16_ARM_FRESH_GRIP"
    assert protocol["candidate_2_authorized"] is False
    assert protocol["method_development_closed"] is True
    assert protocol["cohort"]["primary_tasks"] == {
        "libero_goal": [4, 6, 7, 8, 9],
        "libero_10": [0, 2, 4, 6, 7],
    }
    assert protocol["cohort"]["state_ids"] == list(range(14))
    assert protocol["cohort"]["paired_blocks"] == 140
    assert protocol["rollout"]["new_c2_experimental_episodes"] == 140
    assert protocol["runtime"]["policy_rng_seed"] == 424242
    assert protocol["runtime"]["policy_checkpoint_chunk_size"] == 100
    assert protocol["runtime"]["action_dim"] == 7
    assert protocol["runtime"]["policy_temporal_ensemble"] is False
    assert protocol["runtime"]["action_smoothing"] is False
    assert protocol["condition"]["arm"]["scheduled_source_period_steps"] == 16
    assert protocol["condition"]["policy_query"]["total_policy_query_period_steps"] == 1
    assert protocol["condition"]["policy_query"]["expected_total_policy_query_rate"] == 1.0
    assert protocol["condition"]["gripper"]["prediction_offset"] == 0
    assert protocol["condition"]["gripper"]["source_age_steps"] == 0
    assert "expected_query_rate" not in find_keys(protocol)


def test_seed_sentinels_and_all_task_seed_lists() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["cohort"]["seed_sentinels"] == {
        "libero_goal_task4_state0": 342400,
        "libero_goal_task9_state13": 342913,
        "libero_10_task0_state0": 343000,
        "libero_10_task7_state13": 343713,
    }
    suite_index = protocol["cohort"]["suite_index"]
    for task in protocol["cohort"]["tasks"]:
        assert task["environment_seeds"] == [
            340000 + 1000 * suite_index[task["suite"]] + 100 * task["task_id"] + state_id
            for state_id in range(14)
        ]


def test_checkpoint_paths_match_cross_suite_confirmation_exactly() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    cross_protocol = json.loads(
        (EXPERIMENT_ROOT.parent / "cross_suite_confirmation" / "protocol.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {
        (task["suite"], task["task_id"]): task["checkpoint"]
        for task in cross_protocol["cohort"]["tasks"]
        if task["role"] == "primary_unseen_to_executor_development"
    }
    actual = {
        (task["suite"], task["task_id"]): task["checkpoint"]
        for task in protocol["cohort"]["tasks"]
    }
    assert actual == expected
