"""Validate one completed C1/C2 task shard before analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from asymmetric_executor import C1, C2, H16, METHODS


ROOT = Path(__file__).resolve().parent


def validate_episode(episode: dict, method: str, task_id: int, state_id: int) -> None:
    assert int(episode["task_id"]) == task_id
    assert episode["method"] == method
    assert int(episode["requested_initial_state_id"]) == state_id
    assert episode["fresh_environment_instance"] is True
    assert int(episode["environment_construction_seed"]) == int(episode["environment_seed"])
    assert int(episode["max_episode_steps"]) == 280
    steps = int(episode["environment_steps"])
    assert 1 <= steps <= 280
    rows = episode["step_log"]
    assert len(rows) == steps
    expected_scheduled = list(range(0, steps, H16))
    query_steps = [int(x) for x in episode["query_steps"]]
    scheduled_query_steps = [int(x) for x in episode["scheduled_query_steps"]]
    assert scheduled_query_steps == expected_scheduled
    if method == C1:
        assert query_steps == expected_scheduled
        assert int(episode["policy_queries"]) == (steps + H16 - 1) // H16
        assert episode["compute_parity_assertions"]["valid"] is True
    else:
        assert query_steps == list(range(steps))
        assert int(episode["policy_queries"]) == steps
        assert episode["compute_parity_assertions"]["arm_source_chunk_count_exact"] is True
        assert int(episode["distinct_arm_source_chunks"]) == (steps + H16 - 1) // H16
    assert float(episode["query_rate"]) == len(query_steps) / steps
    assert float(episode["wall_clock_seconds"]) > 0
    assert int(episode["policy_call_count_for_latency"]) == len(query_steps)
    assert float(episode["mean_policy_call_latency_seconds"]) >= 0

    for target_t, row in enumerate(rows):
        required = ("t", "queried", "query_q", "arm_source_q", "arm_offset", "grip_source_q", "grip_offset", "executed_action_7d")
        assert all(key in row for key in required)
        assert int(row["t"]) == target_t
        assert bool(row["queried"]) == (target_t in query_steps)
        assert int(row["arm_source_q"]) + int(row["arm_offset"]) == target_t
        assert int(row["grip_source_q"]) + int(row["grip_offset"]) == target_t
        assert int(row["arm_source_age"]) == target_t - int(row["arm_source_q"])
        assert int(row["gripper_source_age"]) == target_t - int(row["grip_source_q"])
        assert np.asarray(row["executed_action_7d"], dtype=np.float64).shape == (7,)
        assert np.isfinite(np.asarray(row["executed_action_7d"], dtype=np.float64)).all()
        np.testing.assert_array_equal(np.asarray(row["action"], dtype=np.float64), np.asarray(row["executed_action_7d"], dtype=np.float64))
        scheduled_action = np.asarray(row["scheduled_action"], dtype=np.float64)
        action = np.asarray(row["executed_action_7d"], dtype=np.float64)
        assert scheduled_action.shape == (7,)
        if method == C1:
            assert int(row["arm_source_q"]) % H16 == 0
            assert int(row["scheduled_source_q"]) == int(row["arm_source_q"])
            assert int(row["scheduled_chunk_offset"]) == int(row["arm_offset"])
            assert bool(row["gripper_source_chunk_cached"]) is True
            assert int(row["grip_source_q"]) % H16 == 0
            assert int(row["grip_offset"]) < 100
            if target_t < H16:
                assert int(row["grip_source_q"]) == 0
                assert int(row["grip_offset"]) == target_t
                assert row["previous_action"] is None
                np.testing.assert_array_equal(action, scheduled_action)
            else:
                assert 16 <= int(row["grip_offset"]) <= 31
                previous_action = np.asarray(row["previous_action"], dtype=np.float64)
                assert previous_action.shape == (7,)
                expected = scheduled_action.copy()
                expected[6] = previous_action[6]
                np.testing.assert_array_equal(action, expected)
        else:
            assert int(row["arm_source_q"]) % H16 == 0
            assert int(row["scheduled_source_q"]) == int(row["arm_source_q"])
            assert int(row["scheduled_chunk_offset"]) == int(row["arm_offset"])
            assert int(row["grip_source_q"]) == target_t
            assert int(row["grip_offset"]) == 0
            assert row["previous_action"] is None
            fresh_action = np.asarray(row["fresh_action"], dtype=np.float64)
            assert fresh_action.shape == (7,)
            expected = scheduled_action.copy()
            expected[6] = fresh_action[6]
            np.testing.assert_array_equal(action, expected)


def validate_shard(path: Path, protocol_path: Path, marker: Path | None = None) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads(path.read_text(encoding="utf-8"))
    task_id = int(result["task_id"])
    assert task_id in protocol["cohort"]["primary_task_ids"]
    assert result["methods"] == list(METHODS)
    state_ids = [int(x) for x in protocol["cohort"]["state_ids"]]
    assert result["state_ids"] == state_ids
    assert result["finished"] is True
    total = 0
    for method in METHODS:
        episodes = result["episodes"][method]
        assert len(episodes) == len(state_ids)
        observed_states = [int(episode["requested_initial_state_id"]) for episode in episodes]
        assert observed_states == state_ids
        expected_seeds = protocol["cohort"]["environment_seeds_by_task"][str(task_id)]
        for episode, state_id, expected_seed in zip(episodes, state_ids, expected_seeds, strict=True):
            assert int(episode["environment_seed"]) == int(expected_seed)
            validate_episode(episode, method, task_id, state_id)
        total += len(episodes)
    assert total == 28
    summary = {"valid": True, "task_id": task_id, "episodes": total, "result": str(path.resolve())}
    if marker is not None:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--marker", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_shard(args.result, args.protocol, args.marker)))


if __name__ == "__main__":
    main()
