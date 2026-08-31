"""Validate one completed or partial repaired factorial task shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from temporal_reuse import DELAY, FIXED_METHODS, H16, METHODS


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
    query_steps = [int(x) for x in episode["query_steps"]]
    expected_query_steps = list(range(steps)) if method in FIXED_METHODS else list(range(0, steps, H16))
    assert query_steps == expected_query_steps
    assert int(episode["policy_queries"]) == len(query_steps)
    assert float(episode["query_rate"]) == len(query_steps) / steps

    for target_t, row in enumerate(rows):
        assert int(row["physical_target_t"]) == target_t
        arm_q = int(row["arm_source_query_q"])
        arm_offset = int(row["arm_chunk_offset"])
        grip_q = int(row["gripper_source_query_q"])
        grip_offset = int(row["gripper_chunk_offset"])
        assert arm_q + arm_offset == target_t
        assert grip_q + grip_offset == target_t
        assert int(row["arm_source_age"]) == target_t - arm_q
        assert int(row["gripper_source_age"]) == target_t - grip_q
        assert bool(row["policy_queried_at_t"]) == (target_t in query_steps)
        if method in FIXED_METHODS:
            assert int(row["query_physical_step_q"]) == target_t
            if target_t < DELAY:
                assert arm_q == target_t and arm_offset == 0
                assert grip_q == target_t and grip_offset == 0
                assert row["old_action"] is None
            else:
                assert row["old_action"] is not None
                old = np.asarray(row["old_action"], dtype=np.float64)
                fresh = np.asarray(row["fresh_action"], dtype=np.float64)
                action = np.asarray(row["action"], dtype=np.float64)
                assert old.shape == fresh.shape == action.shape == (7,)
                if method == "FRESH":
                    assert (arm_q, arm_offset) == (target_t, 0)
                    assert (grip_q, grip_offset) == (target_t, 0)
                    expected = fresh
                elif method == "FO20":
                    assert (arm_q, arm_offset) == (target_t, 0)
                    assert (grip_q, grip_offset) == (target_t - DELAY, DELAY)
                    expected = fresh.copy()
                    expected[6] = old[6]
                elif method == "REVERSE20":
                    assert (arm_q, arm_offset) == (target_t - DELAY, DELAY)
                    assert (grip_q, grip_offset) == (target_t, 0)
                    expected = old.copy()
                    expected[6] = fresh[6]
                elif method == "FULL_OLD20":
                    assert (arm_q, arm_offset) == (target_t - DELAY, DELAY)
                    assert (grip_q, grip_offset) == (target_t - DELAY, DELAY)
                    expected = old
                np.testing.assert_array_equal(action, expected)
            if target_t < DELAY:
                np.testing.assert_array_equal(
                    np.asarray(row["action"], dtype=np.float64),
                    np.asarray(row["fresh_action"], dtype=np.float64),
                )
        else:
            q = arm_q
            assert q == grip_q
            assert q % H16 == 0
            assert arm_offset == grip_offset == target_t - q
            assert row["query_physical_step_q"] == (target_t if target_t % H16 == 0 else None)
            assert row["fresh_action"] is None
            assert row["old_action"] is None
        action = np.asarray(row["action"], dtype=np.float64)
        assert action.shape == (7,)
        assert np.isfinite(action).all()


def validate_shard(path: Path, protocol_path: Path, marker: Path | None = None) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads(path.read_text(encoding="utf-8"))
    task_id = int(result["task_id"])
    assert task_id in protocol["cohort"]["primary_task_ids"]
    assert result["methods"] == list(METHODS)
    state_ids = [int(x) for x in protocol["cohort"]["state_ids"]]
    assert result["state_ids"] == state_ids
    assert result["finished"] is True
    assert set(result["episodes"]) == set(METHODS)
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
    assert total == 70
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
