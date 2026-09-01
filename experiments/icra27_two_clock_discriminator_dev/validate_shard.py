"""Validate one completed two-clock task shard before analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from fixed_clock_executor import H16, H32, H32_COHERENT, METHODS, TWO_CLOCK  # noqa: E402
from run_fixed_clocks import environment_seed  # noqa: E402


def validate_episode(episode: dict, method: str, task_id: int, state_id: int) -> None:
    assert int(episode["task_id"]) == task_id
    assert episode["method"] == method
    assert int(episode["requested_initial_state_id"]) == state_id
    assert int(episode["environment_seed"]) == environment_seed(task_id, state_id)
    assert int(episode["environment_construction_seed"]) == environment_seed(task_id, state_id)
    assert episode["fresh_environment_instance"] is True
    assert int(episode["policy_rng_seed"]) == 424242
    assert int(episode["max_episode_steps"]) == 280
    steps = int(episode["environment_steps"])
    assert 1 <= steps <= 280
    rows = episode["step_log"]
    assert len(rows) == steps
    period = H32 if method == H32_COHERENT else H16
    assert episode["query_steps"] == list(range(0, steps, period))
    assert int(episode["policy_queries"]) == len(episode["query_steps"])
    assert float(episode["query_rate"]) == len(episode["query_steps"]) / steps
    assert int(episode["policy_call_count_for_latency"]) == len(episode["query_steps"])
    assert float(episode["mean_policy_call_latency_seconds"]) >= 0
    assert float(episode["wall_clock_seconds"]) > 0

    for t, row in enumerate(rows):
        assert int(row["t"]) == t
        assert bool(row["policy_queried_at_t"]) == (t in episode["query_steps"])
        arm_q = int(row["arm_source_query_q"])
        arm_offset = int(row["arm_chunk_offset"])
        grip_q = int(row["gripper_source_query_q"])
        grip_offset = int(row["gripper_chunk_offset"])
        assert arm_q + arm_offset == t
        assert grip_q + grip_offset == t
        assert int(row["arm_source_age"]) == t - arm_q == arm_offset
        assert int(row["gripper_source_age"]) == t - grip_q == grip_offset
        assert arm_offset < 100 and grip_offset < 100
        action = np.asarray(row["action"], dtype=np.float64)
        assert action.shape == (7,) and np.isfinite(action).all()
        if method == H32_COHERENT:
            q = H32 * (t // H32)
            assert (arm_q, grip_q) == (q, q)
            expected_refresh = ["arm", "gripper"] if t % H32 == 0 else []
        else:
            assert arm_q == H16 * (t // H16)
            assert grip_q == H32 * (t // H32)
            expected_refresh = ["arm", "gripper"] if t % H32 == 0 else (["arm"] if t % H16 == 0 else [])
        assert row["refreshed_groups"] == expected_refresh


def validate_shard(path: Path, protocol_path: Path, marker: Path | None = None) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads(path.read_text(encoding="utf-8"))
    task_id = int(result["task_id"])
    assert task_id in protocol["cohort"]["primary_task_ids"]
    assert result["methods"] == list(METHODS)
    states = [int(value) for value in protocol["cohort"]["state_ids"]]
    assert result["state_ids"] == states
    assert result["finished"] is True
    for method in METHODS:
        episodes = result["episodes"][method]
        assert len(episodes) == len(states)
        assert [int(episode["requested_initial_state_id"]) for episode in episodes] == states
        for episode, state_id in zip(episodes, states, strict=True):
            validate_episode(episode, method, task_id, state_id)
    summary = {"valid": True, "task_id": task_id, "episodes": 2 * len(states), "result": str(path.resolve())}
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
