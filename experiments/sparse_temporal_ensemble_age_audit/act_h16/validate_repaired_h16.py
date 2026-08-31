#!/usr/bin/env python3
"""Validate a completed repaired ACT h16 task shard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ACT_ROOT = Path(__file__).resolve().parent
AUDIT_ROOT = ACT_ROOT.parent
sys.path.insert(0, str(AUDIT_ROOT))

from dense_equivalent_executor import dense_equivalent_te_weights  # noqa: E402


METHODS = ("hard_h16", "candidate_index_te_h16", "dense_equivalent_te_h16")


def expected_weights(method: str, sources: list[int]) -> np.ndarray:
    count = len(sources)
    if method == "hard_h16":
        result = np.zeros(count, dtype=np.float64)
        result[-1] = 1.0
        return result
    if method == "candidate_index_te_h16":
        result = np.exp(-0.01 * np.arange(count, dtype=np.float64))
        return result / result.sum()
    if method == "dense_equivalent_te_h16":
        return dense_equivalent_te_weights(np.asarray(sources, dtype=np.int64))
    raise ValueError(method)


def validate_episode(episode: dict, method: str) -> None:
    steps = int(episode["environment_steps"])
    assert episode["fresh_environment_instance"] is True
    assert int(episode["environment_construction_seed"]) == int(episode["environment_seed"])
    assert int(episode["requested_initial_state_id"]) == int(episode["selected_initial_state_id_before_reset"])
    query_steps = [int(value) for value in episode["query_steps"]]
    assert query_steps == list(range(0, steps, 16))
    assert int(episode["query_count"]) == len(query_steps)
    rows = episode["step_log"]
    assert len(rows) == steps
    for t, row in enumerate(rows):
        assert int(row["physical_target_t"]) == t
        assert bool(row["policy_queried_at_t"]) == (t % 16 == 0)
        sources = [int(value) for value in row["ensemble_source_query_ids"]]
        offsets = [int(value) for value in row["candidate_offsets_t_minus_q"]]
        weights = np.asarray(row["normalized_ensemble_weights"], dtype=np.float64)
        assert sources == sorted(sources)
        assert len(sources) == len(offsets) == len(weights) == int(row["ensemble_candidate_count"])
        assert all(q % 16 == 0 and 0 <= t - q < 100 for q in sources)
        assert all(q + offset == t for q, offset in zip(sources, offsets))
        np.testing.assert_allclose(weights, expected_weights(method, sources), rtol=1e-12, atol=1e-12)
        action = np.asarray(row["chosen_executed_action_7d"], dtype=np.float64)
        assert action.shape == (7,) and np.isfinite(action).all()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--marker", type=Path)
    args = parser.parse_args()

    data = json.loads(args.result.read_text())
    assert "finished_at" in data
    assert data["fresh_environment_per_condition_state"] is True
    assert list(data["methods_result"]) == list(METHODS)
    protocol = json.loads((AUDIT_ROOT / "protocol.json").read_text())
    panel = protocol["repaired_h16_panel"]
    expected_pairs = list(
        zip(panel["initial_state_ids"][: args.episodes], panel["environment_seeds"][: args.episodes])
    )
    by_method = {}
    for method in METHODS:
        episodes = data["methods_result"][method]["episodes_detail"]
        assert len(episodes) == args.episodes
        pairs = [
            (int(episode["requested_initial_state_id"]), int(episode["environment_seed"]))
            for episode in episodes
        ]
        assert pairs == expected_pairs
        by_method[method] = dict(zip(pairs, episodes))
        for episode in episodes:
            validate_episode(episode, method)

    reference = by_method["hard_h16"]
    for pair in expected_pairs:
        hard = reference[pair]
        for method in METHODS[1:]:
            candidate = by_method[method][pair]
            np.testing.assert_array_equal(hard["initial_sim_state"], candidate["initial_sim_state"])
            np.testing.assert_array_equal(hard["initial_model_body_pos"], candidate["initial_model_body_pos"])
            assert hard["initial_low_dimensional_observation"] == candidate["initial_low_dimensional_observation"]
            hard_prefix = np.asarray(
                [row["chosen_executed_action_7d"] for row in hard["step_log"][:16]], dtype=np.float32
            )
            candidate_prefix = np.asarray(
                [row["chosen_executed_action_7d"] for row in candidate["step_log"][:16]], dtype=np.float32
            )
            assert hard_prefix.shape == candidate_prefix.shape == (16, 7)
            np.testing.assert_array_equal(hard_prefix, candidate_prefix)

    if args.marker is not None:
        args.marker.parent.mkdir(parents=True, exist_ok=True)
        args.marker.write_text(json.dumps({"valid": True, "result": str(args.result.resolve())}) + "\n")
    print(json.dumps({"valid": True, "task": data["task"], "episodes_per_method": args.episodes}))


if __name__ == "__main__":
    main()
