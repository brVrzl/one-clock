#!/usr/bin/env python3
"""Validate one completed sparse temporal-ensemble task shard."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
METHODS = ("hard_h8", "sparse_te_h8", "hard_h16", "sparse_te_h16")


def query_seed(task: str, state_id: int, env_seed: int, q: int) -> tuple[str, int]:
    key = f"smolvla|{task}|state={state_id}|env_seed={env_seed}|q={q}"
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)
    return key, seed


def expected_weights(method: str, count: int) -> np.ndarray:
    if method.startswith("hard_"):
        weights = np.zeros(count, dtype=np.float64)
        weights[-1] = 1.0
        return weights
    weights = np.exp(-0.01 * np.arange(count, dtype=np.float64))
    return weights / weights.sum()


def validate_episode(episode: dict, *, method: str, task: str, horizon: int, policy: str) -> None:
    cadence = 8 if method.endswith("h8") else 16
    steps = int(episode["environment_steps"])
    query_steps = [int(value) for value in episode["query_steps"]]
    assert query_steps == list(range(0, steps, cadence)), (method, "query schedule", query_steps, steps)
    assert int(episode["query_count"]) == len(query_steps)
    assert int(episode["policy_queries"]) == len(query_steps)
    rows = episode["step_log"]
    assert len(rows) == steps
    state_id = int(episode.get("requested_initial_state_id", episode.get("initial_state_id")))
    env_seed = int(episode.get("environment_seed", episode.get("env_seed")))
    for t, row in enumerate(rows):
        assert int(row["physical_target_t"]) == t
        assert bool(row["policy_queried_at_t"]) == (t % cadence == 0)
        sources = [int(value) for value in row["ensemble_source_query_ids"]]
        offsets = [int(value) for value in row["candidate_offsets_t_minus_q"]]
        weights = np.asarray(row["normalized_ensemble_weights"], dtype=np.float64)
        assert len(sources) == len(offsets) == len(weights) == int(row["ensemble_candidate_count"])
        assert sources == sorted(sources)
        assert int(row["latest_query_q"]) == sources[-1]
        assert all(q % cadence == 0 and 0 <= t - q < horizon for q in sources)
        assert all(q + offset == t for q, offset in zip(sources, offsets))
        np.testing.assert_allclose(weights, expected_weights(method, len(sources)), rtol=1e-10, atol=1e-12)
        action = np.asarray(row["chosen_executed_action_7d"], dtype=np.float64)
        assert action.shape == (7,) and np.isfinite(action).all()
        if policy == "smolvla" and row["policy_queried_at_t"]:
            key, seed = query_seed(task, state_id, env_seed, t)
            actual_key = row.get(
                "query_seed_key", row.get("policy_rng_key", row.get("query_rng_key"))
            )
            actual_seed = row.get("policy_rng_seed", row.get("query_rng_seed"))
            assert actual_key == key, (method, t, actual_key, key)
            assert int(actual_seed) == seed, (method, t, actual_seed, seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--policy", choices=("act", "smolvla"), required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--marker", type=Path)
    args = parser.parse_args()

    data = json.loads(args.result.read_text())
    assert "finished_at" in data
    task = data["task"]
    horizon = 100 if args.policy == "act" else 50
    assert int(data["prediction_horizon"]) == horizon
    assert list(data["methods_result"]) == list(METHODS)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    expected_pairs = list(
        zip(
            protocol["environment_pairing"]["initial_state_ids"][: args.episodes],
            protocol["environment_pairing"]["seeds"][: args.episodes],
        )
    )
    by_method: dict[str, dict[tuple[int, int], dict]] = {}
    for method in METHODS:
        episodes = data["methods_result"][method]["episodes_detail"]
        assert len(episodes) == args.episodes
        pairs = [
            (
                int(ep.get("requested_initial_state_id", ep.get("initial_state_id"))),
                int(ep.get("environment_seed", ep.get("env_seed"))),
            )
            for ep in episodes
        ]
        assert pairs == expected_pairs, (method, pairs, expected_pairs)
        by_method[method] = dict(zip(pairs, episodes))
        for episode in episodes:
            validate_episode(episode, method=method, task=task, horizon=horizon, policy=args.policy)

    for cadence in (8, 16):
        hard = by_method[f"hard_h{cadence}"]
        ensemble = by_method[f"sparse_te_h{cadence}"]
        for pair in expected_pairs:
            hard_episode = hard[pair]
            ensemble_episode = ensemble[pair]
            common_queries = min(len(hard_episode["query_steps"]), len(ensemble_episode["query_steps"]))
            assert hard_episode["query_steps"][:common_queries] == ensemble_episode["query_steps"][:common_queries]
            # Action equality before the first re-query is established by the
            # CPU test with an identical chunk and by the one-state live smoke.
            # It is not a valid full-shard assertion across separate simulator
            # resets when those resets can produce different observations.

    if args.marker is not None:
        args.marker.parent.mkdir(parents=True, exist_ok=True)
        args.marker.write_text(json.dumps({"valid": True, "result": str(args.result.resolve())}) + "\n")
    print(json.dumps({"valid": True, "policy": args.policy, "task": task, "episodes_per_method": args.episodes}))


if __name__ == "__main__":
    main()
