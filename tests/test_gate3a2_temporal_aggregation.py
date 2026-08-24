from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "audit_tools"))

from gate3a2_temporal_aggregation import (  # noqa: E402
    DenseTemporalAggregator,
    temporal_weights,
)


def tagged_chunk(source: int, *, chunk_length: int = 4, action_dim: int = 2) -> np.ndarray:
    chunk = np.empty((chunk_length, action_dim), dtype=np.float64)
    for offset in range(chunk_length):
        chunk[offset] = (100 * source + offset, -(100 * source + offset))
    return chunk


def test_candidate_source_order_age_and_exhaustion() -> None:
    aggregator = DenseTemporalAggregator("newest", chunk_length=4, action_dim=2)
    results = [aggregator.update(step, tagged_chunk(step)) for step in range(6)]

    assert results[0].candidate_ages.tolist() == [0]
    assert results[3].candidate_ages.tolist() == [3, 2, 1, 0]
    assert results[5].candidate_ages.tolist() == [3, 2, 1, 0]
    np.testing.assert_array_equal(results[5].action, tagged_chunk(5)[0])


def test_exact_act_uses_oldest_to_newest_source_order() -> None:
    candidates = np.asarray([[1.0], [4.0], [9.0]])
    ages = np.asarray([2, 1, 0])
    weights = temporal_weights("exact_act_m001", candidates, ages)
    expected = np.exp(-0.01 * np.arange(3))
    expected /= expected.sum()
    np.testing.assert_allclose(weights, expected, rtol=0, atol=1e-15)
    assert weights[0] > weights[-1]


def test_exact_act_matches_pinned_lerobot_online_implementation() -> None:
    pinned_lerobot = Path("/home/thor/projects/embodied_lab/third_party/lerobot/src")
    sys.path.insert(0, str(pinned_lerobot))
    import torch
    from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

    rng = np.random.default_rng(17)
    chunks = rng.normal(size=(7, 4, 2)).astype(np.float32)
    ours = DenseTemporalAggregator("exact_act_m001", chunk_length=4, action_dim=2)
    upstream = ACTTemporalEnsembler(temporal_ensemble_coeff=0.01, chunk_size=4)

    for step, chunk in enumerate(chunks):
        ours_action = ours.update(step, chunk).action
        upstream_action = upstream.update(torch.from_numpy(chunk[None]))[0].numpy()
        np.testing.assert_allclose(ours_action, upstream_action, rtol=2e-6, atol=2e-7)


def test_newest_age_exponential_favors_newest_at_registered_tick_rate() -> None:
    candidates = np.asarray([[1.0], [4.0], [9.0]])
    ages = np.asarray([2, 1, 0])
    weights = temporal_weights("newest_age_exp_b003", candidates, ages)
    expected = np.exp(-0.03 * ages)
    expected /= expected.sum()
    np.testing.assert_allclose(weights, expected, rtol=0, atol=1e-15)
    assert weights[-1] > weights[0]


def test_cogact_matches_released_full_action_cosine_rule() -> None:
    candidates = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    ages = np.asarray([2, 1, 0])
    weights = temporal_weights("cogact_a03", candidates, ages)
    newest = candidates[-1]
    cosine = (candidates @ newest) / (
        np.linalg.norm(candidates, axis=1) * np.linalg.norm(newest) + 1e-7
    )
    expected = np.exp(0.3 * cosine)
    expected /= expected.sum()
    np.testing.assert_allclose(weights, expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    "method",
    ["newest", "exact_act_m001", "cogact_a03", "newest_age_exp_b003"],
)
def test_all_methods_are_identical_with_one_candidate(method: str) -> None:
    aggregator = DenseTemporalAggregator(method, chunk_length=4, action_dim=2)
    result = aggregator.update(0, tagged_chunk(0))
    np.testing.assert_array_equal(result.action, tagged_chunk(0)[0])
    np.testing.assert_array_equal(result.weights, np.ones(1))


def test_rejects_skipped_query_and_bad_chunk() -> None:
    aggregator = DenseTemporalAggregator("newest", chunk_length=4, action_dim=2)
    aggregator.update(0, tagged_chunk(0))
    with pytest.raises(ValueError, match="exactly one ordered"):
        aggregator.update(2, tagged_chunk(2))

    fresh = DenseTemporalAggregator("newest", chunk_length=4, action_dim=2)
    with pytest.raises(ValueError, match="chunk shape"):
        fresh.update(0, np.zeros((3, 2)))
