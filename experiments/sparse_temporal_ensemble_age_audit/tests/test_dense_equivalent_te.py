"""Semantic tests for the ACT-orientation-preserving sparse TE operator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIT_ROOT = REPO_ROOT / "experiments" / "sparse_temporal_ensemble_age_audit"
SPARSE_ROOT = REPO_ROOT / "experiments" / "sparse_temporal_ensemble_dev"
sys.path.insert(0, str(AUDIT_ROOT))
sys.path.insert(0, str(SPARSE_ROOT))

from dense_equivalent_executor import (  # noqa: E402
    DenseEquivalentSparseExecutor,
    dense_equivalent_te_weights,
)
from sparse_executor import SparseExecutor, canonical_temporal_weights  # noqa: E402


def test_h1_matches_validated_canonical_act_weights() -> None:
    """Unit query spacing reproduces canonical ACT oldest-first weights."""

    source_queries = np.arange(6, dtype=np.int64)
    np.testing.assert_allclose(
        dense_equivalent_te_weights(source_queries),
        canonical_temporal_weights(len(source_queries), coefficient=0.01),
        rtol=0,
        atol=1e-15,
    )


def test_h16_two_source_weights_are_oldest_first() -> None:
    actual = dense_equivalent_te_weights(np.asarray([0, 16], dtype=np.int64))
    expected = np.asarray([1.0, np.exp(-0.16)], dtype=np.float64)
    expected /= expected.sum()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
    assert actual[0] > actual[1]


def test_h16_three_source_weights_are_oldest_to_newest() -> None:
    actual = dense_equivalent_te_weights(np.asarray([0, 16, 32], dtype=np.int64))
    expected = np.asarray([1.0, np.exp(-0.16), np.exp(-0.32)], dtype=np.float64)
    expected /= expected.sum()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)


def test_same_target_candidates_satisfy_q_plus_offset_equals_t() -> None:
    executor = DenseEquivalentSparseExecutor(
        cadence=16,
        prediction_horizon=100,
        mode="dense_equivalent_te",
        action_dim=1,
    )
    chunk = np.arange(100, dtype=np.float64)[:, None]
    executor.step(0, lambda: chunk)
    result = executor.step(16, lambda: chunk + 1000.0)

    np.testing.assert_array_equal(
        result.candidates.source_query_steps + result.candidates.offsets,
        np.full(result.candidate_count, 16, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        result.candidates.source_query_steps,
        np.asarray([0, 16], dtype=np.int64),
    )


def test_all_modes_match_before_first_requery() -> None:
    """A single cached chunk makes hard and both TE modes exactly identical."""

    chunk = np.arange(200, dtype=np.float64).reshape(100, 2)
    hard = SparseExecutor(
        cadence=16,
        prediction_horizon=100,
        mode="hard",
        coefficient=0.01,
        action_dim=2,
    )
    candidate_index_te = SparseExecutor(
        cadence=16,
        prediction_horizon=100,
        mode="sparse_te",
        coefficient=0.01,
        action_dim=2,
    )
    dense_equivalent_te = DenseEquivalentSparseExecutor(
        cadence=16,
        prediction_horizon=100,
        mode="dense_equivalent_te",
        action_dim=2,
    )

    for target_step in range(16):
        results = [
            hard.step(target_step, lambda chunk=chunk: chunk),
            candidate_index_te.step(target_step, lambda chunk=chunk: chunk),
            dense_equivalent_te.step(target_step, lambda chunk=chunk: chunk),
        ]
        assert all(result.queried is (target_step == 0) for result in results)
        assert all(result.candidate_count == 1 for result in results)
        for result in results[1:]:
            np.testing.assert_array_equal(result.action, results[0].action)
            np.testing.assert_array_equal(result.weights, results[0].weights)
