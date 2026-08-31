"""CPU semantic tests for the sparse temporal-ensemble executor."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sparse_executor import SparseExecutor, canonical_temporal_weights  # noqa: E402


def chunk(source: int, horizon: int = 100) -> np.ndarray:
    # Distinct values make source IDs, offsets, and target alignment auditable.
    return np.asarray(
        [[1000.0 * source + 10.0 * offset + d for d in range(7)] for offset in range(horizon)],
        dtype=np.float64,
    )


def test_query_schedule_has_no_extra_calls() -> None:
    for cadence, expected in ((8, [0, 8, 16, 24, 32]), (16, [0, 16, 32, 48])):
        executor = SparseExecutor(cadence=cadence, prediction_horizon=100, mode="hard")
        calls: list[int] = []
        for target in range(max(expected) + 1):
            executor.step(target, lambda target=target: calls.append(target) or chunk(target))
        assert executor.query_steps == expected
        assert calls == expected


def test_first_segment_hard_and_sparse_te_are_action_identical() -> None:
    hard = SparseExecutor(cadence=8, prediction_horizon=100, mode="hard")
    te = SparseExecutor(cadence=8, prediction_horizon=100, mode="sparse_te")
    for target in range(8):
        hard_result = hard.step(target, lambda: chunk(0))
        te_result = te.step(target, lambda: chunk(0))
        np.testing.assert_allclose(hard_result.action, te_result.action, rtol=0, atol=1e-12)
        assert hard_result.candidate_count == te_result.candidate_count == 1


def test_first_overlap_uses_oldest_to_newest_same_target_actions() -> None:
    executor = SparseExecutor(cadence=8, prediction_horizon=100, mode="sparse_te")
    for target in range(9):
        result = executor.step(target, lambda: chunk(target))
    assert result.target_step == 8
    assert result.candidates.source_query_steps.tolist() == [0, 8]
    assert result.candidates.offsets.tolist() == [8, 0]
    expected_weights = canonical_temporal_weights(2, coefficient=0.01)
    expected = expected_weights[0] * chunk(0)[8] + expected_weights[1] * chunk(8)[0]
    np.testing.assert_allclose(result.weights, expected_weights, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.action, expected, rtol=1e-12, atol=1e-12)


def test_later_overlap_aligns_all_candidates_to_one_physical_target() -> None:
    executor = SparseExecutor(cadence=8, prediction_horizon=100, mode="sparse_te")
    for target in range(17):
        result = executor.step(target, lambda: chunk(target))
    assert result.candidates.source_query_steps.tolist() == [0, 8, 16]
    assert result.candidates.offsets.tolist() == [16, 8, 0]
    # Every source encodes source*1000 + offset*10.  At target 16, all rows
    # therefore have source*1000 + (target-source)*10 = target*10 + 800*source,
    # and the source/offset pair itself is the auditable alignment check.
    for source, offset, action in zip(
        result.candidates.source_query_steps, result.candidates.offsets, result.candidates.actions
    ):
        np.testing.assert_allclose(action[0], 1000.0 * source + 10.0 * offset)
        assert int(source) + int(offset) == result.target_step


def test_horizon_expiration_is_exact() -> None:
    for cadence, expected_max in ((8, 13), (16, 7)):
        executor = SparseExecutor(cadence=cadence, prediction_horizon=100, mode="hard")
        counts = {}
        for target in range(0, 130):
            result = executor.step(target, lambda target=target: chunk(target))
            counts[target] = result.candidate_count
        assert max(counts.values()) == expected_max
        # q=0 remains valid for target 99 and expires at target 100.
        if cadence == 8:
            assert 0 in executor.same_target_candidates(99).source_query_steps
            assert 0 not in executor.same_target_candidates(100).source_query_steps
        else:
            assert 0 in executor.same_target_candidates(99).source_query_steps
            assert 0 not in executor.same_target_candidates(100).source_query_steps


def test_hard_and_te_have_identical_query_schedule() -> None:
    hard = SparseExecutor(cadence=16, prediction_horizon=100, mode="hard")
    te = SparseExecutor(cadence=16, prediction_horizon=100, mode="sparse_te")
    for target in range(100):
        hard.step(target, lambda: chunk(target))
        te.step(target, lambda: chunk(target))
    assert hard.query_steps == te.query_steps == list(range(0, 100, 16))


def test_smaller_prediction_horizon_changes_only_overlap_expiration() -> None:
    for cadence, expected_max in ((8, 7), (16, 4)):
        executor = SparseExecutor(cadence=cadence, prediction_horizon=50, mode="sparse_te")
        counts = []
        for target in range(0, 80):
            counts.append(executor.step(target, lambda target=target: chunk(target, 50)).candidate_count)
        assert max(counts) == expected_max

