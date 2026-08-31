"""Executable audit of the h1 orientation conflict in the requested kernel."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SPARSE_ROOT = REPO_ROOT / "experiments" / "sparse_temporal_ensemble_dev"
sys.path.insert(0, str(SPARSE_ROOT))

from sparse_executor import canonical_temporal_weights  # noqa: E402


def requested_newest_relative_weights(source_queries: np.ndarray, coefficient: float = 0.01) -> np.ndarray:
    sources = np.asarray(source_queries, dtype=np.float64)
    weights = np.exp(-float(coefficient) * (sources[-1] - sources))
    return weights / weights.sum()


def test_validated_lerobot_h1_favors_the_oldest_candidate() -> None:
    actual = canonical_temporal_weights(2, coefficient=0.01)
    expected = np.asarray([1.0, np.exp(-0.01)], dtype=np.float64)
    expected /= expected.sum()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
    assert actual[0] > actual[1]


def test_requested_newest_relative_formula_reverses_h1_orientation() -> None:
    canonical = canonical_temporal_weights(2, coefficient=0.01)
    requested = requested_newest_relative_weights(np.asarray([0, 1]))
    np.testing.assert_allclose(requested, canonical[::-1], rtol=0, atol=1e-15)
    assert not np.allclose(requested, canonical, rtol=0, atol=1e-15)


def test_oldest_relative_physical_separation_is_the_h1_equivalent_subsampling() -> None:
    sources = np.asarray([0, 1, 2], dtype=np.float64)
    weights = np.exp(-0.01 * (sources - sources[0]))
    weights /= weights.sum()
    np.testing.assert_allclose(weights, canonical_temporal_weights(3), rtol=0, atol=1e-15)
