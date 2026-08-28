"""CPU tests for sparse-query indexing and fixed-period execution."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_fixed_horizon_blind as runner  # noqa: E402


def test_sparse_same_target_uses_chunk_offset_from_query_time() -> None:
    records = [runner.QueryRecord(0, np.arange(70, dtype=float).reshape(10, 7)), runner.QueryRecord(4, np.ones((10, 7)))]
    candidates = runner._candidate_records(records, 5)
    assert [(age, query) for age, _, query in candidates] == [(5, 0), (1, 4)]
    np.testing.assert_array_equal(candidates[-1][1], np.ones(7))


def test_fixed_period_queries_only_on_period_boundary() -> None:
    records = []
    chunks = [np.tile(np.arange(7, dtype=float), (120, 1))]
    action, queried, _, _, age, _ = runner.scheduler_step("fixed_h16", 0, records, lambda: chunks[0])
    assert queried and age == 0
    action, queried, _, _, age, _ = runner.scheduler_step("fixed_h16", 1, records, lambda: chunks[0])
    assert not queried and age == 1
    np.testing.assert_array_equal(action, chunks[0][1])


def test_all_four_horizons_are_exact() -> None:
    chunks = [np.tile(np.arange(7, dtype=float), (120, 1))]
    for method, horizon in runner.HORIZONS.items():
        records = []
        runner.scheduler_step(method, 0, records, lambda: chunks[0])
        _, queried, _, _, _, _ = runner.scheduler_step(method, horizon - 1, records, lambda: chunks[0])
        if horizon == 1:
            assert queried
        else:
            assert not queried
        _, queried, _, _, _, _ = runner.scheduler_step(method, horizon, records, lambda: chunks[0])
        assert queried
