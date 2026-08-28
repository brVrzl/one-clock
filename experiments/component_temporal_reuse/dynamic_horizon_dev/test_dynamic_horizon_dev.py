"""CPU tests for sparse-query indexing and adaptive trigger semantics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_dynamic_horizon_dev as runner  # noqa: E402


def test_sparse_same_target_uses_chunk_offset_from_query_time() -> None:
    records = [runner.QueryRecord(0, np.arange(70, dtype=float).reshape(10, 7)), runner.QueryRecord(4, np.ones((10, 7)))]
    candidates = runner._candidate_records(records, 5)
    assert [(age, query) for age, _, query in candidates] == [(5, 0), (1, 4)]
    np.testing.assert_array_equal(candidates[-1][1], np.ones(7))


def test_adaptive_triggers_are_prequery_and_independent() -> None:
    old = np.tile(np.asarray([1, 0, 0, 0, 0, 0, 1.0]), (20, 1))
    new = np.tile(np.asarray([-1, 0, 0, 0, 0, 0, -1.0]), (20, 1))
    records = [runner.QueryRecord(0, old), runner.QueryRecord(1, new)]
    query, reasons, diagnostics = runner.adaptive_query_decision(records, 2)
    assert query
    assert set(("gripper_sign_disagreement", "arm_cosine_lt_0.90")) <= set(reasons)
    assert diagnostics["candidate_count"] == 2


def test_fixed_horizon_queries_only_on_period_boundary() -> None:
    records = []
    chunks = [np.tile(np.arange(7, dtype=float), (20, 1))]
    action, queried, _, _, age, _ = runner.scheduler_step("fixed_h4", 0, records, lambda: chunks[0])
    assert queried and age == 0
    action, queried, _, _, age, _ = runner.scheduler_step("fixed_h4", 1, records, lambda: chunks[0])
    assert not queried and age == 1
    np.testing.assert_array_equal(action, chunks[0][1])
