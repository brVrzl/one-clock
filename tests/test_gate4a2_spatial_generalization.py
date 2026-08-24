from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/audit_tools"))

from research.audit_tools.gate3a2_temporal_aggregation import temporal_weights
from research.audit_tools.gate3c_temporal_reuse import (
    METHODS,
    Gate3CTemporalExecutor,
    compose_fixed_action,
)
from research.audit_tools.gate4a2_rollout import assert_query_fairness
from research.audit_tools.gate4a2_schedule import (
    COMMON_VALID_STATES,
    SELECTED_STATES,
    STATE_SELECTION_RNG_RESULT,
    build_schedule,
    pending_runs,
)


def chunks(count: int = 25) -> list[np.ndarray]:
    result = []
    for source in range(count):
        chunk = np.zeros((100, 7), dtype=np.float64)
        for offset in range(100):
            chunk[offset] = 1000 * source + 10 * offset + np.arange(7)
        result.append(chunk)
    return result


def test_gate4a2_reuses_gate3c_executor_source_directly() -> None:
    source = inspect.getsourcefile(Gate3CTemporalExecutor)
    assert source is not None and source.endswith("research/audit_tools/gate3c_temporal_reuse.py")


def test_fixed_formulas_source_indexing_and_fresh_prefix() -> None:
    generated = chunks()
    expected_fresh = generated[20][0]
    expected_old = generated[0][20]
    for method in METHODS[:3]:
        executor = Gate3CTemporalExecutor(method)
        for step in range(20):
            observed = executor.update(step, generated[step])
            np.testing.assert_array_equal(observed.action, generated[step][0].astype(np.float32))
            assert observed.old_action is None
        observed = executor.update(20, generated[20])
        assert observed.old_source_step == 0
        assert observed.old_chunk_offset == 20
        np.testing.assert_array_equal(observed.fresh_action, expected_fresh)
        np.testing.assert_array_equal(observed.old_action, expected_old)
    np.testing.assert_array_equal(
        compose_fixed_action("A_NEWEST", expected_fresh, expected_old), expected_fresh
    )
    np.testing.assert_array_equal(
        compose_fixed_action("B_FULL_OLD20", expected_fresh, expected_old), expected_old
    )
    mixed = compose_fixed_action("C_ASYMMETRIC_FO20", expected_fresh, expected_old)
    np.testing.assert_array_equal(mixed[:6], expected_fresh[:6])
    assert mixed[6] == expected_old[6]


def test_frozen_scalar_baselines_use_shared_full_action_weights() -> None:
    candidates = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.2, 0.3, 0.4, -1.0],
            [0.8, 0.2, 0.0, 0.1, 0.4, 0.3, 1.0],
            [0.9, 0.1, 0.1, 0.3, 0.2, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    ages = np.asarray([2.0, 1.0, 0.0])
    age_weights = temporal_weights("newest_age_exp_b003", candidates, ages)
    expected_age = np.exp(-0.03 * ages)
    expected_age /= expected_age.sum()
    np.testing.assert_allclose(age_weights, expected_age, rtol=0, atol=1e-15)
    cog_weights = temporal_weights("cogact_a03", candidates, ages)
    newest = candidates[-1]
    cosine = (candidates @ newest) / (
        np.linalg.norm(candidates, axis=1) * np.linalg.norm(newest) + 1e-7
    )
    expected_cog = np.exp(0.3 * cosine - np.max(0.3 * cosine))
    expected_cog /= expected_cog.sum()
    np.testing.assert_allclose(cog_weights, expected_cog, rtol=0, atol=1e-15)
    for weights in (age_weights, cog_weights):
        action = weights @ candidates
        assert action.shape == (7,)
        assert np.isfinite(action).all()


def test_state_selection_schedule_and_resume_are_frozen() -> None:
    rng = np.random.default_rng(20260825)
    result = tuple(rng.choice(COMMON_VALID_STATES, size=10, replace=False).tolist())
    assert result == STATE_SELECTION_RNG_RESULT
    assert tuple(sorted(result)) == SELECTED_STATES
    first = build_schedule()
    second = build_schedule()
    assert first == second
    assert first["planned_episodes"] == 500
    assert first["planned_blocks"] == 100
    assert len({(r["task_id"], r["state_id"], r["method"]) for r in first["runs"]}) == 500
    assert all(
        r["episode_seed"] == 340000 + 100 * r["task_id"] + r["state_id"]
        for r in first["runs"]
    )
    assert all(len(set(block["method_order"])) == 5 for block in first["blocks"])
    completed = {0, 3, 17, 499}
    pending = pending_runs(first, completed)
    assert [r["run_index"] for r in pending] == [
        index for index in range(500) if index not in completed
    ]


def test_one_query_per_step_bookkeeping() -> None:
    assert_query_fairness(280, 280)
    with pytest.raises(RuntimeError, match="policy_queries == environment_steps"):
        assert_query_fairness(279, 280)
