from __future__ import annotations

import numpy as np
import pytest

from research.audit_tools.gate3c_schedule import build_schedule, pending_runs
from research.audit_tools.gate3c_temporal_reuse import (
    METHODS,
    Gate3CTemporalExecutor,
    compose_fixed_action,
)


def chunks(count: int = 25) -> list[np.ndarray]:
    result = []
    for source in range(count):
        chunk = np.zeros((100, 7), dtype=np.float64)
        for offset in range(100):
            chunk[offset] = 1000 * source + 10 * offset + np.arange(7)
        result.append(chunk)
    return result


def test_fixed_formulas_and_no_source_order_reversal() -> None:
    generated = chunks()
    expected_fresh = generated[20][0]
    expected_old = generated[0][20]
    for method in METHODS[:3]:
        executor = Gate3CTemporalExecutor(method)
        observed = None
        for step in range(21):
            observed = executor.update(step, generated[step])
        assert observed is not None
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


@pytest.mark.parametrize("method", METHODS[:3])
def test_first_twenty_fixed_source_actions_are_full_fresh(method: str) -> None:
    executor = Gate3CTemporalExecutor(method)
    for step, chunk in enumerate(chunks(20)):
        observed = executor.update(step, chunk)
        np.testing.assert_array_equal(observed.action, chunk[0].astype(np.float32))
        assert observed.old_action is None


def test_scalar_baselines_use_one_full_action_weight_vector() -> None:
    generated = chunks(4)
    for method in METHODS[3:]:
        executor = Gate3CTemporalExecutor(method)
        observed = None
        for step in range(4):
            observed = executor.update(step, generated[step])
        assert observed is not None
        assert np.isfinite(observed.weights).all()
        assert observed.weights.sum() == pytest.approx(1.0)
        assert observed.arm_effective_age_ticks == pytest.approx(
            observed.gripper_effective_age_ticks
        )


def test_no_policy_temporal_ensemble_or_smoothing_flags() -> None:
    from research.audit_tools import gate3c_temporal_reuse as module

    assert module.POLICY_TEMPORAL_ENSEMBLE_ACTIVE is False
    assert module.ACTION_SMOOTHING_ACTIVE is False


def test_schedule_and_resume_are_deterministic() -> None:
    first = build_schedule()
    second = build_schedule()
    assert first == second
    assert first["planned_episodes"] == 700
    assert len({(r["task_id"], r["state_id"], r["method"]) for r in first["runs"]}) == 700
    assert all(
        r["episode_seed"] == 330000 + 100 * r["task_id"] + r["state_id"]
        for r in first["runs"]
    )
    completed = {0, 3, 17, 699}
    pending = pending_runs(first, completed)
    assert [r["run_index"] for r in pending] == [
        index for index in range(700) if index not in completed
    ]
