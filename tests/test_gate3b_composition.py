from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "audit_tools"))

from gate3b_composition import (  # noqa: E402
    METHODS,
    SOURCE_AGE_TICKS,
    TEMPORAL_ENSEMBLE_ACTIVE,
    FixedAgeComposer,
    compose_action,
)
from gate3b_schedule import build_schedule, pending_runs  # noqa: E402


def tagged_chunk(source: int) -> np.ndarray:
    chunk = np.empty((100, 7), dtype=np.float64)
    for offset in range(100):
        chunk[offset] = 10_000 * source + 100 * offset + np.arange(7)
    return chunk


def result_at(method: str, target_step: int):
    composer = FixedAgeComposer(method)
    result = None
    for step in range(target_step + 1):
        result = composer.update(step, tagged_chunk(step))
    assert result is not None
    return result


def test_old_source_q_t_minus_20_maps_to_chunk_offset_20() -> None:
    result = result_at("OO", 23)
    np.testing.assert_array_equal(result.old_action, tagged_chunk(3)[20])
    np.testing.assert_array_equal(result.action, tagged_chunk(3)[20])
    assert result.old_source_step == 3
    assert result.old_chunk_offset == SOURCE_AGE_TICKS


@pytest.mark.parametrize("method", METHODS)
def test_before_tick_20_every_method_executes_fresh(method: str) -> None:
    composer = FixedAgeComposer(method)
    for step in range(SOURCE_AGE_TICKS):
        result = composer.update(step, tagged_chunk(step))
        np.testing.assert_array_equal(result.action, tagged_chunk(step)[0])
        assert result.old_action is None
        assert result.arm_source_step == step
        assert result.gripper_source_step == step


def test_four_registered_formulas_are_exact() -> None:
    fresh = np.arange(7, dtype=np.float64) + 0.25
    old = np.arange(7, dtype=np.float64) + 10.75
    np.testing.assert_array_equal(compose_action("FF", fresh, old), fresh)
    np.testing.assert_array_equal(compose_action("OO", fresh, old), old)
    np.testing.assert_array_equal(compose_action("FO", fresh, old)[:6], fresh[:6])
    assert compose_action("FO", fresh, old)[6] == old[6]
    np.testing.assert_array_equal(compose_action("OF", fresh, old)[:6], old[:6])
    assert compose_action("OF", fresh, old)[6] == fresh[6]


def test_source_identity_has_no_order_reversal() -> None:
    expected = {
        "FF": (20, 20),
        "OO": (0, 0),
        "FO": (20, 0),
        "OF": (0, 20),
    }
    for method, sources in expected.items():
        result = result_at(method, 20)
        assert (result.arm_source_step, result.gripper_source_step) == sources
        assert result.fresh_source_step == 20
        assert result.old_source_step == 0


def test_no_temporal_ensemble_is_active() -> None:
    assert TEMPORAL_ENSEMBLE_ACTIVE is False
    fresh = np.zeros(7)
    old = np.ones(7)
    for method in METHODS:
        action = compose_action(method, fresh, old)
        assert np.all((action == fresh) | (action == old))


def test_all_four_first_20_actions_are_identical_for_identical_chunks() -> None:
    composers = {method: FixedAgeComposer(method) for method in METHODS}
    for step in range(SOURCE_AGE_TICKS):
        actions = [composer.update(step, tagged_chunk(step)).action for composer in composers.values()]
        for action in actions[1:]:
            np.testing.assert_array_equal(action, actions[0])


def test_schedule_and_resume_order_are_deterministic() -> None:
    first = build_schedule()
    second = build_schedule()
    assert first == second
    assert first["state_selection"]["selected_state_ids"] == [24, 26, 28, 29, 32, 33, 37, 40, 46, 49]
    assert len(first["blocks"]) == 100
    assert len(first["runs"]) == 400
    cells = {(run["task_id"], run["state_id"], run["method"]) for run in first["runs"]}
    assert len(cells) == 400
    completed = {0, 2, 7, 19}
    resumed_once = pending_runs(first, completed, max_new_runs=8)
    resumed_twice = pending_runs(second, completed, max_new_runs=8)
    assert resumed_once == resumed_twice
    assert [run["run_index"] for run in resumed_once] == [1, 3, 4, 5, 6, 8, 9, 10]


def test_rejects_skipped_query_and_wrong_shape() -> None:
    composer = FixedAgeComposer("FF")
    composer.update(0, tagged_chunk(0))
    with pytest.raises(ValueError, match="exactly one ordered"):
        composer.update(2, tagged_chunk(2))
    with pytest.raises(ValueError, match="chunk shape"):
        FixedAgeComposer("FF").update(0, np.zeros((99, 7)))
