"""Focused semantic tests for bounded group-triggered joint replanning."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from requery_policy import (  # noqa: E402
    MAX_HORIZON,
    action_from_newest_chunk,
    arm_phase_horizon,
    choose_horizon,
    gripper_event_horizon,
)


def _arm_speed_chunk(speeds: list[float]) -> np.ndarray:
    chunk = np.zeros((100, 7), dtype=np.float64)
    cumulative = np.cumsum(np.asarray(speeds, dtype=np.float64))
    chunk[1 : len(speeds) + 1, 0] = cumulative
    chunk[1 : len(speeds) + 1, 3] = cumulative
    return chunk


def test_hard16_is_always_sixteen() -> None:
    chunk = np.zeros((100, 7), dtype=np.float64)
    assert choose_horizon("M0_hard16", chunk)[0] == 16


def test_stationary_arm_profile_is_finite_and_bounded() -> None:
    chunk = np.zeros((100, 7), dtype=np.float64)
    horizon, diagnostics = arm_phase_horizon(chunk)
    assert horizon == 4
    assert np.isfinite(diagnostics["normalized_arm_speed"]).all()
    assert 4 <= horizon <= 16


def test_arm_selects_earliest_qualifying_local_minimum() -> None:
    speeds = [1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    horizon, diagnostics = arm_phase_horizon(_arm_speed_chunk(speeds))
    assert horizon == 4
    assert diagnostics["arm_boundary_candidates"][0] == 4


def test_arm_falls_back_when_no_low_speed_boundary_exists() -> None:
    speeds = [1.0] * 15
    horizon, diagnostics = arm_phase_horizon(_arm_speed_chunk(speeds))
    assert horizon == MAX_HORIZON
    assert diagnostics["arm_boundary_candidates"] == []


def test_gripper_uses_earliest_transition_and_existing_zero_contract() -> None:
    chunk = np.zeros((100, 7), dtype=np.float64)
    chunk[7:, 6] = -1.0
    horizon, diagnostics = gripper_event_horizon(chunk)
    assert horizon == 7
    assert diagnostics["gripper_event_candidates"] == [7]
    zero_to_open = np.zeros((100, 7), dtype=np.float64)
    zero_to_open[8:, 6] = 1.0
    assert gripper_event_horizon(zero_to_open)[0] == 16


def test_gripper_falls_back_without_transition() -> None:
    chunk = np.ones((100, 7), dtype=np.float64)
    assert choose_horizon("M2_gripper_event", chunk)[0] == 16


def test_combined_is_minimum_joint_horizon() -> None:
    chunk = _arm_speed_chunk([1.0, 1.0, 1.0, 0.1] + [1.0] * 11)
    chunk[7:, 6] = -1.0
    horizon, diagnostics = choose_horizon("M3_group_event_joint", chunk)
    assert horizon == min(16, diagnostics["h_arm"], diagnostics["h_grip"])
    assert 4 <= horizon <= 16


def test_next_query_and_action_use_only_newest_chunk_offset() -> None:
    old_chunk = np.zeros((100, 7), dtype=np.float64)
    new_chunk = np.arange(100 * 7, dtype=np.float64).reshape(100, 7)
    q = 12
    h = choose_horizon("M0_hard16", new_chunk)[0]
    assert q + h == 28
    action, offset = action_from_newest_chunk(new_chunk, 16, 20)
    assert offset == 4
    np.testing.assert_array_equal(action, new_chunk[4])
    assert not np.array_equal(action, old_chunk[20])


def test_all_methods_keep_horizon_inclusive_bounds() -> None:
    rng = np.random.default_rng(4)
    for _ in range(20):
        chunk = rng.normal(size=(100, 7))
        for method in ("M0_hard16", "M1_arm_phase", "M2_gripper_event", "M3_group_event_joint"):
            horizon, _ = choose_horizon(method, chunk)
            assert 4 <= horizon <= 16
