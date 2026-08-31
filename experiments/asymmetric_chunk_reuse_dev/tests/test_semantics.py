from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from asymmetric_executor import (  # noqa: E402
    C1,
    C2,
    CHUNK_LENGTH,
    H16,
    METHODS,
    H16ArmFreshGripExecutor,
    PreviousChunkGripExecutor,
    make_executor,
)


def tagged_chunk(source: int) -> np.ndarray:
    return np.asarray(
        [[10000 * source + 100 * offset + dimension for dimension in range(7)] for offset in range(CHUNK_LENGTH)],
        dtype=np.float64,
    )


def test_both_conditions_preserve_same_target_q_plus_k_equals_t() -> None:
    for method in METHODS:
        executor = make_executor(method)
        for target_t in range(65):
            result = executor.step(target_t, lambda target_t=target_t: tagged_chunk(target_t))
            assert result.arm_source_q + result.arm_offset == target_t
            assert result.grip_source_q + result.grip_offset == target_t


def test_h16_schedule_is_exactly_multiples_of_16() -> None:
    for method in METHODS:
        executor = make_executor(method)
        query_steps = []
        for target_t in range(50):
            result = executor.step(target_t, lambda target_t=target_t: (query_steps.append(target_t) or tagged_chunk(target_t)))
            assert result.scheduled_source_q == 16 * (target_t // 16)
            assert result.scheduled_offset == target_t - result.scheduled_source_q
        assert query_steps == ([0, 16, 32, 48] if method == C1 else list(range(50)))


def test_c1_matches_standard_hard_h16_before_first_legitimate_difference() -> None:
    c1 = PreviousChunkGripExecutor()
    hard_chunks: dict[int, np.ndarray] = {}
    hard_queries: list[int] = []
    for target_t in range(16):
        c1_result = c1.step(target_t, lambda target_t=target_t: tagged_chunk(target_t))
        q = H16 * (target_t // H16)
        if q not in hard_chunks:
            hard_chunks[q] = tagged_chunk(target_t)
            hard_queries.append(target_t)
        hard_action = hard_chunks[q][target_t - q]
        np.testing.assert_array_equal(c1_result.action, hard_action)
        assert c1_result.grip_source_q == q
        assert c1_result.grip_offset == target_t - q
    assert hard_queries == [0]


def test_c1_uses_previous_chunk_gripper_offsets_16_through_31() -> None:
    executor = PreviousChunkGripExecutor()
    result = None
    for target_t in range(32):
        result = executor.step(target_t, lambda target_t=target_t: tagged_chunk(target_t))
        if target_t < 16:
            assert result.grip_source_q == 0
            assert result.grip_offset == target_t
        else:
            assert result.grip_source_q == 0
            assert result.grip_offset == target_t
            assert 16 <= result.grip_age <= 31
            expected_current = tagged_chunk(16)[target_t - 16]
            expected_previous = tagged_chunk(0)[target_t]
            np.testing.assert_array_equal(result.action[:6], expected_current[:6])
            assert result.action[6] == expected_previous[6]
    assert result is not None


def test_c1_t_less_than_16_is_identical_in_all_7_dimensions() -> None:
    executor = PreviousChunkGripExecutor()
    for target_t in range(16):
        result = executor.step(target_t, lambda target_t=target_t: tagged_chunk(target_t))
        np.testing.assert_array_equal(result.action, tagged_chunk(0)[target_t])
        assert result.previous_action is None


def test_c2_arm_remains_scheduled_while_gripper_is_fresh() -> None:
    executor = H16ArmFreshGripExecutor()
    query_steps = []
    for target_t in range(35):
        result = executor.step(target_t, lambda target_t=target_t: (query_steps.append(target_t) or tagged_chunk(target_t)))
        q = H16 * (target_t // H16)
        np.testing.assert_array_equal(result.action[:6], tagged_chunk(q)[target_t - q][:6])
        assert result.grip_source_q == target_t
        assert result.grip_offset == 0
        assert result.action[6] == tagged_chunk(target_t)[0, 6]
    assert query_steps == list(range(35))


def test_c2_at_schedule_boundary_uses_one_query_for_both_roles_without_averaging() -> None:
    executor = H16ArmFreshGripExecutor()
    calls = []
    for target_t in range(17):
        result = executor.step(target_t, lambda target_t=target_t: (calls.append(target_t) or tagged_chunk(target_t)))
        assert result.policy_queried is True
        assert result.query_q == target_t
    assert calls == list(range(17))
    np.testing.assert_array_equal(result.action[:6], tagged_chunk(16)[0, :6])
    assert result.action[6] == tagged_chunk(16)[0, 6]


@pytest.mark.parametrize("method", METHODS)
def test_no_temporal_averaging_fields_are_present(method: str) -> None:
    result = make_executor(method).step(0, lambda: tagged_chunk(0))
    assert result.previous_action is None
    if method == C2:
        assert result.fresh_action is not None
    else:
        assert result.fresh_action is None
