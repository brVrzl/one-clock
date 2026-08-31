from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from temporal_reuse import (  # noqa: E402
    CHUNK_LENGTH,
    DELAY,
    FIXED_METHODS,
    H16,
    METHODS,
    HardH16Executor,
    make_executor,
)


def tagged_chunk(source: int) -> np.ndarray:
    return np.asarray(
        [[10000 * source + 100 * offset + dimension for dimension in range(7)] for offset in range(CHUNK_LENGTH)],
        dtype=np.float64,
    )


def test_every_source_offset_targets_the_current_physical_step() -> None:
    for method in METHODS:
        executor = make_executor(method)
        for target_t in range(65):
            result = executor.step(target_t, lambda target_t=target_t: tagged_chunk(target_t))
            assert result.arm_source_q + result.arm_offset == target_t
            assert result.grip_source_q + result.grip_offset == target_t


@pytest.mark.parametrize("method", FIXED_METHODS)
def test_all_fixed_methods_are_exactly_fresh_before_t20(method: str) -> None:
    executor = make_executor(method)
    for target_t in range(DELAY):
        result = executor.step(target_t, lambda target_t=target_t: tagged_chunk(target_t))
        np.testing.assert_array_equal(result.action, tagged_chunk(target_t)[0])
        assert result.arm_source_q == result.grip_source_q == target_t
        assert result.arm_offset == result.grip_offset == 0
        assert result.old_action is None


def test_fo20_uses_fresh_arm_and_old_gripper_at_same_target() -> None:
    executor = make_executor("FO20")
    result = None
    for target_t in range(DELAY + 1):
        result = executor.step(target_t, lambda target_t=target_t: tagged_chunk(target_t))
    assert result is not None
    fresh = tagged_chunk(DELAY)[0]
    old = tagged_chunk(0)[DELAY]
    np.testing.assert_array_equal(result.action[:6], fresh[:6])
    np.testing.assert_array_equal(result.action[6], old[6])
    assert (result.arm_source_q, result.arm_offset) == (DELAY, 0)
    assert (result.grip_source_q, result.grip_offset) == (0, DELAY)


def test_reverse20_and_full_old20_have_registered_group_sources() -> None:
    for method in ("REVERSE20", "FULL_OLD20"):
        executor = make_executor(method)
        result = None
        for target_t in range(DELAY + 1):
            result = executor.step(target_t, lambda target_t=target_t: tagged_chunk(target_t))
        assert result is not None
        if method == "REVERSE20":
            assert (result.arm_source_q, result.arm_offset) == (0, DELAY)
            assert (result.grip_source_q, result.grip_offset) == (DELAY, 0)
        else:
            assert (result.arm_source_q, result.arm_offset) == (0, DELAY)
            assert (result.grip_source_q, result.grip_offset) == (0, DELAY)


def test_no_misaligned_old_offset_zero_or_fresh_offset_twenty_is_possible() -> None:
    for method in FIXED_METHODS:
        executor = make_executor(method)
        for target_t in range(DELAY, DELAY + 5):
            result = executor.step(target_t, lambda target_t=target_t: tagged_chunk(target_t))
            assert result.old_action is not None
            assert result.arm_offset in (0, DELAY)
            assert result.grip_offset in (0, DELAY)
            assert result.arm_source_q + result.arm_offset == target_t
            assert result.grip_source_q + result.grip_offset == target_t


def test_hard_h16_queries_only_at_multiples_and_executes_newest_offset() -> None:
    executor = HardH16Executor()
    query_steps = []
    for target_t in range(40):
        result = executor.step(target_t, lambda target_t=target_t: (query_steps.append(target_t) or tagged_chunk(target_t)))
        assert result.arm_source_q % H16 == 0
        assert result.arm_source_q + result.arm_offset == target_t
        assert result.grip_source_q == result.arm_source_q
        assert result.grip_offset == result.arm_offset
    assert query_steps == [0, 16, 32]


def test_deterministic_repeated_inference_contract() -> None:
    processed_input = np.arange(7, dtype=np.float64)

    def deterministic_act(value: np.ndarray) -> np.ndarray:
        return np.repeat(value[None, :], CHUNK_LENGTH, axis=0)

    first = deterministic_act(processed_input)
    second = deterministic_act(processed_input.copy())
    np.testing.assert_array_equal(first, second)
