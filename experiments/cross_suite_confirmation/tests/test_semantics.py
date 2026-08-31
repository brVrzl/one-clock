from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEMPORAL = ROOT.parent / "group_delay_factorial_act20"
sys.path.insert(0, str(TEMPORAL))

from temporal_reuse import H16, METHODS, make_executor  # noqa: E402


def tagged_chunk(source: int) -> np.ndarray:
    return np.asarray(
        [[10000 * source + 100 * offset + dimension for dimension in range(7)] for offset in range(100)],
        dtype=np.float64,
    )


def test_same_target_and_fixed_source_semantics() -> None:
    for method in METHODS:
        executor = make_executor(method)
        query_steps: list[int] = []
        for t in range(65):
            result = executor.step(t, lambda t=t: (query_steps.append(t) or tagged_chunk(t)))
            assert result.arm_source_q + result.arm_offset == t
            assert result.grip_source_q + result.grip_offset == t
            if method != "HARD_H16" and t < 20:
                np.testing.assert_array_equal(result.action, tagged_chunk(t)[0])
            if method == "FO20" and t >= 20:
                np.testing.assert_array_equal(result.action[:6], tagged_chunk(t)[0, :6])
                assert result.grip_source_q == t - 20 and result.grip_offset == 20
            if method == "REVERSE20" and t >= 20:
                np.testing.assert_array_equal(result.action[:6], tagged_chunk(t - 20)[20, :6])
                assert result.grip_source_q == t and result.grip_offset == 0
            if method == "FULL_OLD20" and t >= 20:
                np.testing.assert_array_equal(result.action, tagged_chunk(t - 20)[20])
        if method == "HARD_H16":
            assert query_steps == [0, 16, 32, 48, 64]
        else:
            assert query_steps == list(range(65))


def test_hard_h16_newest_chunk_offset() -> None:
    executor = make_executor("HARD_H16")
    for t in range(65):
        result = executor.step(t, lambda t=t: tagged_chunk(t))
        q = H16 * (t // H16)
        assert result.arm_source_q == q
        assert result.grip_source_q == q
        assert result.arm_offset == t - q
        assert result.grip_offset == t - q
        np.testing.assert_array_equal(result.action, tagged_chunk(q)[t - q])


def test_no_misaligned_fixed_source_controls() -> None:
    for method in ("FO20", "REVERSE20", "FULL_OLD20"):
        executor = make_executor(method)
        for t in range(40):
            result = executor.step(t, lambda t=t: tagged_chunk(t))
            if t >= 20:
                assert result.arm_source_q + result.arm_offset == result.target_t
                assert result.grip_source_q + result.grip_offset == result.target_t
