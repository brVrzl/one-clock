from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
ASYMMETRIC_ROOT = REPO_ROOT / "experiments" / "asymmetric_chunk_reuse_dev"
TEMPORAL_ROOT = REPO_ROOT / "experiments" / "group_delay_factorial_act20"
sys.path.insert(0, str(ASYMMETRIC_ROOT))
sys.path.insert(0, str(TEMPORAL_ROOT))

from asymmetric_executor import CHUNK_LENGTH, H16ArmFreshGripExecutor  # noqa: E402
from temporal_reuse import make_executor as make_reference_executor  # noqa: E402


def tagged_chunk(source_q: int) -> np.ndarray:
    return np.asarray(
        [
            [10000 * source_q + 100 * offset + dimension for dimension in range(7)]
            for offset in range(CHUNK_LENGTH)
        ],
        dtype=np.float64,
    )


def test_c2_arm_source_and_index_match_hard_h16_for_t0_through_t31() -> None:
    c2 = H16ArmFreshGripExecutor()
    hard_h16 = make_reference_executor("HARD_H16")
    c2_queries: list[int] = []
    hard_queries: list[int] = []
    for t in range(32):
        c2_result = c2.step(t, lambda t=t: (c2_queries.append(t) or tagged_chunk(t)))
        hard_result = hard_h16.step(t, lambda t=t: (hard_queries.append(t) or tagged_chunk(t)))
        q_arm = 16 * (t // 16)
        k_arm = t - q_arm
        assert c2_result.arm_source_q == q_arm == hard_result.arm_source_q
        assert c2_result.arm_offset == k_arm == hard_result.arm_offset
        np.testing.assert_array_equal(c2_result.action[:6], hard_result.action[:6])
        assert c2_result.arm_source_q + c2_result.arm_offset == t
        assert c2_result.grip_source_q + c2_result.grip_offset == t
        assert c2_result.grip_source_q == t
        assert c2_result.grip_offset == 0
        assert c2_result.grip_age == 0
        assert c2_result.policy_queried
        assert c2_result.query_q == t
        assert c2_result.action[6] == tagged_chunk(t)[0, 6]
    assert c2_queries == list(range(32))
    assert hard_queries == [0, 16]


@pytest.mark.parametrize(
    ("t", "expected_q", "expected_k"),
    [(0, 0, 0), (15, 0, 15), (16, 16, 0), (31, 16, 15)],
)
def test_c2_explicit_boundary_sources(t: int, expected_q: int, expected_k: int) -> None:
    c2 = H16ArmFreshGripExecutor()
    result = None
    for target_t in range(t + 1):
        result = c2.step(target_t, lambda target_t=target_t: tagged_chunk(target_t))
    assert result is not None
    assert result.arm_source_q == expected_q
    assert result.arm_offset == expected_k
    assert result.arm_source_q + result.arm_offset == t
    assert result.grip_source_q == t
    assert result.grip_offset == 0
    assert result.grip_source_q + result.grip_offset == t
