from __future__ import annotations

import numpy as np
import torch

from research.audit_tools.robotwin_dcta import (
    ACTION_DIM,
    CHUNK_LENGTH,
    DCTAExecutor,
    DynamicTemporalGate,
    GROUPS,
    aggregate_candidates,
    dcta_action,
    native_act_weights,
)


def batch(count: int = 8):
    torch.manual_seed(7)
    candidates = torch.randn(2, CHUNK_LENGTH, ACTION_DIM)
    mask = torch.zeros(2, CHUNK_LENGTH, dtype=torch.bool)
    mask[:, :count] = True
    lags = torch.zeros(2, CHUNK_LENGTH)
    lags[:, :count] = torch.arange(count - 1, -1, -1)
    ages = lags * 0.06
    qpos = torch.randn(2, ACTION_DIM)
    context = torch.randn(2, 512)
    return candidates, mask, lags, ages, qpos, context


def test_group_contract() -> None:
    assert GROUPS == {
        "left_arm": tuple(range(0, 6)),
        "left_gripper": (6,),
        "right_arm": tuple(range(7, 13)),
        "right_gripper": (13,),
    }


def test_weights_sum_to_one_and_warmup_mask_is_respected() -> None:
    candidates, mask, *_ = batch(count=3)
    _, weights = aggregate_candidates(candidates, mask)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4))
    assert torch.count_nonzero(weights[..., 3:]) == 0


def test_zero_residual_reproduces_native_act() -> None:
    candidates, mask, lags, ages, qpos, context = batch()
    gate = DynamicTemporalGate()
    native, native_group_weights = aggregate_candidates(candidates, mask)
    dcta, dcta_weights = dcta_action(
        gate, candidates, lags, ages, qpos, context, mask, shared=False
    )
    shared, shared_weights = dcta_action(
        gate, candidates, lags, ages, qpos, context, mask, shared=True
    )
    torch.testing.assert_close(dcta, native, rtol=0, atol=1e-7)
    torch.testing.assert_close(shared, native, rtol=0, atol=1e-7)
    torch.testing.assert_close(dcta_weights, native_group_weights, rtol=0, atol=1e-7)
    torch.testing.assert_close(shared_weights, native_group_weights, rtol=0, atol=1e-7)


def test_shared_dynamic_weights_are_identical_across_groups() -> None:
    candidates, mask, lags, ages, qpos, context = batch()
    gate = DynamicTemporalGate()
    with torch.no_grad():
        gate.head.weight.normal_()
    _, weights = dcta_action(
        gate, candidates, lags, ages, qpos, context, mask, shared=True
    )
    for group in range(1, 4):
        torch.testing.assert_close(weights[:, 0], weights[:, group])


def test_dcta_can_use_different_group_weights() -> None:
    candidates, mask, *_ = batch()
    residuals = torch.zeros(2, 4, CHUNK_LENGTH)
    residuals[:, 0, 0] = 2
    residuals[:, 1, 1] = 2
    _, weights = aggregate_candidates(candidates, mask, residuals)
    assert not torch.equal(weights[:, 0], weights[:, 1])


def test_executor_alignment_no_future_and_reset() -> None:
    gate = DynamicTemporalGate()
    executor = DCTAExecutor(gate, "NATIVE_ACT", torch.device("cpu"))
    for decision in range(4):
        chunk = np.empty((50, 14), dtype=np.float32)
        for offset in range(50):
            chunk[offset] = 1000 * decision + 10 * offset + np.arange(14)
        result = executor.update(
            decision,
            0.1 * decision,
            chunk,
            np.zeros(14, dtype=np.float32),
            np.zeros(512, dtype=np.float32),
        )
    assert result.candidate_sources == (0, 1, 2, 3)
    assert result.candidate_offsets == (3, 2, 1, 0)
    assert all(source <= 3 for source in result.candidate_sources)
    executor.reset()
    chunk = np.zeros((50, 14), dtype=np.float32)
    reset = executor.update(0, 1.0, chunk, np.zeros(14), np.zeros(512))
    assert reset.candidate_sources == (0,)
    assert reset.candidate_offsets == (0,)
    np.testing.assert_array_equal(reset.weights, np.ones((4, 1)))


def test_native_weight_order_matches_pinned_oldest_first_decay() -> None:
    mask = torch.zeros(1, 50, dtype=torch.bool)
    mask[0, :4] = True
    weights = native_act_weights(mask, torch.float32)[0, :4]
    assert torch.all(weights[:-1] > weights[1:])
