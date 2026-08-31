"""Small semantic tests for the outcome-blind group-delay audit."""

from __future__ import annotations

import inspect

import numpy as np

from experiments.group_temporal_memory_offline import analyze_outcome_blind as analysis
from research.overnight_pppr_20260828.pppr_metrics import action_at, fit_arm_scales


def test_same_target_alignment_uses_source_query_plus_offset() -> None:
    chunks = np.zeros((6, 5, 7), dtype=float)
    for query in range(6):
        for offset in range(5):
            chunks[query, offset] = 100.0 * query + offset + np.arange(7)
    source_q = 2
    delay = 3
    target_t = source_q + delay
    np.testing.assert_array_equal(
        action_at(chunks, source_q, target_t), chunks[source_q, delay]
    )
    assert source_q + delay == target_t


def test_group_slicing_is_six_dimensional_arm_and_scalar_gripper() -> None:
    action = np.arange(7, dtype=float)
    slices = analysis.group_slices()
    np.testing.assert_array_equal(action[slices["arm"]], np.arange(6, dtype=float))
    np.testing.assert_array_equal(action[slices["gripper"]], np.asarray([6.0]))


def test_delay_availability_masks_source_before_episode_start_or_chunk_end() -> None:
    np.testing.assert_array_equal(
        analysis.valid_target_indices(10, 50, 0), np.arange(10, dtype=np.int64)
    )
    np.testing.assert_array_equal(
        analysis.valid_target_indices(10, 50, 4), np.arange(4, 10, dtype=np.int64)
    )
    np.testing.assert_array_equal(
        analysis.valid_target_indices(10, 50, 32), np.empty(0, dtype=np.int64)
    )


def test_arm_normalization_is_fitted_from_development_input_only() -> None:
    development = np.zeros((8, 2, 7), dtype=float)
    development[:, 0, :6] = np.arange(8, dtype=float)[:, None]
    held_out = np.full((1, 2, 7), 1000.0, dtype=float)
    dev_fit = fit_arm_scales([development])
    contaminated_fit = fit_arm_scales([development, held_out])
    assert not np.allclose(dev_fit.scales, contaminated_fit.scales)
    np.testing.assert_allclose(
        fit_arm_scales([development]).scales, dev_fit.scales
    )


def test_fresh_zero_delay_is_identity_for_revision_metrics() -> None:
    action = np.asarray([[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -1.0]])
    metrics = analysis.pair_metrics(action, action, np.ones(6, dtype=float))
    np.testing.assert_allclose(metrics["arm_revision"], 0.0)
    np.testing.assert_allclose(metrics["gripper_abs_diff"], 0.0)
    np.testing.assert_allclose(metrics["gripper_sign_disagreement"], 0.0)
    np.testing.assert_allclose(metrics["utility_arm"], 1.0)
    np.testing.assert_allclose(metrics["utility_gripper"], 1.0)


def test_h_temp_builder_has_no_closed_loop_success_dependency() -> None:
    source = inspect.getsource(analysis.build_analysis)
    assert "success_rate" not in source
    assert "final_analysis" not in source
    assert "pilot_results" not in source
    assert "intervention" not in source
