#!/usr/bin/env python3
"""Semantic tests for the frozen Phase-0 PPPR operators.

These tests use synthetic chunks whose values encode physical source/target
indices.  They intentionally test indexing and masks independently of the
real cache and never import a simulator or policy.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pppr_metrics import (  # noqa: E402
    _validate_future_family,
    action_at,
    arm_distance,
    component_metrics,
    event_score_from_fresh_chunk,
    fit_arm_scales,
    future_consensus,
    pair_feature,
)


def encoded_chunks(timesteps: int, horizon: int) -> np.ndarray:
    """Action value is a unique function of query and chunk offset."""

    result = np.zeros((timesteps, horizon, 7), dtype=np.float64)
    for q in range(timesteps):
        for d in range(horizon):
            result[q, d, :6] = 1000.0 * q + d + np.arange(6) / 100.0
            result[q, d, 6] = 1.0 if (q + d) % 2 == 0 else -1.0
    return result


class PPPRSemanticsTest(unittest.TestCase):
    def test_physical_target_alignment(self) -> None:
        chunks = encoded_chunks(20, 12)
        # The physical target v=7 from q=3 is offset 4, not row 7.
        np.testing.assert_array_equal(action_at(chunks, 3, 7), chunks[3, 4])
        with self.assertRaises(ValueError):
            action_at(chunks, 7, 3)

    def test_rejects_q_less_than_u_future_family(self) -> None:
        with self.assertRaises(ValueError):
            _validate_future_family(old_query=4, future_query=8, family_queries=[7, 8, 9], radius=2)
        with self.assertRaises(ValueError):
            _validate_future_family(old_query=8, future_query=8, family_queries=[8, 9, 10], radius=2)
        np.testing.assert_array_equal(
            _validate_future_family(old_query=4, future_query=8, family_queries=[8, 9, 10], radius=2),
            np.array([8, 9, 10]),
        )

    def test_window_and_local_geometry(self) -> None:
        chunks = encoded_chunks(30, 12)
        row = pair_feature(chunks, old_query=3, age_steps=4, scales=np.ones(6))
        self.assertIsNotNone(row)
        assert row is not None
        np.testing.assert_array_equal(row["window_targets"], np.array([9, 10, 11, 12]))
        np.testing.assert_array_equal(row["future_family_queries"], np.array([7, 8, 9]))
        # Every family member and the old chunk address the same target v.
        for v, family in zip(row["window_targets"], row["family_actions"]):
            for q, action in zip(row["future_family_queries"], family):
                np.testing.assert_array_equal(action, action_at(chunks, int(q), int(v)))

    def test_component_metrics_and_majority_gripper(self) -> None:
        family = np.zeros((3, 7), dtype=np.float64)
        family[:, :6] = np.array([[1, 1, 1, 0, 0, 0], [3, 3, 3, 2, 2, 2], [5, 5, 5, 4, 4, 4]])
        family[:, 6] = [1.0, -1.0, 1.0]
        consensus = future_consensus(family)
        np.testing.assert_array_equal(consensus[:6], np.array([3, 3, 3, 2, 2, 2]))
        self.assertEqual(consensus[6], 1.0)
        same_grip = np.zeros(7)
        same_grip[6] = 1.0
        matching = np.array([1, 1, 1, 0, 0, 0, 1])
        np.testing.assert_allclose(
            component_metrics(same_grip, matching, np.ones(6)),
            np.array([1.0 / 3.0, 0.0, 1.0 / 6.0]),
        )
        np.testing.assert_allclose(
            component_metrics(np.zeros(7), np.array([1, 1, 1, 0, 0, 0, -1]), np.ones(6)),
            np.array([1.0 / 3.0, 1.0, 2.0 / 3.0]),
        )
        self.assertGreaterEqual(arm_distance(np.zeros(7), np.ones(7), np.ones(6)), 0.0)
        self.assertLess(arm_distance(np.zeros(7), np.ones(7), np.ones(6)), 1.0)
        self.assertEqual(component_metrics(np.zeros(7), np.zeros(7), np.ones(6))[1], 0.0)

    def test_boundary_masks_episode_and_old_chunk(self) -> None:
        # Future query boundary: q=u,u+1,u+2 must exist, while predicted
        # physical targets may extend beyond the final executed query.
        short_episode = encoded_chunks(12, 50)
        self.assertIsNone(pair_feature(short_episode, old_query=9, age_steps=1, scales=np.ones(6)))
        self.assertIsNotNone(pair_feature(short_episode, old_query=4, age_steps=4, scales=np.ones(6)))
        # Old chunk boundary: W's final target is beyond t's retained horizon.
        short_chunk = encoded_chunks(30, 7)
        self.assertIsNone(pair_feature(short_chunk, old_query=0, age_steps=4, scales=np.ones(6)))
        # The same physical geometry is valid with a sufficiently long chunk.
        valid = pair_feature(encoded_chunks(30, 10), old_query=0, age_steps=4, scales=np.ones(6))
        self.assertIsNotNone(valid)

    def test_persistent_margin_is_old_consensus_minus_dispersion(self) -> None:
        chunks = np.zeros((20, 12, 7), dtype=np.float64)
        chunks[..., 6] = 1.0
        # Use the four W targets.  The old prediction is zero; q=u,u+1,u+2
        # predict p0={2,4,6}, so consensus is 4 and dispersion is nonzero.
        for v in range(6, 10):
            chunks[0, v, 0] = 0.0  # a[v|t], t=0
            chunks[4, v - 4, 0] = 2.0  # a[v|u], raw reference
            chunks[5, v - 5, 0] = 4.0
            chunks[6, v - 6, 0] = 6.0
        row = pair_feature(chunks, old_query=0, age_steps=4, scales=np.ones(6))
        self.assertIsNotNone(row)
        assert row is not None
        expected = np.maximum(row["old_to_consensus_per_target"] - row["future_dispersion_per_target"], 0.0)
        np.testing.assert_allclose(row["pppr_per_target"], expected)
        np.testing.assert_allclose(row["pppr"], np.median(expected, axis=0))
        self.assertFalse(np.allclose(row["pppr"], row["raw"]))

    def test_development_only_iqr_scale_fitting(self) -> None:
        development = []
        for offset in (0.0, 10.0):
            episode = np.zeros((4, 3, 7), dtype=np.float64)
            episode[:, 0, :6] = np.arange(4, dtype=float)[:, None] + offset
            development.append(episode)
        held_out = np.full((4, 3, 7), 10000.0, dtype=np.float64)
        fit = fit_arm_scales(development)
        # Development Fresh values are 0..3 and 10..13, giving NumPy's fixed
        # linear-interpolation IQR of 9.5;
        # the held-out extreme values must not affect the fitted scales.
        np.testing.assert_allclose(fit.scales, np.full(6, 9.5))
        fit_with_held_out = fit_arm_scales(development + [held_out])
        self.assertFalse(np.allclose(fit.scales, fit_with_held_out.scales))

        constant = np.zeros((2, 2, 7), dtype=np.float64)
        guarded = fit_arm_scales([constant])
        np.testing.assert_allclose(guarded.scales, np.ones(6))
        self.assertTrue(np.all(guarded.zero_scale_guard_applied))

    def test_event_score_is_fresh_chunk_only_and_fixed(self) -> None:
        chunk = np.zeros((6, 7), dtype=np.float64)
        chunk[:, 6] = 1.0
        chunk[2:, 6] = -1.0
        event = event_score_from_fresh_chunk(chunk, np.ones(6))
        self.assertEqual(event["nearest_gripper_transition_offset"], 2)
        self.assertAlmostEqual(event["gripper_transition_proximity"], 1.0 - 2.0 / 5.0)
        self.assertGreaterEqual(event["event_score"], 0.0)
        self.assertLessEqual(event["event_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
