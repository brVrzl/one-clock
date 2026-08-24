from __future__ import annotations

import unittest

import numpy as np

from one_clock import AffineResidualCalibrator, ExponentialChunkSmoother


class PostPolicyTest(unittest.TestCase):
    def test_exponential_smoothing_is_causal_within_chunk(self) -> None:
        chunk = np.asarray([[0.0], [2.0], [2.0]], dtype=np.float32)

        result = ExponentialChunkSmoother(0.5)(state=np.zeros(1), action_chunk=chunk, task_id=0)

        np.testing.assert_allclose(result.action_chunk[:, 0], [0.0, 1.0, 1.5])
        self.assertTrue(result.gate_active)

    def test_affine_calibrator_recovers_constant_residual(self) -> None:
        action = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        state = np.zeros((4, 1))
        target = action + 0.25
        model = AffineResidualCalibrator.fit(
            action=action,
            state=state,
            position=np.asarray([0, 1, 0, 1]),
            task_id=np.zeros(4, dtype=np.int64),
            target=target,
            chunk_size=2,
            task_count=1,
            ridge=1e-8,
        )

        result = model(state=np.zeros(1), action_chunk=action[:2], task_id=0)

        np.testing.assert_allclose(result.action_chunk, target[:2], atol=1e-6)
        self.assertGreater(result.correction_norm, 0.0)

    def test_affine_calibrator_can_restrict_correction_dimensions(self) -> None:
        action = np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        model = AffineResidualCalibrator.fit(
            action=action,
            state=np.zeros((4, 1)),
            position=np.asarray([0, 1, 0, 1]),
            task_id=np.zeros(4, dtype=np.int64),
            target=action + 0.25,
            chunk_size=2,
            task_count=1,
            ridge=1e-8,
        )
        model.correction_dimensions = (1,)

        result = model(state=np.zeros(1), action_chunk=action[:2], task_id=0)

        np.testing.assert_allclose(result.action_chunk[:, 0], action[:2, 0])
        self.assertTrue(np.any(result.action_chunk[:, 1] != action[:2, 1]))


if __name__ == "__main__":
    unittest.main()
