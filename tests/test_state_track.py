from __future__ import annotations

import unittest

import numpy as np

from one_clock import StateTrackChunk, nominal_eef_trajectory


def observation(position=(0.0, 0.0, 0.0)) -> dict[str, object]:
    return {
        "robot_state": {
            "eef": {
                "pos": np.asarray(position, dtype=np.float64),
                "quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            }
        }
    }


class StateTrackTest(unittest.TestCase):
    def test_nominal_trajectory_uses_osc_scales(self) -> None:
        chunk = np.zeros((3, 7), dtype=np.float64)
        chunk[:, 0] = 1.0
        trajectory = nominal_eef_trajectory(observation(), chunk, active_horizon=3)
        np.testing.assert_allclose(trajectory[:, 0], [0.05, 0.10, 0.15])
        np.testing.assert_allclose(trajectory[:, 1:], 0.0, atol=1e-8)

    def test_progress_is_monotonic_and_only_selects_rows(self) -> None:
        chunk = np.arange(5 * 7, dtype=np.float64).reshape(5, 7) / 100.0
        tracker = StateTrackChunk(lookahead=1, active_horizon=5, max_forward_skip=1)
        tracker.start_chunk(observation(), chunk)
        first, first_diag = tracker.select(observation())
        self.assertEqual(first_diag.progress_index, 0)
        np.testing.assert_array_equal(first, chunk[1])
        later_obs = observation(position=(0.05, 0.0, 0.0))
        second, second_diag = tracker.select(later_obs)
        self.assertGreaterEqual(second_diag.progress_index, first_diag.progress_index)
        self.assertTrue(any(np.array_equal(second, row) for row in chunk))
        self.assertTrue(np.array_equal(second[:6], chunk[second_diag.selected_index, :6]))


if __name__ == "__main__":
    unittest.main()

