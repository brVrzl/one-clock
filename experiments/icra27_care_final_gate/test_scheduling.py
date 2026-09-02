#!/usr/bin/env python3
"""Narrow pure tests for the frozen horizon rules and manifest."""

from __future__ import annotations

import json
import unittest

import numpy as np

from build_manifest import HELD_OUT_STATES, ROOT
from scheduling import M2_DEVELOPMENT_HISTOGRAM, gate_horizon, shuffled_horizon


class SchedulingTest(unittest.TestCase):
    def setUp(self):
        self.cell = {"task_id": 2, "state_id": 24, "environment_seed": 330224}
        self.chunk = np.zeros((50, 7), dtype=np.float32)

    def test_historical_m2_first_event_and_fallback(self):
        self.chunk[:, 6] = 1.0
        self.chunk[7:, 6] = -1.0
        horizon, _ = gate_horizon("M2_GRIPPER_EVENT", self.chunk, self.cell, 0)
        self.assertEqual(horizon, 7)
        self.chunk[:, 6] = 0.0
        horizon, _ = gate_horizon("M2_GRIPPER_EVENT", self.chunk, self.cell, 0)
        self.assertEqual(horizon, 16)

    def test_m2_ignores_transitions_before_minimum(self):
        self.chunk[:, 6] = 1.0
        self.chunk[2, 6] = -1.0
        horizon, _ = gate_horizon("M2_GRIPPER_EVENT", self.chunk, self.cell, 0)
        self.assertEqual(horizon, 16)

    def test_shuffled_is_stable_and_in_frozen_support(self):
        left = [shuffled_horizon(self.cell, i)[0] for i in range(100)]
        right = [shuffled_horizon(self.cell, i)[0] for i in range(100)]
        self.assertEqual(left, right)
        self.assertTrue(set(left) <= set(M2_DEVELOPMENT_HISTOGRAM))

    def test_manifest_counts_and_task6_subtraction(self):
        self.assertEqual(sum(len(states) for states in HELD_OUT_STATES.values()), 130)
        self.assertEqual(set((25, 26, 28, 29)) & set(HELD_OUT_STATES[6]), set())
        manifest = json.loads((ROOT / "queue_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["expected_counts"], {"gate_m": 520, "smolvla_robustness": 320})
        blocks = {}
        for cell in manifest["cells"]:
            blocks.setdefault(cell["block_id"], set()).add(cell["method"])
        gate_blocks = [methods for key, methods in blocks.items() if key.startswith("gate_m__")]
        self.assertEqual(len(gate_blocks), 130)
        self.assertTrue(all(methods == {"M0_HARD16", "M2_GRIPPER_EVENT", "FIXED_H13", "SHUFFLED_TRIGGER"} for methods in gate_blocks))


if __name__ == "__main__":
    unittest.main()

