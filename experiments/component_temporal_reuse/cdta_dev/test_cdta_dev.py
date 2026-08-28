"""CPU semantic/unit tests for the frozen CDTA-16 development runner."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_cdta_dev as runner  # noqa: E402


def test_protocol_is_exactly_the_frozen_200_episode_panel() -> None:
    import json

    protocol = json.loads((ROOT / "protocol.json").read_text())
    assert [(task["suite"], task["task_id"]) for task in protocol["tasks"]] == [
        ("libero_object", 6), ("libero_spatial", 2), ("libero_goal", 1), ("libero_10", 3)
    ]
    assert protocol["environment"]["initial_state_ids"] == list(range(10, 20))
    assert protocol["environment"]["seeds"] == list(range(2000, 2010))
    assert protocol["methods"] == list(runner.METHODS)


def test_cdta_uses_independent_age_limited_weights_and_ordinary_aggregation() -> None:
    actions = np.asarray([
        [0.0, 0, 0, 0, 0, 0, -1.0],
        [1.0, 0, 0, 0, 0, 0, 1.0],
        [2.0, 0, 0, 0, 0, 0, 1.0],
        [3.0, 0, 0, 0, 0, 0, 1.0],
    ], dtype=np.float64)
    candidates = type("C", (), {"actions": actions, "ages": np.asarray([17, 2, 1, 0])})()
    action, arm_age, gripper_age, details = runner.compose_action(
        "cdta_a16_alpha03_beta003", candidates
    )
    assert details["candidate_count"] == 3
    assert 0.0 <= arm_age <= 2.0 and 0.0 <= gripper_age <= 2.0
    expected_weights = np.exp(-0.03 * np.asarray([2.0, 1.0, 0.0]))
    expected_weights /= expected_weights.sum()
    np.testing.assert_allclose(action[0], expected_weights @ np.asarray([1.0, 2.0, 3.0]))
    assert action[6] == 1.0  # sign agreement changes weights, never a discrete vote
