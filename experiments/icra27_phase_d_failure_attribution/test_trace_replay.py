from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


PATH = Path(__file__).with_name("trace_replay.py")
SPEC = importlib.util.spec_from_file_location("phase_d_trace_replay", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_point_to_oriented_box_distance() -> None:
    center = np.zeros(3)
    rotation = np.eye(3)
    half_size = np.ones(3)
    assert MODULE.point_to_oriented_box_distance(np.array([0.0, 0.0, 0.0]), center, rotation, half_size) == 0.0
    assert MODULE.point_to_oriented_box_distance(np.array([2.0, 1.0, 1.0]), center, rotation, half_size) == 1.0
    assert np.isclose(
        MODULE.point_to_oriented_box_distance(np.array([2.0, 2.0, 1.0]), center, rotation, half_size),
        np.sqrt(2.0),
    )


def test_hybrid_commands_common_prefix_and_dimensions() -> None:
    baseline = [[float(10 * t + d) for d in range(7)] for t in range(3)]
    treatment = [[float(100 + 10 * t + d) for d in range(7)] for t in range(2)]
    b_arm = np.asarray(MODULE.hybrid_commands(baseline, treatment, "baseline"))
    t_arm = np.asarray(MODULE.hybrid_commands(baseline, treatment, "treatment"))
    assert b_arm.shape == (2, 7)
    assert np.array_equal(b_arm[:, :6], np.asarray(baseline[:2])[:, :6])
    assert np.array_equal(b_arm[:, 6], np.asarray(treatment)[:, 6])
    assert np.array_equal(t_arm[:, :6], np.asarray(treatment)[:, :6])
    assert np.array_equal(t_arm[:, 6], np.asarray(baseline[:2])[:, 6])


def test_pair_type_retains_harms() -> None:
    assert MODULE.pair_type({"success": False}, {"success": True}) == "rescue"
    assert MODULE.pair_type({"success": True}, {"success": False}) == "harm"
    assert MODULE.pair_type({"success": False}, {"success": False}) == "both_fail"
    assert MODULE.pair_type({"success": True}, {"success": True}) == "both_succeed"


def test_completed_stage_latches_without_bddl_success_use_frozen_manual_fallback() -> None:
    tracker = MODULE.StageTracker.__new__(MODULE.StageTracker)
    tracker.states = [
        {"id": "first", "credited_complete": True, "opportunity_reached": True},
        {"id": "second", "credited_complete": True, "opportunity_reached": True},
    ]
    result = tracker.classification(False)
    assert result["failure_category"] == "BLIND_MANUAL_REVIEW"
    assert result["failed_stage"] is None
    assert result["ever_manipulation_opportunity"] is True
