#!/usr/bin/env python3
"""Pure ACT chunk re-query rules for the bounded group-event experiment.

The operators inspect only the newly predicted chunk.  They return a joint
execution horizon; no previous chunk or action is read by these rules.
"""

from __future__ import annotations

from typing import Any

import numpy as np


MIN_HORIZON = 4
MAX_HORIZON = 16
ARM_PHASE_THRESHOLD = 0.5
EPSILON = 1e-8


def _validate_chunk(chunk: np.ndarray) -> np.ndarray:
    values = np.asarray(chunk, dtype=np.float64)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2 or values.shape[1] != 7 or values.shape[0] < MAX_HORIZON:
        raise ValueError(f"ACT chunk must have shape (H>=16,7), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("ACT chunk must be finite")
    return values


def arm_phase_profile(chunk: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Return v_arm indexed by action offset and its frozen diagnostics."""

    values = _validate_chunk(chunk)
    xyz = np.linalg.norm(values[1:16, :3] - values[:15, :3], axis=1)
    rotation = np.linalg.norm(values[1:16, 3:6] - values[:15, 3:6], axis=1)
    xyz_scale = float(np.median(xyz) + EPSILON)
    rotation_scale = float(np.median(rotation) + EPSILON)
    normalized_xyz = xyz / xyz_scale
    normalized_rotation = rotation / rotation_scale
    combined = 0.5 * normalized_xyz + 0.5 * normalized_rotation
    # Index zero is unused; entry k is the increment from action k-1 to k.
    profile = np.zeros(16, dtype=np.float64)
    profile[1:16] = combined
    return profile, {
        "translation_increments": xyz,
        "rotation_increments": rotation,
        "translation_median_plus_epsilon": xyz_scale,
        "rotation_median_plus_epsilon": rotation_scale,
        "normalized_arm_speed": profile,
    }


def arm_phase_horizon(chunk: np.ndarray) -> tuple[int, dict[str, Any]]:
    """Select the earliest bounded local low-speed arm phase boundary."""

    profile, diagnostics = arm_phase_profile(chunk)
    candidates: list[int] = []
    # Offset 15 has no k+1 neighbor within the frozen first-16 window.
    for k in range(MIN_HORIZON, MAX_HORIZON + 1):
        if k + 1 >= len(profile):
            continue
        if profile[k] <= profile[k - 1] and profile[k] <= profile[k + 1] and profile[k] <= ARM_PHASE_THRESHOLD:
            candidates.append(k)
    horizon = candidates[0] if candidates else MAX_HORIZON
    diagnostics.update(
        {
            "arm_boundary_candidates": candidates,
            "h_arm": int(horizon),
            "arm_triggered": bool(candidates),
            "arm_trigger_offset": int(horizon) if candidates else None,
        }
    )
    return int(horizon), diagnostics


def gripper_intent(values: np.ndarray) -> np.ndarray:
    """Apply the established ACT contract: nonnegative=open, negative=close."""

    commands = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.where(commands >= 0.0, 1, -1).astype(np.int8)


def gripper_event_horizon(chunk: np.ndarray) -> tuple[int, dict[str, Any]]:
    """Select the earliest open/close intent transition in offsets 4..15."""

    values = _validate_chunk(chunk)
    intents = gripper_intent(values[:16, 6])
    events = [k for k in range(MIN_HORIZON, MAX_HORIZON) if intents[k] != intents[k - 1]]
    horizon = events[0] if events else MAX_HORIZON
    diagnostics = {
        "gripper_intents_first_16": intents,
        "gripper_event_candidates": events,
        "h_grip": int(horizon),
        "gripper_triggered": bool(events),
        "gripper_trigger_offset": int(horizon) if events else None,
    }
    if events:
        diagnostics["gripper_intent_before"] = int(intents[events[0] - 1])
        diagnostics["gripper_intent_after"] = int(intents[events[0]])
    return int(horizon), diagnostics


def choose_horizon(method: str, chunk: np.ndarray) -> tuple[int, dict[str, Any]]:
    """Compute one joint horizon from one new chunk for one method."""

    values = _validate_chunk(chunk)
    if method == "M0_hard16":
        return MAX_HORIZON, {
            "h_arm": MAX_HORIZON,
            "h_grip": MAX_HORIZON,
            "h_exec": MAX_HORIZON,
            "trigger_reason": "hard16",
            "arm_triggered": False,
            "gripper_triggered": False,
            "both_nominated": False,
            "both_nearby": False,
        }
    if method == "M1_arm_phase":
        h_arm, arm = arm_phase_horizon(values)
        h_grip = MAX_HORIZON
        grip = {
            "gripper_event_candidates": [],
            "gripper_triggered": False,
            "gripper_trigger_offset": None,
        }
        h_exec = h_arm
        reason = "arm_phase" if arm["arm_triggered"] else "no_arm_phase_fallback"
    elif method == "M2_gripper_event":
        h_arm = MAX_HORIZON
        arm = {
            "arm_boundary_candidates": [],
            "arm_triggered": False,
            "arm_trigger_offset": None,
        }
        h_grip, grip = gripper_event_horizon(values)
        h_exec = h_grip
        reason = "gripper_event" if grip["gripper_triggered"] else "no_gripper_event_fallback"
    elif method == "M3_group_event_joint":
        h_arm, arm = arm_phase_horizon(values)
        h_grip, grip = gripper_event_horizon(values)
        h_exec = min(MAX_HORIZON, h_arm, h_grip)
        both = bool(arm["arm_triggered"] and grip["gripper_triggered"])
        if both and h_arm == h_grip:
            reason = "arm_and_gripper_same_boundary"
        elif both and abs(h_arm - h_grip) <= 1:
            reason = "arm_and_gripper_nearby_boundaries"
        elif arm["arm_triggered"] and h_arm == h_exec:
            reason = "arm_phase"
        elif grip["gripper_triggered"] and h_grip == h_exec:
            reason = "gripper_event"
        else:
            reason = "joint_fallback"
    else:
        raise ValueError(f"unknown method {method!r}")
    if not MIN_HORIZON <= h_exec <= MAX_HORIZON:
        raise AssertionError(f"invalid execution horizon {h_exec}")
    both = bool(arm["arm_triggered"] and grip["gripper_triggered"])
    diagnostics = {
        **arm,
        **grip,
        "h_arm": int(h_arm),
        "h_grip": int(h_grip),
        "h_exec": int(h_exec),
        "trigger_reason": reason,
        "arm_triggered": bool(arm["arm_triggered"]),
        "gripper_triggered": bool(grip["gripper_triggered"]),
        "both_nominated": both,
        "both_nearby": bool(both and abs(h_arm - h_grip) <= 1),
    }
    return int(h_exec), diagnostics


def action_from_newest_chunk(chunk: np.ndarray, query_step: int, target_step: int) -> tuple[np.ndarray, int]:
    """Return only the current chunk's same-query action and its offset."""

    values = _validate_chunk(chunk)
    offset = int(target_step) - int(query_step)
    if offset < 0 or offset >= len(values):
        raise ValueError(f"target {target_step} is outside newest chunk from query {query_step}")
    return values[offset].copy(), offset
