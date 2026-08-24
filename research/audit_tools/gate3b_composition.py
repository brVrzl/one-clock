#!/usr/bin/env python3
"""Fixed-age joint-source and cross-source composition for Gate-3B."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


METHODS = ("FF", "OO", "FO", "OF")
SOURCE_AGE_TICKS = 20
CHUNK_LENGTH = 100
ACTION_DIM = 7
ARM_SLICE = slice(0, 6)
GRIPPER_INDEX = 6
TEMPORAL_ENSEMBLE_ACTIVE = False
ACTION_STD = np.asarray(
    [
        0.2681190073490143,
        0.4384443759918213,
        0.4475117325782776,
        0.024448219686746597,
        0.04936208948493004,
        0.042103495448827744,
        0.9974462985992432,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class ComposedAction:
    action: np.ndarray
    fresh_action: np.ndarray
    old_action: np.ndarray | None
    fresh_source_step: int
    old_source_step: int | None
    fresh_chunk_offset: int
    old_chunk_offset: int | None
    arm_source_step: int
    gripper_source_step: int

    @property
    def intervention_active(self) -> bool:
        return self.old_action is not None


def action_sign(value: float) -> int:
    return 1 if value >= 0.0 else -1


def rotation_geodesic(left: np.ndarray, right: np.ndarray) -> float:
    relative = Rotation.from_rotvec(left).inv() * Rotation.from_rotvec(right)
    return float(relative.magnitude())


def control_semantic_distance(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    """Audited translation/SO(3)/gripper-sign distance decomposition."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != (ACTION_DIM,) or right.shape != (ACTION_DIM,):
        raise ValueError("control-semantic distance requires two 7-D actions")
    translation_delta = left[:3] - right[:3]
    translation_normalized_mse = float(np.mean((translation_delta / ACTION_STD[:3]) ** 2))
    translation_l2 = float(np.linalg.norm(translation_delta))
    rotation_radians = rotation_geodesic(left[3:6], right[3:6])
    rotation_normalized_sq = rotation_radians**2 / float(np.sum(ACTION_STD[3:6] ** 2))
    gripper_sign_disagreement = float(action_sign(float(left[6])) != action_sign(float(right[6])))
    dimension_weighted = (
        3.0 * translation_normalized_mse
        + 3.0 * rotation_normalized_sq
        + gripper_sign_disagreement
    ) / 7.0
    return {
        "dimension_weighted_semantic_distance": float(dimension_weighted),
        "translation_normalized_mse": translation_normalized_mse,
        "translation_l2_action_units": translation_l2,
        "rotation_geodesic_radians": rotation_radians,
        "rotation_normalized_sq": float(rotation_normalized_sq),
        "gripper_sign_disagreement": gripper_sign_disagreement,
        "raw_arm_l2": float(np.linalg.norm(left[:6] - right[:6])),
    }


def compose_action(method: str, fresh: np.ndarray, old: np.ndarray | None) -> np.ndarray:
    """Apply the frozen Gate-3B cell formula without averaging or smoothing."""

    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    fresh = np.asarray(fresh, dtype=np.float64)
    if fresh.shape != (ACTION_DIM,):
        raise ValueError(f"fresh action must have shape ({ACTION_DIM},)")
    if old is None:
        return fresh.copy()
    old = np.asarray(old, dtype=np.float64)
    if old.shape != (ACTION_DIM,):
        raise ValueError(f"old action must have shape ({ACTION_DIM},)")
    if method == "FF":
        return fresh.copy()
    if method == "OO":
        return old.copy()
    action = np.empty(ACTION_DIM, dtype=np.float64)
    if method == "FO":
        action[ARM_SLICE] = fresh[ARM_SLICE]
        action[GRIPPER_INDEX] = old[GRIPPER_INDEX]
    else:
        action[ARM_SLICE] = old[ARM_SLICE]
        action[GRIPPER_INDEX] = fresh[GRIPPER_INDEX]
    return action


class FixedAgeComposer:
    """Cache every full chunk and compose only age-0 and age-20 candidates."""

    def __init__(
        self,
        method: str,
        *,
        source_age_ticks: int = SOURCE_AGE_TICKS,
        chunk_length: int = CHUNK_LENGTH,
        action_dim: int = ACTION_DIM,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if source_age_ticks != SOURCE_AGE_TICKS:
            raise ValueError(f"Gate-3B fixes source age at {SOURCE_AGE_TICKS} ticks")
        if chunk_length <= source_age_ticks:
            raise ValueError("the fixed old-source offset must lie inside the chunk")
        if action_dim != ACTION_DIM:
            raise ValueError(f"Gate-3B requires the verified {ACTION_DIM}-D LIBERO action")
        self.method = method
        self.source_age_ticks = source_age_ticks
        self.chunk_length = chunk_length
        self.action_dim = action_dim
        self._chunks: deque[tuple[int, np.ndarray]] = deque()

    def reset(self) -> None:
        self._chunks.clear()

    def update(self, source_step: int, chunk: np.ndarray) -> ComposedAction:
        source_step = int(source_step)
        chunk = np.asarray(chunk, dtype=np.float64)
        expected_shape = (self.chunk_length, self.action_dim)
        if chunk.shape != expected_shape:
            raise ValueError(f"expected chunk shape {expected_shape}, got {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise ValueError("ACT chunk contains non-finite values")
        if self._chunks and source_step != self._chunks[-1][0] + 1:
            raise ValueError("Gate-3B requires exactly one ordered ACT query per controller step")

        self._chunks.append((source_step, chunk.copy()))
        while self._chunks and source_step - self._chunks[0][0] > self.source_age_ticks:
            self._chunks.popleft()

        fresh = chunk[0].copy()
        old_source_step: int | None = None
        old: np.ndarray | None = None
        if source_step >= self.source_age_ticks:
            expected_old_source = source_step - self.source_age_ticks
            cached_source, cached_chunk = self._chunks[0]
            if cached_source != expected_old_source:
                raise RuntimeError(
                    f"old-source cache mismatch: expected q={expected_old_source}, got q={cached_source}"
                )
            old_source_step = cached_source
            old = cached_chunk[self.source_age_ticks].copy()

        action = compose_action(self.method, fresh, old)
        if not np.isfinite(action).all():
            raise RuntimeError("Gate-3B composition produced a non-finite action")
        if old is None or self.method == "FF":
            arm_source_step = source_step
            gripper_source_step = source_step
        elif self.method == "OO":
            arm_source_step = old_source_step
            gripper_source_step = old_source_step
        elif self.method == "FO":
            arm_source_step = source_step
            gripper_source_step = old_source_step
        else:
            arm_source_step = old_source_step
            gripper_source_step = source_step
        assert arm_source_step is not None and gripper_source_step is not None
        return ComposedAction(
            action=action.astype(np.float32),
            fresh_action=fresh,
            old_action=old,
            fresh_source_step=source_step,
            old_source_step=old_source_step,
            fresh_chunk_offset=0,
            old_chunk_offset=self.source_age_ticks if old is not None else None,
            arm_source_step=arm_source_step,
            gripper_source_step=gripper_source_step,
        )
