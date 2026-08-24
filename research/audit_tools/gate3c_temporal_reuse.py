#!/usr/bin/env python3
"""Frozen Gate-3C asymmetric temporal-source and full-action baselines."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from gate3a2_temporal_aggregation import temporal_weights


METHODS = (
    "A_NEWEST",
    "B_FULL_OLD20",
    "C_ASYMMETRIC_FO20",
    "D_AGE_EXP_B003",
    "E_COGACT_A03",
)
SOURCE_AGE_TICKS = 20
CHUNK_LENGTH = 100
ACTION_DIM = 7
ARM_SLICE = slice(0, 6)
GRIPPER_INDEX = 6
POLICY_TEMPORAL_ENSEMBLE_ACTIVE = False
ACTION_SMOOTHING_ACTIVE = False

SCALAR_METHOD = {
    "D_AGE_EXP_B003": "newest_age_exp_b003",
    "E_COGACT_A03": "cogact_a03",
}


@dataclass(frozen=True)
class ExecutedAction:
    action: np.ndarray
    fresh_action: np.ndarray
    old_action: np.ndarray | None
    fresh_source_step: int
    old_source_step: int | None
    old_chunk_offset: int | None
    arm_effective_age_ticks: float
    gripper_effective_age_ticks: float
    candidate_ages: np.ndarray
    weights: np.ndarray

    @property
    def intervention_active(self) -> bool:
        return self.old_action is not None


def compose_fixed_action(method: str, fresh: np.ndarray, old: np.ndarray | None) -> np.ndarray:
    """Apply A/B/C exactly; B and C fall back to full fresh before d=20."""

    if method not in METHODS[:3]:
        raise ValueError(f"fixed-source method must be one of {METHODS[:3]}, got {method!r}")
    fresh = np.asarray(fresh, dtype=np.float64)
    if fresh.shape != (ACTION_DIM,):
        raise ValueError("fresh action must be 7-D")
    if old is None or method == "A_NEWEST":
        return fresh.copy()
    old = np.asarray(old, dtype=np.float64)
    if old.shape != (ACTION_DIM,):
        raise ValueError("old action must be 7-D")
    if method == "B_FULL_OLD20":
        return old.copy()
    action = fresh.copy()
    action[GRIPPER_INDEX] = old[GRIPPER_INDEX]
    return action


class Gate3CTemporalExecutor:
    """Cache every chunk and execute one frozen Gate-3C condition."""

    def __init__(self, method: str) -> None:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        self.method = method
        self._chunks: deque[tuple[int, np.ndarray]] = deque()

    def reset(self) -> None:
        self._chunks.clear()

    def update(self, source_step: int, chunk: np.ndarray) -> ExecutedAction:
        source_step = int(source_step)
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.shape != (CHUNK_LENGTH, ACTION_DIM):
            raise ValueError(f"expected chunk shape {(CHUNK_LENGTH, ACTION_DIM)}, got {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise ValueError("ACT chunk contains non-finite values")
        if self._chunks and source_step != self._chunks[-1][0] + 1:
            raise ValueError("Gate-3C requires one ordered ACT query per controller step")
        self._chunks.append((source_step, chunk.copy()))
        while self._chunks and source_step - self._chunks[0][0] >= CHUNK_LENGTH:
            self._chunks.popleft()

        source_steps = np.asarray([step for step, _ in self._chunks], dtype=np.int64)
        ages = source_step - source_steps
        candidates = np.stack(
            [saved[int(age)] for age, (_, saved) in zip(ages, self._chunks, strict=True)]
        )
        fresh = candidates[-1].copy()
        old_source_step: int | None = None
        old: np.ndarray | None = None
        if source_step >= SOURCE_AGE_TICKS:
            old_source_step = source_step - SOURCE_AGE_TICKS
            positions = np.flatnonzero(source_steps == old_source_step)
            if len(positions) != 1:
                raise RuntimeError("missing or duplicate q=t-20 source")
            old_position = int(positions[0])
            if int(ages[old_position]) != SOURCE_AGE_TICKS:
                raise RuntimeError("q=t-20 did not map to chunk offset 20")
            old = candidates[old_position].copy()

        if self.method in SCALAR_METHOD:
            weights = temporal_weights(SCALAR_METHOD[self.method], candidates, ages)
            action = weights @ candidates
            effective_age = float(weights @ ages)
            arm_age = effective_age
            gripper_age = effective_age
        else:
            action = compose_fixed_action(self.method, fresh, old)
            weights = np.zeros(len(candidates), dtype=np.float64)
            weights[-1] = 1.0
            if old is not None and self.method == "B_FULL_OLD20":
                weights[np.flatnonzero(ages == SOURCE_AGE_TICKS)[0]] = 1.0
                weights[-1] = 0.0
                arm_age = gripper_age = float(SOURCE_AGE_TICKS)
            elif old is not None and self.method == "C_ASYMMETRIC_FO20":
                # There is no single full-action weight vector for C.
                weights[:] = np.nan
                arm_age = 0.0
                gripper_age = float(SOURCE_AGE_TICKS)
            else:
                arm_age = gripper_age = 0.0

        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise RuntimeError("Gate-3C produced an invalid executed action")
        return ExecutedAction(
            action=action.astype(np.float32),
            fresh_action=fresh,
            old_action=old,
            fresh_source_step=source_step,
            old_source_step=old_source_step,
            old_chunk_offset=SOURCE_AGE_TICKS if old is not None else None,
            arm_effective_age_ticks=arm_age,
            gripper_effective_age_ticks=gripper_age,
            candidate_ages=ages.copy(),
            weights=weights.copy(),
        )
