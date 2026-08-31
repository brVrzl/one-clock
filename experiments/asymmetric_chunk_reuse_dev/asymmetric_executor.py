"""Frozen asymmetric reuse executors for the final h16 development gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


ACTION_DIM = 7
CHUNK_LENGTH = 100
H16 = 16
ARM = slice(0, 6)
GRIPPER_INDEX = 6
C1 = "C1_PREVIOUS_CHUNK_GRIP"
C2 = "C2_H16_ARM_FRESH_GRIP"
METHODS = (C1, C2)


@dataclass(frozen=True)
class StepResult:
    target_t: int
    action: np.ndarray
    queried: bool
    query_q: int | None
    scheduled_query_q: int | None
    fresh_query_q: int | None
    scheduled_source_q: int
    scheduled_offset: int
    arm_source_q: int
    arm_offset: int
    grip_source_q: int
    grip_offset: int
    scheduled_action: np.ndarray
    fresh_action: np.ndarray | None
    previous_action: np.ndarray | None
    gripper_from_cached_chunk: bool

    @property
    def arm_age(self) -> int:
        return self.target_t - self.arm_source_q

    @property
    def grip_age(self) -> int:
        return self.target_t - self.grip_source_q

    @property
    def policy_queried(self) -> bool:
        return self.queried


def checked_chunk(chunk: np.ndarray) -> np.ndarray:
    result = np.asarray(chunk, dtype=np.float64)
    if result.shape != (CHUNK_LENGTH, ACTION_DIM):
        raise ValueError(f"expected chunk shape {(CHUNK_LENGTH, ACTION_DIM)}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("ACT chunk must be finite")
    return result.copy()


def checked_target(target_t: int, previous_t: int | None) -> int:
    target_t = int(target_t)
    if target_t < 0 or (previous_t is not None and target_t != previous_t + 1):
        raise ValueError("target steps must start at zero and increase by one")
    return target_t


class ScheduledH16:
    """One shared q=0,16,32,... chunk clock."""

    def __init__(self) -> None:
        self.chunks: dict[int, np.ndarray] = {}
        self.previous_t: int | None = None

    def reset(self) -> None:
        self.chunks.clear()
        self.previous_t = None

    def has_chunk(self, q: int) -> bool:
        return int(q) in self.chunks

    def step(self, target_t: int, query_fn: Callable[[], np.ndarray]) -> tuple[int, int, bool, np.ndarray]:
        target_t = checked_target(target_t, self.previous_t)
        q = H16 * (target_t // H16)
        queried = q not in self.chunks
        if queried:
            self.chunks[q] = checked_chunk(query_fn())
        offset = target_t - q
        if not 0 <= offset < CHUNK_LENGTH or q + offset != target_t:
            raise RuntimeError("scheduled h16 source violated q+k=t")
        self.previous_t = target_t
        return q, offset, queried, self.chunks[q]


class PreviousChunkGripExecutor:
    """C1: current scheduled h16 arm plus previous scheduled h16 gripper."""

    def __init__(self) -> None:
        self.schedule = ScheduledH16()

    def reset(self) -> None:
        self.schedule.reset()

    def step(self, target_t: int, query_fn: Callable[[], np.ndarray]) -> StepResult:
        q, offset, queried, current_chunk = self.schedule.step(target_t, query_fn)
        scheduled_action = current_chunk[offset].copy()
        action = scheduled_action.copy()
        if q >= H16:
            previous_chunk = self.schedule.chunks[q - H16]
            previous_offset = offset + H16
            if not 16 <= previous_offset <= 31 or (q - H16) + previous_offset != target_t:
                raise RuntimeError("C1 previous gripper source violated q+k=t")
            previous_action = previous_chunk[previous_offset].copy()
            action[GRIPPER_INDEX] = previous_action[GRIPPER_INDEX]
            grip_q, grip_offset = q - H16, previous_offset
        else:
            previous_action = None
            grip_q, grip_offset = q, offset
        if q + offset != target_t or grip_q + grip_offset != target_t:
            raise RuntimeError("C1 source violated q+k=t")
        return StepResult(
            target_t=target_t,
            action=action,
            queried=queried,
            query_q=q if queried else None,
            scheduled_query_q=q if queried else None,
            fresh_query_q=None,
            scheduled_source_q=q,
            scheduled_offset=offset,
            arm_source_q=q,
            arm_offset=offset,
            grip_source_q=grip_q,
            grip_offset=grip_offset,
            scheduled_action=scheduled_action,
            fresh_action=None,
            previous_action=previous_action,
            gripper_from_cached_chunk=True,
        )


class H16ArmFreshGripExecutor:
    """C2: scheduled h16 arm, with a fresh dense same-target gripper query."""

    def __init__(self) -> None:
        self.schedule = ScheduledH16()

    def reset(self) -> None:
        self.schedule.reset()

    def step(self, target_t: int, query_fn: Callable[[], np.ndarray]) -> StepResult:
        q, offset, scheduled_queried, scheduled_chunk = self.schedule.step(target_t, query_fn)
        if scheduled_queried:
            # At q, the one query is both the scheduled A_q chunk and fresh A_t.
            fresh_chunk = scheduled_chunk
        else:
            fresh_chunk = checked_chunk(query_fn())
        scheduled_action = scheduled_chunk[offset].copy()
        fresh_action = fresh_chunk[0].copy()
        action = scheduled_action.copy()
        action[GRIPPER_INDEX] = fresh_action[GRIPPER_INDEX]
        if q + offset != target_t or target_t + 0 != target_t:
            raise RuntimeError("C2 source violated q+k=t")
        return StepResult(
            target_t=target_t,
            action=action,
            queried=True,
            query_q=target_t,
            scheduled_query_q=q if scheduled_queried else None,
            fresh_query_q=target_t,
            scheduled_source_q=q,
            scheduled_offset=offset,
            arm_source_q=q,
            arm_offset=offset,
            grip_source_q=target_t,
            grip_offset=0,
            scheduled_action=scheduled_action,
            fresh_action=fresh_action,
            previous_action=None,
            gripper_from_cached_chunk=False,
        )


def make_executor(method: str) -> PreviousChunkGripExecutor | H16ArmFreshGripExecutor:
    if method == C1:
        return PreviousChunkGripExecutor()
    if method == C2:
        return H16ArmFreshGripExecutor()
    raise ValueError(f"unknown method: {method}")


__all__ = [
    "ACTION_DIM",
    "CHUNK_LENGTH",
    "H16",
    "ARM",
    "GRIPPER_INDEX",
    "C1",
    "C2",
    "METHODS",
    "StepResult",
    "PreviousChunkGripExecutor",
    "H16ArmFreshGripExecutor",
    "make_executor",
    "checked_chunk",
]
