"""Same-target fixed group-delay and hard h16 ACT executors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


ACTION_DIM = 7
CHUNK_LENGTH = 100
DELAY = 20
H16 = 16
ARM = slice(0, 6)
GRIPPER_INDEX = 6
FIXED_METHODS = ("FRESH", "FO20", "REVERSE20", "FULL_OLD20")
METHODS = FIXED_METHODS + ("HARD_H16",)


@dataclass(frozen=True)
class StepResult:
    target_t: int
    action: np.ndarray
    queried: bool
    query_q: int | None
    arm_source_q: int
    arm_offset: int
    grip_source_q: int
    grip_offset: int
    fresh_action: np.ndarray | None
    old_action: np.ndarray | None

    @property
    def arm_age(self) -> int:
        return self.target_t - self.arm_source_q

    @property
    def grip_age(self) -> int:
        return self.target_t - self.grip_source_q


def _checked_chunk(chunk: np.ndarray) -> np.ndarray:
    result = np.asarray(chunk, dtype=np.float64)
    if result.shape != (CHUNK_LENGTH, ACTION_DIM):
        raise ValueError(f"expected chunk shape {(CHUNK_LENGTH, ACTION_DIM)}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("ACT chunk must be finite")
    return result.copy()


def _checked_step(target_t: int, previous_t: int | None) -> int:
    target_t = int(target_t)
    if target_t < 0 or (previous_t is not None and target_t != previous_t + 1):
        raise ValueError("target steps must start at zero and increase by one")
    return target_t


class FixedSourceExecutor:
    """Implement the four dense-query fixed-source conditions."""

    def __init__(self, method: str) -> None:
        if method not in FIXED_METHODS:
            raise ValueError(f"unknown fixed-source method: {method}")
        self.method = method
        self.reset()

    def reset(self) -> None:
        self._chunks: list[np.ndarray] = []
        self._previous_t: int | None = None

    def step(self, target_t: int, query_fn: Callable[[], np.ndarray]) -> StepResult:
        target_t = _checked_step(target_t, self._previous_t)
        chunk = _checked_chunk(query_fn())
        self._chunks.append(chunk)
        fresh = chunk[0].copy()
        old = self._chunks[target_t - DELAY][DELAY].copy() if target_t >= DELAY else None

        if target_t < DELAY or self.method == "FRESH":
            action = fresh.copy()
            arm_q = grip_q = target_t
            arm_offset = grip_offset = 0
        elif self.method == "FO20":
            action = fresh.copy()
            assert old is not None
            action[GRIPPER_INDEX] = old[GRIPPER_INDEX]
            arm_q, arm_offset = target_t, 0
            grip_q, grip_offset = target_t - DELAY, DELAY
        elif self.method == "REVERSE20":
            assert old is not None
            action = old.copy()
            action[GRIPPER_INDEX] = fresh[GRIPPER_INDEX]
            arm_q, arm_offset = target_t - DELAY, DELAY
            grip_q, grip_offset = target_t, 0
        elif self.method == "FULL_OLD20":
            assert old is not None
            action = old.copy()
            arm_q = grip_q = target_t - DELAY
            arm_offset = grip_offset = DELAY
        else:  # pragma: no cover - guarded by constructor
            raise AssertionError(self.method)

        if arm_q + arm_offset != target_t or grip_q + grip_offset != target_t:
            raise RuntimeError("fixed-source action violated q + k = t")
        if not np.isfinite(action).all():
            raise RuntimeError("fixed-source action is non-finite")
        self._previous_t = target_t
        return StepResult(
            target_t=target_t,
            action=action,
            queried=True,
            query_q=target_t,
            arm_source_q=arm_q,
            arm_offset=arm_offset,
            grip_source_q=grip_q,
            grip_offset=grip_offset,
            fresh_action=fresh,
            old_action=old,
        )


class HardH16Executor:
    """Query at multiples of 16 and execute the newest chunk at offset t-q."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._chunks: dict[int, np.ndarray] = {}
        self._previous_t: int | None = None

    @staticmethod
    def should_query(target_t: int) -> bool:
        return int(target_t) % H16 == 0

    def step(self, target_t: int, query_fn: Callable[[], np.ndarray]) -> StepResult:
        target_t = _checked_step(target_t, self._previous_t)
        queried = self.should_query(target_t)
        if queried:
            self._chunks[target_t] = _checked_chunk(query_fn())
        available = [q for q in self._chunks if q <= target_t]
        if not available:
            raise RuntimeError("hard h16 has no source chunk")
        q = max(available)
        offset = target_t - q
        if not 0 <= offset < CHUNK_LENGTH or q + offset != target_t:
            raise RuntimeError("hard h16 violated q + k = t")
        action = self._chunks[q][offset].copy()
        self._previous_t = target_t
        return StepResult(
            target_t=target_t,
            action=action,
            queried=queried,
            query_q=target_t if queried else None,
            arm_source_q=q,
            arm_offset=offset,
            grip_source_q=q,
            grip_offset=offset,
            fresh_action=None,
            old_action=None,
        )


def make_executor(method: str) -> FixedSourceExecutor | HardH16Executor:
    if method in FIXED_METHODS:
        return FixedSourceExecutor(method)
    if method == "HARD_H16":
        return HardH16Executor()
    raise ValueError(f"unknown method: {method}")


__all__ = [
    "ACTION_DIM",
    "CHUNK_LENGTH",
    "DELAY",
    "H16",
    "FIXED_METHODS",
    "METHODS",
    "StepResult",
    "FixedSourceExecutor",
    "HardH16Executor",
    "make_executor",
]
