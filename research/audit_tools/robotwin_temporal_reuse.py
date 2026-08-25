#!/usr/bin/env python3
"""RoboTwin ACT aggregation and same-current-decision-target reuse."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


METHODS = ("NEWEST", "FULL_OLD_17", "FO_17")
PHYSICAL_AGE_METHODS = ("NEWEST", "FULL_OLD_1S", "FO_1S")
GRIPPER_CONTROL_METHODS = ("GRIPPER_HOLD", "GRIPPER_EMA_1S")
NATIVE_METHOD = "NATIVE_ACT"
ACTION_DIM = 14
CHUNK_LENGTH = 50
NOMINAL_SOURCE_AGE_TICKS = 17
PHYSICAL_SOURCE_AGE_SECONDS = 1.0

ACTION_GROUPS = {
    "left_arm": tuple(range(0, 6)),
    "left_gripper": (6,),
    "right_arm": tuple(range(7, 13)),
    "right_gripper": (13,),
}
ARM_GROUPS = ("left_arm", "right_arm")
GRIPPER_GROUPS = ("left_gripper", "right_gripper")


@dataclass(frozen=True)
class RoboTwinTemporalAction:
    method: str
    target_step: int
    action: np.ndarray
    fresh_action: np.ndarray
    old_action: np.ndarray | None
    fresh_source_step: int
    fresh_chunk_offset: int
    old_source_step: int | None
    old_chunk_offset: int | None
    group_source_steps: dict[str, int]
    group_chunk_offsets: dict[str, int]
    group_source_ages: dict[str, int]

    def as_log_record(self) -> dict[str, object]:
        return {
            "method": self.method,
            "target_decision": self.target_step,
            "fresh_source_step": self.fresh_source_step,
            "fresh_chunk_offset": self.fresh_chunk_offset,
            "old_source_step": self.old_source_step,
            "old_chunk_offset": self.old_chunk_offset,
            "effective_source_step_per_group": dict(self.group_source_steps),
            "chunk_offset_per_group": dict(self.group_chunk_offsets),
            "effective_source_age_ticks_per_group": dict(self.group_source_ages),
            "fresh_action": self.fresh_action.tolist(),
            "old_action": None if self.old_action is None else self.old_action.tolist(),
            "executed_composed_action": self.action.tolist(),
        }


class RoboTwinTemporalExecutor:
    """Compose predictions aligned to one current evaluator decision target."""

    def __init__(
        self,
        method: str,
        *,
        source_age_ticks: int = NOMINAL_SOURCE_AGE_TICKS,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"unsupported method: {method!r}")
        if source_age_ticks < 1 or source_age_ticks >= CHUNK_LENGTH:
            raise ValueError(
                f"source_age_ticks must be within [1, {CHUNK_LENGTH - 1}]"
            )
        self.method = method
        self.source_age_ticks = int(source_age_ticks)
        self._chunks: deque[tuple[int, np.ndarray]] = deque()

    def reset(self) -> None:
        self._chunks.clear()

    def update(self, source_step: int, chunk: np.ndarray) -> RoboTwinTemporalAction:
        source_step = int(source_step)
        chunk = np.asarray(chunk)
        if not np.issubdtype(chunk.dtype, np.floating):
            raise ValueError("ACT chunk must have a floating-point dtype")
        if chunk.shape != (CHUNK_LENGTH, ACTION_DIM):
            raise ValueError(
                f"expected chunk shape {(CHUNK_LENGTH, ACTION_DIM)}, got {chunk.shape}"
            )
        if not np.isfinite(chunk).all():
            raise ValueError("ACT chunk contains non-finite values")
        if self._chunks and source_step != self._chunks[-1][0] + 1:
            raise ValueError("RoboTwin temporal reuse requires one ordered query per tick")

        self._chunks.append((source_step, chunk.copy()))
        while self._chunks and source_step - self._chunks[0][0] >= CHUNK_LENGTH:
            self._chunks.popleft()

        target_step = source_step
        fresh = chunk[0].copy()
        old_source_step = target_step - self.source_age_ticks
        old: np.ndarray | None = None
        if old_source_step >= 0:
            matches = [saved for step, saved in self._chunks if step == old_source_step]
            if len(matches) != 1:
                raise RuntimeError("missing or duplicate old-source ACT query")
            old = matches[0][self.source_age_ticks].copy()

        use_old = old is not None and self.method != "NEWEST"
        action = fresh.copy()
        if use_old and self.method == "FULL_OLD_17":
            action = old.copy()
        elif use_old and self.method == "FO_17":
            for group_name in GRIPPER_GROUPS:
                indices = ACTION_GROUPS[group_name]
                action[list(indices)] = old[list(indices)]

        group_source_steps: dict[str, int] = {}
        group_chunk_offsets: dict[str, int] = {}
        group_source_ages: dict[str, int] = {}
        for group_name in ACTION_GROUPS:
            group_uses_old = use_old and (
                self.method == "FULL_OLD_17"
                or (self.method == "FO_17" and group_name in GRIPPER_GROUPS)
            )
            group_source_steps[group_name] = old_source_step if group_uses_old else source_step
            group_chunk_offsets[group_name] = self.source_age_ticks if group_uses_old else 0
            group_source_ages[group_name] = self.source_age_ticks if group_uses_old else 0

        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise RuntimeError("RoboTwin temporal composition produced an invalid action")
        return RoboTwinTemporalAction(
            method=self.method,
            target_step=target_step,
            action=action.astype(np.float32),
            fresh_action=fresh,
            old_action=old,
            fresh_source_step=source_step,
            fresh_chunk_offset=0,
            old_source_step=old_source_step if old is not None else None,
            old_chunk_offset=self.source_age_ticks if old is not None else None,
            group_source_steps=group_source_steps,
            group_chunk_offsets=group_chunk_offsets,
            group_source_ages=group_source_ages,
        )


@dataclass(frozen=True)
class PhysicalAgeSourceSelection:
    target_step: int
    target_query_time_seconds: float
    old_source_step: int
    old_source_query_time_seconds: float
    chunk_offset: int
    realized_source_age_seconds: float
    absolute_age_error_seconds: float


def select_physical_age_source(
    query_times_by_source: dict[int, float],
    target_step: int,
    *,
    target_age_seconds: float = PHYSICAL_SOURCE_AGE_SECONDS,
    chunk_length: int = CHUNK_LENGTH,
) -> PhysicalAgeSourceSelection | None:
    """Select the past query nearest the frozen physical source age.

    Only simulator/query timestamps supplied by the caller enter the rule. If
    two sources have exactly equal absolute error, the more recent source is
    selected, which gives the smaller current-decision-target chunk offset. At decision 0
    there is no past query, so the caller must use NEWEST.
    """

    target_step = int(target_step)
    target_age_seconds = float(target_age_seconds)
    chunk_length = int(chunk_length)
    if target_age_seconds <= 0 or not np.isfinite(target_age_seconds):
        raise ValueError("target_age_seconds must be finite and positive")
    if chunk_length < 2:
        raise ValueError("chunk_length must provide at least one past offset")
    if target_step not in query_times_by_source:
        raise ValueError("target query timestamp is missing")

    target_time = float(query_times_by_source[target_step])
    if not np.isfinite(target_time):
        raise ValueError("query timestamps must be finite")
    candidates: list[tuple[float, int, float, float]] = []
    for source_step, source_time_value in query_times_by_source.items():
        source_step = int(source_step)
        offset = target_step - source_step
        if offset <= 0 or offset >= chunk_length:
            continue
        source_time = float(source_time_value)
        if not np.isfinite(source_time):
            raise ValueError("query timestamps must be finite")
        realized_age = target_time - source_time
        if realized_age < 0:
            raise ValueError("past query timestamp occurs after the target query")
        absolute_error = abs(realized_age - target_age_seconds)
        candidates.append((absolute_error, -source_step, source_time, realized_age))

    if not candidates:
        return None
    absolute_error, neg_source_step, source_time, realized_age = min(candidates)
    source_step = -neg_source_step
    return PhysicalAgeSourceSelection(
        target_step=target_step,
        target_query_time_seconds=target_time,
        old_source_step=source_step,
        old_source_query_time_seconds=source_time,
        chunk_offset=target_step - source_step,
        realized_source_age_seconds=realized_age,
        absolute_age_error_seconds=absolute_error,
    )


@dataclass(frozen=True)
class RoboTwinPhysicalAgeAction:
    method: str
    target_step: int
    target_query_time_seconds: float
    action: np.ndarray
    fresh_action: np.ndarray
    old_action: np.ndarray | None
    fresh_source_step: int
    fresh_chunk_offset: int
    old_source_step: int | None
    old_chunk_offset: int | None
    old_source_query_time_seconds: float | None
    realized_source_age_seconds: float | None
    absolute_age_error_seconds: float | None
    group_source_steps: dict[str, int]
    group_chunk_offsets: dict[str, int]
    group_source_ages_ticks: dict[str, int]
    group_source_ages_seconds: dict[str, float]

    def as_log_record(self) -> dict[str, object]:
        return {
            "method": self.method,
            "target_decision": self.target_step,
            "target_query_time_seconds": self.target_query_time_seconds,
            "fresh_source_step": self.fresh_source_step,
            "fresh_chunk_offset": self.fresh_chunk_offset,
            "old_source_step": self.old_source_step,
            "old_chunk_offset": self.old_chunk_offset,
            "old_source_query_time_seconds": self.old_source_query_time_seconds,
            "candidate_old_source_age_seconds": self.realized_source_age_seconds,
            "candidate_old_absolute_age_error_seconds": self.absolute_age_error_seconds,
            "effective_source_step_per_group": dict(self.group_source_steps),
            "chunk_offset_per_group": dict(self.group_chunk_offsets),
            "effective_source_age_ticks_per_group": dict(
                self.group_source_ages_ticks
            ),
            "effective_source_age_seconds_per_group": dict(
                self.group_source_ages_seconds
            ),
            "fresh_action": self.fresh_action.tolist(),
            "old_action": None if self.old_action is None else self.old_action.tolist(),
            "executed_composed_action": self.action.tolist(),
        }


class RoboTwinPhysicalAgeExecutor:
    """Compose same-current-decision-target actions using simulator timestamps."""

    def __init__(
        self,
        method: str,
        *,
        target_age_seconds: float = PHYSICAL_SOURCE_AGE_SECONDS,
    ) -> None:
        if method not in PHYSICAL_AGE_METHODS:
            raise ValueError(f"unsupported physical-age method: {method!r}")
        if target_age_seconds <= 0 or not np.isfinite(target_age_seconds):
            raise ValueError("target_age_seconds must be finite and positive")
        self.method = method
        self.target_age_seconds = float(target_age_seconds)
        self._queries: deque[tuple[int, float, np.ndarray]] = deque()

    def reset(self) -> None:
        self._queries.clear()

    def update(
        self,
        source_step: int,
        chunk: np.ndarray,
        *,
        query_time_seconds: float,
    ) -> RoboTwinPhysicalAgeAction:
        source_step = int(source_step)
        query_time_seconds = float(query_time_seconds)
        chunk = np.asarray(chunk)
        if not np.isfinite(query_time_seconds):
            raise ValueError("query_time_seconds must be finite")
        if not np.issubdtype(chunk.dtype, np.floating):
            raise ValueError("ACT chunk must have a floating-point dtype")
        if chunk.shape != (CHUNK_LENGTH, ACTION_DIM):
            raise ValueError(
                f"expected chunk shape {(CHUNK_LENGTH, ACTION_DIM)}, got {chunk.shape}"
            )
        if not np.isfinite(chunk).all():
            raise ValueError("ACT chunk contains non-finite values")
        if self._queries:
            if source_step != self._queries[-1][0] + 1:
                raise ValueError(
                    "RoboTwin temporal reuse requires one ordered query per tick"
                )
            if query_time_seconds <= self._queries[-1][1]:
                raise ValueError("simulator query timestamps must strictly increase")

        self._queries.append((source_step, query_time_seconds, chunk.copy()))
        while self._queries and source_step - self._queries[0][0] >= CHUNK_LENGTH:
            self._queries.popleft()

        query_times = {step: timestamp for step, timestamp, _ in self._queries}
        selection = select_physical_age_source(
            query_times,
            source_step,
            target_age_seconds=self.target_age_seconds,
        )
        fresh = chunk[0].copy()
        old: np.ndarray | None = None
        if selection is not None:
            matches = [
                saved
                for step, _, saved in self._queries
                if step == selection.old_source_step
            ]
            if len(matches) != 1:
                raise RuntimeError("missing or duplicate physical-age ACT query")
            old = matches[0][selection.chunk_offset].copy()

        use_old = old is not None and self.method != "NEWEST"
        action = fresh.copy()
        if use_old and self.method == "FULL_OLD_1S":
            action = old.copy()
        elif use_old and self.method == "FO_1S":
            for group_name in GRIPPER_GROUPS:
                indices = ACTION_GROUPS[group_name]
                action[list(indices)] = old[list(indices)]

        old_step = None if selection is None else selection.old_source_step
        old_offset = None if selection is None else selection.chunk_offset
        old_time = (
            None if selection is None else selection.old_source_query_time_seconds
        )
        realized_age = (
            None if selection is None else selection.realized_source_age_seconds
        )
        absolute_error = (
            None if selection is None else selection.absolute_age_error_seconds
        )
        group_source_steps: dict[str, int] = {}
        group_chunk_offsets: dict[str, int] = {}
        group_source_ages_ticks: dict[str, int] = {}
        group_source_ages_seconds: dict[str, float] = {}
        for group_name in ACTION_GROUPS:
            group_uses_old = use_old and (
                self.method == "FULL_OLD_1S"
                or (self.method == "FO_1S" and group_name in GRIPPER_GROUPS)
            )
            group_source_steps[group_name] = old_step if group_uses_old else source_step
            group_chunk_offsets[group_name] = old_offset if group_uses_old else 0
            group_source_ages_ticks[group_name] = old_offset if group_uses_old else 0
            group_source_ages_seconds[group_name] = (
                realized_age if group_uses_old else 0.0
            )

        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise RuntimeError("RoboTwin temporal composition produced an invalid action")
        return RoboTwinPhysicalAgeAction(
            method=self.method,
            target_step=source_step,
            target_query_time_seconds=query_time_seconds,
            action=action.astype(np.float32),
            fresh_action=fresh,
            old_action=old,
            fresh_source_step=source_step,
            fresh_chunk_offset=0,
            old_source_step=old_step,
            old_chunk_offset=old_offset,
            old_source_query_time_seconds=old_time,
            realized_source_age_seconds=realized_age,
            absolute_age_error_seconds=absolute_error,
            group_source_steps=group_source_steps,
            group_chunk_offsets=group_chunk_offsets,
            group_source_ages_ticks=group_source_ages_ticks,
            group_source_ages_seconds=group_source_ages_seconds,
        )


@dataclass(frozen=True)
class RoboTwinGripperControlAction:
    method: str
    target_step: int
    query_time_seconds: float
    action: np.ndarray
    fresh_action: np.ndarray
    previous_executed_grippers: np.ndarray | None
    executed_grippers: np.ndarray
    ema_alpha: float | None

    def as_log_record(self) -> dict[str, object]:
        return {
            "method": self.method,
            "target_decision": self.target_step,
            "simulator_query_time_seconds": self.query_time_seconds,
            "fresh_action": self.fresh_action.tolist(),
            "previous_executed_grippers": (
                None
                if self.previous_executed_grippers is None
                else self.previous_executed_grippers.tolist()
            ),
            "executed_grippers": self.executed_grippers.tolist(),
            "ema_alpha": self.ema_alpha,
            "executed_composed_action": self.action.tolist(),
        }


class RoboTwinGripperControlExecutor:
    """Apply frozen HOLD or physical-time EMA in executed-command space."""

    def __init__(self, method: str, *, ema_tau_seconds: float = 1.0) -> None:
        if method not in GRIPPER_CONTROL_METHODS:
            raise ValueError(f"unsupported gripper control method: {method!r}")
        if ema_tau_seconds <= 0 or not np.isfinite(ema_tau_seconds):
            raise ValueError("ema_tau_seconds must be finite and positive")
        self.method = method
        self.ema_tau_seconds = float(ema_tau_seconds)
        self._last_step: int | None = None
        self._last_query_time_seconds: float | None = None
        self._executed_grippers: np.ndarray | None = None

    def reset(self) -> None:
        self._last_step = None
        self._last_query_time_seconds = None
        self._executed_grippers = None

    def update(
        self,
        source_step: int,
        fresh_action: np.ndarray,
        *,
        query_time_seconds: float,
    ) -> RoboTwinGripperControlAction:
        source_step = int(source_step)
        query_time_seconds = float(query_time_seconds)
        fresh_action = np.asarray(fresh_action)
        if fresh_action.shape != (ACTION_DIM,) or not np.isfinite(fresh_action).all():
            raise ValueError("fresh postprocessed action must be one finite 14-D vector")
        if not np.issubdtype(fresh_action.dtype, np.floating):
            raise ValueError("fresh action must have a floating-point dtype")
        if self._last_step is not None:
            if source_step != self._last_step + 1:
                raise ValueError("gripper control requires one ordered query per tick")
            if query_time_seconds <= self._last_query_time_seconds:
                raise ValueError("simulator query timestamps must strictly increase")
        elif source_step != 0:
            raise ValueError("gripper control episode must start at decision 0")

        fresh_grippers = fresh_action[[6, 13]].copy()
        previous_grippers = (
            None if self._executed_grippers is None else self._executed_grippers.copy()
        )
        ema_alpha: float | None = None
        if previous_grippers is None:
            executed_grippers = fresh_grippers
        elif self.method == "GRIPPER_HOLD":
            executed_grippers = previous_grippers
        else:
            dt = query_time_seconds - self._last_query_time_seconds
            ema_alpha = float(np.exp(-dt / self.ema_tau_seconds))
            executed_grippers = (
                ema_alpha * previous_grippers + (1.0 - ema_alpha) * fresh_grippers
            )

        action = fresh_action.copy()
        action[[6, 13]] = executed_grippers
        self._last_step = source_step
        self._last_query_time_seconds = query_time_seconds
        self._executed_grippers = executed_grippers.copy()
        return RoboTwinGripperControlAction(
            method=self.method,
            target_step=source_step,
            query_time_seconds=query_time_seconds,
            action=action.astype(np.float32),
            fresh_action=fresh_action.copy(),
            previous_executed_grippers=previous_grippers,
            executed_grippers=executed_grippers,
            ema_alpha=ema_alpha,
        )


def native_act_aggregate(
    chunks_by_source: dict[int, np.ndarray],
    target_step: int,
    *,
    weight_decay: float = 0.01,
) -> np.ndarray:
    """Reproduce official ACT temporal aggregation in normalized action space.

    Sources are ordered by their evaluator row index (oldest to newest), exactly
    as in ``detr/act_policy.py``. Each source contributes the chunk entry whose
    target is ``target_step``. The official nonzero-row filter is preserved.
    """

    target_step = int(target_step)
    candidates = []
    for source_step in sorted(chunks_by_source):
        offset = target_step - int(source_step)
        if offset < 0 or offset >= CHUNK_LENGTH:
            continue
        chunk = np.asarray(chunks_by_source[source_step], dtype=np.float64)
        if chunk.shape != (CHUNK_LENGTH, ACTION_DIM):
            raise ValueError(
                f"expected chunk shape {(CHUNK_LENGTH, ACTION_DIM)}, got {chunk.shape}"
            )
        candidate = chunk[offset]
        if np.all(candidate != 0):
            candidates.append(candidate)

    if not candidates:
        raise RuntimeError("official temporal aggregation has no populated candidates")
    candidates_array = np.stack(candidates)
    if not np.isfinite(candidates_array).all():
        raise ValueError("ACT aggregation candidate contains non-finite values")
    weights = np.exp(-float(weight_decay) * np.arange(len(candidates_array)))
    weights /= weights.sum()
    return np.sum(candidates_array * weights[:, None], axis=0)


def postprocess_action(
    normalized_action: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> np.ndarray:
    """Apply the official ACT affine action denormalization."""

    action = np.asarray(normalized_action) * np.asarray(action_std) + np.asarray(
        action_mean
    )
    if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
        raise ValueError("postprocessed ACT action is invalid")
    return action
