"""Sparse-query hard execution and temporal ensembling.

The executor is deliberately policy- and environment-independent.  A caller
provides a postprocessed ``(H_pred, action_dim)`` chunk only when the fixed
query schedule asks for one.  Every action returned by this module is a
prediction for the same physical target step.  This keeps the critical
query-cadence and temporal-alignment semantics testable without importing
LeRobot or MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np


Mode = Literal["hard", "sparse_te"]


@dataclass(frozen=True)
class QueryRecord:
    """One sparse policy query and its postprocessed action chunk."""

    query_step: int
    chunk: np.ndarray

    def __post_init__(self) -> None:
        query_step = int(self.query_step)
        if query_step < 0:
            raise ValueError("query_step must be nonnegative")
        chunk = np.asarray(self.chunk, dtype=np.float64)
        if chunk.ndim == 3 and chunk.shape[0] == 1:
            chunk = chunk[0]
        if chunk.ndim != 2 or chunk.shape[0] == 0 or chunk.shape[1] == 0:
            raise ValueError(f"chunk must have shape (H,D), got {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise ValueError("cached action chunks must be finite")
        object.__setattr__(self, "query_step", query_step)
        object.__setattr__(self, "chunk", chunk.copy())


@dataclass(frozen=True)
class CandidateSet:
    """Same-target candidates ordered from oldest query to newest query."""

    source_query_steps: np.ndarray
    offsets: np.ndarray
    actions: np.ndarray

    def __post_init__(self) -> None:
        sources = np.asarray(self.source_query_steps, dtype=np.int64)
        offsets = np.asarray(self.offsets, dtype=np.int64)
        actions = np.asarray(self.actions, dtype=np.float64)
        if sources.ndim != 1 or offsets.shape != sources.shape:
            raise ValueError("source_query_steps and offsets must be aligned vectors")
        if actions.ndim != 2 or actions.shape[0] != len(sources):
            raise ValueError("actions must have one row per source query")
        if len(sources) == 0:
            raise ValueError("candidate set cannot be empty")
        if not np.isfinite(actions).all():
            raise ValueError("candidate actions must be finite")
        object.__setattr__(self, "source_query_steps", sources.copy())
        object.__setattr__(self, "offsets", offsets.copy())
        object.__setattr__(self, "actions", actions.copy())

    @property
    def ages(self) -> np.ndarray:
        """Physical source ages, synonymous with same-target offsets."""

        return self.offsets

    @property
    def candidate_count(self) -> int:
        return len(self.source_query_steps)


@dataclass(frozen=True)
class StepResult:
    """Action and provenance emitted for one physical target step."""

    action: np.ndarray
    target_step: int
    queried: bool
    latest_query_step: int
    candidates: CandidateSet
    weights: np.ndarray

    def __post_init__(self) -> None:
        action = np.asarray(self.action, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)
        if action.ndim != 1 or action.shape[0] != self.candidates.actions.shape[1]:
            raise ValueError("action must be a vector matching candidate action dimension")
        if weights.shape != (len(self.candidates.source_query_steps),):
            raise ValueError("weights must have one value per candidate")
        if not np.isfinite(action).all() or not np.isfinite(weights).all():
            raise ValueError("result values must be finite")
        object.__setattr__(self, "action", action.copy())
        object.__setattr__(self, "weights", weights.copy())
        object.__setattr__(self, "target_step", int(self.target_step))
        object.__setattr__(self, "latest_query_step", int(self.latest_query_step))

    @property
    def candidate_count(self) -> int:
        return len(self.candidates.source_query_steps)

    @property
    def weighted_source_age(self) -> float:
        return float(self.weights @ self.candidates.offsets)


def canonical_temporal_weights(candidate_count: int, coefficient: float = 0.01) -> np.ndarray:
    """Return canonical ACT weights, oldest candidate first.

    The coefficient is intentionally explicit and defaults to the fixed
    experiment value.  The index is the candidate index, not physical age.
    """

    candidate_count = int(candidate_count)
    coefficient = float(coefficient)
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if not np.isfinite(coefficient) or coefficient < 0:
        raise ValueError("coefficient must be finite and nonnegative")
    logits = -coefficient * np.arange(candidate_count, dtype=np.float64)
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


class SparseExecutor:
    """Fixed-cadence executor for hard newest or canonical sparse TE action."""

    def __init__(
        self,
        *,
        cadence: int,
        prediction_horizon: int,
        mode: Mode,
        coefficient: float = 0.01,
        action_dim: int = 7,
    ) -> None:
        self.cadence = int(cadence)
        self.prediction_horizon = int(prediction_horizon)
        self.mode = mode
        self.coefficient = float(coefficient)
        self.action_dim = int(action_dim)
        if self.cadence <= 0:
            raise ValueError("cadence must be positive")
        if self.prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive")
        if self.mode not in ("hard", "sparse_te"):
            raise ValueError(f"unknown mode: {self.mode}")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if not np.isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError("coefficient must be finite and nonnegative")
        self.records: list[QueryRecord] = []
        self.query_steps: list[int] = []

    def reset(self) -> None:
        """Clear all cached chunks; call this at every episode boundary."""

        self.records.clear()
        self.query_steps.clear()

    def should_query(self, target_step: int) -> bool:
        target_step = int(target_step)
        if target_step < 0:
            raise ValueError("target_step must be nonnegative")
        return target_step % self.cadence == 0

    def same_target_candidates(self, target_step: int) -> CandidateSet:
        """Enumerate every valid sparse chunk prediction for ``target_step``."""

        target_step = int(target_step)
        sources: list[int] = []
        offsets: list[int] = []
        rows: list[np.ndarray] = []
        for record in self.records:
            offset = target_step - record.query_step
            if 0 <= offset < self.prediction_horizon and offset < len(record.chunk):
                sources.append(record.query_step)
                offsets.append(offset)
                rows.append(record.chunk[offset].copy())
        if not rows:
            raise RuntimeError(f"no valid cached prediction for target step {target_step}")
        return CandidateSet(
            source_query_steps=np.asarray(sources, dtype=np.int64),
            offsets=np.asarray(offsets, dtype=np.int64),
            actions=np.stack(rows),
        )

    def step(self, target_step: int, query_fn: Callable[[], np.ndarray]) -> StepResult:
        """Query if scheduled, then execute an action for ``target_step``.

        ``query_fn`` is called exactly once at scheduled steps and never at
        other steps.  Query chunks are inserted before same-target extraction,
        so a re-query at ``t=h`` contributes both ``A_0[h]`` and ``A_h[0]``.
        """

        target_step = int(target_step)
        if target_step < 0:
            raise ValueError("target_step must be nonnegative")
        queried = self.should_query(target_step)
        if queried:
            if self.records and target_step <= self.records[-1].query_step:
                raise RuntimeError("query steps must increase strictly")
            chunk = np.asarray(query_fn(), dtype=np.float64)
            if chunk.ndim == 3 and chunk.shape[0] == 1:
                chunk = chunk[0]
            if chunk.ndim != 2 or chunk.shape[1] != self.action_dim:
                raise ValueError(
                    f"query_fn must return (H,{self.action_dim}), got {chunk.shape}"
                )
            if chunk.shape[0] < self.prediction_horizon:
                raise ValueError(
                    f"query chunk horizon {chunk.shape[0]} is shorter than configured "
                    f"prediction horizon {self.prediction_horizon}"
                )
            record = QueryRecord(target_step, chunk)
            self.records.append(record)
            self.query_steps.append(target_step)

        candidates = self.same_target_candidates(target_step)
        if self.mode == "hard":
            weights = np.zeros(candidates.candidate_count, dtype=np.float64)
            weights[-1] = 1.0
        else:
            weights = canonical_temporal_weights(candidates.candidate_count, self.coefficient)
        action = weights @ candidates.actions
        return StepResult(
            action=action,
            target_step=target_step,
            queried=queried,
            latest_query_step=int(candidates.source_query_steps[-1]),
            candidates=candidates,
            weights=weights,
        )
