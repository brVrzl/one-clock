#!/usr/bin/env python3
"""Scalar temporal aggregation rules for the Gate-3A2 rollout audit."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


METHODS = (
    "newest",
    "exact_act_m001",
    "cogact_a03",
    "newest_age_exp_b003",
)


@dataclass(frozen=True)
class AggregatedAction:
    action: np.ndarray
    candidate_count: int
    candidate_ages: np.ndarray
    weights: np.ndarray

    @property
    def mean_effective_age(self) -> float:
        return float(np.dot(self.weights, self.candidate_ages))


def temporal_weights(method: str, candidates: np.ndarray, ages: np.ndarray) -> np.ndarray:
    """Return normalized scalar weights for oldest-to-newest candidates."""

    candidates = np.asarray(candidates, dtype=np.float64)
    ages = np.asarray(ages, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[0] != ages.shape[0]:
        raise ValueError("candidates must be (sources, action_dim) with one age per source")
    if len(candidates) == 0:
        raise ValueError("at least one temporal candidate is required")
    if not np.isfinite(candidates).all() or not np.isfinite(ages).all():
        raise ValueError("temporal candidates and ages must be finite")

    if method == "newest":
        weights = np.zeros(len(candidates), dtype=np.float64)
        weights[-1] = 1.0
        return weights
    if method == "exact_act_m001":
        # Pinned LeRobot ACT semantics: index zero is the oldest source.
        logits = -0.01 * np.arange(len(candidates), dtype=np.float64)
    elif method == "newest_age_exp_b003":
        # The frozen LIBERO data index is one physical 20 Hz controller tick.
        logits = -0.03 * ages
    elif method == "cogact_a03":
        newest = candidates[-1]
        denominator = np.linalg.norm(candidates, axis=1) * np.linalg.norm(newest) + 1e-7
        cosine = (candidates @ newest) / denominator
        logits = 0.3 * cosine
    else:
        raise ValueError(f"unknown Gate-3A2 method: {method!r}")

    logits -= np.max(logits)
    weights = np.exp(logits)
    weights /= weights.sum()
    return weights


class DenseTemporalAggregator:
    """Cache one full chunk per controller step and aggregate the current action."""

    def __init__(self, method: str, *, chunk_length: int = 100, action_dim: int = 7) -> None:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        self.method = method
        self.chunk_length = int(chunk_length)
        self.action_dim = int(action_dim)
        self._chunks: deque[tuple[int, np.ndarray]] = deque()

    def reset(self) -> None:
        self._chunks.clear()

    def update(self, source_step: int, chunk: np.ndarray) -> AggregatedAction:
        """Insert the current query and return the action for ``source_step``."""

        source_step = int(source_step)
        chunk = np.asarray(chunk, dtype=np.float64)
        expected_shape = (self.chunk_length, self.action_dim)
        if chunk.shape != expected_shape:
            raise ValueError(f"expected chunk shape {expected_shape}, got {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise ValueError("ACT chunk contains non-finite values")
        if self._chunks and source_step != self._chunks[-1][0] + 1:
            raise ValueError("Gate-3A2 requires exactly one ordered ACT query per controller step")

        self._chunks.append((source_step, chunk.copy()))
        while self._chunks and source_step - self._chunks[0][0] >= self.chunk_length:
            self._chunks.popleft()

        source_steps = np.asarray([step for step, _ in self._chunks], dtype=np.int64)
        ages = source_step - source_steps
        candidates = np.stack(
            [saved_chunk[int(age)] for age, (_, saved_chunk) in zip(ages, self._chunks, strict=True)]
        )
        weights = temporal_weights(self.method, candidates, ages)
        action = weights @ candidates
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise RuntimeError("temporal aggregation produced an invalid action")
        return AggregatedAction(
            action=action.astype(np.float32),
            candidate_count=len(candidates),
            candidate_ages=ages,
            weights=weights,
        )
