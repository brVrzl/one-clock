"""Rollout-facing adaptive groupwise execution adapter.

The existing fixed executor is intentionally immutable.  This adapter keeps
the same action-group/source-age semantics and returns the existing
``ExecutionDecision`` record type, while obtaining a new horizon for expired
groups after each fresh frozen-policy query.  It is not wired into benchmark
scripts; that integration remains gated on offline validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from one_clock import ActionGroup, ExecutionDecision, FixedChunkExecutor


@dataclass
class _AdaptiveGroupState:
    chunk: np.ndarray
    chunk_id: int
    query_step: int
    position: int
    remaining: int
    horizon: int


class AdaptiveGroupwiseExecutor:
    """Apply scheduler-generated horizons without changing action semantics."""

    def __init__(
        self,
        *,
        action_dim: int,
        chunk_size: int,
        groups: Sequence[ActionGroup],
        scheduler: object,
    ) -> None:
        if action_dim < 1 or chunk_size < 1:
            raise ValueError("action_dim and chunk_size must be positive")
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.groups = tuple(groups)
        self.scheduler = scheduler
        self._validate_groups()
        self.reset()

    def _validate_groups(self) -> None:
        if not self.groups:
            raise ValueError("at least one group is required")
        names = [group.name for group in self.groups]
        if len(set(names)) != len(names):
            raise ValueError("group names must be unique")
        if any(not group.name for group in self.groups):
            raise ValueError("group names must be non-empty")
        if any(not group.indices for group in self.groups):
            raise ValueError("every group must contain at least one action index")
        indices = [index for group in self.groups for index in group.indices]
        if sorted(indices) != list(range(self.action_dim)):
            raise ValueError("groups must partition every action dimension exactly once")

    def reset(self) -> None:
        self._environment_step = 0
        self._next_chunk_id = 0
        self._states: dict[str, _AdaptiveGroupState] = {}

    def _query_chunk(self, query_policy: Callable[[], np.ndarray]) -> tuple[np.ndarray, int]:
        chunk = np.asarray(query_policy(), dtype=np.float32)
        expected = (self.chunk_size, self.action_dim)
        if chunk.shape != expected:
            raise ValueError(f"policy chunk must have shape {expected}, got {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise ValueError("policy chunk must contain only finite values")
        chunk_id = self._next_chunk_id
        self._next_chunk_id += 1
        return chunk.copy(), chunk_id

    def _get_horizons(
        self,
        observation_embedding: np.ndarray | None,
        chunk: np.ndarray,
    ) -> dict[str, int]:
        predict = getattr(self.scheduler, "predict_horizons", None)
        if predict is None:
            raise TypeError("scheduler must provide predict_horizons(observation_embedding, chunk)")
        raw = predict(observation_embedding, chunk)
        horizons = {str(name): int(value) for name, value in dict(raw).items()}
        expected_names = {group.name for group in self.groups}
        if set(horizons) != expected_names:
            raise ValueError("scheduler horizons must contain exactly the configured groups")
        if any(value < 1 or value > self.chunk_size for value in horizons.values()):
            raise ValueError("scheduler horizons must be within [1, chunk_size]")
        return horizons

    def step(
        self,
        query_policy: Callable[[], np.ndarray],
        *,
        observation_embedding: np.ndarray | None = None,
    ) -> ExecutionDecision:
        expired = [
            group
            for group in self.groups
            if group.name not in self._states or self._states[group.name].remaining == 0
        ]
        queried = bool(expired)
        new_chunk_id: int | None = None
        horizons: dict[str, int] = {
            group.name: self._states[group.name].horizon
            for group in self.groups
            if group.name in self._states
        }
        if queried:
            chunk, new_chunk_id = self._query_chunk(query_policy)
            horizons.update(self._get_horizons(observation_embedding, chunk))
            for group in expired:
                horizon = horizons[group.name]
                self._states[group.name] = _AdaptiveGroupState(
                    chunk=chunk,
                    chunk_id=new_chunk_id,
                    query_step=self._environment_step,
                    position=0,
                    remaining=horizon,
                    horizon=horizon,
                )

        if not self._states:
            raise RuntimeError("adaptive executor has no active group states")
        dtype = self._states[self.groups[0].name].chunk.dtype
        action = np.empty(self.action_dim, dtype=dtype)
        source_chunk_ids: dict[str, int] = {}
        source_ages: dict[str, int] = {}
        source_positions: dict[str, int] = {}
        remaining: dict[str, int] = {}
        configured_horizons: dict[str, int] = {}
        for group in self.groups:
            state = self._states[group.name]
            action[list(group.indices)] = state.chunk[state.position, list(group.indices)]
            source_chunk_ids[group.name] = state.chunk_id
            source_ages[group.name] = self._environment_step - state.query_step
            source_positions[group.name] = state.position
            remaining[group.name] = state.remaining
            configured_horizons[group.name] = state.horizon

        decision = ExecutionDecision(
            environment_step=self._environment_step,
            action=action,
            policy_query=queried,
            new_chunk_id=new_chunk_id,
            source_chunk_ids=source_chunk_ids,
            source_ages=source_ages,
            source_positions=source_positions,
            remaining_commitments=remaining,
            refreshed_groups=tuple(group.name for group in expired),
            configured_horizons=configured_horizons,
        )
        for state in self._states.values():
            state.position += 1
            state.remaining -= 1
        self._environment_step += 1
        return decision


def make_static_groupwise_executor(
    *,
    action_dim: int,
    chunk_size: int,
    groups: Sequence[ActionGroup],
) -> FixedChunkExecutor:
    """Use the existing fixed executor for the static groupwise baseline."""

    return FixedChunkExecutor.groupwise_fixed(
        action_dim=action_dim,
        chunk_size=chunk_size,
        groups=groups,
    )
