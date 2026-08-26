"""Small fixed-commitment executors for full action chunks.

The policy-facing contract is deliberately numeric: a query callback returns a
full ``[chunk_step, action_dim]`` array. Policy observation encoding and
environment-specific action conversion stay in the policy integration layer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class ActionGroup:
    """A semantic physical group and its fixed commitment horizon."""

    name: str
    indices: tuple[int, ...]
    horizon: int


@dataclass(frozen=True)
class ExecutionDecision:
    """One composed environment action and the evidence for its selection."""

    environment_step: int
    action: np.ndarray
    policy_query: bool
    new_chunk_id: int | None
    source_chunk_ids: dict[str, int]
    source_ages: dict[str, int]
    source_positions: dict[str, int]
    remaining_commitments: dict[str, int]
    refreshed_groups: tuple[str, ...]
    configured_horizons: dict[str, int]

    def as_log_record(self) -> dict[str, object]:
        """Return a JSON-serializable record without changing the action."""

        return {
            "environment_step": self.environment_step,
            "policy_query": self.policy_query,
            "new_chunk_id": self.new_chunk_id,
            "source_chunk_ids": dict(self.source_chunk_ids),
            "source_ages": dict(self.source_ages),
            "source_positions": dict(self.source_positions),
            "remaining_commitments": dict(self.remaining_commitments),
            "refreshed_groups": list(self.refreshed_groups),
            "configured_horizons": dict(self.configured_horizons),
            "action": self.action.tolist(),
        }


@dataclass
class _GroupState:
    chunk: np.ndarray
    chunk_id: int
    query_step: int
    position: int
    remaining_commitment: int


class FixedChunkExecutor:
    """Execute either ordinary global or fixed group-wise commitments."""

    def __init__(
        self,
        *,
        strategy: Literal["global_fixed", "groupwise_fixed"],
        action_dim: int,
        chunk_size: int,
        groups: Sequence[ActionGroup],
    ) -> None:
        if strategy not in {"global_fixed", "groupwise_fixed"}:
            raise ValueError(f"unsupported execution strategy: {strategy!r}")
        if action_dim < 1 or chunk_size < 1:
            raise ValueError("action_dim and chunk_size must be positive")
        self.strategy = strategy
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.groups = tuple(groups)
        self._validate_groups()
        self._configured_horizons = {group.name: group.horizon for group in self.groups}
        self._environment_step = 0
        self._next_chunk_id = 0
        self._global_chunk: np.ndarray | None = None
        self._global_position = 0
        self._global_chunk_id: int | None = None
        self._global_query_step: int | None = None
        self._group_states: dict[str, _GroupState] = {}
        if self.strategy == "global_fixed" and len({group.horizon for group in self.groups}) != 1:
            raise ValueError("global_fixed requires one shared horizon")

    @classmethod
    def global_fixed(
        cls,
        *,
        action_dim: int,
        chunk_size: int,
        horizon: int,
        groups: Sequence[ActionGroup] | None = None,
    ) -> "FixedChunkExecutor":
        """Create ordinary fixed-horizon execution with one shared horizon."""

        if groups is None:
            groups = (ActionGroup("all", tuple(range(action_dim)), horizon),)
        else:
            groups = tuple(
                ActionGroup(group.name, group.indices, horizon) for group in groups
            )
        return cls(
            strategy="global_fixed",
            action_dim=action_dim,
            chunk_size=chunk_size,
            groups=groups,
        )

    @classmethod
    def groupwise_fixed(
        cls,
        *,
        action_dim: int,
        chunk_size: int,
        groups: Sequence[ActionGroup],
    ) -> "FixedChunkExecutor":
        """Create fixed execution with an independent horizon per group."""

        return cls(
            strategy="groupwise_fixed",
            action_dim=action_dim,
            chunk_size=chunk_size,
            groups=groups,
        )

    def _validate_groups(self) -> None:
        if not self.groups:
            raise ValueError("at least one action group is required")
        names = [group.name for group in self.groups]
        if len(set(names)) != len(names):
            raise ValueError("action group names must be unique")
        seen: list[int] = []
        for group in self.groups:
            if not group.name:
                raise ValueError("action group names must be non-empty")
            if not group.indices:
                raise ValueError(f"action group {group.name!r} has no indices")
            if group.horizon < 1 or group.horizon > self.chunk_size:
                raise ValueError(
                    f"horizon for {group.name!r} must be within [1, {self.chunk_size}]"
                )
            if any(index < 0 or index >= self.action_dim for index in group.indices):
                raise ValueError(f"indices for {group.name!r} exceed action dimension")
            seen.extend(group.indices)
        if sorted(seen) != list(range(self.action_dim)):
            raise ValueError("action groups must partition every action dimension exactly once")

    def reset(self) -> None:
        """Reset execution state at the start of an environment episode."""

        self._environment_step = 0
        self._next_chunk_id = 0
        self._global_chunk = None
        self._global_position = 0
        self._global_chunk_id = None
        self._global_query_step = None
        self._group_states.clear()

    def _query_chunk(self, query_policy: Callable[[], np.ndarray]) -> tuple[np.ndarray, int]:
        chunk = np.asarray(query_policy())
        expected_shape = (self.chunk_size, self.action_dim)
        if chunk.shape != expected_shape:
            raise ValueError(f"policy chunk must have shape {expected_shape}, got {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise ValueError("policy chunk must contain only finite values")
        chunk_id = self._next_chunk_id
        self._next_chunk_id += 1
        return chunk.copy(), chunk_id

    def _global_step(self, query_policy: Callable[[], np.ndarray]) -> ExecutionDecision:
        queried = self._global_chunk is None or self._global_position == self.groups[0].horizon
        if queried:
            chunk, chunk_id = self._query_chunk(query_policy)
            self._global_chunk = chunk
            self._global_chunk_id = chunk_id
            self._global_query_step = self._environment_step
            self._global_position = 0
        assert self._global_chunk is not None
        assert self._global_chunk_id is not None
        assert self._global_query_step is not None

        action = self._global_chunk[self._global_position].copy()
        source_chunk_ids = {group.name: self._global_chunk_id for group in self.groups}
        source_ages = {
            group.name: self._environment_step - self._global_query_step
            for group in self.groups
        }
        source_positions = {group.name: self._global_position for group in self.groups}
        remaining_commitments = {
            group.name: group.horizon - self._global_position for group in self.groups
        }
        decision = ExecutionDecision(
            environment_step=self._environment_step,
            action=action,
            policy_query=queried,
            new_chunk_id=self._global_chunk_id if queried else None,
            source_chunk_ids=source_chunk_ids,
            source_ages=source_ages,
            source_positions=source_positions,
            remaining_commitments=remaining_commitments,
            refreshed_groups=tuple(group.name for group in self.groups) if queried else (),
            configured_horizons=dict(self._configured_horizons),
        )
        self._global_position += 1
        self._environment_step += 1
        return decision

    def _groupwise_step(self, query_policy: Callable[[], np.ndarray]) -> ExecutionDecision:
        expired = [
            group
            for group in self.groups
            if group.name not in self._group_states
            or self._group_states[group.name].remaining_commitment == 0
        ]
        queried = bool(expired)
        new_chunk_id: int | None = None
        if queried:
            chunk, new_chunk_id = self._query_chunk(query_policy)
            for group in expired:
                self._group_states[group.name] = _GroupState(
                    chunk=chunk,
                    chunk_id=new_chunk_id,
                    query_step=self._environment_step,
                    position=0,
                    remaining_commitment=group.horizon,
                )

        action = np.empty(self.action_dim, dtype=self._group_states[self.groups[0].name].chunk.dtype)
        source_chunk_ids: dict[str, int] = {}
        source_ages: dict[str, int] = {}
        source_positions: dict[str, int] = {}
        remaining_commitments: dict[str, int] = {}
        for group in self.groups:
            state = self._group_states[group.name]
            action[list(group.indices)] = state.chunk[state.position, list(group.indices)]
            source_chunk_ids[group.name] = state.chunk_id
            source_ages[group.name] = self._environment_step - state.query_step
            source_positions[group.name] = state.position
            remaining_commitments[group.name] = state.remaining_commitment

        decision = ExecutionDecision(
            environment_step=self._environment_step,
            action=action,
            policy_query=queried,
            new_chunk_id=new_chunk_id,
            source_chunk_ids=source_chunk_ids,
            source_ages=source_ages,
            source_positions=source_positions,
            remaining_commitments=remaining_commitments,
            refreshed_groups=tuple(group.name for group in expired),
            configured_horizons=dict(self._configured_horizons),
        )
        for state in self._group_states.values():
            state.position += 1
            state.remaining_commitment -= 1
        self._environment_step += 1
        return decision

    def step(self, query_policy: Callable[[], np.ndarray]) -> ExecutionDecision:
        """Compose the next environment action, querying only at commitments."""

        if self.strategy == "global_fixed":
            return self._global_step(query_policy)
        return self._groupwise_step(query_policy)
