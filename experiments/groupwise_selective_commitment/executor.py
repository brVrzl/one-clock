"""Scheduled matched-query executors for the selective-commitment gate.

This module is intentionally separate from :mod:`one_clock.executor`.  The
audited fixed-commitment executor remains unchanged; this experiment needs a
fixed global query cadence and a causal accept/retain decision at each query.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np


METHODS = ("global_replace", "selective_commit")
GROUP_NAMES = ("arm", "gripper")
EPSILON = {"arm": 1.0, "gripper": 1.0}


@dataclass(frozen=True)
class CommitGroup:
    name: str
    indices: tuple[int, ...]


@dataclass(frozen=True)
class ScheduledDecision:
    """One emitted action plus all causal executor evidence."""

    environment_step: int
    policy_query_id: int | None
    policy_query: bool
    fresh_source_generation_id: int | None
    current_source_generation_ids: dict[str, int]
    local_source_ages: dict[str, int]
    source_positions: dict[str, int]
    acceptance: dict[str, str]
    distances: dict[str, float | None]
    epsilons: dict[str, float]
    source_exhausted: dict[str, bool]
    action: np.ndarray

    def as_log_record(self) -> dict[str, object]:
        return {
            "global_timestep": self.environment_step,
            "policy_query_id": self.policy_query_id,
            "query_occurred": self.policy_query,
            "fresh_source_generation_id": self.fresh_source_generation_id,
            "current_source_generation_id": dict(self.current_source_generation_ids),
            "local_source_age": dict(self.local_source_ages),
            "source_position": dict(self.source_positions),
            "acceptance": dict(self.acceptance),
            "d_g": dict(self.distances),
            "epsilon_g": dict(self.epsilons),
            "source_chunk_exhausted": dict(self.source_exhausted),
            "final_emitted_action": self.action.tolist(),
        }


@dataclass
class _GroupState:
    chunk: np.ndarray
    generation_id: int
    query_step: int
    position: int


def _sign(value: float) -> int:
    return 1 if value >= 0.0 else -1


def normalized_current_distance(
    old_current: np.ndarray,
    fresh_current: np.ndarray,
    *,
    group: str,
    action_std: np.ndarray,
) -> float:
    """Return the predeclared current-query distance for one group.

    Arm distance is the maximum of the audited translation and rotation
    dataset-standard-deviation RMS values.  Gripper distance is the audited
    normalized absolute error, with a sign mismatch promoted above epsilon so
    that the existing normalized-error-plus-sign validity rule is preserved.
    """

    old = np.asarray(old_current, dtype=np.float64)
    fresh = np.asarray(fresh_current, dtype=np.float64)
    if old.shape != fresh.shape:
        raise ValueError("old and fresh group actions must have equal shape")
    std = np.asarray(action_std, dtype=np.float64)
    if std.shape != (7,) or not np.isfinite(std).all() or np.any(std <= 0.0):
        raise ValueError("action_std must be a finite positive 7-vector")
    delta = fresh - old
    if group == "arm":
        if old.shape != (6,):
            raise ValueError("arm action must have six dimensions")
        translation = float(np.sqrt(np.mean((delta[:3] / std[:3]) ** 2)))
        rotation = float(np.sqrt(np.mean((delta[3:6] / std[3:6]) ** 2)))
        return max(translation, rotation)
    if group == "gripper":
        if old.shape != (1,):
            raise ValueError("gripper action must have one dimension")
        normalized = float(abs(delta[0]) / std[6])
        if _sign(float(old[0])) != _sign(float(fresh[0])):
            return max(normalized, float(np.nextafter(1.0, np.inf)))
        return normalized
    raise ValueError(f"unsupported group: {group!r}")


class ScheduledCommitExecutor:
    """Execute fixed-cadence full queries with global or selective commits."""

    def __init__(
        self,
        *,
        method: Literal["global_replace", "selective_commit"],
        query_cadence: int,
        chunk_size: int,
        action_dim: int,
        groups: Sequence[CommitGroup],
        action_std: np.ndarray,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"unsupported method: {method!r}")
        if query_cadence < 1:
            raise ValueError("query_cadence must be positive")
        if chunk_size < 1 or action_dim < 1:
            raise ValueError("chunk_size and action_dim must be positive")
        self.method = method
        self.query_cadence = int(query_cadence)
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.groups = tuple(groups)
        self.action_std = np.asarray(action_std, dtype=np.float64).copy()
        self._validate_groups()
        if self.action_std.shape != (self.action_dim,):
            raise ValueError("action_std shape must equal action_dim")
        if not np.isfinite(self.action_std).all() or np.any(self.action_std <= 0.0):
            raise ValueError("action_std must be finite and positive")
        self.reset()

    def _validate_groups(self) -> None:
        if not self.groups:
            raise ValueError("at least one action group is required")
        names = [group.name for group in self.groups]
        if len(set(names)) != len(names):
            raise ValueError("group names must be unique")
        seen: list[int] = []
        for group in self.groups:
            if not group.name or not group.indices:
                raise ValueError("groups require a name and at least one index")
            if any(index < 0 or index >= self.action_dim for index in group.indices):
                raise ValueError("group indices exceed action dimension")
            seen.extend(group.indices)
        if sorted(seen) != list(range(self.action_dim)):
            raise ValueError("groups must partition action dimensions exactly once")

    def reset(self) -> None:
        self.environment_step = 0
        self._next_query_id = 0
        self._next_generation_id = 0
        self._states: dict[str, _GroupState] = {}

    def _query(self, query_policy: Callable[[], np.ndarray]) -> tuple[np.ndarray, int, int]:
        chunk = np.asarray(query_policy(), dtype=np.float32)
        expected = (self.chunk_size, self.action_dim)
        if chunk.shape != expected:
            raise ValueError(f"policy chunk must have shape {expected}, got {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise ValueError("policy chunk must contain only finite values")
        query_id = self._next_query_id
        generation_id = self._next_generation_id
        self._next_query_id += 1
        self._next_generation_id += 1
        return chunk.copy(), query_id, generation_id

    def _current_group_action(self, state: _GroupState, group: CommitGroup) -> tuple[np.ndarray, bool, int]:
        exhausted = state.position >= self.chunk_size
        position = min(state.position, self.chunk_size - 1)
        return state.chunk[position, list(group.indices)].copy(), exhausted, position

    def step(self, query_policy: Callable[[], np.ndarray]) -> ScheduledDecision:
        scheduled_query = self.environment_step % self.query_cadence == 0
        policy_query_id: int | None = None
        fresh_generation_id: int | None = None
        fresh_chunk: np.ndarray | None = None
        if scheduled_query:
            fresh_chunk, policy_query_id, fresh_generation_id = self._query(query_policy)

        distances: dict[str, float | None] = {}
        acceptance: dict[str, str] = {}
        exhausted: dict[str, bool] = {}
        if scheduled_query:
            assert fresh_chunk is not None
            for group in self.groups:
                state = self._states.get(group.name)
                if state is None:
                    distances[group.name] = None
                    acceptance[group.name] = "accept"
                    continue
                old_current, is_exhausted, _ = self._current_group_action(state, group)
                fresh_current = fresh_chunk[0, list(group.indices)]
                distance = normalized_current_distance(
                    old_current,
                    fresh_current,
                    group=group.name,
                    action_std=self.action_std,
                )
                distances[group.name] = distance
                exhausted[group.name] = is_exhausted
                if self.method == "global_replace" or is_exhausted or distance > EPSILON[group.name]:
                    acceptance[group.name] = "accept"
                else:
                    acceptance[group.name] = "retain"
        else:
            for group in self.groups:
                distances[group.name] = None
                acceptance[group.name] = "none"

        assert fresh_chunk is None or fresh_generation_id is not None
        if scheduled_query:
            assert fresh_chunk is not None and fresh_generation_id is not None
            for group in self.groups:
                if acceptance[group.name] == "accept":
                    self._states[group.name] = _GroupState(
                        chunk=fresh_chunk,
                        generation_id=fresh_generation_id,
                        query_step=self.environment_step,
                        position=0,
                    )

        action = np.empty(self.action_dim, dtype=np.float32)
        source_generation_ids: dict[str, int] = {}
        local_ages: dict[str, int] = {}
        source_positions: dict[str, int] = {}
        for group in self.groups:
            state = self._states[group.name]
            group_action, is_exhausted, position = self._current_group_action(state, group)
            action[list(group.indices)] = group_action
            source_generation_ids[group.name] = state.generation_id
            local_ages[group.name] = self.environment_step - state.query_step
            source_positions[group.name] = position
            exhausted[group.name] = bool(is_exhausted)

        decision = ScheduledDecision(
            environment_step=self.environment_step,
            policy_query_id=policy_query_id,
            policy_query=scheduled_query,
            fresh_source_generation_id=fresh_generation_id,
            current_source_generation_ids=source_generation_ids,
            local_source_ages=local_ages,
            source_positions=source_positions,
            acceptance=acceptance,
            distances=distances,
            epsilons=dict(EPSILON),
            source_exhausted=exhausted,
            action=action,
        )
        for state in self._states.values():
            state.position += 1
        self.environment_step += 1
        return decision
