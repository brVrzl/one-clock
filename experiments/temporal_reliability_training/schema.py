"""Typed records exchanged by the temporal reliability preparation stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GroupSpec:
    """A named, non-overlapping slice of the action vector."""

    name: str
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("group name must be non-empty")
        if not self.indices:
            raise ValueError(f"group {self.name!r} must contain an index")
        if any(index < 0 for index in self.indices):
            raise ValueError(f"group {self.name!r} contains a negative index")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError(f"group {self.name!r} contains duplicate indices")


DEFAULT_LIBERO_GROUPS = (
    GroupSpec("arm", tuple(range(6))),
    GroupSpec("gripper", (6,)),
)


def _as_float_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class FrozenTrajectory:
    """One episode's source-time observations and frozen-policy predictions.

    ``policy_chunks`` contains already-materialized ACT predictions keyed by
    the observation step used to query ACT.  Creating those predictions is an
    upstream frozen-inference step; this package only consumes them.  Keeping
    the map explicit allows a target to use a fresh prediction at ``s + k``
    without accidentally treating a future observation as a model feature.
    """

    episode_id: str | int
    task_id: str | int | None
    policy_chunks: Mapping[int, np.ndarray]
    demonstrated_actions: np.ndarray | None = None
    observation_embeddings: np.ndarray | None = None
    source_steps: Sequence[int] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.policy_chunks:
            raise ValueError("at least one frozen policy chunk is required")

        normalized_chunks: dict[int, np.ndarray] = {}
        chunk_shape: tuple[int, int] | None = None
        for raw_step, raw_chunk in self.policy_chunks.items():
            step = int(raw_step)
            if step < 0 or step != raw_step:
                raise ValueError("policy chunk steps must be non-negative integers")
            chunk = _as_float_array(raw_chunk, name=f"policy_chunks[{step}]", ndim=2)
            if chunk.shape[0] < 1 or chunk.shape[1] < 1:
                raise ValueError("policy chunks must have positive dimensions")
            if chunk_shape is None:
                chunk_shape = tuple(chunk.shape)
            elif tuple(chunk.shape) != chunk_shape:
                raise ValueError("all policy chunks must have the same shape")
            normalized_chunks[step] = chunk.copy()
        object.__setattr__(self, "policy_chunks", normalized_chunks)

        if self.source_steps is None:
            source_steps = tuple(sorted(normalized_chunks))
        else:
            source_steps = tuple(int(step) for step in self.source_steps)
            if len(set(source_steps)) != len(source_steps):
                raise ValueError("source_steps must be unique")
            if any(step not in normalized_chunks for step in source_steps):
                raise ValueError("every source step must have a policy chunk")
            if any(step < 0 for step in source_steps):
                raise ValueError("source_steps must be non-negative")
        if not source_steps:
            raise ValueError("at least one source step is required")
        object.__setattr__(self, "source_steps", source_steps)

        if self.demonstrated_actions is not None:
            actions = _as_float_array(
                self.demonstrated_actions, name="demonstrated_actions", ndim=2
            )
            if actions.shape[1] != chunk_shape[1]:
                raise ValueError("demonstrated action dimension must match chunks")
            object.__setattr__(self, "demonstrated_actions", actions.copy())

        if self.observation_embeddings is not None:
            embeddings = _as_float_array(
                self.observation_embeddings,
                name="observation_embeddings",
                ndim=2,
            )
            if embeddings.shape[0] == 0:
                raise ValueError("observation_embeddings must not be empty")
            max_source = max(source_steps)
            if max_source >= embeddings.shape[0]:
                raise ValueError(
                    "observation_embeddings must contain every source step"
                )
            object.__setattr__(self, "observation_embeddings", embeddings.copy())

    @property
    def action_dim(self) -> int:
        return next(iter(self.policy_chunks.values())).shape[1]

    @property
    def chunk_size(self) -> int:
        return next(iter(self.policy_chunks.values())).shape[0]

    def observation_embedding_at(self, step: int) -> np.ndarray | None:
        if self.observation_embeddings is None:
            return None
        if step < 0 or step >= self.observation_embeddings.shape[0]:
            raise KeyError(f"no observation embedding for step {step}")
        return self.observation_embeddings[step].copy()


@dataclass(frozen=True)
class TemporalExample:
    """One source/group/offset example.

    Future fields are intentionally retained for target generation only.  The
    feature encoder reads only ``source_chunk`` and
    ``source_observation_embedding``.
    """

    episode_id: str | int
    task_id: str | int | None
    source_step: int
    future_step: int
    offset: int
    group: str
    source_chunk: np.ndarray
    source_observation_embedding: np.ndarray | None
    future_policy_chunk: np.ndarray | None
    future_demonstrated_action: np.ndarray | None

    def __post_init__(self) -> None:
        chunk = _as_float_array(self.source_chunk, name="source_chunk", ndim=2)
        if self.offset < 0 or self.offset >= chunk.shape[0]:
            raise ValueError("offset must be within source_chunk")
        if self.source_step < 0 or self.future_step < self.source_step:
            raise ValueError("example steps are invalid")
        object.__setattr__(self, "source_chunk", chunk.copy())

        if self.source_observation_embedding is not None:
            embedding = _as_float_array(
                self.source_observation_embedding,
                name="source_observation_embedding",
                ndim=1,
            )
            object.__setattr__(self, "source_observation_embedding", embedding.copy())

        for field_name in ("future_policy_chunk", "future_demonstrated_action"):
            value = getattr(self, field_name)
            if value is not None:
                expected_ndim = 2 if field_name == "future_policy_chunk" else 1
                array = _as_float_array(value, name=field_name, ndim=expected_ndim)
                if array.shape[-1] != chunk.shape[1]:
                    raise ValueError(f"{field_name} action dimension must match chunk")
                object.__setattr__(self, field_name, array.copy())

    @property
    def source_action(self) -> np.ndarray:
        return self.source_chunk[self.offset].copy()
