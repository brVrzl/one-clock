"""Causal source-time feature contract for vector reliability prediction."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from experiments.temporal_reliability_training.features import (
    STATISTIC_NAMES,
    action_chunk_statistics,
)
from experiments.temporal_reliability_training.schema import GroupSpec, TemporalExample


@dataclass(frozen=True)
class CausalFeatureContract:
    """Allowed input layout; future and terminal fields have no API slot."""

    observation_embedding_dim: int = 16
    groups: tuple[GroupSpec, ...] = ()

    @classmethod
    def for_groups(
        cls,
        groups: Sequence[GroupSpec],
        *,
        observation_embedding_dim: int = 16,
    ) -> "CausalFeatureContract":
        contract = cls(observation_embedding_dim, tuple(groups))
        contract.validate()
        return contract

    def validate(self) -> None:
        if self.observation_embedding_dim < 0:
            raise ValueError("observation_embedding_dim must be non-negative")
        if not self.groups:
            raise ValueError("at least one group is required")
        if len({group.name for group in self.groups}) != len(self.groups):
            raise ValueError("group names must be unique")

    @property
    def max_group_width(self) -> int:
        self.validate()
        return max(len(group.indices) for group in self.groups)

    @property
    def feature_names(self) -> tuple[str, ...]:
        self.validate()
        names = [
            *(f"observation_embedding_{index}" for index in range(self.observation_embedding_dim)),
            *(
                f"chunk_{statistic}_channel_{channel}"
                for statistic in STATISTIC_NAMES
                for channel in range(self.max_group_width)
            ),
            *(f"group_embedding_{group.name}" for group in self.groups),
        ]
        return tuple(names)

    @property
    def input_dim(self) -> int:
        return len(self.feature_names)

    def encode(
        self,
        *,
        observation_embedding: np.ndarray | None,
        action_chunk: np.ndarray,
        group: str,
    ) -> np.ndarray:
        """Encode only current observation/history, chunk, and group ID.

        ``observation_embedding`` is the caller's source-time representation of
        current observation/history. This function intentionally accepts no
        offset, future observation, future action, episode length, phase, or
        terminal metadata.
        """

        self.validate()
        group_by_name = {item.name: item for item in self.groups}
        if group not in group_by_name:
            raise KeyError(f"unknown group: {group!r}")
        if observation_embedding is None:
            observation = np.zeros(self.observation_embedding_dim, dtype=np.float32)
        else:
            observation = np.asarray(observation_embedding, dtype=np.float32)
            if observation.shape != (self.observation_embedding_dim,):
                raise ValueError(
                    "observation_embedding shape does not match contract: "
                    f"expected {(self.observation_embedding_dim,)}, got {observation.shape}"
                )
        spec = group_by_name[group]
        stats = action_chunk_statistics(np.asarray(action_chunk), spec.indices)
        padded = np.zeros(len(STATISTIC_NAMES) * self.max_group_width, dtype=np.float32)
        width = len(spec.indices)
        for statistic_index in range(len(STATISTIC_NAMES)):
            padded[statistic_index * self.max_group_width : statistic_index * self.max_group_width + width] = stats[
                statistic_index * width : (statistic_index + 1) * width
            ]
        group_embedding = np.zeros(len(self.groups), dtype=np.float32)
        group_embedding[self.groups.index(spec)] = 1.0
        encoded = np.concatenate((observation, padded, group_embedding)).astype(np.float32, copy=False)
        if encoded.shape != (self.input_dim,) or not np.isfinite(encoded).all():
            raise AssertionError("causal feature contract produced an invalid vector")
        return encoded

    def encode_example(self, example: TemporalExample) -> np.ndarray:
        return self.encode(
            observation_embedding=example.source_observation_embedding,
            action_chunk=example.source_chunk,
            group=example.group,
        )
