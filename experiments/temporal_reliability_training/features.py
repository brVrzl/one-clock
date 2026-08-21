"""Leakage-safe feature extraction for source/group/offset examples."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from .config import FeatureConfig
from .schema import GroupSpec, TemporalExample


STATISTIC_NAMES = (
    "mean",
    "std",
    "min",
    "max",
    "first",
    "last",
    "last_minus_first",
)


def action_chunk_statistics(
    chunk: np.ndarray,
    indices: Sequence[int],
) -> np.ndarray:
    """Return fixed-size per-channel statistics for one action group.

    Statistics are intentionally channelwise.  They are input features, not a
    validity metric, so translation/rotation are not silently collapsed into a
    physical norm here.
    """

    array = np.asarray(chunk, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("chunk must have shape [chunk_step, action_dim]")
    group = array[:, tuple(indices)]
    values = (
        group.mean(axis=0),
        group.std(axis=0),
        group.min(axis=0),
        group.max(axis=0),
        group[0],
        group[-1],
        group[-1] - group[0],
    )
    return np.concatenate(values).astype(np.float32, copy=False)


@dataclass(frozen=True)
class FeatureBatch:
    features: np.ndarray
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        array = np.asarray(self.features, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("features must have shape [example, feature]")
        if not np.isfinite(array).all():
            raise ValueError("features must contain only finite values")
        object.__setattr__(self, "features", array)
        if array.shape[1] != len(self.feature_names):
            raise ValueError("feature_names length must match feature dimension")


class FeatureEncoder:
    """Create estimator inputs using only source-time information."""

    def __init__(self, groups: Sequence[GroupSpec], config: FeatureConfig | None = None):
        if not groups:
            raise ValueError("at least one group is required")
        if len({group.name for group in groups}) != len(groups):
            raise ValueError("group names must be unique")
        self.groups = tuple(groups)
        self.config = config or FeatureConfig()
        self.config.validate()
        self._group_by_name = {group.name: group for group in self.groups}

        names: list[str] = []
        names.extend(
            f"observation_embedding_{index}"
            for index in range(self.config.observation_embedding_dim)
        )
        names.extend(
            f"chunk_{statistic}_channel_{channel}"
            for statistic in STATISTIC_NAMES
            for channel in range(max(len(group.indices) for group in self.groups))
        )
        # The channelwise statistics are padded to the largest group so that
        # every group shares one input layout.
        self._max_group_channels = max(len(group.indices) for group in self.groups)
        if self.config.include_group_one_hot:
            names.extend(f"group_one_hot_{group.name}" for group in self.groups)
        if self.config.include_offset:
            names.extend(("offset", "offset_normalized"))
        self.feature_names = tuple(names)

    @property
    def input_dim(self) -> int:
        return len(self.feature_names)

    def _encode_observation(self, example: TemporalExample) -> np.ndarray:
        dimension = self.config.observation_embedding_dim
        if dimension == 0:
            return np.empty(0, dtype=np.float32)
        if example.source_observation_embedding is None:
            return np.zeros(dimension, dtype=np.float32)
        embedding = np.asarray(example.source_observation_embedding, dtype=np.float32)
        if embedding.shape != (dimension,):
            raise ValueError(
                "source observation embedding shape does not match feature config: "
                f"expected {(dimension,)}, got {embedding.shape}"
            )
        return embedding.copy()

    def encode(self, example: TemporalExample) -> np.ndarray:
        if example.group not in self._group_by_name:
            raise KeyError(f"unknown example group: {example.group!r}")
        group = self._group_by_name[example.group]
        values: list[np.ndarray] = [self._encode_observation(example)]
        stats = action_chunk_statistics(example.source_chunk, group.indices)
        channels = len(group.indices)
        padded = np.zeros(len(STATISTIC_NAMES) * self._max_group_channels, dtype=np.float32)
        for statistic_index in range(len(STATISTIC_NAMES)):
            start = statistic_index * self._max_group_channels
            source_start = statistic_index * channels
            padded[start : start + channels] = stats[source_start : source_start + channels]
        values.append(padded)
        if self.config.include_group_one_hot:
            one_hot = np.zeros(len(self.groups), dtype=np.float32)
            one_hot[self.groups.index(group)] = 1.0
            values.append(one_hot)
        if self.config.include_offset:
            if example.offset > self.config.max_offset:
                raise ValueError("example offset exceeds configured max_offset")
            values.append(
                np.asarray(
                    [example.offset, example.offset / self.config.max_offset],
                    dtype=np.float32,
                )
            )
        result = np.concatenate(values).astype(np.float32, copy=False)
        if result.shape != (self.input_dim,):
            raise AssertionError("feature layout produced an unexpected dimension")
        return result

    def encode_many(self, examples: Sequence[TemporalExample]) -> FeatureBatch:
        if not examples:
            matrix = np.empty((0, self.input_dim), dtype=np.float32)
        else:
            matrix = np.vstack([self.encode(example) for example in examples])
        return FeatureBatch(matrix, self.feature_names)
