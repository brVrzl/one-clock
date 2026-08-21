"""Configurable temporal validity targets.

The target is binary validity at a future offset, while the underlying loss is
always retained.  Thresholds are deliberately external configuration: this
module does not choose a final arm or gripper threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from .config import TargetConfig
from .schema import GroupSpec, TemporalExample


ErrorMetric = Callable[[np.ndarray, np.ndarray, str], float]
ValidityFunction = Callable[[float, str, TemporalExample], bool]


def groupwise_rms_error(
    source_action: np.ndarray,
    reference_action: np.ndarray,
    group: str,
) -> float:
    """A simple scale-sensitive metric for a group slice.

    The group name is accepted so callers can replace this with group-specific
    normalization without changing the target interface.  No translation /
    rotation weighting or gripper threshold is embedded here.
    """

    del group
    source = np.asarray(source_action, dtype=np.float64)
    reference = np.asarray(reference_action, dtype=np.float64)
    if source.shape != reference.shape:
        raise ValueError("source and reference actions must have matching shapes")
    difference = source - reference
    return float(np.sqrt(np.mean(np.square(difference))))


@dataclass(frozen=True)
class TargetBatch:
    losses: np.ndarray
    labels: np.ndarray | None

    def __post_init__(self) -> None:
        losses = np.asarray(self.losses, dtype=np.float32)
        if losses.ndim != 1 or not np.isfinite(losses).all():
            raise ValueError("losses must be a finite one-dimensional array")
        object.__setattr__(self, "losses", losses)
        if self.labels is not None:
            labels = np.asarray(self.labels, dtype=np.int64)
            if labels.shape != losses.shape or not np.isin(labels, [0, 1]).all():
                raise ValueError("labels must be binary and match losses")
            object.__setattr__(self, "labels", labels)


class TemporalValidityTarget:
    """Generate ``Y_g(k)`` from frozen-policy or demonstration references."""

    def __init__(
        self,
        groups: Sequence[GroupSpec],
        config: TargetConfig | None = None,
        *,
        metric: ErrorMetric = groupwise_rms_error,
    ) -> None:
        if not groups:
            raise ValueError("at least one group is required")
        if len({group.name for group in groups}) != len(groups):
            raise ValueError("group names must be unique")
        self.groups = tuple(groups)
        self._group_by_name = {group.name: group for group in groups}
        self.config = config or TargetConfig()
        self.config.validate()
        self.metric = metric

    def _reference_action(self, example: TemporalExample) -> np.ndarray:
        if self.config.mode == "fresh_policy":
            if example.future_policy_chunk is None:
                raise ValueError("fresh-policy target requires a future policy chunk")
            return example.future_policy_chunk[0]
        if example.future_demonstrated_action is None:
            raise ValueError("demonstration target requires a future action")
        return example.future_demonstrated_action

    def loss(self, example: TemporalExample) -> float:
        group = self._group_by_name.get(example.group)
        if group is None:
            raise KeyError(f"unknown example group: {example.group!r}")
        source = example.source_chunk[example.offset, list(group.indices)]
        reference = self._reference_action(example)[list(group.indices)]
        value = float(self.metric(source, reference, group.name))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("target metric must return a finite non-negative value")
        return value

    def generate(
        self,
        examples: Sequence[TemporalExample],
        *,
        validity_fn: ValidityFunction | None = None,
        threshold_by_group: Mapping[str, float] | None = None,
    ) -> TargetBatch:
        """Return losses and optional labels for examples.

        ``validity_fn`` is the most flexible interface.  Otherwise thresholds
        are read first from the explicit call and then from ``TargetConfig``.
        If neither is present, labels are ``None`` rather than silently using a
        guessed threshold.
        """

        losses = np.asarray([self.loss(example) for example in examples], dtype=np.float32)
        if validity_fn is not None and threshold_by_group is not None:
            raise ValueError("provide validity_fn or threshold_by_group, not both")
        thresholds = (
            dict(threshold_by_group)
            if threshold_by_group is not None
            else self.config.threshold_by_group
        )
        if validity_fn is None and thresholds is None:
            return TargetBatch(losses=losses, labels=None)

        labels: list[int] = []
        for loss_value, example in zip(losses, examples):
            if validity_fn is not None:
                valid = validity_fn(float(loss_value), example.group, example)
            else:
                assert thresholds is not None
                if example.group not in thresholds:
                    raise KeyError(f"no validity threshold for group {example.group!r}")
                threshold = float(thresholds[example.group])
                if threshold < 0.0:
                    raise ValueError("target thresholds must be non-negative")
                valid = (
                    float(loss_value) <= threshold
                    if self.config.inclusive
                    else float(loss_value) < threshold
                )
            labels.append(int(bool(valid)))
        return TargetBatch(losses=losses, labels=np.asarray(labels, dtype=np.int64))
