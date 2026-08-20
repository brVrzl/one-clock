"""Offline prior and empirical reliability-curve baselines."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np


def constant_prior_scores(train_labels: Sequence[int], count: int) -> np.ndarray:
    labels = np.asarray(train_labels, dtype=np.int64)
    if labels.ndim != 1 or labels.size == 0 or not np.isin(labels, [0, 1]).all():
        raise ValueError("train_labels must be a non-empty binary vector")
    if count < 1:
        raise ValueError("count must be positive")
    return np.full(count, float(labels.mean()), dtype=np.float64)


class EmpiricalReliabilityPredictor:
    """Predict train-set P(Y=1 | group, offset), with transparent fallbacks."""

    def __init__(self, *, smoothing: float = 0.0) -> None:
        if smoothing < 0.0:
            raise ValueError("smoothing must be non-negative")
        self.smoothing = float(smoothing)
        self._global = 0.0
        self._group: dict[str, float] = {}
        self._cell: dict[tuple[str, int], float] = {}

    @staticmethod
    def _rate(positive: int, total: int, smoothing: float) -> float:
        if total == 0:
            return float("nan")
        if smoothing == 0.0:
            return positive / total
        return (positive + smoothing * 0.5) / (total + smoothing)

    def fit(self, groups: Sequence[str], offsets: Sequence[int], labels: Sequence[int]) -> "EmpiricalReliabilityPredictor":
        group_array = np.asarray(groups).astype(str)
        offset_array = np.asarray(offsets, dtype=np.int64)
        label_array = np.asarray(labels, dtype=np.int64)
        if group_array.ndim != 1 or offset_array.shape != group_array.shape or label_array.shape != group_array.shape:
            raise ValueError("groups, offsets, and labels must have matching vectors")
        if label_array.size == 0 or not np.isin(label_array, [0, 1]).all():
            raise ValueError("labels must be non-empty and binary")
        self._global = self._rate(int(label_array.sum()), label_array.size, self.smoothing)
        for group in sorted(set(group_array)):
            selected = group_array == group
            self._group[group] = self._rate(int(label_array[selected].sum()), int(selected.sum()), self.smoothing)
        for group in sorted(set(group_array)):
            for offset in sorted(set(offset_array)):
                selected = (group_array == group) & (offset_array == offset)
                if selected.any():
                    self._cell[(group, int(offset))] = self._rate(
                        int(label_array[selected].sum()), int(selected.sum()), self.smoothing
                    )
        return self

    def predict(self, groups: Sequence[str], offsets: Sequence[int]) -> np.ndarray:
        group_array = np.asarray(groups).astype(str)
        offset_array = np.asarray(offsets, dtype=np.int64)
        if group_array.shape != offset_array.shape:
            raise ValueError("groups and offsets must have matching shapes")
        values = [
            self._cell.get(
                (str(group), int(offset)),
                self._group.get(str(group), self._global),
            )
            for group, offset in zip(group_array, offset_array)
        ]
        return np.asarray(values, dtype=np.float64)
