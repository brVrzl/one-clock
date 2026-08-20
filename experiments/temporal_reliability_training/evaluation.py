"""Offline reliability and calibration metrics without a training dependency."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np


def _validate_binary_inputs(labels: Sequence[int], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape:
        raise ValueError("labels and scores must be matching one-dimensional arrays")
    if y.size == 0:
        raise ValueError("at least one prediction is required")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("scores must be finite probabilities in [0, 1]")
    return y, p


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute AUROC with average ranks for ties; return NaN if undefined."""

    y, p = _validate_binary_inputs(labels, scores)
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(p, kind="mergesort")
    sorted_scores = p[order]
    ranks = np.empty(y.size, dtype=np.float64)
    start = 0
    while start < y.size:
        end = start + 1
        while end < y.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[y == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


auroc = roc_auc


@dataclass(frozen=True)
class ReliabilityCurve:
    bin_edges: np.ndarray
    mean_score: np.ndarray
    fraction_valid: np.ndarray
    count: np.ndarray

    def as_dict(self) -> dict[str, list[float] | list[int]]:
        return {
            "bin_edges": self.bin_edges.tolist(),
            "mean_score": self.mean_score.tolist(),
            "fraction_valid": self.fraction_valid.tolist(),
            "count": self.count.tolist(),
        }


def reliability_curve(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    n_bins: int = 10,
) -> ReliabilityCurve:
    """Return equal-width probability bins for a reliability diagram."""

    y, p = _validate_binary_inputs(labels, scores)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.minimum(np.floor(p * n_bins).astype(int), n_bins - 1)
    mean_score = np.full(n_bins, np.nan, dtype=np.float64)
    fraction_valid = np.full(n_bins, np.nan, dtype=np.float64)
    count = np.zeros(n_bins, dtype=np.int64)
    for bin_id in range(n_bins):
        selected = bin_ids == bin_id
        count[bin_id] = int(selected.sum())
        if count[bin_id]:
            mean_score[bin_id] = float(p[selected].mean())
            fraction_valid[bin_id] = float(y[selected].mean())
    return ReliabilityCurve(edges, mean_score, fraction_valid, count)


def expected_calibration_error(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    n_bins: int = 10,
) -> float:
    """Compute count-weighted expected calibration error."""

    curve = reliability_curve(labels, scores, n_bins=n_bins)
    total = int(curve.count.sum())
    occupied = curve.count > 0
    return float(
        np.sum(
            curve.count[occupied]
            * np.abs(curve.mean_score[occupied] - curve.fraction_valid[occupied])
        )
        / total
    )


def brier_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute the mean squared probability error."""

    y, p = _validate_binary_inputs(labels, scores)
    return float(np.mean(np.square(p - y)))


calibration_error = expected_calibration_error


@dataclass(frozen=True)
class EvaluationResult:
    auroc: float
    brier_score: float
    calibration_error: float
    curve: ReliabilityCurve

    def as_dict(self) -> dict[str, object]:
        return {
            "auroc": self.auroc,
            "brier_score": self.brier_score,
            "calibration_error": self.calibration_error,
            "reliability_curve": self.curve.as_dict(),
        }


def evaluate_reliability(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    n_bins: int = 10,
) -> EvaluationResult:
    y, p = _validate_binary_inputs(labels, scores)
    curve = reliability_curve(y, p, n_bins=n_bins)
    return EvaluationResult(
        auroc=roc_auc(y, p),
        brier_score=brier_score(y, p),
        calibration_error=expected_calibration_error(y, p, n_bins=n_bins),
        curve=curve,
    )


def evaluate_by_group_offset(
    labels: Sequence[int],
    scores: Sequence[float],
    groups: Sequence[str],
    offsets: Sequence[int],
    *,
    n_bins: int = 10,
) -> dict[str, dict[str, EvaluationResult]]:
    """Evaluate aggregate, per-group, and per-offset slices."""

    y, p = _validate_binary_inputs(labels, scores)
    group_array = np.asarray(groups, dtype=object)
    offset_array = np.asarray(offsets, dtype=np.int64)
    if group_array.shape != y.shape or offset_array.shape != y.shape:
        raise ValueError("groups and offsets must match labels")

    result: dict[str, dict[str, EvaluationResult]] = {
        "overall": {"all": evaluate_reliability(y, p, n_bins=n_bins)}
    }
    for group in sorted({str(value) for value in group_array}):
        selected = np.asarray([str(value) == group for value in group_array])
        result.setdefault("group", {})[group] = evaluate_reliability(
            y[selected], p[selected], n_bins=n_bins
        )
    for offset in sorted({int(value) for value in offset_array}):
        selected = offset_array == offset
        result.setdefault("offset", {})[str(offset)] = evaluate_reliability(
            y[selected], p[selected], n_bins=n_bins
        )
    return result
