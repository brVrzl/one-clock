"""Offline horizon decoding summaries; no rollout-success interpretation."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence

import numpy as np

from .decoder import GroupHorizonDecoder


@dataclass(frozen=True)
class HorizonScheduleSummary:
    source: str
    count: int
    by_group: dict[str, dict[str, float]]

    def as_dict(self) -> dict[str, object]:
        return {"source": self.source, "count": self.count, "by_group": self.by_group}


@dataclass(frozen=True)
class HorizonRegret:
    """Offline discrepancy between a decoded schedule and an oracle schedule.

    Regret is reported in horizon steps.  Positive signed regret means the
    prediction over-commits relative to the oracle; negative means it
    under-commits.  The oracle here is only a held-out target diagnostic and
    is never an executor input.
    """

    count: int
    by_group: dict[str, dict[str, float]]
    overall: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {"count": self.count, "by_group": self.by_group, "overall": self.overall}


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman_correlation(predicted: np.ndarray, oracle: np.ndarray) -> float:
    if predicted.size < 2:
        return float("nan")
    predicted_ranks = _average_ranks(predicted)
    oracle_ranks = _average_ranks(oracle)
    if np.std(predicted_ranks) == 0.0 or np.std(oracle_ranks) == 0.0:
        return float("nan")
    return float(np.corrcoef(predicted_ranks, oracle_ranks)[0, 1])


def horizon_regret(
    predicted_horizons: Sequence[Mapping[str, int]],
    oracle_horizons: Sequence[Mapping[str, int]],
) -> HorizonRegret:
    """Measure decoded horizon error against an offline oracle schedule."""

    if not predicted_horizons or len(predicted_horizons) != len(oracle_horizons):
        raise ValueError("predicted and oracle schedules must be non-empty and aligned")
    groups = sorted({group for mapping in predicted_horizons for group in mapping})
    if groups != sorted({group for mapping in oracle_horizons for group in mapping}):
        raise ValueError("predicted and oracle schedules must contain the same groups")
    by_group: dict[str, dict[str, float]] = {}
    all_predicted: list[float] = []
    all_oracle: list[float] = []
    for group in groups:
        predicted: list[float] = []
        oracle: list[float] = []
        for prediction, reference in zip(predicted_horizons, oracle_horizons):
            if group not in prediction or group not in reference:
                raise ValueError("every schedule must contain every group")
            predicted.append(float(prediction[group]))
            oracle.append(float(reference[group]))
        difference = np.asarray(predicted) - np.asarray(oracle)
        all_predicted.extend(predicted)
        all_oracle.extend(oracle)
        by_group[group] = {
            "mean_absolute_regret": float(np.abs(difference).mean()),
            "mae": float(np.abs(difference).mean()),
            "median_absolute_error": float(np.median(np.abs(difference))),
            "within_plus_minus_2_rate": float(np.mean(np.abs(difference) <= 2.0)),
            "within_plus_minus_5_rate": float(np.mean(np.abs(difference) <= 5.0)),
            "mean_signed_regret": float(difference.mean()),
            "exact_match_rate": float(np.mean(difference == 0.0)),
            "undercommit_rate": float(np.mean(difference < 0.0)),
            "overcommit_rate": float(np.mean(difference > 0.0)),
            "spearman_correlation": _spearman_correlation(
                np.asarray(predicted), np.asarray(oracle)
            ),
        }
    overall_predicted = np.asarray(all_predicted)
    overall_oracle = np.asarray(all_oracle)
    overall_difference = overall_predicted - overall_oracle
    overall = {
        "mean_absolute_regret": float(np.abs(overall_difference).mean()),
        "mae": float(np.abs(overall_difference).mean()),
        "median_absolute_error": float(np.median(np.abs(overall_difference))),
        "within_plus_minus_2_rate": float(np.mean(np.abs(overall_difference) <= 2.0)),
        "within_plus_minus_5_rate": float(np.mean(np.abs(overall_difference) <= 5.0)),
        "mean_signed_regret": float(overall_difference.mean()),
        "exact_match_rate": float(np.mean(overall_difference == 0.0)),
        "undercommit_rate": float(np.mean(overall_difference < 0.0)),
        "overcommit_rate": float(np.mean(overall_difference > 0.0)),
        "spearman_correlation": _spearman_correlation(
            overall_predicted, overall_oracle
        ),
    }
    return HorizonRegret(len(predicted_horizons), by_group, overall)


def summarize_horizon_schedule(
    horizons: Sequence[Mapping[str, int]],
    *,
    source: str,
) -> HorizonScheduleSummary:
    if not horizons:
        raise ValueError("at least one horizon mapping is required")
    groups = sorted({group for mapping in horizons for group in mapping})
    by_group: dict[str, dict[str, float]] = {}
    for group in groups:
        values = np.asarray([mapping[group] for mapping in horizons if group in mapping], dtype=np.float64)
        by_group[group] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "p50": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
        }
    return HorizonScheduleSummary(source, len(horizons), by_group)


def rows_to_curves(
    *,
    episode_ids: Sequence[str],
    source_steps: Sequence[int],
    groups: Sequence[str],
    offsets: Sequence[int],
    scores: Sequence[float],
) -> tuple[dict[str, np.ndarray], ...]:
    """Group row scores into one group/offset curve per source observation."""

    episode_array = np.asarray(episode_ids).astype(str)
    source_array = np.asarray(source_steps, dtype=np.int64)
    group_array = np.asarray(groups).astype(str)
    offset_array = np.asarray(offsets, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    n = episode_array.size
    if any(array.shape != (n,) for array in (source_array, group_array, offset_array, score_array)):
        raise ValueError("curve row fields must have matching shapes")
    curves: list[dict[str, np.ndarray]] = []
    keys = list(dict.fromkeys(zip(episode_array, source_array)))
    for episode_id, source_step in keys:
        selected = (episode_array == episode_id) & (source_array == source_step)
        curve: dict[str, np.ndarray] = {}
        for group in sorted(set(group_array[selected])):
            group_selected = selected & (group_array == group)
            order = np.argsort(offset_array[group_selected], kind="stable")
            group_offsets = offset_array[group_selected][order]
            if group_offsets.size == 0 or not np.array_equal(
                group_offsets, np.arange(group_offsets.size, dtype=np.int64)
            ):
                raise ValueError("each source/group curve must contain contiguous offsets from zero")
            curve[group] = score_array[group_selected][order]
        if curve:
            curves.append(curve)
    if not curves:
        raise ValueError("no source curves could be constructed")
    return tuple(curves)


def vector_rows_to_curves(
    *,
    episode_ids: Sequence[str],
    source_steps: Sequence[int],
    groups: Sequence[str],
    scores: np.ndarray,
) -> tuple[dict[str, np.ndarray], ...]:
    """Group vector-head rows into one curve mapping per source observation."""

    episode_array = np.asarray(episode_ids).astype(str)
    source_array = np.asarray(source_steps, dtype=np.int64)
    group_array = np.asarray(groups).astype(str)
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 2 or any(
        array.shape != (score_array.shape[0],)
        for array in (episode_array, source_array, group_array)
    ):
        raise ValueError("vector curve rows must have matching metadata and score shapes")
    curves: list[dict[str, np.ndarray]] = []
    keys = list(dict.fromkeys(zip(episode_array, source_array)))
    for episode_id, source_step in keys:
        selected = (episode_array == episode_id) & (source_array == source_step)
        curve: dict[str, np.ndarray] = {}
        for group in sorted(set(group_array[selected])):
            group_selected = selected & (group_array == group)
            if int(group_selected.sum()) != 1:
                raise ValueError("vector dataset must have one row per source/group")
            curve[group] = score_array[group_selected][0].copy()
        if curve:
            curves.append(curve)
    if not curves:
        raise ValueError("no vector source curves could be constructed")
    return tuple(curves)


def compare_horizon_sources(
    predicted_curves: Sequence[Mapping[str, Sequence[float]]],
    oracle_curves: Sequence[Mapping[str, Sequence[float]]],
    *,
    decoder: GroupHorizonDecoder,
    static_horizons: Mapping[str, int],
    global_horizon: int | None = None,
) -> dict[str, HorizonScheduleSummary | HorizonRegret]:
    if len(predicted_curves) != len(oracle_curves):
        raise ValueError("predicted and oracle curve collections must match")
    predicted = [decoder.decode_curves(curves) for curves in predicted_curves]
    oracle = [decoder.decode_curves(curves) for curves in oracle_curves]
    static = [dict(static_horizons) for _ in predicted]
    result = {
        "static_group": summarize_horizon_schedule(static, source="static_group"),
        "learned_reliability": summarize_horizon_schedule(predicted, source="learned_reliability"),
        "oracle_reliability": summarize_horizon_schedule(oracle, source="oracle_reliability"),
        "horizon_regret": horizon_regret(predicted, oracle),
    }
    if global_horizon is not None:
        global_static = [
            {group: global_horizon for group in mapping}
            for mapping in predicted
        ]
        result["global_fixed"] = summarize_horizon_schedule(
            global_static, source="global_fixed"
        )
    return result
