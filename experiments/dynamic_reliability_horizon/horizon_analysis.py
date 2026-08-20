"""Offline horizon decoding summaries; no rollout-success interpretation."""

from __future__ import annotations

from dataclasses import dataclass
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


def compare_horizon_sources(
    predicted_curves: Sequence[Mapping[str, Sequence[float]]],
    oracle_curves: Sequence[Mapping[str, Sequence[float]]],
    *,
    decoder: GroupHorizonDecoder,
    static_horizons: Mapping[str, int],
    global_horizon: int | None = None,
) -> dict[str, HorizonScheduleSummary]:
    if len(predicted_curves) != len(oracle_curves):
        raise ValueError("predicted and oracle curve collections must match")
    predicted = [decoder.decode_curves(curves) for curves in predicted_curves]
    oracle = [decoder.decode_curves(curves) for curves in oracle_curves]
    static = [dict(static_horizons) for _ in predicted]
    result = {
        "static_group": summarize_horizon_schedule(static, source="static_group"),
        "learned_reliability": summarize_horizon_schedule(predicted, source="learned_reliability"),
        "oracle_reliability": summarize_horizon_schedule(oracle, source="oracle_reliability"),
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
