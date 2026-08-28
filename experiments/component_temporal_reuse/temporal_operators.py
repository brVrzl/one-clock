#!/usr/bin/env python3
"""CPU-only same-target temporal aggregation and selection operators.

This module is intentionally independent of the frozen pilot runner.  It accepts
already-produced action chunks, extracts predictions for one physical target
time, and applies full-action or component-wise temporal rules.  No policy or
environment is imported here.

Candidate convention throughout: rows are ordered oldest source to newest
source; ``ages`` therefore normally decreases to zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


ARM = slice(0, 6)
GRIPPER = slice(6, 7)
DEFAULT_GROUPS = {"arm": ARM, "gripper": GRIPPER}


@dataclass(frozen=True)
class SameTargetCandidates:
    actions: np.ndarray
    ages: np.ndarray
    source_steps: np.ndarray


def same_target_candidates(query_history: list[np.ndarray], target_step: int) -> SameTargetCandidates:
    """Extract every cached prediction for physical time ``target_step``.

    ``query_history[tau]`` may have shape ``(H, D)`` or ``(1, H, D)``.  The
    prediction from source ``tau`` is row ``target_step - tau`` of that chunk.
    """

    target_step = int(target_step)
    rows: list[np.ndarray] = []
    source_steps: list[int] = []
    for source_step, raw_chunk in enumerate(query_history[: target_step + 1]):
        chunk = np.asarray(raw_chunk, dtype=np.float64)
        if chunk.ndim == 3 and chunk.shape[0] == 1:
            chunk = chunk[0]
        if chunk.ndim != 2:
            raise ValueError("each cached chunk must have shape (H, D) or (1, H, D)")
        offset = target_step - source_step
        if offset < len(chunk):
            rows.append(chunk[offset].copy())
            source_steps.append(source_step)
    if not rows:
        raise ValueError(f"no cached prediction targets step {target_step}")
    actions = np.stack(rows)
    if not np.isfinite(actions).all():
        raise ValueError("cached predictions must be finite")
    sources = np.asarray(source_steps, dtype=np.int64)
    return SameTargetCandidates(
        actions=actions,
        ages=target_step - sources,
        source_steps=sources,
    )


def normalize_weights(weights: np.ndarray, candidate_count: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (candidate_count,):
        raise ValueError(f"expected {candidate_count} weights, got shape {weights.shape}")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("temporal weights must be finite and nonnegative")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("temporal weights must have positive sum")
    return weights / total


def one_hot_age(ages: np.ndarray, requested_age: int) -> np.ndarray:
    """One-hot weight for an exact source age; fail instead of silently rounding."""

    ages = np.asarray(ages, dtype=np.int64)
    positions = np.flatnonzero(ages == int(requested_age))
    if len(positions) != 1:
        raise ValueError(f"source age {requested_age} is not uniquely available")
    result = np.zeros(len(ages), dtype=np.float64)
    result[int(positions[0])] = 1.0
    return result


def act_temporal_weights(candidate_count: int, coefficient: float = 0.01) -> np.ndarray:
    """Pinned LeRobot ACT semantics: index zero is the oldest source.

    This reproduces ``ACTTemporalEnsembler``.  With a positive coefficient,
    later-arriving (newer) candidates receive smaller weights.
    """

    logits = -float(coefficient) * np.arange(int(candidate_count), dtype=np.float64)
    logits -= logits.max()
    return normalize_weights(np.exp(logits), int(candidate_count))


def exponential_age_weights(ages: np.ndarray, beta: float = 0.03) -> np.ndarray:
    """Generic physical-age decay: positive beta favors newer predictions."""

    ages = np.asarray(ages, dtype=np.float64)
    logits = -float(beta) * ages
    logits -= logits.max()
    return normalize_weights(np.exp(logits), len(ages))


def cogact_cosine_weights(candidates: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """CogACT-style full-action similarity weights relative to newest.

    This is the released cosine rule used in the project's prior Gate-3A/3C
    controls.  One scalar weight is applied to the complete action vector.
    """

    candidates = np.asarray(candidates, dtype=np.float64)
    if candidates.ndim != 2 or len(candidates) == 0:
        raise ValueError("candidates must have shape (sources, action_dim)")
    newest = candidates[-1]
    denominator = np.linalg.norm(candidates, axis=1) * np.linalg.norm(newest) + 1e-7
    cosine = (candidates @ newest) / denominator
    logits = float(alpha) * cosine
    logits -= logits.max()
    return normalize_weights(np.exp(logits), len(candidates))


def aggregate_full_action(candidates: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Ordinary temporal aggregation with one weight vector for every dimension."""

    candidates = np.asarray(candidates, dtype=np.float64)
    if candidates.ndim != 2:
        raise ValueError("candidates must have shape (sources, action_dim)")
    normalized = normalize_weights(weights, len(candidates))
    return normalized @ candidates


def aggregate_components(
    candidates: np.ndarray,
    weights_by_group: Mapping[str, np.ndarray],
    groups: Mapping[str, slice] = DEFAULT_GROUPS,
) -> np.ndarray:
    """Apply an independently normalized temporal weight vector to each group."""

    candidates = np.asarray(candidates, dtype=np.float64)
    if candidates.ndim != 2:
        raise ValueError("candidates must have shape (sources, action_dim)")
    if set(weights_by_group) != set(groups):
        raise ValueError("weights_by_group must define exactly the configured groups")
    output = np.empty(candidates.shape[1], dtype=np.float64)
    covered = np.zeros(candidates.shape[1], dtype=bool)
    for name, indices in groups.items():
        normalized = normalize_weights(weights_by_group[name], len(candidates))
        output[indices] = normalized @ candidates[:, indices]
        covered[indices] = True
    if not covered.all():
        raise ValueError("configured groups must cover every action dimension")
    return output


def component_candidate_features(
    candidates: np.ndarray,
    ages: np.ndarray,
    indices: slice,
) -> dict[str, np.ndarray]:
    """Outcome-blind observable features for retrospective source selection."""

    candidates = np.asarray(candidates, dtype=np.float64)
    ages = np.asarray(ages, dtype=np.float64)
    if candidates.ndim != 2 or ages.shape != (len(candidates),):
        raise ValueError("candidates and ages are misaligned")
    values = candidates[:, indices]
    newest = values[-1]
    centroid = values.mean(axis=0)
    pairwise = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=-1)
    return {
        "source_age": ages.copy(),
        "fresh_disagreement_l2": np.linalg.norm(values - newest, axis=1),
        "action_magnitude_l2": np.linalg.norm(values, axis=1),
        "distance_to_cached_centroid_l2": np.linalg.norm(values - centroid, axis=1),
        "mean_pairwise_disagreement_l2": pairwise.mean(axis=1),
    }


def select_component_source(
    features: Mapping[str, np.ndarray],
    *,
    rule: str,
    disagreement_threshold: float | None = None,
) -> int:
    """Select a cached source using observables only, never rollout outcomes.

    Rules are deliberately diagnostic:
    - ``newest``: the ordinary fresh action.
    - ``consensus_medoid``: candidate with minimum mean pairwise disagreement.
    - ``oldest_within_fresh_disagreement``: oldest candidate no farther from the
      fresh prediction than a fixed threshold.  It still requires a fresh query
      and is therefore not a query-saving deployment method.
    """

    ages = np.asarray(features["source_age"], dtype=np.float64)
    if len(ages) == 0:
        raise ValueError("at least one candidate is required")
    if rule == "newest":
        return int(np.argmin(ages))
    if rule == "consensus_medoid":
        scores = np.asarray(features["mean_pairwise_disagreement_l2"], dtype=np.float64)
        return int(np.argmin(scores))
    if rule == "oldest_within_fresh_disagreement":
        if disagreement_threshold is None or disagreement_threshold < 0:
            raise ValueError("a nonnegative disagreement threshold is required")
        disagreement = np.asarray(features["fresh_disagreement_l2"], dtype=np.float64)
        eligible = np.flatnonzero(disagreement <= float(disagreement_threshold))
        if len(eligible) == 0:
            raise RuntimeError("fresh candidate should always satisfy a nonnegative threshold")
        return int(eligible[np.argmax(ages[eligible])])
    raise ValueError(f"unknown outcome-blind selection rule: {rule!r}")


def selected_component_action(
    candidates: np.ndarray,
    selected_by_group: Mapping[str, int],
    groups: Mapping[str, slice] = DEFAULT_GROUPS,
) -> np.ndarray:
    """Compose group actions from selected same-target candidate rows."""

    candidates = np.asarray(candidates, dtype=np.float64)
    weights = {}
    for name, position in selected_by_group.items():
        one_hot = np.zeros(len(candidates), dtype=np.float64)
        one_hot[int(position)] = 1.0
        weights[name] = one_hot
    return aggregate_components(candidates, weights, groups)
