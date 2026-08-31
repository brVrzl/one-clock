#!/usr/bin/env python3
"""CPU-only action fusion operators for the group-memory development ladder.

The module receives already postprocessed same-target candidates ordered from
oldest source query to newest source query.  It imports the repository's
validated cosine and aggregation primitives, while keeping the Sol-selected
shared temporal kernel explicit at the call boundary.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from experiments.component_temporal_reuse.temporal_operators import (
    ARM,
    DEFAULT_GROUPS,
    GRIPPER,
    aggregate_components,
    aggregate_full_action,
    normalize_weights,
)


ARM_DIM = 6
ACTION_DIM = 7
ALPHA = 0.3
BASE_COEFFICIENT = 0.01


def _validate_candidates(candidates: np.ndarray, ages: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(candidates, dtype=np.float64)
    source_ages = np.asarray(ages, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or len(actions) == 0:
        raise ValueError(f"candidates must have shape (N,7), got {actions.shape}")
    if source_ages.shape != (len(actions),) or not np.isfinite(source_ages).all() or np.any(source_ages < 0):
        raise ValueError("ages must be finite, nonnegative, and aligned with candidates")
    if not np.isfinite(actions).all():
        raise ValueError("candidate actions must be finite")
    return actions, source_ages


def _validate_newest_is_last(actions: np.ndarray, ages: np.ndarray) -> None:
    # ``age=t-q`` is the physical target/source delay, so between scheduled
    # queries the newest source has a positive age.  Only a query target
    # (t=q_newest) has newest age zero.  Ordering, not zero-age identity, is
    # the invariant needed by all four operators.
    if np.any(np.diff(ages) > 0.0):
        raise ValueError("candidates must be ordered oldest to newest with non-increasing source ages")


def shared_temporal_prior(ages: np.ndarray, *, kernel_name: str, coefficient: float = BASE_COEFFICIENT) -> np.ndarray:
    """Return the frozen Sol-selected shared prior before compatibility factors."""

    source_ages = np.asarray(ages, dtype=np.float64)
    if source_ages.ndim != 1 or len(source_ages) == 0:
        raise ValueError("ages must be a non-empty vector")
    if not np.isfinite(source_ages).all() or np.any(source_ages < 0):
        raise ValueError("ages must be finite and nonnegative")
    coefficient = float(coefficient)
    if not np.isfinite(coefficient) or coefficient < 0:
        raise ValueError("coefficient must be finite and nonnegative")
    if kernel_name == "physical_age_te":
        logits = -coefficient * source_ages
    elif kernel_name == "dense_equivalent_te":
        # Candidate rows are oldest to newest.  Therefore q-q_oldest is the
        # reverse of age, and this exactly preserves the validated ACT
        # oldest-to-newest orientation under sparse h16 subsampling.
        logits = -coefficient * (source_ages.max() - source_ages)
    elif kernel_name == "candidate_index_te":
        logits = -coefficient * np.arange(len(source_ages), dtype=np.float64)
    else:
        raise ValueError(f"unknown or unresolved shared kernel: {kernel_name!r}")
    logits -= logits.max()
    return normalize_weights(np.exp(logits), len(source_ages))


def _cosine_similarity(values: np.ndarray) -> np.ndarray:
    newest = values[-1]
    denominator = np.linalg.norm(values, axis=1) * np.linalg.norm(newest) + 1e-7
    return (values @ newest) / denominator


def _gripper_intent_compatibility(values: np.ndarray) -> np.ndarray:
    signs = np.sign(values.reshape(-1))
    return signs * signs[-1]


def _compatibility_scores(
    candidates: np.ndarray,
    base_weights: np.ndarray,
    compatibility: np.ndarray,
    *,
    alpha: float = ALPHA,
) -> np.ndarray:
    if base_weights.shape != (len(candidates),) or compatibility.shape != (len(candidates),):
        raise ValueError("base weights and compatibility must align with candidates")
    if np.any(base_weights <= 0) or not np.isfinite(base_weights).all():
        raise ValueError("base weights must be finite and positive")
    if not np.isfinite(compatibility).all():
        raise ValueError("compatibility values must be finite")
    logits = np.log(base_weights) + float(alpha) * compatibility
    logits -= logits.max()
    return normalize_weights(np.exp(logits), len(candidates))


def m0_hard(candidates: np.ndarray, ages: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """M0: newest sparse candidate for every action dimension."""

    actions, source_ages = _validate_candidates(candidates, ages)
    _validate_newest_is_last(actions, source_ages)
    weights = np.zeros(len(actions), dtype=np.float64)
    weights[-1] = 1.0
    return actions[-1].copy(), {"shared_weights": weights, "arm_weights": weights.copy(), "gripper_weights": weights.copy()}


def m1_shared_te(
    candidates: np.ndarray,
    ages: np.ndarray,
    *,
    kernel_name: str,
    coefficient: float = BASE_COEFFICIENT,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """M1: one Sol-selected temporal prior shared by all seven dimensions."""

    actions, source_ages = _validate_candidates(candidates, ages)
    _validate_newest_is_last(actions, source_ages)
    weights = shared_temporal_prior(source_ages, kernel_name=kernel_name, coefficient=coefficient)
    return aggregate_full_action(actions, weights), {
        "shared_weights": weights,
        "arm_weights": weights.copy(),
        "gripper_weights": weights.copy(),
    }


def m2_shared_cogact(
    candidates: np.ndarray,
    ages: np.ndarray,
    *,
    kernel_name: str,
    alpha: float = ALPHA,
    coefficient: float = BASE_COEFFICIENT,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """M2: shared prior times whole-action CogACT compatibility."""

    actions, source_ages = _validate_candidates(candidates, ages)
    _validate_newest_is_last(actions, source_ages)
    base = shared_temporal_prior(source_ages, kernel_name=kernel_name, coefficient=coefficient)
    compatibility = _cosine_similarity(actions)
    weights = _compatibility_scores(actions, base, compatibility, alpha=alpha)
    return aggregate_full_action(actions, weights), {
        "shared_weights": weights,
        "arm_weights": weights.copy(),
        "gripper_weights": weights.copy(),
        "whole_action_compatibility": compatibility,
    }


def m3_group_cogact(
    candidates: np.ndarray,
    ages: np.ndarray,
    *,
    kernel_name: str,
    alpha: float = ALPHA,
    coefficient: float = BASE_COEFFICIENT,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """M3: independently normalized arm and scalar-gripper compatibility."""

    actions, source_ages = _validate_candidates(candidates, ages)
    _validate_newest_is_last(actions, source_ages)
    base = shared_temporal_prior(source_ages, kernel_name=kernel_name, coefficient=coefficient)
    arm_compatibility = _cosine_similarity(actions[:, ARM])
    gripper_compatibility = _gripper_intent_compatibility(actions[:, GRIPPER])
    arm_weights = _compatibility_scores(actions, base, arm_compatibility, alpha=alpha)
    gripper_weights = _compatibility_scores(actions, base, gripper_compatibility, alpha=alpha)
    output = aggregate_components(
        actions,
        {"arm": arm_weights, "gripper": gripper_weights},
        DEFAULT_GROUPS,
    )
    return output, {
        "arm_weights": arm_weights,
        "gripper_weights": gripper_weights,
        "arm_compatibility": arm_compatibility,
        "gripper_compatibility": gripper_compatibility,
        "base_weights": base,
    }


def m4_anchored_group_reliability(*args, reliability: Mapping[str, np.ndarray] | None = None, **kwargs):
    """M4 entry point; refuses to fabricate a missing online reliability interface."""

    if reliability is None:
        raise RuntimeError("UNAVAILABLE_RELIABILITY_INTERFACE")
    raise RuntimeError("M4 reliability interface is not frozen for this development protocol")
