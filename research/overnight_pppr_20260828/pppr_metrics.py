#!/usr/bin/env python3
"""Frozen, CPU-only PPPR Phase-0 feature construction.

The input cache is the output of ``run_component_reuse.py``.  For an episode
array ``chunks`` with shape ``(T, H, 7)``, row ``chunks[q, d]`` is the
postprocessed action predicted by query ``q`` for physical target ``q+d``.
This module deliberately contains no policy, simulator, training, or outcome
code.  It is a small, testable translation of the Phase-0 protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


ARM_DIM = 6
ACTION_DIM = 7
POSITION = slice(0, 3)
ROTATION = slice(3, 6)
ARM = slice(0, 6)
GRIP = 6
SQRT3 = float(np.sqrt(3.0))


@dataclass(frozen=True)
class ArmScaleFit:
    """Robust fixed arm scales fitted on development Fresh actions only."""

    scales: np.ndarray
    q25: np.ndarray
    q50: np.ndarray
    q75: np.ndarray
    method: str
    source_action_definition: str
    zero_scale_guard_value: float
    zero_scale_guard_applied: np.ndarray

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "source_action_definition": self.source_action_definition,
            "arm_scales": [float(x) for x in self.scales],
            "q25": [float(x) for x in self.q25],
            "q50": [float(x) for x in self.q50],
            "q75": [float(x) for x in self.q75],
            "zero_scale_guard_value": float(self.zero_scale_guard_value),
            "zero_scale_guard_applied": [bool(x) for x in self.zero_scale_guard_applied],
            "zero_scale_guard_count": int(self.zero_scale_guard_applied.sum()),
        }


def _as_episode(array: np.ndarray, *, validate_finite: bool = True) -> np.ndarray:
    """Validate and return one cache episode as a float64 ``(T,H,7)`` array."""

    result = np.asarray(array, dtype=np.float64)
    if result.ndim != 3 or result.shape[2] != ACTION_DIM:
        raise ValueError(f"expected episode shape (T,H,7), got {result.shape}")
    if result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError("episode cache must contain at least one query and one chunk row")
    if validate_finite and not np.isfinite(result).all():
        raise ValueError("cache actions must be finite")
    return result


def fresh_actions(episodes: Iterable[np.ndarray]) -> np.ndarray:
    """Extract postprocessed Fresh actions, exactly ``chunk[q,0,:]``."""

    rows = []
    for episode in episodes:
        chunks = _as_episode(episode)
        rows.append(chunks[:, 0, :])
    if not rows:
        raise ValueError("at least one development episode is required")
    return np.concatenate(rows, axis=0)


def fit_arm_scales(
    development_episodes: Iterable[np.ndarray],
    *,
    zero_scale_guard_value: float = 1.0,
) -> ArmScaleFit:
    """Fit per-dimension arm IQR scales on development Fresh actions.

    ``zero_scale_guard_value`` is not a tuning parameter: it is used only for
    an exactly zero or non-finite IQR, which is unreachable for the observed
    development cache but makes the deterministic metric total on a constant
    synthetic dimension.
    """

    if not np.isfinite(zero_scale_guard_value) or zero_scale_guard_value <= 0:
        raise ValueError("zero_scale_guard_value must be finite and positive")
    actions = fresh_actions(development_episodes)[:, :ARM_DIM]
    q25, q50, q75 = np.percentile(actions, [25.0, 50.0, 75.0], axis=0)
    iqr = np.asarray(q75 - q25, dtype=np.float64)
    guard = (~np.isfinite(iqr)) | (iqr <= 0.0)
    scales = np.where(guard, float(zero_scale_guard_value), iqr)
    return ArmScaleFit(
        scales=scales,
        q25=np.asarray(q25),
        q50=np.asarray(q50),
        q75=np.asarray(q75),
        method="IQR (Q75-Q25)",
        source_action_definition="all development Fresh postprocessed chunk[q,0,:6] actions",
        zero_scale_guard_value=float(zero_scale_guard_value),
        zero_scale_guard_applied=np.asarray(guard, dtype=bool),
    )


def _check_scales(scales: Sequence[float]) -> np.ndarray:
    result = np.asarray(scales, dtype=np.float64)
    if result.shape != (ARM_DIM,) or not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("arm scales must be finite and strictly positive with shape (6,)")
    return result


def grip_sign(value: float | np.ndarray) -> np.ndarray:
    """Return the protocol sign ``-1, 0, +1`` for gripper commands."""

    return np.sign(np.asarray(value)).astype(np.int8)


def arm_distance(a: Sequence[float], b: Sequence[float], scales: Sequence[float]) -> float:
    """Return bounded arm distance specified by Phase 0.

    Position and rotation use independent robust scales, each normalized by
    ``sqrt(3)``.  The two normalized distances are averaged and then bounded.
    """

    lhs = np.asarray(a, dtype=np.float64)
    rhs = np.asarray(b, dtype=np.float64)
    scale = _check_scales(scales)
    if lhs.shape[-1] < ARM_DIM or rhs.shape[-1] < ARM_DIM:
        raise ValueError("actions must have at least six arm dimensions")
    dp = float(np.linalg.norm((lhs[..., POSITION] - rhs[..., POSITION]) / scale[POSITION]) / SQRT3)
    dr = float(np.linalg.norm((lhs[..., ROTATION] - rhs[..., ROTATION]) / scale[ROTATION]) / SQRT3)
    d_arm = 0.5 * (dp + dr)
    return float(d_arm / (1.0 + d_arm))


def component_metrics(a: Sequence[float], b: Sequence[float], scales: Sequence[float]) -> np.ndarray:
    """Return ``[arm, grip, joint]`` discrepancy for two seven-D actions."""

    lhs = np.asarray(a, dtype=np.float64)
    rhs = np.asarray(b, dtype=np.float64)
    if lhs.shape[-1] < ACTION_DIM or rhs.shape[-1] < ACTION_DIM:
        raise ValueError("actions must have shape (...,7) or more")
    arm = arm_distance(lhs, rhs, scales)
    grip = float(grip_sign(lhs[..., GRIP]) != grip_sign(rhs[..., GRIP]))
    joint = 0.5 * (arm + grip)
    return np.asarray([arm, grip, joint], dtype=np.float64)


def _action_at_validated(array: np.ndarray, query_step: int, target_step: int) -> np.ndarray:
    """Align a physical target ``v`` to ``chunks[q, v-q]``.

    Negative offsets are rejected rather than silently selecting a future
    query's row.  This is the central anti-look-behind/physical-time guard.
    """

    q = int(query_step)
    v = int(target_step)
    if q < 0 or q >= array.shape[0]:
        raise IndexError(f"query step {q} is outside episode length {array.shape[0]}")
    offset = v - q
    if offset < 0:
        raise ValueError(f"query q={q} cannot predict earlier target v={v} (q<u guard)")
    if offset >= array.shape[1]:
        raise IndexError(
            f"target v={v} is outside query q={q} chunk with horizon {array.shape[1]}"
        )
    return array[q, offset].copy()


def action_at(chunks: np.ndarray, query_step: int, target_step: int) -> np.ndarray:
    """Align a physical target ``v`` to ``chunks[q, v-q]``."""

    return _action_at_validated(_as_episode(chunks), query_step, target_step)


def _validate_future_family(
    *,
    old_query: int,
    future_query: int,
    family_queries: Sequence[int],
    radius: int,
) -> np.ndarray:
    q = np.asarray(family_queries, dtype=np.int64)
    expected = np.arange(int(future_query), int(future_query) + int(radius) + 1, dtype=np.int64)
    if q.shape != expected.shape or not np.array_equal(q, expected):
        raise ValueError(
            "future family must be exactly q=u:u+r in ascending order; q<u is forbidden"
        )
    if np.any(q < int(future_query)) or np.any(q <= int(old_query)):
        raise ValueError("future-family query list contains an old query (q<u is forbidden)")
    return q


def future_consensus(family_actions: np.ndarray) -> np.ndarray:
    """Build one per-target consensus from q=u:u+2 family actions."""

    family = np.asarray(family_actions, dtype=np.float64)
    if family.ndim != 2 or family.shape[1] < ACTION_DIM or family.shape[0] != 3:
        raise ValueError("future family must have shape (3,7)")
    if not np.isfinite(family).all():
        raise ValueError("future family actions must be finite")
    consensus = np.empty(ACTION_DIM, dtype=np.float64)
    consensus[:ARM_DIM] = np.median(family[:, :ARM_DIM], axis=0)
    signs = grip_sign(family[:, GRIP])
    # Three members make a strict majority; this branch also documents the
    # deterministic tie rule if the radius is changed in a future experiment.
    values, counts = np.unique(signs, return_counts=True)
    max_count = int(counts.max())
    winners = set(int(value) for value, count in zip(values, counts) if int(count) == max_count)
    # Three real cache predictions are nonzero and therefore have a strict
    # binary majority.  For synthetic/all-zero ties, use the newest family
    # member's exact sign deterministically.
    majority = next((int(sign) for sign in signs[::-1] if int(sign) in winners), int(signs[-1]))
    consensus[GRIP] = float(majority)
    return consensus


def event_score_from_fresh_chunk(
    fresh_chunk: np.ndarray,
    scales: Sequence[float],
) -> dict[str, float | int]:
    """Compute a fixed, outcome-blind pre-treatment event score.

    The score uses only the current Fresh prediction chunk.  ``transition`` is
    high when the nearest predicted gripper sign transition is close.  The arm
    terms are median chunk-local normalized first differences and second
    differences (curvature), clipped to one.  The frozen score is

    ``0.5*transition + 0.25*arm_change + 0.25*arm_curvature``.

    A chunk with no predicted gripper transition receives transition proximity
    zero.  The arm denominators are the already-fitted development IQR scales;
    no outcome-dependent tuning is performed.
    """

    chunk = np.asarray(fresh_chunk, dtype=np.float64)
    scale = _check_scales(scales)
    if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM or len(chunk) == 0:
        raise ValueError("fresh_chunk must have shape (H,7) with H>0")
    if not np.isfinite(chunk).all():
        raise ValueError("fresh chunk actions must be finite")

    signs = grip_sign(chunk[:, GRIP])
    transition_positions = np.flatnonzero(signs[1:] != signs[:-1]) + 1
    nearest = int(transition_positions[0]) if len(transition_positions) else -1
    horizon = max(1, chunk.shape[0] - 1)
    proximity = 0.0 if nearest < 0 else float(1.0 - min(nearest, horizon) / horizon)

    normalized_arm = chunk[:, :ARM_DIM] / scale
    if len(chunk) >= 2:
        first = np.linalg.norm(np.diff(normalized_arm, axis=0), axis=1) / np.sqrt(float(ARM_DIM))
        arm_change = float(np.clip(np.median(first), 0.0, 1.0))
    else:
        arm_change = 0.0
    if len(chunk) >= 3:
        second = np.diff(normalized_arm, n=2, axis=0)
        curvature = float(np.clip(np.median(np.linalg.norm(second, axis=1) / np.sqrt(float(ARM_DIM))), 0.0, 1.0))
    else:
        curvature = 0.0
    score = 0.5 * proximity + 0.25 * arm_change + 0.25 * curvature
    return {
        "nearest_gripper_transition_offset": nearest,
        "gripper_transition_proximity": proximity,
        "arm_change": arm_change,
        "arm_curvature": curvature,
        "event_score": float(score),
    }


def pair_feature(
    chunks: np.ndarray,
    *,
    old_query: int,
    age_steps: int,
    scales: Sequence[float],
    radius: int = 2,
    window_size: int = 4,
    window_start_offset: int = 2,
) -> dict[str, object] | None:
    """Compute one valid ``(t,u=t+k)`` PPPR row, or ``None`` at a boundary.

    Validity requires all physical targets in W to be actual episode steps,
    inside the old chunk, and present in every future-family chunk.  The
    family is exactly ``q=u,u+1,u+2``; no earlier query is admitted.
    """

    # Build validates each loaded episode once.  Avoid rescanning the full
    # (potentially long) cache for every pair/target here.
    array = _as_episode(chunks, validate_finite=False)
    t = int(old_query)
    k = int(age_steps)
    u = t + k
    r = int(radius)
    m = int(window_size)
    start = int(window_start_offset)
    if k <= 0 or r < 0 or m <= 0 or start < 0:
        raise ValueError("age must be positive; radius/window size/offset must be nonnegative/positive")
    if t < 0 or t >= array.shape[0] or u + r >= array.shape[0]:
        return None
    future_queries = _validate_future_family(
        old_query=t,
        future_query=u,
        family_queries=np.arange(u, u + r + 1),
        radius=r,
    )
    targets = np.arange(u + start, u + start + m, dtype=np.int64)
    # ``v`` may be beyond the final executed environment step: the cache still
    # contains its predicted target row.  Validity is governed by required
    # query records (u:u+r) and chunk offsets, not by executed target rows.
    horizon = array.shape[1]
    if np.any(targets - t >= horizon):
        return None
    if np.any(targets[:, None] - future_queries[None, :] < 0):
        # This catches q>v explicitly instead of permitting look-behind.
        return None
    if np.any(targets[:, None] - future_queries[None, :] >= horizon):
        return None

    scale = _check_scales(scales)
    raw_per_target = []
    pppr_per_target = []
    dispersion_per_target = []
    consensus_actions = []
    family_actions = []
    for v in targets.tolist():
        old = _action_at_validated(array, t, v)
        immediate = _action_at_validated(array, u, v)
        family = np.stack([_action_at_validated(array, int(q), v) for q in future_queries], axis=0)
        consensus = future_consensus(family)
        raw_per_target.append(component_metrics(old, immediate, scale))
        pppr_per_target.append(component_metrics(old, consensus, scale))
        dispersion_per_target.append(
            np.median(np.stack([component_metrics(candidate, consensus, scale) for candidate in family]), axis=0)
        )
        consensus_actions.append(consensus)
        family_actions.append(family)

    raw = np.asarray(raw_per_target)
    pppr = np.asarray(pppr_per_target)
    dispersion = np.asarray(dispersion_per_target)
    # Frozen persistent margin: R is old-to-consensus, C is future-family
    # dispersion.  Raw old-vs-a[v|u] is a separate diagnostic, not R.
    margin = np.maximum(pppr - dispersion, 0.0)
    current_chunk = array[t].copy()
    event = event_score_from_fresh_chunk(current_chunk, scale)
    return {
        "old_query_t": t,
        "future_query_u": u,
        "age_steps": k,
        "future_family_queries": future_queries,
        "window_targets": targets,
        "current_action": current_chunk[0].copy(),
        "current_chunk": current_chunk,
        "raw_per_target": raw,
        "old_to_consensus_per_target": pppr,
        "future_dispersion_per_target": dispersion,
        "pppr_per_target": margin,
        "raw": np.median(raw, axis=0),
        "old_to_consensus": np.median(pppr, axis=0),
        "future_dispersion": np.median(dispersion, axis=0),
        "pppr": np.median(margin, axis=0),
        "consensus_actions": np.asarray(consensus_actions),
        "family_actions": np.asarray(family_actions),
        "event": event,
    }


def pair_feature_rows(
    chunks: np.ndarray,
    *,
    scales: Sequence[float],
    ages_steps: Sequence[int] = (4, 8, 16),
    radius: int = 2,
    window_size: int = 4,
    window_start_offset: int = 2,
) -> list[dict[str, object]]:
    """Generate all valid old/future source pairs for one episode."""

    array = _as_episode(chunks)
    rows = []
    for age in ages_steps:
        # ``u+2`` must exist as a future query and target windows must remain
        # in the actual episode; pair_feature performs all chunk checks.
        for t in range(array.shape[0]):
            row = pair_feature(
                array,
                old_query=t,
                age_steps=int(age),
                scales=scales,
                radius=radius,
                window_size=window_size,
                window_start_offset=window_start_offset,
            )
            if row is not None:
                rows.append(row)
    return rows


METRIC_NAMES = ("arm", "grip", "joint")


def flatten_row(row: Mapping[str, object]) -> dict[str, object]:
    """Convert one pair row to scalar/array columns for the feature table."""

    output: dict[str, object] = {
        "old_query_t": int(row["old_query_t"]),
        "future_query_u": int(row["future_query_u"]),
        "age_steps": int(row["age_steps"]),
        "current_action": np.asarray(row["current_action"], dtype=np.float32),
    }
    for name, key in (
        ("raw_ppr", "raw"),
        ("old_to_consensus", "old_to_consensus"),
        ("future_dispersion", "future_dispersion"),
        ("pppr", "pppr"),
    ):
        values = np.asarray(row[key], dtype=np.float32)
        for index, metric in enumerate(METRIC_NAMES):
            output[f"{name}_{metric}"] = float(values[index])
    event = row["event"]
    for field in (
        "nearest_gripper_transition_offset",
        "gripper_transition_proximity",
        "arm_change",
        "arm_curvature",
        "event_score",
    ):
        output[f"event_{field}"] = event[field]
    return output
