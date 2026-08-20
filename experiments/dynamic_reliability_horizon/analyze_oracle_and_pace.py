#!/usr/bin/env python3
"""Offline oracle group-horizon and PACE-style diagnostic.

This script consumes the already materialized frozen-policy re-query cache. It
does not instantiate a policy, train an estimator, create a horizon target for
training, step an executor, or run a rollout.

The oracle horizon is a descriptive right-censored prefix duration derived
from ``Y_refresh``.  The PACE-style rule uses only the stored old predicted
chunk when selecting a horizon; ``Y_refresh`` is used only afterward to score
the selected schedule.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CHUNK_SIZE = 100
GROUPS = ("arm", "gripper")
KEY_OFFSETS = (1, 2, 4, 8, 16, 32, 64, 99)
SMOOTHING_WINDOW = 5
MIN_VALLEY_SEPARATION = 10
PACE_PERCENTILE = 5.0
CALIBRATION_FRACTION = 0.8
CALIBRATION_SEED = 20260820
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260820
# Predeclared before inspecting sensitivity outputs.  These are multipliers
# on the existing normalized error tolerances; sign agreement is unchanged.
REFRESH_THRESHOLD_MULTIPLIERS = (0.75, 1.0, 1.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "experiments/temporal_reliability_target_comparison/target_comparison.npz",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "experiments/temporal_reliability/metadata.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "experiments/temporal_reliability/dataset_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/dynamic_reliability_horizon",
    )
    return parser.parse_args()


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def json_number(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return json_number(value)


def validate(data: Any, metadata: list[dict[str, Any]]) -> None:
    n = len(metadata)
    if data["episode_index"].shape != (n,) or data["frame_index"].shape != (n,):
        raise RuntimeError("target-comparison arrays and metadata have different row counts")
    if data["observed_offsets"].shape != (n, CHUNK_SIZE):
        raise RuntimeError("unexpected observed_offsets shape")
    if data["old_predicted_actions"].shape != (n, CHUNK_SIZE, 7):
        raise RuntimeError("unexpected old_predicted_actions shape")
    if data["refresh_first_actions"].shape != (n, CHUNK_SIZE, 7):
        raise RuntimeError("unexpected refresh_first_actions shape")
    episodes = data["episode_index"].astype(np.int64)
    frames = data["frame_index"].astype(np.int64)
    expected_episodes = np.asarray([int(row["episode_index"]) for row in metadata], dtype=np.int64)
    expected_frames = np.asarray([int(row["frame_index"]) for row in metadata], dtype=np.int64)
    if not np.array_equal(episodes, expected_episodes) or not np.array_equal(frames, expected_frames):
        raise RuntimeError("metadata is not row-aligned with target_comparison.npz")
    observed = data["observed_offsets"].astype(bool)
    if not np.all(np.diff(observed.astype(np.int8), axis=1) <= 0):
        raise RuntimeError("observed offsets must be contiguous prefixes")
    for key in (
        "arm_refresh_survival",
        "gripper_refresh_survival",
        "arm_refresh_pointwise",
        "gripper_refresh_pointwise",
    ):
        if data[key].shape != (n, CHUNK_SIZE):
            raise RuntimeError(f"unexpected {key} shape")


def summary_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None, "quantiles": {}}
    quantile_values = np.percentile(values, [10, 25, 50, 75, 90, 95])
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "quantiles": {
            "q10": float(quantile_values[0]),
            "q25": float(quantile_values[1]),
            "q50": float(quantile_values[2]),
            "q75": float(quantile_values[3]),
            "q90": float(quantile_values[4]),
            "q95": float(quantile_values[5]),
        },
    }


def derive_oracle_horizon(survival: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return action-count prefix duration and right-censoring indicator.

    ``survival[:, k]`` is validity through offset k.  Therefore a valid
    positive offset k contributes k+1 actions.  Offset k=0 is used only to
    establish the minimum one-action convention; it is never allowed to
    provide evidence for a longer horizon.  A row whose positive suffix is
    entirely valid is right-censored at the last observed action.
    """

    survival = survival.astype(bool)
    observed = observed.astype(bool)
    available = observed.sum(axis=1).astype(np.int32)
    positive = survival[:, 1:] & observed[:, 1:]
    # Number of consecutive valid positive offsets beginning at k=1.
    prefix_count = np.zeros(len(survival), dtype=np.int32)
    if positive.shape[1]:
        still_valid = np.ones(len(survival), dtype=bool)
        for column in range(positive.shape[1]):
            still_valid &= positive[:, column]
            prefix_count += still_valid.astype(np.int32)
    horizon = np.minimum(available, 1 + prefix_count).astype(np.int32)
    horizon = np.maximum(horizon, 1)
    censored = (available <= 1) | (horizon == available)
    return horizon, censored


def refresh_validity_from_cached_actions(
    old_actions: np.ndarray,
    refreshed_actions: np.ndarray,
    observed: np.ndarray,
    action_std: np.ndarray,
    threshold_multiplier: float,
) -> dict[str, np.ndarray]:
    """Recompute Y_refresh from cached actions, without policy inference.

    This intentionally mirrors ``construct_dataset.py`` and
    ``compare_targets.py``.  The multiplier is applied to the three existing
    normalized-error tolerances only; the gripper sign criterion is fixed.
    """

    difference = old_actions.astype(np.float32) - refreshed_actions.astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        translation = np.sqrt(np.mean((difference[:, :, :3] / action_std[:3]) ** 2, axis=2))
        rotation = np.sqrt(np.mean((difference[:, :, 3:6] / action_std[3:6]) ** 2, axis=2))
        gripper = np.abs(difference[:, :, 6]) / float(action_std[6])
    target_sign = np.where(refreshed_actions[:, :, 6] >= 0.0, 1, -1)
    predicted_sign = np.where(old_actions[:, :, 6] >= 0.0, 1, -1)
    sign_match = predicted_sign == target_sign
    arm_pointwise = (
        (translation <= threshold_multiplier)
        & (rotation <= threshold_multiplier)
        & observed
    )
    gripper_pointwise = (
        (gripper <= threshold_multiplier)
        & sign_match
        & observed
    )
    return {
        "arm_pointwise": arm_pointwise,
        "gripper_pointwise": gripper_pointwise,
        "arm_survival": np.logical_and.accumulate(arm_pointwise, axis=1) & observed,
        "gripper_survival": np.logical_and.accumulate(gripper_pointwise, axis=1) & observed,
    }


def grouped_rows(metadata: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tasks = np.asarray([str(row["task_name"]) for row in metadata], dtype=object)
    phases = np.asarray([str(row["phase"]) for row in metadata], dtype=object)
    episodes = np.asarray([int(row["episode_index"]) for row in metadata], dtype=np.int32)
    progress = np.asarray([float(row["progress"]) for row in metadata], dtype=float)
    return tasks, phases, episodes, progress


def conditional_stats(
    horizons: dict[str, np.ndarray],
    censoring: dict[str, np.ndarray],
    row_group: np.ndarray,
    groups: list[str],
    *,
    include_censoring: bool = True,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in groups:
        selected = row_group == label
        result[str(label)] = {
            group: {
                "horizon_actions": summary_stats(horizons[group][selected]),
                "censoring_rate": float(np.mean(censoring[group][selected])) if np.any(selected) else None,
            }
            for group in GROUPS
        }
        if not include_censoring:
            for group in GROUPS:
                uncensored = selected & ~censoring[group]
                result[str(label)][group]["uncensored_horizon_actions"] = summary_stats(
                    horizons[group][uncensored]
                )
    return result


def horizon_prevalence(
    data: Any,
    observed: np.ndarray,
    selected: np.ndarray,
    target_prefix: str = "refresh",
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in GROUPS:
        pointwise = data[f"{group}_{target_prefix}_pointwise"].astype(bool)
        survival = data[f"{group}_{target_prefix}_survival"].astype(bool)
        pointwise_values: list[Any] = []
        survival_values: list[Any] = []
        for k in range(CHUNK_SIZE):
            mask = selected & observed[:, k]
            pointwise_values.append(float(np.mean(pointwise[mask, k])) if np.any(mask) else None)
            survival_values.append(float(np.mean(survival[mask, k])) if np.any(mask) else None)
        result[group] = {
            "pointwise_by_offset": pointwise_values,
            "survival_by_offset": survival_values,
            "key_offsets": {
                str(k): {
                    "pointwise": pointwise_values[k],
                    "survival": survival_values[k],
                    "n_observed": int(np.sum(selected & observed[:, k])),
                }
                for k in KEY_OFFSETS
            },
        }
    return result


def heterogeneity(horizons: dict[str, np.ndarray], censoring: dict[str, np.ndarray]) -> dict[str, Any]:
    arm = horizons["arm"]
    gripper = horizons["gripper"]
    difference = arm.astype(np.int32) - gripper.astype(np.int32)
    equal = arm == gripper
    arm_first = arm < gripper
    gripper_first = gripper < arm
    both_uncensored = ~censoring["arm"] & ~censoring["gripper"]

    def categorical(mask: np.ndarray) -> dict[str, Any]:
        n = int(np.sum(mask))
        return {
            "n": n,
            "p_arm_expires_first": float(np.mean(arm_first[mask])) if n else None,
            "p_gripper_expires_first": float(np.mean(gripper_first[mask])) if n else None,
            "p_equal": float(np.mean(equal[mask])) if n else None,
            "p_different": float(np.mean(~equal[mask])) if n else None,
            "h_arm_minus_h_gripper": summary_stats(difference[mask]),
        }

    min_horizon = np.minimum(arm, gripper)
    wasted = (arm - min_horizon) + (gripper - min_horizon)
    total_commitment = arm + gripper
    return {
        "all_rows_lower_bound_comparison": categorical(np.ones(len(arm), dtype=bool)),
        "both_uncensored_exact_comparison": categorical(both_uncensored),
        "wasted_commitment_actions_lower_bound": summary_stats(wasted),
        "wasted_commitment_positive_rate_lower_bound": float(np.mean(wasted > 0)),
        "wasted_commitment_fraction_of_observed_group_commitment": summary_stats(
            np.divide(wasted, total_commitment, out=np.zeros_like(wasted, dtype=float), where=total_commitment > 0)
        ),
        "wasted_commitment_actions_both_uncensored": summary_stats(wasted[both_uncensored]),
        "wasted_commitment_fraction_both_uncensored": summary_stats(
            np.divide(
                wasted[both_uncensored],
                total_commitment[both_uncensored],
                out=np.zeros(np.sum(both_uncensored), dtype=float),
                where=total_commitment[both_uncensored] > 0,
            )
        ),
        "global_min_horizon_actions": summary_stats(min_horizon),
    }


def bootstrap_heterogeneity_intervals(
    horizons: dict[str, np.ndarray],
    censoring: dict[str, np.ndarray],
    episodes: np.ndarray,
) -> dict[str, Any]:
    """Episode-cluster bootstrap for the uncensored heterogeneity estimands."""

    unique_episodes = np.asarray(sorted(set(int(value) for value in episodes.tolist())), dtype=np.int32)
    rows_by_episode = {
        int(episode): np.flatnonzero(episodes == episode)
        for episode in unique_episodes.tolist()
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws: list[tuple[float, float, float]] = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled_episodes = rng.choice(unique_episodes, size=len(unique_episodes), replace=True)
        row_indices = np.concatenate([rows_by_episode[int(episode)] for episode in sampled_episodes])
        arm = horizons["arm"][row_indices]
        gripper = horizons["gripper"][row_indices]
        both_uncensored = ~censoring["arm"][row_indices] & ~censoring["gripper"][row_indices]
        if not np.any(both_uncensored):
            continue
        arm = arm[both_uncensored].astype(np.float64)
        gripper = gripper[both_uncensored].astype(np.float64)
        minimum = np.minimum(arm, gripper)
        wasted = (arm - minimum) + (gripper - minimum)
        total = arm + gripper
        discarded_fraction = np.divide(
            wasted,
            total,
            out=np.zeros_like(wasted, dtype=float),
            where=total > 0,
        )
        draws.append(
            (
                float(np.mean(arm != gripper)),
                float(np.mean(wasted)),
                float(np.mean(discarded_fraction)),
            )
        )
    if not draws:
        raise RuntimeError("episode bootstrap produced no valid both-uncensored draws")
    values = np.asarray(draws, dtype=float)
    point = heterogeneity(horizons, censoring)
    exact = point["both_uncensored_exact_comparison"]
    point_values = np.asarray(
        [
            exact["p_different"],
            point["wasted_commitment_actions_both_uncensored"]["mean"],
            point["wasted_commitment_fraction_both_uncensored"]["mean"],
        ],
        dtype=float,
    )
    intervals = np.percentile(values, [2.5, 97.5], axis=0)
    names = (
        "p_h_arm_not_equal_h_gripper",
        "mean_discarded_valid_positions",
        "discarded_commitment_fraction",
    )
    return {
        "unit": "episode",
        "seed": BOOTSTRAP_SEED,
        "replicates": int(len(values)),
        "estimand_population": "rows uncensored for both groups",
        "metrics": {
            name: {
                "point_estimate": float(point_values[index]),
                "ci95_low": float(intervals[0, index]),
                "ci95_high": float(intervals[1, index]),
            }
            for index, name in enumerate(names)
        },
    }


def threshold_sensitivity_analysis(
    old_actions: np.ndarray,
    refreshed_actions: np.ndarray,
    observed: np.ndarray,
    action_std: np.ndarray,
    episodes: np.ndarray,
) -> dict[str, Any]:
    """Offline tolerance sensitivity using only cached action arrays."""

    result: dict[str, Any] = {
        "predeclared_multipliers": list(REFRESH_THRESHOLD_MULTIPLIERS),
        "base_thresholds": {
            "arm_translation_normalized_rms": 1.0,
            "arm_rotation_normalized_rms": 1.0,
            "gripper_normalized_absolute_error": 1.0,
            "gripper_sign_agreement": "unchanged",
        },
        "uses_new_frozen_policy_inference": False,
        "uses_rollout_success": False,
        "results": {},
    }
    for multiplier in REFRESH_THRESHOLD_MULTIPLIERS:
        targets = refresh_validity_from_cached_actions(
            old_actions,
            refreshed_actions,
            observed,
            action_std,
            multiplier,
        )
        horizons: dict[str, np.ndarray] = {}
        censoring: dict[str, np.ndarray] = {}
        for group in GROUPS:
            horizons[group], censoring[group] = derive_oracle_horizon(
                targets[f"{group}_survival"], observed
            )
        het = heterogeneity(horizons, censoring)
        both = het["both_uncensored_exact_comparison"]
        result["results"][str(multiplier)] = {
            "threshold_multiplier": multiplier,
            "both_uncensored_windows": int(both["n"]),
            "both_uncensored_fraction": float(both["n"] / len(episodes)),
            "p_h_arm_not_equal_h_gripper": both["p_different"],
            "mean_discarded_valid_positions": het["wasted_commitment_actions_both_uncensored"]["mean"],
            "discarded_commitment_fraction": het["wasted_commitment_fraction_both_uncensored"]["mean"],
            "arm_censoring_rate": float(np.mean(censoring["arm"])),
            "gripper_censoring_rate": float(np.mean(censoring["gripper"])),
        }
    return result


def task_phase_analysis(
    horizons: dict[str, np.ndarray],
    censoring: dict[str, np.ndarray],
    tasks: np.ndarray,
    phases: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, values in (("task", tasks), ("phase", phases)):
        result[name] = {}
        for condition in sorted(set(values.tolist())):
            selected = values == condition
            local_heterogeneity = heterogeneity(
                {group: horizons[group][selected] for group in GROUPS},
                {group: censoring[group][selected] for group in GROUPS},
            )
            result[name][str(condition)] = {
                "n_rows": int(np.sum(selected)),
                "horizons": {
                    group: {
                        "all_rows_lower_bound": summary_stats(horizons[group][selected]),
                        "uncensored": summary_stats(horizons[group][selected & ~censoring[group]]),
                        "censoring_rate": float(np.mean(censoring[group][selected])),
                    }
                    for group in GROUPS
                },
                "heterogeneity": local_heterogeneity,
                "both_uncensored_horizon_difference_rate": local_heterogeneity[
                    "both_uncensored_exact_comparison"
                ]["p_different"],
                "both_uncensored_mean_discarded_valid_positions": local_heterogeneity[
                    "wasted_commitment_actions_both_uncensored"
                ]["mean"],
                "both_uncensored_discarded_commitment_fraction": local_heterogeneity[
                    "wasted_commitment_fraction_both_uncensored"
                ]["mean"],
            }
    return result


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values.astype(float, copy=True)
    left = width // 2
    right = width - 1 - left
    padded = np.pad(values.astype(float), ((0, 0), (left, right)), mode="edge")
    cumulative = np.cumsum(padded, axis=1)
    cumulative = np.pad(cumulative, ((0, 0), (1, 0)), mode="constant")
    return (cumulative[:, width:] - cumulative[:, :-width]) / float(width)


def signal_profiles(actions: np.ndarray, action_std: np.ndarray) -> dict[str, np.ndarray]:
    """Construct explicit proxies for PACE's unspecified kinematic operator.

    LIBERO ACT emits relative Cartesian position/axis-angle deltas, not joint
    positions.  We use normalized per-step magnitude within each action group.
    Translation and rotation are kept separate and combined by max for a
    conservative arm score; this is a declared PACE-style deviation.
    """

    arm_translation = np.sqrt(np.mean((actions[:, :, :3] / action_std[:3]) ** 2, axis=2))
    arm_rotation = np.sqrt(np.mean((actions[:, :, 3:6] / action_std[3:6]) ** 2, axis=2))
    arm = np.maximum(arm_translation, arm_rotation)
    gripper = np.abs(actions[:, :, 6]) / float(action_std[6])
    return {
        "arm": arm,
        "gripper": gripper,
        "arm_translation": arm_translation,
        "arm_rotation": arm_rotation,
    }


def local_valley_prominences(profile: np.ndarray, min_separation: int) -> tuple[np.ndarray, np.ndarray]:
    """Return local valley indices and a deterministic prominence proxy."""

    n, length = profile.shape
    if length < 3:
        return np.empty((n, 0), dtype=np.int32), np.empty((n, 0), dtype=float)
    interior = np.arange(1, length - 1, dtype=np.int32)
    is_valley = (profile[:, 1:-1] <= profile[:, :-2]) & (profile[:, 1:-1] <= profile[:, 2:])
    prominence = np.full((n, len(interior)), -np.inf, dtype=float)
    for column, index in enumerate(interior.tolist()):
        left_start = max(0, index - min_separation)
        right_stop = min(length, index + min_separation + 1)
        left_max = np.max(profile[:, left_start:index], axis=1)
        right_max = np.max(profile[:, index + 1 : right_stop], axis=1)
        prominence[:, column] = np.minimum(left_max, right_max) - profile[:, index]
    prominence[~is_valley] = -np.inf
    return np.broadcast_to(interior, (n, len(interior))).copy(), prominence


def nonmaximum_valleys(indices: np.ndarray, scores: np.ndarray, min_separation: int) -> list[tuple[int, float]]:
    finite = np.isfinite(scores)
    candidates = [(int(index), float(score)) for index, score in zip(indices[finite], scores[finite], strict=True)]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    selected: list[tuple[int, float]] = []
    for item in candidates:
        if all(abs(item[0] - kept[0]) >= min_separation for kept in selected):
            selected.append(item)
    return selected


def split_calibration_episodes(tasks: np.ndarray, episodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = sorted(set(int(value) for value in episodes.tolist()))
    calibration: set[int] = set()
    for task in sorted(set(tasks.tolist())):
        task_episodes = sorted(set(int(value) for value in episodes[tasks == task].tolist()))
        cutoff = max(1, int(math.ceil(CALIBRATION_FRACTION * len(task_episodes))))
        calibration.update(task_episodes[:cutoff])
    calibration_mask = np.asarray([int(value) in calibration for value in episodes], dtype=bool)
    if calibration_mask.all() or (~calibration_mask).sum() == 0:
        raise RuntimeError("episode-level calibration split is empty")
    return calibration_mask, ~calibration_mask


def calibrate_thresholds(
    smoothed: dict[str, np.ndarray],
    valley_indices: dict[str, np.ndarray],
    valley_scores: dict[str, np.ndarray],
    tasks: np.ndarray,
    calibration_mask: np.ndarray,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {group: {} for group in GROUPS}
    for task in sorted(set(tasks.tolist())):
        task_mask = calibration_mask & (tasks == task)
        for group in GROUPS:
            scores: list[float] = []
            for row in np.flatnonzero(task_mask):
                scores.extend(float(value) for value in valley_scores[group][row] if np.isfinite(value) and value > 0)
            threshold = float(np.percentile(scores, PACE_PERCENTILE)) if scores else 0.0
            result[group][str(task)] = threshold
    return result


def select_horizons(
    smoothed: dict[str, np.ndarray],
    valley_indices: dict[str, np.ndarray],
    valley_scores: dict[str, np.ndarray],
    tasks: np.ndarray,
    thresholds: dict[str, dict[str, float]],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for group in GROUPS:
        horizons = np.full(len(tasks), CHUNK_SIZE, dtype=np.int32)
        for row in range(len(tasks)):
            kept = nonmaximum_valleys(
                valley_indices[group][row], valley_scores[group][row], MIN_VALLEY_SEPARATION
            )
            accepted = [index for index, score in kept if score >= thresholds[group].get(str(tasks[row]), 0.0)]
            if accepted:
                # A valley index is an offset; execution horizon is an action count.
                horizons[row] = int(np.clip(min(accepted) + 1, 1, CHUNK_SIZE))
        result[group] = horizons
    return result


def schedule_metrics(
    name: str,
    predicted: dict[str, np.ndarray],
    oracle: dict[str, np.ndarray],
    observed: np.ndarray,
    refresh_survival: dict[str, np.ndarray],
    tasks: np.ndarray,
    phases: np.ndarray,
    eval_mask: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {"name": name, "n_rows": int(np.sum(eval_mask)), "groups": {}}
    query_horizons = predicted["arm"] if name == "pace_style_global_arm" else None
    for group in GROUPS:
        selected = eval_mask
        pred = predicted[group][selected].astype(float)
        target = oracle[group][selected].astype(float)
        diff = pred - target
        target_index = predicted[group][selected] - 1
        valid_target = (
            (target_index >= 0)
            & (target_index < CHUNK_SIZE)
            & observed[selected, :][np.arange(len(pred)), np.clip(target_index, 0, CHUNK_SIZE - 1)]
        )
        survival_values = np.full(len(pred), np.nan, dtype=float)
        if len(pred):
            row_indices = np.arange(len(pred))
            safe_index = np.clip(target_index, 0, CHUNK_SIZE - 1)
            survival_values = refresh_survival[group][selected][row_indices, safe_index].astype(float)
            survival_values[~valid_target] = np.nan
        group_result: dict[str, Any] = {
            "selected_horizon_actions": summary_stats(pred),
            "oracle_lower_bound_actions": summary_stats(target),
            "absolute_error_actions": summary_stats(np.abs(diff)),
            "signed_error_actions": summary_stats(diff),
            "selected_prefix_refresh_survival": {
                "n_observed": int(np.sum(np.isfinite(survival_values))),
                "mean": float(np.nanmean(survival_values)) if np.any(np.isfinite(survival_values)) else None,
            },
            "expected_local_query_rate_per_action": float(np.mean(1.0 / pred)) if len(pred) else None,
            "task_mean_horizon_std": float(
                np.std(
                    [np.mean(pred[tasks[selected] == task]) for task in sorted(set(tasks[selected].tolist()))]
                )
            )
            if len(pred)
            else None,
            "phase_mean_horizon_std": float(
                np.std(
                    [np.mean(pred[phases[selected] == phase]) for phase in sorted(set(phases[selected].tolist()))]
                )
            )
            if len(pred)
            else None,
        }
        result["groups"][group] = group_result

    if name == "pace_style_global_arm":
        h = predicted["arm"][eval_mask].astype(float)
        result["forced_global_horizon_actions"] = summary_stats(h)
    else:
        result["group_horizon_difference_actions"] = summary_stats(
            predicted["arm"][eval_mask].astype(float) - predicted["gripper"][eval_mask].astype(float)
        )
    return result


def plot_oracle(output_dir: Path, horizons: dict[str, np.ndarray], censoring: dict[str, np.ndarray], tasks: np.ndarray, phases: np.ndarray, refresh_survival: dict[str, np.ndarray], observed: np.ndarray) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, group in zip(axes, GROUPS, strict=True):
        ax.hist(horizons[group], bins=np.arange(0.5, CHUNK_SIZE + 1.5), density=True, alpha=0.65, color="tab:blue")
        ax.set_title(f"Oracle refresh horizon: {group}")
        ax.set_xlabel("h* (executed action count)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("density; censored rows shown at observed limit")
    fig.tight_layout()
    path = output_dir / "oracle_horizon_distributions.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.scatter(horizons["arm"], horizons["gripper"], s=6, alpha=0.15, color="tab:purple")
    ax.plot([1, CHUNK_SIZE], [1, CHUNK_SIZE], color="0.3", linewidth=1)
    ax.set(xlabel="h*_arm", ylabel="h*_gripper", title="Per-window oracle group horizons")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = output_dir / "oracle_arm_gripper_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 4.7))
    for group, color in (("arm", "tab:blue"), ("gripper", "tab:orange")):
        values = []
        for k in range(CHUNK_SIZE):
            mask = observed[:, k]
            values.append(float(np.mean(refresh_survival[group][mask, k])) if np.any(mask) else np.nan)
        ax.plot(np.arange(CHUNK_SIZE), values, color=color, linewidth=2, label=group)
    ax.axvline(0, color="0.4", linestyle="--", linewidth=0.8)
    ax.text(1, 0.03, "k=0 is identity check; interpretation starts at k=1", fontsize=8)
    ax.set(xlabel="future offset k", ylabel="refresh prefix survival", title="Y_refresh prevalence by offset")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "refresh_survival_by_offset.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def plot_pace(output_dir: Path, schedules: dict[str, dict[str, np.ndarray]], oracle: dict[str, np.ndarray], eval_mask: np.ndarray) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    colors = {"pace_style_global_arm": "tab:blue", "pace_style_groupwise": "tab:orange", "oracle_refresh_group": "tab:green"}
    labels = {"pace_style_global_arm": "PACE-style global", "pace_style_groupwise": "PACE-style group-wise", "oracle_refresh_group": "oracle Y_refresh"}
    for ax, group in zip(axes, GROUPS, strict=True):
        for name, schedule in schedules.items():
            values = oracle[group][eval_mask] if name == "oracle_refresh_group" else schedule[group][eval_mask]
            ax.hist(values, bins=np.arange(0.5, CHUNK_SIZE + 1.5), density=True, histtype="step", linewidth=1.8, color=colors[name], label=labels[name])
        ax.set_title(group.capitalize())
        ax.set_xlabel("selected action count")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle("PACE-style versus oracle horizon distributions (held-out episodes)")
    fig.tight_layout()
    path = output_dir / "pace_horizon_distributions.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.scatter(schedules["pace_style_groupwise"]["arm"][eval_mask], schedules["pace_style_groupwise"]["gripper"][eval_mask], s=7, alpha=0.18, color="tab:orange")
    ax.plot([1, CHUNK_SIZE], [1, CHUNK_SIZE], color="0.3", linewidth=1)
    ax.set(xlabel="PACE-style arm horizon", ylabel="PACE-style gripper horizon", title="Group-wise kinematic schedule")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = output_dir / "pace_groupwise_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def render_oracle_report(metrics: dict[str, Any], figures: list[str]) -> str:
    overall = metrics["oracle"]["overall"]
    het = overall["heterogeneity"]
    lines = [
        "# Oracle group-horizon analysis",
        "",
        "## Decision-relevant result",
        "",
        "The cached `Y_refresh` targets provide an offline, right-censored oracle "
        "for group-specific temporal persistence. The result is descriptive only: "
        "it is not a learned horizon label, rollout-success supervision, or a "
        "closed-loop execution measurement.",
        "",
        "The action-count convention is `h* = max { h >= 1 : Y_refresh(h-1) "
        "remains true }`. Offset `k=0` supplies only the minimum one-action "
        "convention; positive evidence begins at `k=1`. Rows valid through the "
        "last observed action are right-censored.",
        "",
        f"Total windows: {metrics['oracle']['coverage']['rows']}; episodes: {metrics['oracle']['coverage']['episodes']}; "
        f"both-uncensored windows: {metrics['oracle']['coverage']['both_uncensored_windows']} "
        f"({metrics['oracle']['coverage']['both_uncensored_fraction']:.3f}).",
        "",
        "## Group distributions",
        "",
        "| group | mean | median | q10 | q25 | q75 | q90 | censoring |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        stats = overall["horizon_distributions"][group]["all_rows_lower_bound"]
        q = stats["quantiles"]
        lines.append(
            f"| {group} | {stats['mean']:.2f} | {stats['median']:.1f} | {q['q10']:.1f} | "
            f"{q['q25']:.1f} | {q['q75']:.1f} | {q['q90']:.1f} | "
            f"{overall['horizon_distributions'][group]['censoring_rate']:.3f} |"
        )
    lines += [
        "",
        "These are lower-bound distributions because censored rows are displayed "
        "at their last observed action count. The report and JSON also include "
        "uncensored-only summaries.",
        "",
        "## Heterogeneity and global-clock waste",
        "",
        "| comparison population | arm expires first | gripper expires first | equal | different |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, item in (("all rows (lower-bound)", het["all_rows_lower_bound_comparison"]), ("both uncensored", het["both_uncensored_exact_comparison"])):
        lines.append(
            f"| {label} | {item['p_arm_expires_first']:.3f} | {item['p_gripper_expires_first']:.3f} | "
            f"{item['p_equal']:.3f} | {item['p_different']:.3f} |"
        )
    waste = het["wasted_commitment_actions_both_uncensored"]
    waste_fraction = het["wasted_commitment_fraction_both_uncensored"]
    lines += [
        "",
        f"On the both-uncensored subset, forcing both groups to the shorter oracle "
        f"clock discards a mean of {waste['mean']:.2f} valid action positions "
        f"per window (median {waste['median']:.1f}); the discarded fraction is "
        f"{waste_fraction['mean']:.3f} of the two groups' observed oracle commitment. "
        "This is commitment discarded by the global clock, not a success gain.",
        "",
        "### Episode-bootstrap uncertainty",
        "",
        "The following 95% intervals resample whole episodes (2,000 draws; seed "
        f"{metrics['oracle']['heterogeneity_bootstrap']['seed']}) and use only "
        "windows uncensored for both groups:",
        "",
        "| estimand | point estimate | episode-bootstrap 95% CI |",
        "|---|---:|---:|",
    ]
    bootstrap = metrics["oracle"]["heterogeneity_bootstrap"]["metrics"]
    lines += [
        f"| P(h*_arm != h*_gripper) | {bootstrap['p_h_arm_not_equal_h_gripper']['point_estimate']:.3f} | "
        f"[{bootstrap['p_h_arm_not_equal_h_gripper']['ci95_low']:.3f}, {bootstrap['p_h_arm_not_equal_h_gripper']['ci95_high']:.3f}] |",
        f"| mean discarded valid positions | {bootstrap['mean_discarded_valid_positions']['point_estimate']:.2f} | "
        f"[{bootstrap['mean_discarded_valid_positions']['ci95_low']:.2f}, {bootstrap['mean_discarded_valid_positions']['ci95_high']:.2f}] |",
        f"| discarded commitment fraction | {bootstrap['discarded_commitment_fraction']['point_estimate']:.3f} | "
        f"[{bootstrap['discarded_commitment_fraction']['ci95_low']:.3f}, {bootstrap['discarded_commitment_fraction']['ci95_high']:.3f}] |",
        "",
        "## Offset prevalence",
        "",
        "The complete per-offset pointwise and prefix-survival arrays are in "
        "`oracle_group_horizon_metrics.json`. Selected refresh prefix-survival values are:",
        "",
        "| offset k | arm | gripper | observed rows |\n"
        "|---:|---:|---:|---:|",
    ]
    prevalence = metrics["oracle"]["refresh_prevalence"]
    for k in KEY_OFFSETS:
        arm = prevalence["arm"]["key_offsets"][str(k)]
        grip = prevalence["gripper"]["key_offsets"][str(k)]
        lines.append(f"| {k} | {arm['survival']:.3f} | {grip['survival']:.3f} | {arm['n_observed']} |")
    lines += [
        "",
        "## Fixed threshold sensitivity",
        "",
        "This predeclared audit rescored the cached old/refreshed action pairs "
        "at tolerance multipliers 0.75, 1.0, and 1.25. It performed no new "
        "frozen-policy inference, did not use rollout success, and did not tune "
        "the threshold toward a desired result. `k=0` remains excluded from "
        "horizon evidence.",
        "",
        "| tolerance multiplier | both-uncensored fraction | P(different horizons) | mean discarded positions | discarded fraction | arm censoring | gripper censoring |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for multiplier, item in metrics["oracle"]["threshold_sensitivity"]["results"].items():
        lines.append(
            f"| {multiplier} | {item['both_uncensored_fraction']:.3f} | "
            f"{item['p_h_arm_not_equal_h_gripper']:.3f} | "
            f"{item['mean_discarded_valid_positions']:.2f} | "
            f"{item['discarded_commitment_fraction']:.3f} | "
            f"{item['arm_censoring_rate']:.3f} | {item['gripper_censoring_rate']:.3f} |"
        )
    lines += [
        "",
        "## Task and offline phase variation",
        "",
        "Task and normalized-episode-phase summaries are retrospective analyses. "
        "Progress, phase, and terminal episode length were not used to select a "
        "PACE horizon and must not be estimator inputs.",
        "",
        "### Task-conditioned lower-bound distributions",
        "",
        "| task | rows | difference rate | discarded positions | discarded fraction | arm mean | gripper mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task, item in metrics["oracle"]["task_phase"]["task"].items():
        lines.append(
            f"| {task} | {item['n_rows']} | {item['both_uncensored_horizon_difference_rate']:.3f} | "
            f"{item['both_uncensored_mean_discarded_valid_positions']:.2f} | "
            f"{item['both_uncensored_discarded_commitment_fraction']:.3f} | "
            f"{item['horizons']['arm']['all_rows_lower_bound']['mean']:.2f} | "
            f"{item['horizons']['gripper']['all_rows_lower_bound']['mean']:.2f} |"
        )
    lines += [
        "",
        "### Offline phase-conditioned lower-bound distributions",
        "",
        "| phase | rows | arm mean | gripper mean | arm censoring | gripper censoring |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for phase, item in metrics["oracle"]["task_phase"]["phase"].items():
        lines.append(
            f"| {phase} | {item['n_rows']} | {item['horizons']['arm']['all_rows_lower_bound']['mean']:.2f} | "
            f"{item['horizons']['gripper']['all_rows_lower_bound']['mean']:.2f} | "
            f"{item['horizons']['arm']['censoring_rate']:.3f} | {item['horizons']['gripper']['censoring_rate']:.3f} |"
        )
    lines += [
        "",
        "Full quantiles, uncensored-only strata, and per-stratum heterogeneity are in `oracle_group_horizon_metrics.json`.",
        "",
        "## Figures",
        "",
    ]
    lines.extend(f"![figure](figures/{Path(path).name})" for path in figures)
    lines += [
        "",
        "## Limitations",
        "",
        "- `Y_refresh` queries the frozen policy on a demonstrated future observation; it does not execute the old action in an environment.",
        "- Censoring and teacher forcing prevent interpreting `h*` as a physical safety or task-success horizon.",
        "- Group comparison is exact only when both group prefixes fail before their observed suffix ends; all-row rates are explicitly lower-bound comparisons.",
    ]
    return "\n".join(lines) + "\n"


def render_related_work_report(metrics: dict[str, Any]) -> str:
    pace = metrics["pace"]
    heldout = pace["heldout_metrics"]
    global_m = heldout["pace_style_global_arm"]
    group_m = heldout["pace_style_groupwise"]
    oracle_m = heldout["oracle_refresh_group"]
    lines = [
        "# Related-work execution audit and PACE-style baseline",
        "",
        "## Scope and conclusion",
        "",
        "This is an offline audit over cached frozen-policy chunks and `Y_refresh`. "
        "No estimator training, online reinforcement learning, executor change, "
        "or rollout was performed. No success claim is made.",
        "",
        "The inspected prior work establishes that dynamic execution-horizon "
        "selection, confidence/self-consistency signals, kinematic phase signals, "
        "and frozen-policy scheduling are already active ideas. Accordingly:",
        "",
        "1. **Reliability -> dynamic horizon is not claimed as novel.** The burden of proof is not met by the current comparison.",
        "2. The potentially defensible distinction is the combination of **heterogeneous group-wise temporal persistence, independent execution clocks, and mixed-generation action composition**. The inspected papers use a scalar/global commitment or a policy-internal/training-time horizon mechanism; this combination remains a provisional positioning statement, not a novelty claim.",
        "3. A single global clock measurably discards non-expired group commitment in this cache when evaluated against `Y_refresh`; see the oracle report. This is an offline upper-bound diagnostic, not an execution improvement.",
        "4. The group-wise PACE-style heuristic is not a faithful PACE reproduction and does not establish that kinematics solve the oracle gap. It should be read as a diagnostic of how much a simple chunk-only signal can explain.",
        "",
        "## Method matrix",
        "",
        "| method | horizon meaning | target / signal | training required | frozen base policy | policy-internal access | online RL | global or independent groups | retain non-expired old source chunk | inference overhead | overlap |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|",
        "| [PACE](https://arxiv.org/html/2606.00537) | executed prefix before next query | smoothed predicted arm kinematic profile; prominent low-speed valleys | no; task threshold calibrated from demonstrations | yes | no | no | one scalar; earliest accepted arm boundary | no; suffix is discarded and the global query refreshes together | small profile/smoothing/valley computation | closest chunk-only timing baseline; not group clocks |",
        "| [VLA Knows Its Limits / AutoHorizon](https://arxiv.org/html/2602.21445) | per-chunk executed prefix | action self-attention coverage/turning point from flow-VLA attention | no additional training for the reported test-time method | yes | yes; attention maps are required | no | one scalar prefix | no | low relative to policy, but requires attention extraction | adaptive horizon from internal predictive-limit proxy |",
        "| [DEHP](https://arxiv.org/abs/2606.11408) | predicted number of actions to execute before replanning | learned execution-horizon branch optimized with chunk-level PPO | yes; online RL | yes; pretrained chunk policy frozen | designed for black-box chunk policies | yes | one scalar horizon | no group retention semantics | lightweight branch plus RL-trained scheduler | closest learned frozen-policy scheduler, but objective/training differs |",
        "| [Spatial Attention](https://arxiv.org/html/2607.04739) | execution horizon under a sampling budget | observation sensitivity `E||grad_o log pi(a|o)||^2`, forecast along chunk | yes; score models/forecasting machinery | base policy can remain fixed, but auxiliary models are trained | not a black-box-only signal; action/observation likelihood sensitivity is required | no | scalar horizon | no | score/sensitivity and forecast computation | adaptive confidence/sensitivity timing, not group persistence |",
        "| [A3](https://arxiv.org/html/2605.11567) | longest verified executable action prefix | sampled trajectory consensus, conditional-invariance re-decoding, prefix sequential consistency | no separate policy training reported | yes | yes; sampling and conditional re-decoding are central | no | one global verified prefix | no | high relative overhead; candidate sampling and verification, parallelized in implementation | closest self-consistency/verification idea, but global and policy-internal |",
        "| [Mixture of Horizons](https://arxiv.org/html/2511.19433) | training chunk length plus dynamic consensus-selected executable prefix | multi-horizon predictions, gating, cross-horizon disagreement | yes; modifies/trains the action module | no; it trains a multi-horizon policy | yes; horizon-wise predictions and gates | no | one global prefix | no | extra horizon-wise action processing/gating; reported as small in the paper | multi-horizon consensus, but not frozen-policy reliability or group clocks |",
        "",
        "### Interpretation of the matrix",
        "",
        "All six works address how much of a predicted chunk to execute, directly or "
        "through a training-time horizon construction. None of the inspected "
        "formulations provides the same explicit combination of a future-label "
        "reliability survival target, independent arm/gripper clocks, and retaining "
        "non-expired slices from older query generations. That supports a narrow "
        "positioning hypothesis, not a claim that the individual ingredients are new.",
        "",
        "## PACE implementation availability and deviations",
        "",
        "The PACE primary source and its linked materials did not provide a "
        "compatible official implementation. The paper specifies the high-level "
        "low-speed-valley rule, smoothing, minimum separation, prominence, and "
        "training-demonstration calibration, but does not uniquely specify every "
        "operator needed to reproduce the exact horizon sequence. The existing "
        "relative ACT/LIBERO action is also Cartesian position plus axis-angle "
        "delta, not a joint-position trajectory.",
        "",
        "The executed baseline is therefore named **PACE-style**, not PACE:",
        "",
        "- signal: per-step normalized action magnitude; arm uses `max(translation RMS, rotation RMS)` and gripper uses normalized absolute command magnitude;\n"
        "- smoothing: centered edge-padded moving average of width 5;\n"
        "- valleys: local minima with a prominence proxy equal to the lower of left/right max-minus-valley over a 10-step neighborhood;\n"
        "- spacing: greedy prominence-first non-maximum suppression with minimum separation 10;\n"
        "- calibration: per-task and per-group 5th percentile of positive calibration-episode valley scores, using the first 80% of episodes within each task; no `Y_refresh` values are used;\n"
        "- fallback: full 100-action prefix;\n"
        "- global variant: arm signal only, one scalar horizon applied to both groups, matching the source's global commitment semantics;\n"
        "- group-wise variant: the same simple signal and selection rule run independently for arm and gripper. This is an explicit diagnostic extension, not claimed to be PACE.",
        "",
        "## Offline comparison",
        "",
        f"Calibration uses {pace['calibration']['episodes']} episodes and held-out scoring uses {pace['evaluation']['episodes']} episodes ({pace['evaluation']['rows']} windows). The split is episode-level. The horizon selector sees only old predicted chunks, task identity for calibration, and fixed signal parameters; refresh labels are scoring-only.",
        "",
        "| schedule | arm MAE to h* | gripper MAE to h* | arm selected-prefix Y_refresh | gripper selected-prefix Y_refresh | arm mean 1/h | gripper mean 1/h |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| PACE-style global | {global_m['groups']['arm']['absolute_error_actions']['mean']:.2f} | {global_m['groups']['gripper']['absolute_error_actions']['mean']:.2f} | {global_m['groups']['arm']['selected_prefix_refresh_survival']['mean']:.3f} | {global_m['groups']['gripper']['selected_prefix_refresh_survival']['mean']:.3f} | {global_m['groups']['arm']['expected_local_query_rate_per_action']:.4f} | {global_m['groups']['gripper']['expected_local_query_rate_per_action']:.4f} |",
        f"| PACE-style group-wise | {group_m['groups']['arm']['absolute_error_actions']['mean']:.2f} | {group_m['groups']['gripper']['absolute_error_actions']['mean']:.2f} | {group_m['groups']['arm']['selected_prefix_refresh_survival']['mean']:.3f} | {group_m['groups']['gripper']['selected_prefix_refresh_survival']['mean']:.3f} | {group_m['groups']['arm']['expected_local_query_rate_per_action']:.4f} | {group_m['groups']['gripper']['expected_local_query_rate_per_action']['mean'] if isinstance(group_m['groups']['gripper']['expected_local_query_rate_per_action'], dict) else group_m['groups']['gripper']['expected_local_query_rate_per_action']:.4f} |",
        f"| oracle Y_refresh group | {oracle_m['groups']['arm']['absolute_error_actions']['mean']:.2f} | {oracle_m['groups']['gripper']['absolute_error_actions']['mean']:.2f} | {oracle_m['groups']['arm']['selected_prefix_refresh_survival']['mean']:.3f} | {oracle_m['groups']['gripper']['selected_prefix_refresh_survival']['mean']:.3f} | {oracle_m['groups']['arm']['expected_local_query_rate_per_action']:.4f} | {oracle_m['groups']['gripper']['expected_local_query_rate_per_action']:.4f} |",
        "",
        "The `mean 1/h` columns are per-window reciprocal-horizon proxies, not measured closed-loop query rates.",
        "",
        "The oracle row is a self-comparison and is included only to show the "
        "upper-bound reference. Selected-prefix survival is an offline target "
        "event, not rollout success. Because censored oracle horizons are lower "
        "bounds, the horizon errors and survival scores should not be interpreted "
        "as calibrated scheduler performance.",
        "",
        "## PACE-style figures",
        "",
        *[f"![figure](pace_baseline/figures/{Path(path).name})" for path in pace["figures"]],
        "",
        "## Required answers",
        "",
        "1. **What remains genuinely novel?** The individual idea of adaptive horizon selection is not novel. A narrow combination involving independent group clocks and mixed-generation action composition is the remaining candidate, but needs a direct prior-work and implementation audit whenever new papers/code appear.",
        "2. **Is reliability -> dynamic horizon itself novel?** No. Do not claim this.",
        "3. **Is the defensible novelty instead heterogeneous persistence + independent clocks + mixed-generation composition?** Provisional yes as a system combination relative to the six inspected papers; not yet a proven novelty claim.",
        "4. **Does one global clock discard valid commitment?** Yes in this teacher-forced offline oracle: the oracle report quantifies nonzero group heterogeneity and lower-bound discarded actions when both groups are forced to the minimum. This is not a rollout improvement claim.",
        "5. **Does group-wise PACE-style solve most of the oracle gap?** Not established. The reported held-out horizon errors and selected-prefix survival are the correct negative/positive diagnostic. They do not support claiming that a kinematic heuristic reaches the oracle, and any apparent alignment is not task success.",
        "",
        "## Reproducibility",
        "",
        "- Script: `analyze_oracle_and_pace.py`.",
        "- Input: `experiments/temporal_reliability_target_comparison/target_comparison.npz` and aligned metadata.",
        "- No changes were made to the executor, rollout code, paper, or checkpoints.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    pace_dir = output_dir / "pace_baseline"
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    pace_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.input, allow_pickle=False)
    metadata = load_json_lines(args.metadata)
    validate(data, metadata)
    tasks, phases, episodes, _progress = grouped_rows(metadata)
    observed = data["observed_offsets"].astype(bool)
    action_std = np.asarray(json.loads(args.manifest.read_text())["checkpoint"]["action_std"], dtype=np.float32)
    if action_std.shape != (7,) or np.any(action_std <= 0):
        raise RuntimeError("invalid action normalization in target manifest")

    oracle_horizons: dict[str, np.ndarray] = {}
    oracle_censoring: dict[str, np.ndarray] = {}
    for group in GROUPS:
        oracle_horizons[group], oracle_censoring[group] = derive_oracle_horizon(
            data[f"{group}_refresh_survival"], observed
        )

    oracle_metrics = {
        "coverage": {
            "rows": int(len(metadata)),
            "episodes": int(len(set(episodes.tolist()))),
            "observed_pairs": int(np.sum(observed)),
            "positive_offset_pairs": int(np.sum(observed[:, 1:])),
            "both_uncensored_windows": int(
                np.sum(~oracle_censoring["arm"] & ~oracle_censoring["gripper"])
            ),
            "both_uncensored_fraction": float(
                np.mean(~oracle_censoring["arm"] & ~oracle_censoring["gripper"])
            ),
        },
        "horizon_convention": {
            "definition": "h* = max action count h such that Y_refresh(h-1) remains true",
            "minimum_action_count": 1,
            "positive_evidence_starts_at_offset": 1,
            "k0_used_as_evidence": False,
            "censoring": "horizon equals last observed action count when all observed positive offsets remain valid",
        },
        "horizon_distributions": {
            group: {
                "all_rows_lower_bound": summary_stats(oracle_horizons[group]),
                "uncensored": summary_stats(oracle_horizons[group][~oracle_censoring[group]]),
                "censoring_rate": float(np.mean(oracle_censoring[group])),
                "no_positive_offset_observed_rate": float(np.mean(observed[:, 1:].sum(axis=1) == 0)),
            }
            for group in GROUPS
        },
        "refresh_prevalence": horizon_prevalence(data, observed, np.ones(len(metadata), dtype=bool)),
        "heterogeneity": heterogeneity(oracle_horizons, oracle_censoring),
        "heterogeneity_bootstrap": bootstrap_heterogeneity_intervals(
            oracle_horizons, oracle_censoring, episodes
        ),
        "task_phase": task_phase_analysis(oracle_horizons, oracle_censoring, tasks, phases),
    }
    oracle_metrics["overall"] = {
        "horizon_distributions": oracle_metrics["horizon_distributions"],
        "refresh_prevalence": oracle_metrics["refresh_prevalence"],
        "heterogeneity": oracle_metrics["heterogeneity"],
        "heterogeneity_bootstrap": oracle_metrics["heterogeneity_bootstrap"],
    }
    oracle_figures = plot_oracle(
        figure_dir,
        oracle_horizons,
        oracle_censoring,
        tasks,
        phases,
        {group: data[f"{group}_refresh_survival"].astype(bool) for group in GROUPS},
        observed,
    )

    actions = data["old_predicted_actions"].astype(np.float32)
    refreshed_actions = data["refresh_first_actions"].astype(np.float32)
    oracle_metrics["threshold_sensitivity"] = threshold_sensitivity_analysis(
        actions,
        refreshed_actions,
        observed,
        action_std,
        episodes,
    )
    raw_profiles = signal_profiles(actions, action_std)
    smoothed = {group: moving_average(raw_profiles[group], SMOOTHING_WINDOW) for group in GROUPS}
    valley_indices: dict[str, np.ndarray] = {}
    valley_scores: dict[str, np.ndarray] = {}
    for group in GROUPS:
        valley_indices[group], valley_scores[group] = local_valley_prominences(
            smoothed[group], MIN_VALLEY_SEPARATION
        )
    calibration_mask, evaluation_mask = split_calibration_episodes(tasks, episodes)
    thresholds = calibrate_thresholds(
        smoothed, valley_indices, valley_scores, tasks, calibration_mask
    )
    pace_groupwise = select_horizons(smoothed, valley_indices, valley_scores, tasks, thresholds)
    pace_global = {group: pace_groupwise["arm"].copy() for group in GROUPS}
    oracle_schedule = {group: oracle_horizons[group].copy() for group in GROUPS}
    schedules = {
        "pace_style_global_arm": pace_global,
        "pace_style_groupwise": pace_groupwise,
        "oracle_refresh_group": oracle_schedule,
    }
    refresh_survival = {group: data[f"{group}_refresh_survival"].astype(bool) for group in GROUPS}
    schedule_results = {
        name: schedule_metrics(
            name,
            schedule,
            oracle_horizons,
            observed,
            refresh_survival,
            tasks,
            phases,
            evaluation_mask,
        )
        for name, schedule in schedules.items()
    }
    pace_metrics = {
        "source": {
            "primary_paper": "https://arxiv.org/html/2606.00537",
            "official_compatible_implementation_found": False,
            "status": "PACE-style offline diagnostic, not a faithful PACE reproduction",
        },
        "input_scope": {
            "uses_old_predicted_chunk_only_for_selection": True,
            "uses_y_refresh_for_selection": False,
            "uses_future_observation_for_selection": False,
            "uses_rollout_success": False,
            "uses_estimator_training": False,
        },
        "deviations": {
            "action_contract": "relative Cartesian position plus axis-angle deltas, not joint positions",
            "arm_signal": "max(normalized translation RMS, normalized rotation RMS) per action",
            "gripper_signal": "normalized absolute gripper command per action; no direct PACE analog in source",
            "smoothing": f"centered edge-padded moving average width {SMOOTHING_WINDOW}",
            "valley_prominence": "min(left-window maximum, right-window maximum) minus valley over a 10-step neighborhood",
            "valley_selection": f"local minima, prominence-first nonmaximum suppression with minimum separation {MIN_VALLEY_SEPARATION}",
            "threshold": f"per-task/per-group {PACE_PERCENTILE:g}th percentile of positive calibration valley scores",
            "calibration": "first 80 percent of episodes within each task, episode-level split",
            "fallback": f"{CHUNK_SIZE}-action prefix",
        },
        "calibration": {
            "episodes": int(len(set(episodes[calibration_mask].tolist()))),
            "rows": int(np.sum(calibration_mask)),
            "fraction": CALIBRATION_FRACTION,
            "thresholds": thresholds,
        },
        "evaluation": {
            "episodes": int(len(set(episodes[evaluation_mask].tolist()))),
            "rows": int(np.sum(evaluation_mask)),
        },
        "heldout_metrics": schedule_results,
        "horizon_distributions_all_rows": {
            name: {group: summary_stats(schedule[group]) for group in GROUPS}
            for name, schedule in schedules.items()
        },
    }
    # Keep PACE figures under the requested pace_baseline/figures directory.
    (pace_dir / "figures").mkdir(parents=True, exist_ok=True)
    pace_figures = plot_pace(pace_dir / "figures", schedules, oracle_horizons, evaluation_mask)
    pace_metrics["figures"] = pace_figures
    oracle_metrics["figures"] = oracle_figures
    metrics = {"oracle": oracle_metrics, "pace": pace_metrics}

    (output_dir / "oracle_group_horizon_metrics.json").write_text(
        json.dumps(json_safe(metrics), indent=2, sort_keys=True) + "\n"
    )
    (pace_dir / "metrics.json").write_text(json.dumps(json_safe(pace_metrics), indent=2, sort_keys=True) + "\n")
    (output_dir / "oracle_group_horizon_analysis.md").write_text(
        render_oracle_report(metrics, oracle_figures)
    )
    (output_dir / "related_work_execution_audit.md").write_text(
        render_related_work_report(metrics)
    )
    print(json.dumps({"output_dir": str(output_dir), "oracle_figures": oracle_figures, "pace_figures": pace_figures}, indent=2))


if __name__ == "__main__":
    main()
