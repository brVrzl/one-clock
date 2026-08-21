#!/usr/bin/env python3
"""Recompute reliability targets, sensitivity, information loss, and smoothness.

Run with the LeRobot audit environment because reading the pinned demonstration
Parquet files requires PyArrow.  No policy inference or environment rollout is
performed and historical artifacts are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
DATASET = Path("/home/thor/datasets/libero_object_25_08_23_lerobotv2.1")
CHECKPOINT = Path("/home/thor/projects/checkpoints/zeromidnight_act_libero_object")
RELIABILITY = ROOT / "experiments/temporal_reliability/reliability_dataset.npz"
BUNDLE = ROOT / "experiments/dynamic_reliability_horizon/artifact_handoff/minimal_y_refresh_training_bundle.npz"
ACTION_STD = np.asarray(
    [0.2681190073490143, 0.4384443759918213, 0.4475117325782776,
     0.024448219686746597, 0.04936208948493004, 0.042103495448827744,
     0.9974462985992432],
    dtype=np.float64,
)
THRESHOLDS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
GROUPS = ("arm", "gripper")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "research/audit_outputs")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_episode_actions(dataset: Path, episode_ids: np.ndarray) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for episode_id in sorted(set(int(value) for value in episode_ids)):
        path = dataset / "data/chunk-000" / f"episode_{episode_id:06d}.parquet"
        table = pq.read_table(path, columns=["action", "frame_index", "episode_index"])
        data = table.to_pydict()
        if set(int(value) for value in data["episode_index"]) != {episode_id}:
            raise RuntimeError(f"episode identity mismatch: {path}")
        if data["frame_index"] != list(range(len(data["frame_index"]))):
            raise RuntimeError(f"frame order mismatch: {path}")
        result[episode_id] = np.asarray(data["action"], dtype=np.float32)
    return result


def reconstruct_targets(data: Any, actions: dict[int, np.ndarray]) -> np.ndarray:
    target = np.full((len(data["episode_index"]), 100, 7), np.nan, dtype=np.float32)
    for row, (episode_id, frame, length) in enumerate(
        zip(data["episode_index"], data["frame_index"], data["episode_length"], strict=True)
    ):
        sequence = actions[int(episode_id)]
        if len(sequence) != int(length):
            raise RuntimeError(f"saved episode length mismatch for episode {episode_id}")
        suffix = sequence[int(frame) : int(frame) + 100]
        target[row, : len(suffix)] = suffix
    return target


def labels_for_threshold(
    data: Any,
    threshold: float,
    *,
    gripper_mode: str = "magnitude_and_sign",
) -> dict[str, np.ndarray]:
    observed = np.asarray(data["observed_offsets"], dtype=bool)
    arm = (
        (data["arm_translation_error"] <= threshold)
        & (data["arm_rotation_error"] <= threshold)
        & observed
    )
    if gripper_mode == "magnitude_and_sign":
        gripper = (
            (data["gripper_normalized_absolute_error"] <= threshold)
            & data["gripper_sign_match"]
            & observed
        )
    elif gripper_mode == "sign_only":
        gripper = data["gripper_sign_match"] & observed
    elif gripper_mode == "magnitude_only":
        gripper = (data["gripper_normalized_absolute_error"] <= threshold) & observed
    else:
        raise ValueError(gripper_mode)
    return {
        "arm_pointwise": arm,
        "gripper_pointwise": gripper,
        "arm_survival": np.logical_and.accumulate(arm, axis=1) & observed,
        "gripper_survival": np.logical_and.accumulate(gripper, axis=1) & observed,
    }


def binary_entropy(rate: float) -> float:
    if rate <= 0.0 or rate >= 1.0:
        return 0.0
    return float(-rate * math.log2(rate) - (1.0 - rate) * math.log2(1.0 - rate))


def decoded_prefix_length(survival: np.ndarray, observed: np.ndarray, *, minimum_one: bool) -> np.ndarray:
    values = np.sum(survival & observed, axis=1).astype(np.int32)
    if minimum_one:
        values = np.maximum(values, 1)
    return values


def segmentation_summary(values: np.ndarray, progress: np.ndarray, bins: int) -> list[dict[str, Any]]:
    codes = np.minimum((progress * bins).astype(int), bins - 1)
    return [
        {
            "bin": index,
            "progress_interval": [index / bins, (index + 1) / bins],
            "samples": int(np.sum(codes == index)),
            "mean_prefix_length": float(np.mean(values[codes == index])),
            "median_prefix_length": float(np.median(values[codes == index])),
        }
        for index in range(bins)
        if np.any(codes == index)
    ]


def episode_balanced_km_auc(
    pointwise: np.ndarray,
    observed: np.ndarray,
    episodes: np.ndarray,
    selected: np.ndarray,
    horizon: int = 38,
) -> float:
    episode_ids = np.unique(episodes[selected])
    risks = np.zeros(horizon, dtype=np.float64)
    events = np.zeros(horizon, dtype=np.float64)
    for episode_id in episode_ids:
        rows = np.flatnonzero(selected & (episodes == episode_id))
        if not len(rows):
            continue
        weight = 1.0 / len(rows)
        alive = np.ones(len(rows), dtype=bool)
        for offset in range(horizon):
            eligible = alive & observed[rows, offset]
            risks[offset] += weight * eligible.sum()
            failed = eligible & ~pointwise[rows, offset]
            events[offset] += weight * failed.sum()
            alive[failed] = False
    finite = risks > 0
    curve = np.cumprod(1.0 - np.divide(events[finite], risks[finite]))
    if len(curve) < 2:
        return float("nan")
    return float(np.trapezoid(curve, np.arange(len(curve))) / (len(curve) - 1))


def km_segmentation(
    pointwise: np.ndarray,
    observed: np.ndarray,
    episodes: np.ndarray,
    progress: np.ndarray,
    bins: int,
) -> list[dict[str, Any]]:
    codes = np.minimum((progress * bins).astype(int), bins - 1)
    return [
        {
            "bin": index,
            "progress_interval": [index / bins, (index + 1) / bins],
            "samples": int(np.sum(codes == index)),
            "episode_balanced_km_auc_k0_37": episode_balanced_km_auc(
                pointwise, observed, episodes, codes == index
            ),
        }
        for index in range(bins)
        if np.any(codes == index)
    ]


def smoothness_features(
    episodes: np.ndarray,
    frames: np.ndarray,
    progress: np.ndarray,
    actions: dict[int, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    names = [
        "progress",
        "arm_velocity",
        "arm_acceleration",
        "arm_jerk",
        "arm_curvature",
        "gripper_delta",
        "gripper_sign_change",
        "future_arm_velocity_5",
        "future_arm_variance_5",
        "future_gripper_change_rate_5",
    ]
    features = np.zeros((len(episodes), len(names)), dtype=np.float64)
    features[:, 0] = progress
    for row, (episode_id, frame) in enumerate(zip(episodes, frames, strict=True)):
        sequence = actions[int(episode_id)].astype(np.float64)
        t = int(frame)
        current = sequence[t]
        previous = sequence[max(0, t - 1)]
        previous2 = sequence[max(0, t - 2)]
        previous3 = sequence[max(0, t - 3)]
        v = (current[:6] - previous[:6]) / ACTION_STD[:6]
        v_prev = (previous[:6] - previous2[:6]) / ACTION_STD[:6]
        v_prev2 = (previous2[:6] - previous3[:6]) / ACTION_STD[:6]
        acceleration = v - v_prev
        jerk = acceleration - (v_prev - v_prev2)
        denom = np.linalg.norm(v) * np.linalg.norm(v_prev)
        curvature = 0.0 if denom < 1e-12 else math.acos(float(np.clip(np.dot(v, v_prev) / denom, -1.0, 1.0)))
        features[row, 1] = np.linalg.norm(v) / math.sqrt(6)
        features[row, 2] = np.linalg.norm(acceleration) / math.sqrt(6)
        features[row, 3] = np.linalg.norm(jerk) / math.sqrt(6)
        features[row, 4] = curvature
        features[row, 5] = abs(current[6] - previous[6]) / ACTION_STD[6]
        features[row, 6] = float((current[6] >= 0.0) != (previous[6] >= 0.0))
        future = sequence[t : min(len(sequence), t + 6)]
        if len(future) > 1:
            future_velocity = np.diff(future[:, :6], axis=0) / ACTION_STD[:6]
            features[row, 7] = float(np.mean(np.linalg.norm(future_velocity, axis=1) / math.sqrt(6)))
            features[row, 8] = float(np.mean(np.var(future[:, :6] / ACTION_STD[:6], axis=0)))
            features[row, 9] = float(np.mean((future[1:, 6] >= 0.0) != (future[:-1, 6] >= 0.0)))
    return features, names


def ridge_fit_predict(
    features: np.ndarray,
    target: np.ndarray,
    split: np.ndarray,
    feature_indices: list[int],
) -> dict[str, Any]:
    x = features[:, feature_indices]
    train = split == 0
    validation = split == 1
    test = split == 2
    mean = x[train].mean(axis=0)
    std = x[train].std(axis=0)
    std[std < 1e-9] = 1.0
    scaled = (x - mean) / std
    design = np.column_stack([np.ones(len(x)), scaled])
    best: tuple[float, float, np.ndarray] | None = None
    for regularization in (0.0, 0.01, 0.1, 1.0, 10.0, 100.0):
        penalty = np.eye(design.shape[1]) * regularization
        penalty[0, 0] = 0.0
        weights = np.linalg.pinv(design[train].T @ design[train] + penalty) @ design[train].T @ target[train]
        prediction = np.clip(design @ weights, 1.0, 100.0)
        validation_mae = float(np.mean(np.abs(prediction[validation] - target[validation])))
        item = (validation_mae, regularization, prediction)
        if best is None or item[:2] < best[:2]:
            best = item
    assert best is not None
    prediction = best[2]
    rho = spearmanr(prediction[test], target[test]).statistic
    residual = target[test] - prediction[test]
    baseline = target[test] - target[train].mean()
    return {
        "features": feature_indices,
        "regularization_selected_on_validation": best[1],
        "validation_mae": best[0],
        "test_samples": int(test.sum()),
        "test_mae": float(np.mean(np.abs(residual))),
        "test_median_absolute_error": float(np.median(np.abs(residual))),
        "test_within_plus_minus_2": float(np.mean(np.abs(residual) <= 2.0)),
        "test_within_plus_minus_5": float(np.mean(np.abs(residual) <= 5.0)),
        "test_spearman": None if not np.isfinite(rho) else float(rho),
        "test_r2": float(1.0 - np.sum(residual**2) / np.sum(baseline**2)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(RELIABILITY, allow_pickle=False)
    bundle = np.load(BUNDLE, allow_pickle=False)
    issues: list[str] = []
    observed = np.asarray(data["observed_offsets"], dtype=bool)
    if not np.all(np.diff(observed.astype(np.int8), axis=1) <= 0):
        issues.append("observed_mask_not_contiguous")
    if not np.array_equal(bundle["source_chunk_actions"], data["predicted_actions"]):
        issues.append("refresh_bundle_source_chunk_differs_from_demo_target_cache")
    if not np.array_equal(bundle["episode_index"], data["episode_index"]):
        issues.append("refresh_bundle_episode_alignment")

    actions = load_episode_actions(args.dataset, data["episode_index"])
    target = reconstruct_targets(data, actions)
    predicted = data["predicted_actions"].astype(np.float64)
    difference = predicted - target
    translation_error = np.sqrt(np.mean((difference[:, :, :3] / ACTION_STD[:3]) ** 2, axis=2))
    rotation_error = np.sqrt(np.mean((difference[:, :, 3:6] / ACTION_STD[3:6]) ** 2, axis=2))
    gripper_error = np.abs(difference[:, :, 6]) / ACTION_STD[6]
    sign_match = (predicted[:, :, 6] >= 0.0) == (target[:, :, 6] >= 0.0)
    error_deltas = {
        "arm_translation_max_abs": float(np.nanmax(np.abs(translation_error - data["arm_translation_error"]))),
        "arm_rotation_max_abs": float(np.nanmax(np.abs(rotation_error - data["arm_rotation_error"]))),
        "gripper_max_abs": float(np.nanmax(np.abs(gripper_error - data["gripper_normalized_absolute_error"]))),
        "gripper_sign_mismatches": int(np.sum((sign_match != data["gripper_sign_match"]) & observed)),
    }
    labels = labels_for_threshold(data, 1.0)
    for key in ("arm_pointwise", "gripper_pointwise", "arm_survival", "gripper_survival"):
        saved_key = key.replace("pointwise", "pointwise_valid").replace("survival", "survival_valid")
        if not np.array_equal(labels[key], data[saved_key]):
            issues.append(f"saved_label_mismatch_{key}")

    sensitivity: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        current = labels_for_threshold(data, threshold)
        sensitivity[str(threshold)] = {}
        for group in GROUPS:
            pointwise = current[f"{group}_pointwise"]
            survival = current[f"{group}_survival"]
            horizon = decoded_prefix_length(survival, observed, minimum_one=True)
            point_rate = float(pointwise[observed].mean())
            survival_rate = float(survival[observed].mean())
            recoveries = pointwise & ~survival & observed
            sensitivity[str(threshold)][group] = {
                "pointwise_positive_rate": point_rate,
                "pointwise_binary_entropy_bits": binary_entropy(point_rate),
                "prefix_positive_rate": survival_rate,
                "prefix_binary_entropy_bits": binary_entropy(survival_rate),
                "mean_decoded_prefix_length": float(horizon.mean()),
                "median_decoded_prefix_length": float(np.median(horizon)),
                "pointwise_positive_cells_erased_by_prefix_fraction": float(recoveries.sum() / max(1, pointwise.sum())),
                "samples_with_post_failure_recovery_fraction": float(np.mean(np.any(recoveries, axis=1))),
                "thirds": segmentation_summary(horizon, data["progress"], 3),
                "quartiles": segmentation_summary(horizon, data["progress"], 4),
                "deciles": segmentation_summary(horizon, data["progress"], 10),
                "thirds_censor_aware": km_segmentation(
                    pointwise, observed, data["episode_index"], data["progress"], 3
                ),
                "quartiles_censor_aware": km_segmentation(
                    pointwise, observed, data["episode_index"], data["progress"], 4
                ),
                "deciles_censor_aware": km_segmentation(
                    pointwise, observed, data["episode_index"], data["progress"], 10
                ),
                "continuous_progress_spearman": float(spearmanr(data["progress"], horizon).statistic),
                "task_mean_prefix_lengths": {
                    str(task): float(horizon[data["task_index"] == task].mean()) for task in range(10)
                },
            }
    gripper_variants: dict[str, Any] = {}
    for mode in ("magnitude_and_sign", "sign_only", "magnitude_only"):
        current = labels_for_threshold(data, 1.0, gripper_mode=mode)
        horizon = decoded_prefix_length(current["gripper_survival"], observed, minimum_one=True)
        gripper_variants[mode] = {
            "pointwise_positive_rate": float(current["gripper_pointwise"][observed].mean()),
            "prefix_positive_rate": float(current["gripper_survival"][observed].mean()),
            "mean_prefix_length": float(horizon.mean()),
            "thirds": segmentation_summary(horizon, data["progress"], 3),
        }

    demo_horizons = {
        group: decoded_prefix_length(labels[f"{group}_survival"], observed, minimum_one=True)
        for group in GROUPS
    }
    refresh_observed = bundle["label_observed"].astype(bool)
    refresh = bundle["y_refresh"].astype(bool)
    if np.any((refresh[:, :, 1:] & ~refresh[:, :, :-1]) & refresh_observed[:, :, 1:]):
        issues.append("refresh_survival_not_monotone")
    refresh_horizons = {
        group: 1 + np.sum(refresh[:, index, :] & refresh_observed[:, index, :], axis=1)
        for index, group in enumerate(GROUPS)
    }
    target_comparison = {}
    for index, group in enumerate(GROUPS):
        common_demo = labels[f"{group}_survival"][:, 1:]
        mask = refresh_observed[:, index, :]
        agreement = common_demo == refresh[:, index, :]
        target_comparison[group] = {
            "cell_agreement": float(agreement[mask].mean()),
            "demo_positive_refresh_negative": int(np.sum(common_demo & ~refresh[:, index, :] & mask)),
            "demo_negative_refresh_positive": int(np.sum(~common_demo & refresh[:, index, :] & mask)),
            "horizon_spearman": float(spearmanr(demo_horizons[group], refresh_horizons[group]).statistic),
            "mean_demo_horizon": float(demo_horizons[group].mean()),
            "mean_refresh_horizon": float(refresh_horizons[group].mean()),
            "mean_absolute_horizon_difference": float(np.mean(np.abs(demo_horizons[group] - refresh_horizons[group]))),
        }

    features, feature_names = smoothness_features(
        data["episode_index"], data["frame_index"], data["progress"], actions
    )
    split = bundle["split_membership"].astype(np.int8)
    correlation: dict[str, Any] = {}
    ridge: dict[str, Any] = {}
    causal_indices = list(range(0, 7))
    causal_by_group = {"arm": [0, 1, 2, 3, 4], "gripper": [0, 5, 6]}
    explanatory_by_group = {"arm": [0, 1, 2, 3, 4, 7, 8], "gripper": [0, 5, 6, 9]}
    for group in GROUPS:
        correlation[group] = {
            feature_names[index]: {
                "demo_horizon_spearman": float(spearmanr(features[:, index], demo_horizons[group]).statistic),
                "refresh_horizon_spearman": float(spearmanr(features[:, index], refresh_horizons[group]).statistic),
            }
            for index in range(len(feature_names))
        }
        ridge[group] = {
            "demo_causal_history": ridge_fit_predict(features, demo_horizons[group], split, causal_by_group[group]),
            "demo_with_future_smoothness": ridge_fit_predict(features, demo_horizons[group], split, explanatory_by_group[group]),
            "refresh_causal_history": ridge_fit_predict(features, refresh_horizons[group], split, causal_by_group[group]),
            "refresh_with_future_smoothness": ridge_fit_predict(features, refresh_horizons[group], split, explanatory_by_group[group]),
        }

    dataset_info = json.loads((args.dataset / "meta/info.json").read_text(encoding="utf-8"))
    tree_file = args.dataset / ".cache/huggingface/trees/cbf7122bbdbaa0c50517a6a4b2ae663d0e96e51a.json"
    dump(
        args.output_dir / "reliability_and_smoothness_recomputed.json",
        {
            "audit_script": str(Path(__file__).relative_to(ROOT)),
            "issues": issues,
            "provenance": {
                "reliability_dataset_sha256": sha256(RELIABILITY),
                "refresh_bundle_sha256": sha256(BUNDLE),
                "dataset_revision": "cbf7122bbdbaa0c50517a6a4b2ae663d0e96e51a" if tree_file.is_file() else None,
                "dataset_fps": dataset_info["fps"],
                "rollout_control_frequency": 20,
                "timebase_warning": "Offline demonstration offsets are 10 Hz; rollout execution-horizon steps are 20 Hz. Equal step counts are not equal physical durations.",
                "refresh_raw_cache_status": "Missing: target_comparison.npz, refresh_first_actions.npz, and compare_targets.py were untracked generation artifacts and are absent. Bundle labels are internally auditable but exact target regeneration is NOT REPRODUCIBLE without fresh ACT re-querying.",
            },
            "coverage": {
                "episodes": len(actions),
                "samples": len(data["episode_index"]),
                "predicted_actions": int(np.prod(data["predicted_actions"].shape[:2])),
                "observed_pairs": int(observed.sum()),
                "successful_demonstrations_only": True,
                "failed_trajectories": 0,
            },
            "metric_equations": {
                "arm_translation": "sqrt(mean_{d=0..2}(((prediction_d - demonstration_d)/dataset_std_d)^2))",
                "arm_rotation": "sqrt(mean_{d=3..5}(((prediction_d - demonstration_d)/dataset_std_d)^2))",
                "arm_pointwise": "translation_error <= theta AND rotation_error <= theta",
                "gripper_pointwise": "abs(prediction_6-demonstration_6)/std_6 <= theta AND signs match",
                "prefix_survival": "Y_g(t,k) = AND_{j=0..k} V_g(t,j), with censored offsets masked",
            },
            "recomputed_error_deltas_vs_saved": error_deltas,
            "threshold_sensitivity": sensitivity,
            "gripper_definition_sensitivity": gripper_variants,
            "information_loss": {
                "interpretation": "Binary thresholding retains at most one bit per observed cell; prefix survival additionally discards every post-first-failure recovery. Rates and recovery fractions are reported under threshold_sensitivity.",
                "refresh_vs_demonstration_target": target_comparison,
            },
            "smoothness": {
                "feature_names": feature_names,
                "correlations": correlation,
                "ridge_baselines": ridge,
                "causal_note": "Current/history action differences and progress are causal descriptive baselines. Features prefixed future_ use demonstration futures and are explanatory ceilings only.",
            },
            "group_limit": "LIBERO has one 6D arm and one gripper command; left/right decomposition is not available in these artifacts.",
        },
    )
    print(json.dumps({
        "issues": issues,
        "error_deltas": error_deltas,
        "target_comparison": target_comparison,
        "output": str(args.output_dir / "reliability_and_smoothness_recomputed.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
