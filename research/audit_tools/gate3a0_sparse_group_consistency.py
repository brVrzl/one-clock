#!/usr/bin/env python3
"""Cheap Gate-3A0 audit on the existing sparse teacher-forced ACT cache.

This script is deliberately read-only with respect to historical artifacts.  It
reuses saved chunks whose source observations are roughly 25 dataset steps apart;
it does not run ACT and must not be described as dense temporal ensembling.

The audit replaces generic gripper magnitude MSE with sign/event errors, compares
axis-angle increments through SO(3) geodesic distance, and asks whether semantic
group oracles retain headroom when source-age disparity is constrained.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DATASET = Path("/home/thor/datasets/libero_object_25_08_23_lerobotv2.1")
CACHE = ROOT / "experiments/temporal_reliability/reliability_dataset.npz"
BUNDLE = ROOT / "experiments/dynamic_reliability_horizon/artifact_handoff/minimal_y_refresh_training_bundle.npz"
ACTION_STD = np.asarray(
    [
        0.2681190073490143,
        0.4384443759918213,
        0.4475117325782776,
        0.024448219686746597,
        0.04936208948493004,
        0.042103495448827744,
        0.9974462985992432,
    ],
    dtype=np.float64,
)
EXP_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
SIMILARITY_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
COGACT_ALPHA_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
GATE_GRID = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, float("inf"))
AGE_DISPARITY_GRID = (0, 4, 8, 16, 25, 50, 99)
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260821
DATASET_HZ = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "research/audit_outputs/gate3a0_sparse_group_consistency.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_actions(dataset: Path, episode_ids: np.ndarray) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for episode_id in sorted(set(int(value) for value in episode_ids)):
        path = dataset / "data/chunk-000" / f"episode_{episode_id:06d}.parquet"
        result[episode_id] = np.asarray(
            pq.read_table(path, columns=["action"])["action"].to_pylist(), dtype=np.float64
        )
    return result


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def axis_angle_matrix(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3) + skew(vector)
    axis = vector / angle
    cross = skew(axis)
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def rotation_geodesic(left: np.ndarray, right: np.ndarray) -> float:
    relative = axis_angle_matrix(left).T @ axis_angle_matrix(right)
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.arccos(cosine))


def matrix_axis_angle(matrix: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    vee = np.asarray(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1e-10:
        return 0.5 * vee
    sine = float(np.sin(angle))
    if abs(sine) < 1e-8:
        values, vectors = np.linalg.eig(matrix)
        axis = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
        axis /= np.linalg.norm(axis)
        return axis * angle
    return vee * (angle / (2.0 * sine))


def sign(value: float) -> int:
    return 1 if value >= 0.0 else -1


def semantic_errors(
    prediction: np.ndarray, target: np.ndarray, previous_target: np.ndarray
) -> dict[str, float]:
    translation_delta = prediction[:3] - target[:3]
    translation_normalized_mse = float(np.mean((translation_delta / ACTION_STD[:3]) ** 2))
    translation_l2 = float(np.linalg.norm(translation_delta))
    rotation_radians = rotation_geodesic(prediction[3:6], target[3:6])
    rotation_scale_sq = float(np.sum(ACTION_STD[3:6] ** 2))
    rotation_normalized_sq = float(rotation_radians**2 / rotation_scale_sq)
    predicted_sign = sign(float(prediction[6]))
    target_sign = sign(float(target[6]))
    previous_sign = sign(float(previous_target[6]))
    predicted_transition = predicted_sign != previous_sign
    target_transition = target_sign != previous_sign
    gripper_sign_error = float(predicted_sign != target_sign)
    dimension_weighted = (
        3.0 * translation_normalized_mse
        + 3.0 * rotation_normalized_sq
        + gripper_sign_error
    ) / 7.0
    equal_group = (
        translation_normalized_mse + rotation_normalized_sq + gripper_sign_error
    ) / 3.0
    arm_gripper_balanced = 0.5 * (
        0.5 * (translation_normalized_mse + rotation_normalized_sq)
        + gripper_sign_error
    )
    return {
        "translation_l2_action_units": translation_l2,
        "translation_normalized_mse": translation_normalized_mse,
        "rotation_geodesic_radians": rotation_radians,
        "rotation_normalized_sq": rotation_normalized_sq,
        "gripper_sign_error": gripper_sign_error,
        "gripper_false_transition": float(predicted_transition and not target_transition),
        "gripper_missed_transition": float(target_transition and not predicted_transition),
        "dimension_weighted_semantic_error": float(dimension_weighted),
        "equal_group_semantic_error": float(equal_group),
        "arm_gripper_balanced_semantic_error": float(arm_gripper_balanced),
    }


def semantic_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Control-semantic distance between two action candidates.

    This is the same dimension-weighted construction used for target error, with
    gripper sign disagreement in place of continuous gripper magnitude.
    """

    translation = float(np.mean(((left[:3] - right[:3]) / ACTION_STD[:3]) ** 2))
    rotation = rotation_geodesic(left[3:6], right[3:6]) ** 2 / float(
        np.sum(ACTION_STD[3:6] ** 2)
    )
    grip = float(sign(float(left[6])) != sign(float(right[6])))
    return float((3.0 * translation + 3.0 * rotation + grip) / 7.0)


def group_distance(left: np.ndarray, right: np.ndarray, group: str) -> float:
    if group == "translation":
        return float(np.mean(((left[:3] - right[:3]) / ACTION_STD[:3]) ** 2))
    if group == "rotation":
        return float(
            rotation_geodesic(left[3:6], right[3:6]) ** 2
            / np.sum(ACTION_STD[3:6] ** 2)
        )
    if group == "gripper":
        return float(sign(float(left[6])) != sign(float(right[6])))
    if group == "arm":
        return 0.5 * (
            group_distance(left, right, "translation")
            + group_distance(left, right, "rotation")
        )
    raise ValueError(group)


def mean_metrics(
    predictions: Iterable[np.ndarray], entries: list[dict[str, Any]]
) -> dict[str, float]:
    rows = [
        semantic_errors(prediction, entry["target"], entry["previous_target"])
        for prediction, entry in zip(predictions, entries, strict=True)
    ]
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def episode_bootstrap_difference(
    left: list[float], right: list[float], episode_ids: list[int]
) -> dict[str, Any]:
    by_episode: dict[int, list[float]] = defaultdict(list)
    for a, b, episode_id in zip(left, right, episode_ids, strict=True):
        by_episode[int(episode_id)].append(float(a - b))
    episode_means = np.asarray([np.mean(items) for items in by_episode.values()], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(episode_means), size=(BOOTSTRAP_DRAWS, len(episode_means)))
    draws = episode_means[indices].mean(axis=1)
    return {
        "sample_weighted_mean_difference": float(np.mean(np.asarray(left) - np.asarray(right))),
        "episode_weighted_mean_difference": float(np.mean(episode_means)),
        "episode_bootstrap_ci95": [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ],
        "episodes": int(len(episode_means)),
    }


def weights_from_distance(distances: np.ndarray, temperature: float) -> np.ndarray:
    shifted = distances - float(np.min(distances))
    weights = np.exp(-shifted / temperature)
    return weights / weights.sum()


def semantic_aggregate(experts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Aggregate with one shared source-weight vector and action-aware operators."""

    result = np.zeros(7, dtype=np.float64)
    result[:3] = weights @ experts[:, :3]
    matrix_average = sum(
        float(weight) * axis_angle_matrix(expert[3:6])
        for weight, expert in zip(weights, experts, strict=True)
    )
    left, _, right_t = np.linalg.svd(matrix_average)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(left @ right_t)
    result[3:6] = matrix_axis_angle(left @ correction @ right_t)
    sign_vote = float(np.dot(weights, np.where(experts[:, 6] >= 0.0, 1.0, -1.0)))
    result[6] = 1.0 if sign_vote >= 0.0 else -1.0
    return result


def exponential(entry: dict[str, Any], coefficient: float) -> np.ndarray:
    weights = np.exp(-coefficient * entry["ages"].astype(np.float64))
    weights /= weights.sum()
    return weights @ entry["experts"]


def exponential_semantic_aggregation(
    entry: dict[str, Any], coefficient: float
) -> np.ndarray:
    weights = np.exp(-coefficient * entry["ages"].astype(np.float64))
    weights /= weights.sum()
    return semantic_aggregate(entry["experts"], weights)


def scalar_similarity(entry: dict[str, Any], temperature: float) -> np.ndarray:
    newest = entry["experts"][int(np.argmin(entry["ages"]))]
    distances = np.asarray([semantic_distance(expert, newest) for expert in entry["experts"]])
    return weights_from_distance(distances, temperature) @ entry["experts"]


def scalar_similarity_semantic_aggregation(
    entry: dict[str, Any], temperature: float
) -> np.ndarray:
    newest = entry["experts"][int(np.argmin(entry["ages"]))]
    distances = np.asarray([semantic_distance(expert, newest) for expert in entry["experts"]])
    return semantic_aggregate(
        entry["experts"], weights_from_distance(distances, temperature)
    )


def cogact_cosine(
    entry: dict[str, Any], alpha: float, use_semantic_aggregation: bool
) -> np.ndarray:
    """Released CogACT scalar full-action cosine weighting rule."""

    experts = entry["experts"]
    newest = experts[int(np.argmin(entry["ages"]))]
    norms = np.linalg.norm(experts, axis=1) * np.linalg.norm(newest)
    cosine = np.sum(experts * newest[None, :], axis=1) / (norms + 1e-7)
    logits = alpha * cosine
    logits -= float(np.max(logits))
    weights = np.exp(logits)
    weights /= weights.sum()
    if use_semantic_aggregation:
        return semantic_aggregate(experts, weights)
    return weights @ experts


def group_similarity(entry: dict[str, Any], temperature: float, grouping: str) -> np.ndarray:
    newest = entry["experts"][int(np.argmin(entry["ages"]))]
    groups = ("arm", "gripper") if grouping == "arm_gripper" else (
        "translation",
        "rotation",
        "gripper",
    )
    predictions: dict[str, np.ndarray] = {}
    for group in groups:
        distances = np.asarray(
            [group_distance(expert, newest, group) for expert in entry["experts"]]
        )
        predictions[group] = weights_from_distance(distances, temperature) @ entry["experts"]
    result = newest.copy()
    if grouping == "arm_gripper":
        result[:6] = predictions["arm"][:6]
    else:
        result[:3] = predictions["translation"][:3]
        result[3:6] = predictions["rotation"][3:6]
    result[6] = predictions["gripper"][6]
    return result


def nearest_full_distance(prediction: np.ndarray, experts: np.ndarray) -> float:
    return float(min(semantic_distance(prediction, expert) for expert in experts))


def consistency_gated_group_similarity(
    entry: dict[str, Any], scalar_temperature: float, group_temperature: float,
    gate_scale: float, grouping: str
) -> np.ndarray:
    scalar = scalar_similarity(entry, scalar_temperature)
    grouped = group_similarity(entry, group_temperature, grouping)
    inconsistency = nearest_full_distance(grouped, entry["experts"])
    if gate_scale == 0.0:
        alpha = 0.0
    elif np.isinf(gate_scale):
        alpha = 1.0
    else:
        alpha = float(np.exp(-inconsistency / gate_scale))
    return scalar + alpha * (grouped - scalar)


def select_parameter(
    validation: list[dict[str, Any]],
    candidates: Iterable[Any],
    predictor: Callable[[dict[str, Any], Any], np.ndarray],
) -> Any:
    def score(candidate: Any) -> float:
        return float(
            np.mean(
                [
                    semantic_errors(
                        predictor(entry, candidate), entry["target"], entry["previous_target"]
                    )["dimension_weighted_semantic_error"]
                    for entry in validation
                ]
            )
        )

    return min(candidates, key=score)


def composite_from_indices(
    experts: np.ndarray, indices: tuple[int, ...], grouping: str
) -> np.ndarray:
    result = experts[indices[0]].copy()
    if grouping == "scalar":
        return result
    if grouping == "arm_gripper":
        result[:6] = experts[indices[0], :6]
        result[6] = experts[indices[1], 6]
        return result
    result[:3] = experts[indices[0], :3]
    result[3:6] = experts[indices[1], 3:6]
    result[6] = experts[indices[2], 6]
    return result


def constrained_hard_oracle(
    entry: dict[str, Any], grouping: str, maximum_age_disparity: int
) -> tuple[np.ndarray, dict[str, float]]:
    group_count = {"scalar": 1, "arm_gripper": 2, "semantic_three": 3}[grouping]
    best_prediction: np.ndarray | None = None
    best_indices: tuple[int, ...] | None = None
    best_loss = float("inf")
    for indices in itertools.product(range(len(entry["experts"])), repeat=group_count):
        selected_ages = entry["ages"][list(indices)]
        if int(np.max(selected_ages) - np.min(selected_ages)) > maximum_age_disparity:
            continue
        prediction = composite_from_indices(entry["experts"], indices, grouping)
        loss = semantic_errors(prediction, entry["target"], entry["previous_target"])[
            "dimension_weighted_semantic_error"
        ]
        if loss < best_loss:
            best_loss = loss
            best_prediction = prediction
            best_indices = indices
    if best_prediction is None or best_indices is None:
        raise RuntimeError("same-source candidates should make every disparity constraint feasible")
    chosen_ages = entry["ages"][list(best_indices)]
    return best_prediction, {
        "age_disparity_steps": float(np.max(chosen_ages) - np.min(chosen_ages)),
        "age_disparity_seconds": float(
            (np.max(chosen_ages) - np.min(chosen_ages)) / DATASET_HZ
        ),
        "mixed_source": float(len(set(best_indices)) > 1),
        "nearest_full_semantic_distance": nearest_full_distance(
            best_prediction, entry["experts"]
        ),
        "teacher_forced_boundary_semantic_distance": semantic_distance(
            best_prediction, entry["previous_target"]
        ),
    }


def summarize_method(
    predictions: list[np.ndarray], entries: list[dict[str, Any]]
) -> dict[str, Any]:
    summary: dict[str, Any] = mean_metrics(predictions, entries)
    summary["per_task_dimension_weighted_semantic_error"] = {
        str(task): float(
            np.mean(
                [
                    semantic_errors(prediction, entry["target"], entry["previous_target"])[
                        "dimension_weighted_semantic_error"
                    ]
                    for prediction, entry in zip(predictions, entries, strict=True)
                    if entry["task"] == task
                ]
            )
        )
        for task in sorted(set(int(entry["task"]) for entry in entries))
    }
    return summary


def main() -> None:
    args = parse_args()
    cache = np.load(args.cache, allow_pickle=False)
    bundle = np.load(args.bundle, allow_pickle=False)
    episode_ids = cache["episode_index"].astype(int)
    frames = cache["frame_index"].astype(int)
    tasks = cache["task_index"].astype(int)
    chunks = cache["predicted_actions"].astype(np.float64)
    actions = load_actions(args.dataset, episode_ids)

    episode_split: dict[int, int] = {}
    for episode_id, split in zip(
        bundle["episode_index"], bundle["split_membership"], strict=True
    ):
        episode_split.setdefault(int(episode_id), int(split))
        if episode_split[int(episode_id)] != int(split):
            raise RuntimeError("split membership is not episode-level")

    row_by_episode: dict[int, list[int]] = defaultdict(list)
    for row, episode_id in enumerate(episode_ids):
        row_by_episode[int(episode_id)].append(row)

    entries: list[dict[str, Any]] = []
    source_spacing_counts: Counter[int] = Counter()
    for episode_id, rows in row_by_episode.items():
        rows.sort(key=lambda row: frames[row])
        source_spacing_counts.update(
            int(frames[right] - frames[left]) for left, right in zip(rows, rows[1:])
        )
        sequence = actions[episode_id]
        for target_time in range(len(sequence)):
            sources = [row for row in rows if frames[row] <= target_time < frames[row] + 100]
            if len(sources) < 2:
                continue
            ages = np.asarray([target_time - frames[row] for row in sources], dtype=np.int32)
            experts = np.stack(
                [chunks[row, age] for row, age in zip(sources, ages, strict=True)]
            )
            entries.append(
                {
                    "episode": episode_id,
                    "task": int(tasks[rows[0]]),
                    "split": episode_split[episode_id],
                    "target_time": target_time,
                    "ages": ages,
                    "experts": experts,
                    "target": sequence[target_time],
                    "previous_target": sequence[max(target_time - 1, 0)],
                }
            )

    validation = [entry for entry in entries if entry["split"] == 1]
    test = [entry for entry in entries if entry["split"] == 2]
    selected_exp = select_parameter(validation, EXP_GRID, exponential)
    selected_exp_semantic_aggregation = select_parameter(
        validation, EXP_GRID, exponential_semantic_aggregation
    )
    selected_scalar_similarity = select_parameter(
        validation, SIMILARITY_GRID, scalar_similarity
    )
    selected_scalar_similarity_semantic_aggregation = select_parameter(
        validation, SIMILARITY_GRID, scalar_similarity_semantic_aggregation
    )
    selected_cogact_cosine_raw = select_parameter(
        validation,
        COGACT_ALPHA_GRID,
        lambda entry, alpha: cogact_cosine(entry, alpha, False),
    )
    selected_cogact_cosine_semantic_aggregation = select_parameter(
        validation,
        COGACT_ALPHA_GRID,
        lambda entry, alpha: cogact_cosine(entry, alpha, True),
    )
    selected_group_temperature: dict[str, float] = {}
    selected_gate_parameters: dict[str, tuple[float, float]] = {}
    for grouping in ("arm_gripper", "semantic_three"):
        selected_group_temperature[grouping] = select_parameter(
            validation,
            SIMILARITY_GRID,
            lambda entry, temperature, current=grouping: group_similarity(
                entry, temperature, current
            ),
        )
        selected_gate_parameters[grouping] = select_parameter(
            validation,
            tuple(itertools.product(SIMILARITY_GRID, GATE_GRID)),
            lambda entry, parameters, current=grouping: consistency_gated_group_similarity(
                entry,
                selected_scalar_similarity,
                parameters[0],
                parameters[1],
                current,
            ),
        )

    methods: dict[str, Callable[[dict[str, Any]], np.ndarray]] = {
        "newest": lambda entry: entry["experts"][int(np.argmin(entry["ages"]))],
        "uniform": lambda entry: np.mean(entry["experts"], axis=0),
        "validation_selected_age_exponential": lambda entry: exponential(entry, selected_exp),
        "validation_selected_age_exponential_semantic_aggregation": lambda entry: exponential_semantic_aggregation(
            entry, selected_exp_semantic_aggregation
        ),
        "official_cogact_cosine_alpha_0_1": lambda entry: cogact_cosine(
            entry, 0.1, False
        ),
        "validation_selected_cogact_cosine": lambda entry: cogact_cosine(
            entry, selected_cogact_cosine_raw, False
        ),
        "validation_selected_cogact_cosine_semantic_aggregation": lambda entry: cogact_cosine(
            entry, selected_cogact_cosine_semantic_aggregation, True
        ),
        "validation_selected_scalar_semantic_similarity": lambda entry: scalar_similarity(
            entry, selected_scalar_similarity
        ),
        "validation_selected_scalar_semantic_similarity_semantic_aggregation": lambda entry: scalar_similarity_semantic_aggregation(
            entry, selected_scalar_similarity_semantic_aggregation
        ),
        "validation_selected_arm_gripper_similarity": lambda entry: group_similarity(
            entry, selected_group_temperature["arm_gripper"], "arm_gripper"
        ),
        "validation_selected_semantic_three_similarity": lambda entry: group_similarity(
            entry, selected_group_temperature["semantic_three"], "semantic_three"
        ),
        "consistency_gated_arm_gripper_similarity": lambda entry: consistency_gated_group_similarity(
            entry,
            selected_scalar_similarity,
            selected_gate_parameters["arm_gripper"][0],
            selected_gate_parameters["arm_gripper"][1],
            "arm_gripper",
        ),
        "consistency_gated_semantic_three_similarity": lambda entry: consistency_gated_group_similarity(
            entry,
            selected_scalar_similarity,
            selected_gate_parameters["semantic_three"][0],
            selected_gate_parameters["semantic_three"][1],
            "semantic_three",
        ),
    }

    predictions = {
        name: [method(entry) for entry in test] for name, method in methods.items()
    }
    method_summaries = {
        name: summarize_method(current, test) for name, current in predictions.items()
    }
    validation_predictions = {
        name: [method(entry) for entry in validation] for name, method in methods.items()
    }
    validation_summaries = {
        name: summarize_method(current, validation)
        for name, current in validation_predictions.items()
    }

    metric_names = (
        "dimension_weighted_semantic_error",
        "equal_group_semantic_error",
        "arm_gripper_balanced_semantic_error",
        "gripper_sign_error",
        "rotation_geodesic_radians",
    )
    episode_vector = [int(entry["episode"]) for entry in test]
    comparisons: dict[str, Any] = {}
    comparison_pairs = (
        (
            "validation_selected_scalar_semantic_similarity",
            "validation_selected_age_exponential",
        ),
        (
            "validation_selected_scalar_semantic_similarity_semantic_aggregation",
            "validation_selected_age_exponential_semantic_aggregation",
        ),
        (
            "validation_selected_scalar_semantic_similarity_semantic_aggregation",
            "validation_selected_scalar_semantic_similarity",
        ),
        (
            "validation_selected_scalar_semantic_similarity_semantic_aggregation",
            "validation_selected_cogact_cosine_semantic_aggregation",
        ),
        (
            "validation_selected_cogact_cosine_semantic_aggregation",
            "validation_selected_cogact_cosine",
        ),
        (
            "validation_selected_arm_gripper_similarity",
            "validation_selected_scalar_semantic_similarity",
        ),
        (
            "validation_selected_semantic_three_similarity",
            "validation_selected_scalar_semantic_similarity",
        ),
        (
            "consistency_gated_semantic_three_similarity",
            "validation_selected_scalar_semantic_similarity",
        ),
    )
    for left_name, right_name in comparison_pairs:
        pair_key = f"{left_name}_minus_{right_name}"
        comparisons[pair_key] = {}
        for metric in metric_names:
            left_values = [
                semantic_errors(prediction, entry["target"], entry["previous_target"])[metric]
                for prediction, entry in zip(predictions[left_name], test, strict=True)
            ]
            right_values = [
                semantic_errors(prediction, entry["target"], entry["previous_target"])[metric]
                for prediction, entry in zip(predictions[right_name], test, strict=True)
            ]
            comparisons[pair_key][metric] = episode_bootstrap_difference(
                left_values, right_values, episode_vector
            )

    oracle_frontiers: dict[str, Any] = {}
    oracle_predictions: dict[str, list[np.ndarray]] = {}
    for grouping in ("scalar", "arm_gripper", "semantic_three"):
        oracle_frontiers[grouping] = {}
        for maximum_disparity in AGE_DISPARITY_GRID:
            current_predictions: list[np.ndarray] = []
            diagnostics: list[dict[str, float]] = []
            for entry in test:
                prediction, diagnostic = constrained_hard_oracle(
                    entry, grouping, maximum_disparity
                )
                current_predictions.append(prediction)
                diagnostics.append(diagnostic)
            key = str(maximum_disparity)
            oracle_frontiers[grouping][key] = summarize_method(current_predictions, test)
            oracle_frontiers[grouping][key]["constraint"] = {
                "maximum_age_disparity_dataset_steps": maximum_disparity,
                "maximum_age_disparity_seconds": maximum_disparity / DATASET_HZ,
            }
            oracle_frontiers[grouping][key]["composition_diagnostics"] = {
                name: float(np.mean([row[name] for row in diagnostics]))
                for name in diagnostics[0]
            }
            if maximum_disparity == AGE_DISPARITY_GRID[-1]:
                oracle_predictions[grouping] = current_predictions

    oracle_comparisons: dict[str, Any] = {}
    for grouping in ("arm_gripper", "semantic_three"):
        pair_key = f"oracle_{grouping}_minus_oracle_scalar"
        oracle_comparisons[pair_key] = {}
        for metric in metric_names:
            left_values = [
                semantic_errors(prediction, entry["target"], entry["previous_target"])[metric]
                for prediction, entry in zip(oracle_predictions[grouping], test, strict=True)
            ]
            right_values = [
                semantic_errors(prediction, entry["target"], entry["previous_target"])[metric]
                for prediction, entry in zip(oracle_predictions["scalar"], test, strict=True)
            ]
            oracle_comparisons[pair_key][metric] = episode_bootstrap_difference(
                left_values, right_values, episode_vector
            )

    dump(
        args.output,
        {
            "audit_script": str(Path(__file__).relative_to(ROOT)),
            "decision_scope": "Gate-3A0 sparse-data sanity audit only; no dense cache, policy modification, or rollout.",
            "provenance": {
                "cache": {
                    "path": str(args.cache.resolve()),
                    "bytes": args.cache.stat().st_size,
                    "sha256": sha256(args.cache),
                },
                "split_bundle": {
                    "path": str(args.bundle.resolve()),
                    "bytes": args.bundle.stat().st_size,
                    "sha256": sha256(args.bundle),
                },
                "dataset": str(args.dataset.resolve()),
                "dataset_frequency_hz": DATASET_HZ,
            },
            "cohort": {
                "all_sparse_overlap_targets": len(entries),
                "validation_sparse_overlap_targets": len(validation),
                "test_sparse_overlap_targets": len(test),
                "test_episodes": len(set(episode_vector)),
                "test_tasks": sorted(set(int(entry["task"]) for entry in test)),
                "expert_count_range": [
                    min(len(entry["experts"]) for entry in entries),
                    max(len(entry["experts"]) for entry in entries),
                ],
                "mean_experts": float(np.mean([len(entry["experts"]) for entry in entries])),
                "within_episode_source_spacing_counts_dataset_steps": {
                    str(key): value for key, value in sorted(source_spacing_counts.items())
                },
            },
            "metric_contract": {
                "translation": "Mean squared component error normalized by audited training-action standard deviations; raw L2 action-unit error is also reported.",
                "rotation": "SO(3) geodesic angle between rotations represented by the predicted and demonstrated axis-angle increments; squared angle is normalized by the sum of audited rotation-component variances.",
                "gripper": "Sign error. Transition errors assume the previous demonstrated sign as the teacher-forced prior command; magnitude MSE is not used.",
                "semantic_aggregation": "One shared source-weight vector; Euclidean translation mean, projected SO(3) chordal rotation mean, and weighted gripper-sign vote.",
                "dimension_weighted_semantic_error": "(3*translation_normalized_mse + 3*rotation_normalized_sq + gripper_sign_error)/7.",
                "equal_group_semantic_error": "Equal average over translation, rotation, and gripper semantic losses; sensitivity metric, not a privileged control objective.",
                "arm_gripper_balanced_semantic_error": "Equal arm/gripper average; retained only to expose weighting sensitivity.",
            },
            "validation_selection": {
                "age_exponential_coefficient": selected_exp,
                "age_exponential_semantic_aggregation_coefficient": selected_exp_semantic_aggregation,
                "scalar_semantic_similarity_temperature": selected_scalar_similarity,
                "scalar_semantic_similarity_semantic_aggregation_temperature": selected_scalar_similarity_semantic_aggregation,
                "cogact_cosine_alpha": {
                    "official_released_setting": 0.1,
                    "validation_selected_raw_linear_aggregation": selected_cogact_cosine_raw,
                    "validation_selected_semantic_aggregation": selected_cogact_cosine_semantic_aggregation,
                },
                "group_similarity_temperature": selected_group_temperature,
                "consistency_gate_parameters": {
                    key: {
                        "group_similarity_temperature": value[0],
                        "gate_scale": "infinity" if np.isinf(value[1]) else value[1],
                    }
                    for key, value in selected_gate_parameters.items()
                },
                "selection_rule": "Minimize dimension-weighted semantic error on validation episodes only.",
            },
            "test_methods": method_summaries,
            "validation_methods": validation_summaries,
            "paired_test_comparisons": comparisons,
            "oracle_age_disparity_frontiers": oracle_frontiers,
            "oracle_unconstrained_comparisons": oracle_comparisons,
            "interpretation_limits": [
                "The cache is sparse (source observations roughly 25 dataset steps apart), not dense ACT temporal ensembling.",
                "All rows are teacher-forced demonstration states; offline semantic error is not closed-loop success.",
                "Every oracle observes the demonstrated target and is unattainable at deployment.",
                "The gripper transition metrics use the prior demonstrated gripper sign, not the policy's prior closed-loop command.",
                "Raw baselines linearly average axis-angle vectors because that is the audited policy representation; explicitly named semantic-aggregation ablations instead use a projected SO(3) chordal mean and sign vote.",
                "Nearest-full-source and boundary distances are diagnostics, not validated detectors of physical consistency or failure.",
                "Dataset ages are reported at 10 Hz (0.1 s/step); they must not be equated with the audited rollout controller's 20 Hz ticks.",
            ],
        },
    )
    print(
        json.dumps(
            {
                "entries": len(entries),
                "validation": len(validation),
                "test": len(test),
                "selection": {
                    "exp": selected_exp,
                    "exp_semantic_aggregation": selected_exp_semantic_aggregation,
                    "scalar_similarity": selected_scalar_similarity,
                    "scalar_similarity_semantic_aggregation": selected_scalar_similarity_semantic_aggregation,
                    "cogact_cosine": {
                        "raw": selected_cogact_cosine_raw,
                        "semantic_aggregation": selected_cogact_cosine_semantic_aggregation,
                    },
                    "group_temperature": selected_group_temperature,
                    "gate": {
                        key: {
                            "group_temperature": value[0],
                            "scale": "infinity" if np.isinf(value[1]) else value[1],
                        }
                        for key, value in selected_gate_parameters.items()
                    },
                },
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
