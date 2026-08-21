#!/usr/bin/env python3
"""Preregistered validation selection and held-out analysis for Gate-3A1.

`tune` reads validation episodes only and writes an immutable selection lock.
`evaluate` refuses to run unless that lock is tracked in the current Git HEAD
and its protocol and analysis-script hashes still match.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Callable

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
DATASET = Path("/home/thor/datasets/libero_object_25_08_23_lerobotv2.1")
CACHE_ROOT = ROOT / "experiments/gate3a1_dense_temporal_cache"
INVENTORY = ROOT / "research/audit_outputs/gate3a1_inventory.json"
CACHE_MANIFEST = ROOT / "research/audit_outputs/gate3a1_dense_cache_manifest.json"
PROTOCOL = ROOT / "research/gate3a1_preregistered_protocol.md"
VALIDATION_LOCK = ROOT / "research/audit_outputs/gate3a1_validation_lock.json"
METRICS_OUTPUT = ROOT / "research/audit_outputs/gate3a1_dense_metrics.json"
PER_TASK_OUTPUT = ROOT / "research/audit_outputs/gate3a1_dense_per_task.csv"
PAIRWISE_OUTPUT = ROOT / "research/audit_outputs/gate3a1_dense_pairwise_comparisons.csv"
ORACLE_OUTPUT = ROOT / "research/audit_outputs/gate3a1_dense_oracle_headroom.json"
REGISTRATION_COMMIT = "d163f5a76a46c9368adbb8c2f56f09e248b3a81c"
DATASET_HZ = 10.0
CHUNK_SIZE = 100
ACTION_DIM = 7
BOOTSTRAP_DRAWS = 10000
EPISODE_BOOTSTRAP_SEED = 20260821
TASK_BOOTSTRAP_SEED = 20260822
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
AGE_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
KERNEL_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
ORACLE_LAMBDAS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
METHOD_ORDER = (
    "b0_newest",
    "b1_uniform",
    "b2_exact_act_m0_01",
    "b2_tuned_act_oldest_exponential",
    "b3_tuned_newest_age_exponential",
    "b4_official_cogact_alpha0_1",
    "b5_tuned_cogact_cosine",
    "b6_control_semantic_similarity",
    "b5_tuned_cogact_cosine_semantic_aggregation",
    "b6_control_semantic_similarity_semantic_aggregation",
)
PRIMARY_METHODS = METHOD_ORDER[:8]
METRIC_NAMES = (
    "dimension_weighted_semantic_error",
    "translation_normalized_mse",
    "translation_l2_action_units",
    "rotation_geodesic_radians",
    "rotation_normalized_sq",
    "gripper_sign_error",
    "raw_7d_mse",
    "equal_group_semantic_error",
    "arm_gripper_balanced_semantic_error",
)


@dataclass
class Episode:
    episode_id: int
    task_id: int
    targets: np.ndarray
    previous_targets: np.ndarray
    experts: np.ndarray
    ages: np.ndarray
    mask: np.ndarray
    semantic_distances: np.ndarray
    cosine_similarities: np.ndarray
    translation_disagreement: np.ndarray
    rotation_disagreement: np.ndarray
    gripper_disagreement_fraction: np.ndarray


@dataclass
class Evaluation:
    prediction: np.ndarray
    weights: np.ndarray
    weighted_age: np.ndarray
    metrics: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("self-test", "validate-cache", "tune", "evaluate"))
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument("--cache-manifest", type=Path, default=CACHE_MANIFEST)
    parser.add_argument("--validation-lock", type=Path, default=VALIDATION_LOCK)
    parser.add_argument("--metrics-output", type=Path, default=METRICS_OUTPUT)
    parser.add_argument("--per-task-output", type=Path, default=PER_TASK_OUTPUT)
    parser.add_argument("--pairwise-output", type=Path, default=PAIRWISE_OUTPUT)
    parser.add_argument("--oracle-output", type=Path, default=ORACLE_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], check=True, text=True, capture_output=True
    ).stdout.strip()


def axis_angle_matrices(vectors: np.ndarray) -> np.ndarray:
    shape = vectors.shape[:-1]
    return Rotation.from_rotvec(vectors.reshape(-1, 3)).as_matrix().reshape(*shape, 3, 3)


def rotation_geodesic(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_matrix = axis_angle_matrices(left)
    right_matrix = axis_angle_matrices(right)
    relative = np.swapaxes(left_matrix, -1, -2) @ right_matrix
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cosine)


def action_sign(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0.0, 1, -1)


def target_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    translation_delta = prediction[:, :3] - target[:, :3]
    translation_normalized = np.mean((translation_delta / ACTION_STD[:3]) ** 2, axis=1)
    translation_l2 = np.linalg.norm(translation_delta, axis=1)
    rotation_radians = rotation_geodesic(prediction[:, 3:6], target[:, 3:6])
    rotation_normalized = rotation_radians**2 / float(np.sum(ACTION_STD[3:6] ** 2))
    gripper = (action_sign(prediction[:, 6]) != action_sign(target[:, 6])).astype(np.float64)
    dimension_weighted = (3.0 * translation_normalized + 3.0 * rotation_normalized + gripper) / 7.0
    equal_group = (translation_normalized + rotation_normalized + gripper) / 3.0
    arm_gripper = 0.5 * (0.5 * (translation_normalized + rotation_normalized) + gripper)
    return {
        "dimension_weighted_semantic_error": dimension_weighted,
        "translation_normalized_mse": translation_normalized,
        "translation_l2_action_units": translation_l2,
        "rotation_geodesic_radians": rotation_radians,
        "rotation_normalized_sq": rotation_normalized,
        "gripper_sign_error": gripper,
        "raw_7d_mse": np.mean((prediction - target) ** 2, axis=1),
        "equal_group_semantic_error": equal_group,
        "arm_gripper_balanced_semantic_error": arm_gripper,
    }


def target_metric_one(predictions: np.ndarray, target: np.ndarray) -> np.ndarray:
    repeated = np.broadcast_to(target, predictions.shape)
    return target_metrics(predictions, repeated)["dimension_weighted_semantic_error"]


def semantic_candidate_diagnostics(experts: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, ...]:
    newest_indices = mask.sum(axis=1) - 1
    newest = experts[np.arange(len(experts)), newest_indices]
    translation = np.mean(((experts[:, :, :3] - newest[:, None, :3]) / ACTION_STD[:3]) ** 2, axis=2)
    rotation = rotation_geodesic(experts[:, :, 3:6], newest[:, None, 3:6])
    rotation_normalized = rotation**2 / float(np.sum(ACTION_STD[3:6] ** 2))
    gripper = (
        action_sign(experts[:, :, 6]) != action_sign(newest[:, None, 6])
    ).astype(np.float64)
    semantic = (3.0 * translation + 3.0 * rotation_normalized + gripper) / 7.0
    cosine_numerator = np.sum(experts * newest[:, None, :], axis=2)
    cosine_denominator = np.linalg.norm(experts, axis=2) * np.linalg.norm(newest, axis=1)[:, None] + 1e-7
    cosine = cosine_numerator / cosine_denominator
    for array in (translation, rotation, gripper, semantic, cosine):
        array[~mask] = 0.0
    count = mask.sum(axis=1)
    semantic_mean = semantic.sum(axis=1) / count
    translation_max = np.max(np.where(mask, translation, -np.inf), axis=1)
    rotation_max = np.max(np.where(mask, rotation, -np.inf), axis=1)
    gripper_fraction = gripper.sum(axis=1) / count
    return semantic, cosine, semantic_mean, translation_max, rotation_max, gripper_fraction


def load_episode(dataset: Path, cache_root: Path, split: str, episode_id: int) -> Episode:
    cache_path = cache_root / split / f"episode_{episode_id:06d}.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        chunks = cache["predicted_chunks"].astype(np.float64)
        task_id = int(cache["task_id"].item())
        frames = cache["dataset_frame"].astype(int)
        if str(cache["split"].item()) != split or int(cache["episode_id"].item()) != episode_id:
            raise RuntimeError(f"Cache metadata mismatch in {cache_path}")
    table = pq.read_table(
        dataset / "data/chunk-000" / f"episode_{episode_id:06d}.parquet",
        columns=["action", "frame_index", "task_index", "episode_index"],
    )
    targets = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    dataset_frames = np.asarray(table["frame_index"].to_pylist(), dtype=int)
    task_values = set(int(value) for value in table["task_index"].to_pylist())
    episode_values = set(int(value) for value in table["episode_index"].to_pylist())
    length = len(targets)
    if (
        chunks.shape != (length, CHUNK_SIZE, ACTION_DIM)
        or not np.array_equal(frames, dataset_frames)
        or task_values != {task_id}
        or episode_values != {episode_id}
    ):
        raise RuntimeError(f"Dataset/cache contract mismatch for episode {episode_id}")

    experts = np.zeros((length, CHUNK_SIZE, ACTION_DIM), dtype=np.float64)
    ages = np.zeros((length, CHUNK_SIZE), dtype=np.int16)
    mask = np.zeros((length, CHUNK_SIZE), dtype=bool)
    for target_time in range(length):
        first_source = max(0, target_time - CHUNK_SIZE + 1)
        sources = np.arange(first_source, target_time + 1)
        current_ages = target_time - sources
        count = len(sources)
        experts[target_time, :count] = chunks[sources, current_ages]
        ages[target_time, :count] = current_ages
        mask[target_time, :count] = True
    semantic, cosine, semantic_mean, translation_max, rotation_max, gripper_fraction = (
        semantic_candidate_diagnostics(experts, mask)
    )
    previous = np.concatenate([targets[:1], targets[:-1]], axis=0)
    return Episode(
        episode_id=episode_id,
        task_id=task_id,
        targets=targets,
        previous_targets=previous,
        experts=experts,
        ages=ages,
        mask=mask,
        semantic_distances=semantic,
        cosine_similarities=cosine,
        translation_disagreement=translation_max,
        rotation_disagreement=rotation_max,
        gripper_disagreement_fraction=gripper_fraction,
    )


def verify_cache_split(
    split: str, inventory: dict[str, Any], manifest: dict[str, Any], cache_root: Path
) -> list[int]:
    expected_ids = [int(value) for value in inventory["splits"][split]["episode_ids"]]
    entries = {
        int(entry["episode_id"]): entry
        for entry in manifest["entries"]
        if entry["split"] == split and entry["status"] == "complete"
    }
    if set(entries) != set(expected_ids):
        raise RuntimeError(f"Incomplete {split} cache: expected {len(expected_ids)}, found {len(entries)}")
    total = 0
    for episode_id in expected_ids:
        entry = entries[episode_id]
        path = cache_root / split / f"episode_{episode_id:06d}.npz"
        if path.resolve() != Path(entry["cache_file"]).resolve():
            raise RuntimeError(f"Cache path mismatch for episode {episode_id}")
        if sha256(path) != entry["sha256"] or path.stat().st_size != int(entry["bytes"]):
            raise RuntimeError(f"Cache integrity mismatch for episode {episode_id}")
        if int(entry["completed_frames"]) != int(entry["expected_frames"]):
            raise RuntimeError(f"Incomplete frames for episode {episode_id}")
        total += int(entry["completed_frames"])
    expected_total = int(inventory["splits"][split]["dataset_steps"])
    if total != expected_total:
        raise RuntimeError(f"{split} query total mismatch: {total} != {expected_total}")
    return expected_ids


def load_split(args: argparse.Namespace, split: str) -> tuple[list[Episode], dict[str, Any], dict[str, Any]]:
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    manifest = json.loads(args.cache_manifest.read_text(encoding="utf-8"))
    ids = verify_cache_split(split, inventory, manifest, args.cache_root.resolve())
    episodes = [load_episode(args.dataset.resolve(), args.cache_root.resolve(), split, episode_id) for episode_id in ids]
    return episodes, inventory, manifest


def normalize_weights(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask, logits, -np.inf)
    maximum = np.max(masked, axis=1, keepdims=True)
    weights = np.where(mask, np.exp(masked - maximum), 0.0)
    return weights / weights.sum(axis=1, keepdims=True)


def semantic_aggregate(experts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    prediction = np.zeros((len(experts), ACTION_DIM), dtype=np.float64)
    prediction[:, :3] = np.einsum("nm,nmd->nd", weights, experts[:, :, :3])
    matrices = axis_angle_matrices(experts[:, :, 3:6])
    averages = np.einsum("nm,nmij->nij", weights, matrices)
    left, _, right_t = np.linalg.svd(averages)
    correction = np.tile(np.eye(3), (len(experts), 1, 1))
    correction[:, -1, -1] = np.linalg.det(left @ right_t)
    projected = left @ correction @ right_t
    prediction[:, 3:6] = Rotation.from_matrix(projected).as_rotvec()
    signs = action_sign(experts[:, :, 6]).astype(np.float64)
    prediction[:, 6] = np.where(np.sum(weights * signs, axis=1) >= 0.0, 1.0, -1.0)
    return prediction


def weights_for_method(episode: Episode, method: str, parameter: float | None = None) -> np.ndarray:
    if method == "b0_newest":
        weights = np.zeros_like(episode.ages, dtype=np.float64)
        weights[np.arange(len(weights)), episode.mask.sum(axis=1) - 1] = 1.0
        return weights
    if method == "b1_uniform":
        return episode.mask.astype(np.float64) / episode.mask.sum(axis=1, keepdims=True)
    if method in ("b2_exact_act_m0_01", "b2_tuned_act_oldest_exponential"):
        coefficient = 0.01 if method == "b2_exact_act_m0_01" else float(parameter)
        source_order = np.broadcast_to(np.arange(episode.ages.shape[1]), episode.ages.shape)
        return normalize_weights(-coefficient * source_order, episode.mask)
    if method == "b3_tuned_newest_age_exponential":
        return normalize_weights(-float(parameter) * episode.ages, episode.mask)
    if method in (
        "b4_official_cogact_alpha0_1",
        "b5_tuned_cogact_cosine",
        "b5_tuned_cogact_cosine_semantic_aggregation",
    ):
        alpha = 0.1 if method == "b4_official_cogact_alpha0_1" else float(parameter)
        return normalize_weights(alpha * episode.cosine_similarities, episode.mask)
    if method in (
        "b6_control_semantic_similarity",
        "b6_control_semantic_similarity_semantic_aggregation",
    ):
        return normalize_weights(-episode.semantic_distances / float(parameter), episode.mask)
    raise ValueError(method)


def evaluate_episode(episode: Episode, method: str, parameter: float | None = None) -> Evaluation:
    weights = weights_for_method(episode, method, parameter)
    if method.endswith("semantic_aggregation"):
        prediction = semantic_aggregate(episode.experts, weights)
    else:
        prediction = np.einsum("nm,nmd->nd", weights, episode.experts)
    return Evaluation(
        prediction=prediction,
        weights=weights,
        weighted_age=np.sum(weights * episode.ages, axis=1),
        metrics=target_metrics(prediction, episode.targets),
    )


def episode_weighted_score(episodes: list[Episode], method: str, parameter: float | None = None) -> float:
    return float(
        np.mean(
            [
                np.mean(evaluate_episode(episode, method, parameter).metrics["dimension_weighted_semantic_error"])
                for episode in episodes
            ]
        )
    )


def select_parameter(episodes: list[Episode], method: str, grid: tuple[float, ...]) -> tuple[float, dict[str, float]]:
    scores = {str(value): episode_weighted_score(episodes, method, value) for value in grid}
    best_score = min(scores.values())
    selected = next(value for value in grid if scores[str(value)] <= best_score + 1e-12)
    return selected, scores


def selected_parameters(lock: dict[str, Any]) -> dict[str, float | None]:
    selection = lock["selection"]
    return {
        "b0_newest": None,
        "b1_uniform": None,
        "b2_exact_act_m0_01": None,
        "b2_tuned_act_oldest_exponential": float(selection["b2_tuned_act_oldest_exponential_m"]),
        "b3_tuned_newest_age_exponential": float(selection["b3_tuned_newest_age_exponential_beta"]),
        "b4_official_cogact_alpha0_1": None,
        "b5_tuned_cogact_cosine": float(selection["b5_tuned_cogact_cosine_alpha"]),
        "b6_control_semantic_similarity": float(selection["b6_control_semantic_temperature"]),
        "b5_tuned_cogact_cosine_semantic_aggregation": float(
            selection["b5_semantic_aggregation_alpha"]
        ),
        "b6_control_semantic_similarity_semantic_aggregation": float(
            selection["b6_semantic_aggregation_temperature"]
        ),
    }


def method_evaluations(
    episodes: list[Episode], parameters: dict[str, float | None]
) -> dict[str, list[Evaluation]]:
    return {
        method: [evaluate_episode(episode, method, parameters[method]) for episode in episodes]
        for method in METHOD_ORDER
    }


def transition_counts(episode: Episode, evaluation: Evaluation) -> dict[str, int | float | None]:
    if len(episode.targets) < 2:
        return {
            "false_count": 0,
            "false_denominator": 0,
            "missed_count": 0,
            "missed_denominator": 0,
            "false_rate": None,
            "missed_rate": None,
        }
    previous = action_sign(episode.previous_targets[1:, 6])
    target = action_sign(episode.targets[1:, 6])
    predicted = action_sign(evaluation.prediction[1:, 6])
    target_transition = target != previous
    predicted_transition = predicted != previous
    false_count = int(np.sum(predicted_transition & ~target_transition))
    false_denominator = int(np.sum(~target_transition))
    missed_count = int(np.sum(~predicted_transition & target_transition))
    missed_denominator = int(np.sum(target_transition))
    return {
        "false_count": false_count,
        "false_denominator": false_denominator,
        "missed_count": missed_count,
        "missed_denominator": missed_denominator,
        "false_rate": false_count / false_denominator if false_denominator else None,
        "missed_rate": missed_count / missed_denominator if missed_denominator else None,
    }


def summarize_method(episodes: list[Episode], evaluations: list[Evaluation]) -> dict[str, Any]:
    episode_means = {
        metric: np.asarray([np.mean(evaluation.metrics[metric]) for evaluation in evaluations])
        for metric in METRIC_NAMES
    }
    frame_metrics = {
        metric: np.concatenate([evaluation.metrics[metric] for evaluation in evaluations])
        for metric in METRIC_NAMES
    }
    transitions = [transition_counts(episode, evaluation) for episode, evaluation in zip(episodes, evaluations, strict=True)]
    false_count = sum(int(row["false_count"]) for row in transitions)
    false_denominator = sum(int(row["false_denominator"]) for row in transitions)
    missed_count = sum(int(row["missed_count"]) for row in transitions)
    missed_denominator = sum(int(row["missed_denominator"]) for row in transitions)
    return {
        "episodes": len(episodes),
        "targets": int(sum(len(episode.targets) for episode in episodes)),
        "episode_weighted": {metric: float(np.mean(values)) for metric, values in episode_means.items()},
        "frame_weighted": {metric: float(np.mean(values)) for metric, values in frame_metrics.items()},
        "weighted_source_age_dataset_steps": float(
            np.mean(np.concatenate([evaluation.weighted_age for evaluation in evaluations]))
        ),
        "weighted_source_age_seconds": float(
            np.mean(np.concatenate([evaluation.weighted_age for evaluation in evaluations])) / DATASET_HZ
        ),
        "gripper_transition_diagnostics": {
            "teacher_forced_previous_demonstration_sign": True,
            "first_frame_excluded": True,
            "false_transition_count": false_count,
            "false_transition_denominator": false_denominator,
            "false_transition_rate": false_count / false_denominator if false_denominator else None,
            "missed_transition_count": missed_count,
            "missed_transition_denominator": missed_denominator,
            "missed_transition_rate": missed_count / missed_denominator if missed_denominator else None,
        },
    }


def per_task_rows(
    episodes: list[Episode], evaluations: dict[str, list[Evaluation]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tasks = sorted(set(episode.task_id for episode in episodes))
    for method in METHOD_ORDER:
        for task in tasks:
            indices = [index for index, episode in enumerate(episodes) if episode.task_id == task]
            row: dict[str, Any] = {
                "method": method,
                "task_id": task,
                "episodes": len(indices),
                "targets": int(sum(len(episodes[index].targets) for index in indices)),
            }
            for metric in METRIC_NAMES:
                episode_values = [
                    float(np.mean(evaluations[method][index].metrics[metric])) for index in indices
                ]
                row[metric] = float(np.mean(episode_values))
            rows.append(row)
    return rows


def paired_comparison(
    episodes: list[Episode], left: list[Evaluation], right: list[Evaluation], metric: str
) -> dict[str, Any]:
    differences = np.asarray(
        [
            float(np.mean(a.metrics[metric]) - np.mean(b.metrics[metric]))
            for a, b in zip(left, right, strict=True)
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(EPISODE_BOOTSTRAP_SEED)
    indices = rng.integers(0, len(differences), size=(BOOTSTRAP_DRAWS, len(differences)))
    draws = differences[indices].mean(axis=1)
    per_task = {
        str(task): float(np.mean(differences[[episode.task_id == task for episode in episodes]]))
        for task in sorted(set(episode.task_id for episode in episodes))
    }
    task_values = np.asarray(list(per_task.values()), dtype=np.float64)
    task_rng = np.random.default_rng(TASK_BOOTSTRAP_SEED)
    task_indices = task_rng.integers(0, len(task_values), size=(BOOTSTRAP_DRAWS, len(task_values)))
    task_draws = task_values[task_indices].mean(axis=1)
    leave_one_task_out = {}
    for task in sorted(set(episode.task_id for episode in episodes)):
        keep = np.asarray([episode.task_id != task for episode in episodes])
        leave_one_task_out[str(task)] = float(np.mean(differences[keep]))
    ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    consistent = bool(
        ci[1] < 0.0
        and sum(value < 0.0 for value in per_task.values()) >= 7
        and all(value < 0.0 for value in leave_one_task_out.values())
    )
    return {
        "episode_weighted_mean_difference": float(np.mean(differences)),
        "episode_bootstrap_ci95": ci,
        "task_weighted_mean_difference": float(np.mean(task_values)),
        "task_cluster_bootstrap_ci95": [
            float(np.percentile(task_draws, 2.5)),
            float(np.percentile(task_draws, 97.5)),
        ],
        "per_task_mean_difference": per_task,
        "negative_task_count": int(sum(value < 0.0 for value in per_task.values())),
        "leave_one_task_out_mean_difference": leave_one_task_out,
        "consistent_advantage": consistent,
        "episodes": len(episodes),
        "tasks": len(task_values),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    }


def digitized_summary(values: np.ndarray, bins: np.ndarray, errors: np.ndarray) -> dict[str, Any]:
    labels = np.digitize(values, bins, right=True)
    result: dict[str, Any] = {}
    for label in range(len(bins) + 1):
        select = labels == label
        if np.any(select):
            result[str(label)] = {
                "targets": int(np.sum(select)),
                "mean_dimension_weighted_semantic_error": float(np.mean(errors[select])),
                "value_mean": float(np.mean(values[select])),
            }
    return result


def stratified_summaries(
    episodes: list[Episode], evaluations: dict[str, list[Evaluation]], quartiles: dict[str, list[float]]
) -> dict[str, Any]:
    candidate_count = np.concatenate([episode.mask.sum(axis=1) for episode in episodes])
    overall_disagreement = np.concatenate(
        [episode.semantic_distances.sum(axis=1) / episode.mask.sum(axis=1) for episode in episodes]
    )
    translation = np.concatenate([episode.translation_disagreement for episode in episodes])
    rotation = np.concatenate([episode.rotation_disagreement for episode in episodes])
    gripper = np.concatenate([episode.gripper_disagreement_fraction for episode in episodes])
    normalized_time = np.concatenate(
        [np.arange(len(episode.targets)) / max(len(episode.targets) - 1, 1) for episode in episodes]
    )
    count_bins = np.asarray([1, 4, 8, 16, 32, 64], dtype=np.float64)
    age_bins = np.asarray([0.0, 1.0, 3.0, 7.0, 15.0, 31.0, 63.0], dtype=np.float64)
    result: dict[str, Any] = {}
    for method in PRIMARY_METHODS:
        errors = np.concatenate(
            [evaluation.metrics["dimension_weighted_semantic_error"] for evaluation in evaluations[method]]
        )
        weighted_age = np.concatenate([evaluation.weighted_age for evaluation in evaluations[method]])
        result[method] = {
            "candidate_count": digitized_summary(candidate_count, count_bins, errors),
            "weighted_candidate_age_dataset_steps": digitized_summary(weighted_age, age_bins, errors),
            "weighted_candidate_age_seconds_mean": float(np.mean(weighted_age) / DATASET_HZ),
            "candidate_semantic_disagreement_validation_quartiles": digitized_summary(
                overall_disagreement, np.asarray(quartiles["semantic_mean"]), errors
            ),
            "translation_disagreement_validation_quartiles": digitized_summary(
                translation, np.asarray(quartiles["translation_max"]), errors
            ),
            "rotation_disagreement_validation_quartiles": digitized_summary(
                rotation, np.asarray(quartiles["rotation_max_radians"]), errors
            ),
            "gripper_candidate_sign_disagreement": {
                "none": {
                    "targets": int(np.sum(gripper == 0.0)),
                    "mean_dimension_weighted_semantic_error": float(np.mean(errors[gripper == 0.0])),
                },
                "any": {
                    "targets": int(np.sum(gripper > 0.0)),
                    "mean_dimension_weighted_semantic_error": (
                        float(np.mean(errors[gripper > 0.0])) if np.any(gripper > 0.0) else None
                    ),
                    "mean_disagreeing_candidate_fraction": (
                        float(np.mean(gripper[gripper > 0.0])) if np.any(gripper > 0.0) else None
                    ),
                },
            },
            "normalized_time_deciles_diagnostic_only": digitized_summary(
                normalized_time, np.arange(0.1, 1.0, 0.1), errors
            ),
        }
    return result


def hard_oracle(episode: Episode) -> Evaluation:
    prediction = np.empty_like(episode.targets)
    weights = np.zeros_like(episode.ages, dtype=np.float64)
    for row in range(len(episode.targets)):
        count = int(episode.mask[row].sum())
        losses = target_metric_one(episode.experts[row, :count], episode.targets[row])
        minimum = float(np.min(losses))
        ties = np.flatnonzero(np.isclose(losses, minimum, atol=1e-12, rtol=0.0))
        index = int(ties[-1])
        prediction[row] = episode.experts[row, index]
        weights[row, index] = 1.0
    return Evaluation(
        prediction=prediction,
        weights=weights,
        weighted_age=np.sum(weights * episode.ages, axis=1),
        metrics=target_metrics(prediction, episode.targets),
    )


def convex_greedy_oracle(episode: Episode, hard: Evaluation) -> tuple[Evaluation, dict[str, Any]]:
    prediction = hard.prediction.copy()
    rounds_used = np.zeros(len(prediction), dtype=np.int16)
    improvements = np.zeros(len(prediction), dtype=np.float64)
    for row in range(len(prediction)):
        count = int(episode.mask[row].sum())
        current = prediction[row].copy()
        current_loss = float(target_metric_one(current[None, :], episode.targets[row])[0])
        initial_loss = current_loss
        for round_index in range(32):
            candidates = episode.experts[row, :count]
            mixtures = np.concatenate(
                [
                    (1.0 - step) * current[None, :] + step * candidates
                    for step in ORACLE_LAMBDAS
                ],
                axis=0,
            )
            losses = target_metric_one(mixtures, episode.targets[row])
            best = int(np.argmin(losses))
            best_loss = float(losses[best])
            if current_loss - best_loss < 1e-12:
                break
            current = mixtures[best]
            current_loss = best_loss
            rounds_used[row] = round_index + 1
        prediction[row] = current
        improvements[row] = initial_loss - current_loss
    metrics = target_metrics(prediction, episode.targets)
    if np.any(metrics["dimension_weighted_semantic_error"] > hard.metrics["dimension_weighted_semantic_error"] + 1e-11):
        raise RuntimeError("Convex oracle became worse than its hard-oracle initialization")
    evaluation = Evaluation(
        prediction=prediction,
        weights=np.full_like(episode.ages, np.nan, dtype=np.float64),
        weighted_age=np.full(len(prediction), np.nan, dtype=np.float64),
        metrics=metrics,
    )
    diagnostics = {
        "mean_rounds_used": float(np.mean(rounds_used)),
        "maximum_rounds_used": int(np.max(rounds_used)),
        "targets_improved_beyond_hard": int(np.sum(improvements > 1e-12)),
        "mean_target_loss_improvement_beyond_hard": float(np.mean(improvements)),
    }
    return evaluation, diagnostics


def oracle_outputs(
    episodes: list[Episode], nonoracle: dict[str, list[Evaluation]], best_method: str
) -> tuple[dict[str, Any], list[Evaluation], list[Evaluation]]:
    hard = [hard_oracle(episode) for episode in episodes]
    convex_pairs = [convex_greedy_oracle(episode, current) for episode, current in zip(episodes, hard, strict=True)]
    convex = [pair[0] for pair in convex_pairs]
    convex_diagnostics = [pair[1] for pair in convex_pairs]
    hard_summary = summarize_method(episodes, hard)
    convex_summary = summarize_method(episodes, convex)
    best_summary = summarize_method(episodes, nonoracle[best_method])
    hard_comparison = paired_comparison(
        episodes, nonoracle[best_method], hard, "dimension_weighted_semantic_error"
    )
    convex_comparison = paired_comparison(
        episodes, nonoracle[best_method], convex, "dimension_weighted_semantic_error"
    )
    return (
        {
            "best_validation_selected_nonoracle": best_method,
            "best_nonoracle_episode_weighted_Lsem": best_summary["episode_weighted"][
                "dimension_weighted_semantic_error"
            ],
            "hard_scalar_source_oracle": hard_summary,
            "conservative_scalar_convex_mixture_oracle": convex_summary,
            "scalar_contextual_headroom_hard": hard_comparison,
            "scalar_contextual_headroom_convex": convex_comparison,
            "convex_oracle_algorithm": {
                "certified_global_optimum": False,
                "initialization": "hard scalar source oracle",
                "rounds": 32,
                "lambda_grid": list(ORACLE_LAMBDAS),
                "mean_rounds_used": float(
                    np.mean([row["mean_rounds_used"] for row in convex_diagnostics])
                ),
                "maximum_rounds_used": int(
                    max(row["maximum_rounds_used"] for row in convex_diagnostics)
                ),
                "targets_improved_beyond_hard": int(
                    sum(row["targets_improved_beyond_hard"] for row in convex_diagnostics)
                ),
            },
            "interpretation": "Teacher-forced target-informed upper-bound analysis; not a deployable selector or control result.",
        },
        hard,
        convex,
    )


def cache_tree_summary(manifest: dict[str, Any], cache_root: Path) -> dict[str, Any]:
    entries = sorted(manifest["entries"], key=lambda row: (row["split"], int(row["episode_id"])))
    lines = []
    for entry in entries:
        relative = Path(entry["cache_file"]).resolve().relative_to(cache_root.resolve())
        lines.append(f"{entry['sha256']}  {relative.as_posix()}\n")
    digest = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    return {
        "path": str(cache_root.resolve()),
        "prediction_files": len(entries),
        "total_bytes": int(sum(int(entry["bytes"]) for entry in entries)),
        "content_tree_sha256": digest,
        "content_tree_definition": "SHA256 of sorted '<file_sha256>  <relative_path>\\n' lines for episode NPZ files.",
        "source_queries": int(sum(int(entry["completed_frames"]) for entry in entries)),
        "full_prediction_shape_total": [
            int(sum(int(entry["completed_frames"]) for entry in entries)),
            CHUNK_SIZE,
            ACTION_DIM,
        ],
    }


def self_test() -> None:
    from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
    import torch

    chunks = []
    ensembler = ACTTemporalEnsembler(0.01, 4)
    observed = []
    for source in range(4):
        chunk = np.asarray([[10.0 * source + offset] for offset in range(4)], dtype=np.float32)
        chunks.append(chunk)
        observed.append(float(ensembler.update(torch.from_numpy(chunk[None, :, :])).item()))
        candidates = np.asarray([chunks[q][source - q, 0] for q in range(source + 1)])
        weights = np.exp(-0.01 * np.arange(len(candidates)))
        expected = float(np.sum(weights * candidates) / np.sum(weights))
        if not math.isclose(observed[-1], expected, rel_tol=1e-6, abs_tol=1e-6):
            raise RuntimeError(f"ACT exact-convention self-test failed at t={source}")

    experts = np.zeros((3, 4, 7), dtype=np.float64)
    mask = np.zeros((3, 4), dtype=bool)
    ages = np.zeros((3, 4), dtype=np.int16)
    for row in range(3):
        count = row + 1
        experts[row, :count] = np.arange(count)[:, None]
        mask[row, :count] = True
        ages[row, :count] = np.arange(count - 1, -1, -1)
    semantic, cosine, _, _, _, _ = semantic_candidate_diagnostics(experts, mask)
    dummy = Episode(0, 0, np.zeros((3, 7)), np.zeros((3, 7)), experts, ages, mask, semantic, cosine, np.zeros(3), np.zeros(3), np.zeros(3))
    weights = weights_for_method(dummy, "b2_exact_act_m0_01")
    for row in range(3):
        count = row + 1
        expected = np.exp(-0.01 * np.arange(count))
        expected /= expected.sum()
        if not np.allclose(weights[row, :count], expected):
            raise RuntimeError("Offline ACT weights do not match source-order convention")
    print(json.dumps({"self_test": "passed", "act_outputs": observed}, indent=2))


def tune(args: argparse.Namespace) -> None:
    episodes, inventory, manifest = load_split(args, "validation")
    selections: dict[str, float] = {}
    scores: dict[str, dict[str, float]] = {}
    tuning_specs = (
        ("b2_tuned_act_oldest_exponential", AGE_GRID, "b2_tuned_act_oldest_exponential_m"),
        ("b3_tuned_newest_age_exponential", AGE_GRID, "b3_tuned_newest_age_exponential_beta"),
        ("b5_tuned_cogact_cosine", KERNEL_GRID, "b5_tuned_cogact_cosine_alpha"),
        ("b6_control_semantic_similarity", KERNEL_GRID, "b6_control_semantic_temperature"),
        (
            "b5_tuned_cogact_cosine_semantic_aggregation",
            KERNEL_GRID,
            "b5_semantic_aggregation_alpha",
        ),
        (
            "b6_control_semantic_similarity_semantic_aggregation",
            KERNEL_GRID,
            "b6_semantic_aggregation_temperature",
        ),
    )
    for method, grid, key in tuning_specs:
        selected, current_scores = select_parameter(episodes, method, grid)
        selections[key] = selected
        scores[method] = current_scores
        print(f"selected {method}={selected}", flush=True)

    temporary_lock = {"selection": selections}
    parameters = selected_parameters(temporary_lock)
    validation_scores = {
        method: episode_weighted_score(episodes, method, parameters[method])
        for method in PRIMARY_METHODS
    }
    best_score = min(validation_scores.values())
    best_method = next(
        method for method in PRIMARY_METHODS if validation_scores[method] <= best_score + 1e-12
    )
    semantic_means = np.concatenate(
        [episode.semantic_distances.sum(axis=1) / episode.mask.sum(axis=1) for episode in episodes]
    )
    translation_max = np.concatenate([episode.translation_disagreement for episode in episodes])
    rotation_max = np.concatenate([episode.rotation_disagreement for episode in episodes])
    quartiles = {
        "semantic_mean": np.quantile(semantic_means, [0.25, 0.5, 0.75]).astype(float).tolist(),
        "translation_max": np.quantile(translation_max, [0.25, 0.5, 0.75]).astype(float).tolist(),
        "rotation_max_radians": np.quantile(rotation_max, [0.25, 0.5, 0.75]).astype(float).tolist(),
    }
    validation_entries = [
        entry for entry in manifest["entries"] if entry["split"] == "validation"
    ]
    lock = {
        "schema_version": 1,
        "scope": "Validation-only Gate-3A1 hyperparameter and diagnostic-cutpoint lock; no test metrics read.",
        "registration_commit": REGISTRATION_COMMIT,
        "protocol_sha256": sha256(PROTOCOL),
        "analysis_script": str(Path(__file__).resolve().relative_to(ROOT)),
        "analysis_script_sha256": sha256(Path(__file__)),
        "validation_cache_manifest_sha256_at_selection": sha256(args.cache_manifest),
        "validation_cache_files": {
            str(entry["episode_id"]): entry["sha256"] for entry in validation_entries
        },
        "validation_cohort": {
            "episodes": len(episodes),
            "targets": int(sum(len(episode.targets) for episode in episodes)),
            "episode_ids": [episode.episode_id for episode in episodes],
            "tasks": sorted(set(episode.task_id for episode in episodes)),
        },
        "selection": selections,
        "selection_grids": {
            "act_and_age_coefficient": list(AGE_GRID),
            "cogact_alpha_and_semantic_temperature": list(KERNEL_GRID),
        },
        "validation_grid_scores_episode_weighted_Lsem": scores,
        "validation_selected_primary_method_scores": validation_scores,
        "best_validation_selected_nonoracle": best_method,
        "diagnostic_quartile_cutpoints": quartiles,
        "tie_rule": "Within 1e-12 choose the first value/method in preregistered display order.",
        "test_metrics_exposed": False,
    }
    atomic_json(args.validation_lock, lock)
    print(json.dumps({"selection": selections, "best_method": best_method, "output": str(args.validation_lock)}, indent=2))


def require_committed_lock(lock_path: Path, lock: dict[str, Any]) -> None:
    relative = str(lock_path.resolve().relative_to(ROOT))
    git("ls-files", "--error-unmatch", "--", relative)
    if sha256(PROTOCOL) != lock["protocol_sha256"]:
        raise RuntimeError("Protocol hash changed after validation lock")
    if sha256(Path(__file__)) != lock["analysis_script_sha256"]:
        raise RuntimeError("Analysis script changed after validation lock")
    commit = git("log", "-1", "--format=%H", "--", relative)
    if not commit:
        raise RuntimeError("Validation lock has no containing commit")
    git("merge-base", "--is-ancestor", commit, "HEAD")
    status = git("status", "--porcelain", "--", relative, str(PROTOCOL.relative_to(ROOT)), str(Path(__file__).relative_to(ROOT)))
    if status:
        raise RuntimeError(f"Locked analysis inputs have uncommitted changes:\n{status}")


def evaluate(args: argparse.Namespace) -> None:
    lock = json.loads(args.validation_lock.read_text(encoding="utf-8"))
    require_committed_lock(args.validation_lock, lock)
    episodes, inventory, manifest = load_split(args, "test")
    parameters = selected_parameters(lock)
    evaluations = method_evaluations(episodes, parameters)
    summaries = {
        method: summarize_method(episodes, evaluations[method]) for method in METHOD_ORDER
    }
    task_rows = per_task_rows(episodes, evaluations)

    comparison_pairs = (
        ("b6_control_semantic_similarity", "b2_exact_act_m0_01"),
        ("b6_control_semantic_similarity", "b2_tuned_act_oldest_exponential"),
        ("b6_control_semantic_similarity", "b3_tuned_newest_age_exponential"),
        ("b6_control_semantic_similarity", "b4_official_cogact_alpha0_1"),
        ("b6_control_semantic_similarity", "b5_tuned_cogact_cosine"),
        ("b6_control_semantic_similarity", "b1_uniform"),
        (
            "b6_control_semantic_similarity_semantic_aggregation",
            "b5_tuned_cogact_cosine_semantic_aggregation",
        ),
        (lock["best_validation_selected_nonoracle"], "b0_newest"),
        ("b1_uniform", "b0_newest"),
        ("b2_exact_act_m0_01", "b0_newest"),
        ("b3_tuned_newest_age_exponential", "b0_newest"),
        ("b5_tuned_cogact_cosine", "b0_newest"),
        ("b6_control_semantic_similarity", "b0_newest"),
    )
    comparisons: dict[str, dict[str, Any]] = {}
    comparison_rows: list[dict[str, Any]] = []
    for left, right in comparison_pairs:
        pair = f"{left}_minus_{right}"
        if pair in comparisons:
            continue
        comparisons[pair] = {}
        for metric in METRIC_NAMES:
            result = paired_comparison(episodes, evaluations[left], evaluations[right], metric)
            comparisons[pair][metric] = result
            comparison_rows.append(
                {
                    "left_method": left,
                    "right_method": right,
                    "metric": metric,
                    "episode_weighted_mean_difference": result["episode_weighted_mean_difference"],
                    "episode_ci95_low": result["episode_bootstrap_ci95"][0],
                    "episode_ci95_high": result["episode_bootstrap_ci95"][1],
                    "task_weighted_mean_difference": result["task_weighted_mean_difference"],
                    "task_cluster_ci95_low": result["task_cluster_bootstrap_ci95"][0],
                    "task_cluster_ci95_high": result["task_cluster_bootstrap_ci95"][1],
                    "negative_task_count": result["negative_task_count"],
                    "consistent_advantage": result["consistent_advantage"],
                    "episodes": result["episodes"],
                    "tasks": result["tasks"],
                }
            )

    primary_key = "dimension_weighted_semantic_error"
    temporal_pair = (
        f"{lock['best_validation_selected_nonoracle']}_minus_b0_newest"
    )
    semantic_cogact = comparisons[
        "b6_control_semantic_similarity_minus_b5_tuned_cogact_cosine"
    ][primary_key]
    strong_pairs = [
        comparisons["b6_control_semantic_similarity_minus_b2_exact_act_m0_01"][primary_key],
        comparisons[
            "b6_control_semantic_similarity_minus_b2_tuned_act_oldest_exponential"
        ][primary_key],
        comparisons[
            "b6_control_semantic_similarity_minus_b3_tuned_newest_age_exponential"
        ][primary_key],
    ]
    temporal_pass = comparisons[temporal_pair][primary_key]["consistent_advantage"]
    semantic_pass = semantic_cogact["consistent_advantage"]
    strong_pass = semantic_pass and all(pair["consistent_advantage"] for pair in strong_pairs)
    if not temporal_pass:
        decision = "FAIL-TEMPORAL"
    elif not semantic_pass:
        decision = "FAIL-SEMANTIC"
    elif strong_pass:
        decision = "STRONG-PASS"
    elif any(pair["episode_bootstrap_ci95"][0] <= 0.0 <= pair["episode_bootstrap_ci95"][1] for pair in strong_pairs):
        decision = "PARTIAL"
    else:
        decision = "PASS-SEMANTIC"

    oracle, _, _ = oracle_outputs(
        episodes, evaluations, str(lock["best_validation_selected_nonoracle"])
    )
    strata = stratified_summaries(
        episodes, evaluations, lock["diagnostic_quartile_cutpoints"]
    )
    cache_summary = cache_tree_summary(manifest, args.cache_root.resolve())
    manifest["local_only_cache_integrity"] = cache_summary
    atomic_json(args.cache_manifest, manifest)

    output = {
        "schema_version": 1,
        "decision": decision,
        "scientific_scope": "Dense teacher-forced offline Gate-3A1; no closed-loop or policy-improvement claim.",
        "protocol": {
            "path": str(PROTOCOL.relative_to(ROOT)),
            "registration_commit": REGISTRATION_COMMIT,
            "sha256": sha256(PROTOCOL),
        },
        "analysis": {
            "script": str(Path(__file__).relative_to(ROOT)),
            "sha256": sha256(Path(__file__)),
            "validation_lock": str(args.validation_lock.relative_to(ROOT)),
            "validation_lock_sha256": sha256(args.validation_lock),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "episode_bootstrap_seed": EPISODE_BOOTSTRAP_SEED,
            "task_bootstrap_seed": TASK_BOOTSTRAP_SEED,
        },
        "cohort": {
            "test_episodes": len(episodes),
            "test_targets": int(sum(len(episode.targets) for episode in episodes)),
            "task_ids": sorted(set(episode.task_id for episode in episodes)),
            "dataset_frequency_hz": DATASET_HZ,
            "candidate_count_range": [
                int(min(np.min(episode.mask.sum(axis=1)) for episode in episodes)),
                int(max(np.max(episode.mask.sum(axis=1)) for episode in episodes)),
            ],
        },
        "cache": cache_summary,
        "metric_contract": {
            "primary": "Episode-weighted mean of (3*translation_normalized_mse + 3*rotation_normalized_sq + gripper_sign_error)/7.",
            "action_std": ACTION_STD.tolist(),
            "rotation_normalization": float(np.sum(ACTION_STD[3:6] ** 2)),
            "gripper": "Sign error; zero is positive.",
        },
        "validation_selection": lock,
        "method_parameters": parameters,
        "test_method_summaries": summaries,
        "paired_test_comparisons": comparisons,
        "stratified_test_diagnostics": strata,
        "gate_logic": {
            "best_validation_selected_nonoracle": lock["best_validation_selected_nonoracle"],
            "temporal_consistent_advantage_over_newest": temporal_pass,
            "semantic_consistent_advantage_over_tuned_cogact": semantic_pass,
            "semantic_consistent_advantage_over_all_act_age_baselines": strong_pass,
            "decision": decision,
        },
        "oracle_summary": oracle,
        "interpretation_limits": [
            "All observations are teacher-forced demonstration states.",
            "Offline target error is not closed-loop success or control quality.",
            "The hard and convex oracles observe the demonstrated target and are unattainable.",
            "The convex oracle is a conservative greedy construction, not a certified global optimum.",
            "Gripper transition diagnostics use the previous demonstrated sign.",
            "Dataset ages are 10 Hz steps and are not 20 Hz deployment ticks.",
            "Diagnostic strata cannot change the preregistered primary decision.",
        ],
    }
    atomic_json(args.metrics_output, output)
    atomic_json(args.oracle_output, oracle)
    atomic_csv(args.per_task_output, list(task_rows[0]), task_rows)
    atomic_csv(args.pairwise_output, list(comparison_rows[0]), comparison_rows)
    print(
        json.dumps(
            {
                "decision": decision,
                "test_episodes": len(episodes),
                "test_targets": sum(len(episode.targets) for episode in episodes),
                "outputs": {
                    "metrics": str(args.metrics_output),
                    "per_task": str(args.per_task_output),
                    "pairwise": str(args.pairwise_output),
                    "oracle": str(args.oracle_output),
                },
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.mode == "self-test":
        self_test()
    elif args.mode == "validate-cache":
        inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
        manifest = json.loads(args.cache_manifest.read_text(encoding="utf-8"))
        summary = {}
        for split in ("validation", "test"):
            expected = int(inventory["splits"][split]["episodes"])
            found = len(
                [
                    entry
                    for entry in manifest["entries"]
                    if entry["split"] == split and entry["status"] == "complete"
                ]
            )
            if found == expected:
                ids = verify_cache_split(split, inventory, manifest, args.cache_root.resolve())
                summary[split] = {"status": "complete", "episodes": len(ids)}
            else:
                summary[split] = {"status": "incomplete", "episodes": found, "expected": expected}
        print(json.dumps(summary, indent=2))
    elif args.mode == "tune":
        tune(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
