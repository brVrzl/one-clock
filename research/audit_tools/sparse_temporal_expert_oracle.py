#!/usr/bin/env python3
"""Oracle-first audit of the saved, sparsely sampled overlapping ACT chunks.

The cache contains source observations roughly every 25 demonstration frames,
not dense per-step predictions.  Results are therefore a sparse teacher-forced
upper-bound diagnostic and must not be described as a dense ACT routing result.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DATASET = Path("/home/thor/datasets/libero_object_25_08_23_lerobotv2.1")
CACHE = ROOT / "experiments/temporal_reliability/reliability_dataset.npz"
BUNDLE = ROOT / "experiments/dynamic_reliability_horizon/artifact_handoff/minimal_y_refresh_training_bundle.npz"
ACTION_STD = np.asarray(
    [0.2681190073490143, 0.4384443759918213, 0.4475117325782776,
     0.024448219686746597, 0.04936208948493004, 0.042103495448827744,
     0.9974462985992432],
    dtype=np.float64,
)
AGE_GRID = (0, 1, 2, 4, 8, 16, 25, 50, 75, 99)
EXP_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
SIMILARITY_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260821


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "research/audit_outputs")
    return parser.parse_args()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_actions(dataset: Path, episode_ids: np.ndarray) -> dict[int, np.ndarray]:
    result = {}
    for episode_id in sorted(set(int(value) for value in episode_ids)):
        path = dataset / "data/chunk-000" / f"episode_{episode_id:06d}.parquet"
        result[episode_id] = np.asarray(pq.read_table(path, columns=["action"])["action"].to_pylist(), dtype=np.float64)
    return result


def phase(progress: float) -> int:
    return min(int(progress * 3), 2)


def nearest_expert(ages: np.ndarray, desired: int) -> int:
    distance = np.abs(ages - desired)
    candidates = np.flatnonzero(distance == distance.min())
    return int(candidates[np.argmin(ages[candidates])])


def convex_mixture(experts: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Exact small-simplex least squares by enumerating active expert sets."""

    count = len(experts)
    if count == 1:
        return experts[0].copy()
    best_error = float("inf")
    best_prediction = experts[0].copy()
    for bitmask in range(1, 1 << count):
        indices = [index for index in range(count) if bitmask & (1 << index)]
        active = experts[indices]
        gram = active @ active.T
        rhs = active @ target
        size = len(indices)
        system = np.block([
            [gram, np.ones((size, 1))],
            [np.ones((1, size)), np.zeros((1, 1))],
        ])
        solution = np.linalg.lstsq(system, np.concatenate((rhs, [1.0])), rcond=None)[0][:size]
        if np.any(solution < -1e-8):
            continue
        solution = np.maximum(solution, 0.0)
        solution /= solution.sum()
        prediction = solution @ active
        objective = float(np.sum((prediction - target) ** 2))
        if objective < best_error:
            best_error = objective
            best_prediction = prediction
    return best_prediction


def errors(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    normalized = (prediction - target) / ACTION_STD
    arm = float(np.mean(normalized[:6] ** 2))
    gripper = float(normalized[6] ** 2)
    return {
        "raw_action_mse": float(np.mean((prediction - target) ** 2)),
        "normalized_dim_mse": float(np.mean(normalized**2)),
        "arm_normalized_mse": arm,
        "gripper_normalized_mse": gripper,
        "group_balanced_normalized_mse": 0.5 * (arm + gripper),
        "gripper_sign_error": float((prediction[6] >= 0.0) != (target[6] >= 0.0)),
    }


def choose_on_validation(
    entries: list[dict[str, Any]],
    candidates: tuple[Any, ...],
    prediction: Callable[[dict[str, Any], Any], np.ndarray],
    metric: str,
) -> Any:
    validation = [entry for entry in entries if entry["split"] == 1]
    return min(
        candidates,
        key=lambda candidate: float(np.mean([
            errors(prediction(entry, candidate), entry["target"])[metric] for entry in validation
        ])),
    )


def episode_bootstrap_difference(
    left: list[float], right: list[float], episode_ids: list[int]
) -> list[float]:
    by_episode: dict[int, list[float]] = defaultdict(list)
    for a, b, episode_id in zip(left, right, episode_ids, strict=True):
        by_episode[int(episode_id)].append(float(a - b))
    values = np.asarray([np.mean(items) for items in by_episode.values()])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    draws = values[indices].mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = np.load(CACHE, allow_pickle=False)
    bundle = np.load(BUNDLE, allow_pickle=False)
    episode_ids = cache["episode_index"].astype(int)
    frames = cache["frame_index"].astype(int)
    tasks = cache["task_index"].astype(int)
    predictions = cache["predicted_actions"].astype(np.float64)
    actions = load_actions(args.dataset, episode_ids)
    episode_split: dict[int, int] = {}
    for episode_id, split in zip(bundle["episode_index"], bundle["split_membership"], strict=True):
        episode_split.setdefault(int(episode_id), int(split))
        if episode_split[int(episode_id)] != int(split):
            raise RuntimeError("episode split is not episode-level")

    row_by_episode: dict[int, list[int]] = defaultdict(list)
    for row, episode_id in enumerate(episode_ids):
        row_by_episode[int(episode_id)].append(row)
    entries: list[dict[str, Any]] = []
    age_counts: Counter[int] = Counter()
    for episode_id, rows in row_by_episode.items():
        rows.sort(key=lambda row: frames[row])
        sequence = actions[episode_id]
        task = int(tasks[rows[0]])
        for target_time in range(len(sequence)):
            sources = [row for row in rows if frames[row] <= target_time < frames[row] + 100]
            if len(sources) < 2:
                continue
            ages = np.asarray([target_time - frames[row] for row in sources], dtype=np.int32)
            experts = np.stack([predictions[row, age] for row, age in zip(sources, ages, strict=True)])
            for age in ages:
                age_counts[int(age)] += 1
            entries.append(
                {
                    "episode": episode_id,
                    "task": task,
                    "target_time": target_time,
                    "progress": target_time / len(sequence),
                    "phase": phase(target_time / len(sequence)),
                    "split": episode_split[episode_id],
                    "ages": ages,
                    "experts": experts,
                    "target": sequence[target_time],
                }
            )

    def fixed(entry: dict[str, Any], desired: int) -> np.ndarray:
        return entry["experts"][nearest_expert(entry["ages"], desired)]

    def exponential(entry: dict[str, Any], coefficient: float) -> np.ndarray:
        weights = np.exp(-coefficient * entry["ages"].astype(np.float64))
        weights /= weights.sum()
        return weights @ entry["experts"]

    def similarity(entry: dict[str, Any], temperature: float) -> np.ndarray:
        newest = entry["experts"][int(np.argmin(entry["ages"]))]
        distance = np.mean(((entry["experts"] - newest) / ACTION_STD) ** 2, axis=1)
        weights = np.exp(-distance / temperature)
        weights /= weights.sum()
        return weights @ entry["experts"]

    selected_fixed = choose_on_validation(entries, AGE_GRID, fixed, "group_balanced_normalized_mse")
    selected_exp = choose_on_validation(entries, EXP_GRID, exponential, "group_balanced_normalized_mse")
    selected_similarity = choose_on_validation(entries, SIMILARITY_GRID, similarity, "group_balanced_normalized_mse")
    selected_group_age = {
        "arm": choose_on_validation(entries, AGE_GRID, fixed, "arm_normalized_mse"),
        "gripper": choose_on_validation(entries, AGE_GRID, fixed, "gripper_normalized_mse"),
    }
    selected_phase_age: dict[str, dict[int, int]] = {"arm": {}, "gripper": {}}
    for group, metric in (("arm", "arm_normalized_mse"), ("gripper", "gripper_normalized_mse")):
        for phase_code in range(3):
            selected_phase_age[group][phase_code] = choose_on_validation(
                [entry for entry in entries if entry["phase"] == phase_code], AGE_GRID, fixed, metric
            )

    def mixed_fixed(entry: dict[str, Any], age_map: dict[str, int]) -> np.ndarray:
        arm = fixed(entry, age_map["arm"])
        gripper = fixed(entry, age_map["gripper"])
        return np.concatenate((arm[:6], gripper[6:]))

    def phase_mixed(entry: dict[str, Any]) -> np.ndarray:
        return mixed_fixed(entry, {
            "arm": selected_phase_age["arm"][entry["phase"]],
            "gripper": selected_phase_age["gripper"][entry["phase"]],
        })

    def scalar_oracle(entry: dict[str, Any]) -> np.ndarray:
        scores = [errors(expert, entry["target"])["normalized_dim_mse"] for expert in entry["experts"]]
        return entry["experts"][int(np.argmin(scores))]

    def group_oracle(entry: dict[str, Any]) -> np.ndarray:
        arm_scores = [errors(expert, entry["target"])["arm_normalized_mse"] for expert in entry["experts"]]
        grip_scores = [errors(expert, entry["target"])["gripper_normalized_mse"] for expert in entry["experts"]]
        return np.concatenate((
            entry["experts"][int(np.argmin(arm_scores)), :6],
            entry["experts"][int(np.argmin(grip_scores)), 6:],
        ))

    def scalar_soft_oracle(entry: dict[str, Any]) -> np.ndarray:
        experts = entry["experts"] / ACTION_STD
        return convex_mixture(experts, entry["target"] / ACTION_STD) * ACTION_STD

    def group_soft_oracle(entry: dict[str, Any]) -> np.ndarray:
        arm = convex_mixture(entry["experts"][:, :6] / ACTION_STD[:6], entry["target"][:6] / ACTION_STD[:6]) * ACTION_STD[:6]
        gripper = convex_mixture(entry["experts"][:, 6:] / ACTION_STD[6:], entry["target"][6:] / ACTION_STD[6:]) * ACTION_STD[6:]
        return np.concatenate((arm, gripper))

    methods: dict[str, Callable[[dict[str, Any]], np.ndarray]] = {
        "newest_prediction_only": lambda entry: entry["experts"][int(np.argmin(entry["ages"]))],
        "oldest_valid_prediction": lambda entry: entry["experts"][int(np.argmax(entry["ages"]))],
        "uniform_temporal_ensemble": lambda entry: entry["experts"].mean(axis=0),
        "train_selected_exponential_ensemble": lambda entry: exponential(entry, selected_exp),
        "train_selected_similarity_ensemble": lambda entry: similarity(entry, selected_similarity),
        "train_selected_fixed_age": lambda entry: fixed(entry, selected_fixed),
        "train_selected_fixed_groupwise_age": lambda entry: mixed_fixed(entry, selected_group_age),
        "train_selected_phase_groupwise_age": phase_mixed,
        "oracle_per_sample_scalar_age": scalar_oracle,
        "oracle_per_sample_groupwise_age": group_oracle,
        "oracle_scalar_convex_mixture": scalar_soft_oracle,
        "oracle_groupwise_convex_mixture": group_soft_oracle,
    }
    test_entries = [entry for entry in entries if entry["split"] == 2]
    method_errors: dict[str, list[dict[str, float]]] = {}
    summaries: dict[str, Any] = {}
    for name, method in methods.items():
        current_errors = [errors(method(entry), entry["target"]) for entry in test_entries]
        method_errors[name] = current_errors
        summaries[name] = {
            metric: float(np.mean([item[metric] for item in current_errors]))
            for metric in current_errors[0]
        }
        summaries[name]["per_task_group_balanced_normalized_mse"] = {
            str(task): float(np.mean([
                item["group_balanced_normalized_mse"]
                for item, entry in zip(current_errors, test_entries, strict=True)
                if entry["task"] == task
            ]))
            for task in range(10)
        }
        summaries[name]["per_phase_group_balanced_normalized_mse"] = {
            str(phase_code): float(np.mean([
                item["group_balanced_normalized_mse"]
                for item, entry in zip(current_errors, test_entries, strict=True)
                if entry["phase"] == phase_code
            ]))
            for phase_code in range(3)
        }

    nonoracle_names = list(methods)[:8]
    best_nonoracle = min(nonoracle_names, key=lambda name: summaries[name]["group_balanced_normalized_mse"])
    newest_values = [item["group_balanced_normalized_mse"] for item in method_errors["newest_prediction_only"]]
    episode_vector = [entry["episode"] for entry in test_entries]
    comparisons = {}
    for name in (
        best_nonoracle,
        "oracle_per_sample_scalar_age",
        "oracle_per_sample_groupwise_age",
        "oracle_scalar_convex_mixture",
        "oracle_groupwise_convex_mixture",
    ):
        values = [item["group_balanced_normalized_mse"] for item in method_errors[name]]
        comparisons[name] = {
            "method_minus_newest_mean": float(np.mean(np.asarray(values) - newest_values)),
            "episode_bootstrap_ci95": episode_bootstrap_difference(values, newest_values, episode_vector),
        }

    oracle_group = summaries["oracle_per_sample_groupwise_age"]["group_balanced_normalized_mse"]
    dump(
        args.output_dir / "sparse_temporal_expert_oracle.json",
        {
            "audit_script": str(Path(__file__).relative_to(ROOT)),
            "cohort": {
                "all_sparse_overlap_targets": len(entries),
                "test_sparse_overlap_targets": len(test_entries),
                "test_episodes": len(set(entry["episode"] for entry in test_entries)),
                "minimum_experts": min(len(entry["experts"]) for entry in entries),
                "maximum_experts": max(len(entry["experts"]) for entry in entries),
                "mean_experts": float(np.mean([len(entry["experts"]) for entry in entries])),
                "age_observation_counts": dict(sorted(age_counts.items())),
                "exclusion": "Targets with only one saved expert are excluded. Predictions are sparse source samples, not consecutive per-step ACT queries.",
            },
            "selection": {
                "fixed_age": selected_fixed,
                "exponential_coefficient": selected_exp,
                "similarity_temperature": selected_similarity,
                "fixed_groupwise_age": selected_group_age,
                "phase_groupwise_age": selected_phase_age,
                "selection_split": "validation episodes only",
            },
            "test_metrics": summaries,
            "comparisons_to_newest": comparisons,
            "headroom": {
                "best_nonoracle": best_nonoracle,
                "routing_headroom_best_nonoracle_minus_oracle_groupwise": summaries[best_nonoracle]["group_balanced_normalized_mse"] - oracle_group,
                "groupwise_advantage_oracle_scalar_minus_groupwise": summaries["oracle_per_sample_scalar_age"]["group_balanced_normalized_mse"] - oracle_group,
                "groupwise_advantage_normalized_dim_mse": summaries["oracle_per_sample_scalar_age"]["normalized_dim_mse"] - summaries["oracle_per_sample_groupwise_age"]["normalized_dim_mse"],
                "contextual_advantage_fixed_groupwise_minus_oracle_groupwise": summaries["train_selected_fixed_groupwise_age"]["group_balanced_normalized_mse"] - oracle_group,
            },
            "interpretation_limits": [
                "Every oracle uses the demonstrated action and is therefore an unattainable teacher-forced upper bound.",
                "Convex-mixture oracles can fit demonstration noise and alternative valid actions are not represented.",
                "Offline error is not closed-loop control quality; no rollout-success inference is made.",
                "Group-wise selections may compose action components never jointly predicted and may be physically inconsistent.",
                "The sparse source cadence prevents conclusions about temporal ages 1..99 or standard dense ACT temporal ensembling.",
            ],
        },
    )
    print(json.dumps({
        "entries": len(entries),
        "test_entries": len(test_entries),
        "selection": {
            "fixed": selected_fixed,
            "exp": selected_exp,
            "similarity": selected_similarity,
            "group": selected_group_age,
            "phase_group": selected_phase_age,
        },
        "best_nonoracle": best_nonoracle,
        "headroom": summaries[best_nonoracle]["group_balanced_normalized_mse"] - oracle_group,
        "output": str(args.output_dir / "sparse_temporal_expert_oracle.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
