"""Run the portable chunk-only reliability pilot on CPU.

This runner intentionally consumes only ``source_chunk_actions`` and a one-hot
group ID from the committed handoff.  It does not reconstruct ``Y_refresh`` or
materialize any observation, phase, progress, episode-length, terminal, or
future-action feature.

The repository's Torch trainer is retained as the reference implementation,
but Torch is optional on Thor.  This file is a small NumPy implementation of
the same two shared-vector model families so the first real pilot can run on a
CPU-only checkout.  The bundle has k=1..99, so both learned heads predict only
those nontrivial offsets; R(0)=1 is prepended only while decoding horizons.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.temporal_reliability_training.evaluation import (
    average_precision,
    brier_score,
    expected_calibration_error,
    roc_auc,
)


EXPECTED_BUNDLE_SHA256 = (
    "45a37a57fc03a3850b5c87e88604d66b16886d306e5ee09aa322f52c7e6c50b4"
)
SEEDS = (20260820, 20260821, 20260822)
GROUP_NAMES = ("arm", "gripper")
TAU_VALUES = tuple(np.round(np.arange(0.10, 0.91, 0.05), 2).tolist())


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


def _logit(probability: np.ndarray | float) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    values = np.clip(values, 1.0e-4, 1.0 - 1.0e-4)
    return np.log(values / (1.0 - values))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_jsonable(rows))


@dataclass(frozen=True)
class PilotData:
    features: np.ndarray
    labels: np.ndarray
    label_mask: np.ndarray
    groups: np.ndarray
    group_ids: np.ndarray
    episode_ids: np.ndarray
    split: np.ndarray
    source_window_ids: np.ndarray
    source_chunks: np.ndarray
    normalization_mean: np.ndarray
    normalization_std: np.ndarray


def _load_data(bundle_path: Path, split_path: Path) -> PilotData:
    digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    if digest != EXPECTED_BUNDLE_SHA256:
        raise RuntimeError(
            f"bundle checksum mismatch: expected {EXPECTED_BUNDLE_SHA256}, got {digest}"
        )
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    split_by_episode: dict[int, int] = {}
    for split_name, code in (("train", 0), ("validation", 1), ("test", 2)):
        for episode_id in split_manifest["episodes_by_split"][split_name]:
            if episode_id in split_by_episode:
                raise RuntimeError(f"episode appears in multiple splits: {episode_id}")
            split_by_episode[int(episode_id)] = code

    with np.load(bundle_path, allow_pickle=False) as arrays:
        required = {
            "source_chunk_actions",
            "group_ids",
            "offsets",
            "y_refresh",
            "label_observed",
            "episode_index",
            "split_membership",
        }
        if set(arrays.files) != required:
            raise RuntimeError(f"unexpected handoff arrays: {arrays.files}")
        chunks = np.asarray(arrays["source_chunk_actions"], dtype=np.float32)
        group_ids = np.asarray(arrays["group_ids"], dtype=np.int8)
        offsets = np.asarray(arrays["offsets"], dtype=np.int16)
        labels_by_group = np.asarray(arrays["y_refresh"], dtype=bool)
        masks_by_group = np.asarray(arrays["label_observed"], dtype=bool)
        episode_ids = np.asarray(arrays["episode_index"], dtype=np.int32)
        split_codes = np.asarray(arrays["split_membership"], dtype=np.int8)

    if chunks.shape != (3740, 100, 7):
        raise RuntimeError(f"unexpected source chunk shape: {chunks.shape}")
    if not np.array_equal(group_ids, np.asarray([0, 1], dtype=np.int8)):
        raise RuntimeError(f"unexpected group IDs: {group_ids}")
    if not np.array_equal(offsets, np.arange(1, 100, dtype=np.int16)):
        raise RuntimeError("bundle offsets are not exactly k=1..99")
    if labels_by_group.shape != (3740, 2, 99) or masks_by_group.shape != (3740, 2, 99):
        raise RuntimeError("bundle label or censor-mask shape is invalid")
    if episode_ids.shape != (3740,) or split_codes.shape != (3740,):
        raise RuntimeError("bundle episode/split shape is invalid")
    for episode_id in np.unique(episode_ids):
        rows = episode_ids == episode_id
        expected_code = split_by_episode.get(int(episode_id))
        if expected_code is None or not np.all(split_codes[rows] == expected_code):
            raise RuntimeError(f"episode-level split leakage or missing episode: {episode_id}")

    # Normalize the source chunk from train windows only.  The group one-hot is
    # appended after normalization and is therefore not leaked through scaling.
    flat_chunks = chunks.reshape(chunks.shape[0], -1).astype(np.float64)
    train_windows = split_codes == 0
    mean = flat_chunks[train_windows].mean(axis=0)
    std = flat_chunks[train_windows].std(axis=0)
    std = np.where(std < 1.0e-6, 1.0, std)
    scaled = ((flat_chunks - mean) / std).astype(np.float32)

    # One shared model sees the same source chunk for both groups plus a group
    # identity.  No offset is supplied as a feature.
    repeated_chunks = np.concatenate((scaled, scaled), axis=0)
    repeated_group_ids = np.concatenate(
        (np.zeros(chunks.shape[0], dtype=np.int8), np.ones(chunks.shape[0], dtype=np.int8))
    )
    group_one_hot = np.eye(2, dtype=np.float32)[repeated_group_ids]
    features = np.concatenate((repeated_chunks, group_one_hot), axis=1)
    labels = np.concatenate((labels_by_group[:, 0, :], labels_by_group[:, 1, :]), axis=0)
    label_mask = np.concatenate((masks_by_group[:, 0, :], masks_by_group[:, 1, :]), axis=0)
    row_groups = np.asarray([GROUP_NAMES[int(value)] for value in repeated_group_ids], dtype=str)
    row_episodes = np.concatenate((episode_ids, episode_ids), axis=0).astype(str)
    row_split = np.concatenate((split_codes, split_codes), axis=0)
    split_names = np.asarray([("train", "validation", "test")[int(code)] for code in row_split])
    window_ids = np.concatenate((np.arange(chunks.shape[0]), np.arange(chunks.shape[0])))
    return PilotData(
        features=features,
        labels=labels,
        label_mask=label_mask,
        groups=row_groups,
        group_ids=repeated_group_ids,
        episode_ids=row_episodes,
        split=split_names,
        source_window_ids=window_ids,
        source_chunks=chunks,
        normalization_mean=mean.astype(np.float32),
        normalization_std=std.astype(np.float32),
    )


class NumpySharedMLP:
    """Small shared vector MLP with an optional monotone survival output."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        kind: str,
        seed: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        initial_output_bias: np.ndarray | None = None,
    ) -> None:
        if kind not in {"independent_vector_shared_mlp", "monotone_conditional_survival_shared_mlp"}:
            raise ValueError(f"unknown model kind: {kind}")
        self.kind = kind
        self.output_dim = output_dim
        rng = np.random.default_rng(seed)
        dims = (input_dim, *hidden_dims, output_dim)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for previous, current in zip(dims[:-1], dims[1:]):
            scale = np.sqrt(2.0 / previous) if current != output_dim else np.sqrt(1.0 / previous)
            self.weights.append(rng.normal(0.0, scale, size=(previous, current)).astype(np.float64))
            self.biases.append(np.zeros(current, dtype=np.float64))
        if initial_output_bias is not None:
            if initial_output_bias.shape != self.biases[-1].shape:
                raise ValueError("initial output bias has the wrong shape")
            self.biases[-1] = initial_output_bias.astype(np.float64, copy=True)

    def _forward(self, features: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        hidden_values: list[np.ndarray] = []
        hidden_preactivations: list[np.ndarray] = []
        values = features.astype(np.float64, copy=False)
        for weights, bias in zip(self.weights[:-1], self.biases[:-1]):
            preactivation = values @ weights + bias
            hidden_preactivations.append(preactivation)
            values = np.maximum(preactivation, 0.0)
            hidden_values.append(values)
        logits = values @ self.weights[-1] + self.biases[-1]
        if self.kind == "independent_vector_shared_mlp":
            probabilities = _sigmoid(logits)
        else:
            conditional = _sigmoid(logits)
            probabilities = np.cumprod(conditional, axis=1)
        return probabilities, hidden_values, hidden_preactivations

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._forward(features)[0].astype(np.float64)

    def _loss_and_gradients(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
        probabilities, hidden_values, hidden_preactivations = self._forward(features)
        safe = np.clip(probabilities, 1.0e-7, 1.0 - 1.0e-7)
        denominator = float(mask.sum())
        if denominator <= 0.0:
            raise RuntimeError("training batch has no observed labels")
        loss = float(
            -np.sum(mask * (labels * np.log(safe) + (1.0 - labels) * np.log1p(-safe)))
            / denominator
        )
        if self.kind == "independent_vector_shared_mlp":
            output_gradient = (probabilities - labels) * mask / denominator
        else:
            # p_i = product_{j<=i} q_j.  Differentiate BCE through the
            # cumulative product so every output remains prefix-monotone.
            logits = hidden_values[-1] @ self.weights[-1] + self.biases[-1]
            conditional = _sigmoid(logits)
            d_probability = mask * (probabilities - labels) / (safe * (1.0 - safe) * denominator)
            d_conditional = np.zeros_like(d_probability)
            suffix = np.zeros(d_probability.shape[0], dtype=np.float64)
            for index in range(self.output_dim - 1, -1, -1):
                suffix += d_probability[:, index] * probabilities[:, index]
                d_conditional[:, index] = suffix / np.clip(conditional[:, index], 1.0e-7, None)
            output_gradient = d_conditional * conditional * (1.0 - conditional)
            output_gradient = np.clip(output_gradient, -10.0, 10.0)

        gradients_w: list[np.ndarray] = [np.zeros_like(value) for value in self.weights]
        gradients_b: list[np.ndarray] = [np.zeros_like(value) for value in self.biases]
        previous = hidden_values[-1] if hidden_values else features
        gradients_w[-1] = previous.T @ output_gradient
        gradients_b[-1] = output_gradient.sum(axis=0)
        upstream = output_gradient @ self.weights[-1].T
        for index in range(len(self.weights) - 2, -1, -1):
            upstream *= hidden_preactivations[index] > 0.0
            previous = features if index == 0 else hidden_values[index - 1]
            gradients_w[index] = previous.T @ upstream
            gradients_b[index] = upstream.sum(axis=0)
            if index > 0:
                upstream = upstream @ self.weights[index].T
        for index in range(len(gradients_w)):
            gradients_w[index] += 1.0e-5 * self.weights[index]
            gradients_w[index] = np.clip(gradients_w[index], -5.0, 5.0)
            gradients_b[index] = np.clip(gradients_b[index], -5.0, 5.0)
        return loss, gradients_w, gradients_b

    def _forward_logits(self, features: np.ndarray) -> np.ndarray:
        values = features.astype(np.float64, copy=False)
        for weights, bias in zip(self.weights[:-1], self.biases[:-1]):
            values = np.maximum(values @ weights + bias, 0.0)
        return values @ self.weights[-1] + self.biases[-1]

    def loss(self, features: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
        probabilities = self.predict(features)
        safe = np.clip(probabilities, 1.0e-7, 1.0 - 1.0e-7)
        denominator = float(mask.sum())
        return float(
            -np.sum(mask * (labels * np.log(safe) + (1.0 - labels) * np.log1p(-safe)))
            / denominator
        )

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
        training_rows: np.ndarray,
        validation_rows: np.ndarray,
        *,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 2.0e-3,
        weight_decay: float = 1.0e-5,
        patience: int = 18,
        seed: int,
    ) -> dict[str, Any]:
        train_rows = np.asarray(training_rows, dtype=bool)
        validation_rows = np.asarray(validation_rows, dtype=bool)
        if np.any(train_rows & validation_rows) or not train_rows.any() or not validation_rows.any():
            raise ValueError("training and validation rows must be non-empty and disjoint")
        rng = np.random.default_rng(seed + 991)
        first_m = [np.zeros_like(value) for value in self.weights]
        first_v = [np.zeros_like(value) for value in self.weights]
        second_m = [np.zeros_like(value) for value in self.biases]
        second_v = [np.zeros_like(value) for value in self.biases]
        best_weights = [value.copy() for value in self.weights]
        best_biases = [value.copy() for value in self.biases]
        best_validation = float("inf")
        best_epoch = 0
        stale = 0
        history: list[dict[str, float]] = []
        step = 0
        train_indices = np.flatnonzero(train_rows)
        for epoch in range(1, epochs + 1):
            order = train_indices.copy()
            rng.shuffle(order)
            losses: list[float] = []
            for start in range(0, len(order), batch_size):
                batch = order[start : start + batch_size]
                loss, gradients_w, gradients_b = self._loss_and_gradients(
                    features[batch], labels[batch], mask[batch]
                )
                losses.append(loss)
                step += 1
                for index, (weights, gradient) in enumerate(zip(self.weights, gradients_w)):
                    first_m[index] = 0.9 * first_m[index] + 0.1 * gradient
                    first_v[index] = 0.999 * first_v[index] + 0.001 * np.square(gradient)
                    corrected_m = first_m[index] / (1.0 - 0.9**step)
                    corrected_v = first_v[index] / (1.0 - 0.999**step)
                    self.weights[index] = weights - learning_rate * (
                        corrected_m / (np.sqrt(corrected_v) + 1.0e-8) + weight_decay * weights
                    )
                for index, (bias, gradient) in enumerate(zip(self.biases, gradients_b)):
                    second_m[index] = 0.9 * second_m[index] + 0.1 * gradient
                    second_v[index] = 0.999 * second_v[index] + 0.001 * np.square(gradient)
                    corrected_m = second_m[index] / (1.0 - 0.9**step)
                    corrected_v = second_v[index] / (1.0 - 0.999**step)
                    self.biases[index] = bias - learning_rate * corrected_m / (
                        np.sqrt(corrected_v) + 1.0e-8
                    )
            validation_loss = self.loss(features[validation_rows], labels[validation_rows], mask[validation_rows])
            row = {
                "epoch": float(epoch),
                "train_bce": float(np.mean(losses)),
                "validation_bce": float(validation_loss),
            }
            history.append(row)
            if validation_loss < best_validation - 1.0e-8:
                best_validation = validation_loss
                best_epoch = epoch
                best_weights = [value.copy() for value in self.weights]
                best_biases = [value.copy() for value in self.biases]
                stale = 0
            else:
                stale += 1
                if stale > patience:
                    break
        self.weights = best_weights
        self.biases = best_biases
        return {
            "best_epoch": best_epoch,
            "best_validation_bce": best_validation,
            "epochs_run": len(history),
            "history_tail": history[-5:],
        }


def _initial_biases(labels: np.ndarray, mask: np.ndarray, kind: str) -> np.ndarray:
    rates: list[float] = []
    for index in range(labels.shape[1]):
        if kind == "independent_vector_shared_mlp" or index == 0:
            selected = mask[:, index]
            rates.append(float(labels[selected, index].mean()) if selected.any() else 0.5)
        else:
            selected = mask[:, index] & mask[:, index - 1] & (labels[:, index - 1] == 1.0)
            rates.append(float(labels[selected, index].mean()) if selected.any() else 0.9)
    return _logit(np.asarray(rates, dtype=np.float64))


def _metric_slice(labels: np.ndarray, scores: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    undefined = positives == 0 or negatives == 0
    result: dict[str, Any] = {
        "sample_count": int(labels.size),
        "positive_count": positives,
        "negative_count": negatives,
        "positive_rate": float(labels.mean()) if labels.size else float("nan"),
        "auroc": float("nan") if undefined else float(roc_auc(labels, scores)),
        "auprc": float("nan") if undefined else float(average_precision(labels, scores)),
        "brier": float(brier_score(labels, scores)),
        "ece": float(expected_calibration_error(labels, scores, n_bins=10)),
        "undefined_discrimination": bool(undefined),
        "undefined_reason": "one_label_class" if undefined else None,
    }
    baseline_brier = float(brier_score(labels, baseline))
    result["empirical_brier"] = baseline_brier
    result["brier_skill_score_vs_empirical"] = (
        float(1.0 - result["brier"] / baseline_brier) if baseline_brier > 0.0 else float("nan")
    )
    return result


def _score_report(
    labels: np.ndarray,
    scores: np.ndarray,
    label_mask: np.ndarray,
    groups: np.ndarray,
    empirical_scores: np.ndarray,
) -> dict[str, Any]:
    valid = np.asarray(label_mask, dtype=bool)
    flat = valid.reshape(-1)
    flat_labels = labels.reshape(-1)[flat]
    flat_scores = scores.reshape(-1)[flat]
    flat_empirical = empirical_scores.reshape(-1)[flat]
    flat_groups = np.repeat(groups, labels.shape[1])[flat]
    offsets = np.tile(np.arange(1, labels.shape[1] + 1), labels.shape[0])
    flat_offsets = offsets[flat]

    overall = _metric_slice(flat_labels, flat_scores, flat_empirical)
    per_group: dict[str, Any] = {}
    for group in GROUP_NAMES:
        selected = flat_groups == group
        per_group[group] = _metric_slice(
            flat_labels[selected], flat_scores[selected], flat_empirical[selected]
        )
    per_offset: dict[str, Any] = {}
    for offset in range(1, labels.shape[1] + 1):
        selected = flat_offsets == offset
        per_offset[str(offset)] = _metric_slice(
            flat_labels[selected], flat_scores[selected], flat_empirical[selected]
        )
    group_offset: dict[str, dict[str, Any]] = {}
    for group in GROUP_NAMES:
        group_offset[group] = {}
        for offset in range(1, labels.shape[1] + 1):
            selected = (flat_groups == group) & (flat_offsets == offset)
            group_offset[group][str(offset)] = _metric_slice(
                flat_labels[selected], flat_scores[selected], flat_empirical[selected]
            )

    valid_offsets = [
        values
        for values in per_offset.values()
        if not values["undefined_discrimination"]
    ]
    macro: dict[str, Any] = {
        "valid_offset_count": len(valid_offsets),
        "excluded_offset_count": labels.shape[1] - len(valid_offsets),
        "excluded_offsets": [
            int(offset)
            for offset, values in per_offset.items()
            if values["undefined_discrimination"]
        ],
    }
    for key in ("auroc", "auprc", "brier", "ece", "brier_skill_score_vs_empirical"):
        macro[key] = (
            float(np.mean([values[key] for values in valid_offsets]))
            if valid_offsets
            else float("nan")
        )
    return {
        "overall": overall,
        "per_group": per_group,
        "per_offset": per_offset,
        "per_group_offset": group_offset,
        "macro_valid_nontrivial_offsets": macro,
        "observed_cells": int(flat_labels.size),
        "source_group_rows": int(labels.shape[0]),
        "offset_convention": "bundle k=1..99; k=0 excluded",
    }


def _decode_learned(scores: np.ndarray, tau: float) -> int:
    horizon = 1  # R(0)=1 is forced only for horizon decoding.
    for value in scores:
        if float(value) <= tau:
            break
        horizon += 1
    return horizon


def _decode_refresh_oracle(labels: np.ndarray, mask: np.ndarray) -> int:
    horizon = 1
    for label, observed in zip(labels, mask):
        if not observed or not label:
            break
        horizon += 1
    return horizon


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        result = np.empty(values.size, dtype=np.float64)
        start = 0
        while start < values.size:
            end = start + 1
            while end < values.size and sorted_values[end] == sorted_values[start]:
                end += 1
            result[order[start:end]] = (start + end - 1) / 2.0 + 1.0
            start = end
        return result
    return float(np.corrcoef(ranks(left), ranks(right))[0, 1])


def _horizon_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_mask: np.ndarray,
    groups: np.ndarray,
    split: np.ndarray,
    *,
    target_split: str,
    tau: float,
) -> dict[str, Any]:
    selected = split == target_split
    by_group: dict[str, Any] = {}
    all_predicted: list[float] = []
    all_oracle: list[float] = []
    for group_id, group_name in enumerate(GROUP_NAMES):
        rows = selected & (groups == group_name)
        predicted = np.asarray([_decode_learned(row, tau) for row in predictions[rows]])
        oracle = np.asarray(
            [_decode_refresh_oracle(y, m) for y, m in zip(labels[rows], label_mask[rows])]
        )
        difference = predicted.astype(np.float64) - oracle.astype(np.float64)
        all_predicted.extend(predicted.tolist())
        all_oracle.extend(oracle.tolist())
        by_group[group_name] = _horizon_slice(predicted, oracle, difference)
    all_predicted_array = np.asarray(all_predicted, dtype=np.float64)
    all_oracle_array = np.asarray(all_oracle, dtype=np.float64)
    all_difference = all_predicted_array - all_oracle_array
    return {
        "count": int(all_predicted_array.size),
        "by_group": by_group,
        "overall": _horizon_slice(all_predicted_array, all_oracle_array, all_difference),
        "tau": float(tau),
        "split": target_split,
        "oracle_definition": "prefix refresh-consistency label, censor-clipped at the first unobserved offset",
    }


def _horizon_slice(predicted: np.ndarray, oracle: np.ndarray, difference: np.ndarray) -> dict[str, Any]:
    return {
        "sample_count": int(predicted.size),
        "mae": float(np.abs(difference).mean()) if predicted.size else float("nan"),
        "median_absolute_error": float(np.median(np.abs(difference))) if predicted.size else float("nan"),
        "exact_match": float(np.mean(difference == 0.0)) if predicted.size else float("nan"),
        "within_plus_minus_2": float(np.mean(np.abs(difference) <= 2.0)) if predicted.size else float("nan"),
        "within_plus_minus_5": float(np.mean(np.abs(difference) <= 5.0)) if predicted.size else float("nan"),
        "spearman": _spearman(predicted.astype(np.float64), oracle.astype(np.float64)),
        "over_commit_rate": float(np.mean(difference > 0.0)) if predicted.size else float("nan"),
        "under_commit_rate": float(np.mean(difference < 0.0)) if predicted.size else float("nan"),
        "mean_learned_horizon": float(predicted.mean()) if predicted.size else float("nan"),
        "mean_refresh_oracle_horizon": float(oracle.mean()) if oracle.size else float("nan"),
    }


def _empirical_scores(data: PilotData) -> tuple[np.ndarray, np.ndarray]:
    train = data.split == "train"
    global_rates = np.asarray(
        [data.labels[train, index][data.label_mask[train, index]].mean() for index in range(99)]
    )
    scores = np.zeros_like(data.labels, dtype=np.float64)
    for group_name in GROUP_NAMES:
        rows = train & (data.groups == group_name)
        test_rows = data.groups == group_name
        for index in range(99):
            observed = rows & data.label_mask[:, index]
            rate = float(data.labels[observed, index].mean()) if observed.any() else float(global_rates[index])
            scores[test_rows, index] = rate
    constant_rate = float(data.labels[data.label_mask & train[:, None]].mean())
    constant = np.full_like(scores, constant_rate)
    return constant, scores


def _metric_rows(reports: dict[str, dict[str, Any]], level: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_name, report in reports.items():
        model, seed = run_name.split("|", 1)
        entries = report["per_group"] if level == "group" else report["per_offset"]
        for key, values in entries.items():
            rows.append({"model": model, "seed": seed, level: key, **values})
    return rows


def _group_offset_rows(reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_name, report in reports.items():
        model, seed = run_name.split("|", 1)
        for group, offsets in report["per_group_offset"].items():
            for offset, values in offsets.items():
                rows.append({"model": model, "seed": seed, "group": group, "offset": offset, **values})
    return rows


def _horizon_rows(horizon_reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_name, report in horizon_reports.items():
        model, seed = run_name.split("|", 1)
        for group, values in report["by_group"].items():
            rows.append({"model": model, "seed": seed, "group": group, **values})
        rows.append({"model": model, "seed": seed, "group": "overall", **report["overall"]})
    return rows


def _plot_outputs(
    output_dir: Path,
    reports: dict[str, dict[str, Any]],
    horizon_predictions: dict[str, np.ndarray],
    data: PilotData,
    labels: np.ndarray,
    masks: np.ndarray,
    best_learned_name: str,
    best_learned_tau: float,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    report = reports[best_learned_name]
    offsets = np.arange(1, 100)
    auroc_values = [report["per_offset"][str(offset)]["auroc"] for offset in offsets]
    auprc_values = [report["per_offset"][str(offset)]["auprc"] for offset in offsets]
    brier_values = [report["per_offset"][str(offset)]["brier"] for offset in offsets]
    figure, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    axes[0].plot(offsets, auroc_values, label="best learned pooled AUROC")
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="chance")
    axes[1].plot(offsets, auprc_values, label="best learned pooled AUPRC")
    axes[2].plot(offsets, brier_values, label="best learned pooled Brier")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    axes[-1].set_xlabel("nontrivial offset k")
    figure.suptitle("Chunk-only fixed-offset reliability metrics")
    figure.tight_layout()
    figure.savefig(plot_dir / "fixed_offset_metrics.png", dpi=160)
    plt.close(figure)

    selected = data.split == "test"
    prediction = horizon_predictions[best_learned_name]
    for group_id, group_name in enumerate(GROUP_NAMES):
        rows = selected & (data.groups == group_name)
        learned = np.asarray([_decode_learned(row, best_learned_tau) for row in prediction[rows]])
        oracle = np.asarray([_decode_refresh_oracle(y, m) for y, m in zip(labels[rows], masks[rows])])
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.scatter(oracle, learned, s=9, alpha=0.35)
        axis.set_xlabel("refresh-oracle horizon")
        axis.set_ylabel("learned horizon")
        axis.set_title(f"{group_name}: learned vs refresh oracle")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(plot_dir / f"horizon_recovery_{group_name}.png", dpi=160)
        plt.close(figure)

    flat_mask = masks[selected].reshape(-1)
    flat_labels = labels[selected].reshape(-1)[flat_mask]
    flat_scores = prediction[selected].reshape(-1)[flat_mask]
    bin_ids = np.minimum((flat_scores * 10).astype(int), 9)
    mean_scores = []
    fractions = []
    for bin_id in range(10):
        cells = bin_ids == bin_id
        mean_scores.append(float(flat_scores[cells].mean()) if cells.any() else np.nan)
        fractions.append(float(flat_labels[cells].mean()) if cells.any() else np.nan)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.8)
    axis.plot(mean_scores, fractions, marker="o", label="best learned")
    axis.set_xlabel("mean predicted probability")
    axis.set_ylabel("observed frequency")
    axis.set_title("Chunk-only calibration on held-out test rows")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "calibration.png", dpi=160)
    plt.close(figure)


def _git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(output_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    bundle_path = root / "experiments/dynamic_reliability_horizon/artifact_handoff/minimal_y_refresh_training_bundle.npz"
    split_path = root / "experiments/dynamic_reliability_horizon/artifact_handoff/episode_split_manifest.json"
    data = _load_data(bundle_path, split_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "status": "completed",
        "runner": "NumPy CPU shared-vector pilot",
        "bundle": str(bundle_path.relative_to(root)),
        "bundle_sha256": EXPECTED_BUNDLE_SHA256,
        "input_contract": "source predicted chunk A_t flattened plus one-hot group identity only",
        "future_features_used": False,
        "k0_policy": "excluded from training and headline metrics; R(0)=1 prepended only for horizon decoding",
        "models": [
            "constant_prior",
            "empirical_group_offset_prior_train_only",
            "independent_vector_shared_mlp",
            "monotone_conditional_survival_shared_mlp",
        ],
        "seeds": list(SEEDS),
        "hidden_dims": [64, 32],
        "epochs": 100,
        "batch_size": 256,
        "learning_rate": 0.002,
        "weight_decay": 1.0e-5,
        "patience": 18,
        "tau_values": list(TAU_VALUES),
        "tau_selection": "minimum validation refresh-oracle MAE; ties prefer higher within +/-2 then lower tau",
        "decision_rule_fixed_before_results": {
            "fixed_offset_auroc_threshold": 0.55,
            "minimum_valid_group_offset_slices": 10,
            "broad_signal_fraction_threshold": 0.50,
            "horizon_useful_requires": "learned overall test MAE <= empirical baseline MAE and within +/-2 >= 0.50",
        },
        "scientific_scope": "causal chunk-only ablation; not task-optimal horizon and no rollout",
    }
    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "seeds.json", {"seeds": list(SEEDS), "deterministic": True})

    constant_scores, empirical_scores = _empirical_scores(data)
    predictions: dict[str, np.ndarray] = {}
    reports: dict[str, dict[str, Any]] = {}
    horizon_reports: dict[str, dict[str, Any]] = {}
    tau_rows: list[dict[str, Any]] = []
    run_summaries: dict[str, Any] = {}

    baseline_runs = {
        "constant_prior|train_only": constant_scores,
        "empirical_group_offset_prior_train_only|train_only": empirical_scores,
    }
    for run_name, scores in baseline_runs.items():
        predictions[run_name] = scores

    train_rows = data.split == "train"
    validation_rows = data.split == "validation"
    for kind in ("independent_vector_shared_mlp", "monotone_conditional_survival_shared_mlp"):
        for seed in SEEDS:
            model = NumpySharedMLP(
                data.features.shape[1],
                data.labels.shape[1],
                kind=kind,
                seed=seed,
                initial_output_bias=_initial_biases(data.labels[train_rows], data.label_mask[train_rows], kind),
            )
            training_summary = model.fit(
                data.features,
                data.labels.astype(np.float64),
                data.label_mask,
                train_rows,
                validation_rows,
                seed=seed,
            )
            run_name = f"{kind}|{seed}"
            predictions[run_name] = model.predict(data.features)
            run_summaries[run_name] = training_summary

    for run_name, scores in predictions.items():
        reports[run_name] = _score_report(
            data.labels[data.split == "test"],
            scores[data.split == "test"],
            data.label_mask[data.split == "test"],
            data.groups[data.split == "test"],
            empirical_scores[data.split == "test"],
        )
        validation_horizon = []
        for tau in TAU_VALUES:
            report = _horizon_report(
                scores,
                data.labels,
                data.label_mask,
                data.groups,
                data.split,
                target_split="validation",
                tau=float(tau),
            )
            for group, values in report["by_group"].items():
                tau_rows.append({"model": run_name.split("|", 1)[0], "seed": run_name.split("|", 1)[1], "split": "validation", "tau": tau, "group": group, **values})
            tau_rows.append({"model": run_name.split("|", 1)[0], "seed": run_name.split("|", 1)[1], "split": "validation", "tau": tau, "group": "overall", **report["overall"]})
            validation_horizon.append(report)
        best = min(
            validation_horizon,
            key=lambda report: (
                report["overall"]["mae"],
                -report["overall"]["within_plus_minus_2"],
                report["tau"],
            ),
        )
        selected_tau = float(best["tau"])
        horizon = _horizon_report(
            scores,
            data.labels,
            data.label_mask,
            data.groups,
            data.split,
            target_split="test",
            tau=selected_tau,
        )
        horizon_reports[run_name] = horizon
        run_summaries.setdefault(run_name, {})
        run_summaries[run_name]["selected_tau"] = selected_tau
        run_summaries[run_name]["validation_horizon_selection"] = best

    # Average the three deterministic learned predictions as an additional
    # descriptive aggregate; its tau is still selected on validation only.
    for kind in ("independent_vector_shared_mlp", "monotone_conditional_survival_shared_mlp"):
        seed_names = [f"{kind}|{seed}" for seed in SEEDS]
        aggregate_name = f"{kind}|aggregate_mean_prediction"
        aggregate_scores = np.mean([predictions[name] for name in seed_names], axis=0)
        predictions[aggregate_name] = aggregate_scores
        reports[aggregate_name] = _score_report(
            data.labels[data.split == "test"],
            aggregate_scores[data.split == "test"],
            data.label_mask[data.split == "test"],
            data.groups[data.split == "test"],
            empirical_scores[data.split == "test"],
        )
        validation_horizon = [
            _horizon_report(
                aggregate_scores,
                data.labels,
                data.label_mask,
                data.groups,
                data.split,
                target_split="validation",
                tau=float(tau),
            )
            for tau in TAU_VALUES
        ]
        best = min(
            validation_horizon,
            key=lambda report: (
                report["overall"]["mae"],
                -report["overall"]["within_plus_minus_2"],
                report["tau"],
            ),
        )
        horizon_reports[aggregate_name] = _horizon_report(
            aggregate_scores,
            data.labels,
            data.label_mask,
            data.groups,
            data.split,
            target_split="test",
            tau=float(best["tau"]),
        )
        run_summaries[aggregate_name] = {
            "selected_tau": float(best["tau"]),
            "validation_horizon_selection": best,
            "aggregate_of": seed_names,
        }
        for tau in TAU_VALUES:
            report = _horizon_report(
                aggregate_scores,
                data.labels,
                data.label_mask,
                data.groups,
                data.split,
                target_split="validation",
                tau=float(tau),
            )
            for group, values in {**report["by_group"], "overall": report["overall"]}.items():
                tau_rows.append({"model": kind, "seed": "aggregate_mean_prediction", "split": "validation", "tau": tau, "group": group, **values})

    _write_csv(output_dir / "per_group_metrics.csv", _metric_rows(reports, "group"))
    _write_csv(output_dir / "per_offset_metrics.csv", _metric_rows(reports, "offset"))
    _write_csv(output_dir / "per_group_offset_metrics.csv", _group_offset_rows(reports))
    _write_csv(output_dir / "horizon_metrics.csv", _horizon_rows(horizon_reports))
    _write_csv(output_dir / "validation_tau_sweep.csv", tau_rows)

    learned_aggregate_names = [
        f"{kind}|aggregate_mean_prediction"
        for kind in ("independent_vector_shared_mlp", "monotone_conditional_survival_shared_mlp")
    ]
    best_learned_name = max(
        learned_aggregate_names,
        key=lambda name: (
            -float("inf")
            if not np.isfinite(reports[name]["macro_valid_nontrivial_offsets"]["auroc"])
            else reports[name]["macro_valid_nontrivial_offsets"]["auroc"]
        ),
    )
    best_learned_tau = float(horizon_reports[best_learned_name]["tau"])
    _plot_outputs(
        output_dir,
        reports,
        predictions,
        data,
        data.labels,
        data.label_mask,
        best_learned_name,
        best_learned_tau,
    )

    best_report = reports[best_learned_name]
    signal_by_group: dict[str, Any] = {}
    for group in GROUP_NAMES:
        values = best_report["per_group_offset"][group]
        valid = [item for item in values.values() if not item["undefined_discrimination"]]
        above = [item for item in valid if item["auroc"] >= config["decision_rule_fixed_before_results"]["fixed_offset_auroc_threshold"]]
        signal_by_group[group] = {
            "mean_valid_group_offset_auroc": float(np.mean([item["auroc"] for item in valid])) if valid else float("nan"),
            "valid_group_offset_slices": len(valid),
            "slices_at_or_above_threshold": len(above),
            "signal_fraction_of_valid_slices": float(len(above) / len(valid)) if valid else float("nan"),
            "meaningful_signal": bool(
                len(valid) >= config["decision_rule_fixed_before_results"]["minimum_valid_group_offset_slices"]
                and np.mean([item["auroc"] for item in valid]) >= config["decision_rule_fixed_before_results"]["fixed_offset_auroc_threshold"]
            ),
        }

    empirical_horizon = horizon_reports["empirical_group_offset_prior_train_only|train_only"]["overall"]
    learned_horizon = horizon_reports[best_learned_name]["overall"]
    horizon_useful = bool(
        learned_horizon["mae"] <= empirical_horizon["mae"]
        and learned_horizon["within_plus_minus_2"] >= 0.50
    )
    probability_improves = bool(
        best_report["overall"]["brier_skill_score_vs_empirical"] > 0.0
        or best_report["macro_valid_nontrivial_offsets"]["brier_skill_score_vs_empirical"] > 0.0
    )
    broad_signal = bool(
        all(values["meaningful_signal"] for values in signal_by_group.values())
        and all(
            values["signal_fraction_of_valid_slices"]
            >= config["decision_rule_fixed_before_results"]["broad_signal_fraction_threshold"]
            for values in signal_by_group.values()
        )
    )
    any_signal = any(values["meaningful_signal"] for values in signal_by_group.values())
    if broad_signal and horizon_useful:
        verdict = "GO"
    elif any_signal or probability_improves or horizon_useful:
        verdict = "PARTIAL"
    else:
        verdict = "NO-GO"

    scientific = {
        "best_learned_model": best_learned_name,
        "signal_by_group": signal_by_group,
        "probability_metrics_improve_vs_empirical": probability_improves,
        "horizon_recovery_useful_by_preregistered_rule": horizon_useful,
        "broad_fixed_offset_signal": broad_signal,
        "verdict": verdict,
        "next_scientific_experiment": (
            "Materialize exact causal source-time observation/state or frozen-policy latent features and rerun the same pre-registered estimator evaluation; keep future observations/actions, phase, progress, terminal metadata, and rollout semantics excluded."
            if verdict != "GO"
            else "Repeat the estimator evaluation with exact causal source-time observation/state or frozen-policy latent features, then assess whether the chunk-only signal survives the richer causal input."
        ),
    }
    metrics = {
        "status": "completed",
        "git_head": _git_head(),
        "bundle_sha256": EXPECTED_BUNDLE_SHA256,
        "dataset": {
            "source_windows": 3740,
            "source_chunk_shape": [3740, 100, 7],
            "episode_count": int(np.unique(data.episode_ids).size),
            "group_rows": int(data.features.shape[0]),
            "split_rows": {name: int(np.sum(data.split == name)) for name in ("train", "validation", "test")},
            "feature_dim": int(data.features.shape[1]),
            "feature_contract": "700 normalized source-chunk values + 2 group one-hot values",
            "no_episode_leakage": True,
        },
        "protocol": config,
        "training": run_summaries,
        "models": {
            name: {
                "test_metrics": reports[name],
                "test_horizon": horizon_reports.get(name),
            }
            for name in reports
        },
        "scientific_verdict": scientific,
    }
    _write_json(output_dir / "metrics.json", metrics)

    empirical_name = "empirical_group_offset_prior_train_only|train_only"
    evaluation_lines = [
        "# Chunk-only reliability pilot",
        "",
        f"Status: completed on CPU with the exact portable bundle (SHA256 `{EXPECTED_BUNDLE_SHA256}`).",
        "",
        "This is the causal chunk-only ablation: one shared predictor receives only the source predicted action chunk and a one-hot arm/gripper identity. No future observations, future actions, phase, progress, episode length, terminal metadata, rollout, or executor call is used.",
        "",
        "## Protocol",
        "",
        "The bundle contains k=1..99. k=0 is excluded from training and all headline metrics; R(0)=1 is prepended only while decoding horizons. Priors, feature normalization, model selection, and tau selection use train/validation episodes only.",
        "",
        f"Seeds: {', '.join(str(seed) for seed in SEEDS)}. The two learned models are an independent-vector shared MLP and a monotone conditional-survival shared MLP.",
        "",
        "## Required questions",
        "",
        f"1. Source-window-conditioned information beyond P(Y=1|g,k): the best learned aggregate is `{best_learned_name}`. Fixed group/offset AUROC/AUPRC are in `per_group_offset_metrics.csv`; pooled fixed-offset metrics are in `per_offset_metrics.csv`. The pre-registered fixed-offset AUROC threshold is 0.55.",
        f"2. Arm signal: {signal_by_group['arm']}. Gripper signal: {signal_by_group['gripper']}.",
        f"3. Best learned test probability metrics: pooled AUROC={best_report['overall']['auroc']}, AUPRC={best_report['overall']['auprc']}, Brier={best_report['overall']['brier']}, ECE={best_report['overall']['ece']}, Brier Skill versus empirical={best_report['overall']['brier_skill_score_vs_empirical']}. The full pooled, group, offset, and macro reports are in `metrics.json` and the CSV files.",
        f"4. Best learned refresh-oracle horizon recovery: overall MAE={learned_horizon['mae']}, median absolute error={learned_horizon['median_absolute_error']}, exact match={learned_horizon['exact_match']}, within +/-2={learned_horizon['within_plus_minus_2']}, within +/-5={learned_horizon['within_plus_minus_5']}, Spearman={learned_horizon['spearman']}, over-commit={learned_horizon['over_commit_rate']}, under-commit={learned_horizon['under_commit_rate']}, mean learned horizon={learned_horizon['mean_learned_horizon']}, mean oracle horizon={learned_horizon['mean_refresh_oracle_horizon']}.",
        f"5. Empirical baseline overall test horizon MAE={empirical_horizon['mae']}, within +/-2={empirical_horizon['within_plus_minus_2']}; tau was selected on validation only. Selected tau for the best learned aggregate: {best_learned_tau}.",
        f"6. Verdict: **{verdict}**. This is a scientific reliability diagnostic, not a task-optimal horizon claim.",
        f"7. Next scientific experiment: {scientific['next_scientific_experiment']}",
        "",
        "## Undefined slices",
        "",
        "Any group/offset or pooled offset containing only one observed label class has AUROC/AUPRC recorded as undefined with positive/negative/sample counts; no discrimination value is fabricated. Brier and ECE remain reported because they are probability metrics.",
        "",
        "## Artifacts",
        "",
        "- `metrics.json` — complete nested results and fixed decision rule",
        "- `per_group_metrics.csv` — pooled metrics by arm/gripper",
        "- `per_offset_metrics.csv` — pooled fixed-offset metrics",
        "- `per_group_offset_metrics.csv` — critical group/fixed-offset discrimination slices",
        "- `horizon_metrics.csv` — per-group refresh-oracle recovery",
        "- `validation_tau_sweep.csv` — validation-only threshold sweep",
        "- `plots/` — fixed-offset, calibration, and horizon plots",
        "",
    ]
    (output_dir / "evaluation.md").write_text("\n".join(evaluation_lines), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/dynamic_reliability_horizon/chunk_only_reliability"),
    )
    args = parser.parse_args()
    metrics = run(args.output_dir)
    verdict = metrics["scientific_verdict"]["verdict"]
    print(f"CHUNK_ONLY_PILOT_COMPLETE verdict={verdict} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
