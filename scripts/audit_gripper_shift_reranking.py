#!/usr/bin/env python3
"""Measure the oracle and learned value of discrete gripper-timing candidates."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shifts", default="-8,-4,0,4,8")
    parser.add_argument("--ridge", type=float, default=10.0)
    return parser.parse_args()


def load_audit_module():
    path = ROOT / "scripts/audit_chunk_repair.py"
    spec = importlib.util.spec_from_file_location("chunk_repair_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def shift_sequence(values: np.ndarray, shift: int) -> np.ndarray:
    indices = np.arange(len(values)) - shift
    return values[np.clip(indices, 0, len(values) - 1)]


def candidate_chunks(split, shifts: tuple[int, ...]) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    for start, stop in split.boundaries:
        base = split.prediction[start:stop]
        bank = np.repeat(base[None], len(shifts), axis=0)
        for index, shift in enumerate(shifts):
            bank[index, :, 6] = shift_sequence(base[:, 6], shift)
        candidates.append(bank)
    return candidates


def chunk_features(split) -> np.ndarray:
    rows: list[np.ndarray] = []
    task_count = int(split.task.max()) + 1
    for start, stop in split.boundaries:
        chunk = split.prediction[start:stop]
        gripper = chunk[:, 6]
        sample_indices = np.linspace(0, len(gripper) - 1, 20).round().astype(int)
        task = np.zeros(task_count, dtype=np.float64)
        task[int(split.task[start])] = 1.0
        transitions = np.flatnonzero(np.sign(gripper[1:]) != np.sign(gripper[:-1]))
        transition_summary = np.asarray(
            [len(transitions), transitions[0] if len(transitions) else len(gripper)],
            dtype=np.float64,
        )
        rows.append(
            np.concatenate(
                (
                    split.source_state[start],
                    chunk.mean(axis=0),
                    chunk.std(axis=0),
                    gripper[sample_indices],
                    transition_summary,
                    task,
                )
            )
        )
    return np.stack(rows)


def oracle_labels(split, banks: list[np.ndarray]) -> np.ndarray:
    labels = []
    for (start, stop), bank in zip(split.boundaries, banks, strict=True):
        target = split.target[start:stop]
        losses = np.mean(np.square(bank - target[None]), axis=(1, 2))
        labels.append(int(np.argmin(losses)))
    return np.asarray(labels, dtype=np.int64)


def flatten_selection(split, banks: list[np.ndarray], labels: np.ndarray) -> np.ndarray:
    output = split.prediction.copy()
    for (start, stop), bank, label in zip(split.boundaries, banks, labels, strict=True):
        output[start:stop] = bank[int(label)]
    return output


def fit_selector(features: np.ndarray, labels: np.ndarray, classes: int, ridge: float):
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (features - mean) / scale
    design = np.concatenate((standardized, np.ones((len(features), 1))), axis=1)
    targets = np.eye(classes, dtype=np.float64)[labels]
    regularizer = ridge * np.eye(design.shape[1])
    weights = np.linalg.solve(design.T @ design + regularizer, design.T @ targets)
    return mean, scale, weights


def predict_selector(features: np.ndarray, model) -> np.ndarray:
    mean, scale, weights = model
    standardized = (features - mean) / scale
    design = np.concatenate((standardized, np.ones((len(features), 1))), axis=1)
    return np.argmax(design @ weights, axis=1)


def summarize(split, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - split.target
    return {
        "frame_weighted_mse": float(np.mean(np.square(error))),
        "frame_weighted_gripper_mse": float(np.mean(np.square(error[:, 6]))),
        "gripper_sign_accuracy": float(
            np.mean(np.sign(prediction[:, 6]) == np.sign(split.target[:, 6]))
        ),
    }


def distribution(labels: np.ndarray, shifts: tuple[int, ...]) -> dict[str, float]:
    return {
        str(shift): float(np.mean(labels == index))
        for index, shift in enumerate(shifts)
    }


def main() -> None:
    args = parse_args()
    shifts = tuple(int(value) for value in args.shifts.split(","))
    if 0 not in shifts:
        raise ValueError("candidate shifts must include zero")
    audit = load_audit_module()
    training = audit.load_split(args.cache, args.dataset, "validation")
    evaluation = audit.load_split(args.cache, args.dataset, "test")
    training_banks = candidate_chunks(training, shifts)
    evaluation_banks = candidate_chunks(evaluation, shifts)
    training_oracle = oracle_labels(training, training_banks)
    evaluation_oracle = oracle_labels(evaluation, evaluation_banks)
    model = fit_selector(
        chunk_features(training), training_oracle, len(shifts), float(args.ridge)
    )
    learned = predict_selector(chunk_features(evaluation), model)
    raw_label = np.full(len(evaluation.boundaries), shifts.index(0), dtype=np.int64)
    result = {
        "status": "exploratory_test_inspected_during_method_selection",
        "shifts_steps": list(shifts),
        "training_chunks": len(training.boundaries),
        "evaluation_chunks": len(evaluation.boundaries),
        "methods": {
            "frozen_act": summarize(
                evaluation, flatten_selection(evaluation, evaluation_banks, raw_label)
            ),
            "linear_selector": summarize(
                evaluation, flatten_selection(evaluation, evaluation_banks, learned)
            ),
            "oracle_selector": summarize(
                evaluation, flatten_selection(evaluation, evaluation_banks, evaluation_oracle)
            ),
        },
        "training_oracle_shift_distribution": distribution(training_oracle, shifts),
        "evaluation_oracle_shift_distribution": distribution(evaluation_oracle, shifts),
        "learned_shift_distribution": distribution(learned, shifts),
        "selector_oracle_label_accuracy": float(np.mean(learned == evaluation_oracle)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
