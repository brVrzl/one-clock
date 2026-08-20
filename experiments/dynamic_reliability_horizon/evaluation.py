"""Estimator and offline horizon evaluation against prior baselines."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
import json

import numpy as np

from experiments.temporal_reliability_training.evaluation import evaluate_reliability

from .artifacts import PreparedReliabilityDataset
from .baselines import EmpiricalReliabilityPredictor, constant_prior_scores
from .decoder import GroupHorizonDecoder
from .horizon_analysis import (
    compare_horizon_sources,
    horizon_regret,
    rows_to_curves,
    vector_rows_to_curves,
)
from .training import load_reliability_checkpoint, predict_scores


def _result_dict(result: object) -> dict[str, object]:
    values = result.as_dict()  # type: ignore[no-any-return]
    values["ece"] = values["calibration_error"]
    return values


def evaluate_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    groups: np.ndarray,
    offsets: np.ndarray,
    n_bins: int = 10,
) -> dict[str, object]:
    result = evaluate_reliability(labels, scores, n_bins=n_bins)
    per_group: dict[str, object] = {}
    for group in sorted(set(groups.astype(str))):
        selected = groups == group
        per_group[group] = _result_dict(
            evaluate_reliability(labels[selected], scores[selected], n_bins=n_bins)
        )
    per_offset: dict[str, object] = {}
    for offset in sorted(set(offsets.astype(int))):
        selected = offsets == offset
        per_offset[str(offset)] = _result_dict(
            evaluate_reliability(labels[selected], scores[selected], n_bins=n_bins)
        )
    return {
        "overall": _result_dict(result),
        "group": per_group,
        "offset": per_offset,
        "count": int(labels.size),
        "ece": float(result.calibration_error),
    }


def evaluate_vector_predictions(
    labels: np.ndarray,
    scores: np.ndarray,
    label_mask: np.ndarray,
    *,
    groups: np.ndarray,
    task_ids: np.ndarray | None = None,
    n_bins: int = 10,
) -> dict[str, object]:
    """Evaluate a shared ``[source, offset]`` prediction matrix.

    The mask is applied before flattening, so padded or unavailable future
    offsets never enter AUROC, Brier, ECE, or reliability curves.  Group and
    offset reports retain the same schema as :func:`evaluate_scores`.
    Optional task slices are included for offline task-wise diagnostics.
    """

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    label_mask = np.asarray(label_mask, dtype=bool)
    groups = np.asarray(groups).astype(str)
    if labels.ndim != 2 or scores.shape != labels.shape or label_mask.shape != labels.shape:
        raise ValueError("labels, scores, and label_mask must be matching two-dimensional arrays")
    if groups.shape != (labels.shape[0],):
        raise ValueError("groups must match the source-row dimension")
    if task_ids is not None:
        task_ids = np.asarray(task_ids).astype(str)
        if task_ids.shape != groups.shape:
            raise ValueError("task_ids must match the source-row dimension")
    flat_mask = label_mask.reshape(-1)
    flat_labels = labels.reshape(-1)[flat_mask]
    flat_scores = scores.reshape(-1)[flat_mask]
    repeated_groups = np.repeat(groups, labels.shape[1])[flat_mask]
    tiled_offsets = np.tile(np.arange(labels.shape[1], dtype=np.int64), labels.shape[0])[flat_mask]
    report = evaluate_scores(
        flat_labels,
        flat_scores,
        groups=repeated_groups,
        offsets=tiled_offsets,
        n_bins=n_bins,
    )
    if task_ids is not None:
        repeated_tasks = np.repeat(task_ids, labels.shape[1])[flat_mask]
        by_task: dict[str, object] = {}
        for task in sorted(set(repeated_tasks.astype(str))):
            selected = repeated_tasks == task
            by_task[task] = evaluate_scores(
                flat_labels[selected],
                flat_scores[selected],
                groups=repeated_groups[selected],
                offsets=tiled_offsets[selected],
                n_bins=n_bins,
            )
        report["task"] = by_task
    report["observed_cells"] = int(flat_labels.size)
    report["source_rows"] = int(labels.shape[0])
    report["horizon_dim"] = int(labels.shape[1])
    return report


def evaluate_vector_horizon_regret(
    dataset: "VectorReliabilityDataset",
    scores: np.ndarray,
    *,
    decoder: GroupHorizonDecoder,
    split: str = "test",
    mode: str = "combined",
) -> dict[str, object]:
    """Decode shared-head and label-oracle curves, then report horizon regret."""

    if dataset.split is None:
        raise ValueError("horizon analysis requires an episode-level split")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if mode == "combined":
        mode_rows = np.ones(dataset.features.shape[0], dtype=bool)
    elif mode in {"arm", "gripper"}:
        mode_rows = dataset.groups == mode
    else:
        raise ValueError("mode must be combined, arm, or gripper")
    selected = mode_rows & (dataset.split == split)
    if not selected.any():
        raise ValueError(f"mode {mode!r} has no rows in {split!r}")
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != dataset.labels.shape:
        raise ValueError("scores must match the full vector dataset label shape")
    predicted_curves = vector_rows_to_curves(
        episode_ids=dataset.episode_ids[selected],
        source_steps=dataset.source_steps[selected],
        groups=dataset.groups[selected],
        scores=scores[selected],
    )
    oracle_curves = vector_rows_to_curves(
        episode_ids=dataset.episode_ids[selected],
        source_steps=dataset.source_steps[selected],
        groups=dataset.groups[selected],
        scores=dataset.labels[selected],
    )
    predicted = [decoder.decode_curves(curves) for curves in predicted_curves]
    oracle = [decoder.decode_curves(curves) for curves in oracle_curves]
    return horizon_regret(predicted, oracle).as_dict()


def evaluate_shared_checkpoint(
    dataset: "VectorReliabilityDataset",
    checkpoint_path: str | Path,
    *,
    mode: str,
    decoder: GroupHorizonDecoder,
    n_bins: int = 10,
    device: str = "cpu",
) -> dict[str, object]:
    """Evaluate a shared checkpoint on held-out vector rows only."""

    if dataset.split is None:
        raise ValueError("evaluation requires an episode-level test split")
    mode_rows = np.ones(dataset.features.shape[0], dtype=bool) if mode == "combined" else dataset.groups == mode
    if mode not in {"combined", "arm", "gripper"}:
        raise ValueError("mode must be combined, arm, or gripper")
    test_rows = mode_rows & (dataset.split == "test")
    if not test_rows.any():
        raise ValueError(f"mode {mode!r} has no test rows")
    from .vector_training import load_shared_checkpoint, predict_reliability_curves

    model = load_shared_checkpoint(checkpoint_path, device=device)
    scores = predict_reliability_curves(model, dataset.features, device=device)
    test = dataset.select(test_rows)
    test_scores = scores[test_rows]
    report = evaluate_vector_predictions(
        test.labels,
        test_scores,
        test.label_mask,
        groups=test.groups,
        task_ids=test.task_ids,
        n_bins=n_bins,
    )
    report["offline_horizon_regret"] = evaluate_vector_horizon_regret(
        dataset,
        scores,
        decoder=decoder,
        split="test",
        mode=mode,
    )
    return report


def evaluate_checkpoint(
    dataset: PreparedReliabilityDataset,
    checkpoint_path: str | Path,
    *,
    mode: str,
    n_bins: int = 10,
    device: str = "cpu",
) -> dict[str, object]:
    if dataset.split is None:
        raise ValueError("evaluation requires an episode-level test split")
    mode_rows = np.ones(dataset.labels.shape, dtype=bool) if mode == "combined" else dataset.groups == mode
    train_rows = mode_rows & (dataset.split == "train")
    test_rows = mode_rows & (dataset.split == "test")
    if not train_rows.any() or not test_rows.any():
        raise ValueError(f"mode {mode!r} needs non-empty train and test rows")
    test = dataset.select(test_rows)
    train = dataset.select(train_rows)
    model = load_reliability_checkpoint(checkpoint_path, device=device)
    learned_scores = predict_scores(model, test.features, device=device)
    prior_scores = constant_prior_scores(train.labels, len(test.labels))
    empirical = EmpiricalReliabilityPredictor().fit(train.groups, train.offsets, train.labels)
    empirical_scores = empirical.predict(test.groups, test.offsets)
    return {
        "mode": mode,
        "learned_reliability": evaluate_scores(
            test.labels, learned_scores, groups=test.groups, offsets=test.offsets, n_bins=n_bins
        ),
        "constant_prior": evaluate_scores(
            test.labels, prior_scores, groups=test.groups, offsets=test.offsets, n_bins=n_bins
        ),
        "empirical_reliability_curve": evaluate_scores(
            test.labels, empirical_scores, groups=test.groups, offsets=test.offsets, n_bins=n_bins
        ),
    }


def save_evaluation_report(report: Mapping[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=True, sort_keys=True), encoding="utf-8")


def evaluate_horizon_sources(
    dataset: PreparedReliabilityDataset,
    checkpoint_path: str | Path,
    *,
    mode: str,
    decoder: GroupHorizonDecoder,
    static_horizons: Mapping[str, int],
    global_horizon: int | None = None,
    device: str = "cpu",
) -> dict[str, object]:
    """Compare fixed, learned, and label-oracle decoded schedules offline."""

    if dataset.split is None:
        raise ValueError("horizon analysis requires a stored episode split")
    mode_rows = np.ones(dataset.labels.shape, dtype=bool) if mode == "combined" else dataset.groups == mode
    test_rows = mode_rows & (dataset.split == "test")
    if not test_rows.any():
        raise ValueError(f"mode {mode!r} has no test rows")
    test = dataset.select(test_rows)
    model = load_reliability_checkpoint(checkpoint_path, device=device)
    learned_scores = predict_scores(model, test.features, device=device)
    learned_curves = rows_to_curves(
        episode_ids=test.episode_ids,
        source_steps=test.source_steps,
        groups=test.groups,
        offsets=test.offsets,
        scores=learned_scores,
    )
    oracle_curves = rows_to_curves(
        episode_ids=test.episode_ids,
        source_steps=test.source_steps,
        groups=test.groups,
        offsets=test.offsets,
        scores=test.labels.astype(np.float64),
    )
    available_groups = sorted(set(test.groups.astype(str)))
    static = {group: static_horizons[group] for group in available_groups if group in static_horizons}
    summaries = compare_horizon_sources(
        learned_curves,
        oracle_curves,
        decoder=decoder,
        static_horizons=static,
        global_horizon=global_horizon,
    )
    return {name: summary.as_dict() for name, summary in summaries.items()}


def plot_reliability_diagrams(
    report: Mapping[str, object],
    path: str | Path,
) -> None:
    """Plot overall learned/prior/empirical reliability curves if matplotlib exists."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - host-dependent
        raise ImportError("plotting requires matplotlib") from error
    figure, axis = plt.subplots(figsize=(5.5, 5.0))
    for name, color in (("learned_reliability", "tab:blue"), ("constant_prior", "tab:gray"), ("empirical_reliability_curve", "tab:orange")):
        curve = report[name]["overall"]["reliability_curve"]  # type: ignore[index]
        axis.plot(curve["mean_score"], curve["fraction_valid"], "o-", label=name, color=color)
    axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    axis.set(xlabel="Predicted reliability", ylabel="Observed validity", xlim=(0, 1), ylim=(0, 1))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_vector_reliability_diagram(
    report: Mapping[str, object],
    path: str | Path,
) -> None:
    """Plot the shared-head overall reliability diagram."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - host-dependent
        raise ImportError("plotting requires matplotlib") from error
    curve = report["overall"]["reliability_curve"]  # type: ignore[index]
    figure, axis = plt.subplots(figsize=(5.5, 5.0))
    axis.plot(curve["mean_score"], curve["fraction_valid"], "o-", label="shared reliability")
    axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    axis.set(xlabel="Predicted reliability", ylabel="Observed validity", xlim=(0, 1), ylim=(0, 1))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_calibration_curves(
    report: Mapping[str, object],
    path: str | Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - host-dependent
        raise ImportError("plotting requires matplotlib") from error
    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    learned = report["learned_reliability"]["offset"]  # type: ignore[index]
    x = sorted((int(offset) for offset in learned))
    y = [learned[str(offset)]["brier_score"] for offset in x]
    axis.plot(x, y, "o-", label="learned reliability")
    axis.set(xlabel="Future offset k", ylabel="Brier score")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_vector_calibration_curve(
    report: Mapping[str, object],
    path: str | Path,
) -> None:
    """Plot Brier score by future offset for a shared-head report."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - host-dependent
        raise ImportError("plotting requires matplotlib") from error
    offsets = report["offset"]  # type: ignore[index]
    x = sorted(int(offset) for offset in offsets)
    y = [offsets[str(offset)]["brier_score"] for offset in x]
    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.plot(x, y, "o-", label="shared reliability")
    axis.set(xlabel="Future offset k", ylabel="Brier score")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
