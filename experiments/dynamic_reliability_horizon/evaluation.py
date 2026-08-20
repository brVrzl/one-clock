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
from .horizon_analysis import compare_horizon_sources, rows_to_curves
from .training import load_reliability_checkpoint, predict_scores


def _result_dict(result: object) -> dict[str, object]:
    return result.as_dict()  # type: ignore[no-any-return]


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
    }


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
