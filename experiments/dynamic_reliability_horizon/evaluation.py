"""Estimator and offline horizon evaluation against prior baselines."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
import csv
from dataclasses import replace
import json

import numpy as np

from experiments.temporal_reliability_training.evaluation import (
    average_precision,
    evaluate_reliability,
)

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


def _result_dict(
    result: object,
    *,
    labels: np.ndarray | None = None,
    scores: np.ndarray | None = None,
) -> dict[str, object]:
    values = result.as_dict()  # type: ignore[no-any-return]
    values["ece"] = values["calibration_error"]
    if labels is not None and scores is not None:
        values["auprc"] = float(average_precision(labels, scores))
    return values


def evaluate_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    groups: np.ndarray,
    offsets: np.ndarray,
    n_bins: int = 10,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    offsets = np.asarray(offsets, dtype=np.int64)
    result = evaluate_reliability(labels, scores, n_bins=n_bins)
    per_group: dict[str, object] = {}
    for group in sorted(set(groups.astype(str))):
        selected = groups == group
        per_group[group] = _result_dict(
            evaluate_reliability(labels[selected], scores[selected], n_bins=n_bins),
            labels=labels[selected],
            scores=scores[selected],
        )
    per_offset: dict[str, object] = {}
    for offset in sorted(set(offsets.astype(int))):
        selected = offsets == offset
        per_offset[str(offset)] = _result_dict(
            evaluate_reliability(labels[selected], scores[selected], n_bins=n_bins),
            labels=labels[selected],
            scores=scores[selected],
        )
    valid_offsets = [
        values
        for values in per_offset.values()
        if np.isfinite(float(values["auroc"]))  # type: ignore[index]
    ]
    macro = {
        metric: float(np.mean([float(values[metric]) for values in valid_offsets]))
        if valid_offsets
        else float("nan")
        for metric in ("auroc", "auprc", "brier_score", "ece")
    }
    macro["valid_offset_count"] = float(len(valid_offsets))
    return {
        "overall": _result_dict(result, labels=labels, scores=scores),
        "group": per_group,
        "offset": per_offset,
        "count": int(labels.size),
        "ece": float(result.calibration_error),
        "auprc": float(average_precision(labels, scores)),
        "macro_average_valid_nontrivial_offsets": macro,
    }


def evaluate_vector_predictions(
    labels: np.ndarray,
    scores: np.ndarray,
    label_mask: np.ndarray,
    *,
    groups: np.ndarray,
    task_ids: np.ndarray | None = None,
    n_bins: int = 10,
    include_identity_offset: bool = False,
) -> dict[str, object]:
    """Evaluate a shared ``[source, offset]`` prediction matrix.

    The mask is applied before flattening, so padded or unavailable future
    offsets never enter AUROC, Brier, ECE, or reliability curves. By default,
    offset zero is additionally masked because ``Y_refresh(0)`` is a trivial
    identity event. Group and offset reports retain the same schema as
    :func:`evaluate_scores`.
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
    if not include_identity_offset and labels.shape[1] > 0:
        label_mask = label_mask.copy()
        label_mask[:, 0] = False
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
    report["identity_offset_excluded"] = not include_identity_offset
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

    predicted, oracle = vector_horizon_pairs(
        dataset,
        scores,
        decoder=decoder,
        split=split,
        mode=mode,
    )
    return horizon_regret(predicted, oracle).as_dict()


def vector_horizon_pairs(
    dataset: "VectorReliabilityDataset",
    scores: np.ndarray,
    *,
    decoder: GroupHorizonDecoder,
    split: str,
    mode: str,
) -> tuple[tuple[dict[str, int], ...], tuple[dict[str, int], ...]]:
    """Decode learned and oracle schedules for one episode split."""

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
    if dataset.horizon_dim > 0:
        observed_identity = dataset.label_mask[:, 0]
        if np.any(dataset.labels[observed_identity, 0] != 1.0):
            raise ValueError("Y_refresh(0) must be the identity label 1")
        scores = scores.copy()
        scores[:, 0] = 1.0
        oracle_labels = dataset.labels.copy()
        oracle_labels[:, 0] = 1.0
    else:
        oracle_labels = dataset.labels
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
        scores=oracle_labels[selected],
    )
    return (
        tuple(decoder.decode_curves(curves) for curves in predicted_curves),
        tuple(decoder.decode_curves(curves) for curves in oracle_curves),
    )


def evaluate_tau_sweep(
    dataset: "VectorReliabilityDataset",
    scores: np.ndarray,
    *,
    decoder: GroupHorizonDecoder,
    tau_values: np.ndarray,
    mode: str = "combined",
) -> list[dict[str, object]]:
    """Evaluate threshold sensitivity on validation episodes only."""

    rows: list[dict[str, object]] = []
    for tau in np.asarray(tau_values, dtype=np.float64):
        if not 0.0 <= float(tau) <= 1.0:
            raise ValueError("tau sweep values must lie in [0, 1]")
        tau_decoder = GroupHorizonDecoder(
            replace(decoder.config, threshold_tau=float(tau))
        )
        predicted, oracle = vector_horizon_pairs(
            dataset,
            scores,
            decoder=tau_decoder,
            split="validation",
            mode=mode,
        )
        regret = horizon_regret(predicted, oracle)
        for group, metrics in regret.by_group.items():
            rows.append({"tau": float(tau), "group": group, **metrics})
    return rows


def _constant_vector_scores(
    train: "VectorReliabilityDataset",
    test: "VectorReliabilityDataset",
) -> np.ndarray:
    mask = train.label_mask.copy()
    if mask.shape[1] > 0:
        mask[:, 0] = False
    prior = float(train.labels[mask].mean()) if mask.any() else float("nan")
    scores = np.full(test.labels.shape, prior, dtype=np.float64)
    if scores.shape[1] > 0:
        scores[:, 0] = 1.0
    return scores


def _empirical_vector_scores(
    train: "VectorReliabilityDataset",
    test: "VectorReliabilityDataset",
) -> np.ndarray:
    """Fit P(Y=1 | group, k) on train rows with transparent fallbacks."""

    horizon_dim = train.horizon_dim
    observed = train.label_mask.copy()
    if horizon_dim > 0:
        observed[:, 0] = False
    global_by_offset: dict[int, float] = {}
    group_by_offset: dict[tuple[str, int], float] = {}
    overall = float(train.labels[observed].mean()) if observed.any() else float("nan")
    for offset in range(horizon_dim):
        selected = observed[:, offset]
        if selected.any():
            global_by_offset[offset] = float(train.labels[selected, offset].mean())
            for group in sorted(set(train.groups)):
                group_selected = selected & (train.groups == group)
                if group_selected.any():
                    group_by_offset[(group, offset)] = float(
                        train.labels[group_selected, offset].mean()
                    )
    scores = np.full(test.labels.shape, overall, dtype=np.float64)
    for row, group in enumerate(test.groups):
        for offset in range(horizon_dim):
            if offset == 0:
                scores[row, offset] = 1.0
            else:
                scores[row, offset] = group_by_offset.get(
                    (str(group), offset), global_by_offset.get(offset, overall)
                )
    return scores


def evaluate_shared_checkpoint(
    dataset: "VectorReliabilityDataset",
    checkpoint_path: str | Path,
    *,
    mode: str,
    decoder: GroupHorizonDecoder,
    n_bins: int = 10,
    device: str = "cpu",
    tau_values: np.ndarray | None = None,
) -> dict[str, object]:
    """Evaluate a shared checkpoint and train-fitted reliability baselines."""

    if dataset.split is None:
        raise ValueError("evaluation requires an episode-level test split")
    mode_rows = np.ones(dataset.features.shape[0], dtype=bool) if mode == "combined" else dataset.groups == mode
    if mode not in {"combined", "arm", "gripper"}:
        raise ValueError("mode must be combined, arm, or gripper")
    train_rows = mode_rows & (dataset.split == "train")
    test_rows = mode_rows & (dataset.split == "test")
    if not train_rows.any() or not test_rows.any():
        raise ValueError(f"mode {mode!r} needs non-empty train and test rows")
    from .vector_training import load_shared_checkpoint, predict_reliability_curves

    model = load_shared_checkpoint(checkpoint_path, device=device)
    scores = predict_reliability_curves(model, dataset.features, device=device)
    train = dataset.select(train_rows)
    test = dataset.select(test_rows)
    learned_report = evaluate_vector_predictions(
        test.labels,
        scores[test_rows],
        test.label_mask,
        groups=test.groups,
        task_ids=test.task_ids,
        n_bins=n_bins,
    )
    prior_scores = _constant_vector_scores(train, test)
    empirical_scores = _empirical_vector_scores(train, test)
    prior_report = evaluate_vector_predictions(
        test.labels,
        prior_scores,
        test.label_mask,
        groups=test.groups,
        task_ids=test.task_ids,
        n_bins=n_bins,
    )
    empirical_report = evaluate_vector_predictions(
        test.labels,
        empirical_scores,
        test.label_mask,
        groups=test.groups,
        task_ids=test.task_ids,
        n_bins=n_bins,
    )
    empirical_brier = float(empirical_report["overall"]["brier_score"])  # type: ignore[index]
    learned_brier = float(learned_report["overall"]["brier_score"])  # type: ignore[index]
    brier_skill = (
        float(1.0 - learned_brier / empirical_brier)
        if empirical_brier > 0.0
        else float("nan")
    )
    report = {
        "learned_reliability": learned_report,
        "constant_prior": prior_report,
        "empirical_reliability_curve": empirical_report,
        "brier_skill_score_vs_empirical": brier_skill,
        "identity_offset_policy": "mask k=0 in loss and headline metrics; set k=0 to one for horizon decoding",
    }
    predicted_horizons, oracle_horizons = vector_horizon_pairs(
        dataset,
        scores,
        decoder=decoder,
        split="test",
        mode=mode,
    )
    report["horizon_pairs"] = [
        {"predicted": predicted, "oracle": oracle}
        for predicted, oracle in zip(predicted_horizons, oracle_horizons)
    ]
    report["offline_horizon_regret"] = evaluate_vector_horizon_regret(
        dataset,
        scores,
        decoder=decoder,
        split="test",
        mode=mode,
    )
    if tau_values is None:
        tau_values = np.linspace(0.1, 0.9, 17)
    report["validation_tau_sweep"] = evaluate_tau_sweep(
        dataset,
        scores,
        decoder=decoder,
        tau_values=tau_values,
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


def _metric_rows(
    report: Mapping[str, object],
    level: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_name in ("constant_prior", "empirical_reliability_curve", "learned_reliability"):
        model_report = report.get(model_name)
        if not isinstance(model_report, Mapping):
            continue
        entries = model_report.get(level, {})
        if not isinstance(entries, Mapping):
            continue
        for key, values in entries.items():
            if not isinstance(values, Mapping):
                continue
            row: dict[str, object] = {"model": model_name, level: key}
            row.update(values)
            rows.append(row)
    return rows


def _write_csv(rows: list[dict[str, object]], path: Path, first_columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(first_columns)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation_artifacts(
    report: Mapping[str, object],
    output_dir: str | Path,
) -> None:
    """Write the requested JSON, tables, plots, and concise question report."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_evaluation_report(report, output_dir / "metrics.json")
    _write_csv(
        _metric_rows(report, "offset"),
        output_dir / "per_offset_metrics.csv",
        ("model", "offset"),
    )
    _write_csv(
        _metric_rows(report, "group"),
        output_dir / "per_group_metrics.csv",
        ("model", "group"),
    )
    horizon = report.get("offline_horizon_regret", {})
    horizon_rows: list[dict[str, object]] = []
    if isinstance(horizon, Mapping):
        groups = horizon.get("by_group", {})
        if isinstance(groups, Mapping):
            for group, values in groups.items():
                if isinstance(values, Mapping):
                    horizon_rows.append({"group": group, **values})
    _write_csv(horizon_rows, output_dir / "horizon_metrics.csv", ("group",))
    tau_rows = report.get("validation_tau_sweep", [])
    if isinstance(tau_rows, list):
        _write_csv(
            [row for row in tau_rows if isinstance(row, dict)],
            output_dir / "validation_tau_sweep.csv",
            ("tau", "group"),
        )
    try:
        plot_vector_reliability_diagram(report, output_dir / "reliability_diagram.png")
        plot_vector_calibration_curve(report, output_dir / "calibration_curve.png")
        plot_horizon_comparison(report, output_dir / "learned_vs_oracle_horizon.png")
    except ImportError:
        # Metrics and tables remain valid on hosts without matplotlib.
        pass
    write_evaluation_markdown(report, output_dir / "evaluation.md")


def _display_metric(report: Mapping[str, object], path: tuple[str, ...]) -> str:
    value: object = report
    for key in path:
        if not isinstance(value, Mapping):
            return "unavailable"
        value = value.get(key, "unavailable")
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN"
    return str(value)


def write_evaluation_markdown(report: Mapping[str, object], path: str | Path) -> None:
    """Write conclusions as evidence statements, without success claims."""

    learned = report.get("learned_reliability", {})
    empirical = report.get("empirical_reliability_curve", {})
    lines = [
        "# Y_refresh reliability evaluation",
        "",
        "Headline metrics exclude the trivial identity event k=0. The loss also masks k=0; horizon decoding sets its reliability to one.",
        "No rollout, executor, benchmark, or robot-success conclusion is included.",
        "",
        "## Required questions",
        "",
        "### Q1. State-conditioned signal beyond empirical g,k",
        f"Learned pooled AUROC: {_display_metric(learned, ('overall', 'auroc'))}; learned pooled AUPRC: {_display_metric(learned, ('overall', 'auprc'))}; empirical pooled AUROC: {_display_metric(empirical, ('overall', 'auroc'))}; Brier Skill Score versus empirical: {_display_metric(report, ('brier_skill_score_vs_empirical',))}.",
        "These are predictive-signal diagnostics only; they do not establish execution improvement.",
        "",
        "### Q2. Probability calibration",
        f"Learned pooled Brier: {_display_metric(learned, ('overall', 'brier_score'))}; learned ECE: {_display_metric(learned, ('overall', 'ece'))}. See reliability_diagram.png and calibration_curve.png.",
        "",
        "### Q3. Learned horizon versus oracle",
        f"Test horizon MAE: {_display_metric(report, ('offline_horizon_regret', 'overall', 'mae'))}; median absolute error: {_display_metric(report, ('offline_horizon_regret', 'overall', 'median_absolute_error'))}; exact match: {_display_metric(report, ('offline_horizon_regret', 'overall', 'exact_match_rate'))}.",
        "Over-commit and under-commit are reported separately in horizon_metrics.csv.",
        "",
        "### Q4. Arm and gripper consistency",
        "Per-group AUROC/AUPRC/calibration are in per_group_metrics.csv; per-group horizon errors are in horizon_metrics.csv. Conclusions must be read separately for arm and gripper.",
        "",
        "## Baselines and threshold sensitivity",
        "",
        "The constant prior and empirical P(Y_refresh=1 | g,k) curve are fit using train episodes only. Validation tau sensitivity is in validation_tau_sweep.csv; no test threshold selection is performed.",
        "",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


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
    source = report.get("learned_reliability", report)
    curve = source["overall"]["reliability_curve"]  # type: ignore[index]
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
    source = report.get("learned_reliability", report)
    offsets = source["offset"]  # type: ignore[index]
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


def plot_horizon_comparison(
    report: Mapping[str, object],
    path: str | Path,
) -> None:
    """Plot learned-versus-oracle decoded horizons by action group."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - host-dependent
        raise ImportError("plotting requires matplotlib") from error
    pairs = report.get("horizon_pairs", [])
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("horizon_pairs are required for a horizon comparison plot")
    groups = sorted(
        {
            group
            for pair in pairs
            if isinstance(pair, Mapping)
            for group in pair.get("oracle", {})
        }
    )
    figure, axis = plt.subplots(figsize=(5.5, 5.0))
    for group in groups:
        predicted = [pair["predicted"][group] for pair in pairs]
        oracle = [pair["oracle"][group] for pair in pairs]
        axis.scatter(oracle, predicted, s=18, alpha=0.7, label=group)
    maximum = max(
        max(pair["oracle"].values())
        for pair in pairs
        if isinstance(pair, Mapping)
    )
    axis.plot([0, maximum], [0, maximum], "k--", linewidth=1, label="identity")
    axis.set(xlabel="Oracle horizon (actions)", ylabel="Learned horizon (actions)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
