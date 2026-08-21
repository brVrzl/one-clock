#!/usr/bin/env python3
"""Run the predeclared four-condition source-context ablation.

The primary estimator is the same shared monotone conditional-survival MLP as
the existing CPU pilot.  Only the source feature matrix changes across A--D;
Y_refresh, masks, episode splits, seeds, optimizer, hidden dimensions, early
stopping, and validation-only tau selection are fixed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from experiments.dynamic_reliability_horizon.chunk_only_reliability.run_pilot import (
    GROUP_NAMES,
    SEEDS,
    TAU_VALUES,
    NumpySharedMLP,
    PilotData,
    _empirical_scores,
    _horizon_report,
    _initial_biases,
    _metric_slice,
    _score_report,
    _write_json,
)


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "experiments/dynamic_reliability_horizon/source_context_ablation"
FEATURE_PATH = OUTPUT_DIR / "feature_bundle.npz"
FEATURE_MANIFEST_PATH = OUTPUT_DIR / "feature_manifest.json"
BUNDLE_PATH = ROOT / "experiments/dynamic_reliability_horizon/artifact_handoff/minimal_y_refresh_training_bundle.npz"
EXPECTED_BUNDLE_SHA256 = "45a37a57fc03a3850b5c87e88604d66b16886d306e5ee09aa322f52c7e6c50b4"
EXPECTED_FEATURE_KEYS = {
    "source_chunk_actions",
    "source_state",
    "source_policy_latent",
    "episode_id",
    "source_step",
    "group_ids",
    "offsets",
    "y_refresh",
    "label_observed",
    "split_membership",
}
CONDITIONS = {
    "A_chunk_only": ("source_chunk_actions",),
    "B_chunk_plus_state": ("source_chunk_actions", "source_state"),
    "C_chunk_plus_frozen_ACT_latent": ("source_chunk_actions", "source_policy_latent"),
    "D_chunk_plus_state_plus_frozen_ACT_latent": (
        "source_chunk_actions",
        "source_state",
        "source_policy_latent",
    ),
}
MODEL_KIND = "monotone_conditional_survival_shared_mlp"
HIDDEN_DIMS = (64, 32)
TRAINING_CONFIG = {
    "model_kind": MODEL_KIND,
    "hidden_dims": list(HIDDEN_DIMS),
    "seeds": list(SEEDS),
    "epochs": 100,
    "batch_size": 256,
    "learning_rate": 0.002,
    "weight_decay": 1.0e-5,
    "patience": 18,
    "tau_values": list(TAU_VALUES),
    "tau_selection": "minimum validation refresh-oracle MAE; ties prefer higher within +/-2 then lower tau",
    "normalization": "feature-wise mean/std fitted on train source windows only; group one-hot is not normalized",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_value(value) for key, value in row.items()})


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if np.isfinite(value) else "nan"
    if isinstance(value, float) and not np.isfinite(value):
        return "nan"
    return value


def load_feature_data() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(FEATURE_PATH)
    manifest = json.loads(FEATURE_MANIFEST_PATH.read_text(encoding="utf-8"))
    expected_feature_sha = manifest["feature_artifact"]["sha256"]
    actual_feature_sha = sha256(FEATURE_PATH)
    if actual_feature_sha != expected_feature_sha:
        raise RuntimeError(f"feature artifact checksum mismatch: {actual_feature_sha}")
    if sha256(BUNDLE_PATH) != EXPECTED_BUNDLE_SHA256:
        raise RuntimeError("portable Y_refresh bundle checksum changed")
    with np.load(FEATURE_PATH, allow_pickle=False) as arrays:
        if set(arrays.files) != EXPECTED_FEATURE_KEYS:
            raise RuntimeError(f"unexpected feature artifact arrays: {arrays.files}")
        data = {name: np.asarray(arrays[name]).copy() for name in arrays.files}
    expected_shapes = {
        "source_chunk_actions": (3740, 100, 7),
        "source_state": (3740, 8),
        "source_policy_latent": (3740, 512),
        "episode_id": (3740,),
        "source_step": (3740,),
        "group_ids": (2,),
        "offsets": (99,),
        "y_refresh": (3740, 2, 99),
        "label_observed": (3740, 2, 99),
        "split_membership": (3740,),
    }
    for name, shape in expected_shapes.items():
        if data[name].shape != shape:
            raise RuntimeError(f"{name} shape {data[name].shape} != {shape}")
    if not np.array_equal(data["group_ids"], np.asarray([0, 1], dtype=np.int8)):
        raise RuntimeError("unexpected group IDs")
    if not np.array_equal(data["offsets"], np.arange(1, 100, dtype=np.int16)):
        raise RuntimeError("unexpected offsets")
    if len(set(zip(data["episode_id"].tolist(), data["source_step"].tolist(), strict=True))) != 3740:
        raise RuntimeError("feature artifact source keys are not unique")
    if not np.isfinite(data["source_chunk_actions"]).all() or not np.isfinite(data["source_state"]).all() or not np.isfinite(data["source_policy_latent"]).all():
        raise RuntimeError("non-finite source feature value")
    with np.load(BUNDLE_PATH, allow_pickle=False) as bundle:
        if not np.array_equal(data["source_chunk_actions"], bundle["source_chunk_actions"]):
            raise RuntimeError("feature artifact changed the locked source chunk")
        if not np.array_equal(data["y_refresh"], bundle["y_refresh"]):
            raise RuntimeError("feature artifact changed Y_refresh")
        if not np.array_equal(data["label_observed"], bundle["label_observed"]):
            raise RuntimeError("feature artifact changed censor mask")
        if not np.array_equal(data["split_membership"], bundle["split_membership"]):
            raise RuntimeError("feature artifact changed split membership")
        if not np.array_equal(data["episode_id"], bundle["episode_index"]):
            raise RuntimeError("feature artifact changed episode IDs")
    if len(np.unique(data["episode_id"])) != 454:
        raise RuntimeError("unexpected source episode count")
    split = data["split_membership"]
    split_sets = [set(data["episode_id"][split == code].tolist()) for code in (0, 1, 2)]
    if any(split_sets[left] & split_sets[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("episode leakage in feature artifact")
    return data, manifest


def build_pilot_data(data: dict[str, np.ndarray], feature_names: tuple[str, ...]) -> PilotData:
    source_features = np.concatenate(
        [data[name].reshape(data[name].shape[0], -1).astype(np.float64) for name in feature_names], axis=1
    )
    source_split = data["split_membership"]
    train_windows = source_split == 0
    mean = source_features[train_windows].mean(axis=0)
    std = source_features[train_windows].std(axis=0)
    std = np.where(std < 1.0e-6, 1.0, std)
    scaled = ((source_features - mean) / std).astype(np.float32)
    repeated = np.concatenate((scaled, scaled), axis=0)
    group_ids = np.concatenate(
        (np.zeros(source_features.shape[0], dtype=np.int8), np.ones(source_features.shape[0], dtype=np.int8))
    )
    group_one_hot = np.eye(2, dtype=np.float32)[group_ids]
    features = np.concatenate((repeated, group_one_hot), axis=1)
    labels = np.concatenate((data["y_refresh"][:, 0, :], data["y_refresh"][:, 1, :]), axis=0)
    label_mask = np.concatenate((data["label_observed"][:, 0, :], data["label_observed"][:, 1, :]), axis=0)
    split_codes = np.concatenate((source_split, source_split), axis=0)
    split_names = np.asarray([("train", "validation", "test")[int(code)] for code in split_codes])
    return PilotData(
        features=features,
        labels=labels,
        label_mask=label_mask,
        groups=np.asarray([GROUP_NAMES[int(value)] for value in group_ids], dtype=str),
        group_ids=group_ids,
        episode_ids=np.concatenate((data["episode_id"], data["episode_id"]), axis=0).astype(str),
        split=split_names,
        source_window_ids=np.concatenate((np.arange(3740), np.arange(3740))),
        source_chunks=data["source_chunk_actions"],
        normalization_mean=mean.astype(np.float32),
        normalization_std=std.astype(np.float32),
    )


def train_condition(data: PilotData, condition: str) -> tuple[dict[str, Any], np.ndarray, dict[str, Any], dict[str, Any]]:
    empirical_constant, empirical_scores = _empirical_scores(data)
    train_rows = data.split == "train"
    validation_rows = data.split == "validation"
    seed_predictions: list[np.ndarray] = []
    seed_summaries: dict[str, Any] = {}
    for seed in SEEDS:
        model = NumpySharedMLP(
            data.features.shape[1],
            data.labels.shape[1],
            kind=MODEL_KIND,
            seed=seed,
            hidden_dims=HIDDEN_DIMS,
            initial_output_bias=_initial_biases(data.labels[train_rows], data.label_mask[train_rows], MODEL_KIND),
        )
        summary = model.fit(
            data.features,
            data.labels.astype(np.float64),
            data.label_mask,
            train_rows,
            validation_rows,
            epochs=100,
            batch_size=256,
            learning_rate=0.002,
            weight_decay=1.0e-5,
            patience=18,
            seed=seed,
        )
        prediction = model.predict(data.features)
        seed_predictions.append(prediction)
        seed_summaries[str(seed)] = summary
    predictions = np.mean(seed_predictions, axis=0)
    test = data.split == "test"
    report = _score_report(
        data.labels[test],
        predictions[test],
        data.label_mask[test],
        data.groups[test],
        empirical_scores[test],
    )
    validation_sweeps: list[dict[str, Any]] = []
    for tau in TAU_VALUES:
        validation_report = _horizon_report(
            predictions, data.labels, data.label_mask, data.groups, data.split, target_split="validation", tau=float(tau)
        )
        validation_sweeps.append(validation_report)
    selected = min(
        validation_sweeps,
        key=lambda values: (
            values["overall"]["mae"],
            -values["overall"]["within_plus_minus_2"],
            values["tau"],
        ),
    )
    horizon = _horizon_report(
        predictions, data.labels, data.label_mask, data.groups, data.split, target_split="test", tau=float(selected["tau"])
    )
    baselines = {
        "constant_prior|train_only": {
            "scores": empirical_constant,
            "report": _score_report(data.labels[test], empirical_constant[test], data.label_mask[test], data.groups[test], empirical_scores[test]),
        },
        "empirical_group_offset_prior_train_only|train_only": {
            "scores": empirical_scores,
            "report": _score_report(data.labels[test], empirical_scores[test], data.label_mask[test], data.groups[test], empirical_scores[test]),
            "horizon": _horizon_report(empirical_scores, data.labels, data.label_mask, data.groups, data.split, target_split="test", tau=0.5),
        },
    }
    summary = {
        "condition": condition,
        "model_kind": MODEL_KIND,
        "input_features": list(CONDITIONS[condition]),
        "input_dim": int(data.features.shape[1]),
        "normalization_mean_shape": list(data.normalization_mean.shape),
        "seed_runs": seed_summaries,
        "aggregate_of_seeds": list(SEEDS),
        "selected_tau": float(selected["tau"]),
        "validation_horizon_selection": selected,
    }
    return summary, predictions, {"test": report, "horizon": horizon}, baselines


def fixed_group_macro(report: dict[str, Any], group: str) -> dict[str, Any]:
    values = [item for item in report["per_group_offset"][group].values() if not item["undefined_discrimination"]]
    return {
        "auroc": float(np.mean([item["auroc"] for item in values])) if values else float("nan"),
        "auprc": float(np.mean([item["auprc"] for item in values])) if values else float("nan"),
        "valid_offset_count": len(values),
    }


def comparison_row(condition: str, report: dict[str, Any], horizon: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"condition": condition, "model": MODEL_KIND, "tau": horizon["tau"]}
    for prefix, values in (("pooled", report["overall"]), ("pooled_macro_offset", report["macro_valid_nontrivial_offsets"])):
        for metric in ("auroc", "auprc", "brier", "ece", "brier_skill_score_vs_empirical"):
            row[f"{prefix}_{metric}"] = values[metric]
    for group in GROUP_NAMES:
        macro = fixed_group_macro(report, group)
        for metric in ("auroc", "auprc", "valid_offset_count"):
            row[f"{group}_fixed_offset_macro_{metric}"] = macro[metric]
        for metric, value in horizon["by_group"][group].items():
            row[f"{group}_horizon_{metric}"] = value
    for metric, value in horizon["overall"].items():
        row[f"overall_horizon_{metric}"] = value
    return row


def make_plots(
    output_dir: Path,
    reports: dict[str, dict[str, Any]],
    predictions: dict[str, np.ndarray],
    data: PilotData,
    horizons: dict[str, dict[str, Any]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        (output_dir / "plots" / "README.md").write_text("matplotlib unavailable; numeric CSV/JSON artifacts are authoritative.\n", encoding="utf-8")
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {condition: color for condition, color in zip(CONDITIONS, ("C0", "C1", "C2", "C3"), strict=True)}
    offsets = np.arange(1, 100)

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for condition, report in reports.items():
        axes[0].plot(offsets, [report["per_offset"][str(k)]["auroc"] for k in offsets], label=condition, color=colors[condition])
        axes[1].plot(offsets, [report["per_offset"][str(k)]["brier"] for k in offsets], label=condition, color=colors[condition])
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("pooled AUROC")
    axes[1].set_ylabel("pooled Brier")
    axes[1].set_xlabel("nontrivial offset k")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(plot_dir / "fixed_offset_comparison.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for condition, report in reports.items():
        for axis, group in zip(axes, GROUP_NAMES, strict=True):
            axis.plot(offsets, [report["per_group_offset"][group][str(k)]["auroc"] for k in offsets], label=condition, color=colors[condition])
            axis.set_ylabel(f"{group} AUROC")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
    axes[-1].set_xlabel("nontrivial offset k")
    figure.tight_layout()
    figure.savefig(plot_dir / "fixed_group_offset_auroc.png", dpi=160)
    plt.close(figure)

    selected = data.split == "test"
    for group in GROUP_NAMES:
        figure, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
        for axis, (condition, scores) in zip(axes.flat, predictions.items(), strict=True):
            rows = selected & (data.groups == group)
            learned = np.asarray([_decode(row, horizons[condition]["tau"]) for row in scores[rows]])
            oracle = np.asarray([_oracle(y, m) for y, m in zip(data.labels[rows], data.label_mask[rows], strict=True)])
            axis.scatter(oracle, learned, s=8, alpha=0.35, color=colors[condition])
            axis.set_title(condition)
            axis.grid(alpha=0.25)
            axis.set_xlabel("refresh-oracle horizon")
            axis.set_ylabel("learned horizon")
        figure.suptitle(f"{group}: horizon recovery on held-out test sources")
        figure.tight_layout()
        figure.savefig(plot_dir / f"horizon_recovery_{group}.png", dpi=160)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    flat_mask = data.label_mask[selected].reshape(-1)
    flat_labels = data.labels[selected].reshape(-1)[flat_mask]
    for condition, scores in predictions.items():
        flat_scores = scores[selected].reshape(-1)[flat_mask]
        bins = np.minimum((flat_scores * 10).astype(int), 9)
        means = [float(flat_scores[bins == b].mean()) if np.any(bins == b) else np.nan for b in range(10)]
        observed = [float(flat_labels[bins == b].mean()) if np.any(bins == b) else np.nan for b in range(10)]
        axis.plot(means, observed, marker="o", label=condition, color=colors[condition])
    axis.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axis.set_xlabel("mean predicted probability")
    axis.set_ylabel("observed frequency")
    axis.set_title("Held-out probability calibration")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(plot_dir / "calibration_comparison.png", dpi=160)
    plt.close(figure)


def _decode(scores: np.ndarray, tau: float) -> int:
    horizon = 1
    for value in scores:
        if float(value) <= tau:
            break
        horizon += 1
    return horizon


def _oracle(labels: np.ndarray, mask: np.ndarray) -> int:
    horizon = 1
    for label, observed in zip(labels, mask, strict=True):
        if not observed or not label:
            break
        horizon += 1
    return horizon


def evaluate(output_dir: Path) -> dict[str, Any]:
    data_arrays, feature_manifest = load_feature_data()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "config.json", {
        "status": "completed",
        "conditions": {condition: list(features) for condition, features in CONDITIONS.items()},
        "training": TRAINING_CONFIG,
        "feature_bundle": str(FEATURE_PATH.resolve()),
        "feature_bundle_sha256": sha256(FEATURE_PATH),
        "portable_bundle_sha256": EXPECTED_BUNDLE_SHA256,
        "scientific_scope": "four causal source-context feature conditions; no rollout and no executor changes",
    })
    _write_json(output_dir / "seeds.json", {"seeds": list(SEEDS), "deterministic": True})

    reports: dict[str, dict[str, Any]] = {}
    predictions: dict[str, np.ndarray] = {}
    horizons: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    baselines: dict[str, Any] | None = None
    pilot_data_by_condition: dict[str, PilotData] = {}
    for condition, feature_names in CONDITIONS.items():
        pilot_data = build_pilot_data(data_arrays, feature_names)
        pilot_data_by_condition[condition] = pilot_data
        summary, prediction, report, condition_baselines = train_condition(pilot_data, condition)
        summaries[condition] = summary
        predictions[condition] = prediction
        reports[condition] = report["test"]
        horizons[condition] = report["horizon"]
        if baselines is None:
            baselines = condition_baselines

    if baselines is None:
        raise RuntimeError("no conditions evaluated")
    # A is an exact reproduction check for the existing pilot's same monotone
    # estimator and aggregate-seed protocol.
    prior_metrics_path = ROOT / "experiments/dynamic_reliability_horizon/chunk_only_reliability/metrics.json"
    prior = json.loads(prior_metrics_path.read_text(encoding="utf-8"))
    prior_monotone = prior["models"][f"{MODEL_KIND}|aggregate_mean_prediction"]
    current_a = reports["A_chunk_only"]
    parity_fields = {
        "overall_auroc": (current_a["overall"]["auroc"], prior_monotone["test_metrics"]["overall"]["auroc"]),
        "overall_brier": (current_a["overall"]["brier"], prior_monotone["test_metrics"]["overall"]["brier"]),
        "macro_brier_skill": (current_a["macro_valid_nontrivial_offsets"]["brier_skill_score_vs_empirical"], prior_monotone["test_metrics"]["macro_valid_nontrivial_offsets"]["brier_skill_score_vs_empirical"]),
        "overall_horizon_mae": (horizons["A_chunk_only"]["overall"]["mae"], prior_monotone["test_horizon"]["overall"]["mae"]),
    }
    parity = {
        field: {"current": current, "existing": existing, "absolute_delta": abs(current - existing)}
        for field, (current, existing) in parity_fields.items()
    }
    parity["all_within_1e-12"] = all(values["absolute_delta"] <= 1e-12 for values in parity.values() if isinstance(values, dict))
    if not parity["all_within_1e-12"]:
        raise RuntimeError(f"condition A does not reproduce existing monotone chunk-only result: {parity}")

    group_rows: list[dict[str, Any]] = []
    offset_rows: list[dict[str, Any]] = []
    group_offset_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for condition, report in reports.items():
        for group, values in report["per_group"].items():
            group_rows.append({"condition": condition, "model": MODEL_KIND, "group": group, **values})
        for offset, values in report["per_offset"].items():
            offset_rows.append({"condition": condition, "model": MODEL_KIND, "offset": offset, **values})
        for group, offsets in report["per_group_offset"].items():
            for offset, values in offsets.items():
                group_offset_rows.append({"condition": condition, "model": MODEL_KIND, "group": group, "offset": offset, **values})
        for group, values in horizons[condition]["by_group"].items():
            horizon_rows.append({"condition": condition, "model": MODEL_KIND, "group": group, **values})
        horizon_rows.append({"condition": condition, "model": MODEL_KIND, "group": "overall", **horizons[condition]["overall"]})
        comparison_rows.append(comparison_row(condition, report, horizons[condition]))
    for baseline_name, baseline in baselines.items():
        if "report" in baseline:
            for group, values in baseline["report"]["per_group"].items():
                group_rows.append({"condition": baseline_name, "model": "baseline", "group": group, **values})
        if "horizon" in baseline:
            for group, values in baseline["horizon"]["by_group"].items():
                horizon_rows.append({"condition": baseline_name, "model": "baseline", "group": group, **values})
            horizon_rows.append({"condition": baseline_name, "model": "baseline", "group": "overall", **baseline["horizon"]["overall"]})

    write_csv(output_dir / "per_group_metrics.csv", group_rows)
    write_csv(output_dir / "per_offset_metrics.csv", offset_rows)
    write_csv(output_dir / "per_group_offset_metrics.csv", group_offset_rows)
    write_csv(output_dir / "horizon_metrics.csv", horizon_rows)
    write_csv(output_dir / "comparison_table.csv", comparison_rows)
    make_plots(output_dir, reports, predictions, pilot_data_by_condition["A_chunk_only"], horizons)

    baseline_horizon = baselines["empirical_group_offset_prior_train_only|train_only"]["horizon"]["overall"]
    primary_summaries: dict[str, Any] = {}
    for condition, report in reports.items():
        horizon = horizons[condition]
        signal: dict[str, Any] = {}
        for group in GROUP_NAMES:
            macro = fixed_group_macro(report, group)
            signal[group] = {
                "mean_valid_group_offset_auroc": macro["auroc"],
                "mean_valid_group_offset_auprc": macro["auprc"],
                "valid_group_offset_slices": macro["valid_offset_count"],
                "meaningful_signal": bool(macro["valid_offset_count"] >= 10 and macro["auroc"] >= 0.55),
            }
        horizon_useful = bool(
            horizon["overall"]["mae"] <= baseline_horizon["mae"]
            and horizon["overall"]["within_plus_minus_2"] >= 0.50
        )
        probability_improves = bool(
            report["overall"]["brier_skill_score_vs_empirical"] > 0.0
            or report["macro_valid_nontrivial_offsets"]["brier_skill_score_vs_empirical"] > 0.0
        )
        any_signal = any(values["meaningful_signal"] for values in signal.values())
        primary_summaries[condition] = {
            "signal_by_group": signal,
            "probability_metrics_improve_vs_empirical": probability_improves,
            "horizon_recovery_useful_by_preregistered_rule": horizon_useful,
            "horizon_delta_vs_empirical_mae": horizon["overall"]["mae"] - baseline_horizon["mae"],
            "verdict": "GO" if all(values["meaningful_signal"] for values in signal.values()) and horizon_useful else "PARTIAL" if any_signal or probability_improves or horizon_useful else "NO-GO",
        }
    best_condition = min(
        CONDITIONS,
        key=lambda condition: (
            horizons[condition]["overall"]["mae"],
            -horizons[condition]["overall"]["within_plus_minus_2"],
        ),
    )
    any_useful = any(values["horizon_recovery_useful_by_preregistered_rule"] for values in primary_summaries.values())
    final_verdict = "GO" if any_useful and all(primary_summaries[best_condition]["signal_by_group"][g]["meaningful_signal"] for g in GROUP_NAMES) else "PARTIAL" if any(values["verdict"] != "NO-GO" for values in primary_summaries.values()) else "NO-GO"
    scientific = {
        "condition_summaries": primary_summaries,
        "best_useful_condition_by_test_mae": best_condition,
        "previous_verdict": "PARTIAL",
        "horizon_recovery_materially_improved_enough_to_overturn_previous_verdict": bool(final_verdict == "GO"),
        "final_verdict": final_verdict,
        "criterion_unchanged": "useful horizon requires test overall MAE <= empirical group/offset baseline MAE and within +/-2 >= 0.50; tau selected on validation only",
        "recommended_next_experiment": "Predeclare a causal short-history state representation using only the current and immediately preceding observation.state (including finite differences), with the same frozen Y_refresh, episode split, estimator, validation protocol, and usefulness criterion; do not roll out.",
    }
    metrics = {
        "status": "completed",
        "git_head": git_head(),
        "portable_bundle_sha256": EXPECTED_BUNDLE_SHA256,
        "feature_bundle": {
            "path": str(FEATURE_PATH.resolve()),
            "sha256": sha256(FEATURE_PATH),
            "manifest_sha256": sha256(FEATURE_MANIFEST_PATH),
        },
        "dataset": {
            "source_windows": 3740,
            "source_chunk_shape": [3740, 100, 7],
            "source_state_shape": [3740, 8],
            "source_policy_latent_shape": [3740, 512],
            "episode_count": 454,
            "split_rows": {name: int(np.sum(data_arrays["split_membership"] == code)) for code, name in enumerate(("train", "validation", "test"))},
            "group_rows": 7480,
            "no_episode_leakage": True,
        },
        "protocol": {"conditions": {name: list(features) for name, features in CONDITIONS.items()}, "training": TRAINING_CONFIG, "condition_A_parity": parity},
        "baselines": {name: {key: value for key, value in item.items() if key != "scores"} for name, item in baselines.items()},
        "conditions": {
            condition: {
                "training": summaries[condition],
                "test_metrics": reports[condition],
                "test_horizon": horizons[condition],
            }
            for condition in CONDITIONS
        },
        "scientific_verdict": scientific,
    }
    _write_json(output_dir / "metrics.json", metrics)
    evaluation = build_evaluation_markdown(metrics, comparison_rows, baseline_horizon, feature_manifest)
    (output_dir / "evaluation.md").write_text(evaluation, encoding="utf-8")
    return metrics


def build_evaluation_markdown(metrics: dict[str, Any], rows: list[dict[str, Any]], baseline_horizon: dict[str, Any], feature_manifest: dict[str, Any]) -> str:
    verdict = metrics["scientific_verdict"]["final_verdict"]
    best = metrics["scientific_verdict"]["best_useful_condition_by_test_mae"]
    lines = [
        "# Exact causal source-context ablation",
        "",
        f"Status: completed offline; final scientific verdict: **{verdict}**.",
        "",
        f"The exact portable Y_refresh bundle remains locked at SHA256 `{EXPECTED_BUNDLE_SHA256}`. The feature artifact is `{metrics['feature_bundle']['sha256']}`. No Y_refresh regeneration, rollout, executor change, or paper-claim change occurred.",
        "",
        "## Frozen cohort and features",
        "",
        "The 3,740 source windows are joined by unique `(episode_id, source_step) = (episode_index, frame_index)` keys. State is the causal 8-vector `[EEF position (3), EEF axis-angle orientation (3), gripper qpos (2)]` at source time. The primary frozen-ACT representation is the 512-D first token of the final fused `policy.model.encoder` output, captured before ACT decoding; it includes only the current state and current images and excludes the training-time VAE action-conditioned latent.",
        "",
        f"Latent extraction invariance: max postprocessed action delta `{feature_manifest['extraction']['invariance']['max_abs_action_delta']}`, allclose at `1e-6`: `{feature_manifest['extraction']['invariance']['allclose_atol_1e-6_rtol_1e-6']}`. The locked chunk remains canonical; same-path replay drift is recorded separately with max absolute delta `{feature_manifest['extraction']['source_chunk_parity']['max_abs_source_chunk_delta']}`.",
        "",
        "## Primary results",
        "",
        "All conditions use one shared monotone conditional-survival MLP with the same `(64, 32)` hidden dimensions, seeds, optimizer, episode split, censor mask, and validation-only tau selection. Condition A reproduces the existing same-formulation chunk-only pilot within `1e-12` on the parity fields recorded in `metrics.json`.",
        "",
        "| Condition | pooled AUROC | pooled Brier | pooled Brier Skill | arm fixed-offset AUROC | gripper fixed-offset AUROC | arm MAE | gripper MAE | overall MAE | overall within ±2 | Spearman | tau |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {condition} | {pooled_auroc:.4f} | {pooled_brier:.4f} | {pooled_macro_offset_brier_skill_score_vs_empirical:.4f} | {arm_fixed_offset_macro_auroc:.4f} | {gripper_fixed_offset_macro_auroc:.4f} | {arm_horizon_mae:.2f} | {gripper_horizon_mae:.2f} | {overall_horizon_mae:.2f} | {overall_horizon_within_plus_minus_2:.3f} | {overall_horizon_spearman:.3f} | {tau:.2f} |".format(**row)
        )
    lines.extend(
        [
            "",
            f"Empirical group/offset baseline horizon: overall MAE `{baseline_horizon['mae']:.2f}`, within ±2 `{baseline_horizon['within_plus_minus_2']:.3f}`. The best test-MAE condition is `{best}`; this selection is descriptive after the fixed protocol, while each condition's tau was selected on validation only.",
            "",
            "## Scientific questions",
            "",
            "1. **State beyond chunk-only:** see B versus A in `comparison_table.csv`; fixed-offset and horizon changes are reported without retuning the criterion.",
            "2. **Frozen ACT latent beyond chunk-only:** see C versus A.",
            "3. **Arm versus gripper:** compare the fixed-offset macro columns and group horizon rows; this directly tests whether latent value is concentrated in arm reliability.",
            "4. **Horizon recovery:** the predeclared usefulness criterion is unchanged. High AUROC alone does not overturn PARTIAL; the arm and overall horizon metrics are decisive.",
            "5. **Smallest useful set:** compare A-D in `comparison_table.csv`; if no augmented condition meets the usefulness rule, the simplest useful estimator remains chunk-only and the result is a causal augmentation failure, not a failure of the project-wide chunk signal.",
            "",
            "## Artifacts",
            "",
            "- `feature_bundle.npz`, `feature_manifest.json`, `feature_provenance.md` — immutable source features and provenance.",
            "- `metrics.json`, `comparison_table.csv`, `per_group_metrics.csv`, `per_offset_metrics.csv`, `per_group_offset_metrics.csv`, `horizon_metrics.csv` — held-out metrics and exact protocol.",
            "- `config.json`, `seeds.json`, `plots/` — reproducibility configuration and figures.",
            "",
            "No executor semantics, Y_refresh labels, rollout code, or paper claims were changed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    metrics = evaluate(args.output_dir)
    print(f"SOURCE_CONTEXT_ABLATION_COMPLETE verdict={metrics['scientific_verdict']['final_verdict']} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
