#!/usr/bin/env python3
"""Frozen Phase-0 PPPR control-relevance analysis.

This is a CPU-only, outcome-aware analysis of the already-built Phase-0
feature table and the paired intervention rollout logs.  It deliberately
does not import a policy or simulator and does not train or tune a predictor.

Run from the repository root with the frozen environment, for example::

    /home/wjq/workspace/venvs/libero_act/bin/python \
      research/overnight_pppr_20260828/analyze_control_relevance.py

The default invocation is idempotent after ``phase0_control_relevance.complete``
exists.  Use ``--force`` only when deliberately regenerating the outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_FEATURES = HERE / "phase0_features.npz"
DEFAULT_PROTOCOL = HERE / "phase0_protocol.json"
DEFAULT_ROLLOUT_PROTOCOL = REPO_ROOT / "experiments/component_temporal_reuse/protocol.json"
DEFAULT_PILOT = REPO_ROOT / "experiments/component_temporal_reuse/pilot_results.json"
DEFAULT_JSON = HERE / "phase0_control_relevance.json"
DEFAULT_MARKDOWN = HERE / "phase0_control_relevance.md"
DEFAULT_PAIRS = HERE / "phase0_pairs.csv"
DEFAULT_MARKER = HERE / "phase0_analysis.complete"

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 2_026_0828
METHODS = ("age", "event", "raw_ppr", "pppr")
COMPONENT_GROUPS = ("full_old_joint", "reverse_arm", "fo_grip")
SPLITS = ("development", "held_out")

CONDITION_SPECS: dict[str, dict[str, Any]] = {
    "full_old4": {
        "group": "full_old_joint",
        "component": "joint",
        "event_components": ("arm", "gripper"),
        "age": 4,
        "raw_field": "raw_ppr_joint",
        "pppr_field": "pppr_joint",
    },
    "full_old8": {
        "group": "full_old_joint",
        "component": "joint",
        "event_components": ("arm", "gripper"),
        "age": 8,
        "raw_field": "raw_ppr_joint",
        "pppr_field": "pppr_joint",
    },
    "full_old16": {
        "group": "full_old_joint",
        "component": "joint",
        "event_components": ("arm", "gripper"),
        "age": 16,
        "raw_field": "raw_ppr_joint",
        "pppr_field": "pppr_joint",
    },
    "reverse4": {
        "group": "reverse_arm",
        "component": "arm",
        "age": 4,
        "raw_field": "raw_ppr_arm",
        "pppr_field": "pppr_arm",
    },
    "reverse8": {
        "group": "reverse_arm",
        "component": "arm",
        "age": 8,
        "raw_field": "raw_ppr_arm",
        "pppr_field": "pppr_arm",
    },
    "reverse16": {
        "group": "reverse_arm",
        "component": "arm",
        "age": 16,
        "raw_field": "raw_ppr_arm",
        "pppr_field": "pppr_arm",
    },
    "fo4": {
        "group": "fo_grip",
        "component": "gripper",
        "age": 4,
        "raw_field": "raw_ppr_grip",
        "pppr_field": "pppr_grip",
    },
    "fo8": {
        "group": "fo_grip",
        "component": "gripper",
        "age": 8,
        "raw_field": "raw_ppr_grip",
        "pppr_field": "pppr_grip",
    },
    "fo16": {
        "group": "fo_grip",
        "component": "gripper",
        "age": 16,
        "raw_field": "raw_ppr_grip",
        "pppr_field": "pppr_grip",
    },
}


def _jsonable(value: Any) -> Any:
    """Convert NumPy scalars/arrays to JSON-safe values."""

    if isinstance(value, np.ndarray):
        return [_jsonable(x) for x in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(x) for x in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_features(path: Path) -> dict[str, np.ndarray]:
    """Load the frozen feature table without object-array deserialization."""

    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def build_feature_lookup(features: Mapping[str, np.ndarray]) -> dict[tuple[str, int, int, int], int]:
    """Index feature rows by the protocol's pre-treatment key."""

    required = ("task_key", "episode", "old_query_t", "age_steps", "valid", "future_query_u")
    missing = [name for name in required if name not in features]
    if missing:
        raise ValueError(f"feature table lacks required columns: {missing}")
    lookup: dict[tuple[str, int, int, int], int] = {}
    for index in range(len(features["task_key"])):
        key = (
            str(features["task_key"][index]),
            int(features["episode"][index]),
            int(features["old_query_t"][index]),
            int(features["age_steps"][index]),
        )
        if key in lookup:
            raise ValueError(f"duplicate feature key {key}")
        lookup[key] = index
    return lookup


def _feature_value(features: Mapping[str, np.ndarray], field: str, index: int) -> float:
    value = float(features[field][index])
    if not np.isfinite(value):
        raise ValueError(f"non-finite valid feature value in {field} at row {index}")
    return value


def align_episode_condition(
    *,
    task_key: str,
    episode_index: int,
    condition: str,
    source_events: Sequence[Mapping[str, Any]],
    features: Mapping[str, np.ndarray],
    feature_lookup: Mapping[tuple[str, int, int, int], int],
) -> dict[str, Any]:
    """Align logged active intervention steps to Fresh-reference rows.

    For each event at physical step ``u``, only the relevant component's
    ``actual_source_age_steps == d`` events are active.  The only feature row
    consulted is ``(task, episode, old_query_t=u-d, age_steps=d)``.  Thus no
    prediction from an intervention trajectory can enter this alignment.
    """

    if condition not in CONDITION_SPECS:
        raise ValueError(f"unknown intervention condition {condition}")
    spec = CONDITION_SPECS[condition]
    d = int(spec["age"])
    component = str(spec["component"])
    event_components = tuple(str(x) for x in spec.get("event_components", (component,)))
    active_steps: list[int] = []
    valid_steps: list[int] = []
    missing_steps: list[int] = []
    invalid_feature_steps: list[int] = []
    raw_values: list[float] = []
    pppr_values: list[float] = []
    event_values: list[float] = []

    for event in source_events:
        if str(event.get("condition")) != condition:
            continue
        u = int(event["environment_step"])
        sources = [event.get(name) for name in event_components]
        if any(not isinstance(source, Mapping) for source in sources):
            raise ValueError(f"source event at u={u} has no required {event_components} records")
        if any(int(source["actual_source_age_steps"]) != d for source in sources):
            # Warm-up events are intentionally ignored; they are not active
            # uses of the requested old source age.
            continue
        active_steps.append(u)
        old_query_t = u - d
        key = (task_key, int(episode_index), old_query_t, d)
        row_index = feature_lookup.get(key)
        if row_index is None:
            missing_steps.append(u)
            continue
        if not bool(features["valid"][row_index]) or int(features["future_query_u"][row_index]) != u:
            invalid_feature_steps.append(u)
            continue
        valid_steps.append(u)
        raw_values.append(_feature_value(features, str(spec["raw_field"]), row_index))
        pppr_values.append(_feature_value(features, str(spec["pppr_field"]), row_index))
        event_values.append(_feature_value(features, "event_event_score", row_index))

    missing_any = missing_steps + invalid_feature_steps
    score_valid = bool(valid_steps)
    return {
        "active_logged_steps": int(len(active_steps)),
        "valid_feature_steps": int(len(valid_steps)),
        "missing_feature_steps": int(len(missing_any)),
        "active_steps": active_steps,
        "valid_steps": valid_steps,
        "missing_steps": sorted(missing_any),
        "missing_key_steps": sorted(missing_steps),
        "invalid_feature_steps": sorted(invalid_feature_steps),
        "score_valid": score_valid,
        "age": float(d) if score_valid else float("nan"),
        "event": float(np.mean(event_values)) if score_valid else float("nan"),
        "raw_ppr": float(np.mean(raw_values)) if score_valid else float("nan"),
        "pppr": float(np.mean(pppr_values)) if score_valid else float("nan"),
    }


def make_pair_rows(
    *,
    features: Mapping[str, np.ndarray],
    pilot: Mapping[str, Any],
    split_by_task: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Construct one row per task/episode/intervention condition."""

    lookup = build_feature_lookup(features)
    rows: list[dict[str, Any]] = []
    for task_key, split in split_by_task.items():
        task = pilot.get("tasks", {}).get(task_key)
        if not isinstance(task, Mapping):
            raise ValueError(f"pilot_results has no task {task_key}")
        conditions = task.get("conditions", {})
        fresh = conditions.get("fresh")
        if not isinstance(fresh, Mapping):
            raise ValueError(f"pilot_results has no Fresh condition for {task_key}")
        fresh_successes = list(fresh.get("successes", []))
        for condition, spec in CONDITION_SPECS.items():
            intervention = conditions.get(condition)
            if not isinstance(intervention, Mapping):
                raise ValueError(f"pilot_results has no condition {condition} for {task_key}")
            episodes = intervention.get("episodes_detail", [])
            if len(episodes) != len(fresh_successes):
                raise ValueError(f"episode count mismatch for {task_key}/{condition}")
            intervention_successes = list(intervention.get("successes", []))
            if len(intervention_successes) != len(episodes):
                raise ValueError(f"success count mismatch for {task_key}/{condition}")
            for episode_index, episode in enumerate(episodes):
                if not isinstance(episode, Mapping):
                    raise ValueError(f"malformed episode record {task_key}/{condition}/{episode_index}")
                fresh_success = bool(fresh_successes[episode_index])
                intervention_success = bool(intervention_successes[episode_index])
                delta_y = int(fresh_success) - int(intervention_success)
                alignment = align_episode_condition(
                    task_key=task_key,
                    episode_index=episode_index,
                    condition=condition,
                    source_events=episode.get("source_events", []),
                    features=features,
                    feature_lookup=lookup,
                )
                row = {
                    "task_key": task_key,
                    "split": split,
                    "episode": int(episode_index),
                    "condition": condition,
                    "intervention_group": spec["group"],
                    "component": spec["component"],
                    "age_steps": int(spec["age"]),
                    "fresh_success": fresh_success,
                    "intervention_success": intervention_success,
                    "delta_y": delta_y,
                    "z": int(delta_y == 1),
                    "decisive": bool(delta_y != 0),
                    "cluster_episode": f"{task_key}|episode={episode_index}",
                    "active_logged_steps": alignment["active_logged_steps"],
                    "valid_feature_steps": alignment["valid_feature_steps"],
                    "missing_feature_steps": alignment["missing_feature_steps"],
                    "active_steps": alignment["active_steps"],
                    "valid_steps": alignment["valid_steps"],
                    "missing_steps": alignment["missing_steps"],
                    "missing_key_steps": alignment["missing_key_steps"],
                    "invalid_feature_steps": alignment["invalid_feature_steps"],
                    "score_valid": alignment["score_valid"],
                    "age": alignment["age"],
                    "event": alignment["event"],
                    "raw_ppr": alignment["raw_ppr"],
                    "pppr": alignment["pppr"],
                }
                rows.append(row)
    return rows


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    n_pos = int((labels == 1).sum())
    if n_pos == 0 or int((labels == 0).sum()) == 0:
        return float("nan")
    # Stable descending sort is the same tie treatment as sklearn's AP for
    # finite scores and makes this fallback deterministic.
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order].astype(int)
    cumulative = np.cumsum(y)
    positions = np.flatnonzero(y == 1)
    return float(np.sum(cumulative[positions] / (positions + 1.0)) / n_pos)


try:  # sklearn is available in the frozen environment; retain a small fallback.
    from sklearn.metrics import average_precision_score as _sk_average_precision
    from sklearn.metrics import roc_auc_score as _sk_roc_auc
except Exception:  # pragma: no cover - exercised only without sklearn
    _sk_average_precision = None
    _sk_roc_auc = None


def metric_values(labels: Sequence[int], scores: Sequence[float]) -> tuple[float | None, float | None]:
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    if len(labels_array) == 0 or not np.isfinite(scores_array).all() or len(np.unique(labels_array)) < 2:
        return None, None
    if _sk_roc_auc is not None and _sk_average_precision is not None:
        return float(_sk_roc_auc(labels_array, scores_array)), float(_sk_average_precision(labels_array, scores_array))
    return _roc_auc(labels_array, scores_array), _average_precision(labels_array, scores_array)


def _summary_stats(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"n": 0, "mean": None, "median": None, "q25": None, "q75": None, "min": None, "max": None}
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q25": float(np.percentile(array, 25.0)),
        "q75": float(np.percentile(array, 75.0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _finite_rows(rows: Sequence[Mapping[str, Any]], method: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if bool(row.get("score_valid")) and np.isfinite(float(row[method]))]


def point_metric_summary(rows: Sequence[Mapping[str, Any]], *, bootstrap: dict[str, Any] | None = None) -> dict[str, Any]:
    """Summarize decisive rows, with harmful-old-source as the positive class."""

    result: dict[str, Any] = {
        "n_decisive": int(len(rows)),
        "n_harmful_old_source": int(sum(int(row["z"]) for row in rows)),
        "n_beneficial_or_other": int(sum(int(row["z"] == 0) for row in rows)),
        "harmful_prevalence": (float(np.mean([int(row["z"]) for row in rows])) if rows else None),
        "metrics": {},
    }
    for method in METHODS:
        values = _finite_rows(rows, method)
        labels = [int(row["z"]) for row in values]
        scores = [float(row[method]) for row in values]
        auc, ap = metric_values(labels, scores)
        result["metrics"][method] = {
            "n_score_valid": int(len(values)),
            "n_score_missing": int(len(rows) - len(values)),
            "auroc": auc,
            "average_precision": ap,
        }
    if bootstrap is not None:
        result["bootstrap"] = bootstrap
    return result


def _cluster_rows(rows: Sequence[Mapping[str, Any]], cluster_kind: str) -> list[np.ndarray]:
    groups: dict[Any, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = row["cluster_episode"] if cluster_kind == "episode" else row["task_key"]
        groups[key].append(index)
    return [np.asarray(indexes, dtype=np.int64) for _, indexes in sorted(groups.items(), key=lambda pair: str(pair[0]))]


def _bootstrap_cluster_indices(clusters: Sequence[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Sample clusters with replacement and expand every sampled cluster.

    There is intentionally no ``unique``/deduplication step: repeated sampled
    clusters duplicate all of their intervention rows, as required by the
    frozen paired cluster bootstrap.
    """

    if not clusters:
        return np.empty(0, dtype=np.int64)
    sampled_cluster_ids = rng.integers(0, len(clusters), size=len(clusters))
    return np.concatenate([np.asarray(clusters[int(cluster_id)], dtype=np.int64) for cluster_id in sampled_cluster_ids])


def bootstrap_summary(rows: Sequence[Mapping[str, Any]], *, cluster_kind: str) -> dict[str, Any]:
    """Compute the 10,000-draw paired cluster bootstrap for one population."""

    clusters = _cluster_rows(rows, cluster_kind)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    auc_values: dict[str, list[float]] = {method: [] for method in METHODS}
    ap_values: dict[str, list[float]] = {method: [] for method in METHODS}
    diffs: list[float] = []
    degenerate_draws: dict[str, int] = {method: 0 for method in METHODS}
    paired_degenerate_draws = 0
    if not clusters:
        return {
            "cluster_kind": cluster_kind,
            "draws_requested": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "valid_draw_count": {method: 0 for method in METHODS},
            "class_degenerate_draw_count": {method: BOOTSTRAP_DRAWS for method in METHODS},
            "paired_pppr_minus_raw": {"valid_draw_count": 0, "class_degenerate_draw_count": BOOTSTRAP_DRAWS, "ci95": None},
            "ci95": {method: {"auroc": None, "average_precision": None} for method in METHODS},
        }
    labels_all = np.asarray([int(row["z"]) for row in rows], dtype=int)
    scores_all = {method: np.asarray([float(row[method]) for row in rows], dtype=float) for method in METHODS}
    for _ in range(BOOTSTRAP_DRAWS):
        sample_indices = _bootstrap_cluster_indices(clusters, rng)
        labels = labels_all[sample_indices]
        draw_auc: dict[str, float | None] = {}
        for method in METHODS:
            scores = scores_all[method][sample_indices]
            auc, ap = metric_values(labels, scores)
            draw_auc[method] = auc
            if auc is None or ap is None:
                degenerate_draws[method] += 1
            else:
                auc_values[method].append(float(auc))
                ap_values[method].append(float(ap))
        if draw_auc["raw_ppr"] is None or draw_auc["pppr"] is None:
            paired_degenerate_draws += 1
        else:
            diffs.append(float(draw_auc["pppr"] - draw_auc["raw_ppr"]))

    def ci(values: Sequence[float]) -> list[float] | None:
        return [float(x) for x in np.percentile(np.asarray(values, dtype=float), [2.5, 97.5])] if values else None

    return {
        "cluster_kind": cluster_kind,
        "draws_requested": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "n_clusters": int(len(clusters)),
        "cluster_row_counts": [int(len(cluster)) for cluster in clusters],
        "valid_draw_count": {method: int(len(auc_values[method])) for method in METHODS},
        "class_degenerate_draw_count": degenerate_draws,
        "ci95": {
            method: {"auroc": ci(auc_values[method]), "average_precision": ci(ap_values[method])}
            for method in METHODS
        },
        "paired_pppr_minus_raw": {
            "valid_draw_count": int(len(diffs)),
            "class_degenerate_draw_count": int(paired_degenerate_draws),
            "ci95": ci(diffs),
        },
    }


def _rows_for_population(
    rows: Sequence[Mapping[str, Any]], *, split: str | None = None, group: str | None = None, condition: str | None = None
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if (split is None or row["split"] == split)
        and (group is None or row["intervention_group"] == group)
        and (condition is None or row["condition"] == condition)
        and bool(row["decisive"])
        and bool(row["score_valid"])
    ]


def make_metrics(rows: Sequence[Mapping[str, Any]], *, with_bootstrap: bool = True) -> dict[str, Any]:
    result = point_metric_summary(rows)
    if with_bootstrap and rows:
        result["bootstrap_episode_cluster"] = bootstrap_summary(rows, cluster_kind="episode")
        result["bootstrap_task_cluster"] = bootstrap_summary(rows, cluster_kind="task")
    return result


def make_taskwise(rows: Sequence[Mapping[str, Any]], split_by_task: Mapping[str, str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split in ("development", "held_out", "all_data"):
        tasks = list(split_by_task) if split == "all_data" else [key for key, value in split_by_task.items() if value == split]
        for task_key in tasks:
            for condition in CONDITION_SPECS:
                population = [
                    row for row in rows
                    if row["task_key"] == task_key and row["condition"] == condition
                    and bool(row["decisive"]) and bool(row["score_valid"])
                ]
                record: dict[str, Any] = {
                    "split": split,
                    "task_key": task_key,
                    "condition": condition,
                    "intervention_group": CONDITION_SPECS[condition]["group"],
                    "n_decisive": int(len(population)),
                    "n_harmful_old_source": int(sum(int(row["z"]) for row in population)),
                    "harmful_prevalence": (float(np.mean([int(row["z"]) for row in population])) if population else None),
                    "directions": {},
                }
                for method in METHODS:
                    harmful = [float(row[method]) for row in population if int(row["z"]) == 1]
                    beneficial = [float(row[method]) for row in population if int(row["z"]) == 0]
                    labels = [int(row["z"]) for row in population]
                    scores = [float(row[method]) for row in population]
                    auc, _ = metric_values(labels, scores)
                    record["directions"][method] = {
                        "harmful_mean": (float(np.mean(harmful)) if harmful else None),
                        "harmful_median": (float(np.median(harmful)) if harmful else None),
                        "beneficial_mean": (float(np.mean(beneficial)) if beneficial else None),
                        "beneficial_median": (float(np.median(beneficial)) if beneficial else None),
                        "auroc": auc,
                    }
                output.append(record)
    return output


def make_task_pooled(rows: Sequence[Mapping[str, Any]], split_by_task: Mapping[str, str]) -> list[dict[str, Any]]:
    """Pool all nine component-matched conditions within each task."""

    output: list[dict[str, Any]] = []
    for split in ("development", "held_out", "all_data"):
        tasks = list(split_by_task) if split == "all_data" else [key for key, value in split_by_task.items() if value == split]
        for task_key in tasks:
            population = [
                row for row in rows
                if row["task_key"] == task_key and bool(row["decisive"]) and bool(row["score_valid"])
            ]
            if split != "all_data":
                population = [row for row in population if row["split"] == split]
            record: dict[str, Any] = {
                "split": split,
                "task_key": task_key,
                "n_decisive": int(len(population)),
                "n_harmful_old_source": int(sum(int(row["z"]) for row in population)),
                "harmful_prevalence": (float(np.mean([int(row["z"]) for row in population])) if population else None),
                "directions": {},
            }
            for method in METHODS:
                harmful = [float(row[method]) for row in population if int(row["z"]) == 1]
                beneficial = [float(row[method]) for row in population if int(row["z"]) == 0]
                auc, _ = metric_values([int(row["z"]) for row in population], [float(row[method]) for row in population])
                record["directions"][method] = {
                    "harmful_mean": (float(np.mean(harmful)) if harmful else None),
                    "harmful_median": (float(np.median(harmful)) if harmful else None),
                    "beneficial_mean": (float(np.mean(beneficial)) if beneficial else None),
                    "beneficial_median": (float(np.median(beneficial)) if beneficial else None),
                    "auroc": auc,
                }
            output.append(record)
    return output


def make_distributions(rows: Sequence[Mapping[str, Any]], features: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Descriptive Raw-vs-PPPR distributions for valid rows and episode scores."""

    result: dict[str, Any] = {"feature_rows": {}, "episode_condition_scores": {}}
    # Feature-table distributions use all valid candidate rows at the three
    # intervention ages and are explicitly descriptive, not independent-
    # sample inference.
    for split in (*SPLITS, "all_data"):
        task_mask = np.ones(len(features["task_key"]), dtype=bool) if split == "all_data" else (features["split"] == split)
        valid = np.asarray(features["valid"], dtype=bool) & task_mask & np.isin(features["age_steps"], [4, 8, 16])
        result["feature_rows"][split] = {}
        for group in COMPONENT_GROUPS:
            if group == "full_old_joint":
                raw_field, pppr_field = "raw_ppr_joint", "pppr_joint"
            elif group == "reverse_arm":
                raw_field, pppr_field = "raw_ppr_arm", "pppr_arm"
            else:
                raw_field, pppr_field = "raw_ppr_grip", "pppr_grip"
            result["feature_rows"][split][group] = {
                "raw_ppr": _summary_stats(features[raw_field][valid]),
                "pppr": _summary_stats(features[pppr_field][valid]),
                "event": _summary_stats(features["event_event_score"][valid]),
                "age_steps": _summary_stats(features["age_steps"][valid]),
                "note": "Descriptive valid feature rows; candidate rows are not treated as independent inferential samples.",
            }
    score_rows = [row for row in rows if bool(row["score_valid"])]
    for split in (*SPLITS, "all_data"):
        selected = score_rows if split == "all_data" else [row for row in score_rows if row["split"] == split]
        result["episode_condition_scores"][split] = {}
        for group in COMPONENT_GROUPS:
            group_rows = [row for row in selected if row["intervention_group"] == group]
            result["episode_condition_scores"][split][group] = {
                method: _summary_stats(float(row[method]) for row in group_rows) for method in ("raw_ppr", "pppr", "event")
            }
            result["episode_condition_scores"][split][group]["age_steps"] = _summary_stats(
                float(row["age_steps"]) for row in group_rows
            )
            result["episode_condition_scores"][split][group]["n_episode_condition_rows"] = int(len(group_rows))
            result["episode_condition_scores"][split][group]["note"] = (
                "Descriptive episode-condition summaries; rows are paired and not treated as independent inferential samples."
            )
    return result


def make_gate(metrics_by_split: Mapping[str, Any], metrics_by_component: Mapping[str, Any]) -> dict[str, Any]:
    heldout = metrics_by_split["held_out"]
    pppr_auc = heldout["metrics"]["pppr"]["auroc"]
    raw_auc = heldout["metrics"]["raw_ppr"]["auroc"]
    improvement = (float(pppr_auc - raw_auc) if pppr_auc is not None and raw_auc is not None else None)
    component_checks: dict[str, Any] = {}
    for group in COMPONENT_GROUPS:
        group_metrics = metrics_by_component["held_out"][group]
        pppr_group_auc = group_metrics["metrics"]["pppr"]["auroc"]
        raw_group_auc = group_metrics["metrics"]["raw_ppr"]["auroc"]
        component_checks[group] = {
            "pppr_auroc": pppr_group_auc,
            "raw_ppr_auroc": raw_group_auc,
            "pppr_minus_raw_auroc": (
                float(pppr_group_auc - raw_group_auc)
                if pppr_group_auc is not None and raw_group_auc is not None
                else None
            ),
        }
    # The protocol gives a guide rather than a tuned statistical cutoff.  The
    # conservative operationalization below makes a positive recommendation
    # only when the point estimate reaches roughly .65, with either a roughly
    # .05 gain or similarly strong (>=.65) matched rank separation.  A
    # component below chance is reported as a suite reversal.
    pppr_threshold = 0.65
    improvement_threshold = 0.05
    pppr_reaches_threshold = pppr_auc is not None and float(pppr_auc) >= pppr_threshold
    improvement_reaches_threshold = improvement is not None and improvement >= improvement_threshold
    similarly_strong = bool(
        raw_auc is not None
        and pppr_auc is not None
        and float(raw_auc) >= pppr_threshold
        and float(pppr_auc) >= pppr_threshold
        and all(
            check["pppr_auroc"] is not None and float(check["pppr_auroc"]) >= 0.5
            for check in component_checks.values()
        )
    )
    catastrophic_reversal = any(
        check["pppr_auroc"] is None or float(check["pppr_auroc"]) < 0.5
        for check in component_checks.values()
    )
    passed = bool(pppr_reaches_threshold and (improvement_reaches_threshold or similarly_strong) and not catastrophic_reversal)
    if pppr_auc is None or raw_auc is None:
        recommendation = "FAIL"
        rationale = "Held-out combined/component-matched AUROC is not estimable because one or more classes are absent."
    elif passed:
        recommendation = "PASS"
        rationale = (
            f"Held-out PPPR AUROC={float(pppr_auc):.3f} reaches the ~0.65 guide and "
            f"PPPR-minus-Raw AUROC={float(improvement):+.3f} satisfies the gain/strong-separation guide, "
            "with no component below chance."
        )
    else:
        recommendation = "FAIL"
        rationale = (
            f"Held-out PPPR AUROC={float(pppr_auc):.3f} and RawPPR AUROC={float(raw_auc):.3f}; "
            f"PPPR-minus-Raw AUROC={float(improvement):+.3f}. The frozen guide is not met without forcing a positive conclusion."
        )
    return {
        "population": "held_out_combined_component_matched",
        "recommendation": recommendation,
        "rationale": rationale,
        "held_out_pppr_auroc": pppr_auc,
        "held_out_raw_ppr_auroc": raw_auc,
        "held_out_pppr_minus_raw_auroc": improvement,
        "operational_guide": {
            "pppr_auroc_rough_threshold": pppr_threshold,
            "improvement_rough_threshold": improvement_threshold,
            "similarly_strong_requires_raw_and_pppr_at_least": pppr_threshold,
            "catastrophic_suite_reversal": "any component-matched held-out PPPR AUROC below 0.50",
        },
        "checks": {
            "pppr_reaches_rough_threshold": pppr_reaches_threshold,
            "improvement_reaches_rough_threshold": improvement_reaches_threshold,
            "similarly_strong_consistent_rank_separation": similarly_strong,
            "catastrophic_suite_reversal": catastrophic_reversal,
            "component_mapped": component_checks,
        },
    }


def write_pairs_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "task_key", "split", "episode", "condition", "intervention_group", "component", "age_steps",
        "fresh_success", "intervention_success", "delta_y", "z", "decisive", "score_valid",
        "active_logged_steps", "valid_feature_steps", "missing_feature_steps", "active_steps", "valid_steps",
        "missing_steps", "raw_ppr", "pppr", "event", "age",
    ]
    lines: list[str] = []
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            output = {field: row.get(field) for field in fields}
            for field in ("active_steps", "valid_steps", "missing_steps"):
                output[field] = ";".join(str(x) for x in row[field])
            writer.writerow(output)
    os.replace(temporary, path)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"[{_fmt(value[0], digits)}, {_fmt(value[1], digits)}]"
    return f"{float(value):.{digits}f}"


def _metric_table(metrics: Mapping[str, Any]) -> str:
    lines = ["| signal | n | harmful | prevalence | AUROC | AUPRC |", "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        item = metrics["metrics"][method]
        lines.append(
            f"| {method} | {item['n_score_valid']} | {metrics['n_harmful_old_source']} | "
            f"{_fmt(metrics['harmful_prevalence'])} | {_fmt(item['auroc'])} | {_fmt(item['average_precision'])} |"
        )
    return "\n".join(lines)


def _ci_text(bootstrap: Mapping[str, Any], method: str, metric: str) -> str:
    ci = bootstrap["ci95"][method][metric]
    if ci is None:
        return "NA"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def make_markdown(result: Mapping[str, Any]) -> str:
    runtime = result["run"]
    lines = [
        "# Phase-0 control relevance",
        "",
        "This report uses only the frozen `phase0_features.npz` table and paired intervention outcomes/source-event logs. "
        "For each active source event at physical step `u`, the score is read from the Fresh row "
        "`(task, episode, old_query_t=u-d, age_steps=d)`; intervention-trajectory predictions are never used.",
        "",
        f"- Exact command: `{runtime['command']}`",
        f"- Started (UTC): `{runtime['started_at_utc']}`",
        f"- Finished (UTC): `{runtime['finished_at_utc']}`",
        f"- Runtime seconds: `{runtime['runtime_seconds']:.3f}`",
        f"- Pair rows: `{result['counts']['pair_rows']}`; decisive rows: `{result['counts']['decisive_rows']}`",
        f"- Source features: `{result['inputs']['features']}`",
        f"- Source outcomes/logs: `{result['inputs']['pilot_results']}`",
        "",
        "## Pairing and alignment",
        "",
        "`DeltaY = Y_fresh - Y_intervention`; decisive rows have `DeltaY != 0`, and `Z=1` means harmful old source (`DeltaY=+1`). "
        "FullOld uses joint scores, Reverse uses arm scores, and FO uses gripper scores at ages 4, 8, and 16. "
        "Episode-condition scores are arithmetic means over valid active logged steps. Warm-up age-0 steps are excluded. "
        "Rows with no valid Fresh feature are not filled.",
        "",
        f"- Active logged steps: `{result['counts']['active_logged_steps']}`; valid aligned steps: `{result['counts']['valid_aligned_steps']}`; "
        f"missing/invalid Fresh rows: `{result['counts']['missing_aligned_steps']}`.",
        "",
        "## Primary metrics on decisive pairs",
        "",
    ]
    for split in ("development", "held_out", "all_data"):
        label = "held-out" if split == "held_out" else split
        scope = "descriptive only" if split == "all_data" else "primary"
        lines += [f"### {label} ({scope})", "", _metric_table(result["metrics_by_split"][split]), ""]
        if split != "all_data":
            boot = result["metrics_by_split"][split]["bootstrap_episode_cluster"]
            lines += [
                "Episode-cluster bootstrap (10,000 paired draws; percentile 95% CI): "
                + "; ".join(
                    f"{method} AUROC {_ci_text(boot, method, 'auroc')}, AUPRC {_ci_text(boot, method, 'average_precision')}"
                    for method in METHODS
                )
                + ".",
                f"Valid AUROC draws: {boot['valid_draw_count']}; class-degenerate draws: {boot['class_degenerate_draw_count']}; "
                f"PPPR-minus-Raw AUROC CI: {_fmt(boot['paired_pppr_minus_raw']['ci95']) if boot['paired_pppr_minus_raw']['ci95'] is not None else 'NA'}.",
                "",
            ]
            task_boot = result["metrics_by_split"][split]["bootstrap_task_cluster"]
            lines += [
                "Task-cluster bootstrap (10,000 paired draws; percentile 95% CI): "
                + "; ".join(
                    f"{method} AUROC {_ci_text(task_boot, method, 'auroc')}, AUPRC {_ci_text(task_boot, method, 'average_precision')}"
                    for method in METHODS
                )
                + ".",
                f"Valid AUROC draws: {task_boot['valid_draw_count']}; class-degenerate draws: {task_boot['class_degenerate_draw_count']}; "
                f"PPPR-minus-Raw AUROC CI: {_fmt(task_boot['paired_pppr_minus_raw']['ci95']) if task_boot['paired_pppr_minus_raw']['ci95'] is not None else 'NA'}.",
                "",
            ]
    lines += ["## Held-out component-matched metrics", "", "| component-matched population | n | harmful | PPPR AUROC | RawPPR AUROC | PPPR−Raw |", "|---|---:|---:|---:|---:|---:|"]
    for group in COMPONENT_GROUPS:
        metric = result["metrics_by_component"]["held_out"][group]
        pppr = metric["metrics"]["pppr"]["auroc"]
        raw = metric["metrics"]["raw_ppr"]["auroc"]
        lines.append(f"| {group} | {metric['n_decisive']} | {metric['n_harmful_old_source']} | {_fmt(pppr)} | {_fmt(raw)} | {_fmt(pppr - raw if pppr is not None and raw is not None else None)} |")
    lines += ["", "## Task-wise direction", "", "For each task/condition, the table reports harmful-old-source and beneficial/other score means/medians, plus AUROC only when both classes are present.", ""]
    lines += ["| split | task | condition | n | harmful | PPPR harmful median | PPPR beneficial median | PPPR AUROC |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for record in result["taskwise"]:
        if record["split"] == "all_data":
            continue
        direction = record["directions"]["pppr"]
        lines.append(
            f"| {record['split']} | {record['task_key']} | {record['condition']} | {record['n_decisive']} | "
            f"{record['n_harmful_old_source']} | {_fmt(direction['harmful_median'])} | {_fmt(direction['beneficial_median'])} | {_fmt(direction['auroc'])} |"
        )
    lines += ["", "### Held-out task-pooled PPPR direction", "", "Each row pools the nine component-matched conditions (three ages for FullOld, Reverse, and FO) within one held-out task.", "", "| task | n | harmful | PPPR harmful mean/median | PPPR beneficial mean/median | PPPR AUROC |", "|---|---:|---:|---:|---:|---:|"]
    for record in result["task_pooled"]:
        if record["split"] != "held_out":
            continue
        direction = record["directions"]["pppr"]
        lines.append(
            f"| {record['task_key']} | {record['n_decisive']} | {record['n_harmful_old_source']} | "
            f"{_fmt(direction['harmful_mean'])}/{_fmt(direction['harmful_median'])} | "
            f"{_fmt(direction['beneficial_mean'])}/{_fmt(direction['beneficial_median'])} | {_fmt(direction['auroc'])} |"
        )
    lines += ["", "## RawPPR versus PPPR distributions", "", "These summaries are descriptive. Candidate feature rows and episode-condition rows are not treated as independent inferential samples.", ""]
    lines += ["| split | population | n | RawPPR median [q25,q75] | PPPR median [q25,q75] |", "|---|---|---:|---:|---:|"]
    for split in (*SPLITS, "all_data"):
        for kind, source in (("valid feature rows", result["distributions"]["feature_rows"]), ("episode-condition scores", result["distributions"]["episode_condition_scores"])):
            for group in COMPONENT_GROUPS:
                item = source[split][group]
                if kind == "episode-condition scores":
                    n = item["n_episode_condition_rows"]
                else:
                    n = item["raw_ppr"]["n"]
                raw = item["raw_ppr"]
                pppr = item["pppr"]
                lines.append(f"| {split} | {kind}: {group} | {n} | {_fmt(raw['median'])} [{_fmt(raw['q25'])}, {_fmt(raw['q75'])}] | {_fmt(pppr['median'])} [{_fmt(pppr['q25'])}, {_fmt(pppr['q75'])}] |")
    gate = result["gate"]
    lines += ["", "## Gate recommendation", "", f"**{gate['recommendation']}** for the held-out combined/component-matched gate.", "", gate["rationale"], "", "The gate guide is held-out PPPR AUROC roughly ≥0.65, improvement over RawPPR roughly ≥0.05 or similarly strong consistent rank separation, and no catastrophic suite reversal. The operational checks and all component values are recorded in the JSON output.", ""]
    return "\n".join(lines).rstrip() + "\n"


def analyze(
    *,
    features_path: Path = DEFAULT_FEATURES,
    protocol_path: Path = DEFAULT_PROTOCOL,
    rollout_protocol_path: Path = DEFAULT_ROLLOUT_PROTOCOL,
    pilot_path: Path = DEFAULT_PILOT,
    json_path: Path = DEFAULT_JSON,
    markdown_path: Path = DEFAULT_MARKDOWN,
    pairs_path: Path = DEFAULT_PAIRS,
    marker_path: Path = DEFAULT_MARKER,
    force: bool = False,
) -> dict[str, Any]:
    started_monotonic = time.monotonic()
    started_at = _utc_now()
    command = shlex.join([sys.executable, *sys.argv])
    if marker_path.exists() and not force:
        if json_path.exists():
            return json.loads(json_path.read_text(encoding="utf-8"))
        raise RuntimeError(f"completion marker exists but result JSON is absent: {marker_path}")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    rollout_protocol = json.loads(rollout_protocol_path.read_text(encoding="utf-8"))
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    split = protocol.get("split", {})
    development = tuple(str(x) for x in split.get("development", []))
    heldout = tuple(str(x) for x in split.get("heldout_offline", []))
    if not development or not heldout or set(development) & set(heldout):
        raise ValueError("frozen protocol split is malformed")
    split_by_task = {task: "development" for task in development} | {task: "held_out" for task in heldout}
    if set(split_by_task) != set(pilot.get("tasks", {})):
        raise ValueError("pilot task set differs from frozen protocol split")
    if tuple(int(x) for x in rollout_protocol.get("source_ages_steps", [])) != (4, 8, 16):
        raise ValueError("rollout source ages changed from frozen Phase-0 schedule")
    if int(protocol.get("uncertainty", {}).get("draws", BOOTSTRAP_DRAWS)) != BOOTSTRAP_DRAWS:
        raise ValueError("bootstrap draw count changed from frozen protocol")
    if int(protocol.get("uncertainty", {}).get("seed", BOOTSTRAP_SEED)) != BOOTSTRAP_SEED:
        raise ValueError("bootstrap seed changed from frozen protocol")

    features = load_features(features_path)
    rows = make_pair_rows(features=features, pilot=pilot, split_by_task=split_by_task)
    decisive_rows = [row for row in rows if bool(row["decisive"])]
    active_logged = int(sum(int(row["active_logged_steps"]) for row in rows))
    valid_aligned = int(sum(int(row["valid_feature_steps"]) for row in rows))
    missing_aligned = int(sum(int(row["missing_feature_steps"]) for row in rows))
    missing_records = [
        {
            "task_key": row["task_key"],
            "split": row["split"],
            "episode": row["episode"],
            "condition": row["condition"],
            "missing_steps": row["missing_steps"],
            "missing_key_steps": row["missing_key_steps"],
            "invalid_feature_steps": row["invalid_feature_steps"],
        }
        for row in rows if row["missing_steps"]
    ]

    metrics_by_split: dict[str, Any] = {}
    for split_name in (*SPLITS, "all_data"):
        selected = [row for row in decisive_rows if split_name == "all_data" or row["split"] == split_name]
        selected_valid = [row for row in selected if bool(row["score_valid"])]
        metrics_by_split[split_name] = make_metrics(selected_valid, with_bootstrap=(split_name != "all_data"))

    metrics_by_component: dict[str, dict[str, Any]] = {}
    for split_name in (*SPLITS, "all_data"):
        metrics_by_component[split_name] = {}
        for group in COMPONENT_GROUPS:
            selected = [
                row for row in decisive_rows
                if row["intervention_group"] == group and (split_name == "all_data" or row["split"] == split_name)
                and bool(row["score_valid"])
            ]
            metrics_by_component[split_name][group] = make_metrics(selected, with_bootstrap=(split_name != "all_data"))

    metrics_by_condition: dict[str, dict[str, Any]] = {}
    for split_name in (*SPLITS, "all_data"):
        metrics_by_condition[split_name] = {}
        for condition in CONDITION_SPECS:
            selected = [
                row for row in decisive_rows
                if row["condition"] == condition and (split_name == "all_data" or row["split"] == split_name)
                and bool(row["score_valid"])
            ]
            metrics_by_condition[split_name][condition] = make_metrics(selected, with_bootstrap=(split_name != "all_data"))

    result = {
        "schema_version": 1,
        "protocol": {
            "path": str(protocol_path.resolve()),
            "version": protocol.get("protocol_version"),
            "primary_population": "decisive Fresh-versus-intervention pairs with valid aligned episode-condition scores",
            "label": "DeltaY = Fresh success - intervention success; Z=1 iff DeltaY=+1",
            "signals": list(METHODS),
            "condition_specs": CONDITION_SPECS,
            "bootstrap": {
                "draws": BOOTSTRAP_DRAWS,
                "seed": BOOTSTRAP_SEED,
                "cluster_definitions": {
                    "episode": "(task, episode/state index), retaining every selected intervention row",
                    "task": "task, retaining every selected row in the task",
                },
                "paired_difference": "PPPR minus RawPPR AUROC within identical sampled indices",
                "ci": "percentile 95%",
            },
        },
        "inputs": {
            "features": str(features_path.resolve()),
            "pilot_results": str(pilot_path.resolve()),
            "rollout_protocol": str(rollout_protocol_path.resolve()),
        },
        "run": {
            "command": command,
            "started_at_utc": started_at,
            "finished_at_utc": None,
            "runtime_seconds": None,
        },
        "counts": {
            "task_count": int(len(split_by_task)),
            "development_task_count": int(len(development)),
            "held_out_task_count": int(len(heldout)),
            "pair_rows": int(len(rows)),
            "decisive_rows": int(len(decisive_rows)),
            "decisive_rows_by_split": {
                split_name: int(sum(int(row["decisive"]) for row in rows if split_name == "all_data" or row["split"] == split_name))
                for split_name in (*SPLITS, "all_data")
            },
            "harmful_decisive_rows_by_split": {
                split_name: int(sum(int(row["decisive"] and row["z"]) for row in rows if split_name == "all_data" or row["split"] == split_name))
                for split_name in (*SPLITS, "all_data")
            },
            "active_logged_steps": active_logged,
            "valid_aligned_steps": valid_aligned,
            "missing_aligned_steps": missing_aligned,
            "missing_alignment_records": int(len(missing_records)),
            "missing_alignment_details": missing_records,
        },
        "metrics_by_split": metrics_by_split,
        "metrics_by_component": metrics_by_component,
        "metrics_by_condition": metrics_by_condition,
        "taskwise": make_taskwise(rows, split_by_task),
        "task_pooled": make_task_pooled(rows, split_by_task),
        "distributions": make_distributions(rows, features),
    }
    result["gate"] = make_gate(metrics_by_split, metrics_by_component)
    result["run"]["finished_at_utc"] = _utc_now()
    result["run"]["runtime_seconds"] = float(time.monotonic() - started_monotonic)
    write_pairs_csv(pairs_path, rows)
    _atomic_json(json_path, result)
    _atomic_text(markdown_path, make_markdown(result))
    marker_payload = {
        "completed_at_utc": result["run"]["finished_at_utc"],
        "runtime_seconds": result["run"]["runtime_seconds"],
        "command": command,
        "result_json": str(json_path.resolve()),
        "result_markdown": str(markdown_path.resolve()),
        "pairs_csv": str(pairs_path.resolve()),
    }
    _atomic_json(marker_path, marker_payload)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="regenerate outputs despite completion marker")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--rollout-protocol", type=Path, default=DEFAULT_ROLLOUT_PROTOCOL)
    parser.add_argument("--pilot-results", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--pairs-out", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--marker", type=Path, default=DEFAULT_MARKER)
    args = parser.parse_args()
    result = analyze(
        features_path=args.features,
        protocol_path=args.protocol,
        rollout_protocol_path=args.rollout_protocol,
        pilot_path=args.pilot_results,
        json_path=args.json_out,
        markdown_path=args.markdown_out,
        pairs_path=args.pairs_out,
        marker_path=args.marker,
        force=args.force,
    )
    print(json.dumps({"gate": result.get("gate"), "counts": result.get("counts", {}), "run": result.get("run", {})}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
