#!/usr/bin/env python3
"""Frozen RTX 5080 cross-generation teacher-forced composition audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from gate3a1_dense_analysis import ACTION_STD, action_sign, rotation_geodesic, target_metrics


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "research/fast5080_cross_generation_offline_protocol.md"
INVENTORY = ROOT / "research/audit_outputs/gate3a1_inventory.json"
DEFAULT_DATASET = Path("/home/wjq/datasets/libero_object_25_08_23_lerobotv2.1")
DEFAULT_CHECKPOINT = Path("/home/wjq/checkpoints/zeromidnight_act_libero_object")
DEFAULT_CACHE = ROOT / "experiments/gate3a1_dense_temporal_cache"
DEFAULT_METRICS = ROOT / "research/audit_outputs/fast5080_cross_generation_metrics.json"
DEFAULT_PER_TASK = ROOT / "research/audit_outputs/fast5080_cross_generation_per_task.csv"
DEFAULT_CONTRAST = ROOT / "research/audit_outputs/fast5080_cross_generation_contrast.csv"
DEFAULT_REPORT = ROOT / "research/fast5080_cross_generation_offline_report.md"

PROTOCOL_COMMIT = "d9cac3ab69bd1a6d93608ebbd8311134483bc2ac"
PROTOCOL_SHA256 = "421c8c78921125727c7145941498a31eb8e4c561d1f5f802280427d8a962ca97"
GATE3A1_ANALYSIS_SHA256 = "af6a211db60cf883c3d33c69637ef275b3012c17334855bf2477367041ae2b76"
INVENTORY_SHA256 = "9f0da4d0779fbc37f59258714047f9297efbcfefcdd26cc1067c76975ba691be"
CHECKPOINT_SHA256 = "340071d7497238669459d93517eb3f8690862ad6fdf14207966759dfe6da9410"
CONFIG_SHA256 = "a76eebed357b3cbed8745c3d0f18c1335ecdd5449fcc498257676c9cbd27453d"
DATASET_REVISION = "cbf7122bbdbaa0c50517a6a4b2ae663d0e96e51a"
DATASET_PAYLOAD_SHA256 = "7c5cb7e88722e0aead2fe0853bdf54e076afe77364a3204ecf46f1e5e7a05b7b"
EPISODES_SHA256 = "63c6fb6940f46d0bc74c0242c1cde2a39a945bbe7de7b1709d38f5d9a82fcfea"
EPISODE_STATS_SHA256 = "5bf31fb80b359c9fd1d56a0eaa27f8e7c76a7e39678487fdf76986af8fe88dca"
CACHE_TREE_SHA256 = "7e14e1f341bc2425cb3304cc3f35b0075184b0b1f33225e2dcf05cfe67e50f65"
AGE = 20
EXPECTED_EPISODES = 82
EXPECTED_TARGETS = 10654
EXPECTED_QUERIES = 12294
BOOTSTRAP_DRAWS = 20000
EPISODE_BOOTSTRAP_SEED = 20260826
TASK_BOOTSTRAP_SEED = 20260827
ZERO_TOLERANCE = 1e-12
CONDITIONS = ("FF", "OO", "FO", "OF")
METRICS = (
    "dimension_weighted_semantic_error",
    "translation_normalized_mse",
    "rotation_normalized_sq",
    "rotation_geodesic_radians",
    "gripper_sign_error",
)
ALIASES = {
    "dimension_weighted_semantic_error": "l_sem",
    "translation_normalized_mse": "translation",
    "rotation_normalized_sq": "rotation_normalized",
    "rotation_geodesic_radians": "rotation_geodesic_radians",
    "gripper_sign_error": "gripper_sign",
}
TRANSLATION_QUARTILES = np.asarray(
    [0.09562705994409045, 0.2215019542367106, 0.4240999982124295], dtype=np.float64
)
ROTATION_QUARTILES = np.asarray(
    [0.01704259893576156, 0.026375850435693467, 0.03981205950930166],
    dtype=np.float64,
)
EXPECTED_GRIP_DISAGREEMENT = 3098


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("self-test", "run"), nargs="?", default="run")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--per-task-output", type=Path, default=DEFAULT_PER_TASK)
    parser.add_argument("--contrast-output", type=Path, default=DEFAULT_CONTRAST)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_tree_sha256(root: Path) -> str:
    lines: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or relative.parts[:2] == (".cache", "huggingface"):
            continue
        lines.append(f"{sha256(path)}  {relative.as_posix()}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def cache_tree(entries: list[dict[str, Any]], cache_root: Path) -> tuple[str, int]:
    lines: list[str] = []
    total_bytes = 0
    for entry in sorted(entries, key=lambda row: (str(row["split"]), int(row["episode_id"]))):
        relative = Path(str(entry["cache_file"])).resolve().relative_to(cache_root.resolve())
        lines.append(f"{entry['sha256']}  {relative.as_posix()}\n")
        total_bytes += int(entry["bytes"])
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest(), total_bytes


def verify_provenance(
    dataset: Path, checkpoint: Path, cache_root: Path, cache_manifest: Path
) -> tuple[dict[str, Any], dict[str, Any], list[tuple[str, int]], dict[str, Any]]:
    fixed_files = {
        PROTOCOL: PROTOCOL_SHA256,
        ROOT / "research/audit_tools/gate3a1_dense_analysis.py": GATE3A1_ANALYSIS_SHA256,
        INVENTORY: INVENTORY_SHA256,
        checkpoint / "model.safetensors": CHECKPOINT_SHA256,
        checkpoint / "config.json": CONFIG_SHA256,
        dataset / "meta/episodes.jsonl": EPISODES_SHA256,
        dataset / "meta/episodes_stats.jsonl": EPISODE_STATS_SHA256,
    }
    for path, expected in fixed_files.items():
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"Provenance mismatch for {path}: {observed} != {expected}")

    payload_digest = payload_tree_sha256(dataset)
    if payload_digest != DATASET_PAYLOAD_SHA256:
        raise RuntimeError(f"Dataset payload mismatch: {payload_digest}")
    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    expected_info = {"total_episodes": 454, "total_frames": 66984, "total_tasks": 10}
    if any(info.get(key) != value for key, value in expected_info.items()):
        raise RuntimeError(f"Dataset metadata mismatch: {info}")

    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    cohort = [
        (split, int(episode_id))
        for split in ("validation", "test")
        for episode_id in inventory["splits"][split]["episode_ids"]
    ]
    manifest = json.loads(cache_manifest.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    indexed = {(str(row["split"]), int(row["episode_id"])): row for row in entries}
    if len(cohort) != EXPECTED_EPISODES or set(indexed) != set(cohort):
        raise RuntimeError("Cache cohort does not equal the frozen 82-episode inventory")

    queries = 0
    for split, episode_id in cohort:
        entry = indexed[(split, episode_id)]
        path = cache_root / split / f"episode_{episode_id:06d}.npz"
        if sha256(path) != entry["sha256"] or path.stat().st_size != int(entry["bytes"]):
            raise RuntimeError(f"Cache file mismatch: {path}")
        if int(entry["completed_frames"]) != int(entry["expected_frames"]):
            raise RuntimeError(f"Incomplete cache file: {path}")
        queries += int(entry["completed_frames"])
    digest, npz_bytes = cache_tree(entries, cache_root)
    if digest != CACHE_TREE_SHA256 or queries != EXPECTED_QUERIES:
        raise RuntimeError(f"Cache scope mismatch: digest={digest}, queries={queries}")
    provenance = manifest["provenance"]
    if (
        provenance["checkpoint_files_sha256"]["model.safetensors"] != CHECKPOINT_SHA256
        or provenance["checkpoint_files_sha256"]["config.json"] != CONFIG_SHA256
        or provenance["dataset_revision"] != DATASET_REVISION
    ):
        raise RuntimeError("Cache provenance does not match frozen assets")
    return (
        inventory,
        manifest,
        cohort,
        {
            "checkpoint_model_sha256": CHECKPOINT_SHA256,
            "checkpoint_config_sha256": CONFIG_SHA256,
            "dataset_revision": DATASET_REVISION,
            "dataset_payload_tree_sha256": payload_digest,
            "cache_content_tree_sha256": digest,
            "cache_npz_files": len(entries),
            "cache_npz_bytes": npz_bytes,
            "cache_manifest_sha256": sha256(cache_manifest),
            "cache_generator_provenance": provenance,
            "runtime": manifest["runtime"],
        },
    )


def compositions(fresh: np.ndarray, old: np.ndarray) -> dict[str, np.ndarray]:
    fo = fresh.copy()
    fo[:, 6] = old[:, 6]
    of = old.copy()
    of[:, 6] = fresh[:, 6]
    return {"FF": fresh, "OO": old, "FO": fo, "OF": of}


def flatten_means(metric_arrays: dict[str, dict[str, np.ndarray]]) -> dict[str, float]:
    row: dict[str, float] = {}
    for condition in CONDITIONS:
        for metric in METRICS:
            row[f"{ALIASES[metric]}_{condition}"] = float(np.mean(metric_arrays[condition][metric]))
    sem = {condition: row[f"l_sem_{condition}"] for condition in CONDITIONS}
    row["c_offline"] = 0.5 * (sem["FO"] + sem["OF"]) - 0.5 * (sem["FF"] + sem["OO"])
    return row


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition in CONDITIONS:
        result[condition] = {}
        for metric in METRICS:
            key = f"{ALIASES[metric]}_{condition}"
            values = np.asarray([float(row[key]) for row in rows])
            result[condition][ALIASES[metric]] = {
                "episode_weighted_mean": float(np.mean(values)),
                "episode_sd": float(np.std(values, ddof=1)),
            }
    return result


def task_rows(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id in range(10):
        current = [row for row in episodes if int(row["task_id"]) == task_id]
        row: dict[str, Any] = {
            "task_id": task_id,
            "episodes": len(current),
            "targets": int(sum(int(value["targets"]) for value in current)),
        }
        for condition in CONDITIONS:
            for metric in METRICS:
                key = f"{ALIASES[metric]}_{condition}"
                row[key] = float(np.mean([float(value[key]) for value in current]))
        row["c_offline"] = float(np.mean([float(value["c_offline"]) for value in current]))
        if row["c_offline"] > ZERO_TOLERANCE:
            row["contrast_sign"] = "positive"
        elif row["c_offline"] < -ZERO_TOLERANCE:
            row["contrast_sign"] = "negative"
        else:
            row["contrast_sign"] = "zero"
        rows.append(row)
    return rows


def percentile_interval(values: np.ndarray) -> list[float]:
    return np.quantile(values, [0.025, 0.975], method="linear").astype(float).tolist()


def primary_inference(episodes: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    episode_values = np.asarray([float(row["c_offline"]) for row in episodes])
    episode_rng = np.random.default_rng(EPISODE_BOOTSTRAP_SEED)
    episode_draws = np.mean(
        episode_values[episode_rng.integers(0, len(episode_values), (BOOTSTRAP_DRAWS, len(episode_values)))],
        axis=1,
    )
    task_values = np.asarray([float(row["c_offline"]) for row in tasks])
    task_rng = np.random.default_rng(TASK_BOOTSTRAP_SEED)
    task_draws = np.mean(
        task_values[task_rng.integers(0, len(task_values), (BOOTSTRAP_DRAWS, len(task_values)))],
        axis=1,
    )
    loto = [
        {
            "omitted_task": int(task_id),
            "mean_remaining_nine_tasks": float(np.mean(np.delete(task_values, task_id))),
        }
        for task_id in range(10)
    ]
    return {
        "definition": "0.5*(L_FO+L_OF)-0.5*(L_FF+L_OO); lower loss is better",
        "episode_weighted_mean": float(np.mean(episode_values)),
        "episode_bootstrap_ci95": percentile_interval(episode_draws),
        "episode_bootstrap_draws": BOOTSTRAP_DRAWS,
        "episode_bootstrap_seed": EPISODE_BOOTSTRAP_SEED,
        "task_weighted_mean": float(np.mean(task_values)),
        "task_cluster_bootstrap_ci95": percentile_interval(task_draws),
        "task_cluster_bootstrap_draws": BOOTSTRAP_DRAWS,
        "task_cluster_bootstrap_seed": TASK_BOOTSTRAP_SEED,
        "leave_one_task_out": loto,
        "zero_tolerance": ZERO_TOLERANCE,
        "task_sign_counts": {
            sign: sum(row["contrast_sign"] == sign for row in tasks)
            for sign in ("positive", "zero", "negative")
        },
    }


def stratum_summary(mask: np.ndarray, arrays: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    means = {
        condition: {
            ALIASES[metric]: float(np.mean(arrays[condition][metric][mask]))
            for metric in METRICS
        }
        for condition in CONDITIONS
    }
    sem = {condition: means[condition]["l_sem"] for condition in CONDITIONS}
    contrast = 0.5 * (sem["FO"] + sem["OF"]) - 0.5 * (sem["FF"] + sem["OO"])
    return {"targets": int(np.sum(mask)), "conditions": means, "c_offline": float(contrast)}


def quartile_masks(values: np.ndarray, cutpoints: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [
        ("Q1", values <= cutpoints[0]),
        ("Q2", (values > cutpoints[0]) & (values <= cutpoints[1])),
        ("Q3", (values > cutpoints[1]) & (values <= cutpoints[2])),
        ("Q4", values > cutpoints[2]),
    ]


def render_report(result: dict[str, Any], per_task: list[dict[str, Any]]) -> str:
    conditions = result["condition_summaries"]
    contrast = result["primary_contrast"]
    lines = [
        "# RTX 5080 cross-generation offline composition report",
        "",
        "Audit date: 2026-08-24 (Asia/Shanghai). This is the frozen post-Gate-3A2",
        "exploratory offline audit defined before composition losses were computed.",
        "No Thor Gate-3B outcome was inspected or used.",
        "",
        "## Result",
        "",
        "| Condition | `L_sem` | translation | rotation normalized | rotation rad | gripper sign |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        current = conditions[condition]
        lines.append(
            f"| {condition} | {current['l_sem']['episode_weighted_mean']:.10f} | "
            f"{current['translation']['episode_weighted_mean']:.10f} | "
            f"{current['rotation_normalized']['episode_weighted_mean']:.10f} | "
            f"{current['rotation_geodesic_radians']['episode_weighted_mean']:.10f} | "
            f"{current['gripper_sign']['episode_weighted_mean']:.10f} |"
        )
    lines.extend(
        [
            "",
            f"Across {result['cohort']['episodes']} episodes and {result['cohort']['eligible_targets']} "
            f"eligible targets, mean `C_offline` is `{contrast['episode_weighted_mean']:.17g}`. "
            f"The paired episode-bootstrap 95% CI is "
            f"`[{contrast['episode_bootstrap_ci95'][0]:.17g}, "
            f"{contrast['episode_bootstrap_ci95'][1]:.17g}]`; the macro-task estimate is "
            f"`{contrast['task_weighted_mean']:.17g}` with task-cluster 95% CI "
            f"`[{contrast['task_cluster_bootstrap_ci95'][0]:.17g}, "
            f"{contrast['task_cluster_bootstrap_ci95'][1]:.17g}]`.",
            "",
            "The targetwise 2x2 identity residual is at most "
            f"`{result['identity_check']['maximum_absolute_target_residual']:.3g}`, below the "
            f"frozen `{ZERO_TOLERANCE:g}` tolerance. The result is therefore an exact "
            "offline null by construction: the additive Gate-3A1 metric has no arm-gripper "
            "interaction term with which to score cross-generation coherence. This says "
            "nothing by itself about closed-loop harm or benefit.",
            "",
            "## Task sensitivity",
            "",
            "| Task | Episodes | Targets | `C_offline` | Sign |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for row in per_task:
        lines.append(
            f"| {row['task_id']} | {row['episodes']} | {row['targets']} | "
            f"{float(row['c_offline']):.17g} | {row['contrast_sign']} |"
        )
    loto = [float(row["mean_remaining_nine_tasks"]) for row in contrast["leave_one_task_out"]]
    lines.extend(
        [
            "",
            f"All ten task contrasts are classified zero at tolerance `{ZERO_TOLERANCE:g}`. "
            f"Leave-one-task-out macro means range from `{min(loto):.3g}` to `{max(loto):.3g}`.",
            "",
            "## Components and disagreement diagnostics",
            "",
            "FO inherits FF's translation and rotation losses and OO's gripper-sign loss; "
            "OF inherits the converse. Individual component means can therefore change "
            "between mixed and coherent conditions, but their symmetric additive contrast "
            "cancels. The frozen translation and rotation quartiles and the gripper same/different "
            "strata likewise retain the targetwise zero contrast; they are descriptive only.",
            "",
            "## Provenance and limits",
            "",
            f"The independent RTX cache contains {result['assets']['cache_npz_files']} NPZ files "
            f"with content-tree SHA256 `{result['assets']['cache_content_tree_sha256']}`. "
            f"The frozen ACT model SHA256 is `{CHECKPOINT_SHA256}`. Gate-3A1's important "
            "RTX ordering reproduced before this audit. The complete machine-readable results "
            "and figure-ready data are in the adjacent JSON and CSV outputs.",
            "",
            "This audit evaluates one frozen ACT checkpoint, one demonstration corpus, one age, "
            "and one separable teacher-forced metric. It neither measures task success nor "
            "establishes a causal mechanism. Interpretation with Gate-3B must wait for the "
            "independent closed-loop result.",
            "",
        ]
    )
    return "\n".join(lines)


def self_test() -> None:
    rng = np.random.default_rng(7)
    fresh = rng.normal(size=(32, 7))
    old = rng.normal(size=(32, 7))
    target = rng.normal(size=(32, 7))
    values = {name: target_metrics(action, target) for name, action in compositions(fresh, old).items()}
    residual = values["FO"]["dimension_weighted_semantic_error"] + values["OF"][
        "dimension_weighted_semantic_error"
    ] - values["FF"]["dimension_weighted_semantic_error"] - values["OO"][
        "dimension_weighted_semantic_error"
    ]
    if float(np.max(np.abs(residual))) > ZERO_TOLERANCE:
        raise RuntimeError("Synthetic additive identity self-test failed")
    chunks = np.arange(30 * 100 * 7, dtype=np.float64).reshape(30, 100, 7)
    if not np.array_equal(chunks[20:, 0], chunks[AGE:, 0]) or not np.array_equal(
        chunks[:-AGE, AGE], chunks[np.arange(10), AGE]
    ):
        raise RuntimeError("Source-index self-test failed")
    print(json.dumps({"self_test": "passed", "maximum_absolute_identity_residual": float(np.max(np.abs(residual)))}, indent=2))


def run(args: argparse.Namespace) -> None:
    dataset = args.dataset.resolve()
    checkpoint = args.checkpoint.resolve()
    cache_root = args.cache_root.resolve()
    cache_manifest = (
        args.cache_manifest.resolve()
        if args.cache_manifest is not None
        else cache_root / "manifest.json"
    )
    inventory, manifest, cohort, assets = verify_provenance(
        dataset, checkpoint, cache_root, cache_manifest
    )

    episode_rows: list[dict[str, Any]] = []
    target_metrics_by_condition: dict[str, dict[str, list[np.ndarray]]] = {
        condition: {metric: [] for metric in METRICS} for condition in CONDITIONS
    }
    trans_disagreement: list[np.ndarray] = []
    rotation_disagreement: list[np.ndarray] = []
    grip_disagreement: list[np.ndarray] = []
    identity_residuals: list[np.ndarray] = []

    for split, episode_id in cohort:
        path = cache_root / split / f"episode_{episode_id:06d}.npz"
        with np.load(path, allow_pickle=False) as cache:
            chunks = cache["predicted_chunks"].astype(np.float64)
            frames = cache["dataset_frame"].astype(int)
            task_id = int(cache["task_id"].item())
        table = pq.read_table(
            dataset / "data/chunk-000" / f"episode_{episode_id:06d}.parquet",
            columns=["action", "frame_index", "task_index", "episode_index"],
        )
        target = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        dataset_frames = np.asarray(table["frame_index"].to_pylist(), dtype=int)
        if (
            chunks.shape != (len(target), 100, 7)
            or not np.array_equal(frames, dataset_frames)
            or set(table["task_index"].to_pylist()) != {task_id}
            or set(table["episode_index"].to_pylist()) != {episode_id}
        ):
            raise RuntimeError(f"Dataset/cache alignment failure for episode {episode_id}")

        fresh = chunks[AGE:, 0]
        old = chunks[:-AGE, AGE]
        eligible_target = target[AGE:]
        constructed = compositions(fresh, old)
        current_metrics = {
            condition: target_metrics(action, eligible_target)
            for condition, action in constructed.items()
        }
        residual = (
            current_metrics["FO"]["dimension_weighted_semantic_error"]
            + current_metrics["OF"]["dimension_weighted_semantic_error"]
            - current_metrics["FF"]["dimension_weighted_semantic_error"]
            - current_metrics["OO"]["dimension_weighted_semantic_error"]
        )
        if float(np.max(np.abs(residual))) > ZERO_TOLERANCE:
            raise RuntimeError(f"Additive identity failure in episode {episode_id}")
        identity_residuals.append(residual)

        row: dict[str, Any] = {
            "split": split,
            "episode_id": episode_id,
            "task_id": task_id,
            "targets": len(eligible_target),
        }
        row.update(flatten_means(current_metrics))
        episode_rows.append(row)
        for condition in CONDITIONS:
            for metric in METRICS:
                target_metrics_by_condition[condition][metric].append(current_metrics[condition][metric])

        trans_disagreement.append(np.mean(((fresh[:, :3] - old[:, :3]) / ACTION_STD[:3]) ** 2, axis=1))
        rotation_disagreement.append(rotation_geodesic(fresh[:, 3:6], old[:, 3:6]))
        grip_disagreement.append((action_sign(fresh[:, 6]) != action_sign(old[:, 6])).astype(bool))

    if len(episode_rows) != EXPECTED_EPISODES or sum(row["targets"] for row in episode_rows) != EXPECTED_TARGETS:
        raise RuntimeError("Eligible cohort accounting mismatch")

    arrays = {
        condition: {metric: np.concatenate(values) for metric, values in metrics.items()}
        for condition, metrics in target_metrics_by_condition.items()
    }
    trans = np.concatenate(trans_disagreement)
    rotation = np.concatenate(rotation_disagreement)
    grip = np.concatenate(grip_disagreement)
    observed_trans_q = np.quantile(trans, [0.25, 0.5, 0.75], method="linear")
    observed_rotation_q = np.quantile(rotation, [0.25, 0.5, 0.75], method="linear")
    if not np.allclose(observed_trans_q, TRANSLATION_QUARTILES, atol=1e-15, rtol=0.0):
        raise RuntimeError(f"Translation quartiles changed: {observed_trans_q}")
    if not np.allclose(observed_rotation_q, ROTATION_QUARTILES, atol=1e-15, rtol=0.0):
        raise RuntimeError(f"Rotation quartiles changed: {observed_rotation_q}")
    if int(np.sum(grip)) != EXPECTED_GRIP_DISAGREEMENT:
        raise RuntimeError("Gripper disagreement count changed")

    per_task = task_rows(episode_rows)
    inference = primary_inference(episode_rows, per_task)
    residuals = np.concatenate(identity_residuals)
    condition_summaries = summarize_rows(episode_rows)
    diagnostics = {
        "scope": "Target-weighted descriptive strata; no frame-level inference.",
        "translation_normalized_disagreement": {
            "cutpoints": TRANSLATION_QUARTILES.tolist(),
            "bins": {name: stratum_summary(mask, arrays) for name, mask in quartile_masks(trans, TRANSLATION_QUARTILES)},
        },
        "rotation_geodesic_disagreement_radians": {
            "cutpoints": ROTATION_QUARTILES.tolist(),
            "bins": {name: stratum_summary(mask, arrays) for name, mask in quartile_masks(rotation, ROTATION_QUARTILES)},
        },
        "gripper_sign_disagreement": {
            "same": stratum_summary(~grip, arrays),
            "different": stratum_summary(grip, arrays),
        },
    }
    result = {
        "schema_version": 1,
        "scope": "Frozen post-Gate-3A2 exploratory teacher-forced cross-generation composition audit; no Thor Gate-3B outcome inspected.",
        "protocol": {
            "path": str(PROTOCOL.relative_to(ROOT)),
            "commit": PROTOCOL_COMMIT,
            "sha256": PROTOCOL_SHA256,
            "age_action_indices": AGE,
            "physical_age_seconds": 1.0,
        },
        "analysis": {
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
            "script_sha256": sha256(Path(__file__)),
            "git_head_at_execution": git_head(),
            "gate3a1_metric_script_sha256": GATE3A1_ANALYSIS_SHA256,
        },
        "assets": assets,
        "cohort": {
            "episodes": len(episode_rows),
            "validation_episodes": int(inventory["splits"]["validation"]["episodes"]),
            "test_episodes": int(inventory["splits"]["test"]["episodes"]),
            "eligible_targets": int(sum(row["targets"] for row in episode_rows)),
            "tasks": 10,
        },
        "metric_contract": {
            "l_sem": "(3*translation_normalized_mse + 3*rotation_normalized_sq + gripper_sign_error)/7",
            "action_std": ACTION_STD.tolist(),
            "gripper": "sign error; zero is positive",
            "continuous_gripper_mse": False,
        },
        "condition_summaries": condition_summaries,
        "primary_contrast": inference,
        "identity_check": {
            "definition": "targetwise L_FO+L_OF-L_FF-L_OO",
            "maximum_absolute_target_residual": float(np.max(np.abs(residuals))),
            "all_targets_within_tolerance": bool(np.all(np.abs(residuals) <= ZERO_TOLERANCE)),
            "tolerance": ZERO_TOLERANCE,
            "interpretation": "The frozen additive metric contains no cross-component interaction term.",
        },
        "per_task": per_task,
        "descriptive_disagreement_strata": diagnostics,
        "figure_data": {
            "panel_a": [
                {"condition": condition, "episode_weighted_l_sem": condition_summaries[condition]["l_sem"]["episode_weighted_mean"]}
                for condition in CONDITIONS
            ],
            "panel_b": [
                {
                    "condition": condition,
                    "translation": condition_summaries[condition]["translation"]["episode_weighted_mean"],
                    "rotation_normalized": condition_summaries[condition]["rotation_normalized"]["episode_weighted_mean"],
                    "rotation_geodesic_radians": condition_summaries[condition]["rotation_geodesic_radians"]["episode_weighted_mean"],
                    "gripper_sign": condition_summaries[condition]["gripper_sign"]["episode_weighted_mean"],
                }
                for condition in CONDITIONS
            ],
            "panel_c": [
                {"task_id": int(row["task_id"]), "c_offline": float(row["c_offline"])}
                for row in per_task
            ],
        },
        "interpretation_limit": "Offline additive teacher-forced loss cannot establish a cross-generation closed-loop effect; integrate only after the independent Gate-3B result is complete.",
    }

    atomic_json(args.metrics_output, result)
    atomic_csv(args.per_task_output, per_task)
    atomic_csv(args.contrast_output, episode_rows)
    atomic_text(args.report_output, render_report(result, per_task))
    print(
        json.dumps(
            {
                "episodes": len(episode_rows),
                "targets": EXPECTED_TARGETS,
                "condition_l_sem": {
                    condition: condition_summaries[condition]["l_sem"]["episode_weighted_mean"]
                    for condition in CONDITIONS
                },
                "c_offline": inference,
                "outputs": {
                    "metrics": str(args.metrics_output),
                    "per_task": str(args.per_task_output),
                    "contrast": str(args.contrast_output),
                    "report": str(args.report_output),
                },
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.mode == "self-test":
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()
