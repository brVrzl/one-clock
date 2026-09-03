#!/usr/bin/env python3
"""Frozen B2 training-demonstration temporal-persistence analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as pads
from safetensors import safe_open


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = Path("/home/wjq/research-assets/datasets/HuggingFaceVLA_libero")
TASK_TAGS = ("libero_object_task3", "libero_spatial_task0", "libero_goal_task2", "libero_10_task3")
ACT_ROOT = Path("/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final")
SMOL_NORMALIZER = Path("/home/wjq/checkpoints/HuggingFaceVLA_smolvla_libero/policy_preprocessor_step_5_normalizer_processor.safetensors")
GROUPS = {"translation": (0, 1, 2), "rotation": (3, 4, 5), "gripper": (6,), "arm": (0, 1, 2, 3, 4, 5)}


def norm_std(path: Path) -> np.ndarray:
    with safe_open(path, framework="numpy") as handle:
        value = np.asarray(handle.get_tensor("action.std"), dtype=np.float64)
    if value.shape != (7,) or not np.isfinite(value).all() or np.any(value <= 0):
        raise RuntimeError(f"invalid frozen action scale: {path}")
    return value


def ci(draws: np.ndarray) -> list[float]:
    return np.percentile(draws, [2.5, 97.5]).astype(float).tolist()


def correlation(sx: np.ndarray, sy: np.ndarray, sxx: np.ndarray, syy: np.ndarray, sxy: np.ndarray, n: np.ndarray) -> np.ndarray:
    numerator = sxy - sx * sy / n
    denominator = np.sqrt(np.maximum(sxx - sx * sx / n, 0) * np.maximum(syy - sy * sy / n, 0))
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def km_survival(risk: np.ndarray, events: np.ndarray, windows: tuple[int, ...]) -> np.ndarray:
    """Product-limit survival for one or many episode-weighted samples."""
    if risk.ndim == 1:
        risk = risk[None, :]
        events = events[None, :]
    factors = np.divide(events, risk, out=np.zeros_like(events), where=risk > 0)
    survival = np.cumprod(1.0 - factors, axis=1)
    return survival[:, [window - 1 for window in windows]]


def main() -> None:
    addendum = json.loads((ROOT / "track_b_analysis_addendum.json").read_text())
    b2 = addendum["b2"]
    if addendum.get("status") != "FROZEN_BEFORE_TRACK_B_PREDICTION_INTERPRETATION":
        raise RuntimeError("analysis addendum is not frozen")

    episode_to_task: dict[int, str] = {}
    act_std: dict[str, np.ndarray] = {}
    for tag in TASK_TAGS:
        checkpoint = ACT_ROOT / tag / "checkpoints/100000/pretrained_model"
        config = json.loads((checkpoint / "train_config.json").read_text())
        episodes = [int(x) for x in config["dataset"]["episodes"]]
        expected = int(b2["panel_episode_counts"][tag.replace("_task", ":task")])
        if len(episodes) != expected:
            raise RuntimeError(f"episode-count drift for {tag}")
        for episode in episodes:
            if episode in episode_to_task:
                raise RuntimeError(f"episode assigned to multiple tasks: {episode}")
            episode_to_task[episode] = tag
        act_std[tag] = norm_std(checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors")
    if len(episode_to_task) != 173:
        raise RuntimeError("frozen panel must contain 173 demonstration episodes")
    smol_std = norm_std(SMOL_NORMALIZER)

    dataset = pads.dataset(DATASET_ROOT / "data", format="parquet")
    table = dataset.to_table(
        columns=["episode_index", "frame_index", "action"],
        filter=pads.field("episode_index").isin(sorted(episode_to_task)),
    )
    rows = table.to_pylist()
    by_episode: dict[int, list[tuple[int, list[float]]]] = {episode: [] for episode in episode_to_task}
    for row in rows:
        episode = int(row["episode_index"])
        if episode in by_episode:
            by_episode[episode].append((int(row["frame_index"]), row["action"]))
    actions: dict[int, np.ndarray] = {}
    for episode, values in by_episode.items():
        values.sort(key=lambda x: x[0])
        frames = [x[0] for x in values]
        if frames != list(range(len(frames))):
            raise RuntimeError(f"non-contiguous frames in episode {episode}")
        array = np.asarray([x[1] for x in values], dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 7:
            raise RuntimeError(f"invalid action shape in episode {episode}")
        actions[episode] = array

    episode_ids = sorted(actions)
    if len(episode_ids) != 173:
        raise RuntimeError("dataset scan did not recover all frozen episodes")
    draws = int(b2["bootstrap_draws"])
    rng = np.random.default_rng(int(b2["bootstrap_seed"]))
    tidy: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    for lag in b2["lags"]:
        lag = int(lag)
        n = np.zeros((len(episode_ids), 7), dtype=np.float64)
        sx = np.zeros_like(n); sy = np.zeros_like(n); sxx = np.zeros_like(n); syy = np.zeros_like(n); sxy = np.zeros_like(n)
        normalized_sse = {"ACT": np.zeros_like(n), "SmolVLA": np.zeros_like(n)}
        sign_disagree = np.zeros(len(episode_ids), dtype=np.float64)
        pair_count = np.zeros(len(episode_ids), dtype=np.float64)
        for i, episode in enumerate(episode_ids):
            a = actions[episode]
            x, y = (a, a) if lag == 0 else (a[:-lag], a[lag:])
            count = len(x)
            n[i, :] = count
            sx[i] = x.sum(0); sy[i] = y.sum(0)
            sxx[i] = np.square(x).sum(0); syy[i] = np.square(y).sum(0); sxy[i] = (x * y).sum(0)
            delta = y - x
            normalized_sse["ACT"][i] = np.square(delta / act_std[episode_to_task[episode]]).sum(0)
            normalized_sse["SmolVLA"][i] = np.square(delta / smol_std).sum(0)
            pair_count[i] = count
            sign_disagree[i] = np.not_equal(np.sign(x[:, 6]), np.sign(y[:, 6])).sum()

        weights = rng.multinomial(len(episode_ids), np.full(len(episode_ids), 1 / len(episode_ids)), size=draws).astype(np.float64)
        totals = {name: weights @ value for name, value in (("sx", sx), ("sy", sy), ("sxx", sxx), ("syy", syy), ("sxy", sxy), ("n", n))}
        center_corr = correlation(sx.sum(0), sy.sum(0), sxx.sum(0), syy.sum(0), sxy.sum(0), n.sum(0))
        boot_corr = correlation(totals["sx"], totals["sy"], totals["sxx"], totals["syy"], totals["sxy"], totals["n"])
        lag_summary: dict[str, Any] = {"autocorrelation": {}, "normalized_difference": {}, "gripper_sign": {}}
        for dim in range(7):
            valid = np.isfinite(boot_corr[:, dim])
            lag_summary["autocorrelation"][f"dim_{dim}"] = {
                "pair_count": int(n[:, dim].sum()), "value": float(center_corr[dim]) if np.isfinite(center_corr[dim]) else None,
                "episode_cluster_bootstrap_ci": ci(boot_corr[valid, dim]) if valid.any() else None,
            }
            tidy.append({"lag": lag, "scale": "raw", "quantity": "autocorrelation", "metric": f"dim_{dim}", **lag_summary["autocorrelation"][f"dim_{dim}"]})
        for scale in ("ACT", "SmolVLA"):
            sse = normalized_sse[scale]
            center_dim = np.sqrt(sse.sum(0) / n.sum(0))
            boot_dim = np.sqrt((weights @ sse) / (weights @ n))
            lag_summary["normalized_difference"][scale] = {}
            for dim in range(7):
                record = {"value": float(center_dim[dim]), "episode_cluster_bootstrap_ci": ci(boot_dim[:, dim])}
                lag_summary["normalized_difference"][scale][f"dim_{dim}"] = record
                tidy.append({"lag": lag, "scale": scale, "quantity": "normalized_rms_difference", "metric": f"dim_{dim}", **record})
            for group, dims in GROUPS.items():
                group_sse = sse[:, dims].sum(1)
                group_n = n[:, 0] * len(dims)
                center = float(np.sqrt(group_sse.sum() / group_n.sum()))
                boot = np.sqrt((weights @ group_sse) / (weights @ group_n))
                record = {"value": center, "episode_cluster_bootstrap_ci": ci(boot)}
                lag_summary["normalized_difference"][scale][group] = record
                tidy.append({"lag": lag, "scale": scale, "quantity": "normalized_rms_difference", "metric": group, **record})
        center_sign = float(sign_disagree.sum() / pair_count.sum())
        boot_sign = (weights @ sign_disagree) / (weights @ pair_count)
        lag_summary["gripper_sign"] = {"disagreement_probability": center_sign, "agreement_probability": 1 - center_sign, "episode_cluster_bootstrap_ci": ci(boot_sign)}
        tidy.append({"lag": lag, "scale": "raw_sign", "quantity": "gripper_sign_disagreement", "metric": "gripper", "value": center_sign, "episode_cluster_bootstrap_ci": ci(boot_sign), "pair_count": int(pair_count.sum())})
        summary[str(lag)] = lag_summary

    transition_rows: list[dict[str, Any]] = []
    transition_count = np.zeros(len(episode_ids), dtype=np.float64)
    adjacent_count = np.zeros(len(episode_ids), dtype=np.float64)
    observed_distance_sum = np.zeros(len(episode_ids), dtype=np.float64)
    observed_distance_n = np.zeros(len(episode_ids), dtype=np.float64)
    censored_n = np.zeros(len(episode_ids), dtype=np.float64)
    total_steps = np.zeros(len(episode_ids), dtype=np.float64)
    for i, episode in enumerate(episode_ids):
        signs = np.sign(actions[episode][:, 6])
        changes = np.flatnonzero(signs[1:] != signs[:-1]) + 1
        transition_count[i] = len(changes); adjacent_count[i] = max(len(signs) - 1, 0)
        for t in range(len(signs)):
            future = changes[changes > t]
            if len(future):
                distance = int(future[0] - t)
                observed_distance_sum[i] += distance; observed_distance_n[i] += 1
                transition_rows.append({"episode_index": episode, "task": episode_to_task[episode], "frame_index": t, "duration_steps": distance, "event_observed": True, "right_censored": False})
            else:
                censored_n[i] += 1
                transition_rows.append({"episode_index": episode, "task": episode_to_task[episode], "frame_index": t, "duration_steps": len(signs) - 1 - t, "event_observed": False, "right_censored": True})
            total_steps[i] += 1
    transition_weights = rng.multinomial(len(episode_ids), np.full(len(episode_ids), 1 / len(episode_ids)), size=draws).astype(np.float64)
    transition_rate = float(transition_count.sum() / adjacent_count.sum())
    transition_rate_boot = (transition_weights @ transition_count) / (transition_weights @ adjacent_count)
    censor_fraction = float(censored_n.sum() / total_steps.sum())
    censor_boot = (transition_weights @ censored_n) / (transition_weights @ total_steps)
    observed_mean = float(observed_distance_sum.sum() / observed_distance_n.sum())
    observed_boot = (transition_weights @ observed_distance_sum) / (transition_weights @ observed_distance_n)
    survival_windows = (5, 10, 20)
    risk_counts = np.zeros((len(episode_ids), max(survival_windows)), dtype=np.float64)
    event_counts = np.zeros_like(risk_counts)
    for row in transition_rows:
        episode_i = episode_ids.index(int(row["episode_index"]))
        duration = int(row["duration_steps"])
        for event_time in range(1, max(survival_windows) + 1):
            risk_counts[episode_i, event_time - 1] += duration >= event_time
        if row["event_observed"] and 1 <= duration <= max(survival_windows):
            event_counts[episode_i, duration - 1] += 1
    center_survival = km_survival(risk_counts.sum(0), event_counts.sum(0), survival_windows)[0]
    survival_rng = np.random.default_rng(27302)
    survival_weights = survival_rng.multinomial(
        len(episode_ids), np.full(len(episode_ids), 1 / len(episode_ids)), size=draws
    ).astype(np.float64)
    boot_survival = km_survival(survival_weights @ risk_counts, survival_weights @ event_counts, survival_windows)
    survival_summary = {
        f"{window / 10:.1f}": {
            "window_steps": window,
            "probability_no_transition_within_window": float(center_survival[index]),
            "episode_cluster_bootstrap_ci": ci(boot_survival[:, index]),
        }
        for index, window in enumerate(survival_windows)
    }

    output_dir = ROOT / "track_b" / "demonstration_persistence"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "lag_metrics.csv").open("w", newline="") as handle:
        fieldnames = sorted({key for row in tidy for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader(); writer.writerows(tidy)
    with (output_dir / "transition_distance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(transition_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(transition_rows)
    output = {
        "status": "COMPLETE", "success_outcomes_loaded": False,
        "provenance_label": b2["provenance_label"], "dataset_repo": b2["dataset_repo"],
        "dataset_revision": b2["dataset_revision"], "dataset_split": b2["dataset_split"],
        "dataset_fps": b2["dataset_fps"], "episode_count": len(episode_ids),
        "lag_metrics": summary,
        "gripper_transition": {
            "adjacent_transition_frequency": transition_rate,
            "episode_cluster_bootstrap_ci": ci(transition_rate_boot),
            "observed_distance_to_next_transition_mean_steps": observed_mean,
            "observed_distance_episode_cluster_bootstrap_ci": ci(observed_boot),
            "observed_distance_interpretation": "biased complete-case descriptive value; not a population mean because right censoring is excluded",
            "right_censored_step_fraction": censor_fraction,
            "right_censored_episode_cluster_bootstrap_ci": ci(censor_boot),
            "kaplan_meier_transition_free_survival": survival_summary,
        },
        "bootstrap": {"unit": "demonstration_episode", "draws": draws, "lag_seed": b2["bootstrap_seed"], "survival_seed": 27302},
    }
    (output_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    lines = [
        "# B2 training-demonstration temporal persistence", "",
        "These 173 episodes are training demonstrations, not a held-out split. Dataset actions are at 10 Hz. No rollout success outcome was loaded.", "",
        f"Adjacent gripper sign-transition frequency: `{transition_rate:.6f}` (episode-cluster 95% CI `[{ci(transition_rate_boot)[0]:.6f}, {ci(transition_rate_boot)[1]:.6f}]`).",
        f"Right-censored action-step fraction: `{censor_fraction:.3%}`. The `{observed_mean:.3f}`-step mean among observed transitions is a biased complete-case description, not a population mean.", "",
        "| Window | P(no gripper transition within window) | Episode-cluster 95% CI |", "|---:|---:|---:|",
        *[f"| {seconds} s | {record['probability_no_transition_within_window']:.6f} | [{record['episode_cluster_bootstrap_ci'][0]:.6f}, {record['episode_cluster_bootstrap_ci'][1]:.6f}] |" for seconds, record in survival_summary.items()], "",
        "All per-dimension autocorrelation, normalized-difference, group, and lag results are stored in `lag_metrics.csv` and `summary.json`.", "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))
    print(json.dumps({"episodes": len(episode_ids), "transition_frequency": transition_rate, "right_censored_fraction": censor_fraction, "kaplan_meier_transition_free_survival": survival_summary}, indent=2))


if __name__ == "__main__":
    main()
