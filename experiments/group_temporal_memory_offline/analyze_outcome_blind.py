#!/usr/bin/env python3
"""Build outcome-blind same-target delay profiles for the frozen SmolVLA cache.

This module deliberately accepts only the frozen protocol and Fresh query
caches.  Closed-loop result files are consumed by ``analyze_closed_loop.py``
after this artifact has been written.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.overnight_pppr_20260828.pppr_metrics import (  # noqa: E402
    action_at,
    fit_arm_scales,
)


GROUPS = ("arm", "gripper")
ARM_DIM = 6
GRIPPER_DIFF_RANGE = 2.0
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260831


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=json_default) + "\n")


def task_cache_path(cache_root: Path, task_key: str) -> Path:
    suite, task_number = task_key.split(":task")
    return cache_root / f"{suite}_task{int(task_number)}_fresh.npz"


def load_task_episodes(cache_root: Path, task_key: str) -> list[np.ndarray]:
    path = task_cache_path(cache_root, task_key)
    if not path.exists():
        raise FileNotFoundError(f"missing Fresh cache for {task_key}: {path}")
    with np.load(path, allow_pickle=False) as archive:
        names = sorted(
            archive.files,
            key=lambda name: int(name.removeprefix("episode_"))
            if name.startswith("episode_")
            else name,
        )
        if not names or any(not name.startswith("episode_") for name in names):
            raise ValueError(f"unexpected cache keys in {path}: {archive.files}")
        episodes = [np.asarray(archive[name], dtype=np.float64) for name in names]
    for index, episode in enumerate(episodes):
        if episode.ndim != 3 or episode.shape[2] != 7:
            raise ValueError(f"{task_key} episode {index} has invalid shape {episode.shape}")
        if episode.shape[0] == 0 or episode.shape[1] == 0:
            raise ValueError(f"{task_key} episode {index} is empty")
        if not np.isfinite(episode).all():
            raise ValueError(f"{task_key} episode {index} contains non-finite actions")
    return episodes


def valid_target_indices(episode_length: int, chunk_horizon: int, delay: int) -> np.ndarray:
    """Return physical targets with a valid same-target source at ``t-delay``."""

    if episode_length < 0 or chunk_horizon < 0 or delay < 0:
        raise ValueError("episode length, chunk horizon, and delay must be non-negative")
    if delay >= chunk_horizon:
        return np.empty(0, dtype=np.int64)
    return np.arange(delay, episode_length, dtype=np.int64)


def group_slices() -> dict[str, slice]:
    return {"arm": slice(0, 6), "gripper": slice(6, 7)}


def pair_metrics(
    fresh: np.ndarray,
    historical: np.ndarray,
    arm_scales: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute vectorized B1 metrics for aligned Fresh/historical actions."""

    fresh = np.asarray(fresh, dtype=np.float64)
    historical = np.asarray(historical, dtype=np.float64)
    if fresh.shape != historical.shape or fresh.ndim != 2 or fresh.shape[1] != 7:
        raise ValueError(f"expected matching [N,7] actions, got {fresh.shape} and {historical.shape}")
    if arm_scales.shape != (6,) or not np.isfinite(arm_scales).all() or np.any(arm_scales <= 0):
        raise ValueError("arm_scales must be finite and positive with shape (6,)")
    difference = historical - fresh
    translation = np.linalg.norm(difference[:, :3] / arm_scales[:3], axis=1) / np.sqrt(3.0)
    rotation = np.linalg.norm(difference[:, 3:6] / arm_scales[3:6], axis=1) / np.sqrt(3.0)
    arm_unbounded = 0.5 * (translation + rotation)
    arm_revision = arm_unbounded / (1.0 + arm_unbounded)
    gripper_abs_diff = np.abs(difference[:, 6])
    gripper_sign_disagreement = np.sign(historical[:, 6]) != np.sign(fresh[:, 6])
    return {
        "arm_translation": translation,
        "arm_rotation": rotation,
        "arm_revision": arm_revision,
        "gripper_abs_diff": gripper_abs_diff,
        "gripper_sign_disagreement": gripper_sign_disagreement.astype(np.float64),
        "utility_arm": 1.0 - arm_revision,
        "utility_gripper": 1.0 - np.clip(gripper_abs_diff / GRIPPER_DIFF_RANGE, 0.0, 1.0),
    }


def build_feature_rows(
    task_keys: list[str],
    split_by_task: dict[str, str],
    cache_root: Path,
    delays: list[int],
    arm_scales: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, list[np.ndarray]], dict[str, Any]]:
    """Materialize one row per task/episode/target/requested-delay cell."""

    rows: dict[str, list[Any]] = {
        "task_index": [],
        "task_key": [],
        "split": [],
        "episode": [],
        "target_t": [],
        "source_q": [],
        "delay": [],
        "valid": [],
        "arm_translation": [],
        "arm_rotation": [],
        "arm_revision": [],
        "gripper_abs_diff": [],
        "gripper_sign_disagreement": [],
        "utility_arm": [],
        "utility_gripper": [],
    }
    all_episodes: dict[str, list[np.ndarray]] = {}
    inventory: dict[str, Any] = {}
    for task_index, task_key in enumerate(task_keys):
        episodes = load_task_episodes(cache_root, task_key)
        all_episodes[task_key] = episodes
        task_inventory = {
            "path": str(task_cache_path(cache_root, task_key).resolve()),
            "episodes": len(episodes),
            "episode_shapes": [list(episode.shape) for episode in episodes],
            "chunk_horizon": int(episodes[0].shape[1]),
            "target_steps": int(sum(episode.shape[0] for episode in episodes)),
            "valid_targets_by_delay": {},
        }
        for delay in delays:
            task_inventory["valid_targets_by_delay"][str(delay)] = int(
                sum(len(valid_target_indices(ep.shape[0], ep.shape[1], delay)) for ep in episodes)
            )
        inventory[task_key] = task_inventory

        for episode_index, chunks in enumerate(episodes):
            for delay in delays:
                targets = valid_target_indices(chunks.shape[0], chunks.shape[1], delay)
                all_targets = np.arange(chunks.shape[0], dtype=np.int64)
                valid_lookup = set(int(value) for value in targets)
                for target_t in all_targets.tolist():
                    is_valid = int(target_t) in valid_lookup
                    rows["task_index"].append(task_index)
                    rows["task_key"].append(task_key)
                    rows["split"].append(split_by_task[task_key])
                    rows["episode"].append(episode_index)
                    rows["target_t"].append(target_t)
                    rows["delay"].append(delay)
                    rows["valid"].append(is_valid)
                    if not is_valid:
                        rows["source_q"].append(-1)
                        for field in (
                            "arm_translation",
                            "arm_rotation",
                            "arm_revision",
                            "gripper_abs_diff",
                            "gripper_sign_disagreement",
                            "utility_arm",
                            "utility_gripper",
                        ):
                            rows[field].append(np.nan)
                        continue
                    source_q = int(target_t - delay)
                    fresh = action_at(chunks, int(target_t), int(target_t))
                    historical = action_at(chunks, source_q, int(target_t))
                    metrics = pair_metrics(fresh[None, :], historical[None, :], arm_scales)
                    rows["source_q"].append(source_q)
                    for field, values in metrics.items():
                        rows[field].append(float(values[0]))

    arrays: dict[str, np.ndarray] = {
        "task_index": np.asarray(rows["task_index"], dtype=np.int16),
        "task_key": np.asarray(rows["task_key"], dtype="U32"),
        "split": np.asarray(rows["split"], dtype="U16"),
        "episode": np.asarray(rows["episode"], dtype=np.int16),
        "target_t": np.asarray(rows["target_t"], dtype=np.int32),
        "source_q": np.asarray(rows["source_q"], dtype=np.int32),
        "delay": np.asarray(rows["delay"], dtype=np.int16),
        "valid": np.asarray(rows["valid"], dtype=bool),
    }
    for field in (
        "arm_translation",
        "arm_rotation",
        "arm_revision",
        "gripper_abs_diff",
        "gripper_sign_disagreement",
        "utility_arm",
        "utility_gripper",
    ):
        arrays[field] = np.asarray(rows[field], dtype=np.float32)
    return arrays, all_episodes, inventory


def summarize_delay(
    arrays: dict[str, np.ndarray],
    task_key: str,
    split: str,
    episode_index: int,
    delay: int,
    value_field: str,
) -> tuple[float, int]:
    mask = (
        (arrays["task_key"] == task_key)
        & (arrays["split"] == split)
        & (arrays["episode"] == episode_index)
        & (arrays["delay"] == delay)
        & arrays["valid"]
    )
    values = arrays[value_field][mask]
    if values.size == 0:
        raise ValueError(f"no valid rows for {task_key} episode {episode_index} delay {delay}")
    return float(np.mean(values)), int(values.size)


def bootstrap_h_temp(
    episode_profiles: dict[str, dict[str, dict[int, np.ndarray]]],
    task_key: str,
    delays: list[int],
    draws: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    count = len(next(iter(episode_profiles[task_key]["arm"].values())))
    samples: list[float] = []
    for _ in range(draws):
        indices = rng.integers(0, count, size=count)
        arm = np.asarray(
            [np.mean(np.asarray(episode_profiles[task_key]["arm"][delay])[indices]) for delay in delays], dtype=float
        )
        grip = np.asarray(
            [np.mean(np.asarray(episode_profiles[task_key]["gripper"][delay])[indices]) for delay in delays], dtype=float
        )
        samples.append(float(np.mean(np.abs(arm - grip))))
    return samples


def percentile(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def build_analysis(protocol: dict[str, Any], cache_root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    delays = [int(value) for value in protocol["delays"]["steps"]]
    task_keys = list(protocol["task_split"]["task_order"])
    development = set(protocol["task_split"]["development"])
    held_out = set(protocol["task_split"]["held_out_descriptive"])
    if development & held_out or development | held_out != set(task_keys):
        raise ValueError("task split is not a partition of the frozen task order")
    split_by_task = {key: ("development" if key in development else "held_out") for key in task_keys}
    all_episodes = {
        key: load_task_episodes(cache_root, key)
        for key in task_keys
    }
    arm_scale_fit = fit_arm_scales(
        episode
        for task_key in task_keys
        if task_key in development
        for episode in all_episodes[task_key]
    )
    arm_scales = np.asarray(arm_scale_fit.scales, dtype=np.float64)
    arrays, loaded_episodes, inventory = build_feature_rows(
        task_keys, split_by_task, cache_root, delays, arm_scales
    )
    if set(loaded_episodes) != set(all_episodes):
        raise AssertionError("cache load changed between scale fitting and feature construction")

    episode_profiles: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    task_profiles: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for task_key in task_keys:
        split = split_by_task[task_key]
        episode_profiles[task_key] = {"arm": {}, "gripper": {}}
        task_profiles[task_key] = {"split": split, "groups": {}, "H_temp": None}
        for group, value_field, revision_field in (
            ("arm", "utility_arm", "arm_revision"),
            ("gripper", "utility_gripper", "gripper_abs_diff"),
        ):
            episode_profiles[task_key][group] = {delay: [] for delay in delays}
            task_profiles[task_key]["groups"][group] = {"delays": []}
            for delay in delays:
                episode_values: list[float] = []
                episode_revision_values: list[float] = []
                episode_sign_values: list[float] = []
                counts: list[int] = []
                for episode_index in range(len(loaded_episodes[task_key])):
                    mean_utility, valid_count = summarize_delay(
                        arrays, task_key, split, episode_index, delay, value_field
                    )
                    revision_mean, _ = summarize_delay(
                        arrays, task_key, split, episode_index, delay, revision_field
                    )
                    translation_mean = None
                    rotation_mean = None
                    if group == "arm":
                        translation_mean, _ = summarize_delay(
                            arrays, task_key, split, episode_index, delay, "arm_translation"
                        )
                        rotation_mean, _ = summarize_delay(
                            arrays, task_key, split, episode_index, delay, "arm_rotation"
                        )
                    sign_mask = (
                        (arrays["task_key"] == task_key)
                        & (arrays["split"] == split)
                        & (arrays["episode"] == episode_index)
                        & (arrays["delay"] == delay)
                        & arrays["valid"]
                    )
                    sign_mean = float(np.mean(arrays["gripper_sign_disagreement"][sign_mask]))
                    episode_values.append(mean_utility)
                    episode_revision_values.append(revision_mean)
                    episode_sign_values.append(sign_mean)
                    counts.append(valid_count)
                    episode_profiles[task_key][group][delay].append(mean_utility)
                    episode_rows.append(
                        {
                            "task_key": task_key,
                            "split": split,
                            "group": group,
                            "episode": episode_index,
                            "delay_steps": delay,
                            "valid_target_count": valid_count,
                            "utility": mean_utility,
                            "revision_or_abs_diff": revision_mean,
                            "translation_revision": translation_mean,
                            "rotation_revision": rotation_mean,
                            "gripper_sign_disagreement": sign_mean,
                        }
                    )
                profile_row = {
                    "delay_steps": delay,
                    "valid_target_count": int(sum(counts)),
                    "valid_episodes": int(sum(count > 0 for count in counts)),
                    "utility": float(np.mean(episode_values)),
                    "utility_sd_across_episodes": float(np.std(episode_values, ddof=1))
                    if len(episode_values) > 1
                    else 0.0,
                    "revision_or_abs_diff": float(np.mean(episode_revision_values)),
                    "revision_or_abs_diff_sd_across_episodes": float(np.std(episode_revision_values, ddof=1))
                    if len(episode_revision_values) > 1
                    else 0.0,
                    "translation_revision": None,
                    "rotation_revision": None,
                    "gripper_sign_disagreement": float(np.mean(episode_sign_values)),
                    "episode_utility_values": episode_values,
                }
                if group == "arm":
                    translation_values = [
                        summarize_delay(
                            arrays, task_key, split, episode_index, delay, "arm_translation"
                        )[0]
                        for episode_index in range(len(loaded_episodes[task_key]))
                    ]
                    rotation_values = [
                        summarize_delay(
                            arrays, task_key, split, episode_index, delay, "arm_rotation"
                        )[0]
                        for episode_index in range(len(loaded_episodes[task_key]))
                    ]
                    profile_row["translation_revision"] = float(np.mean(translation_values))
                    profile_row["rotation_revision"] = float(np.mean(rotation_values))
                task_profiles[task_key]["groups"][group]["delays"].append(profile_row)
                summary_rows.append(
                    {
                        "task_key": task_key,
                        "split": split,
                        "group": group,
                        "delay_steps": delay,
                        **{key: value for key, value in profile_row.items() if key != "episode_utility_values"},
                    }
                )

    thresholds = {}
    for group in GROUPS:
        values = [
            next(
                row["utility"]
                for row in task_profiles[task_key]["groups"][group]["delays"]
                if row["delay_steps"] == 16
            )
            for task_key in task_keys
            if task_key in development
        ]
        thresholds[group] = float(np.median(values))

    for task_key in task_keys:
        arm_delays = task_profiles[task_key]["groups"]["arm"]["delays"]
        grip_delays = task_profiles[task_key]["groups"]["gripper"]["delays"]
        arm_utility = np.asarray([row["utility"] for row in arm_delays], dtype=float)
        grip_utility = np.asarray([row["utility"] for row in grip_delays], dtype=float)
        task_profiles[task_key]["H_temp"] = float(np.mean(np.abs(arm_utility - grip_utility)))
        task_profiles[task_key]["H_temp_episode_bootstrap_ci95"] = None
        for group, profile_rows in (("arm", arm_delays), ("gripper", grip_delays)):
            utilities = np.asarray([row["utility"] for row in profile_rows], dtype=float)
            below = [
                int(delay)
                for delay, utility in zip(delays, utilities)
                if utility < thresholds[group]
            ]
            profile_rows_for_summary = {
                "preferred_delay_steps": int(delays[int(np.argmax(utilities))]),
                "preferred_delay_utility": float(np.max(utilities)),
                "best_positive_delay_steps": int(delays[1:][int(np.argmax(utilities[1:]))]),
                "best_positive_delay_utility": float(np.max(utilities[1:])),
                "delay_sensitivity_slope_per_step": float(np.polyfit(delays, utilities, 1)[0]),
                "utility_auc_normalized_0_to_32": float(np.trapezoid(utilities, delays) / (delays[-1] - delays[0])),
                "development_threshold": float(thresholds[group]),
                "first_delay_below_development_threshold": below[0] if below else None,
                "profile_shape": "monotone_nonincreasing"
                if np.all(np.diff(utilities) <= 1e-12)
                else "non_monotonic",
            }
            task_profiles[task_key]["groups"][group]["summary"] = profile_rows_for_summary
        bootstrap = bootstrap_h_temp(
            episode_profiles, task_key, delays, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED + task_keys.index(task_key)
        )
        task_profiles[task_key]["H_temp_episode_bootstrap_ci95"] = [
            percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)
        ]

    task_h = [float(task_profiles[key]["H_temp"]) for key in task_keys]
    split_summary = {}
    for split in ("development", "held_out"):
        values = [
            float(task_profiles[key]["H_temp"])
            for key in task_keys
            if task_profiles[key]["split"] == split
        ]
        split_summary[split] = {
            "tasks": len(values),
            "task_macro_mean_H_temp": float(np.mean(values)),
            "task_macro_sd_H_temp": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "task_values": values,
        }

    non_markovian: dict[str, Any] = {"primary_revision_utility": {}}
    for group, utility_field, revision_field in (
        ("arm", "utility_arm", "arm_revision"),
        ("gripper", "utility_gripper", "gripper_abs_diff"),
    ):
        group_result = {"delays": {}, "task_count": len(task_keys)}
        for delay in delays[1:]:
            better_task_keys = []
            matching_task_keys = []
            better_rows = 0
            matching_rows = 0
            total_rows = 0
            for task_key in task_keys:
                profile = next(
                    row for row in task_profiles[task_key]["groups"][group]["delays"]
                    if row["delay_steps"] == delay
                )
                fresh_utility = next(
                    row["utility"] for row in task_profiles[task_key]["groups"][group]["delays"]
                    if row["delay_steps"] == 0
                )
                if profile["utility"] > fresh_utility + 1e-9:
                    better_task_keys.append(task_key)
                if abs(profile["utility"] - fresh_utility) <= 1e-9:
                    matching_task_keys.append(task_key)
                mask = (arrays["task_key"] == task_key) & (arrays["delay"] == delay) & arrays["valid"]
                values = arrays[revision_field][mask]
                total_rows += int(values.size)
                better_rows += int(np.sum(values < -1e-9))
                matching_rows += int(np.sum(np.abs(values) <= 1e-9))
            group_result["delays"][str(delay)] = {
                "task_level_better_count": len(better_task_keys),
                "task_level_better_tasks": better_task_keys,
                "task_level_match_count": len(matching_task_keys),
                "task_level_match_tasks": matching_task_keys,
                "candidate_rows": total_rows,
                "candidate_rows_better_than_fresh": better_rows,
                "candidate_rows_exactly_matching_fresh": matching_rows,
                "candidate_rows_better_fraction": better_rows / total_rows if total_rows else None,
                "candidate_rows_match_fraction": matching_rows / total_rows if total_rows else None,
                "comparison_metric": "revision distance; lower is better, Fresh d=0 is identity",
            }
        non_markovian["primary_revision_utility"][group] = group_result

    analysis = {
        "analysis_status": "complete_outcome_blind_frozen",
        "protocol_path": str((ROOT / "protocol.json").resolve()),
        "source_cache_root": str(cache_root.resolve()),
        "task_split": protocol["task_split"],
        "delays": delays,
        "arm_scale_fit": arm_scale_fit.as_dict(),
        "gripper_diff_range": GRIPPER_DIFF_RANGE,
        "cache_inventory": inventory,
        "counts": {
            "tasks": len(task_keys),
            "episodes": int(sum(len(value) for value in loaded_episodes.values())),
            "current_target_steps": int(sum(ep.shape[0] for values in loaded_episodes.values() for ep in values)),
            "feature_rows_including_masked": int(len(arrays["valid"])),
            "valid_feature_rows": int(np.sum(arrays["valid"])),
            "valid_candidate_rows_by_delay": {
                str(delay): int(np.sum((arrays["delay"] == delay) & arrays["valid"])) for delay in delays
            },
        },
        "profiles": task_profiles,
        "task_macro_H_temp_ranking": [
            {"task_key": key, "split": task_profiles[key]["split"], "H_temp": task_profiles[key]["H_temp"]}
            for key in sorted(task_keys, key=lambda item: task_profiles[item]["H_temp"], reverse=True)
        ],
        "split_summary": split_summary,
        "non_markovian_evidence": non_markovian,
        "demonstration_agreement": protocol["demonstration_agreement"],
        "reliability_context": protocol["existing_source_reliability"],
        "closed_loop_files_loaded": False,
        "interpretation_limits": [
            "Same-target differences mix observation-delay effects with SmolVLA stochastic flow variation because sampling was not keyed by physical step.",
            "Revision utility is an offline consistency proxy, not a causal estimate of control quality or closed-loop success.",
            "Candidate rows are support counts only; task and episode summaries are the aggregation units.",
        ],
    }
    analysis["_summary_rows"] = summary_rows
    analysis["_episode_rows"] = episode_rows
    analysis["_task_h"] = task_h
    return analysis, arrays


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_figures(analysis: dict[str, Any], figures_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style_path = Path("/home/wjq/.codex/skills/figure-style")
    sys.path.insert(0, str(style_path))
    from kernel import apply_figure_style  # type: ignore

    apply_figure_style(frame="open", sizes=(8, 7, 6), grid=False)
    delays = analysis["delays"]
    task_keys = list(analysis["task_split"]["task_order"])
    colors = {"arm": "#2166ac", "gripper": "#e08214"}

    fig, axes = plt.subplots(4, 2, figsize=(7.2, 9.0), sharex=True, sharey=True)
    for ax, task_key in zip(axes.flat, task_keys):
        profile = analysis["profiles"][task_key]
        for group in GROUPS:
            values = [row["utility"] for row in profile["groups"][group]["delays"]]
            ax.plot(delays, values, marker="o", ms=3, lw=1.3, color=colors[group], label=group)
        short = task_key.replace("libero_", "").replace(":task", "-")
        ax.set_title(f"{short} ({profile['split']})")
        ax.set_ylim(0.0, 1.03)
        ax.set_xticks(delays)
        ax.margins(x=0.05)
        ax.text(0.98, 0.05, f"H={profile['H_temp']:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    axes[-1, 0].set_xlabel("source age d (steps)")
    axes[-1, 1].set_xlabel("source age d (steps)")
    axes[1, 0].set_ylabel("normalized same-target utility")
    axes[3, 0].set_ylabel("normalized same-target utility")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, ["arm (6-D)", "gripper (scalar)"], loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Outcome-blind delay utility by task and action group", y=0.995, x=0.04, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    fig.savefig(figures_dir / "figure_A_delay_profiles.png")
    plt.close(fig)

    ranking = sorted(analysis["task_macro_H_temp_ranking"], key=lambda row: row["H_temp"])
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    y = np.arange(len(ranking))
    bar_colors = ["#2166ac" if row["split"] == "development" else "#e08214" for row in ranking]
    ax.scatter([row["H_temp"] for row in ranking], y, s=30, c=bar_colors, zorder=3)
    ax.hlines(y, 0, [row["H_temp"] for row in ranking], color="#bdbdbd", lw=0.8, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([row["task_key"].replace("libero_", "").replace(":task", "-") for row in ranking])
    ax.set_xlabel("H_temp (mean absolute utility-profile gap)")
    ax.set_title("Task-level temporal heterogeneity is descriptive and outcome-blind")
    ax.set_xlim(0, max(0.1, max(row["H_temp"] for row in ranking) * 1.15))
    ax.legend(
        [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#2166ac", markersize=5),
         plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#e08214", markersize=5)],
        ["development", "held-out descriptive"], loc="lower right",
    )
    ax.margins(y=0.08)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure_B_task_heterogeneity.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    cache_root = (ROOT / protocol["source"]["cache_root"]).resolve()
    analysis, arrays = build_analysis(protocol, cache_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "cached_same_target_features.npz", **arrays)
    write_json(args.output_dir / "normalization.json", {"arm_scale_fit": analysis["arm_scale_fit"], "gripper_diff_range": GRIPPER_DIFF_RANGE})
    write_json(args.output_dir / "outcome_blind.json", {key: value for key, value in analysis.items() if not key.startswith("_")})
    write_json(
        args.output_dir / "h_temp_frozen.json",
        {
            "status": "frozen_before_closed_loop_comparison",
            "definition": protocol["metrics"]["heterogeneity"],
            "task_values": analysis["task_macro_H_temp_ranking"],
            "split_summary": analysis["split_summary"],
            "source_outcome_blind_artifact": str((args.output_dir / "outcome_blind.json").resolve()),
        },
    )
    write_csv(args.output_dir / "delay_profiles.csv", analysis["_summary_rows"])
    write_csv(args.output_dir / "episode_delay_profiles.csv", analysis["_episode_rows"])
    make_figures(analysis, args.output_dir / "figures")
    print(json.dumps({"status": "complete_outcome_blind_frozen", "tasks": len(analysis["profiles"]), "valid_rows": analysis["counts"]["valid_feature_rows"]}, indent=2))


if __name__ == "__main__":
    main()
