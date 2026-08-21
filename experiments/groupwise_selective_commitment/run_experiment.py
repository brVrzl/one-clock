#!/usr/bin/env python3
"""Run the matched-query LIBERO-Object selective-commitment gate."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.groupwise_selective_commitment.executor import (  # noqa: E402
    CommitGroup,
    ScheduledCommitExecutor,
)
from experiments.group_prediction_persistence.audit import (  # noqa: E402
    load_action_normalization,
)
from scripts.run_libero_gate0 import (  # noqa: E402
    batch_robot_state,
    git_commit,
    load_config,
    load_policy_and_processors,
    prepare_policy_observation,
    query_full_act_chunk,
    set_episode_seed,
)


OUTPUT_DIR = ROOT / "experiments/groupwise_selective_commitment"
CHECKPOINT = Path("/home/thor/projects/checkpoints/zeromidnight_act_libero_object")
CONFIG_PATH = ROOT / "configs/gate0_libero_object.yaml"
TASK_IDS = tuple(range(10))
INIT_STATE_IDS = tuple(range(20))
QUERY_CADENCES = (4, 8, 16)
METHODS = ("global_replace", "selective_commit")
BASE_SEED = 1000
BOOTSTRAP_SEED = 20260820
BOOTSTRAP_DRAWS = 10_000
CHUNK_SIZE = 100
ACTION_DIM = 7
GROUPS = (
    CommitGroup("arm", tuple(range(6))),
    CommitGroup("gripper", (6,)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--episodes-per-task", type=int, default=20)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = z * np.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials**2)) / denominator
    return (float(center - radius), float(center + radius))


def paired_bootstrap_delta(
    a_success: np.ndarray,
    b_success: np.ndarray,
    *,
    seed: int,
) -> tuple[float, float, float]:
    if a_success.shape != b_success.shape or a_success.ndim != 1:
        raise ValueError("paired success arrays must be matching vectors")
    delta = b_success.astype(np.float64) - a_success.astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(BOOTSTRAP_DRAWS, len(delta)))
    boot = delta[draws].mean(axis=1)
    return float(delta.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def make_env_and_policy(config: dict[str, Any], checkpoint: Path, task_id: int) -> tuple[Any, Any, Any, Any, Any, Any]:
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv

    suite_name = str(config["task_suite"])
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    runtime_config = dict(config)
    runtime_config["task_id"] = task_id
    runtime_config["task_name"] = task.name
    policy, policy_preprocessor, policy_postprocessor, env_preprocessor, env_postprocessor = (
        load_policy_and_processors(runtime_config, checkpoint)
    )
    env = LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=suite_name,
        obs_type=str(config["obs_type"]),
        camera_name=str(config["camera_name"]),
        camera_name_mapping=dict(config["camera_name_mapping"]),
        observation_width=int(config["observation_width"]),
        observation_height=int(config["observation_height"]),
        control_freq=int(config.get("control_freq", 20)),
        init_states=bool(config["init_states"]),
        hard_reset=bool(config["hard_reset"]),
        control_mode=str(config["control_mode"]),
    )
    return env, policy, policy_preprocessor, policy_postprocessor, env_preprocessor, env_postprocessor


def normalized_discontinuity(previous: np.ndarray, current: np.ndarray, action_std: np.ndarray) -> dict[str, float]:
    delta = (np.asarray(current, dtype=np.float64) - np.asarray(previous, dtype=np.float64)) / action_std
    return {
        "overall": float(np.sqrt(np.mean(delta**2))),
        "arm": float(np.sqrt(np.mean(delta[:6] ** 2))),
        "gripper": float(abs(delta[6])),
    }


def run_episode(
    *,
    env: Any,
    policy: Any,
    policy_preprocessor: Any,
    policy_postprocessor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    method: str,
    query_cadence: int,
    action_std: np.ndarray,
    init_state_id: int,
    seed: int,
    step_log: Any,
    task_id: int,
) -> dict[str, Any]:
    set_episode_seed(seed)
    env.init_state_id = init_state_id
    observation, _ = env.reset(seed=seed)
    policy.reset()
    executor = ScheduledCommitExecutor(
        method=method,  # type: ignore[arg-type]
        query_cadence=query_cadence,
        chunk_size=CHUNK_SIZE,
        action_dim=ACTION_DIM,
        groups=GROUPS,
        action_std=action_std,
    )
    records: list[dict[str, Any]] = []
    query_steps: list[int] = []
    discontinuities: dict[str, list[float]] = {"overall": [], "arm": [], "gripper": []}
    group_switches = {"arm": 0, "gripper": 0}
    previous_generation: dict[str, int | None] = {"arm": None, "gripper": None}
    acceptance_counts = {"both_accept": 0, "only_arm_accept": 0, "only_gripper_accept": 0, "neither_accept": 0}
    age_sums = {"arm": 0, "gripper": 0}
    info: dict[str, Any] = {"is_success": False}
    previous_action: np.ndarray | None = None

    for step in range(int(env._max_episode_steps)):
        def query() -> np.ndarray:
            return query_full_act_chunk(
                observation=observation,
                policy=policy,
                policy_preprocessor=policy_preprocessor,
                policy_postprocessor=policy_postprocessor,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
            )

        decision = executor.step(query)
        if decision.environment_step != step:
            raise RuntimeError("executor/environment step mismatch")
        if decision.policy_query != (step % query_cadence == 0):
            raise RuntimeError("query cadence violated")
        if decision.policy_query:
            query_steps.append(step)
            accepted = {name for name, value in decision.acceptance.items() if value == "accept"}
            if accepted == {"arm", "gripper"}:
                acceptance_counts["both_accept"] += 1
            elif accepted == {"arm"}:
                acceptance_counts["only_arm_accept"] += 1
            elif accepted == {"gripper"}:
                acceptance_counts["only_gripper_accept"] += 1
            elif not accepted:
                acceptance_counts["neither_accept"] += 1
            else:
                raise RuntimeError(f"unexpected acceptance set: {accepted}")
        for group in ("arm", "gripper"):
            generation = decision.current_source_generation_ids[group]
            prior = previous_generation[group]
            if prior is not None and generation != prior:
                group_switches[group] += 1
            previous_generation[group] = generation
            age_sums[group] += decision.local_source_ages[group]

        if previous_action is not None and decision.policy_query:
            for name, value in normalized_discontinuity(previous_action, decision.action, action_std).items():
                discontinuities[name].append(value)
        previous_action = decision.action.copy()

        observation, _, terminated, truncated, info = env.step(decision.action.astype(np.float32))
        log_record = decision.as_log_record()
        log_record.update(
            {
                "task_id": task_id,
                "init_state_id": init_state_id,
                "seed": seed,
                "method": method,
                "query_cadence": query_cadence,
                "is_success": bool(info["is_success"]),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        step_log.write(json.dumps(log_record, separators=(",", ":")) + "\n")
        records.append(log_record)
        if terminated or truncated:
            break

    environment_steps = len(records)
    policy_queries = sum(int(row["query_occurred"]) for row in records)
    success = bool(info["is_success"])
    return {
        "task_id": task_id,
        "init_state_id": init_state_id,
        "seed": seed,
        "method": method,
        "query_cadence": query_cadence,
        "success": success,
        "environment_steps": environment_steps,
        "policy_queries": policy_queries,
        "queries_per_executed_step": policy_queries / environment_steps,
        "query_steps": query_steps,
        "group_switches_arm": group_switches["arm"],
        "group_switches_gripper": group_switches["gripper"],
        "acceptance_counts": acceptance_counts,
        "mean_age_arm": age_sums["arm"] / environment_steps,
        "mean_age_gripper": age_sums["gripper"] / environment_steps,
        "discontinuity_counts": {key: len(value) for key, value in discontinuities.items()},
        "discontinuity_mean": {
            key: float(np.mean(value)) if value else None for key, value in discontinuities.items()
        },
        "discontinuity_median": {
            key: float(np.median(value)) if value else None for key, value in discontinuities.items()
        },
        "source_exhaustion_steps": sum(
            int(row["source_chunk_exhausted"]["arm"] or row["source_chunk_exhausted"]["gripper"])
            for row in records
        ),
    }


def group_rows(rows: Iterable[dict[str, Any]], *, task_id: int | None = None, q: int | None = None, method: str | None = None) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if (task_id is None or int(row["task_id"]) == task_id)
        and (q is None or int(row["query_cadence"]) == q)
        and (method is None or row["method"] == method)
    ]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty row set")
    episodes = len(rows)
    environment_steps = sum(int(row["environment_steps"]) for row in rows)
    policy_queries = sum(int(row["policy_queries"]) for row in rows)
    disc_values = {
        key: [row["discontinuity_mean"][key] for row in rows if row["discontinuity_mean"][key] is not None]
        for key in ("overall", "arm", "gripper")
    }
    accept_totals = {key: sum(int(row["acceptance_counts"][key]) for row in rows) for key in (
        "both_accept", "only_arm_accept", "only_gripper_accept", "neither_accept"
    )}
    switch_totals = {
        "arm": sum(int(row["group_switches_arm"]) for row in rows),
        "gripper": sum(int(row["group_switches_gripper"]) for row in rows),
    }
    total_queries = sum(accept_totals.values())
    return {
        "episodes": episodes,
        "successes": sum(int(row["success"]) for row in rows),
        "success_rate": float(np.mean([bool(row["success"]) for row in rows])),
        "success_rate_ci95": list(wilson_interval(sum(int(row["success"]) for row in rows), episodes)),
        "environment_steps": environment_steps,
        "mean_environment_steps": environment_steps / episodes,
        "policy_queries": policy_queries,
        "queries_per_rollout": policy_queries / episodes,
        "queries_per_executed_step": policy_queries / environment_steps,
        "mean_group_generation_switches": {
            "arm": switch_totals["arm"] / episodes,
            "gripper": switch_totals["gripper"] / episodes,
        },
        "acceptance_counts": accept_totals,
        "acceptance_fractions": {
            key: value / total_queries if total_queries else None for key, value in accept_totals.items()
        },
        "mean_retained_generation_age": {
            "arm": float(np.mean([row["mean_age_arm"] for row in rows])),
            "gripper": float(np.mean([row["mean_age_gripper"] for row in rows])),
        },
        "action_discontinuity_mean": {
            key: float(np.mean(value)) if value else None for key, value in disc_values.items()
        },
        "action_discontinuity_median": {
            key: float(np.median(value)) if value else None for key, value in disc_values.items()
        },
        "source_exhaustion_steps": sum(int(row["source_exhaustion_steps"]) for row in rows),
    }


def aggregate_results(
    *,
    episode_rows: list[dict[str, Any]],
    output_dir: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    per_task_rows: list[dict[str, Any]] = []
    per_q_rows: list[dict[str, Any]] = []
    acceptance_rows: list[dict[str, Any]] = []
    discontinuity_rows: list[dict[str, Any]] = []
    paired_by_q: dict[str, Any] = {}
    for q in QUERY_CADENCES:
        q_rows = group_rows(episode_rows, q=q)
        a_rows = group_rows(q_rows, method="global_replace")
        b_rows = group_rows(q_rows, method="selective_commit")
        a_by_key = {(int(row["task_id"]), int(row["init_state_id"])): row for row in a_rows}
        b_by_key = {(int(row["task_id"]), int(row["init_state_id"])): row for row in b_rows}
        keys = sorted(set(a_by_key) & set(b_by_key))
        a_success = np.asarray([a_by_key[key]["success"] for key in keys], dtype=bool)
        b_success = np.asarray([b_by_key[key]["success"] for key in keys], dtype=bool)
        delta, ci_low, ci_high = paired_bootstrap_delta(a_success, b_success, seed=BOOTSTRAP_SEED + q)
        paired_by_q[str(q)] = {
            "pairs": len(keys),
            "selective_minus_global_success_rate": delta,
            "bootstrap_95ci": [ci_low, ci_high],
            "bootstrap_seed": BOOTSTRAP_SEED + q,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
        }
        for method in METHODS:
            summary = summarize_rows(group_rows(q_rows, method=method))
            per_q_rows.append(
                {
                    "q": q,
                    "method": method,
                    "successes": summary["successes"],
                    "episodes": summary["episodes"],
                    "success_rate": summary["success_rate"],
                    "success_rate_ci95_low": summary["success_rate_ci95"][0],
                    "success_rate_ci95_high": summary["success_rate_ci95"][1],
                    "queries_per_rollout": summary["queries_per_rollout"],
                    "queries_per_executed_step": summary["queries_per_executed_step"],
                    "mean_arm_generation_switches": summary["mean_group_generation_switches"]["arm"],
                    "mean_gripper_generation_switches": summary["mean_group_generation_switches"]["gripper"],
                    "mean_arm_age": summary["mean_retained_generation_age"]["arm"],
                    "mean_gripper_age": summary["mean_retained_generation_age"]["gripper"],
                    "overall_discontinuity_mean": summary["action_discontinuity_mean"]["overall"],
                    "arm_discontinuity_mean": summary["action_discontinuity_mean"]["arm"],
                    "gripper_discontinuity_mean": summary["action_discontinuity_mean"]["gripper"],
                    "source_exhaustion_steps": summary["source_exhaustion_steps"],
                    "paired_selective_minus_global_delta": delta,
                    "paired_bootstrap_ci_low": ci_low,
                    "paired_bootstrap_ci_high": ci_high,
                }
            )
            acceptance_rows.append(
                {
                    "q": q,
                    "method": method,
                    "queries": summary["policy_queries"],
                    **{f"{key}_count": value for key, value in summary["acceptance_counts"].items()},
                    **{f"{key}_fraction": value for key, value in summary["acceptance_fractions"].items()},
                    "mean_arm_generation_switches": summary["mean_group_generation_switches"]["arm"],
                    "mean_gripper_generation_switches": summary["mean_group_generation_switches"]["gripper"],
                    "mean_arm_retained_generation_age": summary["mean_retained_generation_age"]["arm"],
                    "mean_gripper_retained_generation_age": summary["mean_retained_generation_age"]["gripper"],
                }
            )
            discontinuity_rows.append(
                {
                    "q": q,
                    "method": method,
                    "query_boundary_count": sum(
                        int(row["discontinuity_counts"]["overall"])
                        for row in group_rows(q_rows, method=method)
                    ),
                    **{
                        f"{key}_mean": summary["action_discontinuity_mean"][key]
                        for key in ("overall", "arm", "gripper")
                    },
                    **{
                        f"{key}_median": summary["action_discontinuity_median"][key]
                        for key in ("overall", "arm", "gripper")
                    },
                }
            )
        for task_id in TASK_IDS:
            task_q_rows = group_rows(q_rows, task_id=task_id)
            task_a = group_rows(task_q_rows, method="global_replace")
            task_b = group_rows(task_q_rows, method="selective_commit")
            a_map = {int(row["init_state_id"]): row for row in task_a}
            b_map = {int(row["init_state_id"]): row for row in task_b}
            task_keys = sorted(set(a_map) & set(b_map))
            task_a_success = np.asarray([a_map[key]["success"] for key in task_keys], dtype=bool)
            task_b_success = np.asarray([b_map[key]["success"] for key in task_keys], dtype=bool)
            task_delta, task_low, task_high = paired_bootstrap_delta(
                task_a_success, task_b_success, seed=BOOTSTRAP_SEED + q * 100 + task_id
            )
            for method, rows in (("global_replace", task_a), ("selective_commit", task_b)):
                summary = summarize_rows(rows)
                per_task_rows.append(
                    {
                        "task_id": task_id,
                        "q": q,
                        "method": method,
                        "episodes": summary["episodes"],
                        "successes": summary["successes"],
                        "success_rate": summary["success_rate"],
                        "success_rate_ci95_low": summary["success_rate_ci95"][0],
                        "success_rate_ci95_high": summary["success_rate_ci95"][1],
                        "paired_selective_minus_global_delta": task_delta,
                        "paired_bootstrap_ci_low": task_low,
                        "paired_bootstrap_ci_high": task_high,
                    }
                )

    write_csv(output_dir / "per_task.csv", per_task_rows)
    write_csv(output_dir / "per_q.csv", per_q_rows)
    write_csv(output_dir / "acceptance_statistics.csv", acceptance_rows)
    write_csv(output_dir / "discontinuity_statistics.csv", discontinuity_rows)
    figures = make_figures(output_dir / "figures", per_q_rows, acceptance_rows, discontinuity_rows)
    metrics = {
        "status": "completed",
        "verdict": None,
        "metadata": metadata,
        "paired_success_by_q": paired_by_q,
        "pooled_summaries": {
            str(q): {method: summarize_rows(group_rows(episode_rows, q=q, method=method)) for method in METHODS}
            for q in QUERY_CADENCES
        },
        "task_count": len(TASK_IDS),
        "episodes_per_task": metadata["episodes_per_task"],
        "total_primary_rollouts": len(episode_rows),
        "figures": figures,
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def make_figures(figures_dir: Path, per_q_rows: list[dict[str, Any]], acceptance_rows: list[dict[str, Any]], discontinuity_rows: list[dict[str, Any]]) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    for filename, y_key, title, ylabel in (
        ("success_by_q.png", "success_rate", "Matched-query success by cadence", "success rate"),
        ("discontinuity_by_q.png", "overall_discontinuity_mean", "Query-boundary discontinuity", "normalized RMS discontinuity"),
    ):
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        for method, style in (("global_replace", "o-"), ("selective_commit", "s-")):
            rows = [row for row in per_q_rows if row["method"] == method]
            ax.plot([row["q"] for row in rows], [row[y_key] for row in rows], style, label=method)
        ax.set_xlabel("global query cadence q")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(list(QUERY_CADENCES))
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        path = figures_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    selective = [row for row in acceptance_rows if row["method"] == "selective_commit"]
    bottoms = np.zeros(len(selective))
    for key, label, color in (
        ("both_accept_fraction", "both accept", "tab:blue"),
        ("only_arm_accept_fraction", "arm only", "tab:orange"),
        ("only_gripper_accept_fraction", "gripper only", "tab:green"),
        ("neither_accept_fraction", "neither", "tab:red"),
    ):
        values = np.asarray([row[key] or 0.0 for row in selective])
        ax.bar([str(row["q"]) for row in selective], values, bottom=bottoms, label=label, color=color)
        bottoms += values
    ax.set_xlabel("global query cadence q")
    ax.set_ylabel("fraction of fresh queries")
    ax.set_title("Selective-commit acceptance outcomes")
    ax.legend(frameon=False, ncol=2)
    path = figures_dir / "acceptance_outcomes.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))
    return paths


def main() -> None:
    args = parse_args()
    if args.episodes_per_task != 20:
        raise ValueError("the primary run is predeclared at exactly 20 rollouts per task")
    config = load_config(args.config)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir = output_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    normalization = load_action_normalization(checkpoint)
    action_std = np.asarray(normalization["std"], dtype=np.float64)
    if action_std.shape != (ACTION_DIM,):
        raise RuntimeError(f"expected seven action standard deviations, got {action_std.shape}")

    episode_rows: list[dict[str, Any]] = []
    log_manifest: list[dict[str, Any]] = []
    runtime_metadata: dict[str, Any] = {
        "canonical_start_sha": "6ed5d06516aaddb382095e3343430c7e31cd22d7",
        "repository_commit_at_run": git_commit(ROOT),
        "task_suite": "libero_object",
        "task_ids": list(TASK_IDS),
        "init_state_ids": list(INIT_STATE_IDS),
        "episodes_per_task": args.episodes_per_task,
        "base_seed": BASE_SEED,
        "methods": list(METHODS),
        "query_cadences": list(QUERY_CADENCES),
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": sha256(checkpoint / "model.safetensors"),
        "action_groups": {"arm": list(range(6)), "gripper": [6]},
        "action_std": action_std.tolist(),
        "epsilon_g": {"arm": 1.0, "gripper": 1.0},
        "action_distance": "arm=max(translation dataset-std RMS, rotation dataset-std RMS); gripper=normalized absolute error with sign mismatch promoted above epsilon",
        "control_mode": config["control_mode"],
        "control_frequency": int(config["control_freq"]),
        "chunk_size": CHUNK_SIZE,
        "action_dim": ACTION_DIM,
        "no_y_refresh_online": True,
        "no_future_observation": True,
        "learned_reliability_estimator": None,
        "optional_static_control": None,
    }
    write_json(output_dir / "paired_seed_manifest.json", {
        "base_seed": BASE_SEED,
        "task_ids": list(TASK_IDS),
        "init_state_ids": list(INIT_STATE_IDS),
        "seed_rule": "seed = base_seed + init_state_id",
        "pair_key": "(task_id, query_cadence, init_state_id)",
        "methods": list(METHODS),
        "query_cadences": list(QUERY_CADENCES),
        "episodes_per_task": args.episodes_per_task,
    })

    for task_id in TASK_IDS:
        env, policy, policy_preprocessor, policy_postprocessor, env_preprocessor, env_postprocessor = make_env_and_policy(
            config, checkpoint, task_id
        )
        try:
            official_count = len(env._init_states)
            if max(INIT_STATE_IDS) >= official_count:
                raise RuntimeError(f"task {task_id} has only {official_count} official initial states")
            for q in QUERY_CADENCES:
                handles: dict[str, Any] = {}
                episode_paths: dict[str, Path] = {}
                for method in METHODS:
                    run_dir = rollout_dir / f"task_{task_id:02d}" / f"q_{q:02d}" / method
                    run_dir.mkdir(parents=True, exist_ok=True)
                    step_path = run_dir / "steps.jsonl.gz"
                    episode_path = run_dir / "episodes.jsonl"
                    handles[method] = gzip.open(step_path, "wt", encoding="utf-8")
                    episode_paths[method] = episode_path
                episode_files = {method: episode_paths[method].open("w", encoding="utf-8") for method in METHODS}
                try:
                    for init_state_id in INIT_STATE_IDS:
                        seed = BASE_SEED + init_state_id
                        for method in METHODS:
                            result = run_episode(
                                env=env,
                                policy=policy,
                                policy_preprocessor=policy_preprocessor,
                                policy_postprocessor=policy_postprocessor,
                                env_preprocessor=env_preprocessor,
                                env_postprocessor=env_postprocessor,
                                method=method,
                                query_cadence=q,
                                action_std=action_std,
                                init_state_id=init_state_id,
                                seed=seed,
                                step_log=handles[method],
                                task_id=task_id,
                            )
                            episode_rows.append(result)
                            episode_files[method].write(json.dumps(result) + "\n")
                finally:
                    for handle in episode_files.values():
                        handle.close()
                    for handle in handles.values():
                        handle.close()
                for method in METHODS:
                    run_dir = episode_paths[method].parent
                    log_manifest.append(
                        {
                            "task_id": task_id,
                            "q": q,
                            "method": method,
                            "steps_log": str(run_dir / "steps.jsonl.gz"),
                            "steps_log_sha256": sha256(run_dir / "steps.jsonl.gz"),
                            "episodes_log": str(run_dir / "episodes.jsonl"),
                            "episodes_log_sha256": sha256(run_dir / "episodes.jsonl"),
                        }
                    )
        finally:
            env.close()

    runtime_metadata["lerobot_version"] = importlib.metadata.version("lerobot")
    runtime_metadata["libero_version"] = importlib.metadata.version("hf-libero")
    runtime_metadata["python"] = platform.python_version()
    runtime_metadata["architecture"] = platform.machine()
    import torch

    runtime_metadata["torch"] = torch.__version__
    runtime_metadata["torch_cuda"] = torch.version.cuda
    runtime_metadata["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    write_json(output_dir / "rollout_log_manifest.json", {"logs": log_manifest})
    metrics = aggregate_results(episode_rows=episode_rows, output_dir=output_dir, metadata=runtime_metadata)
    metrics["rollout_log_manifest"] = str(output_dir / "rollout_log_manifest.json")
    write_json(output_dir / "metrics.json", metrics)
    print(json.dumps({"status": "completed", "episodes": len(episode_rows), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
