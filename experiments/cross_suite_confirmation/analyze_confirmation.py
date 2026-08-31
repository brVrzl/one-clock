"""Analyze the completed Branch K cross-suite confirmation rollout."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


METHODS = ("FRESH", "FO20", "REVERSE20", "FULL_OLD20", "HARD_H16")
DISPLAY = {
    "FRESH": "Fresh",
    "FO20": "FO20",
    "REVERSE20": "Reverse20",
    "FULL_OLD20": "FullOld20",
    "HARD_H16": "hard h16",
}
FIXED = {"FRESH", "FO20", "REVERSE20", "FULL_OLD20"}
BOOTSTRAP_DRAWS = 20_000


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    draws = values[indices].mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def task_bootstrap_ci(task_values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(task_values), size=(BOOTSTRAP_DRAWS, len(task_values)))
    draws = task_values[indices].mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def task_label(task: dict[str, Any]) -> str:
    return f"{task['suite']}:task{task['task_id']}"


def validate_episode(
    episode: dict[str, Any],
    method: str,
    task: dict[str, Any],
) -> None:
    if episode["method"] != method:
        raise AssertionError(f"method mismatch: {episode['method']} != {method}")
    if episode["suite"] != task["suite"] or int(episode["task_id"]) != int(task["task_id"]):
        raise AssertionError("task identity mismatch")
    state = int(episode["requested_initial_state_id"])
    if state not in range(14):
        raise AssertionError(f"unexpected state id {state}")
    expected_seed = 340000 + 1000 * {"libero_object": 0, "libero_goal": 2, "libero_10": 3}[task["suite"]] + 100 * int(task["task_id"]) + state
    if int(episode["environment_seed"]) != expected_seed:
        raise AssertionError(f"seed mismatch for {task_label(task)} state {state}")
    if int(episode["environment_construction_seed"]) != expected_seed:
        raise AssertionError("environment construction seed mismatch")
    if not episode["fresh_environment_instance"]:
        raise AssertionError("episode was not run in a fresh environment")
    if int(episode["max_episode_steps"]) != int(task["max_episode_steps"]):
        raise AssertionError("episode cap mismatch")
    steps = episode["step_log"]
    if len(steps) != int(episode["environment_steps"]):
        raise AssertionError("step log length mismatch")
    if len(steps) == 0:
        raise AssertionError("empty episode")
    if int(episode["policy_queries"]) != len(episode["query_steps"]):
        raise AssertionError("policy query count mismatch")
    observed_query_steps = [int(row["physical_target_t"]) for row in steps if row["policy_queried_at_t"]]
    if observed_query_steps != [int(x) for x in episode["query_steps"]]:
        raise AssertionError("query step log mismatch")
    for expected_t, row in enumerate(steps):
        t = int(row["physical_target_t"])
        if t != expected_t:
            raise AssertionError(f"non-sequential physical target at {expected_t}: {t}")
        action = np.asarray(row["action"], dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise AssertionError("invalid executed action")
        arm_q = int(row["arm_source_query_q"])
        arm_k = int(row["arm_chunk_offset"])
        grip_q = int(row["gripper_source_query_q"])
        grip_k = int(row["gripper_chunk_offset"])
        if arm_q + arm_k != t or grip_q + grip_k != t:
            raise AssertionError(f"same-target violation at t={t}")
        if int(row["arm_source_age"]) != t - arm_q or int(row["gripper_source_age"]) != t - grip_q:
            raise AssertionError(f"source-age mismatch at t={t}")
        if method in FIXED:
            if not row["policy_queried_at_t"] or int(row["query_physical_step_q"]) != t:
                raise AssertionError(f"dense query tag mismatch at t={t}")
        if method == "FRESH":
            expected = (t, 0, t, 0)
            actual = (arm_q, arm_k, grip_q, grip_k)
            if actual != expected:
                raise AssertionError(f"Fresh source mismatch at t={t}: {actual}")
        elif method == "FO20":
            expected = (t, 0, t, 0) if t < 20 else (t, 0, t - 20, 20)
            actual = (arm_q, arm_k, grip_q, grip_k)
            if actual != expected:
                raise AssertionError(f"FO20 source mismatch at t={t}: {actual}")
        elif method == "REVERSE20":
            expected = (t, 0, t, 0) if t < 20 else (t - 20, 20, t, 0)
            actual = (arm_q, arm_k, grip_q, grip_k)
            if actual != expected:
                raise AssertionError(f"Reverse20 source mismatch at t={t}: {actual}")
        elif method == "FULL_OLD20":
            expected = (t, 0, t, 0) if t < 20 else (t - 20, 20, t - 20, 20)
            actual = (arm_q, arm_k, grip_q, grip_k)
            if actual != expected:
                raise AssertionError(f"FullOld20 source mismatch at t={t}: {actual}")
        elif method == "HARD_H16":
            if not row["policy_queried_at_t"] and row["query_physical_step_q"] is not None:
                raise AssertionError("hard h16 has a query tag on a non-query step")
            if row["policy_queried_at_t"] and int(row["query_physical_step_q"]) != t:
                raise AssertionError("hard h16 query tag does not equal physical step")
            if arm_q % 16 != 0 or grip_q != arm_q or arm_k != grip_k:
                raise AssertionError(f"hard h16 source mismatch at t={t}")
            if arm_q != 16 * (t // 16) or arm_k != t - arm_q:
                raise AssertionError(f"hard h16 schedule mismatch at t={t}")
        if t < 20 and method in FIXED:
            fresh = np.asarray(row["fresh_action"], dtype=float)
            if fresh.shape != (7,) or not np.array_equal(action, fresh):
                raise AssertionError(f"fixed-source prefix differs from Fresh at t={t}")
    if method == "HARD_H16":
        expected_queries = list(range(0, len(steps), 16))
        if episode["query_steps"] != expected_queries:
            raise AssertionError("hard h16 query schedule is not 0,16,32,...")
        if int(episode["policy_queries"]) != int(np.ceil(len(steps) / 16)):
            raise AssertionError("hard h16 query count mismatch")


def load_results(protocol: dict[str, Any], results_root: Path) -> tuple[dict[tuple[str, int, int, str], int], dict[str, Any], list[dict[str, Any]]]:
    outcomes: dict[tuple[str, int, int, str], int] = {}
    metrics: dict[str, Any] = {}
    validation = {"episodes_checked": 0, "same_target": True, "fixed_semantics": True, "fixed_prefix": True, "hard_schedule": True}
    tasks = protocol["cohort"]["tasks"]
    for task in tasks:
        filename = f"{task['suite']}_task{task['task_id']}.json"
        path = results_root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("finished") or data.get("methods") != list(METHODS):
            raise AssertionError(f"incomplete or incompatible result file: {path}")
        if data.get("state_ids") != list(range(14)):
            raise AssertionError(f"state coverage mismatch: {path}")
        task_key = task_label(task)
        metrics[task_key] = {}
        for method in METHODS:
            episodes = data["episodes"].get(method, [])
            if len(episodes) != 14:
                raise AssertionError(f"{task_key} {method} has {len(episodes)} episodes")
            metrics[task_key][method] = episodes
            for episode in episodes:
                validate_episode(episode, method, task)
                validation["episodes_checked"] += 1
                key = (task["suite"], int(task["task_id"]), int(episode["requested_initial_state_id"]), method)
                if key in outcomes:
                    raise AssertionError(f"duplicate outcome {key}")
                outcomes[key] = int(bool(episode["success"]))
    expected = {
        (task["suite"], int(task["task_id"]), state, method)
        for task in tasks for state in range(14) for method in METHODS
    }
    if set(outcomes) != expected:
        raise AssertionError(f"outcome coverage mismatch: {len(outcomes)}")
    return outcomes, metrics, validation


def comparison(
    outcomes: dict[tuple[str, int, int, str], int],
    tasks: list[dict[str, Any]],
    first: str,
    second: str,
    paired_seed: int,
    cluster_seed: int,
) -> dict[str, Any]:
    keys = [(task["suite"], int(task["task_id"]), state) for task in tasks for state in range(14)]
    first_values = np.asarray([outcomes[key + (first,)] for key in keys], dtype=np.int8)
    second_values = np.asarray([outcomes[key + (second,)] for key in keys], dtype=np.int8)
    differences = first_values.astype(float) - second_values.astype(float)
    task_values = np.asarray([
        differences[[suite == task["suite"] and task_id == int(task["task_id"]) for suite, task_id, _ in keys]].mean()
        for task in tasks
    ])
    first_only = int(np.count_nonzero((first_values == 1) & (second_values == 0)))
    second_only = int(np.count_nonzero((first_values == 0) & (second_values == 1)))
    discordant = first_only + second_only
    return {
        "first_method": first,
        "second_method": second,
        "first_display": DISPLAY[first],
        "second_display": DISPLAY[second],
        "blocks": len(keys),
        "first_successes": int(first_values.sum()),
        "second_successes": int(second_values.sum()),
        "first_success_rate": float(first_values.mean()),
        "second_success_rate": float(second_values.mean()),
        "first_only_wins": first_only,
        "second_only_wins": second_only,
        "net_wins": first_only - second_only,
        "discordant_blocks": discordant,
        "success_delta": float(differences.mean()),
        "success_delta_percentage_points": float(100 * differences.mean()),
        "exact_two_sided_mcnemar_p": float(binomtest(first_only, discordant, 0.5).pvalue) if discordant else 1.0,
        "paired_bootstrap_draws": BOOTSTRAP_DRAWS,
        "paired_bootstrap_seed": paired_seed,
        "paired_bootstrap_ci": bootstrap_ci(differences, paired_seed),
        "task_cluster_bootstrap_draws": BOOTSTRAP_DRAWS,
        "task_cluster_bootstrap_seed": cluster_seed,
        "task_cluster_bootstrap_ci": task_bootstrap_ci(task_values, cluster_seed),
        "task_labels": [task_label(task) for task in tasks],
        "task_differences": task_values.tolist(),
        "leave_one_task_out": [
            {"omitted_task": task_label(tasks[i]), "delta": float(np.delete(task_values, i).mean())}
            for i in range(len(tasks))
        ],
    }


def aggregate_metrics(metrics: dict[str, Any], tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        episodes = [episode for task in tasks for episode in metrics[task_label(task)][method]]
        success = sum(int(bool(e["success"])) for e in episodes)
        steps = sum(len(e["step_log"]) for e in episodes)
        queries = sum(int(e["policy_queries"]) for e in episodes)
        ages = [int(row["gripper_source_age"]) for e in episodes for row in e["step_log"]]
        rows.append({
            "method": method,
            "display_name": DISPLAY[method],
            "blocks": len(episodes),
            "successes": success,
            "success_rate": success / len(episodes),
            "total_environment_steps": steps,
            "total_policy_queries": queries,
            "observed_query_rate": queries / steps,
            "observed_mean_gripper_source_age": float(np.mean(ages)),
            "observed_gripper_source_age_histogram": {str(age): ages.count(age) for age in sorted(set(ages))},
            "mean_wall_clock_seconds_per_episode": float(np.mean([e["wall_clock_seconds"] for e in episodes])),
            "mean_recorded_policy_call_latency_seconds": float(np.mean([
                e["mean_policy_call_latency_seconds"] for e in episodes
            ])),
        })
    return rows


def per_task_rows(metrics: dict[str, Any], tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        key = task_label(task)
        counts = {method: sum(int(bool(e["success"])) for e in metrics[key][method]) for method in METHODS}
        row: dict[str, Any] = {"suite": task["suite"], "task_id": task["task_id"], "task_label": key, "role": task["role"], "blocks": 14}
        for method in METHODS:
            row[f"{method}_successes"] = counts[method]
            row[f"{method}_success_rate"] = counts[method] / 14
        row["FO20_minus_REVERSE20"] = counts["FO20"] - counts["REVERSE20"]
        row["FO20_minus_FRESH"] = counts["FO20"] - counts["FRESH"]
        row["FO20_minus_FULL_OLD20"] = counts["FO20"] - counts["FULL_OLD20"]
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def comparison_table_lines(comparisons: dict[str, Any]) -> list[str]:
    lines = [
        "| Contrast | First success | Second success | First-only | Second-only | Net | Delta (pp) | Exact two-sided McNemar p | Paired 95% CI | Task-cluster 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons.values():
        lines.append(
            f"| {row['first_display']} vs {row['second_display']} | {row['first_successes']}/{row['blocks']} | {row['second_successes']}/{row['blocks']} | {row['first_only_wins']} | {row['second_only_wins']} | {row['net_wins']} | {row['success_delta_percentage_points']:.2f} | {row['exact_two_sided_mcnemar_p']:.6g} | [{row['paired_bootstrap_ci'][0]:.3f}, {row['paired_bootstrap_ci'][1]:.3f}] | [{row['task_cluster_bootstrap_ci'][0]:.3f}, {row['task_cluster_bootstrap_ci'][1]:.3f}] |"
        )
    return lines


def task_table_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Task | Fresh | FO20 | Reverse20 | FullOld20 | hard h16 | FO20−Reverse20 | FO20−Fresh | FO20−FullOld20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task_label']} | {row['FRESH_successes']}/14 | {row['FO20_successes']}/14 | {row['REVERSE20_successes']}/14 | {row['FULL_OLD20_successes']}/14 | {row['HARD_H16_successes']}/14 | {row['FO20_minus_REVERSE20']} | {row['FO20_minus_FRESH']} | {row['FO20_minus_FULL_OLD20']} |"
        )
    return lines


def loto_lines(comparisons: dict[str, Any]) -> list[str]:
    labels = list(comparisons)
    lines = ["| Omitted task | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    for i, item in enumerate(comparisons[labels[0]]["leave_one_task_out"]):
        values = [comparisons[label]["leave_one_task_out"][i]["delta"] for label in labels]
        lines.append("| " + item["omitted_task"] + " | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("experiments/cross_suite_confirmation/protocol.json"))
    parser.add_argument("--results-root", type=Path, default=Path("experiments/cross_suite_confirmation/results"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/cross_suite_confirmation"))
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    outcomes, metrics, validation = load_results(protocol, args.results_root)
    tasks = protocol["cohort"]["tasks"]
    primary_tasks = [task for task in tasks if task["role"] == "primary_unseen_to_executor_development"]
    bridge_tasks = [task for task in tasks if task["role"] == "bridge_exposed"]
    seeds = protocol["statistics"]["bootstrap_seeds"]

    primary_comparisons = {
        "FO20_VS_REVERSE20": comparison(outcomes, primary_tasks, "FO20", "REVERSE20", seeds["FO20_VS_REVERSE20"]["paired"], seeds["FO20_VS_REVERSE20"]["task_cluster"]),
        "FO20_VS_FRESH": comparison(outcomes, primary_tasks, "FO20", "FRESH", seeds["FO20_VS_FRESH"]["paired"], seeds["FO20_VS_FRESH"]["task_cluster"]),
        "FO20_VS_FULL_OLD20": comparison(outcomes, primary_tasks, "FO20", "FULL_OLD20", seeds["FO20_VS_FULL_OLD20"]["paired"], seeds["FO20_VS_FULL_OLD20"]["task_cluster"]),
    }
    bridge_comparisons = {
        "FO20_VS_REVERSE20": comparison(outcomes, bridge_tasks, "FO20", "REVERSE20", seeds["FO20_VS_REVERSE20"]["paired"], seeds["FO20_VS_REVERSE20"]["task_cluster"]),
        "FO20_VS_FRESH": comparison(outcomes, bridge_tasks, "FO20", "FRESH", seeds["FO20_VS_FRESH"]["paired"], seeds["FO20_VS_FRESH"]["task_cluster"]),
        "FO20_VS_FULL_OLD20": comparison(outcomes, bridge_tasks, "FO20", "FULL_OLD20", seeds["FO20_VS_FULL_OLD20"]["paired"], seeds["FO20_VS_FULL_OLD20"]["task_cluster"]),
    }
    primary_metrics = aggregate_metrics(metrics, primary_tasks)
    bridge_metrics = aggregate_metrics(metrics, bridge_tasks)
    rows = per_task_rows(metrics, tasks)

    analysis = {
        "schema_version": 1,
        "analysis_type": "frozen_branch_k_confirmation",
        "protocol": str(args.protocol.resolve()),
        "results_root": str(args.results_root.resolve()),
        "rollout": {"episodes_expected": 910, "episodes_analyzed": validation["episodes_checked"], "all_task_files_finished": True},
        "scope": {"primary": {"label": "unseen to executor development", "tasks": [task_label(task) for task in primary_tasks], "blocks": 140}, "bridge": {"label": "exposed bridge", "tasks": [task_label(task) for task in bridge_tasks], "blocks": 42}},
        "validation": validation,
        "primary_aggregate": primary_metrics,
        "bridge_aggregate": bridge_metrics,
        "primary_comparisons": primary_comparisons,
        "bridge_comparisons_descriptive": bridge_comparisons,
        "per_task": rows,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seeds": seeds},
        "interpretation_scope": "Primary inference is Goal + LIBERO-10 only; Object rows are bridge context and are never pooled into the primary aggregate.",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "analysis.json", analysis)
    write_csv(args.output_root / "per_task.csv", rows)

    def metric_table(metrics_rows: list[dict[str, Any]]) -> list[str]:
        denominator = metrics_rows[0]["blocks"]
        lines = ["| Method | Success | Success % | Observed query rate | Observed mean gripper age |", "|---|---:|---:|---:|---:|"]
        for row in metrics_rows:
            lines.append(f"| {row['display_name']} | {row['successes']}/{denominator} | {100*row['success_rate']:.1f}% | {row['observed_query_rate']:.5f} | {row['observed_mean_gripper_source_age']:.3f} |")
        return lines

    report: list[str] = [
        "# Branch K cross-suite confirmation",
        "",
        "## Scope and interpretation",
        "",
        "The confirmation tasks are unseen to executor development, not unseen to policy training: each per-task ACT checkpoint was trained for its corresponding task. Absolute success rates are therefore interpreted only within the confirmation experiment, and scientific inference uses paired executor contrasts.",
        "",
        "States 0..13 were selected by the deterministic outcome-independent rule `first 14 initialization states`; this does not claim that these numerical state IDs were globally unused.",
        "",
        "These are five frozen conditions: Fresh, FO20, Reverse20, FullOld20, and hard h16. The Branch K confirmation used 910 episodes: 140 primary blocks per method (Goal + LIBERO-10) and 42 exposed Object bridge blocks per method.",
        "",
        "## Checkpoint preflight and cohort",
        "",
        "The per-task 100k ACT checkpoint family was loadable for all 13 selected tasks. No missing-checkpoint contingency was invoked. The actual primary cohort is Goal tasks 4, 6, 7, 8, 9 and LIBERO-10 tasks 0, 2, 4, 6, 7, each with states 0..13 and the frozen seed rule `340000 + 1000*suite_index + 100*task_id + state_id`. The Object bridge cohort is tasks 1, 5, 9.",
        "",
        "## Semantic and rollout validation",
        "",
        f"The CPU analyzer checked {validation['episodes_checked']} episodes and every persisted step. It verified `source_q + offset = target_t`, the four fixed-source definitions, fixed-source Fresh prefixes through t=19, hard-h16 query steps 0,16,32,... with newest-chunk offsets, sequential targets, finite 7D actions, exact frozen seeds/caps, and fresh-environment metadata.",
        "",
        "The pre-outcome semantic suite passed 3/3 tests and the required pairing smoke passed for one Goal task x3 states and one LIBERO-10 task x3 states before rollout. All three outcome shards completed without interruption; no C1/C2 or HARD_H16 rerun was performed.",
        "",
        "## Primary unseen-to-executor-development aggregate",
        "",
        *metric_table(primary_metrics),
        "",
        "The observed query rate is total policy calls divided by total environment steps within the indicated 140-block primary aggregate. Gripper age is step-weighted over the realized episode trajectories.",
        "",
        "## Primary paired contrasts",
        "",
        *comparison_table_lines(primary_comparisons),
        "",
        "McNemar p-values are exact two-sided binomial tests on discordant paired task-state blocks. Bootstrap intervals use the preregistered 20,000 draws and seeds. Task-cluster intervals resample task-level mean differences; leave-one-task-out values are reported below.",
        "",
        "### Primary per-task results",
        "",
        *task_table_lines([row for row in rows if row["role"] == "primary_unseen_to_executor_development"]),
        "",
        "### Primary leave-one-task-out deltas",
        "",
        *loto_lines(primary_comparisons),
        "",
        "## Object bridge context",
        "",
        "Object tasks 1, 5, and 9 are exposed bridge tasks only and are not pooled into the primary inference set.",
        "",
        *metric_table(bridge_metrics),
        "",
        "### Bridge paired contrasts (descriptive)",
        "",
        *comparison_table_lines(bridge_comparisons),
        "",
        "### Bridge per-task results",
        "",
        *task_table_lines([row for row in rows if row["role"] == "bridge_exposed"]),
        "",
        "### Bridge leave-one-task-out deltas",
        "",
        *loto_lines(bridge_comparisons),
        "",
        "## Spatial context",
        "",
        "The completed preregistered Gate-4A2 Spatial reanalysis is reported in `experiments/gate4a2_spatial_analysis/report.md`. It is independent second-suite context for FO20, but it has no Reverse20 and no hard-h16 baseline, so it cannot establish the full arm-versus-gripper factorial asymmetry. Its suite-level checkpoint also differs from this confirmation checkpoint family, so absolute success rates are not compared across experiments.",
        "",
        "## Paper artifact registration",
        "",
        "The `{C2, hard h16, C1}` three-point gripper-age table from the completed development gate remains the executor-decomposition artifact. This Branch K table is the frozen confirmation artifact for the five-condition same-target comparison, with primary inference restricted to the unseen-to-executor-development Goal + LIBERO-10 cohort and Object reported separately as bridge context.",
        "",
        "Method development is closed after this confirmation. Negative or heterogeneous results are retained as paper results; no rescue executor or additional rollout is authorized by this protocol.",
        "",
    ]
    (args.output_root / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"analysis": str((args.output_root / "analysis.json").resolve()), "episodes": validation["episodes_checked"], "primary_blocks": 140, "bridge_blocks": 42}, indent=2))


if __name__ == "__main__":
    main()
