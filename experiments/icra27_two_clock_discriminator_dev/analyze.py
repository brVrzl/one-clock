"""Analyze the two new fixed-clock conditions with exact reused H16 and C1 outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
FACTORIAL_ROOT = REPO_ROOT / "experiments" / "group_delay_factorial_act20"
ASYM_ROOT = REPO_ROOT / "experiments" / "asymmetric_chunk_reuse_dev"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from fixed_clock_executor import H16, H32, H32_COHERENT, METHODS, TWO_CLOCK  # noqa: E402
from validate_shard import validate_shard  # noqa: E402


H16_COHERENT = "H16_COHERENT"
C1 = "C1_PREVIOUS_CHUNK_GRIP"
ALL_METHODS = (H16_COHERENT, H32_COHERENT, TWO_CLOCK, C1)
PRIMARY_METHODS = (H16_COHERENT, H32_COHERENT, TWO_CLOCK)
COMPARISONS = (
    ("TWO_CLOCK_VS_H16", TWO_CLOCK, H16_COHERENT),
    ("TWO_CLOCK_VS_H32", TWO_CLOCK, H32_COHERENT),
    ("H32_VS_H16", H32_COHERENT, H16_COHERENT),
)
BOOTSTRAP_DRAWS = 20_000


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def paired_bootstrap(differences: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(BOOTSTRAP_DRAWS, len(differences)))
    draws = differences[indices].mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def cluster_bootstrap(task_differences: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(task_differences), size=(BOOTSTRAP_DRAWS, len(task_differences)))
    draws = task_differences[indices].mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def comparison_statistics(
    outcomes: dict[tuple[int, int, str], int],
    tasks: list[int],
    states: list[int],
    first: str,
    second: str,
    paired_seed: int,
    cluster_seed: int,
) -> dict[str, Any]:
    keys = [(task, state) for task in tasks for state in states]
    first_values = np.asarray([outcomes[(task, state, first)] for task, state in keys], dtype=np.int8)
    second_values = np.asarray([outcomes[(task, state, second)] for task, state in keys], dtype=np.int8)
    differences = first_values.astype(np.float64) - second_values.astype(np.float64)
    task_differences = np.asarray([
        np.mean([outcomes[(task, state, first)] - outcomes[(task, state, second)] for state in states])
        for task in tasks
    ])
    first_only = int(np.count_nonzero((first_values == 1) & (second_values == 0)))
    second_only = int(np.count_nonzero((first_values == 0) & (second_values == 1)))
    discordant = first_only + second_only
    p_value = float(binomtest(first_only, discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "first_method": first,
        "second_method": second,
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
        "exact_two_sided_mcnemar_p": p_value,
        "mcnemar_role": "descriptive development statistic",
        "paired_bootstrap_draws": BOOTSTRAP_DRAWS,
        "paired_bootstrap_seed": paired_seed,
        "paired_bootstrap_ci": paired_bootstrap(differences, paired_seed),
        "task_cluster_bootstrap_draws": BOOTSTRAP_DRAWS,
        "task_cluster_bootstrap_seed": cluster_seed,
        "task_cluster_bootstrap_ci": cluster_bootstrap(task_differences, cluster_seed),
        "task_ids": tasks,
        "task_differences": task_differences.tolist(),
        "leave_one_task_out": [float(np.delete(task_differences, index).mean()) for index in range(len(tasks))],
    }


def episode_record(episode: dict[str, Any], method: str, task_id: int, state_id: int) -> dict[str, Any]:
    if method in METHODS:
        rows = episode["step_log"]
        arm_ages = [int(row["arm_source_age"]) for row in rows]
        grip_ages = [int(row["gripper_source_age"]) for row in rows]
    elif method == H16_COHERENT:
        rows = episode["step_log"]
        arm_ages = [int(row["arm_source_age"]) for row in rows]
        grip_ages = [int(row["gripper_source_age"]) for row in rows]
    elif method == C1:
        rows = episode["step_log"]
        arm_ages = [int(row["arm_source_age"]) for row in rows]
        grip_ages = [int(row["gripper_source_age"]) for row in rows]
    else:
        raise ValueError(method)
    return {
        "task_id": task_id,
        "state_id": state_id,
        "method": method,
        "success": bool(episode["success"]),
        "environment_seed": int(episode["environment_seed"]),
        "environment_steps": int(episode["environment_steps"]),
        "policy_queries": int(episode["policy_queries"]),
        "query_rate": float(episode["query_rate"]),
        "arm_ages": arm_ages,
        "grip_ages": grip_ages,
    }


def collect_new(protocol: dict[str, Any], output_root: Path) -> dict[tuple[int, int, str], dict[str, Any]]:
    tasks = [int(value) for value in protocol["cohort"]["primary_task_ids"]]
    states = [int(value) for value in protocol["cohort"]["state_ids"]]
    records: dict[tuple[int, int, str], dict[str, Any]] = {}
    for task_id in tasks:
        path = output_root / "results" / f"task_{task_id:02d}.json"
        validate_shard(path, ROOT / "protocol.json")
        result = json.loads(path.read_text(encoding="utf-8"))
        for method in METHODS:
            for episode in result["episodes"][method]:
                state_id = int(episode["requested_initial_state_id"])
                key = (task_id, state_id, method)
                if key in records:
                    raise RuntimeError(f"duplicate new result cell: {key}")
                records[key] = episode_record(episode, method, task_id, state_id)
    expected = {(task, state, method) for task in tasks for state in states for method in METHODS}
    if set(records) != expected:
        raise RuntimeError("new result coverage is not exactly 9 x 14 x 2")
    return records


def validate_historical_identity(episode: dict[str, Any], task_id: int, state_id: int, method: str) -> None:
    seed = 330000 + 100 * task_id + state_id
    if int(episode["environment_seed"]) != seed or episode["fresh_environment_instance"] is not True:
        raise RuntimeError(f"historical {method} pairing identity mismatch")
    if int(episode["policy_rng_seed"]) != 424242 or int(episode["max_episode_steps"]) != 280:
        raise RuntimeError(f"historical {method} runtime identity mismatch")


def collect_historical(protocol: dict[str, Any]) -> dict[tuple[int, int, str], dict[str, Any]]:
    tasks = [int(value) for value in protocol["cohort"]["primary_task_ids"]]
    states = [int(value) for value in protocol["cohort"]["state_ids"]]
    records: dict[tuple[int, int, str], dict[str, Any]] = {}
    for task_id in tasks:
        factorial = json.loads((FACTORIAL_ROOT / "results" / f"task_{task_id:02d}.json").read_text(encoding="utf-8"))
        if factorial["methods"] != ["FRESH", "FO20", "REVERSE20", "FULL_OLD20", "HARD_H16"]:
            raise RuntimeError("historical H16 result identity drifted")
        for episode, state_id in zip(factorial["episodes"]["HARD_H16"], states, strict=True):
            validate_historical_identity(episode, task_id, state_id, H16_COHERENT)
            if episode["query_steps"] != list(range(0, int(episode["environment_steps"]), H16)):
                raise RuntimeError("historical H16 query schedule drifted")
            for t, row in enumerate(episode["step_log"]):
                q = int(row["arm_source_query_q"])
                k = int(row["arm_chunk_offset"])
                if q + k != t or int(row["gripper_source_query_q"]) != q or int(row["gripper_chunk_offset"]) != k:
                    raise RuntimeError("historical H16 same-target semantics drifted")
            records[(task_id, state_id, H16_COHERENT)] = episode_record(episode, H16_COHERENT, task_id, state_id)

        asym = json.loads((ASYM_ROOT / "results" / f"task_{task_id:02d}.json").read_text(encoding="utf-8"))
        if C1 not in asym["methods"]:
            raise RuntimeError("historical C1 result identity drifted")
        for episode, state_id in zip(asym["episodes"][C1], states, strict=True):
            validate_historical_identity(episode, task_id, state_id, C1)
            if episode["query_steps"] != list(range(0, int(episode["environment_steps"]), H16)):
                raise RuntimeError("historical C1 query schedule drifted")
            for t, row in enumerate(episode["step_log"]):
                if int(row["arm_source_q"]) + int(row["arm_offset"]) != t:
                    raise RuntimeError("historical C1 arm same-target semantics drifted")
                if int(row["grip_source_q"]) + int(row["grip_offset"]) != t:
                    raise RuntimeError("historical C1 gripper same-target semantics drifted")
            records[(task_id, state_id, C1)] = episode_record(episode, C1, task_id, state_id)
    expected = {(task, state, method) for task in tasks for state in states for method in (H16_COHERENT, C1)}
    if set(records) != expected:
        raise RuntimeError("historical result coverage is not exactly 9 x 14 x 2")
    return records


def method_table(records: dict[tuple[int, int, str], dict[str, Any]], tasks: list[int], states: list[int]) -> list[dict[str, Any]]:
    rows = []
    for method in ALL_METHODS:
        selected = [records[(task, state, method)] for task in tasks for state in states]
        steps = sum(record["environment_steps"] for record in selected)
        queries = sum(record["policy_queries"] for record in selected)
        arm_ages = [age for record in selected for age in record["arm_ages"]]
        grip_ages = [age for record in selected for age in record["grip_ages"]]
        rows.append(
            {
                "method": method,
                "historical_reuse": method in (H16_COHERENT, C1),
                "successes": int(sum(record["success"] for record in selected)),
                "episodes": len(selected),
                "success_rate": float(np.mean([record["success"] for record in selected])),
                "success_percentage": float(100 * np.mean([record["success"] for record in selected])),
                "policy_queries": queries,
                "environment_steps": steps,
                "observed_query_rate": queries / steps,
                "mean_arm_source_age": float(np.mean(arm_ages)),
                "mean_gripper_source_age": float(np.mean(grip_ages)),
                "arm_source_age_range": [int(min(arm_ages)), int(max(arm_ages))],
                "gripper_source_age_range": [int(min(grip_ages)), int(max(grip_ages))],
            }
        )
    return rows


def per_task_rows(outcomes: dict[tuple[int, int, str], int], tasks: list[int], states: list[int]) -> list[dict[str, Any]]:
    rows = []
    for task_id in tasks:
        counts = {method: int(sum(outcomes[(task_id, state, method)] for state in states)) for method in ALL_METHODS}
        rows.append(
            {
                "task_id": task_id,
                "blocks": len(states),
                "H16_COHERENT_successes": counts[H16_COHERENT],
                "H32_COHERENT_successes": counts[H32_COHERENT],
                "TWO_CLOCK_ARM16_GRIP32_successes": counts[TWO_CLOCK],
                "C1_PREVIOUS_CHUNK_GRIP_successes": counts[C1],
            }
        )
    return rows


def interpretation(comparisons: dict[str, dict[str, Any]]) -> tuple[str, str]:
    two_h16 = comparisons["TWO_CLOCK_VS_H16"]
    two_h32 = comparisons["TWO_CLOCK_VS_H32"]
    two_success = int(two_h16["first_successes"])
    h16_success = int(two_h16["second_successes"])
    h32_success = int(two_h32["second_successes"])
    if two_success > h16_success and two_success > h32_success and two_h16["net_wins"] > 0 and two_h32["net_wins"] > 0:
        return "TWO_CLOCK_SIGNAL", "TWO_CLOCK exceeded both coherent references and had positive paired net against both."
    if h16_success >= two_success and h16_success >= h32_success:
        return "H16_REMAINS_BEST", "Coherent H16 matched or exceeded both new conditions."
    if h32_success >= two_success:
        return "LONGER_COHERENT_SUFFICIENT", "Coherent H32 matched or exceeded TWO_CLOCK, so no clear component-specific advantage was observed."
    return "AMBIGUOUS", "None of the predeclared descriptive patterns cleanly applied."


def report_text(analysis: dict[str, Any]) -> str:
    lines = [
        "# ICRA 2027 two-clock discriminator development result",
        "",
        "This is a development-only comparison on the already exposed 126-block LIBERO Object cohort. H16 and C1 are exact historical reuses; H32 and true arm16/grip32 are the only new rollout conditions.",
        "",
        f"Descriptive interpretation: **{analysis['interpretation_label']}**. {analysis['interpretation_reason']}",
        "",
        "## Main results",
        "",
        "| Method | Status | Success /126 | Success % | Queries | Env steps | Query rate | Mean arm age | Mean grip age | Arm age range | Grip age range |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["main_result_table"]:
        status = "historical reuse" if row["historical_reuse"] else "new rollout"
        lines.append(
            f"| {row['method']} | {status} | {row['successes']}/126 | {row['success_percentage']:.1f}% | "
            f"{row['policy_queries']} | {row['environment_steps']} | {row['observed_query_rate']:.5f} | "
            f"{row['mean_arm_source_age']:.3f} | {row['mean_gripper_source_age']:.3f} | "
            f"{row['arm_source_age_range'][0]}–{row['arm_source_age_range'][1]} | "
            f"{row['gripper_source_age_range'][0]}–{row['gripper_source_age_range'][1]} |"
        )
    lines += [
        "",
        "## Primary paired contrasts",
        "",
        "| Contrast | First-only | Second-only | Net | Delta (pp) | Exact McNemar p | Paired 95% CI | Task-cluster 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, _, _ in COMPARISONS:
        row = analysis["comparisons"][label]
        lines.append(
            f"| {row['first_method']} vs {row['second_method']} | {row['first_only_wins']} | {row['second_only_wins']} | "
            f"{row['net_wins']} | {row['success_delta_percentage_points']:.2f} | {row['exact_two_sided_mcnemar_p']:.6g} | "
            f"[{row['paired_bootstrap_ci'][0]:.3f}, {row['paired_bootstrap_ci'][1]:.3f}] | "
            f"[{row['task_cluster_bootstrap_ci'][0]:.3f}, {row['task_cluster_bootstrap_ci'][1]:.3f}] |"
        )
    rescue = analysis["rescue_regression"]
    lines += [
        "",
        "McNemar p-values are descriptive because this is development.",
        "",
        "## Rescue and regression",
        "",
        f"- Against historical H16: {rescue['vs_H16_COHERENT']['rescues']} H16 failures rescued; {rescue['vs_H16_COHERENT']['regressions']} H16 successes regressed.",
        f"- Against coherent H32: {rescue['vs_H32_COHERENT']['rescues']} H32 failures rescued; {rescue['vs_H32_COHERENT']['regressions']} H32 successes regressed.",
        "",
        "## Per-task success counts",
        "",
        "| Task | H16 | H32 | TWO_CLOCK | C1 context |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["per_task"]:
        lines.append(
            f"| {row['task_id']} | {row['H16_COHERENT_successes']}/14 | {row['H32_COHERENT_successes']}/14 | "
            f"{row['TWO_CLOCK_ARM16_GRIP32_successes']}/14 | {row['C1_PREVIOUS_CHUNK_GRIP_successes']}/14 |"
        )
    lines += [
        "",
        "## Leave-one-task-out deltas",
        "",
        "| Omitted task | TWO_CLOCK−H16 | TWO_CLOCK−H32 | H32−H16 |",
        "|---:|---:|---:|---:|",
    ]
    for index, task_id in enumerate(analysis["tasks"]):
        lines.append(
            f"| {task_id} | {analysis['comparisons']['TWO_CLOCK_VS_H16']['leave_one_task_out'][index]:.4f} | "
            f"{analysis['comparisons']['TWO_CLOCK_VS_H32']['leave_one_task_out'][index]:.4f} | "
            f"{analysis['comparisons']['H32_VS_H16']['leave_one_task_out'][index]:.4f} |"
        )
    lines += [
        "",
        "The semantic smoke passed before the full rollout. No additional horizon, adaptive gate, confirmation task, RoboTwin, pi0/pi0.5, SmolVLA, or real-robot experiment was launched.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    tasks = [int(value) for value in protocol["cohort"]["primary_task_ids"]]
    states = [int(value) for value in protocol["cohort"]["state_ids"]]
    records = {**collect_new(protocol, args.output_root), **collect_historical(protocol)}
    outcomes = {key: int(record["success"]) for key, record in records.items()}
    comparisons: dict[str, dict[str, Any]] = {}
    seeds = protocol["statistics"]["bootstrap_seeds"]
    seed_pairs = (
        (seeds["two_clock_vs_h16_paired"], seeds["two_clock_vs_h16_cluster"]),
        (seeds["two_clock_vs_h32_paired"], seeds["two_clock_vs_h32_cluster"]),
        (seeds["h32_vs_h16_paired"], seeds["h32_vs_h16_cluster"]),
    )
    for (label, first, second), (paired_seed, cluster_seed) in zip(COMPARISONS, seed_pairs, strict=True):
        comparisons[label] = comparison_statistics(outcomes, tasks, states, first, second, paired_seed, cluster_seed)
    label, reason = interpretation(comparisons)
    table = method_table(records, tasks, states)
    per_task = per_task_rows(outcomes, tasks, states)
    analysis = {
        "schema_version": 1,
        "scope": "LIBERO Object tasks 1-9 on the frozen exposed 14-state development cohort",
        "tasks": tasks,
        "states": states,
        "blocks": 126,
        "new_conditions": list(METHODS),
        "new_episodes_executed": 252,
        "historical_H16_COHERENT_reused": True,
        "historical_C1_PREVIOUS_CHUNK_GRIP_reused": True,
        "main_result_table": table,
        "comparisons": comparisons,
        "rescue_regression": {
            "vs_H16_COHERENT": {
                "rescues": comparisons["TWO_CLOCK_VS_H16"]["first_only_wins"],
                "regressions": comparisons["TWO_CLOCK_VS_H16"]["second_only_wins"],
            },
            "vs_H32_COHERENT": {
                "rescues": comparisons["TWO_CLOCK_VS_H32"]["first_only_wins"],
                "regressions": comparisons["TWO_CLOCK_VS_H32"]["second_only_wins"],
            },
        },
        "per_task": per_task,
        "interpretation_label": label,
        "interpretation_reason": reason,
        "stop_after_development_label": True,
    }
    write_json(args.output_root / "analysis.json", analysis)
    with (args.output_root / "per_task.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_task[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(per_task)
    shard_root = args.output_root / "condition_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    for method in ALL_METHODS:
        episodes = []
        for task in tasks:
            for state in states:
                record = records[(task, state, method)]
                episodes.append(
                    {
                        "task_id": task,
                        "state_id": state,
                        "environment_seed": record["environment_seed"],
                        "success": record["success"],
                        "environment_steps": record["environment_steps"],
                        "policy_queries": record["policy_queries"],
                        "query_rate": record["query_rate"],
                        "mean_arm_source_age": float(np.mean(record["arm_ages"])),
                        "mean_gripper_source_age": float(np.mean(record["grip_ages"])),
                    }
                )
        write_json(shard_root / f"{method}.json", {"schema_version": 1, "method": method, "episodes": episodes})
    (args.output_root / "report.md").write_text(report_text(analysis), encoding="utf-8")
    print(json.dumps({"interpretation": label, "analysis": str((args.output_root / 'analysis.json').resolve())}))


if __name__ == "__main__":
    main()
