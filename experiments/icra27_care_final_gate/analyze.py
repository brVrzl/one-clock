#!/usr/bin/env python3
"""Validate and analyze the final Gate M and SmolVLA robustness results."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from run_queue import ROOT, validate_result


DRAWS = 20000
PREREGISTRATION_SHA = "08128f54c84a004dd015f24849a28dec966b716c"
GATE_METHODS = ("M0_HARD16", "M2_GRIPPER_EVENT", "FIXED_H13", "SHUFFLED_TRIGGER")
SMOL_METHODS = ("ARM4_GRIP4", "ARM4_GRIP32")
PRIMARY = (
    ("M2_VS_M0", "M2_GRIPPER_EVENT", "M0_HARD16"),
    ("M2_VS_FIXED_H13", "M2_GRIPPER_EVENT", "FIXED_H13"),
    ("M2_VS_SHUFFLED", "M2_GRIPPER_EVENT", "SHUFFLED_TRIGGER"),
)
SECONDARY = (
    ("FIXED_H13_VS_M0", "FIXED_H13", "M0_HARD16"),
    ("SHUFFLED_VS_M0", "SHUFFLED_TRIGGER", "M0_HARD16"),
    ("FIXED_H13_VS_SHUFFLED", "FIXED_H13", "SHUFFLED_TRIGGER"),
)


def key(row: dict) -> tuple[str, int, int]:
    return row["suite"], int(row["task_id"]), int(row["state_id"])


def task_label(row: dict) -> str:
    return f"{row['suite']}:task{int(row['task_id'])}"


def mcnemar(first_only: int, second_only: int) -> float:
    total = first_only + second_only
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, i) for i in range(min(first_only, second_only) + 1)) / (2 ** total)
    return min(1.0, 2 * tail)


def comparison(first: list[dict], second: list[dict], paired_seed: int, cluster_seed: int) -> dict[str, Any]:
    left, right = {key(row): row for row in first}, {key(row): row for row in second}
    if set(left) != set(right):
        raise ValueError(f"paired keys differ: {len(left)} versus {len(right)}")
    keys = sorted(left)
    left_values = np.asarray([bool(left[k]["success"]) for k in keys], dtype=np.int8)
    right_values = np.asarray([bool(right[k]["success"]) for k in keys], dtype=np.int8)
    differences = left_values.astype(float) - right_values.astype(float)
    labels = np.asarray([task_label(left[k]) for k in keys])
    unique_labels = sorted(set(labels))
    by_task = {label: differences[labels == label] for label in unique_labels}
    paired_rng = np.random.default_rng(paired_seed)
    paired = differences[paired_rng.integers(0, len(differences), size=(DRAWS, len(differences)))].mean(axis=1)
    cluster_rng = np.random.default_rng(cluster_seed)
    clustered = np.empty(DRAWS)
    for index in range(DRAWS):
        sampled = cluster_rng.integers(0, len(unique_labels), size=len(unique_labels))
        clustered[index] = np.concatenate([by_task[unique_labels[i]] for i in sampled]).mean()
    first_only = int(np.sum((left_values == 1) & (right_values == 0)))
    second_only = int(np.sum((left_values == 0) & (right_values == 1)))
    per_task = {}
    loto = {}
    for label in unique_labels:
        selected = labels == label
        per_task[label] = {
            "blocks": int(selected.sum()),
            "first_successes": int(left_values[selected].sum()),
            "second_successes": int(right_values[selected].sum()),
            "delta": float(differences[selected].mean()),
            "delta_percentage_points": float(100 * differences[selected].mean()),
        }
        loto[label] = float(differences[~selected].mean())
    paired_ci = [float(x) for x in np.quantile(paired, [0.025, 0.975])]
    cluster_ci = [float(x) for x in np.quantile(clustered, [0.025, 0.975])]
    return {
        "blocks": len(keys),
        "first_successes": int(left_values.sum()),
        "second_successes": int(right_values.sum()),
        "first_only_wins": first_only,
        "second_only_wins": second_only,
        "ties": int(len(keys) - first_only - second_only),
        "delta": float(differences.mean()),
        "delta_percentage_points": float(100 * differences.mean()),
        "exact_two_sided_mcnemar_p": mcnemar(first_only, second_only),
        "paired_bootstrap_ci": paired_ci,
        "task_cluster_bootstrap_ci": cluster_ci,
        "bootstrap_draws": DRAWS,
        "paired_bootstrap_seed": paired_seed,
        "task_cluster_bootstrap_seed": cluster_seed,
        "per_task": per_task,
        "leave_one_task_out": loto,
        "positive_loto_count": sum(value > 0 for value in loto.values()),
        "strict_stable_positive": paired_ci[0] > 0 and cluster_ci[0] > 0 and sum(value > 0 for value in loto.values()) >= 8,
    }


def method_summary(rows: list[dict], *, include_horizons: bool) -> dict[str, Any]:
    steps = sum(int(row["environment_steps"]) for row in rows)
    queries = sum(int(row["policy_queries"]) for row in rows)
    completion = [int(row["completion_length"]) for row in rows if row["completion_length"] is not None]
    result = {
        "successes": sum(bool(row["success"]) for row in rows),
        "episodes": len(rows),
        "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
        "environment_steps": steps,
        "policy_queries": queries,
        "query_rate": queries / steps,
        "wall_clock_seconds": sum(float(row["wall_clock_seconds"]) for row in rows),
        "mean_wall_clock_seconds": float(np.mean([row["wall_clock_seconds"] for row in rows])),
        "successful_completion_lengths": completion,
        "mean_successful_completion_length": float(np.mean(completion)) if completion else None,
        "median_successful_completion_length": float(np.median(completion)) if completion else None,
    }
    ages = {}
    for component in ("arm", "gripper"):
        values = np.asarray([age[component] for row in rows for age in row["source_ages"]], dtype=float)
        ages[component] = {
            "mean": float(values.mean()), "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)), "maximum": int(values.max()),
        }
    result["source_ages"] = ages
    if include_horizons:
        horizons = [int(horizon) for row in rows for horizon in row["execution_horizons"]]
        histogram = Counter(horizons)
        result.update({
            "execution_horizon_count": len(horizons),
            "mean_execution_horizon": float(np.mean(horizons)),
            "median_execution_horizon": float(np.median(horizons)),
            "execution_horizon_histogram": {str(h): histogram[h] for h in range(4, 17)},
        })
    return result


def load_results(manifest: dict) -> dict[str, list[dict]]:
    rows = {phase: [] for phase in manifest["expected_counts"]}
    for cell in manifest["cells"]:
        path = ROOT / "results" / cell["phase"] / f"{cell['cell_id']}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows[cell["phase"]].append(validate_result(cell, path))
    return rows


def validate_identity(manifest: dict, rows: dict[str, list[dict]]) -> dict[str, Any]:
    expected_cells = {cell["cell_id"]: cell for cell in manifest["cells"]}
    observed = {row["cell_id"]: row for phase_rows in rows.values() for row in phase_rows}
    if set(expected_cells) != set(observed):
        raise ValueError("result identity coverage mismatch")
    for cell_id, cell in expected_cells.items():
        row = observed[cell_id]
        expected_seed = (
            330000 + 100 * int(cell["task_id"]) + int(cell["state_id"])
            if cell["phase"] == "gate_m"
            else 360000 + 1000 * {"libero_spatial": 0, "libero_object": 1, "libero_goal": 2, "libero_10": 3}[cell["suite"]] + 100 * int(cell["task_id"]) + int(cell["state_id"])
        )
        if int(row["environment_seed"]) != expected_seed:
            raise ValueError(f"environment seed drift: {cell_id}")
    attempts = []
    for path in sorted((ROOT / "attempts").glob("*/*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        attempts.extend({"cell_id": value["cell_id"], **item} for item in value.get("attempts", []))
    technical_failed = list((ROOT / "markers").glob("*/*.technical_failed"))
    return {
        "all_manifest_identities_exact": True,
        "all_environment_seeds_exact": True,
        "all_action_counts_exact": True,
        "all_query_counts_and_schedules_exact": True,
        "completed": {phase: len(phase_rows) for phase, phase_rows in rows.items()},
        "technical_failure_attempts": attempts,
        "technical_failure_attempt_count": len(attempts),
        "technical_failed_cells": [path.stem for path in technical_failed],
        "technical_failed_cell_count": len(technical_failed),
        "observed_methods": sorted({row["method"] for phase_rows in rows.values() for row in phase_rows}),
    }


def gate_analysis(rows: list[dict], protocol: dict) -> dict[str, Any]:
    by_method = {method: [row for row in rows if row["method"] == method] for method in GATE_METHODS}
    if any(len(values) != 130 for values in by_method.values()):
        raise ValueError("Gate M does not contain exactly 130 episodes per method")
    summaries = {method: method_summary(values, include_horizons=True) for method, values in by_method.items()}
    comparisons = {}
    for index, (label, first, second) in enumerate(PRIMARY + SECONDARY):
        comparisons[label] = comparison(
            by_method[first], by_method[second], 2027090201 + index, 2027090301 + index,
        )
    budget_difference = abs(summaries["M2_GRIPPER_EVENT"]["query_rate"] - summaries["SHUFFLED_TRIGGER"]["query_rate"])
    budget_pass = budget_difference <= float(protocol["gate_m"]["budget_sanity"]["m2_vs_shuffled_absolute_query_rate_difference_maximum"])
    primary = {label: comparisons[label] for label, _, _ in PRIMARY}
    m2_m0 = primary["M2_VS_M0"]
    if all(value["strict_stable_positive"] for value in primary.values()) and budget_pass:
        label = "METHOD_CONFIRMED"
    elif not m2_m0["strict_stable_positive"]:
        label = "METHOD_NULL"
    elif (
        primary["M2_VS_FIXED_H13"]["delta"] <= 0
        or primary["M2_VS_SHUFFLED"]["delta"] <= 0
        or not budget_pass
    ):
        label = "QUERY_BUDGET_OR_HORIZON_EXPLAINS"
    else:
        label = "PROOF_OF_CONCEPT_ONLY"
    return {
        "scope": "held-out Gate M",
        "paired_blocks": 130,
        "episodes": 520,
        "methods": summaries,
        "primary_contrasts": primary,
        "secondary_descriptive_contrasts": {label: comparisons[label] for label, _, _ in SECONDARY},
        "query_budget": {
            "M2_query_rate": summaries["M2_GRIPPER_EVENT"]["query_rate"],
            "SHUFFLED_query_rate": summaries["SHUFFLED_TRIGGER"]["query_rate"],
            "absolute_M2_minus_SHUFFLED": budget_difference,
            "predeclared_maximum": 0.005,
            "budget_match_pass": budget_pass,
            "M2_minus_FIXED_H13": summaries["M2_GRIPPER_EVENT"]["query_rate"] - summaries["FIXED_H13"]["query_rate"],
            "absolute_M2_minus_FIXED_H13": abs(summaries["M2_GRIPPER_EVENT"]["query_rate"] - summaries["FIXED_H13"]["query_rate"]),
        },
        "final_label": label,
    }


def scope_analysis(rows: list[dict], paired_seed: int, cluster_seed: int) -> dict[str, Any]:
    by_method = {method: [row for row in rows if row["method"] == method] for method in SMOL_METHODS}
    return {
        "ARM4_GRIP4": method_summary(by_method["ARM4_GRIP4"], include_horizons=False),
        "ARM4_GRIP32": method_summary(by_method["ARM4_GRIP32"], include_horizons=False),
        "ARM4_GRIP32_VS_ARM4_GRIP4": comparison(
            by_method["ARM4_GRIP32"], by_method["ARM4_GRIP4"], paired_seed, cluster_seed,
        ),
    }


def smolvla_analysis(rows: list[dict]) -> dict[str, Any]:
    if len(rows) != 320:
        raise ValueError("SmolVLA robustness does not contain exactly 320 episodes")
    suites = {}
    for index, suite in enumerate(("libero_spatial", "libero_object", "libero_goal", "libero_10")):
        suites[suite] = scope_analysis(
            [row for row in rows if row["suite"] == suite], 2027090401 + index, 2027090501 + index,
        )
    return {
        "scope": "CROSS_POLICY_ROBUSTNESS",
        "independent_confirmation": False,
        "paired_blocks": 160,
        "episodes": 320,
        "historical_ACT_transfer_target": {
            "ARM4_GRIP32_successes": 131,
            "ARM4_GRIP4_successes": 112,
            "episodes": 180,
            "delta_percentage_points": 10.56,
            "discordance_ARM4_GRIP32_only": 32,
            "discordance_ARM4_GRIP4_only": 13,
        },
        "suites": suites,
        "pooled": scope_analysis(rows, 2027090410, 2027090510),
    }


def contrast_line(name: str, value: dict) -> str:
    paired = [100 * x for x in value["paired_bootstrap_ci"]]
    clustered = [100 * x for x in value["task_cluster_bootstrap_ci"]]
    return (
        f"| {name} | {value['first_successes']}/{value['blocks']} | {value['second_successes']}/{value['blocks']} | "
        f"{value['first_only_wins']}:{value['second_only_wins']} | {value['delta_percentage_points']:+.2f} | "
        f"{value['exact_two_sided_mcnemar_p']:.6g} | [{paired[0]:+.2f}, {paired[1]:+.2f}] | "
        f"[{clustered[0]:+.2f}, {clustered[1]:+.2f}] | {value['positive_loto_count']}/{len(value['leave_one_task_out'])} |"
    )


def method_line(name: str, value: dict) -> str:
    return (
        f"| {name} | {value['successes']}/{value['episodes']} | {100 * value['success_rate']:.2f} | "
        f"{value['environment_steps']} | {value['policy_queries']} | {value['query_rate']:.6f} | "
        f"{value['mean_successful_completion_length'] if value['mean_successful_completion_length'] is not None else 'NA'} |"
    )


def robustness_method_line(name: str, value: dict) -> str:
    arm = value["source_ages"]["arm"]
    grip = value["source_ages"]["gripper"]
    return (
        f"| {name} | {value['successes']}/{value['episodes']} | {value['environment_steps']} | "
        f"{value['policy_queries']} | {value['query_rate']:.6f} | {value['wall_clock_seconds']:.1f} | "
        f"{arm['mean']:.3f}/{arm['maximum']} | {grip['mean']:.3f}/{grip['maximum']} |"
    )


def render_report(analysis: dict) -> str:
    gate = analysis["gate_m"]
    lines = [
        "# ICRA 2027 final CARE method gate",
        "",
        f"Final label: **{gate['final_label']}**.",
        "",
        "The held-out cohort contains 130 paired Object blocks. Four raw historical outcomes omitted by the prior inventory were removed before preregistration: task 6 states 25, 26, 28, and 29. No replacement states were added.",
        "",
        "## Gate M methods",
        "",
        "| Method | Success | Rate (%) | Environment steps | Policy queries | Query rate | Mean execution horizon | Mean successful completion length |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in GATE_METHODS:
        value = gate["methods"][method]
        line = method_line(method, value).rstrip("|")
        fields = line.split("|")
        fields.insert(-1, f" {value['mean_execution_horizon']:.3f} ")
        lines.append("|".join(fields) + "|")
    lines += ["", "Execution-horizon histograms:", ""]
    for method in GATE_METHODS:
        histogram = gate["methods"][method]["execution_horizon_histogram"]
        lines.append(f"- `{method}`: " + ", ".join(f"{h}:{count}" for h, count in histogram.items()))
    lines += [
        "",
        "## Gate M primary contrasts",
        "",
        "| Contrast | First | Second | Discordance | Delta (pp) | McNemar p | Paired 95% CI (pp) | Task-cluster 95% CI (pp) | Positive LOTO |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, _, _ in PRIMARY:
        lines.append(contrast_line(label, gate["primary_contrasts"][label]))
    lines += [
        "",
        "Per-task deltas and leave-one-task-out pooled deltas:",
        "",
        "| Task | Blocks | M2−M0 task / LOTO (pp) | M2−H13 task / LOTO (pp) | M2−SHUFFLED task / LOTO (pp) |",
        "|---|---:|---:|---:|---:|",
    ]
    tasks = gate["primary_contrasts"]["M2_VS_M0"]["per_task"]
    for task, task_value in tasks.items():
        values = []
        for contrast in ("M2_VS_M0", "M2_VS_FIXED_H13", "M2_VS_SHUFFLED"):
            comparison_value = gate["primary_contrasts"][contrast]
            values.append(
                f"{comparison_value['per_task'][task]['delta_percentage_points']:+.2f} / "
                f"{100 * comparison_value['leave_one_task_out'][task]:+.2f}"
            )
        lines.append(f"| `{task}` | {task_value['blocks']} | {values[0]} | {values[1]} | {values[2]} |")
    lines += [
        "",
        "## Gate M secondary descriptive contrasts",
        "",
        "| Contrast | First | Second | Discordance | Delta (pp) | McNemar p | Paired 95% CI (pp) | Task-cluster 95% CI (pp) | Positive LOTO |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, _, _ in SECONDARY:
        lines.append(contrast_line(label, gate["secondary_descriptive_contrasts"][label]))
    budget = gate["query_budget"]
    lines += [
        "",
        "## Query-budget sanity",
        "",
        f"M2 versus SHUFFLED absolute query-rate difference: **{budget['absolute_M2_minus_SHUFFLED']:.6f}** (criterion <= 0.005: **{'PASS' if budget['budget_match_pass'] else 'MISS'}**). M2 minus FIXED_H13: **{budget['M2_minus_FIXED_H13']:+.6f}**.",
        "",
        "## SmolVLA cross-policy robustness",
        "",
        "This cohort is outcome-exposed and is labeled **CROSS_POLICY_ROBUSTNESS**, not independent confirmation.",
    ]
    smol = analysis["smolvla_robustness"]
    for scope in ("libero_spatial", "libero_object", "libero_goal", "libero_10", "pooled"):
        value = smol["pooled"] if scope == "pooled" else smol["suites"][scope]
        lines += [
            "",
            f"### {scope}",
            "",
            "| Method | Success | Environment steps | Policy queries | Query rate | Wall-clock (s) | Arm age mean/max | Gripper age mean/max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            robustness_method_line("ARM4_GRIP4", value["ARM4_GRIP4"]),
            robustness_method_line("ARM4_GRIP32", value["ARM4_GRIP32"]),
            "",
            "| Contrast | First | Second | Discordance | Delta (pp) | McNemar p | Paired 95% CI (pp) | Task-cluster 95% CI (pp) | Positive LOTO |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            contrast_line("ARM4_GRIP32 vs ARM4_GRIP4", value["ARM4_GRIP32_VS_ARM4_GRIP4"]),
        ]
    integrity = analysis["integrity"]
    lines += [
        "",
        "## Integrity",
        "",
        f"Completed Gate M: **{integrity['completed']['gate_m']}/520**. Completed SmolVLA robustness: **{integrity['completed']['smolvla_robustness']}/320**. Technical failure attempts: **{integrity['technical_failure_attempt_count']}**; terminal technical failures: **{integrity['technical_failed_cell_count']}**.",
        "",
        f"Observed methods were exactly: `{integrity['observed_methods']}`. No follow-up condition is present in the frozen manifest or completed results.",
        "",
    ]
    return "\n".join(lines)


def write_per_task_csv(gate: dict) -> None:
    rows = []
    for contrast, value in gate["primary_contrasts"].items():
        for task, task_value in value["per_task"].items():
            rows.append({
                "contrast": contrast,
                "task": task,
                **task_value,
                "leave_one_task_out_delta": value["leave_one_task_out"][task],
            })
    path = ROOT / "gate_m_per_task_and_loto.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_smolvla_per_task_csv(smolvla: dict) -> None:
    rows = []
    for scope, value in [*smolvla["suites"].items(), ("pooled", smolvla["pooled"])]:
        comparison_value = value["ARM4_GRIP32_VS_ARM4_GRIP4"]
        for task, task_value in comparison_value["per_task"].items():
            rows.append({
                "scope": scope,
                "task": task,
                **task_value,
                "leave_one_task_out_delta": comparison_value["leave_one_task_out"][task],
            })
    path = ROOT / "smolvla_per_task_and_loto.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    manifest = json.loads((ROOT / "queue_manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    rows = load_results(manifest)
    integrity = validate_identity(manifest, rows)
    analysis = {
        "schema_version": 1,
        "preregistered_protocol": "protocol.json",
        "preregistration_commit": PREREGISTRATION_SHA,
        "gate_m": gate_analysis(rows["gate_m"], protocol),
        "smolvla_robustness": smolvla_analysis(rows["smolvla_robustness"]),
        "integrity": integrity,
        "method_development_closed": True,
        "forbidden_followup_launched": False,
        "pre_scientific_technical_audit": {
            "reporting_only_exception_count": 1,
            "scientific_environment_steps": 0,
            "outcomes_observed": 0,
            "clean_process_rerun_passed": True,
        },
    }
    atomic = ROOT / "analysis.json"
    atomic.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    write_per_task_csv(analysis["gate_m"])
    write_smolvla_per_task_csv(analysis["smolvla_robustness"])
    (ROOT / "report.md").write_text(render_report(analysis), encoding="utf-8")
    print(json.dumps({
        "gate_m_label": analysis["gate_m"]["final_label"],
        "gate_m_completed": integrity["completed"]["gate_m"],
        "smolvla_completed": integrity["completed"]["smolvla_robustness"],
    }))


if __name__ == "__main__":
    main()
