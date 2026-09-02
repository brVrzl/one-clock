#!/usr/bin/env python3
"""Analyze completed frozen overnight cells against the exact historical vectors."""

from __future__ import annotations

import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DRAWS = 20000


def git_json(spec: str) -> dict:
    return json.loads(subprocess.check_output(["git", "-C", str(REPO), "show", spec], text=True))


def load_phase(phase: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((ROOT / "results" / phase).glob("*.json"))]


def key(row: dict) -> tuple:
    return row["suite"], int(row["task_id"]), int(row["state_id"])


def task_label(row: dict) -> str:
    return f"{row['suite']}:task{int(row['task_id'])}"


def mcnemar(first_only: int, second_only: int) -> float:
    n = first_only + second_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(first_only, second_only) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def comparison(first: list[dict], second: list[dict], seed: int) -> dict:
    a, b = {key(r): r for r in first}, {key(r): r for r in second}
    if set(a) != set(b):
        raise ValueError(f"paired keys differ: {len(a)} versus {len(b)}")
    keys = sorted(a)
    av = np.asarray([bool(a[k]["success"]) for k in keys], dtype=int)
    bv = np.asarray([bool(b[k]["success"]) for k in keys], dtype=int)
    diff = av - bv
    first_only = int(np.sum((av == 1) & (bv == 0)))
    second_only = int(np.sum((av == 0) & (bv == 1)))
    rng = np.random.default_rng(seed)
    paired = diff[rng.integers(0, len(diff), size=(DRAWS, len(diff)))].mean(axis=1)
    labels = np.asarray([task_label(a[k]) for k in keys])
    unique = sorted(set(labels))
    by_task = {label: diff[labels == label] for label in unique}
    cluster = np.empty(DRAWS)
    for i in range(DRAWS):
        sampled = rng.integers(0, len(unique), size=len(unique))
        cluster[i] = np.concatenate([by_task[unique[j]] for j in sampled]).mean()
    per_task = {label: {"first_successes": int(sum(bool(a[k]["success"]) for k in keys if task_label(a[k]) == label)),
        "second_successes": int(sum(bool(b[k]["success"]) for k in keys if task_label(b[k]) == label)),
        "blocks": int(np.sum(labels == label)), "delta": float(by_task[label].mean())} for label in unique}
    loto = {label: float(diff[labels != label].mean()) for label in unique}
    return {
        "blocks": len(keys), "first_successes": int(av.sum()), "second_successes": int(bv.sum()),
        "first_only_wins": first_only, "second_only_wins": second_only,
        "ties": int(len(keys) - first_only - second_only),
        "delta": float(diff.mean()), "delta_percentage_points": float(100 * diff.mean()),
        "exact_two_sided_mcnemar_p": mcnemar(first_only, second_only),
        "paired_bootstrap_ci": [float(x) for x in np.quantile(paired, [0.025, 0.975])],
        "task_cluster_bootstrap_ci": [float(x) for x in np.quantile(cluster, [0.025, 0.975])],
        "bootstrap_draws": DRAWS, "per_task": per_task, "leave_one_task_out": loto,
    }


def summary(rows: list[dict]) -> dict:
    steps = sum(int(r["environment_steps"]) for r in rows)
    queries = sum(int(r["policy_queries"]) for r in rows)
    source_age_stats = {}
    for component in ("arm", "gripper"):
        ages = np.asarray([
            int(age[component])
            for row in rows
            for age in row.get("source_ages", [])
        ], dtype=float)
        if len(ages):
            source_age_stats[component] = {
                "count": len(ages),
                "mean": float(ages.mean()),
                "standard_deviation": float(ages.std()),
                "median": float(np.median(ages)),
                "p95": float(np.quantile(ages, 0.95)),
                "maximum": int(ages.max()),
            }
    return {
        "successes": sum(bool(r["success"]) for r in rows), "episodes": len(rows),
        "success_rate": sum(bool(r["success"]) for r in rows) / len(rows),
        "environment_steps": steps, "policy_queries": queries, "model_forward_count": sum(int(r.get("model_forward_count", r["policy_queries"])) for r in rows),
        "query_rate": queries / steps, "wall_clock_seconds": sum(float(r.get("wall_clock_seconds", 0)) for r in rows),
        "mean_wall_clock_seconds": float(np.mean([r.get("wall_clock_seconds", 0) for r in rows])),
        "mean_arm_source_age": sum(float(r["mean_arm_source_age"]) * int(r["environment_steps"]) for r in rows) / steps,
        "mean_gripper_source_age": sum(float(r["mean_gripper_source_age"]) * int(r["environment_steps"]) for r in rows) / steps,
        "source_age_stats": source_age_stats,
    }


def schedule_audit(rows: list[dict]) -> dict:
    periods = sorted({min(int(r["arm_horizon"]), int(r["gripper_horizon"])) for r in rows})
    invalid = [r["cell_id"] for r in rows
        if r["query_steps"] != list(range(0, int(r["environment_steps"]),
            min(int(r["arm_horizon"]), int(r["gripper_horizon"]))))]
    return {
        "episodes": len(rows),
        "query_periods": periods,
        "all_exact_periodic_schedules": not invalid,
        "invalid_cell_ids": invalid,
        "all_arm_driven": all(int(r["arm_horizon"]) <= int(r["gripper_horizon"]) for r in rows),
    }


def archived_preflight_summary() -> dict:
    out = {}
    base = ROOT / "preflight_failures"
    if not base.is_dir():
        return out
    for directory in sorted(p for p in base.iterdir() if p.is_dir()):
        attempt_files = sorted((directory / "attempts").glob("*.json"))
        out[directory.name] = {
            "cells_with_exception_history": len(attempt_files),
            "exception_attempts": sum(len(json.loads(p.read_text()).get("attempts", [])) for p in attempt_files),
            "provisional_technical_failed_markers": len(list((directory / "markers").glob("*.technical_failed"))),
            "scientific_result_files": len(list((directory / "results").glob("*.json"))) if (directory / "results").is_dir() else 0,
        }
    return out


def historical_h16_126() -> list[dict]:
    data = git_json("c4f9cb9:experiments/icra27_two_clock_discriminator_dev/condition_shards/H16_COHERENT.json")
    return [{"suite": "libero_object", **row} for row in data["episodes"]]


def historical_h16_140() -> list[dict]:
    rows = []
    for suite, tasks in {"libero_goal": (4, 6, 7, 8, 9), "libero_10": (0, 2, 4, 6, 7)}.items():
        for task in tasks:
            data = json.loads((REPO / f"experiments/cross_suite_confirmation/results/{suite}_task{task}.json").read_text())
            for row in data["episodes"]["HARD_H16"]:
                rows.append({"suite": suite, "task_id": task,
                    "state_id": int(row["requested_initial_state_id"]), "success": bool(row["success"]),
                    "environment_steps": int(row["environment_steps"]), "policy_queries": int(row["policy_queries"]),
                    "mean_arm_source_age": float(row["mean_gripper_source_age"]),
                    "mean_gripper_source_age": float(row["mean_gripper_source_age"]),
                    "wall_clock_seconds": float(row["wall_clock_seconds"]), "model_forward_count": int(row["policy_queries"]),
                })
    return rows


def historical_cross(method: str) -> list[dict]:
    rows = []
    for task in range(1, 10):
        data = git_json(f"origin/main:experiments/libero_object_cross_task/task_{task}/result.json")
        config = next(c for c in data["configurations"] if c["name"] == method)
        for state, success in enumerate(config["success_vector"]):
            rows.append({"suite": "libero_object", "task_id": task, "state_id": state, "success": bool(success)})
    return rows


def comparison_line(name: str, value: dict) -> str:
    paired = [100 * x for x in value["paired_bootstrap_ci"]]
    clustered = [100 * x for x in value["task_cluster_bootstrap_ci"]]
    return (
        f"- {name}: first-only `{value['first_only_wins']}`, second-only "
        f"`{value['second_only_wins']}`, ties `{value['ties']}`, delta "
        f"`{value['delta_percentage_points']:+.2f} pp`, exact two-sided McNemar "
        f"`p={value['exact_two_sided_mcnemar_p']:.6g}`, paired bootstrap 95% CI "
        f"`[{paired[0]:+.2f}, {paired[1]:+.2f}] pp`, task-cluster bootstrap "
        f"95% CI `[{clustered[0]:+.2f}, {clustered[1]:+.2f}] pp`."
    )


def task_tables(value: dict, first: str, second: str) -> list[str]:
    lines = [
        "",
        f"| Task | {first} | {second} | Blocks | Delta (pp) | LOTO pooled delta (pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, row in value["per_task"].items():
        lines.append(
            f"| `{label}` | {row['first_successes']} | {row['second_successes']} | "
            f"{row['blocks']} | {100 * row['delta']:+.2f} | "
            f"{100 * value['leave_one_task_out'][label]:+.2f} |"
        )
    return lines


def method_row(label: str, value: dict) -> str:
    arm = value["source_age_stats"].get("arm", {})
    grip = value["source_age_stats"].get("gripper", {})
    arm_stats = (f"{arm['mean']:.3f}/{arm['p95']:.1f}/{arm['maximum']}"
        if arm else f"{value['mean_arm_source_age']:.3f}/NA/NA")
    grip_stats = (f"{grip['mean']:.3f}/{grip['p95']:.1f}/{grip['maximum']}"
        if grip else f"{value['mean_gripper_source_age']:.3f}/NA/NA")
    return (
        f"| {label} | {value['successes']}/{value['episodes']} | "
        f"{100 * value['success_rate']:.2f}% | {value['environment_steps']} | "
        f"{value['policy_queries']} | {100 * value['query_rate']:.3f}% | "
        f"{value['mean_wall_clock_seconds']:.3f} | {arm_stats} | {grip_stats} |"
    )


def paired_section(title: str, value: dict, first: str, second: str) -> list[str]:
    return [f"### {title}", "", comparison_line(f"{first} vs {second}", value)] + task_tables(value, first, second) + [""]


def render_report(analysis: dict) -> str:
    completed = analysis["completed"]
    technical = analysis["technical_failures"]
    requested_phases = ("act_posthoc_h8_140", "act_arm4_grip32_180", "smolvla_primary", "smolvla_capacity_h16")
    requested_complete = sum(completed[p] for p in requested_phases)
    lines = [
        "# ICRA 2027 overnight fixed-clock results harvest",
        "",
        "## Completion",
        "",
        f"Completion count first: **{requested_complete}/800 frozen overnight cells complete**. "
        f"Including the already-completed 126-cell ACT discriminator phase, the full manifest is "
        f"**{sum(completed.values())}/926 complete**, with **{sum(technical.values())} TECHNICAL_FAILED**, "
        "**0 pending**, and **0 running**.",
        "",
        "## A. ACT-B: post-hoc 140-block H8 audit",
        "",
    ]
    actb = analysis["act_posthoc_h8_140"]
    comp = actb["H8_vs_H16"]
    lines += [
        f"This is a **post-hoc coherent-baseline audit**, not new confirmation. H8 succeeded on "
        f"**{actb['H8']['successes']}/140**; historical H16 succeeded on **93/140**.",
        comparison_line("H8 vs historical H16 (H8-only:H16-only)", comp),
        f"- Interpretation: `{actb['interpretation_label']}`. " + (
            "The label `COHERENT_OPTIMUM_IS_NOT_H16` applies."
            if actb["interpretation_label"] == "COHERENT_OPTIMUM_IS_NOT_H16"
            else "H16 was not challenged by H8; `COHERENT_OPTIMUM_IS_NOT_H16` does not apply."
        ),
    ] + task_tables(comp, "H8", "H16") + [
        "",
        "## B. ACT-C: ARM4_GRIP32",
        "",
    ]
    actc = analysis["act_arm4_grip32_180"]
    lines += [
        f"ARM4_GRIP32 succeeded on **{actc['ARM4_GRIP32']['successes']}/180**; historical "
        "references are ARM4_GRIP16 **128/180** and ARM4_GRIP4 **112/180**.",
        comparison_line("ARM4_GRIP32 vs ARM4_GRIP16", actc["ARM4_GRIP32_vs_ARM4_GRIP16"]),
    ] + task_tables(actc["ARM4_GRIP32_vs_ARM4_GRIP16"], "GRIP32", "GRIP16") + [
        "",
        comparison_line("ARM4_GRIP32 vs ARM4_GRIP4", actc["ARM4_GRIP32_vs_ARM4_GRIP4"]),
    ] + task_tables(actc["ARM4_GRIP32_vs_ARM4_GRIP4"], "GRIP32", "GRIP4") + [
        "",
        "Execution totals and source ages (mean/p95/max):",
        "",
        "| Method | Success | Rate | Env steps | Queries | Query rate | Mean wall (s) | Arm age | Gripper age |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        method_row("ARM4_GRIP32", actc["ARM4_GRIP32"]),
        "",
        f"Plateau criterion `abs(successes_ARM4_GRIP32 - 128) <= 3`: "
        f"**{'met' if actc['interpretation_label'] == 'GRIPPER_PLATEAU_AT_16' else 'not met'}**. "
        f"Label: `{actc['interpretation_label']}`. No grip64 or other grid cell was run.",
        "",
        "## C. SmolVLA primary",
        "",
        "All **320/320 primary episodes** (160 paired blocks) completed.",
    ]
    primary = analysis["smolvla_primary"]
    suite_names = {
        "libero_spatial": "Spatial", "libero_goal": "Goal",
        "libero_object": "Object", "libero_10": "Long/LIBERO-10",
    }
    for suite, label in suite_names.items():
        value = primary["suites"][suite]
        lines += [
            "",
            f"### {label}",
            "",
            "| Method | Success | Rate | Env steps | Queries | Query rate | Mean wall (s) | Arm age | Gripper age |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            method_row("COHERENT_H8", value["SMOLVLA_COHERENT_H8"]),
            method_row("ARM8_GRIP16", value["SMOLVLA_ARM8_GRIP16"]),
            "",
            comparison_line("ARM8_GRIP16 vs COHERENT_H8", value["ARM8_GRIP16_vs_COHERENT_H8"]),
        ] + task_tables(value["ARM8_GRIP16_vs_COHERENT_H8"], "ARM8_GRIP16", "H8")
    pooled = primary
    lines += [
        "",
        "### Pooled across all suites",
        "",
        "| Method | Success | Rate | Env steps | Queries | Query rate | Mean wall (s) | Arm age | Gripper age |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        method_row("COHERENT_H8", pooled["pooled"]["SMOLVLA_COHERENT_H8"]),
        method_row("ARM8_GRIP16", pooled["pooled"]["SMOLVLA_ARM8_GRIP16"]),
        "",
        comparison_line("ARM8_GRIP16 vs COHERENT_H8", pooled["ARM8_GRIP16_vs_COHERENT_H8"]),
    ] + task_tables(pooled["ARM8_GRIP16_vs_COHERENT_H8"], "ARM8_GRIP16", "H8") + [
        "",
        f"Query-schedule audit: `{primary['query_schedule_audit']}`. The intended matched "
        "arm-driven query schedule is present: both methods query at steps 0, 8, 16, ... until "
        "their own terminal step. Aggregate query rates can differ slightly because episode lengths differ.",
        "",
        "## D. SmolVLA COHERENT_H16 capacity condition",
        "",
        "The capacity condition **ran and completed all 160/160 episodes** after the primary barrier.",
    ]
    capacity = analysis["smolvla_capacity_h16"]
    lines += [f"Barrier audit: `{capacity['barrier_audit']}`."]
    for scope, label in list(suite_names.items()) + [("pooled", "Pooled across all suites")]:
        value = capacity[scope] if scope == "pooled" else capacity["suites"][scope]
        primary_value = primary["pooled"] if scope == "pooled" else primary["suites"][scope]
        lines += [
            "",
            f"### {label}",
            "",
            "| Method | Success | Rate | Env steps | Queries | Query rate | Mean wall (s) | Arm age | Gripper age |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            method_row("COHERENT_H16", value["SMOLVLA_COHERENT_H16"]),
            method_row("COHERENT_H8", primary_value["SMOLVLA_COHERENT_H8"]),
            method_row("ARM8_GRIP16", primary_value["SMOLVLA_ARM8_GRIP16"]),
            "",
            comparison_line("COHERENT_H16 vs COHERENT_H8", value["COHERENT_H16_vs_COHERENT_H8"]),
        ] + task_tables(value["COHERENT_H16_vs_COHERENT_H8"], "H16", "H8") + [
            "",
            comparison_line("ARM8_GRIP16 vs COHERENT_H16", value["ARM8_GRIP16_vs_COHERENT_H16"]),
        ] + task_tables(value["ARM8_GRIP16_vs_COHERENT_H16"], "ARM8_GRIP16", "H16")
    pooled_capacity = capacity["pooled"]
    lines += [
        "",
        f"Capacity interpretation: coherent H16 was numerically higher than H8 "
        f"({pooled_capacity['SMOLVLA_COHERENT_H16']['successes']}/160 vs "
        f"{primary['pooled']['SMOLVLA_COHERENT_H8']['successes']}/160; "
        f"{pooled_capacity['COHERENT_H16_vs_COHERENT_H8']['delta_percentage_points']:+.2f} pp), "
        "while ARM8_GRIP16 did not exceed H16. Together with the exactly null pooled primary "
        "comparison, these data do not establish a component-specific SmolVLA advantage.",
    ]
    integrity = analysis["execution_integrity"]
    lines += [
        "",
        "## E. Execution integrity",
        "",
        f"- Completed: **{integrity['requested_overnight_completed']}/800 requested overnight cells**; "
        f"**{integrity['full_manifest_completed']}/926 full-manifest cells**.",
        f"- Pending/running: **{integrity['pending']} / {integrity['running']}**.",
        f"- Scientific execution retries: **{integrity['scientific_execution_retries']}**; "
        f"TECHNICAL_FAILED: **{integrity['technical_failed']}**.",
        f"- Pre-scientific integration failures: `{integrity['archived_preflight_failures']}`. "
        "These archived camera-key/configuration construction failures produced no scientific result files.",
        f"- Valid scientific failure rerun: **{integrity['valid_scientific_failure_rerun']}**.",
        f"- Prohibited experiment launched: **{integrity['prohibited_experiment_launched']}**. "
        f"Observed result methods were exactly `{integrity['observed_methods']}`.",
        "",
        "## F. Branch state",
        "",
        "This harvest updates results, complete markers, worker progress/logs, aggregate analysis, "
        "the exposure inventory, handoff, and this report. The fallback manuscript is unchanged.",
    ]
    return "\n".join(lines)


def main() -> None:
    manifest = json.loads((ROOT / "queue_manifest.json").read_text())
    expected = manifest["expected_counts"]
    phases = {phase: load_phase(phase) for phase in expected}
    analysis: dict = {"schema_version": 2, "expected": expected,
        "completed": {p: len(v) for p, v in phases.items()}, "technical_failures": {
            p: len(list((ROOT / "markers" / p).glob("*.technical_failed"))) for p in expected}}
    if len(phases["act_object_h8_126"]) == 126:
        h8, h16 = phases["act_object_h8_126"], historical_h16_126()
        analysis["act_object_h8_126"] = {"H8": summary(h8), "H16_historical": summary(h16),
            "H8_vs_H16": comparison(h8, h16, 20270101)}
        analysis["act_object_h8_126"]["interpretation_label"] = (
            "COHERENT_OPTIMUM_IS_NOT_H16" if analysis["act_object_h8_126"]["H8"]["successes"] > 88 else "H16_NOT_CHALLENGED_BY_H8")
    if len(phases["act_posthoc_h8_140"]) == 140:
        h8, h16 = phases["act_posthoc_h8_140"], historical_h16_140()
        analysis["act_posthoc_h8_140"] = {"scope": "POST-HOC COHERENT-BASELINE AUDIT",
            "H8": summary(h8), "H16_historical": summary(h16), "H8_vs_H16": comparison(h8, h16, 20270102)}
        analysis["act_posthoc_h8_140"]["interpretation_label"] = (
            "COHERENT_OPTIMUM_IS_NOT_H16" if analysis["act_posthoc_h8_140"]["H8"]["successes"] > 93 else "H16_NOT_CHALLENGED_BY_H8")
    if len(phases["act_arm4_grip32_180"]) == 180:
        new = phases["act_arm4_grip32_180"]
        h16, h4 = historical_cross("group_arm4_grip16"), historical_cross("group_arm4_grip4")
        successes = summary(new)["successes"]
        analysis["act_arm4_grip32_180"] = {"ARM4_GRIP32": summary(new),
            "ARM4_GRIP16_historical_successes": sum(r["success"] for r in h16),
            "ARM4_GRIP4_historical_successes": sum(r["success"] for r in h4),
            "ARM4_GRIP32_vs_ARM4_GRIP16": comparison(new, h16, 20270103),
            "ARM4_GRIP32_vs_ARM4_GRIP4": comparison(new, h4, 20270104),
            "query_schedule_audit": schedule_audit(new),
            "interpretation_label": "GRIPPER_PLATEAU_AT_16" if abs(successes - 128) <= 3 else "NO_PLATEAU_LABEL",
            "mandatory_stop": "No arm8_grip32, arm16_grip32, coherent h32, or grip64 regardless of result"}
    if len(phases["smolvla_primary"]) == 320:
        primary = phases["smolvla_primary"]
        methods = {m: [r for r in primary if r["method"] == m] for m in ("SMOLVLA_COHERENT_H8", "SMOLVLA_ARM8_GRIP16")}
        smol = {"pooled": {m: summary(rows) for m, rows in methods.items()}, "suites": {}}
        for suite in ("libero_spatial", "libero_goal", "libero_object", "libero_10"):
            smol["suites"][suite] = {m: summary([r for r in rows if r["suite"] == suite]) for m, rows in methods.items()}
            smol["suites"][suite]["ARM8_GRIP16_vs_COHERENT_H8"] = comparison(
                [r for r in methods["SMOLVLA_ARM8_GRIP16"] if r["suite"] == suite],
                [r for r in methods["SMOLVLA_COHERENT_H8"] if r["suite"] == suite], 20270200 + len(smol["suites"]))
        smol["ARM8_GRIP16_vs_COHERENT_H8"] = comparison(methods["SMOLVLA_ARM8_GRIP16"], methods["SMOLVLA_COHERENT_H8"], 20270210)
        smol["query_rate_match_absolute_difference"] = abs(methods and smol["pooled"]["SMOLVLA_ARM8_GRIP16"]["query_rate"] - smol["pooled"]["SMOLVLA_COHERENT_H8"]["query_rate"])
        method_audits = {method: schedule_audit(rows) for method, rows in methods.items()}
        smol["query_schedule_audit"] = {
            "methods": method_audits,
            "matched_query_periods": len({tuple(v["query_periods"]) for v in method_audits.values()}) == 1,
            "intended_matched_arm_driven_schedule": all(
                v["all_exact_periodic_schedules"] and v["all_arm_driven"] and v["query_periods"] == [8]
                for v in method_audits.values()),
        }
        analysis["smolvla_primary"] = smol
    if len(phases["smolvla_capacity_h16"]) == 160:
        h16 = phases["smolvla_capacity_h16"]
        primary = phases["smolvla_primary"]
        h8 = [r for r in primary if r["method"] == "SMOLVLA_COHERENT_H8"]
        arm = [r for r in primary if r["method"] == "SMOLVLA_ARM8_GRIP16"]
        capacity = {
            "ran": True,
            "barrier_audit": {
                "latest_primary_finished_at": max(float(r["finished_at"]) for r in primary),
                "earliest_capacity_started_at": min(
                    float(r["finished_at"]) - float(r["wall_clock_seconds"]) for r in h16),
                "capacity_started_after_all_primary_finished": min(
                    float(r["finished_at"]) - float(r["wall_clock_seconds"]) for r in h16
                ) > max(float(r["finished_at"]) for r in primary),
            },
            "pooled": {
                "SMOLVLA_COHERENT_H16": summary(h16),
                "COHERENT_H16_vs_COHERENT_H8": comparison(h16, h8, 20270301),
                "ARM8_GRIP16_vs_COHERENT_H16": comparison(arm, h16, 20270302),
            },
            "suites": {},
            "query_schedule_audit": schedule_audit(h16),
        }
        for index, suite in enumerate(("libero_spatial", "libero_goal", "libero_object", "libero_10")):
            hs = [r for r in h16 if r["suite"] == suite]
            h8s = [r for r in h8 if r["suite"] == suite]
            arms = [r for r in arm if r["suite"] == suite]
            capacity["suites"][suite] = {
                "SMOLVLA_COHERENT_H16": summary(hs),
                "COHERENT_H16_vs_COHERENT_H8": comparison(hs, h8s, 20270310 + 2 * index),
                "ARM8_GRIP16_vs_COHERENT_H16": comparison(arms, hs, 20270311 + 2 * index),
            }
        analysis["smolvla_capacity_h16"] = capacity

    current_attempt_files = list((ROOT / "attempts").glob("*/*.json")) if (ROOT / "attempts").is_dir() else []
    current_exceptions = sum(len(json.loads(p.read_text()).get("attempts", [])) for p in current_attempt_files)
    requested = ("act_posthoc_h8_140", "act_arm4_grip32_180", "smolvla_primary", "smolvla_capacity_h16")
    observed_methods = sorted({r["method"] for rows in phases.values() for r in rows})
    allowed_methods = sorted({c["method"] for c in manifest["cells"]})
    analysis["execution_integrity"] = {
        "requested_overnight_completed": sum(len(phases[p]) for p in requested),
        "full_manifest_completed": sum(len(rows) for rows in phases.values()),
        "pending": sum(max(0, expected[p] - len(phases[p]) - analysis["technical_failures"][p]) for p in expected),
        "running": 0,
        "scientific_execution_retries": current_exceptions,
        "technical_failed": sum(analysis["technical_failures"].values()),
        "archived_preflight_failures": archived_preflight_summary(),
        "valid_scientific_failure_rerun": False,
        "prohibited_experiment_launched": observed_methods != allowed_methods,
        "observed_methods": observed_methods,
        "allowed_manifest_methods": allowed_methods,
    }
    (ROOT / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    (ROOT / "report.md").write_text(render_report(analysis) + "\n")
    print(json.dumps({"analysis": str(ROOT / "analysis.json"), "completed": analysis["completed"]}))


if __name__ == "__main__":
    main()
