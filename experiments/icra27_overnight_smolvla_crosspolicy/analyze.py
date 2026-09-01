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
    return {
        "successes": sum(bool(r["success"]) for r in rows), "episodes": len(rows),
        "success_rate": sum(bool(r["success"]) for r in rows) / len(rows),
        "environment_steps": steps, "policy_queries": queries, "model_forward_count": sum(int(r.get("model_forward_count", r["policy_queries"])) for r in rows),
        "query_rate": queries / steps, "wall_clock_seconds": sum(float(r.get("wall_clock_seconds", 0)) for r in rows),
        "mean_wall_clock_seconds": float(np.mean([r.get("wall_clock_seconds", 0) for r in rows])),
        "mean_arm_source_age": sum(float(r["mean_arm_source_age"]) * int(r["environment_steps"]) for r in rows) / steps,
        "mean_gripper_source_age": sum(float(r["mean_gripper_source_age"]) * int(r["environment_steps"]) for r in rows) / steps,
    }


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


def main() -> None:
    manifest = json.loads((ROOT / "queue_manifest.json").read_text())
    expected = manifest["expected_counts"]
    phases = {phase: load_phase(phase) for phase in expected}
    analysis: dict = {"schema_version": 1, "expected": expected,
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
        analysis["smolvla_primary"] = smol
    if len(phases["smolvla_capacity_h16"]) == 160:
        analysis["smolvla_capacity_h16"] = {"SMOLVLA_COHERENT_H16": summary(phases["smolvla_capacity_h16"])}
    (ROOT / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    lines = ["# Overnight fixed-clock analysis", "", f"Completed cells: `{analysis['completed']}`.", ""]
    for name in ("act_object_h8_126", "act_posthoc_h8_140", "act_arm4_grip32_180", "smolvla_primary", "smolvla_capacity_h16"):
        if name in analysis:
            lines += [f"## {name}", "", "```json", json.dumps(analysis[name], indent=2), "```", ""]
    (ROOT / "report.md").write_text("\n".join(lines))
    print(json.dumps({"analysis": str(ROOT / "analysis.json"), "completed": analysis["completed"]}))


if __name__ == "__main__":
    main()
