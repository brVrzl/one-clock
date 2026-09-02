#!/usr/bin/env python3
"""Validate and analyze the complete frozen Track-A paired queue."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent


def exact_mcnemar(first_only: int, second_only: int) -> float:
    n = first_only + second_only
    if n == 0:
        return 1.0
    low = min(first_only, second_only)
    return min(1.0, 2.0 * sum(math.comb(n, k) for k in range(low + 1)) / (2**n))


def ci(values: np.ndarray) -> list[float]:
    return (100 * np.percentile(values, [2.5,97.5])).astype(float).tolist()


def load_all(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    # Gate before reading any result content, so a partial queue cannot be analyzed.
    missing = []
    for cell in manifest["cells"]:
        stem = cell["cell_id"]
        if not (ROOT / "track_a/results" / f"{stem}.json").is_file() or not (ROOT / "track_a/markers" / f"{stem}.complete").is_file():
            missing.append(stem)
    if missing:
        raise RuntimeError(f"Track-A queue is incomplete ({len(missing)} missing); no outcomes loaded")
    results = {}
    for cell in manifest["cells"]:
        path = ROOT / "track_a/results" / f"{cell['cell_id']}.json"
        row = json.loads(path.read_text())
        for key in ("cell_id","block_id","suite","task_id","state_id","environment_seed","policy_seed","method","checkpoint","preregistration_commit"):
            if row.get(key) != cell.get(key):
                raise RuntimeError(f"result identity mismatch {cell['cell_id']}:{key}")
        if row.get("status") != "COMPLETE" or not isinstance(row.get("success"), bool):
            raise RuntimeError(f"invalid completion status: {cell['cell_id']}")
        steps, queries = int(row["environment_steps"]), int(row["policy_queries"])
        if len(row["executed_actions"]) != steps or len(row["query_steps"]) != queries or row["model_forward_count"] != queries:
            raise RuntimeError(f"action/query count mismatch: {cell['cell_id']}")
        if row["method"] == "TE_DENSE":
            if row["query_steps"] != list(range(steps)) or len(row["candidate_counts"]) != steps or row["temporal_ensemble_coeff"] != 0.01:
                raise RuntimeError(f"noncanonical TE result: {cell['cell_id']}")
        else:
            period = min(int(row["arm_horizon"]), int(row["gripper_horizon"]))
            if row["query_steps"] != list(range(0, steps, period)) or len(row["source_ages"]) != steps:
                raise RuntimeError(f"fixed schedule mismatch: {cell['cell_id']}")
        results[cell["cell_id"]] = row
    return results


def contrast(manifest: dict[str, Any], results: dict[str, dict[str, Any]], spec: str) -> dict[str, Any]:
    first, second = spec.split("-")
    blocks = defaultdict(dict)
    for cell in manifest["cells"]:
        if cell["method"] in (first,second):
            blocks[cell["block_id"]][cell["method"]] = results[cell["cell_id"]]
    ordered = [blocks[key] for key in sorted(blocks)]
    x = np.asarray([row[first]["success"] for row in ordered], dtype=float)
    y = np.asarray([row[second]["success"] for row in ordered], dtype=float)
    delta = x-y
    first_only, second_only = int(np.sum((x==1)&(y==0))), int(np.sum((x==0)&(y==1)))
    pair_seed = manifest["statistics"]["paired_bootstrap_seeds"][spec]
    task_seed = manifest["statistics"]["task_cluster_bootstrap_seeds"][spec]
    rng = np.random.default_rng(pair_seed)
    paired_draws = delta[rng.integers(0,len(delta),size=(manifest["statistics"]["bootstrap_draws"],len(delta)))].mean(1)
    by_task = defaultdict(list)
    by_suite = defaultdict(list)
    for block, value in zip(ordered, delta, strict=True):
        row = block[first]
        by_task[(row["suite"],row["task_id"])].append(float(value))
        by_suite[row["suite"]].append(float(value))
    task_keys = sorted(by_task)
    task_values = np.asarray([np.mean(by_task[key]) for key in task_keys])
    rng = np.random.default_rng(task_seed)
    task_draws = task_values[rng.integers(0,len(task_values),size=(manifest["statistics"]["bootstrap_draws"],len(task_values)))].mean(1)
    loto = []
    for index,key in enumerate(task_keys):
        loto.append({"omitted_task": f"{key[0]}:task{key[1]}", "delta_percentage_points": float(100*np.delete(task_values,index).mean())})
    suites = sorted(by_suite)
    suite_descriptive = {suite: float(100*np.mean(by_suite[suite])) for suite in suites}
    leave_suite = {}
    for omitted in suites:
        kept = [v for suite in suites if suite != omitted for v in by_suite[suite]]
        leave_suite[omitted] = float(100*np.mean(kept))
    return {
        "contrast": spec, "first_successes": int(x.sum()), "second_successes": int(y.sum()), "N": len(x),
        "first_only": first_only, "second_only": second_only,
        "delta_percentage_points": float(100*delta.mean()),
        "exact_two_sided_mcnemar_p": exact_mcnemar(first_only,second_only),
        "paired_bootstrap_ci_percentage_points": ci(paired_draws),
        "task_cluster_bootstrap_ci_percentage_points": ci(task_draws),
        "per_task_delta_percentage_points": {f"{key[0]}:task{key[1]}": float(100*value) for key,value in zip(task_keys,task_values,strict=True)},
        "loto": loto, "loto_positive": int(sum(row["delta_percentage_points"]>0 for row in loto)), "loto_total": len(loto),
        "per_suite_descriptive_delta_percentage_points": suite_descriptive,
        "leave_one_suite_out_delta_percentage_points": leave_suite,
        "leave_one_suite_out_positive": int(sum(v>0 for v in leave_suite.values())),
    }


def main() -> None:
    manifest = json.loads((ROOT / "track_a_manifest.json").read_text())
    results = load_all(manifest)
    methods = manifest["condition_order"]
    summaries = {}
    for method in methods:
        rows = [results[cell["cell_id"]] for cell in manifest["cells"] if cell["method"] == method]
        steps, queries = sum(r["environment_steps"] for r in rows), sum(r["policy_queries"] for r in rows)
        summaries[method] = {"successes": sum(r["success"] for r in rows), "N": len(rows), "success_rate": float(np.mean([r["success"] for r in rows])), "environment_steps": steps, "policy_queries": queries, "query_rate": queries/steps, "wall_clock_seconds_sum": float(sum(r["wall_clock_seconds"] for r in rows)), "wall_clock_seconds_mean_episode": float(np.mean([r["wall_clock_seconds"] for r in rows]))}
    contrasts = {spec: contrast(manifest,results,spec) for spec in manifest["statistics"]["contrasts"]}
    c = contrasts
    penalty = c["H16-H4"]["delta_percentage_points"] > 0 and c["H16-H4"]["paired_bootstrap_ci_percentage_points"][0] > 0 and c["H16-H4"]["task_cluster_bootstrap_ci_percentage_points"][0] > 0
    dose = c["H4-H2"]["delta_percentage_points"] > 0 and c["H4-H2"]["paired_bootstrap_ci_percentage_points"][0] > 0 and c["H4-H2"]["task_cluster_bootstrap_ci_percentage_points"][0] > 0
    main_mitigation = c["ARM4_GRIP32-H4"]
    mechanism = penalty and main_mitigation["paired_bootstrap_ci_percentage_points"][0] > 0 and main_mitigation["task_cluster_bootstrap_ci_percentage_points"][0] > 0 and main_mitigation["loto_positive"] >= math.ceil(.9*main_mitigation["loto_total"]) and main_mitigation["leave_one_suite_out_positive"] == 3 and c["ARM2_GRIP16-H2"]["delta_percentage_points"] > 0
    practical = c["ARM4_GRIP32-H16"]
    method_pass = mechanism and practical["paired_bootstrap_ci_percentage_points"][0] > 0 and practical["task_cluster_bootstrap_ci_percentage_points"][0] > 0 and practical["loto_positive"] >= math.ceil(.9*practical["loto_total"])
    te = c["TE_DENSE-ARM4_GRIP32"]
    te_label = te["task_cluster_bootstrap_ci_percentage_points"][1] < 3 and summaries["TE_DENSE"]["query_rate"] >= .95 and .20 <= summaries["ARM4_GRIP32"]["query_rate"] <= .30
    moderator = json.loads((ROOT / "gripper_activity_moderator.json").read_text())
    mod_map = {(r["suite"],r["task_id"]): r["gripper_manipulation_frequency"] for r in moderator["tasks"]}
    effect_map = {(key.split(":task")[0],int(key.split(":task")[1])): value for key,value in main_mitigation["per_task_delta_percentage_points"].items()}
    keys = sorted(effect_map)
    rho,pvalue = spearmanr([mod_map[k] for k in keys],[effect_map[k] for k in keys])
    attempt_files = list((ROOT / "track_a/attempts").glob("*.json")) if (ROOT / "track_a/attempts").exists() else []
    technical_failures = sum(len(json.loads(p.read_text()).get("attempts",[])) for p in attempt_files)
    output = {
        "status": "COMPLETE", "preregistration_commit": manifest["preregistration_commit"],
        "validated_results": len(results), "task_count": manifest["task_count"], "paired_blocks": manifest["paired_block_count"],
        "method_summaries": summaries, "contrasts": contrasts,
        "moderator": {"definition": moderator["moderator_definition"], "task_count": len(keys), "spearman_rho": float(rho), "two_sided_p": float(pvalue)},
        "labels": {"PENALTY_4X_CONFIRMED": penalty, "DOSE_RESPONSE_SUPPORTED": dose, "MECHANISM_PASS_A": mechanism, "METHOD_PASS_A": method_pass, "QUERY_EFFICIENT_TE_LEVEL_PERFORMANCE": te_label},
        "query_rate_ratio_TE_DENSE_over_ARM4_GRIP32": summaries["TE_DENSE"]["query_rate"]/summaries["ARM4_GRIP32"]["query_rate"],
        "wall_clock_ratio_TE_DENSE_over_ARM4_GRIP32": summaries["TE_DENSE"]["wall_clock_seconds_sum"]/summaries["ARM4_GRIP32"]["wall_clock_seconds_sum"],
        "scientific_retry_count": 0, "technical_failed_attempt_count": technical_failures,
    }
    (ROOT / "track_a/analysis.json").write_text(json.dumps(output,indent=2)+"\n")
    lines = ["# Track-A cross-suite query-allocation confirmation", "", "## Decision labels", ""]
    for key,value in output["labels"].items(): lines.append(f"- `{key}`: **{'YES' if value else 'NO'}**")
    lines += ["", "## Conditions", "", "| Condition | Success | Rate | Queries | Query rate | Env steps | Mean wall-clock/episode |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method in methods:
        r=summaries[method]; lines.append(f"| {method} | {r['successes']}/{r['N']} | {100*r['success_rate']:.2f}% | {r['policy_queries']} | {r['query_rate']:.5f} | {r['environment_steps']} | {r['wall_clock_seconds_mean_episode']:.2f}s |")
    lines += ["", "## Paired contrasts", "", "| Contrast | Discordance | Delta pp | Exact p | Paired 95% CI | Task-cluster 95% CI | LOTO + | LOSO + |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for spec in manifest["statistics"]["contrasts"]:
        r=contrasts[spec]; lines.append(f"| {spec} | {r['first_only']}:{r['second_only']} | {r['delta_percentage_points']:+.2f} | {r['exact_two_sided_mcnemar_p']:.6g} | [{r['paired_bootstrap_ci_percentage_points'][0]:+.2f},{r['paired_bootstrap_ci_percentage_points'][1]:+.2f}] | [{r['task_cluster_bootstrap_ci_percentage_points'][0]:+.2f},{r['task_cluster_bootstrap_ci_percentage_points'][1]:+.2f}] | {r['loto_positive']}/{r['loto_total']} | {r['leave_one_suite_out_positive']}/3 |")
    lines += ["", "## All task-level deltas", ""]
    for spec in manifest["statistics"]["contrasts"]:
        lines += [f"### {spec}", "", "| Task | Delta pp |", "|---|---:|"] + [f"| {task} | {value:+.2f} |" for task,value in contrasts[spec]["per_task_delta_percentage_points"].items()] + [""]
    lines += ["## Frozen moderator", "", f"Spearman rho `{rho:.4f}`, two-sided p `{pvalue:.6g}`, all {len(keys)} tasks.", "", "No forbidden post-result method development was launched by this analysis.", ""]
    (ROOT / "track_a/report.md").write_text("\n".join(lines))
    print(json.dumps({"labels": output["labels"], "validated_results": len(results), "technical_failed_attempts": technical_failures},indent=2))


if __name__ == "__main__":
    main()
