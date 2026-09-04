#!/usr/bin/env python3
"""Validate and analyze the frozen amended Phase-1 discriminator."""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(ROOT))

from phase1_conditions import CONDITION_ORDER  # noqa: E402
from run_phase1 import build_cells, effective_protocol, frozen_commit, result_path, validate_result  # noqa: E402


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def percentile_ci(values: np.ndarray, seed: int, draws: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    sampled = values[indices].mean(axis=1)
    return [float(np.percentile(sampled, 2.5)), float(np.percentile(sampled, 97.5))]


def comparison(
    name: str,
    first: str,
    second: str,
    block_order: list[tuple[int, int]],
    outcomes: dict[tuple[int, int, str], int],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    first_values = np.asarray([outcomes[(task, state, first)] for task, state in block_order], dtype=np.int8)
    second_values = np.asarray([outcomes[(task, state, second)] for task, state in block_order], dtype=np.int8)
    differences = first_values.astype(float) - second_values.astype(float)
    first_only = int(np.count_nonzero((first_values == 1) & (second_values == 0)))
    second_only = int(np.count_nonzero((first_values == 0) & (second_values == 1)))
    discordant = first_only + second_only
    task_differences = {}
    for task_id in protocol["task_ids"]:
        positions = [i for i, (task, _) in enumerate(block_order) if task == int(task_id)]
        task_differences[str(task_id)] = float(differences[positions].mean())
    task_values = np.asarray(list(task_differences.values()), dtype=float)
    draws = int(protocol["statistics"]["bootstrap_draws"])
    paired_ci = percentile_ci(
        differences,
        int(protocol["statistics"]["paired_bootstrap_seeds"][name]),
        draws,
    )
    task_ci = percentile_ci(
        task_values,
        int(protocol["statistics"]["task_cluster_bootstrap_seeds"][name]),
        draws,
    )
    risk_difference = float(differences.mean())
    mcnemar_p = float(binomtest(first_only, discordant, 0.5).pvalue) if discordant else 1.0
    credible = risk_difference > 0 and mcnemar_p < 0.05 and paired_ci[0] > 0 and task_ci[0] > 0
    return {
        "contrast": name,
        "first": first,
        "second": second,
        "N": len(block_order),
        "first_successes": int(first_values.sum()),
        "second_successes": int(second_values.sum()),
        "first_success_rate": float(first_values.mean()),
        "second_success_rate": float(second_values.mean()),
        "paired_risk_difference": risk_difference,
        "paired_risk_difference_percentage_points": 100 * risk_difference,
        "discordant_first_only": first_only,
        "discordant_second_only": second_only,
        "discordant_total": discordant,
        "exact_two_sided_mcnemar_p": mcnemar_p,
        "paired_block_bootstrap_ci": paired_ci,
        "task_cluster_bootstrap_ci": task_ci,
        "bootstrap_draws": draws,
        "task_differences": task_differences,
        "credible_positive_evidence": bool(credible),
    }


def main() -> None:
    protocol = effective_protocol()
    commit = frozen_commit()
    cells = build_cells(protocol, commit)
    expected_paths = [result_path(cell) for cell in cells]
    missing = [str(path) for path in expected_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Phase-1 results incomplete: {len(missing)} files missing; first={missing[0]}")

    results = []
    technical_errors = []
    for cell, path in zip(cells, expected_paths, strict=True):
        try:
            results.append(validate_result(cell, path))
        except Exception as exc:
            technical_errors.append({"path": str(path), "error": str(exc)})
    if technical_errors:
        raise RuntimeError(f"Phase-1 result validation failed: {technical_errors[:3]}")

    block_order = [
        (int(task_id), int(state_id))
        for task_id in protocol["task_ids"]
        for state_id in protocol["state_ids_by_task"][str(task_id)]
    ]
    outcomes = {
        (int(result["task_id"]), int(result["state_id"]), str(result["method"])): int(bool(result["success"]))
        for result in results
    }
    if len(outcomes) != int(protocol["scientific_episodes"]):
        raise RuntimeError("result join is not one-to-one over every task-state-condition cell")

    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_task_condition: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        method = str(result["method"])
        task_id = int(result["task_id"])
        by_condition[method].append(result)
        by_task_condition[(task_id, method)].append(result)

    condition_summaries = {}
    query_accounting = {"conditions": {}, "per_task": {}}
    for method in CONDITION_ORDER:
        rows = by_condition[method]
        successes = sum(bool(row["success"]) for row in rows)
        calls = sum(int(row["policy_queries"]) for row in rows)
        steps = sum(int(row["environment_steps"]) for row in rows)
        summary = {
            "successes": int(successes),
            "N": len(rows),
            "success_rate": successes / len(rows),
            "total_policy_calls": calls,
            "total_environment_steps": steps,
            "mean_episode_policy_query_rate": float(np.mean([row["query_rate"] for row in rows])),
            "pooled_policy_query_rate": calls / steps,
        }
        condition_summaries[method] = summary
        query_accounting["conditions"][method] = {key: value for key, value in summary.items() if key not in {"successes", "N", "success_rate"}}

    per_task_rows = []
    for task_id in protocol["task_ids"]:
        query_accounting["per_task"][str(task_id)] = {}
        for method in CONDITION_ORDER:
            rows = by_task_condition[(int(task_id), method)]
            successes = sum(bool(row["success"]) for row in rows)
            calls = sum(int(row["policy_queries"]) for row in rows)
            steps = sum(int(row["environment_steps"]) for row in rows)
            row = {
                "task_id": int(task_id),
                "condition": method,
                "successes": int(successes),
                "N": len(rows),
                "success_rate": successes / len(rows),
                "total_policy_calls": calls,
                "total_environment_steps": steps,
                "mean_episode_policy_query_rate": float(np.mean([result["query_rate"] for result in rows])),
                "pooled_policy_query_rate": calls / steps,
            }
            per_task_rows.append(row)
            query_accounting["per_task"][str(task_id)][method] = {
                "total_policy_calls": calls,
                "total_environment_steps": steps,
                "mean_episode_policy_query_rate": row["mean_episode_policy_query_rate"],
                "pooled_policy_query_rate": row["pooled_policy_query_rate"],
            }

    contrast_specs = [
        ("ARM8_GRIP16-H8", "ARM8_GRIP16", "H8"),
        ("ARM8_GRIP32-H8", "ARM8_GRIP32", "H8"),
        ("ARM8_GRIP16-H16", "ARM8_GRIP16", "H16"),
        ("ARM8_GRIP32-H16", "ARM8_GRIP32", "H16"),
        ("ARM8_GRIP16-ARM8_GRIP32", "ARM8_GRIP16", "ARM8_GRIP32"),
        ("ARM8_GRIP16-ZOH8_GRIP16", "ARM8_GRIP16", "ZOH8_GRIP16"),
        ("ZOH8_GRIP16-H16", "ZOH8_GRIP16", "H16"),
        ("ARM4_GRIP32-H4", "ARM4_GRIP32", "H4"),
    ]
    contrasts = {
        name: comparison(name, first, second, block_order, outcomes, protocol)
        for name, first, second in contrast_specs
    }

    query_schedule_checks = {}
    for candidate in ("ARM8_GRIP16", "ARM8_GRIP32", "ZOH8_GRIP16"):
        exact_matched_length = 0
        matched_length_blocks = 0
        common_prefix_matches = 0
        for task_id, state_id in block_order:
            h8 = next(row for row in by_condition["H8"] if int(row["task_id"]) == task_id and int(row["state_id"]) == state_id)
            other = next(row for row in by_condition[candidate] if int(row["task_id"]) == task_id and int(row["state_id"]) == state_id)
            common_steps = min(int(h8["environment_steps"]), int(other["environment_steps"]))
            h8_prefix = [q for q in h8["query_steps"] if q < common_steps]
            other_prefix = [q for q in other["query_steps"] if q < common_steps]
            common_prefix_matches += int(h8_prefix == other_prefix)
            if int(h8["environment_steps"]) == int(other["environment_steps"]):
                matched_length_blocks += 1
                exact_matched_length += int(h8["query_steps"] == other["query_steps"])
        query_schedule_checks[f"{candidate}-H8"] = {
            "blocks": len(block_order),
            "common_executed_prefix_schedule_matches": common_prefix_matches,
            "matched_episode_length_blocks": matched_length_blocks,
            "exact_schedule_matches_among_matched_lengths": exact_matched_length,
            "status": "PASS" if common_prefix_matches == len(block_order) and exact_matched_length == matched_length_blocks else "FAIL",
        }
    if any(check["status"] != "PASS" for check in query_schedule_checks.values()):
        raise RuntimeError("actual arm8 policy-query schedules are not equivalent to H8")
    query_accounting["actual_arm8_schedule_equivalence"] = query_schedule_checks

    a8g16_strong = contrasts["ARM8_GRIP16-H8"]["credible_positive_evidence"] and contrasts["ARM8_GRIP16-H16"]["credible_positive_evidence"]
    a8g32_strong = contrasts["ARM8_GRIP32-H8"]["credible_positive_evidence"] and contrasts["ARM8_GRIP32-H16"]["credible_positive_evidence"]
    any_matched = contrasts["ARM8_GRIP16-H8"]["credible_positive_evidence"] or contrasts["ARM8_GRIP32-H8"]["credible_positive_evidence"]
    if a8g16_strong or a8g32_strong:
        decision = {"code": "A", "label": "EXECUTOR BRANCH VIABLE", "qualifying_candidates": [name for name, value in (("ARM8_GRIP16", a8g16_strong), ("ARM8_GRIP32", a8g32_strong)) if value]}
    elif any_matched:
        decision = {"code": "B", "label": "CONDITIONAL EXECUTION EFFECT ONLY", "qualifying_candidates": []}
    else:
        decision = {"code": "C", "label": "CLOSE EXECUTOR BRANCH", "qualifying_candidates": []}

    a8g16_vs_h16 = contrasts["ARM8_GRIP16-H16"]["paired_risk_difference"]
    zoh_vs_h16 = contrasts["ZOH8_GRIP16-H16"]["paired_risk_difference"]
    a8g16_vs_zoh = contrasts["ARM8_GRIP16-ZOH8_GRIP16"]["paired_risk_difference"]
    if a8g16_vs_zoh < 0:
        retained_interpretation = "ZOH8_GRIP16 outperformed ARM8_GRIP16; the component-chunk executor contribution is weakened further, and the simpler persistence explanation takes precedence."
    elif a8g16_vs_h16 >= 0 and zoh_vs_h16 < 0:
        retained_interpretation = "ARM8_GRIP16 matched or exceeded H16 while ZOH8_GRIP16 did not. The gain was not reproduced by matched-interval zero-order gripper holding, supporting retained chunk progression over simple persistence as the relevant distinction without establishing strict necessity."
    elif a8g16_vs_h16 >= 0 and zoh_vs_h16 >= 0:
        retained_interpretation = "ARM8_GRIP16 and ZOH8_GRIP16 both matched or exceeded H16. The gain is not attributable to retained chunk structure; the evidence is consistent with reduced gripper update rate or persistence being sufficient."
    else:
        retained_interpretation = "The candidate-specific paired evidence does not establish that retained chunk progression is necessary beyond scalar gripper persistence."

    success_vectors = {
        "block_order": [{"task_id": task, "state_id": state} for task, state in block_order],
        "vectors": {
            method: [outcomes[(task, state, method)] for task, state in block_order]
            for method in CONDITION_ORDER
        },
    }
    atomic_json(ROOT / "success_vectors.json", success_vectors)
    atomic_json(ROOT / "query_accounting.json", query_accounting)
    atomic_json(ROOT / "paired_inference.json", contrasts)
    with (ROOT / "per_task.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_task_rows[0]))
        writer.writeheader()
        writer.writerows(per_task_rows)
    with (ROOT / "task_deltas.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["contrast", "task_id", "paired_risk_difference", "percentage_points"])
        for name, result in contrasts.items():
            for task_id, delta in result["task_differences"].items():
                writer.writerow([name, task_id, delta, 100 * delta])
    raw_index = {
        "schema_version": 1,
        "record_count": len(results),
        "format": "One complete machine-readable JSON result per task-state-condition cell; each includes success, exact query steps, policy-call count, environment-step count, source ages, and executed actions.",
        "paths": [str(path.relative_to(REPO_ROOT)) for path in expected_paths],
    }
    atomic_json(ROOT / "raw_results_index.json", raw_index)

    analysis = {
        "status": "COMPLETE",
        "preregistration_commit": commit,
        "amendment": protocol["amendment"],
        "technical_validation_errors": 0,
        "conditions": condition_summaries,
        "contrasts": contrasts,
        "query_schedule_checks": query_schedule_checks,
        "coherent_horizon_curve": {method: condition_summaries[method] for method in ("H4", "H8", "H16")},
        "retained_sequence_interpretation": retained_interpretation,
        "decision": decision,
    }
    atomic_json(ROOT / "analysis.json", analysis)

    def pct(value: float) -> str:
        return f"{100 * value:.2f}%"

    lines = [
        "# Phase-1 prospective executor result",
        "",
        f"Decision: **{decision['code']} — {decision['label']}**.",
        "",
        "## Condition results",
        "",
        "| Condition | Successes/N | Success rate | Policy calls | Environment steps | Mean episode query rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in CONDITION_ORDER:
        row = condition_summaries[method]
        lines.append(f"| {method} | {row['successes']}/{row['N']} | {pct(row['success_rate'])} | {row['total_policy_calls']} | {row['total_environment_steps']} | {row['mean_episode_policy_query_rate']:.6f} |")
    lines += ["", "## Frozen paired contrasts", "", "| Contrast | Successes | Risk difference | Discordant +/− | Exact McNemar p | Paired 95% CI | Task-cluster 95% CI | Credible + |", "|---|---:|---:|---:|---:|---:|---:|:---:|"]
    for name, _, _ in contrast_specs:
        row = contrasts[name]
        lines.append(
            f"| {name} | {row['first_successes']}/{row['N']} vs {row['second_successes']}/{row['N']} | {row['paired_risk_difference_percentage_points']:+.2f} pp | {row['discordant_first_only']}/{row['discordant_second_only']} | {row['exact_two_sided_mcnemar_p']:.6g} | [{100*row['paired_block_bootstrap_ci'][0]:+.2f}, {100*row['paired_block_bootstrap_ci'][1]:+.2f}] pp | [{100*row['task_cluster_bootstrap_ci'][0]:+.2f}, {100*row['task_cluster_bootstrap_ci'][1]:+.2f}] pp | {'yes' if row['credible_positive_evidence'] else 'no'} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        analysis["retained_sequence_interpretation"],
        "Both arm8 candidates are reported separately; no best-of-candidates test was constructed.",
        "All 10 task-level deltas are in `task_deltas.csv`; per-task success and query accounting are in `per_task.csv`.",
        "",
        f"PHASE-1 EXECUTOR RESULT: {decision['code']} — {decision['label']}",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "conditions": condition_summaries, "contrasts": contrasts}, indent=2))


if __name__ == "__main__":
    main()
