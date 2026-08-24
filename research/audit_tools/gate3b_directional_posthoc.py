#!/usr/bin/env python3
"""Recompute the bounded post-hoc directional characterization of Gate-3B."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "2817411a4210b8611dc8dae5d32ec99fc6b94cf3"
MANIFEST_PATH = "research/audit_outputs/gate3b_rollout_manifest.json"
SUMMARY_PATH = "research/audit_outputs/gate3b_success_summary.json"
OUTPUT_DIR = ROOT / "research/audit_outputs"
BOOTSTRAP_DRAWS = 20_000
PAIRED_BOOTSTRAP_SEED = 20260830
TASK_BOOTSTRAP_SEED = 20261830
METHODS = ("FF", "OO", "FO", "OF")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", default=SOURCE_COMMIT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def git_blob(commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return completed.stdout


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def percentile_interval(draws: list[float]) -> tuple[float, float]:
    ordered = sorted(draws)

    def quantile(probability: float) -> float:
        position = (len(ordered) - 1) * probability
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    return quantile(0.025), quantile(0.975)


def validate_sources(
    manifest: dict[str, Any], final_summary: dict[str, Any]
) -> tuple[list[tuple[int, int]], dict[tuple[int, int, str], int]]:
    if manifest.get("complete") is not True:
        raise RuntimeError("Gate-3B manifest is not marked complete")
    if int(manifest.get("completed_episodes", -1)) != 400:
        raise RuntimeError("Gate-3B manifest does not contain 400 completed episodes")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 400:
        raise RuntimeError("Gate-3B episode list is incomplete")

    outcomes: dict[tuple[int, int, str], int] = {}
    for episode in episodes:
        key = (
            int(episode["task_id"]),
            int(episode["state_id"]),
            str(episode["method"]),
        )
        if key in outcomes:
            raise RuntimeError(f"duplicate task-state-method cell: {key}")
        if key[2] not in METHODS:
            raise RuntimeError(f"unexpected method: {key[2]}")
        if int(episode["steps"]) != int(episode["policy_queries"]):
            raise RuntimeError(f"query-cadence mismatch: {key}")
        if str(episode.get("status")) != "complete":
            raise RuntimeError(f"incomplete manifest cell: {key}")
        outcomes[key] = int(bool(episode["success"]))

    blocks = sorted({(task, state) for task, state, _ in outcomes})
    if len(blocks) != 100:
        raise RuntimeError(f"expected 100 paired blocks, found {len(blocks)}")
    if {task for task, _ in blocks} != set(range(10)):
        raise RuntimeError("task IDs differ from 0 through 9")
    for block in blocks:
        if {method for task, state, method in outcomes if (task, state) == block} != set(
            METHODS
        ):
            raise RuntimeError(f"incomplete factorial block: {block}")

    method_summary = final_summary["method_summary"]
    for method in METHODS:
        observed = sum(outcomes[(*block, method)] for block in blocks)
        if observed != int(method_summary[method]["successes"]):
            raise RuntimeError(f"manifest/summary success mismatch for {method}")
    return blocks, outcomes


def formula_values(
    blocks: list[tuple[int, int]],
    outcomes: dict[tuple[int, int, str], int],
    formula: Callable[[int, int, int, int], float],
) -> list[float]:
    return [
        formula(*(outcomes[(*block, method)] for method in METHODS)) for block in blocks
    ]


def exact_sign_flip(values: list[float]) -> dict[str, Any]:
    """Exact two-sided sign-flip diagnostic conditional on absolute contrasts."""
    scaled = [int(round(value * 2.0)) for value in values if value != 0.0]
    distribution: dict[int, int] = {0: 1}
    for magnitude in map(abs, scaled):
        updated: dict[int, int] = {}
        for total, count in distribution.items():
            updated[total + magnitude] = updated.get(total + magnitude, 0) + count
            updated[total - magnitude] = updated.get(total - magnitude, 0) + count
        distribution = updated
    observed = abs(sum(scaled))
    extreme = sum(count for total, count in distribution.items() if abs(total) >= observed)
    denominator = 2 ** len(scaled)
    return {
        "type": "exact two-sided block sign-flip conditional on absolute contrasts",
        "nonzero_blocks": len(scaled),
        "positive_blocks": sum(value > 0.0 for value in values),
        "negative_blocks": sum(value < 0.0 for value in values),
        "zero_blocks": sum(value == 0.0 for value in values),
        "p_value_raw": extreme / denominator if denominator else 1.0,
    }


def exact_mcnemar(values: list[float]) -> dict[str, Any]:
    positive = sum(value > 0.0 for value in values)
    negative = sum(value < 0.0 for value in values)
    discordant = positive + negative
    if discordant == 0:
        p_value = 1.0
    else:
        distance = abs(positive - discordant / 2.0)
        extreme = sum(
            math.comb(discordant, count)
            for count in range(discordant + 1)
            if abs(count - discordant / 2.0) >= distance - 1e-12
        )
        p_value = extreme / (2**discordant)
    return {
        "type": "exact two-sided McNemar/binomial diagnostic on discordant blocks",
        "discordant_blocks": discordant,
        "first_only_successes": positive,
        "second_only_successes": negative,
        "tied_blocks": len(values) - discordant,
        "p_value_raw": p_value,
    }


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (total - rank) * raw[name])
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def main() -> None:
    args = parse_args()
    manifest_blob = git_blob(args.source_commit, MANIFEST_PATH)
    summary_blob = git_blob(args.source_commit, SUMMARY_PATH)
    manifest = json.loads(manifest_blob)
    final_summary = json.loads(summary_blob)
    blocks, outcomes = validate_sources(manifest, final_summary)

    contrast_specs = {
        "arm_fresh_main": {
            "formula": "0.5*(FF+FO)-0.5*(OO+OF)",
            "status": "post-hoc factorial main effect",
            "values": formula_values(
                blocks, outcomes, lambda ff, oo, fo, of: 0.5 * (ff + fo - oo - of)
            ),
            "diagnostic": "sign_flip",
        },
        "gripper_old_main": {
            "formula": "0.5*(OO+FO)-0.5*(FF+OF)",
            "status": "post-hoc factorial main effect",
            "values": formula_values(
                blocks, outcomes, lambda ff, oo, fo, of: 0.5 * (oo + fo - ff - of)
            ),
            "diagnostic": "sign_flip",
        },
        "FO_minus_FF": {
            "formula": "FO-FF",
            "status": "post-hoc paired cell comparison",
            "values": formula_values(blocks, outcomes, lambda ff, oo, fo, of: fo - ff),
            "diagnostic": "mcnemar",
        },
        "FO_minus_OO": {
            "formula": "FO-OO",
            "status": "post-hoc paired cell comparison",
            "values": formula_values(blocks, outcomes, lambda ff, oo, fo, of: fo - oo),
            "diagnostic": "mcnemar",
        },
        "FO_minus_OF": {
            "formula": "FO-OF",
            "status": "post-hoc paired cell comparison",
            "values": formula_values(blocks, outcomes, lambda ff, oo, fo, of: fo - of),
            "diagnostic": "mcnemar",
        },
    }

    paired_draws = {name: [] for name in contrast_specs}
    paired_rng = random.Random(PAIRED_BOOTSTRAP_SEED)
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [paired_rng.randrange(len(blocks)) for _ in blocks]
        for name, spec in contrast_specs.items():
            values = spec["values"]
            paired_draws[name].append(sum(values[index] for index in sample) / len(sample))

    per_task: dict[str, list[float]] = {}
    for name, spec in contrast_specs.items():
        values = spec["values"]
        per_task[name] = [
            sum(value for value, (task, _) in zip(values, blocks) if task == task_id) / 10.0
            for task_id in range(10)
        ]
    task_draws = {name: [] for name in contrast_specs}
    task_rng = random.Random(TASK_BOOTSTRAP_SEED)
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [task_rng.randrange(10) for _ in range(10)]
        for name, task_values in per_task.items():
            task_draws[name].append(sum(task_values[index] for index in sample) / 10.0)

    diagnostics: dict[str, dict[str, Any]] = {}
    for name, spec in contrast_specs.items():
        if spec["diagnostic"] == "sign_flip":
            diagnostics[name] = exact_sign_flip(spec["values"])
        else:
            diagnostics[name] = exact_mcnemar(spec["values"])
    adjusted = holm_adjust(
        {name: float(diagnostic["p_value_raw"]) for name, diagnostic in diagnostics.items()}
    )

    results: dict[str, Any] = {}
    contrast_rows: list[dict[str, Any]] = []
    loto_rows: list[dict[str, Any]] = []
    for name, spec in contrast_specs.items():
        values = spec["values"]
        task_values = per_task[name]
        paired_ci = percentile_interval(paired_draws[name])
        task_ci = percentile_interval(task_draws[name])
        loto = [
            sum(value for task_id, value in enumerate(task_values) if task_id != omitted) / 9.0
            for omitted in range(10)
        ]
        diagnostic = {**diagnostics[name], "p_value_holm_five_contrasts": adjusted[name]}
        result = {
            "formula": spec["formula"],
            "status": spec["status"],
            "estimate": sum(values) / len(values),
            "paired_state_bootstrap_ci95": list(paired_ci),
            "task_cluster_bootstrap_ci95": list(task_ci),
            "per_task": task_values,
            "task_sign_counts": {
                "positive": sum(value > 0 for value in task_values),
                "zero": sum(value == 0 for value in task_values),
                "negative": sum(value < 0 for value in task_values),
            },
            "leave_one_task_out": loto,
            "leave_one_task_out_range": [min(loto), max(loto)],
            "exact_diagnostic": diagnostic,
        }
        results[name] = result
        contrast_rows.append(
            {
                "contrast": name,
                "status": spec["status"],
                "formula": spec["formula"],
                "estimate": result["estimate"],
                "paired_ci95_low": paired_ci[0],
                "paired_ci95_high": paired_ci[1],
                "task_cluster_ci95_low": task_ci[0],
                "task_cluster_ci95_high": task_ci[1],
                "positive_tasks": result["task_sign_counts"]["positive"],
                "zero_tasks": result["task_sign_counts"]["zero"],
                "negative_tasks": result["task_sign_counts"]["negative"],
                "loto_min": min(loto),
                "loto_max": max(loto),
                "exact_diagnostic": diagnostic["type"],
                "exact_p_raw": diagnostic["p_value_raw"],
                "exact_p_holm_five": diagnostic["p_value_holm_five_contrasts"],
            }
        )
        for omitted, value in enumerate(loto):
            loto_rows.append(
                {"contrast": name, "omitted_task": omitted, "estimate": value}
            )

    method_rates = {
        method: sum(outcomes[(*block, method)] for block in blocks) / len(blocks)
        for method in METHODS
    }
    per_task_rows: list[dict[str, Any]] = []
    for task_id in range(10):
        row: dict[str, Any] = {"task_id": task_id}
        task_blocks = [block for block in blocks if block[0] == task_id]
        for method in METHODS:
            row[method] = sum(outcomes[(*block, method)] for block in task_blocks) / 10.0
        for name in contrast_specs:
            row[name] = per_task[name][task_id]
        row["FO_ge_FF"] = row["FO"] >= row["FF"]
        row["FO_gt_FF"] = row["FO"] > row["FF"]
        row["FO_gt_OO"] = row["FO"] > row["OO"]
        per_task_rows.append(row)

    block_rows: list[dict[str, Any]] = []
    for block_index, (task_id, state_id) in enumerate(blocks):
        row = {"task_id": task_id, "state_id": state_id}
        for method in METHODS:
            row[method] = outcomes[(task_id, state_id, method)]
        for name, spec in contrast_specs.items():
            row[name] = spec["values"][block_index]
        block_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "gate3b_directional_contrasts.csv", contrast_rows)
    write_csv(args.output_dir / "gate3b_directional_per_task.csv", per_task_rows)
    write_csv(args.output_dir / "gate3b_directional_leave_one_task_out.csv", loto_rows)
    write_csv(args.output_dir / "gate3b_directional_block_contrasts.csv", block_rows)
    write_json(
        args.output_dir / "gate3b_directional_summary.json",
        {
            "schema_version": 1,
            "status": "bounded post-hoc characterization; not preregistered Gate-3B evidence",
            "source": {
                "commit": args.source_commit,
                "manifest_path": MANIFEST_PATH,
                "manifest_blob_sha256": sha256_bytes(manifest_blob),
                "final_summary_path": SUMMARY_PATH,
                "final_summary_blob_sha256": sha256_bytes(summary_blob),
                "completed_episodes": 400,
                "paired_task_state_blocks": 100,
            },
            "bootstrap": {
                "draws": BOOTSTRAP_DRAWS,
                "paired_state_seed": PAIRED_BOOTSTRAP_SEED,
                "task_cluster_seed": TASK_BOOTSTRAP_SEED,
                "implementation": "Python 3 standard-library random.Random; percentile interval with linear interpolation",
                "interval_status": "unadjusted exploratory 95% intervals",
            },
            "multiple_comparison_note": (
                "Exact diagnostics are exploratory. Raw p-values and Holm-adjusted "
                "values across the five requested contrasts are both reported."
            ),
            "method_success_rates": method_rates,
            "contrasts": results,
            "task_consistency": {
                "FO_ge_FF_tasks": sum(row["FO_ge_FF"] for row in per_task_rows),
                "FO_gt_FF_tasks": sum(row["FO_gt_FF"] for row in per_task_rows),
                "FO_equal_FF_tasks": sum(row["FO"] == row["FF"] for row in per_task_rows),
                "FO_gt_OO_tasks": sum(row["FO_gt_OO"] for row in per_task_rows),
            },
            "offline_d20_component_losses": {
                "status": "frozen RTX 5080 teacher-forced result from a8d49834650e17ca9cd6d413a7f64d0c5387fe4c",
                "fresh_arm_translation": 0.5957820292,
                "old_arm_translation": 0.5066667976,
                "fresh_arm_rotation_normalized": 1.1296224520,
                "old_arm_rotation_normalized": 1.0987722486,
                "fresh_gripper_sign_error": 0.3076001092,
                "old_gripper_sign_error": 0.2740024419,
                "descriptive_preference": {"arm": "old", "gripper": "old"},
            },
            "closed_loop_gate3b_marginal_preference": {
                "status": "secondary post-hoc pattern pending Gate-3C",
                "arm": "fresh",
                "gripper": "old",
                "FO_is_highest_observed_cell": True,
            },
        },
    )
    print(json.dumps({name: results[name]["estimate"] for name in results}, sort_keys=True))


if __name__ == "__main__":
    main()
