#!/usr/bin/env python3
"""Analyze the complete frozen component-temporal-reuse pilot.

The input must contain all protocol task-condition blocks.  This deliberate
guard prevents a partially written shard from being presented as a completed
experiment.  It writes a JSON analysis, a Markdown report, and three figures.

Example (after merge_pilot_results.py has confirmed all 80 blocks):
  /path/to/python analyze_final_pilot.py \
    --pilot pilot_results.json --protocol protocol.json \
    --output-dir final_analysis

This script never runs a policy or edits a pilot result.  It is CPU-only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


CORE_CONTRASTS = (
    ("fo_vs_fresh", "FO - fresh", "fo", "fresh"),
    ("fo_vs_full_old", "FO - full-old", "fo", "full_old"),
    ("fo_vs_reverse", "FO - reverse", "fo", "reverse"),
    ("reverse_vs_fresh", "reverse - fresh", "reverse", "fresh"),
    ("full_old_vs_fresh", "full-old - fresh", "full_old", "fresh"),
)


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise SystemExit(f"cannot parse JSON {path}: {error}") from error


def condition_names(protocol: dict[str, Any]) -> dict[str, Any]:
    """Return the frozen names for each age and factorial cell."""
    result: dict[str, Any] = {"fresh": None, "by_age": {}}
    for condition in protocol["conditions"]:
        arm = int(condition["arm_source_age"])
        grip = int(condition["gripper_source_age"])
        name = str(condition["name"])
        if arm == grip == 0:
            result["fresh"] = name
        elif arm == 0 and grip > 0:
            result["by_age"].setdefault(grip, {})["fo"] = name
        elif grip == 0 and arm > 0:
            result["by_age"].setdefault(arm, {})["reverse"] = name
        elif arm == grip and arm > 0:
            result["by_age"].setdefault(arm, {})["full_old"] = name
    expected_ages = [int(age) for age in protocol["source_ages_steps"]]
    if result["fresh"] is None or set(result["by_age"]) != set(expected_ages):
        raise SystemExit("protocol does not define fresh plus all requested factorial conditions")
    for age in expected_ages:
        if set(result["by_age"][age]) != {"fo", "reverse", "full_old"}:
            raise SystemExit(f"protocol has incomplete factorial conditions at age {age}")
    return result


def expected_task_keys(protocol: dict[str, Any]) -> list[str]:
    return [f"{task['suite']}:task{int(task['task_id'])}" for task in protocol["tasks"]]


def validate_complete(pilot: dict[str, Any], protocol: dict[str, Any], names: dict[str, Any]) -> None:
    """Reject incomplete or malformed inputs before producing any output."""
    expected_tasks = set(expected_task_keys(protocol))
    found_tasks = set(pilot.get("tasks", {}))
    if found_tasks != expected_tasks:
        missing = sorted(expected_tasks - found_tasks)
        extra = sorted(found_tasks - expected_tasks)
        raise SystemExit(f"pilot task set is incomplete or unexpected; missing={missing}, extra={extra}")
    required_conditions = {names["fresh"]}
    for values in names["by_age"].values():
        required_conditions.update(values.values())
    episodes = int(protocol["environment"]["episodes_per_task"])
    found_blocks = 0
    for task_key in expected_tasks:
        conditions = pilot["tasks"][task_key].get("conditions", {})
        if set(conditions) != required_conditions:
            missing = sorted(required_conditions - set(conditions))
            extra = sorted(set(conditions) - required_conditions)
            raise SystemExit(f"{task_key} has incomplete conditions; missing={missing}, extra={extra}")
        for name, result in conditions.items():
            successes = result.get("successes")
            if not isinstance(successes, list) or len(successes) != episodes:
                raise SystemExit(f"{task_key}/{name} does not contain {episodes} paired outcomes")
            if int(result.get("episodes", -1)) != episodes:
                raise SystemExit(f"{task_key}/{name} has inconsistent episode count")
            if int(result.get("success_count", -1)) != sum(bool(value) for value in successes):
                raise SystemExit(f"{task_key}/{name} success_count disagrees with successes")
            found_blocks += 1
    expected_blocks = len(expected_tasks) * len(required_conditions)
    if found_blocks != expected_blocks or found_blocks != 80:
        raise SystemExit(f"requires exactly 80 complete blocks; found {found_blocks}")


def exact_mcnemar(two_zero: int, reference_only: int, candidate_only: int, two_one: int) -> float | None:
    """Two-sided exact binomial McNemar p-value on discordant paired outcomes."""
    del two_zero, two_one
    discordant = reference_only + candidate_only
    if discordant == 0:
        return None
    lower_tail = sum(math.comb(discordant, i) for i in range(min(reference_only, candidate_only) + 1))
    return min(1.0, 2.0 * lower_tail / (2**discordant))


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile of empty input")
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def bootstrap_mean_ci(values: list[float], *, draws: int, seed: int) -> list[float] | None:
    if not values:
        return None
    generator = random.Random(seed)
    count = len(values)
    samples = [mean([values[generator.randrange(count)] for _ in range(count)]) for _ in range(draws)]
    return [percentile(samples, 0.025), percentile(samples, 0.975)]


def paired_result(candidate: list[bool], reference: list[bool], *, seed: int) -> dict[str, Any]:
    if len(candidate) != len(reference):
        raise ValueError("unpaired outcomes")
    both_failure = reference_only = candidate_only = both_success = 0
    differences: list[float] = []
    for candidate_success, reference_success in zip(candidate, reference):
        candidate_value = int(bool(candidate_success))
        reference_value = int(bool(reference_success))
        differences.append(float(candidate_value - reference_value))
        if candidate_value and reference_value:
            both_success += 1
        elif candidate_value:
            candidate_only += 1
        elif reference_value:
            reference_only += 1
        else:
            both_failure += 1
    count = len(candidate)
    return {
        "episodes": count,
        "candidate_successes": sum(bool(value) for value in candidate),
        "reference_successes": sum(bool(value) for value in reference),
        "absolute_success_difference": mean(differences),
        "paired_contingency": {
            "both_fail": both_failure,
            "reference_only_success": reference_only,
            "candidate_only_success": candidate_only,
            "both_success": both_success,
        },
        "exact_mcnemar_two_sided_p": exact_mcnemar(
            both_failure, reference_only, candidate_only, both_success
        ),
        "paired_episode_bootstrap_ci95": bootstrap_mean_ci(differences, draws=10_000, seed=seed),
        "_paired_differences": differences,
    }


def holm_adjust(rows: list[dict[str, Any]], field: str = "exact_mcnemar_two_sided_p") -> None:
    """Add Holm-adjusted p-values across the planned 15 core contrasts."""
    valid = [(index, row[field]) for index, row in enumerate(rows) if row.get(field) is not None]
    ordered = sorted(valid, key=lambda item: item[1])
    total = len(ordered)
    previous = 0.0
    adjusted: dict[int, float] = {}
    for rank, (index, value) in enumerate(ordered):
        current = min(1.0, (total - rank) * float(value))
        previous = max(previous, current)
        adjusted[index] = previous
    for index, row in enumerate(rows):
        row["exact_mcnemar_p_holm_15_core"] = adjusted.get(index)


def episode_metrics(result: dict[str, Any]) -> list[dict[str, float | int | None]]:
    """Use episode-level logs when present; otherwise retain published aggregates."""
    details = result.get("episodes_detail")
    if isinstance(details, list) and details:
        return [
            {
                "environment_steps": int(item["environment_steps"]),
                "policy_queries": int(item["policy_queries"]),
                "arm_age": float(item["mean_arm_source_age_steps"]),
                "gripper_age": float(item["mean_gripper_source_age_steps"]),
                "completion_steps": item.get("completion_steps"),
                "success": int(bool(item["success"])),
            }
            for item in details
        ]
    successes = [bool(value) for value in result["successes"]]
    return [
        {
            "environment_steps": int(result["environment_steps"]) // len(successes),
            "policy_queries": int(result["policy_queries"]) // len(successes),
            "arm_age": float(result["mean_arm_source_age_steps"]),
            "gripper_age": float(result["mean_gripper_source_age_steps"]),
            "completion_steps": None,
            "success": int(success),
        }
        for success in successes
    ]


def execution_metrics(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    episodes = [episode for result in results for episode in episode_metrics(result)]
    successes = [episode for episode in episodes if episode["success"]]
    total_steps = sum(int(episode["environment_steps"]) for episode in episodes)
    total_queries = sum(int(episode["policy_queries"]) for episode in episodes)
    completions = [float(episode["completion_steps"]) for episode in successes if episode["completion_steps"] is not None]
    return {
        "episodes": len(episodes),
        "successes": len(successes),
        "success_rate": len(successes) / len(episodes),
        "policy_queries": total_queries,
        "environment_steps": total_steps,
        "queries_per_environment_step": total_queries / total_steps if total_steps else None,
        "mean_arm_source_age_steps": mean(float(episode["arm_age"]) for episode in episodes),
        "mean_gripper_source_age_steps": mean(float(episode["gripper_age"]) for episode in episodes),
        "successful_episode_completion_steps": {
            "count": len(completions),
            "mean": mean(completions) if completions else None,
        },
    }


def make_analysis(pilot: dict[str, Any], protocol: dict[str, Any], names: dict[str, Any]) -> dict[str, Any]:
    task_spec = {f"{task['suite']}:task{int(task['task_id'])}": task for task in protocol["tasks"]}
    ages = [int(age) for age in protocol["source_ages_steps"]]
    fresh_name = names["fresh"]
    per_task: list[dict[str, Any]] = []

    for task_key in expected_task_keys(protocol):
        task = pilot["tasks"][task_key]
        specification = task_spec[task_key]
        row: dict[str, Any] = {
            "task_key": task_key,
            "suite": specification["suite"],
            "task_id": int(specification["task_id"]),
            "task_name": specification["task_name"],
            "conditions": {},
            "contrasts": [],
        }
        for name, result in task["conditions"].items():
            row["conditions"][name] = execution_metrics([result])
        for age_index, age in enumerate(ages):
            condition_at_age = names["by_age"][age]
            label_to_name = {"fresh": fresh_name, **condition_at_age}
            for contrast_index, (key, label, candidate_kind, reference_kind) in enumerate(CORE_CONTRASTS):
                candidate_name = label_to_name[candidate_kind]
                reference_name = label_to_name[reference_kind]
                comparison = paired_result(
                    task["conditions"][candidate_name]["successes"],
                    task["conditions"][reference_name]["successes"],
                    seed=10_000 + 100 * int(specification["task_id"]) + 10 * age_index + contrast_index,
                )
                comparison.update(
                    {
                        "age_steps": age,
                        "contrast_key": key,
                        "contrast_label": label,
                        "candidate_condition": candidate_name,
                        "reference_condition": reference_name,
                    }
                )
                row["contrasts"].append(comparison)
        holm_adjust(row["contrasts"])
        per_task.append(row)

    suites = list(dict.fromkeys(task["suite"] for task in protocol["tasks"]))
    aggregate: dict[str, Any] = {}
    for scope in [*suites, "all_tasks"]:
        rows = [row for row in per_task if scope == "all_tasks" or row["suite"] == scope]
        condition_metrics: dict[str, Any] = {}
        all_names = [fresh_name] + [names["by_age"][age][kind] for age in ages for kind in ("fo", "full_old", "reverse")]
        for condition_name in all_names:
            task_values = [row["conditions"][condition_name] for row in rows]
            condition_metrics[condition_name] = {
                "task_macro_success_rate": mean(value["success_rate"] for value in task_values),
                "task_macro_mean_arm_source_age_steps": mean(
                    value["mean_arm_source_age_steps"] for value in task_values
                ),
                "task_macro_mean_gripper_source_age_steps": mean(
                    value["mean_gripper_source_age_steps"] for value in task_values
                ),
                "task_macro_queries_per_environment_step": mean(
                    value["queries_per_environment_step"] for value in task_values
                ),
                "pooled_descriptive": execution_metrics(
                    [pilot["tasks"][row["task_key"]]["conditions"][condition_name] for row in rows]
                ),
            }
        contrasts: list[dict[str, Any]] = []
        for age_index, age in enumerate(ages):
            for contrast_index, (key, label, candidate_kind, reference_kind) in enumerate(CORE_CONTRASTS):
                matching = [
                    contrast
                    for row in rows
                    for contrast in row["contrasts"]
                    if contrast["age_steps"] == age and contrast["contrast_key"] == key
                ]
                differences = [float(item["absolute_success_difference"]) for item in matching]
                pooled_differences = [
                    difference for item in matching for difference in item["_paired_differences"]
                ]
                reference_only = sum(item["paired_contingency"]["reference_only_success"] for item in matching)
                candidate_only = sum(item["paired_contingency"]["candidate_only_success"] for item in matching)
                both_fail = sum(item["paired_contingency"]["both_fail"] for item in matching)
                both_success = sum(item["paired_contingency"]["both_success"] for item in matching)
                contrasts.append(
                    {
                        "age_steps": age,
                        "contrast_key": key,
                        "contrast_label": label,
                        "candidate_condition": names["by_age"][age].get(candidate_kind, fresh_name),
                        "reference_condition": names["by_age"][age].get(reference_kind, fresh_name),
                        "task_macro_absolute_success_difference": mean(differences),
                        "task_macro_bootstrap_ci95": bootstrap_mean_ci(
                            differences, draws=10_000, seed=20_000 + 100 * age_index + contrast_index
                        ),
                        "pooled_descriptive": {
                            "episodes": len(pooled_differences),
                            "absolute_success_difference": mean(pooled_differences),
                            "paired_contingency": {
                                "both_fail": both_fail,
                                "reference_only_success": reference_only,
                                "candidate_only_success": candidate_only,
                                "both_success": both_success,
                            },
                            "exact_mcnemar_two_sided_p": exact_mcnemar(
                                both_fail, reference_only, candidate_only, both_success
                            ),
                            "paired_episode_bootstrap_ci95": bootstrap_mean_ci(
                                pooled_differences, draws=10_000, seed=30_000 + 100 * age_index + contrast_index
                            ),
                        },
                    }
                )
        # Holm correction is applied only to the 15 predeclared core contrasts within a scope.
        # The p-value is nested in the pooled descriptive record, so expose a flat temporary
        # view rather than treating a dotted path as a dictionary key.
        flat_for_holm = [
            {"p": contrast["pooled_descriptive"]["exact_mcnemar_two_sided_p"]} for contrast in contrasts
        ]
        holm_adjust(flat_for_holm, field="p")
        for contrast, adjusted in zip(contrasts, flat_for_holm):
            contrast["pooled_descriptive"]["exact_mcnemar_p_holm_15_core"] = adjusted[
                "exact_mcnemar_p_holm_15_core"
            ]
        aggregate[scope] = {"condition_metrics": condition_metrics, "contrasts": contrasts}

    # JSON should expose data rather than private bootstrap samples.
    for row in per_task:
        for contrast in row["contrasts"]:
            contrast.pop("_paired_differences", None)
    return {
        "analysis_status": "complete_frozen_protocol",
        "analysis_definition": {
            "primary_question": "Does success degrade differently as arm versus gripper source age increases?",
            "inference_unit": "paired initial state within task",
            "task_macro_note": "Task macro summaries weight each task equally; their bootstrap resamples task-level paired contrasts.",
            "pooled_note": "Pooled summaries are descriptive across paired episodes and do not replace task-macro results.",
            "mcnemar_note": "Exact two-sided McNemar values are paired diagnostics; Holm correction is across the 15 core age-by-contrast tests within the displayed scope.",
            "ci_note": "Percentile bootstrap intervals use 10,000 deterministic resamples and are descriptive unless a later confirmatory plan designates a scope.",
        },
        "protocol": {
            "source_ages_steps": ages,
            "source_ages_seconds": {str(age): float(age) / float(protocol["environment"]["fps"]) for age in ages},
            "condition_names": names,
            "tasks": len(per_task),
            "blocks": len(per_task) * (1 + 3 * len(ages)),
        },
        "per_task": per_task,
        "aggregates": aggregate,
    }


def markdown_table(headers: list[str], rows: Iterable[Iterable[str]]) -> list[str]:
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return output


def p_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def ci_text(value: list[float] | None) -> str:
    return "n/a" if value is None else f"[{value[0]:+.3f}, {value[1]:+.3f}]"


def write_report(analysis: dict[str, Any], output: Path) -> None:
    names = analysis["protocol"]["condition_names"]
    ages = analysis["protocol"]["source_ages_steps"]
    all_names = [names["fresh"]] + [
        names["by_age"][age][kind] for age in ages for kind in ("fo", "full_old", "reverse")
    ]
    lines = [
        "# Complete frozen-pilot analysis",
        "",
        "The analysis guard accepted all 80 planned task-condition blocks. Results below are paired by initial state; no partial shard is treated as complete.",
        "",
        "## Per-task success and execution metrics",
        "",
    ]
    headers = ["suite", "task"] + all_names
    rows = []
    for task in analysis["per_task"]:
        values = []
        for name in all_names:
            metric = task["conditions"][name]
            values.append(f"{metric['successes']}/{metric['episodes']} ({metric['success_rate']:.1%})")
        rows.append([task["suite"], str(task["task_id"]), *values])
    lines.extend(markdown_table(headers, rows))

    lines.extend(["", "## Per-task execution provenance", ""])
    provenance_rows = []
    for task in analysis["per_task"]:
        for name in all_names:
            metric = task["conditions"][name]
            completion = metric["successful_episode_completion_steps"]
            provenance_rows.append(
                [
                    task["task_key"],
                    name,
                    f"{metric['queries_per_environment_step']:.3f}",
                    f"{metric['mean_arm_source_age_steps']:.2f}",
                    f"{metric['mean_gripper_source_age_steps']:.2f}",
                    f"{completion['mean']:.1f}" if completion["mean"] is not None else "n/a",
                ]
            )
    lines.extend(
        markdown_table(
            ["task", "condition", "queries/step", "mean arm age", "mean gripper age", "success completion steps"],
            provenance_rows,
        )
    )

    lines.extend(["", "## Per-task core paired contrasts", "", "Candidate minus reference; contingency is candidate-only/reference-only success. Exact McNemar values are Holm-adjusted across the 15 core comparisons within each task.", ""])
    contrast_rows = []
    for task in analysis["per_task"]:
        for contrast in task["contrasts"]:
            contingency = contrast["paired_contingency"]
            contrast_rows.append(
                [
                    task["task_key"],
                    str(contrast["age_steps"]),
                    contrast["contrast_label"],
                    f"{contrast['candidate_successes']}/{contrast['episodes']} vs {contrast['reference_successes']}/{contrast['episodes']}",
                    f"{contingency['candidate_only_success']}/{contingency['reference_only_success']}",
                    f"{contrast['absolute_success_difference']:+.3f}",
                    ci_text(contrast["paired_episode_bootstrap_ci95"]),
                    p_text(contrast["exact_mcnemar_two_sided_p"]),
                    p_text(contrast["exact_mcnemar_p_holm_15_core"]),
                ]
            )
    lines.extend(markdown_table(["task", "age", "contrast", "success", "C-only/R-only", "delta", "paired bootstrap CI", "McNemar p", "Holm p"], contrast_rows))

    for scope, summary in analysis["aggregates"].items():
        lines.extend(["", f"## {scope}: task-macro and pooled descriptive summaries", ""])
        metric_rows = []
        for name, values in summary["condition_metrics"].items():
            pooled = values["pooled_descriptive"]
            metric_rows.append(
                [
                    name,
                    f"{values['task_macro_success_rate']:.3f}",
                    f"{pooled['successes']}/{pooled['episodes']} ({pooled['success_rate']:.1%})",
                    f"{values['task_macro_queries_per_environment_step']:.3f}",
                    f"{values['task_macro_mean_arm_source_age_steps']:.2f}",
                    f"{values['task_macro_mean_gripper_source_age_steps']:.2f}",
                    f"{pooled['successful_episode_completion_steps']['mean']:.1f}" if pooled["successful_episode_completion_steps"]["mean"] is not None else "n/a",
                ]
            )
        lines.extend(markdown_table(["condition", "task-macro success", "pooled success", "queries/step", "arm age", "gripper age", "success completion steps"], metric_rows))
        lines.extend(["", "Task macro is the primary presentation for heterogeneous tasks. Pooled numbers are descriptive only.", ""])
        summary_rows = []
        for contrast in summary["contrasts"]:
            pooled = contrast["pooled_descriptive"]
            contingency = pooled["paired_contingency"]
            summary_rows.append(
                [
                    str(contrast["age_steps"]),
                    contrast["contrast_label"],
                    f"{contrast['task_macro_absolute_success_difference']:+.3f}",
                    ci_text(contrast["task_macro_bootstrap_ci95"]),
                    f"{pooled['absolute_success_difference']:+.3f}",
                    f"{contingency['candidate_only_success']}/{contingency['reference_only_success']}",
                    p_text(pooled["exact_mcnemar_two_sided_p"]),
                    p_text(pooled["exact_mcnemar_p_holm_15_core"]),
                ]
            )
        lines.extend(markdown_table(["age", "contrast", "task-macro delta", "task bootstrap CI", "pooled delta", "C-only/R-only", "pooled McNemar p", "pooled Holm p"], summary_rows))

    lines.extend([
        "",
        "## Prespecified sensitivity interpretation",
        "",
        "For each age, compare FO minus fresh (fresh arm, stale gripper) with reverse minus fresh (stale arm, fresh gripper). The direction and uncertainty of those two task-macro contrasts answer the temporal-sensitivity question; no monotonicity is assumed.",
        "",
        "Figures are saved under `figures/`: `arm_vs_gripper_age_sensitivity.png`, `per_task_source_age_deltas.png`, and `success_vs_source_age.png`. If a disagreement JSON is supplied, a fourth outcome-association plot is added.",
        "",
    ])
    output.write_text("\n".join(lines))


def contrast_lookup(summary: dict[str, Any], age: int, key: str) -> dict[str, Any]:
    for contrast in summary["contrasts"]:
        if contrast["age_steps"] == age and contrast["contrast_key"] == key:
            return contrast
    raise KeyError((age, key))


def render_figures(analysis: dict[str, Any], figures_dir: Path, disagreement: dict[str, Any] | None) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("matplotlib is required for figures; rerun with --skip-figures for tables only") from error

    figures_dir.mkdir(parents=True, exist_ok=True)
    ages = analysis["protocol"]["source_ages_steps"]
    overall = analysis["aggregates"]["all_tasks"]

    # 1. Direct arm-age versus gripper-age sensitivity.
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    for key, label, color, marker in (
        ("fo_vs_fresh", "stale gripper (FO)", "#2166ac", "o"),
        ("reverse_vs_fresh", "stale arm (reverse)", "#b2182b", "s"),
    ):
        values = [contrast_lookup(overall, age, key) for age in ages]
        y = [value["task_macro_absolute_success_difference"] for value in values]
        lower = [value["task_macro_bootstrap_ci95"][0] for value in values]
        upper = [value["task_macro_bootstrap_ci95"][1] for value in values]
        # A task-level bootstrap interval can miss the point estimate for a
        # finite sample.  Matplotlib requires nonnegative error lengths; keep
        # the interval endpoints unchanged in the analysis JSON and clamp only
        # the renderer's invalid side.
        lower_error = [max(0.0, a - b) for a, b in zip(y, lower)]
        upper_error = [max(0.0, b - a) for a, b in zip(upper, y)]
        axis.errorbar(ages, y, yerr=[lower_error, upper_error], label=label, color=color, marker=marker, capsize=3)
    axis.axhline(0, color="0.35", linewidth=1)
    axis.set(xlabel="source age (environment steps)", ylabel="task-macro success difference vs fresh", title="Component temporal sensitivity")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "arm_vs_gripper_age_sensitivity.png", dpi=300)
    plt.close(fig)

    # 2. Per-task factorial deltas, preserving task heterogeneity.
    tasks = analysis["per_task"]
    fig, axes = plt.subplots(1, len(ages), figsize=(4.4 * len(ages), 5.0), sharey=True)
    if len(ages) == 1:
        axes = [axes]
    colours = {"fo_vs_fresh": "#2166ac", "reverse_vs_fresh": "#b2182b", "full_old_vs_fresh": "#4d9221"}
    labels = {"fo_vs_fresh": "FO", "reverse_vs_fresh": "reverse", "full_old_vs_fresh": "full-old"}
    for axis, age in zip(axes, ages):
        positions = list(range(len(tasks)))
        width = 0.23
        for offset, key in zip((-width, 0.0, width), labels):
            values = [contrast_lookup({"contrasts": task["contrasts"]}, age, key)["absolute_success_difference"] for task in tasks]
            axis.bar([position + offset for position in positions], values, width=width, color=colours[key], label=labels[key])
        axis.axhline(0, color="0.35", linewidth=1)
        axis.set_title(f"age {age}")
        axis.set_xticks(positions, [f"{task['suite'].replace('libero_', '')}\n{task['task_id']}" for task in tasks], rotation=45, ha="right")
    axes[0].set_ylabel("paired success difference vs fresh")
    axes[-1].legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(figures_dir / "per_task_source_age_deltas.png", dpi=300)
    plt.close(fig)

    # 3. Success trajectories over source age, with all three factorial cells.
    names = analysis["protocol"]["condition_names"]
    fresh = overall["condition_metrics"][names["fresh"]]["task_macro_success_rate"]
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    for kind, label, color, marker in (
        ("fo", "FO: fresh arm / stale gripper", "#2166ac", "o"),
        ("reverse", "reverse: stale arm / fresh gripper", "#b2182b", "s"),
        ("full_old", "full-old", "#4d9221", "^") ,
    ):
        y = [fresh] + [overall["condition_metrics"][names["by_age"][age][kind]]["task_macro_success_rate"] for age in ages]
        axis.plot([0, *ages], y, label=label, color=color, marker=marker)
    axis.set(xlabel="source age (environment steps)", ylabel="task-macro success rate", title="Success versus source age")
    axis.set_ylim(0, 1)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "success_vs_source_age.png", dpi=300)
    plt.close(fig)

    if disagreement is None:
        return
    rows = [
        row
        for group in ("arm", "gripper")
        for row in disagreement.get("aggregate_outcome_association", {}).get(group, {}).get("rows", [])
    ]
    if not rows:
        return
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    for group, color in (("arm", "#b2182b"), ("gripper", "#2166ac")):
        group_rows = [row for row in rows if row.get("condition", "").startswith("reverse") == (group == "arm") or row.get("condition", "").startswith("fo") == (group == "gripper")]
        if not group_rows:
            continue
        x = [float(row["disagreement_l2"]) for row in group_rows]
        y = [float(row["outcome_changed"]) + (0.03 if group == "arm" else -0.03) for row in group_rows]
        axis.scatter(x, y, label=group, color=color, alpha=0.65)
    axis.set(xlabel="episode mean same-target disagreement", ylabel="paired outcome changed vs fresh", title="Exploratory disagreement/outcome association")
    axis.set_yticks([0, 1], ["same outcome", "outcome changed"])
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "disagreement_vs_outcome_change.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--disagreement", type=Path, help="optional output from analyze_disagreement.py")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    pilot = load_json(args.pilot)
    protocol = load_json(args.protocol)
    names = condition_names(protocol)
    validate_complete(pilot, protocol, names)
    analysis = make_analysis(pilot, protocol, names)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    write_report(analysis, args.output_dir / "report.md")
    if not args.skip_figures:
        disagreement = load_json(args.disagreement) if args.disagreement else None
        render_figures(analysis, args.output_dir / "figures", disagreement)
    print(json.dumps({"output_dir": str(args.output_dir), "status": analysis["analysis_status"]}, indent=2))


if __name__ == "__main__":
    main()
