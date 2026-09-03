#!/usr/bin/env python3
"""Post-hoc command-discontinuity characterization on completed trajectories."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
CONFIRMATION_ROOT = Path("/home/wjq/workspace/one-clock/experiments/cross_suite_confirmation")
OUTPUT = ROOT / "command_discontinuity"
TRACK_A_SHA = "40549d876c0e09fad4e8033b3206f6018f53ece5"
BOOTSTRAP_SEED = 20260903
BOOTSTRAP_DRAWS = 20_000
GROUPS = {"translation": (0, 1, 2), "rotation": (3, 4, 5), "gripper": (6,)}
CONFIRMATION_TASKS = {
    "libero_goal": (4, 6, 7, 8, 9),
    "libero_10": (0, 2, 4, 6, 7),
}
COMPARISONS = (
    ("confirmation", "FRESH", "REVERSE20", "Fresh-A20G0"),
    ("confirmation", "FRESH", "HARD_H16", "Fresh-coherent_H16"),
    ("confirmation", "REVERSE20", "HARD_H16", "A20G0-coherent_H16"),
    ("track_a", "ARM4_GRIP32", "H4", "ARM4_GRIP32-H4"),
    ("track_a", "ARM2_GRIP16", "H2", "ARM2_GRIP16-H2"),
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def group_magnitude(values: np.ndarray, dims: tuple[int, ...]) -> np.ndarray:
    selected = values[:, dims]
    if len(dims) == 1:
        return np.abs(selected[:, 0])
    return np.sqrt(np.mean(np.square(selected), axis=1))


def summarize_trajectory(
    family: str,
    condition: str,
    task: str,
    block_id: str,
    actions: np.ndarray,
    source_queries: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < 2:
        raise RuntimeError(f"invalid executed action array: {family}/{condition}/{block_id}")
    if not np.isfinite(actions).all():
        raise RuntimeError(f"non-finite action: {family}/{condition}/{block_id}")
    d1 = np.diff(actions, axis=0)
    d2 = np.diff(actions, n=2, axis=0)
    grip_switch = np.not_equal(np.sign(actions[1:, 6]), np.sign(actions[:-1, 6]))
    summaries: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for group, dims in GROUPS.items():
        q = source_queries[group]
        if q.shape != (len(actions),):
            raise RuntimeError(f"invalid source-query array: {family}/{condition}/{block_id}/{group}")
        switch = q[1:] != q[:-1]
        magnitude1 = group_magnitude(d1, dims)
        magnitude2 = group_magnitude(d2, dims) if len(d2) else np.empty(0)
        switch_values = magnitude1[switch]
        same_values = magnitude1[~switch]
        both = len(switch_values) > 0 and len(same_values) > 0
        summaries.append({
            "family": family,
            "condition": condition,
            "task": task,
            "block_id": block_id,
            "group": group,
            "environment_steps": len(actions),
            "d1_count": len(magnitude1),
            "d1_mean": float(np.mean(magnitude1)),
            "d1_median": float(np.median(magnitude1)),
            "d2_count": len(magnitude2),
            "d2_mean": float(np.mean(magnitude2)) if len(magnitude2) else "",
            "d2_median": float(np.median(magnitude2)) if len(magnitude2) else "",
            "gripper_state_switch_probability": float(np.mean(grip_switch)) if group == "gripper" else "",
            "source_chunk_switch_count": int(switch.sum()),
            "same_source_chunk_count": int((~switch).sum()),
            "source_comparison_status": "AVAILABLE" if both else "STRUCTURALLY_UNAVAILABLE",
            "d1_switch_mean": float(np.mean(switch_values)) if both else "",
            "d1_same_source_mean": float(np.mean(same_values)) if both else "",
            "d1_switch_minus_same_source": float(np.mean(switch_values) - np.mean(same_values)) if both else "",
        })
        for label, values in (("source_chunk_switch", switch_values), ("same_source_chunk", same_values)):
            transitions.append({
                "family": family,
                "condition": condition,
                "task": task,
                "block_id": block_id,
                "group": group,
                "transition_type": label,
                "count": len(values),
                "d1_sum": float(values.sum()),
            })
    return summaries, transitions


def load_confirmation() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = json.loads((CONFIRMATION_ROOT / "protocol.json").read_text())
    if protocol["cohort"]["primary_blocks_per_condition"] != 140:
        raise RuntimeError("confirmation primary cohort drift")
    summaries: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for suite, tasks in CONFIRMATION_TASKS.items():
        for task_id in tasks:
            path = CONFIRMATION_ROOT / "results" / f"{suite}_task{task_id}.json"
            payload = json.loads(path.read_text())
            task = f"{suite}:task{task_id}"
            if payload["suite"] != suite or int(payload["task_id"]) != task_id:
                raise RuntimeError(f"confirmation task identity drift: {path}")
            for condition in ("FRESH", "REVERSE20", "HARD_H16"):
                episodes = payload["episodes"][condition]
                if len(episodes) != 14:
                    raise RuntimeError(f"confirmation block-count drift: {task}/{condition}")
                for episode in episodes:
                    steps = episode["step_log"]
                    actions = np.asarray([row["action"] for row in steps], dtype=np.float64)
                    arm_q = np.asarray([row["arm_source_query_q"] for row in steps], dtype=np.int64)
                    grip_q = np.asarray([row["gripper_source_query_q"] for row in steps], dtype=np.int64)
                    block_id = f"{task}:state{int(episode['requested_initial_state_id']):02d}"
                    rows, source_rows = summarize_trajectory(
                        "confirmation", condition, task, block_id, actions,
                        {"translation": arm_q, "rotation": arm_q, "gripper": grip_q},
                    )
                    summaries.extend(rows)
                    transitions.extend(source_rows)
    expected = 140 * 3 * len(GROUPS)
    if len(summaries) != expected:
        raise RuntimeError(f"confirmation trajectory count drift: {len(summaries)} != {expected}")
    return summaries, transitions


def load_track_a() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads((ROOT / "track_a_manifest.json").read_text())
    canonical = json.loads((ROOT / "track_a/analysis.json").read_text())
    if canonical.get("status") != "COMPLETE" or canonical.get("validated_results") != 2700:
        raise RuntimeError("Track A is not canonically complete")
    selected = {"H4", "ARM4_GRIP32", "H2", "ARM2_GRIP16"}
    cells = [cell for cell in manifest["cells"] if cell["method"] in selected]
    if len(cells) != 1800:
        raise RuntimeError(f"Track-A selected-cell count drift: {len(cells)}")
    summaries: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for cell in cells:
        path = ROOT / "track_a/results" / f"{cell['cell_id']}.json"
        payload = json.loads(path.read_text())
        if (
            payload.get("status") != "COMPLETE"
            or payload.get("cell_id") != cell["cell_id"]
            or payload.get("block_id") != cell["block_id"]
            or payload.get("preregistration_commit") != TRACK_A_SHA
        ):
            raise RuntimeError(f"Track-A identity drift: {path}")
        actions = np.asarray(payload["executed_actions"], dtype=np.float64)
        ages = payload["source_ages"]
        if len(ages) != len(actions) or int(payload["environment_steps"]) != len(actions):
            raise RuntimeError(f"Track-A step-count drift: {path}")
        targets = np.arange(len(actions), dtype=np.int64)
        arm_q = targets - np.asarray([row["arm"] for row in ages], dtype=np.int64)
        grip_q = targets - np.asarray([row["gripper"] for row in ages], dtype=np.int64)
        task = f"{cell['suite']}:task{int(cell['task_id'])}"
        rows, source_rows = summarize_trajectory(
            "track_a", cell["method"], task, cell["block_id"], actions,
            {"translation": arm_q, "rotation": arm_q, "gripper": grip_q},
        )
        summaries.extend(rows)
        transitions.extend(source_rows)
    expected = 1800 * len(GROUPS)
    if len(summaries) != expected:
        raise RuntimeError(f"Track-A trajectory count drift: {len(summaries)} != {expected}")
    return summaries, transitions


def condition_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["condition"], row["group"])].append(row)
    output: list[dict[str, Any]] = []
    for (family, condition, group), values in sorted(grouped.items()):
        record = {
            "family": family,
            "condition": condition,
            "group": group,
            "trajectory_count": len(values),
        }
        for metric in ("d1_mean", "d1_median", "d2_mean", "d2_median"):
            record[f"mean_trajectory_{metric}"] = float(np.mean([float(row[metric]) for row in values]))
        record["mean_trajectory_gripper_state_switch_probability"] = (
            float(np.mean([float(row["gripper_state_switch_probability"]) for row in values]))
            if group == "gripper" else ""
        )
        output.append(record)
    return output


def source_transition_summaries(
    rows: list[dict[str, Any]], transition_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["condition"], row["group"])].append(row)
    output: list[dict[str, Any]] = []
    for (family, condition, group), values in sorted(grouped.items()):
        switch_count = sum(int(row["source_chunk_switch_count"]) for row in values)
        same_count = sum(int(row["same_source_chunk_count"]) for row in values)
        both = switch_count > 0 and same_count > 0
        # Recompute from transition sufficient statistics so trajectories lacking one class
        # still contribute to the pooled class that they do contain.
        matching = [row for row in transition_rows
                    if row["family"] == family and row["condition"] == condition and row["group"] == group]
        switch_sum = sum(float(row["d1_sum"]) for row in matching if row["transition_type"] == "source_chunk_switch")
        same_sum = sum(float(row["d1_sum"]) for row in matching if row["transition_type"] == "same_source_chunk")
        output.append({
            "family": family,
            "condition": condition,
            "group": group,
            "source_comparison_status": "AVAILABLE" if both else "STRUCTURALLY_UNAVAILABLE",
            "source_chunk_switch_count": switch_count,
            "same_source_chunk_count": same_count,
            "d1_switch_mean": switch_sum / switch_count if both else "",
            "d1_same_source_mean": same_sum / same_count if both else "",
            "d1_switch_minus_same_source": switch_sum / switch_count - same_sum / same_count if both else "",
        })
    return output


def contrast_summaries(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contrast_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metrics = ("d1_mean", "d1_median", "d2_mean", "d2_median", "gripper_state_switch_probability")
    for family, first, second, label in COMPARISONS:
        for group in GROUPS:
            blocks_first = {
                row["block_id"]: row for row in rows
                if row["family"] == family and row["condition"] == first and row["group"] == group
            }
            blocks_second = {
                row["block_id"]: row for row in rows
                if row["family"] == family and row["condition"] == second and row["group"] == group
            }
            if set(blocks_first) != set(blocks_second):
                raise RuntimeError(f"unpaired blocks: {label}/{group}")
            block_ids = sorted(blocks_first)
            tasks = sorted({str(blocks_first[block]["task"]) for block in block_ids})
            for metric in metrics:
                if metric == "gripper_state_switch_probability" and group != "gripper":
                    continue
                deltas = {
                    block: float(blocks_first[block][metric]) - float(blocks_second[block][metric])
                    for block in block_ids
                }
                per_task = {
                    task: float(np.mean([deltas[block] for block in block_ids if blocks_first[block]["task"] == task]))
                    for task in tasks
                }
                task_values = np.asarray([per_task[task] for task in tasks], dtype=np.float64)
                sampled = rng.integers(0, len(tasks), size=(BOOTSTRAP_DRAWS, len(tasks)))
                boot = np.mean(task_values[sampled], axis=1)
                contrast_rows.append({
                    "family": family,
                    "contrast": label,
                    "first": first,
                    "second": second,
                    "group": group,
                    "metric": metric,
                    "paired_block_count": len(block_ids),
                    "task_count": len(tasks),
                    "mean_paired_difference": float(np.mean(list(deltas.values()))),
                    "median_paired_difference": float(np.median(list(deltas.values()))),
                    "task_cluster_ci_low": float(np.percentile(boot, 2.5)),
                    "task_cluster_ci_high": float(np.percentile(boot, 97.5)),
                })
                for task, value in per_task.items():
                    task_rows.append({
                        "family": family,
                        "contrast": label,
                        "group": group,
                        "metric": metric,
                        "task": task,
                        "mean_paired_difference": value,
                    })
    return contrast_rows, task_rows


def main() -> None:
    confirmation, confirmation_transitions = load_confirmation()
    track_a, track_a_transitions = load_track_a()
    rows = confirmation + track_a
    transition_rows = confirmation_transitions + track_a_transitions
    condition_rows = condition_summaries(rows)
    source_rows = source_transition_summaries(rows, transition_rows)
    contrasts, task_contrasts = contrast_summaries(rows)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "trajectory_summaries.csv", rows)
    write_csv(OUTPUT / "condition_summaries.csv", condition_rows)
    write_csv(OUTPUT / "source_transition_summaries.csv", source_rows)
    write_csv(OUTPUT / "contrasts.csv", contrasts)
    write_csv(OUTPUT / "task_contrasts.csv", task_contrasts)
    result = {
        "status": "COMPLETE",
        "label": "POST_HOC_COMMAND_DISCONTINUITY_CHARACTERIZATION",
        "specification": "POST_HOC_COMMAND_DISCONTINUITY_CHARACTERIZATION.md",
        "specification_commit": "4c26241e0b31ca77ee9c895b504d6bb74120c838",
        "no_rerollout": True,
        "reviewer_supplement_inputs_loaded": False,
        "action_units": "controller_native_command",
        "physical_jerk_claimed": False,
        "bootstrap": {"unit": "task", "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED},
        "input_counts": {
            "confirmation_primary_trajectories": 140 * 3,
            "track_a_trajectories": 450 * 4,
        },
        "condition_summaries": condition_rows,
        "source_transition_summaries": source_rows,
        "contrasts": contrasts,
        "post_hoc_interpretation": {
            "arm_reduced_discontinuity_hypothesis": "NOT_SUPPORTED_BY_REQUESTED_D1_COMPARISONS",
            "gripper_sparse_transition_characterization": "COMPATIBLE_AT_ARM4_GRIP32_VS_H4_BUT_NOT_UNIFORM_AT_ARM2_GRIP16_VS_H2",
            "causal_mechanism_claimed": False,
        },
    }
    (OUTPUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    d1_rows = [row for row in contrasts if row["metric"] == "d1_mean"]
    grip_switch_rows = [row for row in contrasts if row["metric"] == "gripper_state_switch_probability"]
    lines = [
        "# Post-hoc command-discontinuity characterization", "",
        "Status: **COMPLETE**", "",
        "Computed only from existing completed trajectories. No reviewer-supplement artifact or success outcome was used.", "",
        "Quantities are controller-native command differences, not physical jerk. Translation, rotation, and gripper are reported separately.", "",
        "Fresh and A20G0 have no same-source transitions by construction; those comparisons are recorded as `STRUCTURALLY_UNAVAILABLE`, not as outcomes.", "",
        "## Mean first-difference contrasts", "",
        "Positive values mean greater command variation in the first named condition. Intervals are task-cluster percentile 95% intervals.", "",
        "| Contrast | Group | Mean paired difference | 95% interval |", "|---|---|---:|---:|",
    ]
    for row in d1_rows:
        lines.append(
            f"| {row['contrast']} | {row['group']} | {row['mean_paired_difference']:.6f} | "
            f"[{row['task_cluster_ci_low']:.6f}, {row['task_cluster_ci_high']:.6f}] |"
        )
    lines.extend(["", "## Gripper state-switch contrasts", "",
                  "| Contrast | Mean paired probability difference | 95% interval |", "|---|---:|---:|"])
    for row in grip_switch_rows:
        lines.append(
            f"| {row['contrast']} | {row['mean_paired_difference']:.6f} | "
            f"[{row['task_cluster_ci_low']:.6f}, {row['task_cluster_ci_high']:.6f}] |"
        )
    lines.extend([
        "", "## Post-hoc interpretation", "",
        "The requested D1 comparisons do not support a simple account in which arm temporal benefits arise from reduced executed arm-command discontinuity. Coherent H16 had greater translation and rotation D1 than both Fresh and A20G0, and both split Track-A methods had slightly greater arm-D1 point estimates than their matched global-horizon comparators.", "",
        "ARM4_GRIP32 reduced gripper D1 and executed gripper state-switch probability relative to H4. The corresponding ARM2_GRIP16-H2 differences were small with task-cluster intervals spanning zero. This is compatible with sparse-transition timing mattering at the first operating point, but is not uniform evidence and does not establish a causal or forecasting mechanism.", "",
        "Canonical numerical outputs are `condition_summaries.csv`, `source_transition_summaries.csv`, `contrasts.csv`, `task_contrasts.csv`, `trajectory_summaries.csv`, and `analysis.json`.", "",
    ])
    (OUTPUT / "report.md").write_text("\n".join(lines))
    print(json.dumps({"status": "COMPLETE", "trajectory_group_rows": len(rows), "contrasts": len(contrasts)}, indent=2))


if __name__ == "__main__":
    main()
