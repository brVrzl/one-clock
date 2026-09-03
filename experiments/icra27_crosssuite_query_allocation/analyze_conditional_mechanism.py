#!/usr/bin/env python3
"""Final prospective task-level persistence/sensitivity associations."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from safetensors import safe_open
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = Path("/home/wjq/research-assets/datasets/HuggingFaceVLA_libero")


def action_std(checkpoint: Path) -> np.ndarray:
    with safe_open(checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors", framework="numpy") as handle:
        value = np.asarray(handle.get_tensor("action.std"), dtype=np.float64)
    if value.shape != (7,) or np.any(value <= 0) or not np.isfinite(value).all():
        raise RuntimeError(f"invalid frozen action std: {checkpoint}")
    return value


def main() -> None:
    amendment = json.loads((ROOT / "final_analysis_only_amendment.json").read_text())
    if amendment["status"] != "FROZEN_BEFORE_TRACK_A_CANONICAL_ANALYSIS_AND_REVIEWER_SUPPLEMENT_OUTCOMES":
        raise RuntimeError("analysis-only amendment is not frozen")
    protocol = json.loads((ROOT.parent / "cross_suite_confirmation/protocol.json").read_text())
    outcome = json.loads((ROOT.parent / "cross_suite_confirmation/analysis.json").read_text())
    tasks = [task for task in protocol["cohort"]["tasks"] if task["role"] == "primary_unseen_to_executor_development"]
    expected = amendment["confirmation_tasks"]
    if [f"{task['suite']}:task{task['task_id']}" for task in tasks] != expected:
        raise RuntimeError("confirmation task order drift")

    info = json.loads((DATASET_ROOT / "meta/info.json").read_text())
    fps = float(info["fps"])
    if fps != 10:
        raise RuntimeError(f"dataset fps drift: {fps}")
    task_table = pq.read_table(DATASET_ROOT / "meta/tasks.parquet").to_pylist()
    language = {int(row["task_index"]): row["__index_level_0__"] for row in task_table}

    episode_owner: dict[int, str] = {}
    task_config: dict[str, dict] = {}
    for task in tasks:
        label = f"{task['suite']}:task{task['task_id']}"
        checkpoint = Path(task["checkpoint"])
        config = json.loads((checkpoint / "train_config.json").read_text())
        if config["dataset"]["repo_id"] != "HuggingFaceVLA/libero":
            raise RuntimeError(f"unexpected dataset for {label}")
        episodes = [int(value) for value in config["dataset"]["episodes"]]
        if not episodes:
            raise RuntimeError(f"empty episode list for {label}")
        for episode in episodes:
            if episode in episode_owner:
                raise RuntimeError(f"episode {episode} assigned twice")
            episode_owner[episode] = label
        task_config[label] = {"checkpoint": checkpoint, "episodes": episodes, "std": action_std(checkpoint), "task_name": task["task_name"]}

    dataset = pads.dataset(DATASET_ROOT / "data", format="parquet")
    table = dataset.to_table(
        columns=["episode_index", "frame_index", "task_index", "action"],
        filter=pads.field("episode_index").isin(sorted(episode_owner)),
    )
    episodes: dict[int, list[dict]] = {episode: [] for episode in episode_owner}
    for row in table.to_pylist():
        episode = int(row["episode_index"])
        if episode in episodes:
            episodes[episode].append(row)

    aggregates = {label: {"transitions": 0, "adjacent": 0, "arm_sse": 0.0, "arm_n": 0} for label in expected}
    for episode, rows in episodes.items():
        label = episode_owner[episode]
        rows.sort(key=lambda row: int(row["frame_index"]))
        if [int(row["frame_index"]) for row in rows] != list(range(len(rows))):
            raise RuntimeError(f"non-contiguous demonstration episode {episode}")
        task_indices = {int(row["task_index"]) for row in rows}
        if len(task_indices) != 1 or language[next(iter(task_indices))] != task_config[label]["task_name"]:
            raise RuntimeError(f"task-language mapping mismatch for {label}, episode {episode}")
        actions = np.asarray([row["action"] for row in rows], dtype=np.float64)
        delta = np.diff(actions, axis=0)
        signs = np.sign(actions[:, 6])
        aggregates[label]["transitions"] += int(np.count_nonzero(signs[1:] != signs[:-1]))
        aggregates[label]["adjacent"] += len(delta)
        normalized_arm_delta = delta[:, :6] / task_config[label]["std"][:6]
        aggregates[label]["arm_sse"] += float(np.square(normalized_arm_delta).sum())
        aggregates[label]["arm_n"] += int(normalized_arm_delta.size)

    outcome_by_task = {row["task_label"]: row for row in outcome["per_task"] if row["role"] == "primary_unseen_to_executor_development"}
    rows = []
    for label in expected:
        counts = aggregates[label]
        result = outcome_by_task[label]
        rows.append({
            "task": label,
            "training_demonstration_episodes": len(task_config[label]["episodes"]),
            "adjacent_pairs": counts["adjacent"],
            "gripper_transitions": counts["transitions"],
            "gripper_transition_density_per_second": counts["transitions"] / (counts["adjacent"] / fps),
            "arm_normalized_adjacent_variation": float(np.sqrt(counts["arm_sse"] / counts["arm_n"])),
            "A0G0_success_rate": result["FRESH_success_rate"],
            "A0G20_success_rate": result["FO20_success_rate"],
            "A20G0_success_rate": result["REVERSE20_success_rate"],
            "Delta_G": result["FO20_success_rate"] - result["FRESH_success_rate"],
            "Delta_A": result["REVERSE20_success_rate"] - result["FRESH_success_rate"],
        })

    gripper = spearmanr(
        [row["gripper_transition_density_per_second"] for row in rows],
        [row["Delta_G"] for row in rows],
    )
    arm = spearmanr(
        [row["arm_normalized_adjacent_variation"] for row in rows],
        [row["Delta_A"] for row in rows],
    )
    output = {
        "status": "COMPLETE",
        "analysis_only_amendment": "e2fb21b",
        "dataset_fps": fps,
        "success_outcomes_source": "completed frozen 140-block cross-suite confirmation analysis",
        "track_a_outcomes_loaded": False,
        "task_count": len(rows),
        "gripper_prediction": {"expected_direction": "rho < 0", "spearman_rho": float(gripper.statistic), "two_sided_p_descriptive": float(gripper.pvalue)},
        "arm_prediction": {"expected_direction": "rho < 0", "spearman_rho": float(arm.statistic), "two_sided_p_descriptive": float(arm.pvalue)},
        "task_values": rows,
    }
    output_dir = ROOT / "track_b/conditional_mechanism"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    with (output_dir / "task_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    lines = [
        "# Conditional task-level mechanism analysis", "",
        "This prospective analysis uses all ten primary tasks from the completed frozen 140-block confirmation and each task checkpoint's exact training-episode list. It does not load Track-A outcomes.", "",
        f"Gripper transition-density versus Delta_G: Spearman rho `{gripper.statistic:.6f}` (expected <0; descriptive two-sided p `{gripper.pvalue:.6g}`).",
        f"Arm temporal-variation versus Delta_A: Spearman rho `{arm.statistic:.6f}` (expected <0; descriptive two-sided p `{arm.pvalue:.6g}`).", "",
        "All ten task values are in `task_metrics.csv` and `summary.json`.", "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))
    print(json.dumps({"status": "COMPLETE", "gripper_rho": float(gripper.statistic), "arm_rho": float(arm.statistic)}, indent=2))


if __name__ == "__main__":
    main()
