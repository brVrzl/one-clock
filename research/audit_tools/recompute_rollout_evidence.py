#!/usr/bin/env python3
"""Independently validate and recompute the project's saved rollout evidence.

Historical experiment directories are read-only inputs.  All generated files are
written below ``research/audit_outputs`` unless ``--output-dir`` is supplied.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments/runs"
SELECTIVE = ROOT / "experiments/groupwise_selective_commitment"
HORIZONS = (1, 2, 4, 8, 16)
COMMON_HORIZONS = (2, 4, 8, 16)
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260821


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "research/audit_outputs",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def config_key(metadata: dict[str, Any]) -> tuple[str, int, int]:
    strategy = str(metadata["strategy"])
    if strategy == "global_fixed":
        horizon = int(metadata["global_horizon"])
        return strategy, horizon, horizon
    horizons = metadata["group_horizons"]
    return strategy, int(horizons["arm"]), int(horizons["gripper"])


def expected_fixed_trace(step: int, horizons: dict[str, int]) -> tuple[bool, list[str]]:
    refreshed = sorted(name for name, horizon in horizons.items() if step % horizon == 0)
    return bool(refreshed), refreshed


def validate_fixed_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    metadata_path = run_dir / "metadata.json"
    episodes_path = run_dir / "episodes.jsonl"
    steps_path = run_dir / "steps.jsonl"
    summary_path = run_dir / "summary.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    if not episodes_path.is_file():
        issues.append("episodes_log_missing")
    if not steps_path.is_file():
        issues.append("steps_log_missing")
    episodes = read_jsonl(episodes_path) if episodes_path.is_file() else []
    steps = read_jsonl(steps_path) if steps_path.is_file() else []

    if tuple(metadata.get("observed_chunk_shape", ())) != (100, 7):
        issues.append("observed_chunk_shape_not_100x7")
    if int(metadata.get("action_dim", -1)) != 7:
        issues.append("action_dim_not_7")
    if len({int(row["init_state_id"]) for row in episodes}) != len(episodes):
        issues.append("duplicate_init_state_ids")

    steps_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in steps:
        steps_by_episode[int(row["episode"])].append(row)
        action = np.asarray(row.get("action", []), dtype=np.float64)
        if action.shape != (7,) or not np.isfinite(action).all():
            issues.append("invalid_action")
            break

    out_of_bounds = np.zeros(7, dtype=np.int64)
    total_actions = 0
    for episode in episodes:
        episode_id = int(episode["episode"])
        records = sorted(steps_by_episode.get(episode_id, []), key=lambda row: int(row["environment_step"]))
        expected_count = int(episode["environment_steps"])
        if len(records) != expected_count:
            issues.append(f"episode_{episode_id}_step_count")
            continue
        if [int(row["environment_step"]) for row in records] != list(range(expected_count)):
            issues.append(f"episode_{episode_id}_noncontiguous_steps")
        if sum(bool(row["policy_query"]) for row in records) != int(episode["policy_queries"]):
            issues.append(f"episode_{episode_id}_query_count")
        if records and bool(records[-1]["is_success"]) != bool(episode["success"]):
            issues.append(f"episode_{episode_id}_success_mismatch")
        for row in records:
            action = np.asarray(row["action"], dtype=np.float64)
            out_of_bounds += np.abs(action) > 1.0
            total_actions += 1
            horizons = {name: int(value) for name, value in row["configured_horizons"].items()}
            query_expected, refreshed_expected = expected_fixed_trace(int(row["environment_step"]), horizons)
            if bool(row["policy_query"]) != query_expected:
                issues.append(f"episode_{episode_id}_query_schedule")
                break
            if sorted(row["refreshed_groups"]) != refreshed_expected:
                issues.append(f"episode_{episode_id}_refresh_schedule")
                break
            for name, horizon in horizons.items():
                position = int(row["source_positions"][name])
                if position != int(row["source_ages"][name]):
                    issues.append(f"episode_{episode_id}_{name}_age_position")
                    break
                if position != int(row["environment_step"]) % horizon:
                    issues.append(f"episode_{episode_id}_{name}_position_cycle")
                    break
                if int(row["remaining_commitments"][name]) != horizon - position:
                    issues.append(f"episode_{episode_id}_{name}_remaining")
                    break

    successes = sum(bool(row["success"]) for row in episodes)
    queries = sum(int(row["policy_queries"]) for row in episodes)
    environment_steps = sum(int(row["environment_steps"]) for row in episodes)
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for field, computed in (
            ("episodes", len(episodes)),
            ("successes", successes),
            ("policy_queries", queries),
            ("environment_steps", environment_steps),
        ):
            if field in summary and int(summary[field]) != computed:
                issues.append(f"summary_{field}_mismatch")

    row = {
        "run_dir": str(run_dir.relative_to(ROOT)),
        "task_id": metadata.get("task_id"),
        "strategy": metadata.get("strategy"),
        "arm_horizon": config_key(metadata)[1],
        "gripper_horizon": config_key(metadata)[2],
        "episodes": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes) if episodes else None,
        "environment_steps": environment_steps,
        "policy_queries": queries,
        "query_rate": queries / environment_steps if environment_steps else None,
        "actions": total_actions,
        "out_of_bounds_by_dimension": out_of_bounds.tolist(),
        "metadata_sha256": sha256(metadata_path),
        "episodes_sha256": sha256(episodes_path) if episodes_path.is_file() else None,
        "steps_sha256": sha256(steps_path) if steps_path.is_file() else None,
        "summary_sha256": sha256(summary_path) if summary_path.is_file() else None,
        "checkpoint": metadata.get("checkpoint"),
        "lerobot_commit": metadata.get("lerobot_commit"),
        "temporal_ensemble_coeff": metadata.get("policy_temporal_ensemble_coeff"),
        "issues": sorted(set(issues)),
    }
    return row, episodes, issues


def load_canonical_static() -> tuple[dict[int, dict[tuple[str, int, int], list[dict[str, Any]]]], list[str]]:
    data: dict[int, dict[tuple[str, int, int], list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    issues: list[str] = []
    roots = (RUNS / "libero_static_grid_20", RUNS / "libero_static_grid_50_extension")
    for root in roots:
        for run_dir in sorted(path for path in root.iterdir() if (path / "metadata.json").is_file()):
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            episodes = read_jsonl(run_dir / "episodes.jsonl")
            data[0][config_key(metadata)].extend(episodes)
    for key, episodes in data[0].items():
        ids = [int(row["init_state_id"]) for row in episodes]
        if sorted(ids) != list(range(50)):
            issues.append(f"task0_{key}_state_coverage")

    cross_root = RUNS / "libero_object_cross_task"
    for task_id in range(1, 10):
        task_root = cross_root / f"task_{task_id}"
        for run_dir in sorted(path for path in task_root.iterdir() if (path / "metadata.json").is_file()):
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            episodes = read_jsonl(run_dir / "episodes.jsonl")
            data[task_id][config_key(metadata)].extend(episodes)
        for key, episodes in data[task_id].items():
            ids = [int(row["init_state_id"]) for row in episodes]
            if ids != list(range(20)):
                issues.append(f"task{task_id}_{key}_state_coverage")
    return data, issues


def successes(episodes: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([bool(row["success"]) for row in sorted(episodes, key=lambda row: int(row["init_state_id"]))], dtype=np.float64)


def task_macro_bootstrap(differences: np.ndarray) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(differences), size=(BOOTSTRAP_DRAWS, len(differences)))
    draws = differences[indices].mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def paired_state_interval(delta: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(BOOTSTRAP_DRAWS, len(delta)))
    draws = delta[indices].mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def static_recomputation(data: dict[int, dict[tuple[str, int, int], list[dict[str, Any]]]]) -> dict[str, Any]:
    task_tables: dict[str, Any] = {}
    flat_rows: list[dict[str, Any]] = []
    for task_id, configs in sorted(data.items()):
        table: dict[str, Any] = {}
        for key, episodes in sorted(configs.items()):
            strategy, arm, gripper = key
            vector = successes(episodes)
            name = f"global_h{arm}" if strategy == "global_fixed" else f"group_arm{arm}_grip{gripper}"
            item = {
                "name": name,
                "strategy": strategy,
                "arm_horizon": arm,
                "gripper_horizon": gripper,
                "episodes": len(vector),
                "successes": int(vector.sum()),
                "success_rate": float(vector.mean()),
                "query_rate": float(sum(int(row["policy_queries"]) for row in episodes) / sum(int(row["environment_steps"]) for row in episodes)),
                "success_vector": vector.astype(int).tolist(),
            }
            table[name] = item
            flat_rows.append({"task_id": task_id, **{k: v for k, v in item.items() if k != "success_vector"}})
        task_tables[str(task_id)] = table

    # Independently justify only those global aliases whose diagonal executor is
    # mathematically identical.  Raw global h=4 is retained when available.
    def get_global(task_id: int, horizon: int) -> dict[str, Any]:
        table = task_tables[str(task_id)]
        raw = table.get(f"global_h{horizon}")
        if raw is not None:
            return raw
        return table[f"group_arm{horizon}_grip{horizon}"]

    global_curves: dict[str, Any] = {}
    for task_id in range(10):
        horizons = HORIZONS if task_id == 0 else COMMON_HORIZONS
        global_curves[str(task_id)] = [
            {
                "horizon": horizon,
                "success_rate": get_global(task_id, horizon)["success_rate"],
                "successes": get_global(task_id, horizon)["successes"],
                "episodes": get_global(task_id, horizon)["episodes"],
                "query_rate": get_global(task_id, horizon)["query_rate"],
            }
            for horizon in horizons
        ]

    common_group_names = set.intersection(
        *[
            {name for name, item in task_tables[str(task_id)].items() if item["strategy"] == "groupwise_fixed"}
            for task_id in range(10)
        ]
    )
    common_group_names.add("group_arm4_grip4")
    universal_global = []
    for horizon in COMMON_HORIZONS:
        rates = np.asarray([get_global(task_id, horizon)["success_rate"] for task_id in range(10)])
        universal_global.append({"horizon": horizon, "macro_success_rate": float(rates.mean()), "task_rates": rates.tolist()})
    universal_groups = []
    for name in sorted(common_group_names):
        arm = int(name.split("arm", 1)[1].split("_", 1)[0])
        gripper = int(name.rsplit("grip", 1)[1])
        rates = []
        for task_id in range(10):
            if name == "group_arm4_grip4" and name not in task_tables[str(task_id)]:
                rates.append(get_global(task_id, 4)["success_rate"])
            else:
                rates.append(task_tables[str(task_id)][name]["success_rate"])
        universal_groups.append({"arm_horizon": arm, "gripper_horizon": gripper, "macro_success_rate": float(np.mean(rates)), "task_rates": rates})

    best_global = sorted(universal_global, key=lambda row: (-row["macro_success_rate"], row["horizon"]))[0]
    best_group = sorted(universal_groups, key=lambda row: (-row["macro_success_rate"], row["arm_horizon"], row["gripper_horizon"]))[0]
    task_differences = np.asarray(best_group["task_rates"]) - np.asarray(best_global["task_rates"])

    leave_one_task_out = []
    for held_out in range(10):
        train_ids = [task_id for task_id in range(10) if task_id != held_out]
        selected_global = sorted(
            universal_global,
            key=lambda row: (-float(np.mean(np.asarray(row["task_rates"])[train_ids])), row["horizon"]),
        )[0]
        selected_group = sorted(
            universal_groups,
            key=lambda row: (
                -float(np.mean(np.asarray(row["task_rates"])[train_ids])),
                row["arm_horizon"],
                row["gripper_horizon"],
            ),
        )[0]
        leave_one_task_out.append(
            {
                "held_out_task": held_out,
                "selected_global_horizon": selected_global["horizon"],
                "selected_group": [selected_group["arm_horizon"], selected_group["gripper_horizon"]],
                "held_out_global_success": selected_global["task_rates"][held_out],
                "held_out_group_success": selected_group["task_rates"][held_out],
                "held_out_difference": selected_group["task_rates"][held_out] - selected_global["task_rates"][held_out],
            }
        )

    task0_global4 = get_global(0, 4)
    task0_group416 = task_tables["0"]["group_arm4_grip16"]
    paired_delta = np.asarray(task0_group416["success_vector"]) - np.asarray(task0_global4["success_vector"])
    discordant = {
        "global_only": int(np.sum(paired_delta == -1)),
        "group_only": int(np.sum(paired_delta == 1)),
        "both_equal": int(np.sum(paired_delta == 0)),
    }

    return {
        "task_tables": task_tables,
        "flat_rows": flat_rows,
        "global_support_curves": global_curves,
        "common_configuration_set": {
            "universal_global": universal_global,
            "universal_group": universal_groups,
            "selected_global": best_global,
            "selected_group": best_group,
            "selected_group_minus_global_task_differences": task_differences.tolist(),
            "selected_group_minus_global_macro_difference": float(task_differences.mean()),
            "task_cluster_bootstrap_ci95": task_macro_bootstrap(task_differences),
            "leave_one_task_out": leave_one_task_out,
        },
        "task0_budget_control_global4_vs_group416": {
            "global_success_rate": task0_global4["success_rate"],
            "group_success_rate": task0_group416["success_rate"],
            "difference": float(paired_delta.mean()),
            "paired_state_bootstrap_ci95": paired_state_interval(paired_delta, BOOTSTRAP_SEED + 1),
            "discordant": discordant,
            "global_query_rate": task0_global4["query_rate"],
            "group_query_rate": task0_group416["query_rate"],
        },
        "seed_limitation": "Each init state has exactly one deterministic seed (1000 + state ID); seed and initial state are confounded, so stochastic seed stability is not estimable.",
    }


def validate_selective_commitment() -> dict[str, Any]:
    manifest = json.loads((SELECTIVE / "rollout_log_manifest.json").read_text(encoding="utf-8"))["logs"]
    issues: list[str] = []
    rows: list[dict[str, Any]] = []
    manifest_hash_matches = 0
    for entry in manifest:
        episodes_path = Path(entry["episodes_log"])
        steps_path = Path(entry["steps_log"])
        if not episodes_path.is_file() or not steps_path.is_file():
            issues.append(f"missing_logs_task{entry['task_id']}_q{entry['q']}_{entry['method']}")
            continue
        if sha256(episodes_path) == entry["episodes_log_sha256"] and sha256(steps_path) == entry["steps_log_sha256"]:
            manifest_hash_matches += 1
        else:
            issues.append(f"hash_mismatch_task{entry['task_id']}_q{entry['q']}_{entry['method']}")
        episodes = read_jsonl(episodes_path)
        steps = read_jsonl(steps_path)
        if len(episodes) != 20:
            issues.append(f"episode_count_task{entry['task_id']}_q{entry['q']}_{entry['method']}")
        steps_by_state: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for step in steps:
            steps_by_state[int(step["init_state_id"])].append(step)
        for episode in episodes:
            state_id = int(episode["init_state_id"])
            trace = steps_by_state[state_id]
            q = int(episode["query_cadence"])
            if len(trace) != int(episode["environment_steps"]):
                issues.append("selective_step_count")
            expected_queries = list(range(0, len(trace), q))
            actual_queries = [int(step["global_timestep"]) for step in trace if step["query_occurred"]]
            if actual_queries != expected_queries or actual_queries != episode["query_steps"]:
                issues.append("selective_query_schedule")
            if trace and bool(trace[-1]["is_success"]) != bool(episode["success"]):
                issues.append("selective_success_mismatch")
        rows.extend(episodes)

    pooled: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for q in (4, 8, 16):
        pooled[str(q)] = {}
        by_method: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
        for method in ("global_replace", "selective_commit"):
            subset = [row for row in rows if int(row["query_cadence"]) == q and row["method"] == method]
            pooled[str(q)][method] = {
                "episodes": len(subset),
                "successes": sum(bool(row["success"]) for row in subset),
                "success_rate": float(np.mean([bool(row["success"]) for row in subset])),
                "environment_steps": sum(int(row["environment_steps"]) for row in subset),
                "policy_queries": sum(int(row["policy_queries"]) for row in subset),
                "source_exhaustion_steps": sum(int(row["source_exhaustion_steps"]) for row in subset),
            }
            by_method[method] = {(int(row["task_id"]), int(row["init_state_id"])): row for row in subset}
        keys = sorted(set(by_method["global_replace"]) & set(by_method["selective_commit"]))
        delta = np.asarray([
            bool(by_method["selective_commit"][key]["success"]) - bool(by_method["global_replace"][key]["success"])
            for key in keys
        ], dtype=np.float64)
        task_delta = np.asarray([
            delta[[key[0] == task_id for key in keys]].mean() for task_id in range(10)
        ])
        paired[str(q)] = {
            "pairs": len(keys),
            "selective_minus_global": float(delta.mean()),
            "paired_episode_bootstrap_ci95": paired_state_interval(delta, BOOTSTRAP_SEED + q),
            "task_cluster_bootstrap_ci95": task_macro_bootstrap(task_delta),
            "task_differences": task_delta.tolist(),
        }
    return {
        "manifest_entries": len(manifest),
        "manifest_hash_matches": manifest_hash_matches,
        "episodes": len(rows),
        "issues": sorted(set(issues)),
        "pooled": pooled,
        "paired": paired,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, Any]] = []
    validation_issues: list[str] = []
    for metadata_path in sorted(RUNS.rglob("metadata.json")):
        run_dir = metadata_path.parent
        row, _, issues = validate_fixed_run(run_dir)
        inventory.append(row)
        validation_issues.extend(f"{row['run_dir']}:{issue}" for issue in issues)

    static_data, coverage_issues = load_canonical_static()
    recomputed = static_recomputation(static_data)
    selective = validate_selective_commitment()
    total_actions = sum(int(row["actions"]) for row in inventory)
    out_of_bounds = np.sum(
        [np.asarray(row["out_of_bounds_by_dimension"], dtype=np.int64) for row in inventory],
        axis=0,
    )

    write_csv(args.output_dir / "rollout_artifact_inventory.csv", inventory)
    write_csv(args.output_dir / "gate0_static_task_configurations.csv", recomputed.pop("flat_rows"))
    json_dump(
        args.output_dir / "rollout_evidence_recomputed.json",
        {
            "audit_script": str(Path(__file__).relative_to(ROOT)),
            "historical_run_directories": len(inventory),
            "historical_validation_issues": validation_issues,
            "canonical_coverage_issues": coverage_issues,
            "historical_action_range_audit": {
                "saved_actions": total_actions,
                "count_abs_greater_than_one_by_dimension": out_of_bounds.tolist(),
                "fraction_abs_greater_than_one_by_dimension": (out_of_bounds / total_actions).tolist(),
            },
            "static_gate": recomputed,
            "selective_commitment": selective,
        },
    )
    print(json.dumps({
        "run_directories": len(inventory),
        "historical_validation_issues": len(validation_issues),
        "coverage_issues": coverage_issues,
        "selective_issues": selective["issues"],
        "output": str(args.output_dir / "rollout_evidence_recomputed.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
