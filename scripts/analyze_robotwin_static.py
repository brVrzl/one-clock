#!/usr/bin/env python3
"""Summarize paired RoboTwin static-horizon runs and draw diagnostics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


CONFIGS = [
    ("G2", "global_fixed", 2, 2),
    ("G4", "global_fixed", 4, 4),
    ("G8", "global_fixed", 8, 8),
    ("G16", "global_fixed", 16, 16),
    ("A2G8", "groupwise_fixed", 2, 8),
    ("A2G16", "groupwise_fixed", 2, 16),
    ("A4G16", "groupwise_fixed", 4, 16),
    ("A8G16", "groupwise_fixed", 8, 16),
    ("A8G2", "groupwise_fixed", 8, 2),
    ("A16G2", "groupwise_fixed", 16, 2),
    ("A16G4", "groupwise_fixed", 16, 4),
    ("A16G8", "groupwise_fixed", 16, 8),
]
SYMMETRIC = [("A2G8", "A8G2"), ("A2G16", "A16G2"), ("A4G16", "A16G4"), ("A8G16", "A16G8")]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_from_seed_dir(seed_dir: Path, seed: int) -> tuple[dict[str, Any], str]:
    """Read one isolated seed, retaining timeout/error cells explicitly."""
    summary_path = seed_dir / "summary.json"
    episode_path = seed_dir / "episodes.jsonl"
    if summary_path.exists() and episode_path.exists():
        rows = [json.loads(line) for line in episode_path.read_text(encoding="utf-8").splitlines()]
        if rows:
            row = dict(rows[0])
            row["seed"] = seed
            row["status"] = "complete"
            return row, "complete"
    step_path = seed_dir / "steps.jsonl"
    partial_steps = step_path.read_text(encoding="utf-8").splitlines() if step_path.exists() else []
    queries = 0
    for line in partial_steps:
        try:
            queries += int(json.loads(line).get("policy_query", 0))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return {
        "episode": 0,
        "seed": seed,
        "success": False,
        "environment_steps": len(partial_steps),
        "policy_queries": queries,
        "policy_query_rate": queries / len(partial_steps) if partial_steps else 0.0,
        "arm_horizon": None,
        "gripper_horizon": None,
        "configured_horizons": {},
        "mean_source_age_by_group": {},
        "status": "timeout_or_error",
    }, "timeout_or_error"


def load_run(root: Path, label: str, strategy: str, arm: int, gripper: int) -> dict[str, Any]:
    run = root / label
    direct_summary = run / "summary.json"
    if direct_summary.exists():
        metadata = read_json(run / "metadata.json")
        summary = read_json(direct_summary)
        episodes = [json.loads(line) for line in (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
        statuses = ["complete"] * len(episodes)
    else:
        seed_dirs = sorted(
            (
                path
                for path in run.glob("seed_*")
                if path.is_dir() and re.fullmatch(r"seed_\d+", path.name)
            ),
            key=lambda path: int(path.name.split("_", 1)[1]),
        )
        metadata = {}
        episodes = []
        statuses = []
        for seed_dir in seed_dirs:
            if not metadata and (seed_dir / "metadata.json").exists():
                metadata = read_json(seed_dir / "metadata.json")
            seed = int(seed_dir.name.split("_", 1)[1].split(".", 1)[0])
            row, status = _episode_from_seed_dir(seed_dir, seed)
            episodes.append(row)
            statuses.append(status)
        expected_seeds = list(range(20))
        seen = {int(row["seed"]) for row in episodes}
        for seed in expected_seeds:
            if seed not in seen:
                episodes.append({"seed": seed, "success": False, "status": "missing"})
                statuses.append("missing")
        episodes.sort(key=lambda row: int(row["seed"]))
        successes = sum(int(bool(row.get("success", False))) for row in episodes)
        steps = sum(int(row.get("environment_steps", 0)) for row in episodes)
        queries = sum(int(row.get("policy_queries", 0)) for row in episodes)
        age_totals: dict[str, float] = {}
        age_weights: dict[str, int] = {}
        for row in episodes:
            weight = int(row.get("environment_steps", 0))
            for group, value in row.get("mean_source_age_by_group", {}).items():
                age_totals[group] = age_totals.get(group, 0.0) + float(value) * weight
                age_weights[group] = age_weights.get(group, 0) + weight
        summary = {
            "episodes": len(episodes),
            "successes": successes,
            "success_rate": successes / len(episodes) if episodes else 0.0,
            "environment_steps": steps,
            "policy_queries": queries,
            "policy_queries_per_episode": queries / len(episodes) if episodes else 0.0,
            "policy_query_rate": queries / steps if steps else 0.0,
            "mean_source_age_by_group": {
                group: age_totals[group] / age_weights[group]
                for group in age_totals if age_weights[group]
            },
        }
        statuses = [str(row.get("status", "missing")) for row in episodes]
        metadata = dict(metadata)
        metadata["isolated_seed_processes"] = True
        metadata["evaluation_seeds"] = expected_seeds
        metadata["complete_seed_count"] = statuses.count("complete")
        metadata["anomaly_seed_count"] = len(statuses) - statuses.count("complete")
    if strategy == "global_fixed":
        metadata = dict(metadata)
        metadata["group_horizons"] = {
            "left_arm": arm,
            "left_gripper": gripper,
            "right_arm": arm,
            "right_gripper": gripper,
        }
    return {
        "label": label,
        "strategy": strategy,
        "arm_horizon": arm,
        "gripper_horizon": gripper,
        "metadata": metadata,
        "summary": summary,
        "episodes": episodes,
        "success_vector": [int(bool(row["success"])) for row in episodes],
        "statuses": statuses,
        "complete": all(status == "complete" for status in statuses),
    }


def paired_delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    valid = [
        index
        for index, (a_status, b_status) in enumerate(zip(a["statuses"], b["statuses"]))
        if a_status == "complete" and b_status == "complete"
    ]
    av = [a["success_vector"][index] for index in valid]
    bv = [b["success_vector"][index] for index in valid]
    return {
        "a": a["label"],
        "b": b["label"],
        "a_success": sum(av),
        "b_success": sum(bv),
        "a_minus_b": sum(av) - sum(bv),
        "a_wins": sum(x > y for x, y in zip(av, bv)),
        "b_wins": sum(y > x for x, y in zip(av, bv)),
        "ties": sum(x == y for x, y in zip(av, bv)),
        "paired_complete_seeds": len(valid),
        "paired_incomplete_seeds": len(a["success_vector"]) - len(valid),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()

    runs = {label: load_run(args.runs, label, strategy, arm, gripper) for label, strategy, arm, gripper in CONFIGS}
    seeds = runs[CONFIGS[0][0]]["metadata"]["evaluation_seeds"]
    symmetric = [paired_delta(runs[a], runs[b]) for a, b in SYMMETRIC]
    global_rows = [runs[label] for label, strategy, _, _ in CONFIGS if strategy == "global_fixed"]
    best_global = max(row["summary"]["success_rate"] for row in global_rows)
    best_global_labels = [row["label"] for row in global_rows if row["summary"]["success_rate"] == best_global]
    offdiag = [runs[label] for label, strategy, _, _ in CONFIGS if strategy == "groupwise_fixed"]
    best_offdiag = max(row["summary"]["success_rate"] for row in offdiag)
    query_rate_by_label = {label: runs[label]["summary"]["policy_query_rate"] for label, *_ in CONFIGS}
    # A conservative empirical Pareto test: no other evaluated point has both
    # at least as many successes and no greater query rate, with one strict.
    pareto = []
    for label, *_ in CONFIGS:
        row = runs[label]
        dominated = any(
            other != label
            and runs[other]["summary"]["success_rate"] >= row["summary"]["success_rate"]
            and query_rate_by_label[other] <= query_rate_by_label[label]
            and (
                runs[other]["summary"]["success_rate"] > row["summary"]["success_rate"]
                or query_rate_by_label[other] < query_rate_by_label[label]
            )
            for other, *_ in CONFIGS
        )
        if not dominated:
            pareto.append(label)
    complete_all = all(row["complete"] for row in runs.values())
    if not complete_all:
        pareto = []

    result: dict[str, Any] = {
        "task": "place_can_basket",
        "task_config": "demo_clean",
        "evaluation_seeds": seeds,
        "configurations": [
            {
                "label": label,
                "strategy": strategy,
                "arm_horizon": arm,
                "gripper_horizon": gripper,
                "metadata": runs[label]["metadata"],
                "summary": runs[label]["summary"],
                "episodes": runs[label]["episodes"],
                "success_vector": runs[label]["success_vector"],
                "statuses": runs[label]["statuses"],
                "complete": runs[label]["complete"],
            }
            for label, strategy, arm, gripper in CONFIGS
        ],
        "symmetric_comparisons": symmetric,
        "best_global": {"labels": best_global_labels, "success_rate": best_global},
        "best_offdiagonal_success_rate": best_offdiag,
        "empirical_pareto_labels": pareto,
        "classification": (
            "BLOCKED"
            if not complete_all
            else ("A" if len({row["summary"]["success_rate"] for row in runs.values()}) > 1 else "C")
        ),
        "complete": complete_all,
        "anomalies": {
            label: [
                int(row.get("seed", -1))
                for row, status in zip(runs[label]["episodes"], runs[label]["statuses"])
                if status != "complete"
            ]
            for label, *_ in CONFIGS
            if not runs[label]["complete"]
        },
        "execution_blocker": {
            "reason": "Pinned headless SAPIEN/MPLIB qpos rollout stalled before terminal completion for some fixed-horizon cells.",
            "timeout_seconds": 120,
            "observed_timeout_attempts": sum(
                1
                for line in (args.runs / "isolation_status.tsv").read_text(encoding="utf-8").splitlines()
                if line and line.split("\t")[2] == "124"
            )
            if (args.runs / "isolation_status.tsv").exists()
            else None,
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    def pct(value: float) -> str:
        return f"{100 * value:.1f}%"

    lines = [
        "# RoboTwin static horizon results",
        "",
        "Task: `place_can_basket`, configuration: `demo_clean`. The 12 rows use the same ordered "
        f"{len(seeds)} evaluation seeds: `{','.join(map(str, seeds))}`.",
        "",
        (
            "**STATUS: complete.** All 240 seed/config cells produced terminal episode records."
            if result["complete"]
            else "**STATUS: blocked/incomplete.** Timeout or error cells are retained as anomalies; no scientific success claim is made from an incomplete sweep."
        ),
        "",
        f"Pinned RoboTwin SHA: `{runs['G2']['metadata'].get('robotwin_commit', 'unknown')}`; XPolicyLab SHA: `{runs['G2']['metadata'].get('xpolicylab_commit', 'unknown')}`; ACT chunk size: `{runs['G2']['metadata'].get('chunk_size', 'unknown')}`.",
        "",
        "## Global horizons",
        "",
        "| Configuration | Arm | Gripper | Success | Queries/episode | Query rate | Mean source age |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in global_rows:
        s = row["summary"]
        ages = ", ".join(f"{k}={v:.2f}" for k, v in sorted(s.get("mean_source_age_by_group", {}).items()))
        lines.append(f"| {row['label']} | {row['arm_horizon']} | {row['gripper_horizon']} | {pct(s['success_rate'])} | {s['policy_queries_per_episode']:.1f} | {s['policy_query_rate']:.4f} | {ages} |")
    lines += ["", "## Group-specific configurations", "", "| Configuration | Arm | Gripper | Success | Queries/episode | Query rate | Mean source age |", "|---|---:|---:|---:|---:|---:|---|"]
    for row in offdiag:
        s = row["summary"]
        ages = ", ".join(f"{k}={v:.2f}" for k, v in sorted(s.get("mean_source_age_by_group", {}).items()))
        lines.append(f"| {row['label']} | {row['arm_horizon']} | {row['gripper_horizon']} | {pct(s['success_rate'])} | {s['policy_queries_per_episode']:.1f} | {s['policy_query_rate']:.4f} | {ages} |")
    lines += ["", "## Arm/gripper success matrix", "", "Rows are arm horizons and columns are gripper horizons.", "", "| Arm \\ Gripper | 2 | 4 | 8 | 16 |", "|---|---:|---:|---:|---:|"]
    for arm in (2, 4, 8, 16):
        cells = []
        for gripper in (2, 4, 8, 16):
            label = next((name for name, _, a, g in CONFIGS if a == arm and g == gripper), None)
            cells.append("—" if label is None else pct(runs[label]["summary"]["success_rate"]))
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += ["", "## Query-rate matrix", "", "| Arm \\ Gripper | 2 | 4 | 8 | 16 |", "|---|---:|---:|---:|---:|"]
    for arm in (2, 4, 8, 16):
        cells = []
        for gripper in (2, 4, 8, 16):
            label = next((name for name, _, a, g in CONFIGS if a == arm and g == gripper), None)
            cells.append("—" if label is None else f"{runs[label]['summary']['policy_query_rate']:.4f}")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired and symmetric comparisons", "", "Counts use only seeds with terminal records for both configurations; incomplete cells are not assigned a success.", "", "| Pair | First wins | Second wins | Ties | Valid paired seeds | Incomplete | Success difference |", "|---|---:|---:|---:|---:|---:|---:|"]
    for pair in symmetric:
        lines.append(f"| `{pair['a']} vs {pair['b']}` | {pair['a_wins']} | {pair['b_wins']} | {pair['ties']} | {pair['paired_complete_seeds']} | {pair['paired_incomplete_seeds']} | {pair['a_minus_b']} |")
    lines += [
        "",
        "## Interpretation",
        "",
        f"- Best global configuration(s): `{', '.join(best_global_labels)}` at {pct(best_global)}.",
        f"- Best off-diagonal success rate: {pct(best_offdiag)}; empirical success/query Pareto labels: `{', '.join(pareto)}`.",
        "- Query-budget matching is reported using the measured policy-query rate, not configured horizon alone.",
        "- Classification is deliberately mechanical: A means success varies across complete evaluated configurations; C means no variation; BLOCKED means at least one required seed/configuration did not terminate.",
        "",
        f"- Anomalies: `{json.dumps(result['anomalies'], sort_keys=True)}`.",
        f"- Execution blocker: {result['execution_blocker']['reason']} Timeout attempts recorded: `{result['execution_blocker']['observed_timeout_attempts']}`.",
        "## Reproducibility",
        "",
        "The JSON artifact contains per-episode seeds, success, environment steps, policy queries, configured horizons, and source-age summaries. Raw step traces remain under `experiments/runs/` and are ignored by git.",
    ]
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib.pyplot as plt
    import numpy as np

    labels = [label for label, *_ in CONFIGS]
    success = [runs[label]["summary"]["success_rate"] for label in labels]
    rates = [query_rate_by_label[label] for label in labels]
    colors = ["tab:blue" if runs[label]["strategy"] == "global_fixed" else "tab:orange" for label in labels]
    horizons = [2, 4, 8, 16]
    success_grid = np.full((len(horizons), len(horizons)), np.nan)
    query_grid = np.full((len(horizons), len(horizons)), np.nan)
    for label, _, arm, gripper in CONFIGS:
        i, j = horizons.index(arm), horizons.index(gripper)
        success_grid[i, j] = runs[label]["summary"]["success_rate"]
        query_grid[i, j] = runs[label]["summary"]["policy_query_rate"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.0), constrained_layout=True)
    axes = axes.ravel()
    axes[0].bar(labels, success, color=colors)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Success rate")
    axes[0].tick_params(axis="x", rotation=55)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, rates, color=colors)
    axes[1].set_ylabel("Policy query rate")
    axes[1].set_xlabel("Blue: global, orange: type-tied arm/gripper")
    axes[1].tick_params(axis="x", rotation=55)
    axes[1].grid(axis="y", alpha=0.25)
    im_success = axes[2].imshow(success_grid, vmin=0, vmax=1, cmap="viridis")
    axes[2].set_title("Success heatmap")
    axes[2].set_xlabel("Gripper horizon")
    axes[2].set_ylabel("Arm horizon")
    axes[2].set_xticks(range(4), horizons)
    axes[2].set_yticks(range(4), horizons)
    fig.colorbar(im_success, ax=axes[2], fraction=0.046)
    for i in range(4):
        for j in range(4):
            if not np.isnan(success_grid[i, j]):
                axes[2].text(j, i, f"{success_grid[i, j]:.2f}", ha="center", va="center", color="white")
    im_query = axes[3].imshow(query_grid, cmap="magma")
    axes[3].set_title("Query-rate heatmap")
    axes[3].set_xlabel("Gripper horizon")
    axes[3].set_ylabel("Arm horizon")
    axes[3].set_xticks(range(4), horizons)
    axes[3].set_yticks(range(4), horizons)
    fig.colorbar(im_query, ax=axes[3], fraction=0.046)
    for i in range(4):
        for j in range(4):
            if not np.isnan(query_grid[i, j]):
                axes[3].text(j, i, f"{query_grid[i, j]:.3f}", ha="center", va="center", color="white")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
