#!/usr/bin/env python3
"""Render the Gate-2B markdown report from summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PHASES = ("early", "middle", "late")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ending-commit", default="recorded in final handoff")
    return parser.parse_args()


def pct(value: float) -> str:
    return f"{value:.3f}"


def selected_global(summary: dict, phase: str) -> dict:
    return summary["phase_global_table"][phase]["selected"]


def selected_group(summary: dict, phase: str) -> dict:
    return summary["phase_group_table"][phase]["selected"]


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    global_rows = [selected_global(summary, phase) for phase in PHASES]
    group_rows = [selected_group(summary, phase) for phase in PHASES]
    global_horizons = [int(row["arm_horizon"]) for row in global_rows]
    arm_horizons = [int(row["arm_horizon"]) for row in group_rows]
    gripper_horizons = [int(row["gripper_horizon"]) for row in group_rows]
    phase_dep = len(set(arm_horizons)) > 1 or len(set(gripper_horizons)) > 1
    combined_global = summary["combined_oracles"]["global"]
    combined_group = summary["combined_oracles"]["group"]
    static_global = summary["static_baselines"]["global_h16"]
    static_group = summary["static_baselines"]["group_arm4_grip16"]
    global_comparison = summary["comparisons"]["phase_oracle_global_vs_static_global"]
    group_comparison = summary["comparisons"]["phase_oracle_group_vs_static_group"]
    def macro_policy_queries(row: dict) -> float:
        return sum(float(task["mean_policy_queries"]) for task in row["task_results"].values()) / len(row["task_results"])
    lines = [
        "# Gate-2B: Phase-conditioned oracle horizon analysis",
        "",
        "**Status: complete offline analysis.** This report evaluates a retrospective "
        "phase-conditioned oracle horizon. It does not implement a scheduler, "
        "dynamic horizon controller, executor behavior, rollout code, or paper changes.",
        "",
        "## 1. Research question",
        "",
        "Does the empirical optimal execution horizon for the same action group "
        "depend on normalized task phase? The result is called a **phase-conditioned "
        "oracle horizon**, not a ground-truth horizon.",
        "",
        "## 2. Provenance and protocol",
        "",
        f"| Item | Value |",
        f"|---|---|",
        f"| Starting commit | `{summary['starting_commit']}` |",
        f"| Ending commit | `{args.ending_commit}` |",
        f"| Checkpoint | `{summary['checkpoint']}` |",
        f"| Dataset | `{summary['dataset']}` |",
        f"| Tasks | {summary['task_coverage']['task_count']} LIBERO Object tasks |",
        f"| Rollout episodes | {summary['task_coverage']['total_episodes']} (task 0: 50; tasks 1–9: 20 each) |",
        f"| Seed rule | `{summary['task_coverage']['seed_rule']}` |",
        "",
        "The frozen ACT policy was evaluated in the existing LIBERO Object runtime "
        "setup. The established state IDs and seeds were preserved. Each candidate "
        "was evaluated as a fresh closed-loop rollout; no training was performed.",
        "",
        "Phase is the deterministic rollout-time proxy "
        "`environment_step / env._max_episode_steps`: early `< 1/3`, middle "
        "`[1/3, 2/3)`, and late `>= 2/3`. A phase horizon is applied only when the "
        "next group commitment expires; no query is forced at a phase boundary, so "
        "an existing commitment may cross a boundary.",
        "",
        "The primary groups are arm=`action[0:6]` and gripper=`action[6]`. "
        "Global candidates use horizons `{1,2,4,8,16}`. Group candidates use all "
        "25 arm/gripper combinations from that set. For each target phase, the "
        "other phases use the fixed controls global `h=16` or group `(4,16)`.",
        "",
        "## 3. Metrics and uncertainty",
        "",
        "Reported metrics are success rate, environment steps, frozen-policy query "
        "count, and query rate (queries/environment steps). Selection uses the "
        "macro mean of per-task success rates, with deterministic query-rate and "
        "horizon tie-breaks. Pooled success is also retained. Per-task success "
        "intervals are Wilson 95% intervals; macro uncertainty uses a 20,000-draw "
        "task bootstrap with seed `20260819`.",
        "",
        "## 4. Phase × global horizon",
        "",
        "| Phase | Selected global h | Macro success | 95% bootstrap CI | Mean env steps | Mean policy queries | Macro query rate |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for phase, row in zip(PHASES, global_rows):
        ci = row["macro_success_rate_bootstrap_ci95"]
        lines.append(f"| {phase} | {int(row['arm_horizon'])} | {pct(row['macro_success_rate'])} | [{pct(ci[0])}, {pct(ci[1])}] | {row['macro_mean_environment_steps']:.1f} | {macro_policy_queries(row):.1f} | {pct(row['macro_query_rate'])} |")
    lines += [
        "",
        "![Phase-conditioned global horizon candidates](phase_conditioned_oracle/phase_global_success.png)",
        "",
        "## 5. Phase × group horizon",
        "",
        "| Phase | Arm h | Gripper h | Macro success | 95% bootstrap CI | Mean env steps | Mean policy queries | Macro query rate |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for phase, row in zip(PHASES, group_rows):
        ci = row["macro_success_rate_bootstrap_ci95"]
        lines.append(f"| {phase} | {int(row['arm_horizon'])} | {int(row['gripper_horizon'])} | {pct(row['macro_success_rate'])} | [{pct(ci[0])}, {pct(ci[1])}] | {row['macro_mean_environment_steps']:.1f} | {macro_policy_queries(row):.1f} | {pct(row['macro_query_rate'])} |")
    lines += [
        "",
        "![Phase-conditioned group horizon heatmaps](phase_conditioned_oracle/phase_group_success_heatmaps.png)",
        "",
        "## 6. Static controls versus combined phase oracle",
        "",
        "The combined oracle uses the selected map for all three phases. This is an "
        "offline selection/evaluation comparison, not evidence that a deployable "
        "dynamic controller improves performance.",
        "",
        "| Configuration | Macro success | Mean env steps | Mean policy queries | Macro query rate |",
        "|---|---:|---:|---:|---:|",
        f"| Static global h=16 | {pct(static_global['macro_success_rate'])} | not logged in baseline | not logged in baseline | {pct(sum(row['policy_query_rate'] for row in static_global['task_results'].values()) / len(static_global['task_results']))} |",
        f"| Phase oracle global | {pct(combined_global['macro_success_rate'])} | {combined_global['macro_mean_environment_steps']:.1f} | {macro_policy_queries(combined_global):.1f} | {pct(combined_global['macro_query_rate'])} |",
        f"| Static group (4,16) | {pct(static_group['macro_success_rate'])} | not logged in baseline | not logged in baseline | {pct(sum(row['policy_query_rate'] for row in static_group['task_results'].values()) / len(static_group['task_results']))} |",
        f"| Phase oracle group | {pct(combined_group['macro_success_rate'])} | {combined_group['macro_mean_environment_steps']:.1f} | {macro_policy_queries(combined_group):.1f} | {pct(combined_group['macro_query_rate'])} |",
        "",
        f"Paired task-bootstrap difference, phase-global minus static-global: "
        f"`{global_comparison['macro_difference_dynamic_minus_static']:+.3f}` "
        f"[{global_comparison['task_bootstrap_ci95'][0]:+.3f}, {global_comparison['task_bootstrap_ci95'][1]:+.3f}].",
        "",
        f"Paired task-bootstrap difference, phase-group minus static-group: "
        f"`{group_comparison['macro_difference_dynamic_minus_static']:+.3f}` "
        f"[{group_comparison['task_bootstrap_ci95'][0]:+.3f}, {group_comparison['task_bootstrap_ci95'][1]:+.3f}].",
        "",
        "![Static controls versus phase-conditioned oracle](phase_conditioned_oracle/phase_oracle_vs_static.png)",
        "",
        "## 7. Answers to the research questions",
        "",
        f"**Does the empirical optimal horizon depend on task phase?** "
        f"At the selected point estimates, **{'yes' if phase_dep else 'no'}**: global selections are "
        f"`{global_horizons}`, group arm selections are `{arm_horizons}`, and group gripper "
        f"selections are `{gripper_horizons}` for early/middle/late.",
        "",
        "**Does the same group select different horizons?** "
        f"{'Yes' if len(set(arm_horizons)) > 1 else 'No'} for arm and "
        f"{'yes' if len(set(gripper_horizons)) > 1 else 'no'} for gripper under the selected point estimates. "
        "This is the requested empirical phase-dependence test; it should be read "
        "with the task-bootstrap intervals and selection limitations below.",
        "",
        "**Is this dynamic horizon improvement?** No claim is made. The combined "
        "phase oracle is retrospective and selected from the same task set used for "
        "evaluation. It only tests whether phase-conditioned horizon motivation is "
        "present.",
        "",
        "## 8. Implications for dynamic horizon design",
        "",
        "If phase-conditioned persistence is retained as a research direction, the "
        "next method study could compare: (1) a training-free estimator derived from "
        "prediction persistence, (2) a self-supervised persistence estimator, and "
        "(3) an uncertainty/confidence-based signal. This audit does not choose among "
        "them and does not implement any scheduler.",
        "",
        "## 9. Limitations",
        "",
        "- Normalized episode time is a rollout proxy, not a semantic task-phase label.",
        "- Oracle maps use phase information retrospectively and are selected/evaluated on the same tasks; held-out selection is still needed.",
        "- No query is forced at phase boundaries; phase exposure depends on episode termination and commitment alignment.",
        "- The task bootstrap treats the ten tasks as the resampling units; it does not remove within-task state correlation.",
        "- The result is specific to this frozen checkpoint, LIBERO Object tasks, action representation, and candidate grid.",
        "- Environment steps and query rates are accounting metrics, not a claim of real-robot efficiency.",
        "",
        "## 10. Artifacts",
        "",
        "- `experiments/phase_conditioned_oracle/phase_oracle.py` — frozen-policy oracle evaluator.",
        "- `experiments/phase_conditioned_oracle/merge_phase_parts.py` and `merge_combined_parts.py` — deterministic partition merges.",
        "- `experiments/phase_conditioned_oracle/summary.json` — full candidate/task aggregates and comparisons.",
        "- `experiments/phase_conditioned_oracle/phase_global_success.png` — global candidate curves.",
        "- `experiments/phase_conditioned_oracle/phase_group_success_heatmaps.png` — group candidate heatmaps.",
        "- `experiments/phase_conditioned_oracle/phase_oracle_vs_static.png` — controls and query-rate comparison.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
