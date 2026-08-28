#!/usr/bin/env python3
"""Write the concise frozen-protocol component-reuse feasibility report."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def pct(successes: int, episodes: int) -> str:
    return f"{successes}/{episodes} ({100.0 * successes / episodes:.1f}%)"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--native-semantics", type=Path, required=True)
    parser.add_argument("--semantic-validation", type=Path, required=True)
    parser.add_argument("--disagreement", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pilot = json.loads(args.pilot.read_text())
    protocol = json.loads(args.protocol.read_text())
    native = json.loads(args.native_semantics.read_text())
    semantic = json.loads(args.semantic_validation.read_text())
    disagreement = json.loads(args.disagreement.read_text()) if args.disagreement and args.disagreement.exists() else None
    conditions = [condition["name"] for condition in protocol["conditions"]]
    task_by_key = {f"{task['suite']}:task{task['task_id']}": task for task in protocol["tasks"]}

    rows = []
    suite_condition = defaultdict(lambda: {name: [0, 0] for name in conditions})
    overall_condition = {name: [0, 0] for name in conditions}
    for task_key, task_result in pilot.get("tasks", {}).items():
        task = task_by_key[task_key]
        condition_results = task_result["conditions"]
        row = {"suite": task["suite"], "task_id": task["task_id"], "task_name": task["task_name"]}
        for name in conditions:
            result = condition_results[name]
            row[name] = (result["success_count"], result["episodes"])
            suite_condition[task["suite"]][name][0] += result["success_count"]
            suite_condition[task["suite"]][name][1] += result["episodes"]
            overall_condition[name][0] += result["success_count"]
            overall_condition[name][1] += result["episodes"]
        rows.append(row)

    lines = [
        "# Component-wise temporal reuse feasibility",
        "",
        "This is a separate research intervention on branch `exp/libero-component-temporal-reuse`; standard baseline artifacts and the baseline supervisor are out of scope.",
        "",
        "## Frozen protocol",
        "",
        f"- Frozen task list: {len(protocol['tasks'])} tasks, selected from native SmolVLA results before intervention outcomes were inspected.",
        f"- Paired initial states: IDs {protocol['environment']['initial_state_ids']} (the LeRobot vector environment uses `n_envs=1`, `episode_index=0`, then increments the init-state ID on reset); seeds {protocol['environment']['seeds']}.",
        f"- Runtime: {protocol['environment']['fps']} Hz, relative control, `init_states=true`, cameras `{protocol['environment']['camera_name']}`, observation mode `{protocol['environment']['obs_type']}`, {protocol['environment']['observation_width']}x{protocol['environment']['observation_height']}.",
        f"- Software: LeRobot {protocol['environment']['software_versions']['lerobot']}, LIBERO {protocol['environment']['software_versions']['libero']}, PyTorch {protocol['environment']['software_versions']['pytorch']}, MuJoCo {protocol['environment']['software_versions']['mujoco']}, Gymnasium {protocol['environment']['software_versions']['gymnasium']}, `MUJOCO_GL={protocol['environment']['software_versions']['mujoco_gl']}`.",
        f"- Frozen SmolVLA checkpoint: `{protocol['checkpoint']}`, revision `{protocol['checkpoint_revision']}`, native `chunk_size={pilot.get('chunk_size')}`, `n_action_steps={pilot.get('n_action_steps')}`.",
        f"- Source ages: {', '.join(f'{age} steps ({age / protocol["environment"]["fps"]:.4f} s)' for age in protocol['source_ages_steps'])}.",
        "- At target step `t`, query `tau=t-d` contributes chunk index `d`; arm is dimensions 0:6 and gripper is dimension 6. Old gripper is therefore a same-target chunk prediction, never a previous-command hold.",
        f"- Full rollouts use {protocol['environment']['episodes_per_task']} paired episodes per task. The cache disagreement summary uses the first ten target states at `t>=16` from episode 0, as frozen in the protocol.",
        "",
        "## Minimal semantic validation",
        "",
        f"- Native `select_action` versus `predict_action_chunk()[0]` with shared flow-matching noise: max absolute error `{native['max_abs_error']:.9g}`, match `{native['fresh_semantics_match']}`.",
        f"- Object task 3 semantic smoke: fresh max error `{semantic['tasks']['libero_object:task3']['conditions']['fresh']['validation']['fresh_semantics_max_abs_error']:.9g}`; full-old8 max error `{semantic['tasks']['libero_object:task3']['conditions']['full_old8']['validation']['full_old_max_abs_error']:.9g}`; FO8 and reverse8 composition checks are exact; retained-gripper versus previous-applied command differed on `{semantic['tasks']['libero_object:task3']['conditions']['fo8']['validation']['old_gripper_vs_previous_applied_non_equal_count']}` logged steps.",
        "",
        "## Per-task success",
        "",
        "| suite | task | fresh | FO4 | FO8 | FO16 | full-old4 | full-old8 | full-old16 | reverse4 | reverse8 | reverse16 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [pct(*row[name]) for name in conditions]
        lines.append(f"| {row['suite']} | {row['task_id']} | " + " | ".join(values) + " |")

    lines.extend(["", "## Suite and overall aggregates", "", "| suite | fresh | FO4 | FO8 | FO16 | full-old4 | full-old8 | full-old16 | reverse4 | reverse8 | reverse16 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        values = [pct(*suite_condition[suite][name]) for name in conditions]
        lines.append(f"| {suite} | " + " | ".join(values) + " |")
    lines.append("| overall | " + " | ".join(pct(*overall_condition[name]) for name in conditions) + " |")

    lines.extend(["", "## Paired comparisons against fresh", "", "Each intervention uses the same frozen initial-state IDs as fresh for that task. `win/tie/loss` counts are intervention relative to fresh on each paired episode.", "", "| condition | wins | ties | losses | intervention rate | fresh-paired rate difference |", "|---|---:|---:|---:|---:|---:|"])
    for name in conditions[1:]:
        wins = ties = losses = 0
        fresh_total = intervention_total = episodes = 0
        for task_key, task_result in pilot.get("tasks", {}).items():
            fresh_successes = task_result["conditions"]["fresh"]["successes"]
            intervention_successes = task_result["conditions"][name]["successes"]
            for fresh_success, intervention_success in zip(fresh_successes, intervention_successes):
                episodes += 1
                fresh_total += int(fresh_success)
                intervention_total += int(intervention_success)
                if intervention_success > fresh_success:
                    wins += 1
                elif intervention_success == fresh_success:
                    ties += 1
                else:
                    losses += 1
        lines.append(f"| {name} | {wins} | {ties} | {losses} | {pct(intervention_total, episodes)} | {(intervention_total - fresh_total) / episodes:+.3f} |")

    lines.extend(["", "## Query budget and source ages", "", "| condition | mean queries/env step | mean arm age (steps) | mean gripper age (steps) |", "|---|---:|---:|---:|"])
    for name in conditions:
        results = [task_result["conditions"][name] for task_result in pilot.get("tasks", {}).values()]
        lines.append(
            f"| {name} | {mean([result['policy_queries_per_environment_step'] for result in results]):.4f} | "
            f"{mean([result['mean_arm_source_age_steps'] for result in results]):.3f} | "
            f"{mean([result['mean_gripper_source_age_steps'] for result in results]):.3f} |"
        )

    lines.extend(["", "## Successful completion steps", "", "| condition | successful episodes | mean completion steps |", "|---|---:|---:|"])
    for name in conditions:
        results = [task_result["conditions"][name] for task_result in pilot.get("tasks", {}).values()]
        completions = [step for result in results for step in result["completion_steps_successful"]]
        lines.append(f"| {name} | {len(completions)} | {mean([float(step) for step in completions]):.2f} |" if completions else f"| {name} | 0 | n/a |")

    lines.extend(["", "## Disagreement analysis", ""])
    if disagreement is None:
        lines.append("Pending: the pilot output is not complete, so the cache analysis has not been run.")
    else:
        lines.append("`D_g(t)` is the L2 distance between the fresh current-query action and the retained same-target action from the age-matched source query. This is exploratory with two paired episodes per task.")
        for group, values in disagreement["aggregate_outcome_association"].items():
            lines.append(f"- {group}: {values['n_outcome_changes']} outcome changes among {values['n_condition_episode_pairs']} paired condition-episode comparisons; correlation with outcome change `{values['correlation_disagreement_with_outcome_change']}`.")
        lines.extend(["", "| task | age (steps) | arm mean L2 | gripper mean L2 |", "|---|---:|---:|---:|"])
        for task_key in sorted(disagreement["task_results"]):
            for age, values in disagreement["task_results"][task_key].items():
                lines.append(f"| {task_key} | {age} | {values['arm']['mean_l2']:.5f} | {values['gripper']['mean_l2']:.5f} |")
        lines.append("- No independent unsafe-event label is available in this rollout log, so the analysis tests association with paired outcome changes, not a causal safety claim.")

    lines.extend(["", "## Conclusion", "", "The conclusion is intentionally left evidence-bounded: compare the paired direction across suites and tasks; a single-task effect is a narrow result, while a repeatable cross-suite directional effect supports the feasibility hypothesis. No adaptive online rule or neural horizon predictor is implemented here.", ""])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(json.dumps({"output": str(args.output), "tasks": len(rows), "overall": {name: pct(*value) for name, value in overall_condition.items()}}, indent=2))


if __name__ == "__main__":
    main()
