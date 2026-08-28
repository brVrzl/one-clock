#!/usr/bin/env python3
"""Write a compact, evidence-separated overnight research report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "experiments/component_temporal_reuse"
BASELINE = ROOT / "experiments/standard_libero_baselines"
OUT = ROOT / "research/overnight_morning_report_20260828.md"


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def pct(x: float) -> str:
    return f"{x:.1%}"


def native_act_summary() -> tuple[list[tuple[str, int, int]], dict]:
    rows = []
    for path in sorted((BASELINE / "act_final").glob("libero_*_task*/eval10/eval_info.json")):
        data = load(path)
        if not data:
            continue
        successes = data["per_task"][0]["metrics"]["successes"]
        rows.append((path.parts[-3], sum(successes), len(successes)))
    state = load(BASELINE / "overnight_state.json") or {}
    return rows, state


def main() -> None:
    frozen = load(RESEARCH / "final_analysis/analysis.json")
    aggregation = load(RESEARCH / "aggregation/analysis.json")
    act = load(RESEARCH / "act_confirmation/analysis.json")
    standard = load(BASELINE / "results.json") or {}
    standard_table = standard.get("summary_table", {})
    native_rows, act_state = native_act_summary()
    lines = [
        "# ICRA27 overnight LIBERO research status",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()} (UTC). Research branch: `exp/libero-component-temporal-reuse`.",
        "",
        "## Frozen 80-block experiment",
        "",
    ]
    if frozen:
        all_tasks = frozen["aggregates"]["all_tasks"]
        metrics = all_tasks["condition_metrics"]
        lines += ["The frozen cohort is complete with exactly 80 task-condition blocks. Task-macro rates are primary; pooled rates are descriptive.", "", "| condition | task-macro | pooled |", "|---|---:|---:|"]
        for condition in ("fresh", "fo4", "full_old4", "reverse4", "fo8", "full_old8", "reverse8", "fo16", "full_old16", "reverse16"):
            m = metrics[condition]["pooled_descriptive"]
            lines.append(f"| {condition} | {pct(metrics[condition]['task_macro_success_rate'])} | {m['successes']}/{m['episodes']} ({pct(m['success_rate'])}) |")
        lines += ["", "Classification: C (non-monotonic temporal-source utility) plus D (strong task/suite heterogeneity). The age-16 FO-versus-reverse direction is not universal, so A, stable component-specific asymmetry, is not established.", ""]
    else:
        lines += ["Final frozen analysis is not yet present.", ""]

    lines += ["## Standard LIBERO baselines", ""]
    lines += ["These are the untouched native-policy baselines, separate from all research interventions.", "", "| model | spatial | object | goal | long | average |", "|---|---:|---:|---:|---:|---:|"]
    for policy in ("SmolVLA", "ACT"):
        values = standard_table.get(policy, {})
        if values:
            lines.append("| {policy} | {spatial} | {object} | {goal} | {long} | {average} |".format(
                policy=policy,
                **{key: pct(value) if value is not None else "pending" for key, value in values.items()},
            ))
    act_records = [row for row in standard.get("records", []) if row.get("policy") == "ACT" and row.get("successes") is not None]
    if act_records:
        act_successes = sum(row["successes"] for row in act_records)
        act_episodes = sum(row["episodes"] for row in act_records)
        lines += ["", f"Completed native ACT total: {act_successes}/{act_episodes} ({pct(act_successes / act_episodes)}), across {len(act_records)} task checkpoints.", ""]
    else:
        lines += ["", "The completed native ACT summary is not present in the baseline results artifact.", ""]

    lines += ["## Fixed temporal aggregation follow-up", ""]
    if aggregation:
        summary = aggregation["aggregates"]["all_tasks"]["condition_metrics"]
        lines += ["| method | task-macro | pooled |", "|---|---:|---:|"]
        for method in ("fresh", "official_act_m001", "physical_exp_beta003", "cogact_alpha03", "component_arm_fresh_gripper_act"):
            x = summary[method]
            lines.append(f"| {method} | {pct(x['task_macro_success_rate'])} | {x['pooled_successes']}/{x['pooled_episodes']} ({pct(x['pooled_success_rate'])}) |")
        lines += ["", "Official ACT temporal aggregation does not explain the historical-source benefit: it is below Fresh (66/80 vs 72/80). CogACT-style shared aggregation captures most of the descriptive gain (73/80), while component-aware aggregation ties it (73/80) and has no clear added advantage.", ""]
    else:
        lines += ["Aggregation follow-up is still running or awaiting analysis.", ""]

    lines += ["## Independent ACT source-age confirmation", ""]
    if act:
        lines += ["| task | fresh | FO16 | full-old16 | reverse16 | FO16−reverse16 |", "|---|---:|---:|---:|---:|---:|"]
        for row in act["per_task"]:
            m = row["methods"]
            c = row["fo16_vs_reverse16"]
            lines.append(f"| {row['task_key']} | {m['fresh']['success_count']}/10 | {m['fo16']['success_count']}/10 | {m['full_old16']['success_count']}/10 | {m['reverse16']['success_count']}/10 | {c['absolute_success_difference']:+.2f} ({c['candidate_only_success']}/{c['reference_only_success']}) |")
        aggregate = act.get("aggregates", {}).get("all_completed_tasks", {})
        if aggregate:
            lines += ["", f"Aggregate confirmation: Fresh {aggregate['fresh']['pooled_successes']}/40, FO16 {aggregate['fo16']['pooled_successes']}/40, full-old16 {aggregate['full_old16']['pooled_successes']}/40, reverse16 {aggregate['reverse16']['pooled_successes']}/40; FO16 vs reverse16 is {aggregate['fo16_vs_reverse16']['pooled']['candidate_only_success']} candidate-only vs {aggregate['fo16_vs_reverse16']['pooled']['reference_only_success']} reference-only, exact McNemar p={aggregate['fo16_vs_reverse16']['pooled']['exact_mcnemar_two_sided_p']:.5f}.", "", "These are matched-query interventions on independent initial states 10–19, not the native ACT baseline. The native baseline retains its installed `n_action_steps`.", ""]
    else:
        lines += ["ACT confirmation is running or awaiting analysis.", ""]

    if frozen and act:
        smolvla_signs = []
        for row in frozen["per_task"]:
            contrast = next(
                item for item in row["contrasts"]
                if item["age_steps"] == 16 and item["contrast_key"] == "fo_vs_reverse"
            )
            smolvla_signs.append((row["suite"], contrast["absolute_success_difference"]))
        act_signs = [(row["suite"], row["fo16_vs_reverse16"]["absolute_success_difference"]) for row in act["per_task"]]
        s_positive = sum(delta > 0 for _, delta in smolvla_signs)
        a_positive = sum(delta > 0 for _, delta in act_signs)
        lines += [
            "## Cross-policy direction",
            "",
            f"At age 16, SmolVLA has FO > reverse on {s_positive}/{len(smolvla_signs)} frozen tasks; ACT has FO > reverse on {a_positive}/{len(act_signs)} confirmation tasks. These cohorts use different tasks, so this is a suite-level directional comparison rather than a same-task replication. Both policies retain fresh as the strongest aggregate reference in the completed cohorts, while reverse is the most fragile condition; the evidence supports structured component sensitivity, not a universal FO improvement.",
            "",
        ]

    lines += [
        "## Scientific interpretation",
        "",
        "The current main framing is broader conditional temporal-source utility. Large source ages often hurt the arm more than the gripper across the tested SmolVLA and ACT tasks, but this is task-dependent rather than a universal component rule. Fresh is not reliably surpassed. The source-age response is non-monotonic in some tasks and strongly heterogeneous across suites, supporting classification C+D rather than a stable global component-asymmetry claim.",
        "",
        "The official ACT temporal ensemble does not explain the effect. CogACT-style shared aggregation captures most of the descriptive aggregation gain, and component-aware aggregation currently has no clear advantage over CogACT.",
        "",
    ]

    lines += ["## Standard ACT baseline", "", f"Supervisor PID: {act_state.get('supervisor', {}).get('pid', 'unknown')}. It remains independent and untouched.", ""]
    if native_rows:
        by_suite = {}
        for tag, successes, episodes in native_rows:
            suite = tag.rsplit("_task", 1)[0]
            by_suite.setdefault(suite, [0, 0])
            by_suite[suite][0] += successes
            by_suite[suite][1] += episodes
        lines += ["| suite | completed native ACT task results |", "|---|---:|"]
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
            if suite in by_suite:
                s, n = by_suite[suite]
                lines.append(f"| {suite} | {s}/{n} ({pct(s/n)}) |")
        lines.append("")
    if act_state.get("jobs"):
        from collections import Counter
        counts = Counter(x.get("state") for x in act_state["jobs"].values())
        lines.append("Queue state: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) + ".")
        lines.append("")
        running = [
            f"{tag}({job.get('stage')}, GPU {job.get('gpu')}, PID {job.get('pid')})"
            for tag, job in act_state["jobs"].items()
            if job.get("state") == "running"
        ]
        lines.append("Baseline jobs currently running: " + (", ".join(running) if running else "none") + ".")
        lines.append("")

    research_processes = []
    for progress_path in sorted((RESEARCH / "aggregation").glob("*.progress.json")):
        data = load(progress_path) or {}
        if data.get("pid") and Path(f"/proc/{data['pid']}").exists():
            research_processes.append(f"aggregation PID {data['pid']} ({data.get('current_task')}, {data.get('current_method')})")
    for progress_path in sorted((RESEARCH / "act_confirmation/progress").glob("*.json")):
        data = load(progress_path) or {}
        if data.get("pid") and Path(f"/proc/{data['pid']}").exists():
            research_processes.append(f"ACT confirmation PID {data['pid']} ({data.get('current_task')}, {data.get('current_condition')})")
    lines += ["Research processes currently running: " + ("; ".join(research_processes) if research_processes else "none") + ".", ""]

    lines += [
        "## Closest-work and novelty boundary",
        "",
        "Action chunking and temporal ensembling are established in [ACT](https://arxiv.org/abs/2304.13705). [Lazzati et al.](https://arxiv.org/abs/2608.02547) is the closest conceptual collision because it explains action-chunking gains through delayed observation-conditioned predictions and implicit temporal ensembling. [CogACT](https://arxiv.org/abs/2411.19650) is a full-action similarity-weighted comparator. [TAS](https://arxiv.org/abs/2511.04421), [AutoHorizon](https://arxiv.org/abs/2602.21445), [AAC](https://arxiv.org/abs/2604.04161), and [PACE](https://arxiv.org/abs/2606.00537) address selection or execution-horizon adaptation rather than the present same-target component assignment.",
        "",
        "RoboTwin was correctly deferred while LIBERO aggregation and ACT confirmation used the GPUs. The official/current path is [RoboTwin-Platform/RoboTwin](https://github.com/robotwin-Platform/robotwin), documented at [robotwin-platform.github.io](https://robotwin-platform.github.io/doc/); no standard RoboTwin run is reported here.",
        "",
        "## Recommended framing and next experiment",
        "",
        "The strongest defensible framing is conditional, non-monotonic temporal-source utility with substantial task heterogeneity. Do not claim a universal gripper advantage or present an intervention as a native baseline. The single highest-value next experiment is an independently frozen confirmatory cohort testing the fixed aggregation/source-age comparison selected before outcomes, with task-level paired analysis.",
        "",
        "Active research artifacts remain under `experiments/component_temporal_reuse/`; baseline artifacts and supervisor remain under `experiments/standard_libero_baselines/`.",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
