#!/usr/bin/env python3
"""Render the frozen offline audit and the secondary outcome association."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
VALID_DECISIONS = {
    "GROUP_TEMPORAL_HETEROGENEITY_STRONG",
    "GROUP_TEMPORAL_HETEROGENEITY_PARTIAL",
    "GROUP_TEMPORAL_HETEROGENEITY_NULL",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        return str(value)
    return f"{float(value):.{digits}f}"


def vector(rows: list[dict[str, Any]], key: str) -> str:
    return " / ".join(fmt(row[key]) for row in rows)


def task_label(task_key: str) -> str:
    return task_key.replace("libero_", "").replace(":task", "-")


def task_profile_rows(outcome_blind: dict[str, Any], group: str) -> list[dict[str, Any]]:
    rows = []
    delays = [int(value) for value in outcome_blind["delays"]]
    for task_key in outcome_blind["task_split"]["task_order"]:
        profile = outcome_blind["profiles"][task_key]
        group_rows = profile["groups"][group]["delays"]
        summary = profile["groups"][group]["summary"]
        rows.append(
            {
                "task_key": task_key,
                "split": profile["split"],
                "support": "/".join(str(row["valid_target_count"]) for row in group_rows),
                "utility": vector(group_rows, "utility"),
                "revision": vector(group_rows, "revision_or_abs_diff"),
                "translation": vector(group_rows, "translation_revision"),
                "rotation": vector(group_rows, "rotation_revision"),
                "sign": vector(group_rows, "gripper_sign_disagreement"),
                "preferred": summary["preferred_delay_steps"],
                "positive_preferred": summary["best_positive_delay_steps"],
                "positive_utility": summary["best_positive_delay_utility"],
                "slope": summary["delay_sensitivity_slope_per_step"],
                "auc": summary["utility_auc_normalized_0_to_32"],
                "threshold_delay": summary["first_delay_below_development_threshold"],
                "shape": summary["profile_shape"],
                "delays": delays,
            }
        )
    return rows


def reliability_context() -> str:
    rows = [
        (1, 1.000, 1.000, 0.931, 0.931),
        (4, 0.996, 0.995, 0.893, 0.818),
        (8, 0.980, 0.971, 0.871, 0.722),
        (16, 0.945, 0.877, 0.788, 0.581),
        (32, 0.933, 0.715, 0.609, 0.291),
        (64, 0.910, 0.395, 0.687, 0.111),
    ]
    lines = [
        "| offset | arm pointwise | arm prefix survival | gripper pointwise | gripper prefix survival |",
        "|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {offset} | {point:.3f} | {survival:.3f} | {gpoint:.3f} | {gsurvival:.3f} |"
        for offset, point, survival, gpoint, gsurvival in rows
    )
    return "\n".join(lines)


def render(protocol: dict[str, Any], outcome_blind: dict[str, Any], relation: dict[str, Any], decision: str) -> str:
    delays = [int(value) for value in outcome_blind["delays"]]
    dev = set(protocol["task_split"]["development"])
    held_out = set(protocol["task_split"]["held_out_descriptive"])
    ranking = outcome_blind["task_macro_H_temp_ranking"]
    arm_rows = task_profile_rows(outcome_blind, "arm")
    grip_rows = task_profile_rows(outcome_blind, "gripper")
    by_task_arm = {row["task_key"]: row for row in arm_rows}
    by_task_grip = {row["task_key"]: row for row in grip_rows}
    counts = outcome_blind["counts"]
    frozen = load_json(ROOT / "h_temp_frozen.json")
    arm_table = [
        "| task | split | valid target counts by d=" + ",".join(str(d) for d in delays) + " | U_arm by d | translation revision by d | rotation revision by d | best positive d (U) | slope | AUC | threshold crossing |",
        "|---|---|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for row in arm_rows:
        profile = outcome_blind["profiles"][row["task_key"]]["groups"]["arm"]["delays"]
        arm_table.append(
            f"| {task_label(row['task_key'])} | {row['split']} | {row['support']} | {row['utility']} | {row['translation']} | {row['rotation']} | {row['positive_preferred']} ({fmt(row['positive_utility'])}) | {fmt(row['slope'], 4)} | {fmt(row['auc'])} | {fmt(row['threshold_delay'], 0)} |"
        )

    grip_table = [
        "| task | split | valid target counts by d=" + ",".join(str(d) for d in delays) + " | U_grip by d | abs difference by d | sign disagreement by d | best positive d (U) | slope | AUC | threshold crossing |",
        "|---|---|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for row in grip_rows:
        grip_table.append(
            f"| {task_label(row['task_key'])} | {row['split']} | {row['support']} | {row['utility']} | {row['revision']} | {row['sign']} | {row['positive_preferred']} ({fmt(row['positive_utility'])}) | {fmt(row['slope'], 4)} | {fmt(row['auc'])} | {fmt(row['threshold_delay'], 0)} |"
        )

    non_markov_lines = []
    for group, label in (("arm", "arm"), ("gripper", "gripper")):
        non_markov_lines.append(f"### {label}")
        non_markov_lines.append("")
        non_markov_lines.append("| d | task utility better than Fresh | candidate rows better than Fresh | exact candidate matches |")
        non_markov_lines.append("|---:|---:|---:|---:|")
        group_evidence = outcome_blind["non_markovian_evidence"]["primary_revision_utility"][group]
        metrics = group_evidence["delays"]
        for delay, values in metrics.items():
            non_markov_lines.append(
                f"| {delay} | {values['task_level_better_count']}/{group_evidence['task_count']} | {values['candidate_rows_better_than_fresh']}/{values['candidate_rows']} ({fmt(values['candidate_rows_better_fraction'])}) | {values['candidate_rows_exactly_matching_fresh']}/{values['candidate_rows']} ({fmt(values['candidate_rows_match_fraction'])}) |"
            )
        non_markov_lines.append("")
        non_markov_lines.append(
            "The revision metric is a distance to the identical Fresh prediction at d=0, so strict improvement is impossible by construction; exact matches are repeat/persistence cases."
        )
        non_markov_lines.append("")

    relation_rows = relation["task_mean_summary"]
    relation_table = [
        "| task | split | H_temp | mean FO−Reverse | mean abs FO−Reverse | mean FullOld−Fresh |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in relation_rows:
        relation_table.append(
            f"| {task_label(row['task_key'])} | {row['split']} | {fmt(row['H_temp'])} | {fmt(row['mean_FO_minus_Reverse'])} | {fmt(row['mean_abs_FO_minus_Reverse'])} | {fmt(row['mean_FullOld_minus_Fresh'])} |"
        )

    delay_relation_table = [
        "| d | mean FO−Reverse | mean abs FO−Reverse | positive / negative / zero tasks | Spearman H vs signed A | Spearman H vs |A| |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for delay, row in relation["delay_summaries"].items():
        delay_relation_table.append(
            f"| {delay} | {fmt(row['mean_FO_minus_Reverse'])} | {fmt(row['mean_abs_FO_minus_Reverse'])} | {row['positive_tasks']} / {row['negative_tasks']} / {row['zero_tasks']} | {fmt(row['spearman_H_temp_vs_FO_minus_Reverse'])} | {fmt(row['spearman_H_temp_vs_abs_FO_minus_Reverse'])} |"
        )

    split_summary = outcome_blind["split_summary"]
    ranking_lines = [
        "| rank | task | split | H_temp | episode bootstrap 95% CI |",
        "|---:|---|---|---:|---:|",
    ]
    profiles = outcome_blind["profiles"]
    for rank, row in enumerate(ranking, start=1):
        ci = profiles[row["task_key"]]["H_temp_episode_bootstrap_ci95"]
        ranking_lines.append(
            f"| {rank} | {task_label(row['task_key'])} | {row['split']} | {fmt(row['H_temp'])} | [{fmt(ci[0])}, {fmt(ci[1])}] |"
        )

    high_h = sorted(relation_rows, key=lambda row: row["H_temp"], reverse=True)
    counterexamples = relation["obvious_counterexamples"]
    counterexample_text = ", ".join(
        f"{task_label(row['task_key'])} (H={fmt(row['H_temp'])}, mean A={fmt(row['mean_FO_minus_Reverse'])})"
        for row in counterexamples
    ) or "none under the recorded rule"

    decision_text = {
        "GROUP_TEMPORAL_HETEROGENEITY_STRONG": "The profiles differ materially on most development tasks and the qualitative pattern persists on held-out tasks without being concentrated in one task; the existing outcome asymmetry is directionally compatible overall.",
        "GROUP_TEMPORAL_HETEROGENEITY_PARTIAL": "The outcome-blind profiles differ materially across the cohort and persist descriptively on held-out tasks, but the existing FO-versus-Reverse outcome association is not directionally consistent enough for a strong label. The heterogeneity signal remains identifiable without outcomes.",
        "GROUP_TEMPORAL_HETEROGENEITY_NULL": "The profiles are not stably different across tasks, or apparent heterogeneity is not supported by the frozen outcome-blind evidence. Group-wise temporal-memory development should stop.",
    }[decision]

    ladder = """
The result supports preparation of the following later, rollout-based ladder, without executing it here: M0 hard sparse h16; M1 shared sparse temporal ensemble; M2 whole-action sparse CogACT-style similarity weighting; M3 group-wise CogACT similarity weighting; M4 group-specific delay prior plus group-wise CogACT; M5 anchored group-wise temporal memory (newest joint action as anchor plus group-specific historical residual/fusion); and M6 optional reliability-weighted anchored group memory. For SmolVLA, an AutoHorizon eligibility mask can be noted as an implementation option. The intended distinction is soft group-specific historical correction around a common newest anchor, not independent hard group replacement.
""".strip()

    return f"""# Group-specific temporal memory: offline audit

Decision: `{decision}`

{decision_text}

This is an offline/development analysis. It does not establish causal control benefit. Same-target SmolVLA prediction differences combine observation-delay effects with stochastic flow sampling variation that was not keyed by physical step.

## 1. Data/protocol

The primary source is the Fresh dense-query SmolVLA cache at `{protocol['source']['cache_root']}`. Each episode has shape `[T, 50, 7]`, and the same-target alignment is `a_{{t|t-d}} = cache[t-d, d, :]`, with source query `q=t-d`. The separate ACT dense cache was inspected but not pooled because it has a different checkpoint/cohort and 100-step chunks. The exact frozen protocol is in [protocol.json](protocol.json).

The cohort contains {counts['tasks']} tasks, {counts['episodes']} episodes, and {counts['current_target_steps']} current target steps. The requested delay set is `d={', '.join(str(d) for d in delays)}` steps, or approximately `{' / '.join(f'{d/30:.3f}s' for d in delays)}` at 30 Hz. Candidate counts are support counts, not inferential N. The development split is `{', '.join(task_label(x) for x in protocol['task_split']['development'])}`; held-out descriptive tasks are `{', '.join(task_label(x) for x in protocol['task_split']['held_out_descriptive'])}`.

The arm is dimensions 0–5, with translation and rotation treated separately and equally combined using the validated PPPR-style IQR normalization fit from development Fresh actions only. The gripper is dimension 6; its continuous difference uses the fixed postprocessed range 2.0 and its sign disagreement is reported separately. B2 demonstration agreement was skipped because no correctly aligned expert-action table exists in the current SmolVLA artifacts, and labels were not reconstructed.

Total materialized feature rows, including masked cells, are {counts['feature_rows_including_masked']}; valid rows are {counts['valid_feature_rows']}. Per-delay support by task is in `delay_profiles.csv` and the cached feature table.

## 2. Arm delay profiles

`U_arm(d)=1-R_arm(d)`, where higher means more same-target agreement with Fresh, not better control. The columns below list values in the frozen delay order. The revision component columns are exported in `delay_profiles.csv`; the table gives the combined revision explicitly and keeps translation/rotation diagnostics available in the cached table.

{chr(10).join(arm_table)}

## 3. Gripper delay profiles

`U_grip(d)=1-clip(mean(abs(g_old-g_fresh))/2, 0, 1)`. Sign disagreement is the fraction of valid target positions with different `np.sign` commands. Delay maxima, slopes, AUCs, and the development-only threshold crossing are descriptive summaries, not exact optimal horizons.

{chr(10).join(grip_table)}

The exact episode-level profiles and per-delay valid support are in [episode_delay_profiles.csv](episode_delay_profiles.csv), and all row-level aligned features are in [cached_same_target_features.npz](cached_same_target_features.npz). Because d=0 is the Fresh identity, the report shows the best positive-delay summary in addition to the full sampled profile; the d=0 maximum is a reference identity, not evidence of an optimal executor horizon.

## 4. Non-Markovian evidence

Under the legitimate offline revision metric, historical sources do not strictly outperform Fresh: Fresh at d=0 is the identical target prediction, so its revision distance is exactly zero and its bounded utility is exactly one. Exact historical matches can occur and are reported below. This does not show that historical observations are useless; it shows that this particular same-target consistency metric is anchored to Fresh and cannot demonstrate historical superiority. B2 expert agreement was unavailable, and no closed-loop outcomes are merged into this section.

{chr(10).join(non_markov_lines)}

## 5. Temporal heterogeneity

The primary task descriptor was frozen before reading the intervention outcome file:

`H_temp(task) = mean_d |U_arm(d) - U_grip(d)|` over the six requested delays, with equal delay weight and equal task weight. It uses the normalized same-target revision utility only. This definition, normalization, split, and freeze status are recorded in [protocol.json](protocol.json) and [h_temp_frozen.json](h_temp_frozen.json). The outcome-blind builder explicitly loaded no success/intervention artifact.

{chr(10).join(ranking_lines)}

Development macro H_temp is {fmt(split_summary['development']['task_macro_mean_H_temp'])} (SD {fmt(split_summary['development']['task_macro_sd_H_temp'])}); held-out macro H_temp is {fmt(split_summary['held_out']['task_macro_mean_H_temp'])} (SD {fmt(split_summary['held_out']['task_macro_sd_H_temp'])}). The score is not dominated by a single task if its rank and bootstrap interval are read together, but the eight-task sample remains descriptive. Figure A shows the profiles and Figure B shows the frozen task score.

## 6. Existing closed-loop relation

Only after [h_temp_frozen.json](h_temp_frozen.json) was written, the existing source-intervention results were opened. For each task and d in 4, 8, and 16, `A_task(d)=success(FO_d)-success(Reverse_d)`, with FullOld reported separately.

{chr(10).join(delay_relation_table)}

{chr(10).join(relation_table)}

Across delay-specific rows, the descriptive Spearman correlations of H_temp with signed FO−Reverse are `d=4: {fmt(relation['delay_summaries']['4']['spearman_H_temp_vs_FO_minus_Reverse'])}`, `d=8: {fmt(relation['delay_summaries']['8']['spearman_H_temp_vs_FO_minus_Reverse'])}`, and `d=16: {fmt(relation['delay_summaries']['16']['spearman_H_temp_vs_FO_minus_Reverse'])}`. The across-delay task-mean correlation is `{fmt(relation['across_delays_task_mean']['spearman_H_temp_vs_mean_FO_minus_Reverse'])}` for signed A and `{fmt(relation['across_delays_task_mean']['spearman_H_temp_vs_mean_abs_FO_minus_Reverse'])}` for |A|. With eight tasks these are descriptive, not significance tests. Obvious counterexamples under the fixed high-H/negative-mean-A rule are: {counterexample_text}. FullOld is not used to define H_temp and is only reported as a separate whole-action comparison. Figure C is the task scatter.

## 7. Reliability reinterpretation

The current branch does not contain a directly comparable SmolVLA source-context reliability table. The validated historical ACT artifact at `{protocol['existing_source_reliability']['source']}` defines reliability as prefix survival of an old cached action against the corresponding Fresh action at the future target: arm is reliable when both normalized translation and rotation discrepancies remain at most 1.0; gripper is reliable when normalized absolute error is at most 1.0 and signs agree. It is semantically compatible with source-age persistence but comes from a different cohort and is not pooled into H_temp.

The historical oracle curves show group-specific decay:

{reliability_context()}

The same artifact reports that the group horizons differ in 95.9% of rows where both are uncensored, with gripper expiring first in 64.9% and arm first in 31.0%. In the frozen chunk-only source-context ablation, fixed-cohort AUROC was 0.838 for arm versus 0.964 for gripper, while horizon MAE was 24.06 versus 7.51 actions, respectively. Thus reliability classification was easier for gripper, whereas converting reliability into a hard predicted horizon was especially poor for arm. Together, these results support treating reliability as a continuous/soft temporal signal candidate, not as evidence for a hard per-group horizon. They do not establish that soft weighting improves control.

## 8. Decision

`{decision}`

{decision_text}

## 9. Recommended next experiment

{ladder if decision != 'GROUP_TEMPORAL_HETEROGENEITY_NULL' else 'No group-wise temporal-memory executor experiment is recommended from this audit. Stop that development line unless new outcome-blind evidence changes the decision.'}

## Files and reproducibility

- [outcome_blind.json](outcome_blind.json) is the frozen primary result and records `closed_loop_files_loaded: false`.
- [h_temp_frozen.json](h_temp_frozen.json) is the frozen task score artifact.
- [closed_loop_relation.json](closed_loop_relation.json) contains the secondary association only.
- [tests/test_offline_semantics.py](tests/test_offline_semantics.py) covers alignment, slicing, availability, normalization, d=0 identity, and outcome-blind independence.
- Figures are [Figure A](figures/figure_A_delay_profiles.png), [Figure B](figures/figure_B_task_heterogeneity.png), and [Figure C](figures/figure_C_h_temp_vs_fo_reverse.png).
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", required=True, choices=sorted(VALID_DECISIONS))
    parser.add_argument("--output", type=Path, default=ROOT / "report.md")
    args = parser.parse_args()
    protocol = load_json(ROOT / "protocol.json")
    outcome_blind = load_json(ROOT / "outcome_blind.json")
    relation = load_json(ROOT / "closed_loop_relation.json")
    report = render(protocol, outcome_blind, relation, args.decision)
    args.output.write_text(report)
    (ROOT / "STATUS.md").write_text(
        f"# Status\n\n- State: complete\n- Decision: `{args.decision}`\n- Primary artifact: `outcome_blind.json`\n- Secondary artifact: `closed_loop_relation.json`\n- Rollouts: none\n- ACT/SmolVLA changes: none\n- Executor implementation: none\n- Tests: `pytest experiments/group_temporal_memory_offline/tests`\n"
    )
    print(json.dumps({"status": "report_written", "decision": args.decision, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
