#!/usr/bin/env python3
"""Build the final frozen experiment handoff from canonical artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CROSS = ROOT.parent / "icra27_crosssuite_query_allocation"


def ci(values: list[float]) -> str:
    return f"[{values[0]:+.2f}, {values[1]:+.2f}]"


def rate(successes: int, n: int) -> str:
    return f"{successes}/{n} ({100 * successes / n:.2f}%)"


def contrast_table(lines: list[str], rows: list[dict]) -> None:
    lines += [
        "| Contrast | Successes | Delta (pp) | Discordance | exact McNemar p | paired 95% CI (pp) | task-cluster 95% CI (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['contrast']}` | {row['first_successes']} vs {row['second_successes']} / {row['N']} | "
            f"{row['delta_percentage_points']:+.2f} | {row['first_only']}:{row['second_only']} | "
            f"{row['exact_two_sided_mcnemar_p']:.6g} | {ci(row['paired_bootstrap_ci_percentage_points'])} | "
            f"{ci(row['task_cluster_bootstrap_ci_percentage_points'])} |"
        )
    lines.append("")


def per_task_matrix(lines: list[str], title: str, rows: list[dict]) -> None:
    tasks = sorted(rows[0]["per_task_delta_percentage_points"])
    lines += [f"### {title}", "", "| Task | " + " | ".join(f"`{row['contrast']}`" for row in rows) + " |",
              "|---|" + "---:|" * len(rows)]
    for task in tasks:
        lines.append("| " + task + " | " + " | ".join(
            f"{row['per_task_delta_percentage_points'][task]:+.2f}" for row in rows
        ) + " |")
    lines.append("")


def b3_value(record: dict, key: str) -> str:
    if key == "sign":
        item = record["gripper_sign_disagreement"]
        center = item["probability"]
    else:
        item = record["groups"][key]
        center = item["rmse"]
    bounds = item["episode_cluster_bootstrap_ci"]
    return f"{center:.4f} [{bounds[0]:.4f}, {bounds[1]:.4f}]"


def b3_table(lines: list[str], policy: str, records: dict[str, dict]) -> None:
    lines += [
        f"### {policy}: complete frozen B3 curve",
        "",
        "All entries are center `[episode-cluster 95% CI]`. RMSE quantities are in that policy's frozen normalized action space.",
        "",
        "| k | seconds | translation RMSE | rotation RMSE | arm RMSE | gripper absolute normalized error | gripper sign disagreement |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in range(33):
        record = records[str(k)]
        lines.append(
            f"| {k} | {k / 20:.2f} | {b3_value(record, 'translation')} | {b3_value(record, 'rotation')} | "
            f"{b3_value(record, 'arm')} | {b3_value(record, 'gripper')} | {b3_value(record, 'sign')} |"
        )
    lines.append("")


def main() -> None:
    supplement = json.loads((ROOT / "canonical_report.json").read_text(encoding="utf-8"))
    canary = json.loads((ROOT / "canaries/r1_prelaunch.json").read_text(encoding="utf-8"))
    repair = json.loads((ROOT / "canaries/r1d_runtime_repair.json").read_text(encoding="utf-8"))
    b3 = json.loads((CROSS / "track_b/forecast/analysis/summary.json").read_text(encoding="utf-8"))
    b3_spec = json.loads((CROSS / "track_b_analysis_addendum.json").read_text(encoding="utf-8"))["b3"]
    relationship = json.loads((CROSS / "track_b/final_mechanism_relationships/summary.json").read_text(encoding="utf-8"))
    track_a = json.loads((CROSS / "track_a/analysis.json").read_text(encoding="utf-8"))
    te = json.loads((CROSS / "track_a/te_dense_characterization/analysis.json").read_text(encoding="utf-8"))
    conditional = json.loads((CROSS / "track_b/conditional_mechanism/summary.json").read_text(encoding="utf-8"))

    if supplement["status"] != "COMPLETE" or supplement["scientific_retries"] != 0:
        raise RuntimeError("supplement canonical analysis is not complete and zero-retry")
    if canary["r1c"]["status"] != "PASS" or repair["status"] != "PASS":
        raise RuntimeError("required integrity canary did not pass")
    if b3["status"] != "COMPLETE" or b3["offsets"] != list(range(33)):
        raise RuntimeError("B3 canonical curve is incomplete")
    if b3["offset_seconds"] != [k / 20 for k in range(33)]:
        raise RuntimeError("B3 physical axis is not corrected")

    summaries = {(row["family"], row["method"]): row for row in supplement["condition_summaries"]}
    contrasts = {(row["family"], row["contrast"]): row for row in supplement["contrasts"]}
    r1a_ds = (2, 4, 8, 12, 16, 20, 32)

    lines = [
        "# Final experiment handoff", "",
        "Status: `FINAL_SCIENTIFIC_CLAIM_FREEZE`", "",
        "Branch: `exp/icra27-crosssuite-query-allocation`", "",
        "Pre-unblinding technical disposition: `FROZEN_BEFORE_REVIEWER_SUPPLEMENT_OUTCOME_INSPECTION`.", "",
        "No manuscript, LaTeX, `CLAIMS.md`, or paper-facing artwork was changed.", "",
        "## 1. Runtime/source provenance", "",
        "### Track A", "",
        "Track A, including TE_DENSE, ran with `/home/wjq/workspace/venvs/libero_act/bin/python` and installed pip/site-packages LeRobot 0.4.4 at `/home/wjq/workspace/venvs/libero_act/lib/python3.12/site-packages/lerobot`. There was no LeRobot `PYTHONPATH`, editable install, or checkout shadow. `ACTTemporalEnsembler`, ACT configuration/loading, the policy factory, and LIBERO construction all resolved to that 0.4.4 package.", "",
        "### R1A--R1C", "",
        "R1A, R1B, and R1C used the same interpreter and installed LeRobot 0.4.4 paths. They imported the full 0.4.4 ACT/config/factory path, not `/home/wjq/workspace/upstreams/lerobot/src`. The launcher set no `PYTHONPATH`; the unrelated editable `verl-vla` path did not contain or shadow `lerobot`.", "",
        "### R1D", "",
        "The failed original R1D initialization selected `/home/wjq/workspace/upstreams/lerobot/src` at clean commit `f66e5128ecb2456e8c54a63d15404fa59c16aebc`. Its import chain was `run_queue.Runtime -> lerobot.policies.factory -> lerobot.policies.__init__ -> eo1.configuration_eo1 -> Qwen2_5_VLTextConfig`; Transformers 4.51.3 lacks that export. EO1/Qwen is unrelated to the frozen ACT evaluator.", "",
        "The neutral repair preloaded only the installed LeRobot 0.4.4 package root, then invoked the unchanged frozen queue. Every ACT/LIBERO submodule consequently resolved to the same 0.4.4 files used by R1A--R1C. The source comparison found no ACT inference, chunk-indexing, or temporal-ensembler change. The checkout explicitly forwards `fps=20`; 0.4.4 omits it and resolves to the same LIBERO/robosuite default 20 Hz. No relevant package, checkout, environment setting, or source changed in the 0.307 s between R1C completion and the original R1D launch.", "",
        "The corrective technical record is `experiments/icra27_reviewer_supplement/RUNTIME_SOURCE_PROVENANCE_AUDIT_20260903.md`; it preserves the failed launch's historical source identity and records the completed run's actual source separately.", "",
        "### Temporal-contract consequence", "",
        "The conclusions remain valid: ACT policy index = 0.05 s; R1A--R1D evaluator step = 0.05 s; `d=20` = 1.00 s; and `q+k=t` is physically same-target.", "",
        "## 2. R1A fixed-source temporal sensitivity", "",
        "> Report successes/rates, `Fresh-A_d_G0`, `Fresh-A0_G_d`, and `A0_G_d-A_d_G0`, with discordances, exact McNemar, paired and task-cluster bootstrap intervals, per-task effects, LOTO, queries, query rate, steps, and wall time for every `d`. No threshold or best `d` is selected.", "",
        "Status: `EXPOSED_DEVELOPMENT_CHARACTERIZATION`. Fresh, A20G0, and A0G20 are audited historical reuse anchors; the other fixed-age conditions are new supplement cells.", "",
        "| d | seconds | `S(A_d G0)` | `S(A0 G_d)` | `S(A0 G_d)-S(A_d G0)` (pp) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for d in r1a_ds:
        arm = summaries[("r1a", f"A{d}_G0")]
        grip = summaries[("r1a", f"A0_G{d}")]
        asym = contrasts[("r1a", f"A0_G{d}-A{d}_G0")]
        lines.append(f"| {d} | {d / 20:.2f} | {rate(arm['successes'], arm['N'])} | {rate(grip['successes'], grip['N'])} | {asym['delta_percentage_points']:+.2f} |")
    fresh = summaries[("r1a", "A0_G0")]
    lines += ["", f"Common Fresh/d=0 anchor: {rate(fresh['successes'], fresh['N'])}.", "", "### Fresh-relative and between-branch inference", ""]
    r1a_rows = []
    for d in r1a_ds:
        r1a_rows += [contrasts[("r1a", f"A0_G0-A{d}_G0")], contrasts[("r1a", f"A0_G0-A0_G{d}")], contrasts[("r1a", f"A0_G{d}-A{d}_G0")]]
    contrast_table(lines, r1a_rows)
    per_task_matrix(lines, "R1A per-task Fresh minus arm-stale effects (pp)", [contrasts[("r1a", f"A0_G0-A{d}_G0")] for d in r1a_ds])
    per_task_matrix(lines, "R1A per-task Fresh minus gripper-stale effects (pp)", [contrasts[("r1a", f"A0_G0-A0_G{d}")] for d in r1a_ds])
    per_task_matrix(lines, "R1A per-task gripper-stale minus arm-stale asymmetry (pp)", [contrasts[("r1a", f"A0_G{d}-A{d}_G0")] for d in r1a_ds])
    lines += [
        "The arm-success curve is not monotone: it falls through d=12, rises at d=16, reaches its minimum at d=20, and rises again at d=32. The gripper-success curve is also not monotone: it rises through d=16 and then declines at d=20 and d=32. The branch ordering is preserved at every frozen d, with no reversal: `S(A0 G_d) > S(A_d G0)` throughout. The largest divergence is at d=20 (1.00 s), +54.76 pp. These are complete-grid characterizations, not lag selection.", "",
        "## 3. R1B translation versus rotation", "",
        "> On the same 126 blocks at `d=20`, run `T20_R0_G0` (translation stale; rotation and gripper Fresh) and `T0_R20_G0` (rotation stale; translation and gripper Fresh).", "",
        "> B1 found the same-target normalized dispersion ordering `rotation > translation > gripper` for both ACT and SmolVLA. If same-target source disagreement predicts behavioral temporal sensitivity, then at d=20 stale rotation should be at least as damaging as stale translation: `success(T0_R20_G0) <= success(T20_R0_G0)`.", "",
        f"Translation-stale: {rate(summaries[('r1b','T20_R0_G0')]['successes'],126)}; rotation-stale: {rate(summaries[('r1b','T0_R20_G0')]['successes'],126)}; Fresh: {rate(summaries[('r1b','A0_G0')]['successes'],126)}.", "",
    ]
    r1b_rows = [contrasts[("r1b", name)] for name in ("T20_R0_G0-T0_R20_G0", "T20_R0_G0-A0_G0", "T0_R20_G0-A0_G0")]
    contrast_table(lines, r1b_rows)
    per_task_matrix(lines, "R1B per-task effects (pp)", r1b_rows)
    lines += [
        "The prospective B1-derived prediction is `MECHANISM_DISSOCIATION`: the direct contrast is -33.33 pp, not non-negative. Translation staleness is strongly damaging; rotation staleness is close to Fresh. Because translation and rotation behave differently on every task in the primary contrast, R1B supports broader within-arm component dependence, while rejecting B1 dispersion ordering as its explanation. The paper need not narrow to only arm-versus-gripper, but must keep this evidence ACT- and Object-development-specific.", "",
        "## 4. R1C dense-query matched H16 factorial", "",
        "> All four conditions make a whole-policy query every step: `C00` Fresh/Fresh; `C10` scheduled-H16 arm/Fresh gripper; `C01` Fresh arm/scheduled-H16 gripper; `C11` scheduled-H16 arm/gripper.", "",
        "> Before bulk rollout a frozen deterministic canary must show that unused dense forward passes leave C11 executed actions, simulator trajectory, terminal result, and length identical to sparse HARD-H16. Canary failure stops R1C.", "",
        "The exact frozen identity gate passed: dense C11 and sparse HARD-H16 had exactly identical executed actions, simulator trajectory from the same initial state, terminal success, completion step, and episode length. Extra discarded policy queries therefore did not change the canary trajectory.", "",
        "| Condition | Reader-facing semantics | Success/N |",
        "|---|---|---:|",
        f"| C00 | dense-query Fresh arm + Fresh gripper | {rate(summaries[('r1c','C00')]['successes'],140)} |",
        f"| C10 | dense-query scheduled-H16 arm + Fresh gripper | {rate(summaries[('r1c','C10')]['successes'],140)} |",
        f"| C01 | dense-query Fresh arm + scheduled-H16 gripper | {rate(summaries[('r1c','C01')]['successes'],140)} |",
        f"| C11 | dense-query scheduled-H16 arm + scheduled-H16 gripper | {rate(summaries[('r1c','C11')]['successes'],140)} |", "",
    ]
    r1c_names = ("C10-C00", "C01-C00", "C11-C10", "C11-C01", "C11-C00", "C10-C01")
    r1c_rows = [contrasts[("r1c", name)] for name in r1c_names]
    contrast_table(lines, r1c_rows)
    per_task_matrix(lines, "R1C per-task frozen effects (pp)", r1c_rows)
    lines += [
        "Per-suite conditional effects are retained in `canonical_report.json`. In particular, C11-C10 is +20.00 pp on LIBERO-10 and +4.29 pp on Goal; C11-C01 is +18.57 pp on LIBERO-10 and -1.43 pp on Goal. The frozen risk-difference interaction `C11-C10-C01+C00` is +9.29 pp.", "",
        "Conclusion: the H16 advantage survives whole-policy query-rate matching. Because C11 exactly matches sparse H16 on the preregistered identity canary, extra discarded forward passes are not an executed-behavior confound under that frozen rule. This does not authorize component percentage attribution.", "",
        "## 5. B3 forecastability", "",
        "> If and only if the temporal-contract audit establishes an unambiguous mapping, compare the complete ACT B3 future-action forecast curves with the complete R1A behavioral sensitivity curves on a seconds-first axis. The provenance-only timebase audit resolved the 20 Hz R1A ages `d={2,4,8,12,16,20,32}` to the exact 10 Hz B3 offsets `k={1,2,4,6,8,10,16}` at matched times `{0.1,0.2,0.4,0.6,0.8,1.0,1.6}` seconds, so no interpolation is used. The table must include B3 arm, translation, rotation, gripper-value, and gripper-sign forecast quantities and R1A `Fresh-A_d_G0` and `Fresh-A0_G_d`. This is cross-cohort descriptive characterization, not a significance-gated hypothesis. If time alignment is ambiguous, report the comparison as not identifiable and keep native step axes separate. The full B3 offset grid `0..32` and every frozen R1A age remain reported in their own canonical analyses; this matched-time table is not a lag-selection rule.", "",
        "The quoted 10 Hz offset mapping is retained verbatim as historical governance. The later corrective temporal audit, which controls the present handoff, established the physical B3 mapping as `k/20` seconds; therefore the exact matched correspondence is `d={2,4,8,12,16,20,32}` to `k={2,4,8,12,16,20,32}`. The characterization rule itself is unchanged.", "",
        "> Episode-cluster percentile bootstrap uses 20,000 draws, 95% intervals, seed 27401 for ACT and 27402 for SmolVLA. No performance outcome, task subset, lag, or offset is selected after results. These associations do not establish that persistence or forecastability causes executor sensitivity.", "",
        "Classification: `B3_NO_FROZEN_DISCRIMINATIVE_CRITERION`. No threshold, correlation, or ordering rule for mechanism support was frozen, so none is constructed after unblinding.", "",
        "Metric and target: at each demonstration anchor t, compare the frozen policy's normalized predicted action at chunk offset k with the recorded demonstrated action at exact target row t+k. Report per-dimension RMSE, translation/rotation/combined-arm normalized RMSE, gripper absolute normalized error, and controller-native gripper sign-disagreement probability. Physical offset is `k/20` seconds for every k=0..32.", "",
        f"Cohort: training-demonstration reference, not held-out; tasks Object 3, Spatial 0, Goal 2, and LIBERO-10 3. The selection rule is `{b3_spec['episode_selection_rule']}`. Episode IDs: " + "; ".join(f"{task}={ids}" for task, ids in b3_spec["episodes"].items()) + ".", "",
        "ACT uses the four frozen task-specific ACT checkpoints; SmolVLA uses local revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`, whose training-data relationship remains unknown. Policies are reported in separate normalized spaces. Uncertainty is the already-generated 20,000-draw demonstration-episode cluster percentile 95% interval (ACT seed 27401; SmolVLA seed 27402).", "",
    ]
    b3_table(lines, "ACT", b3["policy_results"]["ACT"])
    b3_table(lines, "SmolVLA", b3["policy_results"]["SmolVLA"])
    lines += [
        "At the exact R1A-matched offsets k=d, ACT rotation forecast RMSE exceeds translation RMSE at all seven points, while behavior shows translation staleness is much more damaging than rotation staleness. ACT combined-arm error is nearly flat over most of 0..32, whereas R1A arm harm grows and then recedes non-monotonically. ACT gripper sign disagreement remains near zero even where gripper staleness improves behavioral success. Under the frozen descriptive rule, simple forecastability therefore does not provide discriminative mechanism support. B3 is retained as null/mixed mechanism evidence and the mechanism search is closed.", "",
        "## 6. R1D Spatial Reverse20 completion", "",
        "Status: `POST_HOC_SPATIAL_FACTORIAL_COMPLETION`. The import-only repair and canary details are in Sections 1 and 6; `REFERENCE_SEQUENCE_UNAVAILABLE` applies to the optional stronger technical comparison.", "",
        "Required canaries passed: exact ACT imports; exact checkpoint/config identities; policy load; Spatial environment construction/reset; actual 20 Hz clock; expected 256x256 two-camera plus 8D-state preprocessing; frozen normalization/denormalization processors; Reverse20 Fresh-prefix/source-age semantics; physical `q+k=t`; exact manifest; zero prelaunch results, markers, and attempts.", "",
        "Exactly 100 new scientific cells executed; skipped completed cells = 0; retries = 0; duplicates = 0; frozen manifest identity = exact match; terminal validator = PASS.", "",
        "| Condition | Success/N |",
        "|---|---:|",
    ]
    for method in ("A0_G0", "A0_G20", "A20_G0", "A20_G20"):
        row = summaries[("r1d", method)]
        lines.append(f"| `{method}` | {rate(row['successes'],row['N'])} |")
    lines += ["", "### Frozen R1D contrasts", ""]
    r1d_names = ("A0_G20-A20_G0", "A20_G0-A0_G0", "A0_G20-A0_G0", "A20_G20-A0_G0")
    r1d_rows = [contrasts[("r1d", name)] for name in r1d_names]
    contrast_table(lines, r1d_rows)
    per_task_matrix(lines, "R1D per-task effects (pp)", r1d_rows)
    lines += [
        "Spatial completes the same qualitative arm–gripper asymmetry: stale arm reduces success by 28.00 pp from Fresh, stale gripper changes success by 0.00 pp, and stale-gripper minus stale-arm is +28.00 pp. This separate post-hoc 100-block panel is not pooled into the original 140-block confirmation.", "",
        "## 7. R2A", "",
        "`R2A_NOT_RUN_FROZEN_GATE_INELIGIBLE`. Original epoch 1788354953 = 2026-09-02T21:15:53+08:00; eligibility required elapsed <=57,600 s; observed elapsed was 63,630 s. R2A was correctly not launched. This is a permanent governance outcome, not a technical failure.", "",
        "## 8. TE_DENSE", "",
        "Actual runtime provenance is installed LeRobot 0.4.4. TE_DENSE uses canonical upstream `ACTTemporalEnsembler`, coefficient 0.01, chunk length 100. The oldest available prediction receives the first temporal weight; a positive coefficient intentionally weights older predictions more strongly. The runtime canary confirms this actual direction and seven-dimension normalized-space aggregation.", "",
        "Observed characterization: empirical weighted mean age 44.99 steps = 2.249 s; weighted p50 43 steps = 2.15 s; p95 94 steps = 4.70 s; 52.27% normalized weight is older than 2.0 s. `abs(g)<0.50` occurs on 24.41% of executed steps; gripper sign/state-switch rate is 1.02%.", "",
        "Scoped interpretation: under the frozen upstream/runtime coefficient and chunk length, TE_DENSE places substantial aggregate weight on older predictions and produces substantially more near-boundary gripper commands than the other frozen executors. This is not evidence that canonical temporal ensembling is intrinsically harmful, that near-boundary commands causally explain the full loss, or that gripper chatter is the failure mode. The low switch rate is incompatible with a chatter description. No TE tuning or further analysis is authorized.", "",
        "## 9. Track A", "",
        "Track A's main matched-query component-resolved contrasts are ARM4_GRIP32-H4 = +4.667 pp (335/450 vs 314/450; paired 95% CI [+2.00,+7.56], task-cluster CI [+0.67,+9.11]) and ARM2_GRIP16-H2 = +5.778 pp (321/450 vs 295/450; paired CI [+3.56,+8.22], task-cluster CI [+2.67,+9.56]). Policy-query rates are matched within each comparison: approximately 0.251 and 0.501, respectively.", "",
        "H16 remains the boundary: 357/450 (79.33%), exceeding ARM4_GRIP32 by 4.889 pp and ARM2_GRIP16 by 8.000 pp. The measured path `H2 -> ARM2_GRIP16 -> H16` is +5.778 pp followed by an additional +8.000 pp. It is a measured path only, not unique or path-independent component attribution. No analogous symmetric decomposition is allowed for `H4 -> ARM4_GRIP32 -> H16`, because the second edge changes both arm 4->16 and gripper 32->16.", "",
        "H16 absolute suite baselines are LIBERO-10 54.7%, Goal 91.3%, and Spatial 92.0%. ARM4_GRIP32-H4 is +13.333 pp, 0.000 pp, and +0.667 pp; ARM2_GRIP16-H2 is +14.000 pp, +2.000 pp, and +1.333 pp, respectively. The largest gain occurs on the hardest suite, while Goal and Spatial operate near the H16 ceiling. Because suite identity, baseline difficulty, and task semantics covary, the source of this concentration is not identifiable.", "",
        "LOSO minimal-margin disclosure: the minimum leave-one-suite-out effect is +0.333 pp for ARM4_GRIP32-H4 and +1.667 pp for ARM2_GRIP16-H2; all three LOSO estimates are positive for both. This robustness summary is not independent evidence.", "",
        "## 10. Mechanism accounting", "",
        "- B1 same-target source disagreement: `ACT_LOCALIZATION_KILL`; no cross-policy mechanism support. Normalized dispersion orders rotation > translation > gripper for ACT and SmolVLA, but R1B behavior orders translation staleness as far more damaging than rotation staleness. Status: descriptive source disagreement plus mechanism dissociation.", "",
        "- B2 training-demonstration persistence: gripper actions are persistent, but this is training-data characterization, not held-out evidence or a causal mechanism. Under the corrected 20 Hz physical mapping, S(5), S(10), and S(20) correspond to 0.25, 0.50, and 1.00 s; their survival estimates are 0.921765, 0.840878, and 0.675018. The complete-case 30.775-step mean is biased and not a population mean.", "",
        f"- Prospective conditional moderators: gripper transition density versus Delta_G has rho={conditional['gripper_prediction']['spearman_rho']:.3f} (descriptive p={conditional['gripper_prediction']['two_sided_p_descriptive']:.3f}); arm variation versus Delta_A has rho={conditional['arm_prediction']['spearman_rho']:.3f} (p={conditional['arm_prediction']['two_sided_p_descriptive']:.3f}). The first is directionally compatible but uncertain; the second is null.", "",
        "- Frozen Track-A gripper-activity occupancy moderator: rho=0.192, descriptive p=0.309 across all 30 tasks. Status: unsupported.", "",
        "- B3 forecastability: `B3_NO_FROZEN_DISCRIMINATIVE_CRITERION`; the complete curves are descriptively non-corresponding with R1A/R1B behavior. Status: no discriminative mechanism support.", "",
        "- Command discontinuity: `NON_IDENTIFYING_POST_HOC_CHARACTERIZATION`. Between-condition D1 comparisons are confounded by state/trajectory composition and may depend on prediction offset. Coherence is neither supported, falsified, nor causal.", "",
        "- Supported explanations/claims: ACT execution is temporally component-dependent; matched-query gripper commitment can improve success relative to matched global short horizons; translation and rotation staleness differ strongly on the exposed Object cohort; the matched-query R1C result rules out policy-query count alone under the frozen canary contract.", "",
        "- Killed or unsupported explanations: ACT-specific gripper instability/chatter localization; B1 dispersion ordering as a behavioral-sensitivity predictor; frozen occupancy moderation; simple B3 forecastability as discriminative mechanism support; command-coherence causality; canonical TE as intrinsically harmful.", "",
        "- Unresolved: the causal mechanism producing component differences; the source of suite concentration; SmolVLA's physical training timebase and cross-policy generality; whether TE near-boundary commands explain any success loss; generality beyond one checkpoint/training seed per task.", "",
        "## 11. Final paper claim scope", "",
        "Strongest defensible claim: for the evaluated ACT policies, temporal source/execution choices have component-dependent behavioral effects. Cross-suite matched-query allocation improves over matched global short-horizon baselines, while coherent H16 remains better; exposed fixed-age characterization further separates translation, rotation, and gripper responses.", "",
        "The working broad title `Component-Dependent Temporal Effects in Action-Chunked Robot Policies` may remain, because R1B directly shows within-arm translation-versus-rotation dependence rather than only an arm-versus-gripper split. The paper text must narrow the empirical claim to evaluated ACT policies and must not imply universal cross-policy confirmation.", "",
        "Remove or avoid: unique arm/gripper contribution; component percentages; per-dimension additive attribution; path-independent decomposition; a symmetric H4->ARM4_GRIP32->H16 decomposition; causal suite moderators; a forecastability/coherence mechanism claim; gripper chatter; intrinsic harm from canonical temporal ensembling; broad SmolVLA replication; pooling R1D into the original 140 blocks.", "",
        "Principal limitations: exposed/development status of R1A/R1B; post-hoc status of R1C/R1D; task/suite/checkpoint covariation; one checkpoint/training seed per task; no frozen discriminative mechanism criterion for B3; unresolved causal mechanism; SmolVLA timebase provenance gap.", "",
        "## 12. Experiment closure", "",
        "Open-ended scientific search is `CLOSED`. R1A, R1B, R1C, R1D, and B3 are complete and canonically reported; R2A is permanently gate-ineligible. No new executor, rescue method, d sweep, horizon, mechanism analysis, seed, benchmark, TE coefficient, or SmolVLA repair is recommended or authorized. Only a genuine technical-integrity defect capable of invalidating an existing main claim may reopen execution.", "",
        "Next state: `FINAL SCIENTIFIC CLAIM FREEZE -> PAPER WRITING -> FINAL FIGURES`.", "",
    ]

    output = REPO / "FINAL_EXPERIMENT_HANDOFF.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "path": str(output), "lines": len(lines)}, indent=2))


if __name__ == "__main__":
    main()
