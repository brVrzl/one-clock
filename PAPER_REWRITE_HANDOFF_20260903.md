# ICRA 2027 final manuscript rewrite handoff

Date: 2026-09-03 (Asia/Shanghai)

## Title

**Component-Dependent Sensitivity to Stale Predictions in ACT Action-Chunk Execution**

## Abstract

How old can a predicted action be before it should be refreshed? Action-chunk policies forecast a trajectory from each observation, yet execution rules apply one temporal setting to the action vector. We diagnose component dependence by taking arm and gripper commands from ACT chunks while aligning both to the same physical control step. These probes query the policy once per step and are diagnostic interventions, not deployment schedules. In a preregistered 140-block ACT evaluation, moving a one-second-old prediction from the gripper to the arm changed success from 83/140 to 38/140, a 32.14-point difference. On a separate development cohort, the fresh-arm/stale-gripper branch remained above the stale-arm/fresh-gripper branch from 0.10 to 1.60 s, with observed separations of 21.43--54.76 points. Translation staleness was also substantially more damaging than rotation staleness. In executable schedules, extending gripper commitment improved success by 4.667 and 5.778 points at two arm cadences with nearly matched policy-query rates. Coherent H16 nevertheless remained the strongest operating point. Same-target disagreement ranked the arm components in the opposite order from behavior, and the probe couples source age with prediction lookahead. The mechanism and generality beyond the evaluated ACT policies therefore remain unresolved.

## Section structure

1. Introduction centered on the executor refresh decision and component heterogeneity.
2. Related Work grouped by action chunking, replanning, stale/asynchronous execution, and component-structured actions.
3. Method and Experimental Setup with a recipe-like A0G20/A20G0 definition, `q+k=t`, the 20 Hz clock, cohorts, schedules, inference, and a compact reader-facing provenance table.
4. Results organized around four questions: component asymmetry, age/component sensitivity, diagnostic explanation, and executable schedules.
5. Discussion and Limitations covering the required identification, provenance, deployment, suite, mechanism, and policy-scope boundaries.
6. Short Conclusion.
7. Supporting Interaction Accounting appendix; interaction is absent from the main Results and no estimates are pooled.

## Build and page count

- Compile command used: temporary Tectonic 0.17.0, `tectonic -X compile main.tex --keep-logs --keep-intermediates`, with the public Tectonic bundle.
- Build status: successful, including BibTeX and cross-reference reruns.
- Output: `paper/icra2027/main.pdf`.
- Page count: **6 pages including references and the supporting appendix** with the current specification-box figure placeholders.
- Paper size: US Letter.

## Key claims retained

- The preregistered A0G20--A20G0 contrast is +32.14 percentage points (83/140 versus 38/140).
- The Object development ordering persists at all tested ages from 0.10 to 1.60 s; both branches are non-monotone and no lag optimum is claimed.
- Translation-stale minus rotation-stale success is -33.33 points on Object-126, with paired and task-cluster intervals strictly below zero.
- Same-target dispersion ranks rotation above translation, opposite the within-arm behavioral sensitivity ordering.
- Dense-query R1C rules out query schedule alone for the reported conditional gripper effect in the tested deterministic ACT evaluator.
- Track A improves H4 and H2 by +4.667 and +5.778 points when only gripper commitment is extended at nearly matched policy-query rates.
- Coherent H16 remains the strongest overall frozen operating point at 357/450 (79.33%).

## Key limitations retained

- Same-target construction couples source-observation age and prediction lookahead.
- Same-target probes query every executed step and are diagnostics, not deployment executors.
- R1A/R1B are development/reviewer-directed characterizations rather than the primary confirmatory cohort.
- Track-A gains concentrate in LIBERO-10, and suite identity, difficulty, and task semantics covary.
- The tested diagnostics do not identify a positive mechanism.
- Behavioral conclusions are limited to the evaluated ACT policies and LIBERO setting.
- SmolVLA lacks an identifiable physical training/chunk timebase for a matched replication.
- R2A was not run because its prespecified runtime eligibility window expired.

## Figure placeholders

- Fig. 1: same-target concept and primary preregistered factorial.
- Fig. 2: full R1A age curves and the five-point Object-126 characterization, with historical anchors visually distinguished from later probes.
- Fig. 3: compact diagnostic accounting centered on the dispersion/behavior ordering reversal.
- Fig. 4: Track-A success versus log policy-query rate with fixed-cadence pair annotations and a suite-effect inset.

Only compile-ready specification boxes and captions are present. No final paper-facing artwork was created.

## Citations

No bibliography entries were added or fabricated. Related Work uses only the verified entries already present in `paper/icra2027/references.bib`. There are no citation TODOs in this rewrite.

## Prose and layout audit

- No internal governance status labels appear in ordinary manuscript prose.
- No unsupported positive causal mechanism, rotation-is-free claim, generic stale-gripper benefit, lag optimum, equal-compute claim, global component-schedule superiority, or cross-policy generalization remains.
- No em dash appears in the Abstract; no prose paragraph exceeds the one-em-dash limit.
- The compiled log contains **no overfull boxes and no undefined citations or references**.
- Remaining layout warnings are underfull boxes, concentrated in the narrow provenance table and number-dense Results paragraphs. They do not overflow margins. Final figure artwork will change pagination and should trigger a new layout pass.
- Tectonic compiled through XeTeX and substituted Latin Modern for unavailable `TU/ptm` Times font shapes. The source is build-clean, but the 6-page count should be rechecked with the intended PaperCept/pdflatex environment because font metrics and final artwork can change pagination.
