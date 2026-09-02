# One-clock figure specifications

Status: specification only. **No paper-facing artwork is produced or authorized
by this file.** All scientific plots, schematics, and final exports will be made
in a later human/co-PI figure-design session after the relevant analyses finish.

## Shared production contract

The repository's actual ICRA source uses
`\documentclass[letterpaper, 10 pt, conference]{ieeeconf}`. The checked-in class
sets `\textwidth=7.0in` and `\columnsep=0.2in`, giving a current single-column
width of 3.4in and double-column width of 7.0in. The later production session
must re-read the final template before export; these values are template-derived,
not generic IEEE assumptions.

- Use the final manuscript's actual typography and vector output for plots.
- Use a white/light background and a restrained, semantic encoding held fixed
  across all figures.
- Distinguish arm, gripper, and coherent execution with color plus line style,
  marker, fill, or direct label; color cannot be the only discriminator.
- Point estimates are visually primary and uncertainty secondary but visible.
- Put full discordances, p-values, audit statistics, and provenance detail in
  captions or tables rather than crowding panels.
- Never connect fixed-source diagnostic probes to periodic executor conditions
  as though they form one dose-response family.
- Never render the +32.14 pp diagonal as an additive arm or gripper percentage.
- Keep these quantities distinct in labels and captions:
  1. demonstration action temporal persistence;
  2. frozen-policy future-action forecast error;
  3. same-target cross-source prediction disagreement.
- Historical aliases FO20/Reverse20 may appear parenthetically once; primary
  notation is A0G20/A20G0.

## Figure 1 — Same-target component assignment in miniature

### Scientific question and one-sentence job

How can two action components for the same physical target come from different
policy queries, and what empirical asymmetry motivates studying that choice?

The figure's job is to compress the paper into:
`same-target problem -> component source assignment -> robust diagonal
observation`, without implying a deployment recommendation.

### Format and hierarchy

- Target: double-column width (currently 7.0in from the checked-in template).
- Suggested structure: three left-to-right conceptual/evidence modules, not a
  required graphical arrangement.
- Anchor: the component-source assignment at physical target `t`; the reader
  should understand it before seeing the success matrix.

### Panels/modules

**A. Same-target identity — methodological bridge**

- Show source query `q`, chunk offset `k`, source age `d=t-q`, and target `t`.
- State the identity `q+k=t`; under natural fixed-source execution, `k=d`.
- Show that `t<d` uses the exact Fresh prefix `A_t[0]`.
- Define a policy call separately from an executed environment action.

**B. Four component assignments — definition**

- A0G0: fresh arm, fresh gripper.
- A0G20: fresh arm, source-age-20 gripper.
- A20G0: source-age-20 arm, fresh gripper.
- A20G20: source-age-20 arm and gripper.
- Show arm dimensions 0--5 and gripper dimension 6.
- Annotate `policy-query rate = 1` for all four diagnostic cells.
- Explicitly label A0G20 and A20G0 as diagnostic probes, not deployment
  recommendations.

**C. Frozen 140-block observation — claim-supporting evidence**

- Minimal 2x2 absolute-success matrix: A0G0 77/140, A0G20 83/140,
  A20G0 38/140, A20G20 66/140.
- Highlight only the robust diagonal A0G20-A20G0 = +32.14 pp.
- Include a compact qualifier that conditional simple effects are weaker.

### Data source

`experiments/cross_suite_confirmation/{protocol.json,analysis.json}` and the
canonical interaction metadata at
`experiments/icra27_crosssuite_query_allocation/interaction_robustness/analysis.json`.

### Visual emphasis and prohibited implication

Emphasize the equality of physical target and the difference in component
source. Do not make A0G20 look like a named method, do not show it as generally
optimal, and do not attribute the diagonal contrast uniquely to arm freshness
or gripper commitment.

### Candidate caption points

- Define `A_q[k]`, `q+k=t`, and the Fresh prefix.
- State that all four probes query once per environment step.
- Give the 140-block absolute counts and identify the diagonal as the frozen
  primary contrast.
- State that the four-cell conditional dependence prevents additive component
  attribution.

### Matching Results topic sentence

“Predictions for the same physical target show a large component-assignment
asymmetry, but the conditional simple effects do not support a unique additive
arm or gripper attribution.”

## Figure 2 — Frozen factorial: robust diagonal, conditional simple effects

### Scientific question and one-sentence job

Which contrasts in the frozen 2x2 component-source table are stable across
paired states and task clusters?

The figure should establish a robust diagonal assignment asymmetry while making
the uncertainty of the two attribution-relevant simple effects impossible to
miss.

### Format and hierarchy

- Target: single-column width (currently 3.4in) if legible; otherwise
  double-column width. Decide from a print-size proof, not convenience.
- Two panels.
- Anchor: Panel B's A0G20-A20G0 forest row.

### Panels

**A. Absolute success matrix — context**

- 2x2 grid with arm source age on rows `{0,20}` and gripper source age on
  columns `{0,20}`.
- Cell values: 77/140, 83/140, 38/140, 66/140 with percentages secondary.
- Use no heatmap scale that visually implies linear/additive main effects.

**B. Paired contrast forest — claim-supporting evidence and limitation**

Required rows, in this order:

1. A0G20-A20G0: +32.14 pp; paired CI [+23.6,+40.7]; task-cluster CI
   [+21.4,+44.3]. Label as frozen primary diagonal.
2. A0G20-A0G0: +4.29 pp; paired CI [-1.4,+10.0]; task-cluster CI
   [-1.4,+10.7].
3. A0G20-A20G20: +12.14 pp; paired CI [+3.6,+20.7]; task-cluster CI
   [-2.1,+28.6].
4. Other simple effects may appear only if all are shown symmetrically and the
   panel remains legible; otherwise place them in a supplement table.

- Draw paired and task-cluster intervals with distinct glyphs and a shared point
  estimate.
- Use a zero reference line.
- Do not headline the post-hoc difference-in-differences.

### Axes and uncertainty

- X: success difference (percentage points), linear scale.
- Y: explicitly written contrast notation, not internal experiment codes.
- Caption defines paired resampling over task-state blocks and task-cluster
  resampling over tasks.

### Data source

`experiments/cross_suite_confirmation/analysis.json`. Interaction sensitivity,
if mentioned outside the figure, comes from `interaction_robustness/analysis.json`.

### Visual emphasis and prohibited implication

Emphasize the diagonal row and the zero-crossing cluster interval for
A0G20-A20G20. Do not use separate “arm contribution” and “gripper contribution”
areas, brackets, percentages, or causal labels.

### Candidate caption points

- What each matrix cell changes.
- Absolute counts and N=140 paired task-state blocks across ten task-specific
  checkpoints.
- Meaning of the two interval glyphs.
- Takeaway: stable diagonal, conditional and uncertain simple effects.
- Limitation: one trained checkpoint/training seed per task.

### Matching Results topic sentence

“The diagonal source assignment generalized across tasks, whereas the simple
effects needed for a unique component attribution remained conditional and
uncertain.”

## Figure 3 — Temporal scale and three mechanism measurements

### Scientific question and one-sentence job

At what timescale do action components become behaviorally sensitive, and do
demonstration persistence, future-action forecast error, or cross-source
disagreement track that sensitivity?

This should be the key mechanism figure. It aligns four measurements for
comparison without treating them as the same metric or as causal proof.

### Availability gate

Do not finalize this figure until the reviewer-supplement R1.1 curve and frozen
B3 forecast analysis are complete. A missing or negative mechanism analysis is
shown honestly; it is not replaced with a new metric.

### Format and hierarchy

- Target: double-column width (currently 7.0in).
- Candidate layout: four aligned panels A--D; final arrangement follows
  legibility and argument, not a reference paper's layout.
- Anchor: behavioral sensitivity in Panel A, with B--D as rival explanatory
  measurements.

### Panels

**A. Behavioral fixed-source sensitivity — claim-supporting evidence**

- X: source age `d` in control steps using exactly
  `{0,2,4,8,12,16,20,32}`.
- `d=0` appears once as the common Fresh anchor, not as duplicated arm/gripper
  points.
- Series: A_dG0 and A0G_d for positive d.
- Y: success rate.
- Show paired/task-cluster uncertainty for planned contrasts in a visually
  secondary form; do not imply independent binomial samples.
- Visually mark d=20 as the historical frozen anchor without giving it greater
  statistical status within the development curve.

**B. Demonstration action temporal persistence — explanatory evidence**

- X: demonstration lag in control steps; include seconds only after using the
  dataset's own recorded rate.
- Y: either autocorrelation or normalized action difference, with the exact
  metric named. If both are necessary, use separate small subpanels rather than
  a mixed scale.
- Series: translation dimensions 0--2, rotation 3--5, gripper 6.
- State that the audited data are training demonstrations, not held out.

**C. Frozen-policy future-action forecast error — explanatory evidence**

- X: chunk offset `k`, exactly the frozen analysis offsets.
- Y: normalized reference-action error for translation/rotation; gripper uses
  its separately defined continuous error and/or sign disagreement.
- Series: translation, rotation, gripper with group-specific metric names.
- Do not label this panel “same-target disagreement.” Its reference is a
  demonstration action at `t+k`.

**D. Same-target cross-source disagreement — mechanism test and boundary**

- X: source age 0--15 for the frozen primary window.
- Y: normalized dispersion across source predictions for the same physical
  target.
- Series: translation, rotation, gripper; ACT primary, with SmolVLA either
  clearly separated or moved to a supplement if overlay is cluttered.
- Include a compact ACT localization result: R_ACT=0.540,
  episode-cluster CI [0.397,0.703], `ACT_LOCALIZATION_PASS=no`.
- Optional inset/companion: gripper sign disagreement against distance/margin to
  the decision boundary, using the frozen terciles only.

### Data sources

- Panel A: future tidy output from the frozen reviewer-supplement R1.1 manifest.
- Panel B: `track_b/demonstration_persistence/`.
- Panel C: future output from the frozen B3 forecast manifest committed at
  `94657b54591fd1305e8ac888a0c05beb4de2c2cb`.
- Panel D: `track_b/analysis_addendum/` and `track_b/analysis.json`.

### Visual emphasis and prohibited implication

Align x coordinates only when their units/rates are truly commensurate. Do not
collapse the panel titles to “prediction error,” use a shared y-axis across
different metrics, or claim causal explanation because curves have similar
shapes. The failed ACT localization criterion remains visible.

### Candidate caption points

- Define each measurement and its reference target separately.
- State exact d grid and that d=0 is a shared anchor.
- Give dataset rates when translating steps to seconds.
- Define uncertainty and cluster unit panel by panel.
- State whether temporal sensitivity tracks any mechanism descriptively and
  that correlation does not establish causality.

### Matching Results topic sentence

“Component sensitivity emerged over a measurable temporal scale, but the three
mechanism measurements placed different constraints on its interpretation.”

## Figure 4 — Practical consequence or cross-policy scope

### Decision gate

The final scientific job and panel content are **not selected while Track A is
running**. Make the choice only after all Track-A cells complete, integrity
validation passes, the preregistered analysis is final, and the available
cross-policy evidence is assessed without inventing new conditions.

### Option A: Track-A practical consequence

Use if Track A yields a scientifically meaningful matched-query result, whether
positive or a clear negative boundary.

- Question: does component-resolved periodic execution improve over a matched
  coherent horizon at the same policy-query schedule?
- Conditions: H16, H4, ARM4_GRIP32, H2, ARM2_GRIP16, TE_DENSE—exactly these six.
- X: observed policy-query rate, not compute or FLOPs.
- Y: success rate.
- Make H4↔ARM4_GRIP32 and H2↔ARM2_GRIP16 matched comparisons explicit; use
  paired/task-cluster contrast intervals in a companion panel if the rate plot
  obscures pairing.
- Show wall-clock in a separate compact table/panel, never as a hidden synonym
  for query rate.
- TE_DENSE is a standard-practice reference, not query-budget matched.
- Data source: future frozen `track_a` analysis only.

### Option B: ACT versus SmolVLA scope

Use if Track A is not the right main-figure story but the complete frozen
SmolVLA same-target factorial materially clarifies policy scope.

- Question: does the component-assignment asymmetry reproduce across policy
  families?
- Show within-policy A0G20-A20G0 and conditional simple effects with paired
  uncertainty; do not compare raw normalized dispersion across policies.
- State cohort exposure and checkpoint differences explicitly.
- A SmolVLA non-replication is shown as a boundary, not a failed rescue.

### Prohibited choice

Do not preserve a planned figure number by forcing an uninformative aggregate
into the main paper. If neither option carries a necessary main claim, omit or
demote Figure 4.

## Supplement figure specifications

### S1. Per-task factorial effects

- Question: how heterogeneous is the frozen diagonal across the ten tasks?
- Forest/strip display of per-task A0G20-A20G0, A0G20-A0G0, and
  A0G20-A20G20; separate panels or facets, same x scale.
- Show denominators and suite grouping without selecting “winning” tasks.
- Data: `cross_suite_confirmation/analysis.json`.

### S2. Full source-age curves by Object task

- Question: is aggregate temporal sensitivity broad or driven by particular
  tasks?
- Small multiples for all Object tasks 1--9; exact x grid
  `{0,2,4,8,12,16,20,32}`, with d=0 as one Fresh anchor.
- Show both A_dG0 and A0G_d; no outcome-selected ordering or task omission.

### S3. Translation versus rotation control

- Question: does the d=20 effect differ between equally 3D action subgroups?
- Conditions Fresh, T20_R0_G0, T0_R20_G0 with absolute rates and paired
  contrasts.
- State explicitly that the control does not isolate geometry or semantics.

### S4. Dense-query H16 factorial

- Question: what does the 2x2 component-temporal table show when qrate=1 in all
  cells?
- 2x2 C00/C10/C01/C11 matrix plus conditional contrasts.
- Caption distinguishes reused versus newly run cells and labels the analysis
  `POST_HOC QUERY-MATCHED EXTENSION`.

### S5. Spatial factorial completion

- Create only if exact historical reconstruction passes.
- Recovered candidate cohort: Spatial tasks 0--9, states
  `1,13,15,19,21,24,31,37,40,47`, 100 paired blocks, using the immutable
  multi-suite checkpoint revision recorded in the supplement draft.
- Show every reconstructed Spatial cell, including the historical negative
  A0G20 result and newly completed A20G0.
- Label `POST_HOC SPATIAL FACTORIAL COMPLETION`; never pool with N=140.

### S6. ACT versus SmolVLA

- Question: which component-assignment and localization findings are
  policy-specific?
- Within-policy effects/ratios only. Do not compare raw normalized dispersion
  across policy-specific normalization spaces.
- Report the failed `R_ACT-R_SMOLVLA` interval if mechanism ratios appear.

### S7. Interaction-scale sensitivity

- Question: does the post-hoc interaction's interpretation depend on effect
  scale?
- Compact table or coefficient plot for risk difference and log odds with
  paired, cluster, and small-cluster sensitivity intervals/tests.
- State canonical sign formula and `POST_HOC_SUPPORTING_INTERACTION` in the
  caption.

### S8. Query rate and wall-clock

- Prefer a table, not a dual-axis plot.
- Rows: all relevant conditions; columns: successes/N, environment steps,
  policy queries, query rate, mean/total wall-clock, mean call latency where
  identified.
- Missing historical timing remains missing, never zero.

### S9. Failure taxonomy

- Create only if passive state/contact logging makes categories identifiable.
- Show complete predefined categories and an unidentifiable/other category.
- If existing evidence is insufficient, retain the textual result
  `FAILURE_MODE_CLASSIFICATION_NOT_IDENTIFIABLE_FROM_EXISTING_ARTIFACTS` and
  create no taxonomy plot.

## Final artwork handoff checklist

Before any later rendering session:

1. freeze canonical tidy CSV/JSON and verify every row against source results;
2. re-read the final ICRA template for physical dimensions and typography;
3. choose Figure 4 only after its decision gate;
4. write the one-sentence claim and panel role on the production checklist;
5. render at final physical size and inspect labels, intervals, and grayscale;
6. export plots as vector graphics; and
7. verify that captions, Results claims, and evidence-role labels match.
