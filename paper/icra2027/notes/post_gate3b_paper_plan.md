# Post-Gate-3B ICRA 2027 paper plan

Status: paper infrastructure frozen on 2026-08-24 while Gate-3B remains blinded.
This is a planning document, not a historical preregistration and not a result
report.  The existing `main.tex` and section files are obsolete prose.

## Evidence boundary

The current paper question is:

> What happens when components of a jointly predicted robot action chunk are
> refreshed from different temporal source generations?

Confirmed evidence may support the following chain.

1. Execution choices materially affect closed-loop behavior on the frozen
   ACT/LIBERO system.  Task 0 changes from 29/50 to 45/50 successes across the
   evaluated global horizons, with a nonmonotonic curve.
2. A prior matched-query selective-retention rule that independently retained
   arm and gripper sources reduced success by 0.26, 0.20, and 0.29 at query
   periods 4, 8, and 16.  This observation motivates a controlled composition
   test but does not identify its mechanism.
3. Gate-3A1 and Gate-3A2 establish that coherent full-action temporal-source
   weighting is operationally meaningful for this frozen system.  These gates
   are controls and prior-art-aware evidence, not a novel method contribution.
4. The RTX 5080 audit proves that the frozen additive teacher-forced metric is
   structurally blind to the symmetric arm--gripper source interaction.  The
   identity holds target-by-target, and the measured residual is numerical
   roundoff rather than an underpowered statistical null.
5. Gate-3B alone will determine whether the mathematically invisible
   interaction changes closed-loop success.  No direction is assumed here.

The authoritative local sources are the verified fact sheet, the Gate-3A1 and
Gate-3A2 reports, and the RTX offline report.  The Gate-3B protocol was read
without reading outcomes from commit `510908e` on
`origin/exp/gate3b-cross-generation-composition`.

## North Star and claim architecture

**Broad problem.** A chunk policy predicts a joint sequence, but deployment may
refresh only part of the current action from a newer observation.

**Central question.** Does cross-generation arm--gripper recomposition create a
closed-loop interaction that a separable teacher-forced action loss cannot
represent?

**Contribution type.** Controlled empirical finding plus a measurement-limit
result.  This paper does not introduce a new policy or execution method.

**Confirmed main claim before Gate-3B.** Any offline loss that adds independent
arm and gripper terms has exactly zero symmetric 2x2 composition contrast.

**Conditional main claim after Gate-3B.** The highest permitted claim is chosen
by the frozen Gate-3B decision rule.  It must remain specific to one frozen ACT
checkpoint, LIBERO Object, the arm/gripper partition, and a 20-tick source gap.

```text
Joint prediction and component refresh
|-- Execution is consequential
|   `-- Task-0 horizon curve and Gate-3A2 full-action control
|-- Independent retention can be harmful
|   `-- Historical matched-query selective-retention experiment
|-- Additive offline loss cannot encode the symmetric interaction
|   `-- Formal identity plus RTX 5080 targetwise replication
`-- Does the hidden interaction affect closed-loop success?
    `-- Gate-3B 2x2 matched-query experiment: outcome pending
```

Canonical terms are **source generation**, **source-coherent action**,
**cross-generation recomposition**, **component refresh**, and **coherence
contrast**.  Do not rotate these into “off-manifold,” “multi-clock,” or a
universal “incoherence” claim.

## Conditional manuscript versions

### Version A: Gate-3B positive or suggestive enough

Working story: **Jointly Predicted, Separately Refreshed.**

The Introduction should establish that action chunks jointly predict
heterogeneous commands, while most audited adaptive-execution methods ultimately
make one full-action temporal decision.  Independent component refresh is a
natural extension, but it can recombine components generated under different
observations.  The paper then moves from historical motivation to the controlled
2x2 intervention.  The offline identity explains why a common teacher-forced
metric cannot answer the compositionality question.  Gate-3B supplies the only
closed-loop test of whether that missing interaction matters.

If Gate-3B is `COMPOSITION-HARM-CONFIRMED`, the practical implication may state:
component-specific reliability signals should not be converted directly into
independent component refresh decisions on this evaluated system.  If Gate-3B
is only `COMPOSITION-HARM-SUGGESTIVE`, the title and abstract must avoid
“reduces success” and describe a suggestive interaction with the exact frozen
uncertainty qualifiers.

Recommended Results order:

1. Execution decisions alter closed-loop behavior.
2. Independent retention motivates an isolated source-composition test.
3. Separable offline metrics cancel the symmetric composition interaction.
4. Gate-3B tests that interaction under matched query cadence.
5. Secondary action diagnostics remain descriptive and cannot identify the
   mechanism.

Do not invent a consistency constraint, refresh method, or deployment heuristic.
The minimum contribution is the controlled result and its measurement
consequence.

### Version B: Gate-3B null

Working story: **the limits of offline reliability targets for execution
decisions.**  The paper remains finding-led and makes no claim that
cross-generation composition is harmful.  The evidence chain becomes:
execution is sensitive; intuitive offline reliability targets are fragile;
full-action temporal weighting can affect control; and the additive offline
metric cannot measure the symmetric component-source interaction.  Gate-3B
then places a direct null boundary on the proposed interaction for this system
and age.

The strongest defensible result would be methodological caution: component-wise
offline improvements under a separable metric cannot establish a joint
closed-loop benefit, because the corresponding 2x2 interaction is absent from
the metric by construction.  The Gate-3B null must be reported as evidence
against making cross-generation coherence the central mechanism, not explained
away through secondary diagnostics.

Claims that become too weak under Version B and must be deleted are:

- cross-generation recomposition reduces task success;
- execution coherence is a supported control mechanism;
- the historical selective-retention loss is explained by source mixing;
- independent component refresh should generally be avoided;
- a coherence-preserving execution method is justified by current evidence.

Version B should move the detailed Gate-2B phase grids, `Y_refresh` estimator
architecture, semantic-kernel development, PACE audits, and abandoned candidate
methods to project history or omit them entirely.

## Title shortlist

1. **Jointly Predicted, Separately Refreshed: Cross-Generation Composition in Robot Action Chunks** — recommended for Version A.
2. **Cross-Generation Composition of Joint Robot Actions**
3. **Joint Action Chunks Under Independent Component Refresh**
4. **From Joint Prediction to Mixed-Generation Execution**
5. **Component Refresh Across Temporal Generations of Robot Action Chunks**
6. **Execution Coherence Under Component-Wise Action Refresh**
7. **Cross-Generation Robot Actions: Offline Separability and Closed-Loop Interaction**
8. **Joint Prediction, Component Refresh, and the Limits of Separable Action Metrics** — recommended for Version B.
9. **Separable Action Metrics for Cross-Generation Component Refresh**
10. **Temporal Source Composition in Joint Robot Actions**

Titles 1 and 6 require Gate-3B wording calibration if the result is only
suggestive.  Titles 7--10 remain compatible with a null because they name the
question or measurement boundary rather than asserting harm.

## Four-figure backbone

### Figure 1: One joint prediction, four source compositions

**Single claim:** The same fresh and old joint predictions define two
source-coherent actions and two cross-generation recompositions while preserving
the marginal assignment of source ages.

- **Panel A, methodological bridge:** draw the fresh chunk `F_t` and the chunk
  queried 20 controller ticks earlier, `O_t`, both pointing to physical action
  time `t`.
- **Panel B, definition:** show `FF`, `OO`, `FO`, and `OF` as arm/gripper tiles.
  Use one color family for source `F` and another for source `O`.
- **Panel C, design consequence:** brace `FF/OO` as source-coherent and `FO/OF`
  as cross-generation, while showing equal fresh/old arm and gripper margins.

The caption must define source generation and the common fresh prefix.  It must
not call mixed actions off-manifold or imply that they are harmful.

### Figure 2: Historical evidence motivates, but does not prove, composition harm

**Single claim:** Prior closed-loop results establish execution sensitivity and
motivate isolating component-source composition.

- **Panel A, motivation:** task-0 global horizon success for horizons
  1, 2, 4, 8, and 16: 29, 31, 42, 45, and 42 successes out of 50.  Show query
  rate on a secondary aligned strip or in annotations, not a misleading dual
  axis.
- **Panel B, failure mode:** matched-query selective retention minus global
  replacement at q=4, 8, and 16: -0.26, -0.20, and -0.29, with paired-episode
  confidence intervals.
- **Chronology:** label Panel A as exploratory motivation and Panel B as the
  later direct negative observation.  A caption sentence must state that neither
  isolates cross-generation composition as the mechanism.

Gate-2B phase maps and the retrospective cross-task configuration oracle do not
belong in this main figure.

### Figure 3: The offline interaction is structurally unidentifiable

**Single claim:** A separable arm-plus-gripper loss has exactly zero symmetric
composition interaction, independent of sample size.

- **Panel A, definition:** show the additive loss as two tiles,
  `L_arm + L_grip`, and expand all four cells.
- **Panel B, algebraic anchor:** show
  `L(FF)+L(OO)-L(FO)-L(OF)=0` by cancellation.
- **Panel C, replication:** report the four RTX 5080 `L_sem` values and
  `C_offline=-1.49e-17`, with the maximum targetwise residual `1.78e-15`.
  Label these as floating-point residuals around an exact identity.
- **Panel D, task check:** show all ten task contrasts at zero under the frozen
  `1e-12` tolerance.

The figure title and caption must state “structural identity,” not “no effect”
or “insufficient power.”

### Figure 4: Reserved Gate-3B result

**Single claim:** To be written only after applying the frozen Gate-3B decision
rule to the complete validated experiment.

- **Panel A, primary outcomes:** four success rates for `FF`, `OO`, `FO`, `OF`.
- **Panel B, anchor:** `C_coherence` with paired-state and task-cluster 95%
  bootstrap intervals; show zero as the null reference.
- **Panel C, sensitivity:** all ten task-wise coherence contrasts, preserving
  task order 0--9.

`figures/gate3b_figure4_interface.json` contains only placeholders, and
`figures/plot_gate3b_reserved.py` refuses to render until it receives numeric
final data and all ten task values.  No secondary diagnostic may replace Panel
B or rescue a null primary result.

## Fill-ready table plan

The manuscript-ready LaTeX blocks are in
`notes/post_gate3b_manuscript_blocks.tex`.

**Table 1, system and experiment contract.** Pre-fill the frozen ACT hash,
100x7 chunk, 20 Hz action contract, 20-tick source gap, LIBERO Object tasks,
100 paired blocks, randomized four-condition design, and one query per surviving
step.  Separate registered counts from final validation status.

**Table 2, Gate-3B primary result.** Reserve the four exact success placeholders,
the coherence point estimate, and its bootstrap intervals.  Pairwise comparisons
remain secondary and need not enter the main table.

**Table 3, relevant controls and prior evidence.** Keep only task-0 horizon
sensitivity, the matched-query selective-retention negative result, Gate-3A1/3A2
full-action controls, and the offline separability identity.  Do not turn this
table into a chronological inventory of failed gates.

## Related-work rewrite outline

### 1. Action chunking and temporal ensembling

- ACT establishes joint action-chunk prediction and optional exponential
  temporal ensembling over overlapping full-action predictions.
- CogACT similarity-weights current and historical predictions with one scalar
  per complete action.  It is prior art for adaptive temporal ensembling.
- Lazzati et al. analyze delayed temporal relationships and implicit ensembling
  as mechanisms behind action chunking.  Treat their conclusions as
  paper-reported and bounded to their evaluated settings.

### 2. Adaptive execution and replanning

- SGAC uses full-action similarity for retain/replace behavior.
- AutoHorizon, AAC, A3, PACE, DVAC, DEHP, and BCP select a scalar full-action
  prefix, boundary, or continue/replan decision using attention, component
  entropy, consensus, kinematics, denoising variation, or learned value.
- AAC and PACE are important qualifications: their signals use action
  components or arms, but the audited final decision remains synchronized.
- These methods are baselines and context for the decision granularity.  They
  are not targets to rename or claim as absent.

### 3. Stale-chunk correction and consistency

- RTC inpaints around actions committed during asynchronous inference.
- REMAC uses masked or prefix-conditioned generation for continuation.
- SEAM uses the previous unexecuted tail as a consistency reference.
- A2C2 corrects base actions from the newest observation at control rate.
- These works prevent a broad novelty claim around “consistency.”  The present
  question concerns the narrower 2x2 recomposition of arm and gripper components
  from two fixed source generations.

### 4. Bounded distinction

Candidate sentence for later verification:

> In a bounded audit of primary sources, we did not locate a controlled
> evaluation that isolates component-level cross-generation source
> recomposition while holding the marginal arm and gripper source ages fixed.

\TODO{FIRSTNESS CHECK: repeat the component-refresh and cross-generation search
immediately before submission, verify every cited paper's current version and
venue status, and either retain the bounded sentence above or weaken it.}

Every “we are not aware,” “no prior work,” “first,” “previously untested,” or
“unique” sentence is firstness-like and remains blocked by this TODO.  The safer
distinction does not require firstness: this study isolates a component-source
interaction that is absent from separable teacher-forced action metrics.

## Formal method and analysis contract

The exact manuscript-ready definitions and proof are in
`notes/post_gate3b_manuscript_blocks.tex`.  In brief, `F_t=E_{t,t}` and
`O_t=E_{t,t-20}` define fresh and old predictions for the same physical action.
The four deterministic compositions are `FF`, `OO`, `FO`, and `OF`.  For binary
success in a paired task-state block,

`C_coherence = 0.5(success_FF + success_OO) - 0.5(success_FO + success_OF)`.

For any target and separable loss
`L(a_arm,a_grip)=L_arm(a_arm)+L_grip(a_grip)`, the four terms contain each arm
and gripper marginal exactly once on each side.  Therefore
`L(FF)+L(OO)-L(FO)-L(OF)=0`.  This identity does not prove that mixed actions
help, harm, or behave equivalently in closed loop.  It proves only that this
class of additive offline metrics has no term capable of measuring the
symmetric 2x2 interaction.

## Gate-3B replacement checklist

Do not edit the narrative before the complete validated Gate-3B report exists.
Then perform these mechanical replacements once:

- `<GATE3B_SUCCESS_FF>`
- `<GATE3B_SUCCESS_OO>`
- `<GATE3B_SUCCESS_FO>`
- `<GATE3B_SUCCESS_OF>`
- `<GATE3B_COHERENCE>`
- `<GATE3B_CI>`

Also populate the empty ten-task vector in the Figure 4 interface, record final
episode/query validation and provenance, apply the frozen decision label, and
choose Version A or B.  Search the complete paper tree for every placeholder
before building.  Do not use secondary action diagnostics to change the branch.

## Selective inclusion and appendix boundary

Keep in the main paper: task-0 horizon sensitivity as motivation; the
matched-query selective-retention loss as motivating observation; Gate-3A1 and
Gate-3A2 only as full-action controls; the offline separability identity; and
Gate-3B.

Move to appendix only if needed for reviewer questions: detailed Gate-3A1
baseline rows, Gate-3A2 secondary action diagnostics, full task-wise historical
tables, and additional offline component values.

Omit from the submission narrative: Gate-2B phase grids, `Y_refresh`
architecture details, semantic-kernel development history, PACE reproduction
work, and abandoned method families.
