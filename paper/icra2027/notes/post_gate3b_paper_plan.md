# Post-Gate-3B / pre-Gate-3C ICRA 2027 paper plan

Status: manuscript infrastructure revised on 2026-08-24 after the complete
Gate-3B result. Gate-3C remains blinded and pending. This is a paper-planning
artifact, not a protocol or a historical preregistration. The historical
`main.tex` and section files remain obsolete prose.

## Evidence boundary and story pivot

Gate-3B did not confirm the preregistered generic coherence hypothesis. Its
coherence contrast was `+.025`, with paired-state 95% CI `[-.030,+.085]` and
task-cluster CI `[-.005,+.055]`. The paper must not claim that cross-generation
composition is generally harmful.

The completed 2x2 cells instead show a strong directional pattern:

| | Fresh gripper | Old20 gripper |
|---|---:|---:|
| Fresh arm | FF: 44% | FO: 62% |
| Old20 arm | OF: 17% | OO: 40% |

The fresh-arm marginal is 53.0%, compared with 28.5% for the old arm. The
old-gripper marginal is 51.0%, compared with 30.5% for the fresh gripper. These
main effects, all pairwise comparisons centered on `FO`, and the interpretation
of `FO` as the preferred source assignment are post-hoc. They remain candidate
findings until the untouched-state Gate-3C confirmation.

The provisional paper question is:

> Can heterogeneous components of a jointly predicted action chunk prefer
> different temporal source generations?

The candidate thesis is that temporal-source utility need not be shared across
action components. In the frozen Gate-3B sample, fresh arm predictions and
old20 gripper predictions were associated with the highest observed success.
The frozen offline audit provides the complementary measurement result. Its
teacher-forced losses favored old sources for both arm and gripper, so the arm
source preference did not transfer to closed-loop success.

This paper does not introduce a new policy, a dynamic horizon, or a learned
selector. If Gate-3C confirms the pattern, `FO20` becomes a simple training-free
executor. Until then, it is an exploratory cell selected after Gate-3B.

## Claim and evidence architecture

```text
Overlapping action chunks predict the same physical action from different times
|-- Published methods usually make one full-action temporal decision
|-- Heterogeneous components may need different temporal information
|   `-- Gate-3B: exploratory fresh-arm / old-gripper directional pattern
|-- Teacher-forced delayed prediction quality is not closed-loop utility
|   `-- Offline old-arm preference versus Gate-3B fresh-arm preference
`-- Does the directional pattern transfer to untouched states and baselines?
    `-- Gate-3C: pending, blinded confirmation
```

Confirmed claims available now are deliberately narrow.

1. Gate-3B did not confirm generic composition harm.
2. Gate-3B displayed a post-hoc directional source-age asymmetry: `FO=.62`,
   `FF=.44`, `OO=.40`, and `OF=.17`.
3. The post-hoc fresh-arm and old-gripper marginal effects were `+.245` and
   `+.205`. Their paired and task-cluster intervals exclude zero, but these
   intervals are exploratory rather than confirmatory.
4. `FO>=FF` on all ten tasks, with eight improvements and two ties. `FO>OO` on
   all ten tasks.
5. The frozen offline losses favored the old source for arm and gripper, while
   the Gate-3B marginal pattern favored a fresh arm and old gripper.
6. Any additive arm-plus-gripper loss remains structurally incapable of
   measuring the symmetric 2x2 interaction. That identity is a measurement
   limitation, not proof of behavioral harm.

Claims about replication, superiority on untouched states, a practical
executor, and general component-specific preferences remain conditional on
Gate-3C.

## Introduction arc

The final Introduction should use full paragraphs and follow this sequence.

1. Action chunking creates multiple predictions for the same future physical
   action, conditioned on observations from different source times.
2. Temporal ensembling and adaptive-execution methods usually resolve this
   redundancy through one temporal decision for the complete action.
3. Robot actions are heterogeneous. Continuous motion and discrete interaction
   commands may value current feedback and retained temporal context
   differently.
4. A historical independent-retention failure was not mechanistically
   diagnostic, so Gate-3B directly assigned fresh or old20 generations to arm
   and gripper under a matched-query 2x2 intervention.
5. Gate-3B did not confirm generic coherence harm, but it revealed an
   exploratory asymmetry: fresh arm plus old20 gripper was the highest cell.
6. The frozen teacher-forced audit favored old sources for both components and
   therefore did not reproduce the fresh-arm closed-loop preference.
7. Gate-3C supplies the untouched-state confirmation against `FF`, `OO20`,
   age-exponential, and CogACT baselines. No result is stated until its final
   validated report exists.

## Conditional contribution statements

### If Gate-3C confirms the directional pattern

The abstract, Introduction, and Discussion may present four contributions:

1. controlled evidence that temporal-source age need not be shared across
   heterogeneous components of a jointly predicted action chunk;
2. identification of an asymmetric source assignment in which fresh arm
   feedback and older gripper context outperform synchronized temporal-source
   execution in the evaluated system;
3. evidence that separable teacher-forced prediction metrics can disagree with
   closed-loop temporal-source utility, especially for arm motion; and
4. a training-free asymmetric temporal-reuse executor evaluated against
   newest, full-old, age-exponential, and CogACT baselines.

Even after confirmation, these claims remain bounded to the frozen ACT policy,
LIBERO Object, the arm/gripper partition, and the 20-tick source gap. The paper
must not claim universal component incoherence or optimality over source age.

### If Gate-3C does not confirm the directional pattern

Contributions 1, 2, and 4 become too weak and must be removed. The manuscript
becomes a finding-led analysis of three limits: execution sensitivity, the
failure of intuitive offline reliability targets to specify execution choices,
and the inability of separable teacher-forced losses to encode the symmetric
component interaction. Gate-3B's directional pattern remains explicitly
exploratory, and Gate-3C becomes the direct failed confirmation.

The surviving conclusion is narrower: teacher-forced component accuracy does
not establish the closed-loop value of a temporal-source assignment. A failed
Gate-3C would also rule out presenting `FO20` as a method, practical executor,
or reliable component preference.

## Title shortlist

1. **Fresh Feedback, Retained Intent: Asymmetric Temporal Reuse for Action-Chunked Robot Policies**
2. **Asymmetric Temporal Reuse for Action-Chunked Robot Policies**
3. **Temporal Source Asymmetry in Joint Robot Action Chunks**
4. **Group-Specific Temporal Sources for Action-Chunked Robot Policies**
5. **Heterogeneous Temporal Source Age in Joint Robot Actions**
6. **Action-Component Temporal Reuse in Chunked Robot Control**
7. **Joint Prediction with Component-Specific Temporal Sources**
8. **Fresh Motion, Retained Interaction Intent in Robot Action Chunks**
9. **From Delayed Prediction Accuracy to Closed-Loop Temporal Utility**
10. **Teacher-Forced Accuracy and Closed-Loop Temporal Source Utility**
11. **One Temporal Source Does Not Fit All: Asymmetric Temporal Reuse for Action-Chunked Robot Policies**

Title 1 is the recommended confirmation title. “Retained intent” is evocative,
but the body must define it operationally as executing the old20 gripper
prediction. It must not be presented as a measured latent mechanism. Title 10
is the recommended fallback if Gate-3C fails.

Title 11 is not recommended before confirmation. “Does not fit all” sounds
universal and can be mistaken for a query-frequency or execution-horizon claim.
The intervention changes source generation while holding one policy query per
surviving controller step. Titles 3, 5, and 10 remain safe while Gate-3C is
pending.

## Four-figure backbone

### Figure 1: Two source generations, four component assignments

**Claim:** Fresh and old chunks predict the same physical action time and define
four deterministic arm/gripper source assignments.

Panel A shows `F_t=E_{t,t}` and `O_t=E_{t,t-20}` both pointing to physical time
`t`. Panel B forms `FF`, `OO`, `FO`, and `OF` from arm and gripper tiles. Panel C
marks `FO` as the highest observed Gate-3B cell using an “exploratory” badge.
It must not call `FO` an established method or call mixed actions off-manifold.

### Figure 2: Gate-3B directional 2x2 pattern

**Claim:** Gate-3B did not confirm generic coherence harm but displayed a large
post-hoc directional source-age asymmetry.

Panel A is a 2x2 success matrix with arm source on rows and gripper source on
columns: `[[.44,.62],[.17,.40]]` in the order fresh arm then old arm, fresh
gripper then old gripper. Use exact successes out of 100 in the annotations.
Panel B shows the fresh-arm main effect `+.245` and old-gripper main effect
`+.205`, with paired-state and task-cluster exploratory intervals. Panel C may
show `FO-FF` and `FO-OO` task values or sign counts. Every directional panel is
marked post-hoc. The caption opens with the unresolved preregistered coherence
contrast to prevent secondary-result substitution.

### Figure 3: Offline accuracy and closed-loop utility prefer different arm sources

**Claim:** The teacher-forced component metric and Gate-3B marginal success do
not select the same arm source.

Panel A shows offline source preference. Both arm terms favor old20 because
translation loss is `.50667` versus `.59578` and normalized rotation loss is
`1.09877` versus `1.12962`. Gripper sign error also favors old20, `.27400`
versus `.30760`. Panel B shows the Gate-3B marginal preferences: fresh arm,
`.530` versus `.285`, and old gripper, `.510` versus `.305`. Panel C aligns the
two decisions as a simple old/fresh source table. It must not plot incomparable
losses and success rates on one quantitative axis or imply causation.

### Figure 4: Reserved Gate-3C confirmation

**Claim:** unwritten until the complete validated Gate-3C report exists.

Panel A will show success for `FO20`, `FF`, `OO20`, age-exponential, and CogACT
on Gate-3C's new states. Panel B will show the frozen primary contrast or
contrasts with confidence intervals. Panel C will show task-wise effects in the
registered order. The plotting interface rejects nonnumeric placeholders and
incomplete task arrays. No Gate-3B value may be copied into Figure 4.

## Fill-ready table plan

**Table 1, system and experiment contract.** Include the frozen ACT policy,
100x7 chunk, 20 Hz controller contract, 20-tick source gap, arm/gripper split,
paired task-state design, and one policy query per surviving step. Separate the
exploratory Gate-3B cohort from the untouched Gate-3C cohort.

**Table 2, Gate-3B directional result and Gate-3C confirmation.** The Gate-3B
subtable contains all four observed cells, the unresolved preregistered
coherence contrast, and the two post-hoc main effects. The Gate-3C subtable
remains placeholder-only until final validation.

**Table 3, selected controls.** Keep only the task-0 horizon sensitivity,
matched-query independent-retention failure, Gate-3A1/3A2 full-action control,
the additive-loss identity, and the offline/closed-loop arm-source mismatch.
Gate-2B phase grids, the refresh-model architecture, semantic-kernel history,
PACE audits, and abandoned methods stay outside the main paper.

## Related-work architecture

### Action chunking and full-action temporal decisions

ACT predicts joint action chunks and temporally ensembles overlapping complete
action predictions. CogACT reweights complete historical action predictions
using one similarity score per source. Delayed-action analysis by Lazzati et
al. explains how older observation-conditioned predictions can match expert
behavior under non-Markovian demonstrations. This literature supports the
possibility that old predictions can help. It does not establish that the same
source age is useful for every component or that delayed teacher-forced quality
selects the best closed-loop source.

### Adaptive execution and replanning

AAC estimates uncertainty separately for translation, rotation, and gripper,
but aggregates those estimates into one executed chunk prefix. TAS selects one
complete cached action candidate. AutoHorizon and PACE choose one global
execution boundary. These works motivate temporal adaptation but do not answer
the present component-source factorial question.

### Stale-action and chunk-consistency methods

RTC, REMAC, A2C2, and SEAM repair or align stale full-action chunks. SEAM can
apply overlap guidance to a dimension subset, which is the closest technical
nuance in the bounded set. Its reported procedure does not independently
assign arm and gripper predictions from different source observations for a
controlled 2x2 execution test.

### Bounded distinction

Use only the following firstness-like sentence, and preserve its qualification:

> We are not aware of prior controlled evaluations that independently assign
> temporal source generations to heterogeneous action components.

This sentence must be rechecked before submission. Never shorten it to “we are
the first.” The contribution is not that old predictions can help. The
candidate distinction is that temporal-source utility may differ across
heterogeneous components, while teacher-forced delayed prediction quality may
not identify the source assignment that maximizes closed-loop control.

## Gate-3C fill contract

Only the following placeholders may carry Gate-3C-dependent information:

- `<GATE3C_STATUS>`
- `<GATE3C_SUCCESS_FO20>`
- `<GATE3C_SUCCESS_FF>`
- `<GATE3C_SUCCESS_OO20>`
- `<GATE3C_SUCCESS_AGE_EXP>`
- `<GATE3C_SUCCESS_COGACT>`
- `<GATE3C_PRIMARY_CONTRASTS>`
- `<GATE3C_CI>`
- `<GATE3C_PER_TASK>`
- `<GATE3C_FINAL_REPORT>`
- `<GATE3C_EXPERIMENT_CONTRACT>`

After Gate-3C, first validate the final report and replace these placeholders.
Then choose the confirmed or null contribution branch, revise the title and
abstract together, and run a claim-to-evidence audit. No Gate-3C-dependent
claim may be inferred from partial logs or from Gate-3B.
