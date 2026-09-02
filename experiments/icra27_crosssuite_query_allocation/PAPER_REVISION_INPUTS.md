# One-clock paper revision inputs

Status: analysis-only source ledger. This file is not manuscript prose and does
not modify the frozen fallback paper. Track-A scientific outcomes are deliberately
absent while the sealed queue is running.

## 1. Same-target and cold-start contract

For a chunk prediction `A_q[k]`, the physical target is `t=q+k`. The fixed-source
diagnostics use a natural same-target intervention: a source age `d` means
`q=t-d` and therefore `k=d`. For every fixed-source method at `d=20`, execution
falls back **exactly** to Fresh, `A_t[0]`, for `t<20`. The archived analyzers
verified this prefix at every persisted step.

The Object-development and frozen 140-block confirmation protocols run the
controller at 20 Hz. Thus `d=20` is 20 controller ticks, or 1.0 s. This is an
execution-timescale statement; the demonstration dataset used in the Track-B
persistence audit is separately recorded at 10 Hz.

Sources: `experiments/group_delay_factorial_act20/protocol.json`,
`experiments/cross_suite_confirmation/protocol.json`, and their archived reports.

## 2. Object development cohort (N=126)

This is development/hypothesis-generation evidence, not prospective
cross-suite confirmation.

- Suite: LIBERO-Object.
- Tasks: 1--9.
- States per task: `20,21,22,23,27,31,34,35,38,39,44,45,47,48`.
- Blocks per task: 14; paired blocks per condition: 126.
- Episode cap: 280 environment steps.
- Environment seed: `330000 + 100*task_id + state_id`.
- Checkpoint: the suite-level ACT export at
  `/home/wjq/checkpoints/zeromidnight_act_libero_object`, not the later
  task-specific checkpoint bank.

| Condition | Success | Rate | Policy-query rate |
|---|---:|---:|---:|
| Fresh / A0G0 | 56/126 | 44.44% | 1.00000 |
| A0G20 / FO20 | 81/126 | 64.29% | 1.00000 |
| A20G0 / Reverse20 | 12/126 | 9.52% | 1.00000 |
| A20G20 / FullOld20 | 47/126 | 37.30% | 1.00000 |
| coherent H16 | 88/126 | 69.84% | 0.06515 |

Realized episode-length summaries:

| Condition | All episodes: mean / median / range | Successful episodes: mean / median / range |
|---|---:|---:|
| Fresh | 218.51 / 280 / 106--280 | 141.64 / 130 / 106--274 |
| A0G20 | 190.17 / 150.5 / 105--280 | 140.26 / 137 / 105--263 |
| A20G0 | 265.35 / 280 / 110--280 | 126.17 / 127.5 / 110--142 |
| A20G20 | 224.69 / 280 / 111--280 | 131.72 / 127 / 111--209 |
| H16 | 181.50 / 143.5 / 106--280 | 138.97 / 131 / 106--240 |

The archived Object runner did not persist a reliable per-episode wall-clock
field, so wall-clock and mean call latency are unavailable for this cohort and
must not be reported as zero.

## 3. Frozen 140-block cross-suite factorial

The primary cohort is Goal tasks `4,6,7,8,9` and LIBERO-10 tasks `0,2,4,6,7`,
with states `0..13` for every task (14 blocks/task, 140 paired blocks/condition).
Goal episodes were capped at 300 steps and LIBERO-10 episodes at 520 steps.

Checkpoint structure: **ten task-specific ACT checkpoints**, one 100k export per
task. This is not a single-checkpoint experiment. Repository provenance supports
one trained checkpoint/training seed per task, which remains the relevant seed
limitation.

| Condition | Success | Rate | Query rate | Mean wall time/episode (s) | Mean episode-level call latency (s/query) |
|---|---:|---:|---:|---:|---:|
| A0G0 / Fresh | 77/140 | 55.00% | 1.00000 | 24.06 | 0.05797 |
| A0G20 / FO20 | 83/140 | 59.29% | 1.00000 | 19.02 | 0.04039 |
| A20G0 / Reverse20 | 38/140 | 27.14% | 1.00000 | 23.00 | 0.04101 |
| A20G20 / FullOld20 | 66/140 | 47.14% | 1.00000 | 22.81 | 0.04721 |
| coherent H16 | 93/140 | 66.43% | 0.06430 | 7.27 | 0.04893 |

The latency column averages the archived per-episode mean call latencies; it is
not a pooled call-weighted estimator.

Realized episode-length summaries:

| Condition | All episodes: mean / median / range | Successful episodes: mean / median / range |
|---|---:|---:|
| Fresh | 287.90 / 260 / 68--520 | 132.29 / 87 / 68--330 |
| A0G20 | 275.69 / 245.5 / 68--520 | 137.07 / 91 / 68--371 |
| A20G0 | 352.97 / 300 / 65--520 | 101.47 / 80.5 / 65--371 |
| A20G20 | 305.31 / 300 / 70--520 | 147.94 / 113 / 70--429 |
| H16 | 254.39 / 238 / 66--520 | 148.55 / 119 / 66--276 |

### Factorial contrasts and attribution discipline

The only clearly stable component-assignment contrast is:

- `A0G20 - A20G0 = +32.14 pp` (83/140 versus 38/140), discordance
  48:3, exact two-sided McNemar `p=1.96749e-11`, paired 95% CI
  `[+23.6,+40.7] pp`, task-cluster 95% CI `[+21.4,+44.3] pp`.

The simple effects are conditional and substantially weaker:

- `A0G20 - A0G0 = +4.29 pp`, discordance 12:6, paired CI
  `[-1.4,+10.0] pp`, task-cluster CI `[-1.4,+10.7] pp`.
- `A0G20 - A20G20 = +12.14 pp`, discordance 28:11, paired CI
  `[+3.6,+20.7] pp`, task-cluster CI `[-2.1,+28.6] pp`.

Therefore the four-cell table does **not** identify a unique additive attribution
of the diagonal contrast to arm freshness or gripper commitment. Do not call the
32.14 pp contrast "the arm effect" or "the gripper effect," and do not assign
path-dependent percentages to either component.

The current canonical difference-in-differences sign is:

`I_RD = p(A20G20) - p(A20G0) - p(A0G20) + p(A0G0)`.

Under this convention, the 140-block risk-difference interaction is +15.71 pp.
The paired bootstrap CI is `[+6.43,+25.00] pp`; the task-cluster percentile CI is
`[+0.71,+30.71] pp`; a t interval across ten task effects is
`[-2.76,+34.18] pp`; and the exact task-level sign-flip sensitivity test gives
`p=0.101562`. The log-odds interaction is 0.6979 (odds-ratio interaction 2.009),
with paired bootstrap CI `[0.3034,1.1142]`, task-cluster percentile CI
`[0.0007,1.5980]`, and delete-one-task jackknife-t CI `[-0.2089,1.6046]`.
This is a **POST_HOC_SUPPORTING_INTERACTION** and is suggestive under the small
number of task clusters, not confirmatory evidence.

The 126-block Object protocol predeclared the algebraically opposite descriptive
formula, `A0G20-A0G0-A20G20+A20G0`, yielding -7.94 pp. Under the current
canonical sign it is +7.94 pp. That Object analysis remains a predeclared
descriptive interaction; its preregistration does not transfer to the 140-block
post-hoc interaction.

Sources: `experiments/cross_suite_confirmation/{protocol.json,report.md}` and
`experiments/icra27_crosssuite_query_allocation/interaction_robustness/`.

### Separate Spatial context

The completed Gate-4A2 Spatial artifact is a separate 100-block/condition panel:
Spatial tasks 0--9 with states `1,13,15,19,21,24,31,37,40,47`. It used the
suite-level, multi-suite-trained ACT checkpoint
`ishandotsh/act_libero_spatial_test` at immutable revision
`8f04de1472975d62db214238b2fc07e78bde2474`, not the later task-specific bank.
Fresh and A0G20 each achieved 40/100; A20G20 achieved 30/100. The panel contains
no A20G0/Reverse20 or coherent-H16 cell and therefore cannot establish a full
factorial asymmetry or a practical-executor comparison. Its exact missing
Reverse20 completion is drafted as a post-hoc reviewer supplement and will not
be pooled into N=140.

## 4. C2 and moderate-horizon context

On the frozen 140-block cohort, C2 (`H16Arm+FreshGrip`) succeeded on 76/140,
compared with Fresh at 77/140 and coherent H16 at 93/140. Thus C2-Fresh is
-0.71 pp and H16-C2 is +12.14 pp. The Object-development C2 comparison used a
different task distribution and checkpoint family: C2 was 42/126 and H16 was
88/126. Any cross-cohort discrepancy must not be attributed uniquely to task
distribution or checkpoint family without the separately frozen disentangling
experiment.

In the final CARE gate, H13 achieved 95/130 and H16 92/130. The H13-H16
discordance was 8:5, delta +2.31 pp, exact McNemar `p=0.581055`, paired CI
`[-3.08,+7.69] pp`, and task-cluster CI `[-0.75,+5.43] pp`; 9/9 LOTO estimates
were positive. This is directionally consistent but small relative to the
uncertainty, supporting a flat moderate-horizon band rather than a unique
optimum. LOTO positivity and a statement algebraically implied by the same
task-effect signs must not be presented as independent evidence.

## 5. SmolVLA scope evidence

The four-task sparse-temporal-ensemble development panel used Object task 3,
Spatial task 0, Goal task 2, and LIBERO-10 task 3, states 10--19 (40 paired
blocks/method). SmolVLA results were:

| Method | Success | Query rate |
|---|---:|---:|
| coherent H8 | 30/40 | 0.12672 |
| sparse TE H8 | 28/40 | 0.12659 |
| coherent H16 | 30/40 | 0.06465 |
| sparse TE H16 | 29/40 | 0.06468 |

Sparse-TE H8 versus coherent H8 had discordance 2:4 and exact McNemar `p=.6875`;
sparse-TE H16 versus coherent H16 had discordance 3:4 and `p=1`. Bootstrap
intervals were not archived for this pilot, so none should be inferred.

The larger completed SmolVLA scope experiment used all 40 LIBERO tasks with four
states/task (160 paired blocks). At H8, both coherent H8 and ARM8_GRIP16 achieved
106/160; discordance was 12:12, paired CI `[-6.25,+6.25] pp`, task-cluster CI
`[-5.62,+5.00] pp`, and query rates were 0.12643 and 0.12647. At the H4 query
schedule, ARM4_GRIP4 achieved 108/160 and ARM4_GRIP32 110/160; discordance was
17:15, delta +1.25 pp, paired CI `[-5.62,+8.12] pp`, task-cluster CI
`[-5.00,+8.12] pp`, and query rates were 0.251230 and 0.251312. These are
cross-policy scope/robustness results on exposed cells, not independent
confirmation and not evidence that SmolVLA must reproduce ACT.

## 6. Checkpoint selection and demonstration provenance

The task-specific ACT bank uses the terminal 100k export. Exports also exist at
20k, 40k, 60k, and 80k, but no outcome or validation-based checkpoint selection
rule was used: 100k is the fixed terminal checkpoint. Every audited export has a
7D action, chunk size 100, native temporal ensembling disabled, and checkpoint-
frozen MEAN_STD action normalization/unnormalization.

The suite-level Object-development checkpoint was trained for 100k steps with
seed 1000 on all 454 episodes of
`DorayakiLin/libero_object_25_08_23_lerobotv2.1`. The local dataset metadata has
ten tasks, 66,984 frames, 10 Hz, and only a `train:0:454` split. It does not
establish a held-out validation split.

For the ten task-specific checkpoints in the 140-block confirmation, the frozen
training episode subsets contain 412 episodes total:

| Task | Training demonstrations |
|---|---:|
| Goal 4 / 6 / 7 / 8 / 9 | 46 / 40 / 50 / 49 / 36 |
| LIBERO-10 0 / 2 / 4 / 6 / 7 | 33 / 41 / 38 / 36 / 43 |

These are per-task subsets of `HuggingFaceVLA/libero`. The locally audited
dataset exposes only a training split; do not call the demonstrations held out.
The checkpoint configuration does not freeze a dataset revision, although the
Track-B local cache audit records the resolved revision where recoverable.

## 7. Success criterion and action contract

Success is the LIBERO environment's `_check_success()` result exposed as
`info["is_success"]`; the historical runners use positive terminal reward only
as the established fallback when the info field is absent. Episodes terminate
when the underlying environment is done or the success predicate becomes true.

The policy emits a continuous 7D relative action: translation dimensions 0--2,
rotation axis-angle dimensions 3--5, and one continuous gripper scalar at
dimension 6. The checkpoint postprocessor unnormalizes all seven dimensions.
The Panda gripper executor then uses the sign of the gripper scalar to increment
its two-finger command by 0.01 and clips the command to `[-1,1]`. Therefore the
policy output is continuous, while downstream gripper actuation is sign-based;
the policy output itself must not be described as binary.

## 8. Query schedules and timing terminology

- Fresh, A0G20, A20G0, and A20G20 query the complete policy once per executed
  environment step; their policy-query rate is exactly 1.0.
- FO20 and Reverse20 have the same query schedule. Any difference in realized
  total calls arises only from outcome-dependent episode termination; this audit
  was already frozen on `paper/icra27-final-claim-freeze` and is not rerun here.
- A coherent horizon `h` queries at `q=0,h,2h,...` and executes the corresponding
  chunk until the next scheduled query.
- Policy-query rate means policy calls divided by environment steps. It is not
  FLOPs, generic compute, or a hardware-independent cost measure.
- Wall-clock and per-call latency are reported separately wherever archived.

## 9. Preregistration lineage and evidence classes

| Artifact | Commit | Role |
|---|---|---|
| Object d=20 repaired factorial protocol | `7ab52cbc6360ae8436cfe5a04f8d200130d3f7a4` | Exposed development; interaction predeclared descriptive |
| Cross-suite protocol freezing FO20-vs-Reverse20 and cohort | `78ebc4daf3f9893c51bdf9f864283d6c4c11642e` | Pre-outcome frozen primary contrast |
| Frozen cross-suite outcomes | `2cae988fe70e406a795b5aa4bf24af02809496e8` | Completed outcome artifact |
| Frozen fallback science | `7ea83e1c0bea4367cc722a3d7b72ac0ca827e009` | Read-only fallback science |
| Fallback paper claim freeze | `ec8dc325b1ed6d54053b35d411ea92b13108374a` | Read-only manuscript context |
| Track-A query-allocation preregistration | `40549d876c0e09fad4e8033b3206f6018f53ece5` | Current prospective condition/cohort seal |

The 140-block factorial interaction is post-hoc supporting analysis of frozen
outcomes. The 126-block Object interaction was predeclared descriptive. Do not
conflate these evidence classes.

For current Track A, the accurate wording is: **query-allocation conditions
frozen from Object development were prospectively evaluated on non-Object
task-state cells selected without reference to their query-allocation outcomes.**
Do not call the three suites unseen or the 30 policies globally executor-unseen.
All 450 selected cells are `TRACK_A_CELL_PROSPECTIVE=true`; some candidate cells
had other executor exposure and were conservatively excluded.

## 10. Multiplicity and evidence narration

The frozen primary cross-suite contrast was A0G20 versus A20G0. A0G20 versus
Fresh and A0G20 versus A20G20 were also frozen comparisons, but their weaker,
conditional evidence must remain visible. The 140-block interaction and all
nonlinear-scale sensitivities are post-hoc. Current Track-A questions, labels,
and all six conditions were frozen in commit `40549d8` before outcomes.

Report all losing tasks/suites and negative mechanism tests. Do not present
per-task sign summaries and LOTO positivity as independent evidence when one is
algebraically implied by the other.

## 11. Existing evidence tension governing Track A

The frozen 140-block result A0G20-A0G0 is only +4.29 pp with both uncertainty
intervals crossing zero, whereas exposed Object development found
ARM4_GRIP32-H4 = +10.56 pp with positive paired and task-cluster intervals.
These interventions differ in source-age profile, policy-query rate, and
fixed-source versus periodic-schedule semantics. Those are descriptive facts,
not demonstrated explanations.

Track A therefore tests the distinct prospective hypothesis that, under
periodic executable schedules at matched policy-query rate, preserving gripper
commitment while refreshing the arm mitigates the uniform high-frequency
replanning penalty. If Track A fails, that is consistent with the weak existing
cross-suite conditional evidence. If it succeeds, the discrepancy must still be
reported and intervention-semantics explanations remain hypotheses. No ICRA
follow-up experiment is authorized to reconcile it.

## 12. Track-B mechanism result and quantity separation

Track B is **mechanism-only logging on already outcome-exposed development
cells; success outcomes are not used for method selection.** It is not
"outcome-free."

The original ACT localization criterion did not pass: `R_ACT = 0.5396`,
episode-cluster bootstrap CI `[0.3975,0.7034]`, so the gripper/arm normalized
dispersion ratio is not greater than one. The low-minus-high margin-tercile sign
disagreement contrast was positive, 0.1792 with CI `[0.1209,0.2353]`, but both
criteria were required. Thus `ACT_LOCALIZATION_PASS=no`. Cross-policy mechanism
support also failed: `R_SMOLVLA=0.4311` with CI `[0.3511,0.5184]`, and
`R_ACT-R_SMOLVLA=0.1085` with CI `[-0.0422,0.2815]`.

Retain three distinct mechanism quantities throughout analysis and figure specs:

1. demonstration action temporal persistence;
2. frozen-policy future-action forecast error;
3. same-target cross-source prediction disagreement.

These are not interchangeable forms of generic "prediction error." The archived
ACT same-target normalized dispersions were translation 0.1364, rotation 0.1484,
gripper 0.0790, and six-dimensional arm RMS 0.1464. SmolVLA values were
translation 0.3157, rotation 0.4644, gripper 0.1733, and arm RMS 0.4020.

The demonstration persistence audit uses 173 training episodes from the four
Track-B ACT tasks, not a held-out split. Gripper sign/state transition frequency
was 0.01552 per 10-Hz demonstration step (episode-cluster CI
`[0.01446,0.01666]`); mean distance to the next transition was 30.78 steps, with
19.17% right-censored. The frozen-policy future-action forecast analysis is
sealed but deferred until Track A releases the GPUs; no values are available yet.

Existing artifacts do not contain enough object/contact/phase information for a
defensible failure taxonomy. Record:
`FAILURE_MODE_CLASSIFICATION_NOT_IDENTIFIABLE_FROM_EXISTING_ARTIFACTS`.

## 13. Current Track-A status boundary

Track A remains the sealed 30-task x 15-state x six-condition evaluation (2,700
episodes) under preregistration `40549d8`. The conditions, in frozen within-block
order, are H16, H4, ARM4_GRIP32, H2, ARM2_GRIP16, and canonical TE_DENSE. This
file contains no partial Track-A success values. It must be updated only after
all markers exist, the integrity validator passes, and the frozen analysis runs.
