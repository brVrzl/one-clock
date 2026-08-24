# Post-Gate-3C ICRA 2027 paper plan

Status: Gate-3C is complete and confirmatory. This plan is now the active
paper source-of-truth scaffold. The historical `main.tex` and section files
remain obsolete prose until deliberately rewritten from this plan.

## Confirmed scientific question and boundary

The paper asks whether heterogeneous components of a jointly predicted action
chunk can benefit from different temporal source ages. The confirmed
intervention is:

\[
F_t=E_{t,t},\qquad O_t=E_{t,t-20},\qquad
\mathrm{FO20}_t=[F_t[0{:}6],O_t[6]].
\]

`O_t` is the offset-20 prediction for the same physical target time. It is not
a 20-step gripper hold: the gripper prediction may change at every controller
tick while the policy is queried. The result concerns source-observation age,
not query frequency or execution horizon.

Gate-3C's primary cohort was tasks 1--9, 14 common official states per task,
and 126 paired task-state blocks. FO20 achieved 80/126 successes (63.5%). The
comparators achieved 53/126 newest (42.1%), 55/126 full old20 (43.7%), 62/126
age-exponential (49.2%), and 59/126 tuned CogACT (46.8%). All four registered
FO20-minus-comparator contrasts were stable positive under paired-block
bootstrap, task-cluster bootstrap, and leave-one-task-out criteria.

This is a bounded confirmation for the frozen ACT/LIBERO system, the arm and
gripper partition, and a 20-tick source gap. It does not establish universal
arm freshness, universal gripper memory, dynamic temporal adaptation,
20-tick optimality, non-Markovian causation, or generalization to other
policies, embodiments, VLAs, dexterous hands, bimanual systems, or real robots.

## Evidence chain

```text
Action chunks produce multiple source-time predictions for one physical action
|-- Existing temporal aggregation usually makes one full-action decision
|-- Independent retention was a motivating but confounded negative observation
|-- Gate-3B found an exploratory fresh-arm / old-gripper asymmetry
|-- Additive teacher-forced losses are structurally blind to the symmetric 2x2 term
`-- Gate-3C confirms FO20 on untouched states against four full-action controls
```

The preregistered Gate-3B generic coherence result remains negative/suggestive,
not confirmatory: `C_coherence=+.025`, paired CI `[-.030,+.085]`, task-cluster
CI `[-.005,+.055]`. Its FO cell is developmental evidence only. Gate-3C is the
confirmatory source for the asymmetric result.

## Introduction arc

1. Action chunking provides multiple predictions for the same future physical
   action from observations at different source times.
2. ACT, CogACT, temporal ensembling, and adaptive execution typically make one
   temporal decision for the complete action.
3. Robot actions are heterogeneous: continuous arm motion and discrete gripper
   commands can value feedback and retained context differently.
4. A prior independent-retention failure could not isolate source assignment,
   because it also changed retention and execution dynamics.
5. Gate-3B directly crossed fresh and old20 arm/gripper assignments. Its FO
   result was exploratory, not confirmatory.
6. The RTX teacher-forced audit favored old sources for both arm and gripper,
   whereas Gate-3B's closed-loop marginal pattern favored a fresh arm.
7. Gate-3C confirmed FO20 on untouched tasks 1--9 against newest, full old20,
   age-exponential, and CogACT controls.

## Confirmed contribution claims

The paper may state:

1. controlled evidence that temporal-source age need not be shared across
   heterogeneous components of a jointly predicted action chunk;
2. a confirmed, training-free FO20 executor in which fresh arm feedback and an
   older gripper prediction outperform synchronized source assignments and
   full-action temporal controls in the evaluated system;
3. evidence that separable teacher-forced component metrics can disagree with
   closed-loop temporal-source utility, especially for arm motion; and
4. a reproducible matched-query intervention that isolates source generation
   while preserving one policy query per surviving controller step.

These are system-bounded claims. Do not call FO20 an adaptive horizon, a
universal multi-clock controller, or a causal explanation in terms of
non-Markovianity.

## Gate-3C primary result

| Method | Successes / 126 | Rate |
|---|---:|---:|
| FO20, fresh arm + old20 gripper | 80 | 63.5% |
| Newest full action | 53 | 42.1% |
| Full old20 action | 55 | 43.7% |
| Age exponential, beta=.03 | 62 | 49.2% |
| CogACT, alpha=.3 | 59 | 46.8% |

| Contrast | Estimate | Paired 95% CI | Task-cluster 95% CI |
|---|---:|---:|---:|
| FO20 - newest | +.2143 | [.1429,.2937] | [.1270,.3095] |
| FO20 - full old20 | +.1984 | [.0952,.3016] | [.0635,.3175] |
| FO20 - age exponential | +.1429 | [.0476,.2381] | [.0159,.2778] |
| FO20 - CogACT | +.1667 | [.0714,.2619] | [.0397,.2937] |

Every primary leave-one-task-out estimate is positive. Task heterogeneity is
real: FO20-minus-newest is positive on all nine primary tasks, while FO20
versus full old20 includes one negative task and one tie. The aggregate claim
must not be rewritten as uniform improvement on every task.

## Offline-versus-closed-loop interpretation

At `d=20`, the additive offline metric favors old arm translation, old arm
rotation, and old gripper sign prediction. Gate-3C favors the fresh arm and
old gripper in closed-loop success. The safe interpretation is that
teacher-forced component accuracy and closed-loop temporal-source utility are
not equivalent objectives in this audited setting. Do not claim that the
demonstration action is wrong or that offline accuracy causes the mismatch.

## Related-work boundary

ACT and CogACT are full-action temporal aggregation controls. AAC computes
component uncertainty but aggregates it to one prefix. TAS selects one complete
cached action. AutoHorizon and PACE choose a shared execution boundary. RTC,
REMAC, A2C2, and SEAM reconcile or correct stale chunks; SEAM's subset guidance
is the closest nuance but is not the present source-generation factorial.

Lazzati et al. support the possibility that older predictions better match
expert behavior under non-Markovian demonstrations. The distinction here is
narrower: source utility may differ across heterogeneous components, and
teacher-forced delayed prediction quality may not identify closed-loop source
utility.

Use the bounded sentence below, subject to final literature verification:

> We are not aware of prior controlled evaluations that independently assign
> temporal source generations to heterogeneous action components.

Do not claim “we are the first.” Do not claim that older predictions helping is
novel.

## Figure plan after Gate-3C

**Figure 1:** schematic of `F_t`, `O_t`, and the four FF/OO/FO/OF assignments;
mark FO as exploratory Gate-3B evidence and FO20 as the confirmed Gate-3C
executor. Do not call mixed actions off-manifold.

**Figure 2:** Gate-3B 2x2 success matrix and post-hoc marginal effects. Open
the caption with the unresolved generic coherence result.

**Figure 3:** offline source preference versus closed-loop source preference.
Show old arm and old gripper offline, then fresh arm and old gripper in the
Gate-3B/Gate-3C closed-loop directional result. Do not combine incomparable
loss and success axes.

**Figure 4:** Gate-3C confirmation. Show the five primary success rates, the
four FO20 contrasts with paired/task-cluster intervals, and task-wise primary
contrasts. Data are frozen in the updated interface and must not be smoothed.

## Fill status

No Gate-3C placeholders remain in the active manuscript skeleton or Figure 4
interface. The historical Gate-3B planning filename is retained for continuity,
but its contents now describe the completed Gate-3C result. The next edits are
ordinary manuscript completion, citation verification, and submission audit;
no additional experiment is implied by this plan.
