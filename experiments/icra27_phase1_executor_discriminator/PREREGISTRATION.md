# Phase-1 prospective executor discriminator

Status: **FROZEN BEFORE HELD-OUT OUTCOMES**

## Question and scope

On the complete ten-task LIBERO-10 suite, test whether intermediate arm cadence with longer gripper commitment (`ARM8_GRIP32`) improves over matched-query coherent `H8` and over the strong coherent `H16` baseline. This is a single bounded discriminator, not a horizon search.

The experiment contains exactly five conditions, in fixed block order: `H4`, `ARM4_GRIP32`, `H8`, `ARM8_GRIP32`, and `H16`. Arm dimensions are exactly indices 0:6 and the gripper is exactly index 6. Temporal ensembling is disabled.

## Prospective cohort

All ten LIBERO-10 tasks enter. Each task contributes ten official initial states, for 100 paired task-state blocks and 500 scientific episodes. The official installation contains 50 states for every task.

The incoming summary described prior Track A as states 0-14, but the completed local Track-A results directly record task-specific official IDs mostly in 20-44. Other completed local results also expose some IDs in 15-19. To honor the stricter instruction not to repeat historical experiments, the cohort was selected from the observed result inventory: for each task, consider IDs 15-49, exclude every task-state with any recorded prior result, then take the lowest ten eligible IDs. Ten is the maximum uniform eligible count because tasks 1 and 9 each have ten. No success outcome or effect size entered selection.

Exact states:

- task 0: 15-19, 35-39
- task 1: 15-19, 45-49
- task 2: 15-19, 35-39
- task 3: 25-29, 40-44
- task 4: 15-19, 35-39
- task 5: 15-19, 35-39
- task 6: 15-19, 35-39
- task 7: 15-19, 35-39
- task 8: 15-19, 35-39
- task 9: 15-19, 45-49

`prior_exposure_inventory.json` is the machine-readable freeze of the exclusion evidence and selected cohort.

## Pairing and execution contract

Every method in a block uses the same task, official state, environment seed, policy seed, task-specific ACT checkpoint, preprocessing/postprocessing, success criterion, episode cap, and 10 Hz control setting. The environment seed is `390000 + 100*task_id + state_id`, matching the validated Track-A LIBERO-10 seed contract. Each condition receives a fresh synchronous environment. Scientific failures are completed outcomes; only technical failures may be retried, at most twice after the initial attempt.

The existing validated Track-A runtime is reused directly. Before scientific rollout, a bounded technical smoke on an already exposed state must verify exact group semantics, chunk support for gripper horizon 32, fresh/reset determinism, actual H8/A8G32 query triggers, the held-out-state exclusion, and the current deterministic LIBERO contract. Smoke success outcomes are not recorded or used.

## Frozen contrasts and inference

- Primary: `ARM8_GRIP32 - H8`.
- Secondary: `ARM8_GRIP32 - H16`.
- Supporting replication: `ARM4_GRIP32 - H4`.

For each contrast, report successes/N, paired risk difference, discordant directions, two-sided exact McNemar, 20,000-draw paired-block percentile CI, 20,000-draw task-cluster percentile CI, and all ten task deltas. Report condition- and task-level actual policy calls, environment steps, and policy-query rates. Policy-query rate is not called compute.

Credible primary evidence is frozen as: positive primary risk difference, exact McNemar p<0.05, and positive lower bounds for both paired-block and task-cluster 95% bootstrap CIs.

## Decision

- A: credible primary evidence and the observed `ARM8_GRIP32-H16` paired risk difference is positive.
- B: credible primary evidence and the observed `ARM8_GRIP32-H16` paired risk difference is non-positive.
- C: the credible-primary criterion is not met. Close the executor-method branch and do not search more horizon cells.

All ten tasks remain in the report regardless of direction. No other horizons, training, SmolVLA, adaptive executor, or paper edit is authorized in this phase.

## Prospective amendment, 2026-09-04T10:45:20+08:00

Outcome-exposure status at amendment: **NO NEW PHASE-1 HELD-OUT SUCCESS OUTCOME EXPOSED**. No new Phase-1 held-out success outcome had been generated, inspected, parsed, summarized, or displayed. No worker or live smoke had started. This amendment is therefore prospective. The original text above is preserved as the protocol history; explicit fields below supersede it for execution.

The forensic audit identified `ARM8_GRIP16` as the most directly motivated matched-query-rate arm=8 candidate: on Object tasks 1-9, H8 was 114/180, ARM8_GRIP16 was 123/180, and H16 was 123/180. Existing evidence did not establish that gripper horizon 32 was preferable. A4G16 and A4G32 were 128/180 and 131/180, while on Object-126 H16 was 88/126 and A16G32 was 78/126. The already frozen `ARM8_GRIP32` is retained, and two conditions are added:

6. `ARM8_GRIP16`: groupwise fixed arm8/gripper16.
7. `ZOH8_GRIP16`: query the full policy at steps 0,8,16,...; execute each fresh eight-step arm prefix; at gripper boundaries 0,16,32,... take gripper index 0 from the fresh query at that boundary and hold that scalar unchanged for 16 environment steps. It never advances through indices 1-15 of a retained gripper chunk and adds no smoothing, hysteresis, threshold, low-pass, or adaptive logic.

The effective condition order is `H4`, `ARM4_GRIP32`, `H8`, `ARM8_GRIP32`, `H16`, `ARM8_GRIP16`, `ZOH8_GRIP16`.

All ten tasks support exactly 50 official initial states under the installed validated evaluator. The effective held-out range is the full fixed range 15-49 for every task: 35 states × 10 tasks = 350 paired blocks per condition and 2,450 scientific episodes. The range will not be reduced based on runtime or outcomes.

Revised frozen inference:

- Primary: `ARM8_GRIP16 - H8`.
- Key secondary: `ARM8_GRIP32 - H8`.
- Additional secondary: `ARM8_GRIP16 - H16`, `ARM8_GRIP32 - H16`, `ARM8_GRIP16 - ARM8_GRIP32`, and `ARM8_GRIP16 - ZOH8_GRIP16`.
- Supporting replication remains `ARM4_GRIP32 - H4`.
- Both candidates are reported separately regardless of direction. No post-hoc best-of-candidates comparison or pooled candidate test is permitted.

For each frozen contrast, credible positive evidence means a positive paired risk difference, exact two-sided McNemar p<0.05, and positive lower bounds for both paired-block and task-cluster 95% bootstrap CIs. No McNemar power or minimum detectable effect is inferred from N alone.

The effective branch rule is exhaustive and candidate-specific:

- A: at least one frozen component candidate has credible positive evidence versus H8 and the same candidate has credible positive evidence versus H16.
- B: at least one frozen component candidate has credible positive evidence versus H8, but no candidate satisfies A.
- C: neither ARM8_GRIP16 nor ARM8_GRIP32 has credible positive evidence versus H8. Close the executor branch and do not search additional horizon cells.

The exact machine-readable amendment is `amendment_20260904T104520+0800.json`. Everything else in the original protocol remains unchanged.

## Prospective interpretation addendum, 2026-09-04T10:50:29+08:00

No new Phase-1 held-out outcome had been generated or exposed before this addendum. For these interpretation rules, “matches or exceeds” means an observed paired risk difference ≥0, “does not” means <0, and “outperforms” means >0. These descriptive interpretation labels do not replace the frozen credible-evidence criteria.

Apply the following rules in order:

1. If `ZOH8_GRIP16` outperforms `ARM8_GRIP16`, the component-chunk executor contribution is weakened further and the simpler persistence explanation takes precedence.
2. Otherwise, if `ARM8_GRIP16` matches or exceeds H16 while `ZOH8_GRIP16` does not, report that the gain is not reproduced by matched-interval zero-order gripper holding. This supports retained chunk progression over simple persistence as the relevant distinction, but does not establish strict necessity.
3. Otherwise, if both `ARM8_GRIP16` and `ZOH8_GRIP16` match or exceed H16, do not attribute the gain to retained chunk structure; the evidence is consistent with reduced gripper update rate or persistence being sufficient.
4. Otherwise, report the candidate-specific paired evidence without claiming that retained chunk progression is necessary.

`ZOH8_GRIP16-H16` is frozen as an interpretation-only paired contrast, with paired-bootstrap seed 27818 and task-cluster-bootstrap seed 27918. The exact machine-readable addendum is `interpretation_addendum_20260904T105029+0800.json`.
