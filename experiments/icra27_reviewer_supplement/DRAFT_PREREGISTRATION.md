# Reviewer-supplement preregistration draft

Status: **DRAFT ONLY — NOT FROZEN, NOT LAUNCHABLE**

This document records the already authorized reviewer-supplement design while
Track A is running. It does not alter the Track-A preregistration or manifest,
and it does not authorize any rollout. A separate final preregistration commit
and fail-closed manifest may be created only after:

1. all 2,700 Track-A completion markers exist;
2. Track-A integrity validation passes;
3. the frozen Track-A analysis is complete;
4. no Track-A technical rerun is required; and
5. R2 eligibility is decided from technical feasibility, elapsed time, and GPU
   budget without inspecting R1 outcomes.

The scientific rollout window begins only after those gates and is capped at 24
wall-clock hours. After that window, no new scientific condition may start;
technical retries may finish an already frozen condition.

## Global design contract

- Every rollout is paired by policy/checkpoint, task, initialization state,
  environment seed, and policy RNG rule.
- Task-state block is the within-experiment paired unit. Task is the primary
  generality unit wherever more than one task is present; state-level episodes
  are not treated as independent evidence of cross-task generality.
- Every condition for a task-state block uses a fresh environment initialized
  from the same stored state and seed.
- New conditions use task-major workers: load a checkpoint once, run every
  frozen state and condition owned by that task, then destroy policy and
  environment and release GPU memory.
- Condition order, static task shards, seeds, and checkpoint paths will be
  serialized in the final manifest. Scheduling will not depend on outcomes.
- Scientific failures are completed observations. Technical attempts are
  limited to the initial attempt plus at most two retries.
- Existing results are reusable only when checkpoint, cohort, state, seed,
  action semantics, query schedule, prefix, episode cap, and success criterion
  all match exactly. Reuse decisions are technical identity decisions and may
  not inspect success magnitude.
- All long runs require incremental per-episode results, completion markers,
  deterministic resume, logs, PID files, and a fail-closed detached launcher.
- No partial supplement outcome may be analyzed before every frozen cell in its
  preregistered analysis family is complete.

No adaptive executor, debounce, consensus, C-RTC, PACE reproduction, new
horizon search, extra lag, outcome-selected task subset, training seed, or
RoboTwin work is authorized.

## Shared action and same-target semantics

The action is 7D: translation dimensions 0--2, rotation axis-angle dimensions
3--5, and gripper dimension 6. For policy query `q`, chunk offset `k`, and
physical target `t`, the same-target identity is `q+k=t`.

For fixed source age `d`, the stale source is `q=t-d` and `k=d`. For `t<d`,
every fixed-source condition executes the frozen Fresh prefix `A_t[0,:]`.
Fixed-source diagnostics query the whole policy once per executed environment
step; unused predictions are retained only for auditing or discarded as the
condition specifies.

All action normalization and unnormalization use checkpoint-frozen statistics.
No scale is refit on supplement outcomes.

## R1.1 — Fixed-source source-age sensitivity curves

### Question

Is the component-assignment asymmetry present across a temporal scale rather
than only at the historical `d=20` anchor?

### Cohort and checkpoint

- Role: exposed Object development characterization, not confirmation.
- Cohort: the canonical 126-block LIBERO-Object cohort from
  `experiments/group_delay_factorial_act20/protocol.json`.
- Tasks: Object 1--9.
- States/task: `20,21,22,23,27,31,34,35,38,39,44,45,47,48`.
- Seed rule: `330000 + 100*task_id + state_id`.
- Episode cap: 280 steps; controller rate: 20 Hz.
- Checkpoint: `/home/wjq/checkpoints/zeromidnight_act_libero_object`.
- Policy RNG seed: 424242.

### Frozen candidate grid

`d in {2,4,8,12,16,20,32}`.

For every `d`, evaluate:

- `A_dG0`: dimensions 0--5 from `A_{t-d}[d]`, gripper from `A_t[0]`;
- `A0G_d`: dimensions 0--5 from `A_t[0]`, gripper from
  `A_{t-d}[d]`.

`d=0` is represented only by the single common Fresh anchor `A0G0`; it is not
duplicated as two series-specific cells. The validated `d=20` outcomes will be
reused only after the exact identity audit passes. No `d` may be added or
removed after outcomes.

If Fresh and both `d=20` cells reuse exactly, the maximum new rollout is 12 new
conditions x 126 blocks = 1,512 episodes. The final manifest will enumerate only
new cells while the analysis ledger identifies reused anchors.

### Analysis

For every `d`, report absolute successes/N, policy queries, query rate,
environment steps, wall-clock, and:

- `Fresh - A_dG0` (arm-source-age loss);
- `Fresh - A0G_d` (gripper-source-age loss);
- `A0G_d - A_dG0` (component-assignment difference).

For each contrast report paired discordances, exact two-sided McNemar, paired
bootstrap CI, task-cluster bootstrap CI, per-task deltas, and LOTO. Report the
entire grid regardless of sign.

The preregistered descriptive onset rule for each loss curve is the smallest
tested positive `d` for which the loss has a positive center and both paired and
task-cluster CI lower bounds exceed zero. If no `d` meets it, report no detected
onset on the tested grid. This is not an estimate of a continuous threshold.

Report adjacent-grid finite differences and a descriptive task-cluster
bootstrap slope versus numeric `d`; because the grid is nonlinear, the slope is
only a compact trend summary. A plateau will not receive a binary label. Any
plateau-like interpretation must be tied to the complete point/interval pattern
and cannot be inferred merely because one adjacent contrast is nonsignificant.

## R1.2 — Translation versus rotation component control

### Question

Does fixed-source sensitivity differ between the 3D translation and 3D rotation
subgroups when all other action dimensions are Fresh?

Use the same Object 126-block cohort, checkpoint, seeds, 280-step cap, 20-Hz
control rate, dense query schedule, `d=20`, and Fresh prefix as R1.1.

New conditions:

- `T20_R0_G0`: translation 0--2 from `A_{t-20}[20]`; rotation 3--5
  and gripper 6 from `A_t[0]`.
- `T0_R20_G0`: rotation 3--5 from `A_{t-20}[20]`; translation 0--2
  and gripper 6 from `A_t[0]`.

Maximum new rollout: 2 x 126 = 252 episodes. Reuse the exact Fresh results.

Primary descriptive contrast: `T20_R0_G0 - T0_R20_G0`. Secondary descriptive
contrasts compare each condition with Fresh. Report the same paired,
task-cluster, per-task, LOTO, query, and timing fields as R1.1.

This control equalizes subgroup dimensionality at 3D versus 3D but does not
identify dimensionality, geometry, representation, or control semantics as a
unique cause. It must not be used to convert the arm/gripper diagonal contrast
into additive percentages.

## R1.3 — Dense-query matched H16 factorial

### Question

What is the component-temporal 2x2 when all four cells query the policy once per
environment step, removing policy-query schedule as a between-cell difference?

### Cohort

- Role: `POST_HOC QUERY-MATCHED EXTENSION`.
- Exact frozen 140-block cohort only: Goal tasks `4,6,7,8,9` and LIBERO-10 tasks
  `0,2,4,6,7`, states `0..13`.
- Checkpoints: the ten exact task-specific 100k ACT exports recorded in
  `experiments/cross_suite_confirmation/protocol.json`.
- Seed rule: `340000 + 1000*suite_index + 100*task_id + state_id`, with suite
  indices Goal=2 and LIBERO-10=3.
- Episode caps: Goal 300, LIBERO-10 520; controller rate 20 Hz.
- Policy RNG seed: 424242.

### Four cells

Every step produces a whole-policy query. Predictions not selected for
execution are discarded.

- `C00`: arm Fresh, gripper Fresh.
- `C10`: arm from the scheduled H16 committed source, gripper Fresh.
- `C01`: arm Fresh, gripper from the scheduled H16 committed source.
- `C11`: arm and gripper from the scheduled H16 committed source.

For a scheduled source `q_h=16*floor(t/16)`, the committed component uses
`A_{q_h}[t-q_h]`; a Fresh component uses `A_t[0]`.

Existing Fresh may be reused as C00 and existing C2 may be reused as C10 only
if the exact-identity audit confirms all frozen fields. No equality is inferred
from outcome agreement. C01 is expected to require new rollout. C11 must undergo
a deterministic canary against sparse coherent H16: executed actions, simulator
states, terminal result, and episode length must be identical within the already
established numerical tolerance, while extra calls remain execution-inert.
Only if this canary and RNG-isolation audit pass may sparse H16 be reused as C11;
otherwise dense C11 is run as a distinct condition.

The expected new rollout is 140--280 episodes depending only on those technical
identity gates, never on outcome magnitude.

### Analysis

Use the canonical signed risk-difference interaction:

`I_RD = p(C11) - p(C10) - p(C01) + p(C00)`.

Report all four absolute cells, all conditional simple effects, the two
diagonals, paired and task-cluster intervals, per-task effects, LOTO, and
leave-one-suite-out descriptive results. Report the risk-difference interaction
and a log-odds interaction sensitivity. With ten task clusters, include a
transparent task-level sign-flip/permutation-style sensitivity where valid and
do not rely solely on percentile cluster-bootstrap exclusion of zero.

This is post-hoc identification support, not transfer of the original
140-block preregistration and not a new confirmation cohort.

## R1.4 — Spatial Reverse20 completion

### CPU provenance audit completed while Track A ran

The archived manifest and frozen branch recover the historical cohort without
using outcome magnitude:

- source ref: `origin/exp/gate4a2-spatial-act-generalization`, commit
  `e246d3d6baa2adac9b715c9065ff11a6860ad99c`;
- suite: LIBERO-Spatial tasks 0--9;
- states/task: `1,13,15,19,21,24,31,37,40,47`;
- blocks/condition: 100;
- seed rule: `340000 + 100*task_id + state_id`;
- episode cap: 280 steps; controller rate 20 Hz;
- checkpoint: `/home/wjq/checkpoints/ishandotsh_act_libero_spatial_test`,
  repository `ishandotsh/act_libero_spatial_test`, immutable revision
  `8f04de1472975d62db214238b2fc07e78bde2474`;
- checkpoint provenance: MULTI-SUITE ACT, 7D action, chunk size 100, native
  temporal ensembling and action smoothing disabled;
- evaluation dataset revision:
  `zeromidnight/libero_spatial_lerobot_v3.0@38927e939de5d2bfd40effcf27d16710aea6f864`;
- existing cells: Fresh/A0G0, FullOld20/A20G20, A0G20, AGE_EXP_B003, and
  COGACT_A03, all with one policy query per surviving environment step;
- exact runner/protocol sources: `research/audit_tools/gate4a2_rollout.py`,
  `research/audit_tools/gate3c_temporal_reuse.py`, and
  `research/gate4a2_spatial_generalization_protocol.md` on the source ref.

The historical manifest contains 500/500 unique completed cells and preserves
the selected initial-state identities. It contains no A20G0/Reverse20 cell.
Therefore the missing extension is exactly 100 new episodes if the remaining
portability canary below passes.

### Remaining pre-launch technical gate

Before final preregistration, port the frozen runner semantics into the isolated
supplement directory and verify without reading a new outcome:

- checkpoint/preprocessor/postprocessor identity against the archived hashes;
- identical Fresh and A0G20 action chunks on a technical exposed canary;
- A20G0 source mapping `q=t-20`, offset 20 for dimensions 0--5, with gripper
  from `A_t[0,6]`;
- exact Fresh prefix through `t=19`;
- one query per environment step;
- episode-seed/RNG behavior; and
- the archived success criterion and 280-step cap.

If the portability canary passes, freeze the missing A20G0/Reverse20 condition
on exactly the recovered 100-block cohort and label it
`POST_HOC SPATIAL FACTORIAL COMPLETION`. Report every historical and new cell,
including the negative FO20 result, but never pool this extension into the
original 140-block aggregate.

If any identity field is irrecoverable, set
`SPATIAL_FACTORIAL_COMPLETION_NOT_RECONSTRUCTABLE` and run no approximation or
replacement cohort.

## R2 eligibility gate

R2 is secondary. Before reading any R1 scientific outcome, serialize one of:

- `R2_ENABLED`: technical setup passes and sufficient wall-clock/GPU budget
  remains within the 24-hour window;
- `R2_DISABLED_RUNTIME`: insufficient remaining wall-clock/GPU budget; or
- `R2_DISABLED_TECHNICAL`: exact semantics or provenance cannot be established.

Eligibility may not use R1 success, effect direction, confidence intervals, or
task-level patterns. If enabled, the exact R2 subset below is frozen before its
first outcome. R2.3 additionally requires substantial margin after R2.1/R2.2 is
budgeted.

## R2.1 — Complete SmolVLA same-target factorial

Pending technical semantic audit, freeze the already exposed four-task scope
panel used by the historical SmolVLA evaluation and Track B:

- Object task 3, Spatial task 0, Goal task 2, LIBERO-10 task 3;
- states 10--19 and environment seeds 2000--2009;
- checkpoint `/home/wjq/checkpoints/HuggingFaceVLA_smolvla_libero`, revision
  `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`;
- SmolVLA query RNG key exactly as frozen in
  `experiments/sparse_temporal_ensemble_dev/protocol.json`, excluding method
  name from the key;
- four cells A0G0, A0G20, A20G0, A20G20 with the same-target and Fresh-prefix
  semantics above;
- 40 paired blocks/cell, at most 160 new episodes before exact reuse.

Report absolute success, A0G20-A20G0, all conditional simple effects, paired
uncertainty, task-level heterogeneity, queries, query rate, and wall-clock.
Analyze SmolVLA separately from ACT. Non-replication is a valid scope result.

## R2.2 — Object checkpoint/task-distribution disentangling

This condition family is eligible only if, before any R2 outcome is read, the
paper-support claim still discusses the failure of the Object C2 penalty to
reproduce on the confirmation cohort. If that claim is removed or weakened to
an uninterpreted cross-cohort discrepancy, record
`R2_2_OMITTED_CLAIM_NOT_REQUIRED` before R2 analysis.

If retained, use the exposed Object tasks 1--9 and the 14-state canonical Object
cohort, but replace the suite-level development checkpoint with the corresponding
task-specific 100k checkpoint for each task. Run exact Fresh and C2
(`H16Arm+FreshGrip`) semantics with original Object seeds/caps, 126 paired
blocks/condition, at most 252 episodes. This separates checkpoint family while
holding task distribution fixed; it does not estimate a universal checkpoint
effect from multiple training seeds.

## R2.3 — Wrong-target auxiliary control

This is enabled only if R2.1/R2.2 are already budgeted and substantial time
remains, with the decision frozen before their outcomes. Use the canonical
Object 126-block cohort at `d=20`:

- correct-target stale reference: `A_{t-20}[20,:]` (existing A20G20 where exact);
- fresh-source wrong-target control: `A_t[20,:]`.

Both query every step. The wrong-target condition is intentionally
non-deployable and does not preserve `q+k=t`; it illustrates why source age and
chunk offset cannot be independently identified under natural same-target
execution. For `t<20`, use the same Fresh prefix to isolate the post-prefix
comparison. Maximum new rollout is 126 episodes if the reference is reused.

## Passive failure-mode logging

For new supplement rollouts, log gripper command/state, object pose,
end-effector pose, success events, contact indicators, and existing phase
proxies only where already exposed by the environment API with negligible
overhead. Before bulk use, a deterministic identity canary must show identical
executed actions, simulator trajectory within frozen tolerance, terminal result,
and episode length with logging on versus off.

The logger is passive and cannot alter actions. Failure labels may be assigned
only when object/contact/state evidence supports them. Action traces alone are
insufficient. If a category cannot be identified, record it as unidentifiable
rather than infer it.

## Final preregistration checklist

The final post-Track-A seal must add, without changing the scientific design:

- exact new/reused cell ledger and provenance for every reused cell;
- exact Spatial reconstruction decision;
- exact R2 eligibility decision and included R2 families;
- static task shards and within-block condition order;
- retry and fail-closed rules;
- estimated runtime demonstrating compatibility with the 24-hour cap;
- technical canary results;
- analysis bootstrap seeds and draw counts;
- canonical tidy output schema; and
- a statement that no supplement outcome existed or was inspected before the
  seal commit.

The final seal SHA must be pushed before any supplement rollout launches.
