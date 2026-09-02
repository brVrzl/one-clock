# ICRA 2027 reviewer-supplement preregistration

Status: **FROZEN BEFORE TRACK-A OR SUPPLEMENT OUTCOME INSPECTION**

This protocol answers two prospective reviewer critiques: temporal sensitivity
must be characterized beyond a single source age, and component semantics must
be separated from the historical arm/gripper comparison as far as the existing
7D action representation allows. It also completes the previously agreed
dense-query H16 and Spatial controls. The design uses only previously exposed
development cohorts and previously agreed conditions. No Track-A result was
opened while this protocol was prepared.

`protocol.json` is the immutable machine-readable authority. The governing Git
commit is recorded after the freeze in `PREREGISTRATION_COMMIT`; the watcher
requires that commit to be on the pushed remote branch and verifies that the
protocol and execution sources are unchanged from it before any rollout.

## Shared contract

- The action is 7D: translation 0--2, rotation axis-angle 3--5, gripper 6.
- For physical target `t`, same-target sources obey `q+k=t`. A fixed age `d`
  uses `q=t-d,k=d`; for `t<d` all dimensions execute the Fresh `A_t[0]`
  prefix. Every fixed-source and dense-H16 condition queries once per executed
  environment step.
- Each paired cell uses the frozen checkpoint, task, initialization state,
  environment seed, policy/query RNG rule, simulator settings, success
  criterion, and episode cap in `protocol.json`.
- New cells are task-major and statically sharded by frozen task order modulo
  three. Within every block conditions follow the serialized order. A worker
  may resume only an identical cell. Results, attempt ledgers, progress records,
  and completion markers are written incrementally.
- A scientific failure is a completed observation. Only technical failures may
  be retried, with at most three total attempts. An unresolved technical
  failure stops analysis and every downstream phase.
- Historical reuse requires exact checkpoint, cohort, state/seed, action and
  source semantics, query schedule, simulator settings, episode cap, and
  success criterion. Outcome agreement is never an identity test.
- No supplement family is analyzed until all its new and reused cells pass the
  integrity gate. Every frozen condition and contrast is reported regardless
  of sign. There is no best-age search and no outcome-based task exclusion.

## R1A: fixed-source temporal sensitivity

Use the exposed 126-block Object cohort (tasks 1--9; states
`20,21,22,23,27,31,34,35,38,39,44,45,47,48`) with the suite-level ACT
checkpoint. Freeze `d={2,4,8,12,16,20,32}` and, for every `d`, run/report
`A_d_G0` and `A0_G_d`. Exact historical Fresh, `A20G0`, and `A0G20` cells are
reused only after the identity audit; the other 12 conditions are new (1,512
episodes). Report successes/rates, `Fresh-A_d_G0`, `Fresh-A0_G_d`, and
`A0_G_d-A_d_G0`, with discordances, exact McNemar, paired and task-cluster
bootstrap intervals, per-task effects, LOTO, queries, query rate, steps, and
wall time for every `d`. No threshold or best `d` is selected.

## R1B: translation versus rotation

On the same 126 blocks at `d=20`, run `T20_R0_G0` (translation stale; rotation
and gripper Fresh) and `T0_R20_G0` (rotation stale; translation and gripper
Fresh), 252 episodes. The primary descriptive contrast is
`T20_R0_G0-T0_R20_G0`; each is also compared with Fresh. This equalizes the
continuous subgroup dimension count but does not uniquely separate dimension
count, semantics, representation, or continuous/binary structure.

## R1C: dense-query matched H16 factorial

Label: `POST_HOC_QUERY_MATCHED_EXTENSION`. Use only the frozen 140-block Goal
and LIBERO-10 cohort. All four conditions make a whole-policy query every step:
`C00` Fresh/Fresh; `C10` scheduled-H16 arm/Fresh gripper; `C01` Fresh
arm/scheduled-H16 gripper; `C11` scheduled-H16 arm/gripper. Historical dense
Fresh and dense C2 are reused as C00 and C10 after exact identity validation.
C01 and C11 are new (280 episodes). Sparse HARD-H16 is never reused as C11
because the query schedule differs. Before bulk rollout a frozen deterministic
canary must show that unused dense forward passes leave C11 executed actions,
simulator trajectory, terminal result, and length identical to sparse
HARD-H16. Canary failure stops R1C. Report the full factorial, simple effects,
diagonals, risk-difference interaction, paired/task-cluster intervals,
per-task, LOTO, per-suite, and leave-one-suite-out descriptions.

## R1D: Spatial Reverse20 completion

Label: `POST_HOC_SPATIAL_FACTORIAL_COMPLETION`. The historical Gate-4A2
provenance at commit `e246d3d6baa2adac9b715c9065ff11a6860ad99c` reconstructs
Spatial tasks 0--9, states `1,13,15,19,21,24,31,37,40,47`, seeds
`340000+100*task_id+state_id`, the suite ACT checkpoint at immutable revision
`8f04de1472975d62db214238b2fc07e78bde2474`, 20 Hz, 280 steps, dense queries,
and exact state identities. Run only missing Reverse20/A20G0 (100 episodes) if
the checkpoint-hash, runtime, initial-state, prefix, source-map, and portability
canary passes. Otherwise record
`SPATIAL_FACTORIAL_COMPLETION_NOT_RECONSTRUCTABLE` and skip it; no replacement
cohort is permitted. Never pool this extension into the primary 140 blocks.

## Passive failure-mode logging

The current common LeRobot vector wrapper exposes actions, rewards, terminal
success, and full simulator state, but does not expose a stable cross-suite API
for named object poses, EEF pose, gripper state, and contact events without
reaching into task-specific private simulator objects. New instrumentation is
therefore omitted. Required action/source/query/step/success logging remains
enabled; no failure taxonomy will be inferred from action traces.

## R2A: pre-authorized SmolVLA scope factorial

Eligibility decision: **R2_ENABLED_TECHNICALLY** before any R1 outcome exists.
The audited SmolVLA checkpoint/revision and method-independent per-query RNG
rule are present, disk headroom exceeds 1 TB, and the frozen budget permits at
most eight post-R1 wall-clock hours inside a 24-hour watcher window. R1 is never
terminated for R2. Immediately before R2 the watcher confirms checkpoint
loadability, no R1 failures, at least eight hours remaining, and the frozen
wall-time allowance; otherwise it records a runtime-gate skip without changing
R1 or inventing another experiment.

Use the historical scope cohort: Object task 3, Spatial task 0, Goal task 2,
LIBERO-10 task 3; states 10--19; seeds 2000--2009; audited 30 Hz environment;
SmolVLA revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`. Run `A0G0`,
`A0G20`, `A20G0`, and `A20G20` with the same-target/Fresh-prefix contract (160
episodes). Report successes/rates, `A0G20-A20G0`, discordances, paired and
task-cluster intervals where meaningful, per-suite/task heterogeneity, query
counts/rates, steps, and wall time. This is a SCOPE experiment; a null result is
valid and no tuning is allowed.

## Statistics and artifacts

Binary paired contrasts use exact two-sided McNemar and 20,000-draw percentile
bootstrap intervals with frozen seeds in `protocol.json`. Task-cluster draws
sample task policies with replacement and average their within-task paired
effects. Report per-task effects and LOTO; multi-suite families also report
per-suite and leave-one-suite-out effects. Canonical outputs are JSON and CSV
plus `FIGURE_SPEC.md`. No paper-facing PNG, PDF, or SVG is authorized.

## Fail-closed sequence

The detached watcher requires 2,700 unique Track-A completion markers, zero
unresolved failures, exact manifest/preregistration identity, the completion
mode Phase-0 validator, and the one-shot frozen Track-A analysis artifacts.
Only then may supplement canaries and R1 launch. R1A/B, R1C, and R1D each have
terminal barriers; any integrity or technical failure stops downstream launch.
R2A follows R1 only when its pre-authorized runtime gate passes. RoboTwin,
C-RTC, consensus/debounce, adaptive executors, horizon search, training,
wrong-target controls, and any outcome-invented experiment remain forbidden.
