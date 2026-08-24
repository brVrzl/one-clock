# Gate-3B cross-generation composition test — preregistered protocol

Registration date: 2026-08-24

Starting scientific commit: `eb4f6bfeb40a9d1444d3fb1d17c841601ca29a76`

Branch: `exp/gate3b-cross-generation-composition`

Status: **FROZEN BEFORE ANY OFFICIAL GATE-3B SUCCESS OUTCOME IS GENERATED OR READ**

This is a controlled 2×2 compositionality experiment. It is not a temporal
ensembling benchmark, an execution-horizon experiment, or a method search. It
asks whether independently refreshing components of a jointly predicted ACT
action chunk harms closed-loop control by recomposing arm and gripper slices
across policy source generations.

The complete exact 400-run order is part of this registration in
[`gate3b_run_schedule.json`](audit_outputs/gate3b_run_schedule.json), whose
SHA256 is `2cae2712ff00a1bc5bf8c3eb808b69aeaa8208acb62d2f8591e476e7b18ce4ff`.
The schedule contains all 400 ordered run records, not merely a generator seed.
The compact registration manifest records hashes of the protocol, schedule,
runner, composition implementation, analyzer, validator, and deterministic
tests.

## 1. Frozen question, unit, and outcome

At physical controller time `t`, query frozen ACT exactly once from the current
observation. Let the current query's prediction for time `t` be

\[
F_t = E_{t,t},
\]

and, once it exists, let the chunk queried exactly 20 controller ticks earlier
predict the same physical time through offset 20:

\[
O_t = E_{t,t-20}.
\]

The randomized treatment unit and primary inference unit are the paired
`(task_id, state_id)` blocks. Every block receives all four conditions in
randomized order. The primary outcome is binary LIBERO task success. Controller
steps and action diagnostics are repeated measurements, not independent
replicates.

## 2. Frozen system and provenance

- Checkpoint directory:
  `/home/thor/projects/checkpoints/zeromidnight_act_libero_object`.
- Model SHA256:
  `340071d7497238669459d93517eb3f8690862ad6fdf14207966759dfe6da9410`.
- Checkpoint config SHA256:
  `a76eebed357b3cbed8745c3d0f18c1335ecdd5449fcc498257676c9cbd27453d`.
- Pinned LeRobot commit:
  `f66e5128ecb2456e8c54a63d15404fa59c16aebc`.
- ACT chunk length: 100; action dimension: 7; deterministic inference; no
  policy training; policy-internal temporal ensembling disabled.
- LIBERO Object tasks 0–9; relative controller; two 256×256 RGB observations;
  `pixels_agent_pos`; hard reset; official initial states; 280-action maximum.
- Controller frequency: 20 Hz. One action/chunk index is 0.05 s and a
  100-action chunk spans 5.0 s.
- LIBERO action contract: `arm = action[0:6]`, `gripper = action[6]`.

The runner aborts before official execution if the checkpoint/config hashes,
LeRobot commit or cleanliness, policy temporal-ensemble setting, chunk/action
shape, controller frequency, episode limit, schedule hash, or official-state
count differs.

## 3. Fixed source-age contract

The single frozen age is `d=20` controller ticks, equal to 1.0 second at the
audited 20 Hz controller. It is physically interpretable, lies comfortably
inside the 100-action chunk, was not selected by optimizing Gate-3B success,
and is close to the source-age regime that was operationally relevant in
Gate-3A2 without copying a fitted weighting rule. No age sweep or sensitivity
search is authorized in this confirmatory experiment.

For `t<20`, `O_t` does not exist. Every condition therefore executes `F_t` in
full. The intervention begins only at `t>=20`, giving all four conditions an
identical 20-action prefix under identical deterministic trajectories.

## 4. Frozen 2×2 conditions

For `t>=20`, execute exactly:

- `FF` — JOINT_FRESH: `[arm(F_t), gripper(F_t)]`.
- `OO` — JOINT_OLD20: `[arm(O_t), gripper(O_t)]`.
- `FO` — ARM_FRESH_GRIP_OLD: `[arm(F_t), gripper(O_t)]`.
- `OF` — ARM_OLD_GRIP_FRESH: `[arm(O_t), gripper(F_t)]`.

Only `FO` and `OF` are cross-generation recompositions. The implementation
does not average, temporally ensemble, similarity-weight, threshold, select,
smooth, or blend either candidate. It does not change controller frequency,
action scaling, postprocessing, or gripper semantics.

## 5. Frozen cohort, episode seeds, and exact order

Gate-3A2 used state IDs `[0, 7, 11, 13, 25, 30, 36, 41, 42, 43]`. Gate-3B
samples without replacement from `{20,...,49}` after excluding those IDs. A
NumPy `default_rng(20260827)` draw, sorted only for deterministic traversal
readability, freezes:

```text
[24, 26, 28, 29, 32, 33, 37, 40, 46, 49]
```

The unsorted draw order was `[26, 37, 32, 49, 33, 28, 29, 40, 46, 24]`.
Historical per-state success was not inspected during selection. The same ten
states are used for every task. Historical tasks 1–9 primarily used low-index
states, so this higher-index cohort limits direct reuse. Task 0 historically
used all 50 official states; a completely untouched task-0 cohort does not
exist, and this registration does not describe task 0 as fresh.

Every method in block `(task_id,state_id)` uses the same deterministic episode
seed from a namespace distinct from Gate-3A2:

```text
320000 + 100 * task_id + state_id
```

Within every one of the 100 blocks, a continuing NumPy
`default_rng(20260828)` stream independently permutes `FF`, `OO`, `FO`, and
`OF`. Tasks and selected states are traversed in ascending order; treatment
order is never changed after outcome generation. The authoritative schedule
records `run_index`, task, state, shared episode seed, within-block position,
and method for all 400 episodes.

## 6. Query and execution fairness

Every condition receives the current observation, queries frozen ACT exactly
once per surviving controller step, produces the complete fresh `100×7` chunk,
and retains the same 21-source rolling cache needed to resolve `q=t-20`.
Each episode asserts:

```text
policy_queries == environment_steps
```

Closed-loop trajectories and therefore total episode steps/queries may differ.
The fairness claim is one policy query per surviving step, not equal total
episode compute. Policy, checkpoint, preprocessing, postprocessing,
environment, initial state, seed, action scaling, and gripper control remain
fixed. Only the executed source assignment differs.

## 7. Primary estimand and frozen statistics

For each paired task-state block, define

\[
C_{coherence}=\tfrac12(success_{FF}+success_{OO})
-\tfrac12(success_{FO}+success_{OF}).
\]

Positive values mean source-coherent actions outperform cross-generation
recompositions at the same marginal source-age assignment. This is the primary
estimand. The standard 2×2 interaction

\[
I=success_{FF}+success_{OO}-success_{FO}-success_{OF}=2C_{coherence}
\]

is reported without choosing between statistics after seeing results.

The primary analysis reports:

- the mean coherence contrast over the 100 paired blocks;
- ten task-level mean contrasts;
- 20,000 paired task-state bootstrap draws using seed `20260829`;
- 20,000 task-cluster bootstrap draws using seed `20261829`, resampling ten
  whole tasks and retaining all ten states within a sampled task;
- all ten leave-one-task-out mean contrasts.

Percentile 2.5% and 97.5% bounds form the 95% intervals. No frame- or step-level
pseudoreplication enters inference. The four success rates and all six
pairwise comparisons are descriptive secondary results. In particular, the
direct decompositions `FF−FO`, `OO−OF`, `FF−OF`, and `OO−FO` are secondary and
cannot replace the coherence contrast.

## 8. Frozen decision rule

- **COMPOSITION-HARM-CONFIRMED** if `C_coherence>0`, both bootstrap lower
  bounds exceed zero, and every leave-one-task-out contrast remains positive.
- **COMPOSITION-HARM-SUGGESTIVE** if the point estimate is positive, at least
  one bootstrap lower bound does not exceed zero, and every leave-one-task-out
  contrast remains positive, so no single omitted task reverses the effect.
- **COMPOSITION-HARM-CONTRADICTED** by the exact symmetric rule: the point
  estimate is negative, both bootstrap upper bounds are below zero, and every
  leave-one-task-out contrast remains negative.
- **COMPOSITION-HARM-NULL** for every other complete result, including an
  unresolved or near-zero result or one concentrated enough for an omitted
  task to remove/reverse the direction.

No percentage-point threshold is introduced. Secondary diagnostics cannot
rescue a null success result. A null result does not authorize revival of the
group-wise selective-retention method.

## 9. Secondary mechanism diagnostics

Per-step local logs record episode length, policy queries, executed actions,
translation discontinuity, SO(3) rotation-action discontinuity, raw action
acceleration and jerk, gripper transitions, fresh/old arm norm disagreement,
fresh/old translation and SO(3) disagreement, gripper-sign disagreement,
source identity for each group, and distance to `F_t` and `O_t` under the
already audited translation/SO(3)/gripper-sign decomposition. For mixed cells,
the smaller of those two distances is named **distance to the nearest jointly
predicted source action**. It is not called an off-manifold distance, because
no action manifold is learned or measured.

These diagnostics are descriptive along treatment-dependent trajectories. No
mediator or jerk-causation claim is preregistered.

## 10. Integrity, resume, and stop conditions

Synthetic tests must establish the exact `q=t-20`/offset-20 mapping, the common
fresh prefix, all four formulas, source order, absence of temporal ensembling,
and deterministic schedule/resume behavior. Runtime validation must establish
all 400 unique cells, schedule identity, file hashes, finite 7-D actions, exact
source identities/formulas, identical first 20 actions within every block, and
one query per environment step.

Completed episodes are written atomically as local gzip JSON logs. Resume
validates identity, provenance, step count, query equality, and hash before
skipping a completed cell. A malformed or provenance-mismatched artifact
aborts; it is not silently overwritten. An implementation/environment failure
stops execution and requires a dated amendment before any affected official
cell can be replaced. Interruption without invalid data is handled only by the
frozen deterministic resume order. There is no outcome-dependent interim stop.

After all 400 valid episodes, the frozen analysis and validation run once, the
bounded report and source-of-truth records are updated, results are committed
and pushed, and work stops. No consistency-constrained method, adaptive group
horizon, learned router, age sweep, PACE experiment, or additional benchmark
is authorized by this protocol.

## 11. Claim boundary

No firstness claim is made. If the gate is positive, the maximum allowed
conclusion is:

> For this frozen ACT/LIBERO system, recomposing arm and gripper components from
> different temporal source generations reduces task success relative to
> source-coherent execution at the same marginal source-age assignment.

The result cannot establish a policy-manifold violation, a universal
joint-action coherence law, harm from all component-wise adaptive execution,
causation through jerk, or generalization to dexterous, bimanual, VLA, or other
systems.
