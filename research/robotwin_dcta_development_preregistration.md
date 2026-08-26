# RoboTwin DCTA method-development preregistration

This document freezes the first closed-loop development evaluation of Dynamic
Component-wise Temporal Aggregation (DCTA). It does not alter the completed
hard-FO exploratory pilot at result commit
`c32defc222d6d7fc794d2c2aa860ac230570287f`.

## Baseline resolution

Pinned XPolicyLab commit `c37109c500be67d0dea6b36bf7337bbd26e763cd`
does not write `policy_best.ckpt`; the corresponding code is commented out.
Its ACT evaluation class explicitly loads `policy_last.ckpt`. A three-seed,
five-decision trace comparison on `handover_block` found exact equality between
the pinned official evaluation path and the prior pilot's `NATIVE_ACT` path
through observation conversion, qpos normalization, full predicted chunk,
temporal candidate set and weights, denormalized action, and action sent.
Retraining is therefore not part of this development run.

## Frozen policy and gate

The five existing task-specific seed-0, 6000-epoch `policy_last.ckpt` ACT
policies remain frozen. The ACT backbone is never updated by DCTA training.

The shared-dynamic and DCTA gates use the same 40,835-parameter architecture:
a shared MLP, learned group embedding, and a localized 512-D frozen ACT decoder
query-0 feature. Candidate features include decision lag, simulator/demo source
age, candidate action, difference from the newest candidate, cross-candidate
disagreement, normalized qpos, and group identity. The residual head was
initialized to zero, which numerically reproduces native ACT aggregation.

Both gates were fit only on official `demo_clean` trajectories 0--39 from all
five tasks. Trajectories 40--49 were held out for early stopping. The objective
is the mean of four per-group action MSEs so scalar grippers and 6-D arms have
equal group weight. No rollout reward or success was used in fitting or model
selection.

Frozen gate identities:

- `SHARED_DYNAMIC_AGG`: `9ce7e1984cc08ca430dcbe55c4f7c8816cc6943404a412196aa458f18d2af513`
- `DCTA`: `c0257a966b40b43e7b7a3b736815f7f7ace78d1ba9480a5d0e42b6d03e6e879b`

## Methods

1. `NATIVE_ACT`: unchanged pinned official ACT temporal aggregation.
2. `SHARED_DYNAMIC_AGG`: learned residual temporal logits, with one identical
   temporal distribution forced across all four semantic groups.
3. `DCTA`: learned residual temporal logits separately applied to left arm,
   left gripper, right arm, and right gripper using shared gate parameters and
   learned group embeddings.

At decision target `t`, all methods aggregate candidates
`chunk_from_query_q[t-q]`; no candidate uses a future query. DCTA logits are
`log(w_ACT) + r_theta`, masked over available history and normalized over
candidate sources.

## Development matrix

Tasks are fixed as `beat_block_hammer`, `click_alarmclock`, `dump_bin_bigbin`,
`handover_block`, and `open_laptop`. Each task uses the same 20 expert-eligible
seeds already frozen before the hard-FO pilot. All three methods share each
task/seed block. Method order is randomized within blocks using generator seed
`20270827`.

The matrix contains 5 tasks x 20 seeds x 3 methods = 300 cells. The frozen
machine-readable schedule is
`research/audit_outputs/robotwin_dcta_development_schedule.json`, with cell-list
SHA256 `c9ce4651be45a1c21d95e75f966d0464c40cf50c4a2c4f1c5036c221f3af4178`.

## Analysis

Primary development contrasts are `DCTA - NATIVE_ACT` and
`DCTA - SHARED_DYNAMIC_AGG`. The secondary diagnostic is
`SHARED_DYNAMIC_AGG - NATIVE_ACT`. Report pooled paired success difference,
paired wins/losses/ties, task-level differences, a task-cluster bootstrap
interval with 10,000 draws, leave-one-task-out estimates, and rollout effective
source age by semantic group. No confirmatory p-value is claimed.

The full 300-cell matrix is evaluated before success-guided architecture or
hyperparameter changes. A technical failure may rerun only the same frozen
task/method/seed cell.
