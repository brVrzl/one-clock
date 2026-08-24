# Gate-3C asymmetric temporal reuse — preregistered protocol

Registration date: 2026-08-24

Scientific parent: `2817411a4210b8611dc8dae5d32ec99fc6b94cf3`

Branch: `exp/gate3c-asymmetric-temporal-reuse`

Status: **FROZEN BEFORE ANY OFFICIAL GATE-3C SUCCESS OUTCOME IS GENERATED OR READ**

Gate-3B's `C_ASYMMETRIC_FO20` cell succeeded in 62/100 episodes, but that cell
pattern was post-hoc under Gate-3B's symmetric preregistration. It is
development evidence only. Gate-3C is a new-state confirmation of the frozen
directional hypothesis and a matched comparison with two frozen full-action
temporal baselines. It is not a parameter search.

The complete ordered 700-run schedule is
[`gate3c_run_schedule.json`](audit_outputs/gate3c_run_schedule.json), SHA-256
`55c0fc25e830c9084d114bfe7f4639a944e76514a4715b059844dd5a26bc87f4`.
It contains each ordered run record, not only RNG seeds.

## 1. System, unit, and outcome

The randomized treatment and primary inference unit is a paired
`(task_id,state_id)` block. Every block receives all five methods. The primary
outcome is binary task success. Controller steps are repeated measurements,
not independent replicates.

Frozen provenance:

- ACT checkpoint: `/home/thor/projects/checkpoints/zeromidnight_act_libero_object`.
- Model SHA-256: `340071d7497238669459d93517eb3f8690862ad6fdf14207966759dfe6da9410`.
- Checkpoint config SHA-256: `a76eebed357b3cbed8745c3d0f18c1335ecdd5449fcc498257676c9cbd27453d`.
- Preprocessor SHA-256: `e7e3815a9e23eabe88e3dc5697cbccf8c59e61b59cf916d947dd673123426450`.
- Postprocessor SHA-256: `c27cf6f42b42352f9b8f9c40da155fd4459e0ee9b85b9f23072941eb52b3ffb5`.
- Normalizer SHA-256: `3cb90679b116d22c960772f75e567c32b51778df2ca065cc4784bd6cd593e941`.
- Rollout config SHA-256: `81f07ffc0740500c1c70be477e83471c7da50ca40bd50536f368aa4ba6ed9d54`.
- LeRobot commit: `f66e5128ecb2456e8c54a63d15404fa59c16aebc`, required clean.
- LIBERO Object tasks 0–9, official initial states, 20 Hz, relative controller,
  280-action maximum, deterministic frozen policy, chunk length 100, action
  dimension 7, no policy-internal temporal ensemble, no action smoothing.
- LIBERO action contract: arm `action[0:6]`; gripper `action[6]`.

Every method receives the current observation, queries ACT exactly once per
surviving controller step, produces and caches the same complete `100×7`
chunk, and uses identical preprocessing/postprocessing. Every completed
episode must satisfy `policy_queries == environment_steps`.

## 2. Frozen candidates and methods

At physical step `t`, define the fresh candidate `F_t=E_{t,t}`. For `t>=20`,
define `O_t=E_{t,t-20}`, taken from source query `q=t-20` at chunk offset 20
and targeting the same physical time. At 20 Hz, `d=20` is 1.0 second. It is
frozen because it generated the Gate-3B developmental observation; Gate-3C
does not test another age.

Execute exactly:

- **A_NEWEST (A/FF):** `F_t`.
- **B_FULL_OLD20 (B/OO20):** `F_t` for `t<20`; `O_t` for `t>=20`.
- **C_ASYMMETRIC_FO20 (C/FO20):** `F_t` for `t<20`; then
  `[F_t[0:6], O_t[6]]`.
- **D_AGE_EXP_B003 (D):** over every valid full-action prediction for time
  `t`, use one shared scalar vector `w_q proportional to exp(-0.03*(t-q))`
  over all seven action dimensions.
- **E_COGACT_A03 (E):** released full-action cosine weighting with frozen
  `alpha=0.3`, one shared scalar vector over all seven dimensions, exactly as
  used in Gate-3A2.

C is not a 20-step gripper hold: its gripper command may change every control
tick, while the prediction source observation stays 20 ticks older. No age,
threshold, partition, dynamic horizon, semantic weight, learned selector,
PACE rule, gripper vote, or exact ACT `m=+0.01` condition is added.

## 3. Outcome-blind cohort and schedule

The identity-only historical audit is frozen in
[`gate3c_state_usage_audit.md`](gate3c_state_usage_audit.md). Tasks 1–9 share
exactly 14 genuinely unused official IDs:

```text
[20, 21, 22, 23, 27, 31, 34, 35, 38, 39, 44, 45, 47, 48]
```

All 14 are used because the count lies between 10 and 15; reserved state seed
`20260830` is therefore not consumed. Historical success outcomes were not
inspected. Task 0 historically used all 50 IDs and is explicitly secondary,
not an untouched-state confirmation.

Primary scope is tasks 1–9 (`126` paired blocks; `630` episodes). Secondary
scope is task 0 on the same states (`14` blocks; `70` episodes). The fixed
episode seed is `330000 + 100*task_id + state_id`, shared by all five methods
in a block. A continuing `numpy.default_rng(20260831)` stream permutes methods
within blocks traversed by ascending task and state. The frozen schedule has
140 blocks and 700 episodes. It will not be reduced after partial outcomes.

## 4. Frozen hypotheses and statistics

The four and only four primary comparisons are:

- H1-arm: `C-B > 0`.
- H1-gripper: `C-A > 0`.
- H2-age: `C-D > 0`.
- H2-cog: `C-E > 0`.

For each comparison, the analyzer reports success counts/rates, paired block
difference, task-wise differences, every leave-one-primary-task-out estimate,
an exact two-sided McNemar/binomial diagnostic on discordant paired blocks,
20,000 paired-block bootstrap draws, and 20,000 task-cluster bootstrap draws.
The confirmatory analysis uses tasks 1–9 only. Percentile 2.5% and 97.5%
bounds form 95% intervals. Bootstrap seeds by comparison `(C-A,C-B,C-D,C-E)`
are respectively `20260901..20260904` for paired blocks and
`20261901..20261904` for primary task clusters. All-ten-task sensitivity uses
paired seeds `20262901..20262904` and cluster seeds `20263901..20263904`.
No controller-step or frame pseudoreplication enters success inference.

A difference is **stable positive** iff its primary paired-bootstrap lower
bound is above zero, primary task-cluster lower bound is above zero, and every
primary leave-one-task-out estimate is above zero. Stable negative uses the
exact symmetric rule: both upper bounds and every leave-one-task-out estimate
are below zero.

## 5. Frozen gate decision

Decision precedence is:

1. **ASYMMETRIC-REUSE-NEGATIVE** if either directional comparison `C-A` or
   `C-B` is stable negative.
2. **ASYMMETRIC-REUSE-STRONG** if all four comparisons are stable positive.
3. **ASYMMETRIC-REUSE-SUPPORTED** if `C-A` and `C-B` are stable positive, C is
   numerically no worse than both D and E, and neither baseline comparison is
   stable negative.
4. **ASYMMETRIC-REUSE-BASELINE-LIMITED** if both directional comparisons are
   stable positive but C is numerically worse than at least one of D or E (a
   stable-negative baseline contrast is a stronger instance of this case).
5. **ASYMMETRIC-REUSE-NULL** for every other complete result: the new-state
   cohort does not stably reproduce both directional effects and neither
   directional comparison reverses stably.

The prompt's NULL line was textually truncated; item 5 records the only
non-overlapping residual interpretation before outcomes. Labels will not be
altered. No percentage-point threshold will be introduced.

## 6. Diagnostics, integrity, and stopping

Local step logs record effective arm/gripper ages, gripper transitions,
episode length, translation and SO(3) action discontinuity, raw acceleration
and jerk, and F/O gripper-sign, translation, and rotation disagreement.
Diagnostics are secondary along treatment-dependent trajectories and cannot
rescue success or establish jerk as a mechanism.

Synthetic tests and post-result validation require exact `q=t-20`/offset-20
mapping, A/B/C formulas, the common A/B/C fresh prefix, source order, shared
full-action scalar weighting for D/E, no policy temporal ensemble or smoothing,
deterministic schedule/resume identity, finite 7-D actions, file hashes,
complete coverage, and one query per environment step.

Execution stops before outcomes on any provenance/time-contract mismatch.
Resume accepts only identity- and hash-consistent atomic logs. Official
analysis waits for all 700 scheduled episodes; no partial-result decision,
method change, excluded episode, or outcome-triggered rerun is authorized.
After the final report and bounded source-of-truth updates, Gate-3C stops all
method search; the next project phase is manuscript completion.
