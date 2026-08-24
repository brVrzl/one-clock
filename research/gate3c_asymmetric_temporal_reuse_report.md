# Gate-3C asymmetric temporal reuse confirmatory report

Experiment date: 2026-08-24

Scientific parent: `2817411a4210b8611dc8dae5d32ec99fc6b94cf3`

Preregistration/code/schedule commit:
`7de8971df56480f3d9982b8a8714d910f630f8b6`

Frozen schedule SHA-256:
`55c0fc25e830c9084d114bfe7f4639a944e76514a4715b059844dd5a26bc87f4`

Decision: **ASYMMETRIC-REUSE-STRONG**

## 1. Confirmatory result

On the primary untouched-state scope—tasks 1–9, 14 official states per task—
the frozen asymmetric condition succeeded in `80/126` episodes (63.5%). The
success counts were `53/126` (42.1%) for newest full action, `55/126` (43.7%)
for full old20, `62/126` (49.2%) for the frozen newest-age exponential, and
`59/126` (46.8%) for tuned CogACT.

All four preregistered C-minus-comparator contrasts are stable positive: both
20,000-draw bootstrap lower bounds exceed zero and every primary
leave-one-task-out estimate remains positive.

| Frozen comparison | Difference | Paired-block 95% CI | Task-cluster 95% CI | Discordant C-only / comparator-only | Exact two-sided McNemar/binomial p | Stable positive |
|---|---:|---:|---:|---:|---:|---|
| H1-gripper: C−A | +.2143 | `[+.1429,+.2937]` | `[+.1270,+.3095]` | 28 / 1 | `1.12e-7` | yes |
| H1-arm: C−B | +.1984 | `[+.0952,+.3016]` | `[+.0635,+.3175]` | 36 / 11 | `.000346` | yes |
| H2-age: C−D | +.1429 | `[+.0476,+.2381]` | `[+.0159,+.2778]` | 29 / 11 | `.00643` | yes |
| H2-cog: C−E | +.1667 | `[+.0714,+.2619]` | `[+.0397,+.2937]` | 31 / 10 | `.00145` | yes |

The McNemar/binomial values are exact diagnostics; the frozen stability rule
uses the paired bootstrap, task-cluster bootstrap, and leave-one-task-out
criteria. No controller frame is treated as an independent success replicate.

## 2. Design and outcome-blind cohort

Gate-3B's 62% FO cell was post-hoc developmental evidence under a different
symmetric preregistration. Gate-3C froze its directional hypothesis before new
outcomes and did not tune age, threshold, task, group partition, or baseline.

The historical state-identity audit did not inspect success outcomes. For every
task 1–9 it found the same 14 genuinely unused official IDs:

```text
[20, 21, 22, 23, 27, 31, 34, 35, 38, 39, 44, 45, 47, 48]
```

All 14 were used because the count was between 10 and 15. Task 0 historically
used all 50 official states, so its 14 matched episodes per method are a
preregistered secondary sensitivity only, not untouched-state confirmation.
Every one of 140 task-state blocks received all five methods with the same
episode seed and a method order frozen by `default_rng(20260831)`.

At physical step `t`, the candidate used:

```text
F_t = E_{t,t}
O_t = E_{t,t-20}
C_t = [F_t[0:6], O_t[6]]  for t>=20
C_t = F_t                 for t<20
```

`O_t` is the offset-20 prediction for the same physical target time, not a
20-step gripper hold. The gripper command can change at every controller tick.
D and E used their frozen full-action scalar weights over all seven dimensions.

## 3. Success rates by scope

| Method | Primary tasks 1–9 | Secondary task 0 | All ten sensitivity |
|---|---:|---:|---:|
| A — newest | 53/126 (42.1%) | 10/14 (71.4%) | 63/140 (45.0%) |
| B — full old20 | 55/126 (43.7%) | 4/14 (28.6%) | 59/140 (42.1%) |
| C — asymmetric FO20 | 80/126 (63.5%) | 11/14 (78.6%) | 91/140 (65.0%) |
| D — age exponential `beta=.03` | 62/126 (49.2%) | 10/14 (71.4%) | 72/140 (51.4%) |
| E — CogACT `alpha=.3` | 59/126 (46.8%) | 10/14 (71.4%) | 69/140 (49.3%) |

The all-ten sensitivity is directionally consistent. Its C-minus-comparator
differences and paired/task-cluster intervals are respectively:

- C−A: +.2000, `[+.1286,+.2714]`, `[+.1214,+.2929]`;
- C−B: +.2286, `[+.1286,+.3286]`, `[+.1000,+.3500]`;
- C−D: +.1357, `[+.0429,+.2286]`, `[+.0214,+.2571]`;
- C−E: +.1571, `[+.0643,+.2500]`, `[+.0500,+.2786]`.

Task 0 contributes no claim of state novelty.

## 4. Task heterogeneity and leave-one-task-out checks

Task-wise primary differences, in task order 1–9:

| Task | C−A | C−B | C−D | C−E |
|---:|---:|---:|---:|---:|
| 1 | +.3571 | +.2857 | +.2143 | +.2143 |
| 2 | +.2143 | +.2857 | +.5000 | +.4286 |
| 3 | +.2143 | −.2143 | −.1429 | .0000 |
| 4 | +.5000 | +.1429 | +.1429 | +.0714 |
| 5 | +.2857 | .0000 | −.0714 | −.0714 |
| 6 | +.0714 | +.1429 | −.0714 | −.0714 |
| 7 | +.1429 | +.4286 | +.2143 | +.2143 |
| 8 | +.0714 | +.2857 | +.3571 | +.5000 |
| 9 | +.0714 | +.4286 | +.1429 | +.2143 |

The effect is heterogeneous; individual task reversals are visible and are not
hidden by the gate label. Nevertheless, every aggregate estimate after
omitting one primary task stays positive:

| Omitted task | C−A | C−B | C−D | C−E |
|---:|---:|---:|---:|---:|
| 1 | +.1964 | +.1875 | +.1339 | +.1607 |
| 2 | +.2143 | +.1875 | +.0982 | +.1339 |
| 3 | +.2143 | +.2500 | +.1786 | +.1875 |
| 4 | +.1786 | +.2054 | +.1429 | +.1786 |
| 5 | +.2054 | +.2232 | +.1696 | +.1964 |
| 6 | +.2321 | +.2054 | +.1696 | +.1964 |
| 7 | +.2232 | +.1696 | +.1339 | +.1607 |
| 8 | +.2321 | +.1875 | +.1161 | +.1250 |
| 9 | +.2321 | +.1696 | +.1429 | +.1607 |

## 5. Execution and integrity

All 700 scheduled episodes completed; no episode was excluded or rerun. They
contain 145,272 environment steps and exactly 145,272 policy queries. The
post-result validator independently established:

- 700 unique scheduled task-state-method cells;
- exact checkpoint/config, clean pinned LeRobot, and schedule hashes;
- one current-observation ACT query per surviving 20 Hz step;
- exact `q=t-20` and chunk-offset-20 identity for every old candidate;
- exact A/B/C executed formulas with maximum error zero;
- identical first 20 A/B/C actions within all 140 blocks, maximum difference
  zero;
- normalized shared full-action scalar weights for D and E;
- finite seven-dimensional actions, no policy temporal ensemble, and no action
  smoothing.

Compact artifacts:

- rollout manifest SHA-256:
  `618bd8cb9340cf459f670c4911ed082827aa515d6914713a7665f3c9f09b2984`;
- success summary SHA-256:
  `cc9662355f6debc4b23b6b5117882fc91476c80a9c87cd47d2f7fb44d7222ece`;
- pairwise table SHA-256:
  `97ac570bea6e5f71a21b2688cff4418edd29a627c452f1be6fa52d1e349d6c41`;
- per-task table SHA-256:
  `b251480abba3ee28edf7288d5ca45a11d124e0495f1b2efaaf092566da09b6bd`;
- validation output SHA-256:
  `dc9bccd67ffc8f63bd77de98033585fbb21d5866fb534e5a97e1dd5b606f77ec`.

The 700 large compressed traces remain local at
`/home/thor/projects/one-clock/experiments/gate3c_asymmetric_temporal_reuse`,
totaling 60,101,395 bytes. Their content-tree SHA-256 is
`0df106c267c4651ac50182829e85e00a8e2791e68f44982b879facd7df506403`.

## 6. Secondary diagnostics

These episode-averaged all-ten-task measurements describe
treatment-dependent trajectories and do not determine the gate:

| Method | Arm age (ticks) | Gripper age (ticks) | Gripper transitions | Translation delta L2 | SO(3) delta (rad) | Raw jerk L2 |
|---|---:|---:|---:|---:|---:|---:|
| A | .00 | .00 | 3.67 | .0495 | .00544 | .1566 |
| B | 17.88 | 17.88 | 9.02 | .0784 | .00695 | .2490 |
| C | .00 | 17.55 | 4.54 | .0633 | .00669 | .1723 |
| D | 21.53 | 21.53 | 2.78 | .0450 | .00410 | .0203 |
| E | 34.85 | 34.85 | 2.42 | .0449 | .00417 | .0330 |

Mean effective fixed age is below 20 for B/C because every episode begins with
the preregistered full-fresh `t<20` prefix. These descriptive differences do
not establish mediation or causation through jerk, discontinuity, or gripper
transitions.

## 7. Interpretation and stop

The bounded supported interpretation is:

> For this frozen ACT/LIBERO system, the action-chunk policy benefits from
> asymmetric temporal source use: current observation-conditioned arm motion
> together with temporally older gripper predictions for the same physical
> target time.

The result concerns source-observation age while querying the policy every
surviving step. It is not evidence for reduced query frequency, a universal
arm-fresh/gripper-memory law, dexterous or bimanual generalization,
non-Markovianity as the proven mechanism, or jerk causation. Task-level
heterogeneity also remains material.

Gate-3C ends method experimentation. No age sweep, selector, adaptive horizon,
new group partition, PACE variant, or additional benchmark is launched. The
next phase is manuscript completion.
