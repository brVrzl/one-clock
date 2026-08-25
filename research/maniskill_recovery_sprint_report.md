# ManiSkill corrective-recovery sprint

## Decision

**KEEP FRAGILITY / CHANGE AGAIN.** The simulator-measured fragility family
still has a usable causal signal, and the new corrective-bridge generator is
valid at the simulator level. No LOCK is justified because the official ACT
policy-health gate did not produce a nontrivial, stable baseline. Recovery
policy comparisons were therefore not run and are not reported as method
results.

## Policy-health gate

All runs used the official ManiSkill ACT source snapshot documented in
`maniskill_act_provenance.md`, state observations, `pd_ee_pose`, PhysX CPU,
batch size 32, 30-query chunks, seed 1, and the same five evaluation seeds.
The original cached demonstrations contained 10 trajectories per task.

| run | iterations | success once | success at end |
|---|---:|---:|---:|
| PickCube original, temporal aggregation | 0–28k every 2k | 0.2 at 8k, 20k, 26k; 0 otherwise | 0 at every checkpoint |
| StackCube original, temporal aggregation | 0–28k every 2k | 0.2 at 28k; 0 otherwise | 0 at every checkpoint |
| PickCube, 14 successful demos, no temporal aggregation | 0–8k every 2k | 0 at every checkpoint | 0 at every checkpoint |

The 30k PickCube and StackCube runs completed normally. Training loss fell,
but closed-loop success was transient and never reached the requested
40–80% regime. The earlier 7k, 0/5 matched result is explained by the same
failure mode: it was evaluated before this learning curve showed a healthy
policy, not evidence against fragility. Raw TensorBoard logs and checkpoints
remain under `runs/`; the exported curve is
`results/maniskill_act_health_curve.csv`.

The official ManiSkill demo downloader was attempted for PickCube-v1 and
StackCube-v1, but the host returned `OSError: [Errno 101] Network is
unreachable` before any archive was written. This is the next health-gate
dependency. The official README recommends motion-planning demonstrations
and a delta-position control mode; the local causal cache is instead a small
scripted `pd_ee_pose` cache, so a healthy baseline has not yet been
established for this action representation.

## Corrective bridge generation

The old successful-suffix H5s are retired for training because they encode
perturbed state -> original-state action targets. The new bridge generator
uses direct bounded absolute EEF pose commands toward action `t+k`, with
`k ∈ {1,3,5}`, 3 mm object/joint perturbations, 20 mm maximum translation per
bridge command, and 0.12 rad maximum rotation per command. It then replays
the original suffix only from the future waypoint. A saved branch must be
valid, have no teleport violation, reduce EEF position error, and succeed at
the final simulator evaluation.

| task | demos | sampled states | attempts | valid | invalid | teleport violations | full successes | eligible saved branches | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PickCube-v1 | 10 | 72 | 1,296 | 1,296 | 0 | 0 | 452 (34.9%) | 177 (13.7%) | 96.4 s |
| StackCube-v1 | 10 | 120 | 2,160 | 2,160 | 0 | 0 | 1,102 (51.0%) | 594 (27.5%) | 200.5 s |
| total | 20 | 192 | 3,456 | 3,456 | 0 | 0 | 1,554 (45.0%) | 771 (22.3%) | 296.9 s |

The eligibility filter is intentionally conservative. The complete branch
files and per-attempt labels are in
`experiments/counterfactual_tournament/maniskill_recovery_branches/`.

## Equal-budget data products

For each task, RandomRecover, GoalRecover, MotionRecover, and
FragilityRecover were built with exactly 24 successful added trajectories,
plus the same 10 original trajectories. The common simulator-pool costs were
1,296 PickCube attempts and 2,160 StackCube attempts. The H5s are under
`experiments/counterfactual_tournament/act_data/recovery/`.

These are data artifacts, not policy results. ACT training was deliberately
deferred because UniformACT did not pass the health gate. Consequently the
live policy matrix has no new numerical recovery rows.

## Next exact steps

1. Obtain the official ManiSkill motion-planning archive on a networked host,
   or copy it into the local official-demo directory, and reproduce the ACT
   recipe with its matching control mode. First require stable nonzero
   UniformACT success on the same five evaluation seeds.
2. Once healthy, rebuild the four H5s from the same base-demo count and run
   the 24-branch matched test at the selected checkpoint. Only then expand to
   100/300 successful branches.
3. Only after FragilityRecover beats RandomRecover and GoalRecover, port one
   RoboTwin cached expert trajectory: restore an intermediate state, verify
   state hash and branch isolation, replay its original action suffix, and
   record success. Then add the same bounded corrective bridge and fragility
   measurement. No RoboTwin policy method is ported before that gate.
