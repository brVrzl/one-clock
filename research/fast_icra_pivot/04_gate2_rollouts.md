# Gate 2: exploratory three-task rollouts

## Frozen task subset

Tasks 1, 6, and 8 were chosen before this screen from historical global-horizon-8 performance: nonzero and not uniformly saturated. The screen uses official initial states 0–4 and seeds 1000–1004 for every method. It is exploratory, not final evaluation. Every completed episode is retained in JSONL.

Common configuration: LIBERO-Object, frozen `zeromidnight_act_libero_object` ACT, global execution horizon 8, 100-step chunks, relative 7-D actions, clip to `[-1,1]` after the post-policy module, NVIDIA RTX 5080.

## Results

| Task | Method | Episodes | Success | Wilson 95% CI | Rollout wall time |
|---|---|---:|---:|---:|---:|
| 1 cream cheese | Frozen ACT | 5 | 3/5 (60%) | [23.1%, 88.2%] | 29.6 s |
| 1 cream cheese | EMA α=0.25 | 5 | 1/5 (20%) | [3.6%, 62.4%] | 36.4 s |
| 1 cream cheese | Affine full + q25 gate | 5 | 1/5 (20%) | [3.6%, 62.4%] | 28.9 s |
| 1 cream cheese | Gripper-only affine | 5 | 0/5 (0%) | [0.0%, 43.4%] | 46.3 s |
| 6 butter | Frozen ACT | 5 | 5/5 (100%) | [56.6%, 100%] | 29.3 s |
| 6 butter | EMA α=0.25 | 5 | 2/5 (40%) | [11.8%, 76.9%] | 36.2 s |
| 6 butter | Affine scale 0.25 | 5 | 3/5 (60%) | [23.1%, 88.2%] | 36.2 s |
| 6 butter | Gripper-only affine | 5 | 1/5 (20%) | [3.6%, 62.4%] | 37.2 s |
| 8 chocolate pudding | Frozen ACT | 5 | 1/5 (20%) | [3.6%, 62.4%] | 31.8 s |
| 8 chocolate pudding | EMA α=0.25 | 5 | 0/5 (0%) | [0.0%, 43.4%] | 39.4 s |
| 8 chocolate pudding | Affine scale 0.25 | 5 | 0/5 (0%) | [0.0%, 43.4%] | 37.7 s |
| 8 chocolate pudding | Gripper-only affine | 5 | 1/5 (20%) | [3.6%, 62.4%] | 39.3 s |

Rollout wall time is summary-file modification time minus output-directory filesystem birth time. It begins after model/environment setup and includes episodes, logging, and environment close.

Across the common frozen/EMA comparison, frozen ACT is 9/15 and EMA is 3/15. The sequential affine configurations total 4/15 versus the same frozen 9/15, though task 1 used full gated correction while tasks 6/8 used scale 0.25 after the full correction was killed. Gripper-only affine is 2/15. These small samples have wide intervals, but every tested correction is directionally worse and the degradation is large enough to stop.

## Failure modes

- Offline q25 gating activated on every online query, so it did not protect already-good chunks.
- Full affine corrections were large: mean chunk correction norm 6.84 on task 1.
- Conservative 0.25 corrections still reduced task 6 from 5/5 to 3/5 and task 8 from 1/5 to 0/5.
- EMA reduced offline MSE but damaged all three closed-loop tasks, showing that expert-action MSE and smoothness are unsafe surrogate objectives.
- Gripper-only correction left the arm untouched but still changed grasp/release timing enough to degrade task 1 and task 6.

Machine-readable aggregate with per-episode outcomes: `artifacts/gate2_screen_summary.json`.

## Decision

Continuous post-policy residual repair is killed for this checkpoint/benchmark. No 20–30 episode expansion is warranted. The next branch is discrete, structured gripper-timing candidates with a conservative selector and an explicit no-change bias; it must pass a fresh held-out offline protocol before any rollout.
