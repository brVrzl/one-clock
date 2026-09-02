# ICRA 2027 one-clock durable handoff

Snapshot time: 2026-09-02 (Asia/Shanghai). This is a technical-progress
snapshot only; no partial Track-A success outcome was inspected.

## Repository identity

- Worktree: `/home/wjq/workspace/one-clock-icra27-crosssuite-query-allocation`
- Branch: `exp/icra27-crosssuite-query-allocation`
- Parent HEAD immediately before this handoff commit:
  `e82e5ccc7979e45b005aab9355e2a3839be85133`; use `git rev-parse HEAD` for the
  handoff-inclusive branch tip.
- Frozen Track-A preregistration: `40549d876c0e09fad4e8033b3206f6018f53ece5`
- Track-B analysis-addendum commit: `6344960b0cb1164a389eebf2a14927f8fc92cc5f`
- Frozen B3 forecast implementation/manifest commit: `94657b54591fd1305e8ac888a0c05beb4de2c2cb`
- Reviewer-supplement preregistration: not created; supplement work is forbidden
  until Track A completes and its frozen analysis passes integrity checks.

## Track B

- Rollout: complete, 80/80 artifacts, zero technical failures.
- Historical worker PID 3624661: exited normally; do not restart Track B.
- ACT_LOCALIZATION_PASS: no.
- CROSS_POLICY_MECHANISM_SUPPORT: no.
- B1 per-dimension analysis: complete.
- B2 demonstration persistence: complete; training data, not held out.
- B3 frozen-policy forecast: frozen but pending; launch only after all 2,700
  Track-A markers exist and all Track-A processes have exited. The runner also
  fails closed on these conditions.
- B4: `FAILURE_MODE_CLASSIFICATION_NOT_IDENTIFIABLE_FROM_EXISTING_ARTIFACTS`.

## Track A

- Frozen workload: 30 tasks x 15 states x six conditions = 2,700 episodes.
- Conditions: H16, H4, ARM4_GRIP32, H2, ARM2_GRIP16, TE_DENSE.
- Snapshot counts: 1,293 complete, 3 running, 1,404 not currently running;
  equivalently the status tool reports 1,407 not yet complete because running
  cells are included in its `pending` counter.
- Technical failures: 0.
- Attempt/retry artifacts: 0. Every current cell is on initial attempt 1.
- Duplicate/overwritten cells: none found in the recovery-integrity audit.
- Manifest/seeds/checkpoints/condition order: unchanged from preregistration.
- Scheduling remains task-major; all six conditions for each selected state run
  while its task-specific checkpoint remains loaded, followed by full teardown.

Original detached workers (do not duplicate while alive):

| Worker | GPU | PID | Log | Progress marker |
|---|---:|---:|---|---|
| 0 | 0 | 3796812 | `experiments/icra27_crosssuite_query_allocation/track_a/logs/worker_0.log` | `track_a/progress/worker_0.json` |
| 1 | 1 | 3796813 | `experiments/icra27_crosssuite_query_allocation/track_a/logs/worker_1.log` | `track_a/progress/worker_1.json` |
| 2 | 2 | 3796814 | `experiments/icra27_crosssuite_query_allocation/track_a/logs/worker_2.log` | `track_a/progress/worker_2.json` |

Completion markers are under `track_a/markers/`; incremental result files are
under `track_a/results/`. The watcher completed successfully and left
`orchestration/TRACK_A_AUTOLAUNCH_COMPLETE`. Its log is
`orchestration/logs/launch_after_b.log`.

## Track C and reviewer supplement

- RoboTwin/Track C is no longer on the ICRA critical path. No SAPIEN/Vulkan
  process is running. Do not start Track C.
- R1 and R2 are not started and have no preregistration. Do not create or launch
  them until Track A finishes and the frozen Track-A analysis is complete.
- No adaptive executor, consensus, debounce, RTC/PACE reproduction, horizon
  search, or new condition is authorized.

## Paper-support artifacts completed

- `PAPER_STYLE_BENCHMARK.md`
- `ONE_CLOCK_STYLE_GUIDE.md`
- `RELATED_WORK_FACT_CHECK.md`
- `PAPER_REVISION_INPUTS.md`
- interaction robustness data/report under `interaction_robustness/`

`FIGURE_SPEC.md` remains pending because Figure 4 must be chosen only after the
complete Track-A and cross-policy results exist. No paper-facing figure artwork
has been generated. The manuscript and `CLAIMS.md` have not been edited.

## Exact next commands

Technical monitor (safe; does not read success values):

```bash
cd /home/wjq/workspace/one-clock-icra27-crosssuite-query-allocation
/home/wjq/workspace/venvs/libero_act/bin/python \
  experiments/icra27_crosssuite_query_allocation/track_a_status.py
for p in 3796812 3796813 3796814; do
  ps -p "$p" -o pid=,stat=,etimes=,cmd=
done
```

Resume only if technical inspection proves all three original workers have
exited unexpectedly and the queue is incomplete. The launcher reuses the exact
sealed manifest and skips cells with existing result+completion markers:

```bash
cd /home/wjq/workspace/one-clock-icra27-crosssuite-query-allocation
bash experiments/icra27_crosssuite_query_allocation/launch_track_a.sh
```

Do not run that resume command while any original worker is alive. After 2,700
completion markers and zero technical failures, run the existing frozen
Track-A analyzer, then launch the already-frozen B3 forecast. Do not inspect
partial Track-A scientific outcomes before global completion and validation.
