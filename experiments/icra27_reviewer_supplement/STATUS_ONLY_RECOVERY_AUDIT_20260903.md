# Status-only recovery audit

Audit timestamp: 2026-09-03T14:56:08+08:00

Branch: `exp/icra27-crosssuite-query-allocation`

Reviewer-supplement preregistration:
`f44a7605246d4c9ea82f4d19ad61833e8fb13eb8`.

No reviewer-supplement or B3 scientific outcome payload was opened. This audit
uses frozen manifests, filenames, marker contents, technical logs, process
metadata, and filesystem metadata only.

## Technical phase status

| Phase | Frozen cells/shards | Complete | Technical failures | Attempts/retries | Phase completion | Canonical analysis |
|---|---:|---:|---:|---:|---|---|
| R1A | 1,512 | 1,512 | 0 | 0 | present | absent |
| R1B | 252 | 252 | 0 | 0 | present | absent |
| R1C | 280 | 280 | 0 | 0 | present | absent |
| R1D | 100 | 0 | 0 cell markers | 0 | absent | absent |
| B3 | 8 shards | 8 | 0 | 0 | analysis-complete marker present | three canonical files present |
| R2A | 160 | 0 | 0 | 0 | absent | absent |

For R1A, R1B, and R1C, completion-marker and result-filename identity sets
exactly equal their frozen manifest cell-ID sets. Each phase's technical
validator reported `PASS`. There are no extra or missing IDs, duplicate IDs,
zero-byte files, temporary/partial files, attempt records, retry records, or
technical-failure markers. The supplement-wide canonical analyzer has not run,
so `analysis.json`, `condition_summaries.csv`, `contrasts.csv`, and the
supplement-analysis completion marker are absent.

R1D is `TECHNICALLY_FAILED`. Its launch marker was written at
2026-09-03T14:32:54.587455483+08:00. All three workers failed during runtime
initialization, before any frozen cell was attempted. There are zero R1D
results, completion markers, technical-failure cell markers, attempt records,
retry records, zero-byte files, or temporary/partial files. The frozen 100-cell
manifest has no duplicate IDs. No R1D worker is alive.

B3 is `COMPLETE`: all eight expected task-policy shard names have matching
completion markers and the expected JSON/NPZ artifact pair. The canonical
`summary.json`, `forecast_metrics.csv`, and `report.md` exist, and
`B3_ANALYSIS_COMPLETE` is present. The launcher did not create a distinct B3
launch marker. Its recorded PID is stale and no B3 worker is alive. No B3
scientific values were opened.

## Reviewer pipeline failure

Marker:
`experiments/icra27_reviewer_supplement/orchestration/PIPELINE_FAILED`.

Marker timestamp: 2026-09-03T14:33:54.635622733+08:00. Marker stage:
`r1d_workers_exited_early`. The master exited with status 1 during R1D, after
the R1D launch marker and before any R1D cell attempt. Each worker raised
`ImportError` because the configured LeRobot source path requested
`Qwen2_5_VLTextConfig`, which is unavailable in the configured
Transformers 4.51.3 runtime. Technical logs are
`orchestration/logs/master.log` and
`orchestration/logs/r1d_worker_{0,1,2}.log` under the reviewer-supplement
directory.

## R2A frozen runtime gate

R2A was prospectively marked `R2_ENABLED_TECHNICALLY`, subject to the final
runtime gate: at most eight post-R1 hours inside a 24-hour watcher window,
implemented as launch only when elapsed time from the original pipeline epoch
is at most 57,600 seconds. The preserved epoch is `1788354953`, corresponding
to 2026-09-02T21:15:53+08:00. At audit time, elapsed time was 63,630 seconds,
so the unchanged rule resolves R2A as `INELIGIBLE_BY_ORIGINAL_RUNTIME_GATE`.
R2A was not launched and has no result, completion, failure, attempt, retry, or
worker artifact. The runtime-skip marker is absent because the pipeline failed
at R1D before reaching the R2A gate.

## Runtime integrity

No reviewer, B3, or R2A scientific process is alive. NVIDIA reports no compute
process on GPUs 0--2. Across R1D, B3, and R2A there are no duplicate scientific
IDs, unexpected result multiplicity, zero-byte/incomplete results, stale
temporary files, or scientific retry artifacts. The R1D PID files and B3 PID
file are stale records of exited processes. The R1D phase launch lock is also
stale and must be cleared only as part of an explicitly reviewed technical
resume.

## Integrity-safe R1D resume disposition

No currently executable resume is safe because the pinned R1D import canary
fails in the configured interpreter. R1D is resumable in principle after a
technical-only environment repair that changes no governed scientific file.
After such a repair, the following gated command is the exact queue-level
resume; its first command must pass before it can clear the stale phase lock:

```bash
cd /home/wjq/workspace/one-clock-icra27-crosssuite-query-allocation && \
PYTHONDONTWRITEBYTECODE=1 /home/wjq/workspace/venvs/libero_act/bin/python -c 'import sys; sys.path.insert(0, "/home/wjq/workspace/upstreams/lerobot/src"); from lerobot.configs.policies import PreTrainedConfig' && \
test "$(<experiments/icra27_reviewer_supplement/orchestration/pipeline_start_epoch)" = 1788354953 && \
test "$(find experiments/icra27_reviewer_supplement/results/r1d -maxdepth 1 -type f 2>/dev/null | wc -l)" -eq 0 && \
test "$(find experiments/icra27_reviewer_supplement/markers/r1d -maxdepth 1 -type f 2>/dev/null | wc -l)" -eq 0 && \
rmdir experiments/icra27_reviewer_supplement/orchestration/r1d_LAUNCH_LOCK && \
bash experiments/icra27_reviewer_supplement/launch_watcher.sh --resume
```

The frozen runner validates a result plus completion marker before skipping a
cell. In the audited zero-artifact R1D state, the preflight prevents overwriting
an orphan result and the resumed phase can execute only the 100 missing frozen
IDs. R1A/R1B/R1C are already complete and B3 requires no resume. R2A is
ineligible under the original epoch and must not be resumed.

## Public-repository submission risk

GitHub reports `brVrzl/one-clock` as a public repository. Public exposure is
present. `paper/icra2027/main.tex` and section-level manuscript text are on the
following public remote branches:

- `exp/fast5080-adaptive-recency`
- `exp/fast5080-cross-generation-offline`
- `exp/gate3b-cross-generation-composition`
- `exp/gate3c-asymmetric-temporal-reuse`
- `exp/gate4a-spatial-generalization`
- `exp/gate4a2-prefix-rootcause`
- `exp/gate4a2-spatial-act-generalization`
- `exp/groupwise-selective-commit-act`
- `exp/icra27-care-final-gate`
- `exp/icra27-chunkfix-fast`
- `exp/icra27-crosssuite-query-allocation`
- `exp/icra27-overnight-smolvla-crosspolicy`
- `exp/icra27-post-gate3c-5080`
- `exp/icra27-two-clock-discriminator`
- `exp/libero-component-temporal-reuse`
- `exp/robotwin-dcta-development`
- `exp/robotwin-exploratory-sealed`
- `exp/robotwin-static-validation`
- `main`
- `ops/overnight-5080-prep-20260824`
- `ops/thor-to-5080-handoff`
- `paper/icra27-final-claim-freeze`
- `preserve/integrate-5080-into-main-20260826`
- `research/icra27-direction-reset`
- `research/libero-baseline-handoff-20260826`
- `research/robotwin-paused-20260826`

`paper/icra2027/main.pdf` is publicly present on these remote branches:

- `exp/gate4a-spatial-generalization`
- `exp/gate4a2-prefix-rootcause`
- `exp/gate4a2-spatial-act-generalization`
- `exp/icra27-post-gate3c-5080`
- `exp/robotwin-dcta-development`
- `exp/robotwin-exploratory-sealed`
- `ops/overnight-5080-prep-20260824`
- `research/icra27-direction-reset`
- `research/libero-baseline-handoff-20260826`
- `research/robotwin-paused-20260826`

No repository visibility, branch, manuscript file, or PDF was changed.
