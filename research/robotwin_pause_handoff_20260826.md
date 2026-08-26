# RoboTwin pause handoff — 2026-08-26

## Frozen scientific results

The frozen 600-cell RoboTwin exploratory pilot remains **NO_SIGNAL**. Its
result commit is `c32defc222d6d7fc794d2c2aa860ac230570287f`; the later frozen
diagnostic result commit is `f36be2d`. The classification and all outcomes are
historical and must not be changed.

Pooled success (100 paired task-seed blocks per method):

| Method | Success | Rate |
|---|---:|---:|
| NATIVE_ACT | 19/100 | 19% |
| NEWEST | 11/100 | 11% |
| FO_1S | 11/100 | 11% |
| FULL_OLD_1S | 9/100 | 9% |
| GRIPPER_HOLD | 5/100 | 5% |
| GRIPPER_EMA_1S | 10/100 | 10% |

The preregistered primary contrast was FO_1S minus NEWEST: pooled difference
0.000, paired W/L/T 2/2/96, task-cluster 95% interval [-0.03, +0.03].
FO_1S minus NATIVE_ACT was -0.08 (3/11/86; interval [-0.27, +0.05]).
The frozen analysis is at `research/audit_outputs/robotwin_exploratory_analysis.json`
and the report is `research/robotwin_exploratory_results.md`.

## Baseline fidelity status

All five trained task directories contain the same pinned-procedure artifacts:
`policy_last.ckpt` and `policy_epoch_6000_seed_0.ckpt`; no
`policy_best.ckpt` exists. The pinned XPolicyLab training file has best-checkpoint
saving commented out, and the pinned ACT evaluator explicitly loads
`policy_last.ckpt`. Thus a generic upstream claim that ACT evaluation loads a
best checkpoint does not apply to this installed revision. No BEST-vs-LAST
closed-loop comparison was possible from these artifacts.

| Task | Checkpoint used in frozen pilot | Policy-best status | Fidelity evidence |
|---|---|---|---|
| beat_block_hammer | `policy_last.ckpt` | absent in pinned run | shared evaluator path; no separate task equivalence canary |
| click_alarmclock | `policy_last.ckpt` | absent in pinned run | shared evaluator path; no separate task equivalence canary |
| dump_bin_bigbin | `policy_last.ckpt` | absent in pinned run | shared evaluator path; no separate task equivalence canary |
| handover_block | `policy_last.ckpt` | absent in pinned run | exact official-vs-local path match on seeds 100000, 100002, 100003 for 5 decisions each |
| open_laptop | `policy_last.ckpt` | absent in pinned run | shared evaluator path; no separate task equivalence canary |

The equivalence artifact is
`research/audit_outputs/robotwin_native_path_equivalence.json`. In the tested
handover comparison, reset/observation preprocessing, qpos normalization, ACT
chunk, candidate set/order, aggregation, denormalization, and sent action were
exactly equal, with no success outcome recorded. The remaining unresolved item
is only that the equivalence canary was not repeated separately for every task;
it is not evidence of a known semantic mismatch. The low task rates therefore
remain an observed pinned-model/setup result, not a justification for retraining.

Canonical checkpoint SHA256 values are recorded in the existing RoboTwin
manifests/notes: beat `7f3a058419b82464aeeb48d414a8b948eba55220ff5b4b82f16385a0383862fd`,
click `73a475b8a2f97d3998ce1e90d26439cd8a701feb9a2ab1659466bad0da869c1c`,
dump `5d26180719fded89edf5587674281fa4c3d90470ced743ab504bb40074370781`,
handover `dfb2801ab20b820a844cbdb896989ff443abb5499d3c10f9749bfc84619dc78c`,
and open `9e8366772f163c78e18f91cbb1685423c34fca859bad2e03308ee60700444b4e`.

## DCTA status

Completed candidate extraction files exist for all five tasks under
`/home/wjq/research-assets/robotwin/dcta_candidates/`:

- `beat_block_hammer.npz`
- `click_alarmclock.npz`
- `dump_bin_bigbin.npz`
- `handover_block.npz`
- `open_laptop.npz`

The offline DCTA fit used demonstrations only (trajectories 0–39 for training,
40–49 held out), froze the ACT backbone, and produced the shared and
component-wise gates in `research/audit_outputs/robotwin_dcta_offline/`.
The gate has 40,835 parameters; training and held-out reconstruction outputs
are preserved in `offline_summary.json`. The outcome-free closed-loop DCTA
canary passed 20 decisions for NATIVE_ACT, SHARED_DYNAMIC_AGG, and DCTA.
The implementation and tests are `research/audit_tools/robotwin_dcta.py` and
`tests/test_robotwin_dcta.py`.

The later DCTA development rollout is **unfinished and not a scientific
result**: technical records exist for 180/300 cells, covering all 60 cells for
each of `beat_block_hammer`, `click_alarmclock`, and `dump_bin_bigbin`; no
handover or open_laptop cells were completed. Its external result root is
`/home/wjq/research-assets/robotwin/dcta_development_results/` under schedule
hash `c9ce4651be45a1c21d95e75f966d0464c40cf50c4a2c4f1c5036c221f3af4178`.
Partial sealed outcome files are retained but were not opened or analyzed.

## Useful infrastructure

- Pinned native ACT execution and evaluator-equivalence tooling:
  `research/audit_tools/compare_robotwin_native_paths.py`.
- Physical-time source-age selection, same-decision-target indexing, and
  temporal executor: `research/audit_tools/robotwin_temporal_reuse.py` and
  `research/audit_tools/robotwin_dcta.py`.
- Frozen temporal canaries and calibration artifacts:
  `research/audit_outputs/robotwin_beat_*` and
  `research/audit_outputs/robotwin_dcta_closed_loop_canary.json`.
- Candidate extraction, offline fitting, development runner, and analyzer:
  `research/audit_tools/extract_robotwin_dcta_candidates.py`,
  `train_robotwin_dcta.py`, `run_robotwin_dcta_development_task.py`, and
  `analyze_robotwin_dcta_development.py`.
- Frozen hard-FO pilot schedule/results and diagnostic reporting remain under
  `research/audit_outputs/robotwin_exploratory_*` and
  `research/robotwin_no_signal_diagnosis.md`.

## Why paused

DCTA is now treated as an ablation/diagnostic mechanism rather than the
primary ICRA contribution. The hard FO result was NO_SIGNAL, and the DCTA
development rollout is incomplete, so additional DCTA tuning or rollout
expansion is not currently justified.

## Safe resume point

If RoboTwin becomes relevant again:

1. Reconcile baseline fidelity using the pinned `policy_last.ckpt` path and the
   existing official-equivalence evidence.
2. Use verified NATIVE_ACT as the standard baseline.
3. Resume only a preregistered, targeted transfer experiment.
4. Do not search task/age combinations post hoc.

No RoboTwin jobs remain running from this session. The GPUs are free; the
external assets and partial outputs above are intentionally preserved.
