# Repository-wide provenance audit

This audit inspected every `origin/*` ref after `git fetch --all --prune`, including artifacts that survive only in reachable history. It does not use similarity of success rates as provenance evidence.

## Existing groupwise implementation

`src/one_clock/executor.py` already implements `strategy="groupwise_fixed"`. Each action group retains its own source chunk, query step, chunk position, and remaining commitment. The historical Gate-0 runner at commit `2a1f1fa` exposes `--group-horizons arm=K,gripper=K`. No new two-clock executor was implemented.

Archived 50-state diagonal controls compare `global_hK` with `group_armK_gripK` at K=1,2,4,8,16. Success, environment steps, policy queries, and query rates are equal episode-by-episode for every diagonal.

## `libero_static_grid_50`

Classification: **PROVENANCE_COMPATIBLE**.

- Task: LIBERO Object task 0, `pick_up_the_alphabet_soup_and_place_it_in_the_basket`.
- States and seeds: official initial-state IDs 0–49; base seed 1000, so environment seed is `1000 + state_id`.
- Checkpoint: archived raw-run metadata identifies `/home/thor/projects/checkpoints/zeromidnight_act_libero_object`. A later audit records ACT model SHA256 `340071d7497238669459d93517eb3f8690862ad6fdf14207966759dfe6da9410` and config SHA256 `a76eebed357b3cbed8745c3d0f18c1335ecdd5449fcc498257676c9cbd27453d`. Both match `/home/wjq/checkpoints/zeromidnight_act_libero_object` byte-for-byte.
- Evaluator: `scripts/run_libero_gate0.py` with LeRobot LIBERO, synchronous single environment, official initial states, `hard_reset=true`, pixel+agent-state observations at 256×256, agent and wrist cameras.
- Episode/control contract: environment `_max_episode_steps` (280 for Object), 20 Hz, relative 7-D actions, arm indices 0–5 and gripper index 6.
- Success: terminal LIBERO `info["is_success"]` (positive terminal reward is the established vector-evaluator fallback in later runners).
- Execution: `FixedChunkExecutor` global/groupwise fixed commitment, full postprocessed policy chunks, same-target row `chunk_q[t-q]`; policy forward occurs only when the active commitment schedule requests it.
- Temporal aggregation and smoothing: ACT `temporal_ensemble_coeff=null`; no executor smoothing.
- Result artifact: 50 states × 30 conditions = 1,500 episodes. Coherent diagonal counts are h1 29/50, h2 31/50, h4 42/50, h8 45/50, h16 42/50; `arm4_grip16` is 47/50.

The aggregate `libero_static_grid_50.json` omits checkpoint configuration, but the later repository-wide raw-artifact inventory ties every contributing `libero_static_grid_20` and `libero_static_grid_50_extension` run directory to the exact checkpoint path, runner commit `f66e5128ecb2456e8c54a63d15404fa59c16aebc`, and valid metadata/episode/step/summary hashes. This closes the aggregate-file omission.

## `libero_object_cross_task`

Classification: **PROVENANCE_COMPATIBLE**.

- Task/cohort: LIBERO Object tasks 1–9, official states 0–19, paired within task and state.
- Seeds: base seed 1000; every task/method reuses `1000 + state_id`. The archived pairing object records all 20 state IDs, base seed 1000, 50 official initial states, and no initial-observation mismatches.
- Checkpoint/evaluator/action semantics: the same byte-identical authoritative Object ACT checkpoint and Gate-0 runner contract listed above.
- Historical raw-artifact inventory: each task/configuration row records the checkpoint path, runner commit, metadata/episode/step/summary hashes, policy-query counts, and no validation issue for the scientific cells.
- Exact pooled vectors: `arm2_grip2` 96/180; `arm2_grip16` 122/180; `arm4_grip4` 112/180; `arm4_grip16` 128/180; `arm8_grip8` 114/180; `arm8_grip16` 123/180; `arm16_grip16` 123/180.
- Verified budget-matched discordances: arm2 row 29:3 (+14.44 pp), arm4 row 19:3 (+8.89 pp), arm8 row 13:4 (+5.00 pp). The arm16 row has approximately zero incremental gripper-horizon gain.

The selected grid argmax is exploratory. The stable historical evidence is the directional budget-matched effect across fixed arm rows.

## SmolVLA baseline provenance

The standard baseline artifact records public `HuggingFaceVLA/smolvla_libero` revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`, mirrored locally at `/home/wjq/checkpoints/HuggingFaceVLA_smolvla_libero`. Its config is `type=smolvla`, `chunk_size=50`, `n_action_steps=1`, `action_dim=7`, and `temporal_ensemble_coeff=null`. Standard native totals are Spatial 85/100, Object 93/100, Goal 78/100, and Long 42/100. They are unpaired context only and are not used as tonight's paired coherent control.

