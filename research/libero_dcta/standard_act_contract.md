# LIBERO DCTA track: public-source decision and STANDARD_ACT contract

Date: 2026-08-26

This is a new LIBERO-only track. Historical Gate-3C and the invalidated sealed
Spatial rollout outcomes are out of scope and are not inputs to this protocol.

## Public ACT checkpoint search

| Source | Suite | Checkpoint | Public weights | Published success | N episodes | Temporal aggregation | Protocol match | Authority level |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| [verl-vla official ACT recipe](https://verl-vla.readthedocs.io/en/latest/fine-tuning/act/official-libero-spatial.html) | Spatial | Native ACT export at training step 8,275 | No. `assets/hf_models/act_libero` is a config-only initializer with ImageNet ResNet-18 weights. | 85/100 (per task: 10, 9, 9, 9, 9, 10, 10, 6, 8, 5) | 100 | Disabled: `chunk_size=10`, `n_action_steps=10`, `temporal_ensemble_coeff=null` | Environment, data, task identities, training, and action horizon match; requested inference contract does not | Official framework release |
| [LeRobot ACT implementation](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/act) | Not LIBERO-specific | N/A | Public implementation, no released standard-LIBERO ACT checkpoint/result found | N/A | N/A | Native online exponential temporal ensemble; original coefficient 0.01; policy queried each step | Defines the requested aggregation semantics | Official framework release |
| [LeRobot LIBERO benchmark integration](https://huggingface.co/docs/lerobot/libero) | Spatial, Object, Goal, Long | No ACT checkpoint listed | No ACT weights/results | N/A | Recommends 10 episodes/task | Policy-dependent | Environment and evaluation protocol reference only | Official framework release |
| `ishandotsh/act_libero_spatial_test` | Spatial | Community LeRobot checkpoint | Yes | No authoritative, complete published protocol | Not established | Disabled in checkpoint metadata | Insufficient | Community diagnostic |
| `zeromidnight/act_libero_object` | Object | Community LeRobot checkpoint | Yes | No authoritative, complete published protocol | Not established | Disabled in checkpoint metadata | Insufficient | Community diagnostic |
| `Harrysunshine/act-libero-goal` | Goal task 8 only | Best single-task checkpoint near step 950 | Yes | 10/10 on one training task | 10 | Explicitly disabled; executes 100 actions/query | Wrong task scope and inference contract | Community teaching example |

Search conclusion: no public checkpoint satisfies all primary-evidence fields,
and no released copy of verl-vla's trained step-8,275 Spatial export was found in
the project, releases, linked storage, or the public Hugging Face ACT/LIBERO
model index. The track therefore reproduces training rather than adopting a
community checkpoint.

For Spatial only, the resulting step-8,275 weights will first be evaluated once
with the recipe's documented 10-step open-loop executor as a training-fidelity
diagnostic against 85/100. That diagnostic is explicitly not STANDARD_ACT and
will not enter the three-method comparison. The same frozen weights are then
evaluated under the native temporal-ensemble contract below. Object and Goal
have no published ACT success target with a reproducible protocol, so no false
numerical fidelity claim will be made for those suites.

Pinned implementation sources:

- verl-vla commit `856c1d747c19695caaf66f55431c57b27e3c9d8e`
- LeRobot 0.4.4, as required by that verl-vla revision
- LIBERO 0.1.1 plus the verl-vla-pinned simulator assets revision

Pinned training datasets:

| Suite | Dataset | Revision | Episodes / frames | Stored rate | Gripper action |
| --- | --- | --- | ---: | ---: | --- |
| Spatial | `lerobot/libero_spatial_image` | `d86c0b94922572b3b657e1d1a3d01f0952ddeb46` | 432 / 52,970 | 10 Hz | `[-1, +1]` |
| Object | `lerobot/libero_object_image` | `e1e080d7df1d0a359dff5c86c222e047549f447f` | 454 / 66,984 | 10 Hz | `[-1, +1]` |
| Goal | `lerobot/libero_goal_image` | `91a97115558b5b611200a432d9c82e4f30991b60` | 428 / 52,042 | 10 Hz | `[-1, +1]` |

The `nvidia/LIBERO_LeRobot_v3` conversion was considered but rejected for
training this baseline. Its Object and Goal subsets encode the gripper as
`[0, 1]`, whereas the official LeRobot image datasets and the standard LIBERO
OSC-pose environment use `[-1, +1]`. Its metadata also labels the same frames
as 20 Hz rather than the 10 Hz rate used by the selected recipe datasets.

Each selected dataset contains exactly the ten tasks in its corresponding
LIBERO benchmark, but its `task_index` order is a permutation of the benchmark
task-id order. Dataset-index to benchmark-id mappings are Spatial
`[6,4,5,7,0,3,8,1,2,9]`, Object `[9,4,1,3,0,7,2,6,5,8]`, and Goal
`[8,9,3,6,2,5,7,1,4,0]`. ACT does not consume the task string or task index;
the mapping is nevertheless frozen for trajectory splits and per-task reports.

The authoritative recipe trains for five full passes with batch size 32 and
evaluates the latest completed step (its `run_eval.sh` reads
`latest_checkpointed_iteration.txt`); it does not define a validation-best ACT
selection rule. The frozen suite endpoints are therefore Spatial 8,275 steps,
Object 10,465 steps, and Goal 8,130 steps. Using these final endpoints follows
the published rule rather than choosing `policy_last` ad hoc.

## Canonical STANDARD_ACT execution contract

The frozen ACT policy predicts a normalized action chunk at every simulator
decision. Aggregation occurs in normalized ACT action space before the native
LeRobot action postprocessor. Because that postprocessor is affine per action
dimension and the temporal weights sum to one, this ordering is also consistent
with aggregation in environment action space.

| Contract item | Frozen value |
| --- | --- |
| Policy architecture | Native LeRobot ACT: CVAE, ResNet-18 per camera, 512-wide transformer, 8 heads, 4 encoder layers, 1 effective decoder layer, latent width 32 |
| Observations | 256x256 agent view and wrist RGB, each rotated 180 degrees by the verl-vla LIBERO adapter; 8-D state = EEF xyz, EEF axis-angle, two gripper positions. The selected datasets' `rx, ry, rz, rw, gripper` labels are stale metadata; their values and the official evaluator are the 3+3+2 representation. |
| Action | 7-D LIBERO relative OSC pose command: first 6 arm dimensions, final gripper scalar, with checkpoint-native mean/std preprocessing and postprocessing |
| Simulator control rate | 20 Hz, inherited from `OffScreenRenderEnv`; the LeRobot training datasets are indexed at their declared 10 Hz |
| Chunk size | 10 simulator decisions |
| Query frequency | Every simulator decision (`n_action_steps=1`) |
| Same-target indexing | At decision `t`, source query `q` contributes chunk element `t-q`; valid sources are `max(0,t-9) <= q <= t` |
| Candidate order | Oldest valid source to newest valid source |
| Native ACT logits | `-0.01 * i`, where `i=0` is the oldest valid candidate |
| Native ACT weights | Softmax over the valid same-target logits |
| Warmup | Decision 0 uses one candidate; candidate count increases by one per decision until 10; history is reset at every environment reset |
| Reset stabilization | 10 zero-action simulator steps, as in the selected verl-vla recipe |
| Episode limit | 512 simulator decisions |
| Evaluation states | First 10 official reset states per task, in the deterministic verl-vla evaluation queue |

`NEWEST`, hard future-offset/source selection, hold, and EMA are not
STANDARD_ACT in this track.

The local numerical acceptance test compares the new vectorized implementation
against LeRobot's online `ACTTemporalEnsembler` over both warmup and steady-state
history. The required decision is whether the new executor can be used without
changing native ACT; failure blocks DCTA integration.

## Frozen method contrast

All three methods use the same candidate history and executor:

1. `STANDARD_ACT`: native ACT weights only.
2. `SHARED_DYNAMIC_AGG`: one learned residual temporal logit per candidate,
   shared by all seven action dimensions.
3. `DCTA`: the same gate architecture evaluated with separate arm and gripper
   group identities; dimensions 0--5 use arm weights and dimension 6 uses
   gripper weights.

Both learned controls use

`logit = log(w_ACT) + residual`

and masked softmax over valid candidates. Gate output layers are zero
initialized. Thus an untrained shared gate and an untrained DCTA gate both
numerically reproduce `STANDARD_ACT`.

The frozen gate features are normalized candidate query age, physical/source
age, the full normalized 7-D candidate, its difference from the newest
prediction, full cross-candidate variance, normalized current 8-D robot state,
group identity, and a mean-pooled 512-D ACT encoder context. The context is
captured by one localized forward hook during the policy's existing forward
pass, so it adds no vision backbone or duplicate visual forward.

Training uses demonstration trajectories only, freezes ACT, selects checkpoints
on held-out demonstration loss, and averages arm L1 and gripper L1 with equal
group weight.

## Frozen development design

The independent unit for DCTA train/validation splitting is a complete
demonstration trajectory, never a frame. Within every dataset task, a seeded
(`2027`) permutation assigns 20% of trajectories to validation and the rest to
gate training. The resulting split is frozen in
`configs/libero_dcta/demo_splits.json` before any rollout outcome is inspected.

For the first development evaluation, an official reset state is the paired
unit and suite/task are analysis clusters. Every suite x task x trial block is
evaluated under all three methods. Method order within each block is a seeded
random permutation to prevent runtime drift from aligning with a method. The
complete 900-rollout schedule is frozen in
`configs/libero_dcta/rollout_schedule.json`. Episode-level repeats are not
treated as independent task or suite replicates in interval estimation.
