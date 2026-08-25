# Gate-4A2 first-20-prefix root-cause report

## Frozen status

Gate-4A2 remains **INVALIDATED BEFORE OUTCOME ANALYSIS**. This audit did not open, compute, aggregate, print, or infer any task-success field. It did not run the outcome analyzer, alter the 500 historical traces, modify the invalidated Gate-4A2 report, modify the manuscript, or launch Gate-4A3.

The root-cause branch starts at `13f87613fec4aeb99cd48aa3ad03ff7b9850d995` and is `exp/gate4a2-prefix-rootcause`.

## Decision

- Primary classification: **RESET_STATE_NONDETERMINISM**.
- Secondary contributor: **OBSERVATION_RENDER_NONDETERMINISM**.
- Earliest divergent layer overall: **L1**, at task 4 step 0.
- Task 3/state 1 earliest divergent layer: **L2**, at step 8.
- Scientific rerun: **RERUN-NOT-JUSTIFIED**.
- Determinism canary: **FAILED**.
- Gate-4A3: **NOT READY**.

The task-4 defect has a narrow technical correction, but the independent task-3/state-1 renderer divergence does not. Because an exact pairing-only correction has not passed the canary, no Gate-4A3 protocol draft was created.

## Outcome seal

The historical reader uses `json.load(..., object_pairs_hook=...)` to remove `success`, `is_success`, `reward`, `failure_category`, `terminated`, and `truncated` while each JSON object is constructed. It then projects the object to run identity, registered initial-state vector hash, and the first 20 fresh/executed actions. It recursively rejects any forbidden key before returning data.

The traces store L0, one fresh action from L5, and L6. They do not store L1-L4 or the complete L5 chunk. Therefore the sealed historical pass can establish the recorded action discrepancy but cannot locate an earlier layer. Live diagnostic prefixes were needed for that purpose. Diagnostic transitions used controller/physics stepping without reward or success evaluation and never exceeded 20 controller steps.

## Sealed historical-prefix audit

All A/B/C blocks had an identical state ID, registered initial-state vector hash, and episode seed at L0.

| Task | State | First differing step | Earliest stored layer | Max L5 fresh-action difference | Max L6 difference |
|---:|---:|---:|:---:|---:|---:|
| 4 | 1 | 0 | L5 | 0.11424273 | 0.11424273 |
| 4 | 13 | 0 | L5 | 0.11535895 | 0.11535895 |
| 4 | 15 | 0 | L5 | 0.08125186 | 0.08125186 |
| 4 | 19 | 0 | L5 | 0.19246033 | 0.19246033 |
| 4 | 21 | 0 | L5 | 0.14379483 | 0.14379483 |
| 4 | 24 | 0 | L5 | 0.13054425 | 0.13054425 |
| 4 | 31 | 0 | L5 | 0.10562640 | 0.10562640 |
| 4 | 37 | 0 | L5 | 0.19852698 | 0.19852698 |
| 4 | 40 | 0 | L5 | 0.13932577 | 0.13932577 |
| 4 | 47 | 0 | L5 | 0.11511990 | 0.11511990 |
| 3 | 1 | 8 | L5 | 0.00996086 | 0.00996086 |
| 3 | 13 | none | none | 0 | 0 |
| 0 | 1 | none | none | 0 | 0 |
| 5 | 1 | none | none | 0 | 0 |

## Layered live-prefix audit

The live audit fingerprinted the registered official state, persistent and derived simulator/model/controller arrays, raw camera/proprio observations, checkpoint input tensors, raw ACT chunks, postprocessed chunks, and executed fresh actions. MuJoCo exposes some uninitialized scratch buffers; those were inventoried but excluded from persistent-state equality because their unused memory is not simulator state. L1 conclusions below use the flattened simulator state, qpos, qvel, act, ctrl, mocap state, model body poses, and controller state.

| Task | State | First differing step | Earliest layer | Max persistent L1 diff | Max image diff | Max L3 diff | Max L4 diff | Max L5 diff | Max L6 diff |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 0 | L1 | 2.93031898 | 219 | 3.81652629 | 1.65183866 | 1.32592100 | 0.11338907 |
| 4 | 37 | 0 | L1 | 2.49214156 | 221 | 3.86904728 | 1.29952279 | 1.02604371 | 0.11000615 |
| 3 | 1 | 8 | L2 | 0 through t=8; first differs at t=9 | 67 | 1.17296910 | 0.04775447 | 0.03915584 | 0.00996086 |
| 3 | 13 | none | none | 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | none | none | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | 1 | none in stored L5/L6 | not localized live | not stored | not stored | not stored | not stored | 0 | 0 |

These maxima summarize all 20 steps and can exceed the first-difference magnitude because a small earlier perturbation is amplified by closed-loop execution.

## Task-4 diagnosis

Task 4 is `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate`. Its BDDL initial conditions include `(Open wooden_cabinet_1_top_region)`.

LIBERO initializes `object_property_initializers` once in `BddlBaseDomain.__init__`. Every hard-reset model load calls `_setup_placement_initializer()`, which calls `_add_placement_initializer()`. The Open predicate creates and appends an `OpenCloseSampler`, but the persistent list is not cleared on hard reset. `_reset_internal()` samples every retained initializer before object placement. Gate-4A2 reused one environment for all episodes of each task, so the sampler count increased with episode ordinal.

This mechanism was reproduced outcome-blind with the same task, state 1, and seed 340401:

| Repeat | Registered state hash | OpenCloseSampler count | Cabinet body position | Cabinet body quaternion | Simulator-state hash |
|---:|:---|---:|:---|:---|:---|
| 0 | `869b45eb…a61438` | 3 | `[0.02720025, -0.27273010, 0.905]` | `[0.22224891, 0, 0, 0.97498996]` | `6063fd9b…856cfe` |
| 1 | `869b45eb…a61438` | 4 | `[0.02726990, -0.26982238, 0.905]` | `[0.20834577, 0, 0, 0.97805523]` | `4be58fdb…54b367` |
| 2 | `869b45eb…a61438` | 5 | `[0.03017762, -0.26075872, 0.905]` | `[0.21356127, 0, 0, 0.97692967]` | `844c7b1c…5f632a8` |

The registered 92-D official state is identical, but the cabinet model pose, settled simulator state, and both camera hashes differ. The first task-4 live difference is therefore L1, before the first policy query.

The official state restoration calls `sim.set_state_from_flattened`; it does not restore `model.body_pos` or `model.body_quat`. LIBERO does not expose a supported complete snapshot API that includes model arrays, controller internals, renderer state, and mutable reset structures. Manually treating the registered vector as a full state is therefore incorrect.

The narrow task-4 correction is to seed all RNGs before construction and create/destroy a fresh LIBERO environment for every episode. This ensures the task-4 initializer count is the same for every method. It is not a complete Gate-4A3 correction because task 3/state 1 still fails.

## Task-3/state-1 diagnosis

Task 3/state 1 reproduces the historical signature exactly. Repeats are identical through step 7. At step 8:

- the persistent L1 simulator/model/controller state remains exactly equal;
- `pixels.image2` (the wrist camera) first differs at pixel `[80, 208, 0]` by one uint8 intensity unit;
- the corresponding normalized L3 tensor differs by at most 0.01750700;
- the raw ACT chunk differs by at most 0.00023219;
- the postprocessed chunk differs by at most 0.00017413;
- the executed fresh action differs by at most 0.00001884 at that step.

The distinct action then makes persistent simulator state differ at step 9. Feedback amplifies the fresh-action difference to 0.00996086 during the 20-step prefix, matching the sealed historical maximum.

This establishes **OBSERVATION_RENDER_NONDETERMINISM**, not policy stochasticity and not physics nondeterminism before step 8. The following pairing-only attempts all reproduced the same step-8 L2 failure:

1. fresh hard-reset environment per repeat with Python, NumPy, Torch, and CUDA seeded before construction;
2. disabling NVIDIA threaded GL optimization and using non-yielding driver waits before rendering-stack import;
3. a new Python process and EGL context for every repeat;
4. an explicit `glFinish()` barrier after MuJoCo render and before pixel readback.

No tolerance was introduced. The 0.00996086 prefix difference is not treated as numerical noise. No observation quantization, image filtering, alternative renderer, preprocessing change, cached canonical action, or first-20 action override was adopted because each would change the scientific execution/input contract rather than only repair pairing.

## Exact-input ACT determinism

On one immutable valid observation:

- 20 preprocessing calls produced one exact input fingerprint and consumed no monitored RNG state;
- 50 raw ACT calls produced one exact chunk fingerprint and consumed no monitored RNG state;
- 50 postprocessing calls produced one exact output fingerprint and consumed no monitored RNG state;
- 20 policy-reset repetitions produced one exact raw-chunk fingerprint;
- a newly instantiated policy produced the same raw chunk.

The policy was in evaluation mode. All 28 dropout modules were in evaluation mode. No random image augmentation was present. Torch deterministic algorithms and deterministic CuDNN were enabled; CuDNN benchmarking was disabled. This excludes PREPROCESSOR_STOCHASTICITY, POLICY_INFERENCE_NONDETERMINISM, and POLICY_STATE_LEAKAGE for the reproduced failure.

## Runner state and seed audit

Gate-4A2 created one policy, processor set, and LIBERO environment per task and reused them across the task's scheduled episodes. The per-episode function seeded NumPy and Torch CPU/CUDA immediately before `env.reset`; it did not seed Python `random`. Process startup and policy/processor/suite/wrapper construction occurred earlier. The underlying lazy LIBERO environment and renderer were first materialized during reset.

`policy.reset()` occurred after hard reset, official-state restoration, reset settling, and production of the first observation, but before the first policy query. The ACT checkpoint had no internal temporal ensemble. The runner called `predict_action_chunk` directly once per step, so its action queue could not save a query. The processors consist of fixed rename/batch/device/normalization and unnormalization steps; no mutable processor state affecting inference was found.

Method order could nevertheless affect method identity at a given reset ordinal because the environment was reused. This is the task-4 pathway: different methods encountered different accumulated `OpenCloseSampler` counts. No policy-state order effect was reproduced.

Moving seed initialization earlier and constructing fresh environments fixes task 4 but does not fix the task-3 render divergence. Seed placement is therefore an implementation weakness, not the complete root cause.

## Determinism canary

The first proposed correction (fresh environment, earlier complete seeding, and GL driver settings) passed the first 30 task-state blocks, then failed the known task-3/state-1 block at L2 step 8. The run stopped at 31/100 because completing the remaining blocks could not turn the correction into a passing candidate.

A stronger fresh-process check passed task 4/state 1, task 4/state 37, and task 0/state 1, but again failed task 3/state 1 at L2 step 8. Five further task-3/state-1 repeats with an explicit OpenGL completion barrier also failed. Thus the canary status is **FAILED**, exact 100/100 identity is not established, and Gate-4A3 is **NOT READY**.

## Rerun legitimacy

A future Gate-4A3 would preserve the checkpoint, dataset, states, methods, `d=20`, beta 0.03, CogACT alpha 0.3, and all 500 cells. The task-4 correction would only change lifecycle pairing. However, condition 1 and the pre-reregistration canary are not fully satisfied for task 3/state 1 because there is no validated pairing-only renderer correction.

Accordingly, this audit declares **RERUN-NOT-JUSTIFIED**. No old Gate-4A2 episode may be reused, no Gate-4A3 preregistration should be committed, and no Gate-4A3 scientific episode should run unless a separately reviewed deterministic rendering contract passes a new exact 100-block canary without changing observations or method definitions.
