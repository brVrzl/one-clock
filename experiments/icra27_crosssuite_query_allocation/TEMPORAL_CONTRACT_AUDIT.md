# Temporal contract audit

Status: **PASS WITH EXPLICIT MULTI-RATE MAPPING**

This audit uses configuration, code, checkpoint processors, and dataset
metadata only. It does not inspect Track-A scientific outcomes.

## Demonstration and training timebase

- `HuggingFaceVLA/libero` declares 10 Hz for actions, observations, frame
  indices, and timestamps. A direct parquet check confirms contiguous frame
  indices and approximately 0.1-second timestamp increments.
- The standard task-specific ACT checkpoints used by Track A, Track B, the
  140-block confirmation, and R1C were trained from that 10 Hz dataset. ACT's
  `action_delta_indices` are integer offsets `0..chunk_size-1`; the LeRobot
  dataset factory converts them to `offset / dataset_fps` and the reader maps
  them back to the corresponding integer dataset frames. There is no
  subsampling, repetition, or 10-to-20 Hz action resampling in this path.
- Thus ACT chunk offset `k` was supervised against the recorded action at
  `k/10` seconds in these standard checkpoints. B3 uses the exact same mapping
  for offsets 0..32 and is identifiable without interpolation.
- The `env.fps=30` field preserved in standard ACT `train_config.json` is an
  evaluation-environment configuration field. Offline training chunk assembly
  uses the dataset metadata's 10 Hz rate, not that environment field.
- The historical Object checkpoint used by R1A/R1B names a different remote
  training dataset whose local frame metadata is unavailable. Its evaluator
  timing is known, but its native demonstration/training sampling rate is not
  asserted here.

## Evaluation timebases and physical source age

| Family | Evaluator rate | One executed step | Source age / horizon mapping |
|---|---:|---:|---:|
| Track A | 10 Hz | 0.1 s | `d` or `h` steps = `d/10` or `h/10` s |
| Track B rollout | 10 Hz | 0.1 s | source age `a` = `a/10` s |
| Historical 140-block confirmation | 20 Hz | 0.05 s | source age 20 = 1.0 s; H16 = 0.8 s |
| R1A/R1B | 20 Hz | 0.05 s | frozen d grid = 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.6 s |
| R1C/R1D | 20 Hz | 0.05 s | H16 = 0.8 s; d20 = 1.0 s |
| R2A | 30 Hz | 0.0333 s | d20 = 0.6667 s |
| B2/B3 demonstrations | 10 Hz | 0.1 s | lag/offset `k` = `k/10` s |

The rollout executor defines physical target index by `q+k=t`, so at evaluator
rate `f`, a source age or executed chunk offset of `k` spans `k/f` seconds.
Integer source ages must therefore never be compared across families without
their evaluator rate.

At matched physical times, the complete R1A grid
`d={2,4,8,12,16,20,32}` at 20 Hz maps exactly to B3 offsets
`k={1,2,4,6,8,10,16}` at 10 Hz. No interpolation is needed. The comparison is
still cross-cohort and, for R1A/R1B, cross-checkpoint/training-dataset.

## Action contract and rate conversion

- The policy output is seven controller-input values: translation dimensions
  0--2, axis-angle rotation command dimensions 3--5, and gripper command 6.
- Checkpoint postprocessing exactly inverts the frozen MEAN_STD transform as
  `native = normalized * std + mean`. The LIBERO environment processor changes
  observations only and applies no action rescaling.
- `control_mode="relative"` sets the robosuite OSC controller to delta mode.
  Its controller-input scaling is fixed by the OSC configuration and does not
  include evaluator frequency. Changing 10 Hz to 20 Hz changes the duration
  between commands and the number of commands applied per second; it does not
  halve, repeat, interpolate, or otherwise rescale the stored action values.
- The stored/native values are therefore reported as controller-native action
  units. They are not labeled as measured end-effector millimetres or degrees:
  controller goal scaling does not guarantee realized physical displacement.

## Interpretation consequence

B2 and B3 have a direct 10 Hz demonstration-time mapping. Track A and Track B
also run at 10 Hz, so their integer offsets share that elapsed-time scale for
the standard ACT checkpoints. The 20 Hz and 30 Hz rollout families use
different elapsed-time mappings and apply the same controller-input magnitudes
more frequently. Physical-time alignment is allowed only through the explicit
seconds mappings above. This audit does not license a claim that demonstration
persistence alone explains behavioral sensitivity, especially for R1A/R1B's
separately trained Object checkpoint.
