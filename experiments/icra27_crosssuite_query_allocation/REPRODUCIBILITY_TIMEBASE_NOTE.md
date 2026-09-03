# Reproducibility note: LIBERO control timebase

Recorded `2026-09-03`. This note describes the executed software path; it does
not classify the behavior as an upstream software bug.

## Pinned runtime inspected

| Package | Version |
|---|---:|
| LeRobot | 0.4.4 |
| LIBERO | 0.1.1 |
| robosuite | 1.4.0 |
| MuJoCo | 3.3.1 |
| Gymnasium | 1.2.2 |
| PyTorch | 2.7.1+cu128 |
| NumPy | 1.26.4 |

Python runtime:
`/home/wjq/workspace/venvs/libero_act/bin/python`.

R1D separately uses LeRobot source commit
`f66e5128ecb2456e8c54a63d15404fa59c16aebc`.

## Resolved control path

In the pinned LeRobot 0.4.4 installation:

1. `lerobot/envs/configs.py`, `LiberoEnv.gym_kwargs`, does not forward the
   configuration's `fps` value.
2. `lerobot/envs/libero.py`, `_make_envs_task`, calls
   `OffScreenRenderEnv(**env_args)` without `control_freq`.
3. `libero/libero/envs/env_wrapper.py`, `ControlEnv.__init__`, defaults
   `control_freq` to 20.
4. `robosuite/environments/base.py` sets
   `control_timestep = 1.0 / control_freq` and performs the corresponding
   simulator substeps for each `env.step`.

The resolved runtime behavior is therefore 20 Hz, or 0.05 seconds per
environment step. This is true for the installed-runtime paths used by Track
A, Track B, the historical confirmation/Object experiments, R1A, R1B, R1C,
and R2A. The newer LeRobot source used for R1D explicitly forwards the sealed
20-Hz value and also executes at 20 Hz.

## Sealed metadata mismatch

The sealed Track-A and Track-B cells record `control_frequency_hz=10`; R1A-D
record 20; R2A records 30. In LeRobot 0.4.4 these fields did not determine the
LIBERO `control_freq`, so the 10-Hz and 30-Hz records do not describe the
actual executed control cadence. They remain preserved as sealed historical
metadata and must not be silently rewritten.

This mismatch did not change ACT same-target identity: the ACT training
sequence also retains one physical 20-Hz action per chunk index. It does mean
that physical-time labels derived only from sealed nominal FPS fields are not
reliable.

## Reproduction requirement

Future reproduction must explicitly set LIBERO `control_freq=20` through a
code path known to forward it, then verify on the constructed environment that
`env.control_freq == 20` and `env.control_timestep == 0.05` before rollout.
Reproduction must not depend on a particular LeRobot version omitting or
propagating `fps`. It should also verify the dataset's content-level physical
cadence rather than trusting its nominal timestamp metadata alone.
