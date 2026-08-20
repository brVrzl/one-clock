# RoboTwin baseline calibration and stall diagnosis

Date: 2026-08-20  
Task: `place_can_basket`, `demo_clean`  
Checkpoint: `/home/wjq/checkpoints/robotwin/act-place_can_basket/demo_clean-50`  
RoboTwin: `266f3aadf505a4f7fe9af0faa41a20f5f47cd123`  
XPolicyLab: `c37109c500be67d0dea6b36bf7337bbd26e763cd`

This is a bounded calibration, not Stage A. The previous 240-episode sweep was
not resumed. Each seed was run in a fresh process on GPU 0 with a 180-second
shell timeout. Raw JSON and logs are ignored under
`experiments/runs/robotwin_baseline_calibration/`.

## Paths compared

| Label | Execution path | ACT behavior | Planner |
|---|---|---|---|
| official | Pinned XPolicyLab `Model.get_action()`; no one-clock executor | Upstream `temporal_agg=True` (one ACT action/query per environment step, upstream weighted temporal aggregation) | `mplib_RRT` override for this x86 headless diagnosis |
| global8 | `FixedChunkExecutor.global_fixed(horizon=8)` | Full `(50,14)` ACT chunk, `temporal_agg=False` | `mplib_RRT` |
| group416 | `FixedChunkExecutor.groupwise_fixed`; both arms 4, both grippers 16 | Full `(50,14)` ACT chunk, `temporal_agg=False` | `mplib_RRT` |

The same pinned checkpoint, task seed, `qpos` action type, headless settings,
and `safe_qpos` runtime fallback were used for the three paths. The fallback is
the previously audited environment compatibility setting; it is not a new
horizon or policy modification.

## Dependency and initialization probe

With no planner override (the official ALOHA config default), `place_can_basket`
fails during `setup_demo`, before the first observation/action:

```text
ImportError: cannot import name 'kinematics_fused_cu' from curobo.curobolib
RuntimeError: Ninja is required to load C++ extensions (pip install ninja to get it)
RuntimeError: CuroboPlanner is unavailable in this environment
```

Thus Curobo absence does affect this selected task: the default official
environment cannot be constructed on this host. The calibration uses the
already-audited MPLIB fallback solely to determine whether rollouts and the
executor can make progress after initialization.

## Renderer and headless probe

The runtime configuration used by the pinned base task was:

```text
EGL_PLATFORM=surfaceless
SAPIEN camera shader = rt
ray-tracing samples per pixel = 1 (official source requests 32)
ray-tracing path depth = 1 (official source requests 8)
ray-tracing denoiser = none (official source requests oidn)
render_freq = 0
```

The direct SAPIEN probe created a renderer and a 320x240 RGBA camera frame.
The SAPIEN 3.0.0b1 OIDN path emits `unsupported device type: CUDA` / `invalid
handle` on this RTX 5080 host, so the runtime keeps the RT shader but disables
OIDN. Other observed warnings are the deprecated `Engine`/`SapienRenderer`
constructors and a missing optional `pytorch3d` import. They do not prevent the
initial camera probe.

## Bounded rollout results

All complete trials used the RoboTwin evaluation step limit of 700. `success=0`
means the terminal task check returned false; it is not a timeout converted to a
failure.

| Path | Seed | Status | Success | Episode steps | Policy queries | Failure/stall reason |
|---|---:|---|---:|---:|---:|---|
| official | 0 | complete | 0 | 700 | 700 | task not successful at limit |
| official | 1 | **timeout/stall** | — | 554 last completed | — | no progress in `get_obs` after step 554; isolated process terminated |
| official | 2 | complete | 0 | 700 | 700 | task not successful at limit |
| global8 | 0 | complete | 0 | 700 | 88 | task not successful at limit |
| global8 | 1 | complete | 0 | 700 | 88 | task not successful at limit |
| global8 | 2 | complete | 0 | 700 | 88 | task not successful at limit |
| group416 | 0 | complete | 0 | 700 | 175 | task not successful at limit |
| group416 | 1 | complete | 0 | 700 | 175 | task not successful at limit |
| group416 | 2 | complete | 0 | 700 | 175 | task not successful at limit |

The initial global-8 run that hit a driver bug (`Model.all_actions` versus the
wrapped ACT object) is excluded from this table and was rerun successfully; it
was not a benchmark result.

Timing across the complete trials was consistent with successful progression:
`setup_demo` took 1.36--2.13 s; MPLIB/TOPP plus simulation inside
`take_action` had a maximum of 0.51 s; terminal checks took less than 1 ms.
Observation calls occasionally spiked to 7.2--9.0 s, and the official seed-1
stall occurred in that phase, after the preceding `take_action` had returned.
No finite-action, shape, planner exception, or terminal-check exception was
observed in a complete trial. The custom paths produced exactly the expected
query counts (`ceil(700/8)=88` and `ceil(700/4)=175`).

## Diagnosis

| Candidate blocker | Evidence | Classification |
|---|---|---|
| A. Missing dependency | Default Curobo setup fails before first action; MPLIB override constructs the task | **Confirmed for the default path** |
| B. Renderer/SAPIEN issue | Headless RT+OIDN warning; 7--9 s `get_obs` spikes; official seed 1 stalls in `get_obs` | **Confirmed intermittent** |
| C. MPLIB planner deadlock | MPLIB setup completes; all complete trials return from every `take_action`, max 0.51 s | **Not observed in this bounded sample** (historical sweep stalls remain unresolved) |
| D. Task/checkpoint difficulty | Official ACT path also reaches 700 with zero success on two complete seeds | **Plausible contributor, not isolated from environment anomalies** |
| E. Executor/action issue | ACT output is finite; full chunk is `(50,14)`; both fixed paths complete 700-step episodes with expected query counts | **No evidence of an executor/action-contract failure** |

## Conclusions for the requested questions

1. **Is the official ACT checkpoint functional?** Yes as a policy artifact: it
   loads on the RTX 5080, produces finite 14-D actions, and completes two of
   three official-path trials. It is not yet a reliable benchmark runtime
   because one official trial stalls in SAPIEN observation rendering and the
   default Curobo planner cannot initialize.

2. **Is low success caused by task difficulty or our executor modification?**
   The low success is not uniquely caused by the custom executor: the official
   ACT path also has zero success on its complete trials, while global-8 and
   group (4,16) both complete with zero success. It is consistent with task /
   checkpoint difficulty or a broader environment mismatch, but the renderer
   stall means this cannot be attributed to task difficulty alone.

3. **Is RoboTwin suitable for horizon comparison now?** No. The executor and
   action contract are operational under the MPLIB fallback, but the default
   planner dependency and intermittent SAPIEN `get_obs` stall make a success
   comparison scientifically invalid. Do not restart the 240-episode sweep
   until the renderer/planner runtime is made reliable and the official baseline
   completes repeatedly.

No dynamic horizon method, PACE, RoboDojo, or benchmark-logic fix was attempted.

## Commands

Default-planner dependency probe:

```bash
(cd /home/wjq/workspace/upstreams/RoboTwin && \
 EGL_PLATFORM=surfaceless PYTHONPATH=/home/wjq/workspace/one-clock/src:/home/wjq/workspace/one-clock/scripts:/home/wjq/workspace/upstreams/RoboTwin:/home/wjq/workspace/upstreams/RoboTwin/XPolicyLab \
 /home/wjq/workspace/venvs/robotwin/bin/python \
 /home/wjq/workspace/one-clock/scripts/run_gate0.py \
 --robotwin-root /home/wjq/workspace/upstreams/RoboTwin \
 --checkpoint /home/wjq/checkpoints/robotwin/act-place_can_basket/demo_clean-50 \
 --strategy global_fixed --horizon 8 --seeds 0 --skip-expert-check \
 --output-dir /home/wjq/workspace/one-clock/experiments/runs/robotwin_baseline_calibration/default_global8_seed0_probe)
```

Calibration invocation (run once per mode and seed in a fresh process):

```bash
EGL_PLATFORM=surfaceless CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/wjq/workspace/one-clock/src:/home/wjq/workspace/one-clock/scripts:/home/wjq/workspace/upstreams/RoboTwin:/home/wjq/workspace/upstreams/RoboTwin/XPolicyLab \
/home/wjq/workspace/venvs/robotwin/bin/python \
/home/wjq/workspace/one-clock/scripts/run_robotwin_baseline_calibration.py \
 --robotwin-root /home/wjq/workspace/upstreams/RoboTwin \
 --checkpoint /home/wjq/checkpoints/robotwin/act-place_can_basket/demo_clean-50 \
 --mode official \
 --seed 0 --planner mplib_RRT \
 --output /home/wjq/workspace/one-clock/experiments/runs/robotwin_baseline_calibration/<mode>_seed0.json
```

Each invocation was externally wrapped with `timeout 180s` and the necessary
user-space NVIDIA library path from the dedicated venv. No system package or
sudo command was used.
