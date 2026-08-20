# RoboTwin static-validation environment audit

Audit date: 2026-08-20 (Asia/Shanghai)

## Host and operating system

| Field | Value |
|---|---|
| Hostname | `xdl-MS73-HB1-000` |
| Kernel / uname | `Linux xdl-MS73-HB1-000 6.8.0-136-generic #136~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Jul 3 16:29:11 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux` |
| Architecture | `x86_64` |
| OS | Ubuntu 22.04.5 LTS (Jammy) |
| System Python | `/usr/bin/python3`, Python 3.10.12; `python` is not installed as a system alias |
| Dedicated environment | `/home/wjq/workspace/venvs/robotwin` (Python 3.10 venv) |
| Conda / uv | Neither command is available |
| Docker | Not installed (`docker` command unavailable) |
| Git | 2.34.1 |

## GPU and CUDA

`nvidia-smi` reports NVIDIA driver 595.84 and CUDA runtime compatibility 13.2.
All three visible devices are NVIDIA GeForce RTX 5080 GPUs with 16,303 MiB
reported memory each:

```text
GPU 0  NVIDIA GeForce RTX 5080  00000000:16:00.0  18 MiB / 16303 MiB
GPU 1  NVIDIA GeForce RTX 5080  00000000:27:00.0  18 MiB / 16303 MiB
GPU 2  NVIDIA GeForce RTX 5080  00000000:98:00.0  18 MiB / 16303 MiB
Driver Version: 595.84    CUDA Version: 13.2
```

`nvcc` is not installed. The pinned RoboTwin requirement initially installed
`torch==2.4.1+cu121`, but that build failed on the RTX 5080 with
`no kernel image is available for execution on the device` (the wheel did not
contain compute capability 12.0 kernels). The dedicated environment was
therefore upgraded, without system packages, to the official
`torch==2.11.0+cu128` and `torchvision==0.26.0+cu128` wheels plus the required
user-space CUDA 12.8 runtime libraries. The verified runtime probe reports
`torch.cuda.is_available() == True` and device capability `(12, 0)` for GPU 0.
No system CUDA toolkit was installed.

## Memory and storage

At audit time:

```text
RAM: 188 GiB total, 11 GiB used, 136 GiB free, 175 GiB available
Disk (/): 1.8 TiB total, 94 GiB used, 1.7 TiB available (6%)
Swap: 2.0 GiB total, 0 B used
```

## Rendering / headless path

`vulkaninfo` is not installed, but the host exposes NVIDIA and Mesa EGL/GLX
libraries. With `EGL_PLATFORM=surfaceless`, SAPIEN 3.0.0b1 successfully created
a renderer, scene, and 320x240 camera frame. The pinned `place_can_basket`
task also constructed and reset headlessly using the documented asset-mirror
subset.

## Installation provenance

The dedicated venv was built from the pinned RoboTwin checkout's
`scripts/requirements.txt` without sudo or system changes. Installed core
versions used for the final probe/sweep:

```text
torch       2.11.0+cu128
torchvision 0.26.0+cu128
numpy       1.26.4
sapien      3.0.0b1
mplib       0.2.1
```

The upstream checkouts are outside this repository:

```text
/home/wjq/workspace/upstreams/RoboTwin
/home/wjq/workspace/upstreams/RoboTwin/XPolicyLab
```

They are pinned to the commits recorded in
`experiments/roboTwin_act_execution_audit.md` and are not updated during this
experiment.

## Asset and runtime status

The official RoboTwin asset downloader was run against the pinned revision and
Hugging Face dataset `TianxingChen/RoboTwin2.0`. Full Xet-backed archives were
not practical through the server proxy. The required pinned-revision static
asset subset was reproduced outside one-clock from the immutable mirror
`yinchenghust/robotwin_sim`, dataset revision
`bd7e0dc1471c27f265fdf5749af72023f477a832`; this is recorded as an asset
mirror, not as checkpoint provenance. It includes the ALOHA AgileX URDF/SRDF,
meshes/textures, Objaverse index, and the `071_can`/`110_basket` object assets.

The task reset and headless SAPIEN camera path were verified. The pinned base
task requests 32-sample ray tracing with OIDN, but the bundled SAPIEN OIDN
reports `unsupported device type: CUDA` / `invalid handle` on this host. The
runtime fallback keeps the official ray-traced camera shader, sets
samples-per-pixel to 1 for the long headless sweep, and disables the
incompatible denoiser. This is an environment/rendering anomaly and is not an
executor change. Curobo's fused extensions also cannot be compiled because no
`nvcc`/CUDA toolkit is present. A reset-only construction succeeded before
rollout; an MPLIB expert replay was attempted but failed/was too slow. The
sweep therefore uses MPLIB RRT for environment setup/TOPP, skips the expert
replay after the smoke, and defers the qpos success contact check to the
terminal state to avoid a pinned-SAPIEN per-substep stall. The checkpoint,
observation path, action targets, and executor remain unchanged. The pinned
MPLIB qpos fallback clips drive targets to the URDF joint limits before sending
them to SAPIEN and clips the one-step target before MPLIB TOPP sees it; this
prevents an invalid path from stalling the planner. With
`--skip-expert-check`, the runner enables a fixed `safe_qpos` mode that sends
zero velocity targets while retaining the position target. These safeguards
are held fixed across all configurations.

The legitimate official ACT checkpoint is outside one-clock at
`/home/wjq/checkpoints/robotwin/act-place_can_basket/demo_clean-50/`:
policy upload revision
`c544d0d6ebfed8e9032cd2a8e44a46415e913dcf`, policy SHA256
`a8a1b61614788a068c9b266b209e845034050281a0df737965406b3aec3ef1b0`, and
official dataset statistics SHA256
`269981a2f99456f8c64f9237442b4612085d2f73b63824d79ff4b175db365ee6`.
The ACT probe produced a finite `(50, 14)` full chunk.

`pip check` still reports non-fatal binary-environment mismatches from the
RTX-5080 Torch upgrade (the pinned cuDNN/triton versions and an optional
`cuda-bindings` pathfinder dependency). Imports, CUDA inference, SAPIEN, and
the benchmark smoke path are functional; these mismatches are recorded as
installation anomalies rather than hidden.
