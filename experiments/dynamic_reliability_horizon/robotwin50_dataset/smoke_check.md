# RoboTwin SmolVLA smoke check

Status: `blocked_contract` (verified 2026-08-20).

The predeclared representative task subset is:

1. `move_can_pot` — single-side manipulation
2. `pick_dual_bottles` — bimanual coordination
3. `handover_block` — handover
4. `stack_blocks_two` — stacking
5. `place_object_scale` — precision/contact
6. `stack_blocks_three` — longer manipulation

The selection is fixed before observing outcomes. The smoke runner first
checks the actual pinned policy and dataset contracts, then runs only a small
number of frames/episodes. It does not estimate benchmark success.

On this machine the contract gate stops before GPU inference because the
primary checkpoint declares a six-D state while the target dataset declares a
14-D state, with no saved adapter selecting six channels. This satisfies the
structural-failure branch of the run protocol; no large cache was started.

The machine-readable result is `smoke_result.json`. It records the pinned
public config audit: action shape `[14]`, `chunk_size=50`,
`n_action_steps=50`, `n_obs_steps=1`, `num_steps=10`, and the three-camera
rename map all match; only the state shape fails (`[6]` versus `[14]`). Thus
there was no policy forward, no smoke success claim, and no `z_t` result.

Resume after an explicitly compatible checkpoint is available:

```bash
tmux new-session -d -s oneclock_robotwin50 'experiments/dynamic_reliability_horizon/robotwin50_dataset/run_overnight.sh'
```
