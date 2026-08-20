# RoboTwin static-validation task audit

Pre-registration date: 2026-08-20

This audit was completed before inspecting any Stage-A success result. The
candidate set was fixed from the requested diversity categories and evaluated
only for official task presence, frozen ACT artifact availability, and action
compatibility. No task was selected or removed based on expected success.

## Frozen upstream and checkpoint source

- RoboTwin source: `266f3aadf505a4f7fe9af0faa41a20f5f47cd123`.
- XPolicyLab submodule: `c37109c500be67d0dea6b36bf7337bbd26e763cd`.
- Checkpoint/data repository: official RoboTwin dataset
  `TianxingChen/RoboTwin2.0`.
- All audited ACT artifacts use the official
  `act_ckpt/act-<task>/demo_clean-50/` layout and contain
  `policy_last.ckpt` plus `dataset_stats.pkl`.
- The exact file-upload revision for each policy is recorded below. The
  revision is used as the immutable download selector if that task is run.

## Candidate coverage

| Task | Manipulation structure | Official task source | ACT checkpoint (demo_clean-50) | Policy upload revision | 14-D joint contract | Evaluation suitability |
|---|---|---|---|---|---|---|
| `place_can_basket` | Object placement followed by opposite-arm basket lift; dual-arm coordination | `envs/place_can_basket.py` | Available | `c544d0d6ebfed8e9032cd2a8e44a46415e913dcf` | Compatible: `aloha_agilex`, 6+1 per arm | Suitable; Stage A task |
| `handover_block` | Inter-arm object handover / dual-arm coordination | `envs/handover_block.py` | Available | `38ea3190507ad18a11ee3965ad76dc6bebbb16c4` | Compatible: `aloha_agilex`, 6+1 per arm | Suitable |
| `stack_blocks_two` | Sequential object stacking / placement | `envs/stack_blocks_two.py` | Available | `32cbdc37e7880b23dea2114b4506d7231c98f17e` | Compatible: `aloha_agilex`, 6+1 per arm | Suitable |
| `place_dual_shoes` | Two-object / dual-arm placement coordination | `envs/place_dual_shoes.py` | Available | `0a9706660619d8661f4b0b3fd3887ce50e819384` | Compatible: `aloha_agilex`, 6+1 per arm | Suitable |
| `open_microwave` | Articulated-object manipulation (door opening) | `envs/open_microwave.py` | Available | `3faa611a54dd35b728cb78fc2ba13def607147da` | Compatible: `aloha_agilex`, 6+1 per arm | Suitable |
| `place_object_basket` | Object placement into a receptacle | `envs/place_object_basket.py` | Available | `cd9169c5dbc2ba26952bb6ed82d6ebbda3c2b274` | Compatible: `aloha_agilex`, 6+1 per arm | Suitable |

## Objective selection rule

The six rows above form the pre-registered multi-task extension, including
`place_can_basket`. All six have official pinned-revision task modules and
legitimate frozen ACT artifacts. No replacement is needed. If an artifact
becomes inaccessible at evaluation time, the run is blocked; it is not replaced
because of a low success rate. Any future replacement would follow the
pre-registered rule of choosing another official task from the same broad
manipulation category with a valid `demo_clean-50` ACT artifact.

The scientific execution comparison remains frozen within each task: the
checkpoint, observations, task configuration, ordered evaluation seeds, and
action contract are held constant while only static execution horizons change.
