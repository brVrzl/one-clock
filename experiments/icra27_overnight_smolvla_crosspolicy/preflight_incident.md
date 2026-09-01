# ACT Object camera-key preflight incident

The first detached launch attempted to construct the authoritative historical Object ACT policy with the newer default camera key `image2`. The checkpoint requires `wrist_image`. Policy construction failed before environment reset, policy forward, action execution, reward, or outcome observation.

The launcher automatically repeated the same construction check three times for each of 126 manifest cells and initially wrote `TECHNICAL_FAILED` markers. These were not episode attempts. All 126 exception histories and provisional markers were moved, not deleted, to `preflight_failures/act_object_h8_126/`. There were zero scientific result JSONs for the phase.

The integration was corrected to select camera mapping from checkpoint provenance:

- historical `zeromidnight_act_libero_object`: `agentview_image -> image`, `robot0_eye_in_hand_image -> wrist_image`;
- task-specific ACT and SmolVLA: `agentview_image -> image`, `robot0_eye_in_hand_image -> image2`.

The installed LeRobot 0.4.4 policy validator also requires the historical checkpoint's `env_cfg.features_map` to name `observation.images.wrist_image`, while its environment preprocessor still emits the same pixels under `observation.images.image2`. A first mapping-only correction therefore remained a preflight failure. Those additional exception histories/provisional markers were preserved under `preflight_failures/act_object_h8_126_second_mapping/`. The final integration reuses the already-established historical runner behavior: adjust `features_map`, then rename the emitted tensor after environment preprocessing without changing its pixels. A complete exact manifest cell passed validation before detached restart.

The same frozen manifest cells were then restarted. Valid cells from other phases remain durable and are skipped on resume.
