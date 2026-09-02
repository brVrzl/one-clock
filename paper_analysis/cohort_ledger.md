# Cohort and exposure ledger

Internal provenance is retained here. Manuscript and supplementary generation must strip repository ownership, branch names, commit SHAs, local paths, and user/machine identifiers.

| Cohort ID | Policy | Suite | Tasks / states | n | Exposure | Preregistered | Post-hoc | First use | Final-paper entries | Source |
|---|---|---|---|---:|---|---|---|---|---:|---|
| `ACT_FROZEN_CONFIRM_140` | ACT | LIBERO Goal + LIBERO-10 | `{"libero_goal":[4,6,7,8,9],"libero_10":[0,2,4,6,7]}` / `[0,1,2,3,4,5,6,7,8,9,10,11,12,13]` | 140 | `FROZEN_CONFIRMATION` | true | false | cross_suite_confirmation | 5 | `exp/libero-component-temporal-reuse@7ea83e1c0bea:experiments/cross_suite_confirmation/protocol.json` |
| `ACT_FROZEN_CONFIRM_140_POSTHOC_FACTORIAL` | ACT | LIBERO Goal + LIBERO-10 | `{"libero_goal":[4,6,7,8,9],"libero_10":[0,2,4,6,7]}` / `[0,1,2,3,4,5,6,7,8,9,10,11,12,13]` | 140 | `POST_HOC_ON_EXPOSED_COHORT` | false | true | cross_suite_confirmation (factorial decomposition analyzed post hoc) | 2 | `exp/libero-component-temporal-reuse@7ea83e1c0bea:experiments/posthoc_reporting/paired_completeness.json` |
| `ACT_FROZEN_CONFIRM_140_POSTHOC_H8` | ACT | LIBERO Goal + LIBERO-10 | `{"libero_goal":[4,6,7,8,9],"libero_10":[0,2,4,6,7]}` / `[0,1,2,3,4,5,6,7,8,9,10,11,12,13]` | 140 | `POST_HOC_ON_EXPOSED_COHORT` | false | true | cross_suite_confirmation (before H8 was added post hoc) | 1 | `exp/icra27-overnight-smolvla-crosspolicy@eb4e29a62b28:experiments/icra27_overnight_smolvla_crosspolicy/analysis.json` |
| `ACT_OBJECT_DEV_126` | ACT | LIBERO Object | `[1,2,3,4,5,6,7,8,9]` / `[20,21,22,23,27,31,34,35,38,39,44,45,47,48]` | 126 | `EXPOSED_DEVELOPMENT` | false | false | group_delay_factorial_act20 | 5 | `exp/icra27-two-clock-discriminator@c4f9cb9ba081:experiments/icra27_two_clock_discriminator_dev/protocol.json` |
| `ACT_LANDSCAPE_180` | ACT | LIBERO Object | `[1,2,3,4,5,6,7,8,9]` / `"0-19"` | 180 | `EXPOSED_DEVELOPMENT` | false | false | libero_object_cross_task | 5 | `baseline/libero-standard@2a1f1fab34c0:experiments/libero_object_cross_task/task_{1..9}/result.json` |
| `ACT_STATIC_GRID_50` | ACT | LIBERO Object | `[0]` / `"0-49"` | 50 | `EXPOSED_DEVELOPMENT` | false | false | libero_static_grid_50 | 1 | `baseline/libero-standard@2a1f1fab34c0:experiments/libero_static_grid_50.json` |
| `ACT_GATE_M_HELDOUT_130` | ACT | LIBERO Object | `[1,2,3,4,5,6,7,8,9]` / `{"1":[30,32,33,36,37,40,41,42,43,46,49],"2":[24,25,26,28,29,30,32,33,36,37,40,41,42,43,46,49],"3":[24,25,26,28,29,30,32,33,36,37,40,41,42,43,46,49],"4":[30,32,33,36,37,40,41,42,43,46,49],"5":[24,25,26,28,29,30,32,33,36,37,40,41,42,43,46,49],"6":[24,30,32,33,36,37,40,41,42,43,46,49],"7":[24,25,26,28,29,30,32,33,36,37,40,41,42,43,46,49],"8":[24,25,26,28,29,30,32,33,36,37,40,41,42,43,46,49],"9":[24,25,26,28,29,30,32,33,36,37,40,41,42,43,46,49]}` | 130 | `HELD_OUT_PREREGISTERED` | true | false | icra27_care_final_gate Gate M | 6 | `exp/icra27-care-final-gate@92ed6b281b5c:experiments/icra27_care_final_gate/queue_manifest.json` |
| `SMOLVLA_PRIMARY_160` | SmolVLA | All four LIBERO suites | `{"libero_spatial":[0,1,2,3,4,5,6,7,8,9],"libero_object":[0,1,2,3,4,5,6,7,8,9],"libero_goal":[0,1,2,3,4,5,6,7,8,9],"libero_10":[0,1,2,3,4,5,6,7,8,9]}` / `"0-3"` | 160 | `CROSS_POLICY_ROBUSTNESS` | true | false | icra27_overnight_smolvla_crosspolicy primary | 5 | `exp/icra27-overnight-smolvla-crosspolicy@eb4e29a62b28:experiments/icra27_overnight_smolvla_crosspolicy/queue_manifest.json` |
| `SMOLVLA_CAPACITY_H16_160` | SmolVLA | All four LIBERO suites | `{"libero_spatial":[0,1,2,3,4,5,6,7,8,9],"libero_object":[0,1,2,3,4,5,6,7,8,9],"libero_goal":[0,1,2,3,4,5,6,7,8,9],"libero_10":[0,1,2,3,4,5,6,7,8,9]}` / `"0-3"` | 160 | `CROSS_POLICY_ROBUSTNESS` | true | false | icra27_overnight_smolvla_crosspolicy capacity barrier | 10 | `exp/icra27-overnight-smolvla-crosspolicy@eb4e29a62b28:experiments/icra27_overnight_smolvla_crosspolicy/queue_manifest.json` |
| `SMOLVLA_ROBUSTNESS_160` | SmolVLA | All four LIBERO suites | `{"libero_spatial":[0,1,2,3,4,5,6,7,8,9],"libero_object":[0,1,2,3,4,5,6,7,8,9],"libero_goal":[0,1,2,3,4,5,6,7,8,9],"libero_10":[0,1,2,3,4,5,6,7,8,9]}` / `"0-3"` | 160 | `CROSS_POLICY_ROBUSTNESS` | true | false | icra27_care_final_gate SmolVLA robustness | 5 | `exp/icra27-care-final-gate@92ed6b281b5c:experiments/icra27_care_final_gate/queue_manifest.json` |
| `SMOLVLA_STANDARD_400` | SmolVLA | All four LIBERO suites | `{"libero_spatial":[0,1,2,3,4,5,6,7,8,9],"libero_object":[0,1,2,3,4,5,6,7,8,9],"libero_goal":[0,1,2,3,4,5,6,7,8,9],"libero_10":[0,1,2,3,4,5,6,7,8,9]}` / `null` | 400 | `EXPOSED_DEVELOPMENT` | false | false | standard_libero_baselines | 4 | `exp/libero-component-temporal-reuse@7ea83e1c0bea:experiments/standard_libero_baselines/results.json` |
| `ROBOTWIN_FEASIBILITY_600` | XPolicyLab ACT | RoboTwin 2.0 Easy | `["beat_block_hammer","click_alarmclock","dump_bin_bigbin","handover_block","open_laptop"]` / `"first 20 official-expert-eligible seeds in ascending order from 100000, selected per task"` | 600 | `EXPLORATORY_FEASIBILITY` | true | false | RoboTwin sealed exploratory pilot | 2 | `exp/robotwin-exploratory-sealed@f36be2d75d6d:research/audit_outputs/robotwin_exploratory_analysis.json` |

## Protocol details

### ACT_FROZEN_CONFIRM_140

- Environment seed rule: `340000 + 1000 * suite_index + 100 * task_id + state_id`
- Policy seed: `424242`
- Checkpoint: `per-task ACT 100000/pretrained_model checkpoints`
- Max episode steps: `{"libero_goal":300,"libero_10":520}`
- Contrasts: `act_same_target_fo20_vs_reverse20_confirm140`, `act_same_target_fo20_vs_fresh_confirm140`, `act_same_target_fullold20_vs_fo20_confirm140`, `act_c2_vs_fresh_confirm140`, `act_h16_vs_c2_confirm140`

### ACT_FROZEN_CONFIRM_140_POSTHOC_FACTORIAL

- Environment seed rule: `340000 + 1000 * suite_index + 100 * task_id + state_id`
- Policy seed: `424242`
- Checkpoint: `per-task ACT 100000/pretrained_model checkpoints`
- Max episode steps: `{"libero_goal":300,"libero_10":520}`
- Contrasts: `act_same_target_fullold20_vs_reverse20_confirm140`, `act_same_target_factorial_interaction_confirm140`

### ACT_FROZEN_CONFIRM_140_POSTHOC_H8

- Environment seed rule: `340000 + 1000 * suite_index + 100 * task_id + state_id`
- Policy seed: `424242`
- Checkpoint: `per-task ACT 100000/pretrained_model checkpoints`
- Max episode steps: `{"libero_goal":300,"libero_10":520}`
- Contrasts: `act_coherent_h8_vs_h16_posthoc140`

### ACT_OBJECT_DEV_126

- Environment seed rule: `330000 + 100 * task_id + state_id`
- Policy seed: `424242`
- Checkpoint: `/home/wjq/checkpoints/zeromidnight_act_libero_object`
- Max episode steps: `280`
- Contrasts: `act_c2_vs_fresh_dev126`, `act_fixedclock_arm16_grip32_vs_h16_dev126`, `act_fixedclock_h32_vs_h16_dev126`, `act_fixedclock_arm16_grip32_vs_h32_dev126`, `act_coherent_h8_vs_h16_dev126`

### ACT_LANDSCAPE_180

- Environment seed rule: `1000 + state_id (reused within each paired task)`
- Policy seed: `None`
- Checkpoint: `/home/thor/projects/checkpoints/zeromidnight_act_libero_object (byte-identical to /home/wjq/checkpoints/zeromidnight_act_libero_object)`
- Max episode steps: `280`
- Contrasts: `act_landscape_arm2_grip16_vs_arm2_grip2`, `act_landscape_arm4_grip16_vs_arm4_grip4`, `act_landscape_arm8_grip16_vs_arm8_grip8`, `act_landscape_arm4_grip32_vs_arm4_grip4`, `act_landscape_arm4_grip32_vs_arm4_grip16`

### ACT_STATIC_GRID_50

- Environment seed rule: `1000 + state_id`
- Policy seed: `None`
- Checkpoint: `/home/thor/projects/checkpoints/zeromidnight_act_libero_object (byte-identical audited checkpoint)`
- Max episode steps: `280`
- Contrasts: `act_static_grid_task0_context`

### ACT_GATE_M_HELDOUT_130

- Environment seed rule: `330000 + 100 * task_id + state_id`
- Policy seed: `424242`
- Checkpoint: `/home/wjq/checkpoints/zeromidnight_act_libero_object`
- Max episode steps: `280`
- Contrasts: `act_gate_m2_vs_h16_heldout130`, `act_gate_m2_vs_h13_heldout130`, `act_gate_m2_vs_shuffled_heldout130`, `act_coherent_h13_vs_h16_heldout130`, `act_gate_shuffled_vs_h16_heldout130`, `act_gate_h13_vs_shuffled_heldout130`

### SMOLVLA_PRIMARY_160

- Environment seed rule: `360000 + 1000 * suite_index + 100 * task_id + state_id`
- Policy seed: `SHA256(policy,suite,task,state,environment_seed,physical_query_step)`
- Checkpoint: `/home/wjq/checkpoints/HuggingFaceVLA_smolvla_libero`
- Max episode steps: `null`
- Contrasts: `smolvla_arm8_grip16_vs_h8_pooled`, `smolvla_arm8_grip16_vs_h8_libero_spatial`, `smolvla_arm8_grip16_vs_h8_libero_goal`, `smolvla_arm8_grip16_vs_h8_libero_object`, `smolvla_arm8_grip16_vs_h8_libero_10`

### SMOLVLA_CAPACITY_H16_160

- Environment seed rule: `360000 + 1000 * suite_index + 100 * task_id + state_id`
- Policy seed: `SHA256(policy,suite,task,state,environment_seed,physical_query_step)`
- Checkpoint: `/home/wjq/checkpoints/HuggingFaceVLA_smolvla_libero`
- Max episode steps: `null`
- Contrasts: `smolvla_h16_vs_h8_pooled`, `smolvla_arm8_grip16_vs_h16_pooled`, `smolvla_h16_vs_h8_libero_spatial`, `smolvla_arm8_grip16_vs_h16_libero_spatial`, `smolvla_h16_vs_h8_libero_goal`, `smolvla_arm8_grip16_vs_h16_libero_goal`, `smolvla_h16_vs_h8_libero_object`, `smolvla_arm8_grip16_vs_h16_libero_object`, `smolvla_h16_vs_h8_libero_10`, `smolvla_arm8_grip16_vs_h16_libero_10`

### SMOLVLA_ROBUSTNESS_160

- Environment seed rule: `360000 + 1000 * suite_index + 100 * task_id + state_id`
- Policy seed: `SHA256(smolvla,suite,task,state,environment_seed,physical_query_step)`
- Checkpoint: `/home/wjq/checkpoints/HuggingFaceVLA_smolvla_libero`
- Max episode steps: `null`
- Contrasts: `smolvla_arm4_grip32_vs_arm4_grip4_pooled`, `smolvla_arm4_grip32_vs_arm4_grip4_libero_spatial`, `smolvla_arm4_grip32_vs_arm4_grip4_libero_object`, `smolvla_arm4_grip32_vs_arm4_grip4_libero_goal`, `smolvla_arm4_grip32_vs_arm4_grip4_libero_10`

### SMOLVLA_STANDARD_400

- Environment seed rule: `None`
- Policy seed: `None`
- Checkpoint: `HuggingFaceVLA/smolvla_libero`
- Max episode steps: `null`
- Contrasts: `smolvla_standard_baseline_libero_spatial`, `smolvla_standard_baseline_libero_object`, `smolvla_standard_baseline_libero_goal`, `smolvla_standard_baseline_libero_10`

### ROBOTWIN_FEASIBILITY_600

- Environment seed rule: `frozen eligible-seed list; no method-specific replacement`
- Policy seed: `None`
- Checkpoint: `per-task final seed-0 6000-epoch policy_last.ckpt`
- Max episode steps: `null`
- Contrasts: `robotwin_feasibility_600_context`, `robotwin_fo1s_vs_newest_100`
