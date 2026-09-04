# Phase-D scientific report

All rates and counts below retain the complete paired-block denominator.

## Integrity and provenance

The task predicates, failure taxonomy, replay schema, and component-swap protocol were frozen at `2026-09-04T16:31:26+08:00` in commit `b36693db6ec79c80b9cbe4fb3d25ed231f3c85fb`. Phase-1 success outcomes were first read at `2026-09-04T18:04:16+08:00`.

The stored Track-A LIBERO-10 cohort contains task-specific official state IDs mostly in 20--44, not states 0--14. It overlaps Phase-1 states 15--49, so Phase-1 is described as a complete confirmation execution cohort, not as state-held-out.

Exact open-loop reproduction passed for all 248 selected development source episodes and all 288 selected Phase-1 source episodes: zero initial-state, terminal-success, episode-length, or command mismatches.

## development_h4

Paired blocks: 150. Outcomes: rescue 26, harm 6, both fail 63, both succeed 55. Net rescue minus harm: 20.

Baseline failures: 89; never reached a predefined manipulation opportunity: 7; reached opportunity: 82.

Among 82 opportunity-reaching baseline failures, 23 were rescued and 59 still failed.

Rescue failure stages: `{"ACQUISITION_OR_ENGAGEMENT_FAILURE": 2, "LATER_STAGE_FAILURE": 21, "PRE_OPPORTUNITY_FAILURE": 3}`. Harm failure stages: `{"ACQUISITION_OR_ENGAGEMENT_FAILURE": 1, "LATER_STAGE_FAILURE": 5}`.

Rescue task stages: `{"acquire_butter": 1, "acquire_moka_pot": 3, "acquire_right_moka_pot": 5, "acquire_tomato_sauce": 3, "acquire_yellow_white_mug": 3, "close_bottom_drawer": 1, "close_microwave": 3, "place_alphabet_soup": 4, "place_cream_cheese": 1, "place_mug_in_microwave": 2}`. Harm task stages: `{"acquire_chocolate_pudding": 1, "acquire_cream_cheese": 1, "acquire_white_mug": 1, "close_bottom_drawer": 1, "place_chocolate_pudding_right": 1, "place_white_mug_left": 1}`.

Later-stage physical detail among rescues: `{"ACQUISITION_OR_ENGAGEMENT_FAILURE": 7, "POST_ACQUISITION_LOSS": 2, "PRE_OPPORTUNITY_FAILURE": 12}`; among harms: `{"ACQUISITION_OR_ENGAGEMENT_FAILURE": 3, "POST_ACQUISITION_LOSS": 1, "PRE_OPPORTUNITY_FAILURE": 1}`.

Both-direction rescue/harm swap table (SUCCESS/CENSORED): `{"harm": {"baseline_arm_plus_treatment_gripper": {"CENSORED": 6}, "treatment_arm_plus_baseline_gripper": {"CENSORED": 6}}, "rescue": {"baseline_arm_plus_treatment_gripper": {"CENSORED": 26}, "treatment_arm_plus_baseline_gripper": {"CENSORED": 24, "SUCCESS": 2}}}`. Censored swaps do not establish failure.

## development_h2

Paired blocks: 150. Outcomes: rescue 22, harm 1, both fail 75, both succeed 52. Net rescue minus harm: 21.

Baseline failures: 97; never reached a predefined manipulation opportunity: 12; reached opportunity: 85.

Among 85 opportunity-reaching baseline failures, 22 were rescued and 63 still failed.

Rescue failure stages: `{"LATER_STAGE_FAILURE": 22}`. Harm failure stages: `{"LATER_STAGE_FAILURE": 1}`.

Rescue task stages: `{"acquire_butter": 4, "acquire_chocolate_pudding": 1, "acquire_cream_cheese": 1, "acquire_moka_pot": 2, "acquire_tomato_sauce": 2, "acquire_yellow_white_mug": 6, "close_bottom_drawer": 1, "close_microwave": 2, "place_alphabet_soup": 2, "place_mug_in_microwave": 1}`. Harm task stages: `{"acquire_tomato_sauce": 1}`.

Later-stage physical detail among rescues: `{"ACQUISITION_OR_ENGAGEMENT_FAILURE": 10, "INTERACTION_EXECUTION_FAILURE": 2, "POST_ACQUISITION_LOSS": 1, "PRE_OPPORTUNITY_FAILURE": 9}`; among harms: `{"ACQUISITION_OR_ENGAGEMENT_FAILURE": 1}`.

Both-direction rescue/harm swap table (SUCCESS/CENSORED): `{"harm": {"baseline_arm_plus_treatment_gripper": {"CENSORED": 1}, "treatment_arm_plus_baseline_gripper": {"CENSORED": 1}}, "rescue": {"baseline_arm_plus_treatment_gripper": {"CENSORED": 22}, "treatment_arm_plus_baseline_gripper": {"CENSORED": 22}}}`. Censored swaps do not establish failure.

## phase1_h4

Paired blocks: 350. Outcomes: rescue 47, harm 20, both fail 154, both succeed 129. Net rescue minus harm: 27.

Baseline failures: 201; never reached a predefined manipulation opportunity: 21; reached opportunity: 180.

Among 180 opportunity-reaching baseline failures, 43 were rescued and 137 still failed.

Rescue failure stages: `{"ACQUISITION_OR_ENGAGEMENT_FAILURE": 5, "LATER_STAGE_FAILURE": 38, "PRE_OPPORTUNITY_FAILURE": 4}`. Harm failure stages: `{"ACQUISITION_OR_ENGAGEMENT_FAILURE": 2, "LATER_STAGE_FAILURE": 18}`.

Rescue task stages: `{"acquire_butter": 1, "acquire_chocolate_pudding": 1, "acquire_moka_pot": 6, "acquire_right_moka_pot": 8, "acquire_tomato_sauce": 4, "acquire_yellow_white_mug": 5, "close_bottom_drawer": 1, "close_microwave": 7, "place_alphabet_soup": 7, "place_book_back_compartment": 1, "place_cream_cheese": 1, "place_mug_in_microwave": 2, "place_white_mug_left": 1, "place_yellow_white_mug_right": 2}`. Harm task stages: `{"acquire_chocolate_pudding": 1, "acquire_cream_cheese": 2, "acquire_moka_pot": 1, "acquire_white_mug": 1, "close_bottom_drawer": 6, "place_book_back_compartment": 1, "place_chocolate_pudding_right": 4, "place_mug_in_microwave": 1, "place_white_mug_left": 1, "place_white_mug_on_plate": 2}`.

Later-stage physical detail among rescues: `{"ACQUISITION_OR_ENGAGEMENT_FAILURE": 13, "INTERACTION_EXECUTION_FAILURE": 1, "POST_ACQUISITION_LOSS": 3, "PRE_OPPORTUNITY_FAILURE": 21}`; among harms: `{"ACQUISITION_OR_ENGAGEMENT_FAILURE": 4, "INTERACTION_EXECUTION_FAILURE": 3, "POST_ACQUISITION_LOSS": 5, "PRE_OPPORTUNITY_FAILURE": 6}`.

Both-direction rescue/harm swap table (SUCCESS/CENSORED): `{"harm": {"baseline_arm_plus_treatment_gripper": {"CENSORED": 16, "SUCCESS": 4}, "treatment_arm_plus_baseline_gripper": {"CENSORED": 20}}, "rescue": {"baseline_arm_plus_treatment_gripper": {"CENSORED": 46, "SUCCESS": 1}, "treatment_arm_plus_baseline_gripper": {"CENSORED": 44, "SUCCESS": 3}}}`. Censored swaps do not establish failure.

## Interpretation

This analysis localizes recorded command-trace effects. It does not establish that an online ACT policy would preserve its arm trajectory after altered gripper execution.

`PHASE-D RESULT B — STAGE-LOCALIZED RESCUE WITHOUT SUFFICIENCY`
