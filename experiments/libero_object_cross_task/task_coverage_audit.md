# LIBERO Object cross-task checkpoint coverage audit

Audit date: 2026-08-19

The frozen checkpoint records `DorayakiLin/libero_object_25_08_23_lerobotv2.1`
in `train_config.json`. The dataset metadata snapshot inspected locally was
`cbf7122bbdbaa0c50517a6a4b2ae663d0e96e51a`. Its `meta/tasks.jsonl` contains
all ten LIBERO Object task descriptions, and its `meta/episodes.jsonl` contains
the demonstration counts shown below. Dataset task indices are not the same as
the LIBERO benchmark task IDs, so coverage was matched by language description.

The live `hf-libero` 0.1.4 / LeRobot 0.6.2 runtime exposed 50 official initial
states and a `(7,)` action space for every task below. Each environment was
instantiated and closed successfully with `MUJOCO_GL=egl`; therefore all ten
tasks were runtime-compatible. Task 0 is listed for coverage but was not
rerun: its complete 50-state result already exists in the static landscape
artifacts.

| task_id | task_name | dataset task index | present in training dataset | demonstrations | official init states | runtime-compatible |
|---:|---|---:|---|---:|---:|---|
| 0 | `pick_up_the_alphabet_soup_and_place_it_in_the_basket` | 4 | yes | 44 | 50 | yes (not rerun) |
| 1 | `pick_up_the_cream_cheese_and_place_it_in_the_basket` | 2 | yes | 45 | 50 | yes |
| 2 | `pick_up_the_salad_dressing_and_place_it_in_the_basket` | 6 | yes | 47 | 50 | yes |
| 3 | `pick_up_the_bbq_sauce_and_place_it_in_the_basket` | 3 | yes | 46 | 50 | yes |
| 4 | `pick_up_the_ketchup_and_place_it_in_the_basket` | 1 | yes | 45 | 50 | yes |
| 5 | `pick_up_the_tomato_sauce_and_place_it_in_the_basket` | 8 | yes | 42 | 50 | yes |
| 6 | `pick_up_the_butter_and_place_it_in_the_basket` | 7 | yes | 45 | 50 | yes |
| 7 | `pick_up_the_milk_and_place_it_in_the_basket` | 5 | yes | 45 | 50 | yes |
| 8 | `pick_up_the_chocolate_pudding_and_place_it_in_the_basket` | 9 | yes | 50 | 50 | yes |
| 9 | `pick_up_the_orange_juice_and_place_it_in_the_basket` | 0 | yes | 45 | 50 | yes |

Checkpoint/runtime contracts used for the cross-task pass are unchanged from
task 0: ACT chunk size 100, output action shape `(100, 7)` after removing the
batch dimension, relative control, image/state preprocessing from the existing
runner, and no temporal ensembling. The dataset metadata reports 454 total
demonstrations across the ten tasks.
