# Frozen LIBERO-10 task-specific manipulation opportunities

The executable source of truth is `TASK_MANIPULATION_OPPORTUNITIES.json`.
Every predicate is position-only. Object names, sites, joints, and fixture geoms
are exact simulator identifiers. Thresholds use compiled object/site geometry
plus the 0.040 m Panda finger-travel margin; no demonstration quantity is used.

| Task | Manipulated body/site | Target | Ordered manipulation stages | Opportunity variables | Orientation |
|---|---|---|---|---|---|
| 0: alphabet soup + tomato sauce into basket | `alphabet_soup_1`, `tomato_sauce_1` contact geoms | `basket_1_contain_region` | acquire soup; place soup; acquire sauce; place sauce | `gripper0_grip_site`, `geom_xpos`, `geom_rbound`, target `site_xpos/site_xmat/site_size` | position only |
| 1: cream cheese + butter into basket | `cream_cheese_1`, `butter_1` contact geoms | `basket_1_contain_region` | acquire cheese; place cheese; acquire butter; place butter | same geometry variables | position only |
| 2: turn on stove + place moka pot | `flat_stove_1_button` geoms `g5:g22`; `moka_pot_1` contact geoms | `flat_stove_1_cook_region` | engage/turn knob; acquire moka pot; place moka pot | EEF/geom variables; `flat_stove_1_button` qpos; cook-region site pose | position only |
| 3: bowl into bottom drawer + close | `akita_black_bowl_1`; bottom-handle geoms `white_cabinet_1_g40:g42` | `white_cabinet_1_bottom_region` | acquire bowl; place bowl; engage/close drawer | EEF/geom variables; `white_cabinet_1_bottom_level` qpos; drawer-region site pose | position only |
| 4: two mugs onto left/right plates | `porcelain_mug_1`, `white_yellow_mug_1` | `plate_1`, `plate_2` | acquire/place white mug; acquire/place yellow-white mug | EEF and object/plate contact-geom poses/radii | position only |
| 5: book into caddy back | `black_book_1` | `desk_caddy_1_back_contain_region` | acquire book; place in back compartment | EEF/object geoms; caddy target-site pose/size | position only |
| 6: mug on plate + pudding right of plate | `porcelain_mug_1`, `chocolate_pudding_1` | `plate_1`, `living_room_table_plate_right_region` | acquire/place mug; acquire/place pudding | EEF/object/plate geoms; right-region site pose/size | position only |
| 7: soup + cream cheese into basket | `alphabet_soup_1`, `cream_cheese_1` | `basket_1_contain_region` | acquire/place soup; acquire/place cheese | EEF/object geoms; basket target-site pose/size | position only |
| 8: both moka pots on stove | `moka_pot_1`, `moka_pot_2` | `flat_stove_1_cook_region` | acquire/place right pot; acquire/place left pot | EEF/object geoms; cook-region site pose/size | position only |
| 9: mug into microwave + close | `white_yellow_mug_1`; handle geoms `microwave_1_g14:g16` | `microwave_1_heating_region` | acquire mug; place mug; engage/close door | EEF/geom variables; `microwave_1_microjoint` qpos; heating-region site pose | position only |

The ordered stages follow the official language's noun order where conjunctions
do not specify a temporal order. Exact completion is always evaluated with the
corresponding LIBERO BDDL predicate.
