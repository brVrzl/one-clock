# Verified fact sheet

Audit date: 2026-08-21. This file is the replacement research source of truth. Numerical provenance is the named raw artifact and the read-only scripts under [`research/audit_tools/`](audit_tools/).

# Verified facts

## System and data

- The auditable empirical system is a frozen, deterministic-at-inference LeRobot ACT policy on single-arm LIBERO Object. It predicts 100 seven-dimensional actions per query; temporal ensembling is disabled in the audited rollouts. Evidence: per-run metadata, checkpoint config, `scripts/run_libero_gate0.py`, and [`system_contract_reaudit.md`](system_contract_reaudit.md).
- The LIBERO action partition is arm/end-effector indices 0–5 and gripper index 6. The arm command is relative Cartesian translation plus axis-angle rotation; the environment consumes the gripper command by sign. Evidence: environment/controller source and [`system_contract_reaudit.md`](system_contract_reaudit.md).
- Across 709,241 historical saved actions, 551,635 gripper scalars have magnitude above one. Gripper magnitude beyond sign does not change the LIBERO gripper command. Evidence: [`rollout_evidence_recomputed.json`](audit_outputs/rollout_evidence_recomputed.json).
- The demonstration dataset contains 454 successful episodes, 66,984 frames, ten LIBERO Object tasks, and is sampled at 10 Hz. Audited rollouts run at 20 Hz. Evidence: pinned dataset metadata and [`reliability_and_smoothness_recomputed.json`](audit_outputs/reliability_and_smoothness_recomputed.json).
- In fixed group execution, a single ACT query is made when any group expires; only expired groups install their fresh slices. The executed vector can combine predictions made from different observations. Diagonal group schedules are equivalent to global schedules. Evidence: executor source and complete step traces.
- No valid project RoboTwin rollout is linked to an exact checkpoint, upstream commit, and verified policy action contract. RoboTwin scientific conclusions are currently **NOT REPRODUCIBLE**.

## Static rollout observations

- On task 0 over all 50 official states, global success counts for execution horizons 1, 2, 4, 8, and 16 are respectively 29, 31, 42, 45, and 42. Corresponding query rates are approximately 1.000, .501, .252, .128, and .065. Evidence: complete raw traces recomputed by [`recompute_rollout_evidence.py`](audit_tools/recompute_rollout_evidence.py).
- On the same 50 states, fixed group `(arm=4, gripper=16)` succeeds 47 times versus 42 for global 4 at nearly equal query rate. Paired outcomes contain five group-only successes and no global-only success; the original exact two-sided McNemar diagnostic is .0625. The audit’s paired-state bootstrap difference CI is [.02,.18]. This is a post-hoc selected task-0 observation, not a confirmatory generalization.
- Across a common post-hoc configuration set on all ten tasks, global 16 has macro task success .699 and group `(4,16)` has .734. The task-cluster bootstrap CI for the .035 difference is [-.020,.085], and these two schedules are not query matched. Per-task differences are `[+.10,+.10,+.10,-.10,0,0,-.10,+.10,0,+.15]`.
- There is only one seed for each official initial state in the static grids. Seed and state effects cannot be separated.

## Direct matched-query negative result

- The selective-commitment manifest contains 60 entries, all 60 logged file hashes validate, and the experiment contains 1,200 episodes. Fixed query schedules in the traces are correct. Evidence: manifest and [`rollout_evidence_recomputed.json`](audit_outputs/rollout_evidence_recomputed.json).
- Relative to global replacement, the tested group-wise accept/retain rule changes absolute success by −.26 at q=4, −.20 at q=8, and −.29 at q=16 over 200 paired episodes each. Paired-episode 95% bootstrap CIs are `[-.35,-.17]`, `[-.285,-.115]`, and `[-.375,-.205]`; task-cluster CIs are also entirely below zero. This falsifies that specific rule.

## Gate-2B accounting and maps

- Gate-2B contains 90 candidate maps because `3 phases × (5 global + 25 group) = 90`. Each map uses 230 initial-state conditions (50 for task 0; 20 for each other task), giving 20,700 candidate rollouts. Two combined maps add 460, giving 21,160 rollouts total. Evidence: [`gate2b_recomputed.json`](audit_outputs/gate2b_recomputed.json).
- Saved aggregate accounting contains 1,227,551 candidate ACT queries and 14,096 combined-map queries, totaling 1,241,647; it contains 3,796,291 environment steps. The artifacts do not permit per-call duplicate/cache/invalid-output verification.
- Gate-2B phase is fixed elapsed step divided by the 280-step time limit: early steps 0–93, middle 94–186, and late 187–279. It is not normalized realized-rollout duration or semantic task progress.
- Recomputed point-max maps are: global early4/middle16/late8; group early `(4,16)`, middle `(16,16)`, late `(16,8)`. The previously stated late `(16,16)` group cell is false.
- The global point maxima are not sharp: all five global candidates are descriptively tied with the selected candidate by the audit’s unadjusted task bootstrap in every phase. Tied group candidates number 11/25 early, 3/25 middle, and 22/25 late. Full point estimates, sample counts, task rates, query rates, and CIs are in [`gate2b_candidate_support_curves.csv`](audit_outputs/gate2b_candidate_support_curves.csv).
- The combined global phase map differs from static global16 by +.014 macro success with task-bootstrap CI [-.045,.070]. The combined group map differs from static `(4,16)` by +.013 with CI [-.017,.045]. The same 230 conditions were used to select and evaluate the maps.

## Reliability target and estimator

- The demonstration-consistency target is exactly reproducible from the pinned demonstrations and saved ACT chunks. Arm validity requires normalized translation and rotation error below a threshold; gripper validity requires normalized magnitude and sign agreement; support is the cumulative AND of point validity from offset zero through (k). Evidence: [`recompute_reliability_and_smoothness.py`](audit_tools/recompute_reliability_and_smoothness.py).
- At threshold one, arm point validity is .5596 but prefix validity is .1309; gripper point validity is .7186 and prefix validity .2940. Cumulative prefix construction erases 76.6% of later arm-valid and 59.1% of later gripper-valid cells after an earlier failure. Later recoveries occur in 91.5% of arm samples and 86.8% of gripper samples.
- Arm mean prefix horizon is threshold-sensitive: 1.77 at θ=.5, 9.46 at θ=1, 28.30 at θ=1.5, and 51.01 at θ=2.0. At θ=1 the gripper magnitude condition adds essentially no information beyond sign.
- Under demonstration-duration thirds and θ=1, censor-aware k=0..37 AUCs are arm `[.177,.267,.275]` and gripper `[.369,.591,.481]`. Under quartiles they are arm `[.186,.182,.384,.220]` and gripper `[.402,.393,.780,.233]`. The segmentation materially changes the apparent pattern.
- The later Y_refresh target is not demonstration consistency. It compares a source prediction with the same ACT policy freshly queried at a future demonstrated observation. Against the demonstration target, arm cell agreement is .4623 and prefix-horizon Spearman correlation .0224; gripper agreement is .8210 and correlation .4068.
- The original raw fresh-action cache and target-comparison construction code are missing. The portable Y_refresh bundle is internally auditable but exact construction is **NOT REPRODUCIBLE** without recomputing fresh ACT actions.
- The chunk-only estimator rerun reproduces all 60 audited scalars to maximum difference 1.7e-16. On the saved Y_refresh bundle its pooled binary metrics are AUROC .9224, AUPRC .9109, Brier .1136, and ECE .0314. Decoded prefix-horizon MAE is 16.71 and within±2 accuracy is .2009, below the predeclared .50 usefulness threshold.
- The source-context ablation reproduces exactly: generated tables are byte-identical and JSON differs only in recorded Git HEAD. Adding source state, ACT latent, or both does not improve the chunk-only condition on the predeclared primary criteria.

## Offline smoothness and temporal-expert observations

- Demonstration arm prefix horizon correlates with current velocity (Spearman −.284), acceleration (−.249), jerk (−.234), and future-five-step velocity (−.443). A causal-feature linear ridge model has test R² .037; adding future explanatory smoothness features reaches .114. Smoothness is a confound but does not explain most variance under these simple models.
- Demonstration gripper horizon has weak causal-feature relationships; future gripper change rate correlates −.387 and an explanatory ridge model reaches R² .109.
- In a new held-out sparse-overlap audit with 5,118 target actions from 41 test episodes, two to six saved temporal predictions are available per target (mean 4.46). Group-balanced normalized error is .7728 for newest, .7404 for uniform, .7112 for exponential, and .7084 for similarity weighting. The similarity-vs-newest episode-bootstrap improvement CI is `[-.0864,-.0453]`.
- The same sparse audit’s hard oracle errors are .4788 for one scalar temporal expert and .4130 for independently selected arm/gripper experts. The group-balanced groupwise oracle advantage is .0658; under normalized dimension weighting it is .0301. These are teacher-forced upper bounds from sparse saved sources, not closed-loop results.
- A control-semantic Gate-3A0 re-analysis uses normalized translation error, SO(3) rotation error, and gripper sign/event errors on the same 5,118 held-out sparse targets. Its dimension-weighted semantic error is .65754 for newest, .62702 for validation-selected age-exponential weighting, .63401 for validation-selected CogACT cosine, and .62471 for validation-selected control-semantic similarity. Evidence: [`gate3a0_sparse_group_consistency.json`](audit_outputs/gate3a0_sparse_group_consistency.json).
- With the semantic aggregation operator held equal, control-semantic similarity differs from validation-selected CogACT cosine by −.00894 in dimension-weighted semantic error; the paired episode-bootstrap CI is `[-.01613,-.00203]`. Against age-exponential raw aggregation, the difference is −.00231 with CI `[-.00715,.00211]`, so that comparison is unresolved.
- Replacing raw action interpolation with a projected SO(3) rotation mean and gripper-sign vote changes the semantic-similarity primary error by +.00053 with CI `[-.00070,.00174]`. This aggregation modification is not supported as an improvement in the sparse cohort.
- Deployable arm/gripper and translation/rotation/gripper similarity weighting are worse than scalar semantic similarity by +.00460 (CI `[.00267,.00724]`) and +.00771 (CI `[.00019,.01302]`), respectively. The validation-selected consistency-gated semantic-three residual differs by +.00058 (CI `[-.00060,.00207]`), providing no held-out benefit.
- Under the control-semantic hard-oracle objective, complete-action scalar, arm/gripper, and translation/rotation/gripper errors are .48137, .46698, and .41665. The latter two are unattainable teacher-forced oracles; they do not establish a deployable group method.

# Strong evidence

- Static execution horizon matters for at least this ACT checkpoint and LIBERO task 0; the complete curve is nonmonotonic.
- The evaluated static landscape is task-dependent. The best evaluated point varies, but the size of true out-of-task heterogeneity is not isolated from finite-sample selection.
- The particular matched-query selective group retention mechanism is harmful across the tested ten-task suite.
- Thresholded prefix support is a lossy and unstable representation of continuous temporal prediction error.
- Existing saved temporal predictions contain enough diversity for simple temporal ensembling to reduce held-out offline demonstration error in the sparse cohort.
- In the sparse teacher-forced cohort, a control-semantic similarity kernel has a held-out offline advantage over validation-tuned full-vector CogACT cosine when aggregation is held fixed.

# Weak evidence

- A group-specific temporal selector may have extra oracle headroom over scalar selection. The advantage is modest under dimension-balanced error and may be physically inconsistent.
- Translation/rotation/gripper oracle freedom has more control-semantic headroom than arm/gripper freedom, but tested deployable group similarities do not capture it.
- The middle Gate-2B phase favors long point estimates more stably than early/late in split resampling, but the phase definition, in-sample selection, and absent traces prevent a semantic claim.
- Action smoothness accounts for some temporal-support variation, especially with future/noncausal features, but tested simple causal linear predictors explain little.
- Task or state context may affect temporal prediction competence, but current targets and ablations do not isolate the causal context signal.

# Unsupported previous conclusions

- A universal group-wise schedule improves over the best global schedule.
- Gate-2B phase conditioning improves closed-loop success over static controls.
- Early, middle, and late correspond to semantic manipulation phases.
- Marginal arm/gripper optima prove different intrinsic group timescales.
- The late Gate-2B cell demonstrates cross-group coupling or non-additivity.
- Demonstration-consistency support predicts the execution horizon that improves success.
- Offline MSE improvement implies closed-loop policy improvement.
- A binary reliability head’s performance on Y_refresh establishes predictability of demonstration support or control quality.
- Current data support left/right-arm conclusions, dexterous conclusions, RoboTwin performance, real-robot performance, or VLA generalization.
- Dynamic execution horizon alone is a novel ICRA 2027 contribution.

# Contradicted previous conclusions

- Gate-2B used normalized thirds of each realized rollout.
- Gate-2B’s late group point maximum is arm16/gripper16; it is arm16/gripper8.
- The selected Gate-2B optima are sharp.
- Reliability is generally unlearnable; binary Y_refresh cells are predicted well, while horizon decoding fails.
- A continuous gripper-magnitude error is aligned with the LIBERO control contract.
- The tested matched-query group-wise selective-commitment rule improves success.
- A generic temporal-expert gate is novel relative to CogACT and Temporal Action Selection.
- The current consistency-gated group residual improves over scalar semantic temporal aggregation.

# Unknowns

- Whether lower offline temporal-expert error predicts higher closed-loop success.
- Whether dense every-step ACT predictions show the same oracle and non-oracle gains as the sparse cache.
- Whether group-wise routing improves over scalar routing enough to justify cross-group consistency risk.
- Whether a static, training-free similarity ensemble matches any learned router on this checkpoint.
- Whether the sparse control-semantic similarity advantage over CogACT cosine survives dense every-step candidates or produces closed-loop success gains.
- Whether contact events, semantic progress, or task-specific phases explain more than normalized time and action smoothness.
- Whether demonstration action is an appropriate unique deployment target in this multimodal imitation setting.
- Whether a directly defined value of fresh observation/re-querying has exploitable, learnable headroom.
- Whether any result transfers to a VLA, RoboTwin, bimanual manipulation, dexterous hands, or a real robot.
- Which exact early checkpoint bytes produced the oldest rollouts.
