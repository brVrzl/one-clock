# Overnight ICRA handoff

Updated: 2026-09-03 10:07:54 +0800

1. Branch: `exp/icra27-crosssuite-query-allocation`
2. HEAD: `dc19c4110ecb973d3fa076f10d3e90ebb34d778d`
3. Track-A preregistration SHA: `40549d876c0e09fad4e8033b3206f6018f53ece5`
4. Reviewer-supplement preregistration SHA: `f44a7605246d4c9ea82f4d19ad61833e8fb13eb8`
5. Track-A final count: 2700/2700
6. Track-A technical failure count: 0
7. Track-A scientific-analysis status: complete; canonical path `/home/wjq/workspace/one-clock-icra27-crosssuite-query-allocation/experiments/icra27_crosssuite_query_allocation/track_a/analysis.json`
8. Track-A headline preregistered labels: PENALTY_4X_CONFIRMED=PASS, DOSE_RESPONSE_SUPPORTED=PASS, MECHANISM_PASS_A=PASS, METHOD_PASS_A=FAIL, QUERY_EFFICIENT_TE_LEVEL_PERFORMANCE=PASS
9. R1A status: 64/1512 complete; 0 unresolved technical failures
10. R1B status: 0/252 complete; 0 unresolved technical failures
11. R1C status: 0/280 complete; 0 unresolved technical failures
12. R1D status: 0/100 complete; 0 unresolved technical failures
13. R2 eligibility decision: `R2_ENABLED_TECHNICALLY` frozen before R1 outcomes
14. R2 status: 0/160 complete; 0 unresolved technical failures
15. Active/completed PIDs: master=373835 (active), r1a_worker_0=374922 (active), r1a_worker_1=374923 (active), r1a_worker_2=374924 (active)
16. Log paths: `/home/wjq/workspace/one-clock-icra27-crosssuite-query-allocation/experiments/icra27_reviewer_supplement/orchestration/logs`, `/home/wjq/workspace/one-clock-icra27-crosssuite-query-allocation/experiments/icra27_crosssuite_query_allocation/track_a/logs`
17. Completion-marker counts: R1A 64/1512 complete; 0 unresolved technical failures; R1B 0/252 complete; 0 unresolved technical failures; R1C 0/280 complete; 0 unresolved technical failures; R1D 0/100 complete; 0 unresolved technical failures; R2 0/160 complete; 0 unresolved technical failures
18. Technical retries: 0
19. Remaining queue: see `/home/wjq/workspace/one-clock-icra27-crosssuite-query-allocation/experiments/icra27_reviewer_supplement/orchestration` markers and master log
20. Exact resume command: `bash /home/wjq/workspace/one-clock-icra27-crosssuite-query-allocation/experiments/icra27_reviewer_supplement/launch_watcher.sh --resume`

Pipeline failure marker: `none`
Original pipeline start epoch: `1788354953` (preserved on `--resume`)

## Authorized morning continuation

- Technical repair commit: `af1b54dc567973d47f0e234d98c9b83ab68e675b`
- Governing final analysis-only amendment commit: `e2fb21b`
- Track-A conditions: H16 357/450 (79.33%); H4 314/450 (69.78%); ARM4_GRIP32 335/450 (74.44%); H2 295/450 (65.56%); ARM2_GRIP16 321/450 (71.33%); TE_DENSE 288/450 (64.00%)
- Track-A contrasts: H16-H4 +9.56 pp, task-CI [+4.44,+15.33]; H4-H2 +4.22 pp, task-CI [+1.33,+7.11]; ARM4_GRIP32-H4 +4.67 pp, task-CI [+0.67,+9.11]; ARM2_GRIP16-H2 +5.78 pp, task-CI [+2.67,+9.56]; ARM4_GRIP32-H16 -4.89 pp, task-CI [-9.33,-0.22]; ARM2_GRIP16-H16 -8.00 pp, task-CI [-12.22,-3.78]; TE_DENSE-H16 -15.33 pp, task-CI [-21.56,-10.00]; TE_DENSE-ARM4_GRIP32 -10.44 pp, task-CI [-17.78,-3.78]
- B3 status: 0/8 task-policy shards complete; 0 technical failures; canonical analysis pending; worker PID 393797 (active).
- Analysis-followup PID: none; final mechanism relationship analysis pending.
- Temporal contract: `PASS_WITH_EXPLICIT_MULTI_RATE_MAPPING`. Standard ACT training/B2/B3 are 10 Hz; R1A/B/C/D are 20 Hz; R2A is 30 Hz. Seconds are the primary cross-family axis.
- Reviewer prelaunch canary: PASS.
- Scientific configuration confirmation: no condition, manifest, cohort, state, seed, statistic, decision rule, or preregistered launch order was changed. Governed supplement files remain those of `f44a7605246d4c9ea82f4d19ad61833e8fb13eb8`.
