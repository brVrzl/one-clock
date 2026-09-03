# CPU completed-artifact handoff, 2026-09-03

Branch: `exp/icra27-crosssuite-query-allocation`

This handoff contains no partial R1/R2/B3 scientific value. No scientific
worker was signaled or modified.

## A. TE_DENSE effective age

Label: `POST_HOC_TE_EFFECTIVE_AGE_CHARACTERIZATION`.

The exact frozen implementation is not `exp(-0.01*age)`. Runtime rank `i=0`
is the oldest candidate and receives weight `exp(-0.01*i)`, making the
equivalent source-age weight `exp(+0.01*age)`. The coefficient remains 0.01;
chunk length remains 100.

- 105,947 executed TE steps across 450 completed cells.
- Candidate count: mean 79.127, p50 100, p95 100, maximum 100.
- Startup (`count<100`): 42,835 steps, mean 48.374, p50 48, p95 93,
  maximum 99.
- Steady state (`count=100`): 63,112 steps.
- Empirical pooled unweighted candidate age: mean 45.110 steps/2.255 s,
  p50 43/2.15 s, p95 93/4.65 s, maximum 99/4.95 s.
- Empirical normalized-weight age: mean 44.987 steps/2.249 s, p50
  43/2.15 s, p95 94/4.70 s, maximum support 99/4.95 s.
- Theoretical steady-state normalized-weight age: mean 57.697 steps/2.885 s,
  p50 62/3.10 s, p95 96/4.80 s, maximum support 99/4.95 s.
- Empirical normalized weight older than 0.50/1.00/2.00 s:
  0.827438/0.715163/0.522690.
- Theoretical steady-state counterparts: 0.932329/0.864005/0.705044.

Maximum temporal support and weighted mean effective age are distinct.

## B. TE_DENSE gripper aggregation

The executed native gripper command is continuous. Under the frozen upstream
coefficient and chunk length:

| Condition | mean abs(g) | abs(g)<0.25 | abs(g)<0.50 | sign/state-switch rate |
|---|---:|---:|---:|---:|
| H16 | 0.999286 | 0.017991 | 0.039253 | 0.024445 |
| H4 | 1.004481 | 0.015318 | 0.034149 | 0.027316 |
| ARM4_GRIP32 | 0.996363 | 0.018940 | 0.040457 | 0.020052 |
| H2 | 1.001660 | 0.016018 | 0.034631 | 0.020043 |
| ARM2_GRIP16 | 0.996343 | 0.017770 | 0.040325 | 0.023983 |
| TE_DENSE | 0.797359 | 0.142703 | 0.244094 | 0.010247 |

For TE_DENSE, mean `g=-0.153195`, median `g=-0.279394`, median
`abs(g)=1.018770`, and executed range is [-1.349301, 1.291061].

Candidate sign disagreement and weighted minority-sign mass are
`NOT_IDENTIFIABLE_FROM_EXISTING_TRACK_A_ARTIFACTS`: pre-aggregation candidate
chunks were not persisted. No rerollout is authorized. These diagnostics do
not imply that canonical temporal ensembling is intrinsically harmful.

## C. Per-suite absolute Track-A table

| Suite | H16 | H4 | A4G32 | H2 | A2G16 | TE_DENSE |
|---|---:|---:|---:|---:|---:|---:|
| LIBERO-10 | 82/150, 54.667% | 61/150, 40.667% | 81/150, 54.000% | 53/150, 35.333% | 74/150, 49.333% | 55/150, 36.667% |
| Goal | 137/150, 91.333% | 125/150, 83.333% | 125/150, 83.333% | 117/150, 78.000% | 120/150, 80.000% | 120/150, 80.000% |
| Spatial | 138/150, 92.000% | 128/150, 85.333% | 129/150, 86.000% | 125/150, 83.333% | 127/150, 84.667% | 113/150, 75.333% |

All eight suite-level contrasts are in
`track_a/te_dense_characterization/track_a_per_suite_contrasts.csv`.

## D. LOSO and suite concentration

ARM4_GRIP32-H4 is +13.333 pp in LIBERO-10, 0.000 in Goal, and +0.667
in Spatial. The pooled paired and task-cluster contrast is positive and the
all-LOSO-positive criterion technically passes. The leave-LIBERO-10-out point
estimate is +0.333 pp, so the non-LIBERO-10 numerical margin is minimal. This
does not describe the pooled inference as one block from disappearing.

ARM2_GRIP16-H2 is +14.000/+2.000/+1.333 pp in
LIBERO-10/Goal/Spatial. Its cross-suite pattern is not presented as equally
robust to ARM4_GRIP32-H4. Existing LOSO point estimates are reported; no LOSO
intervals existed and none were added.

## E. Moderator provenance

The preregistered moderator is closed-gripper occupancy/duty cycle, not
transition density: fraction of task-specific ACT training actions with
`action[6] < 0`. Canonical result: n=30, Spearman rho 0.192221, p 0.308853,
descriptive/null unsupported.

Transition density and trajectory duration retain the separate label
`OUTCOME_MOTIVATED_POST_HOC_SCOPE_CHARACTERIZATION`. That specification was
not expanded or calculated in this turn.

## F. Command-discontinuity interpretation

The confirmation D1 ordering is descriptively
`coherent H16 > Fresh > A20G0` for all three action groups, matching success
ordering only across three outcome-divergent conditions. This is not an
identified association or reverse mechanism finding.

Trajectory/state composition, treatment-dependent episode length, and
fixed-offset smoothness can all confound the comparison. Existing B1 age
curves measure cross-source disagreement, not intrinsic within-chunk offset
smoothness. Fresh and A20G0 have no same-source transitions, and no
protocol-guaranteed common positive prefix exists for every trajectory;
both proposed confound-free sensitivities are structurally unavailable.

Allowed conclusion: an outcome-confounded post-hoc characterization did not
provide identifiable support for a reduced-discontinuity explanation.

## G. Path and headline guardrails

`H2 -> ARM2_GRIP16 -> H16` is a measured path with +5.778 pp followed by an
additional +8.000 pp. The edges are not unique component contributions and
receive no contribution percentages.

No analogous H4 decomposition is valid: ARM4_GRIP32 to H16 changes both arm
and gripper horizons.

The main Track-A result is +4.667 pp for H4 to ARM4_GRIP32 and +5.778 pp for
H2 to ARM2_GRIP16 at nearly matched per-executed-step policy-query rates.
Coherent H16 remains the strongest frozen operating point.

## H. Scientific-worker status

At `2026-09-03T14:38:57+08:00`, no reviewer-supplement or B3 scientific
worker process was active. No process was stopped or signaled.

High-level technical markers only, without opening scientific outcomes:

- reviewer pipeline: `PIPELINE_FAILED` present; R1A, R1B, and R1C completion
  markers present; R1D launch marker present without an R1D completion marker;
  no R2 launch/completion marker observed;
- B3: eight task-policy completion markers present and no failure marker.

This handoff does not infer or report any R1/R2/B3 scientific value.

## I. Integrity confirmation

No rollout, manifest, cohort, condition, checkpoint, state, seed, statistic,
decision rule, worker, manuscript, LaTeX source, `CLAIMS.md`, or paper-facing
artwork was changed. The superseding 20-Hz temporal audit remains authoritative.
