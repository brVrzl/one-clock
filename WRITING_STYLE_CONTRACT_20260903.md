# Writing style contract

Date: 2026-09-03 (Asia/Shanghai)

This contract governs the next manuscript-writing session. Internal provenance terms remain authoritative for scientific bookkeeping, but ordinary paper prose should use reader-facing scientific English.

## Internal-to-reader language

Internal labels may appear in one governance/provenance table. They should not recur through the Abstract, Introduction, or ordinary Results prose.

| Internal label | Ordinary scientific English |
|---|---|
| `PREREGISTERED_CONFIRMATORY` | “tested in the preregistered confirmatory cohort” |
| `EXPOSED_DEVELOPMENT_CHARACTERIZATION` | “measured on the development cohort” |
| `POST_HOC_DESCRIPTIVE` | “reported as an observed descriptive pattern” |
| `POST_HOC_QUERY_MATCHED_EXTENSION` | “we added a query-matched comparison after the primary study” |
| `POST_HOC_SPATIAL_FACTORIAL_COMPLETION` | “we later completed the missing Spatial comparison” |
| `POST_HOC_SUPPORTING_INTERACTION` | “a later supporting analysis of the frozen outcomes” |
| `NON_IDENTIFYING` or `NON_IDENTIFYING_POST_HOC_CHARACTERIZATION` | “this comparison cannot distinguish the competing explanations” |
| `NO_FROZEN_DISCRIMINATIVE_CRITERION` | “we had not specified in advance what curve pattern would count as support, so we report it descriptively” |
| `CONTRADICTED_AS_AN_ORDERING_PREDICTOR_WITHIN_ARM` | “the metric ranks rotation above translation, while the behavioral effect ranks translation above rotation” |
| `NULL / UNSUPPORTED` | “the prespecified analysis did not support this relationship” |
| `INSUFFICIENT_AS_A_COMPLETE_EXPLANATION` | “this observation alone does not explain the behavioral difference” |
| `UNRESOLVED` | “the tested diagnostics did not identify the mechanism” |
| `FROZEN_GATE_INELIGIBLE` | “the planned evaluation was not run because its prespecified eligibility window had expired” |

## Rule 1: one paragraph, one claim

Every Results paragraph begins with its scientific conclusion. Use a direct result sentence instead of a procedural opening such as “We next investigate...” Each paragraph should have one main claim, the evidence needed to support it, and the qualification needed to bound it.

## Rule 2: result first, qualification second

State the concrete result and its magnitude first. Put the relevant cohort, provenance, or interpretive limitation later in the same paragraph, after the reader understands what happened. Qualification must remain complete, but it should not obscure the result by turning every opening clause into a disclaimer.

## Rule 3: Methods must read like a recipe

Define `A0G20` and `A20G0` through execution before relying on labels. A suitable sequence is:

1. At control step `t`, query the policy to obtain a chunk of predicted future actions.
2. A Fresh component executes the offset-0 entry from the chunk queried at `t`.
3. A d=20 stale component executes the offset-20 entry from the chunk queried 20 steps earlier.
4. Both entries target the same physical control step `t`.
5. For the first 20 steps, when the older source is unavailable, execute the Fresh offset-0 action for every component.

Only after this concrete explanation should the notation `q+k=t` be introduced.

## Rule 4: state the source-age/lookahead coupling plainly

Same-target alignment uses `k=d`, so an older source observation necessarily comes with a longer prediction horizon. The manuscript must include clear wording equivalent to:

> We therefore measure the joint effect of using an older observation and a longer-lookahead prediction; this experiment does not separate those two factors.

This limitation should appear early, near the first explanation of the intervention, rather than being buried in the supplement.

## Rule 5: concentrate defensive detail

Put full preregistration and provenance precision primarily in the Methods/provenance table, Limitations, and Supplement. Abstract, Introduction, and Results prose should state the scientific result in ordinary language and then give the limitation needed to interpret it. Do not repeat internal governance labels as a rhetorical prefix.

The query-rate limitation is important enough to appear early: same-target probes query once per executed step, have policy-query rate 1.0, and are diagnostic interventions rather than deployment executors. Coherent H16 queries much less often. Do not translate query rate into FLOPs, latency, or equal-compute claims.

## Rule 6: no mechanism laundering

When a diagnostic is null, cannot distinguish explanations, lacks a prespecified decision rule, or gives the opposite ordering from behavior, say exactly that in ordinary English. Do not turn a descriptive association into a causal mechanism, treat absence of detection as proof of no effect, or manufacture a positive explanation to simplify the story.

Specific boundaries for this paper:

- dispersion ranks rotation above translation, while behavioral harm ranks translation above rotation;
- persistence alone does not explain the arm-gripper difference;
- the occupancy moderator was not supported;
- forecastability was not tied to a prespecified curve criterion and is descriptive;
- command-discontinuity comparisons cannot distinguish the competing explanations;
- the causal mechanism remains unresolved.

## Sentence-level finish

- Use no em dashes in the Abstract and at most one per paragraph elsewhere.
- Use en dashes for numerical ranges such as `0.10–1.60 s`.
- Hedge once, at the level justified by the evidence.
- Prefer one idea per sentence and vary sentence length.
- Avoid repeated paragraph openings and heavy connective words when “but,” “so,” or a new sentence is clearer.
- Do not call the R1A shape a plateau, d=20 an optimum, rotation staleness free, or gripper staleness generically beneficial.
