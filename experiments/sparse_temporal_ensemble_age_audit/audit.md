# Sparse temporal-ensemble age audit

## Scope and stop condition

This audit preserves the results in `../sparse_temporal_ensemble_dev/` as evidence for the **candidate-index sparse temporal ensemble**. The orientation ambiguity was resolved in favor of validated canonical ACT, and the repaired h16 trio was rerun with fresh environment construction for every condition/state.

## Part A: existing-result audit

### A1. Same-target execution

The executor's target alignment is correct and does not have an indexing bug:

- candidate offset is exactly `t-q`;
- a source is included exactly while `0 <= t-q < H_pred`;
- the scheduled query at `q=t` is inserted before target extraction, so `A_t[0]` is present;
- records and candidates remain oldest to newest;
- hard execution selects the newest source;
- temporal aggregation never calls the policy.

Thus the historical implementation executes `a_{t|q}=A_q[t-q]` with the intended strict horizon endpoint.

### A2. Actual weight semantics

The historical operator is exactly `exp(-0.01*i)` over candidate index `i`, with candidates ordered oldest to newest. It slightly favors older chunks and is not physical-separation weighting.

Typical late-episode normalized weights below are listed oldest to newest. The physical-age column is the formula requested in this audit, `exp(-0.01*(q_newest-q))`, and therefore favors newer chunks.

| policy/cadence | candidates | candidate-index weights | requested newest-relative physical-age weights |
|---|---:|---|---|
| ACT h8 | 13 | .081623, .080810, .080006, .079210, .078422, .077642, .076869, .076104, .075347, .074597, .073855, .073120, .072393 | .045532, .049324, .053432, .057882, .062703, .067925, .073582, .079711, .086350, .093542, .101332, .109772, .118915 |
| ACT h16 | 7 | .147178, .145714, .144264, .142829, .141407, .140000, .138607 | .084031, .098611, .115721, .135800, .159362, .187013, .219462 |
| Smol h8 | 7 | .147178, .145714, .144264, .142829, .141407, .140000, .138607 | .110950, .120191, .130201, .141045, .152792, .165518, .179303 |
| Smol h16 | 4 | .253762, .251237, .248738, .246263 | .193547, .227129, .266538, .312786 |

### A3. Hard-baseline provenance

The prior fixed h8/h16 results of 38/40 did not use the current four-task panel:

| suite | prior 38/40 cohort | current sparse-TE cohort |
|---|---|---|
| libero_object | task6 | task3 |
| libero_spatial | task2 | task0 |
| libero_goal | task1 | task2 |
| libero_10 | task3 | task3 |

Both used states 10--19, seeds 2000--2009, 100k ACT checkpoints, the validated native preprocessing/postprocessing path, and the same success extraction. The three-task cohort replacement explains why pooled 38/40 cannot be treated as the reference for current 34/40 and 33/40. Prior result files did not independently record package versions, although their launch path/protocol used the validated ACT environment.

The shared `libero_10:task3` shard exposed a real pairing problem. Prior fixed h8 and current hard h8 were both 10/10; prior fixed h16 was 10/10, while current hard h16 was 8/10 (failures at states 11 and 16). More directly, all ten stored current hard/TE h16 pairs differed during t=0--15, with maximum action difference 0.311079. The task log records the original validator failure. No invalidated full task-10 shard or later rerun exists; the existing result was accepted only after the prefix assertion was removed.

## Part B: strict paired initialization

An initial object-task implementation smoke passed all comparisons exactly. After Part A localized the stored failure to `libero_10:task3`, the formal audit used task 3, h16, states 10, 11, and 16, and seeds 2000, 2001, and 2006.

Repeated resets of one task-10 environment failed B2--B5 even though the selected state ID, environment seed, policy seed, MuJoCo dynamic state, and low-dimensional observation were identical. The camera maximum differences were 213--224 pixel levels, initial ACT chunks differed by 0.732--2.078, and common-prefix actions differed by 0.093--0.212.

The root cause is reset state outside MuJoCo's flattened dynamic state. Repeated hard resets resampled static fixture positions: the white cabinet moved by up to 0.008339 m and the wine rack by up to 0.004750 m. These model-body positions changed the rendered policy inputs but were not represented in the saved dynamic state or robot proprioception.

The minimal repair is to construct a fresh environment for each condition/state under the same method-independent construction seed, then apply the frozen state/seed and policy RNG reset. With this repair, all three formal states passed:

| comparison | states 10, 11, 16 max difference |
|---|---:|
| flattened simulator state | 0 |
| all low-dimensional observations | 0 |
| agent-view RGB, shape 1×256×256×3 | 0 |
| wrist RGB, shape 1×256×256×3 | 0 |
| processed state/proprio | 0 |
| processed images | 0 |
| task conditioning | exact equality |
| initial postprocessed ACT chunk `A_0` | 0 |
| executed actions t=0--15 | 0 |
| post-action simulator states and observations | 0 |

The original two-method repair artifact is `pairing_audit_task10_fresh_env/summary.json`; the final three-method gate is `pairing_audit_task10_fresh_env_trio/summary.json`. Failed diagnostic artifacts are retained separately and must not be combined with repaired results.

## Weight-orientation resolution

Validated LeRobot 0.4.4 explicitly defines index zero as the oldest action. With positive coefficient 0.01, the h1 two-candidate weights for `[old, new]` are proportional to:

`[1, exp(-0.01)]`.

The proposed formula `exp(-0.01*(q_newest-q))` gives, for the same `[old, new]` sources:

`[exp(-0.01), 1]`.

These are reverses. The canonical ACT orientation was therefore made authoritative.

The ACT-orientation-preserving sparse subsampling used in the repaired experiment is:

`w(q) proportional to exp(-0.01*(q-q_oldest))`,

which gives `[1, exp(-0.01*h), exp(-0.02*h), ...]` in oldest-to-newest order. It is named `dense_equivalent_te`; no newest-relative decay was run.

## Repaired h16 trio outcome

Freshly paired hard h16 achieved 32/40, repaired candidate-index TE achieved 24/40, and dense-equivalent TE achieved 23/40. Dense-equivalent versus hard had 2 dense-only and 11 hard-only successes (net −9; exact two-sided McNemar p=0.02246). Candidate-index versus hard had 3 candidate-only and 11 hard-only successes (net −8; p=0.05737). Dense-equivalent and candidate-index differed on only one state (0 dense-only, 1 candidate-only).

The dense-equivalent deficit was present on every task, so the result is classified `DENSE_EQ_TE_HARMFUL`. Repaired candidate-index TE also remains harmful.
