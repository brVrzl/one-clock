# ICRA 2027 final figure information hierarchy

Freeze date: 2026-09-03 (Asia/Shanghai)

Status: `INFORMATION_HIERARCHY_ONLY`

This artifact specifies scientific content and visual priority. It does not authorize or create figure artwork.

## Global hierarchy rules

- Visually distinguish governance: preregistered confirmatory evidence, frozen exposed-development characterization, and post-hoc controls must not share an undifferentiated visual treatment.
- Keep same-target probes separate from deployable periodic executors. Same-target probes use policy-query rate 1.0 and are diagnostic instruments.
- Use `S(A0 G_d)-S(A_d G0)` consistently for R1A branch asymmetry and state the opposite negative R1B sign convention explicitly where used.
- Do not encode component contribution percentages, causal mechanisms, lag optimality, global executor dominance, or policy-general claims.

## Fig. 1: same-target measurement concept and frozen confirmatory factorial

### Purpose

Establish exactly what the diagnostic measures before presenting success differences.

### Required conceptual content

- A time axis showing source observation/query time `q`, prediction index `k`, and current physical target time `t`.
- The invariant `q+k=t` highlighted as the same-target identity.
- Separate source-age labels for translation/rotation arm dimensions and the gripper dimension.
- Fresh-prefix rule: for `t<d`, fixed-age conditions execute Fresh `A_t[0]` for all dimensions.
- Authoritative 20 Hz evaluator mapping: d=20 = 1.00 s.

### Required condition examples

- `A0G20`: fresh arm from `A_t[0]`; gripper from `A_(t-20)[20]` after the Fresh prefix.
- `A20G0`: arm from `A_(t-20)[20]`; fresh gripper from `A_t[0]` after the Fresh prefix.

### Evidence panel

Show the original frozen 140-block ACT factorial conditions with absolute success counts. Visually emphasize only the preregistered diagonal `A0G20-A20G0=+32.14 pp`, with paired CI `[+23.6,+40.7]` and task-cluster CI `[+21.4,+44.3]`. Do not render the diagonal as an additive arm or gripper contribution.

### Mandatory scope note

The panel or caption concept must say that same-target probes make one policy query per executed step (`policy-query rate=1.0`), are diagnostic rather than deployment executors, and do not show that a qrate=1 executor is practically preferable.

## Fig. 2: temporal scale and within-arm component identity

Use two panels with matched visual priority but explicit `EXPOSED_DEVELOPMENT_CHARACTERIZATION` labeling.

### Left: R1A complete frozen lag curves

Show both absolute branch curves, not only their difference:

- stale arm/fresh gripper: `S(A_d G0)`;
- fresh arm/stale gripper: `S(A0 G_d)`.

Required x values: d=2, 4, 8, 12, 16, 20, 32, mapped to 0.10, 0.20, 0.40, 0.60, 0.80, 1.00, 1.60 s. A Fresh/d=0 anchor may be shown at 44.44% with a visual note that it is audited historical reuse rather than new R1A data.

Required numerical structure:

| Time | `S(A_d G0)` | `S(A0 G_d)` | Separation |
|---:|---:|---:|---:|
| 0.10 s | 38.89% | 60.32% | +21.43 pp |
| 0.20 s | 38.89% | 65.08% | +26.19 pp |
| 0.40 s | 28.57% | 69.84% | +41.27 pp |
| 0.60 s | 16.67% | 69.84% | +53.17 pp |
| 0.80 s | 18.25% | 71.43% | +53.17 pp |
| 1.00 s | 9.52% | 64.29% | +54.76 pp |
| 1.60 s | 25.40% | 62.70% | +37.30 pp |

Mark d=20 as the originally frozen confirmation lag, not as an optimized point. If highlighting that d=20 has the largest observed R1A separation, use descriptive annotation only; do not use a peak symbol, superiority bracket, or lag-versus-lag significance cue. Caption concept: “d=20 attained the largest observed separation on the frozen grid; no preregistered lag-versus-lag inference establishes a statistically distinct peak.”

Make the long-lag decomposition legible without causal arrows: from d=20 to d=32, stale-arm success rises 15.87 pp and stale-gripper success falls 1.59 pp, narrowing separation from +54.76 to +37.30 pp.

### Right: R1B translation versus rotation

Show absolute success for:

- translation-stale `T20_R0_G0`: 11/126 (8.73%);
- rotation-stale `T0_R20_G0`: 53/126 (42.06%);
- Fresh reference: 56/126 (44.44%).

Primary annotation must state the sign: `S(translation-stale)-S(rotation-stale)=-33.33 pp`. Show the paired CI `[-42.06,-24.60]` and task-cluster CI `[-44.44,-22.22]`; both lie strictly below zero.

If B1 is referenced, keep it as a separate diagnostic-source annotation rather than placing dispersion and behavioral success on a shared numerical axis. Make the ordering contrast explicit:

- B1 same-target normalized dispersion: rotation > translation > gripper;
- R1B behavioral temporal sensitivity within arm: translation > rotation.

Label the B1 use narrowly as `CONTRADICTED_AS_AN_ORDERING_PREDICTOR_WITHIN_ARM`. Do not imply that the dispersion measurement itself is invalid.

### Figure-level purpose

The two panels jointly show temporal scale structure and within-arm component identity structure, while preserving their exposed-development governance.

## Fig. 3: compact mechanism accounting

### Inclusion decision

Include only as a compact accounting figure/table if the paper needs a dedicated visual boundary around mechanism claims. Information density is sufficient if the B1/R1B ordering inversion is the focal item and the remaining diagnostics are secondary rows. If space is constrained, move this accounting to the supplement rather than compressing it into causal-looking diagrams.

### Preferred hierarchy

1. Focal comparison: B1 dispersion ordering `rotation > translation > gripper` versus R1B behavioral sensitivity `translation > rotation`.
2. Secondary row: B2 persistence, `P(no gripper transition at 1.00 s)=0.675018`, labeled insufficient as a complete explanation.
3. Secondary row: occupancy moderator, n=30, rho=0.1922, p=0.3089, labeled `NULL / UNSUPPORTED`.
4. Secondary row: B3, labeled `NO_FROZEN_CRITERION`; complete curves are descriptively non-corresponding/mixed with behavioral ordering.
5. Terminal statement: mechanism `UNRESOLVED`.

Do not give command discontinuity a main panel. If mentioned in a note, label it `NON_IDENTIFYING_POST_HOC_CHARACTERIZATION`. Do not use equivalent glyph weight for confirmatory behavioral evidence and post-hoc or training-demonstration diagnostics.

## Fig. 4: Track-A execution consequence

### Main panel

Plot pooled success against `log(policy queries per executed environment step)` for exactly:

- H16;
- H4;
- ARM4_GRIP32;
- H2;
- ARM2_GRIP16;
- TE_DENSE.

Rules:

- No regression.
- No line connecting all conditions.
- Give TE_DENSE a distinct marker because its aggregation semantics differ from periodic chunk commitment.
- Add explicit paired annotations only for `H4 -> ARM4_GRIP32` (+4.667 pp) and `H2 -> ARM2_GRIP16` (+5.778 pp).
- Describe these pairs as nearly matched in policy queries per executed environment step, not identical compute.
- Keep coherent H16 visually identifiable as the strongest overall frozen operating point, 357/450 (79.33%).

### Inset: suite-level concentration

Show both component-resolved contrasts by suite:

| Contrast | LIBERO-10 | Goal | Spatial |
|---|---:|---:|---:|
| `ARM4_GRIP32-H4` | +13.333 pp | +0.000 pp | +0.667 pp |
| `ARM2_GRIP16-H2` | +14.000 pp | +2.000 pp | +1.333 pp |

Include H16 baseline context: LIBERO-10 54.7%, Goal 91.3%, Spatial 92.0%. The inset must make LIBERO-10 concentration visible. Its caption concept must state that suite identity, baseline difficulty/ceiling, task semantics, and gain covary, so the source of heterogeneity is not identifiable.

### Figure-level purpose

Show that component-resolved temporal allocation has an operational consequence under nearly matched replanning cadence without claiming global method dominance. State that if cadence is freely selectable in this static LIBERO benchmark, coherent H16 remains preferable.

## Supplement-facing placement

- Full R1A discordances, exact p-values, paired intervals, task-cluster intervals, and per-task effects belong in tables rather than the main curve panel.
- Full R1C factorial, including the canary's exact identity definition, belongs in the supplement or a compact supporting panel. Its narrow role is query-schedule identification under the deterministic ACT evaluator.
- R1D belongs in a visible scope/generalization table: Spatial `A0G20-A20G0=+28.00 pp` preserves the original sign with a 4.14 pp smaller point estimate. It must remain labeled `POST_HOC_SPATIAL_FACTORIAL_COMPLETION` and separate from the original 140 blocks.
- Complete B3 k=0..32 curves and uncertainty belong in the supplement; any main-text mechanism accounting should summarize only the frozen descriptive mismatch and `NO_FROZEN_CRITERION` status.
- TE_DENSE characterization belongs with execution evidence but must retain its implementation-specific scope and must not be illustrated as a bug or chatter mechanism.
