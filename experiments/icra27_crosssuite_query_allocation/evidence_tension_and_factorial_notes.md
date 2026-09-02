# Frozen prior-evidence tension and factorial conventions

This note is frozen before Track-A outcomes.

## Cross-suite evidence tension

On the frozen 140-block same-target confirmation cohort:

- `FO20 - Reverse20 = +32.14 pp` is the only clearly stable diagonal component-assignment contrast.
- `FO20 - Fresh = +4.29 pp`; paired and task-cluster confidence intervals both cross zero.
- `FO20 - FullOld20 = +12.14 pp`; the paired interval is positive, but the task-cluster interval crosses zero.

The four-cell table therefore does not establish a unique additive attribution of the 32.14 pp diagonal asymmetry to arm freshness or gripper commitment. The exact simple effects depend on the other component's temporal state, and the table exhibits substantial conditional dependence. The diagonal contrast must not be called “the gripper effect,” “the arm effect,” or decomposed into fixed component percentages.

The uncertain cross-suite `FO20 - Fresh = +4.29 pp` also differs from the exposed Object development result `ARM4_GRIP32 - H4 = +10.56 pp`, which was positive. The interventions differ in source-age profile, policy-query rate, and fixed-source versus periodic schedule semantics. Those are descriptive facts, not pre-authorized explanations.

If Track A fails, the failure will be reported as consistent with the weak existing cross-suite conditional evidence. If Track A succeeds, the discrepancy will be reported explicitly and the differing intervention semantics discussed only as hypotheses, not demonstrated causal explanations. No follow-up experiment will be launched in this session to reconcile it.

Track A tests the distinct prospective hypothesis that, under periodic executable schedules at matched policy-query rate, preserving gripper commitment while refreshing the arm mitigates the uniform high-frequency replanning penalty.

## Factorial provenance and sign convention

- The 126-block Object factorial interaction was a predeclared descriptive analysis. Its original report used `FO20 - Fresh - FullOld20 + Reverse20`, giving `-10/126 = -7.94 pp`.
- The 140-block cross-suite factorial interaction was a post-hoc supporting analysis of frozen outcomes. The frozen fallback paper uses the opposite orientation, `(FullOld20 - Reverse20) - (FO20 - Fresh)`, giving `+22/140 = +15.71 pp`.

These values use opposite algebraic signs. Their provenance and sign conventions must remain explicit and must not be conflated.

## Read-only fallback context

The fallback context is `paper/icra27-final-claim-freeze` at `ec8dc325b1ed6d54053b35d411ea92b13108374a`. Its completed audit established that FO20 and Reverse20 both query once per executed environment step and have query rate exactly 1.0; realized total query counts differ only through outcome-dependent termination. That audit is not rerun here. `CLAIMS.md` and the manuscript remain untouched.
