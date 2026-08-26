# SOL HIGH audit 3: feasibility

## Primary

**Validity/recoverability-aware asynchronous partial replanning, developed through a source-age causal gate (I15 → I02 → I06/I07; I08 ablation only).**

It reuses query/chunk buffers, source-age tracking and the asynchronous executor; no new hardware. Start arm–gripper, not RH56 fingers. First evidence in 1–3 days; credible multi-task simulation in 1–2 weeks; JAKA/gripper validation another 1–2 weeks.

**Stage gates:** (1) small arm-age × gripper-age, phase/failure-partitioned grid on two LIBERO tasks; (2) at identical query counts, selective retention/replacement versus whole refresh/native ACT/newest/full-old/hold-smoothing; (3) continue only on stable cross-task interaction and partial-replacement benefit; (4) then test whether recoverability predicts held-out replacement value beyond age/entropy/contact. If not, use the simpler validity-aware partial-replanning formulation.

## Backups

1. Causal source-age calibration plus failure-response taxonomy: fast, credible evidentiary spine but lower standalone ceiling.
2. Plain asynchronous component-wise replanning: narrower fallback with direct/observed validity event, no learned recoverability claim.
3. Contact-selective partial replanning: only after partial replacement works and only if it beats conventional force residuals.

## Decisions

- **Pause RoboTwin DCTA.** Preserve native ACT, age logging, alignment/executor and frozen null. Do not tune/expand/search. Resume for one preregistered transfer only after LIBERO demonstrates reproducible interaction and equal-query selective-replacement benefit.
- **Continue LIBERO baseline narrowly for 2–3 days.** Faithful native ACT aggregation, existing controls, correct whole/global comparator, equal query/execution time, then frozen causal grid. Stop method engineering if FO reduces to a one-task/hold/smoothing effect or loses against native ACT.
- **Reject DCTA as main method and standalone heterogeneous multi-rate control.** Keep DCTA only as an ablation.
- **Pause component-query scheduling.** A joint ACT/pi0.5 query returns all components, so choosing a component does not itself save a full query; a compute claim requires a supported conditional/local interface. Test an oracle first if revisited.
- **Do not start RH56/bimanual/cross-modal/action-representation work** unless the arm–gripper gate succeeds early.
