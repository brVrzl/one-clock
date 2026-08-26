# ICRA 2027 direction reset: raw divergence (Brainstorm pass 1)

## Session register

- **Skill / pass:** `scientific-brainstorming`, pass 1 (independent generation before literature review)
- **Focal question:** Which falsifiable robot-learning phenomenon and minimal method could make the strongest executable ICRA 2027 paper using the stated assets?
- **Decision owner / horizon:** research lead; choose a 72-hour go/no-go direction, not a final method.
- **In scope:** execution, architecture, action representation, asynchronous and component-wise control, contact/force/tactile feedback, VLA computation latency, sensor scheduling, recovery, dexterous and bimanual manipulation.
- **Out of scope:** implementation, new GPU experiments, manuscript changes, weakening any robot safety interlock.
- **Known observations:** LIBERO hard FO strongly helps (63.5% vs 42.1% newest); RoboTwin hard FO is no signal and native ACT aggregation matters; slow VLA inference is operationally relevant.
- **Constraints:** JAKA arm, parallel gripper, RH56 hand, wrist/workspace vision, force/contact signals, ACT/pi0.5, LIBERO/RoboTwin, historical action buffers and asynchronous execution exist. Real evidence must be cheap and safety-preserving.
- **Provenance:** AI-assisted independent ideation, generated before this reset's literature search. These are proposals, not findings.

## Independent idea register

| ID | Idea / one-sentence question | Assumption | Prediction | Disconfirming evidence | Minimal killer experiment |
|---|---|---|---|---|---|
| B01 | **Action-subspace staleness:** Do arm, gripper, and hand action subspaces lose usefulness at different physical ages? | Staleness is measurable separately from accuracy. | Per-subspace age-performance curves cross, and their ordering changes by contact phase. | One shared age curve explains every component/task. | Replay matched ACT queries; replace one component at a controlled source age while holding others fixed. |
| B02 | **Event-triggered selective repair:** Can a slow chunked semantic policy stay reliable by refreshing only the subspace whose contact/state event invalidates it? | A low-cost event detector has useful precision. | Selective repair matches full re-query reliability with fewer full policy calls. | Event detector fires indiscriminately or only full re-query helps. | Impose controlled contact/target perturbations; compare full replan, no replan, and matched-budget selective repair. |
| B03 | **Contact-time action ownership:** Is action ownership supposed to switch from vision-plan to force/tactile controller at contact transitions, rather than be continuously blended? | Contact transitions can be identified safely. | A hand correction expert helps only around contact onset/slip/release and harms outside it. | Constant hand correction is as good or better. | Contact-phase intervention with the same arm plan and gated vs ungated hand correction. |
| B04 | **Compute-budgeted partial replanning:** Under slow VLA inference, which action coordinates should be recomputed when compute is scarce? | Coordinate-level refresh is feasible from existing buffers. | Equal success at lower VLA-call rate than global refresh/chunking. | Any partial refresh loses the benefit of full refresh. | Simulated inference latency/call budget; compare full, fixed chunk, and component refresh schedules. |
| B05 | **Sensor-to-actuator asynchronous graph:** Are different sensors informative at different rates for different action components? | Sensor freshness is separable from control freshness. | Wrist/contact updates improve hand/gripper corrections while workspace vision mainly improves arm replan. | Uniform sensor fusion is never worse. | Artificially delay/mask one sensor stream and selectively refresh compatible action groups. |
| B06 | **Commitment versus correction action representation:** Would representing a chunk as slow task-space commitments plus fast residual contact actions remove the need for a shared action clock? | Demonstrations/support can yield both representations. | Residuals are sparse, phase-local, and generalize to disturbances. | The decomposition is arbitrary or residuals dominate everywhere. | Fit offline decomposition; test disturbance recovery using fixed commitments plus residual loop. |
| B07 | **Temporal coordination as a learned communication problem:** In bimanual work, should each arm refresh locally but synchronize only at interaction events? | Synchrony is required only at coupling events. | Event-synchronous local policies retain coordination with less recomputation. | Continuous shared clock is required even away from coupling. | RoboTwin bimanual handoff/assembly with induced delay on one side. |
| B08 | **Action-chunk confidence calibration by subspace:** Can disagreement among historical action queries predict when a coordinate will become stale? | Ensemble disagreement carries calibration signal. | Subspace disagreement predicts intervention benefit better than global uncertainty. | It fails to predict benefit/only tracks task difficulty. | Offline prediction of counterfactual replacement gain from historical query disagreement. |
| B09 | **Failure taxonomy of temporal methods:** Are chunked-policy failures governed by phase errors (semantic), contact errors (reactive), or execution-latency errors, each requiring a distinct response? | Failure sources can be annotated/operationalized. | Interventions have a phase-specific causal pattern across tasks. | A single intervention fixes every class equally. | Small curated disturbance suite with per-failure causal interventions. |
| B10 | **State-dependent action horizon per subspace:** Should arm and end-effector/finger horizons contract at different task phases? | Horizon, rather than aggregation weights, is the relevant causal variable. | Adaptive horizon helps under phase transitions; static component weights do not. | A global adaptive horizon explains gains. | Train/infer matched actuator horizon schedules with equal policy calls. |
| B11 | **Contact-triggered safety recovery:** Can force/contact signals detect an invalidated action chunk early enough to execute a safe local recovery before a semantic replanning call returns? | A conservative trigger is available. | Fewer contact failures with no increase in unsafe commands and bounded full re-plans. | Signal is too late/noisy or recovery worsens outcomes. | Real/sim contact perturbation, hardware safety limits intact; compare stop/hold/full replan/selective recovery. |
| B12 | **Cross-embodiment temporal morphology:** Does the optimal temporal granularity follow actuator/contact morphology (gripper vs dexterous hand), not model family? | Same task phases can be compared across embodiments. | Fingered hand shows shorter reactive timescale and stronger advantage for local refresh. | Differences vanish after normalizing control rate. | Shared pick/insert/contact tasks with gripper and RH56 under matched policy timing. |
| B13 | **Predictive action repair from source age:** Can an action be repaired locally using a model of age-induced error, avoiding full VLA inference? | Age-induced error has learnable structure. | Local repair closes part of full-refresh gap on controlled perturbations. | Error is too task-specific/unpredictable. | Offline source-age-to-error predictor, then replay/rollout repair test. |
| B14 | **Asynchronous affordance execution:** Can a VLA issue long-lived semantic affordance commitments while action-level controllers execute and correct independently? | Affordance representation can be exposed from pi0.5/ACT-compatible interface. | Stronger latency robustness and generalization than raw action chunks. | It is just a conventional hierarchy without measurable timing phenomenon. | Delayed-inference simulation and a visual real task with contact correction. |
| B15 | **Recoverability as the criterion for refresh:** Rather than update when stale, should a policy update only when the remaining chunk cannot recover from a predicted deviation? | A recoverability proxy is attainable. | It beats age/uncertainty thresholds at the same compute budget. | Proxy cannot separate recoverable from unrecoverable deviations. | Rollout suffix counterfactuals to label recoverability; threshold policy comparison. |

## Pre-evidence structural clusters (no ranking)

1. **Phenomenon-first temporal diagnostics:** B01, B08, B09, B12.
2. **Selective re-execution / compute:** B02, B04, B10, B13, B15.
3. **Contact ownership and local correction:** B03, B06, B11.
4. **Sensor and coordination topology:** B05, B07, B14.

## Explicit dissent / alternative framing

**Idea:** The LIBERO effect is task/distribution-specific behavior of ACT temporal ensembling, not evidence for asynchronous control. **Prediction:** no component-specific pattern survives matched source-age, trajectory phase, and task analysis; DCTA/repair gains will not transfer. **Implication if supported:** stop pursuing component-wise temporal control and focus on a reliable standard ACT baseline or a distinct contact/manipulation question.

## Evidence status

All entries: `not-checked`. Literature search and adversarial review follow in the Nature/scientific-research audit; no novelty claim is made here.
