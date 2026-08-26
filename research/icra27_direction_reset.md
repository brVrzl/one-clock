# ICRA 2027 direction reset: research recommendation

## Decision

**Primary direction:** *Conditional action-subspace commitment: measuring component-wise validity/recoverability and using it for coherence-aware partial replanning.*

**One-sentence paper claim (conditional on the gates below):** In action-chunk manipulation, the value of replacing a committed action is phase- and subspace-dependent; measuring that counterfactual validity enables selective retention/replacement that outperforms synchronous whole-chunk choices at matched policy-query budget.

This deliberately does **not** claim that every robot needs separate clocks, that stale actions are universally better, or that a slow semantic planner plus fast contact controller is new.

## Top 3

### 1. Conditional action-subspace commitment / validity-recoverability-aware partial replanning

**Scientific question.** Is action age a shared global property, or does the counterfactual value of retaining versus replacing an arm, gripper, wrist, or hand commitment depend on phase, subspace and coupling?

**Observed motivation.** Frozen LIBERO shows a large off-diagonal intervention: fresh arm plus ~1-second-old gripper is 80/126 (63.5%), versus newest 53/126 (42.1%), full-old 55/126 (43.7%), age-exp 62/126 and CogACT-like 59/126. Frozen RoboTwin is a no-signal counterexample and native ACT is stronger (19% vs FO 11%). This rules out a universal old-gripper rule.

**Assumption challenged.** One globally synchronous action-chunk replan/aggregation boundary is the right unit of control.

**Method, only after causal gate.** Preserve faithful native aggregation/commitment. Estimate validity or recoverability for a candidate action subspace; retain it when it remains valuable, replace it when it is invalid, and impose only demonstrated coherence constraints. The first prototype should use directly observed/oracle validity labels, not a learned scheduler.

**Why not ACT / RTC / AAC / AutoHorizon / HiPolicy.** ACT ensembles a shared vector. RTC/REMAC/FutureRTC preserve/repair temporal chunks globally. AAC/AutoHorizon/PACE/BCP choose shared horizon/continue-replan decisions. HiPolicy offers global frequency branches. This paper's causal unit is an independently tested semantic subspace, with an equal-query comparison, not a temporal prefix or global clock.

**Nearest five papers/threats.** [RTC](https://arxiv.org/abs/2506.07339), [REMAC](https://arxiv.org/abs/2601.20130), [AAC](https://openaccess.thecvf.com/content/CVPR2026/papers/Liang_Adaptive_Action_Chunking_at_Inference-time_for_Vision-Language-Action_Models_CVPR_2026_paper.pdf), [AutoHorizon](https://hatchetproject.github.io/autohorizon/), [VLA-Corrector](https://arxiv.org/abs/2607.01804). Secondary contact threats: PhaForce/RETAF/TouchWorld/T-Rex.

**Killer figure.** A phase-faceted arm-age × gripper-age intervention surface: color is the causal benefit of continue, component replacement, or whole replacement at the same policy-query count. Overlay a held-out validity/recoverability ordering. Include a null/shared-clock facet.

**Killer experiment.** On frozen policies, factorially replace arm and gripper sources at several physical ages while retaining native aggregation as its own baseline. Compare whole fresh, whole old, native ACT, global replan, selective replacement and hold/smoothing under matched query/execution timing. Test whether phase/contact or recoverability predicts the winning intervention.

**1–3 day go/no-go.** Day 1 grid and phase stratification; Day 2 fixed selective-retention executor comparison; Day 3 held-out LIBERO task plus one low-risk JAKA/gripper contact demonstration only if Days 1–2 pass.

**Kills it.** No reproducible time × subspace interaction on held-out tasks; global/native aggregation matches selective replacement at equal query count; interaction is explained by gripper scaling, filtering, action-mode commitment or baseline mismatch; or mixed replacement violates coherence in the useful regimes.

**Publication-scale evaluation.** Two policy families if feasible (ACT first; a chunked diffusion/VLA executor only after signal), LIBERO plus a verified-fidelity RoboTwin transfer, controlled visual/contact perturbations, equal policy-call and wall-clock budgets, component/phase negative cases, and one small real JAKA task. RH56/bimanual only if it offers a confirmed distinct regime.

**Real-robot story.** A visually legible grasp/insert/press task: global arm intent remains committed while a demonstrated invalid local gripper/contact component is selectively corrected; include an explicit case where global replan is preferable.

**Reuse.** ACT, historical query/chunk buffers, physical-time age tracking, aggregation, component grouping, async executor, LIBERO protocol and JAKA/gripper stack.

**Stop if it wins.** Stop DCTA-as-paper, global-horizon variants and generic fast tactile residual work; retain DCTA as an ablation.

### 2. Causal source-age calibration and failure-response taxonomy

**One-sentence claim.** Standard stale-action tests obscure whether a chunk failure needs continuation, local repair, partial replacement or whole replanning; a causal source-age × subspace × phase protocol exposes that distinction.

**Question / phenomenon.** Why does one strong LIBERO off-diagonal intervention coexist with a RoboTwin null? Candidate mechanisms include phase dependence, ensemble/mode commitment, predictability, recoverability, control-rate/scaling and coupling.

**Method.** A preregistered causal diagnostic matrix, not a learned controller: factorial source age, group, phase/event and response intervention. Dense metrics complement success.

**Why distinct.** AAC/AutoHorizon/RTC use global execution strategies; they do not provide a subspace-level causal calibration protocol. It is less novel than Top 1 but a strong evidence backbone.

**Nearest five papers.** [ACT](https://www.roboticsproceedings.org/rss19/p016.pdf), [AAC](https://openaccess.thecvf.com/content/CVPR2026/papers/Liang_Adaptive_Action_Chunking_at_Inference-time_for_Vision-Language-Action_Models_CVPR_2026_paper.pdf), [PACE](https://arxiv.org/abs/2606.00537), [BCP](https://fleetfootwork.github.io/BCP/), [World Action Models in Real Time](https://arxiv.org/abs/2608.01880).

**Killer figure / experiment.** The same response surface plus a failure-to-remedy matrix. Kill if effects disappear after calibration, are one-task-only, or no measured regime variable explains reversals.

**Evaluation / real story / reuse.** Low-risk, immediately reuses the frozen LIBERO protocol and existing logs. It is a fallback paper only if it becomes cross-policy/cross-embodiment and yields a reusable benchmark; otherwise it supports Top 1.

**Stop if it wins.** Do not build a scheduler until the diagnostic identifies a robust response law.

### 3. Plain asynchronous component-wise replanning, contact as a bounded regime

**One-sentence claim.** At contact transitions, a local action subspace can become invalid while global intent remains useful; selective action-chunk replacement can beat global refresh only when this is demonstrated at equal compute.

**Question / phenomenon.** Is contact a localized validity boundary, or do existing force residuals/global replans already solve it?

**Method.** An observed contact/force event triggers a fixed selective replacement policy; no learned timing claim initially.

**Why not prior slow/fast work.** It must show selective replacement of a stale chunk subspace while retaining another commitment, rather than a fixed-rate tactile residual. If it cannot, it is occupied.

**Nearest five papers.** [PhaForce](https://arxiv.org/abs/2603.08342), [RETAF](https://arxiv.org/abs/2602.10013), [TouchWorld](https://arxiv.org/abs/2607.07287), [T-Rex](https://arxiv.org/abs/2606.17055), [FA-RDP](https://arxiv.org/abs/2607.28596).

**Killer figure / experiment.** Contact-aligned recovery curves for continue, force residual, selective replacement and global refresh at equal policy calls. Kill if conventional residual control explains all benefit or local invalidity is not observed.

**Evaluation / real story / reuse.** JAKA gripper/force first; RH56 only after a positive arm–gripper causal gate. Reuses force/contact signals and executor. This is a backup, not a main claim.

## Mandatory candidate assessment

| Candidate | Disposition |
|---|---|
| A. DCTA | **3: primarily an ablation/baseline** and useful diagnostic readout. Not sufficient as a full ICRA contribution. |
| B. Async component-wise replanning | Necessary execution mechanism within Top 1; not enough without the validity/recoverability phenomenon. |
| C. Contact/force-triggered selective replanning | Backup/ablation only; generic framing is occupied. |
| D. Heterogeneous multi-rate action policy | Not worth pursuing standalone; global and fixed slow/fast versions are occupied. |
| E. Component-wise scheduling | Pause as main direction. A joint ACT/pi0.5 call emits all components, so component choice alone does not save full-model compute. Revisit only after an oracle proves value and a real local/conditional interface exists. |
| F. Stronger formulation | Top 1: conditional validity/recoverability plus coherence-aware partial replanning. |

## Best-paper counterfactual

If FO/DCTA had never existed, the paper chosen today would still be **a causal study of conditional temporal commitment in action-chunk policies**, followed by minimal partial replanning. The unusual LIBERO effect is a lead, not the method. RoboTwin's null becomes an essential boundary condition rather than an inconvenience.

## 72-hour go/no-go plan

**Day 1 — establish a causal surface (go if a reproducible interaction appears).** On two preselected LIBERO tasks/policy seeds, run a small factorial arm-age × gripper-age grid at matched physical time, stratified by independently defined phase/event labels. Include native ACT, newest, full-old, existing age-exp/CogACT, hold/smoothing and a whole-output refresh comparator. Output: interaction estimates and dense source-age/failure traces. **No-go:** effect disappears against native ACT, is entirely filtering/scaling/mode-commitment, or is not stable across the two tasks. Do not tune ages after seeing results.

**Day 2 — minimal execution consequence (go if selective replacement beats global choices).** Use the winning predeclared Day-1 rule as a fixed selector, preserving native aggregation where applicable. Compare selective retain/replace with global refresh, continue/native, newest and hold at identical full-policy-call count and execution timing. Output: paired successes plus component coherence/discontinuity traces. **No-go:** global/native matches it, coherence failures dominate, or gain is within uncertainty.

**Day 3 — boundary validation (go if it transfers).** Freeze rule and run a held-out LIBERO task. If and only if Days 1–2 pass, run a small, safety-preserving JAKA gripper contact task with existing limits/watchdogs unchanged; otherwise use a second simulator task. **No-go:** no held-out transfer, or real contact behavior is fully explained by a conventional fixed force residual. Do not start RH56/bimanual work.

## Running work now

- **CONTINUE:** LIBERO baseline fidelity and the narrow Day-1 causal diagnostic. Preserve frozen controls and all source-age logging.
- **PAUSE:** RoboTwin DCTA development/rollouts. Preserve the faithful native ACT path, null result, schedule and async infrastructure. Resume only after the two LIBERO gates pass, for one preregistered transfer test.
- **STOP:** DCTA-specific architecture search/tuning, new DCTA seed expansion, post-hoc RoboTwin task/age search, standalone multi-rate/global-horizon variants, and generic contact-residual implementation.

## Nearest-work threat

The highest risk is that a close masked-correction method already permits semantic action masks, or that the effect reduces to global phase-aware execution/force residual control. Before paper commitment, inspect A2C2, REMAC and VLA-Corrector implementation-level action masking. The scientific claim survives only if the measured subspace intervention effect remains after those controls.

## Skills used

- **Exact Nature/scientific skill:** `academic-researcher`.
- **Exact Brainstorm skill:** `scientific-brainstorming`.
- **Nature passes completed:** 2 (initial literature/novelty audit; refined-pool final stress test).
- **Brainstorm passes completed:** 2 (raw independent divergence; post-literature constrained divergence).
- **Output artifacts:** `icra27_brainstorm_raw.md`, `icra27_nature_audit.md`, `icra27_brainstorm_refined.md`, `icra27_nature_final_stress_test.md`, landscape, idea pool, three Sol audits and this report.
- **Partially applicable/unavailable:** no required skill unavailable. `academic-researcher` is the installed scientific-research/literature-gap skill selected for the requested Nature role; no separately named “Nature research” skill exists (the installed `nature-figure` skill is figure-specific and not applicable).

## Models and reviewer roles

- **Luna / `gpt-5.6-luna`:** three independent agents completed all eight requested bounded roles sequentially: temporal chunking; multi-rate control; contact/tactile; dexterous/bimanual; VLA latency; two broad ideation passes; failure-driven ideation.
- **Sol High / `gpt-5.6-sol`, high reasoning:** independent novelty, scientific-significance, and feasibility audits.
- **Terra role:** integration and report maintenance performed by the primary orchestrator. A separate Terra subagent was not routed because integration was the main coordinating task; the harness did expose explicit Luna and Sol model routes but not a different main-agent route.
- **Sol Max:** **not used.** The two audits and feasibility pass clearly separate the primary evidence-first direction from DCTA and the backups; a Max tie-break would not change the next 72 hours.

## Human decision requested

Authorize only the Day-1 diagnostic gate. Do not implement the proposed method or start new scientific GPU experiments until its outcome is reviewed.
