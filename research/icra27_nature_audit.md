# ICRA 2027 direction reset: Nature/scientific-research audit, pass 1

## Workflow record

- **Selected skill:** `academic-researcher` (the installed scientific-research/literature-gap skill).
- **Question:** Which raw brainstorm proposals identify a literature-supported, falsifiable gap rather than a renamed temporal-weighting method?
- **Search date:** 2026-08-26. Sources: primary paper/project pages where located; searches cover 2024-2026 and selected foundations.
- **Method:** examine research question/significance, method and unit of adaptation, results/setting, limitations, and nearest overlap for each candidate cluster. This is an evidence synthesis, not proof that no other work exists.
- **Evidence inputs:** independent temporal action-chunking, multi-rate control, and tactile/force landscape passes; direct source links below.

## Located evidence: the occupied landscape

| Theme | Located evidence | What it establishes | Consequence for this reset |
|---|---|---|---|
| ACT temporal ensemble | [ACT, RSS 2023](https://www.roboticsproceedings.org/rss19/p016.pdf) | Fixed action chunks; optional *global* temporal ensemble combines overlapping whole action predictions. | A component-wise aggregation extension is an incremental ACT variant unless it supports a phenomenon beyond smoothing. |
| Global adaptive horizon | [AAC, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Liang_Adaptive_Action_Chunking_at_Inference-time_for_Vision-Language-Action_Models_CVPR_2026_paper.pdf), [AutoHorizon, ECCV 2026](https://hatchetproject.github.io/autohorizon/), [BCP, 2026](https://fleetfootwork.github.io/BCP/), [PACE, 2026](https://arxiv.org/abs/2606.00537) | Adapt global chunk/prefix/replan timing from entropy, attention, learned continuation, or motion phase. | Reject any claim whose method merely chooses a shared dynamic horizon/frequency. |
| Hierarchical multi-frequency chunks | [HiPolicy, 2026](https://hipolicy.github.io/), [Mixture of Horizons, 2025](https://arxiv.org/abs/2511.19433) | Coarse and fine chunk branches, global frequency selection / horizon mixing. | A multi-rate paper needs independent action-subspace clocks, not just high/low global branches. |
| Global asynchronous chunks | [RTC, NeurIPS 2025](https://arxiv.org/abs/2506.07339), [REMAC, ICLR 2026](https://remac-async.github.io/), [FutureRTC, 2026](https://arxiv.org/abs/2607.24008), [DEFLECT, 2026](https://arxiv.org/abs/2605.19294) | Whole chunks can execute during inference; work repairs/anticipates stale global context and intra-chunk mismatch. | "Async VLA execution" is not novel. Need partial refresh as the causal unit and a compute-adjusted comparison. |
| Contact-aware force/tactile correction | [FoAR, 2024](https://arxiv.org/abs/2411.15753), [RETAF, 2026](https://arxiv.org/abs/2602.10013), [PhaForce, 2026](https://arxiv.org/abs/2603.08342), [M2-ResiPolicy, 2026](https://arxiv.org/abs/2603.15152), [TouchWorld, 2026](https://arxiv.org/abs/2607.07287), [T-Rex, 2026](https://arxiv.org/abs/2606.17055) | Slow semantic or chunk policy plus fast force/tactile residual/reaction, sometimes phase/contact gated. | Reject a generic slow arm + fast tactile hand residual as already occupied. |
| Classical multi-rate/event control | [Unified Multi-Rate Control](https://arxiv.org/abs/2012.06558), [multi-rate planning/control](https://arxiv.org/abs/2204.00152), [event-triggered-control survey](https://doi.org/10.1016/j.ins.2018.04.055) | Fixed/nested planner-tracker rates and event-triggered updates are mature control concepts. | Fixed fast/slow actuator loops are a straightforward application, not a paper claim. |
| Bimanual action structure | [InterACT, CoRL 2025](https://proceedings.mlr.press/v270/lee25a.html) | Separates bimanual arm predictions, but does not establish independently asynchronous per-subspace action-chunk refresh. | Useful contrast, not yet evidence of a gap by itself. |

## Candidate stress test

| Raw candidates | Status after audit | Evidence-based reason | Required decisive evidence if retained |
|---|---|---|---|
| DCTA / dynamic component-wise temporal aggregation | **Borderline, demote** | It differs mechanically from global horizon papers, but reviewers can plausibly call it a gating/ensemble variant. LIBERO supports only one hard source-selection diagnostic; RoboTwin hard FO is null. | Demonstrate measured source-age divergence and that dynamic subspace weights are necessary *within* a broader partial-refresh mechanism, at equal query budget. |
| Global adaptive execution horizon / heterogeneous multi-rate policy | **Occupied** | AAC, AutoHorizon, BCP, PACE, HiPolicy and MoH cover global adaptive horizon/frequency. | Only retain if recast around action-subspace validity rather than a global frequency decision. |
| Generic async component replan | **Borderline** | RTC/REMAC/FutureRTC/DEFLECT cover whole-chunk async continuation, stale context and corrective action chunks. | Show full replan is not the right causal action: one component becomes invalid while another remains useful, and partial replan wins at equal compute. |
| Generic contact/force-triggered local policy | **Occupied** | RETAF/PhaForce/TouchWorld/T-Rex and related work already use slow plans with fast contact corrections. | Only retain if the novel variable is learned, component-selective scheduling across non-contact and contact cases, not an added residual. |
| Sensor-rate scheduling | **Open but risky** | Event sensor/control clocks are known classically; direct action-chunked IL evidence was not located in this bounded pass. | Need a clean sensor-to-action causal matrix and more literature verification; high engineering/data risk. |
| Recoverability-triggered refresh | **Potentially open** | Located works use age, entropy, phase/speed, or future-state correction, but no direct source was located for choosing a *partial* refresh from suffix recoverability. | Establish that a recoverability proxy predicts whether re-query versus continued execution matters, beyond action uncertainty. |
| Action-subspace staleness + selective asynchronous replanning | **Survives provisionally** | No located work directly combines semantic subspaces, physical-time source-age/validity, event-triggered partial refresh, and query-normalized evidence. It is nevertheless adjacent to every row above. | Crossed component-specific age/benefit curves; selective refresh beats full/global choices at matched policy calls; at least one clean negative regime. |
| Failure taxonomy / causal temporal diagnostic | **Survives as enabling scientific figure, not necessarily standalone method** | Existing methods rarely organize action-chunk failures by which action subspace lost validity. | A concise causal intervention matrix that predicts which remedy works and where it fails. |
| Contact-time action ownership | **Likely occupied if architecture-first** | Strong overlap with PhaForce, RETAF, TouchWorld and T-Rex. | Keep only as a phase-conditioned trigger/ablation inside the subspace-staleness paper. |
| Compute-budgeted partial replanning | **Survives provisionally** | Existing async VLA work tackles latency globally; selective compute allocation among action subspaces appears less direct. | Measure success versus full-policy-query budget/latency and include global async baselines. |

## Scientific interpretation of the existing observations

**Located evidence + project observation, not a conclusion:** the LIBERO result is compatible with heterogeneous action-subspace validity: a roughly one-second-old gripper prediction combined with a fresh arm beat both newest and full-old alternatives (63.5%, 80/126, versus 42.1% and 43.7%). The RoboTwin no-signal result and the importance of native aggregation rule out the simpler conclusion that an old gripper source is generally better or that aggregation should be removed.

The most useful strengthened hypothesis is therefore not "each group should have a different temporal weighting." It is:

> **Heterogeneous action components have state- and phase-dependent validity horizons; a globally synchronous replan either discards still-valid commitments or continues invalid local commands. Selectively refreshing only the invalid subspace can improve reliability per policy query while preserving native aggregation.**

This remains a proposal. The current evidence establishes a component-asymmetry *diagnostic in one setting*, not this universal mechanism.

## What would make the claim convincing

1. A **phenomenon plot**, not a method-only bar chart: controlled source-age replacement gain for arm/wrist/gripper/fingers across at least approach, contact, and release (including intersections/nulls).
2. A **causal execution comparison at fixed model-query / latency budget**: native ACT/global async replan/global adaptive horizon/partial refresh, while preserving native aggregation where it remains useful.
3. A **negative result**: tasks/regimes where a shared clock is sufficient, selective refresh is neutral/harmful, or the trigger is wrong. This distinguishes the claim from an unconditional design preference.
4. Cross-regime evidence: LIBERO diagnostic plus stable RoboTwin task(s) and one visually compelling force/contact real-robot task. Do not use unreconciled low RoboTwin baselines as decisive evidence.

## Search limits and unresolved threats

- The search is bounded and cannot establish that no component-selective work exists. The final stress test must specifically check partial action masks/coordinate-level action repair, any 2026 preprints, and dexterous/bimanual work.
- Recent preprints create a high novelty-risk environment. T-Rex, TouchWorld, PhaForce, AAC, HiPolicy, PACE, REMAC, FutureRTC and DEFLECT are the closest known threats.
- Any claim of a learned *contact residual* or generic *variable rate policy* should be abandoned. The retained direction must prove selective, action-subspace-specific refresh and compute-normalized benefit.

## Decision for second brainstorm

Reopen divergence only around: (i) a measurable subspace-validity law, (ii) event/uncertainty/age/recoverability triggers for **partial** refresh, and (iii) an evaluation design that can falsify the law. Do not polish DCTA or generic contact hierarchies.
