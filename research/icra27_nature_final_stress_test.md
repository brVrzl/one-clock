# ICRA 2027 direction reset: final Nature/scientific-research stress test

## Workflow record

- **Selected skill:** `academic-researcher`, pass 2.
- **Input:** refined post-literature candidates R01–R10; independent temporal, multi-rate, contact/tactile, dexterous/bimanual, VLA-latency, and failure-driven evidence passes.
- **Question:** Does each candidate state a scientific phenomenon and causal unit materially distinct from the nearest 2024–2026 work?
- **Rule:** `No direct paper located` means the bounded search did not locate it; it does not establish novelty.

## Nearest-work adversarial boundary

1. [AAC (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Liang_Adaptive_Action_Chunking_at_Inference-time_for_Vision-Language-Action_Models_CVPR_2026_paper.pdf), [AutoHorizon](https://hatchetproject.github.io/autohorizon/), [PACE](https://arxiv.org/abs/2606.00537), [HiPolicy](https://hipolicy.github.io/) and [BCP](https://fleetfootwork.github.io/BCP/) make a shared dynamic chunk/horizon/frequency contribution insufficient.
2. [RTC](https://arxiv.org/abs/2506.07339), [REMAC](https://arxiv.org/abs/2601.20130), [FutureRTC](https://arxiv.org/abs/2607.24008), [DEFLECT](https://arxiv.org/abs/2605.19294), [A2C2](https://arxiv.org/abs/2509.23224) and [VLA-Corrector](https://arxiv.org/abs/2607.01804) make generic asynchronous stale-chunk detection, continuation, or cheap whole-action correction insufficient.
3. [PhaForce](https://arxiv.org/abs/2603.08342), [RETAF](https://arxiv.org/abs/2602.10013), [TouchWorld](https://arxiv.org/abs/2607.07287), [T-Rex](https://arxiv.org/abs/2606.17055), [DPA-FTG](https://arxiv.org/abs/2608.03103), [LAG-Fusion](https://arxiv.org/abs/2607.17257) and [FA-RDP](https://arxiv.org/abs/2607.28596) make generic slow vision/semantic plus fast force/tactile correction insufficient.
4. Classical [multi-rate planning/control](https://arxiv.org/abs/2204.00152) and [event-triggered control](https://doi.org/10.1016/j.ins.2018.04.055) make any fixed rate/event-loop architecture insufficient without learned-policy evidence and a distinct empirical law.

## Stress test of the refined pool

| Candidate | Final literature status | What could be a publication-level claim | Non-negotiable falsifier / evidence |
|---|---|---|---|
| R01 Validity-field execution | **Survives, high novelty risk** | Action chunks have a *time × subspace validity field*, not one global horizon; calibrated partial replacement is the causal consequence. | A validity map must predict counterfactual replacement gain beyond a global entropy/horizon, and not reduce to a shared scalar. |
| R02 Recoverability-budgeted partial replanning | **Survives, strongest formulation candidate** | The right execution decision is determined by component-wise recoverability of the remaining commitment, not age/entropy/horizon alone. | Must show recoverability predicts distinct decisions and improves success/query over source-age, phase, entropy and global replan. |
| R03 Subspace-staleness law | **Survives as the phenomenon, not enough as method alone** | Component validity horizons cross by physical phase, explaining the LIBERO success and RoboTwin null. | Factorial source-age × subspace × phase study must reproduce crossings with at least one negative regime. |
| R04 Compute-aware action-refresh allocation | **Survives, viable method layer** | A limited VLA call is a resource that should refresh the highest value invalid subspace. | Equal-query/latency budget must beat global horizon/global correction with preserved coherence. |
| R05 Coherence-constrained partial refresh | **Support module / critical ablation** | Temporal independence has a measurable boundary set by task coupling. | Must find a regime where unconstrained partial refresh fails and constraint restores it, without degenerating to full replan. |
| R06 Cross-modal freshness routing | **Borderline, data/engineering-heavy** | Sensor freshness and action validity form a sparse causal routing structure. | Independent delay/masking must yield a reproducible routing matrix; otherwise it is a scheduler heuristic. |
| R07 Failure-response taxonomy | **Support figure, weak standalone paper** | Different chunk failures have distinct optimal repair interventions. | Categories must yield reliable counterfactual response differences, not annotation labels. |
| R08 Role-conditioned bimanual temporal contract | **Borderline/high evaluation burden** | Synchrony is needed only at role/coupling events. | Needs verified bimanual RoboTwin fidelity and a strong unilateral-delay effect. |
| R09 Hold/refresh action interface | **Method option only** | Explicit commitments/expiry expose an ability a dense vector lacks. | Need evidence that it is not a rebranded action mask/gate; unsuitable as first 72-hour bet. |
| R10 Source-age calibration benchmark | **Supporting artifact, not ICRA main contribution** | Current aggregate success hides causal temporal phenomena. | Valuable if R03 yields a robust cross-system result; otherwise too narrow. |

## Strongest surviving combined direction

**Working formulation:** *Component-wise validity and recoverability for compute-aware partial replanning.*

**Scientific question:** When an action-chunk policy is executing under latency and disturbances, is its value of replanning determined by a single shared horizon, or by phase-dependent validity/recoverability of individual action subspaces?

**Prediction:** Fresh arm plus retained/local gripper or hand commitment will help only in the phases where the corresponding subspace's predicted recovery value is high; naïve fresh/old source rules will fail outside those phases. A scheduler that preserves native temporal aggregation and only replaces an invalid subspace will outperform global refresh/continuation at the same expensive-policy-query budget.

**Why it is distinct if supported:** existing works change a global chunk/horizon, repair a whole action vector, or install fixed slow/fast control layers. This direction makes **the independently measured validity/recoverability of a semantic action subspace** both the experimental object and the execution decision. It does not claim that multi-rate control or asynchronous inference is new.

## Explicit threats that could still defeat it

- A2C2 or VLA-Corrector may effectively perform partial coordinate repair; their papers/code must be checked for action masks/components before submission.
- PhaForce/FA-RDP/T-Rex may expose phase-conditioned heads that invalidate a simplistic subspace framing.
- If simple low-pass filtering, gripper scaling, or mode-commitment constraints explain LIBERO FO, the claimed staleness law fails.
- If a shared phase-aware/global horizon matches performance at the same query budget, partial scheduling is unnecessary.
- If coherence constraints cause broad coupled refresh in contact, the compute advantage may vanish exactly where the real-robot story is strongest.

## Final evidence gate before reviewer handoff

Advance only R01–R05 (as one integrated program) and retain R06–R08 as backups. Do **not** advance DCTA as the central claim, generic contact correction, or heterogeneous global horizons.
