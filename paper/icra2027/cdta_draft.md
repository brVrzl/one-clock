# Component-Decoupled Temporal Aggregation for Action-Chunked Robot Policies

> **Draft status: development gate failed.** The mechanism analysis remains supported, but CDTA-16 achieved only one paired net win over its matched shared control and did not meet the frozen advance gate. The held-out panel was therefore not launched. This file is retained as a research draft, not a submission-ready method claim.

## One-sentence contribution

We identify and measure cross-component interference in same-target temporal aggregation, then introduce CDTA-16, a training-free rule that assigns independent temporal weights to continuous arm motion and near-discrete gripper intent.

## Abstract

Action-chunked robot policies produce multiple predictions for the same physical target time, so deployment must choose or aggregate predictions made from different observations. Existing temporal ensembles commonly apply one weight to the full action vector. We show that this shared weighting couples heterogeneous action components: in 665,714 historical same-target candidates from SmolVLA rollouts, the median absolute gripper command is 0.999 while the median norm of the six-dimensional arm command is 0.652, and the gripper magnitude exceeds the arm norm in 90.2% of candidates. A gripper sign revision can therefore change the similarity weight subsequently applied to every arm dimension. We introduce Component-Decoupled Temporal Aggregation (CDTA-16), which independently scores and aggregates arm and gripper candidates using fixed similarity, age-decay, and maximum-age parameters. On a frozen ACT development panel, CDTA-16 matched Fresh at 37/40 and scored 37/40 versus 36/40 for the matched shared control. The comparison contained only one CDTA-only success and no matched-only success, so the +1/40 paired net advantage missed the predeclared +3/40 advance gate. We therefore do not claim a closed-loop benefit from component decoupling and did not launch the planned held-out panel.

## 1. Introduction

Action chunking asks a policy to predict a sequence of future actions from one observation. It reduces the effective decision horizon and has become a standard design in imitation-learning and vision-language-action policies. When the policy is queried again before the previous chunk ends, its predictions overlap. At environment step \(t\), several chunks may therefore contain an action intended for the same physical time \(t\). ACT averages these same-target predictions with fixed temporal weights, while CogACT assigns larger weights to historical predictions that are more similar to the current prediction ([Zhao et al., 2023](https://arxiv.org/abs/2304.13705); [Li et al., 2024](https://arxiv.org/abs/2411.19650)).

The overlapping predictions are not interchangeable. An old prediction conditions on an old observation, but it can encode a useful temporally delayed view of the target action. Recent analysis attributes part of the benefit of action chunking to learning several temporal relationships, such as \(a_t\mid o_t\), \(a_t\mid o_{t-1}\), and \(a_t\mid o_{t-2}\), which act as an implicit ensemble ([Lazzati et al., 2026](https://arxiv.org/abs/2608.02547)). The deployment question is consequently not only whether to use historical predictions, but how to combine them.

Current aggregation rules usually answer this question once for the entire action vector. For the seven-dimensional actions studied here, a candidate contains six continuous end-effector motion coordinates and one near-binary gripper coordinate. These components differ in both geometry and semantics. A shared full-action similarity implicitly assumes that one scalar notion of revision is appropriate for both.

Our cache audit reveals the concrete failure mode. Across 665,714 historical candidates, the median absolute gripper value is approximately one, whereas the median six-dimensional arm norm is 0.652. The gripper magnitude is larger than the complete arm norm in 90.2% of candidates. In a concatenated cosine, changing the gripper sign can thus dominate the similarity revision and change the weights applied to arm predictions, even when the arm geometry remains compatible. The reverse contamination is also possible: an arm revision can alter the weight used to infer gripper intent.

We address this coupling with Component-Decoupled Temporal Aggregation (CDTA). CDTA-16 retains same-target aggregation but computes an arm score from arm cosine similarity and a gripper score from sign agreement. Each score uses the same fixed physical-age penalty, and each component is normalized and aggregated independently over candidates no older than 16 control steps. The policy, action representation, query frequency, and training procedure remain unchanged.

This paper makes three contributions:

1. We formulate temporal aggregation using explicit same-target semantics and isolate cross-component interference caused by one shared full-action similarity.
2. We provide a cache-scale audit and controlled source-age probes showing that arm and gripper predictions can have different temporal sensitivities, while retaining negative and heterogeneous tasks.
3. We evaluate CDTA-16, a fixed, training-free component-decoupled aggregator, against a matched shared control that differs only in component decoupling. CDTA-16 preserved Fresh performance but produced only one paired net win over the matched control, so it failed the frozen development gate and was not advanced to held-out evaluation.

## 2. Related Work

### Action chunking and same-target ensembling

ACT predicts action sequences and queries the policy at every time step so that overlapping chunks provide multiple predictions for one target time. Its temporal ensemble averages those predictions with exponentially varying weights ([Zhao et al., 2023](https://arxiv.org/abs/2304.13705)). Lazzati et al. analyze action chunking through delayed observation-conditioned policies and implicit ensembling, establishing that historical temporal relationships can be useful without asserting that every source age is equally useful ([Lazzati et al., 2026](https://arxiv.org/abs/2608.02547)). Our focus is narrower: we study whether heterogeneous components of the same target action should inherit one shared temporal weight.

### Adaptive aggregation and action selection

CogACT's Adaptive Action Ensemble weights cached same-target predictions by full-action cosine similarity to the current prediction ([Li et al., 2024](https://arxiv.org/abs/2411.19650)). The published equation uses \(w_k\propto\exp(0.1\cos(\cdot,\cdot))\); its released code normalizes these positive weights before averaging. This full-action rule is the closest aggregation comparator to CDTA. Temporal Action Selection instead learns a lightweight selector over cached chunks ([Weng et al., 2025](https://arxiv.org/abs/2511.04421)), while AutoHorizon adapts how many actions to execute from a predicted chunk ([Wang et al., 2026](https://arxiv.org/abs/2602.21445)). CDTA neither trains a selector nor changes the execution horizon. It modifies only the grouping of same-target aggregation weights.

### Policies and benchmark

We study both ACT and SmolVLA, a compact vision-language-action policy that generates chunked actions and supports asynchronous inference ([Shukor et al., 2025](https://arxiv.org/abs/2506.01844)). Experiments use LIBERO, which provides language-conditioned manipulation tasks organized into Object, Spatial, Goal, and long-horizon suites ([Liu et al., 2023](https://arxiv.org/abs/2306.03310)).

## 3. Problem Formulation

### 3.1 Action chunks and source age

At policy-query time \(q\), an action-chunked policy observes \(o_q\) and predicts a horizon-\(H\) chunk

\[
\mathbf{A}_q
=
\left(
\mathbf{a}_{q\mid q},
\mathbf{a}_{q+1\mid q},
\ldots,
\mathbf{a}_{q+H-1\mid q}
\right).
\]

The notation \(\mathbf{a}_{t\mid q}\) means the action for physical target time \(t\) predicted by the query issued at time \(q\). At execution time \(t\), the available same-target candidate set is

\[
\mathcal{C}_t
=
\left\{
\mathbf{a}_{t\mid q}
\;\middle|\;
\max(0,t-H+1)\le q\le t
\right\}.
\]

Candidate \(q\) has source age

\[
d_q=t-q.
\]

The fresh candidate is \(\mathbf{a}_{t\mid t}\), with \(d_t=0\). A historical candidate of age \(d\) is row \(d\) of the chunk predicted at \(t-d\). It is not the action applied at time \(t-d\), the previously applied gripper command, or a held command. Every candidate in \(\mathcal{C}_t\) targets the same physical time \(t\).

### 3.2 Heterogeneous action components

We split each postprocessed seven-dimensional candidate into

\[
\mathbf{a}_{t\mid q}
=
\left[
\mathbf{a}^{\mathrm{arm}}_{t\mid q},
g_{t\mid q}
\right],
\qquad
\mathbf{a}^{\mathrm{arm}}_{t\mid q}\in\mathbb{R}^{6},
\quad
g_{t\mid q}\in\mathbb{R}.
\]

The arm block contains relative translation and rotation commands. The scalar gripper coordinate is continuous in the policy interface but empirically concentrates near two signed states. We aggregate after the native policy and environment postprocessors so that scores are computed in the action domain actually sent to the simulator.

### 3.3 Shared full-action aggregation

A shared aggregator computes one score \(s_q^{\mathrm{shared}}\), normalizes it over candidates, and applies the resulting scalar weight to all seven dimensions:

\[
w_q^{\mathrm{shared}}
=
\frac{\exp(s_q^{\mathrm{shared}})}
{\sum_{j\in\mathcal{C}_t}\exp(s_j^{\mathrm{shared}})},
\qquad
\hat{\mathbf{a}}_t
=
\sum_{q\in\mathcal{C}_t}
w_q^{\mathrm{shared}}\mathbf{a}_{t\mid q}.
\]

For concatenated actions \(\mathbf{x}=[\mathbf{u},g]\) and \(\mathbf{y}=[\mathbf{v},h]\), full-action cosine similarity is

\[
\cos(\mathbf{x},\mathbf{y})
=
\frac{\mathbf{u}^{\top}\mathbf{v}+gh}
{\sqrt{\lVert\mathbf{u}\rVert_2^2+g^2}
 \sqrt{\lVert\mathbf{v}\rVert_2^2+h^2}}.
\]

The \(gh\) term affects the shared score and therefore the weight subsequently applied to every arm coordinate. When gripper magnitudes are near one and arm norms are smaller, a sign change alters the numerator by approximately \(2|gh|\). The shared score then mixes gripper-intent revision with arm-geometry revision.

## 4. Empirical Motivation

### 4.1 Controlled same-target source probes

We first measured sensitivity to source age without aggregation. The frozen SmolVLA probe covered eight LIBERO tasks, two from each suite, with ten initial states per task and ten source assignments. This produced 800 episodes. The runner created one environment per condition and reset it sequentially, so each condition traversed initial-state IDs 0–9. At every environment step, the policy was queried once and the selected components came from predictions for the same target time.

The probe used Fresh \((d_{\mathrm{arm}},d_{\mathrm{grip}})=(0,0)\) and, at ages \(d\in\{4,8,16\}\), three interventions:

\[
\begin{aligned}
\mathrm{FO}(d)&:(0,d),\\
\mathrm{FullOld}(d)&:(d,d),\\
\mathrm{Reverse}(d)&:(d,0).
\end{aligned}
\]

FO retains a fresh arm and an age-\(d\) gripper prediction. Reverse retains an age-\(d\) arm and a fresh gripper prediction. Before an age-\(d\) candidate exists, the intervention uses the fresh component and records actual age zero.

| source assignment | age 4 | age 8 | age 16 |
|---|---:|---:|---:|
| Fresh | 66/80 (82.5%) | 66/80 (82.5%) | 66/80 (82.5%) |
| FO: fresh arm, historical gripper | 68/80 (85.0%) | 65/80 (81.2%) | 61/80 (76.2%) |
| FullOld: historical arm and gripper | 68/80 (85.0%) | 68/80 (85.0%) | 58/80 (72.5%) |
| Reverse: historical arm, fresh gripper | 65/80 (81.2%) | 64/80 (80.0%) | 50/80 (62.5%) |

The response is neither monotonic nor universal. At age 16, Reverse falls 16 successes below Fresh, while FO falls 5. FO exceeds Reverse on four of eight tasks at this age. Other tasks, especially in the Goal suite, do not follow the same ordering. These interventions support component-sensitive temporal degradation, not a universal claim that historical gripper predictions are better.

The SmolVLA policy is stochastic, and the historical runner did not key its flow noise by physical step. The conditions match initial states but not necessarily the sampling noise at each step. We therefore use these success rates as mechanism motivation and retain all tasks, rather than treating small one- or two-episode differences as definitive performance gains.

### 4.2 Descriptive cross-policy source sensitivity

A historical ACT source probe evaluated four tasks with 40 episodes per condition. Its aggregate condition rates were Fresh 37/40 (92.5%), FO16 33/40 (82.5%), FullOld16 28/40 (70.0%), and Reverse16 17/40 (42.5%). The ordering is directionally consistent with greater sensitivity to a stale arm than to a stale gripper at age 16.

These ACT conditions are not episode-paired. The runner assigned each vector environment's `init_state_id` only once before iterating over conditions, while `LiberoEnv.reset` advances that identifier. The conditions therefore likely used different state cohorts. We report only aggregate condition rates as descriptive cross-policy evidence. We do not report candidate-only/reference-only counts, paired confidence intervals, McNemar tests, or claims of a paired ACT advantage from this run.

### 4.3 Cache-scale magnitude and interference audit

We audit the fresh-query caches from the valid SmolVLA source-probe cohort. The cache contains 80 episodes, 15,586 executed target steps, and horizon \(H=50\). For every target step, we enumerate all available non-fresh predictions for that same target, corresponding to ages 1–49 when present. This yields exactly 665,714 historical candidates:

\[
N_{\mathrm{hist}}
=
\sum_{e}\sum_{t}
\min(t,H-1)
=665{,}714.
\]

All statistics use postprocessed actions. Across these candidates,

\[
\operatorname{median}|g|=0.999355,
\qquad
\operatorname{median}\lVert\mathbf{a}^{\mathrm{arm}}\rVert_2=0.652294.
\]

Moreover,

\[
600{,}302/665{,}714=90.174\%
\]

of historical candidates satisfy

\[
|g|>\lVert\mathbf{a}^{\mathrm{arm}}\rVert_2.
\]

This result corrects the tempting dimensionality argument that a six-dimensional arm must dominate a one-dimensional gripper in full-action similarity. In this action domain, the scalar gripper frequently has the larger magnitude. A shared cosine can consequently use a gripper transition to rewrite the temporal weights for arm motion. The audit establishes exposure to interference at scale; it does not by itself prove that decoupling improves closed-loop success.

### 4.4 Historical aggregation evidence and its boundary

An exploratory SmolVLA aggregation run reported Fresh 72/80, the official ACT ensemble 66/80, physical age decay 71/80, a project-frozen CogACT-style shared rule with α=0.3 at 73/80, and an earlier component-aware rule at 73/80. This run does not establish a paired ten-state comparison. It created a new single-worker environment for every seed and did not set `init_state_id`, so the ten episodes per task likely reused initial state 0. The 72/80 and 73/80 summaries are retained only as exploratory evidence that motivated a clean matched control; they are excluded from the final CDTA performance claim.

## 5. Component-Decoupled Temporal Aggregation

### 5.1 Fixed candidate window

CDTA-16 restricts aggregation to same-target candidates no older than 16 control steps:

\[
\mathcal{C}_t^{16}
=
\left\{
\mathbf{a}_{t\mid q}\in\mathcal{C}_t
\;\middle|\;
0\le d_q\le16
\right\}.
\]

At 30 Hz, the maximum physical source age is approximately 0.533 s. During the first 16 steps, the window contains all available candidates and grows naturally from one candidate.

### 5.2 Independent component scores

Let the newest candidate \(\mathbf{a}_{t\mid t}\) be the reference. CDTA-16 scores arm candidates by their stabilized cosine similarity to the newest arm prediction:

\[
s_q^{\mathrm{arm}}
=
0.3\,
\cos_{\varepsilon}
\left(
\mathbf{a}^{\mathrm{arm}}_{t\mid q},
\mathbf{a}^{\mathrm{arm}}_{t\mid t}
\right)
-0.03d_q,
\]

where

\[
\cos_{\varepsilon}(\mathbf{x},\mathbf{y})
=
\frac{\mathbf{x}^{\top}\mathbf{y}}
{\lVert\mathbf{x}\rVert_2\lVert\mathbf{y}\rVert_2+\varepsilon}
\]

and the implementation uses \(\varepsilon=10^{-7}\). Gripper candidates are scored by intent agreement:

\[
s_q^{\mathrm{grip}}
=
0.3\,
\operatorname{sign}(g_{t\mid q})
\operatorname{sign}(g_{t\mid t})
-0.03d_q.
\]

We use the conventional \(\operatorname{sign}(0)=0\). Positive and negative gripper candidates that agree with the newest intent receive the same similarity term regardless of magnitude, while the physical-age penalty still favors recent evidence.

### 5.3 Independent normalization and aggregation

CDTA normalizes scores separately for each component:

\[
w_q^{c}
=
\frac{\exp(s_q^{c})}
{\sum_{j\in\mathcal{C}_t^{16}}\exp(s_j^{c})},
\qquad
c\in\{\mathrm{arm},\mathrm{grip}\}.
\]

The executed components are

\[
\hat{\mathbf{a}}_t^{\mathrm{arm}}
=
\sum_{q\in\mathcal{C}_t^{16}}
w_q^{\mathrm{arm}}
\mathbf{a}^{\mathrm{arm}}_{t\mid q},
\]

\[
\hat{g}_t
=
\sum_{q\in\mathcal{C}_t^{16}}
w_q^{\mathrm{grip}}
g_{t\mid q},
\qquad
\hat{\mathbf{a}}_t
=
\left[
\hat{\mathbf{a}}_t^{\mathrm{arm}},
\hat{g}_t
\right].
\]

The method has three fixed numerical settings: similarity scale 0.3, physical-age penalty 0.03 per control step, and maximum age 16. It adds no learned head, training loss, task-specific parameter, or additional policy query. We do not apply a hard gripper vote.

### 5.4 Matched shared control

To isolate component decoupling, the matched shared control uses the same candidate window, similarity scale, and age penalty but computes one full-action score:

\[
s_q^{\mathrm{matched}}
=
0.3\,
\cos_{\varepsilon}
\left(
\mathbf{a}_{t\mid q},
\mathbf{a}_{t\mid t}
\right)
-0.03d_q,
\]

\[
w_q^{\mathrm{matched}}
=
\operatorname{softmax}_{q\in\mathcal{C}_t^{16}}
\left(s_q^{\mathrm{matched}}\right),
\qquad
\hat{\mathbf{a}}_t^{\mathrm{matched}}
=
\sum_q w_q^{\mathrm{matched}}\mathbf{a}_{t\mid q}.
\]

Thus, CDTA-16 and the matched shared control differ only in whether arm and gripper receive independent similarity scores, normalization, and aggregation. Their comparison does not confound decoupling with age truncation or decay.

## 6. Experimental Protocol

### 6.1 Research questions and methods

The evaluation asks three questions:

1. Does CDTA-16 improve closed-loop success over the matched shared control?
2. Does CDTA-16 preserve Fresh performance while improving over shared temporal aggregation?
3. Are any gains concentrated in tasks with asynchronous arm and gripper revisions?

Every task-policy panel compares five inference rules:

1. **Fresh:** execute the newest same-target candidate.
2. **Official ACT temporal ensemble:** use the policy implementation's standard ACT same-target ensemble.
3. **CogACT-style shared full-action:** use all available same-target candidates with \(w_q\propto\exp(0.1\cos(\mathbf{a}_{t\mid q},\mathbf{a}_{t\mid t}))\). The coefficient matches CogACT, while the full-history support is a frozen LIBERO control rather than a reproduction of CogACT's embodiment-specific deployment horizons.
4. **Matched shared age≤16:** use the score in Section 5.4.
5. **CDTA-16:** use the component-decoupled scores in Sections 5.1–5.3.

All methods query the policy once per environment step and aggregate postprocessed actions for the same target time. No method retrains or fine-tunes the policy.

### 6.2 ACT development panel

The development panel contains four previously inspected tasks:

- LIBERO-Object task 6;
- LIBERO-Spatial task 2;
- LIBERO-Goal task 1;
- LIBERO-10 task 3.

Each method receives the same ten prespecified initial states and environment seeds on each task, for

\[
4\ \text{tasks}\times10\ \text{states}\times5\ \text{methods}
=200\ \text{episodes}.
\]

The development panel uses initial-state IDs 10--19 and environment seeds 2000--2009. The runner assigns every requested `init_state_id` immediately before each method reset. ACT runs in evaluation mode; its policy state and random-number generators are reset to the same fixed stream before every method.

The primary development comparison is CDTA-16 versus the matched shared control. The method passes the primary gate if paired net wins are at least \(+3/40\), where net wins equal the number of CDTA-only successes minus matched-shared-only successes, and CDTA-16 is no worse on at least three of four tasks. The Fresh safeguard allows a pooled deficit of at most two successes. On every task, Fresh-only successes minus CDTA-only successes must remain below 2/10. We apply this gate once after all 200 episodes and do not tune parameters from interim outcomes.

Passing the gate freezes the method. We do not sweep the similarity scale, age penalty, window, action grouping, or a learned head. Failing to obtain a stable advantage over the matched shared control ends the component-decoupling claim.

### 6.3 Planned held-out panel (not launched)

The preregistered held-out panel would have used eight tasks that received no CDTA intervention tuning:

| suite | held-out task IDs |
|---|---|
| LIBERO-Object | 1, 4 |
| LIBERO-Spatial | 3, 7 |
| LIBERO-Goal | 0, 3 |
| LIBERO-10 | 1, 9 |

Each task uses initial-state IDs 20–29 and the same five methods. We evaluate two frozen policy families, SmolVLA and task-specific ACT checkpoints:

\[
8\ \text{tasks}\times10\ \text{states}\times5\ \text{methods}\times2\ \text{policies}
=800\ \text{episodes}.
\]

The component-decoupling gate failed before this panel began. We therefore preserve the planned task list for auditability but report no held-out CDTA outcomes.

### 6.4 State and randomness control

Every method receives the same environment seed and explicit LIBERO initial-state identifier. The runner reassigns `init_state_id` immediately before every reset because `LiberoEnv.reset` advances its internal state index.

For SmolVLA, sampling noise is stateless and keyed by

\[
(\text{suite},\text{task ID},\text{initial-state ID},\text{environment step}).
\]

All methods therefore receive identical flow noise at the same physical step, even when another method terminates early. A single global random stream is insufficient because early termination would shift later draws. For ACT, the policy RNG stream is reset identically before each matched method episode. These controls apply to the new development and held-out evaluations; historical frozen experiments are not retroactively rerun.

### 6.5 Metrics and reporting

The binary episode success is the primary outcome. We report every task separately, suite-macro averages, policy-specific pooled counts, and the predeclared all-task summary. For matched initial states, we report candidate-only and reference-only successes and paired net wins. Any interval or significance test used in the final paper will state its unit and multiplicity treatment.

The development result and stopped held-out panel are:

| policy/panel | Fresh | official ACT | CogACT | matched shared age≤16 | CDTA-16 | CDTA net wins vs matched shared |
|---|---:|---:|---:|---:|---:|---:|
| ACT development, 40 episodes/method | 37/40 | 31/40 | 28/40 | 36/40 | **37/40** | **1/0 (net +1)** |
| ACT held-out, 80 episodes/method | not run | not run | not run | not run | **not run** | **not run** |
| SmolVLA held-out, 80 episodes/method | not run | not run | not run | not run | **not run** | **not run** |

CDTA was no worse than matched shared on all four development tasks and preserved Fresh overall, so the safeguard criteria passed. The primary component-decoupling criterion failed because 39/40 paired outcomes provided no net evidence distinguishing CDTA from the matched shared control.

## 7. Limitations

CDTA assumes a semantically meaningful partition between a six-dimensional arm block and a scalar gripper command. The same partition is natural for the LIBERO action interface but may not transfer unchanged to joint-space control, mobile manipulators, dexterous hands, or multiple end effectors. Applying CDTA to those domains requires a prespecified component grouping rather than a post hoc grouping selected from task outcomes.

The cache audit demonstrates a strong scale imbalance and an algebraic path for interference, but it is observational. It does not prove that scale imbalance is the only cause of closed-loop failures. The controlled source probes show task-dependent temporal sensitivity, and several Goal tasks are counterexamples to a universal component ordering.

CDTA-16 uses fixed parameters and a fixed age window. This simplicity removes training and task-specific tuning, but it may underperform when relevant temporal scales differ sharply across control frequencies or embodiments. The development gate tests whether the chosen fixed rule is useful; a failed gate cannot be repaired by presenting only favorable held-out tasks.

Independent gripper aggregation can average positive and negative scalar commands into an intermediate value. The sign-agreement score reduces cross-intent mixing but does not impose a hard vote. We intentionally omit the previously considered sign-vote module because it did not materially change executed signs in development diagnostics.

The method still queries the policy at every environment step and caches overlapping chunks. It is an aggregation method, not a query-saving or latency-hiding method. Learned source selectors and adaptive execution horizons address different deployment objectives.

Finally, two historical evaluations have protocol limitations. The ACT condition run advanced LIBERO state identifiers between conditions and cannot support paired tests. The exploratory SmolVLA aggregation run likely repeated state 0 across its nominal seeds. Neither run contributes a CDTA performance value. The valid paired development panel did not support advancing CDTA to held-out evaluation.

## Verified primary references

- Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705). 2023.
- Qixiu Li et al. [CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation](https://arxiv.org/abs/2411.19650). 2024. [Official Adaptive Action Ensemble implementation](https://github.com/microsoft/CogACT/blob/b174a1b86deedfab4d198d935207e7bb0527994e/sim_cogact/adaptive_ensemble.py#L29-L42).
- Mustafa Shukor et al. [SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics](https://arxiv.org/abs/2506.01844). 2025.
- Bo Liu et al. [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://arxiv.org/abs/2306.03310). 2023.
- Filippo Lazzati et al. [Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?](https://arxiv.org/abs/2608.02547). 2026.
- Yueyang Weng et al. [Temporal Action Selection for Action Chunking](https://arxiv.org/abs/2511.04421). 2025.
- Haoxuan Wang et al. [VLA Knows Its Limits: Adaptive Execution Horizons for Robot Policies](https://arxiv.org/abs/2602.21445). 2026.
