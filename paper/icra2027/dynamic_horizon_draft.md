# Component-Agreement Dynamic Horizons for Action-Chunked Robot Policies

> **Negative research draft; both dynamic-horizon gates failed.** Adaptive h8 exactly reproduced fixed h8's 40 paired outcomes while using more policy queries. Adaptive h16 then achieved 36/40 versus 38/40 for fixed h16 while also using more queries. Neither adaptive rule was advanced to held-out evaluation.
>
> **Internal method-selection record, not a paper contribution.** In the preceding frozen ACT development panel, CDTA-16 achieved 37/40 successes and its matched shared control achieved 36/40. There was one CDTA-only success and no matched-only success, so the paired net difference was \(+1/40\), below the predeclared \(+3/40\) gate. We treat this as a negative, inconclusive development result. It does not support the dynamic-horizon method.

## Abstract

Action-chunked robot policies predict several future actions from one observation. Deployment can query the policy again at every control step or execute several actions from an earlier chunk, trading observation-conditioned correction for fewer policy queries. We study whether disagreement among already available predictions for the same target time can determine when a new query is needed. Controlled source-age probes and an audit of 665,714 historical SmolVLA candidates motivate separate arm and gripper signals: arm and gripper sources have heterogeneous age sensitivity, while the near-binary gripper command has greater magnitude than the six-dimensional arm vector in 90.2% of audited candidates. We evaluate a training-free component-agreement scheduler for ACT. On a frozen four-task LIBERO development panel, Fresh, fixed h4, fixed h8, and adaptive h8 achieved 37/40, 37/40, 38/40, and 38/40 successes. Adaptive h8 used a pooled query rate of 0.175, whereas fixed h8 produced identical paired outcomes at 0.128. Extending the maximum age did not reveal hidden value: fixed h16 achieved 38/40 at rate 0.0657, while adaptive h16 achieved 36/40 at rate 0.1144. Component triggering therefore failed both development gates and was not advanced to held-out evaluation. The retained positive hypothesis is a fixed source-age cap, evaluated separately on unseen tasks; this draft makes no adaptive-method claim.

## 1. Introduction

Action chunking asks a robot policy to predict a sequence of future actions from one observation. It shortens the effective prediction horizon during training and is central to policies such as [ACT](https://arxiv.org/abs/2304.13705). A chunk also creates a deployment choice. The system can query the policy at every environment step, obtaining overlapping predictions for the same physical target time, or it can execute several actions from an earlier chunk before querying again. The former uses current observations but pays for a policy query at every step. The latter reduces queries but increases the age of the observation that produced the executed action.

The relevant unit of comparison is a same-target candidate. If a policy query at time \(q\) predicts an action for physical time \(t\), we denote that prediction by \(\mathbf{a}_{t\mid q}\). Two candidates \(\mathbf{a}_{t\mid q_1}\) and \(\mathbf{a}_{t\mid q_2}\) target the same control step even though they were conditioned on different observations. This distinction matters because action chunks encode multiple delayed observation-action relationships rather than repeated estimates with identical information ([Lazzati et al., 2026](https://arxiv.org/abs/2608.02547)).

Existing approaches expose two broad controls. ACT queries at every step and temporally ensembles overlapping predictions. Fixed-horizon execution instead commits to a preset number of actions before the next query. Recent work also learns to adapt an execution horizon ([AutoHorizon](https://arxiv.org/abs/2602.21445)). Our question is narrower: can the predictions already stored in action chunks provide a training-free signal for whether to query now? We focus on a receding decision made before a new query, so the signal cannot depend on the candidate that the new query would produce.

### 1.1 Source-age evidence

Our empirical motivation comes from a frozen SmolVLA source-age probe on LIBERO ([SmolVLA](https://arxiv.org/abs/2506.01844); [LIBERO](https://arxiv.org/abs/2306.03310)). The probe covered eight tasks, two from each LIBERO suite, with ten initial states per task and ten source assignments, for 800 episodes. The runner created one environment per condition and reset it sequentially, so every condition traversed initial-state IDs 0–9. At every environment step, the policy was queried once. Each intervention then selected arm and gripper components from candidates that all targeted the current physical step.

Let source age be \(d=t-q\). Besides Fresh, \((d_{\mathrm{arm}},d_{\mathrm{grip}})=(0,0)\), the probe evaluated three interventions at \(d\in\{4,8,16\}\):

\[
\begin{aligned}
\mathrm{FO}(d)&=(0,d),\\
\mathrm{FullOld}(d)&=(d,d),\\
\mathrm{Reverse}(d)&=(d,0).
\end{aligned}
\]

The aggregate success counts were:

| source assignment | age 4 | age 8 | age 16 |
|---|---:|---:|---:|
| Fresh | 66/80 | 66/80 | 66/80 |
| FO: fresh arm, historical gripper | 68/80 | 65/80 | 61/80 |
| FullOld: historical arm and gripper | 68/80 | 68/80 | 58/80 |
| Reverse: historical arm, fresh gripper | 65/80 | 64/80 | 50/80 |

The response was heterogeneous rather than monotonic. At age 16, replacing only the arm source reduced the aggregate count more than replacing only the gripper source, but this ordering was not universal across tasks. The SmolVLA policy is stochastic, and the historical runner did not key flow noise by physical step. Initial states were matched across conditions, but sampling noise was not necessarily matched. We therefore use the probe as mechanism motivation and do not interpret small count differences as performance effects.

A historical ACT source probe showed aggregate rates of 37/40 for Fresh, 33/40 for FO16, 28/40 for FullOld16, and 17/40 for Reverse16. These conditions are not episode-paired. Its runner set each vector environment's init_state_id once before iterating over conditions, while LiberoEnv.reset advances the identifier. The conditions likely traversed different state blocks. We retain the aggregate ordering only as descriptive cross-policy source-sensitivity evidence and make no paired claim from this run.

### 1.2 Cache-scale component audit

We also audited the valid fresh-query caches from the SmolVLA probe. The cache contains 80 episodes, 15,586 executed target steps, and chunk horizon \(H=50\). Enumerating every available non-fresh prediction for the same physical target produces exactly

\[
N_{\mathrm{hist}}
=
\sum_e\sum_t \min(t,H-1)
=665{,}714
\]

historical candidates. All statistics use postprocessed actions in the simulator action domain. Across these candidates,

\[
\operatorname{median}|g|=0.999355,
\qquad
\operatorname{median}\lVert\mathbf{a}^{\mathrm{arm}}\rVert_2=0.652294,
\]

and

\[
\frac{600{,}302}{665{,}714}=90.174\%
\]

satisfy \(|g|>\lVert\mathbf{a}^{\mathrm{arm}}\rVert_2\). Thus, although the arm has six dimensions and the gripper has one, a gripper sign transition can dominate a similarity computed on the concatenated action. The audit does not show that any scheduler improves control. It motivates keeping the arm geometry and gripper intent tests separate.

### 1.3 From disagreement to query scheduling

A preceding development experiment tested whether separate component similarities should instead produce separate aggregation weights. CDTA-16 matched the Fresh count, but its 37/40 versus 36/40 comparison with a matched shared control contained only one discordant success and missed its frozen advance gate. We therefore do not present component-decoupled aggregation as a positive result.

The Plan-B hypothesis uses the same observation differently. When two recent same-target predictions agree in both arm direction and gripper sign, the latest stored chunk may be sufficient for the current step. When either component revises, a new observation-conditioned query may be more appropriate than mixing the candidates. This yields a minimal scheduler with four fixed triggers and no learned selector, aggregation rule, or parameter sweep.

The candidate contribution is a pre-query, component-aware execution rule that converts disagreement among existing same-target candidates into a decision to refresh the action chunk. Its empirical status is intentionally unresolved pending the frozen development experiment.

## 2. Component-Agreement Dynamic Horizon

### 2.1 Sparse same-target records

At a queried environment step \(q\), an action-chunked policy observes \(o_q\) and predicts

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

At physical target step \(t\), the scheduler first inspects only chunks queried before the decision at \(t\). Let \(\mathcal{Q}_{<t}\) be those query times. The pre-query same-target set is

\[
\mathcal{C}^{-}_t
=
\left\{
(q,\mathbf{a}_{t\mid q})
\;\middle|\;
q\in\mathcal{Q}_{<t},\ 0\le t-q<H
\right\}.
\]

Candidates are ordered by source query time, from oldest to newest. Let \(m=|\mathcal{C}^{-}_t|\), and, when \(m>0\), let \(q_m\) be the newest available query time. Its source age is

\[
d_m=t-q_m.
\]

Each postprocessed seven-dimensional candidate is split into a six-dimensional arm command and a scalar gripper command:

\[
\mathbf{a}_{t\mid q}
=
\left[
\mathbf{a}^{\mathrm{arm}}_{t\mid q},
g_{t\mid q}
\right].
\]

This set contains no prediction conditioned on \(o_t\), because the decision is made before a query at \(t\). This ordering prevents a fresh candidate from leaking into the query trigger.

### 2.2 Fixed component-agreement rule

When at least two candidates exist, let \(q_{m-1}\) and \(q_m\) denote the two most recent source queries. We compute their stabilized arm cosine

\[
c_t^{\mathrm{arm}}
=
\frac{
\left(\mathbf{a}^{\mathrm{arm}}_{t\mid q_{m-1}}\right)^\top
\mathbf{a}^{\mathrm{arm}}_{t\mid q_m}
}{
\left\lVert\mathbf{a}^{\mathrm{arm}}_{t\mid q_{m-1}}\right\rVert_2
\left\lVert\mathbf{a}^{\mathrm{arm}}_{t\mid q_m}\right\rVert_2
+10^{-7}
}
\]

and gripper sign disagreement

\[
r_t^{\mathrm{grip}}
=
\mathbb{1}
\left[
\operatorname{sign}(g_{t\mid q_{m-1}})
\ne
\operatorname{sign}(g_{t\mid q_m})
\right].
\]

The scheduler queries at \(t\) if any of four conditions holds:

\[
z_t
=
\mathbb{1}
\left[
\begin{array}{l}
m<2\\
\quad\lor\ d_m\ge 8\\
\quad\lor\ r_t^{\mathrm{grip}}=1\\
\quad\lor\ c_t^{\mathrm{arm}}<0.90.
\end{array}
\right]
\]

All constants are fixed before the development outcomes: maximum source age 8, arm cosine threshold 0.90, and two candidates for an agreement test. There is no sweep over thresholds or horizons.

If \(z_t=1\), the policy is queried, the new chunk \(\mathbf{A}_t\) is stored, and the fresh action \(\mathbf{a}_{t\mid t}\) is executed. If \(z_t=0\), the scheduler executes the newest existing same-target candidate:

\[
\hat{\mathbf{a}}_t
=
\begin{cases}
\mathbf{a}_{t\mid t}, & z_t=1,\\
\mathbf{a}_{t\mid q_m}, & z_t=0.
\end{cases}
\]

No averaging, voting, or temporal weighting is applied. At the first step, \(m=0\) forces a query. At the next step, \(m=1\) again forces a query, which bootstraps two independently queried chunks. Thereafter, the decision is recomputed at every physical step. The interval between policy queries is therefore an observed outcome rather than a horizon predicted once and committed in advance.

### 2.3 Computational interface

The rule uses only postprocessed actions already present in sparse query records. It adds one six-dimensional cosine, one scalar sign comparison, a candidate-count check, and an age check before a possible policy call. It introduces no learned head, extra training, additional observation encoder, or temporal aggregation. Skipped steps execute an action already predicted for that exact physical target time, not the action applied at a previous step.

## 3. Frozen Development Protocol

### 3.1 Policy, tasks, and paired states

The candidate is evaluated with the official LeRobot ACT implementation at checkpoint step 100,000. The panel contains four LIBERO tasks:

| suite and task | language instruction | initial-state IDs |
|---|---|---|
| Object 6 | pick up the butter and place it in the basket | 10–19 |
| Spatial 2 | pick up the black bowl from table center and place it on the plate | 10–19 |
| Goal 1 | put the bowl on the stove | 10–19 |
| LIBERO-10 3 | put the black bowl in the bottom drawer of the cabinet and close it | 10–19 |

Each task uses ten episodes with environment seeds 2000–2009, paired one-to-one with initial-state IDs 10–19. The runner assigns the requested init_state_id immediately before every reset and checks the assignment. ACT's policy state and random-number generators are reset to seed 424242 before each method-state episode. The experiment uses one synchronous environment. Each policy query calls predict_action_chunk followed by the native policy and environment postprocessors.

### 3.2 Comparators

The comparison contains four execution rules:

1. **Fresh (\(h=1\)).** Query at every physical step and execute the fresh same-target candidate. We reuse the valid 37/40 reference from the completed ACT development panel because it uses the identical tasks, initial-state IDs, environment seeds, checkpoint, and postprocessing path.
2. **Fixed \(h=4\).** Query at steps \(0,4,8,\ldots\). Between queries, execute the newest available same-target action from the latest sparse query record.
3. **Fixed \(h=8\).** Query at steps \(0,8,16,\ldots\), with the same newest-candidate execution rule.
4. **Component agreement.** Apply the four pre-query triggers in Section 2.2.

Fresh contributes 40 reusable reference episodes. The three sparse-query methods contribute \(4\times10\times3=120\) new episodes. No sparse method performs temporal aggregation.

For each episode, the runner records success, environment steps, policy-query count, query steps, executed source ages, and trigger counts. Episode query rate is

\[
\rho_e=\frac{N^{\mathrm{query}}_e}{N^{\mathrm{step}}_e}.
\]

The pooled rate reported for the resource gate is

\[
\rho_{\mathrm{pool}}
=
\frac{\sum_e N^{\mathrm{query}}_e}
{\sum_e N^{\mathrm{step}}_e}.
\]

We also retain the episode-level distribution so that early success termination cannot be hidden by a single pooled statistic.

### 3.3 Frozen advance gate

The candidate advances to a held-out evaluation only if all four conditions pass after all 120 new episodes finish:

\[
\begin{aligned}
S_{\mathrm{adaptive}} &\ge 35/40,\\
\rho_{\mathrm{adaptive}} &\le 0.60,\\
L_k^{\mathrm{Fresh}\rightarrow\mathrm{adaptive}} &<2/10
\quad\text{for every task }k,\\
S_{\mathrm{adaptive}}
-\max(S_{h4},S_{h8})
&\ge 2/40.
\end{aligned}
\]

Here \(L_k^{\mathrm{Fresh}\rightarrow\mathrm{adaptive}}\) is the number of Fresh-only successes minus adaptive-only successes on the ten matched states of task \(k\). Thus the taskwise condition permits at most one paired net loss on any task. The final condition requires at least two more pooled successes than the better fixed-horizon control. The gate is evaluated once, with no interim tuning.

## 4. Development Results

The Fresh values below are the valid reusable reference. The sparse-query values come from the completed frozen 120-episode panel.

| task | Fresh \(h=1\) success | fixed \(h=4\) success | fixed \(h=8\) success | component agreement success | Fresh-only / adaptive-only |
|---|---:|---:|---:|---:|---:|
| Object 6 | 8/10 | 8/10 | 8/10 | 8/10 | 1/1 |
| Spatial 2 | 10/10 | 10/10 | 10/10 | 10/10 | 0/0 |
| Goal 1 | 10/10 | 10/10 | 10/10 | 10/10 | 0/0 |
| LIBERO-10 3 | 9/10 | 9/10 | 10/10 | 10/10 | 0/1 |
| **Pooled** | **37/40** | **37/40** | **38/40** | **38/40** | **1/2** |

| method | pooled query rate | mean executed source age | success count |
|---|---:|---:|---:|
| Fresh \(h=1\) | 1.000 | 0.000 | 37/40 |
| fixed \(h=4\) | 0.252 | 1.493 | 37/40 |
| fixed \(h=8\) | 0.128 | 3.470 | 38/40 |
| component agreement | 0.175 | 3.179 | 38/40 |

| frozen gate | observed value | pass |
|---|---:|:---:|
| adaptive success \(\ge35/40\) | 38/40 | yes |
| adaptive query rate \(\le0.60\) | 0.175 | yes |
| taskwise paired net loss from Fresh \(<2/10\) | worst loss 0 | yes |
| adaptive margin over better fixed control \(\ge2/40\) | 38 - 38 = 0 | **no** |
| **advance only if all pass** | gate failed | **no** |

Adaptive h8 and fixed h8 had no discordant episode outcomes. Adaptive h8 issued 1,099 queries over 6,266 environment steps, whereas fixed h8 issued 748 over 5,859. Thus the component triggers added 351 policy calls, a 46.9% increase over fixed h8, without changing one success outcome. The planned held-out panel was not launched.

### 4.1 Prespecified h16 follow-up

The h8 comparison leaves no interval in which the adaptive trigger can reduce queries relative to fixed h8: it can only query earlier than the maximum age. A final development-only test therefore extended the maximum age to 16 while retaining the same arm-cosine threshold, gripper-sign rule, tasks, states, and seeds. Fixed h16 achieved 38/40 at a pooled query rate of 0.0657. Adaptive h16 achieved 36/40 at 0.1144 and had zero adaptive-only versus two fixed-only successes. It failed both the 37/40 success safeguard and the required +3/40 paired net advantage. This second failure ends the component-agreement scheduler branch.

## 5. Limitations

First, the method is a post hoc Plan-B developed after a component-decoupled aggregation candidate failed its own development gate. The four ACT tasks and states 10–19 are consequently a method-development panel, not held-out evidence. Any claim of improvement requires evaluation on tasks and states untouched by this branch.

Second, the source-age probes motivate the mechanism but do not validate the query trigger. The SmolVLA probe did not match stochastic flow noise by physical step, the historical ACT conditions were not episode-paired, and neither probe tested whether a cosine below 0.90 or a gripper sign revision predicts that a new query will prevent failure.

Third, the cache audit establishes a scale and semantics mismatch for concatenated action similarity. It does not show that component disagreement is calibrated, causal, or sufficient for query scheduling. Near-zero arm vectors can make cosine unstable despite numerical stabilization, and a gripper sign change may be a correct planned transition rather than evidence that an older chunk is unsafe to execute.

Fourth, the constants \(8\) and \(0.90\) are fixed for a rapid discriminative test, not derived from a model of uncertainty. Avoiding a sweep reduces development flexibility but does not make the values universal. They may depend on control frequency, chunk horizon, action normalization, policy, and embodiment.

Fifth, query rate is only a proxy for inference cost. The simulator experiment does not measure wall-clock latency, asynchronous execution, energy, network delay, or physical-robot safety. The scheduler must not be assumed to relax actuator limits, collision handling, watchdogs, or any existing safety interlock.

Finally, the development panel contains one policy family, four tasks, and ten states per task. Its gate is deliberately operational rather than a statistical proof. Even if the candidate passes all four conditions, a held-out panel and confidence intervals on paired outcomes and query usage remain necessary before presenting the scheduler as an effective dynamic-horizon method.
