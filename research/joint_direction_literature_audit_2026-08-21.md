# Joint Direction Literature Audit — 2026-08-21

Cutoff: 2026-08-21

Target venue: ICRA 2027

Status vocabulary: `VERIFIED`, `PARTIALLY VERIFIED`, `CONTRADICTED`, `UNKNOWN`

## Bottom line

**Dynamic execution horizon, cached temporal-candidate selection, scalar
similarity-weighted temporal ensembling, and learned full-action temporal
routing are already occupied ideas.** A paper whose only contribution is one of
those mechanisms is not defensible as novel.

The targeted search did **not locate** a prior method that (i) assigns different
source-time weights to translation, rotation, and gripper components of the same
physical action and (ii) explicitly constrains the resulting cross-source joint
action. This is a bounded-search result, not proof of firstness. More
importantly, the project's current evidence does not show that this freedom is
useful. The immediate low-risk distinction is therefore narrower: a **joint**
temporal ensemble whose one shared weight vector and aggregation operators
respect heterogeneous action semantics. That is only `MARGINAL BUT DEFENSIBLE`
if controlled closed-loop experiments show that the semantic treatment itself
is necessary and improves more than one task/policy setting.

## Search and verification method

The literature-review workflow was used as a bounded systematic search rather
than as a claim of exhaustiveness. Queries covered the exact method names below
and combinations of:

- `group-wise temporal action ensemble`, `actuator-wise temporal ensemble`;
- `component-wise action chunking`, `per-dimension temporal selection`;
- `translation rotation gripper temporal ensemble`;
- `heterogeneous action routing robotics`, `multi-rate action chunking`;
- `consistency-constrained action ensemble`, `cross-source action consistency`;
- `adaptive execution horizon`, `cached temporal action selection`, and
  `action chunk boundary consistency`.

Primary arXiv records, official proceedings, author project pages, and official
repositories were preferred. Paper-reported results are described as such; they
were not independently replicated here. CogACT, TAS, and MoH PDFs were also
inspected at the method-equation level. The citation-verification pass checked
titles, author lists, identifiers, dates/status, and the availability of primary
links. This document extends, rather than silently replaces, the broader
[literature claims re-audit](literature_claims_reaudit.md) and
[related-work matrix](related_work_matrix.md).

The scientific-schematics skill recommended by the literature workflow is not
installed. No external image service was used for unpublished project material;
the comparison matrix below is the evidence-preserving substitute.

## Definitions held fixed

- **Prediction horizon:** future action span modeled by one policy query.
- **Execution horizon:** prefix length committed before replanning.
- **Query interval:** physical time between policy forward passes.
- **Temporal age:** physical action time minus the observation/source time that
  produced a candidate for that action.
- **Temporal candidate/expert:** one prediction for the current physical action
  emitted by a chunk from a particular source observation.
- **Group recomposition:** constructing one joint action from components that
  use different temporal source weights.

The papers below do not all study the same quantity. In particular, MoH mixes
training/prediction horizons, whereas ACT/CogACT/TAS combine or select
source-time-overlapping predictions.

## Closest primary literature

| Work | Public date / status by cutoff | What is actually combined or selected? | Weight/decision granularity | Training needed for adaptation? | Group-specific source weights? | Explicit cross-source group consistency? | Audit result for this project |
|---|---|---|---|---|---|---|---|
| [ACT — Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://roboticsproceedings.org/rss19/p016.html) ([arXiv](https://arxiv.org/abs/2304.13705)) | 2023; RSS 2023 | Overlapping predictions for the same physical action; optional exponential temporal ensemble | One age-based scalar per full action prediction | No | No | No | `VERIFIED`: mandatory temporal-ensemble baseline and source of the audited ACT contract. |
| [CogACT](https://arxiv.org/abs/2411.19650) ([project](https://cogact.github.io/), [released ensemble code](https://github.com/microsoft/CogACT/blob/main/sim_cogact/adaptive_ensemble.py)) | 2024-11-29; arXiv preprint | Present and historical predictions for the same action | One scalar per complete action, `exp(alpha * cosine(full_action,newest))`; released default `alpha=0.1` | No | No | Similarity discourages mode averaging, but no group recomposition occurs | `VERIFIED`: a scalar “similarity ensemble” is not new. Any comparison must implement the released cosine rule rather than use the name loosely. |
| [Temporal Action Selection (TAS)](https://arxiv.org/abs/2511.04421) | 2025-11-06; arXiv v2 2026-06-02 | One complete action selected from cached source-time candidates | Categorical full-action selection using learned state/action embeddings; coherence reward penalizes deviation from the prior chunk continuation | Yes, online PPO; residual RL is an optional extension | No | It avoids recomposition by selecting a complete action; coherence is temporal, not a component-mixing constraint | `VERIFIED`: a generic learned cached-candidate router is too close to TAS. |
| [Mixture of Horizons (MoH)](https://arxiv.org/abs/2511.19433) ([official repository](https://github.com/Timsty1/MixtureOfHorizons)) | 2025-11-24; ICML 2026 | Explicit short/long prediction-horizon branches inside a shared action transformer | Learned per-step, per-horizon scalar gate fuses full action vectors; consensus controls a scalar executable prefix | Yes, action-module training | No independent translation/rotation/gripper gate located | Cross-horizon consensus controls prefix validity, not group recomposition | `VERIFIED`: multi-horizon experts and learned fusion are occupied; source-age experts are distinct but not automatically novel. |
| [Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?](https://arxiv.org/abs/2608.02547) ([project PDF](https://action-chunking.github.io/static/action_chunking.pdf)) | 2026-08-03; arXiv preprint | Mechanistic comparison of chunked, delayed, and explicit/randomized-delay ensembles | Analysis and policy constructions, not an adaptive group router | Varies | No group routing located | No | `VERIFIED`: interpreting `a_t|o_t, a_t|o_{t-1},...` as temporal ensemble members is no longer a novelty claim. |
| [Adaptive Action Chunking (AAC)](https://openaccess.thecvf.com/content/CVPR2026/html/Liang_Adaptive_Action_Chunking_at_Inference-time_for_Vision-Language-Action_Models_CVPR_2026_paper.html) ([arXiv](https://arxiv.org/abs/2604.04161), [project](https://lance-lot.github.io/adaptive-chunking.github.io/)) | 2026-04-05; CVPR 2026 | Multiple stochastic chunks; translation, rotation, and gripper entropy cues | Component uncertainties are aggregated to choose one scalar execution prefix | No policy retraining | **Measures components, but does not independently execute group horizons** | No group recomposition | `VERIFIED`: action-semantic decomposition alone is not enough; AAC is a mandatory component-aware comparator. |
| [PACE](https://arxiv.org/abs/2606.00537) | 2026-05-30; arXiv v2 2026-07-29 | Low-speed valleys in the predicted motion define replanning boundaries | One training-free scalar prefix | No | No | No | `VERIFIED`: kinematic/smoothness scheduling is a strong simple baseline, not semantic phase ground truth. |
| [VLA Knows Its Limits / AutoHorizon](https://arxiv.org/abs/2602.21445) ([project/code](https://hatchetproject.github.io/autohorizon/)) | 2026-02-24; ECCV 2026 | Action-token self-attention estimates predictive limit | One scalar execution horizon | No | No | No | `VERIFIED`: “adaptive execution horizon” is occupied. |
| [DVAC](https://arxiv.org/abs/2606.03847) | 2026-06-02; arXiv preprint | Variation across final flow-denoising estimates | One calibrated stable prefix | No | No | No | `VERIFIED`: uncertainty-driven scalar replanning is occupied and flow-specific. |
| [DEHP](https://arxiv.org/abs/2606.11408) ([project](https://dehp-chunking.github.io/)) | 2026-06-09; arXiv preprint | Current state/chunk mapped to an execution-length distribution | One categorical scalar horizon | Yes, online RL; base policy frozen | No | No | `VERIFIED`: a learned horizon head is occupied. |
| [Bernoulli-Continuation Policy (BCP)](https://arxiv.org/abs/2608.03483) ([project](https://fleetfootwork.github.io/BCP/)) | 2026-08-04; arXiv preprint | Sequential continue/replan decisions over a fixed chunk | Ordinal scalar execution decision | Yes, online RL; base VLA frozen | No | No | `VERIFIED`: value-of-replanning at the operational level is crowded. |

### Verified citation ledger

- Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. “Learning
  Fine-Grained Bimanual Manipulation with Low-Cost Hardware.” RSS 2023;
  [arXiv:2304.13705](https://arxiv.org/abs/2304.13705).
- Qixiu Li, Yaobo Liang, Zeyu Wang, Lin Luo, Xi Chen, Mozheng Liao, Fangyun Wei,
  Yu Deng, Sicheng Xu, Yizhong Zhang, Xiaofan Wang, Bei Liu, Jianlong Fu,
  Jianmin Bao, Dong Chen, Yuanchun Shi, Jiaolong Yang, and Baining Guo.
  “CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition
  and Action in Robotic Manipulation.” 2024 preprint;
  [arXiv:2411.19650](https://arxiv.org/abs/2411.19650).
- Yueyang Weng, Xiaopeng Zhang, Yongjin Mu, Yingcong Zhu, and Yanjie Li.
  “Temporal Action Selection for Action Chunking.” 2025 preprint, revised 2026;
  [arXiv:2511.04421](https://arxiv.org/abs/2511.04421).
- Dong Jing, Gang Wang, Jiaqi Liu, Weiliang Tang, Zelong Sun, Yunchao Yao,
  Zhenyu Wei, Yunhui Liu, Zhiwu Lu, and Mingyu Ding. “Mixture of Horizons in
  Action Chunking.” ICML 2026; [arXiv:2511.19433](https://arxiv.org/abs/2511.19433).
- Filippo Lazzati, Kyle Stachowicz, William Chen, Alberto Maria Metelli, Andrew
  Wagenmaker, and Sergey Levine. “Why Does Action Chunking Improve Behavioral
  Cloning Performance in Robotic Control?” 2026 preprint;
  [arXiv:2608.02547](https://arxiv.org/abs/2608.02547).
- Yuanchang Liang, Xiaobo Wang, Kai Wang, Shuo Wang, Xiaojiang Peng, Haoyu Chen,
  David Kim Huat Chua, and Prahlad Vadakkepat. “Adaptive Action Chunking at
  Inference-time for Vision-Language-Action Models.” CVPR 2026;
  [arXiv:2604.04161](https://arxiv.org/abs/2604.04161).
- Junnan Nie, Jiayi Li, Jiachen Zhang, Junyi Lao, Chenghao Liu, Tianle Zhang,
  Liang Lin, and Songfang Huang. “PACE: Phase-Aware Chunk Execution for Robot
  Policies with Action Chunking.” 2026 preprint;
  [arXiv:2606.00537](https://arxiv.org/abs/2606.00537).
- Haoxuan Wang, Gengyu Zhang, Yan Yan, Ramana Rao Kompella, and Gaowen Liu. “VLA
  Knows Its Limits: Adaptive Execution Horizons for Robot Policies.” ECCV 2026;
  [arXiv:2602.21445](https://arxiv.org/abs/2602.21445).
- Xiangdong Feng, Yuxuan Cheng, Chen Shi, Boyao Han, Yuxuan Yan, Yitong Hong,
  Zhuotao Tian, and Li Jiang. “Denoising Tells When to Replan: Denoising-Variance
  Adaptive Chunking for Flow-Based Robot Policies.” 2026 preprint;
  [arXiv:2606.03847](https://arxiv.org/abs/2606.03847).
- Yuchi Zhao, Miroslav Bogdanovic, Arjun Sohal, Liyu Tao, Kourosh Darvish, Alán
  Aspuru-Guzik, Florian Shkurti, and Animesh Garg. “Dynamic Execution Horizon
  Prediction for Chunk-based Robot Policies.” 2026 preprint;
  [arXiv:2606.11408](https://arxiv.org/abs/2606.11408).
- Weichen Xu, Zhenhua Liu, Lin Luo, Yaobo Liang, Chengtang Yao, Qingyu Mei, Jian
  Cao, Xixin Cao, Xing Zhang, Jiaolong Yang, and Baining Guo. “Continue or
  Replan? Bernoulli-Continuation Policy Learning for Adaptive Horizon
  Execution.” 2026 preprint; [arXiv:2608.03483](https://arxiv.org/abs/2608.03483).

## Adjacent consistency and correction work

The following work makes a broad “consistency” claim non-distinctive even though
none was located doing the proposed component recomposition:

- [RTC](https://proceedings.neurips.cc/paper_files/paper/2025/hash/300ccb2187dedd4edcc07f7e76d8e553-Abstract-Conference.html)
  inpaints/regenerates around actions committed during asynchronous inference.
- [REMAC](https://arxiv.org/abs/2601.20130) uses masked/prefix-conditioned
  training for asynchronous continuation.
- [SEAM](https://arxiv.org/abs/2607.04609) uses the previous unexecuted tail as
  an analytic boundary-consistency reference.
- [POTR](https://arxiv.org/abs/2605.24433) constrains flow guidance for smoother
  chunk transitions.
- [FutureRTC](https://arxiv.org/abs/2607.24008) predicts execution-time context
  and includes a policy-consistency loss.

Accordingly, a proposed consistency score must be defined specifically as a
constraint on recomposed temporal-source groups and must be shown to detect
harm that scalar/source-consistent actions avoid. Generic use of the word
“consistency” is not novelty.

## Answers to the seven novelty questions

### 1. Does anyone already independently weight action groups across temporal predictions?

**UNKNOWN after bounded search; no exact method was located.** AAC computes
translation/rotation/gripper uncertainty but collapses it into one scalar
prefix. CogACT uses one scalar weight per complete source action. TAS selects one
complete candidate. MoH produces one horizon weight per action step and applies
it to the complete action prediction. Action-factorization papers found in the
broader search concern representation or sequential action selection, not
source-age weighting of overlapping action chunks.

### 2. Does anyone explicitly constrain cross-source group composition?

**UNKNOWN after bounded search; no exact method was located.** Existing
consistency mechanisms constrain full chunks, transitions, asynchronous
continuations, or horizon consensus. They do not establish that an arm slice
from one source and a gripper/rotation slice from another form a safe joint
action.

### 3. Does CogACT already do anything equivalent?

**No at the proposed group granularity; yes at the scalar core.** The paper and
released code assign a similarity-derived scalar to each complete historical
action and then take a weighted full-vector sum. Thus “adaptive similarity
temporal ensemble” is CogACT prior art. Replacing cosine with another full-action
distance is an incremental metric/geometry change, not a new ensemble paradigm.

### 4. Does TAS already support component/group-level selection?

**No such support was located.** Its method defines a candidate set of complete
actions, embeds each candidate, and samples/selects one candidate index. Its
coherence penalty compares the selected complete action with the natural
successor from the previous selected chunk. A learned group router would still
be close in machinery and would need a clear reason why group freedom and its
constraint are both necessary.

### 5. Does MoH cover the proposal through its gate?

**Not exactly.** MoH gates explicit prediction-horizon branches within one
current policy query. The documented gate is per action step and horizon, not
per actuator/action group. It does, however, occupy learned expert fusion and
cross-horizon consensus, so “mixture of temporal experts” is not enough as a
contribution statement.

### 6. Would the provisional GATE proposal look trivial?

**Yes unless necessity is demonstrated.** Without a direct failure analysis and
controlled ablation, reviewers can reasonably summarize it as “CogACT weights
plus actuator groups plus a heuristic fallback.” The project's own sparse
sanity audit currently gives them additional grounds: deployable independent
group similarities are worse than scalar semantic similarity, and the
validation-selected consistency gate stays extremely close to the scalar
fallback ([Gate-3A0 output](audit_outputs/gate3a0_sparse_group_consistency.json)).

### 7. What is the smallest defensible additional mechanism or setting?

The lowest-risk candidate is a **joint, semantics-aware temporal ensemble**:

1. one shared temporal-source weight vector preserves full-action source
   coupling;
2. the similarity uses normalized translation distance, SO(3) rotation
   distance, and discrete gripper-sign disagreement;
3. aggregation uses Euclidean translation averaging, a valid SO(3) mean, and a
   gripper sign vote/hysteresis rather than continuous gripper magnitude MSE;
4. group-specific residual freedom is optional and must be activated only if a
   dense, control-aligned oracle shows headroom that survives a predeclared
   consistency constraint.

This is `MARGINAL BUT DEFENSIBLE`, not fundamentally new. It becomes
publishable only if its semantic operators, rather than tuning or extra queries,
produce repeatable closed-loop gains against ACT temporal ensemble, released
CogACT AAE, newest-only, uniform, and a matched scalar heuristic on more than one
task and preferably more than one policy/benchmark.

## Honest contribution language

Acceptable provisional language:

> Building on ACT temporal ensembling and CogACT adaptive action aggregation, we
> study temporal aggregation when robot action components have heterogeneous
> control semantics. We preserve a shared temporal source distribution while
> using geometry- and event-aligned similarity and aggregation operators.

Conditional language, only if later evidence supports the group residual:

> We show that unconstrained cross-source component mixing can degrade control
> and introduce a source-consistency trust region that permits limited semantic
> group adaptation without abandoning the joint temporal mode.

Unsupported language to avoid:

- “the first adaptive temporal ensemble”;
- “the first dynamic horizon method”;
- “we discover temporal experts”;
- “group-specific timescales improve policy performance”;
- “cross-source inconsistency causes failure.”

## Literature-audit decision

| Candidate claim | Novelty judgment | Reason |
|---|---|---|
| Dynamic execution horizon | `NOT NOVEL` | Many peer-reviewed and preprint methods already adapt scalar prefixes. |
| Scalar similarity temporal ensemble | `TOO CLOSE TO PRIOR WORK` | Directly covered by CogACT AAE. |
| Learned full-action temporal selector | `TOO CLOSE TO PRIOR WORK` | Directly covered by TAS. |
| Explicit mixture of prediction horizons | `TOO CLOSE TO PRIOR WORK` | Directly covered by MoH. |
| Unconstrained group-wise cached-source mixing | `MARGINAL BUT DEFENSIBLE` in concept, unsupported empirically | Exact prior method not located, but it is an obvious extension and current project evidence is negative. |
| Consistency-gated group residual | `MARGINAL BUT DEFENSIBLE` if necessity and gain are demonstrated | The combination is distinguishable; current sparse deployable evidence does not support making it primary. |
| Joint control-semantics-aware temporal ensemble | `MARGINAL BUT DEFENSIBLE` if multi-setting rollout gains isolate the semantic operators | Smaller technical delta, lower implementation risk, and honest relationship to CogACT. |

## Limitations and update requirement

- Search engines and arXiv indexing cannot prove absence. The group/component
  search must be repeated immediately before submission.
- Several 2026 entries are preprints; venue status and method versions can
  change.
- Paper-reported rollout gains were not independently replicated.
- This review did not treat classical multi-rate control as equivalent to
  mixing components from cached learned-policy chunks; it remains relevant
  background if the project later claims actuator-rate generality.
- ICRA 2027 uses double-anonymous review and requires disclosure of AI-generated
  paper content in the acknowledgments; the [official call](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/)
  should be rechecked at submission time.
