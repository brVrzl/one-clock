# Literature Claims Re-audit

Audit date: 2026-08-21  
Search boundary: material public by 2026-08-21  
Status vocabulary: `VERIFIED`, `PARTIALLY VERIFIED`, `CONTRADICTED`, `UNKNOWN`

## Method

The search covered action chunking, temporal ensembling, adaptive execution,
cached/overlapping predictions, action correction, asynchronous execution,
multi-horizon models, and multi-rate control. Venue records, arXiv version
histories, author project pages, and author repositories were preferred over
surveys or project notes. A paper's own empirical claims are recorded as
*paper-reported*, not independently replicated facts. Search terms included
`action chunking temporal ensemble`, `adaptive execution horizon`, `dynamic
action chunk`, `temporal action selection`, `mixture of horizons`, `replan
action chunk`, `stale action correction`, and `multi-rate robot control`.

The literature review skill's recommended search-and-screen discipline was
used. Its schematic-image recommendation was not used: no scientific schematic
skill is installed, the requested artifact is a comparison matrix, and sending
project details to an unrelated image service would add no evidentiary value.

## Bibliographic and method verification

| Paper | Exact authors | First public / current status | Method and training regime | Backbone / evaluation | Code status checked 2026-08-21 |
|---|---|---|---|---|---|
| [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)](https://arxiv.org/abs/2304.13705) | Tony Z. Zhao; Vikash Kumar; Sergey Levine; Chelsea Finn | 2023-04-26; RSS 2023 | CVAE Transformer predicts an action chunk; optional exponential temporal ensemble combines overlapping predictions for the same physical time | ACT; six real bimanual tasks | [Official repository](https://github.com/tonyzhaozh/aloha) located |
| [CogACT](https://arxiv.org/abs/2411.19650) | Qixiu Li; Yaobo Liang; Zeyu Wang; Lin Luo; Xi Chen; Mozheng Liao; Fangyun Wei; Yu Deng; Sicheng Xu; Yizhong Zhang; Xiaofan Wang; Bei Liu; Jianlong Fu; Jianmin Bao; Dong Chen; Yuanchun Shi; Jiaolong Yang; Baining Guo | 2024-11-29; arXiv preprint | VLM plus diffusion action transformer; paper includes an adaptive action ensemble that similarity-weights current and prior overlapping predictions | CogACT VLA; simulation and real robots | [Project/code/models](https://cogact.github.io/) located |
| [Real-Time Execution of Action Chunking Flow Policies (RTC)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/300ccb2187dedd4edcc07f7e76d8e553-Abstract-Conference.html) | Kevin Black; Manuel Galliker; Sergey Levine | 2025; NeurIPS 2025 | Training-free asynchronous inpainting: freeze actions that will execute and regenerate the suffix while the previous chunk runs | Diffusion/flow VLAs; Kinetix and six real tasks | Author implementation linked from paper/project |
| [Improving Generative Behavior Cloning via Self-Guidance and Adaptive Chunking (SGAC)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/79ce24f9e8d3c4ff5919240eac78a782-Abstract-Conference.html) | Junhyuk So; Chiwoong Lee; Shinyoung Lee; Jungseul Ok; Eunhyeok Park | 2025; NeurIPS 2025 | Self-guidance plus action-similarity-based adaptive updating of generated chunks | Generative/diffusion behavior cloning; simulated and real tasks | No repository verified in this audit |
| [Leave No Observation Behind: Real-time Correction for VLA Action Chunks (A2C2)](https://arxiv.org/abs/2509.23224) | Kohei Sendai; Maxime Alvarez; Tatsuya Matsushima; Yutaka Matsuo; Yusuke Iwasawa | 2025-09-27; arXiv preprint / ICLR 2026 submission record found | Lightweight per-control-step residual correction from the newest observation, base action, chunk index, and base features; base VLA frozen | Kinetix and LIBERO Spatial | Author-linked implementation was not verified from the arXiv record |
| [Temporal Action Selection for Action Chunking (TAS)](https://arxiv.org/abs/2511.04421) | Yueyang Weng; Xiaopeng Zhang; Yongjin Mu; Yingcong Zhu; Yanjie Li | 2025-11-06; arXiv v2, 2026-06-02 | Caches predictions from multiple source times and learns a lightweight selector over candidate actions; selector is trained with online PPO/reward and may feed residual RL | Multiple chunked base policies; simulation and physical robots | No official public repository located |
| [Mixture of Horizons in Action Chunking (MoH)](https://arxiv.org/abs/2511.19433) | Dong Jing; Gang Wang; Jiaqi Liu; Weiliang Tang; Zelong Sun; Yunchao Yao; Zhenyu Wei; Yunhui Liu; Zhiwu Lu; Mingyu Ding | 2025-11-24; accepted ICML 2026 | Shared action transformer processes explicit horizon segments in parallel; a learned linear gate fuses them; cross-horizon consensus enables adaptive inference | pi0, pi0.5, and one-step regression; LIBERO and real tasks | [Official repository](https://github.com/Timsty1/MixtureOfHorizons) located |
| [REMAC: Real-Time Robot Execution with Masked Action Chunking](https://arxiv.org/abs/2601.20130) | Haoxuan Wang; Gengyu Zhang; Yan Yan; Yuzhang Shang; Ramana Rao Kompella; Gaowen Liu | 2026-01-27; ICLR 2026 | Retrains masked/prefix-conditioned chunk generation for asynchronous execution and boundary continuity | Flow/diffusion VLA settings | [Project and public code link](https://remac-async.github.io/) located |
| [VLA Knows Its Limits: Adaptive Execution Horizons for Robot Policies (AutoHorizon)](https://arxiv.org/abs/2602.21445) | Haoxuan Wang; Gengyu Zhang; Yan Yan; Ramana Rao Kompella; Gaowen Liu | 2026-02-24; ECCV 2026 | Training-free scalar execution horizon inferred from VLA action self-attention | Flow VLAs; simulation and real manipulation | [Official repository](https://github.com/hatchetProject/AutoHorizon) located |
| [Adaptive Action Chunking at Inference-time for VLA Models (AAC)](https://openaccess.thecvf.com/content/CVPR2026/html/Liang_Adaptive_Action_Chunking_at_Inference-time_for_Vision-Language-Action_Models_CVPR_2026_paper.html) | Yuanchang Liang; Xiaobo Wang; Kai Wang; Shuo Wang; Xiaojiang Peng; Haoyu Chen; David Kim Huat Chua; Prahlad Vadakkepat | 2026-04-05; CVPR 2026, pp. 20802–20811 | Samples multiple chunks, computes translation/rotation/gripper entropy, aggregates those components, and selects one scalar prefix length; training-free | VLAs; RoboCasa and real tasks | [Official project/code](https://lance-lot.github.io/adaptive-chunking.github.io/) located |
| [Dynamic Execution Commitment of VLA Models (A3)](https://arxiv.org/abs/2605.11567) | Feng Chen; Xianghui Wang; Yuxuan Chen; Boying Li; Yefei He; Zeyu Zhang; Yicheng Wu | 2026-05-12; arXiv preprint | Samples chunks, scores trajectory consensus, conditionally re-decodes, and accepts the longest prefix passing sequential verification | pi0, pi0.5, GR00T; LIBERO, ManiSkill, MetaWorld, real robot | [Official repository](https://github.com/INCEPTIONwang/A3) located |
| [PACE: Phase-Aware Chunk Execution](https://arxiv.org/abs/2606.00537) | Junnan Nie; Jiayi Li; Jiachen Zhang; Junyi Lao; Chenghao Liu; Tianle Zhang; Liang Lin; Songfang Huang | 2026-05-30; arXiv preprint | Training-free low-speed-valley heuristic on the predicted arm trajectory; selects one scalar prefix | RoboTwin 2.0; ALOHA and Franka | No official public repository located |
| [Denoising Tells When to Replan (DVAC)](https://arxiv.org/abs/2606.03847) | Xiangdong Feng; Yuxuan Cheng; Chen Shi; Boyao Han; Yuxuan Yan; Yitong Hong; Zhuotao Tian; Li Jiang | 2026-06-02; arXiv preprint | Final-denoising clean-action variance and rolling calibration select one scalar stable prefix; training-free | Flow policies; LIBERO, RoboTwin, CALVIN, real robot | No official public repository located |
| [Dynamic Execution Horizon Prediction (DEHP)](https://arxiv.org/abs/2606.11408) | Yuchi Zhao; Miroslav Bogdanovic; Arjun Sohal; Liyu Tao; Kourosh Darvish; Alán Aspuru-Guzik; Florian Shkurti; Animesh Garg | 2026-06-09; arXiv preprint | Frozen chunk policy plus lightweight categorical horizon branch trained with online RL | High-precision and long-horizon simulation tasks | [Project page](https://dehp-chunking.github.io/) located; public code not verified |
| [SEAM](https://arxiv.org/abs/2607.04609) | Dijia Zhan; Xuemiao Xu; Jinyi Li; Jie Tang | 2026-07-06; arXiv preprint | Uses the previous unexecuted tail as a consistency reference and applies analytic flow correction; training-free | pi0.5 on LIBERO-10 | No official public repository verified |
| [Continue or Replan? Bernoulli-Continuation Policy (BCP)](https://arxiv.org/abs/2608.03483) | Weichen Xu; Zhenhua Liu; Lin Luo; Yaobo Liang; Chengtang Yao; Qingyu Mei; Jian Cao; Xixin Cao; Xing Zhang; Jiaolong Yang; Baining Guo | 2026-08-04; arXiv preprint | Frozen base VLA plus a 16.4M-parameter sequential continue/replan head trained by online RL with success/efficiency reward | LingBot-VLA on RoboTwin 2.0; pi0.5 on LIBERO/PRO; real robot | [Project page](https://fleetfootwork.github.io/BCP/) located; downloadable code URL not verified |
| [Why Does Action Chunking Improve Behavioral Cloning Performance?](https://arxiv.org/abs/2608.02547) | Filippo Lazzati; Kyle Stachowicz; William Chen; Alberto Maria Metelli; Andrew Wagenmaker; Sergey Levine | 2026-08-03; arXiv preprint | Mechanistic experiments compare chunking, delayed single-action policies, and explicit/randomized-delay ensembles | Simulation and real-world control | No official public repository verified |

## Prior literature-derived claims

| Claim | Source actually inspected | Actual paper evidence | Audit result |
|---|---|---|---|
| ACT temporal aggregation is an exponential ensemble of overlapping predictions for the same action time. | ACT paper and official code | Yes. Each current action can be predicted by chunks issued at several earlier times; newer predictions receive exponentially larger weights. This is a **temporal ensemble**, not adaptive execution-horizon selection. | VERIFIED |
| ACT's prediction horizon, execution horizon, query interval, and temporal age are interchangeable. | ACT paper | They are distinct: chunk length is a training/prediction quantity; naive deployment may execute a prefix/full chunk; overlapping predictions have source age; query frequency controls when new chunks appear. | CONTRADICTED |
| Dynamic execution horizon is an open novelty by itself. | SGAC, AutoHorizon, AAC, A3, PACE, DVAC, DEHP, BCP, MoH | At least eight distinct 2025–2026 methods already select state-dependent scalar execution lengths, using similarity, attention, entropy, consensus, kinematics, denoising variance, supervised/RL heads, or ordinal RL. | CONTRADICTED |
| Cached predictions from different observation times have not been treated as selectable temporal experts. | ACT, CogACT, TAS, 2026 action-chunking analysis | ACT ensembles them; CogACT similarity-weights them; TAS explicitly caches and selects among them; Lazzati et al. interpret the learned temporal relationships as implicit ensembling. | CONTRADICTED |
| A generic learned temporal-age router would be clearly distinct from prior art. | TAS and CogACT | TAS is a learned selector over cached multi-time action candidates; CogACT is a similarity-weighted overlapping-action ensemble. A generic router would collide directly. | CONTRADICTED |
| Explicitly learning multiple prediction horizons is unexplored. | MoH | MoH trains explicit horizon-specific segments in a shared action transformer and learns a linear fusion gate. | CONTRADICTED |
| AutoHorizon learns a group-specific scheduler. | AutoHorizon paper | It infers a scalar horizon from action self-attention; no independent arm/gripper scheduling was found. | CONTRADICTED |
| AAC's translation, rotation, and gripper scores imply separately executed groups. | AAC paper and supplement | AAC computes component entropies but aggregates them to choose one scalar chunk length. | CONTRADICTED |
| PACE is a learned semantic phase model. | PACE paper | PACE is a training-free kinematic low-speed-boundary heuristic; “phase” is inferred through chunk motion structure, not a learned semantic segmentation label. | CONTRADICTED |
| DVAC directly measures closed-loop control value. | DVAC paper | DVAC uses within-denoising variation as a proxy and reports closed-loop results; the signal itself is not value-of-information or causal control value. | CONTRADICTED |
| DEHP and BCP train on offline action MSE labels. | DEHP and BCP papers | Both use online RL/trajectory outcomes while freezing the base policy; BCP explicitly includes replanning efficiency. | CONTRADICTED |
| RTC is a dynamic-horizon policy. | RTC paper | RTC solves asynchronous latency by inpainting a new chunk around actions guaranteed to execute. It is not primarily a state-dependent execution-horizon selector. | CONTRADICTED |
| Action correction and stale-action mitigation are missing from the closest literature. | A2C2, RTC, REMAC, SEAM | Several methods use latest observations, masked regeneration, previous tails, or asynchronous inpainting to correct or reconcile stale chunks. | CONTRADICTED |
| Existing adaptive-horizon work proves that shorter horizons cause better control near contact. | Adaptive-horizon papers | Papers report correlations and closed-loop gains for their own selectors, but method-specific results do not establish the universal causal mechanism. Phase annotations, uncertainty proxies, query compute, and policy family differ. | PARTIALLY VERIFIED |
| Multi-rate robotics already establishes learned independent arm/gripper chunk execution. | Classical multi-rate control literature, including Lee and Xu, *A Multiple Rate Control Scheme for a Robot Manipulator* (1993) | Classical work separates controller/dynamics rates, often fast feedback and slower model/feedforward updates. It does not establish independently aged action slices from a learned chunk policy. | CONTRADICTED |
| No paper explains why action chunking works. | Lazzati et al. 2026 | A direct 2026 study rejects several common explanations in its tested settings and identifies non-Markovian expressivity, reduced compounding error, and implicit temporal ensembling. Its conclusions are paper-reported and not universal proof. | CONTRADICTED |
| Offline imitation error is the accepted target for adaptive execution. | TAS, DEHP, BCP, A3, PACE, DVAC, AutoHorizon | The closest methods mainly optimize trajectory reward or use policy-internal/kinematic proxies and validate with rollouts. Offline MSE alone is not the field's accepted evidence of control improvement. | CONTRADICTED |

## Direct novelty conclusions

1. **“Dynamic execution horizon” alone is not defensible ICRA 2027 novelty.**
   This is a hard negative conclusion, not a wording problem. The literature
   already includes peer-reviewed SGAC (NeurIPS 2025), AAC (CVPR 2026), and MoH
   (ICML 2026), plus multiple closely overlapping 2026 preprints.
2. **“Temporal expert routing” alone is also not defensible.** TAS is the closest
   collision; CogACT and ACT temporal ensembling are additional mandatory
   baselines.
3. **Group-specific temporal routing is a narrower remaining distinction, not a
   demonstrated contribution.** This audit found no exact prior method that
   independently selects left/right arm/gripper source age from cached ACT
   predictions, but the project's own evidence has not yet shown that this
   freedom improves closed-loop control or survives a dimension-balanced loss.
4. **Value of freshness is crowded at the operational level.** BCP, DEHP,
   AutoHorizon, PACE, DVAC, A3, and A2C2 all estimate or act on reasons to obtain
   fresher information, even when they do not use that phrase. A new method must
   define a distinct estimand and show causal or policy-return relevance.

## Limitations of this literature audit

- Venue status was taken from official proceedings where available and otherwise
  from the current arXiv record; preprint performance claims were not replicated.
- “No code located” means no author repository was found in targeted searches by
  the cutoff, not proof that private or later code does not exist.
- The 2026 literature is moving rapidly. A final submission audit must repeat the
  search immediately before ICRA submission.
