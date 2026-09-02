# Related-work fact-check input

Checked against primary paper/proceedings records on 2026-09-02. This is a factual memo, not manuscript prose.

## ACT

- **Record:** Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, Robotics: Science and Systems 2023 ([official proceedings](https://roboticsproceedings.org/rss19/p016.html)).
- **Relevant contribution:** ACT predicts action chunks to reduce the effective decision horizon. Its dense temporal ensemble queries the policy at every controller step and combines overlapping predictions for the same physical timestep with exponentially decaying weights.
- **Canonical implementation audited locally:** `temporal_agg=true`, `query_frequency=1`, coefficient `0.01`, continuous aggregation across all action dimensions, followed by the frozen environment action contract. The Track-A `TE_DENSE` canary matched the canonical aggregation to approximately `2.38e-7` maximum absolute error.
- **Ablation boundary:** ACT includes chunk-size ablations and reports that action chunking and temporal ensembling matter. It therefore already establishes that execution/aggregation choices affect policy behavior; neither chunk sensitivity nor dense re-querying is itself a `one-clock` novelty.

## BID

- **Record:** Liu et al., *Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling*, ICLR 2025 ([official proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0d78dd998f7b9ac79604d47a2d79bb0d-Abstract-Conference.html)).
- **Relevant contribution:** BID analyzes how chunking better preserves demonstration temporal dependencies while reducing responsiveness to unexpected states. It introduces guided test-time sampling using backward coherence and forward contrast.
- **Boundary for `one-clock`:** The consistency/reactivity trade-off and the possibility that an execution horizon changes performance are prior art. `one-clock` must not claim discovery that frequent replanning can hurt.
- **Distinction:** BID selects among sampled chunks using whole-chunk criteria. The current prospective question instead holds the periodic policy-query schedule fixed and assigns execution commitment differently to arm and gripper channels.

## RTC

- **Record:** Black et al., *Real-Time Execution of Action Chunking Flow Policies*, NeurIPS 2025 Main Conference ([official proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/300ccb2187dedd4edcc07f7e76d8e553-Abstract-Conference.html)).
- **Relevant contribution:** RTC treats asynchronous action-chunk execution under inference latency as an inpainting problem, freezing actions that must execute while generating a consistent continuation.
- **Boundary for `one-clock`:** Chunk-boundary discontinuity, consistency, reactivity, and latency-aware execution are established problems. Track A is not an RTC reproduction and no RTC pivot is authorized.
- **Distinction:** The current work measures same-target component assignment and tests fixed component-resolved periodic schedules; RTC addresses asynchronous whole-policy generation and continuity under latency.

## PACE

- **Record:** Nie et al., *PACE: Phase-Aware Chunk Execution for Robot Policies with Action Chunking*, arXiv:2606.00537v2, 2026-07-29. The arXiv record explicitly labels it a preprint and does not name an archival venue ([arXiv](https://arxiv.org/abs/2606.00537)).
- **Relevant contribution:** PACE chooses an execution horizon online from low-speed valleys in a predicted chunk. It reports task-dependent, non-monotonic fixed-horizon performance and analyzes selected horizons across manipulation phases.
- **Boundary for `one-clock`:** Fixed-horizon sensitivity, non-monotonicity, and adaptive replanning boundaries are not novel claims available to this project. No PACE reproduction or adaptive horizon method is authorized here.
- **Distinction:** PACE selects a shared per-query horizon from predicted kinematics. `one-clock` asks whether component-level temporal sensitivity and component-resolved fixed execution have measurable consequences.

## AutoHorizon / VLA Knows Its Limits

- **Record:** Wang et al., *VLA Knows Its Limits: Adaptive Execution Horizons for Robot Policies*, arXiv:2602.21445v2, 2026-06-20 ([arXiv](https://arxiv.org/abs/2602.21445)). The arXiv author comment says “ECCV 2026”; this memo records that author-supplied venue status rather than claiming an independently checked proceedings publication.
- **Relevant contribution:** The paper reports peaked performance versus execution horizon in flow-based VLAs, studies vision-language and action self-attention, and uses attention as a proxy for a chunk’s predictive limit to select a dynamic horizon.
- **Boundary for `one-clock`:** Execution-horizon sensitivity and prediction-limit motivation are existing ideas. The three `one-clock` mechanism quantities must remain separate from attention-based confidence: demonstration persistence, frozen-policy future-action forecast error, and same-target cross-source disagreement.
- **Distinction:** AutoHorizon adapts a shared execution horizon using model attention. The frozen `one-clock` experiments do not inspect attention and do not adapt horizons online.

## ARP

- **Record:** Zhang et al., *Autoregressive Action Sequence Learning for Robotic Manipulation*, arXiv:2410.03132v5, 2025-03-25 ([arXiv](https://arxiv.org/abs/2410.03132)). The arXiv author comment says “RA-L 2025”; this memo does not substitute that comment for a separately verified IEEE record.
- **Relevant contribution:** ARP’s Chunking Causal Transformer predicts variable numbers of tokens per autoregressive step. It explicitly supports heterogeneous action-sequence formats and manually chosen chunk sizes for different action-token types, including high-level versus low-level actions.
- **Important novelty constraint:** Broad statements that no prior work uses different chunk sizes for different action types would be false. ARP is directly relevant prior art for heterogeneous temporal grouping.
- **Distinction requiring precise wording:** ARP changes a learned autoregressive sequence representation and manually specifies token-group chunk sizes. The present work localizes same-target temporal effects within a frozen 6D relative-arm plus 1D gripper action vector, then tests component-resolved **execution commitment** at a matched policy-query schedule without retraining. This is a narrower distinction and should be described factually, not as universal precedence.

## Defensible novelty boundary

Do not claim:

- that frequent replanning can hurt;
- that action-chunk consistency and reactivity trade off;
- that fixed execution horizons matter;
- that no prior policy has heterogeneous action-token chunk sizes.

The potential contribution supported by the frozen program is narrower:

1. same-physical-target assignment of source age separately to arm and gripper channels;
2. complete fixed-source temporal-sensitivity curves over the preregistered `d` grid;
3. separate measurement of demonstration persistence, future-action forecast error, and cross-source prediction disagreement;
4. a prospective matched-policy-query test of component-resolved fixed execution, if Track A supports it.
