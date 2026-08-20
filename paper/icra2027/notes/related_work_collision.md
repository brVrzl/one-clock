# Related-work and terminology collision audit

Audit date: 2026-08-20. All entries below were checked against the primary paper
PDF or official proceedings page, not a search-engine summary. Section and
equation references refer to the cited version. For 2026 manuscripts, venue
status is not inferred beyond what the primary source states.

## Terminology result

| Concept | Established wording found in primary sources | Usage in this paper |
|---|---|---|
| Predicting multiple controls | **action chunking**, **action sequence**, **action chunk**, **prediction horizon** | Use these terms. ACT explicitly predicts the next `k` target joint positions; Diffusion Policy distinguishes prediction and action-execution horizons. |
| Number of predicted actions followed before a new plan | **action execution horizon** (Diffusion Policy); **execution horizon** (RTC, AutoHorizon, PACE, DVAC, DEHP) | Use **execution horizon** and define it once. |
| Following a predicted prefix without new observations | **open-loop execution** or **open-loop prefix** | Use these conventional terms. |
| Obtaining a new observation and producing a replacement plan | **replanning**, **re-querying the policy**, **receding-horizon control** | Use `replanning` for the control interpretation and `policy query` for measured policy-inference demand. |
| Varying the executed prefix | **adaptive action chunking**, **adaptive chunk execution**, **dynamic execution horizon prediction** | Use these only when describing the corresponding prior work. The present experiments are static. |
| Computing a new chunk concurrently with execution | **asynchronous action-chunk execution**, **real-time chunking** | Reserve for RTC/FutureRTC; the current executor is synchronous at query boundaries. |
| Transition between predicted chunks | **chunk boundary**, **chunk transition** | Use for synchronized replacement; explain explicitly that group-specific execution can create multiple group boundaries. |
| Different horizons for physical components | No established term found in the audited set | Use the transparent descriptive extension **group-specific execution horizons** (occasionally **group-wise execution horizons**). Do not present it as established terminology and introduce no acronym. |

Avoid as technical terms: `structured continuation`, `temporal urgency`, and
`multi-clock scheduling`. They do not appear as standard names in the audited
primary literature.

## Core and directly colliding work

| Work / primary citation | Base policy type | Problem addressed | What is adapted | Execution decision | Decision granularity | Scalar vs other | Static / dynamic | Training-free / learned | Signal | Benchmarks in the paper | Relationship to this project |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ACT — Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, RSS 2023, arXiv:2304.13705 | CVAE Transformer behavioral cloning | Compounding error and temporally coherent imitation | Predicts chunks; temporal ensemble optionally blends overlapping predictions | Naive mode executes a complete `k`-step chunk; reported ACT queries each step and temporally ensembles | Full action vector | One synchronized sequence | Fixed chunk length; temporal ensemble every step | Learned base policy; fixed inference rule | Overlapping predictions and exponential weights | Six real ALOHA tasks plus simulation analyses | Establishes action chunking. Current experiments use a frozen ACT checkpoint but explicitly disable temporal ensembling to isolate execution horizons. |
| Diffusion Policy — Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, arXiv:2303.04137 | Conditional diffusion policy | Multimodal visuomotor behavior with stable receding-horizon control | Prediction horizon and action-execution horizon are deployment parameters | Execute `T_a` actions, then re-observe/replan | Full action vector | One scalar `T_a` | Fixed | Learned policy; fixed execution | Fixed hyperparameter | Robomimic, Push-T, real manipulation | Supplies the conventional distinction between prediction and execution horizons and the receding-horizon framing. |
| RTC — Black, Galliker, Levine, *Real-Time Execution of Action Chunking Flow Policies*, NeurIPS 2025, arXiv:2506.07339 | Diffusion/flow VLA or chunk policy | Inference latency and pauses/discontinuities | Timing and constrained generation of the next chunk | Freeze the unavoidable prefix while the current chunk runs; inpaint the remainder | Full action vector at every time index | One horizon/timing constraint shared by all dimensions | Dynamic with measured inference delay | Training-free | Inference completion time, previous chunk, flow/diffusion inpainting | Kinetix and real bimanual tasks | Direct on asynchronous execution, but it does not assign independent execution schedules to physical action groups. |
| AutoHorizon — Wang et al., *VLA Knows Its Limits: Adaptive Execution Horizons for Robot Policies*, arXiv:2602.21445 | Flow-based VLAs | Task/phase-dependent horizon selection | Executed prefix length | Select one execution horizon for the current full chunk | Chunk as a whole | Scalar | Dynamic | Training-free test-time rule | VLA cross-/self-attention statistics | LIBERO, RoboTwin, real robot | Closely collides on adaptive horizons. The audited formulation selects one prefix length for all action components. |
| AAC — Liang et al., *Adaptive Action Chunking at Inference-time for Vision-Language-Action Models*, CVPR 2026, arXiv:2604.04161 | Diffusion-action VLA | Reactivity/consistency tradeoff | Inference-time chunk size | Sample candidate chunks, compute entropy by future index, choose one prefix length `h*`, execute all dimensions for `h*` steps | Signal is component-aware: translation, rotation, and gripper entropies are computed separately, then summed/averaged before selection | One scalar `h*` | Dynamic | Training-free relative to the trained VLA; multiple action samples required | Action entropy across sampled chunks | RoboCasa, LIBERO/LIBERO-Pro, three real tasks | Important granularity collision: it uses physical-component uncertainty signals, but aggregates them to one synchronized full-vector chunk-size decision rather than separate component schedules. |
| PACE — Nie et al., *Phase-Aware Chunk Execution for Robot Policies with Action Chunking*, arXiv:2606.00537 | Black-box chunk policy / VLA | Fixed horizons fail across manipulation phases | Execution horizon | Detect low-speed candidate boundaries and choose the earliest accepted boundary | Builds speed profiles for individual arms, pools candidate boundaries, then chooses one boundary | One scalar horizon for the full action vector | Dynamic | Training-free | Predicted kinematic speed and phase transitions | RoboTwin2.0; real ALOHA and Franka | Closest physical-group collision. Arm-specific signals are retained during boundary proposal, but all action dimensions still replan at the same selected boundary. |
| DVAC — Feng et al., *Denoising Tells When to Replan*, arXiv:2606.03847 | Diffusion/flow action policy | Detect unreliable suffixes during denoising | Executed prefix length | Replan at the first future index whose denoising variance exceeds a threshold | Variance is summed over action dimensions at each future index | One scalar prefix | Dynamic | Training-free | Denoising trajectory variance | LIBERO, RoboTwin, CALVIN, real robot | Direct adaptive-replanning collision; dimension evidence is aggregated before one full-vector decision. |
| DEHP — Zhao et al., *Dynamic Execution Horizon Prediction for Chunk-based Robot Policies*, arXiv:2606.11408 | Frozen diffusion policy plus horizon head | Learn phase-dependent open-loop commitment | Execution horizon | Categorical head selects `h` and the controller executes the first `h` full actions | Chunk as a whole | One scalar categorical horizon | Dynamic | Learned online with reinforcement learning; base policy frozen | Observation plus flattened predicted chunk | FurnitureBench and IsaacLab insertion tasks | Very close in frozen-policy motivation, but learns one synchronized horizon rather than group-specific horizons. |
| SparkVLA — Lei et al., *Stop-Aware Hierarchical VLA with Adaptive Action Chunking for Long-Horizon Manipulation*, arXiv:2608.16172 | Hierarchical VLA with action decoder and stop selector | Long-horizon subtask progression and adaptive boundaries | Full-chunk stop/boundary decision | Learned ordinal selector ranks stop versus candidate prefix lengths | Chunk as a whole | One scalar boundary | Dynamic | Learned selector | Multimodal history, subtask context, boundary annotations | RoboCerebra, LIBERO, real robot | Latest close collision found before submission drafting; still makes one synchronized boundary decision. It weakens broad novelty claims about adaptive chunking, not the audited group-granularity distinction. |

## Additional execution-adjacent collisions found during search

| Work / primary citation | Decision and signal | Granularity / learning | Relationship |
|---|---|---|---|
| SGAC — So et al., *Improving Generative Behavior Cloning via Self-Guidance and Adaptive Chunking*, NeurIPS 2025, arXiv:2510.12392 | Queries each step; compares the queued next full action with a fresh full action by cosine similarity. If similar it appends a fresh tail action; otherwise it replaces the queue. | Binary retain/replace behavior over the full action vector; training-free execution rule combined with diffusion self-guidance. | Directly relevant to selective replanning, but not independent physical-group execution. |
| ACH — Shin et al., *Adaptive Action Chunking via Multi-Chunk Q Value Estimation*, arXiv:2605.10044 | A causal Transformer estimates values for all prefix lengths and selects/samples a chunk length. | Learned offline-to-online RL; one scalar prefix length. | Adaptive chunk-size selection in RL, not a frozen-policy group-specific executor. |
| AQC — Gireesh, Ju, Wang, *Adaptive Q-Chunking for Offline-to-Online Reinforcement Learning*, arXiv:2605.05544 | Per-horizon critics score candidate prefixes via a normalized advantage; the selected chunk runs open-loop. | Learned offline-to-online RL; one scalar commitment length. | Reinforces that adaptive scalar commitment is crowded; no independent action-group schedules. |
| VLA-Corrector — Pan et al., *VLA-Corrector: An Online Framework for Robotic Manipulation with Pre-trained VLA Models*, arXiv:2607.01804 | Compares predicted and observed visual feature evolution; persistent deviation truncates the current chunk and triggers replanning/correction. | Event trigger applies to the full chunk; learned/fitted monitoring and online correction around a frozen VLA. | Closely related event-triggered replanning, still synchronized across action dimensions. |
| SEAM — Zhan et al., *SEAM: Smooth Execution of Action-Chunked Motion for Vision-Language-Action Policies*, arXiv:2607.04609 | Uses the unexecuted tail of the preceding chunk to steer the next flow-generated chunk and smooth transitions. | Full-vector chunk-boundary consistency; training-free. | Particularly relevant to the stated limitation that mixed source generations can be temporally inconsistent. It addresses consistency, not independent horizon choice. |
| DREAM-Chunk — Chen et al., *DREAM-Chunk: Reactive Action Chunking with Latent World Model*, arXiv:2606.18589 | Samples candidate full chunks and uses predicted latent futures versus observed rollout to select behavior. | Full action chunks; auxiliary learned latent world model/test-time scaling. | Improves reactivity under stochasticity without per-group execution schedules. |
| FutureRTC — Jiang et al., *FutureRTC: Real-Time Robot Execution with Anticipatory-Conditioned Action Chunking*, arXiv:2607.24008 | Predicts execution-time observations/states for asynchronous policy calls and aligns the resulting chunk. | Full-vector asynchronous chunk transitions; learned prediction modules. | Latency/misalignment collision, not horizon granularity. |
| TempoWAM — Ye et al., *Rethink Before You Execute: Adaptive Execution for World Action Models*, arXiv:2608.09492 | A progress monitor judges whether the current world-action-model chunk is advancing the task and triggers replanning. | One full-chunk keep/replan decision; learned monitor with online calibration. | Very recent dynamic scalar execution work; no independent physical-group decisions reported. |

## Adjacent concepts for the proposed estimator

| Concept / primary citation | Decision or signal | Granularity | Relationship |
|---|---|---|---|
| Adaptive Computation Time — Graves, *Adaptive Computation Time for Recurrent Neural Networks*, arXiv:1603.08983 | Learns how many recurrent updates to spend before emitting an output. | Variable internal computation, not robot action groups | Conceptual analogy for input-dependent commitment; it is not an execution-horizon or uncertainty estimator for chunked control. |
| Event-triggered scheduling — Tabuada, *Event-Triggered Real-Time Scheduling of Stabilizing Control Tasks*, IEEE TAC 2007 | Replaces periodic control-task updates with a state-dependent trigger while retaining feedback semantics. | Trigger for a control task | Provides control-theoretic context for event-triggered replanning; it does not supply group-wise prediction reliability. |

The proposed $R_g(k)$ target is intentionally narrower than these adjacent ideas:
it is a group-wise validity probability for a predicted action subsequence.  Its
candidate labels (future-action consistency, fresh-query disagreement, or
calibrated predictive uncertainty) remain future design choices and are not
reported as implemented methods.

## Novelty consequence

The defensible statement is narrow: **among the primary works audited here, the
execution decision ultimately remains one synchronized prefix/boundary decision
for the action vector, even when its signal is computed over physical components
or arms.** AAC and PACE prevent a stronger claim that prior work ignores action
structure entirely. No audited method independently lets arm and gripper retain
or accept predicted components on different schedules.

This is not evidence for a universal priority claim. The manuscript should avoid
`first`, `novel`, and broad claims that all prior work is scalar unless explicitly
qualified by this audit. The present evidence supports a controlled formulation
and empirical characterization, not a dynamic scheduling algorithm.
