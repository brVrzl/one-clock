# Internal hostile-review audit

Review date: 2026-08-24. Review target: the first complete six-page ICRA draft
and its four rendered figures. This audit treats the developmental factorial as
hypothesis-generating and the untouched-state study as the sole confirmatory
experiment.

## Reviewer A: action chunking / ACT / CogACT expert

- **Strongest reason to accept:** The paper isolates a real execution variable
  that temporal ensembling and cached-action selection normally collapse. The
  matched-query design is unusually clean: all methods receive current
  observations and issue the same full ACT query every controller step, while
  only the source generation supplying each component changes. The confirmed
  gain remains positive against newest, full old20, age-exponential, and tuned
  CogACT under paired and task-cluster uncertainty.
- **Strongest reason to reject:** FO20 is a fixed, system-selected executor
  evaluated with one ACT checkpoint. A reviewer can reasonably suspect that the
  effect is a checkpoint-specific interaction between ACT's training data,
  action normalization, and binary gripper representation rather than a useful
  action-chunking principle.
- **Novelty concern:** TAS already selects among temporal contexts, SEAM can
  guide selected dimensions, TRACT combines temporal routing with arm-specific
  correction, and DAM-VLA specializes arm and gripper. The paper is novel only
  under the narrow distinction that it assigns observation-time generations
  independently to components of one frozen jointly predicted action. Any
  broader claim would collide with these works.
- **Experimental concern:** The comparison is strong within one execution
  contract, but there is no second checkpoint, policy family, or task suite.
  There is also no delay sweep, so the paper cannot explain whether 20 ticks is
  special. The manuscript correctly avoids an optimality claim.
- **Claim-overreach concern:** "Retained intent," temporal intent, and generic
  mixed-source incoherence would all exceed the evidence. The active draft
  avoids these claims and states that the developmental coherence contrast was
  unresolved.
- **Highest-value revision:** Run one preregistered second-checkpoint,
  second-suite confirmation under the identical five-method matched-query
  contract. Do not add a learned adaptive method before establishing this
  generalization.

## Reviewer B: general ICRA manipulation reviewer

- **Strongest reason to accept:** The intervention is easy to understand, costs
  no training, changes no query budget, and improves binary task success by
  14.3--21.4 percentage points over four relevant controls on 126 paired blocks.
  Figure 1 prevents the common mistaken reading that the method holds the
  gripper for one second.
- **Strongest reason to reject:** The empirical scope is narrow for a method
  paper: one simulated benchmark suite, one checkpoint, one arm/gripper split,
  and no real robot. The six-page draft reads as a strong controlled finding but
  not yet as a generally established manipulation method.
- **Novelty concern:** The executor is simple enough that reviewers may call it
  an ablation rather than a method. The response is that the contribution is the
  controlled source-age intervention and confirmatory evidence, not algorithmic
  complexity, but that positioning must stay prominent.
- **Experimental concern:** LIBERO Object tasks share embodiment, controller,
  action representation, and training distribution. Nine tasks provide task
  clustering but do not vary these system-level factors.
- **Claim-overreach concern:** Phrases such as "for robot policies" can sound
  universal when the evidence is one ACT/LIBERO system. The title is general,
  but the abstract, results, discussion, and limitations repeatedly bind the
  result to the audited system.
- **Highest-value revision:** Add exactly one independent benchmark/checkpoint
  confirmation, preserving the same source gap and baseline contract so that a
  positive result has an unambiguous interpretation.

## Reviewer C: statistical and reproducibility skeptic

- **Strongest reason to accept:** The primary inference unit is correctly the
  paired task-state block rather than controller frames. The cohort identity was
  audited without reading outcomes; method order and episode seeds were frozen;
  all scheduled episodes completed without exclusion or rerun. Effect sizes,
  paired-block intervals, task-cluster intervals, and leave-one-task-out checks
  are all reported.
- **Strongest reason to reject:** There are only nine independent task clusters,
  so task-cluster bootstrap tails are necessarily coarse and generalization
  beyond these tasks remains uncertain. Four contrast-wise intervals are not a
  family-wise error correction, although the preregistered decision requires
  all four comparisons to pass the stable-positive rule.
- **Novelty concern:** Statistical rigor cannot rescue a post-hoc hypothesis.
  The draft therefore must keep 62/100 in the developmental record and never
  combine it with the 126 confirmatory blocks as though all 226 were one
  replication set. The active draft keeps them separate.
- **Experimental concern:** Binary success obscures partial progress and offers
  limited mechanism information. Logged kinematic diagnostics occur on
  treatment-dependent trajectories and cannot be used as causal mediators.
- **Claim-overreach concern:** Positive task-cluster intervals do not establish
  a population of policies, embodiments, or benchmarks. They support only a
  task-spanning aggregate for the evaluated checkpoint and suite.
- **Highest-value revision:** Release the paired block-level success table,
  frozen schedule, state IDs, analysis script, and checkpoint/config
  provenance with the submission. The manuscript already defines the unit,
  state set, seeds, interval construction, and no-exclusion policy; the public
  artifact should make those statements executable.

## Overall verdict

**Borderline / weak accept.** The controlled intervention, untouched-state
confirmation, relevant full-action baselines, and honest limitations make the
paper scientifically credible for ICRA. The dominant rejection risk is external
validity, not internal validity or numerical support. The paper should not grow
an adaptive method or mechanism story before resolving that risk.

## Generalization decision

**GENERALIZATION-HIGH-VALUE**

A second policy/benchmark experiment is not logically necessary for the current
within-system claim to be credible, so the paper is not classified
GENERALIZATION-REQUIRED. It is nevertheless the single highest-value remaining
experiment because one positive, preregistered replication would directly
address the most likely ICRA rejection: that FO20 is peculiar to one Object-suite
checkpoint.

## One recommended final experiment

No experiment was launched by this manuscript task.

- **Policy:** one independently trained LeRobot ACT checkpoint with the same
  100-step, seven-dimensional output contract, trained only on LIBERO Spatial.
- **Benchmark/tasks:** all ten LIBERO Spatial tasks; select ten official initial
  states per task outcome-blind before any method outcome is generated.
- **Baseline set:** FO20, newest, full old20, age-exponential $\beta=.03$, and
  CogACT $\alpha=.3$, all with one query per surviving 20-Hz controller step.
- **Episodes:** $10$ tasks $\times$ $10$ paired states $\times$ $5$ methods
  $=500$ episodes.
- **Decision criterion:** preregister the same four FO20-minus-comparator
  contrasts and the same stable-positive rule. Call the result a generalization
  only if every aggregate contrast is positive, both its paired-block and
  task-cluster 95% interval lower bounds exceed zero, and every leave-one-task-out
  estimate is positive. Otherwise report the suite as non-generalizing or
  inconclusive without tuning the delay.
- **Required assets:** a frozen independently trained LIBERO Spatial ACT
  checkpoint with chunk length at least 21; its preprocessing, postprocessing,
  and normalization files; the LIBERO Spatial environment and demonstrations;
  official initial states; and the existing matched-query cache, schedule, and
  paired-analysis runner on the RTX 5080.

This is the smallest experiment that changes both checkpoint and benchmark
while preserving the exact source-age intervention. It does not test policy-
family, VLA, dexterous, bimanual, or real-robot generalization.
