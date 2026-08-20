# Open questions for the next stage

These are research questions and TODOs, not components of the current method.
No dynamic scheduler or future result is implied.

## Static sufficiency

1. **Is one universal static group pair sufficient?** The current common-set
   post-hoc analysis selects `(4,16)` as the best universal pair and in every
   leave-one-task-out split. Re-evaluate this conclusion on denser grids and
   outside LIBERO Object before investing in a dynamic selector.
2. **How large is the per-task static oracle gap under a fair common grid?** The
   current .045 macro gap is retrospective, coarse, and based on ten related
   object-pick tasks. Quantify uncertainty and compare against a held-out static
   selection rule.
3. **Does a budget-constrained universal pair dominate a global schedule?** The
   best current universal global and group-specific points have different query
   rates. Construct Pareto-matched comparisons rather than choosing only by
   success.

## Offline persistence findings

4. **What does the phase-conditioned error signal predict?** Gate-1 and Gate-2A
   establish group- and phase-dependent offline prediction-error structure, but
   they use demonstrated future actions and no success labels.  Test calibration
   and held-out prediction of a predeclared validity event before connecting the
   signal to execution.
5. **Can a reliability curve remain useful when it is nonmonotonic?** The phase
   audit does not support a universal late-is-shorter rule.  Compare survival,
   threshold-crossing, and calibrated risk summaries without selecting the
   representation after seeing rollout outcomes.

## Within-task variation

6. **Do preferred horizons vary with manipulation phase within a task?** Offline
   error profiles vary with phase, but static episode success cannot answer this.
   A future study needs an online observable proxy and an evaluation that avoids
   selecting phases after seeing outcomes.
7. **What is the dynamic-readiness gap?** Separate the gain available from
   task-level static selection from any additional gain available only through
   within-episode adaptation. Do not infer the latter from per-task static
   oracle numbers.

## Online selection signal

8. **Which observable signal predicts when a group should refresh?** Candidate
   families include action-distribution uncertainty, disagreement across fresh
   predictions, denoising dynamics, kinematic phase, source-chunk age, and
   observation/prediction mismatch. Each requires a prespecified evaluation and
   latency accounting.
9. **Can selection remain training-free, or is a learned decision rule needed?**
   Compare a transparent fixed rule, a training-free adaptive baseline, and a
   learned selector without changing the frozen base policy.
10. **How should policy-query cost be charged?** A query generates a full chunk
   even when only one group accepts it. Future objectives should report both
   environment success and full-policy query rate/latency.

## Synchronization and consistency

11. **When must groups refresh together?** Independent expiry can combine arm and
   gripper commands from different observations and group-local positions.
   Test explicit synchronization events near contact, grasp transitions, or
   when source-age disparity exceeds a threshold.
12. **How harmful is mixed-generation composition?** Compare independent
    retention with full-vector replacement, tail-conditioned smoothing, and
    constrained blending. Measure discontinuity and task outcomes; do not assume
    smoothness causes success.
13. **What partitions are physically meaningful?** Beyond arm/gripper, study
    translation/rotation, multiple arms, individual fingers, or actuator groups
    only where the action semantics and policy outputs support those partitions.

## Additional sensing and embodiment

14. **Can force/contact sensing improve group refresh decisions?** Treat this as
    a future hypothesis. First define the observable, latency, contact labels,
    and a vision/proprioception-only control.
15. **Do results extend to dexterous hands and cross-embodiment control?** The
    current one-dimensional gripper result is insufficient evidence. Dexterous
    action groups may have stronger coupling and different safety constraints.

## Stronger validation

16. **Does the phenomenon recur beyond LIBERO Object?** Planned directions may
    include other LIBERO suites, RoboTwin, RoboDojo, and a real robot, but none
    is a completed result in this draft.
17. **Does it hold across policy families and checkpoints?** Repeat with at
    least one diffusion/flow policy and independently trained checkpoints.
18. **What statistical design is adequate?** Increase paired initial states or
    seeds, pre-register primary comparisons, report effect sizes and confidence
    intervals, and control family-wise exploration when testing many horizons.

## Manuscript TODOs

- Replace oracle-style cross-task summaries with a deployable held-out static or
  dynamic selection protocol if and when such evidence exists.
- Decide whether exploratory trace metrics belong in a supplement; they are not
  causal evidence and are omitted from the current main paper.
- Re-run the literature collision search immediately before submission because
  adaptive execution work is appearing rapidly in 2026.
- Add RoboTwin/RoboDojo/real-robot sections only after auditable artifacts exist.
