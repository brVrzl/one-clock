# Counterfactual tournament status

## 2026-08-24 first sprint checkpoint

### Completed

* ManiSkill 3.0.1 is now the fast causal-development backend. PickCube-v1 and
  StackCube-v1 completed the 10-seed Gate-0 screen with exact state restore,
  isolated suffix branches, zero-branch controls, and zero invalid branches.
* RoboTwin/SAPIEN remains the intended scalable benchmark and recovery work is
  tracked separately in `research/robotwin_recovery_track.md`.
* Exact SAPIEN physics restore passed in
  `experiments/counterfactual_tournament/stack_audit.json`: a packed state was
  restored after an actor pose change, with a 52-byte physics snapshot and
  matching pose before/after restore.
* The generic fork primitive passed its branch-isolation unit test.
* The RoboTwin runner records project/upstream SHAs, scripted segment snapshots,
  perturbation outcomes, and raw physics states. It has an expert-only mode for
  bounded seed screening and an MPLIB-only import path that avoids editing the
  user-dirty upstream checkout.

### ManiSkill Gate-0 evidence

`research/maniskill_gate0_report.md` records the final counts, distributions,
heuristic correlations, qualitative examples, and limitations. The signal
survives on the two validated tasks. PegInsertionSide-v1 initialized, but its
fallback expert did not reliably grasp the peg and is not included in the
causal decision. No policy result or track ranking is complete; the live
matrix intentionally remains `NA`.

### Engineering blockers

1. The local RoboTwin checkout contained only can/basket object assets. The
   requested three-task screen therefore could not start; the official asset
   download was attempted but failed because the configured SOCKS proxy lacks
   the required client dependency, and direct HTTPS was reset.
2. With the one locally complete task, MPLIB screw planning was fast but failed
   to produce an expert for the tested seed. MPLIB RRT reached task execution
   but required minutes and failed during expert planning for the tested seeds;
   one bounded screening process exited without a final artifact, consistent
   with an unstable heavy simulator path.
3. These failures are infrastructure observations only. They are not negative
   policy results and are not entered into the result matrix.

### Conditional track ranking before numerical evidence

1. **BranchBC — KEEP AS PRIMARY CANDIDATE.** It directly tests the requested
   equal-budget data-selection hypothesis and has the clearest causal contrast.
2. **ContrastBC — KEEP AS BACKUP.** It reuses paired fork labels and is the
   cheapest auxiliary objective, but it is closest to generic failure-aware BC.
3. **CriticalBC — SCREEN ONLY.** It is easy to implement, but weighting can
   exploit existing samples without improving state coverage, which weakens the
   closed-loop mechanism.
4. **GeoAux — SCREEN ONLY.** It is a useful privileged-training baseline, but
   the deployment mechanism is less distinctive and overlaps geometric policy
   supervision literature.

This remains a prioritization for the next policy gate, not a LOCK decision.
The next runnable experiment is matched UniformBC/CriticalBC/ContrastBC on the
validated ManiSkill fork data, followed by equal-budget BranchBC only if the
policy signal is positive.
