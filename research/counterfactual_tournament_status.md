# Counterfactual tournament status

## 2026-08-24 first sprint checkpoint

### Completed

* Environment audit selected RoboTwin/SAPIEN as the cheapest locally available
  stack. The active RoboTwin venv has Torch, SAPIEN, and Gymnasium; the active
  environment does not have ManiSkill, MuJoCo, robosuite, LIBERO, or LeRobot.
* Exact SAPIEN physics restore passed in
  `experiments/counterfactual_tournament/stack_audit.json`: a packed state was
  restored after an actor pose change, with a 52-byte physics snapshot and
  matching pose before/after restore.
* The generic fork primitive passed its branch-isolation unit test.
* The RoboTwin runner records project/upstream SHAs, scripted segment snapshots,
  perturbation outcomes, and raw physics states. It has an expert-only mode for
  bounded seed screening and an MPLIB-only import path that avoids editing the
  user-dirty upstream checkout.

### Not yet evidence

No Gate-0 criticality curve, policy result, closed-loop success rate, or track
ranking is complete. The live matrix intentionally remains `NA`.

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

This is a prioritization for the next runnable environment, not a result or a
LOCK decision. No direction can be locked until the equal-budget Gate 2 is run.
