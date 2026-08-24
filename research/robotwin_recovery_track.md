# RoboTwin recovery track

RoboTwin remains the intended scalable paper benchmark. ManiSkill is only the
fast causal-development sandbox.

Current status inherited from the pre-ManiSkill scaffold:

- the existing SAPIEN state save/restore audit passed;
- required object assets remain incomplete locally, with previous download
  attempts blocked by the network path;
- expert planning is slow and unstable on the blocked asset set;
- no RoboTwin research-method implementation is being added while the causal
  mechanism is screened in ManiSkill.

Lightweight recovery actions to continue in parallel:

1. inventory missing assets and identify tasks whose assets are already local;
2. retry asset repair/download only when the network path is available;
3. profile expert planner success rate, latency, and variance on reliable
   tasks;
4. prefer replaying cached expert demonstrations over replanning;
5. test intermediate-state restore followed by the original cached action
   suffix;
6. cache every usable expert trajectory with seed, task, planner metadata,
   state identifiers, and raw logs.

The ManiSkill Gate 0 result does not change this target-benchmark decision. It
only determines whether the causal weighting/selection mechanism is worth
porting after RoboTwin replay infrastructure is repaired.
