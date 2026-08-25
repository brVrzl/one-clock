# FragilityAug action-alignment audit

Status: retired as a policy-training mechanism; retained as provenance for
the previous matched ACT gate.

The former augmentation builder restored a counterfactual simulator state
and then wrote the original action suffix beginning at the perturbation
timestep.  In `pd_ee_pose`, the first six action values are an absolute EEF
pose target in the robot base frame.  A state perturbation changes the state
but leaves that target unchanged.  Consequently, the resulting pair is

```text
perturbed state -> action selected for the unperturbed state
```

and is not assumed to be a valid imitation label.  This is especially clear
for robot-joint perturbations, where the EEF pose is displaced while the
target action remains identical, and for object perturbations near contact,
where the correct approach or grasp action can change.

The old H5 files under `act_data/augmented/` are therefore not used in the
recovery comparison.  The new generator writes a complete bounded corrective
trajectory, checks that the bridge reduces EEF position error, replays the
expert suffix only after its future waypoint, and retains the branch only
when the full simulator rollout succeeds.  Invalid branches and teleport
violations are reported separately.

The old 0/5 ACT result is not evidence against fragility.  It combines this
action-alignment problem with an undertrained ACT regime, so the new policy
gate starts by measuring an original-demonstration learning curve.
