# Command-discontinuity interpretation note

Status: **POST-HOC, OUTCOME-CONFOUNDED, NOT MECHANISM-IDENTIFYING**

This note does not change the canonical numerical outputs in
`command_discontinuity/`. It constrains their interpretation.

## Descriptive ordering

For the completed 140-block confirmation probes, the between-condition mean
first-difference ordering is descriptively

`coherent H16 > Fresh > A20G0`

for translation, rotation, and gripper controller-native command variation.
That ordering happens to match the three-condition success-rate ordering. With
only three intervention conditions and no causal identification of the D1
mediator, this must not be reported as a positive association or a reverse
mechanism finding.

The allowed conclusion is:

> An outcome-confounded between-condition command-discontinuity
> characterization did not provide identifiable support for a
> reduced-discontinuity explanation.

## Trajectory-composition and state-divergence confounding

The interventions generate different visited states, success frequencies,
episode lengths, and termination times. A poor rollout may become stationary
or locally repetitive, whereas a successful rollout may contain purposeful
motion and terminate earlier. Consequently, per-step D1 averages can differ
because trajectory composition differs, not because source switching caused
the difference. This is also why intervention-dependent episode length changes
absolute query counts even when per-executed-step query rates are matched.

Do not condition this analysis on successful episodes. Success is affected by
treatment, so restricting to successes would add outcome-selection bias rather
than solve state divergence.

## Prediction-offset smoothness confounding

A20G0 executes its arm from the distant fixed chunk offset
`A_(t-20)[20]`. If distant chunk indices are intrinsically smoother or more
regression-to-mean than near-term predictions, A20G0 may mechanically have
lower D1 without a source-coherence effect.

The existing B1 age-resolved output measures fresh-referenced same-target
cross-source disagreement as source age changes. It does **not** measure
within-chunk temporal D1 or intrinsic smoothness as a function of prediction
offset. It therefore does not directly resolve this threat, and no substitute
smoothness metric is introduced here.

## Switch-versus-same-source boundary

Fresh and sliding-source A20G0 change source query at every eligible step.
Their same-source class is structurally absent. Thus the confound-free
within-condition switch-versus-same-source contrast is
`STRUCTURALLY_UNAVAILABLE`, not zero and not a scientific outcome.

No protocol-defined positive prefix length is guaranteed to exist for every
trajectory independently of treatment-dependent success or termination. A
common-prefix sensitivity analysis requiring command differences is therefore
also `STRUCTURALLY_UNAVAILABLE`. Episodes must not be selected merely because
they survive to a chosen post-hoc K.

## Track-A gripper D1

- ARM4_GRIP32-H4 gripper D1: approximately -0.013040; behavioral gain
  +4.667 percentage points.
- ARM2_GRIP16-H2 gripper D1: approximately -0.000462; behavioral gain
  +5.778 percentage points.

The narrow observation is:

> The magnitude of reduced executed gripper command variation does not
> consistently track the behavioral gain across the two frozen operating
> points.

This does not causally rule out reduced gripper variation. D1 remains
rollout-level, state-divergent, and post-hoc. Moreover, longer gripper
commitment directly reduces how often a new gripper source is installed, so a
D1 reduction can partly follow mechanically from the executor contract.

## Paper role

This diagnostic is not a main mechanism result, main figure, independent
Results subsection, or preregistered mechanistic falsification. At most it may
support one compact Discussion paragraph and one supplementary
characterization subsection. Its role is a post-hoc diagnostic that failed to
provide an identifiable simple coherence explanation.

Do not claim that coherence is confirmed or falsified, or that larger D1
causes success or smaller D1 causes failure. B3 remains the final high-value
preplanned mechanism analysis. If B3 is null or non-discriminative, retain that
result and stop mechanism search.
