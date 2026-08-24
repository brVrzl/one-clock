# Related-work audit for the counterfactual tournament

This is a screening note, not a claim that the tournament method is novel.
The central question is which exact supervision or data-selection mechanism is
different and whether it changes closed-loop success under equal data and
compute.

| Work | Closest overlap | What the sprint must distinguish |
|---|---|---|
| *Perfect Demo Makes Poor Teacher* (arXiv:2606.15587) | Critical-motion segments, deliberate/resampled data, recovery coverage | Their critical segments are motion/alignment motivated. BranchBC selects branch starts using measured downstream counterfactual success loss and holds added trajectories fixed across selection rules. |
| ISR (arXiv:2606.22907) | Velocity/acceleration information weighting and resampling | ISR is offline trajectory standardization from kinematic information. It is a prespecified Gate-0 heuristic baseline, not the causal selector. |
| S2I (ICRA 2025, arXiv:2409.19917) | Segment selection and optimization for mixed-quality demonstrations | S2I selects segments using learned quality/contrastive signals; BranchBC uses simulator interventions and downstream success labels at matched branch-data budgets. |
| RESample (arXiv:2510.17640) | OOD state augmentation and recovery data | RESample uses offline RL/action-value guidance for exploratory sampling. BranchBC uses no RL/Q-learning in the first sprint and directly tests the location of branch collection. |
| Dream2Fix (arXiv:2603.13528) | Counterfactual failures and actionable recovery data | Dream2Fix synthesizes photorealistic failures from real demonstrations with a world model. This sprint uses exact simulator state forks, measured success labels, and a lightweight BC family. |
| Geometry-aware Policy Imitation (ICLR 2026) | Geometry/relational structure in imitation | GeoAux is a simple privileged-state auxiliary baseline; it is not the primary contribution and must not be conflated with GPI's distance-field policy. |
| REIM / recovery imitation | Failure/recovery branches and closed-loop recovery | REIM includes a failure detector and recovery policy at deployment. The sprint's primary test is earlier data selection: whether where recovery data is collected matters under equal budget. |

## Candidate defensible statement if BranchBC wins

> We introduce counterfactual-criticality-guided branch collection: exact
> simulator forks estimate the downstream success loss caused by local
> perturbations, and a fixed-budget branch collector allocates recovery
> demonstrations to the states with the largest measured loss. Unlike motion
> resampling or value-guided exploration, the selection signal is a causal
> intervention label from the task simulator, and the central evaluation holds
> additional trajectories and optimization steps fixed across selectors.

This statement is conditional. It is not justified unless Gate 2 shows a
repeatable closed-loop advantage over random and the nearest motion/event
heuristic on at least two screened tasks.

## Sources

* [Perfect Demo Makes Poor Teacher](https://arxiv.org/abs/2606.15587)
* [ISR](https://arxiv.org/abs/2606.22907)
* [S2I](https://arxiv.org/abs/2409.19917)
* [RESample](https://arxiv.org/abs/2510.17640)
* [Dream2Fix](https://arxiv.org/abs/2603.13528)
* [Geometry-aware Policy Imitation](https://proceedings.iclr.cc/paper_files/paper/2026/hash/bf235a1d6780afd979f2f81676f43413-Abstract-Conference.html)
* [REIM repository and evaluation protocol](https://github.com/CC-robotics/REIM)
