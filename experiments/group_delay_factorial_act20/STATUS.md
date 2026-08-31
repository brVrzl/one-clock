# Group-delay factorial ACT20

Status: complete. Protocol frozen before outcomes; semantic, pairing, shard,
and full primary-rollout gates passed.

Primary scope is the historical Gate-3C Object cohort: tasks 1–9, 14 states
per task, 126 paired blocks, and five methods, for 630 episodes. Task 0 is
secondary and is not part of the primary rollout.

The four dense conditions use one ACT query at every controller step and exact
same-target source selection at delay 20. `HARD_H16` queries only at physical
steps 0, 16, 32, … and executes the newest chunk at offset `t-q`.

Required gates before the primary rollout:

- semantic tests pass;
- task 1 states 20, 21, and 22 pairing smoke passes;
- `protocol.json` remains unchanged.

Completed pre-rollout gates:

- semantic tests: 10 passed;
- pairing smoke: task 1, states 20/21/22, all fixed prefixes exact through
  t=19, hard h16 divergence correctly begins at t=1;
- ACT repeated inference at identical processed input: exact equality.

Primary outcome rollout completed: 630/630 episodes across the five frozen
methods. All nine task shards passed the full source/offset/action validator.

Primary result:

- Fresh: 56/126;
- FO20: 81/126;
- Reverse20: 12/126;
- FullOld20: 47/126;
- hard h16: 88/126, observed query rate 0.06515.

Decision: `GROUP_DELAY_METHOD_STRONG`. FO20 was clearly positive versus Fresh,
Reverse20, and FullOld20, and was not clearly below hard h16 under the frozen
paired uncertainty rule. The hard h16 point estimate was higher by 7 net wins,
but its uncertainty interval included zero.

The authoritative outputs are `analysis.json`, `per_task.csv`,
`condition_shards/`, and `report.md`. Historical outcomes remain context only;
task 0 and newly claimed held-out tasks were not included.

The only planned follow-up after this factorial is the project-level decision
from `analysis.json` and `report.md`. No delay sweep, requery method, temporal
fusion sweep, selector, or held-out task is in scope here.
