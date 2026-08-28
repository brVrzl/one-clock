# CDTA-16 ACT development result

Frozen 4-task, 10-state, 5-method panel. All methods used explicit initial-state IDs 10--19 and identical environment seeds.

| task | Fresh | ACT ensemble | CogACT-style | matched shared | CDTA-16 | CDTA vs matched | CDTA vs Fresh |
|---|---:|---:|---:|---:|---:|---:|---:|
| libero_10:task3 | 9/10 | 5/10 | 3/10 | 8/10 | 9/10 | 1/0 (net +1) | 0/0 (net +0) |
| libero_goal:task1 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 0/0 (net +0) | 0/0 (net +0) |
| libero_object:task6 | 8/10 | 6/10 | 6/10 | 8/10 | 8/10 | 0/0 (net +0) | 1/1 (net +0) |
| libero_spatial:task2 | 10/10 | 10/10 | 9/10 | 10/10 | 10/10 | 0/0 (net +0) | 0/0 (net +0) |

## Pooled gate

Successes: Fresh 37/40; ACT ensemble 31/40; CogACT-style 28/40; matched shared 36/40; CDTA-16 37/40.

CDTA versus matched shared: 1/0 discordant pairs (net +1). CDTA versus Fresh: 1/1 (net +0).

Advance decision: **FALSE**. The component-decoupling primary gate failed because the paired net advantage over the matched shared control was +1, below the frozen +3 threshold. The Fresh safeguards and task-direction safeguard passed. The predeclared 800-episode CDTA blind panel should not start.

This development result supports the age/window control, which nearly matched Fresh, but it does not establish a closed-loop benefit from component decoupling.
