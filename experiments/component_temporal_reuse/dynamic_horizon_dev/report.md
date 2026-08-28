# ACT component-agreement dynamic-horizon development result

Frozen 4-task, 10-state panel. The three schedulers used explicit initial-state IDs 10--19 and environment seeds 2000--2009. The valid Fresh reference is the 37/40 query-every-step condition from the completed CDTA development panel on the same task/state/seed pairs.

| task | Fresh (h1) | fixed h4 | fixed h8 | adaptive h8 | adaptive query rate |
|---|---:|---:|---:|---:|---:|
| libero_10:task3 | 9/10 | 9/10 | 10/10 | 10/10 | 0.192 |
| libero_goal:task1 | 10/10 | 10/10 | 10/10 | 10/10 | 0.144 |
| libero_object:task6 | 8/10 | 8/10 | 8/10 | 8/10 | 0.179 |
| libero_spatial:task2 | 10/10 | 10/10 | 10/10 | 10/10 | 0.150 |
| **pooled** | **37/40** | **37/40** | **38/40** | **38/40** | **0.175** |

## Query-efficiency result

| scheduler | successes | policy queries / environment steps | pooled query rate | reduction from query-every-step |
|---|---:|---:|---:|---:|
| Fresh (h1) | 37/40 | one per step | 1.000 | 0.0% |
| fixed h4 | 37/40 | 1,570 / 6,223 | 0.252 | 74.8% |
| fixed h8 | 38/40 | 748 / 5,859 | 0.128 | 87.2% |
| adaptive h8 | 38/40 | 1,099 / 6,266 | 0.175 | 82.5% |

Adaptive h8 and fixed h8 had identical success/failure outcomes on all 40 paired episodes (0/0 discordant pairs). Adaptive h8 issued 351 more queries than fixed h8, a 46.9% increase, without changing an episode outcome. Relative to Fresh, adaptive h8 had two adaptive-only successes and one Fresh-only success (paired net +1).

The adaptive scheduler recorded 663 maximum-age triggers, 300 arm-cosine triggers, 151 gripper-sign triggers, and 80 insufficient-history triggers. Trigger counts can overlap at a query step and therefore do not sum to the total query count.

## Frozen gate

| criterion | result | pass |
|---|---:|:---:|
| adaptive success at least 35/40 | 38/40 | yes |
| pooled query rate at most 0.60 | 0.175 | yes |
| no task has paired net loss of 2/10 versus Fresh | worst loss 0 | yes |
| adaptive exceeds the better fixed scheduler by at least 2/40 | 38 - 38 = 0 | **no** |

Advance decision: **FALSE**. The planned held-out evaluation must not start for this adaptive-h8 rule.

## Interpretation

This panel establishes that querying the ACT policy every eight environment steps can preserve performance on these four development tasks while substantially reducing policy calls. It does not establish a benefit from component-agreement triggering: the adaptive rule reproduced fixed h8's exact closed-loop outcomes while using more queries. Fixed h8 is therefore the relevant control for any subsequent scheduler, and this adaptive-h8 result cannot support a method claim by itself.
