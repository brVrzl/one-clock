# ACT h16 component-agreement development result

This is the second and final scheduler-development iteration on the four ACT development tasks. It changes only the maximum query interval from 8 to 16 and compares fixed h16 against adaptive h16 with the previously frozen arm-cosine and gripper-sign triggers. All 80 new episodes use initial-state IDs 10--19 and environment seeds 2000--2009.

| task | Fresh h1 | fixed h8 reference | fixed h16 | adaptive h16 | adaptive vs fixed h16 |
|---|---:|---:|---:|---:|---:|
| libero_10:task3 | 9/10 | 10/10 | 10/10 | 8/10 | 0/2 (net -2) |
| libero_goal:task1 | 10/10 | 10/10 | 10/10 | 10/10 | 0/0 |
| libero_object:task6 | 8/10 | 8/10 | 8/10 | 8/10 | 0/0 |
| libero_spatial:task2 | 10/10 | 10/10 | 10/10 | 10/10 | 0/0 |
| **pooled** | **37/40** | **38/40** | **38/40** | **36/40** | **0/2 (net -2)** |

## Query-efficiency result

| scheduler | successes | policy queries / environment steps | pooled query rate | mean executed source age |
|---|---:|---:|---:|---:|
| Fresh h1 | 37/40 | one per step | 1.000 | 0.000 |
| fixed h8 reference | 38/40 | 748 / 5,859 | 0.1277 | 3.470 |
| fixed h16 | **38/40** | **377 / 5,736** | **0.0657** | 7.342 |
| adaptive h16 | 36/40 | 737 / 6,443 | 0.1144 | 6.564 |

Fixed h16 and fixed h8 had the same pooled success count while fixed h16 used approximately half as many policy queries. Adaptive h16 was strictly dominated by fixed h16: it used 360 more queries and lost two paired episodes, both on LIBERO-10 task 3. Against Fresh, adaptive h16 had one adaptive-only and two Fresh-only successes (net -1).

## Frozen gate

| criterion | result | pass |
|---|---:|:---:|
| adaptive success at least 37/40 | 36/40 | **no** |
| adaptive paired net wins over fixed h16 at least +3/40 | -2/40 | **no** |
| no task has paired net loss of 2/10 versus Fresh | worst loss 1 | yes |
| adaptive not weakly dominated by fixed h8 | lower success and lower query rate | yes |

Advance decision: **FALSE**. Adaptive h16 must not enter held-out evaluation. Together with the failed adaptive-h8 gate, this ends development of the component-agreement scheduler on this cohort.

## Interpretation

The component triggers did not identify useful early refreshes. At h8 they reproduced fixed h8's outcomes while adding queries; at h16 they added queries and caused two additional failures. The positive development result is instead the fixed source-age cap: fixed h16 preserved 38/40 successes at a pooled query rate of 0.0657. This finding justifies a separately frozen blind evaluation of fixed execution horizons, but it is not evidence for adaptive scheduling.

## Reference-array correction

The declared Fresh and fixed-h8 references come from the completed `cdta_dev` and `dynamic_horizon_dev` result files. Before full h16 launch, copied fixed-h8 arrays were corrected from those source files. A duplicate-key-context edit also altered two copied Fresh arrays; after completion they were restored verbatim from the declared `cdta_dev` source. This clerical correction did not use h16 outcomes and does not change the advance decision: the primary adaptive-success and adaptive-versus-fixed gates both fail.
