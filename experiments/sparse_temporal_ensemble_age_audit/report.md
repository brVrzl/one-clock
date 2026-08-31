# Repaired ACT h16 dense-equivalent temporal-ensemble result

## Result

All methods queried ACT at physical steps `0,16,32,...` and used the same frozen four-task development cohort, states 10--19, seeds 2000--2009, checkpoints, runtime, preprocessing, and success definition. Every condition/state used a freshly constructed environment under the same method-independent seed.

| Method | Success /40 | Object | Spatial | Goal | L10 | Query rate |
|---|---:|---:|---:|---:|---:|---:|
| hard h16 | **32** | 8/10 | 7/10 | 10/10 | 7/10 | 0.06506 |
| candidate-index TE h16 | 24 | 4/10 | 6/10 | 9/10 | 5/10 | 0.06462 |
| dense-equivalent TE h16 | 23 | 4/10 | 5/10 | 9/10 | 5/10 | 0.06468 |

The small query-rate differences arise only from differing episode termination lengths. The scheduled query cadence is identical.

## Paired comparisons

“First-only” means the first named method succeeded and its reference failed.

| Contrast | First-only | Reference-only | Net wins | Exact McNemar p |
|---|---:|---:|---:|---:|
| candidate-index TE vs hard | 3 | 11 | −8 | 0.05737 |
| dense-equivalent TE vs hard | 2 | 11 | −9 | 0.02246 |
| dense-equivalent TE vs candidate-index TE | 0 | 1 | −1 | 1.00000 |

Dense-equivalent TE lost relative to hard on every task: object −4, spatial −2, goal −1, and L10 −2 successes. The result is not driven by one task. Candidate-index TE was likewise nonpositive on all four tasks.

## Execution statistics

| Method | Mean queries/episode | Mean candidate count | Mean weighted source age | Mean successful completion steps | Mean query latency |
|---|---:|---:|---:|---:|---:|
| hard h16 | 11.450 | 4.768 | 7.365 | 127.500 | 77.27 ms |
| candidate-index TE h16 | 14.150 | 5.055 | 40.190 | 127.458 | 75.10 ms |
| dense-equivalent TE h16 | 14.175 | 5.056 | 45.672 | 121.130 | 68.05 ms |

Completion length is summarized only among successful episodes and should not be compared as an unconditional speed metric. Query latency differences are runtime noise: all methods use the same ACT query path.

## Pairing and semantic validation

The repaired task-10 audit compared hard, candidate-index TE, and dense-equivalent TE on states 10, 11, and 16. Both cameras, processed inputs, initial chunk, actions t=0--15, and resulting simulator states were exactly equal across methods. Every full task shard also passed a validator requiring:

- fresh environment construction for every condition/state;
- identical initial simulator state, fixture body positions, and low-dimensional observation;
- exact t=0--15 action equality across all three methods;
- queries only at multiples of 16;
- exact same-target alignment `q+offset=t`;
- candidate-index weights `exp(-0.01*i)` oldest to newest;
- dense-equivalent weights `exp(-0.01*(q_j-q_0))` oldest to newest.

The older candidate-index result of 20/40 remains archived as the result of its implemented variant, but it is not the definitive paired aggregate because its task-10 shard was contaminated. The repaired candidate-index result is 24/40 and independently confirms harm relative to hard h16.

## Interpretation and decision

Rescaling the positive ACT kernel by sparse physical query separation did not recover performance. Dense-equivalent TE was essentially tied with candidate-index TE at the paired outcome level, with only one discordant state between them, and both were clearly below hard execution at the same query cadence.

Within these four exposed development tasks, sparse querying and temporal averaging are in tension for this ACT policy even when the dense ACT kernel is correctly subsampled. The earlier degradation therefore cannot be attributed solely to nearly uniform candidate-index weighting.

Decision: **`DENSE_EQ_TE_HARMFUL`**.

Repaired candidate-index TE remains harmful: **yes**.
