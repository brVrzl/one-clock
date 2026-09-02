# Prior-art boundary audit

Status: **PASS**, frozen before any Track-A query-allocation outcome.

## Boundary

Generic degradation from more frequent replanning is not a novelty claim here. ACT introduced temporal aggregation over overlapping action chunks to smooth action predictions, while retaining per-step observation feedback. BID explicitly frames action horizon as a trade-off: longer chunks preserve temporal consistency but use less recent state information, whereas receding-horizon execution can create jitter or switch among incompatible modes. RTC likewise starts from the consistency/reactivity problem at action-chunk boundaries and addresses it through asynchronous chunk generation and inpainting.

Primary sources checked on 2026-09-02:

- ACT, *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, arXiv:2304.13705: <https://arxiv.org/abs/2304.13705>
- Official ACT evaluation implementation (`temporal_agg`, `query_frequency=1`, coefficient `0.01`): <https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py>
- BID, *Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling*, arXiv:2408.17355: <https://arxiv.org/abs/2408.17355>
- RTC, NeurIPS 2025 paper page: <https://papers.nips.cc/paper_files/paper/2025/hash/300ccb2187dedd4edcc07f7e76d8e553-Abstract-Conference.html>
- Physical Intelligence RTC overview: <https://www.pi.website/research/real_time_chunking>

## Prospective question retained

Track A does not ask whether frequent replanning can hurt. It asks whether that known consistency penalty is non-uniform across action components and whether, under the same periodic policy-query schedule, preserving gripper commitment while refreshing the arm mitigates the uniform high-frequency replanning penalty. Track B separately asks whether same-target prediction instability is localized to the gripper channel.

Allowed terms are `policy-query rate`, `policy-query budget`, and `replanning frequency`. Policy calls are not labeled FLOPs, compute scaling, or “4x compute”; wall-clock is reported separately.
