# RTX 5080 cross-generation offline composition report

Audit date: 2026-08-24 (Asia/Shanghai). This is the frozen post-Gate-3A2
exploratory offline audit defined before composition losses were computed.
No Thor Gate-3B outcome was inspected or used.

## Result

| Condition | `L_sem` | translation | rotation normalized | rotation rad | gripper sign |
|---|---:|---:|---:|---:|---:|
| FF | 0.7834019361 | 0.5957820292 | 1.1296224520 | 0.0617667300 | 0.3076001092 |
| OO | 0.7271885115 | 0.5066667976 | 1.0987722486 | 0.0602820201 | 0.2740024419 |
| FO | 0.7786022694 | 0.5957820292 | 1.1296224520 | 0.0617667300 | 0.2740024419 |
| OF | 0.7319881783 | 0.5066667976 | 1.0987722486 | 0.0602820201 | 0.3076001092 |

Across 82 episodes and 10654 eligible targets, mean `C_offline` is `-1.4893235696191125e-17`. The paired episode-bootstrap 95% CI is `[-3.9263985017231144e-17, 8.1235831070133401e-18]`; the macro-task estimate is `-1.5959455978986624e-17` with task-cluster 95% CI `[-3.6082248300317589e-17, 2.0816681711721684e-18]`.

The targetwise 2x2 identity residual is at most `1.78e-15`, below the frozen `1e-12` tolerance. The result is therefore an exact offline null by construction: the additive Gate-3A1 metric has no arm-gripper interaction term with which to score cross-generation coherence. This says nothing by itself about closed-loop harm or benefit.

## Task sensitivity

| Task | Episodes | Targets | `C_offline` | Sign |
|---:|---:|---:|---:|---|
| 0 | 8 | 994 | -4.163336342344337e-17 | zero |
| 1 | 8 | 1063 | 6.9388939039072284e-18 | zero |
| 2 | 8 | 1019 | -8.3266726846886741e-17 | zero |
| 3 | 8 | 985 | 1.3877787807814457e-17 | zero |
| 4 | 8 | 1121 | 0 | zero |
| 5 | 8 | 952 | 0 | zero |
| 6 | 8 | 1043 | -2.7755575615628914e-17 | zero |
| 7 | 8 | 1117 | -1.3877787807814457e-17 | zero |
| 8 | 8 | 995 | -4.163336342344337e-17 | zero |
| 9 | 10 | 1365 | 2.7755575615628914e-17 | zero |

All ten task contrasts are classified zero at tolerance `1e-12`. Leave-one-task-out macro means range from `-2.08e-17` to `-8.48e-18`.

## Components and disagreement diagnostics

FO inherits FF's translation and rotation losses and OO's gripper-sign loss; OF inherits the converse. Individual component means can therefore change between mixed and coherent conditions, but their symmetric additive contrast cancels. The frozen translation and rotation quartiles and the gripper same/different strata likewise retain the targetwise zero contrast; they are descriptive only.

## Provenance and limits

The independent RTX cache contains 82 NPZ files with content-tree SHA256 `7e14e1f341bc2425cb3304cc3f35b0075184b0b1f33225e2dcf05cfe67e50f65`. The frozen ACT model SHA256 is `340071d7497238669459d93517eb3f8690862ad6fdf14207966759dfe6da9410`. Gate-3A1's important RTX ordering reproduced before this audit. The complete machine-readable results and figure-ready data are in the adjacent JSON and CSV outputs.

This audit evaluates one frozen ACT checkpoint, one demonstration corpus, one age, and one separable teacher-forced metric. It neither measures task success nor establishes a causal mechanism. Interpretation with Gate-3B must wait for the independent closed-loop result.
