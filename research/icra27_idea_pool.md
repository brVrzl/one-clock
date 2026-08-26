# ICRA 2027 direction reset: deduplicated candidate pool

Derived from two `scientific-brainstorming` passes plus three independent Luna ideation passes. Candidate labels are proposals, not conclusions.

| ID | Candidate | Central question | Distinct mechanism / needed evidence | Current status |
|---|---|---|---|---|
| I01 | Dynamic Component-wise Temporal Aggregation (DCTA) | Should ACT ensemble weights vary by action group? | dynamic group weights; needs cross-task gain beyond global weighting | mandatory; demoted |
| I02 | Async component-wise replanning | Does selective subspace re-query beat whole-chunk re-query? | retain valid components, replace invalid one; equal-query study | serious |
| I03 | Contact/force-triggered selective replanning | Should contact events refresh only affected components? | event detector + partial repair; must differ from tactile residual papers | serious only within I02 |
| I04 | Heterogeneous multi-rate action policy | Do components need differing refresh rates? | independently scheduled groups; must differ from HiPolicy/global phase adaptation | demoted/reframe |
| I05 | Component-wise scheduling / compute allocation | Which update earns a scarce VLA call? | scheduler over component refreshes at fixed budget | serious |
| I06 | Validity-field execution | Can time × action-subspace validity predict counterfactual replacement value? | validity map then partial refresh | serious |
| I07 | Recoverability-budgeted refresh | Is refresh value governed by remaining recoverability rather than age? | recoverability estimator + action choice | serious, strongest formulation |
| I08 | Coherence-constrained partial refresh | When is component independence unsafe? | coupling-aware retention/replacement | enabling module / ablation |
| I09 | Cross-modal freshness routing | Which sensor stream should refresh which action group? | sensor-action causal routing | backup, engineering-heavy |
| I10 | Failure-response taxonomy | Which failure warrants continue/local repair/partial/global replan? | causal intervention matrix | enabling figure |
| I11 | Role-conditioned bimanual temporal contract | Can arms synchronize only at coupling events? | role-dependent bimanual timing | backup, high burden |
| I12 | Hold/refresh action representation | Is explicit commitment/expiry better than dense action vectors? | structured action interface | backup, later method option |
| I13 | Contact-graph dexterous actions | Does contact topology transfer better than fingers trajectories? | contact graph + fast controller | independent alternative, high data risk |
| I14 | Query-value / sensor-policy allocation | Is VLA inference itself a value-of-information action? | scheduler across VLA, camera, force, local policy | backup, broad/scope risk |
| I15 | Source-age calibration diagnostic | Can a benchmark expose subspace temporal laws? | standard factorial source-age protocol | supporting artifact, not main paper |

## Mandatory-candidate disposition before reviewer audits

- **A. DCTA:** retained as I01 but only as ablation/module.
- **B. Asynchronous component-wise replanning:** retained as I02.
- **C. Contact / force-triggered selective replanning:** retained as I03, but generic architecture is occupied.
- **D. Heterogeneous multi-rate action policy:** retained as I04, but global framing is occupied.
- **E. Component-wise scheduling:** retained as I05.
- **F. Stronger discovered formulation:** I06 + I07 + I08: *validity/recoverability-aware, coherence-constrained partial replanning under a fixed compute budget*.
