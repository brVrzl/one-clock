# Analysis-only correction after rollout

The preregistered protocol specified separate paired-bootstrap and task-cluster-bootstrap seed bases (`2027090201` and `2027090301`). The preregistered analysis implementation initially instantiated one RNG from the paired seed and continued it for the cluster draws, leaving the declared cluster seed unused.

After all 840 scientific episodes were complete and outcomes were first analyzed, the implementation was corrected to instantiate the task-cluster RNG from the declared cluster seed. No outcome, cohort, contrast, bootstrap procedure, number of draws, decision threshold, or label rule changed. The regenerated Gate M label remained `METHOD_NULL`.
