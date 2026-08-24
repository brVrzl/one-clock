# Final ManiSkill Gate 0 artifacts

This directory contains the final mixed-perturbation causal screen:

- `pick_manifest.json`, `stack_manifest.json`: seeds, action counts, sampled
  states, branch counts, success, and runtime;
- `*_episode_*.pt`: raw actions and cloned ManiSkill state dictionaries;
- `*_timestep_branches.csv`: branch-level outcomes and SHA-256 state IDs;
- `gate0_analysis.json`: summary statistics and heuristic correlations;
- `figures/`: phase curve, histogram, and heuristic scatter diagnostics.

The manifests retain the temporary generation paths used during collection;
the raw files were copied into this directory before analysis and commit.
The canonical interpretation is in `research/maniskill_gate0_report.md`.
