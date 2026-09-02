# Clean-process preflight

After preregistration and before any Gate M episode, SmolVLA and ACT were each constructed in a separate new process as the first and only model in that process. Each process created and reset a fresh environment and produced one postprocessed policy chunk without calling `env.step` or observing an outcome.

- `clean_smolvla_process.json`: PASS, chunk `(50, 7)`, `n_action_steps=1`, zero environment steps.
- `clean_act_process.json`: PASS, chunk `(100, 7)`, `n_action_steps=100`, corrected historical `wrist_image` camera/configuration path, zero environment steps.

The first SmolVLA audit invocation completed construction, reset, and prediction, then raised one reporting-only `AttributeError` while reading a non-existent optional `temporal_ensemble_coeff` attribute. It produced no result file, environment step, reward, or outcome. The audit reporter was changed to use `getattr(..., None)` in commit `9f9cf23`; the successful rerun occurred in a new process with no failed construction state to inherit. This was not a scientific execution retry.
