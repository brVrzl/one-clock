# SmolVLA LIBERO cache audit

Status: `BLOCKED BEFORE REAL CACHE GENERATION`.

The repository now contains a reproducible harness in
`cache_smoke_test.py`. It uses the native LeRobot dataset format, calls
`predict_action_chunk` once per unique frame, captures the complete
de-normalized 50-step action chunk, and keeps the offline labels in a
separate artifact. The label rule used by the smoke harness is explicit and
configurable; it must be replaced by the audited Thor target rule for a
production Y_refresh run.

## Provenance audit

- Checkpoint: `HuggingFaceVLA/smolvla_libero`, revision
  `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`.
- LeRobot source: tag `v0.4.4`, commit
  `8fff0fde7c79f23a93d845d1a50e985de01f8b8a`.
- Dataset: `HuggingFaceVLA/libero`, revision
  `86958911c0f959db2bbbdb107eb3e17c5f9c798e`; v3 metadata reports 1,693
  episodes, 273,465 frames, 40 tasks, 10 Hz, and 7-D actions.
- Planned representative subset: episode 0 only, first 12 consecutive
  frames, offsets 1–4. This is fixed by the command below, not selected by
  result quality. The combined dataset metadata does not encode the original
  LIBERO suite name for each episode, so this is documented as a fixed
  combined-dataset smoke subset rather than being relabeled as Spatial.

The checkpoint configuration specifies two 256×256 RGB images, an 8-D state,
7-D action, 50-step `chunk_size`, and `n_action_steps=1`. The cache therefore
retains the full 50-step output even though the default execution setting
consumes one action before the next normal query.

## What was verified

1. The local XPolicyLab adapter and LeRobot `v0.4.4` source import cleanly in
   the CUDA environment after installing the missing runtime dependencies.
2. LeRobot’s `SmolVLAPolicy.predict_action_chunk` is the correct full-chunk
   API; using `select_action` would expose only the execution queue behavior.
3. The normal source-prefix forward produces a reusable VLM prefix/cache
   object before denoising. The harness observes the final prefix hidden state
   with a forward hook and does not replace model outputs or alter weights.
4. The harness code passes Python compilation and its pure label helper passes
   a synthetic offset/lookup consistency check with the project Python
   environment. No policy-backed cache result was inferred from that test.

## Blocker

The 1.2 GB checkpoint snapshot could not be completed on this machine. The HF
Xet path repeatedly failed with `tls handshake eof` against
`cas-server.xethub.hf.co`; the non-Xet streamed download stalled before
`model.safetensors` arrived. Consequently there is no real frozen-policy
forward count, cache size, throughput, latent tensor, or cached-vs-direct
correctness result in this directory. No fabricated cache artifact was
created.

The attempted weight-transfer commands were:

```bash
HF_HUB_DISABLE_XET=1 \
  /home/xdl/miniforge3/envs/env_isaaclab/bin/python -c \
  "from huggingface_hub import hf_hub_download; hf_hub_download('HuggingFaceVLA/smolvla_libero', filename='model.safetensors', revision='6721902bc4d61e50a3bfdb11dfb4cb626f05d102', local_dir='/home/wjq/workspace/upstreams/XPolicyLab/policy/SmolVLA/checkpoints/smolvla_libero')"
```

An Xet-backed snapshot download of the same pinned revision was also tried;
both paths stopped before the weight file was complete.

## Exact command to resume

After placing the checkpoint and the selected dataset files at the paths in
`cache_manifest.json`, run:

```bash
PYTHONPATH=src:/home/wjq/workspace/upstreams/XPolicyLab/policy/SmolVLA/smovla/src \
  /home/xdl/miniforge3/envs/env_isaaclab/bin/python \
  experiments/dynamic_reliability_horizon/smolvla_cache_audit/cache_smoke_test.py \
  --dataset-root /home/wjq/datasets/huggingfacevla_libero \
  --checkpoint /home/wjq/workspace/upstreams/XPolicyLab/policy/SmolVLA/checkpoints/smolvla_libero \
  --lerobot-source /home/wjq/workspace/upstreams/XPolicyLab/policy/SmolVLA/smovla \
  --output-dir experiments/dynamic_reliability_horizon/smolvla_cache_audit \
  --episodes 0 --max-frames 12 --max-horizon 4 --device cuda \
  --seed 20260820 --linf-tolerance 0.05 --direct-check-windows 2
```

The expected successful output is `cache_manifest.json` with
`cached_equals_direct_labels: true`, `num_cached_policy_forwards` equal to
the number of unique selected frames, and two `.npz` files matching
`schema.md`. The direct correctness path intentionally performs repeated
future queries only on the tiny check subset; the cache path scales with
unique frames.

## Latent decision

The candidate `z_t` is the final hidden state sequence returned by the normal
source-prefix VLM call, mean-pooled over its sequence dimension for the tiny
cache. It is not an arbitrary action-head tensor: it exists before action
denoising, is shared by all action offsets, and is already computed in normal
inference. A later implementation should benchmark the exact prefix KV cache
versus a compact projection before choosing the production estimator input.

## Scaling estimate

The dataset metadata reports 273,465 frames for the complete 40-task LIBERO
conversion. A four-suite run should be planned from the actual suite frame
counts after downloading metadata, with one policy forward per unique frame
and storage approximately equal to:

```text
N_frames * (50 * 7 * sizeof(float32) + sizeof(z_t) + metadata)
```

At the action-chunk field alone this is about 383 MB for 273,465 frames. The
latent field and label-side lookup/targets are additional. This is a planning
estimate, not a completed four-suite extraction.

## References

- [SmolVLA LIBERO checkpoint](https://huggingface.co/HuggingFaceVLA/smolvla_libero/tree/6721902bc4d61e50a3bfdb11dfb4cb626f05d102)
- [SmolVLA checkpoint config](https://huggingface.co/HuggingFaceVLA/smolvla_libero/blob/6721902bc4d61e50a3bfdb11dfb4cb626f05d102/config.json)
- [HuggingFaceVLA LIBERO dataset](https://huggingface.co/datasets/HuggingFaceVLA/libero)
- [LeRobot v0.4.4](https://github.com/huggingface/lerobot/tree/v0.4.4)
