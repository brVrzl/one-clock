#!/usr/bin/env python3
"""Run one bounded, current-frame-only SmolVLA compatibility inference."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path

from dataset_common import DATASET_REVISION, atomic_write_json, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = {
        "dataset_revision": DATASET_REVISION,
        "checkpoint_repo_id": "HuggingFaceVLA/smolvla_libero",
        "checkpoint_revision": "6721902bc4d61e50a3bfdb11dfb4cb626f05d102",
        "machine": platform.machine(),
        "status": "not_run",
        "checks": {},
    }
    try:
        import torch

        result["checks"]["torch"] = {"version": torch.__version__, "cuda_available": bool(torch.cuda.is_available())}
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        from build_policy_cache import read_rows, source_observation
        from smolvla_runtime import CHECKPOINT_REVISION, CHECKPOINT_REPO_ID, infer_full_chunk, load_runtime

        rows = read_rows(args.corpus_dir.resolve())
        first = rows[0]
        result["checks"]["dataset_frame"] = {"frame_id": int(first["frame_id"]), "task_name": str(first["task_name"])}
        observation = source_observation(args.dataset_root.resolve(), first)
        observation["task_name"] = str(first["task_name"])
        result["checks"]["observation_preprocessing"] = {"images": {key: list(value.shape) for key, value in observation["images"].items()}, "state": list(observation["state"].shape)}
        runtime = load_runtime(args.dataset_root.resolve(), args.device)
        result["checks"]["model_load"] = {"device": runtime["device"], "policy_type": str(runtime["policy_config"].type)}
        from huggingface_hub import hf_hub_download

        checkpoint_files = {}
        for filename in ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
            try:
                local_path = Path(hf_hub_download("HuggingFaceVLA/smolvla_libero", filename, revision=CHECKPOINT_REVISION, repo_type="model"))
                checkpoint_files[filename] = {"absolute_path": str(local_path.resolve()), "bytes": local_path.stat().st_size, "sha256": sha256_file(local_path)}
            except Exception as exc:
                checkpoint_files[filename] = {"error": f"{type(exc).__name__}: {exc}"}
        result["checkpoint_files"] = checkpoint_files
        chunk = infer_full_chunk(runtime, observation)
        result["checks"]["full_chunk"] = {"shape": list(chunk.shape), "dtype": str(chunk.dtype), "finite": bool(chunk.size and __import__("numpy").isfinite(chunk).all()), "first_action": chunk[0].tolist()}
        result["checks"]["normalization"] = {"postprocessed_action_range": [float(chunk.min()), float(chunk.max())], "source": "checkpoint policy processor pipelines"}
        result["checks"]["truncation_audit"] = {"captured_full_predict_action_chunk": True, "configured_chunk_length": int(runtime["policy_config"].chunk_size), "normal_execution_n_action_steps": int(runtime["policy_config"].n_action_steps)}
        result["checks"]["latent"] = {"status": "not_exposed_by_generic_policy_interface"}
        timings = []
        for row in rows[1:3]:
            probe = source_observation(args.dataset_root.resolve(), row)
            probe["task_name"] = str(row["task_name"])
            started = time.perf_counter()
            infer_full_chunk(runtime, probe)
            timings.append(time.perf_counter() - started)
        if timings:
            mean_seconds = float(sum(timings) / len(timings))
            estimated_hours = mean_seconds * 273465 / 3600.0
            result["checks"]["small_throughput_probe"] = {"inference_plus_decode_calls": len(timings), "seconds": timings, "mean_seconds": mean_seconds, "estimated_full_cache_hours": estimated_hours, "acceptance_limit_hours": 24.0}
            if estimated_hours > 24.0:
                raise RuntimeError(f"estimated full LIBERO-4 cache time {estimated_hours:.1f}h exceeds bounded Thor usability limit 24h")
        result["status"] = "feasible"
        return_code = 0
    except Exception as exc:  # bounded compatibility attempt; details are durable
        result["status"] = "blocked"
        result["blocker"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        return_code = 2
    atomic_write_json(args.output.resolve(), result)
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}, sort_keys=True), flush=True)
    return return_code


if __name__ == "__main__":
    sys.exit(main())
