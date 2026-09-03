#!/usr/bin/env python3
"""Technical-only preflight for the R1D LeRobot import repair."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from r1d_runtime_repair import EXPECTED_PACKAGE, preload_validated_runtime


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SOURCE_CHECKOUT = Path("/home/wjq/workspace/upstreams/lerobot")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_files(path: Path) -> int:
    return sum(item.is_file() for item in path.glob("*")) if path.is_dir() else 0


def main() -> None:
    preload_validated_runtime()
    installed_lerobot_version = importlib.metadata.version("lerobot")
    sys.path.insert(0, str(ROOT))
    # Reproduce the frozen queue's later source-path insertion. Because the
    # validated package root is already imported, it cannot redirect submodules.
    sys.path.insert(0, str(SOURCE_CHECKOUT / "src"))

    from executors import DenseExecutor
    from frozen_queue import phase_cells, protocol
    from run_queue import Runtime

    subprocess.run(
        [sys.executable, str(ROOT / "validate_supplement.py"), "--static"],
        cwd=REPO,
        check=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )

    frozen = json.loads((ROOT / "manifests/r1d.json").read_text(encoding="utf-8"))
    cells = phase_cells("r1d")
    p = protocol()
    if frozen["cells"] != cells or frozen["cell_count"] != 100 or len(cells) != 100:
        raise RuntimeError("R1D manifest identity mismatch")

    artifact_counts = {
        "results": count_files(ROOT / "results/r1d"),
        "completion_or_failure_markers": count_files(ROOT / "markers/r1d"),
        "attempt_records": count_files(ROOT / "attempts/r1d"),
    }
    if any(artifact_counts.values()):
        raise RuntimeError(f"R1D is not in the frozen zero-artifact state: {artifact_counts}")

    r1d = p["r1d"]
    checkpoint = Path(r1d["checkpoint"])
    if file_sha256(checkpoint / "model.safetensors") != r1d["model_sha256"]:
        raise RuntimeError("R1D model identity mismatch")
    if file_sha256(checkpoint / "config.json") != r1d["config_sha256"]:
        raise RuntimeError("R1D config identity mismatch")
    checkout_commit = subprocess.check_output(
        ["git", "-C", str(SOURCE_CHECKOUT), "rev-parse", "HEAD"], text=True
    ).strip()
    if checkout_commit != r1d["lerobot_commit"]:
        raise RuntimeError("intended R1D source checkout drift")

    synthetic = np.arange(40 * 100 * 7, dtype=np.float32).reshape(40, 100, 7)
    executor = DenseExecutor("A20_G0")
    for t in range(40):
        action, source = executor.step(t, synthetic[t])
        expected = synthetic[t, 0] if t < 20 else np.r_[synthetic[t - 20, 20, :6], synthetic[t, 0, 6]]
        if not np.array_equal(action, expected):
            raise RuntimeError("Reverse20 executor semantics drift")
        for label in ("translation", "rotation", "gripper"):
            if source[f"{label}_q"] + source[f"{label}_k"] != t:
                raise RuntimeError("q+k=t drift")

    cell = cells[0]
    runtime = Runtime("0")
    cfg = runtime.env_cfg(cell)
    runtime.load(cell, cfg)
    env_pre, env_post = runtime.make_env_pre_post_processors(env_cfg=cfg, policy_cfg=runtime.cfg)
    env = runtime.make_env(cfg, n_envs=1, use_async_envs=False)[cell["suite"]][cell["task_id"]]
    try:
        base = env.envs[0]
        offscreen = base._env
        robosuite_env = offscreen.env
        control_freq = float(robosuite_env.control_freq)
        control_timestep = float(robosuite_env.control_timestep)
        if control_freq != 20 or not math.isclose(control_timestep, 0.05, rel_tol=0, abs_tol=1e-12):
            raise RuntimeError(
                f"unexpected evaluator clock: {control_freq} Hz, {control_timestep} s"
            )

        base.init_state_id = int(cell["state_id"])
        runtime.reset_rng(int(cell["environment_seed"]))
        runtime.policy.reset()
        observation, _ = env.reset(seed=[int(cell["environment_seed"])])

        from lerobot.envs.utils import preprocess_observation

        processed = runtime.preprocessor(env_pre(preprocess_observation(observation)))
        expected_shapes = {
            "observation.images.image": (1, 3, 256, 256),
            "observation.images.image2": (1, 3, 256, 256),
            "observation.state": (1, 8),
        }
        shapes = {key: tuple(processed[key].shape) for key in expected_shapes}
        if shapes != expected_shapes:
            raise RuntimeError(f"observation contract drift: {shapes}")

        import torch
        from lerobot.utils.constants import ACTION

        with torch.inference_mode():
            normalized_zero = torch.zeros((1, 7), device=runtime.cfg.device)
            denormalized = runtime.postprocessor(normalized_zero)
            final_action = env_post({ACTION: denormalized})[ACTION]
        action_value = final_action.detach().cpu().numpy()
        if action_value.shape != (1, 7) or not np.isfinite(action_value).all():
            raise RuntimeError("action denormalization contract drift")

        module_names = (
            "lerobot.configs.policies",
            "lerobot.policies.act.configuration_act",
            "lerobot.policies.act.modeling_act",
            "lerobot.envs.configs",
            "lerobot.envs.libero",
            "lerobot.envs.factory",
            "lerobot.policies.factory",
        )
        module_paths = {
            name: str(Path(importlib.import_module(name).__file__).resolve()) for name in module_names
        }
        if any(EXPECTED_PACKAGE.resolve() not in Path(path).parents for path in module_paths.values()):
            raise RuntimeError(f"mixed LeRobot source paths: {module_paths}")

        output = {
            "status": "PASS",
            "excluded_from_scientific_analysis": True,
            "scientific_environment_steps": 0,
            "python": sys.executable,
            "lerobot_version": installed_lerobot_version,
            "transformers_version": importlib.metadata.version("transformers"),
            "module_paths": module_paths,
            "policy_class": inspect.getsourcefile(type(runtime.policy)),
            "policy_type": runtime.cfg.type,
            "chunk_size": int(runtime.cfg.chunk_size),
            "action_dim": int(runtime.cfg.output_features[ACTION].shape[0]),
            "checkpoint_loaded": True,
            "checkpoint_identity": "EXACT_FROZEN_MATCH",
            "manifest_identity": "EXACT_FROZEN_MATCH",
            "artifact_counts_before_launch": artifact_counts,
            "evaluator_clock_hz": control_freq,
            "evaluator_step_seconds": control_timestep,
            "observation_shapes": {key: list(value) for key, value in shapes.items()},
            "policy_preprocessor_steps": [type(step).__name__ for step in runtime.preprocessor.steps],
            "policy_postprocessor_steps": [type(step).__name__ for step in runtime.postprocessor.steps],
            "env_preprocessor_steps": [type(step).__name__ for step in env_pre.steps],
            "env_postprocessor_steps": [type(step).__name__ for step in env_post.steps],
            "action_denormalization": "FROZEN_CHECKPOINT_PROCESSOR_LOADED_AND_FINITE_7D_OUTPUT",
            "reverse20_semantics": "PASS",
            "physical_same_target_q_plus_k_equals_t": True,
            "reference_sequence_comparison": "REFERENCE_SEQUENCE_UNAVAILABLE",
            "intended_source_checkout_commit": checkout_commit,
            "repair_disposition": "SCIENTIFICALLY_NEUTRAL_IMPORT_PATH_REPAIR",
        }
    finally:
        env.close()
        runtime.drop()

    path = ROOT / "canaries/r1d_runtime_repair.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
