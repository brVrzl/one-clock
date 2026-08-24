#!/usr/bin/env python3
"""Run the preregistered, resumable Gate-4A2 LIBERO Spatial rollouts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
LEROBOT_ROOT = Path("/home/wjq/workspace/upstreams/lerobot")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(LEROBOT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate3b_composition import control_semantic_distance  # noqa: E402
from gate3c_temporal_reuse import (  # noqa: E402
    ACTION_DIM,
    ACTION_SMOOTHING_ACTIVE,
    CHUNK_LENGTH,
    METHODS,
    POLICY_TEMPORAL_ENSEMBLE_ACTIVE,
    SOURCE_AGE_TICKS,
    Gate3CTemporalExecutor,
)
from gate4a2_schedule import SELECTED_STATES, pending_runs  # noqa: E402
from scripts.run_libero_gate0 import (  # noqa: E402
    load_config,
    load_policy_and_processors,
    query_full_act_chunk,
    set_episode_seed,
)


SCIENTIFIC_PARENT = "36bebdace1ffbd8d36bacc061feb146cd55f894a"
CHECKPOINT = Path("/home/wjq/checkpoints/ishandotsh_act_libero_spatial_test")
CONFIG = ROOT / "configs/gate4a2_libero_spatial.yaml"
SCHEDULE = ROOT / "research/audit_outputs/gate4a2_spatial_schedule.json"
OUTPUT_ROOT = ROOT / "experiments/gate4a2_spatial_act_generalization"
MANIFEST = ROOT / "research/audit_outputs/gate4a2_spatial_rollout_manifest.json"
HF_CHECKPOINT_REPO = "ishandotsh/act_libero_spatial_test"
HF_CHECKPOINT_REVISION = "8f04de1472975d62db214238b2fc07e78bde2474"
HF_EVALUATION_DATASET_REPO = "zeromidnight/libero_spatial_lerobot_v3.0"
HF_EVALUATION_DATASET_REVISION = "38927e939de5d2bfd40effcf27d16710aea6f864"
HF_TRAINING_DATASET_REPO = "HuggingFaceVLA/libero"
HF_TRAINING_DATASET_REVISION_AUDITED = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"
HF_LIBERO_ASSET_REPO = "lerobot/libero-assets"
HF_LIBERO_ASSET_REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
EXPECTED_HASHES = {
    "model.safetensors": "912f41808962d80ca9084435aa01eccccdd97b7eae3a841c9f4ac71caaf9f8b0",
    "config.json": "0e783369890d33a714cef603185c10dff4215328a9862b181eb7f511f3f1a93c",
    "policy_preprocessor.json": "8a5df04ea1f67ab515898ba211bc64b6c38020e259bc0bd520ddd7b38a660128",
    "policy_postprocessor.json": "c27cf6f42b42352f9b8f9c40da155fd4459e0ee9b85b9f23072941eb52b3ffb5",
    "policy_preprocessor_step_3_normalizer_processor.safetensors": "a002c0df7f79c5b169c5a899ad151d4ea1bed246c7d82bd93ed1556558d517a9",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors": "a002c0df7f79c5b169c5a899ad151d4ea1bed246c7d82bd93ed1556558d517a9",
    "train_config.json": "551dd7bdb8b4ffb109f3ebc40a26856b72953188a74b4a02d597ba2989528b5f",
}
EXPECTED_LEROBOT_COMMIT = "f66e5128ecb2456e8c54a63d15404fa59c16aebc"
CONTROL_HZ = 20.0
MAX_STEPS = 280


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, default=SCHEDULE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--validate-runtime", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def official_state_provenance(schedule: dict[str, Any]) -> dict[str, Any]:
    from libero.libero import benchmark, get_libero_path
    from lerobot.envs.libero import get_task_init_states

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    selected = {int(state) for state in schedule["state_selection"]["selected_state_ids_sorted"]}
    files: dict[str, Any] = {}
    vectors: dict[str, str] = {}
    for task_id in range(10):
        task = suite.get_task(task_id)
        states = get_task_init_states(suite, task_id)
        if len(states) != 50:
            raise RuntimeError(f"task {task_id} does not expose 50 official states")
        path = Path(get_libero_path("init_states")) / task.problem_folder / Path(task.init_states_file).name
        files[str(task_id)] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "array_shape": list(states.shape),
            "array_dtype": str(states.dtype),
        }
        for state_id in selected:
            vectors[f"{task_id}:{state_id}"] = array_sha256(states[state_id])
    return {"files": files, "selected_state_vector_sha256": vectors}


def provenance(
    checkpoint: Path, config_path: Path, schedule_path: Path, schedule: dict[str, Any]
) -> dict[str, Any]:
    observed_hashes = {name: sha256(checkpoint / name) for name in EXPECTED_HASHES}
    mismatches = {
        name: observed for name, observed in observed_hashes.items() if observed != EXPECTED_HASHES[name]
    }
    if mismatches:
        raise RuntimeError(f"checkpoint asset hash mismatch: {mismatches}")
    observed_lerobot_commit = git_commit(LEROBOT_ROOT)
    if observed_lerobot_commit != EXPECTED_LEROBOT_COMMIT:
        raise RuntimeError(f"pinned LeRobot commit mismatch: {observed_lerobot_commit}")
    if subprocess.run(
        ["git", "-C", str(LEROBOT_ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout:
        raise RuntimeError("pinned LeRobot checkout is dirty")
    if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get("PYOPENGL_PLATFORM") != "egl":
        raise RuntimeError("Gate-4A2 requires the frozen EGL rendering backend")
    import torch

    return {
        "scientific_parent": SCIENTIFIC_PARENT,
        "checkpoint_directory": str(checkpoint.resolve()),
        "hf_checkpoint_repository": HF_CHECKPOINT_REPO,
        "hf_checkpoint_revision": HF_CHECKPOINT_REVISION,
        "checkpoint_hashes": observed_hashes,
        "model_sha256": observed_hashes["model.safetensors"],
        "config_sha256": observed_hashes["config.json"],
        "policy_preprocessor_sha256": observed_hashes["policy_preprocessor.json"],
        "policy_postprocessor_sha256": observed_hashes["policy_postprocessor.json"],
        "normalizer_sha256": observed_hashes[
            "policy_preprocessor_step_3_normalizer_processor.safetensors"
        ],
        "unnormalizer_sha256": observed_hashes[
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        ],
        "train_config_sha256": observed_hashes["train_config.json"],
        "training_provenance_category": "MULTI-SUITE",
        "training_dataset_repository": HF_TRAINING_DATASET_REPO,
        "training_dataset_revision_audited": HF_TRAINING_DATASET_REVISION_AUDITED,
        "evaluation_dataset_repository": HF_EVALUATION_DATASET_REPO,
        "evaluation_dataset_revision": HF_EVALUATION_DATASET_REVISION,
        "libero_asset_repository": HF_LIBERO_ASSET_REPO,
        "libero_asset_revision": HF_LIBERO_ASSET_REVISION,
        "project_git_commit_at_rollout_start": git_commit(ROOT),
        "lerobot_git_commit": observed_lerobot_commit,
        "schedule_path": str(schedule_path.resolve()),
        "schedule_sha256": sha256(schedule_path),
        "rollout_config_path": str(config_path.resolve()),
        "rollout_config_sha256": sha256(config_path),
        "temporal_executor_source_sha256": sha256(
            Path(__file__).resolve().parent / "gate3c_temporal_reuse.py"
        ),
        "scalar_weight_source_sha256": sha256(
            Path(__file__).resolve().parent / "gate3a2_temporal_aggregation.py"
        ),
        "official_initial_states": official_state_provenance(schedule),
        "control_frequency_hz": CONTROL_HZ,
        "controller_timestep_seconds": 1.0 / CONTROL_HZ,
        "source_age_ticks": SOURCE_AGE_TICKS,
        "source_age_seconds": SOURCE_AGE_TICKS / CONTROL_HZ,
        "chunk_length": CHUNK_LENGTH,
        "action_dim": ACTION_DIM,
        "action_contract": {"arm_indices": list(range(6)), "gripper_index": 6},
        "policy_temporal_ensemble_active": POLICY_TEMPORAL_ENSEMBLE_ACTIVE,
        "action_smoothing_active": ACTION_SMOOTHING_ACTIVE,
        "rendering_backend": "egl",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_cuda": torch.version.cuda,
        "gpu_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "packages": {
            name: package_version(name)
            for name in (
                "lerobot",
                "hf_libero",
                "torch",
                "numpy",
                "scipy",
                "mujoco",
                "robosuite",
                "gymnasium",
            )
        },
    }


def make_env_and_policy(config: dict[str, Any], checkpoint: Path, task_id: int) -> tuple[Any, ...]:
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv, get_task_init_states

    suite_name = str(config["task_suite"])
    if suite_name != "libero_spatial":
        raise RuntimeError("Gate-4A2 requires vanilla LIBERO Spatial")
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    available_states = len(get_task_init_states(suite, task_id))
    runtime_config = dict(config)
    runtime_config["task_id"] = task_id
    runtime_config["task_name"] = task.name
    components = load_policy_and_processors(runtime_config, checkpoint)
    policy, _, _, _, _ = components
    policy.eval()
    if policy.config.temporal_ensemble_coeff is not None:
        raise RuntimeError("Gate-4A2 requires policy-internal temporal ensembling to be disabled")
    if int(policy.config.chunk_size) != CHUNK_LENGTH:
        raise RuntimeError("checkpoint chunk length differs from preregistration")
    if int(policy.config.n_action_steps) != CHUNK_LENGTH:
        raise RuntimeError("checkpoint n_action_steps differs from complete chunk length")
    if tuple(policy.config.output_features["action"].shape) != (ACTION_DIM,):
        raise RuntimeError("checkpoint action dimension differs from preregistration")
    env = LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=suite_name,
        obs_type=str(config["obs_type"]),
        camera_name=str(config["camera_name"]),
        camera_name_mapping=dict(config["camera_name_mapping"]),
        observation_width=int(config["observation_width"]),
        observation_height=int(config["observation_height"]),
        control_freq=int(config["control_freq"]),
        init_states=bool(config["init_states"]),
        hard_reset=bool(config["hard_reset"]),
        control_mode=str(config["control_mode"]),
    )
    if env.control_freq != CONTROL_HZ or int(env._max_episode_steps) != MAX_STEPS:
        raise RuntimeError("runtime LIBERO Spatial time/horizon contract differs from preregistration")
    if env.control_mode != "relative" or not env.hard_reset:
        raise RuntimeError("runtime LIBERO control/reset contract differs from preregistration")
    if env.action_space.shape != (ACTION_DIM,):
        raise RuntimeError("runtime LIBERO action dimension differs from preregistration")
    return (*components, env, task.name, available_states)


def assert_query_fairness(policy_queries: int, environment_steps: int) -> None:
    if int(policy_queries) != int(environment_steps):
        raise RuntimeError(
            f"Gate-4A2 requires policy_queries == environment_steps, got "
            f"{policy_queries} != {environment_steps}"
        )


def rotation_step_distance(previous: np.ndarray, current: np.ndarray) -> float:
    relative = Rotation.from_rotvec(previous).inv() * Rotation.from_rotvec(current)
    return float(relative.magnitude())


def mean(records: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(record[key]) for record in records])) if records else 0.0


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    actions = np.asarray([record["action"] for record in records], dtype=np.float64)
    active = [record for record in records if record["old_action"] is not None]
    return {
        "mean_arm_effective_age_ticks": mean(records, "arm_effective_age_ticks"),
        "mean_gripper_effective_age_ticks": mean(records, "gripper_effective_age_ticks"),
        "mean_fresh_old_gripper_sign_disagreement": mean(
            active, "fresh_old_gripper_sign_disagreement"
        ),
        "mean_fresh_old_translation_l2": mean(active, "fresh_old_translation_l2"),
        "mean_fresh_old_rotation_radians": mean(active, "fresh_old_rotation_radians"),
        "gripper_transitions": int(
            np.count_nonzero(np.signbit(actions[1:, 6]) != np.signbit(actions[:-1, 6]))
        )
        if len(actions) > 1
        else 0,
        "mean_translation_action_delta_l2": float(
            np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=1).mean()
        )
        if len(actions) > 1
        else 0.0,
        "mean_rotation_action_delta_radians": float(
            np.mean(
                [
                    rotation_step_distance(actions[index - 1, 3:6], actions[index, 3:6])
                    for index in range(1, len(actions))
                ]
            )
        )
        if len(actions) > 1
        else 0.0,
        "mean_raw_action_acceleration_l2": float(
            np.linalg.norm(np.diff(actions, n=2, axis=0), axis=1).mean()
        )
        if len(actions) > 2
        else 0.0,
        "mean_raw_action_jerk_l2": float(
            np.linalg.norm(np.diff(actions, n=3, axis=0), axis=1).mean()
        )
        if len(actions) > 3
        else 0.0,
    }


def run_episode(
    *,
    env: Any,
    policy: Any,
    policy_preprocessor: Any,
    policy_postprocessor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    run: dict[str, Any],
    expected_state_sha256: str,
) -> dict[str, Any]:
    import torch

    seed = int(run["episode_seed"])
    state_id = int(run["state_id"])
    method = str(run["method"])
    observed_state_sha256 = array_sha256(env._init_states[state_id])
    if observed_state_sha256 != expected_state_sha256:
        raise RuntimeError("official initial-state vector identity mismatch")
    set_episode_seed(seed)
    env.init_state_id = state_id
    observation, _ = env.reset(seed=seed)
    policy.reset()
    executor = Gate3CTemporalExecutor(method)
    records: list[dict[str, Any]] = []
    query_seconds = 0.0
    policy_queries = 0
    episode_start = time.perf_counter()
    info: dict[str, Any] = {"is_success": False}
    for step in range(MAX_STEPS):
        query_start = time.perf_counter()
        chunk = query_full_act_chunk(
            observation=observation,
            policy=policy,
            policy_preprocessor=policy_preprocessor,
            policy_postprocessor=policy_postprocessor,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
        )
        policy_queries += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        query_elapsed = time.perf_counter() - query_start
        query_seconds += query_elapsed
        executed = executor.update(step, chunk)
        disagreement = None
        if executed.old_action is not None:
            disagreement = control_semantic_distance(executed.fresh_action, executed.old_action)
        observation, reward, terminated, truncated, info = env.step(executed.action)
        records.append(
            {
                "step": step,
                "method": method,
                "action": executed.action.astype(float).tolist(),
                "fresh_action": executed.fresh_action.astype(float).tolist(),
                "old_action": None
                if executed.old_action is None
                else executed.old_action.astype(float).tolist(),
                "fresh_source_step": executed.fresh_source_step,
                "old_source_step": executed.old_source_step,
                "old_chunk_offset": executed.old_chunk_offset,
                "arm_effective_age_ticks": executed.arm_effective_age_ticks,
                "gripper_effective_age_ticks": executed.gripper_effective_age_ticks,
                "candidate_ages": executed.candidate_ages.astype(int).tolist(),
                "scalar_weights": None
                if np.isnan(executed.weights).any()
                else executed.weights.tolist(),
                "fresh_old_gripper_sign_disagreement": 0.0
                if disagreement is None
                else disagreement["gripper_sign_disagreement"],
                "fresh_old_translation_l2": 0.0
                if disagreement is None
                else disagreement["translation_l2_action_units"],
                "fresh_old_rotation_radians": 0.0
                if disagreement is None
                else disagreement["rotation_geodesic_radians"],
                "query_seconds": query_elapsed,
                "reward": float(reward),
                "is_success": bool(info["is_success"]),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        if terminated or truncated:
            break
    steps = len(records)
    if steps == 0:
        raise RuntimeError("episode executed no actions")
    assert_query_fairness(policy_queries, steps)
    summary = {
        "success": bool(info["is_success"]),
        "failure_category": "success" if bool(info["is_success"]) else "time_limit",
        "steps": steps,
        "policy_queries": policy_queries,
        "policy_queries_per_surviving_step": policy_queries / steps,
        "policy_query_seconds": query_seconds,
        "episode_wall_seconds": time.perf_counter() - episode_start,
        "initial_state_id": state_id,
        "initial_state_vector_sha256": observed_state_sha256,
        **summarize(records),
    }
    return {"run": run, "summary": summary, "steps": records}


def episode_path(output_root: Path, run: dict[str, Any]) -> Path:
    return (
        output_root
        / "episodes"
        / f"task_{int(run['task_id']):02d}"
        / f"state_{int(run['state_id']):02d}"
        / f"{run['method']}.json.gz"
    )


def write_episode(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    os.replace(temporary, path)


def read_valid_episode(
    path: Path, run: dict[str, Any], provenance_data: dict[str, Any]
) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1 or payload.get("run") != run:
        raise RuntimeError(f"existing episode identity/schema mismatch: {path}")
    fields = (
        "hf_checkpoint_revision",
        "model_sha256",
        "config_sha256",
        "policy_preprocessor_sha256",
        "policy_postprocessor_sha256",
        "normalizer_sha256",
        "lerobot_git_commit",
        "schedule_sha256",
        "rollout_config_sha256",
        "temporal_executor_source_sha256",
        "scalar_weight_source_sha256",
    )
    for field in fields:
        if payload.get("provenance", {}).get(field) != provenance_data[field]:
            raise RuntimeError(f"existing episode provenance mismatch for {field}: {path}")
    expected_state = provenance_data["official_initial_states"]["selected_state_vector_sha256"][
        f"{run['task_id']}:{run['state_id']}"
    ]
    if payload["summary"].get("initial_state_vector_sha256") != expected_state:
        raise RuntimeError(f"existing episode initial-state mismatch: {path}")
    if int(payload["summary"]["steps"]) != len(payload["steps"]):
        raise RuntimeError(f"existing episode step count mismatch: {path}")
    assert_query_fairness(payload["summary"]["policy_queries"], payload["summary"]["steps"])
    return payload


def manifest_entry(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    return {
        **payload["run"],
        "status": "complete",
        "success": bool(summary["success"]),
        "failure_category": summary["failure_category"],
        "steps": int(summary["steps"]),
        "policy_queries": int(summary["policy_queries"]),
        "policy_queries_per_surviving_step": float(summary["policy_queries_per_surviving_step"]),
        "initial_state_id": int(summary["initial_state_id"]),
        "initial_state_vector_sha256": summary["initial_state_vector_sha256"],
        "policy_query_seconds": float(summary["policy_query_seconds"]),
        "episode_wall_seconds": float(summary["episode_wall_seconds"]),
        "gripper_transitions": int(summary["gripper_transitions"]),
        **{key: float(value) for key, value in summary.items() if key.startswith("mean_")},
        "log_path": str(path.resolve()),
        "log_bytes": path.stat().st_size,
        "log_sha256": sha256(path),
    }


def write_manifest(
    path: Path,
    schedule: dict[str, Any],
    provenance_data: dict[str, Any],
    entries: dict[int, dict[str, Any]],
) -> None:
    ordered = [entries[index] for index in sorted(entries)]
    atomic_json(
        path,
        {
            "schema_version": 1,
            "scope": "Gate-4A2 preregistered LIBERO Spatial ACT generalization",
            "planned_episodes": int(schedule["planned_episodes"]),
            "completed_episodes": len(ordered),
            "complete": len(ordered) == int(schedule["planned_episodes"]),
            "valid_policy_queries": int(sum(entry["policy_queries"] for entry in ordered)),
            "valid_environment_steps": int(sum(entry["steps"] for entry in ordered)),
            "provenance": provenance_data,
            "episodes": ordered,
        },
    )


def configure_determinism() -> None:
    import torch

    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def main() -> None:
    args = parse_args()
    schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
    if schedule.get("planned_episodes") != 500 or tuple(schedule.get("methods", [])) != METHODS:
        raise RuntimeError("schedule differs from frozen 500-episode method set")
    if tuple(schedule["state_selection"]["selected_state_ids_sorted"]) != SELECTED_STATES:
        raise RuntimeError("schedule state selection differs from the frozen audit")
    config = load_config(args.config)
    if int(config.get("control_freq", -1)) != int(CONTROL_HZ):
        raise RuntimeError("rollout config is not 20 Hz")
    provenance_data = provenance(args.checkpoint, args.config, args.schedule, schedule)
    configure_determinism()
    args.output_root.mkdir(parents=True, exist_ok=True)
    entries: dict[int, dict[str, Any]] = {}
    for run in schedule["runs"]:
        path = episode_path(args.output_root, run)
        if path.exists():
            payload = read_valid_episode(path, run, provenance_data)
            entries[int(run["run_index"])] = manifest_entry(path, payload)
    write_manifest(args.manifest, schedule, provenance_data, entries)
    if args.validate_runtime:
        runtime = make_env_and_policy(config, args.checkpoint, 0)
        runtime[5].close()
        if int(runtime[7]) != 50:
            raise RuntimeError("task 0 does not expose 50 official states")
        print("validated frozen Gate-4A2 runtime without generating an outcome")
    if args.verify_only:
        print(f"verified {len(entries)}/{schedule['planned_episodes']} completed episodes")
        return
    current_task: int | None = None
    runtime: tuple[Any, ...] | None = None
    completed_new = 0
    try:
        for run in pending_runs(schedule, set(entries)):
            task_id = int(run["task_id"])
            if task_id != current_task:
                if runtime is not None:
                    runtime[5].close()
                runtime = make_env_and_policy(config, args.checkpoint, task_id)
                current_task = task_id
                if int(runtime[7]) != 50:
                    raise RuntimeError(f"task {task_id} does not expose 50 official states")
            assert runtime is not None
            policy, policy_pre, policy_post, env_pre, env_post, env, _, _ = runtime
            expected_state = provenance_data["official_initial_states"][
                "selected_state_vector_sha256"
            ][f"{task_id}:{run['state_id']}"]
            payload = run_episode(
                env=env,
                policy=policy,
                policy_preprocessor=policy_pre,
                policy_postprocessor=policy_post,
                env_preprocessor=env_pre,
                env_postprocessor=env_post,
                run=run,
                expected_state_sha256=expected_state,
            )
            path = episode_path(args.output_root, run)
            payload = {"schema_version": 1, "provenance": provenance_data, **payload}
            write_episode(path, payload)
            entries[int(run["run_index"])] = manifest_entry(path, payload)
            write_manifest(args.manifest, schedule, provenance_data, entries)
            completed_new += 1
            print(
                f"[{len(entries):03d}/{schedule['planned_episodes']}] completed "
                f"run_index={run['run_index']}",
                flush=True,
            )
    finally:
        if runtime is not None:
            runtime[5].close()
    print(f"completed {completed_new} new episodes; total valid={len(entries)}")


if __name__ == "__main__":
    main()
