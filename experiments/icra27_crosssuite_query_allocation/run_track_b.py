#!/usr/bin/env python3
"""Run H16 trajectories with outcome-inert dense same-target prediction logging."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from one_clock import ActionGroup, FixedChunkExecutor  # noqa: E402


ACTION_DIM = 7
ARM = tuple(range(6))
GRIPPER = (6,)
HORIZON = 16


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def json_path(cell: dict[str, Any]) -> Path:
    return ROOT / "track_b" / "results" / f"{cell['cell_id']}.json"


def npz_path(cell: dict[str, Any]) -> Path:
    return ROOT / "track_b" / "predictions" / f"{cell['cell_id']}.npz"


def marker_path(cell: dict[str, Any], status: str = "complete") -> Path:
    return ROOT / "track_b" / "markers" / f"{cell['cell_id']}.{status}"


def query_seed(cell: dict[str, Any], q: int, *, logging: bool) -> int:
    domain = "smolvla-log" if logging else "smolvla"
    key = (
        f"{domain}|{cell['suite']}:task{cell['task_id']}|state={cell['state_id']}|"
        f"env_seed={cell['environment_seed']}|q={q}"
    )
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)


def reset_rng(torch: Any, seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def snapshot_rng(torch: Any) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(torch: Any, state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def simulator_state(env: Any) -> np.ndarray:
    return np.asarray(env.envs[0]._env.get_sim_state()).astype(np.float64, copy=True)


class Runtime:
    def __init__(self, gpu: str):
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env, make_env_pre_post_processors
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        self.torch = torch
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        self.PreTrainedConfig = PreTrainedConfig
        self.LiberoEnv = LiberoEnv
        self.make_env = make_env
        self.make_env_pre_post_processors = make_env_pre_post_processors
        self.make_policy = make_policy
        self.make_pre_post_processors = make_pre_post_processors
        self.identity: tuple[str, str] | None = None
        self.policy = None
        self.cfg = None
        self.preprocessor = None
        self.postprocessor = None

    def drop_policy(self) -> None:
        self.identity = None
        self.policy = self.cfg = self.preprocessor = self.postprocessor = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    @staticmethod
    def historical_object_checkpoint(cell: dict[str, Any]) -> bool:
        return str(cell["checkpoint"]).endswith("/zeromidnight_act_libero_object")

    def env_config(self, cell: dict[str, Any]):
        historical = self.historical_object_checkpoint(cell)
        camera_mapping = (
            {"agentview_image": "image", "robot0_eye_in_hand_image": "wrist_image"}
            if historical
            else {"agentview_image": "image", "robot0_eye_in_hand_image": "image2"}
        )
        cfg = self.LiberoEnv(
            task=cell["suite"], task_ids=[int(cell["task_id"])],
            fps=int(cell["control_frequency_hz"]), obs_type="pixels_agent_pos",
            camera_name="agentview_image,robot0_eye_in_hand_image",
            camera_name_mapping=camera_mapping, init_states=True,
            observation_width=256, observation_height=256, control_mode="relative",
        )
        if historical:
            cfg.features_map["pixels/robot0_eye_in_hand_image"] = "observation.images.wrist_image"
        return cfg

    def policy_for(self, cell: dict[str, Any], env_cfg: Any) -> None:
        checkpoint = str(Path(cell["checkpoint"]).resolve())
        identity = (cell["policy"], checkpoint)
        if self.identity == identity:
            return
        self.drop_policy()
        cp = Path(checkpoint)
        if not (cp / "config.json").is_file() or not (cp / "model.safetensors").is_file():
            raise FileNotFoundError(f"checkpoint missing required files: {cp}")
        cfg = self.PreTrainedConfig.from_pretrained(cp)
        cfg.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        cfg.pretrained_path = cp
        expected = "act" if cell["policy"] == "ACT" else "smolvla"
        if getattr(cfg, "type", None) != expected:
            raise RuntimeError(f"expected {expected}, got {getattr(cfg, 'type', None)!r}")
        if getattr(cfg, "temporal_ensemble_coeff", None) is not None:
            raise RuntimeError("diagnostic checkpoints must have temporal aggregation disabled")
        if int(cfg.output_features["action"].shape[0]) != ACTION_DIM:
            raise RuntimeError("action dimension is not 7")
        if int(cfg.chunk_size) < HORIZON:
            raise RuntimeError("chunk is shorter than H16")
        if expected == "act":
            cfg.pretrained_backbone_weights = None
        self.policy = self.make_policy(cfg=cfg, env_cfg=env_cfg)
        self.policy.eval()
        self.preprocessor, self.postprocessor = self.make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(cfg.device)}},
        )
        self.cfg = cfg
        self.identity = identity

    def prepare(self, cell: dict[str, Any]):
        env_cfg = self.env_config(cell)
        self.policy_for(cell, env_cfg)
        assert self.cfg is not None and self.policy is not None
        env_pre, env_post = self.make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=self.cfg)
        reset_rng(self.torch, int(cell["environment_seed"]))
        env = self.make_env(env_cfg, n_envs=1, use_async_envs=False)[cell["suite"]][int(cell["task_id"])]
        env.envs[0].init_state_id = int(cell["state_id"])
        if int(env.envs[0].init_state_id) != int(cell["state_id"]):
            env.close()
            raise RuntimeError("initial-state assignment mismatch")
        execution_seed = 424242 if cell["policy"] == "ACT" else query_seed(cell, 0, logging=False)
        reset_rng(self.torch, execution_seed)
        self.policy.reset()
        observation, _ = env.reset(seed=[int(cell["environment_seed"])])
        return env, env_pre, env_post, observation

    def processed_input(self, cell: dict[str, Any], env: Any, observation: dict[str, Any], env_pre: Any):
        from lerobot.envs.utils import add_envs_task, preprocess_observation

        batch = preprocess_observation(observation)
        if cell["policy"] == "SmolVLA":
            batch = add_envs_task(env, batch)
        batch = env_pre(batch)
        if (
            self.historical_object_checkpoint(cell)
            and "observation.images.image2" in batch
            and "observation.images.wrist_image" not in batch
        ):
            batch["observation.images.wrist_image"] = batch.pop("observation.images.image2")
        assert self.preprocessor is not None
        return self.preprocessor(batch)

    def normalized_chunk(
        self,
        cell: dict[str, Any],
        env: Any,
        observation: dict[str, Any],
        env_pre: Any,
        t: int,
        *,
        logging: bool,
    ):
        assert self.policy is not None and self.cfg is not None
        saved_rng = snapshot_rng(self.torch) if logging else None
        if cell["policy"] == "SmolVLA":
            reset_rng(self.torch, query_seed(cell, t, logging=logging))
        batch = self.processed_input(cell, env, observation, env_pre)
        started = time.perf_counter()
        try:
            with self.torch.inference_mode():
                chunk = self.policy.predict_action_chunk(batch)
        finally:
            if saved_rng is not None:
                restore_rng(self.torch, saved_rng)
        latency = time.perf_counter() - started
        if tuple(chunk.shape) != (1, int(self.cfg.chunk_size), ACTION_DIM):
            raise RuntimeError(f"unexpected normalized chunk shape {tuple(chunk.shape)}")
        return chunk, latency

    def postprocess_chunk(self, normalized: Any, env_post: Any) -> np.ndarray:
        from lerobot.utils.constants import ACTION

        assert self.postprocessor is not None
        with self.torch.inference_mode():
            value = self.postprocessor(normalized)
            value = env_post({ACTION: value})[ACTION]
        return value.detach().cpu().numpy()[0].astype(np.float32, copy=False)

    @staticmethod
    def success_from_step(reward: Any, terminated: Any, truncated: Any, info: Any) -> tuple[bool, bool]:
        done = bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0])
        if not done:
            return False, False
        final = info.get("final_info") if isinstance(info, dict) else None
        if isinstance(final, dict) and "is_success" in final:
            return True, bool(np.asarray(final["is_success"]).reshape(-1)[0])
        return True, bool(np.asarray(reward).reshape(-1)[0] > 0)

    def run_episode(self, cell: dict[str, Any], *, dense_logging: bool) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        env, env_pre, env_post, observation = self.prepare(cell)
        started = time.perf_counter()
        try:
            assert self.cfg is not None
            max_steps = int(cell["max_episode_steps"] or np.asarray(env.call("_max_episode_steps")).reshape(-1)[0])
            groups = (ActionGroup("arm", ARM, HORIZON), ActionGroup("gripper", GRIPPER, HORIZON))
            executor = FixedChunkExecutor.global_fixed(
                action_dim=ACTION_DIM, chunk_size=int(self.cfg.chunk_size), horizon=HORIZON, groups=groups,
            )
            query_steps: list[int] = []
            execution_latencies: list[float] = []
            logging_latencies: list[float] = []
            predictions: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            sim_states: list[np.ndarray] = [simulator_state(env)]
            success = False
            for t in range(max_steps):
                def query() -> np.ndarray:
                    normalized, latency = self.normalized_chunk(
                        cell, env, observation, env_pre, t, logging=False,
                    )
                    execution_latencies.append(latency)
                    return self.postprocess_chunk(normalized, env_post)

                decision = executor.step(query)
                if decision.policy_query:
                    query_steps.append(t)
                if dense_logging:
                    logged, latency = self.normalized_chunk(
                        cell, env, observation, env_pre, t, logging=True,
                    )
                    predictions.append(logged.detach().cpu().numpy()[0].astype(np.float32, copy=False))
                    logging_latencies.append(latency)
                action = decision.action.astype(np.float32, copy=False)
                actions.append(action.copy())
                observation, reward, terminated, truncated, info = env.step(action[None])
                sim_states.append(simulator_state(env))
                done, success = self.success_from_step(reward, terminated, truncated, info)
                if done:
                    break
            steps = len(actions)
            metadata = {key: cell[key] for key in (
                "cell_id", "policy", "suite", "task_id", "state_id", "environment_seed", "checkpoint",
            )}
            metadata.update({
                "status": "COMPLETE",
                "trajectory": "H16",
                "dense_logging": bool(dense_logging),
                "extra_predictions_affect_execution": False,
                "success": bool(success),
                "environment_steps": steps,
                "policy_queries_for_execution": len(query_steps),
                "execution_query_steps": query_steps,
                "diagnostic_policy_queries": steps if dense_logging else 0,
                "execution_query_rate": len(query_steps) / steps,
                "execution_model_forward_seconds": execution_latencies,
                "diagnostic_model_forward_seconds": logging_latencies,
                "wall_clock_seconds": time.perf_counter() - started,
                "chunk_size": int(self.cfg.chunk_size),
                "action_dim": ACTION_DIM,
                "prediction_space": "checkpoint-normalized action space",
                "normalization_refit": False,
                "finished_at": time.time(),
            })
            arrays = {
                "predicted_chunks_normalized": (
                    np.stack(predictions).astype(np.float32, copy=False)
                    if predictions
                    else np.empty((0, int(self.cfg.chunk_size), ACTION_DIM), dtype=np.float32)
                ),
                "executed_actions": np.stack(actions).astype(np.float32, copy=False),
                "simulator_states": np.stack(sim_states).astype(np.float64, copy=False),
            }
            return metadata, arrays
        finally:
            env.close()


def validate(cell: dict[str, Any]) -> dict[str, Any]:
    metadata = json.loads(json_path(cell).read_text(encoding="utf-8"))
    for key in ("cell_id", "policy", "suite", "task_id", "state_id", "environment_seed", "checkpoint"):
        if metadata.get(key) != cell.get(key):
            raise ValueError(f"Track-B {key} mismatch")
    if metadata.get("status") != "COMPLETE" or not metadata.get("dense_logging"):
        raise ValueError("Track-B result is not a complete dense log")
    arrays = np.load(npz_path(cell))
    steps = int(metadata["environment_steps"])
    if arrays["predicted_chunks_normalized"].shape != (steps, int(metadata["chunk_size"]), ACTION_DIM):
        raise ValueError("Track-B prediction tensor shape mismatch")
    if arrays["executed_actions"].shape != (steps, ACTION_DIM):
        raise ValueError("Track-B executed-action tensor shape mismatch")
    if arrays["simulator_states"].shape[0] != steps + 1:
        raise ValueError("Track-B simulator-state tensor length mismatch")
    if metadata["execution_query_steps"] != list(range(0, steps, HORIZON)):
        raise ValueError("Track-B H16 execution schedule mismatch")
    if int(metadata["diagnostic_policy_queries"]) != steps:
        raise ValueError("Track-B logging was not dense")
    return metadata


def is_complete(cell: dict[str, Any]) -> bool:
    if not json_path(cell).is_file() or not npz_path(cell).is_file() or not marker_path(cell).is_file():
        return False
    try:
        validate(cell)
    except Exception:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "track_b_manifest.json")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cells = manifest["cells"]
    runtime = Runtime(args.gpu)
    assigned = [cell for index, cell in enumerate(cells) if index % args.num_workers == args.worker_index]
    progress_path = ROOT / "track_b" / "progress" / f"worker_{args.worker_index}.json"
    for cell in assigned:
        if is_complete(cell) or marker_path(cell, "technical_failed").is_file():
            continue
        apath = ROOT / "track_b" / "attempts" / f"{cell['cell_id']}.json"
        attempts = json.loads(apath.read_text(encoding="utf-8")).get("attempts", []) if apath.is_file() else []
        while len(attempts) < 3 and not is_complete(cell):
            atomic_json(progress_path, {
                "pid": os.getpid(), "gpu": args.gpu, "cell_id": cell["cell_id"],
                "attempt": len(attempts) + 1, "state": "RUNNING",
            })
            try:
                metadata, arrays = runtime.run_episode(cell, dense_logging=True)
                atomic_npz(npz_path(cell), **arrays)
                atomic_json(json_path(cell), metadata)
                validate(cell)
                marker_path(cell).parent.mkdir(parents=True, exist_ok=True)
                marker_path(cell).write_text("COMPLETE\n", encoding="utf-8")
            except Exception as exc:
                attempts.append({
                    "attempt": len(attempts) + 1, "time": time.time(),
                    "type": type(exc).__name__, "message": str(exc),
                    "traceback": traceback.format_exc(),
                })
                atomic_json(apath, {"cell_id": cell["cell_id"], "attempts": attempts})
                runtime.drop_policy()
        if not is_complete(cell):
            marker_path(cell, "technical_failed").parent.mkdir(parents=True, exist_ok=True)
            marker_path(cell, "technical_failed").write_text("TECHNICAL_FAILED\n", encoding="utf-8")
    runtime.drop_policy()
    atomic_json(progress_path, {
        "pid": os.getpid(), "gpu": args.gpu, "state": "SHARD_COMPLETE",
        "assigned_cells": len(assigned),
    })


if __name__ == "__main__":
    main()
