#!/usr/bin/env python3
"""Run the preregistered six-condition Track-A queue with blockwise resume."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(ROOT))

from conditions import (  # noqa: E402
    ACTION_DIM,
    CONDITION_ORDER,
    CONDITIONS,
    TE_COEFFICIENT,
    make_fixed_executor,
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_marker(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def result_path(cell: dict[str, Any]) -> Path:
    return ROOT / "track_a" / "results" / f"{cell['cell_id']}.json"


def marker_path(cell: dict[str, Any], status: str = "complete") -> Path:
    return ROOT / "track_a" / "markers" / f"{cell['cell_id']}.{status}"


def attempt_path(cell: dict[str, Any]) -> Path:
    return ROOT / "track_a" / "attempts" / f"{cell['cell_id']}.json"


def validate_result(cell: dict[str, Any], path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "cell_id", "block_id", "suite", "task_id", "state_id", "environment_seed",
        "method", "checkpoint", "preregistration_commit",
    ):
        if value.get(key) != cell.get(key):
            raise ValueError(f"{key} mismatch: {value.get(key)!r} != {cell.get(key)!r}")
    if value.get("status") != "COMPLETE":
        raise ValueError("result status is not COMPLETE")
    steps = int(value["environment_steps"])
    queries = int(value["policy_queries"])
    if not 1 <= steps <= int(value["resolved_max_episode_steps"]):
        raise ValueError("invalid environment step count")
    if queries != int(value["model_forward_count"]) or queries < 1:
        raise ValueError("policy query/model-forward count mismatch")
    if len(value.get("executed_actions", [])) != steps:
        raise ValueError("executed action count mismatch")
    if len(value.get("query_steps", [])) != queries:
        raise ValueError("query-step count mismatch")
    method = cell["method"]
    if method == "TE_DENSE":
        if value["query_steps"] != list(range(steps)):
            raise ValueError("TE_DENSE did not query at every environment step")
        if value.get("temporal_ensemble_coeff") != TE_COEFFICIENT:
            raise ValueError("TE coefficient drift")
        if len(value.get("candidate_counts", [])) != steps:
            raise ValueError("TE candidate-count log mismatch")
    else:
        period = min(int(cell["arm_horizon"]), int(cell["gripper_horizon"]))
        if value["query_steps"] != list(range(0, steps, period)):
            raise ValueError("fixed-condition query schedule mismatch")
        if len(value.get("source_ages", [])) != steps:
            raise ValueError("fixed-condition source-age log mismatch")
    return value


def is_complete(cell: dict[str, Any]) -> bool:
    path = result_path(cell)
    if not path.is_file() or not marker_path(cell).is_file():
        return False
    try:
        validate_result(cell, path)
    except Exception:
        return False
    return True


def reset_rng(torch: Any, seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sim_state_snapshot(env: Any) -> list[float] | None:
    try:
        return np.asarray(env.envs[0]._env.get_sim_state()).astype(float).tolist()
    except (AttributeError, TypeError):
        return None


class Runtime:
    def __init__(self, gpu: str):
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # Freeze deterministic inference before torch/CUDA initialization.  This
        # is required because paired conditions are separate environment resets.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env, make_env_pre_post_processors
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        self.torch = torch
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        self.PreTrainedConfig = PreTrainedConfig
        self.LiberoEnv = LiberoEnv
        self.make_env = make_env
        self.make_env_pre_post_processors = make_env_pre_post_processors
        self.ACTTemporalEnsembler = ACTTemporalEnsembler
        self.make_policy = make_policy
        self.make_pre_post_processors = make_pre_post_processors
        self.checkpoint: str | None = None
        self.policy = None
        self.cfg = None
        self.preprocessor = None
        self.postprocessor = None

    def drop_policy(self) -> None:
        self.policy = self.cfg = self.preprocessor = self.postprocessor = None
        self.checkpoint = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def env_config(self, cell: dict[str, Any]):
        return self.LiberoEnv(
            task=cell["suite"],
            task_ids=[int(cell["task_id"])],
            fps=int(cell["control_frequency_hz"]),
            obs_type="pixels_agent_pos",
            camera_name="agentview_image,robot0_eye_in_hand_image",
            camera_name_mapping={
                "agentview_image": "image",
                "robot0_eye_in_hand_image": "image2",
            },
            init_states=True,
            observation_width=256,
            observation_height=256,
            control_mode="relative",
        )

    def policy_for(self, cell: dict[str, Any], env_cfg: Any) -> None:
        checkpoint = str(Path(cell["checkpoint"]).resolve())
        if self.checkpoint == checkpoint:
            return
        self.drop_policy()
        cp = Path(checkpoint)
        if not (cp / "config.json").is_file() or not (cp / "model.safetensors").is_file():
            raise FileNotFoundError(f"checkpoint missing required files: {cp}")
        cfg = self.PreTrainedConfig.from_pretrained(cp)
        cfg.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        cfg.pretrained_path = cp
        cfg.pretrained_backbone_weights = None
        if getattr(cfg, "type", None) != "act":
            raise RuntimeError(f"expected ACT checkpoint, got {getattr(cfg, 'type', None)!r}")
        if getattr(cfg, "temporal_ensemble_coeff", None) is not None:
            raise RuntimeError("checkpoint policy must have temporal aggregation disabled")
        if int(cfg.output_features["action"].shape[0]) != ACTION_DIM:
            raise RuntimeError("ACT action dimension is not 7")
        if int(cfg.chunk_size) < 32:
            raise RuntimeError("ACT chunk is shorter than ARM4_GRIP32")
        self.policy = self.make_policy(cfg=cfg, env_cfg=env_cfg)
        self.policy.eval()
        self.preprocessor, self.postprocessor = self.make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(cfg.device)}},
        )
        self.cfg = cfg
        self.checkpoint = checkpoint

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
        reset_rng(self.torch, int(cell["policy_seed"]))
        self.policy.reset()
        observation, _ = env.reset(seed=[int(cell["environment_seed"])])
        return env, env_pre, env_post, observation

    def processed_input(self, env: Any, observation: dict[str, Any], env_pre: Any):
        from lerobot.envs.utils import add_envs_task, preprocess_observation

        batch = preprocess_observation(observation)
        batch = add_envs_task(env, batch)
        batch = env_pre(batch)
        assert self.preprocessor is not None
        return self.preprocessor(batch)

    def predict_normalized_chunk(self, env: Any, observation: dict[str, Any], env_pre: Any):
        assert self.policy is not None and self.cfg is not None
        batch = self.processed_input(env, observation, env_pre)
        started = time.perf_counter()
        with self.torch.inference_mode():
            chunk = self.policy.predict_action_chunk(batch)
        latency = time.perf_counter() - started
        if tuple(chunk.shape) != (1, int(self.cfg.chunk_size), ACTION_DIM):
            raise RuntimeError(f"unexpected normalized ACT chunk shape {tuple(chunk.shape)}")
        return chunk, latency

    def postprocess(self, value: Any, env_post: Any):
        from lerobot.utils.constants import ACTION

        assert self.postprocessor is not None
        with self.torch.inference_mode():
            value = self.postprocessor(value)
            value = env_post({ACTION: value})[ACTION]
        return value

    @staticmethod
    def success_from_step(reward: Any, terminated: Any, truncated: Any, info: Any) -> tuple[bool, bool]:
        done = bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0])
        if not done:
            return False, False
        final = info.get("final_info") if isinstance(info, dict) else None
        if isinstance(final, dict) and "is_success" in final:
            return True, bool(np.asarray(final["is_success"]).reshape(-1)[0])
        return True, bool(np.asarray(reward).reshape(-1)[0] > 0)

    def run(self, cell: dict[str, Any], *, executor_override: Any | None = None) -> dict[str, Any]:
        env, env_pre, env_post, observation = self.prepare(cell)
        started = time.perf_counter()
        try:
            max_steps = int(cell["max_episode_steps"] or np.asarray(env.call("_max_episode_steps")).reshape(-1)[0])
            method = str(cell["method"])
            query_steps: list[int] = []
            query_latencies: list[float] = []
            executed_actions: list[list[float]] = []
            source_ages: list[dict[str, int]] = []
            candidate_counts: list[int] = []
            initial_sim_state = sim_state_snapshot(env)
            success = False

            if method == "TE_DENSE":
                assert self.cfg is not None
                ensembler = self.ACTTemporalEnsembler(TE_COEFFICIENT, int(self.cfg.chunk_size))
                for t in range(max_steps):
                    normalized_chunk, latency = self.predict_normalized_chunk(env, observation, env_pre)
                    query_steps.append(t)
                    query_latencies.append(latency)
                    with self.torch.inference_mode():
                        normalized_action = ensembler.update(normalized_chunk)
                    action_tensor = self.postprocess(normalized_action, env_post)
                    action = action_tensor.detach().cpu().numpy()[0].astype(np.float32, copy=False)
                    executed_actions.append(action.astype(float).tolist())
                    candidate_counts.append(min(t + 1, int(self.cfg.chunk_size)))
                    observation, reward, terminated, truncated, info = env.step(action[None])
                    done, success = self.success_from_step(reward, terminated, truncated, info)
                    if done:
                        break
            else:
                assert self.cfg is not None
                executor = (
                    executor_override(int(self.cfg.chunk_size))
                    if executor_override is not None
                    else make_fixed_executor(method, int(self.cfg.chunk_size))
                )
                for t in range(max_steps):
                    def query() -> np.ndarray:
                        normalized_chunk, latency = self.predict_normalized_chunk(env, observation, env_pre)
                        query_latencies.append(latency)
                        chunk = self.postprocess(normalized_chunk, env_post)
                        return chunk.detach().cpu().numpy()[0].astype(np.float32, copy=False)

                    decision = executor.step(query)
                    if decision.policy_query:
                        query_steps.append(t)
                    action = decision.action.astype(np.float32, copy=False)
                    executed_actions.append(action.astype(float).tolist())
                    source_ages.append({k: int(v) for k, v in decision.source_ages.items()})
                    observation, reward, terminated, truncated, info = env.step(action[None])
                    done, success = self.success_from_step(reward, terminated, truncated, info)
                    if done:
                        break

            steps = len(executed_actions)
            result = {key: cell[key] for key in (
                "cell_id", "block_id", "suite", "task_id", "state_id", "environment_seed",
                "policy_seed", "method", "strategy", "arm_horizon", "gripper_horizon",
                "checkpoint", "preregistration_commit",
            )}
            result.update({
                "status": "COMPLETE",
                "success": bool(success),
                "environment_steps": steps,
                "policy_queries": len(query_steps),
                "model_forward_count": len(query_steps),
                "query_rate": len(query_steps) / steps,
                "query_steps": query_steps,
                "wall_clock_seconds": time.perf_counter() - started,
                "model_forward_seconds": query_latencies,
                "mean_model_forward_seconds": float(np.mean(query_latencies)),
                "source_ages": source_ages,
                "candidate_counts": candidate_counts,
                "executed_actions": executed_actions,
                "initial_sim_state": initial_sim_state,
                "resolved_max_episode_steps": max_steps,
                "chunk_size": int(self.cfg.chunk_size),
                "n_action_steps": int(self.cfg.n_action_steps),
                "action_dim": ACTION_DIM,
                "temporal_ensemble_coeff": TE_COEFFICIENT if method == "TE_DENSE" else None,
                "temporal_ensemble_space": "checkpoint-normalized action space" if method == "TE_DENSE" else None,
                "postprocessing_order": "aggregate-normalized-then-policy-denormalize-then-env-postprocess" if method == "TE_DENSE" else "policy-denormalize-chunk-then-env-postprocess-then-execute",
                "fresh_environment_per_condition": True,
                "finished_at": time.time(),
            })
            return result
        finally:
            env.close()


def verify_preregistration(manifest: dict[str, Any]) -> str:
    sha = str(manifest.get("preregistration_commit", ""))
    if len(sha) != 40:
        raise RuntimeError("manifest lacks a resolved 40-character preregistration commit")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", sha, "HEAD"],
        cwd=REPO_ROOT,
        check=True,
    )
    if manifest.get("outcomes_inspected_before_freeze") is not False:
        raise RuntimeError("manifest does not certify outcome-blind freeze")
    if tuple(manifest.get("condition_order", ())) != CONDITION_ORDER:
        raise RuntimeError("condition order drifted")
    return sha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "track_a_manifest.json")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=3)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    verify_preregistration(manifest)
    cells = manifest["cells"]
    blocks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        blocks[cell["block_id"]].append(cell)
    for block_cells in blocks.values():
        if tuple(cell["method"] for cell in block_cells) != CONDITION_ORDER:
            raise RuntimeError(f"incomplete or reordered paired block {block_cells[0]['block_id']}")
    tasks: dict[tuple[str, int, str], list[str]] = defaultdict(list)
    for block_id, block_cells in blocks.items():
        first = block_cells[0]
        tasks[(first["suite"], int(first["task_id"]), first["checkpoint"])].append(block_id)
    assigned_tasks = [
        key for index, key in enumerate(sorted(tasks))
        if index % args.num_workers == args.worker_index
    ]
    runtime = Runtime(args.gpu)
    progress_path = ROOT / "track_a" / "progress" / f"worker_{args.worker_index}.json"
    for task_key in assigned_tasks:
        for block_id in sorted(tasks[task_key]):
            for cell in blocks[block_id]:
                if is_complete(cell) or marker_path(cell, "technical_failed").is_file():
                    continue
                apath = attempt_path(cell)
                attempts = json.loads(apath.read_text(encoding="utf-8")).get("attempts", []) if apath.is_file() else []
                while len(attempts) < 3 and not is_complete(cell):
                    atomic_json(progress_path, {
                        "pid": os.getpid(), "gpu": args.gpu, "worker_index": args.worker_index,
                        "task": {"suite": task_key[0], "task_id": task_key[1], "checkpoint": task_key[2]},
                        "block_id": block_id, "cell_id": cell["cell_id"],
                        "attempt": len(attempts) + 1, "state": "RUNNING",
                    })
                    try:
                        atomic_json(result_path(cell), runtime.run(cell))
                        validate_result(cell, result_path(cell))
                        write_marker(marker_path(cell), "COMPLETE")
                    except Exception as exc:
                        attempts.append({
                            "attempt": len(attempts) + 1,
                            "time": time.time(),
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        })
                        atomic_json(apath, {"cell_id": cell["cell_id"], "attempts": attempts})
                        runtime.drop_policy()
                if not is_complete(cell):
                    write_marker(marker_path(cell, "technical_failed"), "TECHNICAL_FAILED")
        # The task-major contract releases the checkpoint before the next task.
        runtime.drop_policy()
    runtime.drop_policy()
    atomic_json(progress_path, {
        "pid": os.getpid(), "gpu": args.gpu, "worker_index": args.worker_index,
        "state": "SHARD_COMPLETE", "assigned_tasks": len(assigned_tasks),
        "assigned_blocks": sum(len(tasks[key]) for key in assigned_tasks),
    })


if __name__ == "__main__":
    main()
