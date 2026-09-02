#!/usr/bin/env python3
"""Run the frozen Gate M and SmolVLA robustness queues with blockwise resume."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
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
from one_clock import ActionGroup, FixedChunkExecutor  # noqa: E402
from scheduling import gate_horizon, shuffled_horizon  # noqa: E402


ARM = tuple(range(6))
GRIPPER = (6,)
PHASE_ORDER = ("gate_m", "smolvla_robustness")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def result_path(cell: dict) -> Path:
    return ROOT / "results" / cell["phase"] / f"{cell['cell_id']}.json"


def marker_path(cell: dict, status: str = "complete") -> Path:
    return ROOT / "markers" / cell["phase"] / f"{cell['cell_id']}.{status}"


def write_marker(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def expected_gate_query_steps(horizons: list[int], environment_steps: int) -> list[int]:
    query_steps = []
    physical_step = 0
    for horizon in horizons:
        if physical_step >= environment_steps:
            raise ValueError("horizon exists after terminal step")
        query_steps.append(physical_step)
        physical_step += int(horizon)
    if physical_step < environment_steps:
        raise ValueError("horizon schedule ends before executed actions")
    return query_steps


def validate_result(cell: dict, path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    exact = (
        "cell_id", "block_id", "phase", "policy", "suite", "task_id", "state_id",
        "environment_seed", "method", "checkpoint",
    )
    for key in exact:
        if value.get(key) != cell.get(key):
            raise ValueError(f"{key} mismatch: {value.get(key)!r} != {cell.get(key)!r}")
    if value.get("status") != "COMPLETE":
        raise ValueError("result status is not COMPLETE")
    steps = int(value["environment_steps"])
    queries = int(value["policy_queries"])
    if not 1 <= steps <= int(value["resolved_max_episode_steps"]):
        raise ValueError("invalid environment step count")
    if queries < 1 or queries != int(value["model_forward_count"]):
        raise ValueError("invalid query/model-forward count")
    if len(value.get("executed_actions", [])) != steps:
        raise ValueError("executed action count mismatch")
    if len(value.get("source_ages", [])) != steps:
        raise ValueError("source-age count mismatch")
    if len(value.get("query_steps", [])) != queries:
        raise ValueError("query-step count mismatch")
    if cell["phase"] == "gate_m":
        horizons = [int(x) for x in value.get("execution_horizons", [])]
        diagnostics = value.get("query_diagnostics", [])
        if len(horizons) != queries or len(diagnostics) != queries:
            raise ValueError("Gate M horizon/diagnostic count mismatch")
        if value["query_steps"] != expected_gate_query_steps(horizons, steps):
            raise ValueError("Gate M query schedule mismatch")
        for index, (horizon, diagnostic) in enumerate(zip(horizons, diagnostics, strict=True)):
            if not 4 <= horizon <= 16:
                raise ValueError("Gate M horizon outside frozen bounds")
            method = cell["method"]
            if method == "M0_HARD16" and horizon != 16:
                raise ValueError("M0 horizon drift")
            if method == "FIXED_H13" and horizon != 13:
                raise ValueError("FIXED_H13 horizon drift")
            if method == "SHUFFLED_TRIGGER":
                expected, _ = shuffled_horizon(cell, index)
                if horizon != expected:
                    raise ValueError("shuffled schedule drift")
            if method == "M2_GRIPPER_EVENT":
                intents = [int(x) for x in diagnostic["gripper_intents_first_16"]]
                events = [k for k in range(4, 16) if intents[k] != intents[k - 1]]
                expected = events[0] if events else 16
                if horizon != expected or diagnostic["gripper_event_candidates"] != events:
                    raise ValueError("historical M2 detector semantics drift")
        for t, ages in enumerate(value["source_ages"]):
            q = max(q for q in value["query_steps"] if q <= t)
            expected_age = t - q
            if int(ages["arm"]) != expected_age or int(ages["gripper"]) != expected_age:
                raise ValueError("Gate M coherent source-age drift")
    else:
        if value["query_steps"] != list(range(0, steps, 4)):
            raise ValueError("SmolVLA arm-driven query schedule mismatch")
        if int(value["chunk_size"]) != 50 or int(value["n_action_steps"]) != 1:
            raise ValueError("SmolVLA checkpoint contract drift")
    return value


def is_complete(cell: dict) -> bool:
    path = result_path(cell)
    if not path.is_file() or not marker_path(cell).is_file():
        return False
    try:
        validate_result(cell, path)
    except Exception:
        return False
    return True


def query_seed(cell: dict, physical_query_step: int) -> int:
    key = (
        f"smolvla|{cell['suite']}:task{cell['task_id']}|state={cell['state_id']}|"
        f"env_seed={cell['environment_seed']}|q={physical_query_step}"
    )
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)


def reset_torch_rng(torch, seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def make_smol_executor(cell: dict, chunk_size: int) -> FixedChunkExecutor:
    groups = (
        ActionGroup("arm", ARM, int(cell["arm_horizon"])),
        ActionGroup("gripper", GRIPPER, int(cell["gripper_horizon"])),
    )
    return FixedChunkExecutor.groupwise_fixed(action_dim=7, chunk_size=chunk_size, groups=groups)


class Runtime:
    def __init__(self, gpu: str):
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env, make_env_pre_post_processors
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        self.torch = torch
        self.PreTrainedConfig = PreTrainedConfig
        self.LiberoEnv = LiberoEnv
        self.make_env = make_env
        self.make_env_pre_post_processors = make_env_pre_post_processors
        self.make_policy = make_policy
        self.make_pre_post_processors = make_pre_post_processors
        self.checkpoint: str | None = None
        self.policy = None
        self.cfg = None
        self.preprocessor = None
        self.postprocessor = None

    @staticmethod
    def historical_object_checkpoint(cell: dict) -> bool:
        return str(cell["checkpoint"]).endswith("/zeromidnight_act_libero_object")

    def env_config(self, cell: dict):
        historical_object = self.historical_object_checkpoint(cell)
        camera_mapping = (
            {"agentview_image": "image", "robot0_eye_in_hand_image": "wrist_image"}
            if historical_object
            else {"agentview_image": "image", "robot0_eye_in_hand_image": "image2"}
        )
        env_cfg = self.LiberoEnv(
            task=cell["suite"], task_ids=[int(cell["task_id"])],
            fps=int(cell["control_frequency_hz"]), obs_type="pixels_agent_pos",
            camera_name="agentview_image,robot0_eye_in_hand_image",
            camera_name_mapping=camera_mapping, init_states=True,
            observation_width=256, observation_height=256, control_mode="relative",
        )
        if historical_object:
            env_cfg.features_map["pixels/robot0_eye_in_hand_image"] = "observation.images.wrist_image"
        return env_cfg

    def policy_for(self, cell: dict, env_cfg) -> None:
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
        expected = "act" if cell["policy"] == "ACT" else "smolvla"
        if getattr(cfg, "type", None) != expected:
            raise RuntimeError(f"expected {expected}, got {getattr(cfg, 'type', None)}")
        if getattr(cfg, "temporal_ensemble_coeff", None) is not None:
            raise RuntimeError("temporal aggregation must be disabled")
        if int(cfg.output_features["action"].shape[0]) != 7:
            raise RuntimeError("action_dim must be 7")
        if expected == "act":
            cfg.pretrained_backbone_weights = None
        else:
            if int(cfg.chunk_size) != 50 or int(cfg.n_action_steps) != 1:
                raise RuntimeError("SmolVLA chunk_size/n_action_steps contract drift")
        self.policy = self.make_policy(cfg=cfg, env_cfg=env_cfg)
        self.policy.eval()
        self.preprocessor, self.postprocessor = self.make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(cfg.device)}},
        )
        self.cfg = cfg
        self.checkpoint = checkpoint

    def drop_policy(self) -> None:
        self.policy = self.cfg = self.preprocessor = self.postprocessor = None
        self.checkpoint = None
        gc.collect()
        if hasattr(self, "torch") and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def prepare(self, cell: dict):
        env_cfg = self.env_config(cell)
        self.policy_for(cell, env_cfg)
        assert self.policy is not None and self.cfg is not None
        env_pre, env_post = self.make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=self.cfg)
        env = self.make_env(env_cfg, n_envs=1, use_async_envs=False)[cell["suite"]][int(cell["task_id"])]
        env.envs[0].init_state_id = int(cell["state_id"])
        if int(env.envs[0].init_state_id) != int(cell["state_id"]):
            env.close()
            raise RuntimeError("initial state assignment mismatch")
        reset_torch_rng(self.torch, 424242 if cell["policy"] == "ACT" else query_seed(cell, 0))
        self.policy.reset()
        observation, _ = env.reset(seed=[int(cell["environment_seed"])])
        return env, env_pre, env_post, observation

    def predict_chunk(self, cell: dict, env, observation, env_pre, env_post, physical_step: int):
        from lerobot.envs.utils import add_envs_task, preprocess_observation
        from lerobot.utils.constants import ACTION

        assert self.policy is not None and self.cfg is not None
        if cell["policy"] == "SmolVLA":
            reset_torch_rng(self.torch, query_seed(cell, physical_step))
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
        batch = self.preprocessor(batch)
        started = time.perf_counter()
        with self.torch.inference_mode():
            chunk = self.postprocessor(self.policy.predict_action_chunk(batch))
            chunk = env_post({ACTION: chunk})[ACTION]
        latency = time.perf_counter() - started
        value = chunk.detach().cpu().numpy().astype(np.float32, copy=False)
        if value.shape != (1, int(self.cfg.chunk_size), 7):
            raise RuntimeError(f"unexpected chunk shape {value.shape}")
        return value[0], latency

    @staticmethod
    def success_from_step(reward, terminated, truncated, info) -> tuple[bool, bool]:
        done = bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0])
        if not done:
            return False, False
        final = info.get("final_info") if isinstance(info, dict) else None
        if isinstance(final, dict) and "is_success" in final:
            return True, bool(np.asarray(final["is_success"]).reshape(-1)[0])
        return True, bool(np.asarray(reward).reshape(-1)[0] > 0)

    def run(self, cell: dict) -> dict:
        env, env_pre, env_post, observation = self.prepare(cell)
        started = time.time()
        try:
            max_steps = int(cell["max_episode_steps"] or np.asarray(env.call("_max_episode_steps")).reshape(-1)[0])
            query_steps: list[int] = []
            query_latencies: list[float] = []
            actions: list[list[float]] = []
            source_ages: list[dict[str, int]] = []
            horizons: list[int] = []
            diagnostics: list[dict] = []
            success = False
            task_name = str(env.envs[0].task)
            if cell["phase"] == "gate_m":
                active_chunk = None
                active_query_step = -1
                chunk_offset = 0
                remaining = 0
                for t in range(max_steps):
                    if remaining == 0:
                        active_chunk, latency = self.predict_chunk(cell, env, observation, env_pre, env_post, t)
                        query_latencies.append(latency)
                        active_query_step = t
                        chunk_offset = 0
                        horizon, diagnostic = gate_horizon(cell["method"], active_chunk, cell, len(query_steps))
                        query_steps.append(t)
                        horizons.append(int(horizon))
                        diagnostics.append(jsonable(diagnostic))
                        remaining = int(horizon)
                    assert active_chunk is not None
                    action = active_chunk[chunk_offset].astype(np.float32, copy=False)
                    actions.append(action.astype(float).tolist())
                    age = t - active_query_step
                    source_ages.append({"arm": age, "gripper": age})
                    observation, reward, terminated, truncated, info = env.step(action[None])
                    chunk_offset += 1
                    remaining -= 1
                    done, success = self.success_from_step(reward, terminated, truncated, info)
                    if done:
                        break
            else:
                assert self.cfg is not None
                executor = make_smol_executor(cell, int(self.cfg.chunk_size))
                for t in range(max_steps):
                    def query():
                        chunk, latency = self.predict_chunk(cell, env, observation, env_pre, env_post, t)
                        query_latencies.append(latency)
                        return chunk

                    decision = executor.step(query)
                    if decision.policy_query:
                        query_steps.append(t)
                    action = decision.action.astype(np.float32, copy=False)
                    actions.append(action.astype(float).tolist())
                    source_ages.append({k: int(v) for k, v in decision.source_ages.items()})
                    observation, reward, terminated, truncated, info = env.step(action[None])
                    done, success = self.success_from_step(reward, terminated, truncated, info)
                    if done:
                        break
            steps = len(actions)
            wall = time.time() - started
            result = {k: cell[k] for k in (
                "cell_id", "block_id", "phase", "policy", "suite", "task_id", "state_id",
                "environment_seed", "method", "checkpoint",
            )}
            result.update({
                "status": "COMPLETE",
                "task_name": task_name,
                "success": bool(success),
                "completion_length": steps if success else None,
                "environment_steps": steps,
                "policy_queries": len(query_steps),
                "model_forward_count": len(query_steps),
                "query_rate": len(query_steps) / steps,
                "query_steps": query_steps,
                "execution_horizons": horizons,
                "query_diagnostics": diagnostics,
                "wall_clock_seconds": wall,
                "model_forward_seconds": query_latencies,
                "source_ages": source_ages,
                "executed_actions": actions,
                "resolved_max_episode_steps": max_steps,
                "chunk_size": int(self.cfg.chunk_size),
                "n_action_steps": int(self.cfg.n_action_steps),
                "action_dim": 7,
                "temporal_aggregation": False,
                "smoothing": False,
                "fresh_environment_per_condition_block": True,
                "finished_at": time.time(),
            })
            if cell["phase"] == "smolvla_robustness":
                result.update({
                    "strategy": cell["strategy"],
                    "arm_horizon": int(cell["arm_horizon"]),
                    "gripper_horizon": int(cell["gripper_horizon"]),
                    "scope": "CROSS_POLICY_ROBUSTNESS",
                })
            return result
        finally:
            env.close()

    def preflight(self, cell: dict) -> dict:
        """Construct, reset, and predict once without stepping or observing an outcome."""

        env, env_pre, env_post, observation = self.prepare(cell)
        try:
            chunk, latency = self.predict_chunk(cell, env, observation, env_pre, env_post, 0)
            return {
                "pid": os.getpid(),
                "policy": cell["policy"],
                "checkpoint": cell["checkpoint"],
                "suite": cell["suite"],
                "task_id": cell["task_id"],
                "state_id": cell["state_id"],
                "environment_seed": cell["environment_seed"],
                "chunk_shape": list(chunk.shape),
                "chunk_size": int(self.cfg.chunk_size),
                "n_action_steps": int(self.cfg.n_action_steps),
                "action_dim": int(self.cfg.output_features["action"].shape[0]),
                "temporal_ensemble_coeff": getattr(self.cfg, "temporal_ensemble_coeff", None),
                "first_forward_seconds": latency,
                "environment_steps": 0,
                "outcome_observed": False,
            }
        finally:
            env.close()


def terminal_count(cells: list[dict], phase: str) -> int:
    return sum(
        is_complete(cell) or marker_path(cell, "technical_failed").is_file()
        for cell in cells if cell["phase"] == phase
    )


def phase_blocks(cells: list[dict], phase: str) -> list[list[dict]]:
    blocks: dict[str, list[dict]] = defaultdict(list)
    for cell in cells:
        if cell["phase"] == phase:
            blocks[cell["block_id"]].append(cell)
    return [blocks[key] for key in sorted(blocks)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "queue_manifest.json")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--phases", nargs="*", default=list(PHASE_ORDER))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cells = manifest["cells"]
    runtime = Runtime(args.gpu)
    progress_path = ROOT / "progress" / f"worker_{args.worker_index}.json"
    for phase in PHASE_ORDER:
        if phase not in args.phases:
            continue
        if phase == "smolvla_robustness":
            gate_total = int(manifest["expected_counts"]["gate_m"])
            while terminal_count(cells, "gate_m") < gate_total:
                atomic_json(progress_path, {
                    "pid": os.getpid(), "gpu": args.gpu, "phase": phase,
                    "state": "WAITING_FOR_GATE_M_TERMINAL_BARRIER",
                    "gate_m_terminal": terminal_count(cells, "gate_m"), "gate_m_total": gate_total,
                })
                time.sleep(30)
        blocks = phase_blocks(cells, phase)
        assigned = [block for index, block in enumerate(blocks) if index % args.num_workers == args.worker_index]
        for block in assigned:
            for cell in block:
                if is_complete(cell) or marker_path(cell, "technical_failed").is_file():
                    continue
                attempt_path = ROOT / "attempts" / phase / f"{cell['cell_id']}.json"
                attempts = json.loads(attempt_path.read_text(encoding="utf-8")).get("attempts", []) if attempt_path.is_file() else []
                while len(attempts) < 3 and not is_complete(cell):
                    atomic_json(progress_path, {
                        "pid": os.getpid(), "gpu": args.gpu, "phase": phase,
                        "block_id": cell["block_id"], "cell_id": cell["cell_id"],
                        "attempt": len(attempts) + 1, "state": "RUNNING",
                    })
                    try:
                        value = runtime.run(cell)
                        atomic_json(result_path(cell), value)
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
                        atomic_json(attempt_path, {"cell_id": cell["cell_id"], "attempts": attempts})
                        runtime.drop_policy()
                if not is_complete(cell):
                    write_marker(marker_path(cell, "technical_failed"), "TECHNICAL_FAILED")
        atomic_json(progress_path, {
            "pid": os.getpid(), "gpu": args.gpu, "phase": phase,
            "state": "PHASE_SHARD_COMPLETE", "terminal": terminal_count(cells, phase),
            "total": int(manifest["expected_counts"][phase]),
        })
    runtime.drop_policy()
    atomic_json(progress_path, {
        "pid": os.getpid(), "gpu": args.gpu, "state": "ALL_REQUESTED_PHASES_COMPLETE",
    })


if __name__ == "__main__":
    main()
