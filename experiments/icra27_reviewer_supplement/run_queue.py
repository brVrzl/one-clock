#!/usr/bin/env python3
"""Run one frozen reviewer-supplement phase with deterministic cell resume."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import time
import traceback
import sys
from pathlib import Path
from typing import Any

import numpy as np

from executors import DenseExecutor
from frozen_queue import ROOT, attempt_path, marker_path, phase_cells, protocol, result_path


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def mark(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")


def query_seed(cell: dict[str, Any], t: int) -> int:
    key = f"smolvla|{cell['suite']}:task{cell['task_id']}|state={cell['state_id']}|env_seed={cell['environment_seed']}|q={t}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def validate(cell: dict[str, Any], path: Path) -> dict[str, Any]:
    row = json.loads(path.read_text(encoding="utf-8"))
    for key in ("cell_id", "block_id", "phase", "policy", "suite", "task_id",
                "state_id", "environment_seed", "method", "checkpoint"):
        if row.get(key) != cell.get(key):
            raise ValueError(f"identity mismatch {key}")
    steps = int(row["environment_steps"])
    if row.get("status") != "COMPLETE" or not 1 <= steps <= int(row["resolved_max_episode_steps"]):
        raise ValueError("invalid completion")
    if row["query_steps"] != list(range(steps)) or row["policy_queries"] != steps:
        raise ValueError("dense query schedule drift")
    if len(row["executed_actions"]) != steps or len(row["sources"]) != steps:
        raise ValueError("action/source log length drift")
    for t, source in enumerate(row["sources"]):
        for label in ("translation", "rotation", "gripper"):
            if int(source[f"{label}_q"]) + int(source[f"{label}_k"]) != t:
                raise ValueError("q+k=t drift")
    return row


def complete(cell: dict[str, Any]) -> bool:
    try:
        return marker_path(cell).is_file() and bool(validate(cell, result_path(cell)))
    except Exception:
        return False


class Runtime:
    def __init__(self, gpu: str):
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env, make_env_pre_post_processors
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        self.torch, self.PreTrainedConfig, self.LiberoEnv = torch, PreTrainedConfig, LiberoEnv
        self.make_env, self.make_env_pre_post_processors = make_env, make_env_pre_post_processors
        self.make_policy, self.make_pre_post_processors = make_policy, make_pre_post_processors
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.checkpoint = None
        self.policy = self.cfg = self.preprocessor = self.postprocessor = None

    def drop(self) -> None:
        self.policy = self.cfg = self.preprocessor = self.postprocessor = None
        self.checkpoint = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def env_cfg(self, cell: dict[str, Any]):
        mapping = {"agentview_image": "image", "robot0_eye_in_hand_image":
                   ("wrist_image" if cell["suite"] == "libero_object" and
                    cell["checkpoint"].endswith("zeromidnight_act_libero_object") else "image2")}
        kwargs = dict(task=cell["suite"], task_ids=[cell["task_id"]], fps=cell["fps"],
            obs_type="pixels_agent_pos", camera_name="agentview_image,robot0_eye_in_hand_image",
            camera_name_mapping=mapping, init_states=True,
            observation_width=cell["observation_size"], observation_height=cell["observation_size"],
            control_mode="relative")
        if cell["max_episode_steps"] is not None:
            kwargs["episode_length"] = cell["max_episode_steps"]
        cfg = self.LiberoEnv(**kwargs)
        if mapping["robot0_eye_in_hand_image"] == "wrist_image":
            cfg.features_map["pixels/robot0_eye_in_hand_image"] = "observation.images.wrist_image"
        return cfg

    def load(self, cell: dict[str, Any], env_cfg) -> None:
        cp = str(Path(cell["checkpoint"]).resolve())
        if self.checkpoint == cp:
            return
        self.drop()
        if not (Path(cp) / "config.json").is_file() or not (Path(cp) / "model.safetensors").is_file():
            raise FileNotFoundError(cp)
        cfg = self.PreTrainedConfig.from_pretrained(cp)
        cfg.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        cfg.pretrained_path = Path(cp)
        expected = "act" if cell["policy"] == "ACT" else "smolvla"
        if getattr(cfg, "type", None) != expected or int(cfg.output_features["action"].shape[0]) != 7:
            raise RuntimeError("policy/action contract drift")
        if getattr(cfg, "temporal_ensemble_coeff", None) is not None:
            raise RuntimeError("policy temporal ensemble must be disabled")
        if expected == "act":
            cfg.pretrained_backbone_weights = None
            if int(cfg.chunk_size) != 100:
                raise RuntimeError("ACT chunk size drift")
        elif int(cfg.chunk_size) != 50 or int(cfg.n_action_steps) != 1:
            raise RuntimeError("SmolVLA contract drift")
        self.policy = self.make_policy(cfg=cfg, env_cfg=env_cfg)
        self.policy.eval()
        self.preprocessor, self.postprocessor = self.make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=cp,
            preprocessor_overrides={"device_processor": {"device": str(cfg.device)}})
        self.cfg, self.checkpoint = cfg, cp

    def reset_rng(self, seed: int) -> None:
        random.seed(seed); np.random.seed(seed); self.torch.manual_seed(seed)
        if self.torch.cuda.is_available(): self.torch.cuda.manual_seed_all(seed)

    def predict(self, cell, env, observation, env_pre, env_post, t):
        from lerobot.envs.utils import add_envs_task, preprocess_observation
        from lerobot.utils.constants import ACTION
        if cell["policy"] == "SmolVLA": self.reset_rng(query_seed(cell, t))
        batch = preprocess_observation(observation)
        if cell["policy"] == "SmolVLA": batch = add_envs_task(env, batch)
        batch = env_pre(batch)
        if "observation.images.image2" in batch and self.cfg.type == "act" and cell["checkpoint"].endswith("zeromidnight_act_libero_object"):
            batch["observation.images.wrist_image"] = batch.pop("observation.images.image2")
        batch = self.preprocessor(batch)
        started = time.perf_counter()
        with self.torch.inference_mode():
            chunk = self.postprocessor(self.policy.predict_action_chunk(batch))
            chunk = env_post({ACTION: chunk})[ACTION]
        value = chunk.detach().cpu().numpy()[0].astype(np.float32, copy=False)
        return value, time.perf_counter() - started

    @staticmethod
    def terminal(reward, terminated, truncated, info):
        done = bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0])
        if not done: return False, False
        final = info.get("final_info") if isinstance(info, dict) else None
        if isinstance(final, dict) and "is_success" in final:
            return True, bool(np.asarray(final["is_success"]).reshape(-1)[0])
        if isinstance(info, dict) and "is_success" in info:
            return True, bool(np.asarray(info["is_success"]).reshape(-1)[0])
        return True, bool(np.asarray(reward).reshape(-1)[0] > 0)

    def run(self, cell: dict[str, Any]) -> dict[str, Any]:
        cfg = self.env_cfg(cell); self.load(cell, cfg)
        env_pre, env_post = self.make_env_pre_post_processors(env_cfg=cfg, policy_cfg=self.cfg)
        self.reset_rng(cell["environment_seed"])
        env = self.make_env(cfg, n_envs=1, use_async_envs=False)[cell["suite"]][cell["task_id"]]
        env.envs[0].init_state_id = cell["state_id"]
        if int(env.envs[0].init_state_id) != cell["state_id"]: raise RuntimeError("state drift")
        self.reset_rng(424242 if cell["policy"] == "ACT" else query_seed(cell, 0))
        self.policy.reset(); observation, _ = env.reset(seed=[cell["environment_seed"]])
        max_steps = int(cell["max_episode_steps"] or np.asarray(env.call("_max_episode_steps")).reshape(-1)[0])
        executor = DenseExecutor(cell["method"]); actions=[]; sources=[]; latencies=[]; success=False
        started = time.time()
        try:
            for t in range(max_steps):
                chunk, latency = self.predict(cell, env, observation, env_pre, env_post, t)
                action, source = executor.step(t, chunk)
                latencies.append(latency); actions.append(action.astype(float).tolist()); sources.append(source)
                observation, reward, terminated, truncated, info = env.step(action[None])
                done, success = self.terminal(reward, terminated, truncated, info)
                if done: break
        finally:
            env.close()
        steps = len(actions)
        row = {key: cell[key] for key in ("cell_id", "block_id", "phase", "policy", "suite",
            "task_id", "state_id", "environment_seed", "method", "checkpoint")}
        row.update(status="COMPLETE", success=bool(success), environment_steps=steps,
            policy_queries=steps, model_forward_count=steps, query_rate=1.0,
            query_steps=list(range(steps)), executed_actions=actions, sources=sources,
            model_forward_seconds=latencies, wall_clock_seconds=time.time()-started,
            resolved_max_episode_steps=max_steps, chunk_size=int(self.cfg.chunk_size),
            action_dim=7, fresh_environment_per_condition=True, finished_at=time.time())
        return row


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", required=True)
    ap.add_argument("--gpu", required=True); ap.add_argument("--worker-index", type=int, required=True)
    ap.add_argument("--num-workers", type=int, default=3); args = ap.parse_args()
    if args.phase not in {"r1a", "r1b", "r1c", "r1d", "r2"}: raise SystemExit("bad phase")
    if args.phase == "r1d":
        sys.path.insert(0, "/home/wjq/workspace/upstreams/lerobot/src")
    rows = phase_cells(args.phase)
    tasks = sorted({(r["suite"], r["task_id"]) for r in rows})
    owned = {task for i, task in enumerate(tasks) if i % args.num_workers == args.worker_index}
    rows = [r for r in rows if (r["suite"], r["task_id"]) in owned]
    runtime = Runtime(args.gpu); progress = ROOT / "progress" / f"{args.phase}_worker_{args.worker_index}.json"
    for cell in rows:
        if complete(cell): continue
        if marker_path(cell, "technical_failed").is_file(): continue
        apath = attempt_path(cell)
        attempts = json.loads(apath.read_text()).get("attempts", []) if apath.is_file() else []
        while len(attempts) < int(protocol()["maximum_attempts"]) and not complete(cell):
            atomic_json(progress, {"pid": os.getpid(), "gpu": args.gpu, "phase": args.phase,
                "cell_id": cell["cell_id"], "attempt": len(attempts)+1, "state": "RUNNING"})
            try:
                atomic_json(result_path(cell), runtime.run(cell)); validate(cell, result_path(cell)); mark(marker_path(cell), "COMPLETE")
            except Exception as exc:
                attempts.append({"attempt": len(attempts)+1, "time": time.time(), "type": type(exc).__name__,
                    "message": str(exc), "traceback": traceback.format_exc()})
                atomic_json(apath, {"cell_id": cell["cell_id"], "attempts": attempts}); runtime.drop()
        if not complete(cell): mark(marker_path(cell, "technical_failed"), "TECHNICAL_FAILED")
    runtime.drop(); atomic_json(progress, {"pid": os.getpid(), "gpu": args.gpu, "phase": args.phase,
        "state": "SHARD_COMPLETE", "owned_cells": len(rows)})


if __name__ == "__main__": main()
