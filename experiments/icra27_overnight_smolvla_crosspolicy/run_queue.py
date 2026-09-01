#!/usr/bin/env python3
"""Run frozen ACT and SmolVLA fixed-clock cells with durable per-cell resume."""

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

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from one_clock import ActionGroup, FixedChunkExecutor  # noqa: E402

ARM = tuple(range(6))
GRIPPER = (6,)
PHASE_ORDER = (
    "act_object_h8_126",
    "act_posthoc_h8_140",
    "act_arm4_grip32_180",
    "smolvla_primary",
    "smolvla_capacity_h16",
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def result_path(cell: dict) -> Path:
    return ROOT / "results" / cell["phase"] / f"{cell['cell_id']}.json"


def marker_path(cell: dict, status: str = "complete") -> Path:
    return ROOT / "markers" / cell["phase"] / f"{cell['cell_id']}.{status}"


def validate_result(cell: dict, path: Path) -> dict:
    value = json.loads(path.read_text())
    exact = ("cell_id", "phase", "policy", "suite", "task_id", "state_id",
             "environment_seed", "method", "arm_horizon", "gripper_horizon", "checkpoint")
    for key in exact:
        if value.get(key) != cell.get(key):
            raise ValueError(f"{key} mismatch: {value.get(key)!r} != {cell.get(key)!r}")
    if value.get("status") != "COMPLETE":
        raise ValueError("result status is not COMPLETE")
    steps, queries, forwards = (int(value[k]) for k in ("environment_steps", "policy_queries", "model_forward_count"))
    if steps < 1 or queries < 1 or queries != forwards:
        raise ValueError("invalid step/query/forward counts")
    if len(value.get("executed_actions", [])) != steps:
        raise ValueError("executed action count mismatch")
    if len(value.get("source_ages", [])) != steps:
        raise ValueError("source age count mismatch")
    expected_period = min(int(cell["arm_horizon"]), int(cell["gripper_horizon"]))
    if value.get("query_steps") != list(range(0, steps, expected_period)):
        raise ValueError("query schedule mismatch")
    return value


def is_complete(cell: dict) -> bool:
    rp = result_path(cell)
    mp = marker_path(cell)
    if not rp.is_file() or not mp.is_file():
        return False
    try:
        validate_result(cell, rp)
    except Exception:
        return False
    return True


def query_seed(cell: dict, q: int) -> int:
    key = (
        f"smolvla|{cell['suite']}:task{cell['task_id']}|state={cell['state_id']}|"
        f"env_seed={cell['environment_seed']}|q={q}"
    )
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def reset_torch_rng(torch, seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def make_executor(cell: dict, chunk_size: int) -> FixedChunkExecutor:
    groups = (
        ActionGroup("arm", ARM, int(cell["arm_horizon"])),
        ActionGroup("gripper", GRIPPER, int(cell["gripper_horizon"])),
    )
    if cell["strategy"] == "global_fixed":
        return FixedChunkExecutor.global_fixed(
            action_dim=7, chunk_size=chunk_size,
            horizon=int(cell["arm_horizon"]), groups=groups,
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

    def policy_for(self, cell: dict, env_cfg):
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
        if int(cfg.chunk_size) < max(int(cell["arm_horizon"]), int(cell["gripper_horizon"])):
            raise RuntimeError("checkpoint chunk is shorter than configured horizon")
        self.policy = self.make_policy(cfg=cfg, env_cfg=env_cfg)
        self.policy.eval()
        self.preprocessor, self.postprocessor = self.make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(cfg.device)}},
        )
        self.cfg = cfg
        self.checkpoint = checkpoint

    def drop_policy(self):
        self.policy = self.cfg = self.preprocessor = self.postprocessor = None
        self.checkpoint = None
        gc.collect()
        if hasattr(self, "torch") and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def run(self, cell: dict) -> dict:
        env_cfg = self.LiberoEnv(
            task=cell["suite"], task_ids=[int(cell["task_id"])],
            fps=int(cell["control_frequency_hz"]), obs_type="pixels_agent_pos",
            camera_name="agentview_image,robot0_eye_in_hand_image", init_states=True,
            observation_width=256, observation_height=256, control_mode="relative",
        )
        self.policy_for(cell, env_cfg)
        assert self.policy is not None and self.cfg is not None
        env_pre, env_post = self.make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=self.cfg)
        env = self.make_env(env_cfg, n_envs=1, use_async_envs=False)[cell["suite"]][int(cell["task_id"])]
        started = time.time()
        try:
            env.envs[0].init_state_id = int(cell["state_id"])
            if int(env.envs[0].init_state_id) != int(cell["state_id"]):
                raise RuntimeError("initial state assignment mismatch")
            reset_torch_rng(self.torch, 424242 if cell["policy"] == "ACT" else query_seed(cell, 0))
            self.policy.reset()
            observation, _ = env.reset(seed=[int(cell["environment_seed"])])
            max_steps = int(cell["max_episode_steps"] or np.asarray(env.call("_max_episode_steps")).reshape(-1)[0])
            executor = make_executor(cell, int(self.cfg.chunk_size))
            query_steps: list[int] = []
            actions: list[list[float]] = []
            source_ages: list[dict[str, int]] = []
            query_latencies: list[float] = []
            success = False
            task_name = str(env.envs[0].task)
            for t in range(max_steps):
                def query():
                    from lerobot.envs.utils import add_envs_task, preprocess_observation
                    from lerobot.utils.constants import ACTION
                    if cell["policy"] == "SmolVLA":
                        reset_torch_rng(self.torch, query_seed(cell, t))
                    batch = preprocess_observation(observation)
                    if cell["policy"] == "SmolVLA":
                        batch = add_envs_task(env, batch)
                    batch = env_pre(batch)
                    batch = self.preprocessor(batch)
                    q0 = time.perf_counter()
                    with self.torch.inference_mode():
                        chunk = self.postprocessor(self.policy.predict_action_chunk(batch))
                        chunk = env_post({ACTION: chunk})[ACTION]
                    query_latencies.append(time.perf_counter() - q0)
                    value = chunk.detach().cpu().numpy().astype(np.float32, copy=False)
                    if value.shape != (1, int(self.cfg.chunk_size), 7):
                        raise RuntimeError(f"unexpected chunk shape {value.shape}")
                    return value[0]
                decision = executor.step(query)
                if decision.policy_query:
                    query_steps.append(t)
                action = decision.action.astype(np.float32, copy=False)
                actions.append(action.astype(float).tolist())
                source_ages.append({k: int(v) for k, v in decision.source_ages.items()})
                observation, reward, terminated, truncated, info = env.step(action[None])
                done = bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0])
                if done:
                    final = info.get("final_info") if isinstance(info, dict) else None
                    if isinstance(final, dict) and "is_success" in final:
                        success = bool(np.asarray(final["is_success"]).reshape(-1)[0])
                    else:
                        success = bool(np.asarray(reward).reshape(-1)[0] > 0)
                    break
            steps = len(actions)
            wall = time.time() - started
            result = {k: cell[k] for k in (
                "cell_id", "phase", "policy", "suite", "task_id", "state_id",
                "environment_seed", "method", "strategy", "arm_horizon", "gripper_horizon", "checkpoint")}
            result.update({
                "status": "COMPLETE", "task_name": task_name,
                "success": bool(success), "environment_steps": steps,
                "policy_queries": len(query_steps), "model_forward_count": len(query_steps),
                "query_rate": len(query_steps) / steps, "query_steps": query_steps,
                "wall_clock_seconds": wall, "mean_model_forward_seconds": float(np.mean(query_latencies)),
                "mean_arm_source_age": float(np.mean([x["arm"] for x in source_ages])),
                "mean_gripper_source_age": float(np.mean([x["gripper"] for x in source_ages])),
                "source_ages": source_ages, "executed_actions": actions,
                "chunk_size": int(self.cfg.chunk_size), "n_action_steps": int(self.cfg.n_action_steps),
                "action_dim": 7, "temporal_aggregation": False, "smoothing": False,
                "fresh_environment_per_cell": True, "finished_at": time.time(),
            })
            return result
        finally:
            env.close()


def write_marker(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


def terminal_count(cells: list[dict], phase: str) -> int:
    return sum(
        is_complete(c) or marker_path(c, "technical_failed").is_file()
        for c in cells if c["phase"] == phase
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "queue_manifest.json")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--phases", nargs="*", default=list(PHASE_ORDER))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    cells = manifest["cells"]
    runtime = Runtime(args.gpu)
    progress_path = ROOT / "progress" / f"worker_{args.worker_index}.json"
    for phase in PHASE_ORDER:
        if phase not in args.phases:
            continue
        if phase == "smolvla_capacity_h16":
            while terminal_count(cells, "smolvla_primary") < 320:
                atomic_json(progress_path, {"pid": os.getpid(), "gpu": args.gpu, "phase": phase,
                    "state": "WAITING_FOR_PRIMARY_COMPLETION", "primary_terminal": terminal_count(cells, "smolvla_primary")})
                time.sleep(30)
        phase_cells = [c for c in cells if c["phase"] == phase]
        jobs: dict[tuple, list[dict]] = defaultdict(list)
        for c in phase_cells:
            jobs[(c["suite"], c["task_id"], c["checkpoint"])].append(c)
        assigned = [job for i, job in enumerate(sorted(jobs)) if i % args.num_workers == args.worker_index]
        for job in assigned:
            for c in jobs[job]:
                if is_complete(c) or marker_path(c, "technical_failed").is_file():
                    continue
                attempt_path = ROOT / "attempts" / c["phase"] / f"{c['cell_id']}.json"
                attempts = json.loads(attempt_path.read_text()).get("attempts", []) if attempt_path.is_file() else []
                while len(attempts) < 3 and not is_complete(c):
                    atomic_json(progress_path, {"pid": os.getpid(), "gpu": args.gpu, "phase": phase,
                        "cell_id": c["cell_id"], "attempt": len(attempts) + 1, "state": "RUNNING"})
                    try:
                        value = runtime.run(c)
                        atomic_json(result_path(c), value)
                        validate_result(c, result_path(c))
                        write_marker(marker_path(c), "COMPLETE")
                    except Exception as exc:
                        attempts.append({"attempt": len(attempts) + 1, "time": time.time(),
                            "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
                        atomic_json(attempt_path, {"cell_id": c["cell_id"], "attempts": attempts})
                        runtime.drop_policy()
                if not is_complete(c):
                    write_marker(marker_path(c, "technical_failed"), "TECHNICAL_FAILED")
        atomic_json(progress_path, {"pid": os.getpid(), "gpu": args.gpu, "phase": phase,
            "state": "PHASE_SHARD_COMPLETE", "terminal": terminal_count(cells, phase), "total": len(phase_cells)})
    runtime.drop_policy()
    atomic_json(progress_path, {"pid": os.getpid(), "gpu": args.gpu, "state": "ALL_REQUESTED_PHASES_COMPLETE"})


if __name__ == "__main__":
    main()
