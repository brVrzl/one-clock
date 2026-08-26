#!/usr/bin/env python3
"""Run paired LIBERO rollouts through the common ACT/SHARED/DCTA executor."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_STATE
from libero.libero import get_libero_path
from libero.libero.benchmark import Benchmark, get_benchmark
from libero.libero.envs import OffScreenRenderEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = Path("/home/wjq/workspace/upstreams/verl-vla")
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(UPSTREAM_ROOT / "src"))

from one_clock.libero_dcta import (  # noqa: E402
    ACTEncoderContextCapture,
    DynamicTemporalGate,
    LiberoTemporalExecutor,
)
from verl_vla.envs.libero.utils import (  # noqa: E402
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)

METHODS = ("standard_act", "shared_dynamic", "dcta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["spatial", "object", "goal"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--reset-warmup-steps", type=int, default=10)
    parser.add_argument("--env-seed", type=int, default=42)
    return parser.parse_args()


def load_gate(path: Path, method: str, device: torch.device) -> DynamicTemporalGate:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["method"] != method:
        raise ValueError(f"gate checkpoint {path} is for {payload['method']}, not {method}")
    gate = DynamicTemporalGate(
        context_dim=int(payload["context_dim"]),
        num_groups=1 if method == "shared_dynamic" else 2,
        hidden_dim=int(payload["hidden_dim"]),
        max_age=int(payload["max_age"]),
    )
    gate.load_state_dict(payload["state_dict"], strict=True)
    return gate.to(device).eval()


def make_policy_observation(raw_observation: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    image = np.ascontiguousarray(get_libero_image(raw_observation))
    wrist_image = np.ascontiguousarray(get_libero_wrist_image(raw_observation))
    state = np.concatenate(
        [
            raw_observation["robot0_eef_pos"],
            quat2axisangle(raw_observation["robot0_eef_quat"]),
            raw_observation["robot0_gripper_qpos"],
        ]
    ).astype(np.float32)
    return {
        "observation.images.image": torch.from_numpy(image).permute(2, 0, 1).float().div_(255),
        "observation.images.wrist_image": torch.from_numpy(wrist_image).permute(2, 0, 1).float().div_(255),
        "observation.state": torch.from_numpy(state),
    }


def open_task_environment(task_suite: Benchmark, task_id: int) -> OffScreenRenderEnv:
    task = task_suite.get_task(task_id)
    bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    return OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=256,
        camera_widths=256,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_depths=False,
    )


def reset_to_trial(
    env: OffScreenRenderEnv,
    *,
    task_suite: Benchmark,
    task_id: int,
    trial_id: int,
    seed: int,
    warmup_steps: int,
) -> dict[str, np.ndarray]:
    env.seed(seed)
    env.reset()
    raw_observation = env.set_init_state(task_suite.get_task_init_states(task_id)[trial_id])
    zero_action = np.zeros(7, dtype=np.float32)
    for _ in range(warmup_steps):
        raw_observation, _reward, _done, _info = env.step(zero_action)
    return raw_observation


def completed_keys(path: Path) -> set[tuple[str, int, int, str]]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        keys.add((row["suite"], row["task_id"], row["trial_id"], row["method"]))
    return keys


def group_pair(values: torch.Tensor) -> tuple[float, float]:
    values = values.squeeze(0).detach().float().cpu()
    if values.numel() == 1:
        value = float(values[0])
        return value, value
    return float(values[0]), float(values[1])


def main() -> None:
    args = parse_args()
    if args.max_steps < 1 or args.reset_warmup_steps < 0:
        raise ValueError("max steps must be positive and warmup steps non-negative")
    if any(method != "standard_act" for method in args.methods) and args.gate_dir is None:
        raise ValueError("--gate-dir is required for learned aggregation methods")

    device = torch.device(args.device)
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    if policy.config.chunk_size != 10 or policy.config.action_feature.shape != (7,):
        raise ValueError("rollouts require the canonical 10x7 LIBERO ACT checkpoint")
    policy.config.device = args.device
    policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(args.checkpoint)
    )
    gates = {
        method: load_gate(args.gate_dir / f"{method}.pt", method, device)
        for method in args.methods
        if method != "standard_act"
    }

    schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
    blocks = [block for block in schedule["blocks"] if block["suite"] == args.suite]
    task_suite = get_benchmark(f"libero_{args.suite}")()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done_keys = completed_keys(args.output)
    current_task_id = None
    env = None

    try:
        with ACTEncoderContextCapture(policy.model.encoder) as context_capture:
            for block in blocks:
                task_id = int(block["benchmark_task_id"])
                trial_id = int(block["trial_id"])
                if current_task_id != task_id:
                    if env is not None:
                        env.close()
                    env = open_task_environment(task_suite, task_id)
                    current_task_id = task_id

                for method in block["method_order"]:
                    if method not in args.methods:
                        continue
                    key = (args.suite, task_id, trial_id, method)
                    if key in done_keys:
                        continue
                    assert env is not None
                    raw_observation = reset_to_trial(
                        env,
                        task_suite=task_suite,
                        task_id=task_id,
                        trial_id=trial_id,
                        seed=args.env_seed,
                        warmup_steps=args.reset_warmup_steps,
                    )
                    executor = LiberoTemporalExecutor(
                        method=method,
                        chunk_size=10,
                        gate=gates.get(method),
                        postprocessor=postprocessor,
                    )
                    trajectory = []
                    success = False
                    with torch.inference_mode():
                        for step_index in range(args.max_steps):
                            policy_batch = preprocessor(make_policy_observation(raw_observation))
                            normalized_chunk = policy.predict_action_chunk(policy_batch)[0].float()
                            context = context_capture.pop(expected_batch_size=1).float()
                            decision = executor.step(
                                normalized_chunk=normalized_chunk,
                                normalized_robot_state=policy_batch[OBS_STATE].float(),
                                physical_source_step=step_index,
                                current_physical_step=step_index,
                                act_context=context,
                            )
                            raw_observation, _reward, done, _info = env.step(
                                decision.environment_action[0].numpy()
                            )
                            query_age_arm, query_age_gripper = group_pair(
                                decision.aggregation.effective_query_age
                            )
                            physical_age_arm, physical_age_gripper = group_pair(
                                decision.aggregation.effective_physical_age
                            )
                            entropy_arm, entropy_gripper = group_pair(decision.aggregation.entropy)
                            trajectory.append(
                                {
                                    "step": step_index,
                                    "weights": decision.aggregation.weights[0].float().cpu().tolist(),
                                    "effective_query_age_arm": query_age_arm,
                                    "effective_query_age_gripper": query_age_gripper,
                                    "effective_physical_age_arm": physical_age_arm,
                                    "effective_physical_age_gripper": physical_age_gripper,
                                    "entropy_arm": entropy_arm,
                                    "entropy_gripper": entropy_gripper,
                                    "arm_gripper_kernel_distance": float(
                                        decision.aggregation.arm_gripper_kernel_distance[0].cpu()
                                    ),
                                }
                            )
                            if bool(done):
                                success = True
                                break

                    record = {
                        "suite": args.suite,
                        "task_id": task_id,
                        "task": task_suite.get_task(task_id).language,
                        "trial_id": trial_id,
                        "method": method,
                        "success": success,
                        "steps": len(trajectory),
                        "trajectory": trajectory,
                    }
                    with args.output.open("a", encoding="utf-8") as output_file:
                        output_file.write(json.dumps(record, separators=(",", ":")) + "\n")
                    done_keys.add(key)
                    print(
                        f"{args.suite} task={task_id} trial={trial_id} method={method} "
                        f"success={int(success)} steps={len(trajectory)}",
                        flush=True,
                    )
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
