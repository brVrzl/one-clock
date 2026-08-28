#!/usr/bin/env python3
"""Compare native SmolVLA select_action with its first predicted chunk action."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch


def prepare(observation, env, env_preprocessor, preprocessor):
    from lerobot.envs.utils import add_envs_task, preprocess_observation

    value = preprocess_observation(copy.deepcopy(observation))
    value = add_envs_task(env, value)
    value = env_preprocessor(value)
    return preprocessor(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="2")
    args = parser.parse_args()
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.constants import ACTION

    checkpoint = args.checkpoint.resolve()
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.pretrained_path = checkpoint
    env_cfg = LiberoEnv(task="libero_object", task_ids=[3], fps=30, obs_type="pixels_agent_pos", camera_name="agentview_image,robot0_eye_in_hand_image", init_states=True, observation_width=360, observation_height=360, control_mode="relative")
    env = make_env(env_cfg, n_envs=1, use_async_envs=False)["libero_object"][3]
    policy = make_policy(cfg=cfg, env_cfg=env_cfg)
    policy.eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(checkpoint), preprocessor_overrides={"device_processor": {"device": str(cfg.device)}})
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
    observation, _ = env.reset(seed=[1000])
    # SmolVLA flow matching samples Gaussian noise at inference. Use one
    # explicit noise tensor for both calls so this checks API semantics rather
    # than comparing two independent stochastic predictions.
    noise = torch.zeros(
        (1, cfg.chunk_size, cfg.max_action_dim),
        dtype=torch.float32,
        device=next(policy.parameters()).device,
    )
    policy.reset()
    with torch.inference_mode():
        batch = prepare(observation, env, env_pre, pre)
        chunk = env_post({ACTION: post(policy.predict_action_chunk(batch, noise=noise))})[ACTION]
    policy.reset()
    with torch.inference_mode():
        native = env_post({ACTION: post(policy.select_action(batch, noise=noise))})[ACTION]
    chunk_first = chunk[:, 0, :]
    result = {
        "checkpoint": str(checkpoint),
        "task": "libero_object:3",
        "seed": 1000,
        "chunk_size": int(cfg.chunk_size),
        "n_action_steps": int(cfg.n_action_steps),
        "predict_action_chunk_first_shape": list(chunk_first.shape),
        "select_action_shape": list(native.shape),
        "max_abs_error": float(torch.max(torch.abs(chunk_first - native)).item()),
        "fresh_semantics_match": bool(torch.allclose(chunk_first, native, atol=1e-6, rtol=0.0)),
    }
    env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
