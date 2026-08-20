"""Pinned SmolVLA loading and current-observation preprocessing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from dataset_common import DATASET_REVISION


CHECKPOINT_REPO_ID = "HuggingFaceVLA/smolvla_libero"
CHECKPOINT_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"


def load_runtime(dataset_root: Path, device: str | None = None) -> dict[str, Any]:
    import lerobot.policies  # noqa: F401 - registers policy subclasses
    import torch
    from lerobot.configs import PreTrainedConfig
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.policies import make_policy, make_pre_post_processors

    device_name = device or os.environ.get("ONECLOCK_SMOLVLA_DEVICE", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    metadata = LeRobotDatasetMetadata(
        repo_id="lerobot/libero",
        root=dataset_root,
        revision=DATASET_REVISION,
        token=False,
    )
    policy_config = PreTrainedConfig.from_pretrained(CHECKPOINT_REPO_ID, revision=CHECKPOINT_REVISION)
    policy_config.device = device_name
    policy_config.pretrained_path = CHECKPOINT_REPO_ID
    policy_config.pretrained_revision = CHECKPOINT_REVISION
    policy = make_policy(policy_config, ds_meta=metadata)
    policy.eval()
    policy_preprocessor, policy_postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=CHECKPOINT_REPO_ID,
        pretrained_revision=CHECKPOINT_REVISION,
        preprocessor_overrides={"device_processor": {"device": device_name}},
    )
    return {
        "policy": policy,
        "policy_config": policy_config,
        "policy_preprocessor": policy_preprocessor,
        "policy_postprocessor": policy_postprocessor,
        "device": device_name,
    }


def infer_full_chunk(runtime: dict[str, Any], observation: dict[str, Any]) -> np.ndarray:
    import torch
    from lerobot.envs.utils import preprocess_observation
    from lerobot.utils.constants import ACTION

    raw = {
        "pixels": observation["images"],
        "agent_pos": np.asarray(observation["state"], dtype=np.float32),
    }
    prepared = preprocess_observation(raw)
    prepared["task"] = str(observation.get("task_name", "libero"))
    model_observation = runtime["policy_preprocessor"](prepared)
    with torch.inference_mode():
        normalized = runtime["policy"].predict_action_chunk(model_observation)
        processed = runtime["policy_postprocessor"](normalized)
    if isinstance(processed, dict):
        processed = processed[ACTION]
    if hasattr(processed, "detach"):
        processed = processed.detach().cpu().numpy()
    array = np.asarray(processed, dtype=np.float32)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2 or array.shape[1] != 7 or not np.all(np.isfinite(array)):
        raise RuntimeError(f"SmolVLA returned invalid full chunk shape={array.shape}")
    return array
