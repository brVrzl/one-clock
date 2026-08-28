#!/usr/bin/env python3
"""Small teacher-forced held-out check for a native LeRobot ACT checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def stats(x: np.ndarray) -> dict[str, object]:
    return {
        "min": x.min(axis=0).astype(float).tolist(),
        "max": x.max(axis=0).astype(float).tolist(),
        "mean": x.mean(axis=0).astype(float).tolist(),
        "std": x.std(axis=0).astype(float).tolist(),
    }


def errors(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    arm_pred, arm_target = pred[:, :6], target[:, :6]
    grip_pred, grip_target = pred[:, 6], target[:, 6]
    return {
        "all7_mae": float(np.abs(pred - target).mean()),
        "all7_rmse": float(np.sqrt(np.square(pred - target).mean())),
        "arm_mae": float(np.abs(arm_pred - arm_target).mean()),
        "arm_rmse": float(np.sqrt(np.square(arm_pred - arm_target).mean())),
        "gripper_mae": float(np.abs(grip_pred - grip_target).mean()),
        "gripper_rmse": float(np.sqrt(np.square(grip_pred - grip_target).mean())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    horizon = 100
    dataset = LeRobotDataset(
        "HuggingFaceVLA/libero",
        root=args.dataset_root,
        episodes=args.episodes,
        delta_timestamps={"action": [i / 10 for i in range(horizon)]},
        download_videos=False,
    )
    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.pretrained_path = args.checkpoint
    policy = make_policy(config, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": config.device}},
    )

    one_pred, one_target = [], []
    chunk_pred, chunk_target = [], []
    chunk_masks = []
    trace = None
    observation_shapes = {}
    observation_ranges = {}
    for start in range(0, len(dataset), args.batch_size):
        samples = [dataset[i] for i in range(start, min(start + args.batch_size, len(dataset)))]
        observations = {
            key: torch.stack([sample[key] for sample in samples])
            for key in ("observation.images.image", "observation.images.image2", "observation.state")
        }
        targets = torch.stack([sample["action"] for sample in samples]).float()
        masks = torch.stack([sample["action_is_pad"] for sample in samples]).logical_not()
        if trace is None:
            trace = {
                "target": targets[0, : min(10, horizon)].numpy().astype(float).tolist(),
            }
        for key, value in observations.items():
            observation_shapes[key] = list(value.shape[1:])
            observation_ranges[key] = {
                "min": float(value.min()),
                "max": float(value.max()),
            }
        batch = preprocessor(observations)
        with torch.inference_mode():
            predicted = postprocessor(policy.predict_action_chunk(batch))
        predicted = predicted.detach().cpu().numpy().astype(np.float32)
        target_array = targets.numpy().astype(np.float32)
        mask_array = masks.numpy()
        if predicted.shape != target_array.shape:
            raise RuntimeError(f"shape mismatch: predicted={predicted.shape}, target={target_array.shape}")
        one_valid = mask_array[:, 0]
        one_pred.append(predicted[one_valid, 0])
        one_target.append(target_array[one_valid, 0])
        chunk_pred.append(predicted[mask_array])
        chunk_target.append(target_array[mask_array])
        chunk_masks.append(mask_array)
        if trace is not None and "prediction" not in trace:
            trace["prediction"] = predicted[0, : min(10, horizon)].astype(float).tolist()

    one_pred_array = np.concatenate(one_pred)
    one_target_array = np.concatenate(one_target)
    chunk_pred_array = np.concatenate(chunk_pred)
    chunk_target_array = np.concatenate(chunk_target)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": "HuggingFaceVLA/libero",
        "dataset_root": str(args.dataset_root.resolve()),
        "episodes": args.episodes,
        "num_dataset_items": len(dataset),
        "observation_shapes": observation_shapes,
        "observation_ranges": observation_ranges,
        "action_shape": [7],
        "chunk_size": horizon,
        "one_step": {
            "valid_steps": len(one_pred_array),
            "errors": errors(one_pred_array, one_target_array),
            "target_stats": stats(one_target_array),
            "predicted_stats": stats(one_pred_array),
        },
        "chunk": {
            "valid_steps": len(chunk_pred_array),
            "errors": errors(chunk_pred_array, chunk_target_array),
            "target_stats": stats(chunk_target_array),
            "predicted_stats": stats(chunk_pred_array),
        },
        "qualitative_trace_first_sample": trace,
        "contract": {
            "state_dim": 8,
            "state_semantics": ["eef_position_xyz", "eef_axis_angle_xyz", "gripper_joint_positions"],
            "action_dim": 7,
            "action_semantics": ["translation_delta_xyz", "rotation_delta_xyz", "gripper_command"],
            "relative_control": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "one_step": result["one_step"]["errors"],
        "chunk": result["chunk"]["errors"],
        "valid_steps": result["chunk"]["valid_steps"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
