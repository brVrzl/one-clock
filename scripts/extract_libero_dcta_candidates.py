#!/usr/bin/env python3
"""Extract frozen-ACT same-query chunks and contexts from LIBERO demonstrations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from one_clock.libero_dcta import ACTEncoderContextCapture  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size must be positive and num workers non-negative")

    policy = ACTPolicy.from_pretrained(args.checkpoint)
    if policy.config.chunk_size != 10 or policy.config.action_feature.shape != (7,):
        raise ValueError("candidate extraction requires the canonical 10x7 LIBERO ACT checkpoint")
    policy.config.device = args.device
    policy.to(args.device).eval()
    preprocessor, _postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(args.checkpoint)
    )
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
        revision=args.revision,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )

    chunks: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    contexts: list[torch.Tensor] = []
    episode_ids: list[torch.Tensor] = []
    frame_indices: list[torch.Tensor] = []
    task_ids: list[torch.Tensor] = []
    with torch.inference_mode(), ACTEncoderContextCapture(policy.model.encoder) as context_capture:
        for batch in loader:
            policy_batch = preprocessor(
                {
                    key: value
                    for key, value in batch.items()
                    if key in policy.config.input_features or key == ACTION
                }
            )
            predicted_chunk = policy.predict_action_chunk(policy_batch)
            context = context_capture.pop(expected_batch_size=predicted_chunk.shape[0])
            chunks.append(predicted_chunk.float().cpu())
            states.append(policy_batch[OBS_STATE].float().cpu())
            targets.append(policy_batch[ACTION].float().cpu())
            contexts.append(context.float().cpu())
            episode_ids.append(batch["episode_index"].to(torch.int64).cpu())
            frame_indices.append(batch["frame_index"].to(torch.int64).cpu())
            task_ids.append(batch["task_index"].to(torch.int64).cpu())

    cache = {
        "format_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "repo_id": args.repo_id,
        "revision": args.revision,
        "predicted_chunks": torch.cat(chunks),
        "normalized_robot_states": torch.cat(states),
        "normalized_target_actions": torch.cat(targets),
        "act_contexts": torch.cat(contexts),
        "episode_ids": torch.cat(episode_ids),
        "frame_indices": torch.cat(frame_indices),
        "task_ids": torch.cat(task_ids),
    }
    expected_frames = len(dataset)
    for key, value in cache.items():
        if isinstance(value, torch.Tensor) and value.shape[0] != expected_frames:
            raise RuntimeError(f"candidate cache field {key} has {value.shape[0]} rows, expected {expected_frames}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.output)
    print(f"Saved {expected_frames} frozen ACT candidate rows to {args.output}")


if __name__ == "__main__":
    main()
