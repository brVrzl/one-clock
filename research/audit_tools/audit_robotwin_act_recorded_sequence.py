#!/usr/bin/env python3
"""Outcome-free ACT audit on one recorded RoboTwin observation sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from research.audit_tools.robotwin_temporal_reuse import (
    ACTION_GROUPS,
    METHODS,
    NATIVE_METHOD,
    NOMINAL_SOURCE_AGE_TICKS,
    RoboTwinTemporalExecutor,
    native_act_aggregate,
    postprocess_action,
)


CAMERAS = ("cam_head", "cam_right_wrist", "cam_left_wrist")


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def build_policy(act_root: Path, checkpoint_dir: Path, device: torch.device):
    sys.path.insert(0, str(act_root))
    from detr.act_policy import ACTPolicy

    os.environ["ACT_ACTION_DIM"] = "14"
    config = {
        "lr": 1e-5,
        "num_queries": 50,
        "kl_weight": 10,
        "hidden_dim": 512,
        "dim_feedforward": 3200,
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": list(CAMERAS),
    }
    model_config = SimpleNamespace(
        action_dim=14,
        chunk_size=50,
        camera_names=list(CAMERAS),
        lr=1e-5,
        lr_backbone=1e-5,
        weight_decay=1e-4,
        backbone="resnet18",
        dilation=False,
        position_embedding="sine",
        enc_layers=4,
        dec_layers=7,
        dim_feedforward=3200,
        hidden_dim=512,
        dropout=0.1,
        nheads=8,
        pre_norm=False,
        masks=False,
    )
    policy = ACTPolicy(config, model_config).to(device)
    state = torch.load(checkpoint_dir / "policy_last.ckpt", map_location=device)
    policy.load_state_dict(state)
    policy.eval()
    return policy


def infer_chunk(
    policy,
    qpos: np.ndarray,
    images: np.ndarray,
    stats: dict[str, np.ndarray],
    device: torch.device,
) -> np.ndarray:
    normalized_qpos = (qpos - stats["qpos_mean"]) / stats["qpos_std"]
    qpos_tensor = torch.from_numpy(normalized_qpos).float().unsqueeze(0).to(device)
    image_tensor = torch.from_numpy(images).float().unsqueeze(0).to(device)
    with torch.no_grad():
        chunk = policy(qpos_tensor, image_tensor)[0]
    return chunk.detach().cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--act-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)
    policy = build_policy(args.act_root, args.checkpoint_dir, device)
    with (args.checkpoint_dir / "dataset_stats.pkl").open("rb") as handle:
        stats = pickle.load(handle)

    qposes = []
    images = []
    with h5py.File(args.episode, "r") as episode:
        for step in range(NOMINAL_SOURCE_AGE_TICKS + 1):
            qposes.append(np.asarray(episode["observations/qpos"][step]))
            camera_images = [
                np.moveaxis(np.asarray(episode[f"observations/images/{camera}"][step]), -1, 0)
                / 255.0
                for camera in CAMERAS
            ]
            images.append(np.stack(camera_images))

    first_chunk = infer_chunk(policy, qposes[0], images[0], stats, device)
    repeated_chunk = infer_chunk(policy, qposes[0], images[0], stats, device)
    chunks = [first_chunk]
    for step in range(1, NOMINAL_SOURCE_AGE_TICKS + 1):
        chunks.append(infer_chunk(policy, qposes[step], images[step], stats, device))

    target = NOMINAL_SOURCE_AGE_TICKS
    normalized_chunks = dict(enumerate(chunks))
    native_normalized = native_act_aggregate(normalized_chunks, target)
    native_action = postprocess_action(
        native_normalized, stats["action_mean"], stats["action_std"]
    )

    method_records = {}
    for method in METHODS:
        executor = RoboTwinTemporalExecutor(method)
        result = None
        for source_step, chunk in enumerate(chunks):
            result = executor.update(source_step, chunk)
        assert result is not None
        record = result.as_log_record()
        record["target_decision"] = record.pop("target_physical_step")
        record["source_age_simulator_time"] = None
        record["executed_composed_action"] = postprocess_action(
            result.action, stats["action_mean"], stats["action_std"]
        ).tolist()
        record["fresh_action"] = postprocess_action(
            result.fresh_action, stats["action_mean"], stats["action_std"]
        ).tolist()
        record["old_action"] = (
            None
            if result.old_action is None
            else postprocess_action(
                result.old_action, stats["action_mean"], stats["action_std"]
            ).tolist()
        )
        method_records[method] = record

    processed_input = np.concatenate(
        [
            ((qposes[0] - stats["qpos_mean"]) / stats["qpos_std"]).ravel(),
            images[0].ravel(),
        ]
    )
    output = {
        "scope": "Outcome-free recorded-sequence semantic audit",
        "checkpoint": str(args.checkpoint_dir / "policy_last.ckpt"),
        "episode": str(args.episode),
        "device": str(device),
        "camera_inputs": list(CAMERAS),
        "action_groups": {key: list(value) for key, value in ACTION_GROUPS.items()},
        "action_dimension": 14,
        "chunk_length": 50,
        "source_age_ticks": NOMINAL_SOURCE_AGE_TICKS,
        "source_age_nominal_demo_seconds": 1.02,
        "source_age_physical_seconds": None,
        "frozen_input_determinism": {
            "processed_input_sha256": array_sha256(processed_input),
            "first_full_chunk_sha256": array_sha256(first_chunk),
            "repeated_full_chunk_sha256": array_sha256(repeated_chunk),
            "full_chunk_exact_equal": bool(np.array_equal(first_chunk, repeated_chunk)),
            "full_chunk_max_abs_difference": float(
                np.max(np.abs(first_chunk - repeated_chunk))
            ),
            "postprocessing_exact_equal": bool(
                np.array_equal(
                    postprocess_action(first_chunk[0], stats["action_mean"], stats["action_std"]),
                    postprocess_action(repeated_chunk[0], stats["action_mean"], stats["action_std"]),
                )
            ),
        },
        "chunk_sha256_by_source": {
            str(source): array_sha256(chunk)
            for source, chunk in normalized_chunks.items()
        },
        NATIVE_METHOD: {
            "target_decision": target,
            "candidate_source_steps": list(range(target + 1)),
            "candidate_chunk_offsets": [target - source for source in range(target + 1)],
            "weight_decay": 0.01,
            "executed_aggregated_action": native_action.tolist(),
        },
        "experimental_methods": method_records,
        "scientific_outcomes_inspected": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["frozen_input_determinism"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
