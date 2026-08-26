#!/usr/bin/env python3
"""Extract frozen-ACT demonstration candidates and decoder context for DCTA."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch

from research.audit_tools.audit_robotwin_act_recorded_sequence import CAMERAS, build_policy


DEMO_DT_SECONDS = 15.0 / 250.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--act-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)
    policy = build_policy(args.act_root, args.checkpoint_dir, device)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    with (args.checkpoint_dir / "dataset_stats.pkl").open("rb") as handle:
        stats = pickle.load(handle)

    captured: list[torch.Tensor] = []

    def capture_decoder_input(_module, inputs) -> None:
        captured.append(inputs[0].detach())

    hook = policy.model.action_head.register_forward_pre_hook(capture_decoder_input)
    episode_ids = []
    decision_ids = []
    query_times = []
    chunks = []
    qposes = []
    contexts = []
    actions = []
    try:
        for episode_id in range(50):
            path = args.data_root / f"episode_{episode_id}.hdf5"
            with h5py.File(path, "r") as episode:
                length = len(episode["action"])
                raw_qpos = np.asarray(episode["observations/qpos"], dtype=np.float32)
                raw_action = np.asarray(episode["action"], dtype=np.float32)
                for start in range(0, length, args.batch_size):
                    stop = min(start + args.batch_size, length)
                    images = np.stack(
                        [
                            np.stack(
                                [
                                    np.moveaxis(
                                        np.asarray(
                                            episode[f"observations/images/{camera}"][step]
                                        ),
                                        -1,
                                        0,
                                    )
                                    / 255.0
                                    for camera in CAMERAS
                                ]
                            )
                            for step in range(start, stop)
                        ]
                    ).astype(np.float32)
                    normalized_qpos = (
                        raw_qpos[start:stop] - stats["qpos_mean"]
                    ) / stats["qpos_std"]
                    captured.clear()
                    with torch.inference_mode():
                        predicted = policy(
                            torch.from_numpy(normalized_qpos).to(device),
                            torch.from_numpy(images).to(device),
                        )
                    if len(captured) != 1:
                        raise RuntimeError("ACT decoder context hook did not fire exactly once")
                    hidden = captured[0]
                    if hidden.shape != (stop - start, 50, 512):
                        raise RuntimeError(f"unexpected ACT decoder feature shape {hidden.shape}")
                    chunks.append(predicted.cpu().numpy().astype(np.float32))
                    contexts.append(hidden[:, 0].cpu().numpy().astype(np.float32))
                    qposes.append(normalized_qpos.astype(np.float32))
                    actions.append(
                        ((raw_action[start:stop] - stats["action_mean"]) / stats["action_std"])
                        .astype(np.float32)
                    )
                    count = stop - start
                    episode_ids.extend([episode_id] * count)
                    decision_ids.extend(range(start, stop))
                    query_times.extend(np.arange(start, stop) * DEMO_DT_SECONDS)
            print(f"extracted {args.task} episode {episode_id}/49", flush=True)
    finally:
        hook.remove()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        task=np.asarray(args.task),
        episode_id=np.asarray(episode_ids, dtype=np.int16),
        decision=np.asarray(decision_ids, dtype=np.int16),
        query_time_seconds=np.asarray(query_times, dtype=np.float32),
        normalized_chunk=np.concatenate(chunks),
        normalized_qpos=np.concatenate(qposes),
        act_context=np.concatenate(contexts),
        normalized_demonstrated_action=np.concatenate(actions),
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
