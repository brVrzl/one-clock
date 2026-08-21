"""Build a tiny, frame-cached SmolVLA response audit for a LeRobot dataset.

This is deliberately an audit tool, not a LIBERO evaluator.  It queries the
frozen policy once per selected frame, constructs refresh-consistency labels
from those cached chunks, and then compares a small direct re-query path.
Future frames are used only by the label-side code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


GROUPS = {"arm": np.arange(0, 6, dtype=np.int64), "gripper": np.asarray([6], dtype=np.int64)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lerobot-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0])
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--max-horizon", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--linf-tolerance", type=float, default=0.05)
    parser.add_argument("--direct-check-windows", type=int, default=2)
    return parser


def _json_hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class FrozenPolicyRunner:
    """One normal policy call, with a non-invasive prefix-latent probe."""

    def __init__(self, checkpoint: Path, lerobot_source: Path, device: str, seed: int) -> None:
        sys.path.insert(0, str(lerobot_source / "src"))
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.torch = torch
        self.device = torch.device(device)
        self.seed = int(seed)
        self.forward_count = 0
        self.latents: list[np.ndarray] = []
        self.latent_shapes: list[list[int]] = []
        self.latent_dtype: str | None = None
        self.policy = SmolVLAPolicy.from_pretrained(str(checkpoint)).to(self.device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        self._hook = self.policy.model.vlm_with_expert.register_forward_hook(
            self._capture_prefix_latent, with_kwargs=True
        )

    def close(self) -> None:
        self._hook.remove()

    def _capture_prefix_latent(self, _module: Any, _args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        inputs_embeds = kwargs.get("inputs_embeds")
        if not isinstance(inputs_embeds, list) or len(inputs_embeds) != 2 or inputs_embeds[1] is not None:
            return
        prefix_output = output[0][0]
        if prefix_output is None:
            return
        vector = prefix_output.detach().float().mean(dim=1).cpu().numpy()
        self.latents.append(np.asarray(vector[0], dtype=np.float32))
        self.latent_shapes.append([int(value) for value in prefix_output.shape])
        self.latent_dtype = str(prefix_output.dtype)

    def _noise(self, frame_id: int) -> Any:
        generator = self.torch.Generator(device=self.device)
        generator.manual_seed(self.seed + int(frame_id))
        shape = (
            1,
            int(self.policy.config.chunk_size),
            int(self.policy.config.max_action_dim),
        )
        return self.torch.randn(shape, generator=generator, device=self.device, dtype=self.torch.float32)

    def predict(self, frame: dict[str, Any], frame_id: int) -> np.ndarray:
        """Return a de-normalized full action chunk for one source frame."""

        with self.torch.inference_mode():
            self.policy.reset()
            batch = self.preprocessor(dict(frame))
            action = self.policy.predict_action_chunk(batch, noise=self._noise(frame_id))
            action = self.postprocessor(action)
        self.forward_count += 1
        return action.detach().cpu().float().numpy()[0]


def _label_from_chunks(
    chunks: np.ndarray,
    source_indices: list[int],
    max_horizon: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = tuple(GROUPS)
    labels = np.zeros((len(source_indices), len(groups), max_horizon), dtype=np.uint8)
    censor = np.zeros_like(labels, dtype=np.uint8)
    future_frame = np.full_like(labels, -1, dtype=np.int64)
    for row, source in enumerate(source_indices):
        for offset in range(1, max_horizon + 1):
            future = source + offset
            if future >= len(chunks) or offset >= chunks.shape[1]:
                continue
            for group_index, group in enumerate(groups):
                indices = GROUPS[group]
                error = np.max(np.abs(chunks[source, offset, indices] - chunks[future, 0, indices]))
                labels[row, group_index, offset - 1] = int(error <= tolerance)
                censor[row, group_index, offset - 1] = 1
                future_frame[row, group_index, offset - 1] = future
    return labels, censor, future_frame


def _direct_labels(
    runner: FrozenPolicyRunner,
    frames: list[dict[str, Any]],
    frame_ids: list[int],
    source_indices: list[int],
    max_horizon: int,
    tolerance: float,
) -> np.ndarray:
    labels = np.zeros((len(source_indices), len(GROUPS), max_horizon), dtype=np.uint8)
    for row, source in enumerate(source_indices):
        old = runner.predict(frames[source], frame_ids[source])
        for offset in range(1, max_horizon + 1):
            future = source + offset
            if future >= len(frames) or offset >= old.shape[0]:
                continue
            fresh = runner.predict(frames[future], frame_ids[future])
            for group_index, group in enumerate(GROUPS):
                indices = GROUPS[group]
                labels[row, group_index, offset - 1] = int(
                    np.max(np.abs(old[offset, indices] - fresh[0, indices])) <= tolerance
                )
    return labels


def main() -> None:
    args = _parser().parse_args()
    if args.max_frames < args.max_horizon + 1:
        raise ValueError("max-frames must be at least max-horizon + 1")
    if args.linf_tolerance < 0:
        raise ValueError("linf-tolerance must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.lerobot_source / "src"))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        "HuggingFaceVLA/libero",
        root=args.dataset_root,
        episodes=args.episodes,
        download_videos=True,
        video_backend="pyav",
    )
    frames = [dataset[index] for index in range(min(args.max_frames, len(dataset)))]
    frame_ids = [int(frame["index"].item()) for frame in frames]
    if len(frames) < args.max_horizon + 1:
        raise ValueError("selected subset does not contain enough consecutive frames")
    source_indices = list(range(len(frames) - args.max_horizon))

    runner = FrozenPolicyRunner(args.checkpoint, args.lerobot_source, args.device, args.seed)
    start = time.perf_counter()
    chunks = np.stack([runner.predict(frame, frame_id) for frame, frame_id in zip(frames, frame_ids)])
    elapsed = time.perf_counter() - start
    cache_forward_count = runner.forward_count
    latent_vectors = np.stack(runner.latents) if len(runner.latents) == len(frames) else np.empty((0, 0), dtype=np.float32)
    labels, censor, future_frame = _label_from_chunks(
        chunks, source_indices, args.max_horizon, args.linf_tolerance
    )

    runner.forward_count = 0
    direct = _direct_labels(
        runner,
        frames,
        frame_ids,
        source_indices[: args.direct_check_windows],
        args.max_horizon,
        args.linf_tolerance,
    )
    direct_count = runner.forward_count
    cached_check = labels[: direct.shape[0]]
    correctness = bool(np.array_equal(cached_check, direct))
    runner.close()

    np.savez_compressed(
        args.output_dir / "model_side_cache.npz",
        source_frame_id=np.asarray(frame_ids, dtype=np.int64),
        predicted_action_chunk=chunks.astype(np.float32),
        z_t=latent_vectors.astype(np.float32),
        group_names=np.asarray(tuple(GROUPS), dtype=str),
    )
    np.savez_compressed(
        args.output_dir / "label_side_cache.npz",
        source_frame_row=np.asarray(source_indices, dtype=np.int64),
        future_frame_row=future_frame,
        y_refresh=labels,
        censor_mask=censor,
        offset_k=np.arange(1, args.max_horizon + 1, dtype=np.int64),
        group_names=np.asarray(tuple(GROUPS), dtype=str),
    )

    manifest = {
        "status": "pass" if correctness else "fail",
        "dataset_repo": "HuggingFaceVLA/libero",
        "dataset_root": str(args.dataset_root),
        "checkpoint": str(args.checkpoint),
        "checkpoint_config_sha256": _json_hash(args.checkpoint / "config.json"),
        "lerobot_source": str(args.lerobot_source),
        "groups": {name: indices.tolist() for name, indices in GROUPS.items()},
        "offset_convention": "offset_k is 1..K; no k=0 label is stored",
        "label_rule": "max absolute action error within group <= linf_tolerance",
        "linf_tolerance": args.linf_tolerance,
        "num_unique_frames": len(frames),
        "num_source_windows": len(source_indices),
        "num_cached_policy_forwards": cache_forward_count,
        "num_direct_check_forwards": direct_count,
        "naive_direct_forward_count_for_all_windows": len(source_indices) * (args.max_horizon + 1),
        "cache_forward_reuse_factor_vs_naive_direct": float(
            len(source_indices) * (args.max_horizon + 1) / max(1, cache_forward_count)
        ),
        "inference_seconds": elapsed,
        "frames_per_second": float(len(frames) / max(elapsed, 1e-12)),
        "latent_available": bool(latent_vectors.size),
        "latent_shape_per_frame": runner.latent_shapes[0] if runner.latent_shapes else None,
        "latent_dtype": runner.latent_dtype,
        "latent_definition": "mean over sequence of final source-prefix VLM hidden states captured from the normal prefix call",
        "latent_behavior_change": "none; forward hook observes and does not replace outputs",
        "correctness_check_windows": min(args.direct_check_windows, len(source_indices)),
        "cached_equals_direct_labels": correctness,
        "model_side_cache": "model_side_cache.npz",
        "label_side_cache": "label_side_cache.npz",
    }
    (args.output_dir / "cache_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
