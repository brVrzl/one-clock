#!/usr/bin/env python3
"""Analyze the frozen B3 demonstration-reference forecast audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
TASK_TAGS = ("libero_object_task3", "libero_spatial_task0", "libero_goal_task2", "libero_10_task3")
GROUPS = {"translation": (0, 1, 2), "rotation": (3, 4, 5), "gripper": (6,), "arm": (0, 1, 2, 3, 4, 5)}


def ci(values: np.ndarray) -> list[float]:
    return np.percentile(values, [2.5, 97.5]).astype(float).tolist()


def main() -> None:
    addendum = json.loads((ROOT / "track_b_analysis_addendum.json").read_text())
    b3 = addendum["b3"]
    result_root = ROOT / "track_b/forecast/results"
    marker_root = ROOT / "track_b/forecast/markers"
    payloads: dict[str, list[dict[str, Any]]] = {"ACT": [], "SmolVLA": []}
    for policy in payloads:
        for tag in TASK_TAGS:
            slug = f"{policy.lower()}-{tag}"
            metadata_path = result_root / f"{slug}.json"
            npz_path = result_root / f"{slug}.npz"
            marker = marker_root / f"{slug}.complete"
            if not (metadata_path.is_file() and npz_path.is_file() and marker.is_file()):
                raise RuntimeError(f"B3 incomplete: {slug}")
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("policy") != policy or metadata.get("tag") != tag or metadata.get("success_outcomes_loaded") is not False:
                raise RuntimeError(f"B3 identity drift: {slug}")
            with np.load(npz_path) as data:
                record = {key: np.asarray(data[key]) for key in data.files}
            if record["squared_error"].shape[1:] != (33, 7):
                raise RuntimeError(f"B3 shape drift: {slug}")
            record["tag"] = tag
            payloads[policy].append(record)

    tidy: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    draws = int(b3["bootstrap_draws"])
    for policy in payloads:
        episode_records: dict[int, dict[str, Any]] = {}
        for payload in payloads[policy]:
            for episode in np.unique(payload["episode_index"]):
                mask = payload["episode_index"] == episode
                episode_records[int(episode)] = {
                    "sse": payload["squared_error"][mask].sum(axis=0),
                    "anchors": int(mask.sum()),
                    "sign_disagreement": payload["gripper_sign_disagreement"][mask].sum(axis=0),
                    "tag": payload["tag"],
                }
        episodes = sorted(episode_records)
        if len(episodes) != 40:
            raise RuntimeError(f"B3 expected 40 episodes for {policy}, got {len(episodes)}")
        sse = np.stack([episode_records[e]["sse"] for e in episodes])
        anchors = np.asarray([episode_records[e]["anchors"] for e in episodes], dtype=np.float64)
        sign = np.stack([episode_records[e]["sign_disagreement"] for e in episodes]).astype(np.float64)
        rng = np.random.default_rng(int(b3[f"{policy}_seed"]))
        weights = rng.multinomial(len(episodes), np.full(len(episodes), 1 / len(episodes)), size=draws).astype(np.float64)
        boot_sse = np.tensordot(weights, sse, axes=(1, 0))
        boot_anchor = weights @ anchors
        boot_sign = weights @ sign
        policy_summary: dict[str, Any] = {}
        for offset in range(33):
            offset_summary: dict[str, Any] = {"dimensions": {}, "groups": {}}
            for dim in range(7):
                center = float(np.sqrt(sse[:, offset, dim].sum() / anchors.sum()))
                boot = np.sqrt(boot_sse[:, offset, dim] / boot_anchor)
                record = {"rmse": center, "episode_cluster_bootstrap_ci": ci(boot)}
                offset_summary["dimensions"][f"dim_{dim}"] = record
                tidy.append({"policy": policy, "offset": offset, "offset_seconds": offset / 10, "quantity": "normalized_rmse", "metric": f"dim_{dim}", **record})
            for group, dims in GROUPS.items():
                center = float(np.sqrt(sse[:, offset][:, dims].sum() / (anchors.sum() * len(dims))))
                boot = np.sqrt(boot_sse[:, offset][:, dims].sum(axis=1) / (boot_anchor * len(dims)))
                record = {"rmse": center, "episode_cluster_bootstrap_ci": ci(boot)}
                offset_summary["groups"][group] = record
                tidy.append({"policy": policy, "offset": offset, "offset_seconds": offset / 10, "quantity": "normalized_rmse", "metric": group, **record})
            grip_center = float(sign[:, offset].sum() / anchors.sum())
            grip_boot = boot_sign[:, offset] / boot_anchor
            offset_summary["gripper_sign_disagreement"] = {"probability": grip_center, "episode_cluster_bootstrap_ci": ci(grip_boot)}
            tidy.append({"policy": policy, "offset": offset, "offset_seconds": offset / 10, "quantity": "gripper_sign_disagreement", "metric": "gripper", "probability": grip_center, "episode_cluster_bootstrap_ci": ci(grip_boot)})
            policy_summary[str(offset)] = offset_summary
        summary[policy] = policy_summary

    output_dir = ROOT / "track_b/forecast/analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "forecast_metrics.csv").open("w", newline="") as handle:
        fields = sorted({key for row in tidy for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(tidy)
    output = {
        "status": "COMPLETE", "success_outcomes_loaded": False,
        "provenance_label": b3["provenance_label"],
        "episodes_per_policy": 40, "anchor_stride_frames": b3["anchor_stride_frames"],
        "dataset_fps": 10, "offsets": b3["offsets"], "offset_seconds": [k / 10 for k in b3["offsets"]],
        "target_alignment": "exact dataset frame t+k; no interpolation, resampling, or repetition",
        "gripper_sign_contract": "controller-native action sign after exact checkpoint MEAN_STD inverse; zero is separate",
        "bootstrap": {"unit": b3["bootstrap_unit"], "draws": draws, "ACT_seed": b3["ACT_seed"], "SmolVLA_seed": b3["SmolVLA_seed"]},
        "policy_results": summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    lines = [
        "# B3 open-loop future-action predictability", "",
        "This is a training-demonstration reference analysis, not held-out evaluation. ACT and SmolVLA are reported in their own frozen normalized spaces; no rollout success outcome was loaded.", "",
        "Chunk offset k is an exact 10 Hz dataset-frame target at k/10 seconds. No interpolation, resampling, or repetition is used.", "",
        "All per-offset translation, rotation, gripper, per-dimension, sign-disagreement, and episode-cluster interval results are in `forecast_metrics.csv` and `summary.json`.", "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))
    print(json.dumps({"status": "COMPLETE", "episodes_per_policy": 40, "offsets": 33}, indent=2))


if __name__ == "__main__":
    main()
