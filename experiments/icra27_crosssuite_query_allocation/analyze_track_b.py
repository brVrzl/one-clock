#!/usr/bin/env python3
"""Analyze the frozen same-target instability diagnostic without using success."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
DRAW_COUNT = 20_000


def percentile(values: np.ndarray) -> list[float]:
    return np.percentile(values, [2.5, 97.5]).astype(float).tolist()


def ratio_bootstrap(rows: list[dict[str, Any]], seed: int) -> tuple[float, list[float]]:
    arm = np.asarray([row["arm_dispersion"] for row in rows], dtype=float)
    grip = np.asarray([row["gripper_dispersion"] for row in rows], dtype=float)
    denominator = float(arm.mean())
    center = float(grip.mean() / denominator) if denominator > 0 else math.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(DRAW_COUNT, len(rows)))
    boot_arm, boot_grip = arm[indices].mean(1), grip[indices].mean(1)
    valid = boot_arm > 0
    if not valid.all():
        raise RuntimeError("zero arm-dispersion denominator in bootstrap")
    return center, percentile(boot_grip / boot_arm)


def matched_ratio_difference(act: list[dict[str, Any]], smol: list[dict[str, Any]], seed: int) -> tuple[float, list[float]]:
    by_key = lambda rows: {(r["suite"], r["task_id"], r["state_id"]): r for r in rows}
    aa, ss = by_key(act), by_key(smol)
    keys = sorted(aa)
    if keys != sorted(ss):
        raise RuntimeError("ACT/SmolVLA diagnostic panels are not paired")
    arm_a = np.asarray([aa[k]["arm_dispersion"] for k in keys])
    grip_a = np.asarray([aa[k]["gripper_dispersion"] for k in keys])
    arm_s = np.asarray([ss[k]["arm_dispersion"] for k in keys])
    grip_s = np.asarray([ss[k]["gripper_dispersion"] for k in keys])
    center = float(grip_a.mean()/arm_a.mean() - grip_s.mean()/arm_s.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(DRAW_COUNT, len(keys)))
    draws = grip_a[idx].mean(1)/arm_a[idx].mean(1) - grip_s[idx].mean(1)/arm_s[idx].mean(1)
    return center, percentile(draws)


def margin_difference(targets: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    # Exact terciles by stable rank. Metadata breaks numerical ties without
    # using success or any behavioral outcome.
    ranked = sorted(targets, key=lambda r: (r["margin"], r["suite"], r["task_id"], r["state_id"], r["target_time"]))
    n = len(ranked)
    low_ids = {r["target_id"] for r in ranked[: n//3]}
    high_ids = {r["target_id"] for r in ranked[n - n//3 :]}
    episode = defaultdict(lambda: {"low_sum": 0.0, "low_n": 0, "high_sum": 0.0, "high_n": 0})
    for row in targets:
        key = row["episode_id"]
        if row["target_id"] in low_ids:
            episode[key]["low_sum"] += row["sign_disagreement"]
            episode[key]["low_n"] += 1
        if row["target_id"] in high_ids:
            episode[key]["high_sum"] += row["sign_disagreement"]
            episode[key]["high_n"] += 1
    values = list(episode.values())
    def difference(indices: np.ndarray) -> float:
        chosen = [values[int(i)] for i in indices]
        low_n, high_n = sum(x["low_n"] for x in chosen), sum(x["high_n"] for x in chosen)
        return sum(x["low_sum"] for x in chosen)/low_n - sum(x["high_sum"] for x in chosen)/high_n
    center = difference(np.arange(len(values)))
    rng = np.random.default_rng(seed)
    draws = np.asarray([difference(rng.integers(0, len(values), size=len(values))) for _ in range(DRAW_COUNT)])
    return {"low_target_count": len(low_ids), "high_target_count": len(high_ids), "low_minus_high": center, "episode_cluster_bootstrap_ci": percentile(draws)}


def analyze_episode(cell: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata_path = ROOT / "track_b/results" / f"{cell['cell_id']}.json"
    predictions_path = ROOT / "track_b/predictions" / f"{cell['cell_id']}.npz"
    marker = ROOT / "track_b/markers" / f"{cell['cell_id']}.complete"
    if not (metadata_path.is_file() and predictions_path.is_file() and marker.is_file()):
        raise RuntimeError(f"incomplete Track-B cell: {cell['cell_id']}")
    metadata = json.loads(metadata_path.read_text())
    for key in ("cell_id","policy","suite","task_id","state_id","environment_seed","checkpoint"):
        if metadata.get(key) != cell.get(key):
            raise RuntimeError(f"Track-B identity mismatch {cell['cell_id']}:{key}")
    data = np.load(predictions_path)
    chunks = data["predicted_chunks_normalized"]
    steps = int(metadata["environment_steps"])
    if chunks.shape != (steps, int(metadata["chunk_size"]), 7):
        raise RuntimeError("Track-B prediction shape mismatch")
    targets = []
    for target in range(15, steps):
        candidates = np.stack([chunks[target-age, age] for age in range(16)])
        mean = candidates.mean(0)
        dispersion = np.sqrt(np.mean(np.square(candidates - mean), axis=0))
        arm = float(np.sqrt(np.mean(np.square(dispersion[:6]))))
        grip = float(dispersion[6])
        signs = np.sign(candidates[:, 6])
        disagreement = float(sum(signs[i] != signs[j] for i in range(16) for j in range(i+1,16)) / 120)
        _, counts = np.unique(signs, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
        target_id = f"{cell['cell_id']}:t{target}"
        targets.append({
            "target_id": target_id, "episode_id": cell["cell_id"], "policy": cell["policy"],
            "suite": cell["suite"], "task_id": cell["task_id"], "state_id": cell["state_id"],
            "target_time": target, "arm_dispersion": arm, "gripper_dispersion": grip,
            "sign_disagreement": disagreement, "sign_entropy_bits": entropy,
            "margin": float(abs(mean[6])),
        })
    if not targets:
        raise RuntimeError(f"episode shorter than primary window: {cell['cell_id']}")
    episode = {
        "cell_id": cell["cell_id"], "policy": cell["policy"], "suite": cell["suite"],
        "task_id": cell["task_id"], "state_id": cell["state_id"], "eligible_targets": len(targets),
        "arm_dispersion": float(np.mean([x["arm_dispersion"] for x in targets])),
        "gripper_dispersion": float(np.mean([x["gripper_dispersion"] for x in targets])),
        "sign_disagreement": float(np.mean([x["sign_disagreement"] for x in targets])),
        "sign_entropy_bits": float(np.mean([x["sign_entropy_bits"] for x in targets])),
        "absolute_margin": float(np.mean([x["margin"] for x in targets])),
    }
    return episode, targets


def main() -> None:
    manifest = json.loads((ROOT / "track_b_manifest.json").read_text())
    episodes, targets = [], []
    for cell in manifest["cells"]:
        episode, episode_targets = analyze_episode(cell)
        episodes.append(episode)
        targets.extend(episode_targets)
    by_policy = {policy: [x for x in episodes if x["policy"] == policy] for policy in ("ACT","SmolVLA")}
    target_by_policy = {policy: [x for x in targets if x["policy"] == policy] for policy in ("ACT","SmolVLA")}
    act_ratio, act_ci = ratio_bootstrap(by_policy["ACT"], manifest["bootstrap"]["ACT_seed"])
    smol_ratio, smol_ci = ratio_bootstrap(by_policy["SmolVLA"], manifest["bootstrap"]["SmolVLA_seed"])
    difference, difference_ci = matched_ratio_difference(by_policy["ACT"], by_policy["SmolVLA"], manifest["bootstrap"]["difference_seed"])
    margins = {policy: margin_difference(target_by_policy[policy], manifest["bootstrap"]["margin_seed"] + index) for index, policy in enumerate(("ACT","SmolVLA"))}
    act_pass = act_ci[0] > 1 and margins["ACT"]["episode_cluster_bootstrap_ci"][0] > 0
    cross_support = difference_ci[0] > 0
    output = {
        "status": "COMPLETE", "success_outcomes_loaded": False,
        "episodes": len(episodes), "episodes_by_policy": {k: len(v) for k,v in by_policy.items()},
        "primary_window": manifest["primary_window"],
        "policy_results": {
            "ACT": {"arm_normalized_dispersion": float(np.mean([x["arm_dispersion"] for x in by_policy["ACT"]])), "gripper_normalized_dispersion": float(np.mean([x["gripper_dispersion"] for x in by_policy["ACT"]])), "R": act_ratio, "R_episode_cluster_bootstrap_ci": act_ci, "margin_conditioning": margins["ACT"]},
            "SmolVLA": {"arm_normalized_dispersion": float(np.mean([x["arm_dispersion"] for x in by_policy["SmolVLA"]])), "gripper_normalized_dispersion": float(np.mean([x["gripper_dispersion"] for x in by_policy["SmolVLA"]])), "R": smol_ratio, "R_episode_cluster_bootstrap_ci": smol_ci, "margin_conditioning": margins["SmolVLA"]},
        },
        "R_ACT_minus_R_SMOLVLA": difference, "R_difference_paired_episode_bootstrap_ci": difference_ci,
        "labels": {"ACT_LOCALIZATION_PASS": act_pass, "ACT_LOCALIZATION_KILL": not act_pass, "CROSS_POLICY_MECHANISM_SUPPORT": cross_support},
        "episode_metrics": episodes,
    }
    (ROOT / "track_b/analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    lines = ["# Track-B same-target instability diagnostic", "", f"ACT localization: **{'PASS' if act_pass else 'KILL'}**. Cross-policy mechanism support: **{'YES' if cross_support else 'NO'}**.", "", "These are mechanism-only diagnostics on already outcome-exposed development cells. Success outcomes were not loaded or used for method selection.", "", "| Policy | Arm dispersion | Gripper dispersion | R | Episode-cluster 95% CI | Low-minus-high margin disagreement | 95% CI |", "|---|---:|---:|---:|---:|---:|---:|"]
    for policy in ("ACT","SmolVLA"):
        r=output["policy_results"][policy]; md=r["margin_conditioning"]
        lines.append(f"| {policy} | {r['arm_normalized_dispersion']:.6f} | {r['gripper_normalized_dispersion']:.6f} | {r['R']:.3f} | [{r['R_episode_cluster_bootstrap_ci'][0]:.3f}, {r['R_episode_cluster_bootstrap_ci'][1]:.3f}] | {md['low_minus_high']:.4f} | [{md['episode_cluster_bootstrap_ci'][0]:.4f}, {md['episode_cluster_bootstrap_ci'][1]:.4f}] |")
    lines += ["", f"`R_ACT - R_SMOLVLA = {difference:.3f}`, paired episode-cluster 95% CI `[{difference_ci[0]:.3f}, {difference_ci[1]:.3f}]`.", ""]
    (ROOT / "track_b/report.md").write_text("\n".join(lines))
    print(json.dumps({"labels": output["labels"], "R_ACT": act_ratio, "R_SMOLVLA": smol_ratio, "difference": difference}, indent=2))


if __name__ == "__main__":
    main()
