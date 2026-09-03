#!/usr/bin/env python3
"""Frozen B1 per-dimension and source-age same-target analysis."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open


ROOT = Path(__file__).resolve().parent
GROUPS = {
    "translation": (0, 1, 2),
    "rotation": (3, 4, 5),
    "gripper": (6,),
    "arm_original": (0, 1, 2, 3, 4, 5),
}


def action_transform(checkpoint: str) -> tuple[np.ndarray, np.ndarray]:
    root = Path(checkpoint)
    config = json.loads((root / "policy_preprocessor.json").read_text())
    normalizers = [step for step in config["steps"] if step["registry_name"] == "normalizer_processor"]
    if len(normalizers) != 1 or normalizers[0]["config"]["norm_map"].get("ACTION") != "MEAN_STD":
        raise RuntimeError(f"unverified action normalization contract: {root}")
    state_file = root / normalizers[0]["state_file"]
    with safe_open(state_file, framework="numpy") as handle:
        mean = np.asarray(handle.get_tensor("action.mean"), dtype=np.float64)
        std = np.asarray(handle.get_tensor("action.std"), dtype=np.float64)
    if mean.shape != (7,) or std.shape != (7,) or not np.isfinite(mean).all() or np.any(std <= 0):
        raise RuntimeError(f"invalid action normalization statistics: {state_file}")
    return mean, std


def pairwise_sign_disagreement(values: np.ndarray) -> np.ndarray:
    signs = np.sign(values)
    return np.asarray([
        sum(row[i] != row[j] for i in range(16) for j in range(i + 1, 16)) / 120
        for row in signs
    ], dtype=np.float64)


def interval(values: list[float], *, seed: int, draws: int) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all():
        raise RuntimeError("invalid episode metric vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(x), size=(draws, len(x)))
    boot = x[indices].mean(axis=1)
    return {
        "episode_count": int(len(x)),
        "mean": float(x.mean()),
        "episode_sd": float(x.std(ddof=1)),
        "episode_cluster_bootstrap_ci": np.percentile(boot, [2.5, 97.5]).astype(float).tolist(),
    }


def main() -> None:
    manifest = json.loads((ROOT / "track_b_manifest.json").read_text())
    addendum = json.loads((ROOT / "track_b_analysis_addendum.json").read_text())
    if addendum.get("status") != "FROZEN_BEFORE_TRACK_B_PREDICTION_INTERPRETATION":
        raise RuntimeError("Track-B analysis addendum is not frozen")
    missing = [
        c["cell_id"] for c in manifest["cells"]
        if not (ROOT / "track_b/results" / f"{c['cell_id']}.json").is_file()
        or not (ROOT / "track_b/predictions" / f"{c['cell_id']}.npz").is_file()
        or not (ROOT / "track_b/markers" / f"{c['cell_id']}.complete").is_file()
    ]
    if missing:
        raise RuntimeError(f"Track B incomplete: {len(missing)} cells")

    dispersion_rows: list[dict[str, Any]] = []
    age_rows: list[dict[str, Any]] = []
    native_dispersion_rows: list[dict[str, Any]] = []
    native_age_rows: list[dict[str, Any]] = []
    native_sign_rows: list[dict[str, Any]] = []
    transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cell in manifest["cells"]:
        metadata = json.loads((ROOT / "track_b/results" / f"{cell['cell_id']}.json").read_text())
        for key in ("cell_id", "policy", "suite", "task_id", "state_id", "checkpoint"):
            if metadata.get(key) != cell.get(key):
                raise RuntimeError(f"identity mismatch {cell['cell_id']}:{key}")
        with np.load(ROOT / "track_b/predictions" / f"{cell['cell_id']}.npz") as payload:
            chunks = np.asarray(payload["predicted_chunks_normalized"], dtype=np.float64)
        if cell["checkpoint"] not in transforms:
            transforms[cell["checkpoint"]] = action_transform(cell["checkpoint"])
        action_mean, action_std = transforms[cell["checkpoint"]]
        native_chunks = chunks * action_std + action_mean
        steps = int(metadata["environment_steps"])
        if chunks.shape != (steps, int(metadata["chunk_size"]), 7):
            raise RuntimeError(f"prediction shape mismatch: {cell['cell_id']}")
        candidates = np.stack(
            [np.stack([chunks[t - age, age] for age in range(16)]) for t in range(15, steps)]
        )
        native_candidates = np.stack(
            [np.stack([native_chunks[t - age, age] for age in range(16)]) for t in range(15, steps)]
        )
        target_dispersion = candidates.std(axis=1, ddof=0)
        base = {
            "cell_id": cell["cell_id"], "policy": cell["policy"], "suite": cell["suite"],
            "task_id": int(cell["task_id"]), "state_id": int(cell["state_id"]),
            "eligible_targets": int(len(target_dispersion)),
        }
        for dim in range(7):
            dispersion_rows.append({**base, "metric": f"dim_{dim}", "value": float(target_dispersion[:, dim].mean())})
        for name, dims in GROUPS.items():
            grouped = np.sqrt(np.mean(np.square(target_dispersion[:, dims]), axis=1))
            dispersion_rows.append({**base, "metric": name, "value": float(grouped.mean())})
            native_dispersion = native_candidates[:, :, dims].std(axis=1, ddof=0)
            native_grouped = np.sqrt(np.mean(np.square(native_dispersion), axis=1))
            native_dispersion_rows.append({**base, "metric": name, "value": float(native_grouped.mean()), "units": "controller_native_action"})

        native_sign = pairwise_sign_disagreement(native_candidates[:, :, 6])
        normalized_margin = np.abs(candidates[:, :, 6].mean(axis=1))
        for index, target in enumerate(range(15, steps)):
            native_sign_rows.append({
                **base, "target_time": target,
                "target_id": f"{cell['cell_id']}:t{target}",
                "normalized_margin": float(normalized_margin[index]),
                "native_sign_disagreement": float(native_sign[index]),
            })

        fresh_delta = candidates - candidates[:, :1, :]
        native_fresh_delta = native_candidates - native_candidates[:, :1, :]
        for age in range(16):
            for dim in range(7):
                value = float(np.sqrt(np.mean(np.square(fresh_delta[:, age, dim]))))
                age_rows.append({**base, "source_age": age, "metric": f"dim_{dim}", "value": value})
            for name, dims in GROUPS.items():
                value = float(np.sqrt(np.mean(np.square(fresh_delta[:, age][:, dims]))))
                age_rows.append({**base, "source_age": age, "metric": name, "value": value})
                native_value = float(np.sqrt(np.mean(np.square(native_fresh_delta[:, age][:, dims]))))
                native_age_rows.append({**base, "source_age": age, "metric": name, "value": native_value, "units": "controller_native_action"})

    output_dir = ROOT / "track_b" / "analysis_addendum"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("episode_dimension_dispersion.csv", dispersion_rows),
        ("episode_age_disagreement.csv", age_rows),
        ("episode_native_dispersion.csv", native_dispersion_rows),
        ("episode_native_age_difference.csv", native_age_rows),
        ("native_gripper_sign_disagreement.csv", native_sign_rows),
    ):
        with (output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    summaries: dict[str, Any] = {"dimension_dispersion": {}, "age_disagreement": {}, "native_dimension_dispersion": {}, "native_age_difference": {}, "native_gripper": {}}
    draws = int(addendum["b1"]["bootstrap_draws"])
    policies = ("ACT", "SmolVLA")
    metrics = tuple([f"dim_{i}" for i in range(7)] + list(GROUPS))
    for policy_index, policy in enumerate(policies):
        summaries["dimension_dispersion"][policy] = {}
        for metric_index, metric in enumerate(metrics):
            values = [r["value"] for r in dispersion_rows if r["policy"] == policy and r["metric"] == metric]
            seed = int(addendum["b1"]["dimension_seed"]) + 1000 * policy_index + metric_index
            summaries["dimension_dispersion"][policy][metric] = interval(values, seed=seed, draws=draws)
        summaries["age_disagreement"][policy] = {}
        summaries["native_dimension_dispersion"][policy] = {}
        for metric_index, metric in enumerate(GROUPS):
            values = [r["value"] for r in native_dispersion_rows if r["policy"] == policy and r["metric"] == metric]
            summaries["native_dimension_dispersion"][policy][metric] = interval(values, seed=int(addendum["b1"]["dimension_seed"]) + 20000 + 1000 * policy_index + metric_index, draws=draws)
        summaries["native_age_difference"][policy] = {}
        for age in range(16):
            summaries["age_disagreement"][policy][str(age)] = {}
            summaries["native_age_difference"][policy][str(age)] = {}
            for metric_index, metric in enumerate(metrics):
                values = [r["value"] for r in age_rows if r["policy"] == policy and r["source_age"] == age and r["metric"] == metric]
                seed = int(addendum["b1"]["age_curve_seed"]) + 10000 * policy_index + 100 * age + metric_index
                summaries["age_disagreement"][policy][str(age)][metric] = interval(values, seed=seed, draws=draws)
            for metric_index, metric in enumerate(GROUPS):
                values = [r["value"] for r in native_age_rows if r["policy"] == policy and r["source_age"] == age and r["metric"] == metric]
                summaries["native_age_difference"][policy][str(age)][metric] = interval(values, seed=int(addendum["b1"]["age_curve_seed"]) + 20000 + 10000 * policy_index + 100 * age + metric_index, draws=draws)

        sign_rows = [r for r in native_sign_rows if r["policy"] == policy]
        ranked = sorted(sign_rows, key=lambda r: (r["normalized_margin"], r["suite"], r["task_id"], r["state_id"], r["target_time"]))
        tercile_n = len(ranked) // 3
        low = ranked[:tercile_n]; high = ranked[-tercile_n:]
        summaries["native_gripper"][policy] = {
            "pairwise_sign_state_disagreement_probability": float(np.mean([r["native_sign_disagreement"] for r in sign_rows])),
            "low_margin_disagreement_probability": float(np.mean([r["native_sign_disagreement"] for r in low])),
            "high_margin_disagreement_probability": float(np.mean([r["native_sign_disagreement"] for r in high])),
            "low_minus_high_margin_disagreement": float(np.mean([r["native_sign_disagreement"] for r in low]) - np.mean([r["native_sign_disagreement"] for r in high])),
            "margin_terciles": "original frozen normalized mean-gripper absolute-margin stable rank",
        }

    output = {
        "status": "COMPLETE",
        "analysis_addendum_status": addendum["status"],
        "success_outcomes_loaded": False,
        "episodes": len(manifest["cells"]),
        "normalization": manifest["normalization"],
        "primary_window": manifest["primary_window"],
        "age_curve_reference": "same-target age-0 prediction",
        "native_action_contract": "exact inverse checkpoint MEAN_STD transform; controller-native action values, not measured millimetres/degrees",
        "bootstrap": {"unit": "episode", "draws": draws, "ci_percentiles": [2.5, 97.5]},
        **summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n")

    lines = [
        "# Track-B B1 per-dimension same-target analysis", "",
        "All values use each checkpoint's frozen normalized action space and the frozen 16-source primary window. These explanatory analyses do not alter the original Track-B labels.", "",
        "| Policy | Translation | Rotation | Gripper | Original arm |", "|---|---:|---:|---:|---:|",
    ]
    for policy in policies:
        s = output["dimension_dispersion"][policy]
        lines.append(f"| {policy} | {s['translation']['mean']:.6f} | {s['rotation']['mean']:.6f} | {s['gripper']['mean']:.6f} | {s['arm_original']['mean']:.6f} |")
    lines += ["", "## Controller-native diagnostics", "", "Values below invert the exact checkpoint MEAN_STD transform and remain in controller-native action units.", "", "| Policy | Translation source dispersion | Rotation source dispersion | Gripper source dispersion | Gripper sign disagreement | Low-minus-high margin disagreement |", "|---|---:|---:|---:|---:|---:|"]
    for policy in policies:
        n = output["native_dimension_dispersion"][policy]; g = output["native_gripper"][policy]
        lines.append(f"| {policy} | {n['translation']['mean']:.6f} | {n['rotation']['mean']:.6f} | {n['gripper']['mean']:.6f} | {g['pairwise_sign_state_disagreement_probability']:.6f} | {g['low_minus_high_margin_disagreement']:.6f} |")
    lines += ["", "Age-resolved normalized and controller-native differences are in the accompanying tidy CSV/JSON files. Normalized dispersion remains descriptive context; no success outcome was loaded.", ""]
    (output_dir / "report.md").write_text("\n".join(lines))
    print(json.dumps({policy: {metric: output["dimension_dispersion"][policy][metric]["mean"] for metric in ("translation", "rotation", "gripper", "arm_original")} for policy in policies}, indent=2))


if __name__ == "__main__":
    main()
