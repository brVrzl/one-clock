#!/usr/bin/env python3
"""Generate outcome-free Phase-0 implementation, normalization, and moderator audits."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.component_temporal_reuse.temporal_operators import weighted_gripper_vote  # noqa: E402
from experiments.group_temporal_memory_dev.group_memory_operators import (  # noqa: E402
    m1_shared_te,
    m2_shared_cogact,
    m3_group_cogact,
)
from lerobot.policies.act.modeling_act import ACTTemporalEnsembler  # noqa: E402


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def processor_state(checkpoint: Path, prefix: str) -> Path:
    matches = sorted(checkpoint.glob(f"{prefix}*.safetensors"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {prefix} state in {checkpoint}, found {matches}")
    return matches[0]


def action_stats(path: Path) -> dict[str, list[float]]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {
            key: handle.get_tensor(key).detach().cpu().numpy().astype(float).tolist()
            for key in ("action.mean", "action.std")
        }


def normalization_audit() -> dict[str, Any]:
    inventory = json.loads((ROOT / "checkpoint_inventory.json").read_text(encoding="utf-8"))
    rows = []
    for policy in inventory["policies"]:
        checkpoint = Path(policy["exact_local_path"])
        pre = processor_state(checkpoint, "policy_preprocessor_step_")
        post = processor_state(checkpoint, "policy_postprocessor_step_")
        pre_stats, post_stats = action_stats(pre), action_stats(post)
        if pre_stats != post_stats:
            raise RuntimeError(f"ACT pre/post action statistics differ: {checkpoint}")
        rows.append({
            "suite": policy["suite"], "task_id": policy["task_id"],
            "checkpoint": str(checkpoint), "action_dimension": 7,
            "normalization": "MEAN_STD", "preprocessor_buffer": str(pre),
            "postprocessor_buffer": str(post), "pre_post_action_stats_identical": True,
            "action_mean": pre_stats["action.mean"], "action_std": pre_stats["action.std"],
        })
    smol = Path("/home/wjq/checkpoints/HuggingFaceVLA_smolvla_libero")
    config = json.loads((smol / "config.json").read_text(encoding="utf-8"))
    pre = processor_state(smol, "policy_preprocessor_step_")
    post = processor_state(smol, "policy_postprocessor_step_")
    pre_stats, post_stats = action_stats(pre), action_stats(post)
    if pre_stats != post_stats:
        raise RuntimeError("SmolVLA pre/post action statistics differ")
    smol_row = {
        "checkpoint": str(smol), "policy_type": config["type"],
        "action_dimension": config["output_features"]["action"]["shape"][0],
        "chunk_size": config["chunk_size"], "normalization": config["normalization_mapping"]["ACTION"],
        "preprocessor_buffer": str(pre), "postprocessor_buffer": str(post),
        "pre_post_action_stats_identical": True,
        "action_mean": pre_stats["action.mean"], "action_std": pre_stats["action.std"],
    }
    return {
        "status": "PASS", "normalization_refit_allowed": False,
        "prediction_logging_space": "native checkpoint-normalized model output before checkpoint postprocessor",
        "act_checkpoints": rows, "smolvla_checkpoint": smol_row,
    }


def te_audit() -> dict[str, Any]:
    coefficient, horizon, action_dim, steps = 0.01, 100, 7, 37
    generator = torch.Generator().manual_seed(20270902)
    chunks = [torch.randn(1, horizon, action_dim, generator=generator) for _ in range(steps)]
    ensembler = ACTTemporalEnsembler(coefficient, horizon)
    observed, expected = [], []
    for target, chunk in enumerate(chunks):
        observed.append(ensembler.update(chunk).clone())
        candidates = torch.stack([chunks[source][0, target - source] for source in range(target + 1)])
        weights = torch.exp(-coefficient * torch.arange(target + 1, dtype=candidates.dtype))
        expected.append(((weights[:, None] * candidates).sum(0) / weights.sum())[None])
    maximum_error = max(float(torch.max(torch.abs(a - b))) for a, b in zip(observed, expected, strict=True))
    source = inspect.getsource(ACTTemporalEnsembler)
    checks = {
        "coefficient_0_01": coefficient == 0.01,
        "oldest_candidate_is_weight_index_0": "w₀ is the oldest action" in source,
        "online_matches_explicit_oldest_to_newest_weighting": maximum_error < 2e-6,
        "all_seven_dimensions_aggregated_together": all(x.shape == (1, 7) for x in observed),
    }
    if not all(checks.values()):
        raise RuntimeError(f"canonical TE audit failed: {checks}, max_error={maximum_error}")
    return {
        "status": "PASS", "implementation": "lerobot.policies.act.modeling_act.ACTTemporalEnsembler",
        "coefficient": coefficient, "query_frequency": 1,
        "candidate_semantics": "for physical target t, sources q=0..t (bounded by chunk_size), oldest q receives index 0",
        "weight_formula": "w_i = exp(-0.01*i), i=0 for oldest source",
        "aggregation_space": "checkpoint-normalized action space, all 7 dimensions jointly",
        "postprocessing_order": "ensemble normalized predictions, then checkpoint unnormalize, then environment postprocess",
        "synthetic_steps": steps, "maximum_absolute_error": maximum_error, "checks": checks,
    }


def historical_operator_audit() -> dict[str, Any]:
    candidates = np.asarray([
        [1, 0, 0, 0, 0, 0, -0.7],
        [0, 1, 0, 0, 0, 0, 0.2],
        [1, 1, 0, 0, 0, 0, 0.8],
    ], dtype=float)
    ages = np.asarray([32, 16, 0], dtype=float)
    m1, _ = m1_shared_te(candidates, ages, kernel_name="dense_equivalent_te")
    m2, _ = m2_shared_cogact(candidates, ages, kernel_name="dense_equivalent_te")
    m3, m3_diag = m3_group_cogact(candidates, ages, kernel_name="dense_equivalent_te")
    voted, representative, winning_sign, support = weighted_gripper_vote(candidates, np.ones(3))
    rapid = REPO_ROOT / "experiments/component_temporal_reuse/rapid_component_smoke/run_act_groupwise_smoke.py"
    rapid_text = rapid.read_text(encoding="utf-8")
    results = {
        "status": "PASS",
        "operators": {
            "M1_shared_te_h16": {"query_rate_nominal": 1/16, "candidate_age_range": "native same-target sparse sources, up to ACT chunk validity", "aggregation": "continuous weighted average", "affected_dimensions": "0..6 with one shared weight vector", "executes_sign_vote": False},
            "M2_shared_cogact": {"query_rate_nominal": 1/16, "candidate_age_range": "same pool as M1", "aggregation": "continuous weighted average", "affected_dimensions": "0..6 with one compatibility-weighted vector", "executes_sign_vote": False},
            "M3_group_cogact": {"query_rate_nominal": 1/16, "candidate_age_range": "same pool as M1", "aggregation": "continuous weighted average with separately normalized arm/gripper weights", "affected_dimensions": "arm 0..5 and gripper 6", "np_sign_role": "gripper compatibility weights only", "executes_sign_vote": False},
            "canonical_dense_ACT_TE": {"query_rate_nominal": 1.0, "candidate_age_range": "dense sources ages 0..min(t,99)", "aggregation": "continuous exponential weighted average", "affected_dimensions": "0..6 with one shared temporal weight vector", "executes_sign_vote": False},
        },
        "historical_sign_vote_exists": True,
        "historical_sign_vote_path": str(rapid),
        "historical_sign_vote_method": "SIGN_VOTE in rapid_component_smoke",
        "historical_sign_vote_semantics": "weighted open/close support, newest tie break, then execute an original scalar representative from the winning sign",
        "do_not_reinvent": True,
        "np_sign_findings": [
            "group_temporal_memory_dev uses np.sign only to form M3 gripper compatibility weights; M3 output remains a continuous weighted scalar average",
            "cdta_dev uses np.sign only in compatibility/weight computation",
            "dynamic_horizon_dev and dynamic_horizon_h16_dev use np.sign only as a requery trigger",
            "group_temporal_memory_offline uses np.sign only for diagnostic disagreement metrics",
            "the distinct rapid_component_smoke SIGN_VOTE path does execute a discrete sign-supported gripper representative",
        ],
        "synthetic_outputs": {"M1": m1.tolist(), "M2": m2.tolist(), "M3": m3.tolist(), "M3_gripper_weights": m3_diag["gripper_weights"].tolist(), "sign_vote_scalar": voted.tolist(), "sign_vote_representative": representative, "sign_vote_winning_sign": winning_sign, "sign_vote_support": support},
        "rapid_path_calls_weighted_gripper_vote": "weighted_gripper_vote(" in rapid_text,
    }
    if not results["rapid_path_calls_weighted_gripper_vote"]:
        raise RuntimeError("historical sign-vote call disappeared")
    return results


def moderator_audit() -> dict[str, Any]:
    manifest = json.loads((REPO_ROOT / "experiments/standard_libero_baselines/act_task_manifest_corrected.json").read_text(encoding="utf-8"))
    episode_path = hf_hub_download("HuggingFaceVLA/libero", "meta/episodes/chunk-000/file-000.parquet", repo_type="dataset", revision="v3.0")
    table = pq.read_table(episode_path, columns=["episode_index", "length", "stats/action/min", "stats/action/max", "stats/action/mean", "stats/action/count"])
    by_episode = {int(row["episode_index"]): row for row in table.to_pylist()}
    rows = []
    for task in manifest:
        if task["suite"] == "libero_object":
            continue
        closed_steps, total_steps = 0.0, 0
        for episode in task["episodes"]:
            row = by_episode[int(episode)]
            minimum, maximum = row["stats/action/min"][6], row["stats/action/max"][6]
            if minimum < -1.000001 or maximum > 1.000001:
                raise RuntimeError("gripper command outside frozen +/-1 contract")
            count = int(row["stats/action/count"][0])
            mean = float(row["stats/action/mean"][6])
            closed_steps += count * (1.0 - mean) / 2.0
            total_steps += count
        rows.append({
            "suite": task["suite"], "task_id": task["task_id"], "dataset_task_index": task["dataset_task_index"],
            "training_episode_count": len(task["episodes"]), "training_action_steps": total_steps,
            "gripper_manipulation_frequency": closed_steps / total_steps,
        })
    return {
        "status": "FROZEN_BEFORE_TRACK_A_OUTCOMES", "source_dataset": "HuggingFaceVLA/libero@v3.0",
        "source_revision": "86958911c0f959db2bbbdb107eb3e17c5f9c798e",
        "source": "training-demonstration episode metadata only; no evaluation outcomes",
        "moderator_definition": "fraction of task-specific ACT training action steps with closed gripper intent (action[6] < 0); LIBERO gripper commands are +/-1, reconstructed exactly from frozen episode mean and count",
        "hypothesis": "exploratory Spearman correlation between task-level ARM4_GRIP32-H4 success delta and frozen gripper manipulation frequency",
        "no_posthoc_categories": True, "tasks": rows,
    }


def write_markdown(te: dict[str, Any], historical: dict[str, Any], normalization: dict[str, Any]) -> None:
    lines = [
        "# Historical operator and implementation audit", "",
        "Status: **PASS**. This audit used source code and synthetic predictions only; it loaded no Track-A outcomes.", "",
        "## Canonical dense ACT temporal ensembling", "",
        f"- Implementation: `{te['implementation']}`.",
        "- Query every environment step (`query_frequency=1`).",
        "- For a physical target, candidates are ordered oldest source to newest source and weighted `exp(-0.01*i)`, with the oldest at `i=0`.",
        "- The online LeRobot implementation matched an explicit offline same-target weighted sum in normalized action space across all seven dimensions.",
        f"- Synthetic maximum absolute discrepancy: `{te['maximum_absolute_error']:.3g}`.",
        "- The checkpoint postprocessor/unnormalizer and environment postprocessor are applied only after temporal aggregation.", "",
        "This is canonical dense ACT TE. It is not the historical sparse approximation in `temporal_operators.py`.", "",
        "## Historical operators", "",
        "| Operator | Nominal query rate | Candidate ages | Aggregation | Dimensions | Sign-vote output? |", "|---|---:|---|---|---|---|",
    ]
    for name, row in historical["operators"].items():
        lines.append(f"| {name} | {row['query_rate_nominal']:.4g} | {row['candidate_age_range']} | {row['aggregation']} | {row['affected_dimensions']} | {row['executes_sign_vote']} |")
    lines += ["", "M3's `np.sign` forms compatibility weights only; its executed gripper scalar is still a continuous weighted average. CDTA likewise uses signs for weights, dynamic-horizon code uses them for triggers, and offline code uses them for metrics.", "", "A distinct historical discrete output already exists: `rapid_component_smoke`'s `SIGN_VOTE` calls `weighted_gripper_vote`, chooses weighted open/close support, and executes an original candidate scalar from the winning sign. It must not be reinvented in this session.", "", "## ACT and SmolVLA normalization buffers", "", f"- ACT: {len(normalization['act_checkpoints'])}/40 audited checkpoint exports have matching preprocessor and postprocessor action mean/std buffers.", "- SmolVLA: its frozen MEAN_STD action buffers match between preprocessing and postprocessing.", "- Track B logs native model outputs before the checkpoint postprocessor, so dispersion is measured in each checkpoint's own frozen normalized space. No scale is refit on the diagnostic panel.", ""]
    (ROOT / "historical_operator_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    te = te_audit()
    historical = historical_operator_audit()
    normalization = normalization_audit()
    moderator = moderator_audit()
    dump(ROOT / "te_dense_audit.json", te)
    dump(ROOT / "historical_operator_audit.json", historical)
    dump(ROOT / "normalization_buffer_audit.json", normalization)
    dump(ROOT / "gripper_activity_moderator.json", moderator)
    write_markdown(te, historical, normalization)
    print(json.dumps({"TE_DENSE": te["status"], "historical": historical["status"], "normalization": normalization["status"], "moderator_tasks": len(moderator["tasks"])}, indent=2))


if __name__ == "__main__":
    main()
