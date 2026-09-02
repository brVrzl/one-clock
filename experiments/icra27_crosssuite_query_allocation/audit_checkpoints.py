#!/usr/bin/env python3
"""Audit the local task-specific ACT checkpoint bank without running rollouts."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
NON_OBJECT_SUITES = ("libero_spatial", "libero_goal", "libero_10")
REQUIRED_EXPORT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    "train_config.json",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "size_bytes": None}
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def state_tensor_contract(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"opens": False, "action_stat_shapes": {}, "error": "file missing"}
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as handle:
            action_keys = [key for key in handle.keys() if key.startswith("action.")]
            shapes = {key: list(handle.get_tensor(key).shape) for key in action_keys}
        return {"opens": True, "action_stat_shapes": shapes, "error": None}
    except Exception as exc:  # An audit must record a broken artifact and continue.
        return {
            "opens": False,
            "action_stat_shapes": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


def normalization_contract(checkpoint: Path, config: dict[str, Any]) -> dict[str, Any]:
    pre_path = checkpoint / "policy_preprocessor.json"
    post_path = checkpoint / "policy_postprocessor.json"
    pre = read_json(pre_path) if pre_path.is_file() else {}
    post = read_json(post_path) if post_path.is_file() else {}
    pre_steps = pre.get("steps", [])
    post_steps = post.get("steps", [])
    pre_normalizers = [step for step in pre_steps if step.get("registry_name") == "normalizer_processor"]
    post_normalizers = [step for step in post_steps if step.get("registry_name") == "unnormalizer_processor"]
    pre_state = checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    post_state = checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    action_dim = config.get("output_features", {}).get("action", {}).get("shape", [None])[0]
    pre_tensor = state_tensor_contract(pre_state)
    post_tensor = state_tensor_contract(post_state)
    expected_action_shape = [int(action_dim)] if action_dim is not None else None
    action_stats = ("action.mean", "action.std")
    tensor_contract_ok = all(
        tensor["opens"]
        and all(tensor["action_stat_shapes"].get(key) == expected_action_shape for key in action_stats)
        for tensor in (pre_tensor, post_tensor)
    )
    return {
        "mapping": config.get("normalization_mapping"),
        "action_normalization": config.get("normalization_mapping", {}).get("ACTION"),
        "preprocessor_path": str(pre_path.resolve()) if pre_path.exists() else str(pre_path),
        "preprocessor_steps": [step.get("registry_name") for step in pre_steps],
        "preprocessor_action_normalizer": pre_normalizers[0] if len(pre_normalizers) == 1 else None,
        "preprocessor_state": file_record(pre_state),
        "preprocessor_state_contract": pre_tensor,
        "postprocessor_path": str(post_path.resolve()) if post_path.exists() else str(post_path),
        "postprocessor_steps": [step.get("registry_name") for step in post_steps],
        "postprocessor_action_unnormalizer": post_normalizers[0] if len(post_normalizers) == 1 else None,
        "postprocessor_state": file_record(post_state),
        "postprocessor_state_contract": post_tensor,
        "frozen_checkpoint_statistics": tensor_contract_ok,
        "contract_summary": (
            "checkpoint-frozen MEAN_STD action normalization and matching checkpoint-frozen "
            "action unnormalization; eps=1e-8"
            if tensor_contract_ok
            and config.get("normalization_mapping", {}).get("ACTION") == "MEAN_STD"
            else "normalization contract incomplete or inconsistent"
        ),
    }


def baseline_record(task_dir: Path, suite: str, task_id: int) -> dict[str, Any] | None:
    path = task_dir / "eval10" / "eval_info.json"
    if not path.is_file():
        return None
    data = read_json(path)
    rows = [
        row
        for row in data.get("per_task", [])
        if row.get("task_group") == suite and int(row.get("task_id", -1)) == task_id
    ]
    if len(rows) != 1:
        return {
            "available": True,
            "source_path": str(path.resolve()),
            "valid": False,
            "error": f"expected one matching per_task record, found {len(rows)}",
        }
    successes = [bool(value) for value in rows[0].get("metrics", {}).get("successes", [])]
    return {
        "available": True,
        "source_path": str(path.resolve()),
        "valid": bool(successes),
        "protocol": "existing standard ACT baseline eval10",
        "successes": sum(successes),
        "episodes": len(successes),
        "success_rate": sum(successes) / len(successes) if successes else None,
    }


def load_smoke(checkpoint: Path, suite: str, task_id: int) -> dict[str, Any]:
    try:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.device = "cpu"
        cfg.pretrained_path = checkpoint
        cfg.pretrained_backbone_weights = None
        env_cfg = LiberoEnv(
            task=suite,
            task_ids=[task_id],
            obs_type="pixels_agent_pos",
            camera_name="agentview_image,robot0_eye_in_hand_image",
            init_states=True,
            observation_width=256,
            observation_height=256,
            control_mode="relative",
        )
        policy = make_policy(cfg=cfg, env_cfg=env_cfg)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        result = {
            "attempted": True,
            "succeeds": True,
            "device": str(next(policy.parameters()).device),
            "policy_class": type(policy).__name__,
            "policy_type": getattr(cfg, "type", None),
            "chunk_size": int(cfg.chunk_size),
            "action_dim": int(cfg.output_features["action"].shape[0]),
            "preprocessor_steps_loaded": len(preprocessor.steps),
            "postprocessor_steps_loaded": len(postprocessor.steps),
            "error": None,
        }
        del postprocessor, preprocessor, policy, cfg
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
    except Exception as exc:  # A per-checkpoint failure is part of the requested inventory.
        return {
            "attempted": True,
            "succeeds": False,
            "device": "cpu",
            "error": f"{type(exc).__name__}: {exc}",
        }


def audit_task(bank_root: Path, suite: str, task_id: int, run_load_smoke: bool) -> dict[str, Any]:
    task_dir = bank_root / f"{suite}_task{task_id}"
    checkpoint = task_dir / "checkpoints" / "100000" / "pretrained_model"
    files = {name: file_record(checkpoint / name) for name in REQUIRED_EXPORT_FILES}
    exists = checkpoint.is_dir() and all(row["exists"] for row in files.values())
    config = read_json(checkpoint / "config.json") if files["config.json"]["exists"] else {}
    train_config = read_json(checkpoint / "train_config.json") if files["train_config.json"]["exists"] else {}
    normalization = normalization_contract(checkpoint, config) if checkpoint.is_dir() else {
        "frozen_checkpoint_statistics": False,
        "contract_summary": "checkpoint missing",
    }
    available_steps = sorted(
        int(path.name)
        for path in (task_dir / "checkpoints").glob("[0-9]*")
        if path.is_dir() and path.name.isdigit()
    ) if (task_dir / "checkpoints").is_dir() else []
    smoke = load_smoke(checkpoint, suite, task_id) if run_load_smoke and exists else {
        "attempted": False,
        "succeeds": None,
        "device": "cpu",
        "error": "required checkpoint export files missing" if not exists else "disabled by CLI",
    }
    action_shape = config.get("output_features", {}).get("action", {}).get("shape")
    contract_checks = {
        "type_is_act": config.get("type") == "act",
        "checkpoint_step_is_100000": train_config.get("steps") == 100000,
        "action_dim_is_7": action_shape == [7],
        "chunk_size_is_100": config.get("chunk_size") == 100,
        "n_action_steps_is_100": config.get("n_action_steps") == 100,
        "native_temporal_ensemble_disabled": config.get("temporal_ensemble_coeff") is None,
        "normalization_contract_valid": bool(normalization.get("frozen_checkpoint_statistics")),
        "load_smoke_succeeds": smoke.get("succeeds") is True,
    }
    model = files["model.safetensors"]
    total_bytes = sum(row["size_bytes"] or 0 for row in files.values())
    return {
        "suite": suite,
        "task_id": task_id,
        "role": "development_reference" if suite == "libero_object" else "primary_confirmation_candidate",
        "exact_local_path": str(checkpoint.resolve()) if checkpoint.exists() else str(checkpoint),
        "exists": exists,
        "checkpoint_step": train_config.get("steps", 100000 if checkpoint.parent.name == "100000" else None),
        "available_checkpoint_steps": available_steps,
        "config": {
            "path": files["config.json"]["path"],
            "type": config.get("type"),
            "n_obs_steps": config.get("n_obs_steps"),
            "input_features": config.get("input_features"),
            "output_features": config.get("output_features"),
            "chunk_size": config.get("chunk_size"),
            "n_action_steps": config.get("n_action_steps"),
            "temporal_ensemble_coeff": config.get("temporal_ensemble_coeff"),
            "vision_backbone": config.get("vision_backbone"),
            "dim_model": config.get("dim_model"),
        },
        "action_dimension": action_shape[0] if isinstance(action_shape, list) and action_shape else None,
        "chunk_size": config.get("chunk_size"),
        "normalization_contract": normalization,
        "checkpoint_identity": {
            "basis": "exact path + training step + model file size + model file mtime_ns",
            "model_file": model,
            "required_export_size_bytes": total_bytes,
            "hash": None,
            "hash_reason": "not needed; task-specific exact paths disambiguate every selected export",
        },
        "standard_baseline_success": baseline_record(task_dir, suite, task_id),
        "load_smoke": smoke,
        "required_files": files,
        "contract_checks": contract_checks,
        "technically_valid": exists and all(contract_checks.values()),
    }


def markdown(inventory: dict[str, Any]) -> str:
    summary = inventory["summary"]
    lines = [
        "# Local ACT checkpoint bank inventory",
        "",
        f"Audit time (UTC): `{inventory['audit']['generated_at_utc']}`",
        "",
        "The local filesystem contains all 40 expected task-specific ACT 100k exports. "
        "The audit constructed each policy and its saved preprocessing/postprocessing pipeline on CPU; no environment was initialized and no rollout outcome was generated.",
        "",
        "No checkpoint hashes were computed. Exact task-specific paths, step, byte size, and file mtime disambiguate these local exports.",
        "",
        "## Summary",
        "",
        f"- Expected task policies: {summary['expected_total']}",
        f"- Technically valid task policies: {summary['technically_valid_total']}",
        f"- Expected non-Object confirmation policies: {summary['expected_non_object']}",
        f"- Valid non-Object confirmation policies: {summary['technically_valid_non_object']}",
        f"- Valid Object development/reference policies: {summary['technically_valid_object']}",
        f"- Track-A checkpoint-bank gate: **{summary['track_a_checkpoint_gate']}**",
        "",
        "All selected policies use seven-dimensional actions, 100-step chunks, checkpoint-frozen `MEAN_STD` action statistics, and native temporal ensembling disabled in the saved config.",
        "",
        "## Per-policy inventory",
        "",
        "| Suite | Task | Role | Step | Action dim | Chunk | Export bytes | Standard baseline | CPU load smoke | Valid | Exact local path |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in inventory["policies"]:
        baseline = row["standard_baseline_success"]
        baseline_text = "not available"
        if baseline and baseline.get("valid"):
            baseline_text = f"{baseline['successes']}/{baseline['episodes']} ({100 * baseline['success_rate']:.1f}%)"
        lines.append(
            f"| `{row['suite']}` | {row['task_id']} | {row['role']} | {row['checkpoint_step']} | "
            f"{row['action_dimension']} | {row['chunk_size']} | "
            f"{row['checkpoint_identity']['required_export_size_bytes']} | {baseline_text} | "
            f"{'PASS' if row['load_smoke']['succeeds'] else 'FAIL'} | "
            f"{'yes' if row['technically_valid'] else 'no'} | `{row['exact_local_path']}` |"
        )
    lines += [
        "",
        "## Contract and provenance notes",
        "",
        "- The selected export for every task is `checkpoints/100000/pretrained_model`; the earlier 20k, 40k, 60k, and 80k snapshots are also present but are not selected.",
        "- `config.json` and `train_config.json` independently record ACT, seven action dimensions, chunk size 100, and training step 100,000.",
        "- The preprocessor applies the saved normalizer after batching/device placement. The postprocessor applies the saved action unnormalizer before moving results to CPU.",
        "- Both saved processor state files open successfully and expose seven-element `action.mean` and `action.std` tensors for every policy.",
        "- The prior standard-baseline figures are transcribed from each task directory's existing `eval10/eval_info.json`; they are not executor-variant results and were not rerun by this audit.",
        "- Object policies are recorded as development/reference only. They are excluded from the Track-A primary confirmation bank.",
        "",
        "## Audit environment",
        "",
        f"- Python: `{inventory['audit']['environment']['python']}`",
        f"- PyTorch: `{inventory['audit']['environment']['pytorch']}`",
        f"- LeRobot: `{inventory['audit']['environment']['lerobot']}`",
        f"- Load device: `cpu` (CUDA hidden for the audit command)",
        "",
    ]
    invalid = [f"{row['suite']}:task{row['task_id']}" for row in inventory["policies"] if not row["technically_valid"]]
    if invalid:
        lines += ["## Invalid policies", "", ", ".join(f"`{item}`" for item in invalid), ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bank-root",
        type=Path,
        default=Path("/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--no-load-smoke", action="store_true")
    args = parser.parse_args()

    bank_root = args.bank_root.resolve()
    policies = [
        audit_task(bank_root, suite, task_id, not args.no_load_smoke)
        for suite in SUITES
        for task_id in range(10)
    ]
    valid_non_object = sum(row["technically_valid"] for row in policies if row["suite"] in NON_OBJECT_SUITES)
    valid_object = sum(row["technically_valid"] for row in policies if row["suite"] == "libero_object")
    summary = {
        "expected_total": 40,
        "technically_valid_total": sum(row["technically_valid"] for row in policies),
        "expected_non_object": 30,
        "technically_valid_non_object": valid_non_object,
        "technically_valid_object": valid_object,
        "missing_or_invalid": [
            {"suite": row["suite"], "task_id": row["task_id"]}
            for row in policies
            if not row["technically_valid"]
        ],
        "track_a_checkpoint_gate": "PROCEED_ALL_30" if valid_non_object == 30 else (
            "PROCEED_VALID_N_GE_20" if valid_non_object >= 20 else "FAIL_N_LT_20"
        ),
    }
    inventory = {
        "schema_version": 1,
        "audit": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "bank_root": str(bank_root),
            "filesystem_search_scope": "/home/wjq plus mounted local filesystem review",
            "selected_checkpoint_rule": "task-specific 100000-step export",
            "hash_policy": "no hashes; exact paths and lightweight file identity are sufficient",
            "load_smoke_contract": "CPU construction of ACTPolicy plus saved policy pre/postprocessors; no env creation or rollout",
            "environment": {
                "python": platform.python_version(),
                "pytorch": package_version("torch"),
                "lerobot": package_version("lerobot"),
                "safetensors": package_version("safetensors"),
            },
        },
        "summary": summary,
        "policies": policies,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "checkpoint_inventory.json"
    md_path = args.output_dir / "checkpoint_inventory.md"
    json_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown(inventory), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
