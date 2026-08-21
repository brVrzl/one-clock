"""Run the predeclared RoboTwin smoke preflight.

The smoke subset is fixed in source and is not selected from observed policy
outcomes.  A real forward pass is attempted only when a local checkpoint is
supplied and its contract matches the local dataset.  A contract mismatch is
recorded as a blocked smoke check rather than converted into a large cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .cache_builder import ContractMismatch, audit_checkpoint_contract


PREDECLARED_TASKS = [
    {"task": "move_can_pot", "coverage": "single-side manipulation"},
    {"task": "pick_dual_bottles", "coverage": "bimanual coordination"},
    {"task": "handover_block", "coverage": "handover"},
    {"task": "stack_blocks_two", "coverage": "stacking"},
    {"task": "place_object_scale", "coverage": "precision/contact"},
    {"task": "stack_blocks_three", "coverage": "longer manipulation"},
]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _persist_preflight_progress(path: Path, result: dict[str, Any], args: argparse.Namespace) -> None:
    """Keep detached-run state visible even when preflight blocks inference."""
    if path.is_file():
        progress = _read_json(path)
    else:
        progress = {
            "tasks_completed": [],
            "episodes_completed": [],
            "frames_inferred": 0,
            "failures": [],
            "throughput_frames_per_second": None,
            "cache_bytes": 0,
            "checkpoint_revision": args.checkpoint_revision,
            "dataset_revision": args.dataset_revision,
        }
    progress["status"] = result["status"]
    if result["status"].startswith("blocked"):
        progress.setdefault("failures", []).append(
            {"stage": "smoke_preflight", "reason": result.get("contract_error", result.get("note"))}
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(progress, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    manifests = path.parent / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for name in ("completed_shards.json", "failed_shards.json"):
        target = manifests / name
        if not target.exists():
            target.write_text(json.dumps({"shards": []}, indent=2), encoding="utf-8")


def _snapshot_contract(dataset_info: dict[str, Any], config: dict[str, Any], preprocessor: dict[str, Any]) -> dict[str, Any]:
    state_shape = list(dataset_info["features"]["observation.state"]["shape"])
    action_shape = list(dataset_info["features"]["action"]["shape"])
    policy_state = list(config["input_features"]["observation.state"]["shape"])
    policy_action = list(config["output_features"]["action"]["shape"])
    expected_cameras = {
        "observation.images.cam_high": "observation.images.camera1",
        "observation.images.cam_left_wrist": "observation.images.camera2",
        "observation.images.cam_right_wrist": "observation.images.camera3",
    }
    rename_map = dict(preprocessor.get("camera_rename_map", {}))
    mismatches: list[str] = []
    if state_shape != policy_state:
        mismatches.append(f"state shape dataset={state_shape} policy={policy_state}")
    if action_shape != policy_action:
        mismatches.append(f"action shape dataset={action_shape} policy={policy_action}")
    if int(config.get("chunk_size", -1)) != 50:
        mismatches.append("primary cache schema requires chunk_size=50")
    if int(config.get("n_action_steps", -1)) != int(config.get("chunk_size", -2)):
        mismatches.append("primary cache schema requires n_action_steps=chunk_size")
    if rename_map != expected_cameras:
        mismatches.append("camera rename map does not match the RoboTwin camera keys")
    return {
        "dataset_state_shape": state_shape,
        "dataset_action_shape": action_shape,
        "policy_state_shape": policy_state,
        "policy_action_shape": policy_action,
        "policy_chunk_size": int(config.get("chunk_size", -1)),
        "policy_n_action_steps": int(config.get("n_action_steps", -1)),
        "policy_n_obs_steps": int(config.get("n_obs_steps", -1)),
        "policy_num_steps": int(config.get("num_steps", -1)),
        "normalization_mapping": config.get("normalization_mapping"),
        "camera_rename_map": rename_map,
        "mismatches": mismatches,
    }


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    dataset_info = _read_json(args.dataset_info)
    result: dict[str, Any] = {
        "status": "not_run",
        "smoke_tasks": PREDECLARED_TASKS,
        "selection_policy": "fixed source-declared subset; no outcome-based selection",
        "checkpoint_revision": args.checkpoint_revision,
        "dataset_revision": args.dataset_revision,
        "dataset_root": str(args.dataset_root),
    }

    if args.checkpoint:
        try:
            result["contract"] = audit_checkpoint_contract(args.dataset_root, args.checkpoint)
        except ContractMismatch as exc:
            result["status"] = "blocked_contract"
            result["contract_error"] = str(exc)
            return result
        result["status"] = "contract_valid_forward_pending"
        result["note"] = "This command validates the checkpoint contract; use the bounded evaluator before full cache generation."
        return result

    config = _read_json(args.config_snapshot)
    preprocessor = _read_json(args.preprocessor_snapshot)
    contract = _snapshot_contract(dataset_info, config, preprocessor)
    result["contract"] = contract
    if contract["mismatches"]:
        result["status"] = "blocked_contract"
        result["contract_error"] = "\n".join(contract["mismatches"])
        result["note"] = "Pinned public config was audited, but no model forward was attempted because the dataset/policy contract is invalid."
    else:
        result["status"] = "contract_valid_checkpoint_missing"
        result["note"] = "No local checkpoint was supplied; no GPU forward was attempted."
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-info", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config-snapshot", type=Path, default=Path(__file__).with_name("smolvla_config_snapshot.json"))
    parser.add_argument("--preprocessor-snapshot", type=Path, default=Path(__file__).with_name("smolvla_preprocessor_snapshot.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_smoke(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if args.progress:
        _persist_preflight_progress(args.progress, result, args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "blocked_contract":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
