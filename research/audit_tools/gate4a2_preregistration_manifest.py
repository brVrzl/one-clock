#!/usr/bin/env python3
"""Hash the frozen Gate-4A2 registration bundle before rollouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "research/audit_outputs/gate4a2_spatial_preregistration_manifest.json"
FILES = (
    "research/gate4a_spatial_asset_audit.md",
    "research/audit_outputs/gate4a_spatial_asset_audit.json",
    "research/gate4a2_spatial_asset_audit.md",
    "research/audit_outputs/gate4a2_spatial_asset_audit.json",
    "research/gate4a2_spatial_state_audit.md",
    "research/gate4a2_spatial_generalization_protocol.md",
    "research/audit_outputs/gate4a2_spatial_schedule.json",
    "configs/gate4a2_libero_spatial.yaml",
    "research/audit_tools/gate3a2_temporal_aggregation.py",
    "research/audit_tools/gate3c_temporal_reuse.py",
    "research/audit_tools/gate4a2_schedule.py",
    "research/audit_tools/gate4a2_rollout.py",
    "research/audit_tools/gate4a2_analyze.py",
    "research/audit_tools/gate4a2_validate_rollouts.py",
    "tests/test_gate4a2_spatial_generalization.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    entries = []
    for relative in FILES:
        path = ROOT / relative
        entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    value = {
        "schema_version": 1,
        "status": "frozen before any official Gate-4A2 outcome",
        "scientific_parent": "36bebdace1ffbd8d36bacc061feb146cd55f894a",
        "checkpoint_repository": "ishandotsh/act_libero_spatial_test",
        "checkpoint_revision": "8f04de1472975d62db214238b2fc07e78bde2474",
        "model_sha256": "912f41808962d80ca9084435aa01eccccdd97b7eae3a841c9f4ac71caaf9f8b0",
        "training_provenance_category": "MULTI-SUITE",
        "schedule_sha256": sha256(
            ROOT / "research/audit_outputs/gate4a2_spatial_schedule.json"
        ),
        "planned_episodes": 500,
        "files": entries,
    }
    OUTPUT.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
