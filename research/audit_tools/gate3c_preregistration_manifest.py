#!/usr/bin/env python3
"""Hash the frozen Gate-3C registration bundle before rollouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "research/audit_outputs/gate3c_preregistration_manifest.json"
FILES = (
    "research/gate3c_state_usage_audit.md",
    "research/gate3c_asymmetric_temporal_reuse_protocol.md",
    "research/audit_outputs/gate3c_run_schedule.json",
    "research/audit_tools/gate3c_temporal_reuse.py",
    "research/audit_tools/gate3c_schedule.py",
    "research/audit_tools/gate3c_rollout.py",
    "research/audit_tools/gate3c_analyze.py",
    "research/audit_tools/gate3c_validate_rollouts.py",
    "tests/test_gate3c_temporal_reuse.py",
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
        "status": "frozen before any official Gate-3C outcome",
        "scientific_parent": "2817411a4210b8611dc8dae5d32ec99fc6b94cf3",
        "schedule_sha256": sha256(ROOT / "research/audit_outputs/gate3c_run_schedule.json"),
        "planned_episodes": 700,
        "files": entries,
    }
    OUTPUT.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
