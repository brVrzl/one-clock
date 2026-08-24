#!/usr/bin/env python3
"""Write the compact content-addressed Gate-3B preregistration manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STARTING_SHA = "eb4f6bfeb40a9d1444d3fb1d17c841601ca29a76"
SCHEDULE = ROOT / "research/audit_outputs/gate3b_run_schedule.json"
OUTPUT = ROOT / "research/audit_outputs/gate3b_preregistration_manifest.json"
CHECKPOINT = Path("/home/thor/projects/checkpoints/zeromidnight_act_libero_object")
LEROBOT_ROOT = Path("/home/thor/projects/embodied_lab/third_party/lerobot")
FILES = (
    ROOT / "research/gate3b_cross_generation_preregistered_protocol.md",
    SCHEDULE,
    ROOT / "research/audit_tools/gate3b_composition.py",
    ROOT / "research/audit_tools/gate3b_schedule.py",
    ROOT / "research/audit_tools/gate3b_rollout.py",
    ROOT / "research/audit_tools/gate3b_analyze.py",
    ROOT / "research/audit_tools/gate3b_validate_rollouts.py",
    ROOT / "tests/test_gate3b_composition.py",
    ROOT / "configs/gate0_libero_object.yaml",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    value = {
        "schema_version": 1,
        "status": "frozen before official Gate-3B success outcomes",
        "starting_scientific_commit": STARTING_SHA,
        "working_tree_parent_at_generation": git_commit(ROOT),
        "branch": "exp/gate3b-cross-generation-composition",
        "checkpoint": {
            "directory": str(CHECKPOINT),
            "model_sha256": sha256(CHECKPOINT / "model.safetensors"),
            "config_sha256": sha256(CHECKPOINT / "config.json"),
        },
        "lerobot_commit": git_commit(LEROBOT_ROOT),
        "source_age_ticks": 20,
        "source_age_seconds": 1.0,
        "state_selection_seed": 20260827,
        "method_order_seed": 20260828,
        "paired_bootstrap_seed": 20260829,
        "task_cluster_bootstrap_seed": 20261829,
        "selected_state_ids": schedule["state_selection"]["selected_state_ids"],
        "planned_blocks": schedule["planned_blocks"],
        "planned_episodes": schedule["planned_episodes"],
        "files": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in FILES
        ],
    }
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
