#!/usr/bin/env python3
"""Verify that dense diagnostic queries do not change H16 execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_track_b import ROOT, Runtime, atomic_json


def compare(cell: dict, runtime: Runtime) -> dict:
    ordinary_meta, ordinary = runtime.run_episode(cell, dense_logging=False)
    logged_meta, logged = runtime.run_episode(cell, dense_logging=True)
    state_difference = np.abs(ordinary["simulator_states"] - logged["simulator_states"])
    maximum_state_difference = float(state_difference.max(initial=0.0))
    checks = {
        "executed_actions": bool(np.array_equal(ordinary["executed_actions"], logged["executed_actions"])),
        # SAPIEN may differ at roundoff level when two separately constructed
        # environments execute bit-identical actions.  The absolute 1e-12
        # threshold is fixed technical precision, not outcome tolerance.
        "simulator_states": bool(np.allclose(ordinary["simulator_states"], logged["simulator_states"], rtol=0.0, atol=1e-12)),
        "terminal_success": ordinary_meta["success"] == logged_meta["success"],
        "episode_length": ordinary_meta["environment_steps"] == logged_meta["environment_steps"],
        "execution_query_steps": ordinary_meta["execution_query_steps"] == logged_meta["execution_query_steps"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"dense-logging canary failed for {cell['cell_id']}: {checks}")
    return {
        "cell_id": cell["cell_id"],
        "policy": cell["policy"],
        "checks": checks,
        "success": logged_meta["success"],
        "episode_length": logged_meta["environment_steps"],
        "maximum_simulator_state_absolute_difference": maximum_state_difference,
        "simulator_state_absolute_tolerance": 1e-12,
        "ordinary_wall_clock_seconds": ordinary_meta["wall_clock_seconds"],
        "logged_wall_clock_seconds": logged_meta["wall_clock_seconds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "track_b_manifest.json")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    canary_ids = set(manifest["canary_cell_ids"])
    cells = [cell for cell in manifest["cells"] if cell["cell_id"] in canary_ids]
    if {cell["policy"] for cell in cells} != {"ACT", "SmolVLA"}:
        raise RuntimeError("canary must include ACT and SmolVLA")
    runtime = Runtime(args.gpu)
    try:
        comparisons = [compare(cell, runtime) for cell in cells]
    finally:
        runtime.drop_policy()
    result = {"status": "PASS", "comparisons": comparisons}
    atomic_json(ROOT / "track_b" / "canary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
