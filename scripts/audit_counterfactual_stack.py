#!/usr/bin/env python3
"""Audit installed stacks and exact SAPIEN physics snapshot restoration."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packages = ["torch", "sapien", "gymnasium", "mani_skill", "mujoco", "robosuite", "libero", "lerobot"]
    result = {
        "python": sys.executable,
        "packages": {name: bool(importlib.util.find_spec(name)) for name in packages},
        "exact_state_restore": "not_run",
    }
    try:
        import numpy as np
        import sapien.core as sapien

        scene = sapien.Scene()
        scene.set_timestep(1 / 250)
        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.02, 0.02, 0.02])
        actor = builder.build(name="audit_box")
        actor.set_pose(sapien.Pose(p=[0.0, 0.0, 0.25]))
        scene.step()
        physics = scene.get_physx_system()
        snapshot = physics.pack()
        before = np.asarray(actor.get_pose().p).copy()
        actor.set_pose(sapien.Pose(p=[0.4, -0.2, 0.8]))
        physics.unpack(snapshot)
        after = np.asarray(actor.get_pose().p).copy()
        result.update({
            "snapshot_bytes": len(snapshot),
            "pose_before": before.tolist(),
            "pose_after_restore": after.tolist(),
            "exact_state_restore": bool(np.allclose(before, after, atol=1e-7)),
        })
    except Exception as exc:
        result.update({"exact_state_restore": "failed", "error": f"{type(exc).__name__}: {exc}"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["exact_state_restore"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
