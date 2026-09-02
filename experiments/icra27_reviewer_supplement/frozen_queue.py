"""Expand the compact frozen protocol into deterministic per-cell manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def protocol() -> dict[str, Any]:
    return json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))


def _cell(phase: str, policy: str, suite: str, task_id: int, state_id: int,
          seed: int, method: str, checkpoint: str, fps: int,
          max_steps: int | None, size: int) -> dict[str, Any]:
    cid = f"{phase}-{suite}-task{task_id:02d}-state{state_id:02d}-{method}"
    return {"cell_id": cid, "block_id": f"{phase}-{suite}-task{task_id:02d}-state{state_id:02d}",
            "phase": phase, "policy": policy, "suite": suite, "task_id": task_id,
            "state_id": state_id, "environment_seed": seed, "method": method,
            "checkpoint": checkpoint, "fps": fps, "max_episode_steps": max_steps,
            "observation_size": size}


def cells() -> list[dict[str, Any]]:
    p = protocol()
    out: list[dict[str, Any]] = []
    obj = p["r1_object"]
    for phase, methods in (("r1a", obj["conditions_r1a_new"]),
                           ("r1b", obj["conditions_r1b_new"])):
        for task in obj["task_ids"]:
            for method in methods:
                for state in obj["state_ids"]:
                    out.append(_cell(phase, "ACT", obj["suite"], task, state,
                        330000 + 100 * task + state, method, obj["checkpoint"],
                        obj["fps"], obj["max_episode_steps"], obj["observation_size"]))
    r1c = p["r1c"]
    for task in r1c["tasks"]:
        for method in r1c["new_conditions"]:
            for state in r1c["state_ids"]:
                seed = 340000 + 1000 * r1c["suite_index"][task["suite"]] + 100 * task["task_id"] + state
                out.append(_cell("r1c", "ACT", task["suite"], task["task_id"], state,
                    seed, method, task["checkpoint"], r1c["fps"],
                    task["max_episode_steps"], r1c["observation_size"]))
    r1d = p["r1d"]
    for task in r1d["task_ids"]:
        for method in r1d["condition_order"]:
            for state in r1d["state_ids"]:
                out.append(_cell("r1d", "ACT", r1d["suite"], task, state,
                    340000 + 100 * task + state, method, r1d["checkpoint"],
                    r1d["fps"], r1d["max_episode_steps"], 256))
    r2 = p["r2"]
    for task in r2["tasks"]:
        for method in r2["condition_order"]:
            for state, seed in zip(r2["state_ids"], r2["environment_seeds"], strict=True):
                out.append(_cell("r2", "SmolVLA", task["suite"], task["task_id"],
                    state, seed, method, r2["checkpoint"], r2["fps"], None,
                    r2["observation_size"]))
    return out


def phase_cells(phase: str) -> list[dict[str, Any]]:
    return [cell for cell in cells() if cell["phase"] == phase]


def result_path(cell: dict[str, Any]) -> Path:
    return ROOT / "results" / cell["phase"] / f"{cell['cell_id']}.json"


def marker_path(cell: dict[str, Any], status: str = "complete") -> Path:
    return ROOT / "markers" / cell["phase"] / f"{cell['cell_id']}.{status}"


def attempt_path(cell: dict[str, Any]) -> Path:
    return ROOT / "attempts" / cell["phase"] / f"{cell['cell_id']}.json"


def write_manifests() -> None:
    p = protocol()
    for phase in ("r1a", "r1b", "r1c", "r1d", "r2"):
        rows = phase_cells(phase)
        value = {"schema_version": 1, "status": p["status"], "phase": phase,
                 "task_major_static_sharding": "task order modulo 3",
                 "condition_order_is_frozen": True, "cell_count": len(rows), "cells": rows}
        path = ROOT / "manifests" / f"{phase}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_manifests()
