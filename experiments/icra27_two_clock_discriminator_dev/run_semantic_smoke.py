"""One exposed-state runtime smoke for fixed-clock prefix and boundary semantics."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
ASYM_ROOT = REPO_ROOT / "experiments" / "asymmetric_chunk_reuse_dev"
FACTORIAL_ROOT = REPO_ROOT / "experiments" / "group_delay_factorial_act20"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ASYM_ROOT))
sys.path.insert(0, str(FACTORIAL_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from asymmetric_executor import C1, make_executor as make_c1_executor  # noqa: E402
from fixed_clock_executor import CHUNK_LENGTH, H32_COHERENT, TWO_CLOCK, make_executor  # noqa: E402
from run_factorial import (  # noqa: E402
    construct_env,
    freeze_value,
    load_task_runtime,
    query_act_chunk,
    reset_policy_rng,
    sim_state_snapshot,
    values_equal,
)
from temporal_reuse import make_executor as make_historical_executor  # noqa: E402


H16_COHERENT = "H16_COHERENT"


def arrays_equal(left: Any, right: Any) -> bool:
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(arrays_equal(a, b) for a, b in zip(left, right, strict=True))
    return np.array_equal(left, right)


def run_trace(runtime: dict[str, Any], method: str, state_id: int, seed: int, steps: int) -> dict[str, Any]:
    import torch

    env = construct_env(runtime, seed)
    try:
        env.envs[0].init_state_id = state_id
        random.seed(seed)
        np.random.seed(seed)
        reset_policy_rng(torch, runtime["policy_rng_seed"])
        runtime["policy"].reset()
        observation, _ = env.reset(seed=[seed])
        initial_pixels = freeze_value(observation["pixels"])
        initial_state = sim_state_snapshot(env)
        if method in (H32_COHERENT, TWO_CLOCK):
            executor = make_executor(method)
            new_executor = True
        elif method == C1:
            executor = make_c1_executor(C1)
            new_executor = False
        elif method == H16_COHERENT:
            executor = make_historical_executor("HARD_H16")
            new_executor = False
        else:
            raise ValueError(method)
        query_steps: list[int] = []
        rows = []
        for t in range(steps):
            captured: dict[str, Any] = {"chunk": None, "processed": None}

            def query() -> np.ndarray:
                chunk, processed = query_act_chunk(observation, env, runtime)
                captured["chunk"] = chunk.copy()
                captured["processed"] = freeze_value(processed)
                return chunk

            before = sim_state_snapshot(env)
            if new_executor:
                decision = executor.step(query)
                if decision.policy_query:
                    query_steps.append(t)
                source_q = {group: query_steps[chunk_id] for group, chunk_id in decision.source_chunk_ids.items()}
                arm_q = source_q["arm"]
                grip_q = source_q["gripper"]
                arm_offset = int(decision.source_positions["arm"])
                grip_offset = int(decision.source_positions["gripper"])
                action = decision.action
                queried = decision.policy_query
                refreshed_groups = list(decision.refreshed_groups)
            else:
                result = executor.step(t, query)
                if result.queried:
                    query_steps.append(t)
                arm_q, grip_q = int(result.arm_source_q), int(result.grip_source_q)
                arm_offset, grip_offset = int(result.arm_offset), int(result.grip_offset)
                action = result.action
                queried = result.queried
                refreshed_groups = None
            if arm_q + arm_offset != t or grip_q + grip_offset != t:
                raise AssertionError(f"{method} violated q+k=t at t={t}")
            observation, _, terminated, truncated, _ = env.step(action.astype(np.float32)[None])
            if bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]):
                raise AssertionError(f"{method} smoke terminated before t={steps - 1}")
            rows.append(
                {
                    "t": t,
                    "pixels": freeze_value(observation["pixels"]),
                    "before_sim": before,
                    "after_sim": sim_state_snapshot(env),
                    "action": action.copy(),
                    "chunk": captured["chunk"],
                    "processed": captured["processed"],
                    "queried": bool(queried),
                    "arm_q": arm_q,
                    "arm_offset": arm_offset,
                    "grip_q": grip_q,
                    "grip_offset": grip_offset,
                    "refreshed_groups": refreshed_groups,
                }
            )
        return {
            "fresh_environment_instance": True,
            "initial_pixels": initial_pixels,
            "initial_state": initial_state,
            "query_steps": query_steps,
            "rows": rows,
        }
    finally:
        env.close()


def assert_exact_prefix(left: dict[str, Any], right: dict[str, Any], through_t: int, label: str) -> None:
    if not values_equal(left["initial_pixels"], right["initial_pixels"]):
        raise AssertionError(f"{label} initial cameras differ")
    if not arrays_equal(left["initial_state"], right["initial_state"]):
        raise AssertionError(f"{label} initial simulator states differ")
    for t in range(through_t + 1):
        a, b = left["rows"][t], right["rows"][t]
        for field in ("pixels", "processed"):
            if not values_equal(a[field], b[field]):
                raise AssertionError(f"{label} {field} differs at t={t}")
        for field in ("before_sim", "after_sim"):
            if not arrays_equal(a[field], b[field]):
                raise AssertionError(f"{label} {field} differs at t={t}")
        for field in ("action", "chunk"):
            if not np.array_equal(a[field], b[field]):
                raise AssertionError(f"{label} {field} differs at t={t}")


def smoke(protocol_path: Path, gpu: str) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    spec = protocol["semantic_smoke"]
    task_id = int(spec["task_id"])
    state_id = int(spec["state_id"])
    seed = int(spec["environment_seed"])
    steps = int(spec["trace_steps"])
    runtime = load_task_runtime(protocol, task_id, gpu)
    if int(runtime["policy"].config.chunk_size) != CHUNK_LENGTH or CHUNK_LENGTH != 100:
        raise AssertionError("checkpoint chunk size is not 100")
    if runtime["policy"].config.temporal_ensemble_coeff is not None:
        raise AssertionError("hidden ACT temporal ensemble is enabled")

    traces = {
        method: run_trace(runtime, method, state_id, seed, steps)
        for method in (H16_COHERENT, H32_COHERENT, TWO_CLOCK, C1)
    }
    if not all(trace["fresh_environment_instance"] for trace in traces.values()):
        raise AssertionError("a smoke trace did not use a fresh environment")
    assert_exact_prefix(traces[H16_COHERENT], traces[H32_COHERENT], 15, "H16/H32")
    assert_exact_prefix(traces[TWO_CLOCK], traces[H16_COHERENT], 15, "TWO_CLOCK/H16")
    assert_exact_prefix(traces[C1], traces[TWO_CLOCK], 31, "C1/TWO_CLOCK")

    for t in range(32):
        c1 = traces[C1]["rows"][t]
        two = traces[TWO_CLOCK]["rows"][t]
        if (c1["arm_q"], c1["arm_offset"], c1["grip_q"], c1["grip_offset"]) != (
            two["arm_q"], two["arm_offset"], two["grip_q"], two["grip_offset"]
        ):
            raise AssertionError(f"C1/TWO_CLOCK source semantics differ before t=32, at t={t}")
    c1_boundary = traces[C1]["rows"][32]
    two_boundary = traces[TWO_CLOCK]["rows"][32]
    if (c1_boundary["grip_q"], c1_boundary["grip_offset"]) != (16, 16):
        raise AssertionError("C1 t=32 gripper boundary is not q=16,k=16")
    if (two_boundary["grip_q"], two_boundary["grip_offset"]) != (32, 0):
        raise AssertionError("TWO_CLOCK t=32 gripper boundary is not q=32,k=0")
    if traces[H32_COHERENT]["query_steps"] != [0, 32]:
        raise AssertionError("H32 smoke query schedule drifted")
    if traces[TWO_CLOCK]["query_steps"] != [0, 16, 32]:
        raise AssertionError("TWO_CLOCK smoke query schedule drifted")

    return {
        "status": "PASS",
        "protocol": str(protocol_path.resolve()),
        "task_id": task_id,
        "state_id": state_id,
        "environment_seed": seed,
        "checkpoint_chunk_size": CHUNK_LENGTH,
        "offset_31_legal": True,
        "fresh_environment_per_method_state": True,
        "policy_rng_reset_per_method": True,
        "temporal_ensemble_disabled": True,
        "action_smoothing_disabled": True,
        "same_target_asserted_every_group_step": True,
        "exact_prefixes": {
            "H16_COHERENT_vs_H32_COHERENT": "t=0..15",
            "TWO_CLOCK_ARM16_GRIP32_vs_H16_COHERENT": "t=0..15",
            "C1_PREVIOUS_CHUNK_GRIP_vs_TWO_CLOCK_ARM16_GRIP32": "t=0..31",
        },
        "query_steps": {method: trace["query_steps"] for method, trace in traces.items()},
        "boundary_t32": {
            "C1_PREVIOUS_CHUNK_GRIP": {"arm_q": c1_boundary["arm_q"], "arm_offset": c1_boundary["arm_offset"], "grip_q": 16, "grip_offset": 16},
            "TWO_CLOCK_ARM16_GRIP32": {"arm_q": two_boundary["arm_q"], "arm_offset": two_boundary["arm_offset"], "grip_q": 32, "grip_offset": 0},
        },
        "note": "The rollout path passes executor actions directly to env.step; no averaging, interpolation, selector, temporal ensemble, or smoothing is present.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", type=Path, default=ROOT / "semantic_smoke.json")
    args = parser.parse_args()
    result = smoke(args.protocol, args.gpu)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
