"""Runtime smoke for C1/hard common-prefix pairing and C2's causal boundary."""

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
OLD_ROOT = REPO_ROOT / "experiments" / "group_delay_factorial_act20"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OLD_ROOT))

from asymmetric_executor import C1, C2, H16, make_executor  # noqa: E402
from run_factorial import (  # noqa: E402
    construct_env,
    load_task_runtime,
    query_act_chunk,
    reset_policy_rng,
    sim_state_snapshot,
    values_equal,
    freeze_value,
)
from run_factorial import load_protocol as load_factorial_protocol  # noqa: E402
from temporal_reuse import make_executor as make_historical_executor  # noqa: E402


def run_trace(runtime: dict[str, Any], method: str, state_id: int, seed: int, steps: int) -> dict[str, Any]:
    import torch

    env = construct_env(runtime, seed)
    try:
        env.envs[0].init_state_id = int(state_id)
        random.seed(seed)
        np.random.seed(seed)
        reset_policy_rng(torch, runtime["policy_rng_seed"])
        runtime["policy"].reset()
        observation, _ = env.reset(seed=[seed])
        initial_pixels = freeze_value(observation["pixels"])
        initial_state = sim_state_snapshot(env)
        executor = (
            make_executor(method)
            if method in (C1, C2)
            else make_historical_executor("HARD_H16")
        )
        snapshots = []
        deterministic_t0 = False
        for target_t in range(steps):
            captured: dict[str, Any] = {"chunk": None, "processed": None}

            def query() -> np.ndarray:
                chunk, processed = query_act_chunk(observation, env, runtime)
                captured["chunk"] = chunk.copy()
                captured["processed"] = freeze_value(processed)
                return chunk

            before = sim_state_snapshot(env)
            result = executor.step(target_t, query)
            if method == "HARD_H16":
                # The inherited executor predates the C1/C2 smoke fields. Its
                # real queried/query_q pair is the scheduled h16 query event.
                scheduled_query_q = result.query_q if result.queried else None
                fresh_query_q = None
            else:
                scheduled_query_q = result.scheduled_query_q
                fresh_query_q = result.fresh_query_q
            if target_t == 0:
                if captured["chunk"] is None or captured["processed"] is None:
                    raise AssertionError("t=0 did not capture an ACT inference")
                repeat_chunk, repeat_processed = query_act_chunk(observation, env, runtime)
                if not np.array_equal(captured["chunk"], repeat_chunk) or not values_equal(captured["processed"], freeze_value(repeat_processed)):
                    raise AssertionError("ACT repeated inference was not bit-identical under identical processed input")
                deterministic_t0 = True
            observation, _, terminated, truncated, _ = env.step(result.action.astype(np.float32)[None])
            after = sim_state_snapshot(env)
            snapshots.append(
                {
                    "pixels": freeze_value(observation["pixels"]),
                    "before_sim": before,
                    "after_sim": after,
                    "action": result.action.copy(),
                    "chunk": captured["chunk"],
                    "processed": captured["processed"],
                    "policy_queried": bool(result.queried),
                    "query_q": result.query_q,
                    "scheduled_query_q": scheduled_query_q,
                    "fresh_query_q": fresh_query_q,
                    "arm_q": result.arm_source_q,
                    "arm_offset": result.arm_offset,
                    "grip_q": result.grip_source_q,
                    "grip_offset": result.grip_offset,
                }
            )
            if terminated or truncated:
                raise AssertionError(f"smoke episode terminated before t={steps - 1}")
        return {
            "initial_pixels": initial_pixels,
            "initial_state": initial_state,
            "snapshots": snapshots,
            "deterministic_t0": deterministic_t0,
        }
    finally:
        env.close()


def arrays_equal(left: Any, right: Any) -> bool:
    if isinstance(left, tuple) and isinstance(right, tuple):
        return all(arrays_equal(a, b) for a, b in zip(left, right, strict=True))
    return np.array_equal(left, right)


def smoke(protocol_path: Path, gpu: str) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    # The old loader is used only for its runtime-independent schema guard and
    # the shared runtime constructor; this smoke deliberately runs no old method.
    load_factorial_protocol(REPO_ROOT / "experiments/group_delay_factorial_act20/protocol.json")
    spec = protocol["pairing_smoke"]
    task_id = int(spec["task_id"])
    state_ids = [int(x) for x in spec["state_ids"]]
    seeds = [330000 + 100 * task_id + state_id for state_id in state_ids]
    runtime = load_task_runtime(protocol, task_id, gpu)
    c1_checks = []
    c2_checks = []

    for state_id, seed in zip(state_ids, seeds, strict=True):
        c1 = run_trace(runtime, C1, state_id, seed, 17)
        hard = run_trace(runtime, "HARD_H16", state_id, seed, 17)
        if not values_equal(c1["initial_pixels"], hard["initial_pixels"]):
            raise AssertionError(f"C1/hard initial cameras differ, state {state_id}")
        if not arrays_equal(c1["initial_state"], hard["initial_state"]):
            raise AssertionError(f"C1/hard initial simulator states differ, state {state_id}")
        for target_t in range(16):
            observed, expected = c1["snapshots"][target_t], hard["snapshots"][target_t]
            for field in ("pixels", "processed"):
                if not values_equal(observed[field], expected[field]):
                    raise AssertionError(f"C1/hard {field} differs at state={state_id}, t={target_t}")
            for field in ("before_sim", "after_sim"):
                if not arrays_equal(observed[field], expected[field]):
                    raise AssertionError(f"C1/hard {field} differs at state={state_id}, t={target_t}")
            for field in ("action", "chunk"):
                if not np.array_equal(observed[field], expected[field]):
                    raise AssertionError(f"C1/hard {field} differs at state={state_id}, t={target_t}")
        if not arrays_equal(c1["snapshots"][15]["after_sim"], hard["snapshots"][15]["after_sim"]):
            raise AssertionError(f"C1/hard simulator state is not identical through physical t=16, state={state_id}")
        if not c1["deterministic_t0"] or not hard["deterministic_t0"]:
            raise AssertionError("C1/hard ACT determinism check failed")
        c1_checks.append({
            "state_id": state_id,
            "exact_common_prefix_t0_to_t15": True,
            "fields": ["initial_camera_observations", "processed_scheduled_query_inputs", "scheduled_raw_chunks", "executed_actions", "simulator_states"],
            "simulator_state_identical_through_t16": True,
            "boundary_t16": "C1 may diverge from hard from t=16 onward because the gripper source switches to the previous chunk; no step-wise comparison is made from t=16 onward",
        })

        c2 = run_trace(runtime, C2, state_id, seed, 34)
        if not values_equal(c2["initial_pixels"], hard["initial_pixels"]):
            raise AssertionError(f"C2/hard initial cameras differ, state {state_id}")
        if not np.array_equal(c2["snapshots"][0]["chunk"], hard["snapshots"][0]["chunk"]):
            raise AssertionError(f"C2 scheduled q=0 chunk differs from hard, state {state_id}")
        if not np.array_equal(c2["snapshots"][0]["action"], hard["snapshots"][0]["action"]):
            raise AssertionError(f"C2 t=0 action differs from hard, state {state_id}")
        first_dense = c2["snapshots"][1]
        if first_dense["scheduled_query_q"] is not None or first_dense["fresh_query_q"] != 1:
            raise AssertionError("C2 dense query boundary was not recorded at t=1")
        if (first_dense["arm_q"], first_dense["arm_offset"]) != (0, 1):
            raise AssertionError("C2 arm did not reuse scheduled q=0 at t=1")
        if (first_dense["grip_q"], first_dense["grip_offset"]) != (1, 0):
            raise AssertionError("C2 gripper did not use fresh q=1 at t=1")
        c2_arm_sources = {int(row["arm_q"]) for row in c2["snapshots"]}
        if not all(source_q % H16 == 0 for source_q in c2_arm_sources):
            raise AssertionError("C2 used a non-h16 arm source")
        if len(c2_arm_sources) != 3:
            raise AssertionError("C2 did not use ceil(34/16) scheduled arm chunks")
        if not c2["deterministic_t0"]:
            raise AssertionError("C2 ACT determinism check failed")
        c2_checks.append({
            "state_id": state_id,
            "exact_common_prefix_with_hard_t0": True,
            "first_legitimate_difference_boundary": "t=1 may differ because C2 refreshes gripper; arm source remains q=0 offset1",
            "c2_deterministic_t0": True,
            "c2_distinct_arm_source_chunks": len(c2_arm_sources),
        })

    return {
        "status": "PASS",
        "protocol": str(protocol_path.resolve()),
        "task_id": task_id,
        "state_ids": state_ids,
        "environment_seeds": seeds,
        "fresh_environment_per_condition_state": True,
        "deterministic_repeated_inference": "bit-identical ACT chunks and processed inputs were asserted at identical t=0 inputs for C1, C2, and hard h16",
        "c1_vs_hard": c1_checks,
        "c2_vs_hard": c2_checks,
        "causal_boundary": "C1/hard are required identical through t=15; C1 may change gripper at t=16. C2 may diverge at t=1, while its arm source remains the scheduled h16 chunk.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", type=Path, default=ROOT / "pairing_smoke.json")
    args = parser.parse_args()
    result = smoke(args.protocol, args.gpu)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
