"""Runtime pairing smoke for fixed-source methods and hard h16."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from run_factorial import (
    EXPERIMENT_ROOT,
    construct_env,
    freeze_value,
    load_protocol,
    load_task_runtime,
    query_act_chunk,
    reset_policy_rng,
    sim_state_snapshot,
    values_equal,
)
from temporal_reuse import METHODS, make_executor


def smoke(protocol_path: Path, gpu: str) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    smoke_spec = protocol["pairing_smoke"]
    task_id = int(smoke_spec["task_id"])
    state_ids = [int(x) for x in smoke_spec["state_ids"]]
    seeds = [330000 + 100 * task_id + state_id for state_id in state_ids]
    runtime = load_task_runtime(protocol, task_id, gpu)
    torch = runtime["torch"]
    references: dict[int, dict[str, Any]] = {}
    method_checks: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}

    for method in METHODS:
        for state_id, environment_seed in zip(state_ids, seeds, strict=True):
            env = construct_env(runtime, environment_seed)
            try:
                env.envs[0].init_state_id = state_id
                random.seed(environment_seed)
                np.random.seed(environment_seed)
                reset_policy_rng(torch, runtime["policy_rng_seed"])
                runtime["policy"].reset()
                observation, _ = env.reset(seed=[environment_seed])
                initial_pixels = freeze_value(observation["pixels"])
                initial_state = sim_state_snapshot(env)
                executor = make_executor(method)
                captured = {"chunk": None, "processed": None}
                snapshots: list[dict[str, Any]] = []

                for target_t in range(20):
                    captured["chunk"] = None
                    captured["processed"] = None

                    def query() -> np.ndarray:
                        chunk, processed = query_act_chunk(observation, env, runtime)
                        captured["chunk"] = chunk.copy()
                        captured["processed"] = freeze_value(processed)
                        return chunk

                    before = sim_state_snapshot(env)
                    result = executor.step(target_t, query)
                    if result.queried and method == "FRESH" and target_t == 0:
                        repeat_chunk, _ = query_act_chunk(observation, env, runtime)
                        if not np.array_equal(captured["chunk"], repeat_chunk):
                            raise AssertionError("ACT repeated inference was not deterministic at identical input")
                    if captured["chunk"] is not None:
                        captured["chunk"] = np.asarray(captured["chunk"]).copy()
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
                            "queried": result.queried,
                            "query_q": result.query_q,
                            "arm_q": result.arm_source_q,
                            "arm_offset": result.arm_offset,
                            "grip_q": result.grip_source_q,
                            "grip_offset": result.grip_offset,
                        }
                    )
                    if terminated or truncated:
                        raise AssertionError("pairing smoke episode terminated before t=20")

                if method == "FRESH":
                    references[state_id] = {
                        "initial_pixels": initial_pixels,
                        "initial_state": initial_state,
                        "snapshots": snapshots,
                    }
                    continue

                reference = references[state_id]
                if not values_equal(initial_pixels, reference["initial_pixels"]):
                    raise AssertionError(f"initial camera observations differ for {method}, state {state_id}")
                if not all(np.array_equal(a, b) for a, b in zip(initial_state, reference["initial_state"], strict=True)):
                    raise AssertionError(f"initial simulator state differs for {method}, state {state_id}")

                if method in ("FO20", "REVERSE20", "FULL_OLD20"):
                    for target_t, (observed, expected) in enumerate(zip(snapshots, reference["snapshots"], strict=True)):
                        if not values_equal(observed["pixels"], expected["pixels"]):
                            raise AssertionError(f"camera trajectory differs for {method}, state {state_id}, t={target_t}")
                        if not all(np.array_equal(a, b) for a, b in zip(observed["before_sim"], expected["before_sim"], strict=True)):
                            raise AssertionError(f"pre-step simulator state differs for {method}, state {state_id}, t={target_t}")
                        if not all(np.array_equal(a, b) for a, b in zip(observed["after_sim"], expected["after_sim"], strict=True)):
                            raise AssertionError(f"post-step simulator state differs for {method}, state {state_id}, t={target_t}")
                        if not np.array_equal(observed["action"], expected["action"]):
                            raise AssertionError(f"executed action differs before t=20 for {method}, state {state_id}, t={target_t}")
                        if not np.array_equal(observed["chunk"], expected["chunk"]):
                            raise AssertionError(f"raw ACT chunk differs before t=20 for {method}, state {state_id}, t={target_t}")
                        if not values_equal(observed["processed"], expected["processed"]):
                            raise AssertionError(f"processed ACT input differs before t=20 for {method}, state {state_id}, t={target_t}")
                    fixed_check = {
                        "state_id": state_id,
                        "initial_camera_equal": True,
                        "processed_inputs_equal_t0_to_t19": True,
                        "raw_chunks_equal_t0_to_t19": True,
                        "executed_actions_equal_t0_to_t19": True,
                        "simulator_trajectory_equal_t0_to_t20": True,
                    }
                else:
                    observed = snapshots[0]
                    expected = reference["snapshots"][0]
                    for field in ("action", "before_sim", "after_sim", "chunk"):
                        if field == "chunk":
                            equal = np.array_equal(observed[field], expected[field])
                        elif field in ("before_sim", "after_sim"):
                            equal = all(np.array_equal(a, b) for a, b in zip(observed[field], expected[field], strict=True))
                        else:
                            equal = np.array_equal(observed[field], expected[field])
                        if not equal:
                            raise AssertionError(f"hard h16 differs from Fresh at t=0 in {field}, state {state_id}")
                    if not values_equal(observed["processed"], expected["processed"]):
                        raise AssertionError(f"hard h16 processed input differs from Fresh at t=0, state {state_id}")
                    if snapshots[1]["queried"] or snapshots[1]["query_q"] is not None:
                        raise AssertionError("hard h16 queried at t=1")
                    if snapshots[1]["arm_q"] != 0 or snapshots[1]["arm_offset"] != 1:
                        raise AssertionError("hard h16 did not first reuse q=0 at offset 1")
                    if snapshots[1]["grip_q"] != 0 or snapshots[1]["grip_offset"] != 1:
                        raise AssertionError("hard h16 gripper source was not q=0 at offset 1")
                    fixed_check = {
                        "state_id": state_id,
                        "initial_camera_equal": True,
                        "exact_common_prefix_through_t0": True,
                        "first_legitimate_difference_t": 1,
                        "first_difference_cause": "sparse query/source schedule",
                        "hard_query_steps": [0, 16],
                    }
                method_checks[method].append(fixed_check)
            finally:
                env.close()

    return {
        "status": "PASS",
        "protocol": str(protocol_path.resolve()),
        "task_id": task_id,
        "state_ids": state_ids,
        "environment_seeds": seeds,
        "methods": list(METHODS),
        "fresh_environment_per_condition_state": True,
        "deterministic_repeated_inference_t0": True,
        "fixed_method_checks": method_checks,
        "hard_h16_semantics": "compared with Fresh only through the last identical step t=0; t=1 is the first legitimate sparse-source difference",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=EXPERIMENT_ROOT / "protocol.json")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", type=Path, default=EXPERIMENT_ROOT / "pairing_smoke.json")
    args = parser.parse_args()
    result = smoke(args.protocol, args.gpu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
