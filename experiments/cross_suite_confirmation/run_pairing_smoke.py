"""Minimal runtime smoke for the two frozen confirmation suites."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from run_confirmation import (
    METHODS,
    build_task_runtime,
    load_protocol,
    make_fresh_env,
    processed_equal,
    query_act_chunk,
    reset_policy_rng,
)
from temporal_reuse import H16, make_executor


ROOT = Path(__file__).resolve().parent


def sim_state(env: Any) -> np.ndarray:
    return np.asarray(env.envs[0]._env.get_sim_state()).copy()


def run_trace(runtime: dict[str, Any], method: str, state_id: int, seed: int, steps: int) -> dict[str, Any]:
    import torch

    env = make_fresh_env(runtime, seed)
    try:
        env.envs[0].init_state_id = int(state_id)
        random.seed(seed)
        np.random.seed(seed)
        reset_policy_rng(torch, 424242)
        runtime["policy"].reset()
        observation, _ = env.reset(seed=[seed])
        initial_pixels = {key: np.asarray(value).copy() for key, value in observation["pixels"].items()}
        initial_state = sim_state(env)
        executor = make_executor(method)
        trace = []
        query_steps = []
        deterministic = False
        for t in range(steps):
            capture: dict[str, Any] = {}

            def query() -> np.ndarray:
                chunk, processed = query_act_chunk(observation, env, runtime)
                capture["chunk"] = chunk.copy()
                capture["processed"] = processed
                return chunk

            before = sim_state(env)
            result = executor.step(t, query)
            if result.queried:
                query_steps.append(t)
            if t == 0:
                repeat_chunk, repeat_processed = query_act_chunk(observation, env, runtime)
                deterministic = np.array_equal(capture["chunk"], repeat_chunk) and processed_equal(capture["processed"], repeat_processed)
                if not deterministic:
                    raise AssertionError("ACT repeated inference was not bit-identical at identical input")
            observation, _, terminated, truncated, _ = env.step(result.action.astype(np.float32)[None])
            after = sim_state(env)
            trace.append({
                "action": result.action.copy(),
                "chunk": capture.get("chunk"),
                "processed": capture.get("processed"),
                "before_sim": before,
                "after_sim": after,
                "queried": bool(result.queried),
                "query_q": result.query_q,
                "arm_q": int(result.arm_source_q),
                "arm_offset": int(result.arm_offset),
                "grip_q": int(result.grip_source_q),
                "grip_offset": int(result.grip_offset),
            })
            if terminated or truncated:
                raise AssertionError(f"{method} smoke terminated before t={steps - 1}")
        return {"initial_pixels": initial_pixels, "initial_state": initial_state, "trace": trace, "query_steps": query_steps, "deterministic": deterministic}
    finally:
        env.close()


def smoke(protocol_path: Path, gpu: str) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    task_keys = ["libero_goal:task4", "libero_10:task0"]
    states = [0, 1, 2]
    results = []
    for task_key in task_keys:
        task = next(task for task in protocol["cohort"]["tasks"] if f"{task['suite']}:task{int(task['task_id'])}" == task_key)
        runtime = build_task_runtime(task, gpu)
        for state_id in states:
            seed = next(int(s) for s, st in zip(task["environment_seeds"], protocol["cohort"]["state_ids"], strict=True) if int(st) == state_id)
            traces = {method: run_trace(runtime, method, state_id, seed, 20) for method in METHODS}
            reference = traces["FRESH"]
            for method in METHODS:
                if not all(np.array_equal(reference["initial_pixels"][key], traces[method]["initial_pixels"][key]) for key in reference["initial_pixels"]):
                    raise AssertionError(f"initial cameras differ for {task_key} state {state_id} method {method}")
            for method in ("FO20", "REVERSE20", "FULL_OLD20"):
                for t in range(20):
                    left, right = reference["trace"][t], traces[method]["trace"][t]
                    if not np.array_equal(left["action"], right["action"]) or not np.array_equal(left["after_sim"], right["after_sim"]):
                        raise AssertionError(f"fixed-source prefix differs from Fresh at {task_key} state={state_id} method={method} t={t}")
                    if not np.array_equal(left["chunk"], right["chunk"]):
                        raise AssertionError(f"fixed-source raw ACT chunk differs at {task_key} state={state_id} method={method} t={t}")
            hard = traces["HARD_H16"]
            if hard["query_steps"] != [0, 16]:
                raise AssertionError(f"hard h16 smoke schedule mismatch: {hard['query_steps']}")
            if not np.array_equal(reference["trace"][0]["chunk"], hard["trace"][0]["chunk"]):
                raise AssertionError(f"hard h16 q=0 chunk differs from Fresh at {task_key} state={state_id}")
            if not all(row["arm_q"] % H16 == 0 for row in hard["trace"]):
                raise AssertionError("hard h16 arm source is not scheduled")
            results.append({"task": task_key, "state_id": state_id, "seed": seed, "fixed_prefix_t0_t19_exact": True, "initial_cameras_exact": True, "hard_query_steps": hard["query_steps"], "act_deterministic": all(traces[m]["deterministic"] for m in METHODS)})
    output = {
        "status": "PASS",
        "protocol": str(protocol_path.resolve()),
        "tasks": task_keys,
        "state_ids": states,
        "methods": list(METHODS),
        "fresh_environment_per_method_state": True,
        "fixed_source_boundary": "all fixed-source methods are required identical to Fresh through t=19; t=20 is the first eligible delayed-source step",
        "hard_h16_boundary": "only schedule and q=0 chunk identity are compared; no post-divergence trajectory equality is required",
        "results": results,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", type=Path, default=ROOT / "pairing_smoke.json")
    args = parser.parse_args()
    result = smoke(args.protocol, args.gpu)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
