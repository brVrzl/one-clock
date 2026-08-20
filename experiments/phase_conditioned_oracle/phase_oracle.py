#!/usr/bin/env python3
"""Analysis-only phase-conditioned oracle horizon evaluation on LIBERO Object.

This script deliberately keeps the production executor untouched. It runs an
oracle evaluation plan in this analysis directory: when a commitment expires,
the predeclared horizon for the current normalized rollout phase is applied to
the next frozen ACT chunk. The phase is known from rollout time and is not a
deployable scheduler signal. Constant phase maps reduce to the existing fixed
execution semantics and serve as controls.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.run_libero_gate0 import (  # noqa: E402
    load_config,
    load_policy_and_processors,
    query_full_act_chunk,
    set_episode_seed,
)


PHASES = ("early", "middle", "late")
HORIZONS = (1, 2, 4, 8, 16)
BASE_GLOBAL_HORIZON = 16
BASE_GROUP_HORIZONS = {"arm": 4, "gripper": 16}
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260819
Z95 = 1.959963984540054


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/gate0_libero_object.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/thor/projects/checkpoints/zeromidnight_act_libero_object"),
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=ROOT / "experiments/libero_object_dynamic_readiness.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/phase_conditioned_oracle",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        help="Comma-separated task IDs for a smoke run; default is all ten tasks.",
    )
    parser.add_argument(
        "--state-count",
        type=int,
        help="Use only the first N established states per task for a smoke run.",
    )
    parser.add_argument(
        "--state-start",
        type=int,
        default=0,
        help="Start offset within each task's established state list.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        help="Limit initial phase candidates for a smoke run; no oracle map is produced.",
    )
    parser.add_argument(
        "--configs-json",
        type=Path,
        help="Run an explicit JSON list of oracle configurations instead of the initial grid.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume task/config aggregates from config_results.json.")
    return parser.parse_args()


def parse_task_ids(raw: str | None) -> list[int]:
    if raw is None:
        return list(range(10))
    task_ids = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not task_ids or any(task_id < 0 or task_id > 9 for task_id in task_ids):
        raise ValueError("task IDs must be a non-empty subset of 0..9")
    return task_ids


def phase_for_step(environment_step: int, max_episode_steps: int) -> str:
    progress = environment_step / max_episode_steps
    if progress < 1 / 3:
        return "early"
    if progress < 2 / 3:
        return "middle"
    return "late"


def wilson_interval(successes: int, episodes: int) -> list[float]:
    proportion = successes / episodes
    denominator = 1.0 + Z95**2 / episodes
    center = (proportion + Z95**2 / (2.0 * episodes)) / denominator
    half_width = (
        Z95
        * math.sqrt(
            proportion * (1.0 - proportion) / episodes
            + Z95**2 / (4.0 * episodes**2)
        )
        / denominator
    )
    return [center - half_width, center + half_width]


def phase_map_string(phase_map: dict[str, dict[str, int]]) -> str:
    return ";".join(
        f"{phase}:arm{phase_map[phase]['arm']}:gripper{phase_map[phase]['gripper']}"
        for phase in PHASES
    )


class OraclePlanRunner:
    """Local evaluator for a known phase map; not part of production execution."""

    def __init__(self, phase_map: dict[str, dict[str, int]], strategy: str, chunk_size: int) -> None:
        self.phase_map = phase_map
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.reset()

    def reset(self) -> None:
        self.environment_step = 0
        self.next_chunk_id = 0
        self.global_chunk: np.ndarray | None = None
        self.global_position = 0
        self.global_remaining = 0
        self.global_chunk_id: int | None = None
        self.global_query_step: int | None = None
        self.group_states: dict[str, dict[str, Any]] = {}

    def _query(self, query_policy: Any) -> tuple[np.ndarray, int]:
        chunk = np.asarray(query_policy())
        expected_shape = (self.chunk_size, 7)
        if chunk.shape != expected_shape:
            raise ValueError(f"policy chunk must have shape {expected_shape}, got {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise ValueError("policy chunk must be finite")
        chunk_id = self.next_chunk_id
        self.next_chunk_id += 1
        return chunk.copy(), chunk_id

    def step(self, query_policy: Any, phase: str) -> dict[str, Any]:
        if self.strategy == "global":
            horizon = int(self.phase_map[phase]["arm"])
            queried = self.global_chunk is None or self.global_remaining == 0
            if queried:
                chunk, chunk_id = self._query(query_policy)
                self.global_chunk = chunk
                self.global_position = 0
                self.global_remaining = horizon
                self.global_chunk_id = chunk_id
                self.global_query_step = self.environment_step
            assert self.global_chunk is not None
            assert self.global_chunk_id is not None
            assert self.global_query_step is not None
            action = self.global_chunk[self.global_position].copy()
            source_age = self.environment_step - self.global_query_step
            source_position = self.global_position
            active_horizon = self.global_remaining + self.global_position
            self.global_position += 1
            self.global_remaining -= 1
            source_chunk_ids = {"arm": self.global_chunk_id, "gripper": self.global_chunk_id}
            source_ages = {"arm": source_age, "gripper": source_age}
            source_positions = {"arm": source_position, "gripper": source_position}
            remaining = {"arm": self.global_remaining, "gripper": self.global_remaining}
        else:
            expired = [
                group
                for group in ("arm", "gripper")
                if group not in self.group_states or self.group_states[group]["remaining"] == 0
            ]
            queried = bool(expired)
            new_chunk_id: int | None = None
            if queried:
                chunk, new_chunk_id = self._query(query_policy)
                for group in expired:
                    self.group_states[group] = {
                        "chunk": chunk,
                        "chunk_id": new_chunk_id,
                        "query_step": self.environment_step,
                        "position": 0,
                        "remaining": int(self.phase_map[phase][group]),
                    }
            action = np.empty(7, dtype=np.float32)
            source_chunk_ids = {}
            source_ages = {}
            source_positions = {}
            remaining = {}
            active_horizon = {}
            for group, indices in (("arm", list(range(6))), ("gripper", [6])):
                state = self.group_states[group]
                action[indices] = state["chunk"][state["position"], indices]
                source_chunk_ids[group] = int(state["chunk_id"])
                source_ages[group] = self.environment_step - int(state["query_step"])
                source_positions[group] = int(state["position"])
                remaining[group] = int(state["remaining"])
                active_horizon[group] = int(state["remaining"]) + int(state["position"])
            for group in ("arm", "gripper"):
                self.group_states[group]["position"] += 1
                self.group_states[group]["remaining"] -= 1

        decision = {
            "environment_step": self.environment_step,
            "policy_query": queried,
            "phase": phase,
            "action": action,
            "source_chunk_ids": source_chunk_ids,
            "source_ages": source_ages,
            "source_positions": source_positions,
            "remaining_commitments": remaining,
            "active_horizon": active_horizon,
        }
        self.environment_step += 1
        return decision


def phase_map_for_global(target_phase: str, horizon: int) -> dict[str, dict[str, int]]:
    return {
        phase: {"arm": int(horizon if phase == target_phase else BASE_GLOBAL_HORIZON), "gripper": int(horizon if phase == target_phase else BASE_GLOBAL_HORIZON)}
        for phase in PHASES
    }


def phase_map_for_group(target_phase: str, arm: int, gripper: int) -> dict[str, dict[str, int]]:
    return {
        phase: {
            "arm": int(arm if phase == target_phase else BASE_GROUP_HORIZONS["arm"]),
            "gripper": int(gripper if phase == target_phase else BASE_GROUP_HORIZONS["gripper"]),
        }
        for phase in PHASES
    }


def build_initial_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for phase in PHASES:
        for horizon in HORIZONS:
            configs.append(
                {
                    "name": f"phase_{phase}_global_h{horizon}",
                    "kind": "global_phase_candidate",
                    "strategy": "global",
                    "target_phase": phase,
                    "arm_horizon": horizon,
                    "gripper_horizon": horizon,
                    "phase_map": phase_map_for_global(phase, horizon),
                }
            )
        for arm, gripper in itertools.product(HORIZONS, repeat=2):
            configs.append(
                {
                    "name": f"phase_{phase}_group_arm{arm}_grip{gripper}",
                    "kind": "group_phase_candidate",
                    "strategy": "group",
                    "target_phase": phase,
                    "arm_horizon": arm,
                    "gripper_horizon": gripper,
                    "phase_map": phase_map_for_group(phase, arm, gripper),
                }
            )
    return configs


def run_episode(
    *,
    env: Any,
    policy: Any,
    policy_preprocessor: Any,
    policy_postprocessor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    oracle: OraclePlanRunner,
    init_state_id: int,
    seed: int,
) -> dict[str, Any]:
    set_episode_seed(seed)
    env.init_state_id = init_state_id
    observation, _ = env.reset(seed=seed)
    policy.reset()
    oracle.reset()
    max_steps = int(env._max_episode_steps)
    phase_steps = {phase: 0 for phase in PHASES}
    phase_queries = {phase: 0 for phase in PHASES}
    phase_active_horizons: dict[str, dict[str, list[int]]] = {
        phase: {"arm": [], "gripper": []} for phase in PHASES
    }
    last_info: dict[str, Any] = {"is_success": False}
    for environment_step in range(max_steps):
        phase = phase_for_step(environment_step, max_steps)

        def query() -> np.ndarray:
            return query_full_act_chunk(
                observation=observation,
                policy=policy,
                policy_preprocessor=policy_preprocessor,
                policy_postprocessor=policy_postprocessor,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
            )

        decision = oracle.step(query, phase)
        phase_steps[phase] += 1
        phase_queries[phase] += int(decision["policy_query"])
        if decision["policy_query"]:
            active = decision["active_horizon"]
            if isinstance(active, dict):
                phase_active_horizons[phase]["arm"].append(int(active["arm"]))
                phase_active_horizons[phase]["gripper"].append(int(active["gripper"]))
            else:
                phase_active_horizons[phase]["arm"].append(int(active))
                phase_active_horizons[phase]["gripper"].append(int(active))
        observation, _, terminated, truncated, last_info = env.step(decision["action"].astype(np.float32))
        if terminated or truncated:
            break
    return {
        "init_state_id": int(init_state_id),
        "seed": int(seed),
        "success": bool(last_info["is_success"]),
        "environment_steps": int(sum(phase_steps.values())),
        "policy_queries": int(sum(phase_queries.values())),
        "phase_steps": phase_steps,
        "phase_queries": phase_queries,
        "phase_active_horizons": phase_active_horizons,
        "max_episode_steps": max_steps,
    }


def aggregate_task(task_id: int, task_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = len(records)
    environment_steps = sum(int(record["environment_steps"]) for record in records)
    policy_queries = sum(int(record["policy_queries"]) for record in records)
    phase_summary: dict[str, Any] = {}
    for phase in PHASES:
        steps = sum(int(record["phase_steps"][phase]) for record in records)
        queries = sum(int(record["phase_queries"][phase]) for record in records)
        reached = sum(int(record["phase_steps"][phase]) > 0 for record in records)
        phase_summary[phase] = {
            "episodes_reaching_phase": reached,
            "phase_environment_steps": steps,
            "phase_policy_queries": queries,
            "phase_query_rate": queries / steps if steps else None,
            "mean_phase_steps_per_episode": steps / episodes,
            "mean_phase_queries_per_episode": queries / episodes,
        }
    return {
        "task_id": int(task_id),
        "task_name": task_name,
        "episodes": episodes,
        "successes": sum(bool(record["success"]) for record in records),
        "success_rate": sum(bool(record["success"]) for record in records) / episodes,
        "success_rate_ci95": wilson_interval(sum(bool(record["success"]) for record in records), episodes),
        "success_vector": [bool(record["success"]) for record in records],
        "state_ids": [int(record["init_state_id"]) for record in records],
        "environment_steps": environment_steps,
        "policy_queries": policy_queries,
        "policy_query_rate": policy_queries / environment_steps,
        "mean_environment_steps": environment_steps / episodes,
        "mean_policy_queries": policy_queries / episodes,
        "phase_summary": phase_summary,
    }


def run_config_for_task(
    *,
    config: dict[str, Any],
    task_id: int,
    task_name: str,
    state_ids: list[int],
    seed_base: int,
    env: Any,
    policy: Any,
    policy_preprocessor: Any,
    policy_postprocessor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
) -> dict[str, Any]:
    oracle = OraclePlanRunner(config["phase_map"], config["strategy"], chunk_size=100)
    records = [
        run_episode(
            env=env,
            policy=policy,
            policy_preprocessor=policy_preprocessor,
            policy_postprocessor=policy_postprocessor,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            oracle=oracle,
            init_state_id=state_id,
            seed=seed_base + state_id,
        )
        for state_id in state_ids
    ]
    return aggregate_task(task_id, task_name, records)


def load_static_baseline(path: Path, task_ids: list[int]) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    rows = {int(row["task_id"]): row for row in artifact["tasks"]}
    result = {}
    for task_id in task_ids:
        row = rows[task_id]
        group = row["groupwise"]["(4,16)"]
        global_row = row["global"]["16"]
        result[str(task_id)] = {
            "task_id": task_id,
            "task_name": row["task_name"],
            "state_ids": row["state_ids"],
            "episodes": int(group["episodes"]),
            "group_success_vector": list(group["success_vector"]),
            "group_success_rate": float(group["success_rate"]),
            "group_query_rate": float(group["policy_query_rate"]),
            "global_success_vector": list(global_row["success_vector"]),
            "global_success_rate": float(global_row["success_rate"]),
            "global_query_rate": float(global_row["policy_query_rate"]),
        }
    return result


def task_macro(task_results: dict[str, dict[str, Any]], field: str = "success_rate") -> float:
    return float(np.mean([float(row[field]) for row in task_results.values()]))


def task_bootstrap(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    estimates = values[draws].mean(axis=1)
    return [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]


def summarize_config(name: str, config: dict[str, Any], task_results: dict[str, dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    success_values = np.asarray([float(row["success_rate"]) for row in task_results.values()])
    query_values = np.asarray([float(row["policy_query_rate"]) for row in task_results.values()])
    step_values = np.asarray([float(row["mean_environment_steps"]) for row in task_results.values()])
    phase_macro: dict[str, Any] = {}
    for phase in PHASES:
        phase_steps = np.asarray([float(row["phase_summary"][phase]["mean_phase_steps_per_episode"]) for row in task_results.values()])
        phase_queries = np.asarray([float(row["phase_summary"][phase]["mean_phase_queries_per_episode"]) for row in task_results.values()])
        phase_rates = np.asarray(
            [
                float(row["phase_summary"][phase]["phase_query_rate"])
                for row in task_results.values()
                if row["phase_summary"][phase]["phase_query_rate"] is not None
            ]
        )
        phase_macro[phase] = {
            "episodes_reaching_phase": int(sum(row["phase_summary"][phase]["episodes_reaching_phase"] for row in task_results.values())),
            "mean_phase_steps_per_episode": float(phase_steps.mean()),
            "mean_phase_queries_per_episode": float(phase_queries.mean()),
            "macro_phase_query_rate": float(phase_rates.mean()) if len(phase_rates) else None,
        }
    return {
        "name": name,
        "kind": config.get("kind", "combined"),
        "strategy": config["strategy"],
        "target_phase": config.get("target_phase"),
        "arm_horizon": config.get("arm_horizon"),
        "gripper_horizon": config.get("gripper_horizon"),
        "phase_map": config["phase_map"],
        "task_results": task_results,
        "episodes": int(sum(int(row["episodes"]) for row in task_results.values())),
        "success_rate_pooled": float(sum(int(row["successes"]) for row in task_results.values()) / sum(int(row["episodes"]) for row in task_results.values())),
        "macro_success_rate": float(success_values.mean()),
        "macro_success_rate_bootstrap_ci95": task_bootstrap(success_values, rng),
        "macro_query_rate": float(query_values.mean()),
        "pooled_query_rate": float(
            sum(int(row["policy_queries"]) for row in task_results.values())
            / sum(int(row["environment_steps"]) for row in task_results.values())
        ),
        "macro_mean_environment_steps": float(step_values.mean()),
        "phase_macro": phase_macro,
    }


def load_cache(path: Path, resume: bool) -> dict[str, dict[str, dict[str, Any]]]:
    if not resume or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_cache(path: Path, cache: dict[str, dict[str, dict[str, Any]]]) -> None:
    path.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")


def run_configs(
    *,
    configs: list[dict[str, Any]],
    task_ids: list[int],
    task_rows: dict[int, dict[str, Any]],
    config_cache: dict[str, dict[str, dict[str, Any]]],
    cache_path: Path,
    args: argparse.Namespace,
) -> None:
    for task_id in task_ids:
        row = task_rows[task_id]
        config = load_config(args.config)
        runtime_config = dict(config)
        runtime_config["task_id"] = task_id
        runtime_config["task_name"] = row["task_name"]
        from libero.libero import benchmark
        from lerobot.envs.libero import LiberoEnv

        suite = benchmark.get_benchmark_dict()[str(config["task_suite"])]()
        task = suite.get_task(task_id)
        if task.name != row["task_name"]:
            raise ValueError(f"task metadata mismatch for task {task_id}: {task.name} != {row['task_name']}")
        policy, policy_preprocessor, policy_postprocessor, env_preprocessor, env_postprocessor = load_policy_and_processors(
            runtime_config, args.checkpoint
        )
        env = LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=str(config["task_suite"]),
            obs_type=str(config["obs_type"]),
            camera_name=str(config["camera_name"]),
            camera_name_mapping=dict(config["camera_name_mapping"]),
            observation_width=int(config["observation_width"]),
            observation_height=int(config["observation_height"]),
            control_freq=int(config.get("control_freq", 20)),
            init_states=bool(config["init_states"]),
            hard_reset=bool(config["hard_reset"]),
            control_mode=str(config["control_mode"]),
        )
        official_count = len(env._init_states)
        if args.state_start < 0:
            raise ValueError("state-start must be non-negative")
        state_ids = list(row["state_ids"])[args.state_start :]
        if args.state_count is not None:
            if args.state_count < 1:
                raise ValueError("state-count must be positive")
            state_ids = state_ids[: args.state_count]
        if not state_ids:
            raise ValueError(f"task {task_id} has no states after requested slice")
        if state_ids[-1] >= official_count:
            raise ValueError(f"task {task_id} state exceeds official count {official_count}")
        try:
            for config_index, candidate in enumerate(configs, start=1):
                if candidate["name"] in config_cache and str(task_id) in config_cache[candidate["name"]]:
                    continue
                task_result = run_config_for_task(
                    config=candidate,
                    task_id=task_id,
                    task_name=task.name,
                    state_ids=state_ids,
                    seed_base=int(config["seed"]),
                    env=env,
                    policy=policy,
                    policy_preprocessor=policy_preprocessor,
                    policy_postprocessor=policy_postprocessor,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                )
                config_cache.setdefault(candidate["name"], {})[str(task_id)] = task_result
                save_cache(cache_path, config_cache)
                print(
                    f"task {task_id} config {config_index}/{len(configs)} "
                    f"{candidate['name']} success={task_result['success_rate']:.3f}",
                    flush=True,
                )
        finally:
            env.close()


def choose_best_phase_configs(
    *,
    configs: list[dict[str, Any]],
    cache: dict[str, dict[str, dict[str, Any]]],
    task_ids: list[int],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summarized: dict[str, dict[str, Any]] = {}
    for config in configs:
        task_results = cache[config["name"]]
        summarized[config["name"]] = summarize_config(config["name"], config, task_results, rng)
    selected_global: dict[str, dict[str, Any]] = {}
    selected_group: dict[str, dict[str, Any]] = {}
    for phase in PHASES:
        global_entries = [
            value for value in summarized.values()
            if value["target_phase"] == phase and value["kind"] == "global_phase_candidate"
        ]
        group_entries = [
            value for value in summarized.values()
            if value["target_phase"] == phase and value["kind"] == "group_phase_candidate"
        ]
        selected_global[phase] = sorted(global_entries, key=lambda value: (-value["macro_success_rate"], value["macro_query_rate"], int(value["arm_horizon"])))[0]
        selected_group[phase] = sorted(group_entries, key=lambda value: (-value["macro_success_rate"], value["macro_query_rate"], int(value["arm_horizon"]), int(value["gripper_horizon"])))[0]
    return selected_global, selected_group


def combined_config(name: str, strategy: str, selected: dict[str, dict[str, Any]]) -> dict[str, Any]:
    phase_map = {phase: dict(selected[phase]["phase_map"][phase]) for phase in PHASES}
    return {
        "name": name,
        "kind": "phase_oracle_combined",
        "strategy": strategy,
        "target_phase": None,
        "arm_horizon": None,
        "gripper_horizon": None,
        "phase_map": phase_map,
    }


def static_task_results(static_baseline: dict[str, Any], task_ids: list[int], group: bool) -> dict[str, dict[str, Any]]:
    result = {}
    for task_id in task_ids:
        row = static_baseline[str(task_id)]
        vector = row["group_success_vector"] if group else row["global_success_vector"]
        rate = row["group_success_rate"] if group else row["global_success_rate"]
        query_rate = row["group_query_rate"] if group else row["global_query_rate"]
        episodes = int(row["episodes"])
        result[str(task_id)] = {
            "task_id": task_id,
            "task_name": row["task_name"],
            "episodes": episodes,
            "successes": int(sum(vector)),
            "success_rate": float(rate),
            "success_vector": vector,
            "state_ids": row["state_ids"],
            "policy_query_rate": float(query_rate),
            "mean_environment_steps": None,
            "mean_policy_queries": None,
            "phase_summary": None,
        }
    return result


def paired_task_bootstrap(
    dynamic: dict[str, dict[str, Any]],
    static: dict[str, dict[str, Any]],
    task_ids: list[int],
) -> dict[str, Any]:
    task_differences = []
    paired_counts = {}
    for task_id in task_ids:
        dynamic_vector = np.asarray(dynamic[str(task_id)]["success_vector"], dtype=bool)
        static_vector = np.asarray(static[str(task_id)]["success_vector"], dtype=bool)
        if len(dynamic_vector) != len(static_vector):
            raise ValueError(f"state coverage mismatch for task {task_id}")
        task_differences.append(float(np.mean(dynamic_vector.astype(float) - static_vector.astype(float))))
        paired_counts[str(task_id)] = {
            "dynamic_only": int(np.sum(dynamic_vector & ~static_vector)),
            "static_only": int(np.sum(static_vector & ~dynamic_vector)),
            "both_success": int(np.sum(dynamic_vector & static_vector)),
            "both_fail": int(np.sum(~dynamic_vector & ~static_vector)),
        }
    values = np.asarray(task_differences, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return {
        "macro_difference_dynamic_minus_static": float(values.mean()),
        "task_bootstrap_ci95": task_bootstrap(values, rng),
        "task_differences": {str(task_id): float(value) for task_id, value in zip(task_ids, values)},
        "paired_counts_by_task": paired_counts,
    }


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.config = args.config.resolve()
    args.baseline_json = args.baseline_json.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)

    task_ids = parse_task_ids(args.task_ids)
    baseline = load_static_baseline(args.baseline_json, task_ids)
    task_rows = {
        task_id: {"task_name": baseline[str(task_id)]["task_name"], "state_ids": baseline[str(task_id)]["state_ids"]}
        for task_id in task_ids
    }
    if args.configs_json is not None:
        configs = json.loads(args.configs_json.read_text(encoding="utf-8"))
        if not isinstance(configs, list) or not configs:
            raise ValueError("configs-json must contain a non-empty JSON list")
    else:
        configs = build_initial_configs()
    if args.max_configs is not None and args.configs_json is None:
        configs = configs[: args.max_configs]
    cache_path = args.output_dir / "config_results.json"
    cache = load_cache(cache_path, args.resume)
    run_configs(
        configs=configs,
        task_ids=task_ids,
        task_rows=task_rows,
        config_cache=cache,
        cache_path=cache_path,
        args=args,
    )

    smoke = (
        args.configs_json is not None
        or args.max_configs is not None
        or args.task_ids is not None
        or args.state_count is not None
        or args.state_start != 0
    )
    if smoke:
        summary = {
            "status": "smoke_run",
            "task_ids": task_ids,
            "configs_run": [config["name"] for config in configs],
            "coverage": {name: sorted(int(task_id) for task_id in results) for name, results in cache.items() if name in {config["name"] for config in configs}},
            "note": "No full phase-conditioned oracle was selected from this restricted run.",
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    selected_global, selected_group = choose_best_phase_configs(configs=configs, cache=cache, task_ids=task_ids)
    combined = [
        combined_config("phase_oracle_global_combined", "global", selected_global),
        combined_config("phase_oracle_group_combined", "group", selected_group),
    ]
    run_configs(
        configs=combined,
        task_ids=task_ids,
        task_rows=task_rows,
        config_cache=cache,
        cache_path=cache_path,
        args=args,
    )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summarized: dict[str, dict[str, Any]] = {}
    for config in configs + combined:
        summarized[config["name"]] = summarize_config(config["name"], config, cache[config["name"]], rng)
    selected_global_summary = {phase: selected_global[phase] for phase in PHASES}
    selected_group_summary = {phase: selected_group[phase] for phase in PHASES}
    static_group = static_task_results(baseline, task_ids, group=True)
    static_global = static_task_results(baseline, task_ids, group=False)
    combined_group = summarized["phase_oracle_group_combined"]
    combined_global = summarized["phase_oracle_global_combined"]
    summary = {
        "status": "complete",
        "starting_commit": "ba20d60adf8d5f03f1b1d3615266f81b788805c7",
        "checkpoint": str(args.checkpoint),
        "dataset": "LIBERO Object runtime setup; demonstrations used only for prior provenance",
        "task_coverage": {
            "task_ids": task_ids,
            "task_count": len(task_ids),
            "episodes_per_task": {str(task_id): len(task_rows[task_id]["state_ids"]) for task_id in task_ids},
            "seed_rule": "seed = 1000 + init_state_id",
        },
        "protocol": {
            "phase_definition": "environment_step / env._max_episode_steps; early < 1/3, middle < 2/3, late otherwise",
            "phase_decision_timing": "phase horizon applies when a group commitment expires and its next chunk is queried",
            "phase_transition": "no forced query at phase boundary; existing commitment continues",
            "global_horizons": list(HORIZONS),
            "group_horizon_grid": [[arm, gripper] for arm, gripper in itertools.product(HORIZONS, repeat=2)],
            "baseline_outside_target_phase": {"global": BASE_GLOBAL_HORIZON, "group": BASE_GROUP_HORIZONS},
            "static_baseline_source": str(args.baseline_json),
            "oracle_name": "phase-conditioned oracle horizon",
            "training": False,
            "videos": False,
        },
        "action_groups": {"arm": "action[0:6]", "gripper": "action[6]"},
        "metrics": {
            "success_rate": "macro mean of per-task success rates; pooled rate also reported",
            "environment_steps": "sum and macro mean over runtime rollouts",
            "policy_queries": "sum and macro mean of frozen ACT full-chunk calls",
            "query_rate": "policy queries / environment steps",
            "confidence_intervals": "task-level bootstrap 95% CI, 20,000 draws, seed 20260819; per-task Wilson intervals",
        },
        "phase_global_table": {
            phase: {
                "selected": selected_global[phase],
                "candidate_summaries": [
                    summarized[name]
                    for name in sorted(summarized)
                    if summarized[name]["kind"] == "global_phase_candidate" and summarized[name]["target_phase"] == phase
                ],
            }
            for phase in PHASES
        },
        "phase_group_table": {
            phase: {
                "selected": selected_group[phase],
                "candidate_summaries": [
                    summarized[name]
                    for name in sorted(summarized)
                    if summarized[name]["kind"] == "group_phase_candidate" and summarized[name]["target_phase"] == phase
                ],
            }
            for phase in PHASES
        },
        "combined_oracles": {"global": combined_global, "group": combined_group},
        "static_baselines": {
            "global_h16": {"task_results": static_global, "macro_success_rate": float(np.mean([row["success_rate"] for row in static_global.values()]))},
            "group_arm4_grip16": {"task_results": static_group, "macro_success_rate": float(np.mean([row["success_rate"] for row in static_group.values()]))},
        },
        "comparisons": {
            "phase_oracle_global_vs_static_global": paired_task_bootstrap(combined_global["task_results"], static_global, task_ids),
            "phase_oracle_group_vs_static_group": paired_task_bootstrap(combined_group["task_results"], static_group, task_ids),
        },
        "runtime": {"python": platform.python_version()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "selected_global": selected_global, "selected_group": selected_group}, indent=2))


if __name__ == "__main__":
    main()
