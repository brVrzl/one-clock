"""Frozen Candidate-1 paired analysis and claim mapping."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
CROSS_SUITE_PROTOCOL = REPO_ROOT / "experiments" / "cross_suite_confirmation" / "protocol.json"
CROSS_SUITE_RESULTS = REPO_ROOT / "experiments" / "cross_suite_confirmation" / "results"
C2 = "C2_H16_ARM_FRESH_GRIP"
BOOTSTRAP_DRAWS = 20_000
OUTCOME_SENTENCES = {
    "OUTCOME_A": "Across the Object development cohort and the frozen cross-suite confirmation cohort, committing only the arm to the fixed h16 schedule while keeping the gripper fully reactive underperforms fully reactive execution, whereas coherent fixed-h16 execution performs substantially better.",
    "OUTCOME_B": "On the frozen cross-suite confirmation cohort, arm commitment alone does not reproduce the benefit of coherent fixed-h16 execution, although it does not reliably underperform fully reactive execution as it did on the Object development cohort.",
    "OUTCOME_C": "The Object executor decomposition does not transfer to the frozen cross-suite confirmation cohort; it remains a bounded development-setting observation, and we do not claim that arm commitment alone fails to explain the fixed-h16 gain in general.",
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") not in {"pre_outcome_draft", "frozen_before_outcome_rollout"}:
        raise RuntimeError("unexpected Candidate-1 protocol status")
    if protocol["cohort"]["state_ids"] != list(range(14)):
        raise RuntimeError("Candidate-1 state cohort drifted")
    if len(protocol["cohort"]["tasks"]) != 10 or protocol["cohort"]["paired_blocks"] != 140:
        raise RuntimeError("Candidate-1 task cohort is not 140 blocks")
    return protocol


def task_label(task: dict[str, Any]) -> str:
    return f"{task['suite']}:task{int(task['task_id'])}"


def block_keys(protocol: dict[str, Any]) -> list[tuple[str, int, int]]:
    return [
        (task["suite"], int(task["task_id"]), int(state_id))
        for task in protocol["cohort"]["tasks"]
        for state_id in protocol["cohort"]["state_ids"]
    ]


def bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    draws = values[indices].mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def task_bootstrap_ci(task_values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(task_values), size=(BOOTSTRAP_DRAWS, len(task_values)))
    draws = task_values[indices].mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def load_frozen_cross_suite_outcomes(
    candidate_protocol: dict[str, Any], methods: tuple[str, ...]
) -> tuple[dict[tuple[str, int, int, str], int], dict[str, Any]]:
    cross_protocol = json.loads(CROSS_SUITE_PROTOCOL.read_text(encoding="utf-8"))
    if cross_protocol.get("status") != "frozen_before_outcome_rollout":
        raise AssertionError("cross-suite reference protocol is not frozen")
    if cross_protocol["cohort"]["state_ids"] != candidate_protocol["cohort"]["state_ids"]:
        raise AssertionError("cross-suite and Candidate-1 state cohorts differ")
    cross_primary = [
        task
        for task in cross_protocol["cohort"]["tasks"]
        if task["role"] == "primary_unseen_to_executor_development"
    ]
    cross_task_keys = [(task["suite"], int(task["task_id"])) for task in cross_primary]
    candidate_task_keys = [
        (task["suite"], int(task["task_id"])) for task in candidate_protocol["cohort"]["tasks"]
    ]
    if cross_task_keys != candidate_task_keys:
        raise AssertionError("Candidate-1 cohort identity differs from the frozen primary cohort")

    expected_blocks = set(block_keys(candidate_protocol))
    outcomes: dict[tuple[str, int, int, str], int] = {}
    duplicate_counts = {method: 0 for method in methods}
    observed_blocks = {method: set() for method in methods}
    for task in candidate_protocol["cohort"]["tasks"]:
        path = CROSS_SUITE_RESULTS / f"{task['suite']}_task{int(task['task_id'])}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("finished"):
            raise AssertionError(f"incomplete frozen reference file: {path}")
        for method in methods:
            episodes = data.get("episodes", {}).get(method, [])
            for episode in episodes:
                block = (
                    task["suite"],
                    int(task["task_id"]),
                    int(episode["requested_initial_state_id"]),
                )
                key = block + (method,)
                if key in outcomes:
                    duplicate_counts[method] += 1
                    continue
                outcomes[key] = int(bool(episode["success"]))
                observed_blocks[method].add(block)

    join: dict[str, Any] = {}
    for method in methods:
        missing = sorted(expected_blocks - observed_blocks[method])
        extra = sorted(observed_blocks[method] - expected_blocks)
        join[method] = {
            "unique_blocks": len(observed_blocks[method]),
            "expected_blocks": len(expected_blocks),
            "missing_blocks": len(missing),
            "duplicate_blocks": duplicate_counts[method],
            "extra_blocks": len(extra),
        }
        if join[method] != {
            "unique_blocks": 140,
            "expected_blocks": 140,
            "missing_blocks": 0,
            "duplicate_blocks": 0,
            "extra_blocks": 0,
        }:
            raise AssertionError(f"frozen {method} reference join mismatch: {join[method]}")
    return outcomes, join


def comparison(
    outcomes: dict[tuple[str, int, int, str], int],
    protocol: dict[str, Any],
    first: str,
    second: str,
    paired_seed: int,
    cluster_seed: int,
) -> dict[str, Any]:
    keys = block_keys(protocol)
    first_values = np.asarray([outcomes[key + (first,)] for key in keys], dtype=np.int8)
    second_values = np.asarray([outcomes[key + (second,)] for key in keys], dtype=np.int8)
    differences = first_values.astype(float) - second_values.astype(float)
    task_values = np.asarray(
        [
            differences[
                [
                    suite == task["suite"] and task_id == int(task["task_id"])
                    for suite, task_id, _ in keys
                ]
            ].mean()
            for task in protocol["cohort"]["tasks"]
        ]
    )
    first_only = int(np.count_nonzero((first_values == 1) & (second_values == 0)))
    second_only = int(np.count_nonzero((first_values == 0) & (second_values == 1)))
    discordant = first_only + second_only
    return {
        "first_method": first,
        "second_method": second,
        "blocks": len(keys),
        "first_successes": int(first_values.sum()),
        "second_successes": int(second_values.sum()),
        "first_only_wins": first_only,
        "second_only_wins": second_only,
        "discordant_blocks": discordant,
        "success_delta": float(differences.mean()),
        "success_delta_percentage_points": float(100 * differences.mean()),
        "exact_two_sided_mcnemar_p": float(binomtest(first_only, discordant, 0.5).pvalue)
        if discordant
        else 1.0,
        "paired_bootstrap_draws": BOOTSTRAP_DRAWS,
        "paired_bootstrap_seed": int(paired_seed),
        "paired_bootstrap_ci": bootstrap_ci(differences, paired_seed),
        "task_cluster_bootstrap_draws": BOOTSTRAP_DRAWS,
        "task_cluster_bootstrap_seed": int(cluster_seed),
        "task_cluster_bootstrap_ci": task_bootstrap_ci(task_values, cluster_seed),
        "task_labels": [task_label(task) for task in protocol["cohort"]["tasks"]],
        "task_differences": task_values.tolist(),
    }


def classify_outcome(c2_fresh_ci: list[float], hard_c2_ci: list[float]) -> str:
    if c2_fresh_ci[1] < 0 and hard_c2_ci[0] > 0:
        return "OUTCOME_A"
    if c2_fresh_ci[0] <= 0 <= c2_fresh_ci[1] and hard_c2_ci[0] > 0:
        return "OUTCOME_B"
    return "OUTCOME_C"


def frozen_regression(protocol: dict[str, Any]) -> dict[str, Any]:
    outcomes, _ = load_frozen_cross_suite_outcomes(protocol, ("FO20", "REVERSE20"))
    cross_protocol = json.loads(CROSS_SUITE_PROTOCOL.read_text(encoding="utf-8"))
    seeds = cross_protocol["statistics"]["bootstrap_seeds"]["FO20_VS_REVERSE20"]
    result = comparison(
        outcomes,
        protocol,
        "FO20",
        "REVERSE20",
        int(seeds["paired"]),
        int(seeds["task_cluster"]),
    )
    expected = protocol["statistics"]["frozen_regression_test"]
    checks = {
        "delta_percentage_points": np.isclose(
            result["success_delta_percentage_points"],
            expected["expected_delta_percentage_points"],
            rtol=0.0,
            atol=1e-12,
        ),
        "first_only": result["first_only_wins"] == expected["expected_first_only"],
        "second_only": result["second_only_wins"] == expected["expected_second_only"],
        "mcnemar": np.isclose(
            result["exact_two_sided_mcnemar_p"],
            expected["expected_exact_two_sided_mcnemar_p"],
            rtol=1e-14,
            atol=0.0,
        ),
        "paired_ci": np.allclose(
            result["paired_bootstrap_ci"],
            expected["expected_paired_bootstrap_ci"],
            rtol=0.0,
            atol=5e-11,
        ),
        "cluster_ci": np.allclose(
            result["task_cluster_bootstrap_ci"],
            expected["expected_task_cluster_bootstrap_ci"],
            rtol=0.0,
            atol=5e-11,
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"frozen FO20-vs-Reverse20 regression mismatch: {checks}; {result}")
    return {"status": "PASS", "checks": {key: bool(value) for key, value in checks.items()}, **result}


def validate_pre_outcome(protocol: dict[str, Any]) -> dict[str, Any]:
    _, reference_join = load_frozen_cross_suite_outcomes(protocol, ("FRESH", "HARD_H16"))
    regression = frozen_regression(protocol)
    return {
        "status": "PASS",
        "reference_join": reference_join,
        "frozen_fo20_vs_reverse20_regression": regression,
    }


def record_pre_outcome_analysis(protocol_path: Path, result: dict[str, Any]) -> None:
    protocol = load_protocol(protocol_path)
    if protocol["status"] != "pre_outcome_draft":
        raise RuntimeError("pre-outcome analysis validation cannot modify a frozen protocol")
    results = protocol["pre_outcome_validation"].get("results") or {}
    results["analysis_and_reference_validation"] = result
    protocol["pre_outcome_validation"]["results"] = results
    atomic_json(protocol_path, protocol)


def validate_c2_episode(episode: dict[str, Any], task: dict[str, Any], state_id: int, seed: int) -> None:
    if episode["method"] != C2:
        raise AssertionError("Candidate-1 result contains a non-C2 method")
    if int(episode["requested_initial_state_id"]) != state_id or int(episode["environment_seed"]) != seed:
        raise AssertionError("Candidate-1 task/state/seed identity mismatch")
    if not episode["fresh_environment_instance"]:
        raise AssertionError("Candidate-1 episode did not use a fresh environment")
    if int(episode["max_episode_steps"]) != int(task["max_episode_steps"]):
        raise AssertionError("Candidate-1 episode cap drifted")
    steps = int(episode["environment_steps"])
    if int(episode["policy_queries"]) != steps or episode["query_steps"] != list(range(steps)):
        raise AssertionError("C2 did not issue one whole-policy query per controller step")
    if float(episode["query_rate"]) != 1.0:
        raise AssertionError("C2 total policy query rate is not 1.0")
    rows = episode["step_log"]
    if len(rows) != steps:
        raise AssertionError("C2 step log length mismatch")
    for t, row in enumerate(rows):
        q_arm = 16 * (t // 16)
        k_arm = t - q_arm
        if (
            int(row["physical_target_t"]) != t
            or int(row["query_physical_step_q"]) != t
            or int(row["arm_source_query_q"]) != q_arm
            or int(row["arm_chunk_offset"]) != k_arm
            or int(row["gripper_source_query_q"]) != t
            or int(row["gripper_chunk_offset"]) != 0
            or q_arm + k_arm != t
        ):
            raise AssertionError("persisted C2 source/index semantics mismatch")


def load_c2_outcomes(
    protocol: dict[str, Any], results_root: Path
) -> dict[tuple[str, int, int, str], int]:
    outcomes: dict[tuple[str, int, int, str], int] = {}
    for task in protocol["cohort"]["tasks"]:
        path = results_root / f"{task['suite']}_task{int(task['task_id'])}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("finished") or data.get("methods") != [C2]:
            raise AssertionError(f"incomplete or incompatible Candidate-1 result: {path}")
        episodes = data["episodes"].get(C2, [])
        if len(episodes) != 14:
            raise AssertionError(f"Candidate-1 result does not contain 14 C2 episodes: {path}")
        by_state = {int(episode["requested_initial_state_id"]): episode for episode in episodes}
        if len(by_state) != 14:
            raise AssertionError(f"duplicate Candidate-1 task/state key: {path}")
        for state_id, seed in zip(
            protocol["cohort"]["state_ids"], task["environment_seeds"], strict=True
        ):
            episode = by_state[int(state_id)]
            validate_c2_episode(episode, task, int(state_id), int(seed))
            key = (task["suite"], int(task["task_id"]), int(state_id), C2)
            if key in outcomes:
                raise AssertionError(f"duplicate Candidate-1 outcome: {key}")
            outcomes[key] = int(bool(episode["success"]))
    if len(outcomes) != 140:
        raise AssertionError(f"Candidate-1 outcome coverage is {len(outcomes)}, expected 140")
    return outcomes


def deadline_passed(protocol: dict[str, Any], now: datetime | None = None) -> bool:
    current = now or datetime.now(timezone.utc)
    deadline = datetime.fromisoformat(protocol["claim_mapping"]["analysis_deadline"])
    return current > deadline


def analyze_candidate(
    protocol: dict[str, Any], results_root: Path, now: datetime | None = None
) -> dict[str, Any]:
    if deadline_passed(protocol, now):
        return {
            "schema_version": 1,
            "analysis_type": "candidate1_c2_cross_suite",
            "deadline_forced": True,
            "claim_outcome": "OUTCOME_C",
            "manuscript_sentence": OUTCOME_SENTENCES["OUTCOME_C"],
        }
    reference_outcomes, reference_join = load_frozen_cross_suite_outcomes(
        protocol, ("FRESH", "HARD_H16")
    )
    c2_outcomes = load_c2_outcomes(protocol, results_root)
    outcomes = {**reference_outcomes, **c2_outcomes}
    seeds = protocol["statistics"]["candidate_1_bootstrap_seeds"]
    c2_fresh = comparison(
        outcomes,
        protocol,
        C2,
        "FRESH",
        int(seeds["C2_MINUS_FRESH"]["paired"]),
        int(seeds["C2_MINUS_FRESH"]["task_cluster"]),
    )
    hard_c2 = comparison(
        outcomes,
        protocol,
        "HARD_H16",
        C2,
        int(seeds["HARD_H16_MINUS_C2"]["paired"]),
        int(seeds["HARD_H16_MINUS_C2"]["task_cluster"]),
    )
    outcome = classify_outcome(c2_fresh["paired_bootstrap_ci"], hard_c2["paired_bootstrap_ci"])
    return {
        "schema_version": 1,
        "analysis_type": "candidate1_c2_cross_suite",
        "deadline_forced": False,
        "blocks": 140,
        "reference_join": reference_join,
        "comparisons": {
            "C2_MINUS_FRESH": c2_fresh,
            "HARD_H16_MINUS_C2": hard_c2,
        },
        "claim_outcome": outcome,
        "manuscript_sentence": OUTCOME_SENTENCES[outcome],
        "claim_mapping_order": ["OUTCOME_A", "OUTCOME_B", "OUTCOME_C"],
        "exact_mcnemar_note": "the exact McNemar test depends only on discordant blocks; we therefore report their counts explicitly",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-pre-outcome")
    validate.add_argument("--record-protocol", action="store_true")
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--results-root", type=Path, default=ROOT / "results")
    analyze.add_argument("--output", type=Path, default=ROOT / "analysis.json")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    if args.command == "validate-pre-outcome":
        result = validate_pre_outcome(protocol)
        if args.record_protocol:
            record_pre_outcome_analysis(args.protocol, result)
    else:
        if protocol["status"] != "frozen_before_outcome_rollout":
            raise RuntimeError("Candidate-1 outcomes may only be analyzed against the frozen protocol")
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite existing Candidate-1 analysis: {args.output}")
        result = analyze_candidate(protocol, args.results_root)
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
