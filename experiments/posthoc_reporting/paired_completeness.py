"""Zero-rollout post-hoc completeness statistics from frozen result artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "experiments"
OUTPUT = Path(__file__).with_suffix(".json")

CONFIRMATION = EXPERIMENTS / "cross_suite_confirmation"
CANDIDATE1 = EXPERIMENTS / "candidate1_c2_cross_suite"
OBJECT_FACTORIAL = EXPERIMENTS / "group_delay_factorial_act20"
OBJECT_ASYMMETRIC = EXPERIMENTS / "asymmetric_chunk_reuse_dev"
OBJECT_DECOMPOSITION = EXPERIMENTS / "object_executor_decomposition"

PRIMARY_TASKS = [
    ("libero_goal", 4),
    ("libero_goal", 6),
    ("libero_goal", 7),
    ("libero_goal", 8),
    ("libero_goal", 9),
    ("libero_10", 0),
    ("libero_10", 2),
    ("libero_10", 4),
    ("libero_10", 6),
    ("libero_10", 7),
]
PRIMARY_STATES = list(range(14))
PRIMARY_METHODS = ("FRESH", "FO20", "REVERSE20", "FULL_OLD20", "HARD_H16")

OBJECT_TASKS = list(range(1, 10))
OBJECT_STATES = [20, 21, 22, 23, 27, 31, 34, 35, 38, 39, 44, 45, 47, 48]
C2 = "C2_H16_ARM_FRESH_GRIP"
BOOTSTRAP_DRAWS = 20_000

# Frozen before this analysis is executed. Do not change after inspecting output.
SEEDS = {
    "HARD_H16_MINUS_FRESH": {"paired": 20261201, "task_cluster": 20261202},
    "HARD_H16_MINUS_FO20": {"paired": 20261203, "task_cluster": 20261204},
    "REVERSE20_MINUS_FRESH": {"paired": 20261205, "task_cluster": 20261206},
    "FULL_OLD20_MINUS_FO20": {"paired": 20261207, "task_cluster": 20261208},
    "FO20_MINUS_FRESH": {"paired": 20261209, "task_cluster": 20261210},
    "FULL_OLD20_MINUS_REVERSE20": {"paired": 20261211, "task_cluster": 20261212},
    "FACTORIAL_INTERACTION": {"paired": 20261213, "task_cluster": 20261214},
    "OBJECT_HARD_H16_MINUS_C2": {"paired": 20261215, "task_cluster": 20261216},
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def task_label(task: tuple[str, int]) -> str:
    return f"{task[0]}:task{task[1]}"


def bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    draws = values[indices].mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def join_summary(
    observed: Counter[tuple[Any, ...]], expected: set[tuple[Any, ...]]
) -> dict[str, int]:
    observed_keys = set(observed)
    return {
        "unique_blocks": len(observed_keys),
        "expected_blocks": len(expected),
        "missing_blocks": len(expected - observed_keys),
        "duplicate_blocks": sum(count - 1 for count in observed.values() if count > 1),
        "extra_blocks": len(observed_keys - expected),
    }


def require_exact_join(label: str, summary: dict[str, int], blocks: int) -> None:
    expected = {
        "unique_blocks": blocks,
        "expected_blocks": blocks,
        "missing_blocks": 0,
        "duplicate_blocks": 0,
        "extra_blocks": 0,
    }
    if summary != expected:
        raise AssertionError(f"{label} input join differs from the frozen cohort: {summary}")


def load_confirmation() -> tuple[
    dict[tuple[str, int, int, str], int], dict[str, dict[str, int]]
]:
    protocol = read_json(CONFIRMATION / "protocol.json")
    primary = [
        (task["suite"], int(task["task_id"]))
        for task in protocol["cohort"]["tasks"]
        if task["role"] == "primary_unseen_to_executor_development"
    ]
    if (
        protocol.get("status") != "frozen_before_outcome_rollout"
        or primary != PRIMARY_TASKS
        or protocol["cohort"]["state_ids"] != PRIMARY_STATES
    ):
        raise AssertionError("frozen primary confirmation cohort metadata drifted")

    expected = {(suite, task_id, state) for suite, task_id in PRIMARY_TASKS for state in PRIMARY_STATES}
    observed = {method: Counter() for method in PRIMARY_METHODS}
    outcomes: dict[tuple[str, int, int, str], int] = {}
    for suite, task_id in PRIMARY_TASKS:
        path = CONFIRMATION / "results" / f"{suite}_task{task_id}.json"
        data = read_json(path)
        if (
            not data.get("finished")
            or data.get("methods") != list(PRIMARY_METHODS)
            or data.get("state_ids") != PRIMARY_STATES
        ):
            raise AssertionError(f"incomplete or incompatible frozen confirmation result: {path}")
        for method in PRIMARY_METHODS:
            for episode in data.get("episodes", {}).get(method, []):
                block = (
                    str(episode["suite"]),
                    int(episode["task_id"]),
                    int(episode["requested_initial_state_id"]),
                )
                if episode["method"] != method:
                    raise AssertionError(f"method label mismatch in {path}")
                observed[method][block] += 1
                key = block + (method,)
                if key not in outcomes:
                    outcomes[key] = int(bool(episode["success"]))

    joins = {method: join_summary(observed[method], expected) for method in PRIMARY_METHODS}
    for method, summary in joins.items():
        require_exact_join(f"confirmation {method}", summary, 140)
    return outcomes, joins


def paired_comparison(
    outcomes: dict[tuple[Any, ...], int],
    blocks: list[tuple[Any, ...]],
    task_labels: list[str],
    first: str,
    second: str,
    paired_seed: int,
    cluster_seed: int,
) -> dict[str, Any]:
    first_values = np.asarray([outcomes[block + (first,)] for block in blocks], dtype=np.int8)
    second_values = np.asarray([outcomes[block + (second,)] for block in blocks], dtype=np.int8)
    differences = first_values.astype(float) - second_values.astype(float)
    task_differences = differences.reshape(len(task_labels), -1).mean(axis=1)
    first_only = int(np.count_nonzero((first_values == 1) & (second_values == 0)))
    second_only = int(np.count_nonzero((first_values == 0) & (second_values == 1)))
    discordant = first_only + second_only
    effect = float(differences.mean())
    return {
        "post_hoc": True,
        "first_method": first,
        "second_method": second,
        "method_direction": f"{first} - {second}",
        "inference_unit": "paired task-state block",
        "blocks": len(blocks),
        "first_successes": int(first_values.sum()),
        "second_successes": int(second_values.sum()),
        "first_only_wins": first_only,
        "second_only_wins": second_only,
        "discordant_count": discordant,
        "effect": effect,
        "effect_percentage_points": 100.0 * effect,
        "exact_two_sided_mcnemar_p": (
            float(binomtest(first_only, discordant, 0.5).pvalue) if discordant else 1.0
        ),
        "paired_bootstrap_draws": BOOTSTRAP_DRAWS,
        "paired_bootstrap_seed": paired_seed,
        "paired_bootstrap_ci": bootstrap_ci(differences, paired_seed),
        "task_cluster_bootstrap_draws": BOOTSTRAP_DRAWS,
        "task_cluster_bootstrap_seed": cluster_seed,
        "task_cluster_bootstrap_ci": bootstrap_ci(task_differences, cluster_seed),
        "task_level_differences": [
            {"task": label, "effect": float(value)}
            for label, value in zip(task_labels, task_differences, strict=True)
        ],
    }


def load_candidate_c2(
    expected_blocks: set[tuple[str, int, int]],
) -> dict[tuple[str, int, int, str], int]:
    protocol = read_json(CANDIDATE1 / "protocol.json")
    tasks = [(task["suite"], int(task["task_id"])) for task in protocol["cohort"]["tasks"]]
    if (
        protocol.get("status") != "frozen_before_outcome_rollout"
        or tasks != PRIMARY_TASKS
        or protocol["cohort"]["state_ids"] != PRIMARY_STATES
    ):
        raise AssertionError("frozen Candidate-1 cohort metadata drifted")

    observed: Counter[tuple[str, int, int]] = Counter()
    outcomes: dict[tuple[str, int, int, str], int] = {}
    for suite, task_id in PRIMARY_TASKS:
        path = CANDIDATE1 / "results" / f"{suite}_task{task_id}.json"
        data = read_json(path)
        if not data.get("finished") or data.get("methods") != [C2] or data.get("state_ids") != PRIMARY_STATES:
            raise AssertionError(f"incomplete or incompatible frozen Candidate-1 result: {path}")
        for episode in data["episodes"][C2]:
            block = (
                str(episode["suite"]),
                int(episode["task_id"]),
                int(episode["requested_initial_state_id"]),
            )
            if episode["method"] != C2:
                raise AssertionError(f"Candidate-1 condition label mismatch in {path}")
            observed[block] += 1
            key = block + (C2,)
            if key not in outcomes:
                outcomes[key] = int(bool(episode["success"]))
    require_exact_join("Candidate-1 C2", join_summary(observed, expected_blocks), 140)
    return outcomes


def load_object_development() -> tuple[
    dict[tuple[int, int, str], int], dict[str, Any]
]:
    canonical = read_json(OBJECT_DECOMPOSITION / "analysis.json")
    factorial_protocol = read_json(OBJECT_FACTORIAL / "protocol.json")
    asymmetric_protocol = read_json(OBJECT_ASYMMETRIC / "protocol.json")
    compatibility = canonical.get("compatibility", {})
    if (
        not canonical.get("compatible")
        or canonical["scope"]["tasks"] != OBJECT_TASKS
        or canonical["scope"]["states"] != OBJECT_STATES
        or canonical["scope"]["blocks"] != 126
        or factorial_protocol["cohort"]["primary_task_ids"] != OBJECT_TASKS
        or factorial_protocol["cohort"]["state_ids"] != OBJECT_STATES
        or asymmetric_protocol["cohort"]["primary_task_ids"] != OBJECT_TASKS
        or asymmetric_protocol["cohort"]["state_ids"] != OBJECT_STATES
        or compatibility.get("factorial_commit") != "7ab52cbc6360ae8436cfe5a04f8d200130d3f7a4"
        or compatibility.get("asym_commit") != "4cf1cbf97411e0cd7face0974c26adc1b25de37d"
    ):
        raise AssertionError("canonical Object development source metadata is ambiguous or drifted")

    expected = {(task_id, state) for task_id in OBJECT_TASKS for state in OBJECT_STATES}
    methods = ("HARD_H16", C2)
    observed = {method: Counter() for method in methods}
    outcomes: dict[tuple[int, int, str], int] = {}
    for task_id in OBJECT_TASKS:
        sources = {
            "HARD_H16": read_json(OBJECT_FACTORIAL / "results" / f"task_{task_id:02d}.json"),
            C2: read_json(OBJECT_ASYMMETRIC / "results" / f"task_{task_id:02d}.json"),
        }
        for method, data in sources.items():
            if not data.get("finished") or data.get("state_ids") != OBJECT_STATES:
                raise AssertionError(f"incomplete canonical Object source for task {task_id}: {method}")
            for episode in data.get("episodes", {}).get(method, []):
                block = (int(episode["task_id"]), int(episode["requested_initial_state_id"]))
                if episode["method"] != method:
                    raise AssertionError(f"Object condition label mismatch for task {task_id}: {method}")
                observed[method][block] += 1
                key = block + (method,)
                if key not in outcomes:
                    outcomes[key] = int(bool(episode["success"]))

    joins = {method: join_summary(observed[method], expected) for method in methods}
    for method, summary in joins.items():
        require_exact_join(f"Object {method}", summary, 126)
    joined = set(observed["HARD_H16"]) & set(observed[C2])
    if joined != expected:
        raise AssertionError(f"Object joined cohort differs from the canonical 126 blocks: {len(joined)}")
    totals = {
        method: sum(outcomes[block + (method,)] for block in expected) for method in methods
    }
    if totals != {"HARD_H16": 88, C2: 42}:
        raise AssertionError(f"Object success anchors failed: {totals}")
    frozen_pair = canonical["comparisons"]["HARD_H16_VS_C2"]
    if (
        frozen_pair["first_method"] != "HARD_H16"
        or frozen_pair["second_method"] != C2
        or frozen_pair["first_successes"] != 88
        or frozen_pair["second_successes"] != 42
    ):
        raise AssertionError("canonical Object decomposition condition mapping drifted")
    return outcomes, {
        "canonical_source_identified": True,
        "canonical_metadata": "experiments/object_executor_decomposition/analysis.json",
        "hard_h16_source": "experiments/group_delay_factorial_act20/results/task_01.json through task_09.json",
        "c2_source": "experiments/asymmetric_chunk_reuse_dev/results/task_01.json through task_09.json",
        "condition_joins": joins,
        "joined_unique_blocks": len(joined),
        "expected_joined_blocks": 126,
        "HARD_H16_successes": totals["HARD_H16"],
        f"{C2}_successes": totals[C2],
    }


def assert_close(label: str, actual: float, expected: float) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
        raise AssertionError(f"sanity anchor failed for {label}: {actual} != {expected}")


def main() -> None:
    confirmation, confirmation_joins = load_confirmation()
    primary_blocks = [
        (suite, task_id, state)
        for suite, task_id in PRIMARY_TASKS
        for state in PRIMARY_STATES
    ]
    primary_labels = [task_label(task) for task in PRIMARY_TASKS]

    totals = {
        method: sum(confirmation[block + (method,)] for block in primary_blocks)
        for method in PRIMARY_METHODS
    }
    expected_totals = {
        "FRESH": 77,
        "FO20": 83,
        "REVERSE20": 38,
        "FULL_OLD20": 66,
        "HARD_H16": 93,
    }
    if totals != expected_totals:
        raise AssertionError(f"primary confirmation success anchors failed: {totals}")
    primary_anchor_effects = {
        "FO20_MINUS_REVERSE20": (totals["FO20"] - totals["REVERSE20"]) * 100 / 140,
        "FO20_MINUS_FRESH": (totals["FO20"] - totals["FRESH"]) * 100 / 140,
        "FULL_OLD20_MINUS_FRESH": (totals["FULL_OLD20"] - totals["FRESH"]) * 100 / 140,
    }
    assert_close("FO20 - REVERSE20", primary_anchor_effects["FO20_MINUS_REVERSE20"], 45 * 100 / 140)
    assert_close("FO20 - FRESH", primary_anchor_effects["FO20_MINUS_FRESH"], 6 * 100 / 140)
    assert_close("FULL_OLD20 - FRESH", primary_anchor_effects["FULL_OLD20_MINUS_FRESH"], -11 * 100 / 140)

    pairwise_specs = {
        "HARD_H16_MINUS_FRESH": ("HARD_H16", "FRESH"),
        "HARD_H16_MINUS_FO20": ("HARD_H16", "FO20"),
    }
    confirmation_pairwise = {
        label: paired_comparison(
            confirmation,
            primary_blocks,
            primary_labels,
            first,
            second,
            SEEDS[label]["paired"],
            SEEDS[label]["task_cluster"],
        )
        for label, (first, second) in pairwise_specs.items()
    }

    simple_specs = {
        "ARM_WITH_FRESH_GRIPPER_REVERSE20_MINUS_FRESH": (
            "REVERSE20_MINUS_FRESH",
            "REVERSE20",
            "FRESH",
        ),
        "ARM_WITH_OLD_GRIPPER_FULL_OLD20_MINUS_FO20": (
            "FULL_OLD20_MINUS_FO20",
            "FULL_OLD20",
            "FO20",
        ),
        "GRIPPER_WITH_FRESH_ARM_FO20_MINUS_FRESH": (
            "FO20_MINUS_FRESH",
            "FO20",
            "FRESH",
        ),
        "GRIPPER_WITH_OLD_ARM_FULL_OLD20_MINUS_REVERSE20": (
            "FULL_OLD20_MINUS_REVERSE20",
            "FULL_OLD20",
            "REVERSE20",
        ),
    }
    simple_effects = {
        label: paired_comparison(
            confirmation,
            primary_blocks,
            primary_labels,
            first,
            second,
            SEEDS[seed_label]["paired"],
            SEEDS[seed_label]["task_cluster"],
        )
        for label, (seed_label, first, second) in simple_specs.items()
    }

    full_old = np.asarray(
        [confirmation[block + ("FULL_OLD20",)] for block in primary_blocks], dtype=float
    )
    reverse = np.asarray(
        [confirmation[block + ("REVERSE20",)] for block in primary_blocks], dtype=float
    )
    fo20 = np.asarray([confirmation[block + ("FO20",)] for block in primary_blocks], dtype=float)
    fresh = np.asarray([confirmation[block + ("FRESH",)] for block in primary_blocks], dtype=float)
    interaction_values = full_old - reverse - fo20 + fresh
    task_interactions = interaction_values.reshape(len(PRIMARY_TASKS), -1).mean(axis=1)
    interaction = {
        "post_hoc": True,
        "formula": "(FULL_OLD20 - REVERSE20) - (FO20 - FRESH)",
        "block_level_formula": "FULL_OLD20_i - REVERSE20_i - FO20_i + FRESH_i",
        "blocks": 140,
        "mean_interaction": float(interaction_values.mean()),
        "interaction_percentage_points": float(100 * interaction_values.mean()),
        "paired_bootstrap_draws": BOOTSTRAP_DRAWS,
        "paired_bootstrap_seed": SEEDS["FACTORIAL_INTERACTION"]["paired"],
        "paired_bootstrap_ci": bootstrap_ci(
            interaction_values, SEEDS["FACTORIAL_INTERACTION"]["paired"]
        ),
        "task_cluster_bootstrap_draws": BOOTSTRAP_DRAWS,
        "task_cluster_bootstrap_seed": SEEDS["FACTORIAL_INTERACTION"]["task_cluster"],
        "task_cluster_bootstrap_ci": bootstrap_ci(
            task_interactions, SEEDS["FACTORIAL_INTERACTION"]["task_cluster"]
        ),
        "task_level_mean_interactions": [
            {"task": label, "mean_interaction": float(value)}
            for label, value in zip(primary_labels, task_interactions, strict=True)
        ],
        "mcnemar_applied": False,
    }

    expected_primary_blocks = set(primary_blocks)
    candidate_c2 = load_candidate_c2(expected_primary_blocks)
    candidate_totals = {
        "C2": sum(candidate_c2[block + (C2,)] for block in primary_blocks),
        "FRESH": totals["FRESH"],
        "HARD_H16": totals["HARD_H16"],
    }
    if candidate_totals != {"C2": 76, "FRESH": 77, "HARD_H16": 93}:
        raise AssertionError(f"Candidate-1 success anchors failed: {candidate_totals}")
    candidate_effects = {
        "C2_MINUS_FRESH": (candidate_totals["C2"] - candidate_totals["FRESH"]) * 100 / 140,
        "HARD_H16_MINUS_C2": (candidate_totals["HARD_H16"] - candidate_totals["C2"]) * 100 / 140,
    }
    assert_close("Candidate-1 C2 - FRESH", candidate_effects["C2_MINUS_FRESH"], -1 * 100 / 140)
    assert_close("Candidate-1 HARD_H16 - C2", candidate_effects["HARD_H16_MINUS_C2"], 17 * 100 / 140)

    frozen_candidate = read_json(CANDIDATE1 / "analysis.json")
    if frozen_candidate.get("claim_outcome") != "OUTCOME_B":
        raise AssertionError("frozen Candidate-1 OUTCOME_B classification drifted")
    frozen_hard_c2 = frozen_candidate["comparisons"]["HARD_H16_MINUS_C2"]
    if (
        frozen_hard_c2["first_successes"] != 93
        or frozen_hard_c2["second_successes"] != 76
        or len(frozen_hard_c2["task_differences"]) != 10
    ):
        raise AssertionError("frozen Candidate-1 HARD_H16 - C2 analysis drifted")
    candidate_task_differences = np.asarray(frozen_hard_c2["task_differences"], dtype=float)
    candidate_loto = np.asarray(
        [np.delete(candidate_task_differences, index).mean() for index in range(10)]
    )

    object_outcomes, object_gate = load_object_development()
    object_blocks = [(task_id, state) for task_id in OBJECT_TASKS for state in OBJECT_STATES]
    object_result = paired_comparison(
        object_outcomes,
        object_blocks,
        [f"libero_object:task{task_id}" for task_id in OBJECT_TASKS],
        "HARD_H16",
        C2,
        SEEDS["OBJECT_HARD_H16_MINUS_C2"]["paired"],
        SEEDS["OBJECT_HARD_H16_MINUS_C2"]["task_cluster"],
    )
    assert_close("Object HARD_H16 - C2", object_result["effect_percentage_points"], 46 * 100 / 126)

    artifact = {
        "schema_version": 1,
        "analysis_role": "post_hoc_reporting_completeness",
        "post_hoc": True,
        "changes_frozen_candidate1_classification": False,
        "input_validation": {
            "confirmation_input_joins": confirmation_joins,
            "object_development_input_gate": object_gate,
        },
        "sanity_anchors": {
            "status": "PASS",
            "primary_confirmation_successes": {
                method: {"successes": successes, "blocks": 140}
                for method, successes in totals.items()
            },
            "existing_primary_effects_percentage_points": primary_anchor_effects,
            "candidate1": {
                "classification": frozen_candidate["claim_outcome"],
                "successes": candidate_totals,
                "effects_percentage_points": candidate_effects,
            },
            "object_development": {
                "C2_successes": 42,
                "HARD_H16_successes": 88,
                "HARD_H16_MINUS_C2_percentage_points": 46 * 100 / 126,
            },
        },
        "confirmation_posthoc_pairwise": confirmation_pairwise,
        "confirmation_factorial_simple_effects": {
            "interpretation_scope": (
                "Descriptive post-hoc decompositions of the frozen factorial; they do not "
                "independently identify observation-age effects."
            ),
            "comparisons": simple_effects,
        },
        "confirmation_factorial_interaction": interaction,
        "object_development_posthoc": {
            "interpretation_scope": (
                "Cross-cohort decomposition reporting completeness only; this does not modify "
                "any frozen development or Candidate-1 claim."
            ),
            "comparison": object_result,
        },
        "descriptive_robustness": {
            "HARD_H16_MINUS_C2_CONFIRMATION_LOTO": {
                "post_hoc": True,
                "source": "frozen Candidate-1 task-level differences",
                "leave_one_task_out_effects_positive": int(np.count_nonzero(candidate_loto > 0)),
                "minimum_leave_one_task_out_effect": float(candidate_loto.min()),
                "maximum_leave_one_task_out_effect": float(candidate_loto.max()),
            }
        },
    }
    OUTPUT.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "sanity_anchors": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
