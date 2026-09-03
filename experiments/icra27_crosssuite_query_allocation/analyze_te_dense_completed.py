#!/usr/bin/env python3
"""Analyze frozen Track-A TE_DENSE timing and gripper behavior from completed artifacts."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "track_a/results"
OUTPUT = ROOT / "track_a/te_dense_characterization"
TRACK_A_SHA = "40549d876c0e09fad4e8033b3206f6018f53ece5"
LABEL = "POST_HOC_TE_EFFECTIVE_AGE_CHARACTERIZATION"
METHODS = ("H16", "H4", "ARM4_GRIP32", "H2", "ARM2_GRIP16", "TE_DENSE")
SUITES = ("libero_10", "libero_goal", "libero_spatial")
CONTRASTS = (
    ("H16", "H4"),
    ("H4", "H2"),
    ("ARM4_GRIP32", "H4"),
    ("ARM2_GRIP16", "H2"),
    ("ARM4_GRIP32", "H16"),
    ("ARM2_GRIP16", "H16"),
    ("TE_DENSE", "H16"),
    ("TE_DENSE", "ARM4_GRIP32"),
)
DT_SECONDS = 0.05


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def discrete_quantile(mass: np.ndarray, probability: float) -> int:
    normalized = mass / mass.sum()
    return int(np.searchsorted(np.cumsum(normalized), probability, side="left"))


def count_summary(values: np.ndarray) -> dict[str, Any]:
    if len(values) == 0:
        return {"step_count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "step_count": int(len(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": int(np.max(values)),
    }


def distribution_summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def age_summary(mass: np.ndarray) -> dict[str, Any]:
    ages = np.arange(len(mass), dtype=np.float64)
    total = mass.sum()
    maximum = int(np.flatnonzero(mass > 0)[-1])
    return {
        "mean_steps": float(np.dot(ages, mass) / total),
        "p50_steps": discrete_quantile(mass, 0.50),
        "p95_steps": discrete_quantile(mass, 0.95),
        "max_steps": maximum,
        "mean_seconds": float(np.dot(ages, mass) / total * DT_SECONDS),
        "p50_seconds": discrete_quantile(mass, 0.50) * DT_SECONDS,
        "p95_seconds": discrete_quantile(mass, 0.95) * DT_SECONDS,
        "max_seconds": maximum * DT_SECONDS,
    }


def main() -> None:
    manifest = json.loads((ROOT / "track_a_manifest.json").read_text())
    canonical = json.loads((ROOT / "track_a/analysis.json").read_text())
    if canonical.get("status") != "COMPLETE" or canonical.get("validated_results") != 2700:
        raise RuntimeError("Track A is not canonically complete")
    if canonical.get("preregistration_commit") != TRACK_A_SHA:
        raise RuntimeError("Track-A preregistration identity drift")

    suite_success: dict[tuple[str, str], int] = defaultdict(int)
    suite_n: dict[tuple[str, str], int] = defaultdict(int)
    gripper_values: dict[str, list[np.ndarray]] = defaultdict(list)
    gripper_switches: dict[str, int] = defaultdict(int)
    gripper_pairs: dict[str, int] = defaultdict(int)
    candidate_counts: list[int] = []
    unweighted_age_hist = np.zeros(100, dtype=np.int64)
    normalized_weight_hist = np.zeros(100, dtype=np.float64)
    old_mass_sums = {10: 0.0, 20: 0.0, 40: 0.0}
    te_cells = 0

    for cell in manifest["cells"]:
        path = RESULT_ROOT / f"{cell['cell_id']}.json"
        payload = json.loads(path.read_text())
        if (
            payload.get("status") != "COMPLETE"
            or payload.get("cell_id") != cell["cell_id"]
            or payload.get("preregistration_commit") != TRACK_A_SHA
        ):
            raise RuntimeError(f"Track-A result identity drift: {path}")
        method = str(cell["method"])
        suite = str(cell["suite"])
        actions = np.asarray(payload["executed_actions"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) != int(payload["environment_steps"]):
            raise RuntimeError(f"invalid executed actions: {path}")
        if not np.isfinite(actions).all():
            raise RuntimeError(f"non-finite executed action: {path}")
        suite_n[(suite, method)] += 1
        suite_success[(suite, method)] += int(bool(payload["success"]))
        gripper = actions[:, 6]
        gripper_values[method].append(gripper)
        if len(gripper) > 1:
            gripper_switches[method] += int(np.count_nonzero(np.sign(gripper[1:]) != np.sign(gripper[:-1])))
            gripper_pairs[method] += len(gripper) - 1

        if method != "TE_DENSE":
            continue
        te_cells += 1
        if (
            float(payload["temporal_ensemble_coeff"]) != 0.01
            or int(payload["chunk_size"]) != 100
            or payload["temporal_ensemble_space"] != "checkpoint-normalized action space"
            or payload["postprocessing_order"] != "aggregate-normalized-then-policy-denormalize-then-env-postprocess"
        ):
            raise RuntimeError(f"TE_DENSE implementation metadata drift: {path}")
        counts = np.asarray(payload["candidate_counts"], dtype=np.int64)
        expected = np.minimum(np.arange(len(actions), dtype=np.int64) + 1, 100)
        if not np.array_equal(counts, expected):
            raise RuntimeError(f"TE_DENSE candidate-count drift: {path}")
        if payload["query_steps"] != list(range(len(actions))) or int(payload["policy_queries"]) != len(actions):
            raise RuntimeError(f"TE_DENSE query-schedule drift: {path}")
        if payload.get("source_ages") or any(key in payload for key in ("predicted_chunks", "candidate_actions", "candidate_gripper_values")):
            raise RuntimeError(f"unexpected TE candidate-level persistence contract: {path}")
        candidate_counts.extend(counts.tolist())
        for count in counts:
            count = int(count)
            ages = np.arange(count, dtype=np.float64)
            # Runtime rank i=0 is oldest; i=count-1 is newest. Therefore
            # exp(-0.01*i), expressed by source age, is proportional to exp(+0.01*age).
            weights = np.exp(0.01 * ages)
            weights /= weights.sum()
            unweighted_age_hist[:count] += 1
            normalized_weight_hist[:count] += weights
            for threshold in old_mass_sums:
                old_mass_sums[threshold] += float(weights[ages > threshold].sum())

    if te_cells != 450 or len(candidate_counts) != canonical["method_summaries"]["TE_DENSE"]["environment_steps"]:
        raise RuntimeError("TE_DENSE empirical support count drift")

    counts_array = np.asarray(candidate_counts, dtype=np.int64)
    startup = counts_array[counts_array < 100]
    steady = counts_array[counts_array == 100]
    theoretical_weights = np.exp(0.01 * np.arange(100, dtype=np.float64))
    theoretical_weights /= theoretical_weights.sum()
    theoretical_unweighted = np.ones(100, dtype=np.float64)

    empirical_weight_mass = normalized_weight_hist / normalized_weight_hist.sum()
    empirical_unweighted_mass = unweighted_age_hist / unweighted_age_hist.sum()
    age_rows = []
    for age in range(100):
        age_rows.append({
            "age_steps": age,
            "age_seconds": age * DT_SECONDS,
            "empirical_unweighted_candidate_count": int(unweighted_age_hist[age]),
            "empirical_unweighted_candidate_fraction": float(empirical_unweighted_mass[age]),
            "empirical_normalized_weight_mass_fraction": float(empirical_weight_mass[age]),
            "theoretical_steady_state_unweighted_fraction": 0.01,
            "theoretical_steady_state_normalized_weight": float(theoretical_weights[age]),
        })

    gripper_rows = []
    for method in METHODS:
        values = np.concatenate(gripper_values[method])
        absolute = np.abs(values)
        native = distribution_summary(values)
        absolute_summary = distribution_summary(absolute)
        gripper_rows.append({
            "condition": method,
            "executed_steps": len(values),
            **{f"g_{key}": value for key, value in native.items() if key != "count"},
            **{f"abs_g_{key}": value for key, value in absolute_summary.items() if key != "count"},
            "fraction_abs_g_lt_0_25": float(np.mean(absolute < 0.25)),
            "fraction_abs_g_lt_0_50": float(np.mean(absolute < 0.50)),
            "gripper_sign_state_switches": gripper_switches[method],
            "eligible_adjacent_pairs": gripper_pairs[method],
            "gripper_sign_state_switch_rate": gripper_switches[method] / gripper_pairs[method],
        })

    suite_rows = []
    for suite in SUITES:
        for method in METHODS:
            n = suite_n[(suite, method)]
            successes = suite_success[(suite, method)]
            if n != 150:
                raise RuntimeError(f"suite/method N drift: {suite}/{method}={n}")
            suite_rows.append({
                "suite": suite,
                "condition": method,
                "successes": successes,
                "N": n,
                "success_rate": successes / n,
            })
    suite_lookup = {(row["suite"], row["condition"]): row for row in suite_rows}
    suite_contrasts = []
    for suite in SUITES:
        for first, second in CONTRASTS:
            first_row = suite_lookup[(suite, first)]
            second_row = suite_lookup[(suite, second)]
            suite_contrasts.append({
                "suite": suite,
                "contrast": f"{first}-{second}",
                "first_successes": first_row["successes"],
                "second_successes": second_row["successes"],
                "N": first_row["N"],
                "delta_percentage_points": round(
                    100 * (first_row["success_rate"] - second_row["success_rate"]), 12
                ),
            })
    for row in suite_contrasts:
        canonical_delta = canonical["contrasts"][row["contrast"]]["per_suite_descriptive_delta_percentage_points"][row["suite"]]
        if not np.isclose(row["delta_percentage_points"], canonical_delta):
            raise RuntimeError(f"canonical suite-contrast mismatch: {row}")

    theoretical = {
        "candidate_count": 100,
        "unweighted_candidate_age": age_summary(theoretical_unweighted),
        "normalized_weight_effective_age": age_summary(theoretical_weights),
        "normalized_weight_mass_older_than": {
            "0.50_seconds_strictly_older": float(theoretical_weights[11:].sum()),
            "1.00_seconds_strictly_older": float(theoretical_weights[21:].sum()),
            "2.00_seconds_strictly_older": float(theoretical_weights[41:].sum()),
        },
    }
    empirical = {
        "executed_steps": len(counts_array),
        "candidate_availability": {
            "all": count_summary(counts_array),
            "startup_count_lt_100": count_summary(startup),
            "steady_state_count_eq_100": count_summary(steady),
            "startup_fraction": float(len(startup) / len(counts_array)),
            "steady_state_fraction": float(len(steady) / len(counts_array)),
        },
        "unweighted_candidate_age": age_summary(unweighted_age_hist.astype(np.float64)),
        "normalized_weight_effective_age": age_summary(normalized_weight_hist),
        "normalized_weight_mass_older_than": {
            "0.50_seconds_strictly_older": old_mass_sums[10] / len(counts_array),
            "1.00_seconds_strictly_older": old_mass_sums[20] / len(counts_array),
            "2.00_seconds_strictly_older": old_mass_sums[40] / len(counts_array),
        },
    }
    result = {
        "status": "COMPLETE",
        "label": LABEL,
        "post_hoc": True,
        "track_a_preregistration_commit": TRACK_A_SHA,
        "reviewer_supplement_inputs_loaded": False,
        "b3_inputs_loaded": False,
        "rerollout_performed": False,
        "physical_clock_hz": 20,
        "seconds_per_step": DT_SECONDS,
        "verified_runtime_implementation": {
            "class": "lerobot.policies.act.modeling_act.ACTTemporalEnsembler",
            "coefficient": 0.01,
            "chunk_size": 100,
            "runtime_rank_weight": "exp(-0.01*i), where i=0 is the oldest candidate",
            "equivalent_source_age_weight": "exp(+0.01*age), normalized over available ages",
            "requested_exp_negative_age_equivalence": False,
            "aggregation_space": "checkpoint-normalized action space",
            "postprocessing": "aggregate, then checkpoint inverse normalization, then environment postprocess",
        },
        "theoretical_steady_state": theoretical,
        "empirical_realized": empirical,
        "gripper_candidate_disagreement": {
            "status": "NOT_IDENTIFIABLE_FROM_EXISTING_TRACK_A_ARTIFACTS",
            "reason": "candidate chunks and pre-aggregation candidate gripper values were not persisted; only candidate counts and executed aggregate actions were recorded",
            "fraction_steps_with_both_signs": None,
            "weighted_minority_sign_mass": None,
        },
        "gripper_condition_summaries": gripper_rows,
        "per_suite_absolute": suite_rows,
        "per_suite_contrasts": suite_contrasts,
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    write_csv(OUTPUT / "candidate_age_distribution.csv", age_rows)
    write_csv(OUTPUT / "gripper_condition_summaries.csv", gripper_rows)
    write_csv(OUTPUT / "track_a_per_suite_absolute.csv", suite_rows)
    write_csv(OUTPUT / "track_a_per_suite_contrasts.csv", suite_contrasts)

    e_age = empirical["normalized_weight_effective_age"]
    t_age = theoretical["normalized_weight_effective_age"]
    e_unweighted = empirical["unweighted_candidate_age"]
    t_unweighted = theoretical["unweighted_candidate_age"]
    availability = empirical["candidate_availability"]
    e_mass = empirical["normalized_weight_mass_older_than"]
    t_mass = theoretical["normalized_weight_mass_older_than"]
    lines = [
        "# TE_DENSE effective-age and gripper characterization", "",
        f"Label: `{LABEL}`", "", "Status: **COMPLETE**", "",
        "This is explanatory post-hoc characterization after observing the frozen TE_DENSE result. It uses completed Track-A artifacts only and does not tune the coefficient.", "",
        "## Verified implementation", "",
        "The frozen implementation is **not** equivalent to `exp(-0.01*age)`. LeRobot assigns rank `i=0` to the oldest candidate and weights rank as `exp(-0.01*i)`. In source-age coordinates this is `exp(+0.01*age)`, normalized over the candidates available at that step. Aggregation occurs in checkpoint-normalized action space before inverse normalization.", "",
        "## Candidate availability", "",
        "| Segment | Steps | Mean count | p50 | p95 | Maximum |", "|---|---:|---:|---:|---:|---:|",
        f"| All executed TE steps | {availability['all']['step_count']} | {availability['all']['mean']:.3f} | {availability['all']['p50']:.0f} | {availability['all']['p95']:.0f} | {availability['all']['max']} |",
        f"| Startup (`count<100`) | {availability['startup_count_lt_100']['step_count']} | {availability['startup_count_lt_100']['mean']:.3f} | {availability['startup_count_lt_100']['p50']:.0f} | {availability['startup_count_lt_100']['p95']:.0f} | {availability['startup_count_lt_100']['max']} |",
        f"| Steady state (`count=100`) | {availability['steady_state_count_eq_100']['step_count']} | 100.000 | 100 | 100 | 100 |", "",
        "## Effective age", "",
        "| Distribution | Mean | p50 | p95 | Maximum support |", "|---|---:|---:|---:|---:|",
        f"| Theoretical steady-state unweighted candidates | {t_unweighted['mean_steps']:.3f} steps / {t_unweighted['mean_seconds']:.3f} s | {t_unweighted['p50_steps']} / {t_unweighted['p50_seconds']:.2f} s | {t_unweighted['p95_steps']} / {t_unweighted['p95_seconds']:.2f} s | {t_unweighted['max_steps']} / {t_unweighted['max_seconds']:.2f} s |",
        f"| Empirical pooled unweighted candidates | {e_unweighted['mean_steps']:.3f} steps / {e_unweighted['mean_seconds']:.3f} s | {e_unweighted['p50_steps']} / {e_unweighted['p50_seconds']:.2f} s | {e_unweighted['p95_steps']} / {e_unweighted['p95_seconds']:.2f} s | {e_unweighted['max_steps']} / {e_unweighted['max_seconds']:.2f} s |",
        f"| Theoretical steady state | {t_age['mean_steps']:.3f} steps / {t_age['mean_seconds']:.3f} s | {t_age['p50_steps']} / {t_age['p50_seconds']:.2f} s | {t_age['p95_steps']} / {t_age['p95_seconds']:.2f} s | {t_age['max_steps']} / {t_age['max_seconds']:.2f} s |",
        f"| Empirical realized | {e_age['mean_steps']:.3f} steps / {e_age['mean_seconds']:.3f} s | {e_age['p50_steps']} / {e_age['p50_seconds']:.2f} s | {e_age['p95_steps']} / {e_age['p95_seconds']:.2f} s | {e_age['max_steps']} / {e_age['max_seconds']:.2f} s |",
        "", "The pooled unweighted row counts every candidate occurrence once. The normalized-weight row gives every executed step total mass one before pooling. Maximum support is not the weighted mean effective age.", "",
        "## Normalized weight assigned to old predictions", "",
        "Thresholds are strict: older than 0.50 s means age >10 control steps.", "",
        "| Age threshold | Theoretical steady state | Empirical realized |", "|---|---:|---:|",
        f"| >0.50 s | {t_mass['0.50_seconds_strictly_older']:.6f} | {e_mass['0.50_seconds_strictly_older']:.6f} |",
        f"| >1.00 s | {t_mass['1.00_seconds_strictly_older']:.6f} | {e_mass['1.00_seconds_strictly_older']:.6f} |",
        f"| >2.00 s | {t_mass['2.00_seconds_strictly_older']:.6f} | {e_mass['2.00_seconds_strictly_older']:.6f} |", "",
        "## Executed gripper diagnostics", "",
        "The native gripper command is continuous. These are pooled executed-step summaries without new inferential tests.", "",
        "| Condition | Steps | mean(g) | mean(abs(g)) | p50(abs(g)) | abs(g)<0.25 | abs(g)<0.50 | sign/state-switch rate |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in gripper_rows:
        lines.append(
            f"| {row['condition']} | {row['executed_steps']} | {row['g_mean']:.6f} | {row['abs_g_mean']:.6f} | "
            f"{row['abs_g_p50']:.6f} | {row['fraction_abs_g_lt_0_25']:.6f} | "
            f"{row['fraction_abs_g_lt_0_50']:.6f} | {row['gripper_sign_state_switch_rate']:.6f} |"
        )
    lines.extend([
        "",
        "## Candidate-level gripper disagreement", "",
        "`NOT_IDENTIFIABLE_FROM_EXISTING_TRACK_A_ARTIFACTS`: candidate chunks and pre-aggregation gripper values were not persisted. Computing sign disagreement or minority-sign mass would require a rerollout, which is not authorized.", "",
        "## Scope", "",
        "Any interpretation is limited to the frozen upstream coefficient and chunk length in this ACT/LIBERO evaluation. These results do not show that canonical temporal ensembling is intrinsically harmful.", "",
        "Canonical values are in `analysis.json` and the accompanying CSV files.", "",
    ])
    (OUTPUT / "report.md").write_text("\n".join(lines))
    print(json.dumps({"status": "COMPLETE", "te_steps": len(counts_array), "te_cells": te_cells}, indent=2))


if __name__ == "__main__":
    main()
