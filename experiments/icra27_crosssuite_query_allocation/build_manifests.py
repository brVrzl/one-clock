#!/usr/bin/env python3
"""Freeze Track-A and Track-B manifests after every Phase-0 technical gate passes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from conditions import CONDITION_ORDER, CONDITIONS


SUITE_SEED_INDEX = {"libero_spatial": 0, "libero_goal": 1, "libero_10": 2}
BOOTSTRAP_DRAWS = 20_000


def load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def dump(name: str, value: object) -> None:
    (ROOT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def phase0_gate() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    inventory, cohort, exposure = load("checkpoint_inventory.json"), load("confirmation_cohort.json"), load("exposure_audit.json")
    if inventory["summary"]["technically_valid_non_object"] != 30:
        raise RuntimeError("checkpoint gate is not 30/30")
    if cohort["task_count"] != 30 or cohort["selected_block_count"] != 450:
        raise RuntimeError("cohort is not the frozen 30-task/450-block cohort")
    if not cohort["all_selected_track_a_cell_prospective"] or not cohort["all_selected_conservatively_executor_outcome_unexposed"]:
        raise RuntimeError("selected cells do not pass exposure gates")
    if len(exposure["exact_query_allocation_condition_exposure_on_non_object_task_specific_act"]) != 0:
        raise RuntimeError("exact query-allocation exposure exists")
    for name in ("te_dense_audit.json", "historical_operator_audit.json", "normalization_buffer_audit.json"):
        if load(name)["status"] != "PASS":
            raise RuntimeError(f"{name} did not pass")
    if load("gripper_activity_moderator.json")["status"] != "FROZEN_BEFORE_TRACK_A_OUTCOMES":
        raise RuntimeError("moderator is not frozen")
    for suite in SUITE_SEED_INDEX:
        smoke = json.loads((ROOT / "phase0_smoke" / f"{suite}.json").read_text(encoding="utf-8"))
        if smoke["status"] != "PASS" or smoke["scientific_outcomes_used"]:
            raise RuntimeError(f"Phase-0 smoke failed for {suite}")
    return inventory, cohort, exposure


def runtime_prediction(state_count: int) -> dict[str, Any]:
    smokes = {suite: json.loads((ROOT / "phase0_smoke" / f"{suite}.json").read_text()) for suite in SUITE_SEED_INDEX}
    per_task: dict[tuple[str, int], float] = {}
    for suite in SUITE_SEED_INDEX:
        timing = smokes[suite]["timing"]
        seconds_per_block = sum(float(timing[m]["seconds_per_env_step"]) * int(timing[m]["resolved_max_episode_steps"]) for m in CONDITION_ORDER)
        for task_id in range(10):
            per_task[(suite, task_id)] = state_count * seconds_per_block
    workers = defaultdict(float)
    for index, key in enumerate(sorted(per_task)):
        workers[index % 3] += per_task[key]
    return {
        "basis": "exposed-state measured seconds/env-step multiplied by each condition's full episode cap; task-major static sharding; conservative because early terminal episodes are ignored",
        "worker_predicted_seconds": {str(k): v for k, v in sorted(workers.items())},
        "maximum_worker_hours": max(workers.values()) / 3600,
        "bounded_track_a_window_hours": 18.0,
        "fits_window": max(workers.values()) <= 18 * 3600,
    }


def track_a_manifest(inventory: dict[str, Any], cohort: dict[str, Any], state_count: int, prediction: dict[str, Any]) -> dict[str, Any]:
    checkpoints = {(p["suite"], int(p["task_id"])): p["exact_local_path"] for p in inventory["policies"] if p["suite"] != "libero_object" and p["load_smoke"]["succeeds"]}
    cells = []
    for task_key, frozen_states in cohort["selected_states_by_task"].items():
        suite, task_text = task_key.split(":task")
        task_id = int(task_text)
        for state_id in frozen_states[:state_count]:
            block_id = f"{suite}-task{task_id:02d}-state{state_id:02d}"
            env_seed = 370000 + 10000 * SUITE_SEED_INDEX[suite] + 100 * task_id + state_id
            for method in CONDITION_ORDER:
                condition = CONDITIONS[method]
                cells.append({
                    "cell_id": f"{block_id}-{method}", "block_id": block_id,
                    "suite": suite, "task_id": task_id, "state_id": state_id,
                    "environment_seed": env_seed, "policy_seed": 424242,
                    "method": method, "strategy": condition.strategy,
                    "arm_horizon": condition.arm_horizon, "gripper_horizon": condition.gripper_horizon,
                    "checkpoint": checkpoints[(suite, task_id)], "control_frequency_hz": 10,
                    "max_episode_steps": None, "preregistration_commit": "PENDING_PREREGISTRATION_COMMIT",
                })
    return {
        "schema_version": 1, "status": "FROZEN_BEFORE_TRACK_A_OUTCOMES",
        "preregistration_commit": "PENDING_PREREGISTRATION_COMMIT",
        "base_science_commit": "92ed6b281b5c85ff526c2e84ce47b684e86c9d7a",
        "outcomes_inspected_before_freeze": False,
        "scientific_wording": "query-allocation conditions frozen from Object development were prospectively evaluated on non-Object task-state cells selected without reference to their query-allocation outcomes",
        "task_count": 30, "states_per_task": state_count, "paired_block_count": 30 * state_count,
        "condition_count": 6, "scientific_episode_count": 30 * state_count * 6,
        "condition_order": list(CONDITION_ORDER),
        "conditions": {name: CONDITIONS[name].__dict__ for name in CONDITION_ORDER},
        "cohort_source": "confirmation_cohort.json", "checkpoint_source": "checkpoint_inventory.json",
        "state_order_rule": cohort["state_selection_rule"],
        "environment_seed_rule": "370000 + 10000*suite_index(spatial=0,goal=1,libero_10=2) + 100*task_id + state_id; identical across six conditions",
        "policy_seed": 424242,
        "execution": {"workers": 3, "gpus": [0,1,2], "task_major": True, "checkpoint_lifecycle": "load one task checkpoint once, run frozen states x six conditions, fully release policy/environment before next task", "static_task_assignment": "sorted task key index modulo 3", "maximum_attempts": 3, "maximum_technical_retries_after_initial": 2, "scientific_failures_are_completed_outcomes": True, "result_dependent_scheduling": False},
        "runtime_prediction": prediction,
        "statistics": {
            "primary_inference_unit": "task policy", "paired_unit": "task-state block",
            "contrasts": ["H16-H4", "H4-H2", "ARM4_GRIP32-H4", "ARM2_GRIP16-H2", "ARM4_GRIP32-H16", "ARM2_GRIP16-H16", "TE_DENSE-H16", "TE_DENSE-ARM4_GRIP32"],
            "report_every_contrast": ["successes/N", "success rate", "paired discordances", "delta percentage points", "exact two-sided McNemar", "paired percentile bootstrap CI", "task-cluster percentile bootstrap CI", "per-task delta", "leave-one-task-out", "per-suite descriptive delta", "leave-one-suite-out", "policy queries", "query rate", "environment steps", "wall-clock"],
            "bootstrap_draws": BOOTSTRAP_DRAWS, "ci_percentiles": [2.5,97.5],
            "paired_bootstrap_seeds": {"H16-H4": 27001, "H4-H2": 27002, "ARM4_GRIP32-H4": 27003, "ARM2_GRIP16-H2": 27004, "ARM4_GRIP32-H16": 27005, "ARM2_GRIP16-H16": 27006, "TE_DENSE-H16": 27007, "TE_DENSE-ARM4_GRIP32": 27008},
            "task_cluster_bootstrap_seeds": {"H16-H4": 27101, "H4-H2": 27102, "ARM4_GRIP32-H4": 27103, "ARM2_GRIP16-H2": 27104, "ARM4_GRIP32-H16": 27105, "ARM2_GRIP16-H16": 27106, "TE_DENSE-H16": 27107, "TE_DENSE-ARM4_GRIP32": 27108},
            "loto_definition": "omit each task policy in turn and recompute the unweighted mean over remaining paired blocks",
            "leave_one_suite_out_definition": "omit each of the three suites in turn and recompute the unweighted mean over remaining paired blocks",
        },
        "labels": {
            "PENALTY_4X_CONFIRMED": "H16-H4 center >0 and both paired/task-cluster CI lower bounds >0; report LOTO signs",
            "DOSE_RESPONSE_SUPPORTED": "H4-H2 center >0 and both paired/task-cluster CI lower bounds >0",
            "MECHANISM_PASS_A": "PENALTY_4X_CONFIRMED; ARM4_GRIP32-H4 both CI lowers >0, >=90% LOTO positive, all 3 leave-one-suite-out positive; ARM2_GRIP16-H2 center >0",
            "METHOD_PASS_A": "MECHANISM_PASS_A; ARM4_GRIP32-H16 both CI lowers >0 and >=90% LOTO positive",
            "QUERY_EFFICIENT_TE_LEVEL_PERFORMANCE": "upper task-cluster CI for TE_DENSE-ARM4_GRIP32 < +3 pp while TE qrate approximately 1.0 and ARM4_GRIP32 approximately 0.25",
        },
        "practical_equivalence_reference": {"margin_percentage_points": 3.0, "descriptive_only": True, "justification": "At 450 paired blocks, 3 pp is at most about 14 net successes and is predeclared as a small practical materiality tolerance against an approximately fourfold policy-query-rate difference; it is not a formal equivalence theorem."},
        "moderator": {"source": "gripper_activity_moderator.json", "status": "frozen before Track-A outcomes", "analysis": "Spearman correlation using all valid tasks between task-level ARM4_GRIP32-H4 delta and training-demonstration gripper manipulation frequency", "exploratory": True, "posthoc_categories": False},
        "cells": cells,
    }


def track_b_manifest() -> dict[str, Any]:
    panel = [("libero_object",3),("libero_spatial",0),("libero_goal",2),("libero_10",3)]
    smol = "/home/wjq/checkpoints/HuggingFaceVLA_smolvla_libero"
    cells = []
    for policy in ("ACT", "SmolVLA"):
        for suite, task_id in panel:
            checkpoint = smol if policy == "SmolVLA" else f"/home/wjq/workspace/one-clock/experiments/standard_libero_baselines/act_final/{suite}_task{task_id}/checkpoints/100000/pretrained_model"
            for state_id in range(10,20):
                cells.append({
                    "cell_id": f"{policy.lower()}-{suite}-task{task_id:02d}-state{state_id:02d}",
                    "policy": policy, "suite": suite, "task_id": task_id, "state_id": state_id,
                    "environment_seed": 2000 + state_id - 10, "checkpoint": checkpoint,
                    "control_frequency_hz": 10, "max_episode_steps": None,
                })
    return {
        "schema_version": 1, "status": "FROZEN_MECHANISM_DIAGNOSTIC",
        "scope": "mechanism-only logging on already outcome-exposed development cells; outcomes are not used for method selection",
        "success_confirmation": False, "policies": ["ACT","SmolVLA"], "panel": [f"{s}:task{t}" for s,t in panel],
        "states": list(range(10,20)), "cells_per_policy": 40, "total_episodes": 80,
        "trajectory": "execute exactly H16; additionally query every environment step for logging only",
        "execution_canary": "ordinary H16 and dense-logged H16 must exactly match executed actions, simulator states, terminal result, and episode length",
        "canary_cell_ids": ["act-libero_object-task03-state10", "smolvla-libero_object-task03-state10"],
        "normalization": "checkpoint-frozen own-policy normalized action space; no refit",
        "primary_window": {"source_ages": [0,15], "physical_target_time_minimum": 15, "candidate_count": 16},
        "dispersion": "For each physical target and dimension, population RMS deviation across its 16 source predictions. ARM is RMS across dimensions 0..5; GRIPPER is dimension 6. Episode values average eligible targets.",
        "gripper_discrete_metrics": {"sign_disagreement_probability": "fraction of the 120 unordered source pairs with unequal signs; zero is its own sign", "sign_entropy": "Shannon entropy in bits of {-1,0,+1} source signs", "absolute_margin_to_zero": "absolute mean normalized gripper prediction over the 16 sources", "conditioned_disagreement": "within-policy low/middle/high terciles of target-level absolute margin; freeze tercile membership without success outcomes"},
        "ratio": "R_policy = mean episode GRIPPER dispersion / mean episode ARM dispersion; undefined if denominator is zero; no regularizer",
        "bootstrap": {"unit": "episode", "draws": 20000, "ACT_seed": 27201, "SmolVLA_seed": 27202, "difference_seed": 27203, "margin_seed": 27204, "ci_percentiles": [2.5,97.5]},
        "labels": {"ACT_LOCALIZATION_PASS": "R_ACT lower CI >1 and low-margin minus high-margin sign-disagreement CI lower >0", "ACT_LOCALIZATION_KILL": "either ACT criterion fails", "CROSS_POLICY_MECHANISM_SUPPORT": "R_ACT-R_SMOLVLA CI lower >0"},
        "longer_age_curves": "descriptive supplementary only", "method_development_authorized": False,
        "cells": cells,
    }


def write_preregistration(a: dict[str, Any], b: dict[str, Any], exposure: dict[str, Any]) -> None:
    states = defaultdict(list)
    for cell in a["cells"]:
        key = f"{cell['suite']}:task{cell['task_id']}"
        if cell["method"] == "H16":
            states[key].append(cell["state_id"])
    lines = [
        "# Track-A cross-suite query-allocation preregistration", "",
        "Status: **FROZEN BEFORE TRACK-A OUTCOMES**", "",
        "This preregistration governs a prospective evaluation of query-allocation conditions frozen from Object development on non-Object task-state cells selected without reference to their query-allocation outcomes. It does not describe the suites or policies as globally unseen or globally executor-unexposed.", "",
        "## Scientific question and prior-art boundary", "",
        "Frequent replanning degradation and the action-chunk consistency/reactivity trade-off are prior art (ACT, BID, RTC). The new question is whether the penalty is non-uniform across action components and whether preserving gripper commitment while refreshing the arm mitigates it under the same periodic policy-query schedule. We report policy-query rate/budget and replanning frequency; wall-clock is separate. Policy queries are not FLOPs or compute scaling.", "",
        "## Frozen cohort", "",
        f"- Valid task-specific non-Object ACT policies: {a['task_count']}.",
        f"- Frozen states per task: {a['states_per_task']}.",
        f"- Paired task-state blocks: {a['paired_block_count']}.",
        f"- Conditions per block: {a['condition_count']}.",
        f"- Planned scientific episodes: {a['scientific_episode_count']}.",
        "- Object checkpoints/results are development and hypothesis-generation evidence only.",
        "- The deterministic selection rule in `confirmation_cohort.json` conservatively excluded every state with any prior executor-variant outcome. No success rate, difficulty, or previous effect entered selection.",
        f"- Historical non-Object task-specific ACT outcomes under any exact H4/H2/ARM4_GRIP32/ARM2_GRIP16 condition before freeze: {len(exposure['exact_query_allocation_condition_exposure_on_non_object_task_specific_act'])} cells.", "",
        "Exact ordered states:", "",
    ]
    for key in sorted(states):
        lines.append(f"- `{key}`: {states[key]}")
    lines += ["", "## Six frozen conditions", "", "1. `H16`: coherent arm16/gripper16.", "2. `H4`: coherent arm4/gripper4.", "3. `ARM4_GRIP32`: arm refresh 4, gripper commitment 32; exact historical 4x Object contrast.", "4. `H2`: coherent arm2/gripper2.", "5. `ARM2_GRIP16`: arm refresh 2, gripper commitment 16; exact historical 8x Object contrast.", "6. `TE_DENSE`: canonical dense upstream ACT temporal ensembling, query every step, coefficient 0.01, oldest-to-newest exponential weights, all seven normalized action dimensions aggregated before checkpoint denormalization; no tuning and no sparse approximation.", "", "H4 must equal group arm4/grip4 and H2 must equal group arm2/grip2 step-by-step on exposed technical canaries. The frozen workers are task-major: load one task-specific checkpoint once, run all ordered states and six methods, release policy/environment, then load the next task. Static sorted-task modulo-three sharding is fixed. There is no result-dependent scheduling. Scientific failures are outcomes; only technical failures may be retried, at most twice after the initial attempt.", "", "## Questions, contrasts, and inference", "", "The task policy is the primary generality unit and the task-state cell is the paired block. Every contrast reports success counts/rates, paired discordances, percentage-point delta, exact two-sided McNemar, 20,000-draw paired percentile CI, 20,000-draw task-cluster percentile CI, all per-task deltas, leave-one-task-out, per-suite descriptive deltas, leave-one-suite-out, policy queries/rate, environment steps, and wall-clock.", "", "- Q-A1 primary: `H16-H4`; secondary dose response: `H4-H2`.", "- Q-A2 primary matched schedule: `ARM4_GRIP32-H4`; secondary: `ARM2_GRIP16-H2`.", "- Q-A3 primary: `ARM4_GRIP32-H16`; secondary: `ARM2_GRIP16-H16`.", "- Q-A4 standard-practice reference: `TE_DENSE-H16` and `TE_DENSE-ARM4_GRIP32`; TE is not query-budget matched.", "", "Decision-label definitions are frozen verbatim in `track_a_manifest.json`. Strict H4>H2 monotonicity is not required to interpret the 4x mechanism. The 3 pp TE practical-equivalence reference is descriptive only: at this cohort size it is at most about 14 net successes, predeclared as a small materiality tolerance against an approximately fourfold query-rate difference, not a universal equivalence or Pareto theorem.", "", "## Throughput decision", "", f"The conservative task-major prediction is {a['runtime_prediction']['maximum_worker_hours']:.2f} h for the slowest worker against the predeclared 18 h Track-A window. Therefore {a['states_per_task']} states/task are frozen uniformly for all six conditions. TE_DENSE is retained.", "", "## Frozen exploratory moderator", "", "`gripper_activity_moderator.json` fixes, from training-demonstration metadata only, each task's fraction of action steps with closed gripper intent. After outcome freeze, all valid tasks enter one exploratory Spearman correlation with task-level `ARM4_GRIP32-H4`; there are no post-hoc categories or selected tasks.", "", "## Track-B diagnostic", "", f"Track B has {b['total_episodes']} mechanism-only episodes on already outcome-exposed development cells; outcomes are not used for method selection. ACT and SmolVLA execute exact H16 trajectories while extra per-step queries are logging-only. The primary window is target t>=15 and source ages 0..15. Dispersion, sign metrics, bootstrap, and ACT/cross-policy labels are fixed in `track_b_manifest.json`. Dense logging must pass exact action/state/terminal/length canaries. Passing Track B does not authorize debounce, consensus, or any new ICRA method development.", "", "## Prior evidence and interpretation constraints", "", "`evidence_tension_and_factorial_notes.md` is incorporated by reference. In particular, the +32.14 pp FO20-Reverse20 diagonal is not assigned uniquely to either component, the weak FO20-Fresh evidence is kept in tension with Object ARM4_GRIP32-H4 development, differing semantics are hypotheses only, and no reconciliation experiment is authorized. The 126-block Object and 140-block cross-suite factorial interactions retain distinct provenance and opposite sign conventions.", "", "## Frozen exclusions", "", "No fixed-clock search, CARE/M2 variant, grip64, horizon sweep, adaptive threshold, outcome-selected task search, consensus/debounce, RTC pivot, PACE reproduction, or new horizon cell may be launched. The manuscript and `CLAIMS.md` remain untouched.", ""]
    (ROOT / "PREREGISTRATION.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-count", type=int, choices=(10,15), default=None)
    args = parser.parse_args()
    inventory, cohort, exposure = phase0_gate()
    prediction15 = runtime_prediction(15)
    automatic = 15 if prediction15["fits_window"] else 10
    state_count = args.state_count or automatic
    if state_count == 15 and not prediction15["fits_window"]:
        raise RuntimeError("15-state queue does not fit the predeclared 18-hour window")
    prediction = prediction15 if state_count == 15 else runtime_prediction(10)
    a = track_a_manifest(inventory, cohort, state_count, prediction)
    b = track_b_manifest()
    dump("track_a_manifest.json", a)
    dump("track_b_manifest.json", b)
    write_preregistration(a, b, exposure)
    print(json.dumps({"track_a_tasks": a["task_count"], "states_per_task": state_count, "blocks": a["paired_block_count"], "episodes": a["scientific_episode_count"], "predicted_hours": prediction["maximum_worker_hours"], "track_b_episodes": b["total_episodes"]}, indent=2))


if __name__ == "__main__":
    main()
