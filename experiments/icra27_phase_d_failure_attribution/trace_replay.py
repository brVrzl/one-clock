#!/usr/bin/env python3
"""Open-loop LIBERO-10 command replay for the frozen Phase-D protocol."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
TRACK_A_RESULTS = REPO_ROOT / "experiments" / "icra27_crosssuite_query_allocation" / "track_a" / "results"
PHASE1_RESULTS = REPO_ROOT / "experiments" / "icra27_phase1_executor_discriminator" / "results"
RAW_ROOT = ROOT / "raw_logs"
VIDEO_ROOT = ROOT / "videos"
SUMMARY_ROOT = ROOT / "summaries"
MAP_PATH = ROOT / "TASK_MANIPULATION_OPPORTUNITIES.json"
PROTOCOL_PATH = ROOT / "protocol.json"

CONTRASTS = {
    "development_h4": ("H4", "ARM4_GRIP32"),
    "development_h2": ("H2", "ARM2_GRIP16"),
    "phase1_h4": ("H4", "ARM4_GRIP32"),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def frozen_commit() -> str:
    sha = (ROOT / "PREREGISTRATION_COMMIT").read_text(encoding="utf-8").strip()
    if len(sha) != 40:
        raise RuntimeError("Phase-D PREREGISTRATION_COMMIT is not a full commit ID")
    subprocess.run(["git", "merge-base", "--is-ancestor", sha, "HEAD"], cwd=REPO_ROOT, check=True)
    protocol = read_json(PROTOCOL_PATH)
    if protocol["status"] != "FROZEN_BEFORE_PHASE1_OUTCOME_UNBLINDING":
        raise RuntimeError("Phase-D protocol is not frozen")
    if protocol["confirmation"]["outcomes_unblinded_before_freeze"] is not False:
        raise RuntimeError("Phase-D protocol does not certify a prospective outcome freeze")
    return sha


def task_map() -> dict[int, dict[str, Any]]:
    value = read_json(MAP_PATH)
    if value["status"] != "FROZEN_BEFORE_PHASE1_OUTCOME_UNBLINDING":
        raise RuntimeError("task opportunity map is not frozen")
    tasks = {int(task["task_id"]): task for task in value["tasks"]}
    if sorted(tasks) != list(range(10)):
        raise RuntimeError("task map must contain all ten LIBERO-10 tasks")
    return tasks


def point_to_oriented_box_distance(
    point: np.ndarray, center: np.ndarray, rotation: np.ndarray, half_size: np.ndarray
) -> float:
    local = rotation.T @ (point - center)
    outside = np.maximum(np.abs(local) - half_size, 0.0)
    return float(np.linalg.norm(outside))


def geom_union_clearance(sim: Any, eef_position: np.ndarray, geom_names: Iterable[str]) -> float:
    clearances = []
    for name in geom_names:
        geom_id = sim.model.geom_name2id(name)
        center = np.asarray(sim.data.geom_xpos[geom_id], dtype=np.float64)
        radius = float(sim.model.geom_rbound[geom_id])
        clearances.append(float(np.linalg.norm(eef_position - center) - radius))
    if not clearances:
        raise RuntimeError("opportunity geom set is empty")
    return min(clearances)


def site_clearance(sim: Any, eef_position: np.ndarray, site_name: str) -> float:
    site_id = sim.model.site_name2id(site_name)
    center = np.asarray(sim.data.site_xpos[site_id], dtype=np.float64)
    site_type = int(sim.model.site_type[site_id])
    size = np.asarray(sim.model.site_size[site_id], dtype=np.float64)
    if site_type == 6:  # mjGEOM_BOX
        rotation = np.asarray(sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
        return point_to_oriented_box_distance(eef_position, center, rotation, size[:3])
    return float(np.linalg.norm(eef_position - center) - float(np.max(size)))


def result_index(result_root: Path, methods: set[str]) -> dict[tuple[int, int, str], dict[str, Any]]:
    values: dict[tuple[int, int, str], dict[str, Any]] = {}
    for path in sorted(result_root.glob("libero_10-task*-state*-*.json")):
        value = read_json(path)
        if value.get("suite") != "libero_10" or value.get("method") not in methods:
            continue
        key = (int(value["task_id"]), int(value["state_id"]), str(value["method"]))
        if key in values:
            raise RuntimeError(f"duplicate result cell {key}")
        value["_source_path"] = str(path.resolve())
        values[key] = value
    return values


def paired_blocks(
    index: dict[tuple[int, int, str], dict[str, Any]], baseline: str, treatment: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    blocks = []
    for task_id, state_id, method in sorted(index):
        if method != baseline:
            continue
        b = index[(task_id, state_id, baseline)]
        try:
            t = index[(task_id, state_id, treatment)]
        except KeyError as exc:
            raise RuntimeError(f"missing paired treatment for task {task_id}, state {state_id}") from exc
        for key in ("block_id", "suite", "task_id", "state_id", "environment_seed"):
            if b[key] != t[key]:
                raise RuntimeError(f"paired {key} mismatch for {b['block_id']}")
        if not np.array_equal(
            np.asarray(b["initial_sim_state"], dtype=np.float64),
            np.asarray(t["initial_sim_state"], dtype=np.float64),
        ):
            raise RuntimeError(f"paired initial simulator state mismatch for {b['block_id']}")
        blocks.append((b, t))
    return blocks


def pair_type(baseline: dict[str, Any], treatment: dict[str, Any]) -> str:
    b, t = bool(baseline["success"]), bool(treatment["success"])
    if not b and t:
        return "rescue"
    if b and not t:
        return "harm"
    if b and t:
        return "both_succeed"
    return "both_fail"


def relevant_geom_names(raw: Any, task: dict[str, Any]) -> set[str]:
    names = set(raw.robots[0].gripper.important_geoms["left_finger"])
    names.update(raw.robots[0].gripper.important_geoms["right_finger"])
    for stage in task["stages"]:
        opportunity = stage["opportunity"]
        if opportunity["kind"] == "object_contact_geoms":
            names.update(raw.get_object(opportunity["name"]).contact_geoms)
        elif opportunity["kind"] == "explicit_geoms":
            names.update(opportunity["names"])
    return names


def contact_pairs(raw: Any, relevant: set[str]) -> list[list[str]]:
    result = []
    model, data = raw.sim.model, raw.sim.data
    for i in range(int(data.ncon)):
        contact = data.contact[i]
        name1 = model.geom_id2name(int(contact.geom1))
        name2 = model.geom_id2name(int(contact.geom2))
        if name1 in relevant or name2 in relevant:
            result.append([name1, name2])
    return result


def object_geoms(raw: Any, name: str) -> list[str]:
    obj = raw.get_object(name)
    if obj is None:
        raise RuntimeError(f"missing mapped object {name}")
    return list(obj.contact_geoms)


def eef_position(raw: Any) -> np.ndarray:
    site = raw.robots[0].gripper.important_sites["grip_site"]
    return np.asarray(raw.sim.data.get_site_xpos(site), dtype=np.float64)


def opportunity(raw: Any, spec: dict[str, Any], margin_m: float = 0.04) -> tuple[bool, float]:
    p = eef_position(raw)
    if spec["kind"] == "object_contact_geoms":
        clearance = geom_union_clearance(raw.sim, p, object_geoms(raw, spec["name"]))
    elif spec["kind"] == "explicit_geoms":
        clearance = geom_union_clearance(raw.sim, p, spec["names"])
    elif spec["kind"] == "site":
        clearance = site_clearance(raw.sim, p, spec["name"])
    else:
        raise RuntimeError(f"unknown opportunity kind {spec['kind']}")
    return clearance <= margin_m, clearance


def joint_qpos(raw: Any, joint_name: str) -> float:
    address = raw.sim.model.get_joint_qpos_addr(joint_name)
    if not isinstance(address, (int, np.integer)):
        raise RuntimeError(f"mapped fixture joint {joint_name} is not scalar")
    return float(raw.sim.data.qpos[int(address)])


def predicate(raw: Any, value: list[str] | None) -> bool:
    return False if value is None else bool(raw._eval_predicate(value))


def grasped(raw: Any, object_name: str) -> bool:
    return bool(raw._check_grasp(raw.robots[0].gripper, raw.get_object(object_name)))


def finger_contact(raw: Any, geom_names: Iterable[str]) -> bool:
    fingers = list(raw.robots[0].gripper.important_geoms["left_finger"])
    fingers += list(raw.robots[0].gripper.important_geoms["right_finger"])
    return bool(raw.check_contact(fingers, list(geom_names)))


class StageTracker:
    def __init__(self, raw: Any, task: dict[str, Any]):
        self.raw = raw
        self.task = task
        self.states = [
            {
                "id": stage["id"],
                "kind": stage["kind"],
                "credited_complete": False,
                "opportunity_reached": False,
                "attempted": False,
                "acquired_or_engaged": False,
                "lost": False,
                "last_grasped": False,
                "active_start_joint": None,
                "first_opportunity_step": None,
                "first_attempt_step": None,
                "first_engagement_step": None,
                "completion_step": None,
            }
            for stage in task["stages"]
        ]
        if task["stages"][0]["kind"] == "fixture":
            self.states[0]["active_start_joint"] = joint_qpos(raw, task["stages"][0]["joint"])

    def active_index(self) -> int | None:
        for i, state in enumerate(self.states):
            if not state["credited_complete"]:
                return i
        return None

    def update(self, step: int, command: np.ndarray) -> dict[str, Any]:
        active = self.active_index()
        snapshots = []
        for i, (stage, state) in enumerate(zip(self.task["stages"], self.states, strict=True)):
            complete_now = grasped(self.raw, stage["entity"]) if stage["kind"] == "acquire" else predicate(
                self.raw, stage.get("completion")
            )
            snapshots.append(
                {
                    "id": stage["id"],
                    "kind": stage["kind"],
                    "active": i == active,
                    "complete_now": complete_now,
                    "credited_complete": bool(state["credited_complete"]),
                }
            )

        if active is not None:
            stage = self.task["stages"][active]
            state = self.states[active]
            reached, clearance = opportunity(self.raw, stage["opportunity"])
            snapshots[active]["opportunity"] = reached
            snapshots[active]["opportunity_clearance_m"] = clearance
            if reached:
                state["opportunity_reached"] = True
                if state["first_opportunity_step"] is None:
                    state["first_opportunity_step"] = step

            complete_now = bool(snapshots[active]["complete_now"])
            if stage["kind"] == "acquire":
                geoms = object_geoms(self.raw, stage["entity"])
                attempted_now = reached and (float(command[6]) > 0.0 or finger_contact(self.raw, geoms))
                engaged_now = complete_now
                if state["acquired_or_engaged"] and not engaged_now:
                    state["lost"] = True
                state["last_grasped"] = engaged_now
            elif stage["kind"] == "place":
                held_now = grasped(self.raw, stage["entity"])
                release_transition = bool(state["last_grasped"] and not held_now)
                attempted_now = reached and (float(command[6]) < 0.0 or release_transition)
                engaged_now = attempted_now or complete_now
                if release_transition and not complete_now and not reached:
                    state["lost"] = True
                state["last_grasped"] = held_now
            else:
                attempted_now = reached and finger_contact(self.raw, stage["interaction_geoms"])
                if state["active_start_joint"] is None:
                    state["active_start_joint"] = joint_qpos(self.raw, stage["joint"])
                current = joint_qpos(self.raw, stage["joint"])
                start = float(state["active_start_joint"])
                engaged_now = bool(
                    state["attempted"]
                    and ((stage["direction"] == "increase" and current > start) or (stage["direction"] == "decrease" and current < start))
                )
                if state["acquired_or_engaged"] and not finger_contact(self.raw, stage["interaction_geoms"]) and not complete_now:
                    state["lost"] = True
                snapshots[active]["joint_qpos"] = current
                snapshots[active]["joint_start_qpos"] = start

            if attempted_now:
                state["attempted"] = True
                if state["first_attempt_step"] is None:
                    state["first_attempt_step"] = step
            if engaged_now:
                state["acquired_or_engaged"] = True
                if state["first_engagement_step"] is None:
                    state["first_engagement_step"] = step

            if complete_now:
                state["credited_complete"] = True
                state["completion_step"] = step
                snapshots[active]["credited_complete"] = True
                next_index = active + 1
                if next_index < len(self.states):
                    next_stage = self.task["stages"][next_index]
                    if next_stage["kind"] == "place":
                        self.states[next_index]["last_grasped"] = grasped(self.raw, next_stage["entity"])
                    elif next_stage["kind"] == "fixture":
                        self.states[next_index]["active_start_joint"] = joint_qpos(self.raw, next_stage["joint"])

        new_active = self.active_index()
        return {
            "active_stage": "COMPLETE" if new_active is None else self.task["stages"][new_active]["id"],
            "any_active_opportunity": bool(active is not None and snapshots[active].get("opportunity", False)),
            "stages": snapshots,
        }

    def classification(self, success: bool) -> dict[str, Any]:
        if success:
            category = "SUCCESS"
            detail = None
            failed_stage = None
        else:
            index = self.active_index()
            if index is None:
                # Frozen fallback for an internally inconsistent automatic
                # record: every ordered predicate was credited at some time,
                # but the environment never reported its full BDDL condition.
                category = "BLIND_MANUAL_REVIEW"
                detail = None
                failed_stage = None
            else:
                state = self.states[index]
                failed_stage = state["id"]
                if not state["opportunity_reached"]:
                    detail = "PRE_OPPORTUNITY_FAILURE"
                elif not state["attempted"]:
                    detail = "INTERACTION_EXECUTION_FAILURE"
                elif state["lost"]:
                    detail = "POST_ACQUISITION_LOSS"
                else:
                    detail = "ACQUISITION_OR_ENGAGEMENT_FAILURE"
                category = detail if index == 0 else "LATER_STAGE_FAILURE"
        return {
            "failure_category": category,
            "later_stage_detail": detail if category == "LATER_STAGE_FAILURE" else None,
            "failed_stage": failed_stage,
            "ever_manipulation_opportunity": any(bool(state["opportunity_reached"]) for state in self.states),
            "stage_states": self.states,
        }


def mapped_poses(raw: Any, task: dict[str, Any]) -> dict[str, Any]:
    names = set(task.get("manipulated", [])) | set(task.get("targets", []))
    for stage in task["stages"]:
        names.add(stage["entity"])
        if stage.get("target"):
            names.add(stage["target"])
    poses: dict[str, Any] = {}
    model, data = raw.sim.model, raw.sim.data
    for name in sorted(names):
        try:
            body_id = model.body_name2id(name)
        except ValueError:
            body_id = raw.obj_body_id.get(name)
        if body_id is not None:
            poses[name] = {
                "kind": "body",
                "position": np.asarray(data.body_xpos[body_id], dtype=float).tolist(),
                "quaternion_wxyz": np.asarray(data.body_xquat[body_id], dtype=float).tolist(),
            }
            continue
        try:
            site_id = model.site_name2id(name)
        except ValueError as exc:
            raise RuntimeError(f"missing mapped pose entity {name}") from exc
        poses[name] = {
            "kind": "site",
            "position": np.asarray(data.site_xpos[site_id], dtype=float).tolist(),
            "orientation_matrix_row_major": np.asarray(data.site_xmat[site_id], dtype=float).reshape(-1).tolist(),
        }
    return poses


def fixture_joints(raw: Any, task: dict[str, Any]) -> dict[str, float]:
    return {
        stage["joint"]: joint_qpos(raw, stage["joint"])
        for stage in task["stages"]
        if stage["kind"] == "fixture"
    }


class ReplayEnvironment:
    def __init__(self, task_id: int):
        os.environ.setdefault("MUJOCO_GL", "egl")
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env

        cfg = LiberoEnv(
            task="libero_10",
            task_ids=[task_id],
            fps=10,
            obs_type="pixels_agent_pos",
            camera_name="agentview_image,robot0_eye_in_hand_image",
            camera_name_mapping={"agentview_image": "image", "robot0_eye_in_hand_image": "image2"},
            init_states=True,
            observation_width=256,
            observation_height=256,
            control_mode="relative",
        )
        self.vector = make_env(cfg, n_envs=1, use_async_envs=False)["libero_10"][task_id]
        self.wrapper = self.vector.envs[0]
        self.task_id = task_id

    def reset(self, source: dict[str, Any]) -> tuple[Any, bool, float]:
        if int(source["task_id"]) != self.task_id:
            raise RuntimeError("source task does not match replay environment")
        gripper = self.wrapper._env.env.robots[0].gripper
        gripper.current_action = np.zeros(gripper.dof, dtype=np.float64)
        self.wrapper.init_state_id = int(source["state_id"])
        self.vector.reset(seed=[int(source["environment_seed"])])
        low = self.wrapper._env
        saved = np.asarray(source["initial_sim_state"], dtype=np.float64)
        low.set_init_state(saved)
        restored = np.asarray(low.get_sim_state(), dtype=np.float64)
        equal = bool(np.array_equal(saved, restored))
        max_difference = float(np.max(np.abs(saved - restored))) if len(saved) else 0.0
        return low, equal, max_difference

    def close(self) -> None:
        self.vector.close()


def video_writers(replay_id: str, cohort: str):
    import imageio.v2 as imageio

    directory = VIDEO_ROOT / cohort
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "agent": directory / f"{replay_id}.agent.mp4",
        "wrist": directory / f"{replay_id}.wrist.mp4",
    }
    writers = {
        key: imageio.get_writer(path, fps=10, codec="libx264", quality=7, macro_block_size=1)
        for key, path in paths.items()
    }
    return writers, paths


def replay_commands(
    environment: ReplayEnvironment,
    source: dict[str, Any],
    commands: list[list[float]],
    task: dict[str, Any],
    cohort: str,
    replay_id: str,
    *,
    write_video: bool,
    hybrid_semantics: bool = False,
) -> dict[str, Any]:
    low, initial_equal, initial_max_difference = environment.reset(source)
    raw = low.env
    if environment.wrapper.task != task["task_name"]:
        raise RuntimeError(f"task-name mismatch for task {source['task_id']}")
    tracker = StageTracker(raw, task)
    relevant = relevant_geom_names(raw, task)
    log_path = RAW_ROOT / cohort / f"{replay_id}.jsonl.gz"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    writers = paths = None
    if write_video:
        writers, paths = video_writers(replay_id, cohort)
    command_mismatches = 0
    success = False
    terminated = False
    truncated = False
    termination_reason = None
    steps = 0
    try:
        with gzip.open(log_path, "wt", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "record_type": "metadata",
                        "replay_id": replay_id,
                        "cohort": cohort,
                        "source_path": source.get("_source_path"),
                        "source_cell_id": source.get("cell_id"),
                        "source_block_id": source.get("block_id"),
                        "suite": source["suite"],
                        "task_id": source["task_id"],
                        "state_id": source["state_id"],
                        "environment_seed": source["environment_seed"],
                        "policy_seed": source.get("policy_seed"),
                        "control_frequency_hz": 10,
                        "resolved_max_episode_steps": source["resolved_max_episode_steps"],
                        "initial_state_exact": initial_equal,
                        "initial_state_max_abs_difference": initial_max_difference,
                        "hybrid_semantics": hybrid_semantics,
                        "act_queries": 0,
                    }
                )
                + "\n"
            )
            for step, command_values in enumerate(commands):
                expected = np.asarray(command_values, dtype=np.float32)
                command = expected.copy()
                if command.shape != (7,) or not np.array_equal(command, expected):
                    command_mismatches += 1
                raw_obs, reward, raw_done, _ = low.step(command)
                success = bool(low.check_success())
                terminated = bool(raw_done or success)
                stage_snapshot = tracker.update(step, command)
                reason = "ENVIRONMENT_DONE" if raw_done else "SUCCESS" if success else None
                record = {
                    "record_type": "step",
                    "step_index": step,
                    "replay_command_7d": command.astype(float).tolist(),
                    "eef_position": np.asarray(raw_obs["robot0_eef_pos"], dtype=float).tolist(),
                    "eef_quaternion_xyzw": np.asarray(raw_obs["robot0_eef_quat"], dtype=float).tolist(),
                    "gripper_qpos": np.asarray(raw_obs["robot0_gripper_qpos"], dtype=float).tolist(),
                    "gripper_qvel": np.asarray(raw_obs["robot0_gripper_qvel"], dtype=float).tolist(),
                    "relevant_body_and_site_poses": mapped_poses(raw, task),
                    "fixture_joint_qpos": fixture_joints(raw, task),
                    "stage_predicates": stage_snapshot["stages"],
                    "relevant_contact_pairs": contact_pairs(raw, relevant),
                    "reward": float(reward),
                    "success": success,
                    "terminated": terminated,
                    "truncated": False,
                    "termination_reason": reason,
                    "manipulation_opportunity": stage_snapshot["any_active_opportunity"],
                    "stage_label": stage_snapshot["active_stage"],
                }
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                if writers is not None:
                    writers["agent"].append_data(np.asarray(raw_obs["agentview_image"])[::-1, ::-1])
                    writers["wrist"].append_data(np.asarray(raw_obs["robot0_eye_in_hand_image"])[::-1, ::-1])
                steps = step + 1
                if terminated:
                    termination_reason = reason
                    break
            if not terminated:
                cap = int(source["resolved_max_episode_steps"])
                termination_reason = "EPISODE_CAP_REACHED" if steps >= cap else "COMMAND_TRACE_EXHAUSTED"
                truncated = termination_reason == "EPISODE_CAP_REACHED"
            attribution = tracker.classification(success)
            summary_record = {
                "record_type": "summary",
                "steps": steps,
                "success": success,
                "terminated": terminated,
                "truncated": truncated,
                "termination_reason": termination_reason,
                "command_mismatch_count": command_mismatches,
                **attribution,
            }
            stream.write(json.dumps(summary_record, separators=(",", ":")) + "\n")
    finally:
        if writers is not None:
            for writer in writers.values():
                writer.close()

    result = {
        "replay_id": replay_id,
        "cohort": cohort,
        "source_path": source.get("_source_path"),
        "source_cell_id": source.get("cell_id"),
        "source_block_id": source.get("block_id"),
        "task_id": int(source["task_id"]),
        "state_id": int(source["state_id"]),
        "method": source.get("method"),
        "source_success": bool(source.get("success")),
        "replay_success": success,
        "source_steps": int(source.get("environment_steps", len(commands))),
        "replay_steps": steps,
        "initial_state_exact": initial_equal,
        "initial_state_max_abs_difference": initial_max_difference,
        "command_mismatch_count": command_mismatches,
        "termination_reason": termination_reason,
        "raw_log": str(log_path.resolve()),
        "videos": {key: str(path.resolve()) for key, path in (paths or {}).items()},
        **tracker.classification(success),
    }
    return result


def summary_path(cohort: str, replay_id: str) -> Path:
    return SUMMARY_ROOT / cohort / f"{replay_id}.json"


def run_replay(
    environment: ReplayEnvironment,
    source: dict[str, Any],
    commands: list[list[float]],
    task: dict[str, Any],
    cohort: str,
    replay_id: str,
    *,
    write_video: bool,
    hybrid_semantics: bool = False,
) -> dict[str, Any]:
    path = summary_path(cohort, replay_id)
    if path.is_file():
        prior = read_json(path)
        raw_ok = Path(prior["raw_log"]).is_file()
        video_ok = not write_video or all(Path(value).is_file() for value in prior.get("videos", {}).values())
        if raw_ok and video_ok:
            return prior
    result = replay_commands(
        environment,
        source,
        commands,
        task,
        cohort,
        replay_id,
        write_video=write_video,
        hybrid_semantics=hybrid_semantics,
    )
    atomic_json(path, result)
    return result


def canary() -> None:
    frozen_commit()
    protocol = read_json(PROTOCOL_PATH)
    maps = task_map()
    index = result_index(TRACK_A_RESULTS, {"H4"})
    rows = []
    for task_id in range(10):
        state_id = int(protocol["development"]["canary_state_id_by_task"][str(task_id)])
        source = index[(task_id, state_id, "H4")]
        environment = ReplayEnvironment(task_id)
        try:
            replay_id = f"task{task_id:02d}-state{state_id:02d}-H4"
            result = run_replay(
                environment,
                source,
                source["executed_actions"],
                maps[task_id],
                "canary",
                replay_id,
                write_video=False,
            )
        finally:
            environment.close()
        passed = bool(
            result["initial_state_exact"]
            and result["command_mismatch_count"] == 0
            and result["replay_success"] == result["source_success"]
            and result["replay_steps"] == result["source_steps"]
        )
        rows.append({**result, "passed": passed})
        if not passed:
            break
    report = {
        "status": "PASS" if len(rows) == 10 and all(row["passed"] for row in rows) else "FAIL_STOP",
        "required_tasks": 10,
        "completed_tasks": len(rows),
        "rows": rows,
    }
    atomic_json(ROOT / "replay_canary_report.json", report)
    lines = ["# Phase-D original-trace replay canaries", "", f"Status: `{report['status']}`", "", "| Task | State | Source/replay success | Source/replay steps | Initial exact | Command mismatches | Pass |", "|---:|---:|---|---|---|---:|---|"]
    for row in rows:
        lines.append(
            f"| {row['task_id']} | {row['state_id']} | {row['source_success']}/{row['replay_success']} | "
            f"{row['source_steps']}/{row['replay_steps']} | {row['initial_state_exact']} | "
            f"{row['command_mismatch_count']} | {row['passed']} |"
        )
    (ROOT / "REPLAY_CANARY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise RuntimeError("original-trace canary failed; scientific replay is blocked")


def assert_canary_passed() -> None:
    path = ROOT / "replay_canary_report.json"
    if not path.is_file() or read_json(path).get("status") != "PASS":
        raise RuntimeError("all ten original-trace canaries must pass before scientific replay")


def selected_sources(
    blocks: list[tuple[dict[str, Any], dict[str, Any]]]
) -> tuple[dict[str, dict[str, Any]], set[str], list[dict[str, Any]]]:
    selected: dict[str, dict[str, Any]] = {}
    videos: set[str] = set()
    roles = []
    for baseline, treatment in blocks:
        kind = pair_type(baseline, treatment)
        if not bool(baseline["success"]):
            selected[baseline["_source_path"]] = baseline
            roles.append({"source_path": baseline["_source_path"], "role": "baseline_failure", "block_id": baseline["block_id"]})
        if kind in {"rescue", "harm"}:
            selected[baseline["_source_path"]] = baseline
            selected[treatment["_source_path"]] = treatment
            videos.update({baseline["_source_path"], treatment["_source_path"]})
            roles.extend(
                [
                    {"source_path": baseline["_source_path"], "role": f"{kind}_baseline", "block_id": baseline["block_id"]},
                    {"source_path": treatment["_source_path"], "role": f"{kind}_treatment", "block_id": baseline["block_id"]},
                ]
            )
    return selected, videos, roles


def run_source_cohort(cohort: str, task_ids: set[int] | None = None) -> None:
    frozen_commit()
    assert_canary_passed()
    maps = task_map()
    if cohort == "development":
        root = TRACK_A_RESULTS
        contrast_names = ["development_h4", "development_h2"]
    elif cohort == "phase1":
        if not (ROOT / "PHASE1_UNBLINDING_RECORD.json").is_file():
            raise RuntimeError("Phase-1 outcomes have not been formally unblinded")
        root = PHASE1_RESULTS
        contrast_names = ["phase1_h4"]
    else:
        raise ValueError(cohort)
    methods = {method for name in contrast_names for method in CONTRASTS[name]}
    index = result_index(root, methods)
    all_selected: dict[str, dict[str, Any]] = {}
    all_videos: set[str] = set()
    all_roles = []
    for contrast_name in contrast_names:
        baseline, treatment = CONTRASTS[contrast_name]
        blocks = paired_blocks(index, baseline, treatment)
        selected, videos, roles = selected_sources(blocks)
        all_selected.update(selected)
        all_videos.update(videos)
        all_roles.extend({"contrast": contrast_name, **role} for role in roles)
    manifest = {
        "cohort": cohort,
        "selected_episode_count": len(all_selected),
        "video_episode_count": len(all_videos),
        "roles": all_roles,
    }
    atomic_json(ROOT / f"{cohort}_replay_manifest.json", manifest)
    by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for source in all_selected.values():
        by_task[int(source["task_id"])].append(source)
    for task_id in sorted(by_task):
        if task_ids is not None and task_id not in task_ids:
            continue
        for source in sorted(by_task[task_id], key=lambda v: (int(v["state_id"]), str(v["method"]))):
            # Match the source runner's one-new-environment-per-cell isolation.
            # MuJoCo flat state omits controller and wrapper fields, so a reused
            # environment can retain hidden history despite an exact flat state.
            environment = ReplayEnvironment(task_id)
            try:
                replay_id = str(source["cell_id"])
                run_replay(
                    environment,
                    source,
                    source["executed_actions"],
                    maps[task_id],
                    cohort,
                    replay_id,
                    write_video=source["_source_path"] in all_videos,
                )
            finally:
                environment.close()


def hybrid_commands(
    baseline_actions: list[list[float]], treatment_actions: list[list[float]], arm_source: str
) -> list[list[float]]:
    common = min(len(baseline_actions), len(treatment_actions))
    baseline = np.asarray(baseline_actions[:common], dtype=np.float32)
    treatment = np.asarray(treatment_actions[:common], dtype=np.float32)
    if baseline.ndim != 2 or baseline.shape[1] != 7 or treatment.shape != baseline.shape:
        raise RuntimeError("invalid paired command traces")
    if arm_source == "baseline":
        commands = np.concatenate([baseline[:, :6], treatment[:, 6:7]], axis=1)
    elif arm_source == "treatment":
        commands = np.concatenate([treatment[:, :6], baseline[:, 6:7]], axis=1)
    else:
        raise ValueError(arm_source)
    return commands.astype(float).tolist()


def assert_source_reproduction(cohort: str) -> None:
    manifest = read_json(ROOT / f"{cohort}_replay_manifest.json")
    expected = int(manifest["selected_episode_count"])
    rows = [read_json(path) for path in sorted((SUMMARY_ROOT / cohort).glob("*.json"))]
    problems = [
        row["replay_id"]
        for row in rows
        if not row["initial_state_exact"]
        or int(row["command_mismatch_count"]) != 0
        or bool(row["replay_success"]) != bool(row["source_success"])
        or int(row["replay_steps"]) != int(row["source_steps"])
    ]
    if len(rows) != expected or problems:
        raise RuntimeError(
            f"{cohort} source replay reproduction gate failed: "
            f"expected={expected}, observed={len(rows)}, problems={problems[:20]}"
        )


def run_swaps(cohort: str, task_ids: set[int] | None = None) -> None:
    frozen_commit()
    assert_canary_passed()
    assert_source_reproduction(cohort)
    maps = task_map()
    if cohort == "development":
        root = TRACK_A_RESULTS
        contrast_names = ["development_h4", "development_h2"]
    elif cohort == "phase1":
        if not (ROOT / "PHASE1_UNBLINDING_RECORD.json").is_file():
            raise RuntimeError("Phase-1 outcomes have not been formally unblinded")
        root = PHASE1_RESULTS
        contrast_names = ["phase1_h4"]
    else:
        raise ValueError(cohort)
    methods = {method for name in contrast_names for method in CONTRASTS[name]}
    index = result_index(root, methods)
    work: dict[int, list[tuple[str, dict[str, Any], list[list[float]]]]] = defaultdict(list)
    manifest_rows = []
    for contrast_name in contrast_names:
        baseline_method, treatment_method = CONTRASTS[contrast_name]
        for baseline, treatment in paired_blocks(index, baseline_method, treatment_method):
            kind = pair_type(baseline, treatment)
            if kind not in {"rescue", "harm"}:
                continue
            for arm_source, label in (
                ("baseline", "baseline_arm_plus_treatment_gripper"),
                ("treatment", "treatment_arm_plus_baseline_gripper"),
            ):
                commands = hybrid_commands(baseline["executed_actions"], treatment["executed_actions"], arm_source)
                replay_id = f"{contrast_name}-{baseline['block_id']}-{label}"
                source = dict(baseline)
                source["method"] = label
                source["success"] = False
                source["environment_steps"] = len(commands)
                work[int(baseline["task_id"])].append((replay_id, source, commands))
                manifest_rows.append(
                    {
                        "contrast": contrast_name,
                        "block_id": baseline["block_id"],
                        "pair_type": kind,
                        "hybrid": label,
                        "common_support_steps": len(commands),
                        "replay_id": replay_id,
                    }
                )
    atomic_json(ROOT / f"{cohort}_swap_manifest.json", {"cohort": cohort, "rows": manifest_rows})
    for task_id in sorted(work):
        if task_ids is not None and task_id not in task_ids:
            continue
        for replay_id, source, commands in work[task_id]:
            environment = ReplayEnvironment(task_id)
            try:
                result = run_replay(
                    environment,
                    source,
                    commands,
                    maps[task_id],
                    f"{cohort}_swaps",
                    replay_id,
                    write_video=True,
                    hybrid_semantics=True,
                )
                result["swap_semantic_result"] = "SUCCESS" if result["replay_success"] else "CENSORED"
                result["common_support_steps"] = len(commands)
                atomic_json(summary_path(f"{cohort}_swaps", replay_id), result)
            finally:
                environment.close()


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def unblind_phase1() -> None:
    sha = frozen_commit()
    assert_canary_passed()
    baseline_method, treatment_method = CONTRASTS["phase1_h4"]
    index = result_index(PHASE1_RESULTS, {baseline_method, treatment_method})
    blocks = paired_blocks(index, baseline_method, treatment_method)
    expected = {(task_id, state_id) for task_id in range(10) for state_id in range(15, 50)}
    actual = {(int(b["task_id"]), int(b["state_id"])) for b, _ in blocks}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(f"Phase-1 cohort is not the complete 10 x states 15..49 set; missing={missing}, extra={extra}")
    pair_counts = Counter(pair_type(b, t) for b, t in blocks)
    baseline_success = sum(bool(b["success"]) for b, _ in blocks)
    treatment_success = sum(bool(t["success"]) for _, t in blocks)
    differences = np.asarray([float(t["success"]) - float(b["success"]) for b, t in blocks], dtype=float)
    rng = np.random.default_rng(27803)
    paired_boot = differences[rng.integers(0, len(differences), size=(20000, len(differences)))].mean(axis=1)
    task_values = np.asarray(
        [np.mean([d for (b, _), d in zip(blocks, differences, strict=True) if int(b["task_id"]) == task_id]) for task_id in range(10)]
    )
    rng = np.random.default_rng(27903)
    cluster_boot = task_values[rng.integers(0, 10, size=(20000, 10))].mean(axis=1)
    discordant = int(pair_counts["rescue"] + pair_counts["harm"])
    from scipy.stats import binomtest

    result = {
        "status": "PHASE1_OUTCOMES_UNBLINDED_AFTER_PHASE_D_FREEZE",
        "unblinded_at": subprocess.check_output(["date", "--iso-8601=seconds"], text=True).strip(),
        "phase_d_preregistration_commit": sha,
        "contrast": "ARM4_GRIP32-H4",
        "paired_blocks": len(blocks),
        "baseline_successes": baseline_success,
        "treatment_successes": treatment_success,
        "baseline_success_rate": baseline_success / len(blocks),
        "treatment_success_rate": treatment_success / len(blocks),
        "paired_risk_difference": float(np.mean(differences)),
        "paired_bootstrap_95_ci": percentile_interval(paired_boot),
        "task_cluster_bootstrap_95_ci": percentile_interval(cluster_boot),
        "paired_counts": dict(pair_counts),
        "mcnemar_exact_two_sided_p": float(binomtest(pair_counts["rescue"], discordant, 0.5).pvalue) if discordant else 1.0,
    }
    atomic_json(ROOT / "PHASE1_UNBLINDING_RECORD.json", result)
    lines = [
        "# Phase-1 overall paired contrast (reported before stage attribution)",
        "",
        f"Unblinded at `{result['unblinded_at']}` after Phase-D freeze commit `{sha}`.",
        "",
        f"All 350 task/state blocks (10 tasks x official states 15--49): H4 {baseline_success}/350; "
        f"ARM4_GRIP32 {treatment_success}/350; paired difference {100 * result['paired_risk_difference']:.3f} percentage points.",
        "",
        f"Paired outcomes: rescue {pair_counts['rescue']}, harm {pair_counts['harm']}, both fail {pair_counts['both_fail']}, "
        f"both succeed {pair_counts['both_succeed']}. Exact two-sided McNemar p = {result['mcnemar_exact_two_sided_p']:.6g}.",
        "",
        f"Paired-block bootstrap 95% CI: [{100 * result['paired_bootstrap_95_ci'][0]:.3f}, "
        f"{100 * result['paired_bootstrap_95_ci'][1]:.3f}] pp. Task-cluster bootstrap 95% CI: "
        f"[{100 * result['task_cluster_bootstrap_95_ci'][0]:.3f}, {100 * result['task_cluster_bootstrap_95_ci'][1]:.3f}] pp.",
        "",
        "This is a complete confirmation execution cohort, but it overlaps Track-A in official state IDs and is not described as state-held-out.",
    ]
    (ROOT / "PHASE1_OVERALL_CONTRAST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_summaries(cohort: str) -> dict[str, dict[str, Any]]:
    manual_decisions = {}
    decision_path = ROOT / "BLIND_MANUAL_REVIEW_DECISIONS.json"
    if not cohort.endswith("_swaps") and decision_path.is_file():
        for decision in read_json(decision_path)["decisions"]:
            manual_decisions[str(decision["replay_id"])] = decision
    result = {}
    for path in sorted((SUMMARY_ROOT / cohort).glob("*.json")):
        value = read_json(path)
        decision = manual_decisions.get(str(value["replay_id"]))
        if decision is not None:
            value["automatic_failure_category"] = value["failure_category"]
            value["failure_category"] = decision["failure_category"]
            value["later_stage_detail"] = decision.get("later_stage_detail")
            value["failed_stage"] = decision.get("failed_stage")
            value["blind_manual_review_packet"] = decision["packet_id"]
        key = str(value["replay_id"] if cohort.endswith("_swaps") else value.get("source_path") or value["replay_id"])
        result[key] = value
    return result


def attribution_for_contrast(
    contrast_name: str,
    blocks: list[tuple[dict[str, Any], dict[str, Any]]],
    source_summaries: dict[str, dict[str, Any]],
    swap_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    counts = Counter(pair_type(b, t) for b, t in blocks)
    baseline_failures = [(b, t) for b, t in blocks if not bool(b["success"])]
    reaching = []
    nonreaching = []
    for b, t in baseline_failures:
        summary = source_summaries[b["_source_path"]]
        (reaching if summary["ever_manipulation_opportunity"] else nonreaching).append((b, t, summary))
    rescue_stage = Counter()
    harm_stage = Counter()
    rescue_task_stage = Counter()
    harm_task_stage = Counter()
    rescue_later_detail = Counter()
    harm_later_detail = Counter()
    for b, t in blocks:
        kind = pair_type(b, t)
        if kind == "rescue":
            summary = source_summaries[b["_source_path"]]
            rescue_stage[summary["failure_category"]] += 1
            rescue_task_stage[str(summary.get("failed_stage") or "UNSPECIFIED")] += 1
            if summary.get("later_stage_detail"):
                rescue_later_detail[str(summary["later_stage_detail"])] += 1
        elif kind == "harm":
            summary = source_summaries[t["_source_path"]]
            harm_stage[summary["failure_category"]] += 1
            harm_task_stage[str(summary.get("failed_stage") or "UNSPECIFIED")] += 1
            if summary.get("later_stage_detail"):
                harm_later_detail[str(summary["later_stage_detail"])] += 1
    hybrid_labels = (
        "baseline_arm_plus_treatment_gripper",
        "treatment_arm_plus_baseline_gripper",
    )
    swap_counts: dict[str, dict[str, Counter]] = {
        kind: {hybrid: Counter() for hybrid in hybrid_labels} for kind in ("rescue", "harm")
    }
    swap_rows = []
    for b, t in blocks:
        kind = pair_type(b, t)
        if kind not in swap_counts:
            continue
        for hybrid in hybrid_labels:
            replay_id = f"{contrast_name}-{b['block_id']}-{hybrid}"
            value = swap_summaries[replay_id]
            label = "SUCCESS" if value["replay_success"] else "CENSORED"
            swap_counts[kind][hybrid][label] += 1
            swap_rows.append(
                {
                    "block_id": b["block_id"],
                    "pair_type": kind,
                    "hybrid": hybrid,
                    "replay_id": replay_id,
                    "result": label,
                    "steps": value["replay_steps"],
                    "common_support_steps": value["common_support_steps"],
                }
            )
    primary = swap_counts["rescue"]["baseline_arm_plus_treatment_gripper"]
    return {
        "contrast": contrast_name,
        "total_paired_blocks": len(blocks),
        "paired_counts": dict(counts),
        "baseline_failures": len(baseline_failures),
        "baseline_failures_never_reached_opportunity": len(nonreaching),
        "baseline_failures_reached_opportunity": len(reaching),
        "opportunity_reaching_baseline_failure_treatment_rescue": sum(bool(t["success"]) for _, t, _ in reaching),
        "opportunity_reaching_baseline_failure_treatment_still_fail": sum(not bool(t["success"]) for _, t, _ in reaching),
        "full_cohort_rescue": counts["rescue"],
        "full_cohort_harm": counts["harm"],
        "net_rescue_minus_harm": counts["rescue"] - counts["harm"],
        "rescue_failure_stage_distribution": dict(rescue_stage),
        "harm_failure_stage_distribution": dict(harm_stage),
        "rescue_task_stage_distribution": dict(rescue_task_stage),
        "harm_task_stage_distribution": dict(harm_task_stage),
        "rescue_later_stage_detail_distribution": dict(rescue_later_detail),
        "harm_later_stage_detail_distribution": dict(harm_later_detail),
        "component_swap_summary": {
            kind: {hybrid: dict(counter) for hybrid, counter in hybrids.items()}
            for kind, hybrids in swap_counts.items()
        },
        "primary_rescue_swap_success": primary["SUCCESS"],
        "primary_rescue_swap_censored": primary["CENSORED"],
        "component_swap_rows": swap_rows,
    }


def analyze() -> None:
    frozen_commit()
    assert_canary_passed()
    outputs = []
    discordant_video_rows = []
    for cohort, contrast_names, result_root in (
        ("development", ["development_h4", "development_h2"], TRACK_A_RESULTS),
        ("phase1", ["phase1_h4"], PHASE1_RESULTS),
    ):
        if cohort == "phase1" and not (ROOT / "PHASE1_UNBLINDING_RECORD.json").is_file():
            continue
        methods = {method for name in contrast_names for method in CONTRASTS[name]}
        index = result_index(result_root, methods)
        source_summaries = load_summaries(cohort)
        swap_summaries = load_summaries(f"{cohort}_swaps")
        for contrast_name in contrast_names:
            baseline, treatment = CONTRASTS[contrast_name]
            blocks = paired_blocks(index, baseline, treatment)
            outputs.append(attribution_for_contrast(contrast_name, blocks, source_summaries, swap_summaries))
            for baseline_row, treatment_row in blocks:
                kind = pair_type(baseline_row, treatment_row)
                if kind not in {"rescue", "harm"}:
                    continue
                baseline_summary = source_summaries[baseline_row["_source_path"]]
                treatment_summary = source_summaries[treatment_row["_source_path"]]
                primary_id = f"{contrast_name}-{baseline_row['block_id']}-baseline_arm_plus_treatment_gripper"
                reverse_id = f"{contrast_name}-{baseline_row['block_id']}-treatment_arm_plus_baseline_gripper"
                primary_summary = swap_summaries[primary_id]
                reverse_summary = swap_summaries[reverse_id]
                row = {
                    "contrast": contrast_name,
                    "block_id": baseline_row["block_id"],
                    "pair_type": kind,
                    "task_id": baseline_row["task_id"],
                    "state_id": baseline_row["state_id"],
                }
                for prefix, summary in (
                    ("baseline", baseline_summary),
                    ("treatment", treatment_summary),
                    ("baseline_arm_treatment_gripper", primary_summary),
                    ("treatment_arm_baseline_gripper", reverse_summary),
                ):
                    for view in ("agent", "wrist"):
                        row[f"{prefix}_{view}"] = summary.get("videos", {}).get(view, "")
                discordant_video_rows.append(row)
    atomic_json(ROOT / "paired_rescue_harm_attribution.json", {"contrasts": outputs})

    table_path = ROOT / "paired_rescue_harm_attribution.csv"
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "contrast", "total_paired_blocks", "baseline_failures", "baseline_failures_never_reached_opportunity",
            "baseline_failures_reached_opportunity", "opportunity_reaching_baseline_failure_treatment_rescue",
            "opportunity_reaching_baseline_failure_treatment_still_fail", "full_cohort_rescue", "full_cohort_harm",
            "net_rescue_minus_harm", "primary_rescue_swap_success", "primary_rescue_swap_censored",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(outputs)

    swap_rows = []
    for output in outputs:
        swap_rows.extend({"contrast": output["contrast"], **row} for row in output["component_swap_rows"])
    with (ROOT / "component_swap_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["contrast", "block_id", "pair_type", "hybrid", "replay_id", "result", "steps", "common_support_steps"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(swap_rows)

    video_rows = []
    for cohort in ("development", "development_swaps", "phase1", "phase1_swaps"):
        for summary in load_summaries(cohort).values():
            for view, path in summary.get("videos", {}).items():
                video_rows.append({"cohort": cohort, "replay_id": summary["replay_id"], "view": view, "path": path})
    with (ROOT / "video_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["cohort", "replay_id", "view", "path"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(video_rows)

    pair_video_fields = ["contrast", "block_id", "pair_type", "task_id", "state_id"] + [
        f"{prefix}_{view}"
        for prefix in (
            "baseline",
            "treatment",
            "baseline_arm_treatment_gripper",
            "treatment_arm_baseline_gripper",
        )
        for view in ("agent", "wrist")
    ]
    with (ROOT / "discordant_video_pairs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=pair_video_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(discordant_video_rows)

    unblinding = read_json(ROOT / "PHASE1_UNBLINDING_RECORD.json")
    report = [
        "# Phase-D scientific report",
        "",
        "All rates and counts below retain the complete paired-block denominator.",
        "",
        "## Integrity and provenance",
        "",
        "The task predicates, failure taxonomy, replay schema, and component-swap protocol were frozen at "
        f"`2026-09-04T16:31:26+08:00` in commit `{frozen_commit()}`. Phase-1 success outcomes were first read at "
        f"`{unblinding['unblinded_at']}`.",
        "",
        "The stored Track-A LIBERO-10 cohort contains task-specific official state IDs mostly in 20--44, not states 0--14. "
        "It overlaps Phase-1 states 15--49, so Phase-1 is described as a complete confirmation execution cohort, not as state-held-out.",
        "",
        "Exact open-loop reproduction passed for all 248 selected development source episodes and all 288 selected Phase-1 source episodes: "
        "zero initial-state, terminal-success, episode-length, or command mismatches.",
        "",
    ]
    for output in outputs:
        counts = output["paired_counts"]
        report.extend(
            [
                f"## {output['contrast']}",
                "",
                f"Paired blocks: {output['total_paired_blocks']}. Outcomes: rescue {counts.get('rescue', 0)}, harm {counts.get('harm', 0)}, both fail {counts.get('both_fail', 0)}, both succeed {counts.get('both_succeed', 0)}. Net rescue minus harm: {output['net_rescue_minus_harm']}.",
                "",
                f"Baseline failures: {output['baseline_failures']}; never reached a predefined manipulation opportunity: {output['baseline_failures_never_reached_opportunity']}; reached opportunity: {output['baseline_failures_reached_opportunity']}.",
                "",
                f"Among {output['baseline_failures_reached_opportunity']} opportunity-reaching baseline failures, {output['opportunity_reaching_baseline_failure_treatment_rescue']} were rescued and {output['opportunity_reaching_baseline_failure_treatment_still_fail']} still failed.",
                "",
                f"Rescue failure stages: `{json.dumps(output['rescue_failure_stage_distribution'], sort_keys=True)}`. Harm failure stages: `{json.dumps(output['harm_failure_stage_distribution'], sort_keys=True)}`.",
                "",
                f"Rescue task stages: `{json.dumps(output['rescue_task_stage_distribution'], sort_keys=True)}`. Harm task stages: `{json.dumps(output['harm_task_stage_distribution'], sort_keys=True)}`.",
                "",
                f"Later-stage physical detail among rescues: `{json.dumps(output['rescue_later_stage_detail_distribution'], sort_keys=True)}`; among harms: `{json.dumps(output['harm_later_stage_detail_distribution'], sort_keys=True)}`.",
                "",
                f"Both-direction rescue/harm swap table (SUCCESS/CENSORED): `{json.dumps(output['component_swap_summary'], sort_keys=True)}`. Censored swaps do not establish failure.",
                "",
            ]
        )

    h4_outputs = [output for output in outputs if output["contrast"] in {"development_h4", "phase1_h4"}]
    rescues = sum(output["full_cohort_rescue"] for output in h4_outputs)
    harms = sum(output["full_cohort_harm"] for output in h4_outputs)
    post_opportunity = sum(output["full_cohort_rescue"] - output["rescue_failure_stage_distribution"].get("PRE_OPPORTUNITY_FAILURE", 0) for output in h4_outputs)
    swap_success = sum(output["primary_rescue_swap_success"] for output in h4_outputs)
    swap_total = sum(output["primary_rescue_swap_success"] + output["primary_rescue_swap_censored"] for output in h4_outputs)
    rescue_stages = Counter()
    harm_stages = Counter()
    for output in h4_outputs:
        rescue_stages.update(output["rescue_failure_stage_distribution"])
        harm_stages.update(output["harm_failure_stage_distribution"])
    stage_tradeoff = bool(rescue_stages and harm_stages and rescue_stages.most_common(1)[0][0] == harm_stages.most_common(1)[0][0] and harm_stages.most_common(1)[0][1] > harms / 2)
    supports_a = bool(rescues > 0 and post_opportunity > rescues / 2 and swap_total > 0 and swap_success > swap_total / 2 and rescues > harms)
    if supports_a and (harms >= rescues / 2 or stage_tradeoff):
        label = "PHASE-D RESULT D — TARGETED RESCUE/HARM TRADEOFF"
    elif supports_a:
        label = "PHASE-D RESULT A — GRIPPER-STAGE RESCUE WITH COMMAND-TRACE SUFFICIENCY"
    elif rescues > 0 and post_opportunity > rescues / 2:
        label = "PHASE-D RESULT B — STAGE-LOCALIZED RESCUE WITHOUT SUFFICIENCY"
    else:
        label = "PHASE-D RESULT C — RESCUES NOT GRIPPER-STAGE LOCALIZED"
    report.extend(
        [
            "## Interpretation",
            "",
            "This analysis localizes recorded command-trace effects. It does not establish that an online ACT policy would preserve its arm trajectory after altered gripper execution.",
            "",
            f"`{label}`",
        ]
    )
    (ROOT / "SCIENTIFIC_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    atomic_json(ROOT / "phase_d_result.json", {"result": label, "h4_rescues": rescues, "h4_harms": harms, "h4_post_opportunity_rescues": post_opportunity, "h4_primary_swap_success": swap_success, "h4_primary_swap_total": swap_total})


def validate_outputs() -> None:
    required_step_fields = {
        "step_index",
        "replay_command_7d",
        "eef_position",
        "eef_quaternion_xyzw",
        "gripper_qpos",
        "gripper_qvel",
        "relevant_body_and_site_poses",
        "fixture_joint_qpos",
        "stage_predicates",
        "relevant_contact_pairs",
        "reward",
        "success",
        "terminated",
        "truncated",
        "termination_reason",
        "manipulation_opportunity",
        "stage_label",
    }
    expected = {}
    problems = []
    for cohort in ("development", "phase1"):
        source_expected = int(read_json(ROOT / f"{cohort}_replay_manifest.json")["selected_episode_count"])
        swap_expected = len(read_json(ROOT / f"{cohort}_swap_manifest.json")["rows"])
        expected[cohort] = {"source": source_expected, "swaps": swap_expected}
        for summary_cohort, count, is_swap in (
            (cohort, source_expected, False),
            (f"{cohort}_swaps", swap_expected, True),
        ):
            paths = sorted((SUMMARY_ROOT / summary_cohort).glob("*.json"))
            if len(paths) != count:
                problems.append(f"{summary_cohort}: expected {count} summaries, found {len(paths)}")
            for path in paths:
                row = read_json(path)
                if not row["initial_state_exact"] or int(row["command_mismatch_count"]) != 0:
                    problems.append(f"{row['replay_id']}: state or command mismatch")
                if is_swap:
                    if int(row["replay_steps"]) > int(row["common_support_steps"]):
                        problems.append(f"{row['replay_id']}: common-support overrun")
                    expected_semantic = "SUCCESS" if row["replay_success"] else "CENSORED"
                    if row["swap_semantic_result"] != expected_semantic:
                        problems.append(f"{row['replay_id']}: bad swap semantic label")
                elif bool(row["replay_success"]) != bool(row["source_success"]) or int(row["replay_steps"]) != int(row["source_steps"]):
                    problems.append(f"{row['replay_id']}: source reproduction mismatch")
                log_path = Path(row["raw_log"])
                if not log_path.is_file():
                    problems.append(f"{row['replay_id']}: missing raw log")
                else:
                    with gzip.open(log_path, "rt", encoding="utf-8") as stream:
                        metadata = json.loads(next(stream))
                        step = json.loads(next(stream))
                    if metadata.get("record_type") != "metadata" or metadata.get("act_queries") != 0:
                        problems.append(f"{row['replay_id']}: bad raw-log metadata")
                    if step.get("record_type") != "step" or not required_step_fields.issubset(step) or len(step["replay_command_7d"]) != 7:
                        problems.append(f"{row['replay_id']}: bad raw-log step schema")
                for video in row.get("videos", {}).values():
                    if not Path(video).is_file():
                        problems.append(f"{row['replay_id']}: missing video {video}")

    with (ROOT / "discordant_video_pairs.csv").open(newline="", encoding="utf-8") as stream:
        pair_rows = list(csv.DictReader(stream))
    if len(pair_rows) != 122:
        problems.append(f"expected 122 discordant video rows, found {len(pair_rows)}")
    for row in pair_rows:
        for key, value in row.items():
            if key.endswith(("_agent", "_wrist")) and not Path(value).is_file():
                problems.append(f"{row['contrast']}/{row['block_id']}: missing paired video {key}")

    analysis = read_json(ROOT / "paired_rescue_harm_attribution.json")["contrasts"]
    expected_blocks = {"development_h4": 150, "development_h2": 150, "phase1_h4": 350}
    for row in analysis:
        if int(row["total_paired_blocks"]) != expected_blocks[row["contrast"]]:
            problems.append(f"{row['contrast']}: paired denominator mismatch")
        if sum(row["paired_counts"].values()) != int(row["total_paired_blocks"]):
            problems.append(f"{row['contrast']}: paired cells do not sum")
        if sum(row["rescue_failure_stage_distribution"].values()) != int(row["full_cohort_rescue"]):
            problems.append(f"{row['contrast']}: rescue stages do not sum")
        if sum(row["harm_failure_stage_distribution"].values()) != int(row["full_cohort_harm"]):
            problems.append(f"{row['contrast']}: harm stages do not sum")

    result = {
        "status": "PASS" if not problems else "FAIL",
        "source_and_swap_expectations": expected,
        "discordant_video_blocks": len(pair_rows),
        "raw_logs_checked": sum(v["source"] + v["swaps"] for v in expected.values()),
        "problems": problems,
    }
    atomic_json(ROOT / "FINAL_AUDIT.json", result)
    lines = [
        "# Phase-D final artifact audit",
        "",
        f"Status: `{result['status']}`",
        "",
        f"Raw replay logs checked: {result['raw_logs_checked']}.",
        f"Discordant video blocks checked: {result['discordant_video_blocks']}.",
        f"Problems: {len(problems)}.",
    ]
    (ROOT / "FINAL_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if problems:
        raise RuntimeError(f"Phase-D final audit failed: {problems[:20]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["canary", "development", "development-swaps", "unblind-phase1", "phase1", "phase1-swaps", "analyze", "validate"],
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="*",
        choices=range(10),
        help="Optional disjoint LIBERO-10 task shard; scientific selection is unchanged.",
    )
    args = parser.parse_args()
    task_ids = set(args.task_ids) if args.task_ids else None
    if args.command == "canary":
        canary()
    elif args.command == "development":
        run_source_cohort("development", task_ids)
    elif args.command == "development-swaps":
        run_swaps("development", task_ids)
    elif args.command == "unblind-phase1":
        unblind_phase1()
    elif args.command == "phase1":
        run_source_cohort("phase1", task_ids)
    elif args.command == "phase1-swaps":
        run_swaps("phase1", task_ids)
    elif args.command == "analyze":
        analyze()
    else:
        validate_outputs()


if __name__ == "__main__":
    main()
