#!/usr/bin/env python3
"""Minimal detached, resumable ACT train -> native-eval supervisor."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from run_act_task_queue import load_tasks, latest_checkpoint


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = Path("/home/wjq/research-assets/datasets/HuggingFaceVLA_libero")
DEFAULT_OUTPUT_ROOT = ROOT / "act_final"
DEFAULT_STATE = ROOT / "overnight_state.json"
DEFAULT_LOG_ROOT = ROOT / "logs"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def alive(pid: int | None, tag: str) -> bool:
    if not pid or pid <= 0:
        return False


def find_live_job(tag: str) -> tuple[int, str | None] | None:
    """Find a LeRobot child by its explicit job name if state was stale/lost."""
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            command = proc.joinpath("cmdline").read_bytes().decode(errors="ignore")
            if f"--job_name={tag}" not in command or "lerobot-" not in command:
                continue
            environment = proc.joinpath("environ").read_bytes().decode(errors="ignore")
            gpu = next((x.split("=", 1)[1] for x in environment.split("\0") if x.startswith("CUDA_VISIBLE_DEVICES=")), None)
            return int(proc.name), gpu
        except (OSError, ValueError):
            continue
    return None
    try:
        os.kill(pid, 0)
        command = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore")
        return tag in command or "lerobot-" in command
    except (OSError, FileNotFoundError):
        return False


class AdoptedProcess:
    """Poll a child that was started by an earlier supervisor instance."""

    def __init__(self, pid: int, tag: str):
        self.pid = pid
        self.tag = tag

    @property
    def pid(self) -> int:
        return self._pid

    @pid.setter
    def pid(self, value: int) -> None:
        self._pid = value

    def poll(self) -> int | None:
        return None if alive(self.pid, self.tag) else 0


def complete_artifacts(task_dir: Path) -> bool:
    checkpoint = task_dir / "checkpoints/100000/pretrained_model"
    evaluation = task_dir / "eval10/eval_info.json"
    return all((checkpoint / name).is_file() for name in ("config.json", "model.safetensors")) and evaluation.is_file()


def checkpoint_complete(task_dir: Path) -> bool:
    checkpoint = task_dir / "checkpoints/100000/pretrained_model"
    return all((checkpoint / name).is_file() for name in ("config.json", "model.safetensors", "train_config.json"))


def task_args(task: dict, args: argparse.Namespace, task_dir: Path) -> list[str]:
    tag = f"{task['suite']}_task{task['task_id']}"
    return [
        str(args.train_bin),
        "--dataset.repo_id=HuggingFaceVLA/libero",
        f"--dataset.root={args.dataset_root}",
        f"--dataset.episodes={json.dumps(task['episodes'])}",
        "--policy.type=act",
        "--policy.device=cuda",
        "--policy.use_amp=false",
        "--policy.push_to_hub=false",
        "--env.type=libero",
        f"--env.task={task['suite']}",
        f"--env.task_ids=[{task['task_id']}]",
        "--env.obs_type=pixels_agent_pos",
        "--env.control_mode=relative",
        "--env.camera_name=agentview_image,robot0_eye_in_hand_image",
        "--env.init_states=true",
        "--batch_size=8",
        "--num_workers=2",
        "--steps=100000",
        "--eval_freq=0",
        "--log_freq=200",
        "--save_freq=20000",
        f"--output_dir={task_dir}",
        f"--job_name={tag}",
        "--seed=1000",
    ]


def eval_args(task: dict, args: argparse.Namespace, checkpoint: Path, task_dir: Path) -> list[str]:
    tag = f"{task['suite']}_task{task['task_id']}"
    return [
        str(args.eval_bin),
        f"--policy.path={checkpoint}",
        "--policy.device=cuda",
        "--policy.use_amp=false",
        "--env.type=libero",
        f"--env.task={task['suite']}",
        f"--env.task_ids=[{task['task_id']}]",
        "--env.camera_name=agentview_image,robot0_eye_in_hand_image",
        "--env.init_states=true",
        "--env.obs_type=pixels_agent_pos",
        "--env.control_mode=relative",
        "--eval.n_episodes=10",
        "--eval.batch_size=1",
        "--eval.use_async_envs=false",
        f"--output_dir={task_dir / 'eval10'}",
        f"--job_name={tag}_eval10",
        "--seed=1000",
    ]


def prepare_state(tasks: list[dict], state_path: Path, output_root: Path) -> dict:
    previous = json.loads(state_path.read_text()) if state_path.is_file() else {}
    jobs = previous.get("jobs", {})
    for task in tasks:
        tag = f"{task['suite']}_task{task['task_id']}"
        job = jobs.setdefault(tag, {
            "tag": tag,
            "suite": task["suite"],
            "task_id": task["task_id"],
            "task_name": task["task_name"],
            "state": "pending",
            "history": [],
        })
        task_dir = output_root / tag
        if complete_artifacts(task_dir):
            job.update({"state": "completed", "pid": None, "ended_at": job.get("ended_at", now()), "reason": "artifacts_verified"})
        elif job.get("state") == "completed":
            job.update({"state": "pending", "reason": "completion_artifacts_missing"})
    state = {
        "version": 1,
        "supervisor": previous.get("supervisor", {}),
        "jobs": jobs,
        "updated_at": now(),
    }
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--train-bin", type=Path, default=Path(sys.executable).with_name("lerobot-train"))
    parser.add_argument("--eval-bin", type=Path, default=Path(sys.executable).with_name("lerobot-eval"))
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2"])
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.output_root = args.output_root.resolve()
    args.state = args.state.resolve()
    args.log_root = args.log_root.resolve()
    tasks = load_tasks(args.dataset_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    atomic_write(args.output_root / "act_task_manifest.json", tasks)
    state = prepare_state(tasks, args.state, args.output_root)
    state["supervisor"] = {"pid": os.getpid(), "started_at": state.get("supervisor", {}).get("started_at", now())}
    atomic_write(args.state, state)
    if args.dry_run:
        print(json.dumps({"jobs": len(tasks), "gpus": args.gpus, "state": str(args.state)}, indent=2))
        return

    task_by_tag = {f"{task['suite']}_task{task['task_id']}": task for task in tasks}
    running: dict[str, tuple[str, object, object]] = {}
    # Adopt live jobs recorded by a previous supervisor rather than launching
    # a duplicate. A vanished process is recorded as failed unless its expected
    # artifacts already make the stage complete.
    for tag, job in state["jobs"].items():
        if job.get("state") == "pending":
            found = find_live_job(tag)
            if found:
                pid, gpu = found
                job.update({"state": "running", "stage": "eval" if "lerobot-eval" in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore") else "train", "gpu": gpu, "pid": pid, "started_at": job.get("started_at", now()), "reason": "adopted_from_process_scan"})
        if job.get("state") != "running":
            continue
        pid = job.get("pid")
        if alive(pid, tag):
            log_handle = (args.log_root / f"{tag}.{job.get('stage', 'unknown')}.log").open("a")
            running[str(job.get("gpu"))] = (tag, AdoptedProcess(int(pid), tag), log_handle)
        else:
            task_dir = args.output_root / tag
            if job.get("stage") == "train" and checkpoint_complete(task_dir):
                job.update({"state": "pending", "next_stage": "eval", "reason": "adopted_train_artifact_verified"})
            elif job.get("stage") == "eval" and (task_dir / "eval10/eval_info.json").is_file():
                job.update({"state": "completed", "reason": "adopted_eval_artifact_verified"})
            else:
                job.update({"state": "failed", "pid": None, "ended_at": now(), "exit_code": 1, "reason": "supervisor_restart_process_missing"})
    while True:
        for gpu, (tag, process, log_handle) in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            log_handle.close()
            job = state["jobs"][tag]
            stage = job["stage"]
            history = job.setdefault("history", [])
            if history:
                history[-1].update({"ended_at": now(), "exit_code": returncode})
            job.update({"pid": None, "gpu": gpu, "ended_at": now(), "exit_code": returncode})
            task_dir = args.output_root / tag
            if stage == "train":
                if checkpoint_complete(task_dir) and (returncode == 0 or isinstance(process, AdoptedProcess)):
                    job.update({"state": "pending", "next_stage": "eval", "reason": "checkpoint_verified"})
                else:
                    job.update({"state": "failed", "reason": "training_exit_or_checkpoint_validation", "failure_stage": "train"})
            else:
                if (task_dir / "eval10/eval_info.json").is_file() and (returncode == 0 or isinstance(process, AdoptedProcess)):
                    job.update({"state": "completed", "reason": "evaluation_artifact_verified", "next_stage": None})
                else:
                    job.update({"state": "failed", "reason": "evaluation_exit_or_artifact_missing", "failure_stage": "eval"})
            del running[gpu]

        for gpu in args.gpus:
            if gpu in running:
                continue
            candidate = None
            for tag, job in state["jobs"].items():
                if job.get("state") != "pending" or tag in {x[0] for x in running.values()}:
                    continue
                candidate = (tag, job)
                break
            if candidate is None:
                continue
            tag, job = candidate
            task = task_by_tag[tag]
            task_dir = args.output_root / tag
            checkpoint = task_dir / "checkpoints/100000/pretrained_model"
            if job.get("next_stage") == "eval" or checkpoint_complete(task_dir):
                stage = "eval"
                command = eval_args(task, args, checkpoint, task_dir)
                log_path = args.log_root / f"{tag}.eval.log"
            else:
                stage = "train"
                partial = latest_checkpoint(task_dir)
                if partial and (partial.parent / "training_state").is_dir():
                    command = [str(args.train_bin), f"--config_path={partial / 'train_config.json'}", "--resume=true"]
                else:
                    command = task_args(task, args, task_dir)
                log_path = args.log_root / f"{tag}.train.log"
            log_handle = log_path.open("a")
            environment = {**os.environ, "MUJOCO_GL": "egl", "CUDA_VISIBLE_DEVICES": gpu}
            process = subprocess.Popen(command, env=environment, stdout=log_handle, stderr=subprocess.STDOUT)
            job.update({"state": "running", "stage": stage, "gpu": gpu, "pid": process.pid, "started_at": now(), "command": command, "log": str(log_path)})
            job.setdefault("history", []).append({"stage": stage, "gpu": gpu, "pid": process.pid, "started_at": job["started_at"], "log": str(log_path)})
            running[gpu] = (tag, process, log_handle)

        state["updated_at"] = now()
        state["supervisor"]["running_gpus"] = sorted(running)
        atomic_write(args.state, state)
        if not running and not any(job.get("state") == "pending" for job in state["jobs"].values()):
            break
        time.sleep(args.poll_seconds)
    state["supervisor"]["ended_at"] = now()
    state["supervisor"]["running_gpus"] = []
    atomic_write(args.state, state)


if __name__ == "__main__":
    main()
