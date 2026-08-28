#!/usr/bin/env python3
"""Small resumable queue for task-specific LeRobot ACT LIBERO baselines."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import queue
import subprocess
import threading
from pathlib import Path

import pyarrow.parquet as pq

from libero.libero import benchmark


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def load_tasks(dataset_root: Path) -> list[dict]:
    task_table = pq.read_table(dataset_root / "meta/tasks.parquet").to_pandas()
    episode_rows = pq.read_table(
        dataset_root / "meta/episodes/chunk-000/file-000.parquet",
        columns=["episode_index", "tasks"],
    ).to_pylist()
    # LIBERO benchmark task ids are suite-local and do not match the public
    # dataset's global task_index order. Match by the task identity carried in
    # the language description, then use that task's dataset index.
    dataset_task_index = {
        str(name).strip().lower(): int(row.task_index)
        for name, row in task_table.iterrows()
    }
    episodes_by_name: dict[str, list[int]] = {}
    for row in episode_rows:
        episodes_by_name.setdefault(str(row["tasks"][0]).strip().lower(), []).append(
            int(row["episode_index"])
        )

    tasks = []
    for suite in SUITES:
        suite_tasks = benchmark.get_benchmark_dict()[suite]().tasks
        for local_id, suite_task in enumerate(suite_tasks):
            name = suite_task.language
            name_key = name.strip().lower()
            if name_key not in dataset_task_index:
                raise KeyError(f"LIBERO task is absent from dataset metadata: {name}")
            global_id = dataset_task_index[name_key]
            episodes = episodes_by_name.get(name_key, [])
            if not episodes:
                raise ValueError(f"LIBERO task has no dataset episodes: {name}")
            tasks.append(
                {
                    "suite": suite,
                    "task_id": local_id,
                    "dataset_task_index": global_id,
                    "task_name": name,
                    "episodes": episodes,
                }
            )
    return tasks


def append_status(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def latest_checkpoint(task_dir: Path) -> Path | None:
    checkpoints = sorted(
        (p for p in (task_dir / "checkpoints").glob("*") if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    for checkpoint in reversed(checkpoints):
        pretrained = checkpoint / "pretrained_model"
        if (pretrained / "config.json").is_file() and (pretrained / "model.safetensors").is_file():
            return pretrained
    return None


def run_one(task: dict, gpu: str, args: argparse.Namespace, status_path: Path) -> None:
    tag = f"{task['suite']}_task{task['task_id']}"
    task_dir = args.output_root / tag
    task_dir.mkdir(parents=True, exist_ok=True)
    train_log = task_dir / "train.log"
    eval_dir = task_dir / "eval10"
    eval_log = task_dir / "eval10.log"
    env = {**os.environ, "MUJOCO_GL": "egl", "CUDA_VISIBLE_DEVICES": gpu}

    checkpoint = latest_checkpoint(task_dir)
    if checkpoint and checkpoint.parent.name == "100000":
        train_rc = 0
        train_mode = "already_complete"
    else:
        train_args = [
            str(args.train_bin),
            f"--dataset.repo_id=HuggingFaceVLA/libero",
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
        if checkpoint and (checkpoint.parent / "training_state").is_dir():
            train_args = [
                str(args.train_bin),
                f"--config_path={checkpoint / 'train_config.json'}",
                "--resume=true",
            ]
            train_mode = f"resume_{checkpoint.parent.name}"
        else:
            train_mode = "new"
        append_status(status_path, {"tag": tag, "state": "training", "gpu": gpu, "mode": train_mode})
        with train_log.open("a") as log:
            train_rc = subprocess.run(train_args, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        if train_rc != 0:
            append_status(status_path, {"tag": tag, "state": "train_failed", "gpu": gpu, "returncode": train_rc})
            return
        checkpoint = latest_checkpoint(task_dir)

    if checkpoint is None:
        append_status(status_path, {"tag": tag, "state": "train_failed", "gpu": gpu, "reason": "no checkpoint"})
        return

    append_status(status_path, {"tag": tag, "state": "evaluating", "gpu": gpu, "checkpoint": str(checkpoint)})
    eval_args = [
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
        f"--output_dir={eval_dir}",
        f"--job_name={tag}_eval10",
        "--seed=1000",
    ]
    with eval_log.open("a") as log:
        eval_rc = subprocess.run(eval_args, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    append_status(
        status_path,
        {
            "tag": tag,
            "state": "complete" if eval_rc == 0 else "eval_failed",
            "gpu": gpu,
            "checkpoint": str(checkpoint),
            "returncode": eval_rc,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-bin", type=Path, required=True)
    parser.add_argument("--eval-bin", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2"])
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    data_files = sorted((dataset_root / "data").glob("*/*.parquet"))
    info = json.loads((dataset_root / "meta/info.json").read_text())
    # This public dataset currently has 377 parquet shards; the metadata does
    # not encode the shard count because shard sizes are variable.
    expected_files = 377
    if len(data_files) < expected_files:
        raise SystemExit(f"dataset incomplete: found {len(data_files)} data files, expected at least {expected_files}")

    tasks = load_tasks(dataset_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "act_task_manifest.json").write_text(json.dumps(tasks, indent=2) + "\n")
    status_path = args.output_root / "queue_status.jsonl"

    work = queue.Queue()
    for task in tasks:
        work.put(task)

    def worker(gpu: str) -> None:
        while True:
            try:
                task = work.get_nowait()
            except queue.Empty:
                return
            try:
                run_one(task, gpu, args, status_path)
            finally:
                work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in args.gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
