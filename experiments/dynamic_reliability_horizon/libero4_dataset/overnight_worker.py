#!/usr/bin/env python3
"""Resumable overnight orchestrator for the complete LIBERO-4 foundation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dataset_common import DATASET_REVISION, atomic_write_json


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PYTHON = Path("/home/thor/projects/upstreams/lerobot-env/bin/python")
DATASET_ROOT = Path("/home/thor/datasets/lerobot_libero_a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4")
CORPUS_DIR = HERE
POLICY_CACHE_DIR = Path("/home/thor/large_artifacts/one-clock/libero4/policy_cache/smolvla_libero")
LABEL_DIR = Path("/home/thor/large_artifacts/one-clock/libero4/reliability_labels/smolvla_libero")
PROGRESS_PATH = HERE / "progress.json"


def save_progress(step: str, status: str, **extra: object) -> None:
    current = {}
    if PROGRESS_PATH.is_file():
        current = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    current.update({"dataset_revision": DATASET_REVISION, "worker_pid": os.getpid(), "step": step, "status": status, "updated_unix": int(time.time())})
    current.update(extra)
    atomic_write_json(PROGRESS_PATH, current)


def run(command: list[str], *, env: dict[str, str] | None = None, timeout: int | None = None) -> int:
    print("RUN", " ".join(command), flush=True)
    try:
        completed = subprocess.run(command, cwd=REPO_ROOT, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        print("TIMEOUT", " ".join(command), flush=True)
        return 124
    print("RETURN", completed.returncode, flush=True)
    return completed.returncode


def main() -> int:
    if not PYTHON.is_file():
        save_progress("startup", "blocked", blocker=f"missing environment: {PYTHON}")
        return 2
    save_progress("download", "running", dataset_root=str(DATASET_ROOT), dataset_root_bytes=None)
    download_code = run([str(PYTHON), str(HERE / "download_pinned_dataset.py"), "--dataset-root", str(DATASET_ROOT), "--include-videos"])
    save_progress("download", "completed" if download_code == 0 else "completed_with_failures", downloader_return_code=download_code)

    try:
        save_progress("corpus", "running")
        corpus_code = run([str(PYTHON), str(HERE / "build_libero4_corpus.py"), "--dataset-root", str(DATASET_ROOT), "--output-dir", str(CORPUS_DIR)])
        if corpus_code != 0:
            raise RuntimeError(f"corpus builder returned {corpus_code}")
        save_progress("corpus", "completed")
    except Exception as exc:
        save_progress("corpus", "blocked", blocker=repr(exc))
        return 2

    test_code = run([str(PYTHON), "-m", "pytest", "-q", str(HERE / "tests")])
    save_progress("tests", "completed" if test_code == 0 else "failed", return_code=test_code)

    compat_output = HERE / "smolvla_compatibility.json"
    save_progress("smolvla_compatibility", "running", output=str(compat_output))
    compat_env = dict(os.environ)
    # The Thor environment has repeatedly failed the Xet CAS TLS handshake.
    # Keep this bounded compatibility attempt on the regular Hub transport.
    compat_env["HF_HUB_DISABLE_XET"] = "1"
    compat_code = run([
        str(PYTHON), str(HERE / "check_smolvla_compatibility.py"),
        "--dataset-root", str(DATASET_ROOT), "--corpus-dir", str(CORPUS_DIR),
        "--output", str(compat_output), "--device", "cuda",
    ], env=compat_env, timeout=20 * 60)
    compatibility = json.loads(compat_output.read_text(encoding="utf-8")) if compat_output.is_file() else {"status": "missing"}
    feasible = compatibility.get("status") == "feasible" and compat_code == 0
    save_progress("smolvla_compatibility", "feasible" if feasible else "blocked", return_code=compat_code, compatibility_status=compatibility.get("status"), blocker=compatibility.get("blocker"))
    if not feasible:
        save_progress("handoff", "ready", policy_cache_status="not_started", handoff=str(HERE / "handoff_to_5080.md"))
        return 0

    POLICY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_env = dict(os.environ)
    cache_env.update({"ONECLOCK_SMOLVLA_DATASET_ROOT": str(DATASET_ROOT), "ONECLOCK_SMOLVLA_DEVICE": "cuda", "HF_HUB_DISABLE_XET": "1"})
    save_progress("smolvla_cache", "running", policy_cache_dir=str(POLICY_CACHE_DIR))
    cache_code = run([
        str(PYTHON), str(HERE / "build_policy_cache.py"),
        "--dataset-root", str(DATASET_ROOT), "--corpus-dir", str(CORPUS_DIR),
        "--output-dir", str(POLICY_CACHE_DIR),
        "--adapter", "smolvla_adapter:make_adapter", "--episodes-per-shard", "20",
    ], env=cache_env)
    cache_manifest = POLICY_CACHE_DIR / "manifest.json"
    save_progress("smolvla_cache", "completed" if cache_code == 0 else "completed_with_failures", return_code=cache_code, policy_cache_manifest=str(cache_manifest))
    if cache_code == 0:
        save_progress("reliability_labels", "running", label_dir=str(LABEL_DIR))
        label_code = run([
            str(PYTHON), str(HERE / "build_reliability_targets.py"),
            "--corpus-dir", str(CORPUS_DIR), "--policy-cache", str(POLICY_CACHE_DIR),
            "--output-dir", str(LABEL_DIR),
            "--action-std", "0.335523718", "0.378446991", "0.444728601", "0.039243541", "0.063392964", "0.077970275", "0.998767139",
        ])
        save_progress("reliability_labels", "completed" if label_code == 0 else "failed", return_code=label_code)
    save_progress("handoff", "ready", policy_cache_dir=str(POLICY_CACHE_DIR), label_dir=str(LABEL_DIR))
    return 0


if __name__ == "__main__":
    sys.exit(main())
