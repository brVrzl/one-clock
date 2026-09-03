#!/usr/bin/env python3
"""Run the frozen R1D queue with the validated LeRobot 0.4.4 package path."""

from __future__ import annotations

import importlib.metadata
import json
import runpy
import sys
from pathlib import Path


EXPECTED_PYTHON = Path("/home/wjq/workspace/venvs/libero_act/bin/python")
EXPECTED_PACKAGE = Path(
    "/home/wjq/workspace/venvs/libero_act/lib/python3.12/site-packages/lerobot"
)
RUN_QUEUE = Path(__file__).resolve().with_name("run_queue.py")


def preload_validated_runtime() -> None:
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError(f"unexpected interpreter: {sys.executable}")
    if importlib.metadata.version("lerobot") != "0.4.4":
        raise RuntimeError("R1D repair requires installed LeRobot 0.4.4")

    # Import only the package root before the frozen queue prepends the newer
    # checkout. Python then resolves all LeRobot submodules from this package's
    # validated __path__, exactly as it did for R1A--R1C.
    import lerobot

    package_file = Path(lerobot.__file__).resolve()
    if EXPECTED_PACKAGE.resolve() not in package_file.parents:
        raise RuntimeError(f"unexpected LeRobot package: {package_file}")
    print(
        json.dumps(
            {
                "r1d_runtime_repair": "PRELOADED_INSTALLED_LEROBOT_PACKAGE_ROOT",
                "python": sys.executable,
                "lerobot_version": importlib.metadata.version("lerobot"),
                "lerobot_file": str(package_file),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    preload_validated_runtime()
    runpy.run_path(str(RUN_QUEUE), run_name="__main__")
