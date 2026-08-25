#!/usr/bin/env python3
"""Generate the authorized NO_SIGNAL diagnostics from frozen aggregate results."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


REPO = Path("/home/wjq/workspace/one-clock")
ANALYSIS = REPO / "research/audit_outputs/robotwin_exploratory_analysis.json"
STATUS = REPO / "research/audit_outputs/robotwin_exploratory_followup_status.json"
MEMO = REPO / "research/robotwin_no_signal_diagnosis.md"
PNG = REPO / "research/audit_outputs/robotwin_fo_control_deltas.png"
PDF = REPO / "research/audit_outputs/robotwin_fo_control_deltas.pdf"
STYLE = Path("/home/wjq/.codex/skills/figure-style/kernel.py")


def main() -> None:
    result = json.loads(ANALYSIS.read_text())
    if result["classification"] != "NO_SIGNAL":
        raise RuntimeError("NO_SIGNAL diagnostics called for another classification")
    spec = importlib.util.spec_from_file_location("figure_style_kernel", STYLE)
    style = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(style)
    style.apply_figure_style(sizes=(8, 7, 6))

    tasks = [row["task"] for row in result["task_method"] if row["method"] == "NATIVE_ACT"]
    comparators = ["NEWEST", "NATIVE_ACT", "FULL_OLD_1S", "GRIPPER_HOLD", "GRIPPER_EMA_1S"]
    matrix = np.asarray(
        [[result["contrasts"][comparator]["task_deltas"][task] for comparator in comparators] for task in tasks]
    )
    limit = max(0.05, float(np.max(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(7.1, 3.2), constrained_layout=True)
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(column, row, f"{value:+.2f}", ha="center", va="center", color="white" if abs(value) > 0.55 * limit else "black")
    ax.set_xticks(range(len(comparators)), [item.replace("_", "\n") for item in comparators])
    ax.set_yticks(range(len(tasks)), [item.replace("_", " ") for item in tasks])
    ax.set_xlabel("Comparator")
    ax.set_ylabel("RoboTwin task")
    ax.set_title("FO paired success differences across tasks and controls", loc="left")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("FO minus comparator success")
    PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG, dpi=300)
    fig.savefig(PDF)
    plt.close(fig)
    with Image.open(PNG) as rendered:
        if rendered.width < 1000 or rendered.height < 500 or np.asarray(rendered).std() == 0:
            raise RuntimeError("diagnostic figure render failed")

    primary = result["contrasts"]["NEWEST"]
    lines = [
        "# RoboTwin exploratory NO_SIGNAL diagnosis",
        "",
        f"The preregistered classification is **NO_SIGNAL**. FO minus NEWEST was "
        f"{primary['pooled_paired_difference']:+.3f}, with task-cluster 95% interval "
        f"[{primary['task_cluster_bootstrap_95_interval'][0]:+.3f}, "
        f"{primary['task_cluster_bootstrap_95_interval'][1]:+.3f}].",
        "",
        "## Control diagnosis",
        "",
    ]
    for comparator in ("FULL_OLD_1S", "GRIPPER_HOLD", "GRIPPER_EMA_1S", "NATIVE_ACT"):
        contrast = result["contrasts"][comparator]
        lines.append(f"- FO minus {comparator}: {contrast['pooled_paired_difference']:+.3f}.")
    lines.extend(
        [
            "",
            "The control closest to FO indicates whether globally old predictions, command retention, "
            "simple smoothing, or official shared-clock temporal aggregation accounts for the observed pattern. "
            "No source age, task, seed, or smoothing parameter was retuned.",
            "",
            "## LIBERO versus RoboTwin",
            "",
            "LIBERO Object Gate-3C remains frozen: FO20 63.5%, NEWEST 42.1%, FULL_OLD20 43.7%, "
            "Age-exp 49.2%, and CogACT 46.8%. The RoboTwin table and heatmap report the independently "
            "preregistered bimanual result without altering those figures.",
            "",
            "## Closest-method interpretation",
            "",
            "- NATIVE_ACT is the official global temporal-aggregation comparator.",
            "- FULL_OLD_1S tests a shared old temporal source.",
            "- GRIPPER_HOLD and GRIPPER_EMA_1S test retention and smoothing explanations.",
            "- FO_1S is supported only if its paired advantage survives those controls; the frozen gate did not.",
            "",
            "## Reviewer risk",
            "",
            "The cross-benchmark intervention preserves physical source age but aligns chunks by decision target "
            "under variable-duration TOPP. A null or control-explained result therefore bounds generality; it must "
            "not be reframed through post-hoc age tuning or task selection.",
            "",
        ]
    )
    MEMO.write_text("\n".join(lines))
    status = json.loads(STATUS.read_text())
    status["status"] = "NO_SIGNAL_DIAGNOSTICS_COMPLETE"
    status["diagnostic_memo"] = str(MEMO.relative_to(REPO))
    status["diagnostic_figure"] = str(PNG.relative_to(REPO))
    STATUS.write_text(json.dumps(status, indent=2) + "\n")
    subprocess.run(
        ["git", "add", str(MEMO.relative_to(REPO)), str(PNG.relative_to(REPO)), str(PDF.relative_to(REPO)), str(STATUS.relative_to(REPO))],
        cwd=REPO,
        check=True,
    )
    subprocess.run(["git", "commit", "-m", "results: add RoboTwin no-signal diagnostics"], cwd=REPO, check=True)


if __name__ == "__main__":
    main()
