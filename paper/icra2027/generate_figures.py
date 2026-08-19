#!/usr/bin/env python3
"""Generate paper figures from committed one-clock experiment aggregates."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parents[1]
FIGURE_DIR = PAPER_DIR / "figures"
TASK0_JSON = REPO_ROOT / "experiments" / "libero_static_grid_50.json"
CROSS_TASK_JSON = (
    REPO_ROOT / "experiments" / "libero_object_cross_task_summary.json"
)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_chunk_row(ax, y: float, spans: list[tuple[int, int, int]], label: str) -> None:
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]
    for start, end, source_id in spans:
        ax.add_patch(
            Rectangle(
                (start, y - 0.30),
                end - start,
                0.60,
                facecolor=colors[source_id % len(colors)],
                edgecolor="white",
                linewidth=0.8,
            )
        )
        ax.text(
            (start + end) / 2,
            y,
            f"C{source_id}",
            ha="center",
            va="center",
            color="white",
            fontsize=7,
            fontweight="bold",
        )
    ax.text(-0.08, y, label, ha="right", va="center", fontsize=8)


def make_timeline() -> None:
    fig, ax = plt.subplots(figsize=(7.05, 1.85))
    global_spans = [(0, 4, 0), (4, 8, 1), (8, 12, 2), (12, 16, 3), (16, 20, 4)]
    arm_spans = global_spans
    gripper_spans = [(0, 16, 0), (16, 20, 4)]

    draw_chunk_row(ax, 3.25, global_spans, "Arm")
    draw_chunk_row(ax, 2.55, global_spans, "Gripper")
    draw_chunk_row(ax, 1.25, arm_spans, "Arm")
    draw_chunk_row(ax, 0.55, gripper_spans, "Gripper")

    ax.text(-1.18, 2.90, "Global\n$h=4$", ha="right", va="center", fontweight="bold")
    ax.text(
        -1.18,
        0.90,
        "Group-specific\n$(h_a,h_g)=(4,16)$",
        ha="right",
        va="center",
        fontweight="bold",
    )

    for x in range(0, 21, 4):
        ax.axvline(x, color="#888888", linewidth=0.45, linestyle=":" if x else "-")
    for x in (4, 8, 12):
        ax.text(x, 1.72, "query; arm accepts", ha="center", va="bottom", fontsize=6.2)
    ax.text(16, 1.72, "query; both accept", ha="center", va="bottom", fontsize=6.2)

    ax.set_xlim(-2.85, 20)
    ax.set_ylim(0.05, 3.75)
    ax.set_xticks(range(0, 21, 4))
    ax.set_xlabel("Environment step")
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", length=2, pad=1)
    ax.set_title(
        "Source chunks under synchronized and group-specific fixed execution",
        pad=1,
    )
    fig.tight_layout(pad=0.25)
    fig.savefig(FIGURE_DIR / "execution_timeline.pdf", bbox_inches="tight")
    plt.close(fig)


def make_results_overview() -> None:
    with TASK0_JSON.open() as f:
        task0 = json.load(f)
    with CROSS_TASK_JSON.open() as f:
        cross = json.load(f)

    horizons = task0["horizons"]
    success = np.asarray(task0["matrices"]["success_rate"], dtype=float)
    tasks = cross["tasks"]
    global_best = np.asarray([task["best_global_success_rate"] for task in tasks])
    offdiag_best = np.asarray(
        [task["best_off_diagonal_success_rate"] for task in tasks]
    )

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(7.05, 2.55),
        gridspec_kw={"width_ratios": [1.0, 1.22], "wspace": 0.42},
    )

    image = ax0.imshow(success, vmin=0.50, vmax=0.95, cmap="YlGnBu", aspect="equal")
    for row in range(success.shape[0]):
        for col in range(success.shape[1]):
            color = "white" if success[row, col] >= 0.82 else "black"
            ax0.text(
                col,
                row,
                f"{100 * success[row, col]:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )
    ax0.set_xticks(range(len(horizons)), horizons)
    ax0.set_yticks(range(len(horizons)), horizons)
    ax0.set_xlabel("Gripper horizon $h_g$")
    ax0.set_ylabel("Arm horizon $h_a$")
    ax0.set_title("(a) Task 0 success (%, 50 states)")
    colorbar = fig.colorbar(image, ax=ax0, fraction=0.046, pad=0.03)
    colorbar.set_label("Success rate")
    colorbar.set_ticks([0.5, 0.7, 0.9])

    y = np.arange(len(tasks))
    for i, (left, right) in enumerate(zip(global_best, offdiag_best)):
        ax1.plot([left, right], [i, i], color="#B7B7B7", linewidth=1.2, zorder=1)
    ax1.scatter(
        global_best,
        y,
        s=25,
        marker="o",
        color="#4C78A8",
        label=f"Global (macro {global_best.mean():.3f})",
        zorder=2,
    )
    ax1.scatter(
        offdiag_best,
        y,
        s=28,
        marker="D",
        color="#F58518",
        label=f"Off-diagonal (macro {offdiag_best.mean():.3f})",
        zorder=3,
    )
    ax1.set_yticks(y, [f"T{task['task_id']}" for task in tasks])
    ax1.invert_yaxis()
    ax1.set_xlim(0.25, 1.04)
    ax1.set_xticks([0.3, 0.5, 0.7, 0.9, 1.0])
    ax1.set_xlabel("Success rate")
    ax1.set_title("(b) Tasks 1--9: retrospective per-task best")
    ax1.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        frameon=False,
        handletextpad=0.35,
        ncol=2,
        columnspacing=0.8,
    )
    for spine in ("top", "right", "left"):
        ax1.spines[spine].set_visible(False)
    ax1.tick_params(axis="y", length=0)

    fig.subplots_adjust(left=0.075, right=0.995, top=0.90, bottom=0.23, wspace=0.40)
    fig.savefig(FIGURE_DIR / "results_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    make_timeline()
    make_results_overview()


if __name__ == "__main__":
    main()
