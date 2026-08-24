#!/usr/bin/env python3
"""Generate the four active ICRA manuscript figures from frozen interfaces."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
STYLE_PATH = Path("/home/wjq/.codex/skills/figure-style/kernel.py")

spec = importlib.util.spec_from_file_location("figure_style_kernel", STYLE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load figure-style helper: {STYLE_PATH}")
style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)

# Mandatory editable-text rules, set before any figure is created.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
style.apply_figure_style(frame="open", font="Arial", sizes=(8, 7, 6), grid=False)
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

INK = "#252525"
MID = "#707070"
LIGHT = "#D6D6D6"
PALE = "#F2F2F2"
FOCAL = "#2F5D8C"
FOCAL_PALE = "#DCE7F2"
WHITE = "#FFFFFF"


def load_json(name: str) -> dict:
    with (FIG_DIR / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def export(fig: plt.Figure, stem: str) -> None:
    """Export editable SVG, vector PDF, and a 300-dpi inspection PNG."""
    for suffix in ("svg", "pdf"):
        fig.savefig(
            FIG_DIR / f"{stem}.{suffix}",
            bbox_inches="tight",
            pad_inches=0.035,
        )
    fig.savefig(
        FIG_DIR / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.035,
    )
    plt.close(fig)


def rounded_box(ax, xy, width, height, *, face, edge=INK, linewidth=0.8,
                hatch=None, radius=0.03, zorder=2):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.015,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        hatch=hatch,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def figure1() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.01, 0.94, "Repeated queries predict the same physical time", weight="bold",
            ha="left", va="top")
    source_x = [0.08, 0.23, 0.38]
    source_labels = [r"$q=t-20$", r"$q=t-10$", r"$q=t$"]
    for index, (x, label) in enumerate(zip(source_x, source_labels)):
        old = index < 2
        rounded_box(ax, (x, 0.58), 0.105, 0.17,
                    face=WHITE if old else INK,
                    edge=INK,
                    hatch="///" if old else None)
        ax.text(x + 0.0525, 0.665, label, color=INK if old else WHITE,
                ha="center", va="center", weight="bold")
        arrow = FancyArrowPatch(
            (x + 0.105, 0.665), (0.56, 0.665),
            arrowstyle="-|>", mutation_scale=8, color=MID, linewidth=0.8,
            connectionstyle=f"arc3,rad={0.12 * (1-index)}",
        )
        ax.add_patch(arrow)
    rounded_box(ax, (0.56, 0.56), 0.12, 0.21, face=PALE, edge=INK)
    ax.text(0.62, 0.68, r"target $t$", ha="center", va="center", weight="bold")
    ax.text(0.62, 0.61, r"$E_{t,q}$", ha="center", va="center")
    ax.text(0.75, 0.67, "one query every\ncontroller step", ha="left", va="center",
            color=MID)

    ax.plot([0.01, 0.99], [0.48, 0.48], color=LIGHT, linewidth=0.7)
    ax.text(0.01, 0.44, "Execution source assignment", weight="bold",
            ha="left", va="top", fontsize=7.5)
    methods = [
        (0.15, "Full fresh", [("Arm", INK, None, WHITE), ("Grip", INK, None, WHITE)]),
        (0.43, "Full old20", [("Arm", WHITE, "///", INK), ("Grip", WHITE, "///", INK)]),
        (0.71, "FO20", [("Arm", INK, None, WHITE), ("Grip", WHITE, "///", INK)]),
    ]
    for x, label, parts in methods:
        ax.text(x, 0.30, label, ha="center", va="center", fontsize=7.3,
                weight="bold", color=FOCAL if label == "FO20" else INK)
        x0 = x - 0.105
        for j, (part, face, hatch, text_color) in enumerate(parts):
            edge = FOCAL if label == "FO20" else INK
            rounded_box(ax, (x0 + j * 0.105, 0.08), 0.10, 0.13,
                        face=face, edge=edge, linewidth=1.0 if label == "FO20" else 0.7,
                        hatch=hatch, radius=0.02)
            ax.text(x0 + j * 0.105 + 0.05, 0.145, part, ha="center", va="center",
                    color=text_color, weight="bold", fontsize=7)
    ax.text(0.71, 0.035, "fresh arm + old-source gripper; value may change each tick",
            ha="center", va="top", color=FOCAL, fontsize=6.5)
    ax.text(0.97, 0.145, "dark = fresh\nhatched = old20", ha="right", va="center",
            color=MID, fontsize=6.5)
    export(fig, "fig1_temporal_source")


def figure2(data: dict) -> None:
    rates = np.asarray(data["matrix"]["rates"], dtype=float)
    labels = np.asarray(data["matrix"]["condition_labels"])
    successes = np.asarray(data["matrix"]["successes"])
    fig, ax = plt.subplots(figsize=(3.28, 2.65))
    ax.set_xlim(-0.65, 1.72)
    ax.set_ylim(-0.98, 2.08)
    ax.axis("off")
    ax.text(-0.62, 2.02, "Registered coherence unresolved", weight="bold",
            ha="left", va="top")
    ax.text(-0.62, 1.84,
            r"$C_{coh}=+.025$; paired CI $[-.030,.085]$; task CI $[-.005,.055]$",
            ha="left", va="top", fontsize=6.2, color=MID)
    ax.text(0.73, 1.43, "Gripper source", ha="center", va="center", weight="bold")
    ax.text(0.3, 1.25, "Fresh", ha="center", va="center")
    ax.text(1.15, 1.25, "Old20", ha="center", va="center")
    ax.text(-0.53, 0.33, "Arm source", rotation=90, ha="center", va="center", weight="bold")
    row_labels = ["Fresh", "Old20"]
    for i, row_name in enumerate(row_labels):
        y = 0.55 - 0.82 * i
        ax.text(-0.18, y + 0.35, row_name, ha="right", va="center")
        for j in range(2):
            x = 0.85 * j
            focal = labels[i, j] == "FO"
            rect = Rectangle(
                (x, y), 0.6, 0.68,
                facecolor=FOCAL_PALE if focal else PALE,
                edgecolor=FOCAL if focal else LIGHT,
                linewidth=1.5 if focal else 0.7,
            )
            ax.add_patch(rect)
            ax.text(x + 0.3, y + 0.47, labels[i, j], ha="center", va="center",
                    weight="bold", color=FOCAL if focal else INK)
            ax.text(x + 0.3, y + 0.25,
                    f"{successes[i,j]}/100  ({rates[i,j]*100:.0f}%)",
                    ha="center", va="center", fontsize=7)
            if focal:
                ax.text(x + 0.3, y + 0.07, "post-hoc direction", ha="center", va="bottom",
                        fontsize=5.8, color=FOCAL)
    ax.text(-0.62, -0.74, "Post-hoc marginals", weight="bold", ha="left", va="center")
    ax.text(-0.62, -0.90, "fresh arm +24.5 pp   |   old gripper +20.5 pp",
            color=FOCAL, ha="left", va="center", fontsize=6.4)
    export(fig, "fig2_developmental_factorial")


def preference_cell(ax, x0, y0, preferred, detail):
    left, right = x0 + 0.07, x0 + 0.37
    ax.plot([left, right], [y0, y0], color=LIGHT, linewidth=1.3, solid_capstyle="round")
    ax.scatter([left, right], [y0, y0], s=18, facecolor=WHITE, edgecolor=MID,
               linewidth=0.7, zorder=2)
    px = left if preferred == "fresh" else right
    ax.scatter(px, y0, s=40, facecolor=FOCAL, edgecolor=WHITE, linewidth=0.7, zorder=3)
    ax.text(x0 + 0.22, y0 - 0.11, detail, ha="center", va="top", fontsize=5.6,
            color=INK)


def figure3(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(3.28, 2.60))
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-0.08, 1)
    ax.axis("off")
    ax.text(0.0, 0.98, "Favored source depends on the objective", weight="bold",
            ha="left", va="top")
    ax.text(0.40, 0.81, "Offline error", ha="center", va="center", weight="bold")
    ax.text(0.86, 0.81, "Closed-loop post-hoc", ha="center", va="center", weight="bold")
    ax.text(0.40, 0.73, "lower is better", ha="center", va="center", color=MID, fontsize=6)
    ax.text(0.86, 0.73, "higher is better", ha="center", va="center", color=MID, fontsize=6)
    for x in (0.18, 0.64):
        ax.text(x + 0.07, 0.65, "Fresh", ha="center", va="center", fontsize=5.8)
        ax.text(x + 0.37, 0.65, "Old20", ha="center", va="center", fontsize=5.8)
    ax.text(0.01, 0.50, "Arm", ha="left", va="center", weight="bold")
    ax.text(0.01, 0.20, "Gripper", ha="left", va="center", weight="bold")
    preference_cell(ax, 0.18, 0.50, "old20", "T: .596→.507\nR: 1.130→1.099")
    preference_cell(ax, 0.64, 0.50, "fresh", ".530 vs .285")
    preference_cell(ax, 0.18, 0.20, "old20", ".308→.274")
    preference_cell(ax, 0.64, 0.20, "old20", ".305 vs .510")
    ax.text(0.54, -0.055, "Filled marker = source favored within that metric",
            ha="center", va="bottom", color=MID, fontsize=6)
    export(fig, "fig3_offline_closed_loop")


def figure4a(data: dict) -> None:
    keys = data["condition_order"]
    display = ["FO20", "Newest", "Full\nold20", "Age\nexp.", "CogACT"]
    rates = np.array([data["success_rates"][key] for key in keys])
    counts = np.rint(rates * 126).astype(int)
    colors = style.focal_palette(display, "FO20", FOCAL, other="grey")
    fig, ax = plt.subplots(figsize=(3.05, 2.45))
    x = np.arange(len(keys))
    bars = ax.bar(x, rates * 100, color=colors, edgecolor=INK, linewidth=0.55, width=0.72)
    for bar, count, rate in zip(bars, counts, rates):
        ax.text(bar.get_x() + bar.get_width()/2, rate*100 + 1.8,
                f"{count}/126\n{rate*100:.1f}%", ha="center", va="bottom", fontsize=6)
    ax.set_ylabel("Task success (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(display)
    ax.set_ylim(0, 73)
    ax.set_yticks([0, 20, 40, 60])
    ax.margins(x=0.03)
    style.panel_letter(ax, "a", dx=-0.16, dy=1.01)
    export(fig, "fig4a_success_rates")


def figure4b(data: dict) -> None:
    keys = [
        "FO20_minus_newest",
        "FO20_minus_full_old20",
        "FO20_minus_age_exp",
        "FO20_minus_CogACT",
    ]
    labels = ["Newest", "Full old20", "Age exp.", "CogACT"]
    estimates = np.array([data["primary_contrasts"][key] for key in keys]) * 100
    paired = np.array([data["confidence_intervals"][key]["paired_state"] for key in keys]) * 100
    cluster = np.array([data["confidence_intervals"][key]["task_cluster"] for key in keys]) * 100
    fig, ax = plt.subplots(figsize=(4.05, 2.45))
    y = np.arange(len(keys))[::-1]
    for yi, est, pci, cci in zip(y, estimates, paired, cluster):
        ax.plot(cci, [yi, yi], color=LIGHT, linewidth=4.2, solid_capstyle="butt", zorder=1)
        ax.plot(pci, [yi, yi], color=FOCAL, linewidth=1.8, solid_capstyle="butt", zorder=2)
        ax.scatter(est, yi, s=28, color=FOCAL, edgecolor=WHITE, linewidth=0.6, zorder=3)
        ax.text(cci[1] + 0.8, yi, f"+{est:.1f}", ha="left", va="center", fontsize=6.3)
    ax.axvline(0, color=MID, linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(-2, 36)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel("FO20 success advantage (percentage points)")
    ax.legend(
        handles=[
            Line2D([0], [0], color=FOCAL, lw=1.8, label="paired-block 95% CI"),
            Line2D([0], [0], color=LIGHT, lw=4.2, label="task-cluster 95% CI"),
        ],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.05),
        ncol=2,
        fontsize=6,
        handlelength=2.0,
    )
    style.panel_letter(ax, "b", dx=-0.23, dy=1.01)
    export(fig, "fig4b_contrasts")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    developmental = load_json("gate3b_directional_figure_interface.json")
    offline = load_json("offline_closed_loop_temporal_source_interface.json")
    confirmation = load_json("gate3c_figure4_interface.json")
    figure1()
    figure2(developmental)
    figure3(offline)
    figure4a(confirmation)
    figure4b(confirmation)


if __name__ == "__main__":
    main()
