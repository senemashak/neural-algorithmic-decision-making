"""
Generate landscape diagram: three regimes of online decision-making.

2D plot with axes:
  X — Decision recoverability (irrevocable ↔ recoverable)
  Y — Feedback richness (sparse ↔ per-step)

Three regimes as shaded ellipses with examples and suitable approaches.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path


def plot_landscape(out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))

    # --- Axes setup ---
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Decision recoverability", fontsize=13, fontweight="bold")
    ax.set_ylabel("Feedback richness", fontsize=13, fontweight="bold")

    # Arrow-style axes
    ax.annotate("", xy=(1.12, 0), xytext=(-0.05, 0),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))
    ax.annotate("", xy=(0, 1.12), xytext=(0, -0.05),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))

    # Axis labels at ends
    ax.text(1.12, -0.07, "recoverable", fontsize=9, ha="right", color="0.3")
    ax.text(-0.02, -0.07, "irrevocable", fontsize=9, ha="left", color="0.3")
    ax.text(-0.08, 1.10, "per-step", fontsize=9, ha="center", color="0.3", rotation=90)
    ax.text(-0.08, 0.02, "sparse", fontsize=9, ha="center", color="0.3", rotation=90)

    # --- Three regime boxes ---
    box_kw = dict(boxstyle="round,pad=0.03", linewidth=1.5)

    # Region 1: Data-rich, decision-rich (top-right)
    r1 = FancyBboxPatch((0.55, 0.55), 0.48, 0.48, **box_kw,
                         facecolor="#BBDEFB", edgecolor="#1565C0", alpha=0.55)
    ax.add_patch(r1)
    ax.text(0.79, 0.90, "Data-rich,\ndecision-rich", fontsize=12,
            fontweight="bold", ha="center", va="center", color="#0D47A1")
    ax.text(0.79, 0.74, "Online paging, experts\n"
            "MWU, online learning", fontsize=9,
            ha="center", va="center", color="#1565C0", style="italic")
    ax.text(0.79, 0.62, "per-step feedback → hedge", fontsize=8,
            ha="center", va="center", color="#1565C0",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#90CAF9", alpha=0.8))

    # Region 2: Data-poor, decision-poor (bottom-left) — FOCUS
    r2 = FancyBboxPatch((-0.03, -0.03), 0.52, 0.52, **box_kw,
                         facecolor="#C8E6C9", edgecolor="#2E7D32", alpha=0.55)
    ax.add_patch(r2)
    ax.text(0.23, 0.36, "Data-poor,\ndecision-poor", fontsize=12,
            fontweight="bold", ha="center", va="center", color="#1B5E20")
    ax.text(0.23, 0.20, "Optimal stopping, ski rental\n"
            "Value-to-go + robustness wrapper", fontsize=9,
            ha="center", va="center", color="#2E7D32", style="italic")
    ax.text(0.23, 0.08, "few irrevocable decisions", fontsize=8,
            ha="center", va="center", color="#2E7D32",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#A5D6A7", alpha=0.8))

    # Star marker for "focus of this report"
    ax.plot(0.23, 0.44, marker="*", markersize=16, color="#FFD600",
            markeredgecolor="#F57F17", markeredgewidth=0.8, zorder=5)
    ax.text(0.34, 0.445, "focus of this report", fontsize=8,
            color="#F57F17", fontweight="bold", va="center")

    # Region 3: Data-poor, action-rich (bottom-right)
    r3 = FancyBboxPatch((0.55, -0.03), 0.48, 0.48, **box_kw,
                         facecolor="#FFE0B2", edgecolor="#E65100", alpha=0.55)
    ax.add_patch(r3)
    ax.text(0.79, 0.32, "Data-poor,\naction-rich", fontsize=12,
            fontweight="bold", ha="center", va="center", color="#BF360C")
    ax.text(0.79, 0.16, "Revenue management, demand\n"
            "Cumulative reward optimization", fontsize=9,
            ha="center", va="center", color="#E65100", style="italic")
    ax.text(0.79, 0.04, "many actions, no per-step feedback", fontsize=8,
            ha="center", va="center", color="#E65100",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#FFCC80", alpha=0.8))

    # --- Clean up ---
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved landscape diagram to {out_path}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "plots" / "landscape.png"
    plot_landscape(out)
