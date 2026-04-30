"""
Attention uniformity by training regime — data-dependence diagnostic for §5.5.

For each of the six models, load the saved attention_snapshot.npz (256
sequences × 8 layers × 4 heads × 64 query × 64 key) and compute two
per-head summary statistics:

  D_{l,h}        := E_t [ E_seq [ sum_{s <= t} | A_{l,h}[t,s] - 1/(t+1) | ] ]
                    — total-variation deviation from the uniform-on-[0..t]
                    distribution. Smaller = more uniform. Averaged over
                    t in [3, n-2] (skipping the first 3 steps, where t+1
                    is small and the metric is noisy) and over the 256
                    sequences.

  W_recent_{l,h} := E_t [ E_seq [ A_{l,h}[t, t] ] ]
                    — average self-attention weight on the most-recent
                    position. The uniform reference is 1/(t+1), so a
                    horizontal line at the mean over t of 1/(t+1)
                    marks "uniform" (it depends on n).

For each statistic we draw a 6-violin row, one per model in canonical
order (D1_cv, D2_cv, D3_cv, D1_act, D2_act, D3_act), filling each violin
with the train-regime hue from the existing palette. Each violin is
annotated with the count of "uniform-candidate" heads from the saved
attention_summary.json (entropy ≥ 0.9 log(t+1) averaged over t).

Output: sweep/experiments/figures/fig_attention_uniformity_by_regime.png
"""

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


import config
SWEEP = config.SWEEP_ROOT
OUTDIR = SWEEP / "experiments" / "figures"

REGIME = {
    1: {"name": "data-dominant",   "rho": "ρ=0.1",  "color": "#F4A6A6"},
    2: {"name": "balanced",        "rho": "ρ=1",    "color": "#F4D88A"},
    3: {"name": "prior-dominant",  "rho": "ρ=10",   "color": "#A6C8F4"},
}

MODELS = [
    ("D1_cv",  1, "cv"),
    ("D2_cv",  2, "cv"),
    ("D3_cv",  3, "cv"),
    ("D1_act", 1, "act"),
    ("D2_act", 2, "act"),
    ("D3_act", 3, "act"),
]

T_LO, T_HI = 3, config.N - 2     # skip the first three rows (small t -> noisy);
                                  # T_HI = n-2 so we average over t in [3, n-2]


def per_head_stats(attn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """attn: (N, L, H, n, n). Returns (D, W_recent), each (L, H)."""
    N, L, H, n, _ = attn.shape
    D = np.zeros((L, H), dtype=np.float64)
    W = np.zeros((L, H), dtype=np.float64)
    t_range = range(T_LO, T_HI)
    for t in t_range:
        unif = 1.0 / (t + 1)
        # |A[t, s] - 1/(t+1)| summed over s in [0..t], averaged over sequences
        # attn[:, l, h, t, :t+1] has shape (N, t+1)
        slc = attn[:, :, :, t, : t + 1]            # (N, L, H, t+1)
        dev = np.abs(slc - unif).sum(axis=-1)      # (N, L, H)
        D += dev.mean(axis=0)
        # Most-recent weight: A[t, t]
        W += attn[:, :, :, t, t].mean(axis=0)
    n_t = T_HI - T_LO
    return D / n_t, W / n_t


def load_uniform_count(model_id: str) -> int:
    p = SWEEP / model_id / "experiments" / model_id / "attention_summary.json"
    with open(p) as f:
        return len(json.load(f)["uniform_candidates"])


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"writing to: {OUTDIR}")

    stats = []
    for mid, train_id, sup in MODELS:
        snap = SWEEP / mid / f"{mid}_attention_snapshot.npz"
        print(f"  loading {mid}...")
        with np.load(snap) as d:
            attn = d["attn"]
        D, W = per_head_stats(attn)
        n_uniform = load_uniform_count(mid)
        stats.append({
            "model_id": mid, "train_id": train_id, "sup": sup,
            "D": D.flatten(), "W": W.flatten(), "n_uniform": n_uniform,
        })

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    positions = np.arange(1, len(MODELS) + 1)
    colors = [REGIME[s["train_id"]]["color"] for s in stats]
    labels = [s["model_id"] for s in stats]

    # ---------------- PANEL A — deviation from uniform ----------------
    axA = axes[0]
    parts = axA.violinplot(
        [s["D"] for s in stats], positions=positions, widths=0.78,
        showmeans=True, showextrema=True,
    )
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color); body.set_edgecolor("black")
        body.set_linewidth(0.6); body.set_alpha(0.85)
    for key in ("cmeans", "cmaxes", "cmins", "cbars"):
        if key in parts:
            parts[key].set_color("black"); parts[key].set_linewidth(0.7)
    # Annotate uniform-candidate count near the top
    ymax = max(s["D"].max() for s in stats)
    for x, s in zip(positions, stats):
        axA.text(x, ymax * 1.04, f"{s['n_uniform']}/32",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    axA.set_ylim(0, ymax * 1.12)
    axA.set_xticks(positions)
    axA.set_xticklabels(labels, fontsize=10)
    axA.set_ylabel(r"$D_{\ell,h}$  (smaller = more uniform)")
    axA.set_title(
        "Panel A — per-head deviation from uniform attention.\n"
        r"$D_{\ell,h} := \mathbb{E}_t\,\mathbb{E}_{\mathrm{seq}}\,"
        r"\sum_{s\leq t}|A_{\ell,h}[t,s] - 1/(t+1)|$, "
        rf"averaged over $t \in [{T_LO}, {T_HI}]$ and 256 test sequences. "
        "Bold annotation: uniform-candidate count out of 32 heads.",
        fontsize=10,
    )
    axA.grid(True, axis="y", alpha=0.3)
    # Vertical separator between cv and act blocks
    axA.axvline(3.5, color="0.6", linewidth=0.7, linestyle=":")

    # ---------------- PANEL B — most-recent-position weight ----------------
    axB = axes[1]
    parts = axB.violinplot(
        [s["W"] for s in stats], positions=positions, widths=0.78,
        showmeans=True, showextrema=True,
    )
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color); body.set_edgecolor("black")
        body.set_linewidth(0.6); body.set_alpha(0.85)
    for key in ("cmeans", "cmaxes", "cmins", "cbars"):
        if key in parts:
            parts[key].set_color("black"); parts[key].set_linewidth(0.7)
    # Reference uniform: mean over t in [T_LO, T_HI] of 1/(t+1)
    unif_ref = float(np.mean([1.0 / (t + 1) for t in range(T_LO, T_HI)]))
    axB.axhline(unif_ref, color="black", linewidth=0.9, linestyle="--",
                alpha=0.6, label=f"uniform reference = "
                                  f"$\\overline{{1/(t+1)}}$ = {unif_ref:.3f}")
    axB.set_xticks(positions)
    axB.set_xticklabels(labels, fontsize=10)
    axB.set_ylabel(r"$W_{\mathrm{recent},\,\ell,h} = "
                   r"\mathbb{E}_t\,A_{\ell,h}[t,t]$")
    axB.set_title(
        "Panel B — average self-attention weight on the most-recent position. "
        "Heads above the dashed line over-weight the latest offer; "
        "heads near the line attend uniformly across the past.",
        fontsize=10,
    )
    axB.grid(True, axis="y", alpha=0.3)
    axB.axvline(3.5, color="0.6", linewidth=0.7, linestyle=":")
    axB.legend(loc="upper right", fontsize=8)

    # Regime legend on the figure
    regime_handles = [
        mpatches.Patch(color=REGIME[i]["color"],
                       label=f"trained on {REGIME[i]['name']} ({REGIME[i]['rho']})")
        for i in (1, 2, 3)
    ]
    fig.legend(handles=regime_handles, loc="upper center",
               ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.985),
               title="violin colour = training regime")
    # Block labels (cv / act)
    fig.text(0.21, 0.95, "cv supervision", ha="center", fontsize=10,
             fontweight="bold", color="#444")
    fig.text(0.62, 0.95, "act supervision", ha="center", fontsize=10,
             fontweight="bold", color="#444")

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = OUTDIR / "fig_attention_uniformity_by_regime.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
