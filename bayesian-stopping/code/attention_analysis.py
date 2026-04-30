"""
Attention analysis (sec. 5.5) on the calibration model's saved snapshot.

Loads attention_snapshot.npz (produced by train.py) and produces:
  attention_heatmaps.png             — L*M grid of mean attention patterns.
  attention_uniformity_vs_position.png — per-t L1 deviation from uniform,
                                         one curve per uniform-candidate head.
  attention_layer_progression.png    — mean entropy per layer (with min/max
                                         band across heads).
  attention_summary.json             — entropies, uniform-candidate list,
                                         per-head mean deviations.

Notation:
  attn shape (B, L, M, n, n), causal so attn[b,l,h,t,s] = 0 for s > t.
  At query position t (0-indexed), the attention is over s in [0, t], so
  uniform attention has entropy log(t+1) and weight 1/(t+1) at each s.
  We use position t = 0 as a degenerate case (entropy = 0 trivially) and
  focus on t in [1, n-1].
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eval_common import ensure_writable


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def per_position_deviation(attn: np.ndarray) -> np.ndarray:
    """L1 deviation from uniform attention at each query position t.

    attn: (B, L, M, n, n).
    Returns dev: (L, M, n) averaged over B sequences. dev[l,h,t] = mean_b
    sum_{s=0..t} |attn[b,l,h,t,s] - 1/(t+1)|. Range [0, 2]; 0 is uniform.
    """
    B, L, M, n, _ = attn.shape
    dev = np.zeros((L, M, n), dtype=np.float64)
    for t in range(n):
        uni = 1.0 / (t + 1)
        a_t = attn[:, :, :, t, : t + 1]                        # (B, L, M, t+1)
        dev[:, :, t] = np.abs(a_t - uni).sum(axis=-1).mean(axis=0)
    return dev


def per_position_entropy(attn: np.ndarray) -> np.ndarray:
    """Entropy at each query position. Returns (L, M, n)."""
    eps = 1e-12
    H = -np.where(attn > 0, attn * np.log(attn + eps), 0.0).sum(axis=-1)  # (B, L, M, n)
    return H.mean(axis=0)                                       # (L, M, n)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_heatmaps(mean_attn, H_per_lh, uniform_mask, ref, out_path):
    L, M, n, _ = mean_attn.shape
    fig = plt.figure(figsize=(13, 18))
    gs = fig.add_gridspec(L, M + 1, width_ratios=[1] * M + [0.06],
                          wspace=0.18, hspace=0.45)
    vmax = float(mean_attn.max())
    last_im = None
    for l in range(L):
        for h in range(M):
            ax = fig.add_subplot(gs[l, h])
            last_im = ax.imshow(
                mean_attn[l, h], cmap="viridis", vmin=0.0, vmax=vmax,
                aspect="auto",
            )
            ent = float(H_per_lh[l, h])
            mark = "*" if uniform_mask[l, h] else ""
            ax.text(
                0.97, 0.97, f"H={ent:.2f}{mark}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6),
            )
            ax.set_xticks([]); ax.set_yticks([])
            if h == 0: ax.set_ylabel(f"L{l}", fontsize=10)
            if l == 0: ax.set_title(f"H{h}", fontsize=10)
    cax = fig.add_subplot(gs[:, -1])
    plt.colorbar(last_im, cax=cax)
    cax.set_ylabel("attention weight", fontsize=9)
    fig.suptitle(
        f"Mean attention pattern (256 val seqs; * = uniform candidate, "
        f"H ≥ 0.9·{ref:.2f})",
        fontsize=12, y=0.995,
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_uniformity_vs_position(deviation, candidates, n, out_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    pos = np.arange(n)
    cmap = plt.get_cmap("tab20" if len(candidates) > 10 else "tab10")
    for idx, (l, h) in enumerate(candidates):
        ax.plot(pos, deviation[l, h], label=f"L{l}H{h}",
                color=cmap(idx % cmap.N), linewidth=1.2, alpha=0.85)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.6,
               label="uniform (=0)")
    ax.set_xlabel("query position t (0-indexed)")
    ax.set_ylabel("L1 deviation from uniform")
    ax.set_title(
        f"Per-position deviation from uniform attention "
        f"({len(candidates)} candidate heads)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_layer_progression(H_per_lh, ref, out_path):
    L, M = H_per_lh.shape
    layers = np.arange(L)
    mean_per = H_per_lh.mean(axis=1)
    min_per = H_per_lh.min(axis=1)
    max_per = H_per_lh.max(axis=1)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(layers, mean_per, "o-", color="tab:blue", label="mean across heads",
            linewidth=2)
    ax.fill_between(layers, min_per, max_per, color="tab:blue", alpha=0.2,
                    label="min-max across heads")
    ax.axhline(ref, color="black", linestyle="--",
               label=f"uniform reference = {ref:.3f}")
    ax.axhline(0.9 * ref, color="gray", linestyle=":",
               label=f"0.9·uniform = {0.9*ref:.3f}")
    for l in range(L):
        for h in range(M):
            ax.plot(l, H_per_lh[l, h], "o", color="tab:blue",
                    markersize=4, alpha=0.5)
    ax.set_xlabel("layer index")
    ax.set_ylabel("mean attention entropy")
    ax.set_title("Attention entropy per layer (one dot per head)")
    ax.set_xticks(layers)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", default=None,
                   help="path to <prefix>_attention_snapshot.npz; "
                        "if omitted, --model_dir + --file_prefix is used")
    p.add_argument("--model_id", default=None)
    p.add_argument("--file_prefix", default=None,
                   help="defaults to model_id")
    p.add_argument("--model_dir", default=None,
                   help="dir containing <file_prefix>_attention_snapshot.npz")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    if args.snapshot is None:
        if args.model_dir is None or (args.model_id is None and args.file_prefix is None):
            sys.exit("FATAL: provide --snapshot or "
                     "(--model_dir and one of --model_id / --file_prefix)")
        prefix = args.file_prefix or args.model_id
        snap_path = Path(args.model_dir) / f"{prefix}_attention_snapshot.npz"
    else:
        snap_path = Path(args.snapshot)

    out = ensure_writable(Path(args.output_dir))

    print(f"loading: {snap_path}")
    data = np.load(snap_path, allow_pickle=False)
    attn = data["attn"]                                         # (B, L, M, n, n)
    B, L, M, n, _ = attn.shape
    print(f"  attn shape: {attn.shape}  (B={B}, L={L}, M={M}, n={n})")

    # Aggregate stats.
    mean_attn = attn.mean(axis=0)                               # (L, M, n, n)
    H_per_pos = per_position_entropy(attn)                      # (L, M, n)
    H_per_lh = H_per_pos[:, :, 1:].mean(axis=-1)                # mean over t in [1, n-1]
    deviation = per_position_deviation(attn)                    # (L, M, n)
    mean_dev_per_lh = deviation[:, :, 1:].mean(axis=-1)         # (L, M)

    # Reference: mean over t in [1, n-1] of log(t+1).
    pos_t = np.arange(1, n)
    log_t_plus_1 = np.log(pos_t + 1)
    ref_log_t = float(log_t_plus_1.mean())

    uniform_mask = H_per_lh >= 0.9 * ref_log_t                  # (L, M)
    candidates = [(int(l), int(h))
                  for l in range(L) for h in range(M)
                  if uniform_mask[l, h]]

    print(f"  reference mean log(t+1) over t=1..{n-1}: {ref_log_t:.4f}")
    print(f"  uniform-candidate threshold (0.9 * ref): {0.9 * ref_log_t:.4f}")
    print(f"  uniform candidates: {len(candidates)} / {L*M}")

    # Plots.
    plot_heatmaps(mean_attn, H_per_lh, uniform_mask, ref_log_t,
                  out / "attention_heatmaps.png")
    print(f"  wrote: attention_heatmaps.png")

    if candidates:
        plot_uniformity_vs_position(deviation, candidates, n,
                                    out / "attention_uniformity_vs_position.png")
        print(f"  wrote: attention_uniformity_vs_position.png")
    else:
        print("  no uniform candidates — skipping uniformity_vs_position plot")

    plot_layer_progression(H_per_lh, ref_log_t,
                           out / "attention_layer_progression.png")
    print(f"  wrote: attention_layer_progression.png")

    # JSON summary.
    summary = {
        "snapshot":              str(snap_path.resolve()),
        "n_sequences":           int(B),
        "n_layers":              int(L),
        "n_heads":               int(M),
        "horizon":               int(n),
        "reference_mean_log_t_plus_1": ref_log_t,
        "uniform_threshold":     0.9 * ref_log_t,
        "per_lh_entropy":        H_per_lh.tolist(),
        "per_lh_mean_deviation": mean_dev_per_lh.tolist(),
        "uniform_candidates": [
            {
                "layer": l, "head": h,
                "entropy": float(H_per_lh[l, h]),
                "mean_deviation": float(mean_dev_per_lh[l, h]),
            }
            for l, h in candidates
        ],
        "summary_stats": {
            "mean_entropy_overall": float(H_per_lh.mean()),
            "min_entropy":          float(H_per_lh.min()),
            "max_entropy":          float(H_per_lh.max()),
            "mean_deviation_overall": float(mean_dev_per_lh.mean()),
        },
    }
    json_path = out / "attention_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote: {json_path}")


if __name__ == "__main__":
    main()
