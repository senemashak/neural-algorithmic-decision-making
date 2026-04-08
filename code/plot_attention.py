"""
Generate attention visualizations from saved attention map files.

Reads .pt files from attention/ directory (saved by sweep_weights.py)
and generates plots for mechanistic interpretation.

Usage:
    python plot_attention.py results/sweep/sweep_<ts>/attention/stopping_value+action.pt
    python plot_attention.py results/sweep/sweep_<ts>/attention/ --configs value+action chain_only
    python plot_attention.py results/sweep/sweep_<ts>/attention/ --all
"""

import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def plot_config_attention(attn_path, out_dir):
    """Generate all attention plots for one config."""
    d = torch.load(attn_path, weights_only=False)
    attn = d["attn_weights"]
    info = d["info"]
    n = d["n"]
    config = d["config"]
    labels = info["position_labels"]
    decision_pos = info["decision_pos"].tolist()
    t_idx = info["t_idx"]
    j_idx = info["j_idx"]
    L = attn[0].shape[2]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  {config}: n={n}, {L} positions, 2 layers x 2 heads")

    # 1. Full attention maps
    for layer in range(len(attn)):
        for head in range(attn[0].shape[1]):
            fig, ax = plt.subplots(figsize=(12, 10))
            A = attn[layer][0, head].numpy()
            im = ax.imshow(A, cmap="Blues", vmin=0, aspect="auto")
            ax.axhline(n - 0.5, color="red", linewidth=1.5, linestyle="--")
            ax.axvline(n - 0.5, color="red", linewidth=1.5, linestyle="--")
            mid, cmid = n // 2, n + (L - n) // 2
            ax.text(mid, mid, "obs\u2192obs", ha="center", va="center",
                    fontsize=10, color="red", alpha=0.6, fontweight="bold")
            ax.text(cmid, mid, "obs\u2192chain\n(blocked)", ha="center", va="center",
                    fontsize=9, color="gray", alpha=0.5)
            ax.text(mid, cmid, "chain\u2192obs\n(dist. inference)", ha="center", va="center",
                    fontsize=9, color="red", alpha=0.6, fontweight="bold")
            ax.text(cmid, cmid, "chain\u2192chain\n(DP recurrence)", ha="center", va="center",
                    fontsize=9, color="red", alpha=0.6, fontweight="bold")
            ax.set_xlabel("Key (attending to)")
            ax.set_ylabel("Query (attending from)")
            ax.set_title(f"{config}: Layer {layer}, Head {head}")
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            fig.savefig(out_dir / f"full_L{layer}_H{head}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # 2. Decision step observation attention
    test_steps = [0, n//4, n//2, 3*n//4, n-2]
    for layer in range(len(attn)):
        fig, axes = plt.subplots(2, len(test_steps), figsize=(18, 6), sharey="row")
        for col, t in enumerate(test_steps):
            if t >= len(decision_pos):
                continue
            dp = decision_pos[t]
            gp = n + dp
            for head in range(attn[0].shape[1]):
                ax = axes[head, col]
                weights = attn[layer][0, head, gp, :n].numpy()
                bars = ax.bar(range(n), weights, color="#90CAF9", alpha=0.7)
                for i in range(min(t + 1, n)):
                    bars[i].set_color("#1565C0")
                    bars[i].set_alpha(0.9)
                ax.set_xlim(-0.5, n - 0.5)
                if head == 0:
                    ax.set_title(f"V({t},{t})", fontsize=9)
                if col == 0:
                    ax.set_ylabel(f"Head {head}")
                if head == 1:
                    ax.set_xlabel("Obs position")
        fig.suptitle(f"{config} Layer {layer}: V(t,t) attention to observations\n"
                     f"(dark = visible x_0..x_t)", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"decision_obs_L{layer}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 3. Sub-chain recurrence
    from model import OnlineDecisionTransformer
    for t_show in [0, n//4, n//2]:
        sub_len = n - 1 - t_show
        if sub_len <= 1:
            continue
        off = OnlineDecisionTransformer._chain2d_offset(t_show, n)
        chain_positions = list(range(n + off, n + off + sub_len))

        for layer in range(len(attn)):
            fig, axes = plt.subplots(1, attn[0].shape[1], figsize=(12, 5))
            if attn[0].shape[1] == 1:
                axes = [axes]
            for head in range(attn[0].shape[1]):
                ax = axes[head]
                sub_attn = np.zeros((sub_len, sub_len))
                for i in range(sub_len):
                    for j in range(sub_len):
                        sub_attn[i, j] = attn[layer][0, head, chain_positions[i], chain_positions[j]].item()
                im = ax.imshow(sub_attn, cmap="Blues", vmin=0)
                tick_labels = [f"V({t_show},{n-2-j})" for j in range(sub_len)]
                step = max(1, sub_len // 8)
                ax.set_xticks(range(0, sub_len, step))
                ax.set_xticklabels([tick_labels[i] for i in range(0, sub_len, step)],
                                   fontsize=7, rotation=45, ha="right")
                ax.set_yticks(range(0, sub_len, step))
                ax.set_yticklabels([tick_labels[i] for i in range(0, sub_len, step)], fontsize=7)
                ax.set_title(f"Head {head}")
                ax.set_xlabel("Key")
                ax.set_ylabel("Query")
            fig.suptitle(f"{config} Layer {layer}: Sub-chain t={t_show} internal attention",
                         fontsize=11)
            fig.tight_layout()
            fig.savefig(out_dir / f"subchain_t{t_show}_L{layer}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # 4. Head specialization
    n_layers = len(attn)
    n_heads = attn[0].shape[1]
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(6 * n_heads, 4 * n_layers))
    if n_layers == 1 and n_heads == 1:
        axes = np.array([[axes]])
    elif n_layers == 1:
        axes = axes[np.newaxis, :]
    elif n_heads == 1:
        axes = axes[:, np.newaxis]
    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer][head]
            obs_f, chain_f = [], []
            for t in range(len(decision_pos)):
                gp = n + decision_pos[t]
                w = attn[layer][0, head, gp, :].numpy()
                obs_f.append(w[:n].sum())
                chain_f.append(w[n:].sum())
            ax.bar(range(len(decision_pos)), obs_f, label="obs", color="#42A5F5", alpha=0.8)
            ax.bar(range(len(decision_pos)), chain_f, bottom=obs_f, label="chain",
                   color="#66BB6A", alpha=0.8)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Decision step t")
            ax.set_ylabel("Attention fraction")
            ax.set_title(f"Layer {layer}, Head {head}")
            if layer == 0 and head == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f"{config}: Head Specialization (obs vs chain attention)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "head_specialization.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    n_plots = len(list(out_dir.glob("*.png")))
    print(f"  {n_plots} plots saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to .pt file or attention/ directory")
    parser.add_argument("--configs", nargs="+", default=None, help="Config names to plot")
    parser.add_argument("--all", action="store_true", help="Plot all configs in directory")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (default: plots/attention_<config>/)")
    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        # Single .pt file
        config = path.stem.replace("stopping_", "").replace("ski_", "")
        out = Path(args.out_dir) if args.out_dir else path.parent.parent.parent.parent / "plots" / f"attention_{config}"
        plot_config_attention(path, out)

    elif path.is_dir():
        # Directory of .pt files
        files = sorted(path.glob("*.pt"))
        if args.configs:
            files = [f for f in files if any(c in f.stem for c in args.configs)]
        elif not args.all:
            print("Specify --configs or --all")
            return

        for f in files:
            config = f.stem.replace("stopping_", "").replace("ski_", "")
            out = Path(args.out_dir) / f"attention_{config}" if args.out_dir else f.parent.parent.parent.parent / "plots" / f"attention_{config}"
            plot_config_attention(f, out)


if __name__ == "__main__":
    main()
