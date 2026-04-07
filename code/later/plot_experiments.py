"""
Plotting functions for all experiments.

Each function takes results (dicts/DataFrames from CSVs) and saves PNGs.
Consistent style: matplotlib with seaborn-inspired colors, error bars/shading
for multi-seed results, bold best values in tables.

Output directory: ../new_plots/
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "new_plots"
OUT_DIR.mkdir(exist_ok=True)

# ── Style: blue-green-purple pastel palette ──
COLORS = {
    "dp": "#90CAF9",           # pastel blue
    "learned": "#A5D6A7",      # pastel green
    "robust": "#B39DDB",       # pastel purple
    "deterministic": "#80CBC4", # pastel teal
    "offline": "#E0E0E0",      # light gray
    "dynkin": "#CE93D8",       # pastel lavender
    "standard": "#90CAF9",     # pastel blue
    "masked": "#B39DDB",       # pastel purple
    "chain_sup": "#A5D6A7",    # pastel green
    "chain_unsup": "#80DEEA",  # pastel cyan
}

# Darker variants for lines
COLORS_LINE = {
    "dp": "#1565C0",
    "learned": "#2E7D32",
    "robust": "#5E35B1",
    "deterministic": "#00695C",
    "offline": "#616161",
    "dynkin": "#7B1FA2",
    "standard": "#1565C0",
    "masked": "#5E35B1",
    "chain_sup": "#2E7D32",
    "chain_unsup": "#00838F",
}

def _savefig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Standard evaluations
# ═══════════════════════════════════════════════════════════════════════════

def plot_in_distribution_bars(results, problem="stopping", prefix=""):
    """
    Bar chart: CR (stopping) or additive loss (ski) per policy.
    results: dict policy_name -> {mean, std} or {cr, ...}
    Bars are pastel-filled. DP shown as dashed hline.
    ±1 std shown as thin black whiskers.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    metric = "cr" if problem == "stopping" else "mean_additive_loss"
    ylabel = "Competitive Ratio" if problem == "stopping" else "Additive Loss Δ"

    names = list(results.keys())
    means = [results[n].get(f"mean_{metric}", results[n].get(metric, 0)) for n in names]
    stds = [results[n].get(f"std_{metric}", 0) for n in names]

    # Pastel blue-green-purple palette with distinct colors per bar
    _PASTEL_CYCLE = [
        "#E0E0E0",  # gray (offline)
        "#90CAF9",  # pastel blue (dp)
        "#A5D6A7",  # pastel green (learned)
        "#B39DDB",  # pastel purple
        "#80CBC4",  # pastel teal
        "#CE93D8",  # pastel lavender
        "#80DEEA",  # pastel cyan
        "#C5CAE9",  # pastel indigo
        "#B2DFDB",  # pastel mint
        "#D1C4E9",  # pastel violet
        "#B3E5FC",  # pastel sky
        "#C8E6C9",  # pastel lime
        "#E1BEE7",  # pastel orchid
    ]
    bar_colors = [_PASTEL_CYCLE[i % len(_PASTEL_CYCLE)] for i in range(len(names))]

    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=3,
                  color=bar_colors, edgecolor="#616161", linewidth=0.5,
                  error_kw={"linewidth": 0.8, "color": "#424242"})

    # DP as horizontal dashed line if present
    if "dp" in results:
        dp_val = results["dp"].get(f"mean_{metric}", results["dp"].get(metric, 0))
        ax.axhline(dp_val, color="#1565C0", linestyle="--", linewidth=1.5,
                   alpha=0.7, label="Bayes DP")
        ax.legend()

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(f"In-Distribution Evaluation — {'Optimal Stopping' if problem == 'stopping' else 'Ski Rental'}")
    fig.tight_layout()
    _savefig(fig, f"{prefix}in_dist_{problem}.png")


def plot_per_family_heatmap(family_results, problem="stopping", prefix=""):
    """
    Heatmap: rows=policies, columns=distribution families.
    family_results: dict[family][policy] -> value
    """
    families = sorted(family_results.keys())
    policies = sorted({p for f in family_results.values() for p in f.keys()})

    data = np.zeros((len(policies), len(families)))
    for j, fam in enumerate(families):
        for i, pol in enumerate(policies):
            data[i, j] = family_results.get(fam, {}).get(pol, 0)

    fig, ax = plt.subplots(figsize=(max(10, len(families)), max(4, len(policies) * 0.5)))
    # Pastel green-white-purple colormap
    from matplotlib.colors import LinearSegmentedColormap
    _pastel_cmap = LinearSegmentedColormap.from_list(
        "pastel_gp", ["#E1BEE7", "#F3E5F5", "#FFFFFF", "#E8F5E9", "#A5D6A7"])
    cmap_use = _pastel_cmap if problem == "stopping" else _pastel_cmap.reversed()
    im = ax.imshow(data, aspect="auto", cmap=cmap_use)

    ax.set_xticks(range(len(families)))
    ax.set_xticklabels([f.replace("_", "\n") for f in families], fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(policies)))
    ax.set_yticklabels(policies, fontsize=8)

    # Annotate cells
    for i in range(len(policies)):
        for j in range(len(families)):
            val = data[i, j]
            # Bold best per column
            col_vals = data[:, j]
            is_best = (val == col_vals.max()) if problem == "stopping" else (val == col_vals.min())
            weight = "bold" if is_best else "normal"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=6,
                    fontweight=weight, color="black")

    metric_label = "Competitive Ratio" if problem == "stopping" else "Additive Loss"
    ax.set_title(f"Per-Family {metric_label} — {'Optimal Stopping' if problem == 'stopping' else 'Ski Rental'}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    _savefig(fig, f"{prefix}per_family_{problem}.png")


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 1: Loss weighting ablation
# ═══════════════════════════════════════════════════════════════════════════

def plot_weight_sweep(configs, metric_vals, metric_stds, metric_name, prefix=""):
    """
    Horizontal bar chart: configs ranked by metric, with error bars.
    configs: list of config names
    metric_vals, metric_stds: parallel lists
    """
    # Sort by metric
    order = np.argsort(metric_vals)[::-1] if "cr" in metric_name.lower() else np.argsort(metric_vals)
    configs = [configs[i] for i in order]
    metric_vals = [metric_vals[i] for i in order]
    metric_stds = [metric_stds[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(4, len(configs) * 0.4)))
    ax.barh(range(len(configs)), metric_vals, xerr=metric_stds, capsize=3,
            color=COLORS["learned"], edgecolor="#616161", height=0.6,
            error_kw={"linewidth": 0.8, "color": "#424242"})
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=8)
    ax.set_xlabel(metric_name)
    ax.set_title("Experiment 1: Loss Weighting Ablation — Which $(w_v, w_a, w_c)$ Works Best?")
    ax.invert_yaxis()
    fig.tight_layout()
    _savefig(fig, f"{prefix}exp1_weight_sweep.png")


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 2: Chain-of-thought vs standard
# ═══════════════════════════════════════════════════════════════════════════

def plot_chain_comparison(conditions, standard_vals, chain_vals, std_s, std_c,
                          metric_name, prefix=""):
    """
    Grouped bar chart: standard vs chain across conditions.
    """
    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, standard_vals, width, yerr=std_s, capsize=3,
           label="w_chain=0 (unsupervised scratchpad)", color=COLORS["chain_unsup"],
           edgecolor="#616161", linewidth=0.5,
           error_kw={"linewidth": 0.8, "color": "#424242"})
    ax.bar(x + width/2, chain_vals, width, yerr=std_c, capsize=3,
           label="w_chain>0 (supervised chain)", color=COLORS["chain_sup"],
           edgecolor="#616161", linewidth=0.5,
           error_kw={"linewidth": 0.8, "color": "#424242"})

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.set_ylabel(metric_name)
    ax.set_title("Experiment 2: Supervised vs Unsupervised Chain-of-Thought")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, f"{prefix}exp2_chain_vs_standard.png")


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 3: Robust-aware training sweep
# ═══════════════════════════════════════════════════════════════════════════

def plot_robust_sweep(params, standard_vals, masked_vals, std_s, std_m,
                      param_name, metric_name, prefix=""):
    """
    Line plot: x=robustness parameter, y=metric, two lines ± shading.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    params = np.array(params)

    ax.plot(params, standard_vals, "o-", color="#42A5F5", label="Standard training",
            markersize=6, linewidth=2)
    ax.fill_between(params,
                    np.array(standard_vals) - np.array(std_s),
                    np.array(standard_vals) + np.array(std_s),
                    alpha=0.15, color="#42A5F5")

    ax.plot(params, masked_vals, "s-", color="#7E57C2", label="Robust-aware training",
            markersize=6, linewidth=2)
    ax.fill_between(params,
                    np.array(masked_vals) - np.array(std_m),
                    np.array(masked_vals) + np.array(std_m),
                    alpha=0.15, color="#7E57C2")

    ax.set_xlabel(param_name)
    ax.set_ylabel(metric_name)
    ax.set_title("Experiment 3: Robust-Aware vs Standard Training")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, f"{prefix}exp3_robust_sweep.png")


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 4: Horizon and cost-ratio generalization
# ═══════════════════════════════════════════════════════════════════════════

def plot_horizon_generalization(horizons, curves, metric_name, train_range=(20, 200), prefix=""):
    """
    Line plot: x=test horizon, y=metric.
    curves: dict name -> (means, stds)
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    _H_PASTELS = ["#42A5F5", "#66BB6A", "#AB47BC", "#26A69A", "#7E57C2", "#29B6F6"]
    markers_h = ["o", "s", "^", "D", "v", "P"]
    for idx, (name, (means, stds)) in enumerate(curves.items()):
        color = _H_PASTELS[idx % len(_H_PASTELS)]
        marker = markers_h[idx % len(markers_h)]
        ax.plot(horizons, means, f"{marker}-", label=name, color=color, markersize=5)
        ax.fill_between(horizons,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.1, color=color)

    # Training range shading
    ax.axvspan(train_range[0], train_range[1], alpha=0.05, color="#A5D6A7",
               label=f"Training range [{train_range[0]}, {train_range[1]}]")

    ax.set_xlabel("Test Horizon n")
    ax.set_ylabel(metric_name)
    ax.set_title("Experiment 4: Horizon Generalization (Train n ∈ [20, 200])")
    ax.legend(fontsize=8)
    ax.set_xscale("log")
    fig.tight_layout()
    _savefig(fig, f"{prefix}exp4_horizon.png")


def plot_cost_ratio_generalization(ratios, curves, train_range=(10, 100), prefix=""):
    """
    Line plot: x=B/r, y=additive loss.
    curves: dict name -> (means, stds)
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    _C_PASTELS = ["#42A5F5", "#66BB6A", "#AB47BC", "#26A69A", "#7E57C2", "#29B6F6"]
    markers_c = ["o", "s", "^", "D", "v", "P"]
    for idx, (name, (means, stds)) in enumerate(curves.items()):
        color = _C_PASTELS[idx % len(_C_PASTELS)]
        marker = markers_c[idx % len(markers_c)]
        ax.plot(ratios, means, f"{marker}-", label=name, color=color, markersize=5)
        ax.fill_between(ratios,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.1, color=color)

    ax.axvspan(train_range[0], train_range[1], alpha=0.05, color="#A5D6A7",
               label=f"Training range [{train_range[0]}, {train_range[1]}]")

    ax.set_xlabel("Cost Ratio B/r")
    ax.set_ylabel("Additive Loss Δ")
    ax.set_title("Experiment 4: Cost-Ratio Generalization — Ski Rental (Train B/r ∈ [10, 100])")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, f"{prefix}exp4_cost_ratio.png")


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 5: Attention analysis
# ═══════════════════════════════════════════════════════════════════════════

def plot_attention_map(attn_matrix, n_obs, layer_idx=0, head_idx=0, prefix=""):
    """
    Heatmap: 2n×2n attention matrix with observation/chain regions annotated.
    attn_matrix: (2n, 2n) numpy array
    """
    n = n_obs
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(attn_matrix, cmap="Blues", vmin=0)

    # Draw region boundaries
    ax.axhline(n - 0.5, color="red", linewidth=2, linestyle="--")
    ax.axvline(n - 0.5, color="red", linewidth=2, linestyle="--")

    # Labels
    ax.set_xlabel("Key (attending to)")
    ax.set_ylabel("Query (attending from)")
    ax.set_title(f"Attention Map (Layer {layer_idx}, Head {head_idx})")

    # Region annotations
    ax.text(n/2, n/2, "Obs→Obs", ha="center", va="center", fontsize=12,
            color="red", fontweight="bold", alpha=0.5)
    ax.text(n + n/2, n/2, "Obs→Chain\n(blocked)", ha="center", va="center",
            fontsize=10, color="gray", alpha=0.5)
    ax.text(n/2, n + n/2, "Chain→Obs\n(dist. inference)", ha="center", va="center",
            fontsize=10, color="red", fontweight="bold", alpha=0.5)
    ax.text(n + n/2, n + n/2, "Chain→Chain\n(DP recurrence)", ha="center", va="center",
            fontsize=10, color="red", fontweight="bold", alpha=0.5)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    _savefig(fig, f"{prefix}exp5_attention_L{layer_idx}_H{head_idx}.png")


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 6: Depth scaling
# ═══════════════════════════════════════════════════════════════════════════

def plot_depth_scaling(layer_counts, means, stds, metric_name, prefix=""):
    """
    Line plot: x=number of layers, y=metric, with error bars.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(layer_counts, means, "o-", color="#43A047", markersize=8, linewidth=2)
    ax.fill_between(layer_counts,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.2, color=COLORS["learned"])
    ax.set_xlabel("Number of Layers L")
    ax.set_ylabel(metric_name)
    ax.set_title("Experiment 6: Depth Scaling — Does Chain-of-Thought Compensate for Fewer Layers?")
    ax.set_xticks(layer_counts)
    fig.tight_layout()
    _savefig(fig, f"{prefix}exp6_depth_scaling.png")


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 7: Consistency-robustness frontier
# ═══════════════════════════════════════════════════════════════════════════

def plot_frontier(params, curves, param_name, metric_name, prefix=""):
    """
    Line plot: x=robustness parameter, y=metric.
    curves: dict config_name -> (means, stds)
    Four configurations: standard, standard+robust, chain, chain+robust
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Pastel line colors
    _LINE_PASTELS = ["#42A5F5", "#66BB6A", "#AB47BC", "#26A69A",
                     "#7E57C2", "#29B6F6", "#5C6BC0", "#26C6DA",
                     "#8D6E63", "#EC407A"]
    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]

    for idx, (name, (means, stds)) in enumerate(curves.items()):
        color = _LINE_PASTELS[idx % len(_LINE_PASTELS)]
        marker = markers[idx % len(markers)]
        linestyle = "--" if "robust" in name.lower() else "-"
        ax.plot(params, means, f"{marker}{linestyle}", label=name, color=color, markersize=6)
        ax.fill_between(params,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.1, color=color)

    ax.set_xlabel(param_name)
    ax.set_ylabel(metric_name)
    ax.set_title("Experiment 7: Consistency–Robustness Frontier — Which Training Yields the Best Tradeoff?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, f"{prefix}exp7_frontier.png")


# ═══════════════════════════════════════════════════════════════════════════
# Training curves (appendix)
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_curves(logs, config_name, prefix="", all_seed_logs=None):
    """
    Two-panel plot: total loss + decomposition (L_value, L_action, L_chain).
    logs: list of epoch dicts from train() (single seed, used as mean if all_seed_logs not given)
    all_seed_logs: list of lists of epoch dicts (one per seed). If provided,
                   plots mean with ±1 std shading across seeds.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if all_seed_logs is not None and len(all_seed_logs) > 1:
        # Multi-seed: compute mean ± std across seeds
        n_epochs = len(all_seed_logs[0])
        epochs = [all_seed_logs[0][e]["epoch"] for e in range(n_epochs)]

        def _extract(key):
            arr = np.array([[sl[e].get(key, 0) for e in range(n_epochs)] for sl in all_seed_logs])
            return arr.mean(axis=0), arr.std(axis=0)

        train_mean, train_std = _extract("train_total")
        val_mean, val_std = _extract("val_total")

        ax1.plot(epochs, train_mean, label="Train", color="#42A5F5")
        ax1.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                         alpha=0.15, color="#42A5F5")
        ax1.plot(epochs, val_mean, label="Val", color="#AB47BC")
        ax1.fill_between(epochs, val_mean - val_std, val_mean + val_std,
                         alpha=0.15, color="#AB47BC")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Total Loss")
        ax1.set_title(f"Training Curves: {config_name} ({len(all_seed_logs)} seeds)")
        ax1.legend(fontsize=8)

        comp_key = "train_C" if "train_C" in all_seed_logs[0][0] else "train_J"
        for key, label, color in [(comp_key, "L_value", "#42A5F5"),
                                   ("train_a", "L_action", "#7E57C2"),
                                   ("train_chain", "L_chain", "#66BB6A")]:
            if key not in all_seed_logs[0][0]:
                continue
            m, s = _extract(key)
            ax2.plot(epochs, m, label=label, color=color)
            ax2.fill_between(epochs, m - s, m + s, alpha=0.15, color=color)
    else:
        # Single seed: plain lines
        epochs = [l["epoch"] for l in logs]
        train_total = [l["train_total"] for l in logs]
        val_total = [l["val_total"] for l in logs]

        ax1.plot(epochs, train_total, label="Train", color="#42A5F5")
        ax1.plot(epochs, val_total, label="Val", color="#AB47BC")
        best_epoch = min(logs, key=lambda l: l["val_total"])["epoch"]
        ax1.axvline(best_epoch, color="#AB47BC", linestyle="--", alpha=0.5,
                    label=f"Best epoch {best_epoch}")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Total Loss")
        ax1.set_title(f"Training Curves: {config_name}")
        ax1.legend(fontsize=8)

        comp_key = "train_C" if "train_C" in logs[0] else "train_J"
        ax2.plot(epochs, [l[comp_key] for l in logs], label="L_value", color="#42A5F5")
        ax2.plot(epochs, [l["train_a"] for l in logs], label="L_action", color="#7E57C2")
        if "train_chain" in logs[0]:
            ax2.plot(epochs, [l["train_chain"] for l in logs], label="L_chain", color="#66BB6A")

    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Component Loss")
    ax2.set_title("Loss Decomposition")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    safe_name = config_name.replace(' ', '_').replace('/', '_')
    _savefig(fig, f"{prefix}training_curve_{safe_name}.png")


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Available plotting functions:")
    print("  plot_in_distribution_bars()   — Standard: in-dist bar chart")
    print("  plot_per_family_heatmap()     — Standard: per-family heatmap")
    print("  plot_weight_sweep()           — Exp 1: loss weighting ablation")
    print("  plot_chain_comparison()       — Exp 2: chain supervision comparison")
    print("  plot_robust_sweep()           — Exp 3: robust-aware training sweep")
    print("  plot_horizon_generalization() — Exp 4: horizon generalization lines")
    print("  plot_cost_ratio_generalization() — Exp 4: cost-ratio generalization")
    print("  plot_attention_map()          — Exp 5: attention heatmap")
    print("  plot_depth_scaling()          — Exp 6: depth scaling")
    print("  plot_frontier()              — Exp 7: consistency-robustness frontier")
    print("  plot_training_curves()        — Appendix: training curves")
    print(f"\nOutput directory: {OUT_DIR}")
