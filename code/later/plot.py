"""
Generate all figures for both optimal stopping and ski rental experiments.

Multi-seed aware: aggregates across the `seed` column in CSVs.
  - Bar charts with 95% CI error bars
  - Line charts → mean line + shaded ±1 std region
  - Heatmaps → mean values across seeds, best per column bolded

Saves into out_dir/plots/.
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


_PALETTE = {
    "offline": "#BDBDBD",
    "dp": "#1E88E5",
    "learned": "#43A047",
    "deterministic": "#FFB300",
    "offline_opt": "#BDBDBD",
}

# Gradient palettes: light → dark for distinguishing variants
_ROBUST_SHADES = ["#FFCCBC", "#FFAB91", "#FF8A65", "#FF7043", "#E64A19", "#BF360C"]
_DYNKIN_SHADES = ["#E1BEE7", "#CE93D8", "#BA68C8", "#AB47BC", "#8E24AA", "#6A1B9A"]

_STYLE = {
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "figure.dpi": 150,
}

_PASTEL_RDYLGN = LinearSegmentedColormap.from_list(
    "pastel_rdylgn", ["#EF9A9A", "#FFF59D", "#A5D6A7"])

_LOSS_CMAP = LinearSegmentedColormap.from_list(
    "loss", ["#A5D6A7", "#FFF59D", "#EF9A9A"])


def _assign_colors(policies):
    """Assign distinct colors, with gradients for robust/dynkin variants."""
    n_robust = sum(1 for p in policies if p.startswith("robust"))
    n_dynkin = sum(1 for p in policies if p.startswith("dynkin"))
    ri, di = 0, 0
    colors = []
    for p in policies:
        if p.startswith("robust"):
            idx = min(ri, len(_ROBUST_SHADES) - 1)
            # spread across available shades
            if n_robust > 1:
                idx = int(ri * (len(_ROBUST_SHADES) - 1) / max(n_robust - 1, 1))
            colors.append(_ROBUST_SHADES[idx])
            ri += 1
        elif p.startswith("dynkin"):
            idx = min(di, len(_DYNKIN_SHADES) - 1)
            if n_dynkin > 1:
                idx = int(di * (len(_DYNKIN_SHADES) - 1) / max(n_dynkin - 1, 1))
            colors.append(_DYNKIN_SHADES[idx])
            di += 1
        elif p == "dp":
            colors.append(_PALETTE["dp"])
        elif p == "learned":
            colors.append(_PALETTE["learned"])
        elif p == "deterministic":
            colors.append(_PALETTE["deterministic"])
        elif p in ("offline", "offline_opt"):
            colors.append(_PALETTE["offline"])
        else:
            colors.append("#B0BEC5")
    return colors


def _short_name(name):
    if name == "offline": return "Prophet"
    if name == "offline_opt": return "Offline OPT"
    if name == "dp": return "Bayes DP"
    if name == "learned": return "Transformer"
    if name == "deterministic": return "Deterministic"
    if name.startswith("robust β="): return f"Robust β={name.split('=')[-1]}"
    if name.startswith("robust λ="): return f"Robust λ={name.split('=')[-1]}"
    if name.startswith("dynkin β="): return f"Dynkin β={name.split('=')[-1]}"
    return name


def _read_csv(path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(d, key, default=0.0):
    try:
        return float(d[key])
    except (KeyError, ValueError, TypeError):
        return default


def _save(fig, path):
    fig.savefig(str(path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    [plot] {path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Multi-seed aggregation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _ci95(values):
    """95% confidence interval half-width."""
    n = len(values)
    if n <= 1:
        return 0.0
    std = np.std(values, ddof=1)
    t_crit = 2.262 if n <= 10 else 2.045 if n <= 30 else 1.96
    return t_crit * std / np.sqrt(n)


def _agg_by_policy(rows, metric):
    """Aggregate metric by policy across seeds. Returns (policies, means, cis) in order."""
    groups = defaultdict(list)
    policy_order = list(dict.fromkeys(r["policy"] for r in rows))
    for r in rows:
        groups[r["policy"]].append(_f(r, metric))
    means = [np.mean(groups[p]) for p in policy_order]
    cis = [_ci95(groups[p]) for p in policy_order]
    return policy_order, means, cis


# ═══════════════════════════════════════════════════════════════════════════
# Exp 1 bar chart helper (shared by stopping and ski)
# ═══════════════════════════════════════════════════════════════════════════

def _plot_exp1_bars(ax, policies, means, cis, colors, metric_label,
                    dp_as_hline=True, higher_is_better=True):
    """
    Bar chart for Exp 1.
    - DP shown as horizontal blue dashed line, not a bar.
    - Prophet/Offline OPT also shown as horizontal dashed line.
    """
    bar_policies, bar_means, bar_cis, bar_colors = [], [], [], []
    hlines = []  # (value, color, label)
    for p, m, ci, c in zip(policies, means, cis, colors):
        if dp_as_hline and p == "dp":
            hlines.append((m, _PALETTE["dp"], f"Bayes DP ({m:.3f})"))
        elif p in ("offline", "offline_opt"):
            lbl = "Prophet" if p == "offline" else "Offline OPT"
            hlines.append((m, "#9E9E9E", f"{lbl} ({m:.3f})"))
        else:
            bar_policies.append(p)
            bar_means.append(m)
            bar_cis.append(ci)
            bar_colors.append(c)

    x = np.arange(len(bar_policies))
    bars = ax.bar(x, bar_means, color=bar_colors, width=0.6, edgecolor="white",
                  linewidth=0.8, zorder=3)
    ax.errorbar(x, bar_means, yerr=bar_cis, fmt="none",
                capsize=4, capthick=1.5, elinewidth=1.5,
                color="black", zorder=5)
    for i, (m, ci) in enumerate(zip(bar_means, bar_cis)):
        ax.text(i, m + ci + max(bar_means) * 0.015, f"{m:.3f}",
                ha="center", va="bottom", fontsize=7)

    for val, col, lbl in hlines:
        ax.axhline(val, color=col, ls="--", lw=2, label=lbl, zorder=4)

    labels = [_short_name(p) for p in bar_policies]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(metric_label)
    if higher_is_better:
        ax.set_ylim(0, max(bar_means + [h[0] for h in hlines]) * 1.18)
    ax.legend(fontsize=8)


# ═══════════════════════════════════════════════════════════════════════════
# Heatmap helper with bolded best values
# ═══════════════════════════════════════════════════════════════════════════

def _plot_heatmap(ax, matrix, xlabels, ylabels, cmap, vmin, vmax, label,
                  lower_is_better=False, fmt=".2f"):
    """Plot heatmap with best value per column bolded."""
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label=label, shrink=0.8)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)

    # Find best per column
    for fi in range(matrix.shape[1]):
        col = matrix[:, fi]
        if lower_is_better:
            best_val = np.min(col)
        else:
            best_val = np.max(col)
        for pi in range(matrix.shape[0]):
            is_best = abs(matrix[pi, fi] - best_val) < 1e-6
            ax.text(fi, pi, f"{matrix[pi, fi]:{fmt}}", ha="center", va="center",
                    fontsize=8, fontweight="bold" if is_best else "normal")


# ═══════════════════════════════════════════════════════════════════════════
# Stopping plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_stopping_training(out_dir, plots_dir):
    rows = _read_csv(out_dir / "stopping" / "training_log.csv")
    if not rows:
        return
    epochs = [int(r["epoch"]) for r in rows]
    tr_tot = [_f(r, "train_total") for r in rows]
    tr_C = [_f(r, "train_C") for r in rows]
    tr_a = [_f(r, "train_a") for r in rows]
    val = [_f(r, "val_total") for r in rows]
    best_epoch = epochs[val.index(min(val))]

    with plt.rc_context(_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(epochs, tr_tot, color=_PALETTE["dp"], lw=2, label="Train")
        ax1.plot(epochs, val, color=_PALETTE["offline"], lw=2, ls="--", label="Val")
        ax1.axvline(best_epoch, color="#E57373", ls=":", lw=1.5, label=f"Best (epoch {best_epoch})")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.set_title("Stopping — Total Loss"); ax1.legend(fontsize=8)
        ax2.plot(epochs, tr_C, color=_PALETTE["dp"], lw=2, label="C (MSE)")
        ax2.plot(epochs, tr_a, color="#FF8A65", lw=2, label="a (CE)")
        ax2.axvline(best_epoch, color="#E57373", ls=":", lw=1.5)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
        ax2.set_title("Stopping — Loss Breakdown"); ax2.legend(fontsize=8)
        fig.tight_layout()
        _save(fig, plots_dir / "training_curve_stopping.png")


def plot_in_distribution(out_dir, plots_dir):
    rows = _read_csv(out_dir / "stopping" / "exp1_in_distribution.csv")
    if not rows:
        return
    policies, cr_means, cr_cis = _agg_by_policy(rows, "cr")
    _, pb_means, pb_cis = _agg_by_policy(rows, "prob_best")
    colors = _assign_colors(policies)

    n_val = int(_f(rows[0], "n", 0))
    with plt.rc_context(_STYLE):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
        _plot_exp1_bars(ax1, policies, cr_means, cr_cis, colors, "CR")
        ax1.set_title(f"Exp 1 — In-Distribution CR (n={n_val})")

        _plot_exp1_bars(ax2, policies, pb_means, pb_cis, colors, "P(best)")
        ax2.set_title(f"Exp 1 — In-Distribution P(best) (n={n_val})")
        fig.tight_layout()
        _save(fig, plots_dir / "exp1_cr_bar.png")


def plot_family_heatmap(out_dir, plots_dir):
    rows = _read_csv(out_dir / "stopping" / "exp2_by_family.csv")
    if not rows:
        return
    families = list(dict.fromkeys(r["family"] for r in rows))
    policies = list(dict.fromkeys(r["policy"] for r in rows))

    for metric, suffix, label in [("cr", "", "CR"), ("prob_best", "_pbest", "P(best)")]:
        agg = defaultdict(list)
        for r in rows:
            agg[(r["policy"], r["family"])].append(_f(r, metric))
        matrix = np.zeros((len(policies), len(families)))
        for pi, p in enumerate(policies):
            for fi, f in enumerate(families):
                vals = agg.get((p, f), [0.0])
                matrix[pi, fi] = np.mean(vals)

        ylabels = [_short_name(p) for p in policies]
        with plt.rc_context({**_STYLE, "axes.grid": False}):
            fig, ax = plt.subplots(figsize=(max(8, len(families)*1.1), max(4, len(policies)*0.45)))
            _plot_heatmap(ax, matrix, families, ylabels, _PASTEL_RDYLGN,
                          0.0, 1.0, label, lower_is_better=False)
            n_val = int(_f(rows[0], "n", 0))
            ax.set_title(f"Exp 2 — {label} by Family × Policy (n={n_val})", fontsize=11)
            fig.tight_layout()
            _save(fig, plots_dir / f"exp2_family{suffix}_heatmap.png")


def plot_horizon(out_dir, plots_dir):
    rows = _read_csv(out_dir / "stopping" / "exp4_horizon.csv")
    if not rows:
        return
    train_n = _f(rows[0], "train_n") if rows else None

    policies = list(dict.fromkeys(r["policy"] for r in rows))
    test_ns = sorted(set(_f(r, "test_n") for r in rows))
    colors = _assign_colors(policies)

    agg = defaultdict(list)
    for r in rows:
        agg[(r["policy"], _f(r, "test_n"))].append(_f(r, "cr"))

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        for policy, color in zip(policies, colors):
            xs, ys_mean, ys_std = [], [], []
            for tn in test_ns:
                vals = agg.get((policy, tn), [])
                if vals:
                    xs.append(tn)
                    ys_mean.append(np.mean(vals))
                    ys_std.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            if not xs:
                continue
            xs, ys_mean, ys_std = np.array(xs), np.array(ys_mean), np.array(ys_std)
            ax.plot(xs, ys_mean, color=color, lw=2, marker="o", markersize=5,
                    label=_short_name(policy))
            ax.fill_between(xs, ys_mean - ys_std, ys_mean + ys_std,
                            color=color, alpha=0.18)
        if train_n:
            ax.axvline(train_n, color="#E57373", ls="--", lw=1.8, label=f"Train n={int(train_n)}")
        ax.set_xlabel("Test horizon n"); ax.set_ylabel("CR")
        ax.set_title("Exp 4 — Horizon Generalization")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        _save(fig, plots_dir / "exp4_horizon_lines.png")


def plot_robustness_sweep(out_dir, plots_dir):
    rows = _read_csv(out_dir / "stopping" / "exp5_robustness_sweep.csv")
    if not rows:
        return

    agg_robust = defaultdict(list)
    agg_dynkin = defaultdict(list)
    agg_learned = defaultdict(list)
    for r in rows:
        beta = _f(r, "beta")
        agg_robust[beta].append(_f(r, "cr_robust"))
        agg_dynkin[beta].append(_f(r, "cr_dynkin"))
        agg_learned[beta].append(_f(r, "cr_learned"))

    betas = sorted(agg_robust.keys())
    rob_mean = np.array([np.mean(agg_robust[b]) for b in betas])
    rob_std = np.array([np.std(agg_robust[b], ddof=1) if len(agg_robust[b]) > 1 else 0.0 for b in betas])
    dyn_mean = np.array([np.mean(agg_dynkin[b]) for b in betas])
    dyn_std = np.array([np.std(agg_dynkin[b], ddof=1) if len(agg_dynkin[b]) > 1 else 0.0 for b in betas])
    all_learned = []
    for b in betas:
        all_learned.extend(agg_learned[b])
    cr_learned = np.mean(all_learned)

    betas = np.array(betas)
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        all_cr = list(rob_mean) + list(dyn_mean) + [cr_learned]
        cr_min, cr_max = min(all_cr) - 0.03, max(all_cr) + 0.04
        ax.set_ylim(cr_min, cr_max)
        beta_fill = sorted(set(list(betas) + [0.0, 1/np.e]))
        ax.fill_between(beta_fill, beta_fill, cr_min, alpha=0.12, color="#FF8A65",
                         label="Certified floor (CR ≥ β)")
        ax.axhline(cr_learned, color=_PALETTE["learned"], ls="--", lw=2,
                   label=f"Transformer CR={cr_learned:.3f}")

        ax.plot(betas, rob_mean, color="#E64A19", lw=2.5, marker="o", markersize=6, label="Robust (Alg 1)")
        ax.fill_between(betas, rob_mean - rob_std, rob_mean + rob_std,
                        color="#E64A19", alpha=0.18)

        ax.plot(betas, dyn_mean, color="#8E24AA", lw=2.5, marker="s", markersize=6, label="Dynkin")
        ax.fill_between(betas, dyn_mean - dyn_std, dyn_mean + dyn_std,
                        color="#8E24AA", alpha=0.18)

        ax.set_xlabel("β = −(r/n)ln(r/n)"); ax.set_ylabel("CR")
        n_val = int(_f(rows[0], "n", 0))
        ax.set_title(f"Exp 5 — Robustness–Consistency (n={n_val})")
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        _save(fig, plots_dir / "exp5_robustness_curve.png")


def plot_heavy_tail(out_dir, plots_dir):
    rows = _read_csv(out_dir / "stopping" / "exp6_heavy_tail.csv")
    if not rows:
        return
    families = list(dict.fromkeys(r["family"] for r in rows))
    policies = list(dict.fromkeys(r["policy"] for r in rows))

    for metric, suffix, label in [("cr", "", "CR"), ("prob_best", "_pbest", "P(best)")]:
        agg = defaultdict(list)
        for r in rows:
            agg[(r["policy"], r["family"])].append(_f(r, metric))
        matrix = np.zeros((len(policies), len(families)))
        for pi, p in enumerate(policies):
            for fi, f in enumerate(families):
                vals = agg.get((p, f), [0.0])
                matrix[pi, fi] = np.mean(vals)
        ylabels = [_short_name(p) for p in policies]
        family_labels = [f.replace("_", " ").title() for f in families]
        with plt.rc_context({**_STYLE, "axes.grid": False}):
            fig, ax = plt.subplots(figsize=(max(6, len(families)*1.8), max(4, len(policies)*0.45)))
            _plot_heatmap(ax, matrix, family_labels, ylabels, _PASTEL_RDYLGN,
                          0.0, 1.0, label, lower_is_better=False)
            n_val = int(_f(rows[0], "n", 0))
            ax.set_title(f"Exp 6 — Heavy-Tail {label} (n={n_val})", fontsize=11)
            fig.tight_layout()
            _save(fig, plots_dir / f"exp6_heavy_tail{suffix}_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
# Ski rental plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_ski_training(out_dir, plots_dir):
    rows = _read_csv(out_dir / "ski" / "training_log.csv")
    if not rows:
        return
    epochs = [int(r["epoch"]) for r in rows]
    tr_tot = [_f(r, "train_total") for r in rows]
    tr_J = [_f(r, "train_J") for r in rows]
    tr_a = [_f(r, "train_a") for r in rows]
    val = [_f(r, "val_total") for r in rows]
    best_epoch = epochs[val.index(min(val))]

    with plt.rc_context(_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(epochs, tr_tot, color=_PALETTE["dp"], lw=2, label="Train")
        ax1.plot(epochs, val, color=_PALETTE["offline"], lw=2, ls="--", label="Val")
        ax1.axvline(best_epoch, color="#E57373", ls=":", lw=1.5, label=f"Best (epoch {best_epoch})")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.set_title("Ski Rental — Total Loss"); ax1.legend(fontsize=8)
        ax2.plot(epochs, tr_J, color=_PALETTE["dp"], lw=2, label="J (MSE)")
        ax2.plot(epochs, tr_a, color="#FF8A65", lw=2, label="a (CE)")
        ax2.axvline(best_epoch, color="#E57373", ls=":", lw=1.5)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
        ax2.set_title("Ski Rental — Loss Breakdown"); ax2.legend(fontsize=8)
        fig.tight_layout()
        _save(fig, plots_dir / "training_curve_ski.png")


def plot_ski_in_distribution(out_dir, plots_dir):
    rows = _read_csv(out_dir / "ski" / "exp1_in_distribution.csv")
    if not rows:
        return
    policies, loss_means, loss_cis = _agg_by_policy(rows, "mean_additive_loss")
    colors = _assign_colors(policies)

    n_val = int(_f(rows[0], "n", 0))
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))

        # Separate DP and offline_opt as horizontal lines; rest as bars
        bar_policies, bar_means, bar_cis, bar_colors = [], [], [], []
        hlines = []
        for p, m, ci, c in zip(policies, loss_means, loss_cis, colors):
            if p == "dp":
                hlines.append((m, _PALETTE["dp"], f"Bayes DP ({m:.3f})"))
            elif p == "offline_opt":
                hlines.append((m, "#9E9E9E", f"Offline OPT ({m:.3f})"))
            else:
                bar_policies.append(p)
                bar_means.append(m)
                bar_cis.append(ci)
                bar_colors.append(c)

        x = np.arange(len(bar_policies))
        ax.bar(x, bar_means, color=bar_colors, width=0.6, edgecolor="white",
               linewidth=0.8, zorder=3)
        ax.errorbar(x, bar_means, yerr=bar_cis, fmt="none",
                    capsize=4, capthick=1.5, elinewidth=1.5,
                    color="black", zorder=5)
        for i, (m, ci) in enumerate(zip(bar_means, bar_cis)):
            offset = max(bar_means) * 0.015 if bar_means else 0.01
            ax.text(i, m + ci + offset, f"{m:.3f}", ha="center", va="bottom", fontsize=7)
        for val, col, lbl in hlines:
            ax.axhline(val, color=col, ls="--", lw=2, label=lbl, zorder=4)

        labels = [_short_name(p) for p in bar_policies]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Additive Loss Δ(P)")
        ax.set_title(f"Exp 1 — Ski Rental In-Distribution (n={n_val})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        _save(fig, plots_dir / "exp1_ski_bar.png")


def plot_ski_family_heatmap(out_dir, plots_dir):
    rows = _read_csv(out_dir / "ski" / "exp2_by_family.csv")
    if not rows:
        return
    families = list(dict.fromkeys(r["family"] for r in rows))
    policies = list(dict.fromkeys(r["policy"] for r in rows))

    agg = defaultdict(list)
    for r in rows:
        agg[(r["policy"], r["family"])].append(_f(r, "mean_additive_loss"))
    matrix = np.zeros((len(policies), len(families)))
    for pi, p in enumerate(policies):
        for fi, f in enumerate(families):
            vals = agg.get((p, f), [0.0])
            matrix[pi, fi] = np.mean(vals)

    ylabels = [_short_name(p) for p in policies]
    family_labels = [f.replace("ski_", "").replace("_", " ").title() for f in families]
    vmax = max(1.0, np.percentile(matrix, 95))

    with plt.rc_context({**_STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(max(8, len(families)*1.1), max(4, len(policies)*0.45)))
        _plot_heatmap(ax, matrix, family_labels, ylabels, _LOSS_CMAP,
                      0.0, vmax, "Additive Loss", lower_is_better=True)
        n_val = int(_f(rows[0], "n", 0))
        ax.set_title(f"Exp 2 — Additive Loss by Family (n={n_val})", fontsize=11)
        fig.tight_layout()
        _save(fig, plots_dir / "exp2_ski_family_heatmap.png")


def plot_ski_frontier(out_dir, plots_dir):
    rows = _read_csv(out_dir / "ski" / "exp3_frontier.csv")
    if not rows:
        return

    agg_robust = defaultdict(list)
    agg_det = defaultdict(list)
    agg_learned = defaultdict(list)
    for r in rows:
        lam = _f(r, "lambda")
        agg_robust[lam].append(_f(r, "loss_robust"))
        agg_det[lam].append(_f(r, "loss_deterministic"))
        agg_learned[lam].append(_f(r, "loss_learned"))

    lambdas = np.array(sorted(agg_robust.keys()))
    rob_mean = np.array([np.mean(agg_robust[l]) for l in lambdas])
    rob_std = np.array([np.std(agg_robust[l], ddof=1) if len(agg_robust[l]) > 1 else 0.0 for l in lambdas])
    all_det, all_learned = [], []
    for l in lambdas:
        all_det.extend(agg_det[l])
        all_learned.extend(agg_learned[l])
    loss_det = np.mean(all_det)
    loss_learned = np.mean(all_learned)

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(lambdas, rob_mean, color="#E64A19", lw=2.5, marker="o", markersize=6,
                label="Robust wrapper")
        ax.fill_between(lambdas, rob_mean - rob_std, rob_mean + rob_std,
                        color="#E64A19", alpha=0.18)
        ax.axhline(loss_det, color=_PALETTE["deterministic"], ls="--", lw=2,
                   label=f"Deterministic Δ={loss_det:.3f}")
        ax.axhline(loss_learned, color=_PALETTE["learned"], ls="--", lw=2,
                   label=f"Transformer Δ={loss_learned:.3f}")
        ax.set_xlabel("λ (robustness parameter)")
        ax.set_ylabel("Additive Loss Δ(P)")
        n_val = int(_f(rows[0], "n", 0))
        ax.set_title(f"Exp 3 — Consistency–Robustness Frontier (n={n_val})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        _save(fig, plots_dir / "exp3_ski_frontier.png")


def plot_ski_cost_ratio(out_dir, plots_dir):
    rows = _read_csv(out_dir / "ski" / "exp4_cost_ratio.csv")
    if not rows:
        return
    policies = list(dict.fromkeys(r["policy"] for r in rows))
    ratios = sorted(set(_f(r, "B_over_r") for r in rows))
    colors = _assign_colors(policies)

    agg = defaultdict(list)
    for r in rows:
        agg[(r["policy"], _f(r, "B_over_r"))].append(_f(r, "mean_additive_loss"))

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        for policy, color in zip(policies, colors):
            xs, ys_mean, ys_std = [], [], []
            for ratio in ratios:
                vals = agg.get((policy, ratio), [])
                if vals:
                    xs.append(ratio)
                    ys_mean.append(np.mean(vals))
                    ys_std.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            if not xs:
                continue
            xs, ys_mean, ys_std = np.array(xs), np.array(ys_mean), np.array(ys_std)
            ax.plot(xs, ys_mean, color=color, lw=2, marker="o", markersize=5,
                    label=_short_name(policy))
            ax.fill_between(xs, ys_mean - ys_std, ys_mean + ys_std,
                            color=color, alpha=0.18)
        ax.set_xlabel("B/r (cost ratio)")
        ax.set_ylabel("Additive Loss")
        ax.set_title("Exp 4 — Cost-Ratio Sensitivity")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        _save(fig, plots_dir / "exp4_ski_cost_ratio.png")


def plot_ski_heavy_tail(out_dir, plots_dir):
    rows = _read_csv(out_dir / "ski" / "exp5_heavy_tail.csv")
    if not rows:
        return
    families = list(dict.fromkeys(r["family"] for r in rows))
    policies = list(dict.fromkeys(r["policy"] for r in rows))

    agg = defaultdict(list)
    for r in rows:
        agg[(r["policy"], r["family"])].append(_f(r, "mean_additive_loss"))
    matrix = np.zeros((len(policies), len(families)))
    for pi, p in enumerate(policies):
        for fi, f in enumerate(families):
            vals = agg.get((p, f), [0.0])
            matrix[pi, fi] = np.mean(vals)

    ylabels = [_short_name(p) for p in policies]
    family_labels = [f.replace("ski_", "").replace("_", " ").title() for f in families]
    vmax = max(1.0, np.percentile(matrix, 95))

    with plt.rc_context({**_STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(max(6, len(families)*1.8), max(4, len(policies)*0.45)))
        _plot_heatmap(ax, matrix, family_labels, ylabels, _LOSS_CMAP,
                      0.0, vmax, "Additive Loss", lower_is_better=True)
        n_val = int(_f(rows[0], "n", 0))
        ax.set_title(f"Exp 5 — Heavy-Tail Additive Loss (n={n_val})", fontsize=11)
        fig.tight_layout()
        _save(fig, plots_dir / "exp5_ski_heavy_tail.png")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_plots(out_dir):
    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"  Generating plots → {plots_dir}")

    # Stopping
    plot_stopping_training(out_dir, plots_dir)
    plot_in_distribution(out_dir, plots_dir)
    plot_family_heatmap(out_dir, plots_dir)
    plot_horizon(out_dir, plots_dir)
    plot_robustness_sweep(out_dir, plots_dir)
    plot_heavy_tail(out_dir, plots_dir)

    # Ski rental
    plot_ski_training(out_dir, plots_dir)
    plot_ski_in_distribution(out_dir, plots_dir)
    plot_ski_family_heatmap(out_dir, plots_dir)
    plot_ski_frontier(out_dir, plots_dir)
    plot_ski_cost_ratio(out_dir, plots_dir)
    plot_ski_heavy_tail(out_dir, plots_dir)

    print(f"  All plots saved in {plots_dir}/")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot.py results/run_<timestamp>/")
        sys.exit(1)
    generate_all_plots(Path(sys.argv[1]))
