"""
Result visualizations for the 6-model sweep — regime-coloured edition.

Reads the existing per-model JSON eval files; emits 3 PNGs to
sweep/experiments/figures/. No retraining, no re-evaluation.

Color scheme. Each dataset gets a regime-emphasising hue:
    D_1 (rho = 0.1, prior weight ~9%)  — "data-dominant"   peach   #F4A6A6
    D_2 (rho = 1.0, prior weight 50%)  — "balanced"        beige   #F4D88A
    D_3 (rho = 10,  prior weight ~91%) — "prior-dominant"  blue    #A6C8F4

Within each hue family, saturation/edge style separates policies:
    bayes_optimal : full saturation
    plug_in       : 50% mixed with white
    prior_only    : 70% mixed with white
    model         : white fill + regime-color edge (cv = solid, act = hatched)

Axis labels and subplot titles use regime names, not bare D_i labels —
the visual mapping is what carries the story (red→data, beige→between,
blue→prior).

Figures:
  fig_ood_heatmap.png       — payoff gap to oracle, per (supervision, train).
                               Cell colour stays diverging RdBu (gap is
                               directional). Subplot titles tinted by train
                               regime; eval-axis labels tinted by eval regime.
  fig_payoff_bars.png       — grouped bar chart per (train, eval) cell of
                               BO / plug_in / prior_only / model_cv / model_act.
                               Bar colour = eval-regime hue; saturation by
                               policy.
  fig_agreement_matrix.png  — 2×3 of bar groups; per-step agreement with
                               BO / plug-in / prior-only at each eval
                               distribution. Same regime-tint scheme.
"""

import json
from pathlib import Path

import matplotlib.colors as mc
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


import config
SWEEP = config.SWEEP_ROOT
OUTDIR = SWEEP / "experiments" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "D1_cv":  (SWEEP / "D1_cv/experiments/D1_cv",   1, "cv"),
    "D2_cv":  (SWEEP / "D2_cv/experiments/D2_cv",   2, "cv"),
    "D3_cv":  (SWEEP / "D3_cv/experiments/D3_cv",   3, "cv"),
    "D1_act": (SWEEP / "D1_act/experiments/D1_act", 1, "act"),
    "D2_act": (SWEEP / "D2_act/experiments/D2_act", 2, "act"),
    "D3_act": (SWEEP / "D3_act/experiments/D3_act", 3, "act"),
}

# ---------------------------------------------------------------------------
# Regime palette
# ---------------------------------------------------------------------------

REGIME = {
    1: {"name": "data-dominant",   "rho": "ρ=0.1",  "color": "#F4A6A6"},
    2: {"name": "balanced",        "rho": "ρ=1",    "color": "#F4D88A"},
    3: {"name": "prior-dominant",  "rho": "ρ=10",   "color": "#A6C8F4"},
}


def regime_label(i: int, multiline: bool = False) -> str:
    r = REGIME[i]
    sep = "\n" if multiline else " "
    return f"{r['name']}{sep}({r['rho']})"


def regime_color(i: int) -> str:
    return REGIME[i]["color"]


def tint(color, frac: float):
    """Mix `frac` of white into `color`. frac=0 -> color, frac=1 -> white."""
    rgb = mc.to_rgb(color)
    return tuple(c + frac * (1.0 - c) for c in rgb)


POLICY_TINT = {
    "bayes_optimal": 0.0,
    "plug_in":       0.5,
    "prior_only":    0.7,
}


def policy_color(eval_i: int, policy_name: str):
    """Hue from eval regime, saturation by policy."""
    return tint(regime_color(eval_i), POLICY_TINT[policy_name])


def model_bar_kwargs(eval_i: int, sup: str):
    """Outlined bar style for model entries: white fill, regime edge.
    cv = solid edge; act = hatched edge."""
    base = regime_color(eval_i)
    if sup == "act":
        return dict(facecolor="white", edgecolor=base, linewidth=1.3,
                    hatch="///")
    return dict(facecolor="white", edgecolor=base, linewidth=1.6)


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

def load_eval(mid: str, eval_id: int) -> dict:
    pmd, train_id, _ = MODELS[mid]
    if eval_id == train_id:
        path = pmd / "indist_eval.json"
    else:
        path = pmd / f"ood_eval_D{eval_id}.json"
    with open(path) as f:
        return json.load(f)["test"]


# ---------------------------------------------------------------------------
# fig 1 — OOD payoff-gap heatmap
# ---------------------------------------------------------------------------

def fig_ood_heatmap():
    fig, axes = plt.subplots(2, 3, figsize=(14, 5.5))
    sups = ["cv", "act"]
    trains = [1, 2, 3]

    grid = np.full((2, 3, 3), np.nan)
    all_gaps = []
    for i, sup in enumerate(sups):
        for j, tr in enumerate(trains):
            mid = f"D{tr}_{sup}"
            for k, ev in enumerate(trains):
                m = load_eval(mid, ev)
                gap = m["model_payoff"] - m["baseline_payoffs"]["bayes_optimal"]
                grid[i, j, k] = gap
                all_gaps.append(gap)
    vabs = max(abs(min(all_gaps)), abs(max(all_gaps)))
    vmin, vmax = -vabs, vabs

    im = None
    for i, sup in enumerate(sups):
        for j, tr in enumerate(trains):
            ax = axes[i, j]
            im = ax.imshow(grid[i, j].reshape(1, 3), cmap="RdBu",
                           vmin=vmin, vmax=vmax, aspect="auto")
            # Eval-axis tick labels: regime-named, regime-coloured
            ax.set_xticks(range(3))
            ax.set_xticklabels(
                [regime_label(ev, multiline=True) for ev in trains],
                fontsize=9,
            )
            for tick_lbl, ev in zip(ax.get_xticklabels(), trains):
                tick_lbl.set_color(regime_color(ev))
                tick_lbl.set_fontweight("bold")
            ax.set_yticks([])
            for k, val in enumerate(grid[i, j]):
                txt = f"{val:+.3f}"
                if k == j:
                    txt += "\n(in-dist)"
                ax.text(k, 0, txt, ha="center", va="center",
                        color="black" if abs(val) < 0.4 * vabs else "white",
                        fontsize=10)
            # Subplot title: train regime + supervision; tinted with train hue.
            title = f"trained on {regime_label(tr)}  ({sup})"
            ax.set_title(title, fontsize=10, color=regime_color(tr),
                         fontweight="bold")

    fig.text(0.025, 0.72, "cv supervision", rotation="vertical",
             ha="center", va="center", fontsize=11, weight="bold")
    fig.text(0.025, 0.28, "act supervision", rotation="vertical",
             ha="center", va="center", fontsize=11, weight="bold")

    fig.suptitle(
        "OOD generalization gap: model_payoff − bayes_optimal_payoff on each "
        "eval distribution.\nIn-distribution cells sit at 0; OOD cells expose "
        "the asymmetric failure pattern, with prior-dominant → data-dominant "
        "the largest.",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.07, right=0.88, top=0.84, bottom=0.16,
                        hspace=0.95, wspace=0.18)
    cax = fig.add_axes([0.91, 0.16, 0.018, 0.68])
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label("model − BO payoff  (red = below oracle)", fontsize=9)

    out = OUTDIR / "fig_ood_heatmap.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# fig 2 — grouped payoff bars
# ---------------------------------------------------------------------------

def fig_payoff_bars():
    pairs = []
    for tr in (1, 2, 3):
        pairs.append((tr, tr, True))                            # in-dist first
    for tr in (1, 2, 3):
        for ev in (1, 2, 3):
            if ev != tr:
                pairs.append((tr, ev, False))

    fig, ax = plt.subplots(figsize=(17, 7))
    n_groups = len(pairs)
    n_bars = 5
    bar_w = 0.16
    group_w = (n_bars + 0.6) * bar_w
    xc = np.arange(n_groups) * group_w
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * bar_w

    series = ["bayes_optimal", "plug_in", "prior_only", "model_cv", "model_act"]

    for k, name in enumerate(series):
        for i, (tr, ev, _) in enumerate(pairs):
            if name == "model_cv":
                val = load_eval(f"D{tr}_cv", ev)["model_payoff"]
                kw = model_bar_kwargs(ev, "cv")
            elif name == "model_act":
                val = load_eval(f"D{tr}_act", ev)["model_payoff"]
                kw = model_bar_kwargs(ev, "act")
            else:
                val = load_eval(f"D{tr}_cv", ev)["baseline_payoffs"][name]
                kw = dict(color=policy_color(ev, name))
            ax.bar(xc[i] + offsets[k], val, width=bar_w, **kw)

    # X tick labels: train→eval, regime-coloured
    labels = [f"{regime_label(tr)}\n→ {regime_label(ev)}"
              + ("\n(in-dist)" if in_dist else "")
              for tr, ev, in_dist in pairs]
    ax.set_xticks(xc)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
    ax.set_ylabel("E[payoff]")
    ax.set_title(
        "Test-split expected payoff per policy and per (train, eval) cell.\n"
        "Bar fill = EVAL regime (peach / beige / blue); saturation falls from "
        "BO → plug-in → prior-only; model bars are outlined (cv solid, "
        "act hatched).",
        fontsize=10,
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 2.6)

    # Two legends: one for policy (saturation pattern), one for regime hue.
    # Use the balanced (D_2) hue as the demonstrator for the policy legend.
    legend_demo = 2
    policy_handles = [
        mpatches.Patch(color=policy_color(legend_demo, "bayes_optimal"),
                       label="BO (oracle)"),
        mpatches.Patch(color=policy_color(legend_demo, "plug_in"),
                       label="plug-in"),
        mpatches.Patch(color=policy_color(legend_demo, "prior_only"),
                       label="prior-only"),
        mpatches.Patch(facecolor="white",
                       edgecolor=regime_color(legend_demo), linewidth=1.6,
                       label="model (cv)"),
        mpatches.Patch(facecolor="white",
                       edgecolor=regime_color(legend_demo), linewidth=1.3,
                       hatch="///", label="model (act)"),
    ]
    regime_handles = [
        mpatches.Patch(color=regime_color(i),
                       label=f"{REGIME[i]['name']} ({REGIME[i]['rho']})")
        for i in (1, 2, 3)
    ]
    leg1 = ax.legend(handles=policy_handles, title="policy (within group)",
                     loc="upper right", fontsize=8, title_fontsize=9, ncol=1)
    ax.add_artist(leg1)
    ax.legend(handles=regime_handles,
              title="eval regime (bar fill hue)",
              loc="upper left", fontsize=8, title_fontsize=9, ncol=1)

    fig.tight_layout()
    out = OUTDIR / "fig_payoff_bars.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# fig 2b — same as fig 2 but normalised by the eval-distribution oracle
# ---------------------------------------------------------------------------

def fig_payoff_bars_normalized():
    pairs = []
    for tr in (1, 2, 3):
        pairs.append((tr, tr, True))
    for tr in (1, 2, 3):
        for ev in (1, 2, 3):
            if ev != tr:
                pairs.append((tr, ev, False))

    fig, ax = plt.subplots(figsize=(17, 7))
    n_groups = len(pairs)
    n_bars = 5
    bar_w = 0.16
    group_w = (n_bars + 0.6) * bar_w
    xc = np.arange(n_groups) * group_w
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * bar_w

    series = ["bayes_optimal", "plug_in", "prior_only", "model_cv", "model_act"]

    for k, name in enumerate(series):
        for i, (tr, ev, _) in enumerate(pairs):
            ev_data = load_eval(f"D{tr}_cv", ev)
            denom = ev_data["baseline_payoffs"]["bayes_optimal"]
            if name == "model_cv":
                val = load_eval(f"D{tr}_cv", ev)["model_payoff"] / denom
                kw = model_bar_kwargs(ev, "cv")
            elif name == "model_act":
                val = load_eval(f"D{tr}_act", ev)["model_payoff"] / denom
                kw = model_bar_kwargs(ev, "act")
            else:
                val = ev_data["baseline_payoffs"][name] / denom
                kw = dict(color=policy_color(ev, name))
            ax.bar(xc[i] + offsets[k], val, width=bar_w, **kw)

    # Reference line at 1.0 (oracle)
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.5,
               linestyle="--", zorder=0)

    labels = [f"{regime_label(tr)}\n→ {regime_label(ev)}"
              + ("\n(in-dist)" if in_dist else "")
              for tr, ev, in_dist in pairs]
    ax.set_xticks(xc)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
    ax.set_ylabel(r"$R(\pi)\,/\,R(\pi^\star)$  (1.0 = oracle)")
    ax.set_title(
        "Test-split expected payoff normalised by the eval-distribution oracle.\n"
        "Same content as the raw-payoff figure, dataset scale removed: "
        "1.0 is the oracle ceiling, the prior-only bar shows how bad a pure "
        "prior policy is on each regime, and the OOD model bars are now "
        "directly comparable across train→eval cells.",
        fontsize=10,
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)

    legend_demo = 2
    policy_handles = [
        mpatches.Patch(color=policy_color(legend_demo, "bayes_optimal"),
                       label="BO (oracle)"),
        mpatches.Patch(color=policy_color(legend_demo, "plug_in"),
                       label="plug-in"),
        mpatches.Patch(color=policy_color(legend_demo, "prior_only"),
                       label="prior-only"),
        mpatches.Patch(facecolor="white",
                       edgecolor=regime_color(legend_demo), linewidth=1.6,
                       label="model (cv)"),
        mpatches.Patch(facecolor="white",
                       edgecolor=regime_color(legend_demo), linewidth=1.3,
                       hatch="///", label="model (act)"),
    ]
    regime_handles = [
        mpatches.Patch(color=regime_color(i),
                       label=f"{REGIME[i]['name']} ({REGIME[i]['rho']})")
        for i in (1, 2, 3)
    ]
    leg1 = ax.legend(handles=policy_handles, title="policy (within group)",
                     loc="lower right", fontsize=8, title_fontsize=9, ncol=1)
    ax.add_artist(leg1)
    ax.legend(handles=regime_handles,
              title="eval regime (bar fill hue)",
              loc="lower left", fontsize=8, title_fontsize=9, ncol=1)

    fig.tight_layout()
    out = OUTDIR / "fig_payoff_bars_normalized.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# fig 3 — per-(supervision, train) agreement bar groups
# ---------------------------------------------------------------------------

def fig_agreement_matrix():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    sups = ["cv", "act"]
    trains = [1, 2, 3]
    bar_w = 0.25
    series = ["bayes_optimal", "plug_in", "prior_only"]

    for i, sup in enumerate(sups):
        for j, tr in enumerate(trains):
            ax = axes[i, j]
            mid = f"D{tr}_{sup}"
            x = np.arange(3)
            for k, name in enumerate(series):
                vals = []
                colors = []
                for ev in trains:
                    vals.append(load_eval(mid, ev)["agreements"][name])
                    colors.append(policy_color(ev, name))
                ax.bar(x + (k - 1) * bar_w, vals, width=bar_w,
                       color=colors, edgecolor="white", linewidth=0.4)
            # Shade in-dist eval column lightly
            ax.axvspan(j - 0.45, j + 0.45, color=regime_color(tr),
                       alpha=0.10, zorder=0)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [regime_label(ev, multiline=True) for ev in trains],
                fontsize=8,
            )
            for tick_lbl, ev in zip(ax.get_xticklabels(), trains):
                tick_lbl.set_color(regime_color(ev))
                tick_lbl.set_fontweight("bold")
            ax.set_xlabel("eval distribution")
            ax.set_title(
                f"trained on {regime_label(tr)}  ({sup})",
                fontsize=10, color=regime_color(tr), fontweight="bold",
            )
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_ylim(0.5, 1.02)

    axes[0, 0].set_ylabel("per-step agreement")
    axes[1, 0].set_ylabel("per-step agreement")
    fig.text(0.025, 0.72, "cv supervision", rotation="vertical",
             ha="center", va="center", fontsize=11, weight="bold")
    fig.text(0.025, 0.28, "act supervision", rotation="vertical",
             ha="center", va="center", fontsize=11, weight="bold")

    fig.suptitle(
        "Per-step action agreement of each model with three reference "
        "policies on each eval distribution.\n"
        "Bar hue = eval regime; saturation falls BO → plug-in → prior-only. "
        "OOD cells expose what the model actually learned: "
        "prior-dominant→data-dominant has agree(prior-only) ≥ agree(BO).",
        fontsize=10, y=0.995,
    )
    # Policy legend below suptitle, in a clear band
    legend_demo = 2
    policy_handles = [
        mpatches.Patch(color=policy_color(legend_demo, "bayes_optimal"),
                       label="agree(BO)"),
        mpatches.Patch(color=policy_color(legend_demo, "plug_in"),
                       label="agree(plug-in)"),
        mpatches.Patch(color=policy_color(legend_demo, "prior_only"),
                       label="agree(prior-only)"),
    ]
    fig.legend(handles=policy_handles,
               title="bar saturation = policy "
                     "(hue varies with eval regime)",
               loc="upper center", ncol=3, fontsize=9, title_fontsize=9,
               bbox_to_anchor=(0.5, 0.91))
    fig.tight_layout(rect=(0.04, 0, 1, 0.86))
    out = OUTDIR / "fig_agreement_matrix.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    print(f"writing to: {OUTDIR}")
    print(f"  wrote {fig_ood_heatmap()}")
    print(f"  wrote {fig_payoff_bars()}")
    print(f"  wrote {fig_payoff_bars_normalized()}")
    print(f"  wrote {fig_agreement_matrix()}")


if __name__ == "__main__":
    main()
