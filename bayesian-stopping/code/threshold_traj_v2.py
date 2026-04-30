"""
Threshold trajectory plots — second iteration.

Differences vs threshold_traj.py:
  - Drops myopic, secretary, and offline. Only bayes_optimal, plug_in,
    prior_only as baselines.
  - All three cv-supervised models on the same panel:
      D1_cv (red dashed),  D2_cv (red solid),  D3_cv (red dotted).
  - Hard ylim (1.5, 2.2) on every panel, shared across the three datasets
    so the plots are directly comparable. Curves whose mean leaves that
    band are clipped at the boundary; an annotation in the panel lists
    them with their actual mean value at that position.
  - Per-panel title includes the eval distribution and its oracle payoff.
  - Per-panel annotation: BO / plug_in / prior_only payoffs.

Output: a single threshold_trajectories_v2.png (no zoomed variant —
with the fixed (1.5, 2.2) band, "zoomed" no longer adds anything).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from dataset import DATASETS, build_val_test
from eval_common import ensure_writable, load_model
from oracle import C_hat_lin, compute_eta, posterior_path_batch


COLOR_BO       = "black"
COLOR_PLUG     = "tab:blue"
COLOR_PRIOR    = "tab:green"

# One distinct colour per model so the three trained-model curves are
# legible when they overlap. Picked from the standard tab10 palette,
# avoiding the colours already used by baselines (black, blue, green).
MODEL_COLORS = {
    1: "tab:red",        # D1_cv (data-dominant)
    2: "tab:orange",     # D2_cv (balanced)
    3: "tab:purple",     # D3_cv (prior-dominant)
}

MODEL_LINESTYLES = {  # train_dataset -> linestyle (kept as redundant cue)
    1: "--",
    2: "-",
    3: ":",
}


def baseline_thresholds_min(X, cfg, C_hat, grids):
    """Just the 3 we plot: bayes_optimal, plug_in, prior_only."""
    N, n = X.shape
    eta = compute_eta(n)
    th_prior = np.broadcast_to(cfg.mu_0 + cfg.sigma * eta, (N, n - 1)).copy()
    cum = X.cumsum(axis=1)
    xbar = cum / np.arange(1, n + 1, dtype=float)
    th_plug = xbar[:, : n - 1] + cfg.sigma * eta[None, :]
    mu_path, _ = posterior_path_batch(X, cfg.mu_0, cfg.tau0_2, cfg.sigma2)
    th_bo = np.empty((N, n - 1))
    for i in range(n - 1):
        th_bo[:, i] = C_hat_lin(i, mu_path[:, i], C_hat, grids)
    return {"bayes_optimal": th_bo, "plug_in": th_plug, "prior_only": th_prior}


@torch.no_grad()
def model_thresholds(model, X, device, batch_size=4096) -> np.ndarray:
    N, n = X.shape
    out = np.empty((N, n - 1), dtype=np.float32)
    for i in range(0, N, batch_size):
        s = slice(i, i + batch_size)
        Xb = torch.as_tensor(X[s], device=device, dtype=torch.float32)
        out[s] = model(Xb)[:, : n - 1].cpu().numpy()
    return out


def thresholds_to_payoff(X, thresh) -> float:
    N, n = X.shape
    accept = X[:, : n - 1] >= thresh
    any_acc = accept.any(axis=1)
    first_idx = accept.argmax(axis=1)
    stop = np.where(any_acc, first_idx, n - 1)
    return float(X[np.arange(N), stop].mean())


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

YLO, YHI = 1.5, 2.2                                             # hard fixed band


def _summarize_offaxis(label: str, mean: np.ndarray):
    """If the curve's mean leaves [YLO, YHI] for any t in [3, n-2], return
    (direction, value) where direction in {'below', 'above'} and value is
    the most-extreme mean value in that direction. Else return None."""
    sl = mean[3:]
    if not np.isfinite(sl).any():
        return None
    if sl.min() < YLO:
        return ("below", float(sl.min()), label)
    if sl.max() > YHI:
        return ("above", float(sl.max()), label)
    return None


def plot_panel(ax, eval_cfg, eval_id, baseline_th, models_th, baseline_payoffs):
    """Draw one panel with hard ylim (YLO, YHI). Curves leaving the band
    are clipped (matplotlib does this automatically with set_ylim);
    out-of-band curves are listed in a side annotation."""
    n = eval_cfg.n
    pos = np.arange(1, n)

    offaxis = []  # list of ("below"|"above", val, label)

    for name, color, label in (
        ("bayes_optimal", COLOR_BO,    "bayes_optimal"),
        ("plug_in",       COLOR_PLUG,  "plug_in"),
        ("prior_only",    COLOR_PRIOR, "prior_only"),
    ):
        th = baseline_th[name]
        mean = th.mean(axis=0); std = th.std(axis=0)
        ax.plot(pos, mean, color=color, linewidth=1.6, label=label)
        if name != "prior_only":
            ax.fill_between(pos, mean - std, mean + std,
                            color=color, alpha=0.13)
        item = _summarize_offaxis(label, mean)
        if item: offaxis.append(item)

    for tr in (1, 2, 3):
        if tr not in models_th: continue
        mth = models_th[tr]
        mean = mth.mean(axis=0); std = mth.std(axis=0)
        ls = MODEL_LINESTYLES[tr]
        color = MODEL_COLORS[tr]
        tag = "in-dist" if tr == eval_id else "OOD"
        label = f"D{tr}_cv [{tag}]"
        ax.plot(pos, mean, color=color, linewidth=2.0, linestyle=ls,
                label=label)
        ax.fill_between(pos, mean - std, mean + std,
                        color=color, alpha=0.10)
        item = _summarize_offaxis(label, mean)
        if item: offaxis.append(item)

    ax.set_ylim(YLO, YHI)
    bo_pay = baseline_payoffs["bayes_optimal"]
    ax.set_title(f"eval = {eval_cfg.name}, rho={eval_cfg.rho}  "
                 f"(BO payoff = {bo_pay:.4f})", fontsize=11)
    ax.set_xlabel("step t (1-indexed)")
    ax.set_ylabel("threshold value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # Off-axis annotations: stack near the relevant edge
    below = [(v, lbl) for d, v, lbl in offaxis if d == "below"]
    above = [(v, lbl) for d, v, lbl in offaxis if d == "above"]
    if below:
        lines = ["off-axis below ↓"]
        for v, lbl in sorted(below, key=lambda t: t[0]):
            lines.append(f"  {lbl} → ~{v:.2f}")
        ax.text(0.02, 0.02, "\n".join(lines),
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=7, color="#444",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.85, edgecolor="0.6"))
    if above:
        lines = ["off-axis above ↑"]
        for v, lbl in sorted(above, key=lambda t: -t[0]):
            lines.append(f"  {lbl} → ~{v:.2f}")
        ax.text(0.02, 0.98, "\n".join(lines),
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7, color="#444",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.85, edgecolor="0.6"))

    # Payoff anchor (centered, just above the bottom edge)
    anno = (
        f"payoffs (test):  "
        f"BO {baseline_payoffs['bayes_optimal']:.4f}  |  "
        f"plug-in {baseline_payoffs['plug_in']:.4f}  |  "
        f"prior {baseline_payoffs['prior_only']:.4f}"
    )
    ax.text(0.5, 0.02, anno, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.9, edgecolor="0.5"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_d1", required=True,
                   help="path to D1_cv best_model.pt")
    p.add_argument("--ckpt_d2", required=True,
                   help="path to D2_cv best_model.pt")
    p.add_argument("--ckpt_d3", required=True,
                   help="path to D3_cv best_model.pt")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_test", type=int, default=10_000)
    args = p.parse_args()

    out = ensure_writable(Path(args.output_dir))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Load 3 cv models; assert each is cv-supervised.
    models = {}
    for tr, ckpt in ((1, args.ckpt_d1), (2, args.ckpt_d2), (3, args.ckpt_d3)):
        m = load_model(ckpt, n=config.N, device=device)
        if m.supervision != "cv":
            raise SystemExit(f"FATAL: {ckpt} is not cv-supervised; got {m.supervision}")
        models[tr] = m
        print(f"  loaded D{tr}_cv from {ckpt}")

    # Compute thresholds + baseline payoffs once per eval dataset.
    panels = {}
    for eval_id in (1, 2, 3):
        eval_cfg = DATASETS[eval_id]
        print(f"  computing thresholds on {eval_cfg.name}...")
        bundle = build_val_test(eval_cfg, seed_val=42, seed_test=43,
                                N_val=10_000, N_test=args.n_test)
        bths = baseline_thresholds_min(bundle.X_test, eval_cfg,
                                        bundle.C_hat, bundle.grids)
        models_th = {tr: model_thresholds(models[tr], bundle.X_test, device)
                     for tr in (1, 2, 3)}
        bp = {name: thresholds_to_payoff(bundle.X_test, bths[name])
              for name in ("bayes_optimal", "plug_in", "prior_only")}
        panels[eval_id] = (eval_cfg, bths, models_th, bp)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    for ax_idx, eval_id in enumerate((1, 2, 3)):
        eval_cfg, bths, mths, bp = panels[eval_id]
        plot_panel(axes[ax_idx], eval_cfg, eval_id, bths, mths, bp)
    fig.suptitle(
        "Threshold trajectories v2 — three cv models overlaid (D1_cv dashed, "
        "D2_cv solid, D3_cv dotted) against bayes_optimal / plug_in / prior_only.\n"
        f"Mean ±1 std over 10,000 test sequences. y-axis fixed to "
        f"({YLO}, {YHI}) on every panel — curves leaving that band are clipped "
        "and listed in a side note.",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    p1 = out / "threshold_trajectories_v2.png"
    fig.savefig(p1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote: {p1}")

    # Remove the now-stale zoomed file if it still exists from a prior run.
    stale = out / "threshold_trajectories_v2_zoomed.png"
    if stale.exists():
        stale.unlink()
        print(f"removed stale: {stale}")


if __name__ == "__main__":
    main()
