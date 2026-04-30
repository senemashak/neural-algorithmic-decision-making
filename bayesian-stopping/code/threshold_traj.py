"""
Cross-model threshold trajectories (sec. 5.4).

Loads ONE OR MORE cv-supervised checkpoints and produces a 3-panel plot
(D_1 / D_2 / D_3 evaluation) with the per-step model threshold trajectory
for each checkpoint, overlaid on the four threshold-form baselines plus
secretary. Also produces a zoomed companion that drops secretary and
auto-tightens the y-axis.

Act-supervised models are silently skipped — they don't emit a scalar
threshold (the head outputs a logit, not a continuation value).

CLI (simple form, single model):
    --checkpoint        path to <prefix>_best_model.pt
    --train_dataset     1, 2, or 3
    --supervision       cv | act (auto-detected; act -> skip the script)
    --model_label       displayed in legend (default: stem of checkpoint)
    --output_dir        target dir for both PNGs

CLI (sweep form, multiple models):
    --models            "id1:path1:train_ds1,id2:path2:train_ds2,..."
                        Each entry is "<label>:<ckpt_path>:<train_ds>".
                        act-supervised checkpoints are skipped.
    --output_dir        target dir.

If both flags are given, --models wins. The default colour for the model
curve is red; multiple models are auto-coloured from a tab10 palette.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from dataset import DATASETS, build_val_test
from eval_common import ensure_writable, load_model
from oracle import C_hat_lin, compute_eta, posterior_path_batch


BASELINE_COLORS = {
    "bayes_optimal": "black",
    "plug_in":       "tab:blue",
    "prior_only":    "tab:green",
    "myopic":        "tab:orange",
    "secretary":     "tab:purple",
    "offline":       "tab:brown",
}
MODEL_PALETTE = ["tab:red", "tab:cyan", "tab:olive", "tab:pink",
                 "tab:brown", "tab:gray"]


# ---------------------------------------------------------------------------
# Threshold computation
# ---------------------------------------------------------------------------

def baseline_thresholds(X, cfg, C_hat, grids):
    N, n = X.shape
    eta = compute_eta(n)
    th_prior = np.broadcast_to(cfg.mu_0 + cfg.sigma * eta, (N, n - 1)).copy()
    cum = X.cumsum(axis=1)
    xbar = cum / np.arange(1, n + 1, dtype=float)
    th_plug = xbar[:, : n - 1] + cfg.sigma * eta[None, :]
    mu_path, _ = posterior_path_batch(X, cfg.mu_0, cfg.tau0_2, cfg.sigma2)
    th_myopic = mu_path[:, : n - 1]
    th_bo = np.empty((N, n - 1))
    for i in range(n - 1):
        th_bo[:, i] = C_hat_lin(i, mu_path[:, i], C_hat, grids)
    r = int(np.floor(n / np.e))
    th_secr = np.full((N, n - 1), -np.inf)
    if r > 0:
        M_r = X[:, :r].max(axis=1)
        for i in range(r, n - 1):
            th_secr[:, i] = M_r
    # Offline (hindsight) is not a per-step threshold — its realized stopping
    # value is max_t X_t per sequence. We carry it as a constant trajectory
    # so it can share the plotting code and serve as a horizontal ceiling.
    th_offline = np.broadcast_to(X.max(axis=1, keepdims=True), (N, n - 1)).copy()
    return {
        "prior_only":    th_prior,
        "plug_in":       th_plug,
        "myopic":        th_myopic,
        "bayes_optimal": th_bo,
        "secretary":     th_secr,
        "offline":       th_offline,
    }


@torch.no_grad()
def model_thresholds(model, X, device, batch_size=4096) -> np.ndarray:
    N, n = X.shape
    out = np.empty((N, n - 1), dtype=np.float32)
    for i in range(0, N, batch_size):
        s = slice(i, i + batch_size)
        Xb = torch.as_tensor(X[s], device=device, dtype=torch.float32)
        Cb = model(Xb)
        out[s] = Cb[:, : n - 1].cpu().numpy()
    return out


def thresholds_to_payoff(X, thresh) -> float:
    N, n = X.shape
    accept = X[:, : n - 1] >= thresh
    any_acc = accept.any(axis=1)
    first_idx = accept.argmax(axis=1)
    stop = np.where(any_acc, first_idx, n - 1)
    return float(X[np.arange(N), stop].mean())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_panel(ax, eval_cfg, baseline_th, models_th_dict, eval_id, train_ids,
               *, zoom: bool, baseline_payoffs: dict = None):
    """`models_th_dict` maps label -> (model_threshold (N, n-1), color, train_id)."""
    n = eval_cfg.n
    pos = np.arange(1, n)

    if zoom:
        plot_order = ["bayes_optimal", "plug_in", "prior_only", "myopic"]
    else:
        plot_order = ["bayes_optimal", "plug_in", "prior_only",
                      "myopic", "secretary", "offline"]

    ymin, ymax = float("inf"), float("-inf")
    SLICE_FROM = 3

    def update_ylim(mean, std):
        nonlocal ymin, ymax
        lo = mean[SLICE_FROM:] - 0.5 * std[SLICE_FROM:]
        hi = mean[SLICE_FROM:] + 0.5 * std[SLICE_FROM:]
        lo = lo[np.isfinite(lo)]
        hi = hi[np.isfinite(hi)]
        if lo.size: ymin = min(ymin, float(lo.min()))
        if hi.size: ymax = max(ymax, float(hi.max()))

    for name in plot_order:
        th = baseline_th[name]
        with np.errstate(invalid="ignore"):
            mean = th.mean(axis=0)
            std = th.std(axis=0)
        color = BASELINE_COLORS[name]
        if name == "secretary":
            r = int(np.floor(n / np.e))
            sl = np.arange(r, n - 1)
            x = pos[sl]
            ax.plot(x, mean[sl], color=color, linewidth=1.5,
                    label=f"secretary (t > {r})")
            ax.fill_between(x, (mean - std)[sl], (mean + std)[sl],
                            color=color, alpha=0.15)
        elif name == "prior_only":
            ax.plot(pos, mean, color=color, linewidth=1.5, label="prior_only")
            if zoom: update_ylim(mean, std)
        elif name == "offline":
            # Hindsight ceiling: max_t X_t per sequence, constant in t.
            # Dashed to flag that it is not an online threshold.
            ax.plot(pos, mean, color=color, linewidth=1.5, linestyle="--",
                    label=f"offline (hindsight, E[max] = {mean[0]:.3f})")
            ax.fill_between(pos, mean - std, mean + std, color=color, alpha=0.10)
        else:
            ax.plot(pos, mean, color=color, linewidth=1.5, label=name)
            ax.fill_between(pos, mean - std, mean + std, color=color, alpha=0.15)
            if zoom: update_ylim(mean, std)

    for label, (mth, color, train_id) in models_th_dict.items():
        m_mean = mth.mean(axis=0)
        m_std = mth.std(axis=0)
        tag = "in-dist" if train_id == eval_id else "OOD"
        ax.plot(pos, m_mean, color=color, linewidth=2.0, linestyle="--",
                label=f"model {label} [{tag}]")
        ax.fill_between(pos, m_mean - m_std, m_mean + m_std,
                        color=color, alpha=0.15)
        if zoom: update_ylim(m_mean, m_std)

    if zoom and ymin < ymax:
        ax.set_ylim(ymin, ymax)

    title = f"{eval_cfg.name}, rho={eval_cfg.rho}"
    ax.set_title(title)
    ax.set_xlabel("step t (1-indexed)")
    ax.set_ylabel("threshold value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7)

    if zoom and baseline_payoffs is not None:
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

def parse_models_arg(s: str):
    """Parse "label1:path1:train_ds1,label2:path2:train_ds2" -> list of dicts."""
    out = []
    for entry in s.split(","):
        entry = entry.strip()
        if not entry: continue
        parts = entry.split(":")
        if len(parts) != 3:
            sys.exit(f"FATAL: bad --models entry {entry!r}; "
                     f"expected 'label:path:train_ds'")
        label, path, train_ds = parts
        out.append({"label": label, "path": path, "train_ds": int(train_ds)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--train_dataset", type=int, default=None)
    p.add_argument("--supervision", default=None)
    p.add_argument("--model_label", default=None)
    p.add_argument("--models", default=None,
                   help='"label1:path1:train_ds1,label2:path2:train_ds2,..."')
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_test", type=int, default=10_000)
    args = p.parse_args()

    if args.models:
        models_in = parse_models_arg(args.models)
    elif args.checkpoint:
        if args.train_dataset is None:
            sys.exit("FATAL: --train_dataset required with --checkpoint")
        label = args.model_label or Path(args.checkpoint).stem
        models_in = [{"label": label, "path": args.checkpoint,
                      "train_ds": args.train_dataset}]
    else:
        sys.exit("FATAL: provide --checkpoint or --models")

    out = ensure_writable(Path(args.output_dir))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load all models, dropping any with supervision='act'.
    loaded = []
    for i, m in enumerate(models_in):
        sup_override = args.supervision if len(models_in) == 1 else None
        try:
            mdl = load_model(m["path"], n=config.N, device=device,
                             supervision=sup_override)
        except SystemExit:
            raise
        except Exception as e:
            sys.exit(f"FATAL: failed to load {m['path']}: {e}")
        if mdl.supervision == "act":
            print(f"  skipping {m['label']}: act-supervised "
                  f"(no scalar threshold)")
            continue
        color = MODEL_PALETTE[i % len(MODEL_PALETTE)]
        loaded.append({**m, "model": mdl, "color": color})
        print(f"  loaded {m['label']} (train_ds=D_{m['train_ds']}, "
              f"supervision=cv) -> color={color}")

    if not loaded:
        sys.exit("FATAL: no cv-supervised checkpoints to plot")

    # Compute thresholds + payoffs once per eval dataset (shared across plots).
    panels = {}
    for eval_id in [1, 2, 3]:
        eval_cfg = DATASETS[eval_id]
        print(f"  computing thresholds on {eval_cfg.name}...")
        bundle = build_val_test(eval_cfg, seed_val=42, seed_test=43,
                                N_val=10_000, N_test=args.n_test)
        bths = baseline_thresholds(bundle.X_test, eval_cfg,
                                    bundle.C_hat, bundle.grids)
        models_th = {}
        for m in loaded:
            mth = model_thresholds(m["model"], bundle.X_test, device)
            models_th[m["label"]] = (mth, m["color"], m["train_ds"])
        bp = {name: thresholds_to_payoff(bundle.X_test, bths[name])
              for name in ("bayes_optimal", "plug_in", "prior_only", "myopic")}
        panels[eval_id] = (eval_cfg, bths, models_th, bp)

    train_ids = [m["train_ds"] for m in loaded]
    n_models = len(loaded)
    sup_str = f"{n_models}-model cv sweep" if n_models > 1 else "1 model, cv supervision"

    # ---- unzoomed ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    for ax_idx, eval_id in enumerate([1, 2, 3]):
        eval_cfg, bths, models_th, _ = panels[eval_id]
        plot_panel(axes[ax_idx], eval_cfg, bths, models_th,
                   eval_id=eval_id, train_ids=train_ids,
                   zoom=False)
    fig.suptitle(
        f"Threshold trajectories ({sup_str}; "
        f"mean ±1 std over {args.n_test:,} test sequences)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    p1 = out / "threshold_trajectories.png"
    fig.savefig(p1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote: {p1}")

    # ---- zoomed ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    for ax_idx, eval_id in enumerate([1, 2, 3]):
        eval_cfg, bths, models_th, bp = panels[eval_id]
        plot_panel(axes[ax_idx], eval_cfg, bths, models_th,
                   eval_id=eval_id, train_ids=train_ids,
                   zoom=True, baseline_payoffs=bp)
    fig.suptitle(
        f"Threshold trajectories — zoomed ({sup_str}; "
        f"mean ±1 std over {args.n_test:,} test sequences;\n"
        f"secretary excluded · y-axis tightened to mean ±0.5·std over t ≥ 4 · "
        f"first 3 timesteps excluded for visualization, computed from t=4 onward)",
        fontsize=11, y=1.04,
    )
    fig.tight_layout()
    p2 = out / "threshold_trajectories_zoomed.png"
    fig.savefig(p2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote: {p2}")


if __name__ == "__main__":
    main()
