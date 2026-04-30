"""
Shared evaluation utilities for eval_indist.py and eval_ood.py.

Provides:
    ensure_writable(path)            — fail loudly if path isn't writable.
    load_model(ckpt, n, sup, dev)    — instantiate GPTStopper, load weights.
    model_actions_batch(...)         — per-step accept-actions and 0-indexed
                                       stopping times from the cv-supervised
                                       model on a batch of sequences.
    evaluate_on_dataset(...)         — pull together model + 6-baseline
                                       payoffs, stop-times, and per-step
                                       agreements on a fixed test set.
"""

import sys
from pathlib import Path

import numpy as np
import torch

import config
from model import GPTStopper
from oracle import compute_eta, posterior_path_batch


# ---------------------------------------------------------------------------
# Strict path verification
# ---------------------------------------------------------------------------

def ensure_writable(path: Path) -> Path:
    """Print absolute path, verify writability via touch+delete, abort if not."""
    abs_path = path.resolve()
    print(f"output dir: {abs_path}", flush=True)
    if not abs_path.exists():
        print(f"  -> directory does not exist; creating...", flush=True)
        try:
            abs_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            sys.exit(f"FATAL: cannot create output dir {abs_path}: {exc}")
    test_file = abs_path / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
    except Exception as exc:
        sys.exit(
            f"FATAL: output dir {abs_path} is not writable: {exc}\n"
            f"  Refusing to silently fall back to a different directory."
        )
    print(f"  -> verified writable", flush=True)
    return abs_path


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    ckpt_path: str, n: int, device: torch.device,
    supervision: str = None,
) -> GPTStopper:
    """Load a checkpoint. If `supervision` is None, read it from the saved
    metadata (default 'cv' for legacy checkpoints without the field)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = ckpt["state_dict"]
    ckpt_sup = ckpt.get("supervision", "cv")
    if supervision is None:
        supervision = ckpt_sup
    elif supervision != ckpt_sup:
        print(f"WARNING: --supervision={supervision} overrides checkpoint "
              f"metadata supervision={ckpt_sup}", flush=True)
    if supervision not in ("cv", "act"):
        sys.exit(f"FATAL: unknown supervision: {supervision!r}")
    if "cv_head.weight" not in sd:
        sys.exit(f"FATAL: checkpoint {ckpt_path} missing cv_head.weight")

    # Constructor sigma is irrelevant — load_state_dict overwrites the
    # input_scale / output_scale buffers from the checkpoint.
    model = GPTStopper(
        n=n,
        d_emb=config.D_EMB, n_layers=config.N_LAYERS, n_heads=config.N_HEADS,
        supervision=supervision, sigma=1.0,
    ).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Model policy (cv-supervised: accept iff X_t >= predicted C_hat_t)
# ---------------------------------------------------------------------------

@torch.no_grad()
def model_actions_batch(
    model: GPTStopper, X: np.ndarray, device: torch.device, batch_size: int = 4096,
):
    """Per-step actions and 0-indexed stopping times. Dispatches on
    `model.supervision`:
        cv  -> accept iff X[t] >= predicted C_hat[t]
        act -> accept iff predicted logit[t] > 0
    Returns (model_act: (N, n-1) bool, model_stop: (N,) int64)."""
    sup = getattr(model, "supervision", "cv")
    N, n = X.shape
    model_act = np.empty((N, n - 1), dtype=bool)
    model_stop = np.empty(N, dtype=np.int64)
    for i in range(0, N, batch_size):
        s = slice(i, i + batch_size)
        Xb = torch.as_tensor(X[s], device=device, dtype=torch.float32)
        out = model(Xb)
        if sup == "act":
            accept = out[:, : n - 1] > 0.0
        else:
            accept = (Xb[:, : n - 1] >= out[:, : n - 1])
        any_acc = accept.any(dim=1)
        first_idx = accept.float().argmax(dim=1)
        last = torch.full_like(first_idx, n - 1)
        stop = torch.where(any_acc, first_idx, last)
        model_act[s] = accept.cpu().numpy()
        model_stop[s] = stop.cpu().numpy()
    return model_act, model_stop


# ---------------------------------------------------------------------------
# End-to-end metric block
# ---------------------------------------------------------------------------

def baseline_actions(X: np.ndarray, cfg, y_act_oracle: np.ndarray):
    """Per-step accept actions for plug-in, prior-only, myopic, BO, secretary.
    All baselines are evaluated under `cfg` (eval-dataset config) — caller is
    responsible for passing the correct cfg for OOD.
    """
    N, n = X.shape
    eta = compute_eta(n)
    cum = X.cumsum(axis=1)
    xbar = cum / np.arange(1, n + 1, dtype=float)

    a_plug = X[:, : n - 1] >= xbar[:, : n - 1] + cfg.sigma * eta
    a_prior = X[:, : n - 1] >= cfg.mu_0 + cfg.sigma * eta
    mu_path, _ = posterior_path_batch(X, cfg.mu_0, cfg.tau0_2, cfg.sigma2)
    a_myopic = X[:, : n - 1] >= mu_path[:, : n - 1]
    a_bo = y_act_oracle.astype(bool)

    r = int(np.floor(n / np.e))
    a_secr = np.zeros((N, n - 1), dtype=bool)
    if r > 0:
        M_r = X[:, :r].max(axis=1)
        for i in range(r, n - 1):
            a_secr[:, i] = X[:, i] > M_r

    return {
        "plug_in": a_plug, "prior_only": a_prior, "myopic": a_myopic,
        "bayes_optimal": a_bo, "secretary": a_secr,
    }


def actions_to_stop(actions: np.ndarray, n: int) -> np.ndarray:
    any_acc = actions.any(axis=1)
    first_idx = actions.argmax(axis=1)
    return np.where(any_acc, first_idx, n - 1)


def evaluate_on_dataset(model, cfg, X, y_act_oracle, device):
    """Compute model + baseline payoffs and per-step / stop-time agreements.
    `cfg` is the EVAL-dataset config; `y_act_oracle` is its bayes_optimal labels.
    """
    N, n = X.shape

    # Model
    model_act, model_stop = model_actions_batch(model, X, device)
    model_payoff_arr = X[np.arange(N), model_stop]

    # Baselines
    a = baseline_actions(X, cfg, y_act_oracle)
    stops = {name: actions_to_stop(act, n) for name, act in a.items()}
    stops["offline"] = X.argmax(axis=1)

    payoffs = {name: float(X[np.arange(N), stops[name]].mean()) for name in stops}
    payoff_ses = {
        name: float(X[np.arange(N), stops[name]].std(ddof=1) / np.sqrt(N))
        for name in stops
    }
    stop_means = {name: float(stops[name].mean() + 1) for name in stops}  # 1-indexed

    agreements = {name: float((model_act == act).mean()) for name, act in a.items()}
    stoptime_matches = {
        "offline_stoptime_match":      float((model_stop == stops["offline"]).mean()),
        "bayes_optimal_stoptime_match": float((model_stop == stops["bayes_optimal"]).mean()),
    }

    return {
        "model_payoff":         float(model_payoff_arr.mean()),
        "model_payoff_se":      float(model_payoff_arr.std(ddof=1) / np.sqrt(N)),
        "model_stop_mean":      float(model_stop.mean() + 1),  # 1-indexed
        "baseline_payoffs":     payoffs,
        "baseline_payoff_ses":  payoff_ses,
        "baseline_stop_means":  stop_means,
        "agreements":           agreements,
        "stoptime_matches":     stoptime_matches,
    }


# ---------------------------------------------------------------------------
# CSV writer (shared by indist/ood)
# ---------------------------------------------------------------------------

def write_summary_csv(metrics: dict, csv_path: Path):
    import csv
    rows = [
        ("baseline", "payoff", "payoff_se", "stop_mean", "agreement"),
    ]
    for name in ("bayes_optimal", "plug_in", "prior_only",
                 "myopic", "secretary", "offline"):
        if name == "offline":
            agr = f"{metrics['stoptime_matches']['offline_stoptime_match']:.4f} (stop-match)"
        else:
            agr = f"{metrics['agreements'][name]:.4f}"
        rows.append((
            name,
            f"{metrics['baseline_payoffs'][name]:.4f}",
            f"{metrics['baseline_payoff_ses'][name]:.4f}",
            f"{metrics['baseline_stop_means'][name]:.2f}",
            agr,
        ))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)


def print_summary_table(metrics: dict, header: str):
    print(f"\n=== {header} ===")
    print(f"  model payoff: {metrics['model_payoff']:.4f}  "
          f"(SE {metrics['model_payoff_se']:.4f})  "
          f"E[tau]={metrics['model_stop_mean']:.2f}")
    print(f"  vs Bayes-optimal: "
          f"{metrics['model_payoff'] - metrics['baseline_payoffs']['bayes_optimal']:+.4f}")
    print(f"  vs offline:       "
          f"{metrics['model_payoff'] - metrics['baseline_payoffs']['offline']:+.4f}")
    print()
    print(f"  {'baseline':<14} {'payoff':>9} {'E[tau]':>7} {'agreement':>10}")
    print(f"  {'-'*14} {'-'*9} {'-'*7} {'-'*10}")
    for name in ("bayes_optimal", "plug_in", "prior_only",
                 "myopic", "secretary"):
        print(f"  {name:<14} "
              f"{metrics['baseline_payoffs'][name]:>9.4f} "
              f"{metrics['baseline_stop_means'][name]:>7.2f} "
              f"{metrics['agreements'][name]:>10.4f}")
    name = "offline"
    sm = metrics["stoptime_matches"]["offline_stoptime_match"]
    print(f"  {name:<14} "
          f"{metrics['baseline_payoffs'][name]:>9.4f} "
          f"{metrics['baseline_stop_means'][name]:>7.2f} "
          f"{sm:>10.4f} (stop-match)")
