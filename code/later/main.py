"""
End-to-end experimental pipeline for both optimal stopping and ski rental.

Each evaluation experiment is repeated across multiple random seeds (default 10)
to capture variance. CSVs store per-seed results; plots show mean ± CI.

Results saved under:
    results/run_<YYYYMMDD_HHMMSS>/
        stopping/   — stopping model checkpoint + experiment CSVs
        ski/        — ski model checkpoint + experiment CSVs
        plots/      — all PNG figures

Usage:
    python main.py
    python main.py --device cuda --epochs 50 --n_seeds 20
    python main.py --skip_train --checkpoint_stop results/.../stopping/checkpoint.pt
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from sampling import (
    sample_stopping_batch, STOPPING_SAMPLERS, STOPPING_TRAIN_FAMILIES,
    sample_ski_batch, SKI_SAMPLERS, SKI_HEAVY_TAIL_FAMILIES,
)
from dataset import StoppingDataset, SkiRentalDataset, make_dataloader
from model import OnlineDecisionTransformer
from train import train as run_train
from deployment import (
    compare_stopping, compare_ski, print_stopping_results, print_ski_results,
    find_lambdas, compute_U, get_stopping_predictions, get_ski_predictions,
    stop_policy_learned, stop_policy_robust, stop_policy_dynkin,
    ski_policy_learned, ski_policy_robust, ski_policy_deterministic,
    ski_policy_cost, ski_optimal_cost,
)
from plot import generate_all_plots


# ═══════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════

_T0 = time.time()

def _ts():
    return datetime.now().strftime("%H:%M:%S")

def log(msg):
    print(f"  [{_ts()}] {msg}", flush=True)

def banner(title):
    line = "═" * 60
    print(f"\n{line}\n  {title}  [{_ts()}]\n{line}", flush=True)

def elapsed():
    s = int(time.time() - _T0)
    return f"{s//60}m {s%60}s"


# ═══════════════════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════════════════

def save_csv(rows, path):
    if not rows:
        return
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)
    log(f"Saved {len(rows)} rows → {path.name}")

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def _results_to_rows(results, extra=None):
    rows = []
    for policy, m in results.items():
        row = {"policy": policy, **m}
        if extra:
            row = {**extra, **row}
        rows.append({k: v for k, v in row.items() if not isinstance(v, (list, dict))})
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Multi-seed helpers
# ═══════════════════════════════════════════════════════════════════════════

def eval_seeds(args):
    """Return list of evaluation seeds."""
    return [args.seed + 10000 + i for i in range(args.n_seeds)]


# ═══════════════════════════════════════════════════════════════════════════
# Stopping experiments (multi-seed)
# ═══════════════════════════════════════════════════════════════════════════

def stopping_experiments(model, args, out_dir):
    sdir = out_dir / "stopping"
    sdir.mkdir(exist_ok=True)
    n, M = args.n, args.M
    seeds = eval_seeds(args)

    # ── Exp 1: In-distribution ──
    banner(f"STOPPING EXP 1 — IN-DISTRIBUTION  ({args.n_seeds} seeds)")
    all_rows = []
    for si, seed in enumerate(seeds):
        log(f"  seed {si+1}/{args.n_seeds}")
        insts = sample_stopping_batch(args.n_eval, n, M, rng=np.random.default_rng(seed))
        res = compare_stopping(insts, model, args.betas, args.r_fractions, args.device, use_chain=True)
        all_rows.extend(_results_to_rows(res, {"experiment": "in_dist", "n": n, "seed": seed}))
    save_csv(all_rows, sdir / "exp1_in_distribution.csv")

    # ── Exp 2: Per-family ──
    banner(f"STOPPING EXP 2 — PER-FAMILY  ({args.n_seeds} seeds)")
    all_rows = []
    for family in STOPPING_SAMPLERS:
        log(f"  Family: {family}")
        for si, seed in enumerate(seeds):
            insts = sample_stopping_batch(args.n_eval // len(STOPPING_SAMPLERS), n, M,
                                           dist_type=family, rng=np.random.default_rng(seed))
            res = compare_stopping(insts, model, args.betas, args.r_fractions, args.device, use_chain=True)
            all_rows.extend(_results_to_rows(res, {"experiment": "by_family",
                                                    "family": family, "n": n, "seed": seed}))
    save_csv(all_rows, sdir / "exp2_by_family.csv")

    # ── Exp 3: OOD hard instance ──
    banner(f"STOPPING EXP 3 — OOD HARD INSTANCE  ({args.n_seeds} seeds)")
    all_rows = []
    for si, seed in enumerate(seeds):
        log(f"  seed {si+1}/{args.n_seeds}")
        insts = sample_stopping_batch(args.n_eval, n, M, dist_type="hard_instance",
                                       rng=np.random.default_rng(seed))
        res = compare_stopping(insts, model, args.betas, args.r_fractions, args.device, use_chain=True)
        all_rows.extend(_results_to_rows(res, {"experiment": "ood", "n": n, "seed": seed}))
    save_csv(all_rows, sdir / "exp3_ood.csv")

    # ── Exp 4: Horizon generalization ──
    banner(f"STOPPING EXP 4 — HORIZON GENERALIZATION  ({args.n_seeds} seeds)")
    test_ns = sorted(set([n] + args.test_horizons))
    test_ns = [tn for tn in test_ns if tn <= args.max_n]
    all_rows = []
    r_fracs = [0.1, 1 / np.e]
    for test_n in test_ns:
        log(f"  n = {test_n}")
        for si, seed in enumerate(seeds):
            insts = sample_stopping_batch(args.n_eval, test_n, M,
                                           rng=np.random.default_rng(seed))
            res = compare_stopping(insts, model, [0.20], r_fracs, args.device, use_chain=True)
            renamed = {}
            r_to_frac = {max(0, int(f * test_n)): f for f in r_fracs}
            for key, val in res.items():
                if key.startswith("dynkin r="):
                    r_val = int(key.split("=")[1])
                    frac = r_to_frac.get(r_val, r_val / test_n)
                    beta = -frac * np.log(frac) if frac > 0 else 0.0
                    renamed[f"dynkin β={beta:.2f}"] = val
                else:
                    renamed[key] = val
            all_rows.extend(_results_to_rows(renamed, {"experiment": "horizon",
                                                        "train_n": n, "test_n": test_n, "seed": seed}))
    save_csv(all_rows, sdir / "exp4_horizon.csv")

    # ── Exp 5: Robustness sweep ──
    banner(f"STOPPING EXP 5 — ROBUSTNESS SWEEP  ({args.n_seeds} seeds)")
    r_max = max(1, int(np.floor(n / np.e)))
    all_rows = []
    for si, seed in enumerate(seeds):
        log(f"  seed {si+1}/{args.n_seeds}")
        insts = sample_stopping_batch(args.n_eval, n, M, rng=np.random.default_rng(seed))
        V_hat_all = get_stopping_predictions(model, insts, device=args.device, use_chain=True)
        realized_maxes = np.array([float(inst.values.max()) for inst in insts])
        mean_max = float(realized_maxes.mean())

        def _cr(rewards):
            return float(np.mean(rewards)) / mean_max if mean_max > 0 else 0.0

        lrn_rewards = []
        for i, inst in enumerate(insts):
            _, rw = stop_policy_learned(inst.values, V_hat_all[i], M)
            lrn_rewards.append(rw)
        cr_learned = _cr(lrn_rewards)

        for r_val in range(1, r_max + 1):
            frac = r_val / n
            beta = -frac * np.log(frac)
            lam1, lam2 = find_lambdas(beta)

            rob_rewards = []
            for i, inst in enumerate(insts):
                _, rw = stop_policy_robust(inst.values, V_hat_all[i], beta, M)
                rob_rewards.append(rw)

            dyn_rewards = []
            for inst in insts:
                _, rw = stop_policy_dynkin(inst.values, r_val)
                dyn_rewards.append(rw)

            all_rows.append({
                "r": r_val, "beta": round(beta, 5),
                "lambda1": round(lam1, 5), "lambda2": round(lam2, 5),
                "cr_robust": round(_cr(rob_rewards), 5),
                "cr_dynkin": round(_cr(dyn_rewards), 5),
                "cr_learned": round(cr_learned, 5),
                "n": n, "seed": seed,
            })
    save_csv(all_rows, sdir / "exp5_robustness_sweep.csv")

    # ── Exp 6: Heavy-tail ──
    banner(f"STOPPING EXP 6 — HEAVY-TAIL FAMILIES  ({args.n_seeds} seeds)")
    heavy_families = ["lognormal", "bimodal", "weibull", "hard_instance"]
    all_rows = []
    for family in heavy_families:
        log(f"  Family: {family}")
        for si, seed in enumerate(seeds):
            insts = sample_stopping_batch(500, n, M, dist_type=family,
                                           rng=np.random.default_rng(seed))
            res = compare_stopping(insts, model, args.betas, args.r_fractions, args.device, use_chain=True)
            all_rows.extend(_results_to_rows(res, {"experiment": "heavy_tail",
                                                    "family": family, "n": n, "seed": seed}))
    save_csv(all_rows, sdir / "exp6_heavy_tail.csv")


# ═══════════════════════════════════════════════════════════════════════════
# Ski rental experiments (multi-seed)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_training_pmf(n, grid_size=200):
    """
    Compute the marginal PMF of T under the uniform mixture of 10 training
    families by numerical integration over each family's hyperparameter prior
    on a dense grid. No sampling — fully deterministic.

    P(t) = (1/10) * sum_f  integral P_f(t | theta) d(prior_f(theta))

    For continuous hyperparameters: uniform grid quadrature.
    For discrete hyperparameters: exact enumeration.
    """
    from sampling import _make_ski_pmf

    t = np.arange(1, n + 1, dtype=float)
    log_fact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, n + 1)))])
    G = grid_size  # grid points per continuous dimension

    family_pmfs = []

    # --- 1. Geometric: p in [0.05, 0.5] ---
    acc = np.zeros(n)
    for p in np.linspace(0.05, 0.5, G):
        log_w = (t - 1) * np.log1p(-p) + np.log(p)
        acc += _make_ski_pmf(log_w)
    family_pmfs.append(acc / G)

    # --- 2. Poisson: lam in [1, n/2] ---
    acc = np.zeros(n)
    for lam in np.linspace(1.0, n / 2, G):
        log_w = t * np.log(lam) - lam - log_fact[t.astype(int)]
        acc += _make_ski_pmf(log_w)
    family_pmfs.append(acc / G)

    # --- 3. Binomial: p in [0.1, 0.9] ---
    acc = np.zeros(n)
    s = np.arange(n)  # s = T-1
    for p in np.linspace(0.1, 0.9, G):
        log_w = (log_fact[n - 1] - log_fact[s] - log_fact[n - 1 - s]
                 + s * np.log(p + 1e-300) + (n - 1 - s) * np.log(1 - p + 1e-300))
        acc += _make_ski_pmf(log_w)
    family_pmfs.append(acc / G)

    # --- 4. Uniform: K in {2, ..., n} (discrete, exact) ---
    acc = np.zeros(n)
    for K in range(2, n + 1):
        pmf = np.zeros(n)
        pmf[:K] = 1.0 / K
        acc += pmf
    family_pmfs.append(acc / (n - 1))

    # --- 5. Zipf: alpha in [1, 3] ---
    acc = np.zeros(n)
    for alpha in np.linspace(1.0, 3.0, G):
        log_w = -alpha * np.log(t)
        acc += _make_ski_pmf(log_w)
    family_pmfs.append(acc / G)

    # --- 6. Log-normal: mu in [0, ln(n)], sigma in [0.3, 1.5] ---
    G2 = max(G // 5, 40)  # coarser grid for 2D
    acc = np.zeros(n)
    for mu in np.linspace(0, np.log(n), G2):
        for sigma in np.linspace(0.3, 1.5, G2):
            log_w = -(np.log(t) - mu) ** 2 / (2 * sigma ** 2) - np.log(t)
            acc += _make_ski_pmf(log_w)
    family_pmfs.append(acc / (G2 * G2))

    # --- 7. Weibull: beta in [0.3, 2], lam in [1, n/2] ---
    acc = np.zeros(n)
    for beta in np.linspace(0.3, 2.0, G2):
        for lam in np.linspace(1.0, n / 2, G2):
            log_w = -(t / lam) ** beta
            acc += _make_ski_pmf(log_w)
    family_pmfs.append(acc / (G2 * G2))

    # --- 8. Bimodal: lam1 in [1,n/4], lam2 in [n/2,n], p_high in [0.1,0.4] ---
    G3 = max(G // 10, 20)  # coarser for 3D
    acc = np.zeros(n)
    for lam1 in np.linspace(1.0, n / 4, G3):
        for lam2 in np.linspace(n / 2, float(n), G3):
            for p_high in np.linspace(0.1, 0.4, G3):
                log_w1 = t * np.log(lam1) - lam1 - log_fact[np.minimum(t.astype(int), n)]
                log_w2 = t * np.log(lam2) - lam2 - log_fact[np.minimum(t.astype(int), n)]
                pmf1 = _make_ski_pmf(log_w1)
                pmf2 = _make_ski_pmf(log_w2)
                pmf = (1 - p_high) * pmf1 + p_high * pmf2
                pmf /= pmf.sum()
                acc += pmf
    family_pmfs.append(acc / (G3 ** 3))

    # --- 9. Spike: k in {1,...,n} (discrete), eps in [0.01, 0.2] ---
    acc = np.zeros(n)
    count = 0
    for k in range(1, n + 1):
        for eps in np.linspace(0.01, 0.2, G2):
            pmf = np.full(n, eps / n)
            pmf[k - 1] += (1 - eps)
            pmf /= pmf.sum()
            acc += pmf
            count += 1
    family_pmfs.append(acc / count)

    # --- 10. Two-point: a in {1,...,n/2}, b in {n/2,...,n}, p_b in [0.05, 0.5] ---
    acc = np.zeros(n)
    count = 0
    a_max = max(1, n // 2)
    b_min = max(2, n // 2)
    for a in range(1, a_max + 1):
        for b in range(b_min, n + 1):
            for p_b in np.linspace(0.05, 0.5, G3):
                pmf = np.zeros(n)
                pmf[a - 1] = 1 - p_b
                pmf[b - 1] += p_b
                pmf /= pmf.sum()
                acc += pmf
                count += 1
    family_pmfs.append(acc / count)

    # Uniform mixture of 10 families
    mixture_pmf = np.mean(family_pmfs, axis=0)
    mixture_pmf /= mixture_pmf.sum()
    return mixture_pmf


def ski_experiments(model, args, out_dir):
    sdir = out_dir / "ski"
    sdir.mkdir(exist_ok=True)
    n, B, r = args.n, args.B, args.r
    seeds = eval_seeds(args)

    # Estimate training distribution PMF for Algorithm 2's U computation
    pmf_train = _compute_training_pmf(n)
    U_train = compute_U(pmf_train, B, r)
    log(f"Training PMF estimated, U={U_train} (tail threshold for Algorithm 2)")

    # ── Exp 1: In-distribution ──
    banner(f"SKI EXP 1 — IN-DISTRIBUTION  ({args.n_seeds} seeds)")
    all_rows = []
    for si, seed in enumerate(seeds):
        log(f"  seed {si+1}/{args.n_seeds}")
        insts = sample_ski_batch(args.n_eval, n, B, r, rng=np.random.default_rng(seed))
        res = compare_ski(insts, model, args.ski_lambdas, args.device, use_chain=True, pmf_train=pmf_train)
        all_rows.extend(_results_to_rows(res, {"experiment": "ski_in_dist", "n": n, "seed": seed}))
    save_csv(all_rows, sdir / "exp1_in_distribution.csv")

    # ── Exp 2: Per-family ──
    banner(f"SKI EXP 2 — PER-FAMILY  ({args.n_seeds} seeds)")
    all_rows = []
    for family in SKI_SAMPLERS:
        log(f"  Family: {family}")
        for si, seed in enumerate(seeds):
            insts = sample_ski_batch(200, n, B, r, dist_type=family,
                                      rng=np.random.default_rng(seed))
            res = compare_ski(insts, model, args.ski_lambdas, args.device, use_chain=True, pmf_train=pmf_train)
            all_rows.extend(_results_to_rows(res, {"experiment": "ski_by_family",
                                                    "family": family, "n": n, "seed": seed}))
    save_csv(all_rows, sdir / "exp2_by_family.csv")

    # ── Exp 3: Consistency-robustness frontier ──
    banner(f"SKI EXP 3 — FRONTIER  ({args.n_seeds} seeds)")
    lam_values = np.arange(0, 1.05, 0.1)
    all_rows = []
    for si, seed in enumerate(seeds):
        log(f"  seed {si+1}/{args.n_seeds}")
        insts = sample_ski_batch(args.n_eval, n, B, r, rng=np.random.default_rng(seed))
        V_hat_all = get_ski_predictions(model, insts, device=args.device, use_chain=True)
        learned_days = [ski_policy_learned(V_hat_all[i]) for i in range(len(insts))]
        opt_costs = np.array([ski_optimal_cost(inst.pmf_T, inst.n, inst.B, inst.r) for inst in insts])

        for lam in lam_values:
            robust_days = [ski_policy_robust(inst, learned_days[i], lam, U=U_train) for i, inst in enumerate(insts)]
            robust_costs = np.array([ski_policy_cost(robust_days[i], inst.pmf_T, inst.n, inst.B, inst.r)
                                      for i, inst in enumerate(insts)])
            det_K = ski_policy_deterministic(B, r)
            det_costs = np.array([ski_policy_cost(det_K, inst.pmf_T, inst.n, inst.B, inst.r) for inst in insts])
            learned_costs = np.array([ski_policy_cost(learned_days[i], inst.pmf_T, inst.n, inst.B, inst.r)
                                       for i, inst in enumerate(insts)])
            all_rows.append({
                "lambda": round(float(lam), 2),
                "loss_robust": round(float(np.mean(robust_costs - opt_costs)), 5),
                "loss_deterministic": round(float(np.mean(det_costs - opt_costs)), 5),
                "loss_learned": round(float(np.mean(learned_costs - opt_costs)), 5),
                "n": n, "seed": seed,
            })
    save_csv(all_rows, sdir / "exp3_frontier.csv")

    # ── Exp 4: Cost-ratio sensitivity ──
    banner(f"SKI EXP 4 — COST-RATIO SENSITIVITY  ({args.n_seeds} seeds)")
    all_rows = []
    # Test ratios: inside training range [10,100] and extrapolation (5, 150, 200)
    for ratio in [5, 10, 20, 50, 75, 100, 150, 200]:
        B_test = ratio * r
        log(f"  B/r = {ratio}")
        for si, seed in enumerate(seeds):
            insts = sample_ski_batch(args.n_eval, n, B_test, r, rng=np.random.default_rng(seed))
            res = compare_ski(insts, model, [0.0, 0.5, 1.0], args.device, use_chain=True, pmf_train=pmf_train)
            all_rows.extend(_results_to_rows(res, {"experiment": "ski_cost_ratio",
                                                    "B_over_r": ratio, "n": n, "seed": seed}))
    save_csv(all_rows, sdir / "exp4_cost_ratio.csv")

    # ── Exp 5: Heavy-tail ──
    banner(f"SKI EXP 5 — HEAVY-TAIL  ({args.n_seeds} seeds)")
    all_rows = []
    for family in SKI_HEAVY_TAIL_FAMILIES:
        log(f"  Family: {family}")
        for si, seed in enumerate(seeds):
            insts = sample_ski_batch(500, n, B, r, dist_type=family,
                                      rng=np.random.default_rng(seed))
            res = compare_ski(insts, model, args.ski_lambdas, args.device, use_chain=True, pmf_train=pmf_train)
            all_rows.extend(_results_to_rows(res, {"experiment": "ski_heavy_tail",
                                                    "family": family, "n": n, "seed": seed}))
    save_csv(all_rows, sdir / "exp5_heavy_tail.csv")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Task
    p.add_argument("--n", type=int, default=20, help="Training sequence length")
    p.add_argument("--M", type=int, default=1000, help="Domain size for stopping")
    p.add_argument("--B", type=float, default=10.0, help="Default buy cost for ski rental")
    p.add_argument("--r", type=float, default=1.0, help="Rent cost for ski rental")
    p.add_argument("--B_min", type=int, default=10, help="Min buy cost for training (inclusive)")
    p.add_argument("--B_max", type=int, default=100, help="Max buy cost for training (inclusive)")
    p.add_argument("--train_size", type=int, default=10_000)
    p.add_argument("--val_size", type=int, default=1_000)
    p.add_argument("--n_eval", type=int, default=2_000)
    # Model
    p.add_argument("--d_model", type=int, default=None, help="Hidden dim (default: auto from M)")
    p.add_argument("--n_heads", type=int, default=2)
    p.add_argument("--n_layers", type=int, default=2, help="Transformer layers (2 = minimum for Prop 1)")
    p.add_argument("--d_ff", type=int, default=None, help="FFN width (default: auto from M, B, n)")
    p.add_argument("--max_n", type=int, default=501)
    p.add_argument("--train_n_min", type=int, default=20, help="Min training horizon (inclusive)")
    p.add_argument("--train_n_max", type=int, default=200, help="Max training horizon (inclusive)")
    p.add_argument("--dropout", type=float, default=0.1)
    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--w_value", type=float, default=1.0, help="Weight on L_value (value-to-go MSE)")
    p.add_argument("--w_action", type=float, default=0.5, help="Weight on L_action (decision CE)")
    p.add_argument("--w_chain", type=float, default=0.0, help="Weight on L_chain (DP chain MSE)")
    # Evaluation
    p.add_argument("--betas", type=float, nargs="+", default=[0.1, 0.15, 0.2, 0.25])
    p.add_argument("--r_fractions", type=float, nargs="+", default=[0.1, 0.2, 0.368, 0.5])
    p.add_argument("--ski_lambdas", type=float, nargs="+", default=[0.0, 0.2, 0.5, 0.8, 1.0])
    p.add_argument("--test_horizons", type=int, nargs="+",
                   default=[5, 10, 15, 20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500])
    p.add_argument("--n_seeds", type=int, default=10, help="Number of eval seeds per experiment")
    # Misc
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--checkpoint_stop", type=str, default=None)
    p.add_argument("--checkpoint_ski", type=str, default=None)
    p.add_argument("--only", type=str, default=None, help="Run only 'stopping' or 'ski'")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"run_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), out_dir / "config.json")

    banner("ONLINE DECISION TRANSFORMER — PIPELINE")
    log(f"Output: {out_dir}")
    log(f"n={args.n} M={args.M} B={args.B} r={args.r} layers={args.n_layers} "
        f"train_horizon=[{args.train_n_min},{args.train_n_max}] device={args.device}")
    log(f"Eval seeds per experiment: {args.n_seeds}")

    # ── Stopping ──
    if args.only is None or args.only == "stopping":
        stop_dir = out_dir / "stopping"
        stop_dir.mkdir(exist_ok=True)

        if args.skip_train and args.checkpoint_stop:
            banner("LOADING STOPPING MODEL")
            stop_model = OnlineDecisionTransformer(
                M=args.M, d_model=args.d_model, n_heads=args.n_heads,
                n_layers=args.n_layers, d_ff=args.d_ff, max_n=args.max_n, dropout=args.dropout)
            ckpt = torch.load(args.checkpoint_stop, map_location=args.device)
            stop_model.load_state_dict(ckpt["model_state"])
            stop_model.to(args.device)
        else:
            banner("TRAINING STOPPING MODEL")
            log(f"Training horizon range: [{args.train_n_min}, {args.train_n_max}]")
            train_ds = StoppingDataset(args.train_size, args.n, args.M, seed=args.seed,
                                       n_min=args.train_n_min, n_max=args.train_n_max)
            val_ds = StoppingDataset(args.val_size, args.n, args.M, seed=args.seed + 1,
                                     n_min=args.train_n_min, n_max=args.train_n_max)
            stop_model = OnlineDecisionTransformer(
                M=args.M, d_model=args.d_model, n_heads=args.n_heads,
                n_layers=args.n_layers, d_ff=args.d_ff, max_n=args.max_n, dropout=args.dropout)
            n_params = sum(p.numel() for p in stop_model.parameters())
            log(f"Parameters: {n_params:,}")
            stop_model, stop_logs = run_train(
                stop_model, make_dataloader(train_ds, args.batch_size),
                make_dataloader(val_ds, args.batch_size, shuffle=False),
                problem="stopping", n=args.n, M=args.M,
                lr=args.lr, epochs=args.epochs,
                w_value=args.w_value, w_action=args.w_action, w_chain=args.w_chain,
                device=args.device, checkpoint_path=str(stop_dir / "checkpoint.pt"))
            save_csv(stop_logs, stop_dir / "training_log.csv")

        stopping_experiments(stop_model, args, out_dir)

    # ── Ski rental ──
    if args.only is None or args.only == "ski":
        ski_dir = out_dir / "ski"
        ski_dir.mkdir(exist_ok=True)

        if args.skip_train and args.checkpoint_ski:
            banner("LOADING SKI MODEL")
            ski_model = OnlineDecisionTransformer(
                M=args.M, d_model=args.d_model, n_heads=args.n_heads,
                n_layers=args.n_layers, d_ff=args.d_ff, max_n=args.max_n, dropout=args.dropout)
            ckpt = torch.load(args.checkpoint_ski, map_location=args.device)
            ski_model.load_state_dict(ckpt["model_state"])
            ski_model.to(args.device)
        else:
            banner("TRAINING SKI RENTAL MODEL")
            log(f"Training horizon range: [{args.train_n_min}, {args.train_n_max}], "
                f"B range: [{args.B_min}, {args.B_max}], r={args.r}")
            train_ds = SkiRentalDataset(args.train_size, args.n, args.B, args.r, seed=args.seed + 10,
                                         n_min=args.train_n_min, n_max=args.train_n_max,
                                         B_min=args.B_min, B_max=args.B_max)
            val_ds = SkiRentalDataset(args.val_size, args.n, args.B, args.r, seed=args.seed + 11,
                                       n_min=args.train_n_min, n_max=args.train_n_max,
                                       B_min=args.B_min, B_max=args.B_max)
            ski_model = OnlineDecisionTransformer(
                M=args.M, d_model=args.d_model, n_heads=args.n_heads,
                n_layers=args.n_layers, d_ff=args.d_ff, max_n=args.max_n, dropout=args.dropout)
            n_params = sum(p.numel() for p in ski_model.parameters())
            log(f"Parameters: {n_params:,}")
            ski_model, ski_logs = run_train(
                ski_model, make_dataloader(train_ds, args.batch_size),
                make_dataloader(val_ds, args.batch_size, shuffle=False),
                problem="ski", n=args.n, B=args.B,
                lr=args.lr, epochs=args.epochs,
                w_value=args.w_value, w_action=args.w_action, w_chain=args.w_chain,
                device=args.device, checkpoint_path=str(ski_dir / "checkpoint.pt"))
            save_csv(ski_logs, ski_dir / "training_log.csv")

        ski_experiments(ski_model, args, out_dir)

    # ── Plots ──
    banner("PLOTS")
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    generate_all_plots(out_dir)

    # Copy plots to project-level plots/ folder
    import shutil
    proj_plots = Path(__file__).parent.parent / "plots"
    proj_plots.mkdir(exist_ok=True)
    for png in plots_dir.glob("*.png"):
        shutil.copy2(png, proj_plots / png.name)
    log(f"Plots also copied to {proj_plots}/")

    banner("DONE")
    log(f"Total elapsed: {elapsed()}")


if __name__ == "__main__":
    main()
