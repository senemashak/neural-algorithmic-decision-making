"""
Sweep robust training for BOTH optimal stopping and ski rental.

For each robustness parameter (beta for stopping, lambda for ski):
  - Standard models: trained ONCE (no masking), evaluated at each param
  - Robust-masked models: trained per param, loss masked to useful positions

Every model is saved and evaluated. Plotting is fully decoupled from training.

Output structure:
    results/sweep_robust/sweep_<timestamp>/
        results.json
        models/
            stopping_standard_s0.pt          — one per seed (reused across betas)
            stopping_masked_beta0p10_s0.pt   — one per beta × seed
            ski_standard_s0.pt               — one per seed (reused across lambdas)
            ski_masked_lam0p5_s0.pt          — one per lambda × seed

Usage:
    python sweep_robust.py
    python sweep_robust.py --device cuda --epochs 30
    python sweep_robust.py --w_value 0.33 --w_action 0.33 --w_chain 0.33
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from sampling import (
    sample_stopping_batch, sample_ski_batch,
    STOPPING_SAMPLERS, SKI_SAMPLERS,
)
from dataset import StoppingDataset, SkiRentalDataset, make_dataloader
from model import OnlineDecisionTransformer
from train import train as run_train
from deployment import (
    compare_stopping, compare_ski, compute_U, find_lambdas,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--M", type=int, default=1000)
    p.add_argument("--B", type=float, default=10.0)
    p.add_argument("--r", type=float, default=1.0)
    p.add_argument("--B_min", type=int, default=10)
    p.add_argument("--B_max", type=int, default=100)
    p.add_argument("--train_size", type=int, default=5000)
    p.add_argument("--val_size", type=int, default=500)
    p.add_argument("--n_eval", type=int, default=1000)
    p.add_argument("--n_min", type=int, default=20)
    p.add_argument("--n_max", type=int, default=200)
    p.add_argument("--d_model", type=int, default=None, help="Hidden dim (default: auto from M)")
    p.add_argument("--n_heads", type=int, default=2)
    p.add_argument("--n_layers", type=int, default=2, help="Transformer layers (2 = minimum for Prop 1)")
    p.add_argument("--d_ff", type=int, default=None, help="FFN width (default: auto from M, B, n)")
    p.add_argument("--max_n", type=int, default=501)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    # Loss weights — set to whatever sweep_weights found best
    p.add_argument("--w_value", type=float, default=1.0)
    p.add_argument("--w_action", type=float, default=0.5)
    p.add_argument("--w_chain", type=float, default=0.0)
    p.add_argument("--training_mode", type=str, default="teacher_forcing",
                   choices=["teacher_forcing", "autoregressive"])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_train_seeds", type=int, default=3)
    p.add_argument("--n_eval_seeds", type=int, default=5)
    p.add_argument("--out_dir", type=str, default="results/sweep_robust")
    return p.parse_args()


# Betas where masked training makes sense (0 < beta < 1/e)
STOPPING_BETAS_MASKED = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
# All betas to evaluate (including boundaries)
# beta=0: no wrapper (pure learned policy)
# beta=0.367: near 1/e ≈ 0.3679 (wrapper ignores model almost entirely — approaches Dynkin)
STOPPING_BETAS_ALL = [0.0] + STOPPING_BETAS_MASKED + [0.367]

# Lambdas where masked training makes sense (0 < lambda < 1)
SKI_LAMBDAS_MASKED = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# All lambdas to evaluate (including boundaries)
SKI_LAMBDAS_ALL = [0.0] + SKI_LAMBDAS_MASKED + [1.0]


def _make_model(args):
    return OnlineDecisionTransformer(
        M=args.M, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff, max_n=args.max_n)


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def _eval_stopping_robust(model, args, beta, eval_seeds):
    """Eval stopping model with robust wrapper at given beta.
    Tests on all 11 families. beta=0 means learned policy only (no wrapper)."""
    n, M = args.n, args.M
    use_wrapper = beta > 0
    rob_key = f"robust β={beta:.2f}" if use_wrapper else "learned"
    betas_arg = [beta] if use_wrapper else []

    result = {
        "in_dist_cr_learned": [], "in_dist_cr_robust": [],
        "in_dist_pbest_learned": [], "in_dist_pbest_robust": [],
        "per_family_cr_learned": {f: [] for f in STOPPING_SAMPLERS},
        "per_family_cr_robust": {f: [] for f in STOPPING_SAMPLERS},
        "per_family_pbest_learned": {f: [] for f in STOPPING_SAMPLERS},
        "per_family_pbest_robust": {f: [] for f in STOPPING_SAMPLERS},
        "baselines": {"dp_cr": [], "dp_prob_best": [], "offline_cr": []},
    }

    for seed in eval_seeds:
        rng = np.random.default_rng(seed)
        insts = sample_stopping_batch(args.n_eval, n, M, rng=rng)
        res = compare_stopping(insts, model, betas=betas_arg, r_fractions=[], device=args.device)
        result["in_dist_cr_learned"].append(res["learned"]["cr"])
        result["in_dist_pbest_learned"].append(res["learned"]["prob_best"])
        result["in_dist_cr_robust"].append(res[rob_key]["cr"])
        result["in_dist_pbest_robust"].append(res[rob_key]["prob_best"])
        result["baselines"]["dp_cr"].append(res["dp"]["cr"])
        result["baselines"]["dp_prob_best"].append(res["dp"]["prob_best"])
        result["baselines"]["offline_cr"].append(res["offline"]["cr"])

        for fam in STOPPING_SAMPLERS:
            fam_rng = np.random.default_rng(seed + hash(fam) % 10000)
            fam_insts = sample_stopping_batch(
                args.n_eval // len(STOPPING_SAMPLERS), n, M,
                dist_type=fam, rng=fam_rng)
            fam_res = compare_stopping(fam_insts, model, betas=betas_arg, r_fractions=[],
                                       device=args.device)
            result["per_family_cr_learned"][fam].append(fam_res["learned"]["cr"])
            result["per_family_cr_robust"][fam].append(fam_res[rob_key]["cr"])
            result["per_family_pbest_learned"][fam].append(fam_res["learned"]["prob_best"])
            result["per_family_pbest_robust"][fam].append(fam_res[rob_key]["prob_best"])

    return result


def _eval_ski_robust(model, args, lam, eval_seeds, pmf_train):
    """Eval ski model with robust wrapper at given lambda.
    Tests on all 10 families. lam=0 means learned only, lam=1 means deterministic."""
    n, B, r = args.n, args.B, args.r
    # lam=0: robust wrapper returns K_hat unchanged → same as "learned"
    # lam=1: robust wrapper returns floor(B/r) → deterministic, but we still
    #        route through the wrapper so it shows up in results consistently
    rob_key = f"robust λ={lam:.1f}" if lam > 0 else "learned"
    lambdas_arg = [lam] if lam > 0 else []

    result = {
        "in_dist_cr_learned": [], "in_dist_cr_robust": [],
        "in_dist_loss_learned": [], "in_dist_loss_robust": [],
        "per_family_cr_learned": {f: [] for f in SKI_SAMPLERS},
        "per_family_cr_robust": {f: [] for f in SKI_SAMPLERS},
        "per_family_loss_learned": {f: [] for f in SKI_SAMPLERS},
        "per_family_loss_robust": {f: [] for f in SKI_SAMPLERS},
        "baselines": {"dp_cr": [], "dp_loss": [], "deterministic_cr": [], "deterministic_loss": []},
    }

    for seed in eval_seeds:
        rng = np.random.default_rng(seed)
        insts = sample_ski_batch(args.n_eval, n, B, r, rng=rng)
        res = compare_ski(insts, model, lambdas=lambdas_arg, device=args.device, pmf_train=pmf_train)
        result["in_dist_cr_learned"].append(res["learned"]["cr"])
        result["in_dist_loss_learned"].append(res["learned"]["mean_additive_loss"])
        result["in_dist_cr_robust"].append(res[rob_key]["cr"])
        result["in_dist_loss_robust"].append(res[rob_key]["mean_additive_loss"])
        result["baselines"]["dp_cr"].append(res["dp"]["cr"])
        result["baselines"]["dp_loss"].append(res["dp"]["mean_additive_loss"])
        result["baselines"]["deterministic_cr"].append(res["deterministic"]["cr"])
        result["baselines"]["deterministic_loss"].append(res["deterministic"]["mean_additive_loss"])

        for fam in SKI_SAMPLERS:
            fam_rng = np.random.default_rng(seed + hash(fam) % 10000)
            fam_insts = sample_ski_batch(
                args.n_eval // len(SKI_SAMPLERS), n, B, r,
                dist_type=fam, rng=fam_rng)
            fam_res = compare_ski(fam_insts, model, lambdas=lambdas_arg, device=args.device,
                                  pmf_train=pmf_train)
            result["per_family_cr_learned"][fam].append(fam_res["learned"]["cr"])
            result["per_family_cr_robust"][fam].append(fam_res[rob_key]["cr"])
            result["per_family_loss_learned"][fam].append(fam_res["learned"]["mean_additive_loss"])
            result["per_family_loss_robust"][fam].append(fam_res[rob_key]["mean_additive_loss"])

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Training helpers
# ═══════════════════════════════════════════════════════════════════════════

def _train_stopping(args, model_dir, train_seed, ckpt_name,
                    robust_train=False, robust_beta=None):
    train_ds = StoppingDataset(args.train_size, args.n, args.M, seed=train_seed,
                               n_min=args.n_min, n_max=args.n_max)
    val_ds = StoppingDataset(args.val_size, args.n, args.M, seed=train_seed + 1,
                             n_min=args.n_min, n_max=args.n_max)
    torch.manual_seed(train_seed)
    model = _make_model(args)
    ckpt = str(model_dir / ckpt_name)
    model, logs = run_train(
        model, make_dataloader(train_ds, args.batch_size),
        make_dataloader(val_ds, args.batch_size, shuffle=False),
        problem="stopping", n=args.n, M=args.M,
        lr=args.lr, epochs=args.epochs,
        w_value=args.w_value, w_action=args.w_action, w_chain=args.w_chain,
        training_mode=args.training_mode,
        robust_train=robust_train, robust_beta=robust_beta,
        device=args.device, checkpoint_path=ckpt)
    return model, logs, ckpt


def _train_ski(args, model_dir, train_seed, ckpt_name, U_train,
               robust_train=False, robust_lambda=None):
    train_ds = SkiRentalDataset(args.train_size, args.n, args.B, args.r, seed=train_seed,
                                n_min=args.n_min, n_max=args.n_max,
                                B_min=args.B_min, B_max=args.B_max)
    val_ds = SkiRentalDataset(args.val_size, args.n, args.B, args.r, seed=train_seed + 1,
                              n_min=args.n_min, n_max=args.n_max,
                              B_min=args.B_min, B_max=args.B_max)
    torch.manual_seed(train_seed)
    model = _make_model(args)
    ckpt = str(model_dir / ckpt_name)
    model, logs = run_train(
        model, make_dataloader(train_ds, args.batch_size),
        make_dataloader(val_ds, args.batch_size, shuffle=False),
        problem="ski", n=args.n, M=args.M, B=args.B, r=args.r,
        lr=args.lr, epochs=args.epochs,
        w_value=args.w_value, w_action=args.w_action, w_chain=args.w_chain,
        training_mode=args.training_mode,
        robust_train=robust_train, robust_lambda=robust_lambda, robust_U=U_train,
        device=args.device, checkpoint_path=ckpt)
    return model, logs, ckpt


# ═══════════════════════════════════════════════════════════════════════════
# Main sweep
# ═══════════════════════════════════════════════════════════════════════════

def run_sweep(args):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"sweep_{stamp}"
    model_dir = out_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    eval_seeds = [args.seed + 50000 + i for i in range(args.n_eval_seeds)]

    from main import _compute_training_pmf
    pmf_train = _compute_training_pmf(args.n)
    U_train = compute_U(pmf_train, args.B, args.r)

    output = {
        "meta": {
            "stopping_betas_all": STOPPING_BETAS_ALL,
            "stopping_betas_masked": STOPPING_BETAS_MASKED,
            "ski_lambdas_all": SKI_LAMBDAS_ALL,
            "ski_lambdas_masked": SKI_LAMBDAS_MASKED,
            "modes": ["standard", "robust_masked"],
            "loss_weights": {"w_value": args.w_value, "w_action": args.w_action, "w_chain": args.w_chain},
            "training_mode": args.training_mode,
            "n_train_seeds": args.n_train_seeds,
            "n_eval_seeds": args.n_eval_seeds,
            "eval_seeds": eval_seeds,
            "stopping_families_tested": list(STOPPING_SAMPLERS.keys()),
            "ski_families_tested": list(SKI_SAMPLERS.keys()),
            "U_train": U_train,
            "args": vars(args),
            "timestamp": stamp,
        },
        "stopping": [],
        "ski": [],
    }

    print(f"{'='*60}")
    print(f"  Robust Training Sweep — stopping + ski rental")
    print(f"  Loss weights: w_v={args.w_value}, w_a={args.w_action}, w_c={args.w_chain}")
    print(f"  Training mode: {args.training_mode}")
    print(f"  {len(STOPPING_BETAS_ALL)} betas (eval) / {len(STOPPING_BETAS_MASKED)} (masked train)")
    print(f"  {len(SKI_LAMBDAS_ALL)} lambdas (eval) / {len(SKI_LAMBDAS_MASKED)} (masked train)")
    print(f"  {args.n_train_seeds} train seeds × {args.n_eval_seeds} eval seeds")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")

    # ══════════════════════════════════════════════════════════════════════════
    # STOPPING
    # ══════════════════════════════════════════════════════════════════════════

    # --- Train standard models ONCE (no masking, reused across all betas) ---
    print(f"\n{'='*60}")
    print(f"  STOPPING — training standard models (shared across all betas)")
    print(f"{'='*60}")

    std_stop_models = []   # (model, logs, ckpt_path) per seed
    for ts in range(args.n_train_seeds):
        train_seed = args.seed + ts * 1000
        print(f"  train seed {ts+1}/{args.n_train_seeds}")
        ckpt_name = f"stopping_standard_s{ts}.pt"
        model, logs, ckpt = _train_stopping(args, model_dir, train_seed, ckpt_name)
        std_stop_models.append((model, logs, ckpt))

    # --- Evaluate standard models at ALL betas, train+eval masked at valid betas ---
    for beta in STOPPING_BETAS_ALL:
        if beta > 0:
            lam1, lam2 = find_lambdas(beta)
            phase_str = f"  (lam1={lam1:.3f}, lam2={lam2:.3f})"
        else:
            lam1, lam2 = 0.0, 1.0  # beta=0: entire horizon is "middle phase"
            phase_str = "  (no wrapper — learned policy only)"
        print(f"\n{'='*60}")
        print(f"  STOPPING  beta={beta:.2f}{phase_str}")
        print(f"{'='*60}")

        # Standard: evaluate only (already trained)
        print(f"\n  --- standard (eval only) ---")
        entry_std = {
            "beta": beta, "lambda1": lam1, "lambda2": lam2,
            "mode": "standard",
            "seeds": [],
            "best_seed_idx": None,
        }
        for ts, (model, logs, ckpt) in enumerate(std_stop_models):
            print(f"    evaluating seed {ts+1}/{args.n_train_seeds}...")
            eval_result = _eval_stopping_robust(model, args, beta, eval_seeds)
            cr_r = float(np.mean(eval_result["in_dist_cr_robust"]))
            print(f"      CR(robust)={cr_r:.4f}")
            entry_std["seeds"].append({
                "train_seed": args.seed + ts * 1000,
                "training_curve": logs,
                "val_loss": min(log["val_total"] for log in logs),
                "model_path": f"models/stopping_standard_s{ts}.pt",
                "eval": eval_result,
            })

        scores = [np.mean(s["eval"]["in_dist_cr_robust"]) for s in entry_std["seeds"]]
        entry_std["best_seed_idx"] = int(np.argmax(scores))
        output["stopping"].append(entry_std)

        # Masked: train + evaluate (only for valid betas)
        if beta in STOPPING_BETAS_MASKED:
            print(f"\n  --- robust_masked (train + eval) ---")
            entry_msk = {
                "beta": beta, "lambda1": lam1, "lambda2": lam2,
                "mode": "robust_masked",
                "seeds": [],
                "best_seed_idx": None,
            }
            for ts in range(args.n_train_seeds):
                train_seed = args.seed + ts * 1000
                print(f"    train seed {ts+1}/{args.n_train_seeds}")
                safe = f"stopping_masked_beta{beta:.2f}_s{ts}".replace('.', 'p')
                model, logs, ckpt = _train_stopping(
                    args, model_dir, train_seed, f"{safe}.pt",
                    robust_train=True, robust_beta=beta)

                print(f"      evaluating...")
                eval_result = _eval_stopping_robust(model, args, beta, eval_seeds)
                cr_r = float(np.mean(eval_result["in_dist_cr_robust"]))
                print(f"      CR(robust)={cr_r:.4f}")

                entry_msk["seeds"].append({
                    "train_seed": train_seed,
                    "training_curve": logs,
                    "val_loss": min(log["val_total"] for log in logs),
                    "model_path": f"models/{safe}.pt",
                    "eval": eval_result,
                })

            scores = [np.mean(s["eval"]["in_dist_cr_robust"]) for s in entry_msk["seeds"]]
            entry_msk["best_seed_idx"] = int(np.argmax(scores))
            output["stopping"].append(entry_msk)

    # ══════════════════════════════════════════════════════════════════════════
    # SKI RENTAL
    # ══════════════════════════════════════════════════════════════════════════

    # --- Train standard models ONCE (reused across all lambdas) ---
    print(f"\n{'='*60}")
    print(f"  SKI RENTAL — training standard models (shared across all lambdas)")
    print(f"{'='*60}")

    std_ski_models = []
    for ts in range(args.n_train_seeds):
        train_seed = args.seed + ts * 1000
        print(f"  train seed {ts+1}/{args.n_train_seeds}")
        ckpt_name = f"ski_standard_s{ts}.pt"
        model, logs, ckpt = _train_ski(args, model_dir, train_seed, ckpt_name, U_train)
        std_ski_models.append((model, logs, ckpt))

    # --- Evaluate standard at ALL lambdas, train+eval masked at valid lambdas ---
    for lam in SKI_LAMBDAS_ALL:
        if lam == 0:
            lam_str = "  (no wrapper — learned policy only)"
        elif lam == 1:
            lam_str = "  (fully deterministic — floor(B/r))"
        else:
            lam_str = ""
        print(f"\n{'='*60}")
        print(f"  SKI RENTAL  lambda={lam:.1f}{lam_str}")
        print(f"{'='*60}")

        # Standard: eval only
        print(f"\n  --- standard (eval only) ---")
        entry_std = {
            "lambda": lam,
            "mode": "standard",
            "seeds": [],
            "best_seed_idx": None,
        }
        for ts, (model, logs, ckpt) in enumerate(std_ski_models):
            print(f"    evaluating seed {ts+1}/{args.n_train_seeds}...")
            eval_result = _eval_ski_robust(model, args, lam, eval_seeds, pmf_train)
            lo_r = float(np.mean(eval_result["in_dist_loss_robust"]))
            print(f"      loss(robust)={lo_r:.4f}")
            entry_std["seeds"].append({
                "train_seed": args.seed + ts * 1000,
                "training_curve": logs,
                "val_loss": min(log["val_total"] for log in logs),
                "model_path": f"models/ski_standard_s{ts}.pt",
                "eval": eval_result,
            })

        scores = [np.mean(s["eval"]["in_dist_loss_robust"]) for s in entry_std["seeds"]]
        entry_std["best_seed_idx"] = int(np.argmin(scores))
        output["ski"].append(entry_std)

        # Masked: train + eval (only for valid lambdas)
        if lam in SKI_LAMBDAS_MASKED:
            print(f"\n  --- robust_masked (train + eval) ---")
            entry_msk = {
                "lambda": lam,
                "mode": "robust_masked",
                "seeds": [],
                "best_seed_idx": None,
            }
            for ts in range(args.n_train_seeds):
                train_seed = args.seed + ts * 1000
                print(f"    train seed {ts+1}/{args.n_train_seeds}")
                safe = f"ski_masked_lam{lam:.1f}_s{ts}".replace('.', 'p')
                model, logs, ckpt = _train_ski(
                    args, model_dir, train_seed, f"{safe}.pt", U_train,
                    robust_train=True, robust_lambda=lam)

                print(f"      evaluating...")
                eval_result = _eval_ski_robust(model, args, lam, eval_seeds, pmf_train)
                lo_r = float(np.mean(eval_result["in_dist_loss_robust"]))
                print(f"      loss(robust)={lo_r:.4f}")

                entry_msk["seeds"].append({
                    "train_seed": train_seed,
                    "training_curve": logs,
                    "val_loss": min(log["val_total"] for log in logs),
                    "model_path": f"models/{safe}.pt",
                    "eval": eval_result,
                })

            scores = [np.mean(s["eval"]["in_dist_loss_robust"]) for s in entry_msk["seeds"]]
            entry_msk["best_seed_idx"] = int(np.argmin(scores))
            output["ski"].append(entry_msk)

    # ── Save ──────────────────────────────────────────────────────────────────
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=float)

    # Print summary
    print(f"\n{'='*72}")
    print(f"  STOPPING — standard vs robust-masked (best-seed robust CR)")
    print(f"{'='*72}")
    print(f"{'beta':<8} {'Standard CR':>14} {'Masked CR':>14} {'Winner':>10}")
    print(f"{'─'*50}")
    for beta in STOPPING_BETAS_ALL:
        std = next(e for e in output["stopping"] if e["beta"] == beta and e["mode"] == "standard")
        s_cr = np.mean(std["seeds"][std["best_seed_idx"]]["eval"]["in_dist_cr_robust"])
        msk = next((e for e in output["stopping"] if e["beta"] == beta and e["mode"] == "robust_masked"), None)
        if msk is not None:
            m_cr = np.mean(msk["seeds"][msk["best_seed_idx"]]["eval"]["in_dist_cr_robust"])
            winner = "masked" if m_cr > s_cr else "standard"
            print(f"{beta:<8.2f} {s_cr:>14.4f} {m_cr:>14.4f} {winner:>10}")
        else:
            print(f"{beta:<8.2f} {s_cr:>14.4f} {'—':>14} {'—':>10}")

    print(f"\n{'='*72}")
    print(f"  SKI RENTAL — standard vs robust-masked (best-seed robust loss)")
    print(f"{'='*72}")
    print(f"{'lambda':<8} {'Standard loss':>14} {'Masked loss':>14} {'Winner':>10}")
    print(f"{'─'*50}")
    for lam in SKI_LAMBDAS_ALL:
        std = next(e for e in output["ski"] if e["lambda"] == lam and e["mode"] == "standard")
        s_lo = np.mean(std["seeds"][std["best_seed_idx"]]["eval"]["in_dist_loss_robust"])
        msk = next((e for e in output["ski"] if e["lambda"] == lam and e["mode"] == "robust_masked"), None)
        if msk is not None:
            m_lo = np.mean(msk["seeds"][msk["best_seed_idx"]]["eval"]["in_dist_loss_robust"])
            winner = "masked" if m_lo < s_lo else "standard"
            print(f"{lam:<8.1f} {s_lo:>14.4f} {m_lo:>14.4f} {winner:>10}")
        else:
            print(f"{lam:<8.1f} {s_lo:>14.4f} {'—':>14} {'—':>10}")

    print(f"\nResults → {json_path}")
    print(f"Models  → {model_dir}/")
    n_models = len(list(model_dir.glob("*.pt")))
    print(f"  ({n_models} model checkpoints saved)")


def main():
    args = parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
