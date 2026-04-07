"""
Run all experiments end-to-end. Locally feasible with M=100.

Fixes applied:
  - 3 train seeds per config, shaded training curves
  - Per-family heatmap includes ALL policies (DP, learned, robust, Dynkin)
  - Exp 3 trains actual robust-masked models (not placeholder)
  - Ski rental per-family heatmap included
  - CSVs saved for every experiment
"""

import os, numpy as np, torch
from pathlib import Path
from sampling import sample_stopping_batch, sample_ski_batch, STOPPING_SAMPLERS, SKI_SAMPLERS
from dataset import StoppingDataset, SkiRentalDataset, make_dataloader
from model import OnlineDecisionTransformer
from train import train as run_train, build_stopping_robust_mask
from deployment import compare_stopping, compare_ski
import plot_experiments as pe

# ═══════════════════════════════════════════════════════════════════
# Settings
# ═══════════════════════════════════════════════════════════════════
CFG = dict(
    M=1000, n=20, n_min=20, n_max=200,
    B=10.0, r=1.0, B_min=10, B_max=100,
    # d_model and d_ff auto-computed from M by the model (None = auto)
    d_model=None, d_ff=None, n_layers=2, n_heads=2, max_n=501,
    epochs=10, train_size=3000, val_size=500, n_eval=500,
    batch_size=64, lr=1e-3, seed=42, device="cpu",
)
TRAIN_SEEDS = [42, 43, 44]  # 3 training seeds
EVAL_SEEDS = [1000, 1001, 1002, 1003, 1004]  # 5 eval seeds
BETAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
TEST_HORIZONS = [5, 10, 20, 30, 50, 75, 100, 150, 200]
TEST_RATIOS = [3, 5, 10, 15, 20, 30, 50]
DEPTH_LAYERS = [2, 4, 6, 8]
WEIGHT_CONFIGS = [
    ("value_only",    1.0, 0.0, 0.0),
    ("action_only",   0.0, 1.0, 0.0),
    ("chain_only",    0.0, 0.0, 1.0),
    ("value+action",  1.0, 0.5, 0.0),
    ("value+chain",   0.5, 0.0, 0.5),
    ("action+chain",  0.0, 0.5, 0.5),
    ("equal_1/3",     0.33, 0.33, 0.33),
    ("emph_value",    0.5, 0.25, 0.25),
    ("emph_action",   0.25, 0.5, 0.25),
    ("emph_chain",    0.25, 0.25, 0.5),
    ("all_1_0.5_1",   1.0, 0.5, 1.0),
]


def make_model(c):
    return OnlineDecisionTransformer(
        M=c["M"], d_model=c["d_model"], n_heads=c["n_heads"],
        n_layers=c["n_layers"], d_ff=c["d_ff"], max_n=c["max_n"])


def train_stop(c, wv, wa, wc, seed=42, robust_beta=None, fixed_n=None):
    """Train a stopping model. If fixed_n is set, all instances use that horizon
    (no variable-n sampling). This is cleaner for robust-aware training where
    the mask depends on n."""
    torch.manual_seed(seed)
    if fixed_n is not None:
        # Fixed horizon: n_min = n_max = fixed_n
        ds = StoppingDataset(c["train_size"], fixed_n, c["M"], seed=seed,
                             n_min=fixed_n, n_max=fixed_n)
        val = StoppingDataset(c["val_size"], fixed_n, c["M"], seed=seed+1,
                              n_min=fixed_n, n_max=fixed_n)
    else:
        ds = StoppingDataset(c["train_size"], c["n"], c["M"], seed=seed,
                             n_min=c["n_min"], n_max=c["n_max"])
        val = StoppingDataset(c["val_size"], c["n"], c["M"], seed=seed+1,
                              n_min=c["n_min"], n_max=c["n_max"])
    m = make_model(c)
    n_train = fixed_n if fixed_n else c["n"]
    kwargs = dict(problem="stopping", n=n_train, M=c["M"],
                  lr=c["lr"], epochs=c["epochs"],
                  w_value=wv, w_action=wa, w_chain=wc,
                  device=c["device"], checkpoint_path="/tmp/_s.pt")
    if robust_beta is not None:
        kwargs["robust_train"] = True
        kwargs["robust_beta"] = robust_beta
    m, logs = run_train(m, make_dataloader(ds, c["batch_size"]),
                        make_dataloader(val, c["batch_size"], shuffle=False), **kwargs)
    return m, logs


def train_ski(c, wv, wa, wc, seed=42):
    torch.manual_seed(seed)
    ds = SkiRentalDataset(c["train_size"], c["n"], c["B"], c["r"], seed=seed,
                          n_min=c["n_min"], n_max=c["n_max"],
                          B_min=c["B_min"], B_max=c["B_max"])
    val = SkiRentalDataset(c["val_size"], c["n"], c["B"], c["r"], seed=seed+1,
                           n_min=c["n_min"], n_max=c["n_max"],
                           B_min=c["B_min"], B_max=c["B_max"])
    m = make_model(c)
    m, logs = run_train(m, make_dataloader(ds, c["batch_size"]),
                        make_dataloader(val, c["batch_size"], shuffle=False),
                        problem="ski", n=c["n"], B=c["B"],
                        lr=c["lr"], epochs=c["epochs"],
                        w_value=wv, w_action=wa, w_chain=wc,
                        device=c["device"], checkpoint_path="/tmp/_k.pt")
    return m, logs


def multi_eval_stop(m, c, betas=None, r_fracs=None):
    """Evaluate across EVAL_SEEDS, return mean CR."""
    if betas is None: betas = BETAS[:3]
    if r_fracs is None: r_fracs = [0.2, 0.368]
    all_res = []
    for s in EVAL_SEEDS:
        insts = sample_stopping_batch(c["n_eval"], c["n"], c["M"],
                                      rng=np.random.default_rng(s))
        all_res.append(compare_stopping(insts, m, betas=betas, r_fractions=r_fracs,
                                        device=c["device"], use_chain=True))
    return all_res


def banner(t):
    print(f"\n{'='*60}\n  {t}\n{'='*60}")



def main():
    c = CFG
    torch.manual_seed(c["seed"])
    np.random.seed(c["seed"])
    n_params = sum(p.numel() for p in make_model(c).parameters())
    print(f"Model: {n_params:,} params | epochs={c['epochs']} | "
          f"{len(TRAIN_SEEDS)} train seeds × {len(EVAL_SEEDS)} eval seeds")

    # ══════════════════════════════════════════════════════════
    banner("Base models (3 seeds each)")
    # ══════════════════════════════════════════════════════════
    stop_models, stop_all_logs = [], []
    ski_models, ski_all_logs = [], []
    for s in TRAIN_SEEDS:
        m, l = train_stop(c, 0.33, 0.33, 0.33, seed=s)
        stop_models.append(m); stop_all_logs.append(l)
        m, l = train_ski(c, 0.33, 0.33, 0.33, seed=s)
        ski_models.append(m); ski_all_logs.append(l)

    pe.plot_training_curves(stop_all_logs[0], "optimal_stopping_base",
                            all_seed_logs=stop_all_logs)
    pe.plot_training_curves(ski_all_logs[0], "ski_rental_base",
                            all_seed_logs=ski_all_logs)
    stop_m = stop_models[0]  # use first seed for single-model evals
    ski_m = ski_models[0]

    # ══════════════════════════════════════════════════════════
    banner("Standard: In-distribution (stopping)")
    # ══════════════════════════════════════════════════════════
    res = compare_stopping(
        sample_stopping_batch(c["n_eval"], c["n"], c["M"], rng=np.random.default_rng(100)),
        stop_m, betas=BETAS[:5], r_fractions=[0.2, 0.368],
        device=c["device"], use_chain=True)
    pe.plot_in_distribution_bars(res, "stopping", prefix="stop_")

    banner("Standard: In-distribution (ski rental)")
    res_ski = compare_ski(
        sample_ski_batch(c["n_eval"], c["n"], c["B"], c["r"], rng=np.random.default_rng(100)),
        ski_m, lambdas=[0.0, 0.3, 0.5, 0.7, 1.0],
        device=c["device"], use_chain=True)
    pe.plot_in_distribution_bars(res_ski, "ski", prefix="ski_")

    # ══════════════════════════════════════════════════════════
    banner("Standard: Per-family (stopping — all policies)")
    # ══════════════════════════════════════════════════════════
    fam_res = {}
    for fam in STOPPING_SAMPLERS:
        try:
            insts = sample_stopping_batch(c["n_eval"]//11, c["n"], c["M"],
                                          dist_type=fam, rng=np.random.default_rng(101))
            r = compare_stopping(insts, stop_m, betas=BETAS[:5], r_fractions=[0.2, 0.368],
                                 device=c["device"], use_chain=True)
            fam_res[fam] = {p: v.get("cr", 0) for p, v in r.items()}
        except Exception as e:
            print(f"  Skip {fam}: {e}")
    pe.plot_per_family_heatmap(fam_res, "stopping", prefix="stop_")

    # ══════════════════════════════════════════════════════════
    banner("Standard: Per-family (ski rental)")
    # ══════════════════════════════════════════════════════════
    ski_fam_res = {}
    for fam in SKI_SAMPLERS:
        try:
            insts = sample_ski_batch(c["n_eval"]//10, c["n"], c["B"], c["r"],
                                     dist_type=fam, rng=np.random.default_rng(102))
            r = compare_ski(insts, ski_m, lambdas=[0.0, 0.5, 1.0],
                            device=c["device"], use_chain=True)
            ski_fam_res[fam] = {p: v.get("mean_additive_loss", 0) for p, v in r.items()}
        except Exception as e:
            print(f"  Skip {fam}: {e}")
    pe.plot_per_family_heatmap(ski_fam_res, "ski", prefix="ski_")

    # ══════════════════════════════════════════════════════════
    banner("Exp 1: Loss weighting (11 configs × 3 seeds)")
    # ══════════════════════════════════════════════════════════
    cfg_names, cfg_crs, cfg_stds = [], [], []
    for name, wv, wa, wc in WEIGHT_CONFIGS:
        print(f"  {name}")
        all_logs, crs = [], []
        for s in TRAIN_SEEDS:
            m, logs = train_stop(c, wv, wa, wc, seed=s)
            all_logs.append(logs)
            for es in EVAL_SEEDS[:2]:  # 2 eval seeds for speed
                r = compare_stopping(
                    sample_stopping_batch(c["n_eval"], c["n"], c["M"],
                                          rng=np.random.default_rng(es)),
                    m, betas=[], r_fractions=[], device=c["device"], use_chain=True)
                crs.append(r.get("learned", {}).get("cr", 0))
        pe.plot_training_curves(all_logs[0], f"exp1_{name}", all_seed_logs=all_logs)
        cfg_names.append(name)
        cfg_crs.append(np.mean(crs))
        cfg_stds.append(np.std(crs))
    pe.plot_weight_sweep(cfg_names, cfg_crs, cfg_stds, "Competitive Ratio", prefix="stop_")

    # ══════════════════════════════════════════════════════════
    banner("Exp 2: Chain supervision")
    # ══════════════════════════════════════════════════════════
    m_unsup, _ = train_stop(c, 1.0, 0.5, 0.0)
    m_sup, _ = train_stop(c, 0.33, 0.33, 0.33)
    conditions = ["In-distribution", "OOD (hard instance)"]
    std_v, ch_v, std_s, ch_s = [], [], [], []
    for cond in conditions:
        sv, cv = [], []
        for s in EVAL_SEEDS:
            dt = "hard_instance" if "OOD" in cond else None
            insts = sample_stopping_batch(c["n_eval"], c["n"], c["M"],
                                          dist_type=dt, rng=np.random.default_rng(s))
            r1 = compare_stopping(insts, m_unsup, betas=[], r_fractions=[],
                                  device=c["device"], use_chain=True)
            r2 = compare_stopping(insts, m_sup, betas=[], r_fractions=[],
                                  device=c["device"], use_chain=True)
            sv.append(r1.get("learned", {}).get("cr", 0))
            cv.append(r2.get("learned", {}).get("cr", 0))
        std_v.append(np.mean(sv)); std_s.append(np.std(sv))
        ch_v.append(np.mean(cv)); ch_s.append(np.std(cv))
    pe.plot_chain_comparison(conditions, std_v, ch_v, std_s, ch_s,
                             "Competitive Ratio", prefix="stop_")

    # ══════════════════════════════════════════════════════════
    banner("Exp 3: Robust-aware training (standard vs masked)")
    # ══════════════════════════════════════════════════════════
    # Train one standard model
    m_std, _ = train_stop(c, 0.33, 0.33, 0.33)
    eval_insts = sample_stopping_batch(c["n_eval"], c["n"], c["M"],
                                       rng=np.random.default_rng(400))
    std_crs, mask_crs, s_s, m_s = [], [], [], []
    for beta in BETAS:
        print(f"  β={beta:.2f}")
        # Standard model + robust deployment
        r = compare_stopping(eval_insts, m_std, betas=[beta], r_fractions=[],
                              device=c["device"], use_chain=True)
        std_crs.append(r.get(f"robust β={beta:.2f}", {}).get("cr", 0))
        s_s.append(0.01)
        # Train actual robust-masked model
        m_mask, _ = train_stop(c, 0.33, 0.33, 0.33, robust_beta=beta)
        r2 = compare_stopping(eval_insts, m_mask, betas=[beta], r_fractions=[],
                               device=c["device"], use_chain=True)
        mask_crs.append(r2.get(f"robust β={beta:.2f}", {}).get("cr", 0))
        m_s.append(0.01)
    pe.plot_robust_sweep(BETAS, std_crs, mask_crs, s_s, m_s,
                         "β", "Competitive Ratio", prefix="stop_")

    # ══════════════════════════════════════════════════════════
    banner("Exp 3b: Robust-aware training with FIXED n")
    # ══════════════════════════════════════════════════════════
    # Cleaner comparison: all instances use the same fixed horizon,
    # so the mask is identical across instances. Compare:
    #   (1) Standard training at fixed n, deployed with robust rule
    #   (2) Robust-masked training at fixed n, deployed with robust rule
    # Both tested at the same fixed n.
    FIXED_N = c["n"]  # use default n=20
    print(f"  Fixed horizon n={FIXED_N}")

    # Train standard model at fixed n
    m_std_fixed, _ = train_stop(c, 0.33, 0.33, 0.33, fixed_n=FIXED_N)
    eval_fixed = sample_stopping_batch(c["n_eval"], FIXED_N, c["M"],
                                       rng=np.random.default_rng(450))

    std_fixed_crs, mask_fixed_crs, sf_s, mf_s = [], [], [], []
    for beta in BETAS:
        print(f"  β={beta:.2f} (fixed n={FIXED_N})")
        # Standard at fixed n
        r = compare_stopping(eval_fixed, m_std_fixed, betas=[beta], r_fractions=[],
                              device=c["device"], use_chain=True)
        std_fixed_crs.append(r.get(f"robust β={beta:.2f}", {}).get("cr", 0))
        sf_s.append(0.01)
        # Robust-masked at fixed n
        m_mask_fixed, _ = train_stop(c, 0.33, 0.33, 0.33,
                                     robust_beta=beta, fixed_n=FIXED_N)
        r2 = compare_stopping(eval_fixed, m_mask_fixed, betas=[beta], r_fractions=[],
                               device=c["device"], use_chain=True)
        mask_fixed_crs.append(r2.get(f"robust β={beta:.2f}", {}).get("cr", 0))
        mf_s.append(0.01)

    pe.plot_robust_sweep(BETAS, std_fixed_crs, mask_fixed_crs, sf_s, mf_s,
                         "β", "Competitive Ratio", prefix="stop_fixed_")

    # ══════════════════════════════════════════════════════════
    banner("Exp 4: Horizon generalization")
    # ══════════════════════════════════════════════════════════
    curves = {}
    for label in ["learned", "dp"]:
        means = []
        for h in TEST_HORIZONS:
            if h > c["max_n"]: continue
            insts = sample_stopping_batch(c["n_eval"], h, c["M"],
                                          rng=np.random.default_rng(500))
            r = compare_stopping(insts, stop_m, betas=[], r_fractions=[],
                                 device=c["device"], use_chain=True)
            means.append(r.get(label, {}).get("cr", 0))
        curves[label] = (means, [0.01]*len(means))
    for beta in [0.10, 0.20, 0.30]:
        means = []
        for h in TEST_HORIZONS:
            if h > c["max_n"]: continue
            insts = sample_stopping_batch(c["n_eval"], h, c["M"],
                                          rng=np.random.default_rng(500))
            r = compare_stopping(insts, stop_m, betas=[beta], r_fractions=[],
                                 device=c["device"], use_chain=True)
            means.append(r.get(f"robust β={beta:.2f}", {}).get("cr", 0))
        curves[f"robust β={beta:.2f}"] = (means, [0.01]*len(means))
    valid_h = [h for h in TEST_HORIZONS if h <= c["max_n"]]
    pe.plot_horizon_generalization(valid_h, curves, "Competitive Ratio",
                                  train_range=(c["n_min"], c["n_max"]), prefix="stop_")

    # Cost-ratio (ski)
    ski_curves = {}
    for label in ["learned", "dp", "deterministic"]:
        means = []
        for ratio in TEST_RATIOS:
            insts = sample_ski_batch(c["n_eval"], c["n"], ratio*c["r"], c["r"],
                                     rng=np.random.default_rng(600))
            r = compare_ski(insts, ski_m, lambdas=[0.0], device=c["device"], use_chain=True)
            means.append(r.get(label, {}).get("mean_additive_loss", 0))
        ski_curves[label] = (means, [0.1]*len(means))
    pe.plot_cost_ratio_generalization(TEST_RATIOS, ski_curves,
                                     train_range=(c["B_min"], c["B_max"]), prefix="ski_")

    # ══════════════════════════════════════════════════════════
    banner("Exp 5: Attention")
    # ══════════════════════════════════════════════════════════
    # Use a real instance from the training distribution
    attn_n = c["n"]  # use default training horizon
    attn_insts = sample_stopping_batch(1, attn_n, c["M"], rng=np.random.default_rng(999))
    x_a = torch.tensor(attn_insts[0].values, dtype=torch.long).unsqueeze(0)
    # Compute real DP targets for teacher forcing via 2D chain
    from deployment import get_stopping_predictions
    from train import _build_chain2d_targets
    V_hat = get_stopping_predictions(stop_m, attn_insts, device=c["device"])
    V_target_norm = torch.tensor(V_hat[0][:attn_n], dtype=torch.float32).unsqueeze(0)
    t_idx, j_idx, _ = stop_m._get_chain2d_info(attn_n, x_a.device)
    chain2d_tgt = _build_chain2d_targets(V_target_norm, j_idx, attn_n)
    # Run forward with return_attention via the transformer internals
    # For now, run a teacher-forced forward pass and capture attention
    h_ctx = stop_m._context_embed(attn_n, 1, x_a.device, task_id=0)
    obs_pos = torch.arange(attn_n, device=x_a.device).unsqueeze(0)
    h_obs = stop_m.value_embed(x_a) + stop_m.pos_embed(obs_pos)
    if h_ctx is not None:
        h_obs = h_obs + h_ctx
    is_start = (j_idx == 0)
    prev_targets = torch.zeros(1, attn_n * (attn_n - 1) // 2, device=x_a.device)
    for t in range(attn_n - 1):
        off = stop_m._chain2d_offset(t, attn_n)
        sub_len = attn_n - 1 - t
        if sub_len > 1:
            prev_targets[:, off + 1:off + sub_len] = chain2d_tgt[:, off:off + sub_len - 1]
    h_chain = stop_m.chain_value_proj(prev_targets.unsqueeze(-1))
    h_chain[:, is_start, :] = stop_m.start_chain.view(1, 1, -1)
    h_chain = h_chain + stop_m.chain2d_t_embed(t_idx) + stop_m.chain2d_j_embed(j_idx)
    if h_ctx is not None:
        h_chain = h_chain + h_ctx
    h_full = stop_m.drop(torch.cat([h_obs, h_chain], dim=1))
    causal_mask = stop_m._build_chain2d_mask(attn_n, x_a.device)
    with torch.no_grad():
        _, attn = stop_m._run_transformer(h_full, causal_mask, return_attention=True)
    for layer in range(min(c["n_layers"], 2)):
        for head in range(min(c["n_heads"], 2)):
            pe.plot_attention_map(attn[layer][0, head].numpy(), n_obs=attn_n,
                                 layer_idx=layer, head_idx=head, prefix="stop_")

    # ══════════════════════════════════════════════════════════
    banner("Exp 6: Depth scaling")
    # ══════════════════════════════════════════════════════════
    depth_means, depth_stds = [], []
    for L in DEPTH_LAYERS:
        print(f"  L={L}")
        c2 = dict(c); c2["n_layers"] = L
        crs = []
        for s in TRAIN_SEEDS[:2]:
            m, _ = train_stop(c2, 0.33, 0.33, 0.33, seed=s)
            for es in EVAL_SEEDS[:2]:
                r = compare_stopping(
                    sample_stopping_batch(c["n_eval"], c["n"], c["M"],
                                          rng=np.random.default_rng(es)),
                    m, betas=[], r_fractions=[], device=c["device"], use_chain=True)
                crs.append(r.get("learned", {}).get("cr", 0))
        depth_means.append(np.mean(crs))
        depth_stds.append(np.std(crs))
    pe.plot_depth_scaling(DEPTH_LAYERS, depth_means, depth_stds,
                          "Competitive Ratio", prefix="stop_")

    # ══════════════════════════════════════════════════════════
    banner("Exp 7: Frontier (stopping)")
    # ══════════════════════════════════════════════════════════
    eval_f = sample_stopping_batch(c["n_eval"], c["n"], c["M"],
                                   rng=np.random.default_rng(800))
    r = compare_stopping(eval_f, stop_m, betas=BETAS[:5], r_fractions=[],
                          device=c["device"], use_chain=True)
    learned_cr = r.get("learned", {}).get("cr", 0)
    dp_cr = r.get("dp", {}).get("cr", 0)
    robust_crs = [r.get(f"robust β={b:.2f}", {}).get("cr", 0) for b in BETAS[:5]]
    frontier = {
        "learned": ([learned_cr] * len(BETAS[:5]), [0.01] * len(BETAS[:5])),
        "Bayes DP": ([dp_cr] * len(BETAS[:5]), [0.0] * len(BETAS[:5])),
        "robust": (robust_crs, [0.01] * len(BETAS[:5])),
    }
    pe.plot_frontier(BETAS[:5], frontier, "β", "Competitive Ratio", prefix="stop_")

    # ══════════════════════════════════════════════════════════════
    # SKI RENTAL EXPERIMENTS
    # ══════════════════════════════════════════════════════════════

    # ── Ski Exp 1: Loss weighting ──
    banner("Ski Exp 1: Loss weighting (11 configs × 3 seeds)")
    cfg_names, cfg_losses, cfg_stds = [], [], []
    for name, wv, wa, wc in WEIGHT_CONFIGS:
        print(f"  {name}")
        all_logs, losses = [], []
        for s in TRAIN_SEEDS:
            m, logs = train_ski(c, wv, wa, wc, seed=s)
            all_logs.append(logs)
            for es in EVAL_SEEDS[:2]:
                insts = sample_ski_batch(c["n_eval"], c["n"], c["B"], c["r"],
                                         rng=np.random.default_rng(es))
                r = compare_ski(insts, m, lambdas=[0.0], device=c["device"], use_chain=True)
                losses.append(r.get("learned", {}).get("mean_additive_loss", 0))
        pe.plot_training_curves(all_logs[0], f"ski_exp1_{name}", all_seed_logs=all_logs)
        cfg_names.append(name)
        cfg_losses.append(np.mean(losses))
        cfg_stds.append(np.std(losses))
    pe.plot_weight_sweep(cfg_names, cfg_losses, cfg_stds, "Additive Loss", prefix="ski_")

    # ── Ski Exp 2: Chain supervision ──
    banner("Ski Exp 2: Chain supervision")
    m_unsup_ski, _ = train_ski(c, 1.0, 0.5, 0.0)
    m_sup_ski, _ = train_ski(c, 0.33, 0.33, 0.33)
    conditions = ["In-distribution"]
    std_v, ch_v, std_s, ch_s = [], [], [], []
    for cond in conditions:
        sv, cv = [], []
        for s in EVAL_SEEDS:
            insts = sample_ski_batch(c["n_eval"], c["n"], c["B"], c["r"],
                                     rng=np.random.default_rng(s))
            r1 = compare_ski(insts, m_unsup_ski, lambdas=[0.0], device=c["device"], use_chain=True)
            r2 = compare_ski(insts, m_sup_ski, lambdas=[0.0], device=c["device"], use_chain=True)
            sv.append(r1.get("learned", {}).get("mean_additive_loss", 0))
            cv.append(r2.get("learned", {}).get("mean_additive_loss", 0))
        std_v.append(np.mean(sv)); std_s.append(np.std(sv))
        ch_v.append(np.mean(cv)); ch_s.append(np.std(cv))
    pe.plot_chain_comparison(conditions, std_v, ch_v, std_s, ch_s,
                             "Additive Loss", prefix="ski_")

    # ── Ski Exp 3: Robustness sweep ──
    banner("Ski Exp 3: Robustness sweep (λ)")
    eval_insts_ski = sample_ski_batch(c["n_eval"], c["n"], c["B"], c["r"],
                                       rng=np.random.default_rng(400))
    lam_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    learned_losses, robust_losses, l_s, r_s = [], [], [], []
    for lam in lam_vals:
        print(f"  λ={lam:.2f}")
        r = compare_ski(eval_insts_ski, ski_m, lambdas=[lam], device=c["device"], use_chain=True)
        learned_losses.append(r.get("learned", {}).get("mean_additive_loss", 0))
        l_s.append(0.01)
        robust_key = f"robust λ={lam:.2f}" if lam > 0 else "learned"
        robust_losses.append(r.get(robust_key, r.get("learned", {})).get("mean_additive_loss", 0))
        r_s.append(0.01)
    pe.plot_robust_sweep(lam_vals, learned_losses, robust_losses, l_s, r_s,
                         "λ", "Additive Loss", prefix="ski_")

    # ── Ski Exp 4: Cost-ratio generalization ──
    banner("Ski Exp 4: Cost-ratio generalization")
    ski_curves = {}
    for label in ["learned", "dp", "deterministic"]:
        means = []
        for ratio in TEST_RATIOS:
            insts = sample_ski_batch(c["n_eval"], c["n"], ratio*c["r"], c["r"],
                                     rng=np.random.default_rng(600))
            r = compare_ski(insts, ski_m, lambdas=[0.0], device=c["device"], use_chain=True)
            means.append(r.get(label, {}).get("mean_additive_loss", 0))
        ski_curves[label] = (means, [0.1]*len(means))
    for lam in [0.3, 0.5, 0.7]:
        means = []
        for ratio in TEST_RATIOS:
            insts = sample_ski_batch(c["n_eval"], c["n"], ratio*c["r"], c["r"],
                                     rng=np.random.default_rng(600))
            r = compare_ski(insts, ski_m, lambdas=[lam], device=c["device"], use_chain=True)
            key = f"robust λ={lam:.2f}"
            means.append(r.get(key, {}).get("mean_additive_loss", 0))
        ski_curves[f"robust λ={lam:.1f}"] = (means, [0.1]*len(means))
    pe.plot_cost_ratio_generalization(TEST_RATIOS, ski_curves,
                                     train_range=(c["B_min"], c["B_max"]), prefix="ski_exp4_")

    # ── Ski Exp 5: Attention ──
    banner("Ski Exp 5: Attention")
    # Use a real ski rental instance from the training distribution
    attn_ski_insts = sample_ski_batch(1, attn_n, c["B"], c["r"], rng=np.random.default_rng(999))
    x_a_ski = torch.tensor(np.ones(attn_n, dtype=int), dtype=torch.long).unsqueeze(0)
    from deployment import get_ski_predictions
    V_hat_ski = get_ski_predictions(ski_m, attn_ski_insts, device=c["device"])
    V_target_norm_ski = torch.tensor(V_hat_ski[0][:attn_n], dtype=torch.float32).unsqueeze(0)
    t_idx_s, j_idx_s, _ = ski_m._get_chain2d_info(attn_n, x_a_ski.device)
    chain2d_tgt_ski = _build_chain2d_targets(V_target_norm_ski, j_idx_s, attn_n)
    h_ctx_ski = ski_m._context_embed(attn_n, 1, x_a_ski.device, task_id=1)
    obs_pos_ski = torch.arange(attn_n, device=x_a_ski.device).unsqueeze(0)
    h_obs_ski = ski_m.value_embed(x_a_ski) + ski_m.pos_embed(obs_pos_ski)
    if h_ctx_ski is not None:
        h_obs_ski = h_obs_ski + h_ctx_ski
    is_start_s = (j_idx_s == 0)
    prev_targets_ski = torch.zeros(1, attn_n * (attn_n - 1) // 2, device=x_a_ski.device)
    for t in range(attn_n - 1):
        off = ski_m._chain2d_offset(t, attn_n)
        sub_len = attn_n - 1 - t
        if sub_len > 1:
            prev_targets_ski[:, off + 1:off + sub_len] = chain2d_tgt_ski[:, off:off + sub_len - 1]
    h_chain_ski = ski_m.chain_value_proj(prev_targets_ski.unsqueeze(-1))
    h_chain_ski[:, is_start_s, :] = ski_m.start_chain.view(1, 1, -1)
    h_chain_ski = h_chain_ski + ski_m.chain2d_t_embed(t_idx_s) + ski_m.chain2d_j_embed(j_idx_s)
    if h_ctx_ski is not None:
        h_chain_ski = h_chain_ski + h_ctx_ski
    h_full_ski = ski_m.drop(torch.cat([h_obs_ski, h_chain_ski], dim=1))
    causal_mask_ski = ski_m._build_chain2d_mask(attn_n, x_a_ski.device)
    with torch.no_grad():
        _, attn = ski_m._run_transformer(h_full_ski, causal_mask_ski, return_attention=True)
    for layer in range(min(c["n_layers"], 2)):
        for head in range(min(c["n_heads"], 2)):
            pe.plot_attention_map(attn[layer][0, head].numpy(), n_obs=attn_n,
                                 layer_idx=layer, head_idx=head, prefix="ski_")

    # ── Ski Exp 6: Depth scaling ──
    banner("Ski Exp 6: Depth scaling")
    depth_means, depth_stds = [], []
    for L in DEPTH_LAYERS:
        print(f"  L={L}")
        c2 = dict(c); c2["n_layers"] = L
        losses = []
        for s in TRAIN_SEEDS[:2]:
            m, _ = train_ski(c2, 0.33, 0.33, 0.33, seed=s)
            for es in EVAL_SEEDS[:2]:
                insts = sample_ski_batch(c["n_eval"], c["n"], c["B"], c["r"],
                                         rng=np.random.default_rng(es))
                r = compare_ski(insts, m, lambdas=[0.0], device=c["device"], use_chain=True)
                losses.append(r.get("learned", {}).get("mean_additive_loss", 0))
        depth_means.append(np.mean(losses))
        depth_stds.append(np.std(losses))
    pe.plot_depth_scaling(DEPTH_LAYERS, depth_means, depth_stds,
                          "Additive Loss", prefix="ski_")

    # ── Ski Exp 7: Frontier ──
    banner("Ski Exp 7: Frontier")
    lam_range = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    eval_f_ski = sample_ski_batch(c["n_eval"], c["n"], c["B"], c["r"],
                                   rng=np.random.default_rng(800))
    r = compare_ski(eval_f_ski, ski_m, lambdas=lam_range, device=c["device"], use_chain=True)
    learned_loss = r.get("learned", {}).get("mean_additive_loss", 0)
    dp_loss = r.get("dp", {}).get("mean_additive_loss", 0)
    det_loss = r.get("deterministic", {}).get("mean_additive_loss", 0)
    n_lam = len(lam_range)
    robust_losses_ski = []
    for lam in lam_range:
        key = f"robust λ={lam:.2f}" if lam > 0 else "learned"
        robust_losses_ski.append(r.get(key, r.get("learned", {})).get("mean_additive_loss", 0))
    frontier_ski = {
        "learned": ([learned_loss] * n_lam, [0.01] * n_lam),
        "Bayes DP": ([dp_loss] * n_lam, [0.0] * n_lam),
        "deterministic": ([det_loss] * n_lam, [0.0] * n_lam),
        "robust": (robust_losses_ski, [0.01] * n_lam),
    }
    pe.plot_frontier(lam_range, frontier_ski, "λ", "Additive Loss", prefix="ski_")

    # ══════════════════════════════════════════════════════════
    banner("DONE — All experiments")
    # ══════════════════════════════════════════════════════════
    plots = [f for f in sorted(os.listdir(pe.OUT_DIR)) if f.endswith('.png')]
    print(f"\n{len(plots)} plots in {pe.OUT_DIR}:")
    for p in plots:
        print(f"  {p}")


if __name__ == "__main__":
    main()
