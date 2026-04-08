# Neural Algorithmic Decision Making

A causal transformer with 2D chain-of-thought that learns to solve online decision problems by explicitly performing backward induction in a scratchpad. Trained on exact dynamic programming labels, the model generalizes across distribution families and horizon lengths.

## Key idea

Online decision problems like optimal stopping require computing continuation values via backward induction — a process that runs *backward* from the horizon but must be applied *online* as observations arrive. A standard transformer processes tokens left-to-right, so it cannot execute backward induction in a single pass. Our solution: append a chain-of-thought scratchpad where the model explicitly carries out the DP computation, one step per token.

For each decision step t, a sub-chain computes V(t, n-2) → V(t, n-3) → ... → V(t, t), producing V(t) = V(t,t) as the decision output. The attention mask enforces the online constraint: sub-chain t sees only observations x_0,...,x_t (the Bayesian posterior), with future observations blocked.

## Architecture

- **Model**: decoder-only transformer, 2 layers, 2 heads, ~20M parameters
- **Dimensions**: d_model = 1002, d_ff = 2259 (satisfies Propositions 1-2 for exact/approximate DP realization)
- **Why 2 layers**: the stopping Bellman backup requires two sequential operations per chain step (FFN computes ReLU terms, attention computes PMF-weighted sum)
- **Why 2 heads**: one for chain recurrence (reading previous DP value), one for observation reading (inferring the distribution)
- **Single output**: V(t) per decision step, no separate action head. Stop if x_t >= V(t)*M (stopping) or buy if V(t) >= 1 (ski rental)

## Training data

Each instance samples a random distribution family (10 families per problem), random hyperparameters, random horizon n ~ U[20,50], and draws observations. For 6 of 10 stopping families, the PMF is support-subsampled to create sparse, irregular distributions. Labels are computed by exact backward induction on the known PMF. The model never sees the PMF — it must infer it from observations.

## Training loss

Three terms: L_value (MSE on V(t) vs DP target), L_action (soft cross-entropy on implied decisions), L_chain (MSE on all intermediate chain values V(t,k)). The sweep experiment tests 11 weight configurations.

## Results (Experiment 1: Loss Weight Sweep)

11 configurations, 3 training seeds each, evaluated with 5 eval seeds at variable horizons n ~ U[20,50] on all 11 distribution families (including OOD hard_instance).

| Config | Normalized weights | CR |
|---|---|---|
| **value+action** | **(0.67, 0.33, 0)** | **0.846** |
| emph action | (0.25, 0.5, 0.25) | 0.755 |
| value only | (1, 0, 0) | 0.749 |
| action+chain | (0, 0.5, 0.5) | 0.715 |
| chain only | (0, 0, 1) | 0.711 |
| emph value | (0.5, 0.25, 0.25) | 0.705 |
| equal third | (0.33, 0.33, 0.33) | 0.699 |
| emph chain | (0.25, 0.25, 0.5) | 0.648 |
| value+chain | (0.5, 0, 0.5) | 0.638 |
| all 1 0.5 1 | (0.4, 0.2, 0.4) | 0.610 |
| action only | (0, 1, 0) | 0.452 |

DP oracle (knows the distribution): 0.940

**Key findings:**
- L_value + L_action without chain supervision is the clear winner (CR 0.846)
- The action loss provides a strong complementary signal to the value loss
- Chain supervision alone works decently (0.711) but hurts when combined with L_value
- Action-only training fails completely (0.452) — can't learn values from action signal alone

## Code structure

```
code/
  sampling.py          — 10 distribution families per problem, support subsampling
  dp.py                — exact backward induction for labels
  dataset.py           — PyTorch datasets with variable horizons, DP labels
  model.py             — 2D chain-of-thought transformer (2 layers, 2 heads, ~20M params)
  train.py             — training loop: 3 losses, early stopping, progress bar, robust masking
  deployment.py        — inference, policy evaluation (learned, DP, robust, Dynkin, deterministic)
  sweep_weights.py     — loss weight sweep: 11 configs x 3 seeds x 5 eval seeds
  sweep_weights_plots.py — plotting from results.json (fully decoupled from training)
  later/               — sweep_robust, main pipeline, old plotting (not yet updated)

results/               — saved checkpoints, attention maps, results.json
plots/                 — generated figures
experiment.text        — detailed experiment writeup (LaTeX)
```

## Running

```bash
# Full sweep (11 configs, ~12h on 1 GPU)
CUDA_VISIBLE_DEVICES=0 python sweep_weights.py --device cuda --only stopping --epochs 30

# Generate plots from saved results (no GPU needed)
python sweep_weights_plots.py results/sweep/sweep_<timestamp>/results.json

# Re-evaluate saved models at different horizons
# (models saved at results/sweep/sweep_<timestamp>/models/)
```

## Compute

- GPU: NVIDIA RTX A6000 (48GB)
- ~60s per epoch, ~1.7h per config (3 seeds x 30 epochs)
- Full sweep: ~12h on 1 GPU, or ~6h split across 2 GPUs with `--only`
