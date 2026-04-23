# Neural Algorithmic Decision Making

A causal transformer with 2D chain-of-thought that learns to solve online decision problems by performing backward induction in a scratchpad. Trained on exact dynamic programming labels, evaluated on optimal stopping and ski rental.

## Key idea

Online decision problems require computing continuation values via backward induction — a process that runs *backward* but must be applied *online*. We append a chain-of-thought scratchpad where the model explicitly carries out the DP computation, one step per token. A 2-layer, 2-head transformer (~20M params) with architecture dimensions satisfying the theoretical construction requirements (Propositions 1-2).

## Experiments

1. **Loss weight sweep** — which combination of value, action, and chain losses works best
2. **Robust-aware training** — does masking training to positions the robust wrapper uses help
3. **Attention analysis** — mechanistic interpretation of learned attention patterns
4. **Horizon generalization** — does the model generalize to unseen sequence lengths

## Structure

```
code/
  core/                — model, training, evaluation infrastructure
  experiments/         — experiment scripts (train, eval, plot)
results/               — checkpoints, results.json, attention maps
logs/                  — training and eval logs
plots/
  exp1_loss_weights/   — loss weight sweep results
  exp2_robust/         — robust training results
  exp3_attention/      — attention visualizations
  exp4_horizon/        — horizon generalization
experiment.tex         — detailed experiment writeup
```

## Running

```bash
cd code/

# Experiment 1: loss weight sweep
python -m experiments.sweep_weights --device cuda --only stopping

# Experiment 2: robust training + evaluation
python -m experiments.train_robust_stopping --device cuda
python -m experiments.eval_robust_stopping --device cuda

# Experiment 3: attention visualization
python -m core.attention both --checkpoint model.pt --device cuda

# Experiment 4: horizon generalization
python -m experiments.eval_horizon --device cuda
```
