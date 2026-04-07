# Neural Algorithmic Decision Making

A causal transformer with 2D chain-of-thought for online decision problems: **optimal stopping** and **ski rental**.

The model learns continuation values via dynamic programming supervision and generalizes across distribution families and horizon lengths.

## Structure

- `code/model.py` — Decoder-style transformer with 2D chain-of-thought
- `code/train.py` — Training loop (value + chain + action losses)
- `code/dataset.py` — Data generation with exact DP labels
- `code/dp.py` — Exact dynamic programming solvers
- `code/sampling.py` — Distribution family sampling
- `code/deployment.py` — Evaluation policies and metrics
- `code/sweep_weights.py` — Hyperparameter sweeps
