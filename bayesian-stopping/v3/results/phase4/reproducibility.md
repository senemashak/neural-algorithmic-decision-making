# Reproducibility — Stage B trained models

How to reload any of the 10 trained models for downstream analysis.

## Reload contract

```python
from train.io import load_checkpoint

model, head_name, metadata = load_checkpoint('D_disc_cv', which='best')
# model: GPTStopper instance, both heads loaded, on CPU, in eval() mode
# head_name: 'cv' or 'act' — query only this head
# metadata: dict with config, environment, val_loss, step, etc.

model = model.cuda()      # move to GPU
out = model(X)            # X: (B, n) torch tensor of raw observations
                          # out: {'cv': (B, n) Ĉ_t, 'act': (B, n) logits}
predictions = out[head_name]
```

`which` accepts `'best'`, `'final'`, or `'step_50k'`/`'step_100k'`/etc.
For Stage B's 200k-step static cv runs, periodic checkpoints at
50k/100k/150k/200k are available. For 300k random cv runs, also at 250k
and 300k. Act runs save periodics every 50k up through their step count
(100k for static, 150k for random).

## Canonical fields

The `trained_head` field in each `.pt` is authoritative — filenames are
hints. Step 5+ should never assume the head from the filename.

`metadata['config']` contains the full RunConfig dump:
distribution, supervision, step_count, val_every, periodic_every, lr,
batch_size, warmup_frac, seed, n, d_emb, n_layers, n_heads.

`metadata['environment']` contains: torch_version, numpy_version,
cuda_version, gpu_name, hostname, python_version, training_start,
training_end, total_wall_s.

## Per-σ-group val loss fields (random-variance runs)

Each `val` event in `log.jsonl` carries:

  - `val_loss` — aggregate (per-sequence loss averaged over the val set)
  - `val_loss_sigma_1`, `val_loss_sigma_10`, `val_loss_sigma_100` — for
    D_disc; mean per-sequence loss within each σ group
  - `val_loss_sigma_low`, `val_loss_sigma_mid`, `val_loss_sigma_high` —
    for D_logu; bins are log_10 σ ∈ [0, 2/3), [2/3, 4/3), [4/3, 2]

For static-σ runs, `val_loss_sigma_{σ}` duplicates the aggregate.

These are also exposed in `curves.npz` keyed by the same names.

## Loss form

cv loss is **per-sequence MSE divided by σ_i** (NOT σ_i²) — see
"Spec corrections" entry of 2026-05-06 in v3/README.md. Validation
val_loss is in the same units. For random-variance runs, val_loss is
dominated by σ=100 sequences (since MSE scales as σ² and we divide by
σ — leaving a σ factor in the absolute scale).

act loss is per-sequence mean BCE-with-logits. Naturally regime-invariant.

## Per-run files

Each `v3/checkpoints/<run_name>/` contains:
  - `best.pt`, `final.pt`, `step_*k.pt`
  - `config.json`  (RunConfig dump)
  - `metadata.json`  (env + timing)

Each `v3/results/phase4/<run_name>/` contains:
  - `log.jsonl` (one event per line; kinds: start, train, val, checkpoint, end)
  - `curves.npz` (dense step / loss arrays)
