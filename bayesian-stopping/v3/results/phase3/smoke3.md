# Smoke 3 — model forward + loss + backward sanity

Setting: GPTStopper (n=256, d_emb=128, L=8, M=4 heads), random init. Batch size 64. Device: cuda:0.
Param count: 1,619,714 (target ~1.6M).

## Forward + loss

| batch | finite outputs | cv loss | act loss |
|---|---|---|---|
| D_1   | cv=True, act=True | 7.1313e+01 | 7.2765e-01 |
| D_disc | cv=True, act=True | 4.8822e+01 | 7.2997e-01 |

## Gradient norms (backward pass)

| batch | sigma in batch | cv-grad norm | act-grad norm |
|---|---|---|---|
| D_1   | 1.0                      | 1.5082e+02 | 2.0324e+00 |
| D_disc | sigma_max=100 | 8.5274e+01 | 1.4322e+00 |

Regime-invariance check (cv-grad ratio D_disc / D_1): **0.57**.
Without per-sequence 1/sigma_i^2 normalization, this would be ~sigma_max^2 / sigma_D1^2 = 10000; observed ratio close to unity confirms the loss is regime-invariant.

## Gates

- finite outputs (D_1 + D_disc, cv + act): **PASS**
- cv loss in [1e-3, 1e3] for both batches: **PASS**
- act loss in [0.4, 1.0] for both batches (near log 2 ≈ 0.693): **PASS**
- gradient norms finite, cv-grad ratio < 100: **PASS**

**Overall: PASS**
