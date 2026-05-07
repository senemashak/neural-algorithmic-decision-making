# Pre-Step-5 data-separation check

Generated: 2026-05-06T15:39:36

Hashing: SHA-256 of each float64 sequence's byte representation.
Per Step-5 prelude, training streams are sampled probabilistically at 100,000 sequences per (distribution, supervision) — the same RNG state the trainer started from, so this is the start of each stream rather than uniformly across; with continuous iid sampling the collision probability is uniform across the run, and finding any here would already be diagnostic.

## Check A — val ∩ training stream

| training distribution | run | seed | val set size | training sample size | overlap | wall (s) | result |
|---|---|---:|---:|---:|---:|---:|---|
| D_1 | D_1_cv | 1000 | 10,000 | 100,000 | 0 | 0.64 | PASS |
| D_1 | D_1_act | 1001 | 10,000 | 100,000 | 0 | 0.61 | PASS |
| D_2 | D_2_cv | 1002 | 10,000 | 100,000 | 0 | 0.64 | PASS |
| D_2 | D_2_act | 1003 | 10,000 | 100,000 | 0 | 0.61 | PASS |
| D_3 | D_3_cv | 1004 | 10,000 | 100,000 | 0 | 0.61 | PASS |
| D_3 | D_3_act | 1005 | 10,000 | 100,000 | 0 | 0.61 | PASS |
| D_disc | D_disc_cv | 1006 | 10,000 | 100,000 | 0 | 0.64 | PASS |
| D_disc | D_disc_act | 1007 | 10,000 | 100,000 | 0 | 0.60 | PASS |
| D_logu | D_logu_cv | 1008 | 10,000 | 100,000 | 0 | 0.60 | PASS |
| D_logu | D_logu_act | 1009 | 10,000 | 100,000 | 0 | 0.62 | PASS |

Check A overall: **PASS** (expected: zero overlap on every row).

## Check B(i) — test ∩ val

| test regime | val distribution | test size | val size | overlap | result |
|---|---|---:|---:|---:|---|
| D_1 | D_1 | 10,000 | 10,000 | 0 | PASS |
| D_1 | D_2 | 10,000 | 10,000 | 0 | PASS |
| D_1 | D_3 | 10,000 | 10,000 | 0 | PASS |
| D_1 | D_disc | 10,000 | 10,000 | 0 | PASS |
| D_1 | D_logu | 10,000 | 10,000 | 0 | PASS |
| D_2 | D_1 | 10,000 | 10,000 | 0 | PASS |
| D_2 | D_2 | 10,000 | 10,000 | 0 | PASS |
| D_2 | D_3 | 10,000 | 10,000 | 0 | PASS |
| D_2 | D_disc | 10,000 | 10,000 | 0 | PASS |
| D_2 | D_logu | 10,000 | 10,000 | 0 | PASS |
| D_3 | D_1 | 10,000 | 10,000 | 0 | PASS |
| D_3 | D_2 | 10,000 | 10,000 | 0 | PASS |
| D_3 | D_3 | 10,000 | 10,000 | 0 | PASS |
| D_3 | D_disc | 10,000 | 10,000 | 0 | PASS |
| D_3 | D_logu | 10,000 | 10,000 | 0 | PASS |

Check B(i) overall: **PASS**.

## Check B(ii) — test ∩ training stream

| test regime | training run | seed | test size | training sample size | overlap | wall (s) | result |
|---|---|---:|---:|---:|---:|---:|---|
| D_1 | D_1_cv | 1000 | 10,000 | 100,000 | 0 | 0.59 | PASS |
| D_1 | D_1_act | 1001 | 10,000 | 100,000 | 0 | 0.64 | PASS |
| D_1 | D_2_cv | 1002 | 10,000 | 100,000 | 0 | 0.60 | PASS |
| D_1 | D_2_act | 1003 | 10,000 | 100,000 | 0 | 0.60 | PASS |
| D_1 | D_3_cv | 1004 | 10,000 | 100,000 | 0 | 0.61 | PASS |
| D_1 | D_3_act | 1005 | 10,000 | 100,000 | 0 | 0.64 | PASS |
| D_1 | D_disc_cv | 1006 | 10,000 | 100,000 | 0 | 0.61 | PASS |
| D_1 | D_disc_act | 1007 | 10,000 | 100,000 | 0 | 0.62 | PASS |
| D_1 | D_logu_cv | 1008 | 10,000 | 100,000 | 0 | 0.61 | PASS |
| D_1 | D_logu_act | 1009 | 10,000 | 100,000 | 0 | 0.67 | PASS |
| D_2 | D_1_cv | 1000 | 10,000 | 100,000 | 0 | 0.61 | PASS |
| D_2 | D_1_act | 1001 | 10,000 | 100,000 | 0 | 0.60 | PASS |
| D_2 | D_2_cv | 1002 | 10,000 | 100,000 | 0 | 0.60 | PASS |
| D_2 | D_2_act | 1003 | 10,000 | 100,000 | 0 | 0.63 | PASS |
| D_2 | D_3_cv | 1004 | 10,000 | 100,000 | 0 | 0.60 | PASS |
| D_2 | D_3_act | 1005 | 10,000 | 100,000 | 0 | 0.59 | PASS |
| D_2 | D_disc_cv | 1006 | 10,000 | 100,000 | 0 | 0.60 | PASS |
| D_2 | D_disc_act | 1007 | 10,000 | 100,000 | 0 | 0.65 | PASS |
| D_2 | D_logu_cv | 1008 | 10,000 | 100,000 | 0 | 0.61 | PASS |
| D_2 | D_logu_act | 1009 | 10,000 | 100,000 | 0 | 0.60 | PASS |
| D_3 | D_1_cv | 1000 | 10,000 | 100,000 | 0 | 0.60 | PASS |
| D_3 | D_1_act | 1001 | 10,000 | 100,000 | 0 | 0.63 | PASS |
| D_3 | D_2_cv | 1002 | 10,000 | 100,000 | 0 | 0.60 | PASS |
| D_3 | D_2_act | 1003 | 10,000 | 100,000 | 0 | 0.60 | PASS |
| D_3 | D_3_cv | 1004 | 10,000 | 100,000 | 0 | 0.60 | PASS |
| D_3 | D_3_act | 1005 | 10,000 | 100,000 | 0 | 0.63 | PASS |
| D_3 | D_disc_cv | 1006 | 10,000 | 100,000 | 0 | 0.61 | PASS |
| D_3 | D_disc_act | 1007 | 10,000 | 100,000 | 0 | 0.61 | PASS |
| D_3 | D_logu_cv | 1008 | 10,000 | 100,000 | 0 | 0.62 | PASS |
| D_3 | D_logu_act | 1009 | 10,000 | 100,000 | 0 | 0.65 | PASS |

Check B(ii) overall: **PASS**.

## Check C — eval cache routing

The Step-5 payoff matrix evaluates every model on the three test regimes D_1, D_2, D_3 (each: 10⁴ sequences from σ ∈ {1, 10, 100}, seed 43-derived). The per-σ payoff breakdown for random-variance models uses the corresponding training-distribution val cache (seed 42-derived) as the σ-binned test set — those caches already exist from Step 3.

### Payoff-matrix test caches (used for the 5 × 3 cell grid)

| test regime | path | exists | size (bytes) |
|---|---|---|---:|
| D_1 | `/home/senemi/neural-algorithmic-decision-making/bayesian-stopping/v3/data/cache/D_1_test.npz` | yes | 19,717,399 |
| D_2 | `/home/senemi/neural-algorithmic-decision-making/bayesian-stopping/v3/data/cache/D_2_test.npz` | yes | 19,785,754 |
| D_3 | `/home/senemi/neural-algorithmic-decision-making/bayesian-stopping/v3/data/cache/D_3_test.npz` | yes | 19,696,846 |

### Per-σ payoff caches (Step-5 figure 4, random-variance only)

| training distribution | path (val cache) | exists |
|---|---|---|
| D_disc | `/home/senemi/neural-algorithmic-decision-making/bayesian-stopping/v3/data/cache/D_disc_val.npz` | yes |
| D_logu | `/home/senemi/neural-algorithmic-decision-making/bayesian-stopping/v3/data/cache/D_logu_val.npz` | yes |

### Val caches (background, used during training)

| training distribution | path | exists | size (bytes) |
|---|---|---|---:|
| D_1 | `/home/senemi/neural-algorithmic-decision-making/bayesian-stopping/v3/data/cache/D_1_val.npz` | yes | 19,715,270 |
| D_2 | `/home/senemi/neural-algorithmic-decision-making/bayesian-stopping/v3/data/cache/D_2_val.npz` | yes | 19,783,745 |
| D_3 | `/home/senemi/neural-algorithmic-decision-making/bayesian-stopping/v3/data/cache/D_3_val.npz` | yes | 19,696,338 |
| D_disc | `/home/senemi/neural-algorithmic-decision-making/bayesian-stopping/v3/data/cache/D_disc_val.npz` | yes | 19,765,928 |
| D_logu | `/home/senemi/neural-algorithmic-decision-making/bayesian-stopping/v3/data/cache/D_logu_val.npz` | yes | 19,844,990 |

## Overall verdict: **PASS**

## Check B(iii) — fresh per-σ test caches (seed 44, added 2026-05-06)

Two new caches generated for the Step-5 figure-4 per-σ payoff breakdown,
seeded independently from val (seed 42-derived) and test (seed 43-derived)
so the figure-4 evaluation isn't run on data the trainer's checkpoint
selection saw:

  v3/data/cache/D_disc_persigma_test.npz  (10⁴ sequences, σ ∈ {1, 10, 100})
  v3/data/cache/D_logu_persigma_test.npz  (10⁴ sequences, σ ∈ [1, 100] log-uniform)

Each row aggregates 5 (val) + 10 (training-stream) overlap checks per
new cache, plus the cross-cache check; all underlying values are 0.

| persigma test cache         | size  | overlap with any val cache | overlap with any training stream | overlap with sibling cache | result |
|-----------------------------|------:|---------------------------:|---------------------------------:|---------------------------:|--------|
| D_disc_persigma_test (seed 44) | 10000 | 0 (max across 5 val caches) | 0 (max across 10 training streams) | 0 (vs D_logu_persigma_test) | PASS |
| D_logu_persigma_test (seed 44) | 10000 | 0 (max across 5 val caches) | 0 (max across 10 training streams) | 0 (vs D_disc_persigma_test) | PASS |

Check B(iii) overall: **PASS**.

## Updated overall verdict: **PASS** (Checks A + B(i) + B(ii) + B(iii) + C all pass; eval driver routing is unambiguous).
