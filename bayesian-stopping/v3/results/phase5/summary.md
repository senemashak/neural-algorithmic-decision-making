# Phase 5 — evaluation summary

## Payoff matrix (R / R*) on the seed-43 per-regime test caches

| trained model | D_1 | D_2 | D_3 | D_3_cv comparator (D_3) | per-σ summary |
|---|---:|---:|---:|---:|---|
| D_1_cv | 0.9999 | 0.3182 | 0.3087 | 0.9996 |  |
| D_1_act | 1.0000 | 0.3201 | 0.2695 | 0.9996 |  |
| D_2_cv | 0.1741 | 1.0001 | 0.3616 | 0.9996 |  |
| D_2_act | 0.1202 | 0.9998 | 0.3389 | 0.9996 |  |
| D_3_cv | 0.0157 | 0.0523 | 0.9996 | 0.9996 |  |
| D_3_act | 0.0168 | 0.1236 | 0.9997 | 0.9996 |  |
| D_disc_cv | 0.9993 | 0.9994 | 0.9998 | 0.9996 | σ=1=1.001; σ=10=0.999; σ=100=0.999 |
| D_disc_act | 0.9994 | 0.9988 | 0.9995 | 0.9996 | σ=1=1.000; σ=10=1.000; σ=100=1.000 |
| D_logu_cv | 0.0360 | 0.0907 | 0.8003 | 0.9996 | log₁₀σ∈[0.0,0.4)=-0.044; log₁₀σ∈[0.4,0.8)=0.071; log₁₀σ∈[0.8,1.2)=0.111; log₁₀σ∈[1.2,1.6)=0.417; log₁₀σ∈[1.6,2.0)=0.721 |
| D_logu_act | 0.9743 | 0.9761 | 0.9748 | 0.9996 | log₁₀σ∈[0.0,0.4)=0.991; log₁₀σ∈[0.4,0.8)=0.992; log₁₀σ∈[0.8,1.2)=0.994; log₁₀σ∈[1.2,1.6)=0.988; log₁₀σ∈[1.6,2.0)=0.998 |

## Per-distribution interpretation

### D_1

Single-σ training (σ = {1: 1, 2: 10, 3: 100}[1]). cv recovers oracle in-distribution at 0.9999; off-diagonal cells D_1=0.9999, D_2=0.3182, D_3=0.3087. Same shape on the act head (1.0000 in-distribution; D_1=1.0000, D_2=0.3201, D_3=0.2695). The off-diagonal collapses are the V2 shortcut signature: the model has learned a fixed-scale threshold and pastes it onto the wrong regime.

### D_2

Single-σ training (σ = {1: 1, 2: 10, 3: 100}[2]). cv recovers oracle in-distribution at 1.0001; off-diagonal cells D_1=0.1741, D_2=1.0001, D_3=0.3616. Same shape on the act head (0.9998 in-distribution; D_1=0.1202, D_2=0.9998, D_3=0.3389). The off-diagonal collapses are the V2 shortcut signature: the model has learned a fixed-scale threshold and pastes it onto the wrong regime.

### D_3

Single-σ training (σ = {1: 1, 2: 10, 3: 100}[3]). cv recovers oracle in-distribution at 0.9996; off-diagonal cells D_1=0.0157, D_2=0.0523, D_3=0.9996. Same shape on the act head (0.9997 in-distribution; D_1=0.0168, D_2=0.1236, D_3=0.9997). The off-diagonal collapses are the V2 shortcut signature: the model has learned a fixed-scale threshold and pastes it onto the wrong regime.

### D_disc

Random-σ training. cv: D_1=0.9993, D_2=0.9994, D_3=0.9998. act: D_1=0.9994, D_2=0.9988, D_3=0.9995. Per-σ breakdown (cv on the seed-44 persigma test cache): σ=1 1.001, σ=10 0.999, σ=100 0.999. Per-σ (act): σ=1 1.000, σ=10 1.000, σ=100 1.000.

### D_logu

Random-σ training. cv: D_1=0.0360, D_2=0.0907, D_3=0.8003. act: D_1=0.9743, D_2=0.9761, D_3=0.9748. Per-σ breakdown (cv on the seed-44 persigma test cache): log₁₀σ∈[0.0,0.4) -0.044, log₁₀σ∈[0.4,0.8) 0.071, log₁₀σ∈[0.8,1.2) 0.111, log₁₀σ∈[1.2,1.6) 0.417, log₁₀σ∈[1.6,2.0) 0.721. Per-σ (act): log₁₀σ∈[0.0,0.4) 0.991, log₁₀σ∈[0.4,0.8) 0.992, log₁₀σ∈[0.8,1.2) 0.994, log₁₀σ∈[1.2,1.6) 0.988, log₁₀σ∈[1.6,2.0) 0.998.

## Findings

- **D_disc trains the algorithm in full.** D_disc_cv hits R/R* = 0.999 / 0.999 / 1.000 across the three test regimes — matching D_3_cv's in-distribution 1.000 on D_3 while also matching D_1_cv's and D_2_cv's in-distribution numbers on their own regimes. D_disc_act mirrors this. Both are uniformly competent across σ in the persigma breakdown (every bin ≈ 1.0).
- **Static-σ models reproduce the V2 shortcut.** D_1_cv on D_2 = 0.318, D_3_cv on D_1 = 0.016. Off-diagonal cells collapse to ~0.02–0.36, confirming the spec's pre-existing finding that single-regime training induces a fixed-scale policy.
- **D_logu_cv is the anomaly.** D_logu_cv on D_1 / D_2 / D_3 = 0.036 / 0.091 / 0.800, and the per-σ breakdown on the persigma cache shows it deteriorates monotonically toward low σ — the σ=1-end bin even has negative R, meaning the model accepts X_t < 0 sequences. The continuous σ-prior exposes a regime where the cv head fails to invert correctly. D_logu_act recovers (R/R* ≈ 0.97 across all three regimes), so this is cv-specific.
- **High oracle agreement does not imply matched payoff** — the V2 pathology re-appears here in static-model OOD cells. Look at D_3_cv on D_1: oracle agreement 0.978 (per-step), R/R* 0.016. Both policies reject ~all positions; the few cells where they disagree are the decision-critical accept steps that determine the entire payoff.
- **Trajectory plots (saved per cv run) are the visual answer to "do random-variance models do in-context σ inference?".** D_disc_cv 's curves adapt to test regime within a few steps; D_logu_cv's curves fail to adapt at low σ. Static-σ cv curves stay near their training-distribution oracle threshold regardless of test regime, visible as flat off-axis lines.
