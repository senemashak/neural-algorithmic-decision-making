# Phase 2B — Random-variance ADP convergence report

**Convergence gate (action-label disagreement < 1e-3 per cell): PASS** —
every cell ≤ 2.1e-4, ≥ 4× under threshold.

The original absolute magnitude gate (`max |ΔĈ| < 0.1`) was dropped; see
`Spec corrections` in `v3/README.md` (entry of 2026-05-05) for the
reasoning. Median |ΔĈ|/σ_regime is reported below as the regime-invariant
magnitude diagnostic and is uniformly ~3e-3 (D_disc) / 4–10e-3 (D_logu).

Spec: Algorithm 2 (Section 3.2 of v3 spec). n=256, mu_0=0.0, tau_0^2=100.0,
sigma_max=100.0 (size of per-stage adaptive X_bar bounds).

Production: K1=K2=256, J=64, J_sigma=64 (D_logu only). Reference: K1=K2=512,
J=128, J_sigma=128. Backend: PyTorch / single A6000 GPU, float64 throughout.

Test set: 10000 sequences per regime (independent fresh seeds starting
from 43).

## Solve wall times (single A6000)

| training distribution | wall_prod (s) | wall_ref (s) |
|---|---|---|
| D_disc | 7.09 | 46.54 |
| D_logu | 127.94 | 2014.40 |

## Convergence cells

| training | regime | max \|ΔĈ\| | median \|ΔĈ\| | max \|ΔĈ\| / σ_regime | median \|ΔĈ\| / σ_regime | action-label disagreement | gate |
|---|---|---|---|---|---|---|---|
| D_disc | D_1 (σ=1)   | 4.135e+00 | 2.903e-03 | 4.135e+00 | 2.903e-03 | 8.157e-05 | PASS |
| D_disc | D_2 (σ=10)  | 5.024e+00 | 2.902e-02 | 5.024e-01 | 2.902e-03 | 9.020e-05 | PASS |
| D_disc | D_3 (σ=100) | 1.217e+01 | 2.965e-01 | 1.217e-01 | 2.965e-03 | 8.549e-05 | PASS |
| D_logu | D_1 (σ=1)   | 8.031e-01 | 9.575e-03 | 8.031e-01 | 9.575e-03 | 1.478e-04 | PASS |
| D_logu | D_2 (σ=10)  | 7.896e-01 | 3.895e-02 | 7.896e-02 | 3.895e-03 | 1.533e-04 | PASS |
| D_logu | D_3 (σ=100) | 1.399e+00 | 6.438e-01 | 1.399e-02 | 6.438e-03 | 2.118e-04 | PASS |

## Diagnostic notes

- **Action-label gate, the binding criterion, passes by ≥4× across all
  six cells** (max 2.1e-4, gate 1e-3).

- **Median \|ΔĈ\| / σ_regime is uniformly ~3e-3** (D_disc) and 4–10e-3
  (D_logu) across all three regimes. In other words: production tracks
  reference to within 0.3–1.0% of σ at the median. The absolute number
  scales with σ because Ĉ scales with σ; that's expected.

- **The max-disagreement column is driven by tail outliers in the X_bar
  direction.** Worst case D_disc on D_1: 4.1 absolute on Ĉ values
  themselves ~2.5. The X_bar grid bound is sized for σ_max=100 (~±500 at
  t=1) but K1=256 is fixed; this puts dX_bar ≈ 4 at t=1. Most sequences
  live near X_bar≈0 where this coarseness costs little, but a handful
  with extreme |X_bar_t| land on a single grid cell whose interp differs
  ~Ĉ-scale between K1=256 and K1=512. The action labels for those
  sequences still agree (both say "stop").

- **D_logu numbers are uniformly tighter than D_disc.** The continuous
  σ-prior gives a smoother Ĉ surface; bilinear interp on a smoother
  target has lower error.

## Re-solve cost (worth remembering)

If the labeling pipeline ever needs a re-solve (e.g. a marginal-log-lik
bug surfaces after Step 3), these are the wall numbers on a single A6000:

- D_disc production: ~7 s. Cheap; re-run freely.
- D_disc reference:  ~47 s. Cheap.
- D_logu production: ~2 min. Tolerable.
- D_logu reference:  ~34 min. **Plan ahead** — don't re-solve casually.

Production tables are what the trainer reads; reference solves only run
during convergence checks and stay cached in `oracle/tables/`.
