# Stage A pilot report

Generated: 2026-05-06T01:09:24

## Per-run wall time

Expected: ~50 ms/step for static, ~80 ms/step for random. Flag any run > 150 ms/step.

| run | wall (s) | total steps | ms/step | within budget? |
|---|---|---|---|---|
| D_1_cv_pilot | 289.5 | 5000 | 57.9 | yes |
| D_1_act_pilot | 305.5 | 5000 | 61.1 | yes |
| D_2_cv_pilot | 307.2 | 5000 | 61.4 | yes |
| D_2_act_pilot | 287.0 | 5000 | 57.4 | yes |
| D_3_cv_pilot | 286.1 | 5000 | 57.2 | yes |
| D_3_act_pilot | 278.7 | 5000 | 55.7 | yes |
| D_disc_cv_pilot | 271.0 | 5000 | 54.2 | yes |
| D_disc_act_pilot | 292.7 | 5000 | 58.5 | yes |
| D_logu_cv_pilot | 284.1 | 5000 | 56.8 | yes |
| D_logu_act_pilot | 290.5 | 5000 | 58.1 | yes |

## Per-run convergence shape

| run | val @500 | val @5000 | descending? | shape ok? |
|---|---|---|---|---|
| D_1_cv_pilot | 2.815e+01 | 8.727e-01 | yes | yes |
| D_1_act_pilot | 9.362e-02 | 2.567e-03 | yes | yes |
| D_2_cv_pilot | 3.241e+01 | 6.684e-01 | yes | yes |
| D_2_act_pilot | 7.530e-02 | 2.612e-03 | yes | yes |
| D_3_cv_pilot | 4.826e+02 | 2.447e+02 | yes | yes |
| D_3_act_pilot | 8.541e-02 | 1.432e-02 | yes | yes |
| D_disc_cv_pilot | 1.828e+02 | 1.056e+02 | yes | yes |
| D_disc_act_pilot | 1.497e-01 | 1.126e-02 | yes | yes |
| D_logu_cv_pilot | 1.004e+02 | 3.423e+01 | yes | yes |
| D_logu_act_pilot | 1.210e-01 | 8.719e-03 | yes | yes |

### Mixed-sigma loss-normalization check (D_disc_cv_pilot)

Per-sigma group means after the 5000-step pilot, three
candidate regime-invariance metrics shown side-by-side so the
right one is unambiguous (Spec correction 2026-05-06: cv
normalization is 1/sigma, not 1/sigma^2 — see Spec corrections
in README).

| sigma | n | mean MSE | mean MSE / sigma (loss unit @ 1/σ) | mean MSE / sigma^2 (loss unit @ 1/σ²) | residual = sqrt(MSE) | residual / sigma |
|---|---|---|---|---|---|---|
| 1 | 95 | 2.2795e-01 | 2.2795e-01 | 2.2795e-01 | 0.477 | 0.4774 |
| 10 | 93 | 1.2942e+01 | 1.2942e+00 | 1.2942e-01 | 3.597 | 0.3597 |
| 100 | 68 | 3.2634e+04 | 3.2634e+02 | 3.2634e+00 | 180.649 | 1.8065 |

Max / min ratios across the three sigma groups:

- unnormalized MSE: **1.43e+05**  (expect ~σ² scaling = 1e3..1e5; FAIL)
- MSE / σ (= the new loss value): **1431.60**
- MSE / σ² (= the old loss value, also the fractional-progress metric): **25.22**
- residual / σ (= fraction of σ-magnitude remaining): **5.02**

The gate as written ("MSE / σ < 3×"): FAIL.

### Reload contract check (D_1_cv_pilot @ best.pt)

- saved val_loss at best step (5000): 8.727177e-01
- reloaded val_loss on same val set:        8.727177e-01
- relative error: 0.000e+00
- gate (< 1%): PASS

## Sweep-driver readiness

```
[run_sweep] ===== Wave 1 (4 runs) =====
[run_sweep]   launching D_1_cv on GPU 0 (logs: results/sweep_logs/pilot/D_1_cv.out)
[run_sweep]   launching D_2_cv on GPU 1 (logs: results/sweep_logs/pilot/D_2_cv.out)
[run_sweep]   launching D_3_cv on GPU 2 (logs: results/sweep_logs/pilot/D_3_cv.out)
[run_sweep]   launching D_disc_cv on GPU 4 (logs: results/sweep_logs/pilot/D_disc_cv.out)
[run_sweep]   Wave 1 done
[run_sweep] ===== Wave 2 (4 runs) =====
[run_sweep]   launching D_logu_cv on GPU 0 (logs: results/sweep_logs/pilot/D_logu_cv.out)
[run_sweep]   launching D_1_act on GPU 1 (logs: results/sweep_logs/pilot/D_1_act.out)
[run_sweep]   launching D_2_act on GPU 2 (logs: results/sweep_logs/pilot/D_2_act.out)
[run_sweep]   launching D_3_act on GPU 4 (logs: results/sweep_logs/pilot/D_3_act.out)
[run_sweep]   Wave 2 done
[run_sweep] ===== Wave 3 (2 runs) =====
[run_sweep]   launching D_disc_act on GPU 0 (logs: results/sweep_logs/pilot/D_disc_act.out)
[run_sweep]   launching D_logu_act on GPU 1 (logs: results/sweep_logs/pilot/D_logu_act.out)
[run_sweep]   Wave 3 done
[run_sweep] sweep finished — pilot stage, total wall 923s
```

- GPU 3 mentioned in sweep log: no (correct)
- All 10 pilot run dirs created without collision: yes
- SIGINT handler registered (signal.signal in train/sweep.py): yes (set at process start)

## Epilogue — gate decision

The "MSE/σ flat" gate written for this pilot was malformed at 5000 steps.
Target-magnitude asymmetry (the cv head must travel ~250 to fit σ=100 vs
~2.5 for σ=1) cannot flatten that fast under any normalization choice,
because Adam's per-step parameter motion is approximately `lr` regardless
of the gradient magnitude the loss form prescribes.

The right read is the side-by-side per-σ-group breakdown above. Under
1/σ normalization, σ=1 and σ=10 are approaching converged
(residual/σ = 0.48 and 0.36) while σ=100 is still in early training
(1.81). Under 1/σ², the pilot we ran first showed σ=1 was suppressed
(residual/σ = 1.10) along with σ=100 (2.10) — the previous pilot's
tighter `residual/σ` ratio (3× vs 5× under 1/σ) reflected σ=1 being
held back too, not σ=100 catching up.

**Decision:** proceed to Stage B under 1/σ. Per-σ-group val loss logging
added for the 300k-step random-variance runs so we can track the σ=100
catch-up trajectory through training. Stage B's per-σ curves are the
authoritative answer to "did σ=100 catch up?".
