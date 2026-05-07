# Phase 2A — Static ADP convergence report
Spec: Algorithm 1 (Section 3.1 of v3 spec). n=256, mu_0=0.0, tau_0^2=100.0.

Production: K=2048, J=128. Reference: K=4096, J=256.
Test set: 10000 sequences per regime, seed=43.
Gate: action-label disagreement < 1e-03 per regime.

| regime | sigma | wall_prod (s) | wall_ref (s) | max |Ĉ_prod − Ĉ_ref|  | (rel σ) | argmax t | action-label disagreement | pass |
|---|---|---|---|---|---|---|---|---|
| D_1 | 1 | 1.90 | 7.93 | 3.308e-02 | 3.308e-02 | 32 | 3.373e-05 | PASS |
| D_2 | 10 | 0.78 | 4.01 | 3.901e-02 | 3.901e-03 | 103 | 3.216e-05 | PASS |
| D_3 | 100 | 0.84 | 5.28 | 2.367e-01 | 2.367e-03 | 111 | 2.745e-05 | PASS |

Overall gate: **PASS**
