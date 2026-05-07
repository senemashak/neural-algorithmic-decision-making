# Smoke 1 — V3 static labeler vs V2 reference

Setting: 1000 sequences per regime, n=256, static-ADP at K=2048, J=128.
Gates: y_act exact equality; max |y_cv_v3 − y_cv_v2| < 1e-12.

| regime | sigma | max |Δy_cv| | mean |Δy_cv| | y_act disagreement | gate |
|---|---|---|---|---|---|
| D_1 | 1 | 0.000e+00 | 0.000e+00 | 0 / 255000 | PASS |
| D_2 | 10 | 0.000e+00 | 0.000e+00 | 0 / 255000 | PASS |
| D_3 | 100 | 0.000e+00 | 0.000e+00 | 0 / 255000 | PASS |

**Overall: PASS**
