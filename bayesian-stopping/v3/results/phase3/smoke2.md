# Smoke 2 — random labeler at point-mass sigma=10 vs static labeler

Setting: 1000 sequences with sigma_i = 10.0, n=256.
Random ADP: K1=K2=256, J=64, point-mass sigma=10. Static ADP: K=2048, J=128, sigma=10.
Comparison restricted to t=2..n-1 (random labeler hardcodes y_act_1=0 by design).

| metric | value |
|---|---|
| action agreement (t=2..n-1) | 0.999921 |
| action disagreement (t=2..n-1) | 7.8740e-05 |
| max |y_cv_random - y_cv_static| (t=2..n-1) | 4.806e-02 |
| median |y_cv_random - y_cv_static| (t=2..n-1) | 2.908e-02 |
| static labeler accepts at t=1 (count / 1000) | 0 |
| random labeler accepts at t=1 (count / 1000) | 0 |

**Gate: action agreement >= 0.9999: PASS**
