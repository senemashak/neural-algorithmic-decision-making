"""
Validation tests for dataset.py.

Three checks:
  1. Shapes/dtypes of sample_sequences and label_sequences output.
  2. Per dataset, on 10^4 fixed-seed sequences:
       - oracle policy derived from y_cv labels matches bayes_optimal payoff
         (from baselines.py) sequence-by-sequence,
       - y_act = 1[X[:, :n-1] >= y_cv] holds with exact equality.
  3. Throughput of stream_batches: 1000 batches of 64 sequences.

Run as:  python3 dataset_validate.py
"""

import time

import numpy as np

from baselines import bayes_optimal
from dataset import (
    DATASETS,
    build_val_test,
    label_sequences,
    sample_sequences,
    stream_batches,
)
from oracle import solve_adp


# ---------------------------------------------------------------------------
# Test 1 — shapes and dtypes
# ---------------------------------------------------------------------------

def test_shapes_dtypes():
    print("[Test 1] shapes and dtypes")
    cfg = DATASETS[2]
    rng = np.random.default_rng(42)
    X = sample_sequences(cfg, 8, rng)
    assert X.shape == (8, cfg.n), X.shape
    assert X.dtype == np.float64, X.dtype

    C_hat, grids = solve_adp(cfg.n, cfg.mu_0, cfg.sigma2, cfg.tau0_2)
    y_cv, y_act = label_sequences(X, cfg, C_hat, grids)
    assert y_cv.shape == (8, cfg.n - 1), y_cv.shape
    assert y_act.shape == (8, cfg.n - 1), y_act.shape
    assert y_cv.dtype == np.float64, y_cv.dtype
    assert y_act.dtype == np.float64, y_act.dtype

    print(f"  X     shape={X.shape}  dtype={X.dtype}")
    print(f"  y_cv  shape={y_cv.shape}  dtype={y_cv.dtype}")
    print(f"  y_act shape={y_act.shape}  dtype={y_act.dtype}")
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 2 — oracle-via-labels matches baselines.py.bayes_optimal
# ---------------------------------------------------------------------------

def oracle_payoff_from_labels(X: np.ndarray, y_act: np.ndarray) -> np.ndarray:
    """Stop at first t in {1..n-1} where X_t >= y_cv_{t-1}, else stop at n.
    Equivalent to the policy 1[X_t >= y_cv]; stopping time and payoff are
    derived directly from y_act = 1[X[:, :n-1] >= y_cv].
    """
    N, n = X.shape
    accept = y_act.astype(bool)                                   # (N, n-1)
    any_accept = accept.any(axis=1)
    first_idx = np.argmax(accept, axis=1)                         # (N,) 0-indexed
    stop_idx = np.where(any_accept, first_idx, n - 1)             # 0-indexed
    return X[np.arange(N), stop_idx]


def test_per_dataset():
    print("\n[Test 2] oracle-via-labels matches bayes_optimal (10^4 sequences each)")
    print(f"  {'dataset':<6} {'rho':>5}  "
          f"{'oracle (labels)':>15}  {'bayes_opt (baselines)':>22}  "
          f"{'|delta|':>9}  {'label sanity':>13}")
    print("  " + "-" * 84)

    for k in (1, 2, 3):
        cfg = DATASETS[k]
        bundle = build_val_test(
            cfg, seed_val=10_000 + k, seed_test=20_000 + k,
            N_val=10_000, N_test=10_000,
        )
        X = bundle.X_val
        y_cv = bundle.y_cv_val
        y_act = bundle.y_act_val
        N, n = X.shape

        # Label sanity: y_act should be exactly 1[X[:, :n-1] >= y_cv].
        recompute = (X[:, :n - 1] >= y_cv).astype(y_act.dtype)
        label_ok = np.array_equal(recompute, y_act)

        # Oracle policy from labels.
        oracle_payoff = oracle_payoff_from_labels(X, y_act).mean()

        # bayes_optimal via baselines.py (independent path: recursive posterior
        # update + per-step grid lookup).
        bo_payoffs = np.empty(N)
        for j in range(N):
            tau = bayes_optimal(
                X[j], n, cfg.mu_0, cfg.tau0_2, cfg.sigma2,
                bundle.C_hat, bundle.grids,
            )
            bo_payoffs[j] = X[j, tau - 1]
        bo_payoff = bo_payoffs.mean()

        delta = abs(oracle_payoff - bo_payoff)
        print(f"  {cfg.name:<6} {cfg.rho:>5.2f}  "
              f"{oracle_payoff:>15.6f}  {bo_payoff:>22.6f}  "
              f"{delta:>9.2e}  {str(label_ok):>13}")

        assert delta < 1e-6, f"oracle/bayes_optimal payoff mismatch on {cfg.name}"
        assert label_ok, f"label sanity broken on {cfg.name}"


# ---------------------------------------------------------------------------
# Test 3 — streaming throughput
# ---------------------------------------------------------------------------

def test_streaming_throughput():
    print("\n[Test 3] streaming throughput")
    cfg = DATASETS[2]
    C_hat, grids = solve_adp(cfg.n, cfg.mu_0, cfg.sigma2, cfg.tau0_2)
    rng = np.random.default_rng(0)
    it = stream_batches(cfg, batch_size=64, rng=rng, C_hat=C_hat, grids=grids)

    # warmup
    for _ in range(5):
        next(it)

    n_batches, B = 1000, 64
    t0 = time.perf_counter()
    for _ in range(n_batches):
        X, y_cv, y_act = next(it)
    elapsed = time.perf_counter() - t0
    seq_per_sec = n_batches * B / elapsed

    print(f"  {n_batches} batches x {B} sequences = {n_batches * B} sequences")
    print(f"  total time: {elapsed:.2f}s")
    print(f"  throughput: {seq_per_sec:,.0f} seq/sec ({1e6 * elapsed / (n_batches * B):.1f} us/seq)")


if __name__ == "__main__":
    print("=" * 72)
    print("dataset.py validation")
    print("=" * 72 + "\n")
    test_shapes_dtypes()
    test_per_dataset()
    test_streaming_throughput()
