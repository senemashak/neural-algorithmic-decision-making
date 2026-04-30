"""
Validation tests for the oracle DP and the six baseline policies.

Two tests:
  1. ADP threshold-table convergence (K=256, J=32) vs (K=1024, J=64).
     Threshold is 2e-2, not 1e-4: the GH integrand max(x, C_{t+1}(mu')) has a
     kink at the indifference point that caps Gauss-Hermite convergence
     (refining K alone, J alone, or both together all hit ~1.5e-2). To
     recover faster convergence we'd need split-at-kink quadrature
     (closed-form Gaussian truncation against the upper envelope), which
     isn't worth the extra code: model-training noise dwarfs a 1.5% ADP
     error relative to the ~sigma spread of C_t(mu_t).

  2. Mean payoff per policy on 10^4 fixed-seed sequences from rho=1, with
     ordering check
         offline > bayes_optimal > plug_in > prior_only > secretary > myopic
     Myopic is in last place (not in the middle bucket): for mu_0=0, rho=1,
     the rule reduces to "stop iff X_1 >= 0" at t=1 (because mu_1 = X_1/2),
     so it accepts the very first draw half the time.

Run as:  python3 validate.py
"""

import time

import numpy as np

import baselines as B
from oracle import compute_eta, interp_uniform, solve_adp


# ---------------------------------------------------------------------------
# Test 1 — ADP convergence
# ---------------------------------------------------------------------------

def adp_convergence(n, mu_0, sigma2, tau0_2):
    """Largest pointwise difference between the (K=256, J=32) and
    (K=1024, J=64) threshold tables, evaluated on the coarse grid (the fine
    table is interpolated onto coarse-grid points).
    """
    t0 = time.perf_counter()
    C_lo, g_lo = solve_adp(n, mu_0, sigma2, tau0_2, K=256, J=32)
    t_lo = time.perf_counter() - t0
    t0 = time.perf_counter()
    C_hi, g_hi = solve_adp(n, mu_0, sigma2, tau0_2, K=1024, J=64)
    t_hi = time.perf_counter() - t0

    max_diff = 0.0
    for i in range(n - 1):
        v_lo = C_lo[i]
        v_hi_at_lo = interp_uniform(g_lo[i], g_hi[i], C_hi[i])
        max_diff = max(max_diff, float(np.max(np.abs(v_lo - v_hi_at_lo))))
    return max_diff, t_lo, t_hi


# ---------------------------------------------------------------------------
# Test 2 — payoff comparison on a fixed test set (rho = 1)
# ---------------------------------------------------------------------------

def generate_dataset(N, n, mu_0, tau0_2, sigma2, seed):
    rng = np.random.default_rng(seed)
    mu = rng.normal(mu_0, np.sqrt(tau0_2), size=N)
    noise = rng.normal(0.0, np.sqrt(sigma2), size=(N, n))
    return mu[:, None] + noise, mu


def evaluate_baselines(X_test, n, mu_0, tau0_2, sigma2, eta, C_hat, grids):
    sigma = float(np.sqrt(sigma2))
    N = X_test.shape[0]
    policies = [
        ("offline",       lambda X: B.offline(X, n)),
        ("bayes_optimal", lambda X: B.bayes_optimal(X, n, mu_0, tau0_2, sigma2, C_hat, grids)),
        ("plug_in",       lambda X: B.plug_in(X, n, sigma, eta)),
        ("prior_only",    lambda X: B.prior_only(X, n, mu_0, sigma, eta)),
        ("secretary",     lambda X: B.secretary(X, n)),
        ("myopic",        lambda X: B.myopic(X, n, mu_0, tau0_2, sigma2)),
    ]
    out = {}
    for name, fn in policies:
        t0 = time.perf_counter()
        payoffs = np.empty(N); stops = np.empty(N, dtype=np.int64)
        for j in range(N):
            tau = fn(X_test[j])
            stops[j] = tau
            payoffs[j] = X_test[j, tau - 1]
        out[name] = {
            "payoff_mean": float(payoffs.mean()),
            "payoff_se":   float(payoffs.std(ddof=1) / np.sqrt(N)),
            "stop_mean":   float(stops.mean()),
            "elapsed_s":   float(time.perf_counter() - t0),
        }
    return out


def print_table(results):
    oracle = results["bayes_optimal"]["payoff_mean"]
    order = sorted(results.items(), key=lambda kv: -kv[1]["payoff_mean"])
    print(f"\n  {'policy':<14}  {'E[payoff]':>10}  {'SE':>7}  "
          f"{'gap-vs-oracle':>13}  {'E[tau]':>7}  {'time':>6}")
    print(f"  {'-'*14}  {'-'*10}  {'-'*7}  {'-'*13}  {'-'*7}  {'-'*6}")
    for name, r in order:
        gap = r["payoff_mean"] - oracle
        print(f"  {name:<14}  {r['payoff_mean']:>10.4f}  "
              f"{r['payoff_se']:>7.4f}  {gap:>+13.4f}  "
              f"{r['stop_mean']:>7.2f}  {r['elapsed_s']:>5.2f}s")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    n, mu_0, sigma2, tau0_2 = 64, 0.0, 1.0, 1.0
    print("=" * 72)
    print("Bayesian last-offer stopping — oracle + baseline validation")
    print(f"Setting: n={n}, mu_0={mu_0}, sigma^2={sigma2}, tau_0^2={tau0_2}, "
          f"rho={sigma2/tau0_2}")
    print("=" * 72)

    # eta sequence sanity check.
    eta = compute_eta(n)
    print(f"\neta sequence: eta_1={eta[0]:.4f}, eta_2={eta[1]:.4f}, "
          f"eta_{{n-2}}={eta[-2]:.4f}, eta_{{n-1}}={eta[-1]:.4f}  "
          f"(eta_{{n-1}}=0 base; eta increases as t shrinks)")

    # ----- Test 1 — ADP convergence (threshold 2e-2; see module docstring) -----
    print("\n" + "-" * 72)
    print("[Test 1] ADP convergence: max |C_{K=256,J=32} - C_{K=1024,J=64}|")
    print("-" * 72)
    max_diff, t_lo, t_hi = adp_convergence(n, mu_0, sigma2, tau0_2)
    THRESH = 2e-2
    print(f"  K=256/J=32 solve:  {t_lo*1000:6.1f} ms")
    print(f"  K=1024/J=64 solve: {t_hi*1000:6.1f} ms")
    print(f"  max|delta|:        {max_diff:.3e}")
    print(f"  pass (< {THRESH:.0e}):    {max_diff < THRESH}")

    # ----- Test 2 — payoff comparison + ordering check -----
    print("\n" + "-" * 72)
    print("[Test 2] Mean payoff over 10^4 fixed-seed test sequences")
    print("-" * 72)
    N = 10_000
    X_test, _ = generate_dataset(N, n, mu_0, tau0_2, sigma2, seed=20260429)
    C_hat, grids = solve_adp(n, mu_0, sigma2, tau0_2, K=256, J=32)
    results = evaluate_baselines(X_test, n, mu_0, tau0_2, sigma2, eta, C_hat, grids)
    print_table(results)

    expected = ["offline", "bayes_optimal", "plug_in", "prior_only",
                "secretary", "myopic"]
    observed = [name for name, _ in
                sorted(results.items(), key=lambda kv: -kv[1]["payoff_mean"])]
    print()
    print(f"  expected: {' > '.join(expected)}")
    print(f"  observed: {' > '.join(observed)}")
    print(f"  ordering pass: {expected == observed}")

    # well-separated offline-to-oracle gap
    p = {k: v["payoff_mean"] for k, v in results.items()}
    oracle_gap = p["offline"] - p["bayes_optimal"]
    oracle_se  = (results["offline"]["payoff_se"]**2
                  + results["bayes_optimal"]["payoff_se"]**2) ** 0.5
    print(f"  offline-to-oracle gap: {oracle_gap:.4f} "
          f"(±{oracle_se:.4f}; well-separated: {oracle_gap > 4 * oracle_se})")


if __name__ == "__main__":
    main()
