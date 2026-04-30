"""
Oracle for Bayesian last-offer stopping with a Normal-Normal conjugate prior.

Implements:
    - Closed-form posterior update (Proposition 1 of research-notes.tex).
    - Known-mu threshold sequence eta_t (eq. 1, section 1.3).
    - Approximate dynamic program for the Bayes-optimal threshold C_t(mu)
      (Algorithm 1, section 2.3).

Indexing convention. The notes use 1-indexed time t in {1, ..., n}. In code we
use 0-indexed time i in {0, ..., n-1} so that X[i] is the (i+1)-th observation
in the notes. Arrays returned by `compute_eta` and `solve_adp` are indexed by
0-indexed *decision step* i in {0, ..., n-2}; at step i = n-1 the agent must
accept and no threshold is used.
"""

from typing import Tuple

import numpy as np
from scipy.special import ndtr, roots_hermite

import config


# ---------------------------------------------------------------------------
# Posterior update (Proposition 1)
# ---------------------------------------------------------------------------

def posterior_update(
    mu_prev: float, tau2_prev: float, x: float, sigma2: float
) -> Tuple[float, float]:
    """One-step conjugate update: (mu_{t-1}, tau2_{t-1}, X_t) -> (mu_t, tau2_t)."""
    tau2_new = 1.0 / (1.0 / tau2_prev + 1.0 / sigma2)
    mu_new = tau2_new * (mu_prev / tau2_prev + x / sigma2)
    return mu_new, tau2_new


def posterior_path_batch(
    X_batch: np.ndarray, mu_0: float, tau0_2: float, sigma2: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized closed-form posterior path.

    Args:
        X_batch: (B, n) observations.

    Returns:
        mu_path:   (B, n) where mu_path[b, i] = posterior mean after observing
                   X_batch[b, :i+1]   (i.e. mu_{i+1} in 1-indexed notation).
        tau2_path: (n,)  posterior variance after i+1 observations; data-free.
    """
    B, n = X_batch.shape
    t = np.arange(1, n + 1, dtype=float)                       # 1, 2, ..., n
    tau2_path = 1.0 / (1.0 / tau0_2 + t / sigma2)              # (n,)
    S = np.cumsum(X_batch, axis=1)                             # (B, n)
    mu_path = tau2_path[None, :] * (mu_0 / tau0_2 + S / sigma2)
    return mu_path, tau2_path


# ---------------------------------------------------------------------------
# Known-mu threshold sequence (eq. 1)
# ---------------------------------------------------------------------------

def _psi(z: np.ndarray) -> np.ndarray:
    """psi(z) = z * Phi(z) + phi(z)."""
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    return z * ndtr(z) + phi


def compute_eta(n: int) -> np.ndarray:
    """Standardized known-mu thresholds eta_t for the plug-in / prior-only rules.

    Returns an array of length n-1 with eta[i] = eta_{i+1} (1-indexed). The base
    case is eta[n-2] = 0 (i.e. eta_{n-1} = 0); earlier entries are filled by
    eta[i] = psi(eta[i+1]). At step n-1 the agent must accept, so no eta is
    needed there.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    eta = np.zeros(n - 1)
    for i in range(n - 3, -1, -1):
        eta[i] = _psi(eta[i + 1])
    return eta


# ---------------------------------------------------------------------------
# Uniform-grid linear interpolation with boundary clipping
# ---------------------------------------------------------------------------

def interp_uniform(
    x: np.ndarray, grid: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Linear interp of `values` defined on a uniform `grid`, with boundary
    clipping (returns the boundary value for x outside the grid). Vectorized:
    `x` may have any shape; `grid` and `values` must be 1-D of equal length.
    """
    K = grid.shape[0]
    h = (grid[-1] - grid[0]) / (K - 1)
    idx = (x - grid[0]) / h
    idx_clip = np.clip(idx, 0.0, K - 1.0)
    lo = np.floor(idx_clip).astype(np.int64)
    lo = np.clip(lo, 0, K - 2)
    frac = idx_clip - lo
    return (1.0 - frac) * values[lo] + frac * values[lo + 1]


# ---------------------------------------------------------------------------
# Approximate DP (Algorithm 1)
# ---------------------------------------------------------------------------

def solve_adp(
    n: int,
    mu_0: float,
    sigma2: float,
    tau0_2: float,
    K: int = None,
    J: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the approximate backward induction of Algorithm 1.

    The decision-step index i in {0, ..., n-2} maps to 1-indexed stage t = i+1.

    Per-stage grid I_t = [mu_0 - 5 s_t, mu_0 + 5 s_t] with
        s_t = sqrt(max(tau_0^2 - tau_t^2, 0))   (the *marginal* std of mu_t
                                                 across sequences).
    The state being discretized is mu_t as a random variable over data, whose
    marginal variance is Var(mu_t) = tau_0^2 - tau_t^2 by the tower property
    (mu = mu_t + noise_t with the two terms uncorrelated). The conditional
    posterior std tau_t is the wrong scale: it is the spread of mu given the
    data, not the spread of mu_t across realizations of the data.

    Returns:
        C_hat: (n-1, K) threshold table; C_hat[i, k] approximates
               C_{i+1}(m_{i+1}^{(k+1)}).
        grids: (n-1, K) per-stage grids m_t.
    """
    if K is None:
        K = config.K
    if J is None:
        J = config.J
    if n < 2 or K < 2 or J < 1:
        raise ValueError("require n >= 2, K >= 2, J >= 1")

    # Per-stage posterior std, predictive variance, update coefficient.
    t_arr = np.arange(1, n, dtype=float)                       # 1, ..., n-1
    tau2 = 1.0 / (1.0 / tau0_2 + t_arr / sigma2)               # (n-1,)
    v = tau2 + sigma2                                          # predictive var
    alpha = sigma2 / v                                         # update weight on prior mean

    # Per-stage half-widths: marginal std of mu_t across sequences.
    half_w = np.sqrt(np.maximum(tau0_2 - tau2, 0.0))           # (n-1,)

    grids = np.empty((n - 1, K))
    for i in range(n - 1):
        grids[i] = np.linspace(mu_0 - 5.0 * half_w[i], mu_0 + 5.0 * half_w[i], K)

    # Gauss-Hermite nodes/weights for int f(z) e^{-z^2} dz.
    z_nodes, w_nodes = roots_hermite(J)

    C_hat = np.empty((n - 1, K))
    # Base case: at i = n-2 (stage t = n-1), C_{n-1}(mu) = mu.
    C_hat[n - 2] = grids[n - 2]

    inv_sqrt_pi = 1.0 / np.sqrt(np.pi)
    for i in range(n - 3, -1, -1):
        m = grids[i]                                           # (K,)
        s2v = np.sqrt(2.0 * v[i])
        # x[k, j] = m[k] + sqrt(2 v_t) z_j ; mu'[k, j] = alpha m[k] + (1-alpha) x[k, j]
        x = m[:, None] + s2v * z_nodes[None, :]                # (K, J)
        mu_prime = alpha[i] * m[:, None] + (1.0 - alpha[i]) * x
        # Lookup C_{t+1}^{lin}(mu'), interpolated with boundary clipping.
        C_next = interp_uniform(mu_prime, grids[i + 1], C_hat[i + 1])
        # Quadrature: (1/sqrt pi) sum_j w_j max(x_j, C_next_j).
        S = (w_nodes[None, :] * np.maximum(x, C_next)).sum(axis=1)
        C_hat[i] = inv_sqrt_pi * S

    return C_hat, grids


def C_hat_lin(
    t: int, mu_prime: np.ndarray, C_hat: np.ndarray, grids: np.ndarray
) -> np.ndarray:
    """Off-grid lookup at stage `t` (0-indexed in {0, ..., n-2}) with boundary
    clipping. `mu_prime` may be a scalar or any-shape array.
    """
    return interp_uniform(np.asarray(mu_prime), grids[t], C_hat[t])
