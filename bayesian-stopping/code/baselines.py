"""
Six baseline policies for Bayesian last-offer stopping (section 1.3 of the
research notes).

All policies share the signature `policy(X_seq, n, ...) -> stopping_time`,
where `X_seq` has shape (n,) and `stopping_time` is a 1-indexed integer in
[1, n]. Stopping at step t collects payoff X_seq[t-1].
"""

import numpy as np

from oracle import interp_uniform, posterior_update


# ---------------------------------------------------------------------------
# Hindsight upper bound and prior-free heuristic
# ---------------------------------------------------------------------------

def offline(X_seq: np.ndarray, n: int) -> int:
    """Pick argmax in hindsight (not implementable online)."""
    return int(np.argmax(X_seq)) + 1


def secretary(X_seq: np.ndarray, n: int) -> int:
    """Skip the first r = floor(n/e) offers, then accept the first that
    exceeds the running max from the skip phase. Force-accept at step n.
    """
    r = int(np.floor(n / np.e))
    if r == 0:
        return 1
    M_r = float(X_seq[:r].max())
    for i in range(r, n - 1):                                  # 1-indexed t = r+1, ..., n-1
        if X_seq[i] > M_r:
            return i + 1
    return n


# ---------------------------------------------------------------------------
# Threshold-of-form X_t >= mu_hat + sigma * eta_t
# ---------------------------------------------------------------------------

def plug_in(X_seq: np.ndarray, n: int, sigma: float, eta: np.ndarray) -> int:
    """Plug-in / frequentist: replace mu by the running MLE bar X_t."""
    cum = np.cumsum(X_seq)
    for i in range(n - 1):
        x_bar = cum[i] / (i + 1)
        if X_seq[i] >= x_bar + sigma * eta[i]:
            return i + 1
    return n


def prior_only(
    X_seq: np.ndarray, n: int, mu_0: float, sigma: float, eta: np.ndarray
) -> int:
    """Prior-only: replace mu by the prior mean mu_0 (deterministic threshold)."""
    for i in range(n - 1):
        if X_seq[i] >= mu_0 + sigma * eta[i]:
            return i + 1
    return n


# ---------------------------------------------------------------------------
# Bayesian thresholds: myopic and Bayes-optimal oracle
# ---------------------------------------------------------------------------

def myopic(
    X_seq: np.ndarray, n: int, mu_0: float, tau0_2: float, sigma2: float
) -> int:
    """Stop at first t with X_t >= E[X_{t+1} | b_t] = mu_t (one-step lookahead)."""
    mu, tau2 = mu_0, tau0_2
    for i in range(n - 1):
        mu, tau2 = posterior_update(mu, tau2, X_seq[i], sigma2)
        if X_seq[i] >= mu:
            return i + 1
    return n


def bayes_optimal(
    X_seq: np.ndarray,
    n: int,
    mu_0: float,
    tau0_2: float,
    sigma2: float,
    C_hat: np.ndarray,
    grids: np.ndarray,
) -> int:
    """Reservation rule with the precomputed approximate threshold C_hat_t(mu_t)."""
    mu, tau2 = mu_0, tau0_2
    for i in range(n - 1):
        mu, tau2 = posterior_update(mu, tau2, X_seq[i], sigma2)
        thresh = float(interp_uniform(np.asarray(mu), grids[i], C_hat[i]))
        if X_seq[i] >= thresh:
            return i + 1
    return n
