"""
Dataset generation, oracle labeling, and held-out / streaming pipelines for
the three regimes of section 3 of research-notes.tex (v2 spec).

v2 spec varies sigma at fixed tau_0 (=10), so rho = sigma^2 / tau_0^2:
    D_1 — data-dominant  (sigma = 1,   rho = 0.01)
    D_2 — balanced       (sigma = 10,  rho = 1.00)
    D_3 — prior-dominant (sigma = 100, rho = 100.0)

For each (X_1, ..., X_n) we emit two oracle label streams of length n-1:
    y^cv[t-1]  := hat C_t^{lin}(mu_t)            (regression target)
    y^act[t-1] := 1[X_t >= hat C_t^{lin}(mu_t)]   (binary action target)
At t = n the agent must accept (forced acceptance carries no learning signal),
so labels are only produced for t in {1, ..., n-1}.
"""

from dataclasses import dataclass
from typing import Iterator, NamedTuple, Tuple

import numpy as np

import config
from oracle import (
    C_hat_lin,
    interp_uniform,
    posterior_path_batch,
    solve_adp,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetConfig:
    name: str
    sigma: float
    tau0_2: float = config.TAU0_2
    mu_0: float = config.MU_0
    n: int = config.N

    @property
    def sigma2(self) -> float:
        return self.sigma * self.sigma

    @property
    def rho(self) -> float:
        return self.sigma2 / self.tau0_2


DATASETS = {
    i: DatasetConfig(name=config.regime_name(i), sigma=config.SIGMA_VALUES[i])
    for i in (1, 2, 3)
}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_sequences(
    cfg: DatasetConfig, N: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw N sequences from the generative model.

    For each i: mu_i ~ N(mu_0, tau_0^2); X_i,1..X_i,n iid ~ N(mu_i, sigma^2).
    Vectorized; uses the supplied numpy Generator for reproducibility.
    """
    mu = rng.normal(cfg.mu_0, np.sqrt(cfg.tau0_2), size=N)              # (N,)
    noise = rng.normal(0.0, cfg.sigma, size=(N, cfg.n))                 # (N, n)
    return mu[:, None] + noise                                          # (N, n)


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------

def label_sequences(
    X: np.ndarray,
    cfg: DatasetConfig,
    C_hat: np.ndarray,
    grids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Oracle labels for every step t in {1, ..., n-1}.

    Args:
        X:     (N, n) observations.
        cfg:   DatasetConfig used for the generative model.
        C_hat: (n-1, K) precomputed ADP threshold table for `cfg`.
        grids: (n-1, K) per-stage grids for `cfg`.

    Returns:
        y_cv:  (N, n-1) float64. y_cv[i, t-1] = hat C_t^{lin}(mu_t^{(i)}).
        y_act: (N, n-1) float64. y_act = 1[X[:, :n-1] >= y_cv].
    """
    N, n = X.shape
    # Closed-form posterior path: mu_path[i, t-1] = posterior mean after
    # observing X[i, :t]   (i.e. mu_t in 1-indexed notation).
    mu_path, _ = posterior_path_batch(X, cfg.mu_0, cfg.tau0_2, cfg.sigma2)

    # Grid lookup, one stage at a time (grids are different per stage so we
    # can't trivially batch over stages; the per-stage call is already
    # vectorized over the N sequences).
    y_cv = np.empty((N, n - 1), dtype=np.float64)
    for i in range(n - 1):
        y_cv[:, i] = interp_uniform(mu_path[:, i], grids[i], C_hat[i])

    y_act = (X[:, :n - 1] >= y_cv).astype(np.float64)
    return y_cv, y_act


# ---------------------------------------------------------------------------
# Held-out val/test pipeline
# ---------------------------------------------------------------------------

class HeldOutSet(NamedTuple):
    X_val: np.ndarray
    y_cv_val: np.ndarray
    y_act_val: np.ndarray
    X_test: np.ndarray
    y_cv_test: np.ndarray
    y_act_test: np.ndarray
    C_hat: np.ndarray
    grids: np.ndarray


def build_val_test(
    cfg: DatasetConfig,
    *,
    seed_val: int = config.SEED_VAL,
    seed_test: int = config.SEED_TEST,
    N_val: int = config.N_VAL,
    N_test: int = config.N_TEST,
    K: int = config.K,
    J: int = config.J,
) -> HeldOutSet:
    """Solve the ADP once for `cfg`, then sample + label fixed val and test
    splits with distinct seeds. C_hat and grids are returned for reuse at
    evaluation time (per the streaming pipeline).
    """
    C_hat, grids = solve_adp(cfg.n, cfg.mu_0, cfg.sigma2, cfg.tau0_2, K=K, J=J)

    X_val = sample_sequences(cfg, N_val, np.random.default_rng(seed_val))
    y_cv_val, y_act_val = label_sequences(X_val, cfg, C_hat, grids)

    X_test = sample_sequences(cfg, N_test, np.random.default_rng(seed_test))
    y_cv_test, y_act_test = label_sequences(X_test, cfg, C_hat, grids)

    return HeldOutSet(
        X_val, y_cv_val, y_act_val,
        X_test, y_cv_test, y_act_test,
        C_hat, grids,
    )


# ---------------------------------------------------------------------------
# Streaming training iterator
# ---------------------------------------------------------------------------

def stream_batches(
    cfg: DatasetConfig,
    batch_size: int,
    rng: np.random.Generator,
    C_hat: np.ndarray,
    grids: np.ndarray,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Infinite generator of fresh (X, y_cv, y_act) batches.

    Sampling and labeling happen inline; no DP solve per batch. Pass C_hat
    and grids precomputed for `cfg` (e.g. via `build_val_test`).
    """
    while True:
        X = sample_sequences(cfg, batch_size, rng)
        y_cv, y_act = label_sequences(X, cfg, C_hat, grids)
        yield X, y_cv, y_act
