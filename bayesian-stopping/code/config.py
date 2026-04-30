"""
Central source of truth for v2 sweep constants. Every other module imports
from here rather than redeclaring.

Spec change from v1: the previous spec varied tau_0 with sigma fixed; v2
varies sigma with tau_0 fixed. Rationale: tau_0 governs cross-sequence
spread of mu and is not identifiable from a single in-context sequence;
sigma is the within-sequence noise and IS identifiable from the sample
variance of X_{1:t}. With sigma varying, the algorithm hypothesis predicts
in-context regime detection: a model that has learned the algorithm should
adapt its threshold based on the in-context variance estimate, while a
shortcut model cannot. This makes the OOD diagnostic substantially sharper.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Problem instance (sec. 1, sec. 3.1 of research-notes.tex, v2)
# ---------------------------------------------------------------------------

MU_0   = 0.0
TAU0   = 10.0           # FIXED across all regimes
TAU0_2 = TAU0 * TAU0    # = 100.0

# Three regimes; index = dataset_id (1, 2, 3).
SIGMA_VALUES = {1: 1.0, 2: 10.0, 3: 100.0}              # within-sequence noise
RHO_VALUES   = {i: (s * s) / TAU0_2 for i, s in SIGMA_VALUES.items()}
                                                        # = {0.01, 1.0, 100.0}

# Marginal sd of X_t across the joint (mu, X_t): sqrt(sigma^2 + tau_0^2).
MARGINAL_X_SD = {i: (s * s + TAU0_2) ** 0.5 for i, s in SIGMA_VALUES.items()}
                                                        # ≈ {10.05, 14.14, 100.5}

# Sequence horizon (was 64 in v1).
N = 256


# ---------------------------------------------------------------------------
# ADP solver (sec. 2.3)
# ---------------------------------------------------------------------------

K = 2048                # grid points per stage     (was 256 in v1, 1024 in initial v2)
J = 128                 # Gauss-Hermite nodes        (was  32 in v1,   64 in initial v2)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

BATCH_SIZE   = 64
LR           = 1e-4
WARMUP_FRAC  = 0.20             # fraction of n_steps spent in linear warmup

# v1 used 500k cv / 200k act steps at n=64. At n=256 each step is ~4x slower
# (attention is O(n^2)); to stay within the ~2h-per-model budget on an
# RTX A6000 we reduce step counts proportionally.
N_STEPS_CV   = 200_000
N_STEPS_ACT  = 100_000


# ---------------------------------------------------------------------------
# Held-out splits
# ---------------------------------------------------------------------------

N_VAL     = 10_000
N_TEST    = 10_000
SEED_VAL  = 42
SEED_TEST = 43


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

D_EMB    = 128
N_LAYERS = 8
N_HEADS  = 4


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

ATTENTION_SNAPSHOT_N_SEQS = 256


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

# All v2 artifacts live under SWEEP_ROOT. Visualization scripts derive
# their figure-output directories from SWEEP_ROOT / "experiments" / "figures".
SWEEP_ROOT = Path("/home/senemi/checkpoints/sweep_v2")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def regime_name(dataset_id: int) -> str:
    return f"D_{dataset_id}"


def regime_label(dataset_id: int) -> str:
    """Long human-readable label keyed by dataset_id."""
    return {1: "data-dominant", 2: "balanced", 3: "prior-dominant"}[dataset_id]
