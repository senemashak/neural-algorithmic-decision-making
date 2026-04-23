"""Trace generators for LRU-, LFU-, and ARC-favoring families."""

import numpy as np
from config import GenConfig


def _zipf_weights(n, alpha):
    ranks = np.arange(1, n + 1, dtype=np.float64)
    w = ranks ** (-alpha)
    return w / w.sum()


def gen_lru_trace(rng: np.random.Generator, cfg: GenConfig) -> np.ndarray:
    """Drifting working set: hot set changes between phases, mostly disjoint."""
    k, U, T = cfg.k, cfg.U, cfg.T
    w = int(round(rng.uniform(*cfg.lru_w_range) * k))
    eps = rng.uniform(*cfg.lru_noise_range)
    overlap = cfg.lru_overlap

    trace = np.empty(T, dtype=np.int64)
    pool = np.arange(U)
    active = rng.choice(pool, size=w, replace=False)
    t = 0
    while t < T:
        L = int(rng.integers(*cfg.lru_phase_len_range))
        end = min(t + L, T)
        n_in = max(1, int(round((end - t) * (1 - eps))))
        n_out = (end - t) - n_in
        in_reqs = rng.choice(active, size=n_in, replace=True)
        outside = np.setdiff1d(pool, active, assume_unique=False)
        out_reqs = rng.choice(outside, size=n_out, replace=True) if n_out > 0 else np.empty(0, dtype=np.int64)
        block = np.concatenate([in_reqs, out_reqs])
        rng.shuffle(block)
        trace[t:end] = block
        t = end

        # Replace most of active set; keep `overlap` fraction
        keep = max(0, int(round(w * overlap)))
        keep_idx = rng.choice(len(active), size=keep, replace=False) if keep > 0 else np.empty(0, dtype=np.int64)
        kept = active[keep_idx]
        candidates = np.setdiff1d(pool, kept, assume_unique=False)
        new_active = rng.choice(candidates, size=w - keep, replace=False)
        active = np.concatenate([kept, new_active])
    return trace


def gen_lfu_trace(rng: np.random.Generator, cfg: GenConfig) -> np.ndarray:
    """Stationary Zipf popularity over a hot subset; optional block-shuffle."""
    k, U, T = cfg.k, cfg.U, cfg.T
    alpha = rng.uniform(*cfg.lfu_alpha_range)
    U_eff = min(U, int(round(cfg.lfu_universe_factor * k)))
    items = rng.choice(U, size=U_eff, replace=False)
    rng.shuffle(items)  # decouple popularity rank from item id
    probs = _zipf_weights(U_eff, alpha)
    trace = rng.choice(items, size=T, replace=True, p=probs)

    bs = cfg.lfu_block_shuffle
    if bs and bs > 1:
        for s in range(0, T, bs):
            block = trace[s:s + bs]
            rng.shuffle(block)
            trace[s:s + bs] = block
    return trace.astype(np.int64)


def gen_arc_trace(rng: np.random.Generator, cfg: GenConfig) -> np.ndarray:
    """Persistent core + drifting phase set + periodic scan blocks."""
    k, U, T = cfg.k, cfg.U, cfg.T
    core_size = max(1, int(round(rng.uniform(*cfg.arc_core_frac_range) * k)))
    phase_size = max(1, int(round(rng.uniform(*cfg.arc_phase_frac_range) * k)))
    pool = np.arange(U)
    core = rng.choice(pool, size=core_size, replace=False)
    non_core = np.setdiff1d(pool, core)
    phase_set = rng.choice(non_core, size=phase_size, replace=False)

    p_core = cfg.arc_p_core
    p_phase = cfg.arc_p_phase
    p_scan = max(0.0, 1.0 - p_core - p_phase)  # used for noise outside scan blocks

    trace = np.empty(T, dtype=np.int64)
    t = 0
    phase_idx = 0
    while t < T:
        L = int(rng.integers(*cfg.arc_phase_len_range))
        end = min(t + L, T)
        n = end - t

        # Compose phase requests: core / phase / noise
        choice = rng.choice(3, size=n, p=[p_core, p_phase, p_scan]) if p_scan > 0 else \
                 rng.choice(2, size=n, p=[p_core / (p_core + p_phase), p_phase / (p_core + p_phase)])
        block = np.empty(n, dtype=np.int64)
        from_core = (choice == 0)
        from_phase = (choice == 1)
        from_noise = (choice == 2) if p_scan > 0 else np.zeros(n, dtype=bool)
        if from_core.any():
            block[from_core] = rng.choice(core, size=int(from_core.sum()), replace=True)
        if from_phase.any():
            block[from_phase] = rng.choice(phase_set, size=int(from_phase.sum()), replace=True)
        if from_noise.any():
            outside = np.setdiff1d(pool, np.concatenate([core, phase_set]), assume_unique=False)
            block[from_noise] = rng.choice(outside, size=int(from_noise.sum()), replace=True)
        trace[t:end] = block
        t = end
        phase_idx += 1

        # Insert scan block every few phases
        if t < T and (phase_idx % cfg.arc_scan_every_phases == 0):
            scan_len = int(rng.integers(*cfg.arc_scan_len_range))
            scan_len = min(scan_len, T - t)
            if scan_len > 0:
                # Fresh-ish items, not in core/phase, ideally not seen recently
                outside = np.setdiff1d(pool, np.concatenate([core, phase_set]), assume_unique=False)
                if len(outside) >= scan_len:
                    scan = rng.choice(outside, size=scan_len, replace=False)
                else:
                    scan = rng.choice(outside, size=scan_len, replace=True)
                trace[t:t + scan_len] = scan
                t += scan_len

        # Drift the phase set: replace fully, mostly disjoint from core
        outside_core = np.setdiff1d(pool, core)
        phase_set = rng.choice(outside_core, size=phase_size, replace=False)
    return trace


GENERATORS = {"LRU": gen_lru_trace, "LFU": gen_lfu_trace, "ARC": gen_arc_trace}
