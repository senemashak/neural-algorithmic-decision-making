"""Default parameters for the cache-policy trace dataset."""

from dataclasses import dataclass, field


@dataclass
class GenConfig:
    # Cache and universe
    k: int = 32
    U: int = 512
    T: int = 16_000

    # Acceptance margin: winner must beat runner-up by this absolute hit-rate gap.
    # Per-family override; LFU vs ARC tops out around 0.05 empirically.
    margin: float = 0.04
    margin_per_family: dict = field(default_factory=lambda: {"LRU": 0.04, "LFU": 0.04, "ARC": 0.04})
    warmup_frac: float = 0.10  # fraction of trace ignored when scoring

    # How many to keep per family
    n_per_family: int = 1000

    # LRU family: drifting working set just past cache capacity, fully-disjoint
    # phases. Forces eviction; LRU's pure recency wins; ARC's T2 wastes slots.
    lru_w_range: tuple = (1.0, 1.15)       # fraction of k
    lru_phase_len_range: tuple = (120, 280)
    lru_noise_range: tuple = (0.02, 0.06)
    lru_overlap: float = 0.0

    # LFU family: stationary, moderately-skewed Zipf over a large universe so
    # popular items have long inter-arrival gaps that LRU/ARC can't bridge.
    lfu_alpha_range: tuple = (0.95, 1.15)
    lfu_universe_factor: float = 16.0      # U_eff = factor * k (capped at U)
    lfu_block_shuffle: int = 256           # 0 disables; else shuffle within blocks

    # ARC family: persistent core + phase set + scans
    arc_core_frac_range: tuple = (0.2, 0.4)
    arc_phase_frac_range: tuple = (0.5, 0.8)
    arc_phase_len_range: tuple = (600, 1000)
    arc_p_core: float = 0.25
    arc_p_phase: float = 0.55
    arc_scan_every_phases: int = 2
    arc_scan_len_range: tuple = (128, 256)

    seed: int = 0
