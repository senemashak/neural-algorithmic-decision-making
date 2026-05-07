"""
Step 4 sweep configs.

One RunConfig per (distribution, supervision). Stage A pilot uses 5000
steps + 500-step val cadence; Stage B is the spec-pinned full sweep.

Run names:
    Stage A: D_<dist>_<sup>_pilot   (e.g. D_disc_cv_pilot)
    Stage B: D_<dist>_<sup>         (e.g. D_disc_cv)

Wave layout (4 GPUs available — 0, 1, 2, 4; GPU 3 reserved for another job):

    Wave 1: D_1_cv     D_2_cv     D_3_cv     D_disc_cv
    Wave 2: D_logu_cv  D_1_act    D_2_act    D_3_act
    Wave 3: D_disc_act D_logu_act
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

from data.distributions import (
    ALL_DISTRIBUTIONS,
    RANDOM_DISTRIBUTIONS,
    STATIC_DISTRIBUTIONS,
)


# Path roots — derive everything else from these two constants.
V3_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_ROOT = V3_ROOT / 'checkpoints'
RESULTS_PHASE4 = V3_ROOT / 'results' / 'phase4'
RESULTS_PHASE4_PILOT = V3_ROOT / 'results' / 'phase4_pilot'
DATA_CACHE = V3_ROOT / 'data' / 'cache'
ORACLE_TABLES = V3_ROOT / 'oracle' / 'tables'


# ---------------------------------------------------------------------------
# Spec-pinned hyperparameters (Section 5.2 of v3 spec, plus the pilot deltas)
# ---------------------------------------------------------------------------

LR = 1e-4
BATCH_SIZE = 64
WARMUP_FRAC = 0.20
TRAIN_LOG_EVERY = 100

PILOT_STEPS = 5_000
PILOT_VAL_EVERY = 500
PILOT_PERIODIC_EVERY = None         # No periodic checkpoints in pilot

FULL_VAL_EVERY_STATIC = 2_000
FULL_VAL_EVERY_RANDOM = 3_000
FULL_PERIODIC_EVERY = 50_000

# Step counts for Stage B.
_FULL_STEPS = {
    'cv':  {'D_1': 200_000, 'D_2': 200_000, 'D_3': 200_000,
            'D_disc': 300_000, 'D_logu': 300_000},
    'act': {'D_1': 100_000, 'D_2': 100_000, 'D_3': 100_000,
            'D_disc': 150_000, 'D_logu': 150_000},
}


# Wave layout: each tuple lists run names, length must match GPU_IDS for full
# parallelism (Wave 3 has only 2 runs, uses first 2 GPUs).
WAVES = [
    ('D_1_cv',     'D_2_cv',     'D_3_cv',     'D_disc_cv'),    # Wave 1
    ('D_logu_cv',  'D_1_act',    'D_2_act',    'D_3_act'),       # Wave 2
    ('D_disc_act', 'D_logu_act'),                                # Wave 3
]
GPU_IDS = (0, 1, 2, 4)              # NEVER 3 — that GPU is in use


# ---------------------------------------------------------------------------
# RunConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    distribution: str
    supervision: str                 # 'cv' or 'act'
    stage: str                       # 'pilot' or 'full'
    step_count: int
    val_every: int
    periodic_every: int | None       # None = no periodic checkpoints
    seed: int
    train_log_every: int = TRAIN_LOG_EVERY
    lr: float = LR
    batch_size: int = BATCH_SIZE
    warmup_frac: float = WARMUP_FRAC
    n: int = 256
    d_emb: int = 128
    n_layers: int = 8
    n_heads: int = 4

    @property
    def base_name(self) -> str:
        return f'{self.distribution}_{self.supervision}'

    @property
    def run_name(self) -> str:
        return f'{self.base_name}_pilot' if self.stage == 'pilot' else self.base_name

    @property
    def checkpoint_dir(self) -> Path:
        return CHECKPOINT_ROOT / self.run_name

    @property
    def log_dir(self) -> Path:
        root = RESULTS_PHASE4_PILOT if self.stage == 'pilot' else RESULTS_PHASE4
        return root / self.run_name

    @property
    def is_random(self) -> bool:
        return self.distribution in RANDOM_DISTRIBUTIONS

    def model_kwargs(self) -> dict:
        return {
            'n': self.n, 'd_emb': self.d_emb,
            'n_layers': self.n_layers, 'n_heads': self.n_heads,
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        d['run_name'] = self.run_name
        d['base_name'] = self.base_name
        d['checkpoint_dir'] = str(self.checkpoint_dir)
        d['log_dir'] = str(self.log_dir)
        return d


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def make_run_configs(stage: str) -> List[RunConfig]:
    """All 10 RunConfig objects for `stage` ('pilot' or 'full')."""
    if stage not in ('pilot', 'full'):
        raise ValueError(f'stage must be pilot or full, got {stage!r}')
    cfgs: List[RunConfig] = []
    run_idx = 0
    for dist in ALL_DISTRIBUTIONS:
        for sup in ('cv', 'act'):
            if stage == 'pilot':
                cfg = RunConfig(
                    distribution=dist, supervision=sup, stage='pilot',
                    step_count=PILOT_STEPS, val_every=PILOT_VAL_EVERY,
                    periodic_every=PILOT_PERIODIC_EVERY, seed=1000 + run_idx,
                )
            else:
                val_every = (FULL_VAL_EVERY_RANDOM if dist in RANDOM_DISTRIBUTIONS
                             else FULL_VAL_EVERY_STATIC)
                cfg = RunConfig(
                    distribution=dist, supervision=sup, stage='full',
                    step_count=_FULL_STEPS[sup][dist],
                    val_every=val_every,
                    periodic_every=FULL_PERIODIC_EVERY, seed=1000 + run_idx,
                )
            cfgs.append(cfg)
            run_idx += 1
    return cfgs


def get_run_config(base_name: str, stage: str) -> RunConfig:
    """Look up a config by base name (e.g. 'D_disc_cv', no stage suffix)."""
    base = base_name.removesuffix('_pilot')
    for cfg in make_run_configs(stage):
        if cfg.base_name == base:
            return cfg
    raise KeyError(f'No run for base_name={base!r}, stage={stage!r}')


def parse_run_name(name: str) -> Tuple[str, str]:
    """'D_disc_cv' -> ('D_disc', 'cv'); 'D_logu_act_pilot' -> ('D_logu', 'act')."""
    base = name.removesuffix('_pilot')
    parts = base.split('_')
    if len(parts) < 2:
        raise ValueError(f'Bad run name: {name!r}')
    sup = parts[-1]
    dist = '_'.join(parts[:-1])
    return dist, sup
