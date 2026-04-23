"""Generate, simulate, filter, and save the labeled trace dataset.

A trace is accepted into family F only if F's hit rate (after warmup) beats both
other algorithms by at least cfg.margin in absolute terms.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from algorithms import hit_rates
from config import GenConfig
from generators import GENERATORS


def build(cfg: GenConfig, out_dir: Path, max_attempts_factor: int = 20, log_every: int = 50):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    summary = {"config": cfg.__dict__, "families": {}}

    for family in ("LRU", "LFU", "ARC"):
        gen = GENERATORS[family]
        traces = []
        scores = []
        attempts = 0
        accepted = 0
        margin = cfg.margin_per_family.get(family, cfg.margin)
        max_attempts = cfg.n_per_family * max_attempts_factor
        t0 = time.time()
        print(f"[{family}] target={cfg.n_per_family}, margin={margin}, max_attempts={max_attempts}")
        while accepted < cfg.n_per_family and attempts < max_attempts:
            attempts += 1
            trace = gen(rng, cfg)
            hr = hit_rates(trace, cfg.k, cfg.warmup_frac)
            best = max(hr, key=hr.get)
            if best != family:
                continue
            second = max(v for k, v in hr.items() if k != family)
            if hr[family] - second < margin:
                continue
            traces.append(trace.astype(np.int32))
            scores.append(hr)
            accepted += 1
            if accepted % log_every == 0:
                elapsed = time.time() - t0
                print(f"  [{family}] accepted {accepted}/{cfg.n_per_family} "
                      f"(attempts={attempts}, accept_rate={accepted/attempts:.2%}, "
                      f"{elapsed:.1f}s)")

        elapsed = time.time() - t0
        print(f"[{family}] done: kept {accepted} of {attempts} attempts "
              f"({accepted/attempts:.1%}) in {elapsed:.1f}s")

        # Save: stack traces (uniform length T) + per-trace hit-rate triples
        arr = np.stack(traces, axis=0) if traces else np.zeros((0, cfg.T), dtype=np.int32)
        np.save(out_dir / f"{family}_traces.npy", arr)
        with open(out_dir / f"{family}_scores.json", "w") as f:
            json.dump(scores, f)

        summary["families"][family] = {
            "margin": margin,
            "accepted": accepted,
            "attempts": attempts,
            "accept_rate": accepted / max(attempts, 1),
            "elapsed_sec": elapsed,
            "mean_hit_rate": float(np.mean([s[family] for s in scores])) if scores else None,
            "mean_runner_up_gap": float(np.mean([
                s[family] - max(v for k, v in s.items() if k != family) for s in scores
            ])) if scores else None,
        }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved dataset + summary to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data")
    p.add_argument("--n", type=int, default=None, help="override n_per_family")
    p.add_argument("--T", type=int, default=None, help="override trace length")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--margin", type=float, default=None)
    p.add_argument("--max-attempts-factor", type=int, default=20,
                   help="cap = n_per_family * factor")
    args = p.parse_args()

    cfg = GenConfig(seed=args.seed)
    if args.n is not None:
        cfg.n_per_family = args.n
    if args.T is not None:
        cfg.T = args.T
    if args.margin is not None:
        cfg.margin = args.margin

    build(cfg, Path(args.out), max_attempts_factor=args.max_attempts_factor)
