"""Quick smoke test: a handful of traces from each family, confirm intended winner trends."""

import numpy as np
from algorithms import hit_rates
from config import GenConfig
from generators import GENERATORS


def main():
    cfg = GenConfig(n_per_family=0)  # use real T=16000
    rng = np.random.default_rng(0)
    n_per = 20
    for family, gen in GENERATORS.items():
        wins = 0
        gaps = []
        rates = {"LRU": [], "LFU": [], "ARC": []}
        for _ in range(n_per):
            tr = gen(rng, cfg)
            hr = hit_rates(tr, cfg.k, cfg.warmup_frac)
            for k, v in hr.items():
                rates[k].append(v)
            best = max(hr, key=hr.get)
            second = max(v for k, v in hr.items() if k != family)
            if best == family:
                wins += 1
                gaps.append(hr[family] - second)
        avg = {k: float(np.mean(v)) for k, v in rates.items()}
        print(f"[{family}-family] {wins}/{n_per} traces won by {family}; "
              f"mean gap when won = {np.mean(gaps) if gaps else float('nan'):.3f}; "
              f"mean rates: {avg}")


if __name__ == "__main__":
    main()
