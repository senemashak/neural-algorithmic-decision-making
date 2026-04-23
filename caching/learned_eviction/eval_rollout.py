"""Rollout hit-rate evaluation: the transformer plays itself on the held-out
test set.

The model starts each trace with an empty cache and on every full-cache miss
picks the slot to evict via argmax over its logits. The cache state evolves
under the model's own decisions (closed loop), so eviction errors cascade.

Reports mean hit rate for the model vs. LRU / LFU / ARC, overall and per
dominating family (dominating family derived per-trace from the recorded
*_scores.json from dataset generation time).

Usage (from caching/):
    python3 -m learned_eviction.eval_rollout
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .model import CacheEvictionTransformer


@torch.no_grad()
def rollout_model(
    model: CacheEvictionTransformer,
    traces: torch.Tensor,     # (B, T) long, raw items in [0, U-1]
    k: int,
    w: int,
    warmup_frac: float,
    verbose_every: int = 1000,
) -> torch.Tensor:
    """Run the model as the cache policy across B traces in parallel.

    Returns (B,) float tensor of per-trace hit rates, measured after the first
    warmup_frac of the trace.
    """
    device = traces.device
    B, T = traces.shape
    traces_shift = traces + 1  # (B, T) — cache tokens are item+1; 0 reserved for empty

    cache = torch.zeros(B, k, dtype=torch.long, device=device)
    window = torch.zeros(B, w, dtype=torch.long, device=device)  # left-padded with 0
    hits = torch.zeros(B, T, dtype=torch.bool, device=device)

    for t in range(T):
        curr = traces_shift[:, t]                          # (B,)

        # Slide window: drop oldest, append current request.
        window = torch.cat([window[:, 1:], curr.unsqueeze(1)], dim=1)

        hit_mask = (cache == curr.unsqueeze(1)).any(dim=1)  # (B,)
        hits[:, t] = hit_mask
        miss_mask = ~hit_mask

        if not miss_mask.any():
            if verbose_every > 0 and t % verbose_every == 0:
                print(f"  t={t}  mean_hit_so_far={hits[:, :t+1].float().mean().item():.4f}")
            continue

        empty_slot = (cache == 0)                           # (B, k)
        has_empty = empty_slot.any(dim=1)                   # (B,)

        miss_empty_mask = miss_mask & has_empty
        if miss_empty_mask.any():
            empty_idx = empty_slot.int().argmax(dim=1)      # (B,)
            b_idx = torch.nonzero(miss_empty_mask, as_tuple=True)[0]
            cache[b_idx, empty_idx[b_idx]] = curr[b_idx]

        miss_full_mask = miss_mask & (~has_empty)
        if miss_full_mask.any():
            b_idx = torch.nonzero(miss_full_mask, as_tuple=True)[0]
            cache_in = cache[b_idx]
            seq_in = window[b_idx]
            logits = model(cache_in, seq_in)                # (n_active, k)
            evict_slot = logits.argmax(dim=1)               # (n_active,)
            cache[b_idx, evict_slot] = curr[b_idx]

        if verbose_every > 0 and t % verbose_every == 0:
            print(f"  t={t}  mean_hit_so_far={hits[:, :t+1].float().mean().item():.4f}")

    start = int(T * warmup_frac)
    hit_rates = hits[:, start:].float().mean(dim=1)         # (B,)
    return hit_rates


def main(args):
    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    data_dir = Path(args.data_dir)

    # ── Load split, scores, and model ──────────────────────────────────
    with open(run_dir / "split.json") as f:
        split = json.load(f)
    trace_files = [data_dir / name for name in split["files"]]

    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    args_dict = ckpt["args"]
    k = args_dict["k"]
    w = args_dict["context_window"]

    model = CacheEvictionTransformer(
        vocab_size=args_dict["vocab_size"],
        cache_size=k,
        context_window=w,
        d_model=args_dict["d_model"],
        d_ff=args_dict["d_ff"],
        n_layers=args_dict["n_layers"],
        dropout=args_dict["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ── Load test traces + baseline scores ─────────────────────────────
    test_traces = []
    test_scores = []
    for fi, trace_path in enumerate(trace_files):
        traces = np.load(trace_path)
        score_path = trace_path.parent / trace_path.name.replace("_traces.npy", "_scores.json")
        with open(score_path) as f:
            scores = json.load(f)
        for r in split["test"][fi]:
            test_traces.append(traces[r])
            test_scores.append(scores[r])

    traces_np = np.stack(test_traces, axis=0)  # (N, T)
    traces_t = torch.from_numpy(traces_np.astype(np.int64)).to(device)
    N = traces_t.shape[0]
    print(f"rollout over {N} test traces, T={traces_t.shape[1]}, k={k}, w={w}")

    # ── Rollout ────────────────────────────────────────────────────────
    t0 = time.time()
    model_hits = rollout_model(
        model, traces_t, k=k, w=w,
        warmup_frac=args.warmup_frac,
        verbose_every=args.verbose_every,
    ).cpu().numpy()
    dt = time.time() - t0
    print(f"rollout done in {dt:.1f}s")

    # ── Aggregate ──────────────────────────────────────────────────────
    families = ["LRU", "LFU", "ARC"]
    baseline = {f: np.array([s[f] for s in test_scores]) for f in families}
    dom_family = np.array([max(s.items(), key=lambda kv: kv[1])[0] for s in test_scores])

    overall = {
        "model": float(model_hits.mean()),
        **{f: float(baseline[f].mean()) for f in families},
    }
    per_dom = {}
    for dom in families:
        mask = (dom_family == dom)
        per_dom[dom] = {
            "n": int(mask.sum()),
            "model": float(model_hits[mask].mean()) if mask.any() else float("nan"),
            **{f: float(baseline[f][mask].mean()) if mask.any() else float("nan")
               for f in families},
        }

    # ── Report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("ROLLOUT HIT-RATE RESULTS")
    print("=" * 64)
    header = f"{'group':>10}  {'n':>5}  {'model':>8}  {'LRU':>8}  {'LFU':>8}  {'ARC':>8}"
    print(header)
    print("-" * len(header))
    print(f"{'overall':>10}  {N:>5}  {overall['model']:>8.4f}  "
          f"{overall['LRU']:>8.4f}  {overall['LFU']:>8.4f}  {overall['ARC']:>8.4f}")
    for dom in families:
        d = per_dom[dom]
        print(f"{dom+'-dom':>10}  {d['n']:>5}  {d['model']:>8.4f}  "
              f"{d['LRU']:>8.4f}  {d['LFU']:>8.4f}  {d['ARC']:>8.4f}")

    out = {
        "overall": overall,
        "per_dominating_family": per_dom,
        "warmup_frac": args.warmup_frac,
        "rollout_seconds": dt,
    }
    out_path = run_dir / "eval_rollout.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved results to {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, default="learned_eviction/runs/default")
    p.add_argument("--data-dir", type=str, default="data/run_20260422")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--verbose-every", type=int, default=1000)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
