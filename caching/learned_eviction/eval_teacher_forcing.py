"""Teacher-forced evaluation on the held-out test set.

The model is evaluated on every full-cache timestep of each held-out test
trace, always receiving Belady's own cache state as input (teacher forcing).
This matches the training distribution under ``--label-mode all_timesteps``.

Reports:

    (1) Test loss and slot-selection accuracy (agreement with Belady),
        broken down by dominating family (LRU/LFU/ARC) and by timestep type
        (all full-cache / miss-full event only / hit only).

    (2) Slot-level agreement with LRU and LFU computed from Belady's cache
        state — i.e. "given the cache the model sees, which slot would
        LRU/LFU pick?". These are well-defined per test example (coverage is
        trivially 100%). ARC is intentionally excluded: its eviction rule
        depends on its internal T1/T2/B1/B2 partition, which is not a
        function of Belady's cache state alone. ARC appears only in the
        rollout hit-rate comparison (eval_rollout.py).

Usage (from caching/):
    python3 -m learned_eviction.eval_teacher_forcing \\
        --run-dir learned_eviction/runs/all_timesteps --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import HypotheticalEvictionDataset, _labels_for_trace_file
from .model import CacheEvictionTransformer


def _precompute_baselines(test_ds, split, trace_files, cache_root, k):
    """For every test example, compute:
        - is_event:   True iff timestep t is a miss-full event (actual eviction)
        - family:     dominating family of the source trace (LRU/LFU/ARC)
        - lru_slot:   slot LRU would evict from Belady's cache at t
        - lfu_slot:   slot LFU would evict from Belady's cache at t
                      (argmin count, LRU tiebreak)

    Works by walking each test trace once in trace order and snapshotting
    the LRU/LFU rankings at every test timestep.
    """
    N = len(test_ds.index)
    is_event = np.zeros(N, dtype=bool)
    family = np.empty(N, dtype="U3")
    lru_slot = np.full(N, -1, dtype=np.int64)
    lfu_slot = np.full(N, -1, dtype=np.int64)

    # Group test events by (fi, r) so we walk each trace once.
    events_per_trace: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for idx, (fi, r, t) in enumerate(test_ds.index):
        events_per_trace[(fi, r)].append((idx, t))

    # miss_full_mask + dominating family from *_scores.json
    miss_full_per_trace = {}
    dom_family_per_trace = {}
    for fi, trace_path in enumerate(trace_files):
        lab = _labels_for_trace_file(trace_path, k, cache_root)
        mm = lab["miss_full_mask"]
        score_path = trace_path.parent / trace_path.name.replace(
            "_traces.npy", "_scores.json"
        )
        with open(score_path) as f:
            scores = json.load(f)
        for r in split["test"][fi]:
            miss_full_per_trace[(fi, r)] = mm[r]
            dom_family_per_trace[(fi, r)] = max(scores[r].items(), key=lambda kv: kv[1])[0]

    for (fi, r), events in events_per_trace.items():
        events.sort(key=lambda x: x[1])
        trace = test_ds.traces_per_file[fi][r]
        cache_states = test_ds.cache_states_per_file[fi]
        mm = miss_full_per_trace[(fi, r)]
        dom = dom_family_per_trace[(fi, r)]

        last_access: dict[int, int] = {}   # item -> most recent t' < t with trace[t'] == item
        counts: dict[int, int] = {}        # item -> number of accesses in trace[:t]

        ei = 0
        for t in range(len(trace)):
            # Snapshot state for every test event at this exact t.
            while ei < len(events) and events[ei][1] == t:
                idx, _ = events[ei]
                cache_at_t = cache_states[r, t]  # (k,) shifted: 0 = empty

                best_lru_time = float("inf"); best_lru_slot = 0
                best_lfu_count = float("inf"); best_lfu_time = float("inf"); best_lfu_slot = 0

                for slot in range(k):
                    tok = int(cache_at_t[slot])
                    if tok == 0:
                        continue
                    item = tok - 1
                    la = last_access.get(item, -1)
                    cnt = counts.get(item, 0)

                    if la < best_lru_time:
                        best_lru_time = la
                        best_lru_slot = slot

                    if cnt < best_lfu_count or (cnt == best_lfu_count and la < best_lfu_time):
                        best_lfu_count = cnt
                        best_lfu_time = la
                        best_lfu_slot = slot

                lru_slot[idx] = best_lru_slot
                lfu_slot[idx] = best_lfu_slot
                is_event[idx] = bool(mm[t])
                family[idx] = dom
                ei += 1

            # Update running access statistics AFTER snapshot (so they reflect [0, t)).
            x = int(trace[t])
            last_access[x] = t
            counts[x] = counts.get(x, 0) + 1

    return is_event, family, lru_slot, lfu_slot


@torch.no_grad()
def evaluate(args):
    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    data_dir = Path(args.data_dir)

    # ── Load checkpoint + split ─────────────────────────────────────────
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    args_dict = ckpt["args"]
    k = args_dict["k"]
    w = args_dict["context_window"]

    with open(run_dir / "split.json") as f:
        split = json.load(f)
    trace_files = [data_dir / name for name in split["files"]]
    cache_root = data_dir.parent / "belady_cache"

    # ── Test set: all full-cache timesteps, no subsample ─────────────────
    test_ds = HypotheticalEvictionDataset(
        trace_files,
        cache_size=k,
        context_window=w,
        cache_root=cache_root,
        trace_indices=split["test"],
        max_per_trace=None,
    )
    print(f"test set: {len(test_ds):,} full-cache timesteps from {sum(len(s) for s in split['test'])} traces")

    # ── Baselines (LRU, LFU slot picks; event/family flags) ─────────────
    print("precomputing LRU/LFU from Belady cache + event/family metadata...")
    t0 = time.time()
    is_event, family, lru_slot, lfu_slot = _precompute_baselines(
        test_ds, split, trace_files, cache_root, k
    )
    print(f"  done in {time.time() - t0:.1f}s  "
          f"(miss-full events: {int(is_event.sum()):,} = {100*is_event.mean():.1f}%)")

    # ── Model ────────────────────────────────────────────────────────────
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
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} params")

    # ── Forward over the whole test set ─────────────────────────────────
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    N = len(test_ds)
    preds = np.empty(N, dtype=np.int64)
    labels = np.empty(N, dtype=np.int64)
    per_loss = np.empty(N, dtype=np.float64)

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    idx = 0
    t0 = time.time()
    n_batches = len(loader)
    for i, batch in enumerate(loader):
        cache = batch["cache"].to(device, non_blocking=True)
        seq = batch["seq"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        logits = model(cache, seq)
        p = logits.argmax(-1)
        l = loss_fn(logits, label)
        bs = label.shape[0]
        preds[idx:idx + bs] = p.cpu().numpy()
        labels[idx:idx + bs] = label.cpu().numpy()
        per_loss[idx:idx + bs] = l.cpu().numpy()
        idx += bs
        if (i + 1) % max(1, n_batches // 20) == 0:
            elapsed = time.time() - t0
            eta = elapsed * (n_batches - i - 1) / (i + 1)
            print(f"  {100*(i+1)/n_batches:5.1f}%  elapsed {elapsed:5.0f}s  eta {eta:4.0f}s")
    print(f"forward done in {time.time() - t0:.1f}s")

    correct = (preds == labels)
    agree_lru = (preds == lru_slot)
    agree_lfu = (preds == lfu_slot)

    # ── Aggregate into (group × subset) table ───────────────────────────
    # Coverage is 100% by construction: the test dataset only contains
    # full-cache timesteps and LRU/LFU are deterministic functions of the
    # (cache, access-history) pair, so both have a well-defined pick at every
    # example. We therefore don't report coverage.
    def _agg(mask):
        n = int(mask.sum())
        if n == 0:
            return None
        return {
            "n": n,
            "loss": float(per_loss[mask].mean()),
            "acc": float(correct[mask].mean()),
            "agree_LRU": float(agree_lru[mask].mean()),
            "agree_LFU": float(agree_lfu[mask].mean()),
        }

    families = ["LRU", "LFU", "ARC"]
    subsets = [("all", np.ones(N, dtype=bool)),
               ("event", is_event),
               ("hit", ~is_event)]

    result = {"overall": {}, "per_family": {f: {} for f in families}}
    for name, mask in subsets:
        result["overall"][name] = _agg(mask)
        for f in families:
            fm = (family == f)
            result["per_family"][f][name] = _agg(fm & mask)

    # ── Print report ─────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("TEACHER-FORCED TEST RESULTS  (Belady cache as model input at every full-cache timestep)")
    print("=" * 92)

    def _row(label, r):
        if r is None:
            return f"  {label:<20}  {'—':>10}"
        return (f"  {label:<20}  {r['n']:>10,}  "
                f"{r['loss']:>8.4f}  {r['acc']:>8.4f}  "
                f"{r['agree_LRU']:>8.4f}  {r['agree_LFU']:>8.4f}")

    print(f"  {'group':<20}  {'n':>10}  {'loss':>8}  {'acc':>8}  {'agr_LRU':>8}  {'agr_LFU':>8}")
    print("  " + "-" * 80)
    for name, _ in subsets:
        print(_row(f"overall / {name}", result["overall"][name]))
    print()
    for f in families:
        for name, _ in subsets:
            print(_row(f"{f}-dom / {name}", result["per_family"][f][name]))
        print()

    # ── Save JSON ────────────────────────────────────────────────────────
    out_path = run_dir / "eval_teacher_forcing.json"
    with open(out_path, "w") as fp:
        json.dump(result, fp, indent=2)
    print(f"saved to {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, default="learned_eviction/runs/all_timesteps")
    p.add_argument("--data-dir", type=str, default="data/run_20260422")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
