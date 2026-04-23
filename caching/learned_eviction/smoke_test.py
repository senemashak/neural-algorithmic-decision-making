"""End-to-end sanity check: Belady sim → dataset item → model forward → loss step.

Run from caching-experiments/ as:
    python3 -m learned_eviction.smoke_test
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .belady import simulate_belady
from .dataset import EvictionDataset
from .model import CacheEvictionTransformer


def _belady_sanity():
    rng = np.random.default_rng(0)
    trace = rng.integers(0, 8, size=200)
    cs, mm, el = simulate_belady(trace, k=4)
    assert cs.shape == (200, 4)
    assert mm.dtype == bool and el.dtype == np.int64
    assert (el[~mm] == -1).all(), "non-miss steps should have label -1"
    assert (el[mm] >= 0).all() and (el[mm] < 4).all()
    print(f"[belady] ok — misses-with-full-cache: {int(mm.sum())} / {len(mm)}")


def _dataset_sanity(data_dir: Path, k: int, w: int):
    files = sorted(data_dir.glob("*_traces.npy"))
    assert files, f"no *_traces.npy in {data_dir}"
    # Use a small slice of each file so the first run is fast.
    small_indices = [[0, 1] for _ in files]
    ds = EvictionDataset(
        files,
        cache_size=k,
        context_window=w,
        cache_root=data_dir.parent / "belady_cache",
        trace_indices=small_indices,
    )
    print(f"[dataset] {len(ds):,} eviction steps across {len(files)} files × 2 traces")
    sample = ds[0]
    assert sample["cache"].shape == (k,)
    assert sample["seq"].shape == (w,)
    assert sample["label"].ndim == 0
    assert 0 <= int(sample["label"]) < k
    return ds


def _model_sanity(ds, k: int, w: int, device: str):
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    batch = next(iter(loader))
    model = CacheEvictionTransformer(
        cache_size=k, context_window=w, d_model=64, d_ff=128, n_layers=2,
    ).to(device)

    cache = batch["cache"].to(device)
    seq = batch["seq"].to(device)
    label = batch["label"].to(device)

    logits = model(cache, seq)
    assert logits.shape == (cache.shape[0], k)

    loss = nn.CrossEntropyLoss()(logits, label)
    loss.backward()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] forward ok — logits {tuple(logits.shape)}  loss {loss.item():.3f}  "
          f"params {n_params:,}")


def main():
    _belady_sanity()
    data_dir = Path(__file__).resolve().parents[1] / "data" / "run_20260422"
    ds = _dataset_sanity(data_dir, k=32, w=128)
    _model_sanity(ds, k=32, w=128, device="cpu")
    print("all smoke tests passed.")


if __name__ == "__main__":
    main()
