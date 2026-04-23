"""Eviction-step dataset built from cache-policy traces.

A training example is a single timestep at which Belady had to evict (the step
is a miss and the cache was full). Each example provides:

  cache:  (k,) int64 — cache contents at the start of that step (shifted, so
                       0 = empty slot)
  seq:    (w,) int64 — the w most recent requests ending with the current one.
                       Early steps are left-padded with 0.
  label:  int        — which of the k slots Belady evicts at this step.

Belady simulation is expensive, so results are cached to disk. The cache key
is the trace file's path + mtime + the cache size.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .belady import simulate_belady_batch


def _cache_path(trace_path: Path, k: int, cache_root: Path) -> Path:
    stat = trace_path.stat()
    key = f"{trace_path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|k={k}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return cache_root / f"{trace_path.stem}.k{k}.{digest}.npz"


def _labels_for_trace_file(trace_path: Path, k: int, cache_root: Path) -> dict:
    """Return dict with cache_states, miss_full_mask, eviction_labels.
    Runs Belady on every trace in the file, caching the result on disk."""
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(trace_path, k, cache_root)
    if cache_file.exists():
        z = np.load(cache_file)
        return {
            "cache_states": z["cache_states"],
            "miss_full_mask": z["miss_full_mask"],
            "eviction_labels": z["eviction_labels"],
        }
    traces = np.load(trace_path)
    cs, mm, el = simulate_belady_batch(traces, k=k)
    np.savez(
        cache_file,
        cache_states=cs.astype(np.int32),
        miss_full_mask=mm,
        eviction_labels=el.astype(np.int16),
    )
    return {"cache_states": cs, "miss_full_mask": mm, "eviction_labels": el}


class EvictionDataset(Dataset):
    """Concatenates eviction steps from one or more .npy trace files.

    Args:
        trace_files:    iterable of paths to (N, T) int trace arrays.
        cache_size:     k
        context_window: w — sequence window length ending at the current step.
        cache_root:     where Belady simulation results are cached.
        trace_indices:  optional per-file list of trace row indices to include
                        (for splitting). Same order as trace_files. If None,
                        use all rows.
    """

    def __init__(
        self,
        trace_files,
        cache_size: int = 32,
        context_window: int = 512,
        cache_root: str | Path = "learned_eviction/belady_cache",
        trace_indices: list[list[int]] | None = None,
    ):
        self.k = cache_size
        self.w = context_window
        self.cache_root = Path(cache_root)
        self.trace_files = [Path(p) for p in trace_files]

        # Per-file storage (lazy numpy arrays + per-row selection)
        self.traces_per_file: list[np.ndarray] = []
        self.cache_states_per_file: list[np.ndarray] = []
        self.labels_per_file: list[np.ndarray] = []
        self.row_indices_per_file: list[list[int]] = []

        # Flat index of eviction steps: list of (file_idx, row_in_file, t)
        self.index: list[tuple[int, int, int]] = []

        for fi, p in enumerate(self.trace_files):
            traces = np.load(p)                        # (N, T) int32
            lab = _labels_for_trace_file(p, self.k, self.cache_root)
            cs = lab["cache_states"]                   # (N, T, k) int
            mm = lab["miss_full_mask"]                 # (N, T) bool
            el = lab["eviction_labels"]                # (N, T) int

            if trace_indices is not None:
                rows = list(trace_indices[fi])
            else:
                rows = list(range(traces.shape[0]))

            self.traces_per_file.append(traces)
            self.cache_states_per_file.append(cs)
            self.labels_per_file.append(el)
            self.row_indices_per_file.append(rows)

            for r in rows:
                # Every timestep where an eviction occurred in Belady's rollout.
                miss_ts = np.nonzero(mm[r])[0]
                for t in miss_ts:
                    self.index.append((fi, int(r), int(t)))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        fi, r, t = self.index[i]
        traces = self.traces_per_file[fi]              # (N, T) int32, values in [0, U-1]
        cs = self.cache_states_per_file[fi]            # shifted: 0=empty, item x stored as x+1
        el = self.labels_per_file[fi]

        T = traces.shape[1]
        w = self.w

        # Sequence window ending at (and including) t.
        start = t - (w - 1)
        if start >= 0:
            window = traces[r, start : t + 1].astype(np.int64) + 1    # shift: items in [1, U]
        else:
            pad = np.zeros(-start, dtype=np.int64)                    # 0 = padding
            window = np.concatenate([pad, traces[r, : t + 1].astype(np.int64) + 1])
        assert window.shape[0] == w

        cache = cs[r, t].astype(np.int64)                             # already shifted
        label = int(el[r, t])

        return {
            "cache": torch.from_numpy(cache),
            "seq": torch.from_numpy(window),
            "label": torch.tensor(label, dtype=torch.long),
        }


def default_split(
    trace_files,
    cache_size: int = 32,
    context_window: int = 512,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
    cache_root: str | Path = "learned_eviction/belady_cache",
):
    """Split trace rows (not timesteps) into train/val/test per file.

    Splitting by trace, not by timestep, prevents train/test leakage across the
    same trace's history.
    """
    rng = np.random.default_rng(seed)
    per_file_train, per_file_val, per_file_test = [], [], []
    for p in trace_files:
        N = np.load(p, mmap_mode="r").shape[0]
        perm = rng.permutation(N).tolist()
        n_test = int(round(test_frac * N))
        n_val = int(round(val_frac * N))
        test_ids = perm[:n_test]
        val_ids = perm[n_test : n_test + n_val]
        train_ids = perm[n_test + n_val :]
        per_file_train.append(train_ids)
        per_file_val.append(val_ids)
        per_file_test.append(test_ids)

    def _make(indices):
        return EvictionDataset(
            trace_files,
            cache_size=cache_size,
            context_window=context_window,
            cache_root=cache_root,
            trace_indices=indices,
        )

    return _make(per_file_train), _make(per_file_val), _make(per_file_test)
