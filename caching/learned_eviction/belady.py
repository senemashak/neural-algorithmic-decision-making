"""Belady (furthest-in-future) oracle simulator.

Given a trace and cache size k, simulate Belady's optimal policy and record,
at every timestep, the cache state, the true eviction decisions (at full-cache
misses), and a hypothetical eviction label (the slot Belady would evict if
forced to evict right now, defined at every timestep where the cache is full).

Items in the raw trace are ints in [0, U-1]. We shift by +1 so that 0 can
serve as a sentinel "empty slot" token in the model's vocabulary.
"""

from __future__ import annotations

import numpy as np


def simulate_belady(trace: np.ndarray, k: int):
    """Run Belady's optimal eviction on a single trace.

    Args:
        trace: (T,) int array of item ids in [0, U-1]
        k:     cache capacity

    Returns:
        cache_states:       (T, k) int64 — cache contents at the START of step t,
                            shifted by +1 (0 = empty slot, item x stored as x+1).
                            The t-th row is what the model sees BEFORE processing
                            trace[t].
        miss_full_mask:     (T,) bool — step t is a miss AND the cache was full
                            (i.e. an eviction was required).
        eviction_labels:    (T,) int64 — slot index in [0, k) evicted at step t,
                            or -1 if no eviction happened.
        full_cache_labels:  (T,) int64 — slot index in [0, k) that Belady would
                            evict IF forced to evict at step t. Defined at every
                            step where the cache is full (regardless of whether
                            trace[t] is a hit or a miss); -1 when the cache is
                            not full. Computed as argmax over cache_next, so on
                            hit steps the slot holding trace[t] (whose
                            next-occurrence equals t) can never be the argmax
                            for k>=2.
    """
    T = len(trace)
    trace = np.asarray(trace, dtype=np.int64)

    # next_at[i] = next index j > i with trace[j] == trace[i], or T if none.
    next_at = np.full(T, T, dtype=np.int64)
    last_seen: dict[int, int] = {}
    for i in range(T - 1, -1, -1):
        x = int(trace[i])
        if x in last_seen:
            next_at[i] = last_seen[x]
        last_seen[x] = i

    cache_items = [-1] * k     # -1 = empty; otherwise raw item id
    cache_next = [T] * k       # next occurrence of that cached item (after now)
    n_filled = 0               # count of non-empty slots

    cache_states = np.zeros((T, k), dtype=np.int64)
    miss_full_mask = np.zeros(T, dtype=bool)
    eviction_labels = np.full(T, -1, dtype=np.int64)
    full_cache_labels = np.full(T, -1, dtype=np.int64)

    for t in range(T):
        # Snapshot state at the start of step t.
        for i in range(k):
            cache_states[t, i] = cache_items[i] + 1 if cache_items[i] >= 0 else 0

        # Hypothetical-eviction label: argmax of cache_next over slots,
        # defined only when all slots are filled.
        if n_filled == k:
            best_slot = 0
            best_next = cache_next[0]
            for s in range(1, k):
                if cache_next[s] > best_next:
                    best_next = cache_next[s]
                    best_slot = s
            full_cache_labels[t] = best_slot

        x = int(trace[t])

        try:
            idx = cache_items.index(x)
            # Hit: refresh next-occurrence pointer for this slot.
            cache_next[idx] = int(next_at[t])
            continue
        except ValueError:
            pass

        # Miss.
        try:
            empty = cache_items.index(-1)
            cache_items[empty] = x
            cache_next[empty] = int(next_at[t])
            n_filled += 1
        except ValueError:
            # Cache full: evict the slot whose next occurrence is furthest away.
            evict_idx = int(np.argmax(cache_next))
            miss_full_mask[t] = True
            eviction_labels[t] = evict_idx
            cache_items[evict_idx] = x
            cache_next[evict_idx] = int(next_at[t])

    return cache_states, miss_full_mask, eviction_labels, full_cache_labels


def simulate_belady_batch(traces: np.ndarray, k: int):
    """Run Belady on each row of a (N, T) trace array.

    Returns arrays with a leading N dim: cache_states (N, T, k),
    miss_full_mask (N, T), eviction_labels (N, T), full_cache_labels (N, T).
    """
    N, T = traces.shape
    cache_states = np.zeros((N, T, k), dtype=np.int64)
    miss_full_mask = np.zeros((N, T), dtype=bool)
    eviction_labels = np.full((N, T), -1, dtype=np.int64)
    full_cache_labels = np.full((N, T), -1, dtype=np.int64)
    for n in range(N):
        cs, mm, el, fcl = simulate_belady(traces[n], k)
        cache_states[n] = cs
        miss_full_mask[n] = mm
        eviction_labels[n] = el
        full_cache_labels[n] = fcl
    return cache_states, miss_full_mask, eviction_labels, full_cache_labels


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    trace = rng.integers(0, 10, size=50)
    cs, mm, el, fcl = simulate_belady(trace, k=3)
    print("trace :", trace)
    print("miss_full:", mm.astype(int))
    print("eviction labels (event-only):", el)
    print("full-cache labels (all ts):  ", fcl)
    print("first full-cache step:", int(np.nonzero(fcl >= 0)[0][0]))
