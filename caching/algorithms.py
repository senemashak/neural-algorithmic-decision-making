"""Cache simulators: LRU, LFU, ARC. Unit-size objects, return hit/miss per request."""

from collections import OrderedDict, defaultdict
import numpy as np


def simulate_lru(trace, k):
    cache = OrderedDict()
    hits = np.zeros(len(trace), dtype=np.uint8)
    for t, x in enumerate(trace):
        if x in cache:
            cache.move_to_end(x)
            hits[t] = 1
        else:
            cache[x] = True
            if len(cache) > k:
                cache.popitem(last=False)
    return hits


def simulate_lfu(trace, k):
    """Plain LFU with LRU tiebreak among min-frequency items."""
    freq = {}                                # x -> count
    buckets = defaultdict(OrderedDict)       # count -> OrderedDict of items (MRU at end)
    min_count = 0
    hits = np.zeros(len(trace), dtype=np.uint8)
    in_cache = set()

    for t, x in enumerate(trace):
        if x in in_cache:
            c = freq[x]
            del buckets[c][x]
            if not buckets[c]:
                del buckets[c]
                if min_count == c:
                    min_count = c + 1
            freq[x] = c + 1
            buckets[c + 1][x] = True
            hits[t] = 1
        else:
            if len(in_cache) >= k:
                # Evict LRU among min-count bucket
                evict_x, _ = buckets[min_count].popitem(last=False)
                if not buckets[min_count]:
                    del buckets[min_count]
                in_cache.discard(evict_x)
                del freq[evict_x]
            freq[x] = 1
            buckets[1][x] = True
            in_cache.add(x)
            min_count = 1
    return hits


class _ARC:
    """Standard ARC (Megiddo & Modha, 2003). Lists are OrderedDicts: LRU at front, MRU at end."""

    def __init__(self, c):
        self.c = c
        self.p = 0
        self.T1 = OrderedDict()
        self.T2 = OrderedDict()
        self.B1 = OrderedDict()
        self.B2 = OrderedDict()

    def _replace(self, x_in_B2):
        if self.T1 and (len(self.T1) > self.p or (x_in_B2 and len(self.T1) == self.p)):
            k, _ = self.T1.popitem(last=False)
            self.B1[k] = True
        else:
            k, _ = self.T2.popitem(last=False)
            self.B2[k] = True

    def access(self, x):
        c = self.c
        if x in self.T1:
            del self.T1[x]
            self.T2[x] = True
            return 1
        if x in self.T2:
            self.T2.move_to_end(x)
            return 1
        if x in self.B1:
            delta = max(len(self.B2) // max(len(self.B1), 1), 1)
            self.p = min(c, self.p + delta)
            self._replace(x_in_B2=False)
            del self.B1[x]
            self.T2[x] = True
            return 0
        if x in self.B2:
            delta = max(len(self.B1) // max(len(self.B2), 1), 1)
            self.p = max(0, self.p - delta)
            self._replace(x_in_B2=True)
            del self.B2[x]
            self.T2[x] = True
            return 0
        # Case IV: not in any list
        L1 = len(self.T1) + len(self.B1)
        L2 = len(self.T2) + len(self.B2)
        if L1 == c:
            if len(self.T1) < c:
                self.B1.popitem(last=False)
                self._replace(x_in_B2=False)
            else:
                self.T1.popitem(last=False)
        elif L1 < c and L1 + L2 >= c:
            if L1 + L2 == 2 * c:
                self.B2.popitem(last=False)
            self._replace(x_in_B2=False)
        self.T1[x] = True
        return 0


def simulate_arc(trace, k):
    arc = _ARC(k)
    hits = np.zeros(len(trace), dtype=np.uint8)
    for t, x in enumerate(trace):
        hits[t] = arc.access(int(x))
    return hits


ALGOS = {"LRU": simulate_lru, "LFU": simulate_lfu, "ARC": simulate_arc}


def hit_rates(trace, k, warmup_frac=0.1):
    """Return dict of algo -> hit rate after warmup prefix."""
    start = int(len(trace) * warmup_frac)
    out = {}
    for name, fn in ALGOS.items():
        h = fn(trace, k)
        out[name] = float(h[start:].mean())
    return out
