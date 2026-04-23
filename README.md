# Neural Algorithmic Decision Making

Transformers that learn classical online algorithms from exact oracle labels.

## Subprojects

### [`stopping/`](stopping/)

Causal transformer with a **2D chain-of-thought scratchpad** for online
decision problems — **optimal stopping** and **ski rental**. The model performs
backward induction in-place, one continuation value per scratchpad token, under
an attention mask that enforces the online posterior at every DP step. 2-layer,
2-head, ~20M params; trained on exact DP labels. Details in
[`stopping/README.md`](stopping/README.md) and [`stopping/experiment.tex`](stopping/experiment.tex).

### [`caching/`](caching/)

Transformer that learns the **Belady (furthest-in-future) eviction policy**
for caching. Each attention block has two heads — one keyed on the full cache
(k=32 slots, always visible), the other keyed on a sliding window of the
request sequence (so the model scales to T=16,000-long traces while only
attending over `k + w` tokens). Trained on eviction decisions extracted from
an oracle pass over the full trace. Generator for LRU/LFU/ARC-favoring
workloads lives alongside the model in
[`caching/`](caching/) under
[`caching/learned_eviction/`](caching/learned_eviction/).

## Shared idea

Both subprojects imitate an **oracle that sees information the model cannot**:
- Stopping/ski: the oracle solves the DP with knowledge of the full distribution.
- Caching: the oracle (Belady) sees the full future request sequence.

In both cases the learned model has access only to the online posterior and must
recover the oracle's decisions from partial information. 