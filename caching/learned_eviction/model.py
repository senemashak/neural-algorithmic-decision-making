"""Dual-attention transformer for cache-eviction prediction.

The model sees:
  * the full cache         (k tokens, always visible)
  * a recent sequence window (w tokens, causal within the window)

Joint input length is k + w, independent of the full trace length T.

Each attention block has two heads:
  * Head A ("sequence"): K/V come from sequence tokens only; causal within the
    sequence block.
  * Head B ("cache"):    K/V come from cache tokens only.

Queries come from ALL tokens, for both heads. So a cache token's representation
after Head A is informed by the sequence, and a sequence token's representation
after Head B is informed by the cache. The two heads then mix through the
concat + output projection + FFN + residual stream. Stacking two layers gives
the heads a second round of interaction over the already-mixed residual stream.

Output: a logit per cache slot (shape (B, k)). Softmax over these is an
eviction distribution. The training target is the slot index Belady would evict.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualAttentionLayer(nn.Module):
    """One pre-norm transformer layer with one sequence-head and one cache-head."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be divisible by 2 for the two heads"
        self.d_model = d_model
        self.d_head = d_model // 2

        # Separate Q/K/V per head because the attended set differs between heads.
        self.q_seq = nn.Linear(d_model, self.d_head, bias=False)
        self.k_seq = nn.Linear(d_model, self.d_head, bias=False)
        self.v_seq = nn.Linear(d_model, self.d_head, bias=False)

        self.q_cache = nn.Linear(d_model, self.d_head, bias=False)
        self.k_cache = nn.Linear(d_model, self.d_head, bias=False)
        self.v_cache = nn.Linear(d_model, self.d_head, bias=False)

        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, k_cache: int, seq_causal_mask: torch.Tensor):
        """
        h:               (B, k_cache + w, d_model) — cache tokens then sequence tokens
        k_cache:         int, number of leading cache tokens
        seq_causal_mask: (w, w) additive causal mask for the sequence sub-block
        """
        B, L, _ = h.shape
        w = L - k_cache

        h_norm = self.norm1(h)

        # ── Head A: sequence K/V, causal within sequence ────────────────────
        q_a = self.q_seq(h_norm)                           # (B, L, dh)
        k_a = self.k_seq(h_norm[:, k_cache:])              # (B, w, dh)
        v_a = self.v_seq(h_norm[:, k_cache:])              # (B, w, dh)

        scores_a = (q_a @ k_a.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, L, w)
        # Build (L, w) additive mask: cache rows attend to all sequence positions
        # freely; sequence rows attend causally within the sequence block.
        mask_a = torch.zeros(L, w, device=h.device, dtype=scores_a.dtype)
        mask_a[k_cache:, :] = seq_causal_mask
        scores_a = scores_a + mask_a
        attn_a = F.softmax(scores_a, dim=-1)
        out_a = attn_a @ v_a                               # (B, L, dh)

        # ── Head B: cache K/V ──────────────────────────────────────────────
        q_b = self.q_cache(h_norm)
        k_b = self.k_cache(h_norm[:, :k_cache])
        v_b = self.v_cache(h_norm[:, :k_cache])
        scores_b = (q_b @ k_b.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, L, k)
        attn_b = F.softmax(scores_b, dim=-1)
        out_b = attn_b @ v_b

        attn_out = torch.cat([out_a, out_b], dim=-1)       # (B, L, d_model)
        h = h + self.dropout(self.out_proj(attn_out))
        h = h + self.dropout(self.ff(self.norm2(h)))
        return h


class CacheEvictionTransformer(nn.Module):
    """Predict which of k cache slots to evict.

    Args:
        vocab_size:     number of distinct item ids + 1 (the +1 reserves id 0
                        for "empty slot"). For U=512 pass vocab_size=513.
        cache_size:     k
        context_window: w — how many of the most recent requests the model sees.
                        The current request is the LAST token of the window.
        d_model, d_ff, n_layers, dropout: standard transformer hyperparameters.
    """

    def __init__(
        self,
        vocab_size: int = 513,
        cache_size: int = 32,
        context_window: int = 1024,
        d_model: int = 128,
        d_ff: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.cache_size = cache_size
        self.context_window = context_window
        self.d_model = d_model

        self.item_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.cache_pos_embed = nn.Embedding(cache_size, d_model)
        self.seq_pos_embed = nn.Embedding(context_window, d_model)
        # Segment ids: 0 = cache token, 1 = sequence token.
        self.segment_embed = nn.Embedding(2, d_model)

        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [DualAttentionLayer(d_model, d_ff, dropout=dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.evict_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, cache: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """
        cache: (B, k) int64 — current cache contents, tokens in [0, vocab_size),
                               with 0 indicating an empty slot.
        seq:   (B, w) int64 — the w most recent requests including the current
                               one. seq[:, -1] is the item being requested now.

        Returns:
            logits: (B, k) — eviction scores per cache slot.
        """
        B, k = cache.shape
        _, w = seq.shape
        device = cache.device
        assert k == self.cache_size, f"cache size {k} != {self.cache_size}"
        assert w <= self.context_window, (
            f"sequence window {w} exceeds context_window {self.context_window}"
        )

        cache_pos = torch.arange(k, device=device).unsqueeze(0).expand(B, -1)
        seq_pos = torch.arange(w, device=device).unsqueeze(0).expand(B, -1)

        cache_emb = (
            self.item_embed(cache)
            + self.cache_pos_embed(cache_pos)
            + self.segment_embed(torch.zeros(B, k, dtype=torch.long, device=device))
        )
        seq_emb = (
            self.item_embed(seq)
            + self.seq_pos_embed(seq_pos)
            + self.segment_embed(torch.ones(B, w, dtype=torch.long, device=device))
        )

        h = self.drop(torch.cat([cache_emb, seq_emb], dim=1))

        seq_causal_mask = torch.triu(
            torch.full((w, w), float("-inf"), device=device), diagonal=1
        )

        for layer in self.layers:
            h = layer(h, k_cache=k, seq_causal_mask=seq_causal_mask)

        h = self.final_norm(h)
        cache_out = h[:, :k]                              # (B, k, d_model)
        logits = self.evict_head(cache_out).squeeze(-1)   # (B, k)
        return logits


if __name__ == "__main__":
    B, k, w = 4, 32, 128
    model = CacheEvictionTransformer(cache_size=k, context_window=w)
    cache = torch.randint(0, 513, (B, k))
    seq = torch.randint(1, 513, (B, w))
    logits = model(cache, seq)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"logits: {logits.shape}  params: {n_params:,}")
