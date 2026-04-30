"""
GPT-2-style decoder for Bayesian last-offer stopping (sec. 4 of research-notes).

Architecture (sec. 4.1, 4.2):
    L = 8 layers, d_emb = 128, M = 4 heads (head_dim = 32).
    Causal mask, learned absolute position embeddings, no dropout.
    Pre-norm GPT-2 block (LN -> attn / LN -> MLP-4x with GELU; residual).
    Affine input projection: scalar X_t -> R^d_emb.
    Single linear head, scalar output per position; meaning is set by
    the `supervision` flag (cv: continuation-value; act: accept logit).

Input / output scaling (v2). Because v2 varies sigma across regimes and X
values can be ~100 in magnitude on D_3, raw-scale training is unstable
(MSE on cv targets blows up by sigma^2). We bake two non-learnable
buffers into the model so that internal computation always operates at
unit scale but callers see raw X -> raw C_hat / logit:

    input_scale  = 1 / sigma          (applied before input projection)
    output_scale = sigma   if cv       (applied after the output head)
                 = 1.0     if act      (logit doesn't need rescaling)

Both buffers are saved/loaded with the checkpoint, so OOD evaluation at a
different sigma intentionally exposes the OOD scale gap — which is the
whole point of varying sigma in v2.

Forward signature:
    model(X, return_attn=False)
        X:              (B, n) float, RAW scale (caller-side units of X)
        returns C_hat:  (B, n) float, RAW scale
        if return_attn:
            also returns attn: (B, L, M, n, n) float — softmax weights per
            (layer, head); rows correspond to query positions, columns to
            key positions; lower-triangular by causal mask.

The training objective uses positions 0..n-2 (= 1-indexed t = 1..n-1); the
output at position n-1 is unused (forced-acceptance step has no label).
"""

import math
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_emb: int, n_heads: int, n_max: int):
        super().__init__()
        if d_emb % n_heads != 0:
            raise ValueError("d_emb must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_emb // n_heads
        self.qkv = nn.Linear(d_emb, 3 * d_emb)
        self.out = nn.Linear(d_emb, d_emb)
        # Causal mask (upper triangle blocked); shape (1, 1, n_max, n_max).
        mask = torch.tril(torch.ones(n_max, n_max, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, n_max, n_max), persistent=False)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, n, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        q = q.view(B, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, n, self.n_heads, self.head_dim).transpose(1, 2)
        if return_attn:
            # Manual path: surface the (B, M, n, n) softmax weights for analysis.
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~self.mask[:, :, :n, :n], float("-inf"))
            attn = F.softmax(scores, dim=-1)                              # (B, M, n, n)
            y = attn @ v
            y = y.transpose(1, 2).contiguous().view(B, n, d)
            return self.out(y), attn
        # Training/inference path: fused causal attention (much faster).
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)       # (B, M, n, hd)
        y = y.transpose(1, 2).contiguous().view(B, n, d)
        return self.out(y)


class MLP(nn.Module):
    def __init__(self, d_emb: int):
        super().__init__()
        self.fc1 = nn.Linear(d_emb, 4 * d_emb)
        self.fc2 = nn.Linear(4 * d_emb, d_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_emb: int, n_heads: int, n_max: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_emb)
        self.attn = CausalSelfAttention(d_emb, n_heads, n_max)
        self.ln2 = nn.LayerNorm(d_emb)
        self.mlp = MLP(d_emb)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        if return_attn:
            attn_out, attn = self.attn(self.ln1(x), return_attn=True)
            x = x + attn_out
            x = x + self.mlp(self.ln2(x))
            return x, attn
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class GPTStopper(nn.Module):
    """GPT-2 decoder with a single scalar output head per position.

    `supervision` selects how that scalar is interpreted:
      - "cv":  predicted continuation value hat C_hat_t  (MSE target)
      - "act": logit for the accept-action; sigmoid(>0.5) ⟺ logit>0  (BCE-with-logits target)

    Architecture is identical for both — one linear head named `cv_head`.
    At policy time, the eval scripts dispatch on supervision; the
    architecture itself is supervision-agnostic.

    Input/output scaling (v2). Pass `sigma` at construction. The model
    internally normalizes inputs by 1/sigma and (for cv) rescales outputs
    by sigma, so callers always see raw X -> raw C_hat / logit. The two
    scale factors are non-learnable buffers; they serialize with the
    checkpoint and OOD eval sees the training-scale buffers regardless
    of the eval distribution's sigma.
    """

    def __init__(
        self,
        n: int = 64,
        d_emb: int = 128,
        n_layers: int = 8,
        n_heads: int = 4,
        supervision: str = "cv",
        sigma: float = 1.0,
    ):
        super().__init__()
        if supervision not in ("cv", "act"):
            raise ValueError(f"unknown supervision: {supervision!r}")
        self.n = n
        self.d_emb = d_emb
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.supervision = supervision
        self.sigma = float(sigma)

        # Non-learnable scale buffers: input always normalized by 1/sigma;
        # cv output rescaled by sigma so the head outputs raw-scale C_hat,
        # act output unscaled (logit > 0 is scale-invariant).
        self.register_buffer(
            "input_scale", torch.tensor(1.0 / float(sigma)), persistent=True,
        )
        self.register_buffer(
            "output_scale",
            torch.tensor(float(sigma) if supervision == "cv" else 1.0),
            persistent=True,
        )

        self.input_proj = nn.Linear(1, d_emb)
        self.pos_emb = nn.Embedding(n, d_emb)
        self.blocks = nn.ModuleList(
            [Block(d_emb, n_heads, n) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_emb)
        self.cv_head = nn.Linear(d_emb, 1)                     # scalar output

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, X: torch.Tensor, return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, n = X.shape
        pos = torch.arange(n, device=X.device)
        # Internal-scale input: divide by sigma so the transformer always
        # sees roughly unit-magnitude values regardless of training regime.
        X_normalized = X * self.input_scale
        x = self.input_proj(X_normalized.unsqueeze(-1)) + self.pos_emb(pos)  # (B, n, d_emb)

        if return_attn:
            attns = []
            for block in self.blocks:
                x, a = block(x, return_attn=True)
                attns.append(a)
            x = self.ln_f(x)
            C_hat = self.cv_head(x).squeeze(-1) * self.output_scale       # (B, n) raw scale
            attn = torch.stack(attns, dim=1)                              # (B, L, M, n, n)
            return C_hat, attn

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.cv_head(x).squeeze(-1) * self.output_scale            # (B, n) raw scale


# ---------------------------------------------------------------------------
# Policy extraction (deployment-aligned reservation rule)
# ---------------------------------------------------------------------------

@torch.no_grad()
def model_policy_batch(
    model: GPTStopper, X: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized model policy. Dispatches on `model.supervision`.

    cv:  accept iff X[b, t] >= predicted C_hat[b, t]
    act: accept iff sigmoid(logit[b, t]) > 0.5  ⟺  logit[b, t] > 0

    Args:  X: (B, n) float tensor on the model's device.
    Returns: (stop_idx (B,), payoff (B,)).
    """
    B, n = X.shape
    out = model(X)                                                        # (B, n)
    if getattr(model, "supervision", "cv") == "act":
        accept = out[:, : n - 1] > 0.0                                    # logits > 0
    else:
        accept = X[:, : n - 1] >= out[:, : n - 1]                         # cv threshold rule
    any_accept = accept.any(dim=1)
    first_idx = accept.float().argmax(dim=1)                              # 0 if all-False
    last = torch.full_like(first_idx, n - 1)
    stop_idx = torch.where(any_accept, first_idx, last)
    payoff = X.gather(1, stop_idx.unsqueeze(1)).squeeze(1)
    return stop_idx, payoff


def model_policy(model: GPTStopper, X_seq, device) -> int:
    """1-indexed stopping time for a single sequence (numpy or torch input)."""
    X = torch.as_tensor(X_seq, device=device, dtype=torch.float32)
    if X.ndim == 1:
        X = X.unsqueeze(0)
    stop_idx, _ = model_policy_batch(model, X)
    return int(stop_idx[0].item()) + 1
