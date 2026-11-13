"""
CECA: Claim-Evidence Cross-Attention module.

Each claim attends over evidence items to produce a claim-conditioned
evidence representation, then the attended representations are aggregated
to video level.

Input:
  claims_seq   [N_c, D]   — per-claim CLIP embeddings
  evidence_seq [K, D]     — per-evidence CLIP embeddings (K = top-K, fixed)
  claim_mask   [N_c] bool — True = valid claim (False = padding)

Output:
  h_ce [D]  — video-level claim-conditioned evidence representation

Architecture:
  Q = proj_q(claims_seq)   → [N_c, D]
  K = proj_k(evidence_seq) → [K, D]
  V = proj_v(evidence_seq) → [K, D]

  For each claim i:
    alpha_i = softmax(Q_i · K^T / sqrt(D))  → [K]
    ctx_i   = alpha_i · V                   → [D]

  h_ce = masked_mean(ctx_1, ..., ctx_{N_c}) → [D]
  + residual: h_ce += evidence_seq.mean(0)   (skip connection)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CECA(nn.Module):
    """Claim-Evidence Cross-Attention (single-head for simplicity; multi-head via num_heads)."""

    def __init__(self, dim: int = 768, num_heads: int = 4, dropout: float = 0.1,
                 mask_padding: bool = False):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        # True: all-zero (padding) evidence rows are excluded from the attention
        # keys and from the ev_mean skip connection.
        self.mask_padding = mask_padding
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.proj_q = nn.Linear(dim, dim, bias=False)
        self.proj_k = nn.Linear(dim, dim, bias=False)
        self.proj_v = nn.Linear(dim, dim, bias=False)
        self.proj_out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, claims_seq: torch.Tensor, evidence_seq: torch.Tensor,
                claim_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        claims_seq:   [N_c, D]
        evidence_seq: [K, D]
        claim_mask:   [N_c] bool

        returns: [D]
        """
        N_c, D = claims_seq.shape
        K = evidence_seq.size(0)
        H, Dh = self.num_heads, self.head_dim

        # Evidence skip (for residual)
        ev_valid = evidence_seq.abs().sum(-1) > 1e-8      # [K]
        if self.mask_padding and ev_valid.any():
            ev_mean = evidence_seq[ev_valid].mean(0)      # [D]
        else:
            ev_mean = evidence_seq.mean(0)  # [D]

        # Handle degenerate case: all-zero evidence
        if evidence_seq.abs().sum() < 1e-6:
            return ev_mean

        Q = self.proj_q(claims_seq).view(N_c, H, Dh)      # [N_c, H, Dh]
        K_ = self.proj_k(evidence_seq).view(K, H, Dh)     # [K,   H, Dh]
        V  = self.proj_v(evidence_seq).view(K, H, Dh)     # [K,   H, Dh]

        # Attention: [N_c, H, K]
        attn = torch.einsum("nhd,khd->nhk", Q, K_) / self.scale
        if self.mask_padding:
            attn = attn.masked_fill(~ev_valid.view(1, 1, K), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Context: [N_c, H, Dh] → [N_c, D]
        ctx = torch.einsum("nhk,khd->nhd", attn, V).reshape(N_c, D)
        ctx = self.proj_out(ctx)

        # Residual + LayerNorm
        ctx = self.norm(ctx + claims_seq)

        # Masked mean pool over claims
        if claim_mask is not None:
            valid = claim_mask.float().unsqueeze(-1)   # [N_c, 1]
            h_ce = (ctx * valid).sum(0) / valid.sum().clamp(min=1)
        else:
            h_ce = ctx.mean(0)   # [D]

        # Evidence skip connection
        return h_ce + ev_mean
