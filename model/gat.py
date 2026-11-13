"""
ClaimGAT: Graph Attention Network over Claim Graph.

Input:
  x        [N_c, D]   — per-claim CLIP text embeddings
  adj      [N_c, N_c] — conflict-score adjacency (with self-loops, values in [0,1])
  mask     [N_c]      — True = valid node, False = padding

Output:
  h_graph  [D_out]    — video-level aggregated claim representation (mean over valid nodes)

Architecture:
  Layer 1: GATConv(D → hidden, heads=4, concat=True)  → [N_c, 4*hidden]
  Layer 2: GATConv(4*hidden → D_out, heads=1, concat=False) → [N_c, D_out]
  Pool:    masked mean-pool → [D_out]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATConv(nn.Module):
    """Single-head or multi-head graph attention convolution (pure PyTorch, no PyG dependency)."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 1, concat: bool = True,
                 dropout: float = 0.3, negative_slope: float = 0.2,
                 keep_self_loop: bool = False):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.concat = concat
        # True: attention dropout never drops the self-loop, so a node (and a
        # self-loop-only graph) can not collapse to an all-zero output.
        self.keep_self_loop = keep_self_loop

        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        self.leaky = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x:   [N, D_in]
        adj: [N, N]  adjacency weights (0 = no edge)
        returns [N, heads*out_dim] if concat else [N, out_dim]
        """
        N = x.size(0)
        Wh = self.W(x).view(N, self.heads, self.out_dim)   # [N, H, D]

        # Attention scores: e_ij = LeakyReLU(a_src · Wh_i + a_dst · Wh_j)
        src_score = (Wh * self.a_src).sum(dim=-1)   # [N, H]
        dst_score = (Wh * self.a_dst).sum(dim=-1)   # [N, H]
        # Broadcast: e[i,j,h] = src_score[i,h] + dst_score[j,h]
        e = self.leaky(src_score.unsqueeze(1) + dst_score.unsqueeze(0))  # [N, N, H]

        # Mask non-edges with -inf
        mask = (adj == 0).unsqueeze(-1).expand_as(e)  # [N, N, H]
        e = e.masked_fill(mask, float("-inf"))

        # Edge weight modulation: scale by adjacency value
        adj_w = adj.unsqueeze(-1).expand_as(e)   # [N, N, H]
        e = e + (adj_w.clamp(min=1e-6).log())     # bias by log(adj_weight)

        alpha = F.softmax(e, dim=1)    # [N, N, H]  — normalize over neighbors
        # Replace NaN (all-masked rows) with 0
        alpha = torch.nan_to_num(alpha, nan=0.0)
        if self.keep_self_loop:
            eye = torch.eye(N, dtype=torch.bool, device=alpha.device).unsqueeze(-1)
            alpha = torch.where(eye, alpha, self.dropout(alpha))
        else:
            alpha = self.dropout(alpha)

        # Aggregate: h_i = sum_j alpha_ij * Wh_j
        # [N, N, H, D] x alpha [N, N, H] → [N, H, D]
        out = (alpha.unsqueeze(-1) * Wh.unsqueeze(0)).sum(dim=1)  # [N, H, D]

        if self.concat:
            return out.view(N, self.heads * self.out_dim)   # [N, H*D]
        else:
            return out.mean(dim=1)                          # [N, D]


class ClaimGAT(nn.Module):
    """2-layer GAT for claim graph → video-level representation."""

    def __init__(self, in_dim: int = 768, hidden_dim: int = 128, out_dim: int = 256,
                 heads: int = 4, dropout: float = 0.3, keep_self_loop: bool = False):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=dropout,
                            keep_self_loop=keep_self_loop)
        self.norm1 = nn.LayerNorm(heads * hidden_dim)
        self.gat2 = GATConv(heads * hidden_dim, out_dim, heads=1, concat=False, dropout=dropout,
                            keep_self_loop=keep_self_loop)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x:    [N_c, D_in]
        adj:  [N_c, N_c]
        mask: [N_c] bool — True = valid

        returns: [D_out]
        """
        h = self.gat1(x, adj)        # [N_c, 4*hidden]
        h = F.gelu(self.norm1(h))
        h = self.dropout(h)
        h = self.gat2(h, adj)        # [N_c, out_dim]
        h = self.norm2(h)

        # Masked mean pool
        if mask is not None:
            h = h * mask.unsqueeze(-1).float()
            return h.sum(0) / mask.float().sum().clamp(min=1)
        return h.mean(0)             # [out_dim]
