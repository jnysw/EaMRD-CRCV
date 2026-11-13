"""2-layer MLP binary classification head."""

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    Input dimension is determined lazily via LazyLinear.
    h_final = [h_v_pooled | h_c_pooled | h_e_pooled | h_r]  →  [B, 768*3+768]
    """

    def __init__(self, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D_in] → logits: [B, 2]"""
        return self.net(x)
