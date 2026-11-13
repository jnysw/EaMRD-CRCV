"""InfoNCE contrastive loss with in-batch and hard-negative variants."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Claim–Evidence InfoNCE loss.

    In-batch mode:
        Positive pair: (c_i, e_i)  — same video's claim and supporting evidence
        Negatives:     all other e_j in the batch (j ≠ i)

    Hard-negative mode:
        Positive pair: (c_i, e_i⁺)
        Hard negative: (c_i, e_i⁻)  — contradiction evidence from Contradiction Query
        In-batch negatives also included
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature

    def forward(
        self,
        h_c: torch.Tensor,         # [B, D]  claim embeddings
        h_e_pos: torch.Tensor,     # [B, D]  positive evidence
        h_e_neg: torch.Tensor | None = None,  # [B, D] hard negatives (optional)
        valid_mask: torch.Tensor | None = None,  # [B] bool, True = sample has real positive evidence
    ) -> torch.Tensor:
        """
        Returns scalar InfoNCE loss.
        All inputs should be L2-normalised before being passed in.

        valid_mask=None keeps the original behaviour (every sample is an anchor and a key).
        Otherwise masked-out samples are removed both as anchors and as in-batch keys;
        with fewer than 2 valid samples the loss is 0.
        """
        if valid_mask is not None:
            if int(valid_mask.sum()) < 2:
                return h_c.sum() * 0.0
            h_c, h_e_pos = h_c[valid_mask], h_e_pos[valid_mask]
            if h_e_neg is not None:
                h_e_neg = h_e_neg[valid_mask]

        h_c = F.normalize(h_c, dim=-1)
        h_e_pos = F.normalize(h_e_pos, dim=-1)

        B = h_c.size(0)

        if h_e_neg is not None:
            h_e_neg = F.normalize(h_e_neg, dim=-1)
            # Keys: [positive, all in-batch positives, hard negatives]
            keys = torch.cat([h_e_pos, h_e_neg], dim=0)  # [2B, D]
        else:
            keys = h_e_pos  # [B, D]

        # similarity matrix: [B, len(keys)]
        logits = torch.matmul(h_c, keys.T) / self.tau

        # Positive indices are the diagonal (first B columns)
        labels = torch.arange(B, device=h_c.device)

        loss = F.cross_entropy(logits, labels)
        return loss


class ContrastiveLossFactory:
    """Build the appropriate loss variant from config."""

    @staticmethod
    def build(mode: str = "in_batch", temperature: float = 0.07) -> InfoNCELoss:
        assert mode in ("in_batch", "hard_neg"), f"Unknown mode: {mode}"
        return InfoNCELoss(temperature=temperature)
