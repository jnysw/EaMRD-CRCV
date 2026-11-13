"""
RelationAwareFusion (v3 - parameter-light)

Relation-aware replacement for the final `torch.cat([pv, pc, pe, pr])` fusion
in EaMRDCRCV.

Why parameter-light (lesson from earlier heavy versions):
    The upstream (GAT + CECA + filtered evidence) already over-fits hard
    (train Acc -> 0.99 within ~8 epochs). Every heavy fusion head we tried
    (256-d bottleneck MLP, 4D residual MLP) *added free parameters* and made the
    single-seed result worse than plain concat. In this regime the winning move
    is NOT "a smarter, bigger fusion head" but injecting relational *inductive
    bias* with (almost) zero new parameters, so we test whether the relation
    *information* helps without giving the model new ways to over-fit.

Two switchable, parameter-light mechanisms:

    S1  Scalar reliability gate  (use_evidence_gate=True)     +2 params
        Reliability of evidence is driven by claim<->evidence agreement:
            g = sigmoid(a * cos(pc, pe) + b)          # a, b are scalars
            pe' = g * pe                              # scalar scales evidence
        Init a=1, b=4 -> g ~= 0.98 at start (=> ~= keep evidence => ~= concat),
        and the gate can learn to *down-weight* evidence that disagrees with the
        claim. Fully interpretable (cos vs gate).

    S2  Parameter-free interaction augmentation  (use_ce_interaction=True)
        Append the classic element-wise matching features (as in ESIM/NLI):
            [ pc * pe , |pc - pe| ]
        straight into the final concat. Zero fusion parameters; only the
        classifier's (Lazy) first layer grows a bit.

    S3  = S1 + S2 (both flags True).
    All flags off  ->  h_final == cat([pv, pc, pe, pr])  (exactly concat).

Conflict is intentionally out of scope: the refuting-evidence branch (`pe_neg`)
is NOT consumed here; it stays offline via `h_r` and the InfoNCE hard-negative.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationAwareFusion(nn.Module):
    """
    Inputs (all [B, dim]):
        pv : video       (proj_v(h_v))
        pc : claim       (ClaimGAT / proj_c)
        pe : evidence    (CECA / proj_e)
        pr : reasoning   (proj_r(h_r))

    Returns:
        h_final     : [B, W]  (W = 4*dim, or 6*dim when interaction is on)
        fusion_info : dict with "evidence_gate" (scalar per sample) or None
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,          # unused; kept for signature compat
        dropout: float = 0.1,           # unused; kept for signature compat
        use_ce_interaction: bool = True,   # S2: parameter-free interaction
        use_evidence_gate: bool = True,    # S1: scalar reliability gate
        use_reasoning_gate: bool = False,  # deprecated in light design; ignored
    ):
        super().__init__()

        self.dim = dim
        self.use_ce_interaction = use_ce_interaction
        self.use_evidence_gate = use_evidence_gate

        # S1: two scalar parameters only.
        # a=1, b=4 -> sigmoid(cos + 4) in ~[0.95, 0.99] at init (~= keep evidence)
        if use_evidence_gate:
            self.gate_a = nn.Parameter(torch.tensor(1.0))
            self.gate_b = nn.Parameter(torch.tensor(4.0))

    def forward(
        self,
        pv: torch.Tensor,
        pc: torch.Tensor,
        pe: torch.Tensor,
        pr: torch.Tensor,
    ):
        # ----------------------------------------------------
        # S1: scalar reliability gate on evidence
        # ----------------------------------------------------
        if self.use_evidence_gate:
            cos = F.cosine_similarity(pc, pe, dim=-1, eps=1e-8)   # [B]
            g = torch.sigmoid(self.gate_a * cos + self.gate_b)    # [B] in (0,1)
            pe_used = g.unsqueeze(-1) * pe
            evidence_gate = g
        else:
            pe_used = pe
            evidence_gate = None

        # ----------------------------------------------------
        # Base concat (identical to the concat baseline when all off)
        # ----------------------------------------------------
        parts = [pv, pc, pe_used, pr]

        # ----------------------------------------------------
        # S2: parameter-free interaction augmentation
        # ----------------------------------------------------
        if self.use_ce_interaction:
            parts.append(pc * pe)
            parts.append(torch.abs(pc - pe))

        h_final = torch.cat(parts, dim=-1)

        fusion_info = {"evidence_gate": evidence_gate}

        return h_final, fusion_info
