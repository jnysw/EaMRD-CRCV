"""
EaMRD-CRCV P1
Evidence-aware Multimodal Rumor Detection with
Claim-centric Retrieval, Credibility Modeling and Verification.

P1 modification:
    Evidence Credibility Modeling

Core idea:
    Instead of treating retrieved evidence equally, learn a
    credibility score for each evidence item conditioned on the claim.

    credibility =
        relevance
        + consistency
        + optional authority metadata

Important:
    P0 Adaptive Gate / Claim Auxiliary Loss / Branch Dropout
    are NOT used in this version.

Original fusion is preserved:

    h_final =
        concat(proj_v,
               h_c_out,
               h_e_pos_out,
               proj_r)

    -> 2-layer MLP
    -> logits

P1 only changes how evidence is aggregated before fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import MLPClassifier
from .contrastive_loss import InfoNCELoss
from .gat import ClaimGAT
from .ceca import CECA
from .relation_fusion import RelationAwareFusion


class _Proj(nn.Module):
    """
    Simple projection:
        LazyLinear -> LayerNorm -> GELU
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EvidenceCredibility(nn.Module):
    """
    P1 Evidence Credibility Modeling.

    Input:
        claim_seq:
            [B, N, D]

        evidence_seq:
            [B, K, D]

        evidence_cred_seq (optional):
            [B, K, 3]

            [:, :, 0] = authority
            [:, :, 1] = relevance
            [:, :, 2] = consistency

    Output:
        weighted evidence:
            [B, N, D]

        credibility weights:
            [B, N, K]

    Design:
        Each evidence item is scored conditioned on each claim.

        score(c_i, e_j)
            = learned interaction
            + optional credibility metadata

    This module does NOT change the final multimodal fusion.
    """

    def __init__(
        self,
        dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_metadata: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.use_metadata = use_metadata

        self.claim_proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.evidence_proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Learned semantic interaction:
        # [claim, evidence, claim * evidence]
        self.interaction = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Optional credibility metadata:
        # authority / relevance / consistency
        if use_metadata:
            self.metadata_proj = nn.Sequential(
                nn.Linear(3, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )

    def forward(
        self,
        claim_seq: torch.Tensor,
        evidence_seq: torch.Tensor,
        claim_mask: torch.Tensor | None = None,
        evidence_cred_seq: torch.Tensor | None = None,
    ):
        """
        Args:
            claim_seq:
                [B, N, D]

            evidence_seq:
                [B, K, D]

            claim_mask:
                [B, N]

            evidence_cred_seq:
                [B, K, 3], optional

        Returns:
            evidence_out:
                [B, N, D]

            weights:
                [B, N, K]
        """

        B, N, D = claim_seq.shape
        K = evidence_seq.size(1)

        # ----------------------------------------------------
        # Project
        # ----------------------------------------------------

        c = self.claim_proj(
            claim_seq
        )                       # [B, N, H]

        e = self.evidence_proj(
            evidence_seq
        )                       # [B, K, H]

        # ----------------------------------------------------
        # Pairwise interaction
        # ----------------------------------------------------

        c_expand = c.unsqueeze(2).expand(
            -1,
            -1,
            K,
            -1,
        )

        e_expand = e.unsqueeze(1).expand(
            -1,
            N,
            -1,
            -1,
        )

        interaction = torch.cat(
            [
                c_expand,
                e_expand,
                c_expand * e_expand,
            ],
            dim=-1,
        )

        scores = self.interaction(
            interaction
        ).squeeze(-1)             # [B, N, K]

        # ----------------------------------------------------
        # Optional credibility metadata
        # ----------------------------------------------------

        if (
            self.use_metadata
            and evidence_cred_seq is not None
        ):

            metadata_score = self.metadata_proj(
                evidence_cred_seq
            ).squeeze(-1)         # [B, K]

            scores = scores + metadata_score.unsqueeze(1)

        # ----------------------------------------------------
        # Mask invalid claims
        # ----------------------------------------------------

        if claim_mask is not None:

            invalid_claim = ~claim_mask

            scores = scores.masked_fill(
                invalid_claim.unsqueeze(-1),
                0.0,
            )

        # ----------------------------------------------------
        # Evidence credibility distribution
        # ----------------------------------------------------

        weights = torch.softmax(
            scores,
            dim=-1,
        )                          # [B, N, K]

        # ----------------------------------------------------
        # Weighted evidence aggregation
        # ----------------------------------------------------

        evidence_out = torch.bmm(
            weights,
            evidence_seq,
        )                           # [B, N, D]

        # ----------------------------------------------------
        # Mask invalid claims
        # ----------------------------------------------------

        if claim_mask is not None:

            evidence_out = evidence_out * (
                claim_mask
                .unsqueeze(-1)
                .float()
            )

        return evidence_out, weights


class EaMRDCRCV(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.5,
        temperature: float = 0.05,
        lambda_contrastive: float = 0.1,
        use_contrastive: bool = True,
        contrastive_mode: str = "in_batch",

        # ----------------------------------------------------
        # GAT
        # ----------------------------------------------------

        use_gat: bool = False,
        gat_hidden: int = 128,
        gat_heads: int = 4,
        gat_dropout: float = 0.3,

        # ----------------------------------------------------
        # CECA
        # ----------------------------------------------------

        use_ceca: bool = False,
        ceca_heads: int = 4,
        ceca_dropout: float = 0.1,

        # ----------------------------------------------------
        # P1
        # ----------------------------------------------------

        use_credibility: bool = False,
        credibility_hidden: int = 256,
        credibility_dropout: float = 0.1,
        use_credibility_metadata: bool = True,

        # ----------------------------------------------------
        # Ablation
        # ----------------------------------------------------

        use_claim: bool = True,
        use_evidence: bool = True,
        use_reasoning: bool = True,
        use_conflict: bool = True,

        # ----------------------------------------------------
        # Final fusion
        # ----------------------------------------------------

        fusion_type: str = "concat",
        use_ce_interaction: bool = True,
        use_evidence_gate: bool = True,
        use_reasoning_gate: bool = True,
        fusion_dropout: float = 0.1,
        return_fusion_gates: bool = False,

        # ----------------------------------------------------
        # E1 missing-evidence / numerical fixes (all off = original Full)
        # ----------------------------------------------------

        contrastive_valid_mask: bool = False,
        gat_keep_self_loop: bool = False,
        ceca_mask_padding: bool = False,
    ):

        super().__init__()

        self.name = "EaMRDCRCV"

        self.lambda_contrastive = (
            lambda_contrastive
        )

        self.use_contrastive = (
            use_contrastive
        )

        self.contrastive_mode = (
            contrastive_mode
        )

        self.use_gat = use_gat
        self.use_ceca = use_ceca

        # P1
        self.use_credibility = (
            use_credibility
        )

        self.use_claim = use_claim
        self.use_evidence = use_evidence
        self.use_reasoning = use_reasoning
        self.use_conflict = use_conflict

        # Final fusion
        self.fusion_type = fusion_type
        self.return_fusion_gates = return_fusion_gates

        # E1: InfoNCE only on samples with real support evidence and a non-zero pc
        self.contrastive_valid_mask = contrastive_valid_mask

        # ----------------------------------------------------
        # Projection heads
        # ----------------------------------------------------

        self.proj_v = _Proj(
            hidden_dim
        )

        self.proj_r = _Proj(
            hidden_dim
        )

        self.proj_c = _Proj(
            hidden_dim
        )

        self.proj_e = _Proj(
            hidden_dim
        )

        # ----------------------------------------------------
        # GAT
        # ----------------------------------------------------

        if use_gat:

            self.claim_gat = ClaimGAT(
                in_dim=768,
                hidden_dim=gat_hidden,
                out_dim=hidden_dim,
                heads=gat_heads,
                dropout=gat_dropout,
                keep_self_loop=gat_keep_self_loop,
            )

        # ----------------------------------------------------
        # CECA
        # ----------------------------------------------------

        if use_ceca:

            self.ceca_pos = CECA(
                dim=768,
                num_heads=ceca_heads,
                dropout=ceca_dropout,
                mask_padding=ceca_mask_padding,
            )

            self.ceca_neg = CECA(
                dim=768,
                num_heads=ceca_heads,
                dropout=ceca_dropout,
                mask_padding=ceca_mask_padding,
            )

            self.proj_ce_pos = _Proj(
                hidden_dim
            )

            self.proj_ce_neg = _Proj(
                hidden_dim
            )

        # ----------------------------------------------------
        # P1 Evidence Credibility
        # ----------------------------------------------------

        if use_credibility:

            self.credibility = EvidenceCredibility(
                dim=768,
                hidden_dim=credibility_hidden,
                dropout=credibility_dropout,
                use_metadata=use_credibility_metadata,
            )

        # ----------------------------------------------------
        # Relation-aware final fusion (optional)
        # ----------------------------------------------------

        if fusion_type == "relation_aware":

            self.fusion = RelationAwareFusion(
                dim=hidden_dim,
                hidden_dim=hidden_dim,
                dropout=fusion_dropout,
                use_ce_interaction=use_ce_interaction,
                use_evidence_gate=use_evidence_gate,
                use_reasoning_gate=use_reasoning_gate,
            )

        # ----------------------------------------------------
        # Original classifier
        # ----------------------------------------------------

        self.classifier = MLPClassifier(
            hidden_dim=hidden_dim * 4,
            dropout=dropout,
        )

        self.contrastive_loss = InfoNCELoss(
            temperature=temperature
        )

    # ========================================================
    # Claims
    # ========================================================

    def _encode_claims(
        self,
        h_c: torch.Tensor,
        h_c_seq: torch.Tensor | None,
        graph_adj: torch.Tensor | None,
        claim_mask: torch.Tensor | None,
    ) -> torch.Tensor:

        if (
            self.use_gat
            and h_c_seq is not None
        ):

            B = h_c.size(0)

            out = []

            for i in range(B):

                n = (
                    claim_mask[i].sum().item()
                    if claim_mask is not None
                    else h_c_seq.size(1)
                )

                n = max(
                    int(n),
                    1,
                )

                xi = h_c_seq[
                    i,
                    :n
                ]

                adji = graph_adj[
                    i,
                    :n,
                    :n
                ]

                mi = (
                    claim_mask[i, :n]
                    if claim_mask is not None
                    else None
                )

                out.append(
                    self.claim_gat(
                        xi,
                        adji,
                        mi,
                    )
                )

            return torch.stack(
                out,
                dim=0,
            )

        return self.proj_c(
            h_c
        )

    # ========================================================
    # Evidence
    # ========================================================

    def _encode_evidence(
        self,
        h_e_pos: torch.Tensor,
        h_e_neg: torch.Tensor,
        h_c_seq: torch.Tensor | None,
        h_e_pos_seq: torch.Tensor | None,
        h_e_neg_seq: torch.Tensor | None,
        claim_mask: torch.Tensor | None,
        h_e_pos_cred_seq: torch.Tensor | None = None,
        h_e_neg_cred_seq: torch.Tensor | None = None,
    ):

        # ----------------------------------------------------
        # Original mean-pooled / fallback path
        # ----------------------------------------------------

        if not self.use_ceca:

            pe = self.proj_e(
                h_e_pos
            )

            pne = self.proj_e(
                h_e_neg
            )

            return (
                pe,
                pne,
                None,
                None,
            )

        # ----------------------------------------------------
        # CECA
        # ----------------------------------------------------

        if (
            h_c_seq is None
            or h_e_pos_seq is None
        ):

            pe = self.proj_e(
                h_e_pos
            )

            pne = self.proj_e(
                h_e_neg
            )

            return (
                pe,
                pne,
                None,
                None,
            )

        B = h_e_pos.size(0)

        pos_out = []
        neg_out = []

        pos_weights = []
        neg_weights = []

        for i in range(B):

            n = (
                claim_mask[i].sum().item()
                if claim_mask is not None
                else h_c_seq.size(1)
            )

            n = max(
                int(n),
                1,
            )

            ci = h_c_seq[
                i,
                :n
            ]

            mi = (
                claim_mask[i, :n]
                if claim_mask is not None
                else None
            )

            # ------------------------------------------------
            # P1:
            # credibility-weighted evidence
            # ------------------------------------------------

            if self.use_credibility:

                pos_e = h_e_pos_seq[i]

                pos_cred, pos_w = (
                    self.credibility(
                        ci.unsqueeze(0),
                        pos_e.unsqueeze(0),
                        (
                            mi.unsqueeze(0)
                            if mi is not None
                            else None
                        ),
                        (
                            h_e_pos_cred_seq[i]
                            .unsqueeze(0)
                            if h_e_pos_cred_seq is not None
                            else None
                        ),
                    )
                )

                pos_e = pos_cred.squeeze(0)

                pos_weights.append(
                    pos_w.squeeze(0)
                )

            else:

                pos_e = h_e_pos_seq[i]

            # ------------------------------------------------
            # CECA: Claim <-> Evidence
            # ------------------------------------------------

            pos_out.append(
                self.ceca_pos(
                    ci,
                    pos_e,
                    mi,
                )
            )

            # ------------------------------------------------
            # Negative evidence
            # ------------------------------------------------

            if (
                h_e_neg_seq is not None
                and self.use_conflict
            ):

                neg_e = h_e_neg_seq[i]

                if self.use_credibility:

                    neg_cred, neg_w = (
                        self.credibility(
                            ci.unsqueeze(0),
                            neg_e.unsqueeze(0),
                            (
                                mi.unsqueeze(0)
                                if mi is not None
                                else None
                            ),
                            (
                                h_e_neg_cred_seq[i]
                                .unsqueeze(0)
                                if h_e_neg_cred_seq is not None
                                else None
                            ),
                        )
                    )

                    neg_e = neg_cred.squeeze(0)

                    neg_weights.append(
                        neg_w.squeeze(0)
                    )

            else:

                neg_e = h_e_pos_seq[i]

            neg_out.append(
                self.ceca_neg(
                    ci,
                    neg_e,
                    mi,
                )
            )

        pos_h = self.proj_ce_pos(
            torch.stack(
                pos_out
            )
        )

        neg_h = self.proj_ce_neg(
            torch.stack(
                neg_out
            )
        )

        # ----------------------------------------------------
        # Return credibility weights for analysis
        # ----------------------------------------------------

        if pos_weights:

            pos_weights = torch.stack(
                pos_weights
            )

        else:

            pos_weights = None

        if neg_weights:

            neg_weights = torch.stack(
                neg_weights
            )

        else:

            neg_weights = None

        return (
            pos_h,
            neg_h,
            pos_weights,
            neg_weights,
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        h_v: torch.Tensor,
        h_c: torch.Tensor,
        h_e_pos: torch.Tensor,
        h_r: torch.Tensor,
        h_e_neg: torch.Tensor | None = None,

        h_c_seq: torch.Tensor | None = None,
        h_e_pos_seq: torch.Tensor | None = None,
        h_e_neg_seq: torch.Tensor | None = None,

        graph_adj: torch.Tensor | None = None,
        claim_mask: torch.Tensor | None = None,

        # P1 optional credibility metadata
        #
        # [B, K, 3]
        #
        # authority / relevance / consistency
        h_e_pos_cred_seq: torch.Tensor | None = None,
        h_e_neg_cred_seq: torch.Tensor | None = None,
    ):

        B = h_v.size(0)
        dev = h_v.device

        # ----------------------------------------------------
        # Vision
        # ----------------------------------------------------

        pv = self.proj_v(
            h_v
        )

        H = pv.size(-1)

        # ----------------------------------------------------
        # Claim
        # ----------------------------------------------------

        pc_raw = self._encode_claims(
            h_c,
            h_c_seq,
            graph_adj,
            claim_mask,
        )

        pc = (
            pc_raw
            if self.use_claim
            else torch.zeros_like(pc_raw)
        )

        # ----------------------------------------------------
        # Evidence
        # ----------------------------------------------------

        _h_neg = (
            h_e_neg
            if (
                h_e_neg is not None
                and self.use_conflict
            )
            else torch.zeros_like(
                h_e_pos
            )
        )

        _h_e_neg_seq = (
            h_e_neg_seq
            if self.use_conflict
            else None
        )

        (
            pe_raw,
            pe_neg,
            pos_weights,
            neg_weights,
        ) = self._encode_evidence(
            h_e_pos,
            _h_neg,
            h_c_seq,
            h_e_pos_seq,
            _h_e_neg_seq,
            claim_mask,
            h_e_pos_cred_seq,
            h_e_neg_cred_seq,
        )

        pe = (
            pe_raw
            if self.use_evidence
            else torch.zeros_like(
                pe_raw
            )
        )

        # ----------------------------------------------------
        # Reasoning
        # ----------------------------------------------------

        pr_raw = self.proj_r(
            h_r
        )

        pr = (
            pr_raw
            if self.use_reasoning
            else torch.zeros_like(
                pr_raw
            )
        )

        # ----------------------------------------------------
        # Final fusion
        #
        #   fusion_type == "concat":
        #       original behaviour, cat([pv, pc, pe, pr]) -> classifier
        #
        #   fusion_type == "relation_aware":
        #       RelationAwareFusion (Claim-Evidence interaction +
        #       evidence reliability gate + reasoning gate) -> classifier
        # ----------------------------------------------------

        fusion_info = None

        if self.fusion_type == "relation_aware":

            h_final, fusion_info = self.fusion(
                pv,
                pc,
                pe,
                pr,
            )

        else:

            h_final = torch.cat(
                [
                    pv,
                    pc,
                    pe,
                    pr,
                ],
                dim=-1,
            )

        logits = self.classifier(
            h_final
        )

        # ----------------------------------------------------
        # Contrastive loss
        # ----------------------------------------------------

        contrastive = torch.tensor(
            0.0,
            device=dev,
        )

        if (
            self.use_contrastive
            and self.use_claim
            and self.use_evidence
            and self.training
        ):

            hard_neg = (
                pe_neg
                if self.contrastive_mode
                == "hard_neg"
                else None
            )

            valid = None

            if self.contrastive_valid_mask:

                ev = (
                    h_e_pos_seq
                    if h_e_pos_seq is not None
                    else h_e_pos.unsqueeze(1)
                )

                valid = (
                    (ev.abs().sum(dim=(1, 2)) > 1e-8)
                    & (pc.detach().norm(dim=-1) > 1e-6)
                )

            contrastive = (
                self.contrastive_loss(
                    pc,
                    pe,
                    h_e_neg=hard_neg,
                    valid_mask=valid,
                )
            )

        out = {
            "pred": logits,
            "contrastive": contrastive,

            # P1 analysis outputs
            "evidence_credibility_pos":
                pos_weights,

            "evidence_credibility_neg":
                neg_weights,
        }

        # Optional relation-aware fusion gate outputs (analysis only)
        if (
            self.return_fusion_gates
            and fusion_info is not None
        ):
            out["evidence_gate"] = fusion_info.get("evidence_gate")
            out["reasoning_gate"] = fusion_info.get("reasoning_gate")

        return out

    # ========================================================
    # Loss
    # ========================================================

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        contrastive: torch.Tensor,
        class_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:

        cls_loss = F.cross_entropy(
            logits,
            labels,
            weight=class_weight,
        )

        return (
            cls_loss
            + self.lambda_contrastive
            * contrastive
        )