"""CLIP ViT-L/14 encoder for three feature streams: h_v, h_c, h_e."""

import torch
import torch.nn as nn
import clip  # openai/clip


class CLIPEncoder(nn.Module):
    """Wraps CLIP ViT-L/14 and exposes encode_image / encode_text helpers."""

    def __init__(self, model_path: str = "ViT-L/14", device: str = "cuda"):
        super().__init__()
        self.device = device
        model, preprocess = clip.load(model_path, device=device)
        self.model = model.float()
        self.preprocess = preprocess
        # freeze CLIP weights — features are pre-cached; this encoder is used
        # only during 07_encode_features.py offline pass, NOT during training.
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: [B, T, C, H, W]  T=16 key-frames
        returns h_v: [B, T, 768]
        """
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W).to(self.device)
        feats = self.model.encode_image(flat)           # [B*T, 768]
        feats = feats.view(B, T, -1)                   # [B, T, 768]
        return feats.float()

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """
        texts: list of strings
        returns: [N, 768]
        """
        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        feats = self.model.encode_text(tokens)
        return feats.float()

    @torch.no_grad()
    def encode_claims(self, claims: list[str]) -> torch.Tensor:
        """h_c: encode claim texts → [N, 768]"""
        return self.encode_text(claims)

    @torch.no_grad()
    def encode_evidence(self, evidence_list: list[str]) -> torch.Tensor:
        """h_e: encode top-K evidence snippets → [N, 768]"""
        return self.encode_text(evidence_list)
