"""
SigLIP2 Encoder Module for EaMRD-CRCV
======================================
Wraps google/siglip2-base-patch16-224 (or any Siglip2 variant) to provide
the same interface as the Chinese-CLIP encoder used in 07_encode_features.py.

Output dimensions for siglip2-base-patch16-224:
  vision: 768-d (pooled image embedding)
  text:   768-d (pooled text embedding)

Usage (offline feature encoding):
    encoder = SigLIP2Encoder("google/siglip2-base-patch16-224", device="cuda",
                              local_dir=r"F:\\Bishe_workspace\\hf_cache\\...")
    h_v = encoder.encode_image_from_paths(frame_paths)    # [768]
    h_c = encoder.encode_texts(claim_texts)               # [768]
    h_c_seq = encoder.encode_texts_seq(claim_texts)       # [N, 768]
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image


def _extract_tensor(raw) -> torch.Tensor:
    """Handle both old (tensor) and new (dataclass) transformers output formats."""
    if torch.is_tensor(raw):
        return raw.float()
    if hasattr(raw, "pooler_output") and raw.pooler_output is not None:
        return raw.pooler_output.float()
    if hasattr(raw, "image_embeds"):
        return raw.image_embeds.float()
    if hasattr(raw, "text_embeds"):
        return raw.text_embeds.float()
    if hasattr(raw, "last_hidden_state"):
        return raw.last_hidden_state[:, 0].float()
    raise ValueError(f"Cannot extract tensor from {type(raw)}: {dir(raw)}")


class SigLIP2Encoder:
    """Frozen SigLIP2 encoder for offline feature extraction.

    Args:
        model_name_or_path: HuggingFace model id or local directory.
        device: 'cuda' or 'cpu'.
        max_text_len: max token length for text encoding.
    """

    def __init__(
        self,
        model_name_or_path: str = "google/siglip2-base-patch16-224",
        device: str = "cuda",
        max_text_len: int = 64,
    ):
        # Import both SigLIP v1 and v2 classes; SigLIP2 uses SiglipModel architecture
        # but GemmaTokenizer for text, so we load components separately.
        from transformers import SiglipModel, SiglipImageProcessor, AutoTokenizer

        self.device = device
        self.max_text_len = max_text_len

        print(f"[SigLIP2Encoder] Loading {model_name_or_path} ...")
        self.model = (
            SiglipModel.from_pretrained(model_name_or_path)
            .to(device)
            .eval()
        )
        # SigLIP2 uses GemmaTokenizer (not SiglipTokenizer)
        self.tokenizer   = AutoTokenizer.from_pretrained(model_name_or_path)
        self.img_proc    = SiglipImageProcessor.from_pretrained(model_name_or_path)

        self.dim = self.model.config.vision_config.hidden_size
        print(f"[SigLIP2Encoder] Loaded. vision_dim={self.dim}, "
              f"text_dim={self.model.config.text_config.hidden_size}, "
              f"tokenizer={type(self.tokenizer).__name__}")

    # ------------------------------------------------------------------
    # Visual encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_image_from_paths(self, frame_paths: list[str]) -> Optional[torch.Tensor]:
        """Encode video frames; returns mean-pooled [dim] tensor, or None if empty."""
        images = []
        for fp in frame_paths:
            try:
                images.append(Image.open(fp).convert("RGB"))
            except Exception:
                continue
        if not images:
            return None
        inputs = self.img_proc(images=images, return_tensors="pt").to(self.device)
        raw = self.model.get_image_features(**inputs)
        feats = _extract_tensor(raw)          # [N_frames, dim]
        feats = F.normalize(feats, dim=-1)
        return feats.mean(dim=0).cpu()        # [dim]

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Mean-pool text encodings; returns [dim]."""
        if not texts:
            return torch.zeros(self.dim)
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_text_len,
        ).to(self.device)
        raw = self.model.get_text_features(**inputs)
        feats = _extract_tensor(raw)          # [N, dim]
        feats = F.normalize(feats, dim=-1)
        return feats.mean(dim=0).cpu()        # [dim]

    @torch.no_grad()
    def encode_texts_seq(
        self, texts: list[str], fixed_k: Optional[int] = None
    ) -> torch.Tensor:
        """Per-text encodings; returns [N, dim] or [fixed_k, dim] with zero padding."""
        if not texts:
            n = fixed_k if fixed_k else 1
            return torch.zeros(n, self.dim)
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_text_len,
        ).to(self.device)
        raw = self.model.get_text_features(**inputs)
        feats = _extract_tensor(raw)          # [N, dim]
        feats = F.normalize(feats, dim=-1).cpu()
        if fixed_k is not None:
            if feats.size(0) < fixed_k:
                pad = torch.zeros(fixed_k - feats.size(0), self.dim)
                feats = torch.cat([feats, pad], dim=0)
            else:
                feats = feats[:fixed_k]
        return feats
