"""Sentence-BERT encoder for MLLM reasoning output → h_r."""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class ReasoningEncoder(nn.Module):
    """Encode structured MLLM reasoning text with all-mpnet-base-v2."""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", device: str = "cuda"):
        super().__init__()
        self.sbert = SentenceTransformer(model_name, device=device)
        self.hidden_dim = 768  # all-mpnet output dim
        for p in self.sbert.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """
        texts: list of reasoning strings (one per video)
        returns h_r: [N, 768]
        """
        embs = self.sbert.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        return embs.float()
