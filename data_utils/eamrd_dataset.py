"""
EaMRD-CRCV Dataset & Collator
Reads pre-cached feature .pt files produced by 07_encode_features.py.

Cache layout:
  data/cache/{dataset}/features/feat_{vid}.pt
    → {
        "h_v":     tensor([D_v])   — visual (1024 FakeSV / 768 FakeTT)
        "h_c":     tensor([D_c])   — claim text (512 FakeSV / 768 FakeTT)
        "h_e_pos": tensor([D_c])   — semantic evidence
        "h_e_neg": tensor([D_c])   — contradiction evidence
        "h_r":     tensor([768])   — SBERT reasoning
      }

Zero tensors are returned for any missing file so that training can proceed
even with partially completed preprocessing.
"""

import os
from typing import Optional

import torch
from torch.utils.data import Dataset

from .baseline_data import FakeSVDataset, FakeTTDataset, WildFakeDataset

CACHE_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "cache")

# Zero-tensor defaults (actual dims set per dataset + encoder)
_ZERO_V_FAKESV = torch.zeros(1024)   # Chinese-CLIP ViT-L/14 vision
_ZERO_CE_FAKESV = torch.zeros(768)
_ZERO_V_FAKETT = torch.zeros(768)
_ZERO_CE_FAKETT = torch.zeros(768)
_ZERO_R = torch.zeros(768)

# SigLIP2-base outputs 768-d for both vision and text
_ZERO_V_SIGLIP2 = torch.zeros(768)
_ZERO_CE_SIGLIP2 = torch.zeros(768)

_EV_K = 5  # fixed number of evidence items


def _cache_key(dataset: str, encoder_type: str = "clip",
               feature_cache: Optional[str] = None) -> str:
    """Map (dataset, encoder_type) -> cache subdirectory name.

    If feature_cache is set explicitly (e.g. "FakeSV_FilteredEv"), it
    overrides the automatic naming so any features directory can be used
    without changing the dataset split logic.
    """
    if feature_cache:
        return feature_cache
    if encoder_type == "siglip2":
        return f"{dataset}_SigLIP2"
    return dataset   # default: original Chinese-CLIP / openai-CLIP cache


def _zero_tensors(dataset: str, encoder_type: str = "clip",
                  feature_cache: Optional[str] = None) -> dict:
    """Return appropriately-sized zero tensors for a given dataset+encoder."""
    # FakeSV_FilteredEv and WildFake_FilteredEv use same dims as FakeSV (Chinese-CLIP 768/1024)
    effective_dataset = dataset if not feature_cache else (
        "FakeSV" if ("FakeSV" in feature_cache or "WildFake" in feature_cache) else
        "FakeTT" if "FakeTT" in feature_cache else
        dataset
    )
    # WildFake base dataset also uses Chinese-CLIP (same dims as FakeSV)
    if dataset == "WildFake" and not feature_cache:
        effective_dataset = "FakeSV"
    if encoder_type == "siglip2":
        zv  = _ZERO_V_SIGLIP2
        zce = _ZERO_CE_SIGLIP2
    elif effective_dataset == "FakeSV":
        zv  = _ZERO_V_FAKESV
        zce = _ZERO_CE_FAKESV
    else:
        zv  = _ZERO_V_FAKETT
        zce = _ZERO_CE_FAKETT
    return {"h_v": zv, "h_c": zce, "h_e_pos": zce, "h_e_neg": zce, "h_r": _ZERO_R,
            "h_c_seq": torch.zeros(1, zce.shape[0]),
            "h_e_pos_seq": torch.zeros(_EV_K, zce.shape[0]),
            "h_e_neg_seq": torch.zeros(_EV_K, zce.shape[0]),
            "graph_adj": torch.zeros(1, 1)}


def _load_feat(dataset: str, vid: str, encoder_type: str = "clip",
               feature_cache: Optional[str] = None) -> dict:
    cache_dir = _cache_key(dataset, encoder_type, feature_cache)
    path = os.path.join(CACHE_ROOT, cache_dir, "features", f"feat_{vid}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=True)
    return _zero_tensors(dataset, encoder_type, feature_cache)


def _encoded_vids(dataset: str, encoder_type: str = "clip",
                  feature_cache: Optional[str] = None) -> set:
    cache_dir = _cache_key(dataset, encoder_type, feature_cache)
    feat_dir = os.path.join(CACHE_ROOT, cache_dir, "features")
    if not os.path.isdir(feat_dir):
        return set()
    return {
        f[len("feat_"):-3]
        for f in os.listdir(feat_dir)
        if f.startswith("feat_") and f.endswith(".pt")
    }


class FakeSVDataset_EaMRD(FakeSVDataset):
    def __init__(self, fold, split: str, encoder_type: str = "clip",
                 feature_cache: Optional[str] = None, **kwargs):
        super().__init__()
        self.split = split
        self.encoder_type = encoder_type
        self.feature_cache = feature_cache   # e.g. "FakeSV_FilteredEv"
        self.data = self._get_data(fold, split)
        self.data["video_id"] = self.data["video_id"].astype(str)

        enc = _encoded_vids("FakeSV", encoder_type, feature_cache)
        before = len(self.data)
        if enc:
            self.data = self.data[self.data["video_id"].isin(enc)]
        after = len(self.data)
        cache_tag = feature_cache or encoder_type
        print(f"[FakeSV/{split}] cache={cache_tag} {before} -> {after} (filtered by encoded features)")
        self.data = self.data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid  = str(item["video_id"])
        label = torch.tensor(int(item["label"]), dtype=torch.long)
        feat = _load_feat("FakeSV", vid, self.encoder_type, self.feature_cache)
        return {
            "vid":         vid,
            "label":       label,
            "h_v":         feat["h_v"].float(),
            "h_c":         feat["h_c"].float(),
            "h_e_pos":     feat["h_e_pos"].float(),
            "h_e_neg":     feat["h_e_neg"].float(),
            "h_r":         feat["h_r"].float(),
            "h_c_seq":     feat.get("h_c_seq",     torch.zeros(1, 768)).float(),
            "h_e_pos_seq": feat.get("h_e_pos_seq", torch.zeros(_EV_K, 768)).float(),
            "h_e_neg_seq": feat.get("h_e_neg_seq", torch.zeros(_EV_K, 768)).float(),
            "graph_adj":   feat.get("graph_adj",   torch.zeros(1, 1)).float(),
        }


class FakeTTDataset_EaMRD(FakeTTDataset):
    def __init__(self, fold, split: str, encoder_type: str = "clip",
                 feature_cache: Optional[str] = None, **kwargs):
        super().__init__()
        self.split = split
        self.encoder_type = encoder_type
        self.feature_cache = feature_cache   # e.g. "FakeTT_FilteredEv" or "FakeTT_woCMR"
        self.data = self._get_data(fold, split)
        self.data["video_id"] = self.data["video_id"].astype(str)

        enc = _encoded_vids("FakeTT", encoder_type, feature_cache)
        before = len(self.data)
        if enc:
            self.data = self.data[self.data["video_id"].isin(enc)]
        after = len(self.data)
        cache_tag = feature_cache or encoder_type
        print(f"[FakeTT/{split}] cache={cache_tag} {before} -> {after} (filtered by encoded features)")
        self.data = self.data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid  = str(item["video_id"])
        label = torch.tensor(int(item["label"]), dtype=torch.long)
        feat = _load_feat("FakeTT", vid, self.encoder_type, self.feature_cache)
        return {
            "vid":         vid,
            "label":       label,
            "h_v":         feat["h_v"].float(),
            "h_c":         feat["h_c"].float(),
            "h_e_pos":     feat["h_e_pos"].float(),
            "h_e_neg":     feat["h_e_neg"].float(),
            "h_r":         feat["h_r"].float(),
            "h_c_seq":     feat.get("h_c_seq",     torch.zeros(1, 768)).float(),
            "h_e_pos_seq": feat.get("h_e_pos_seq", torch.zeros(_EV_K, 768)).float(),
            "h_e_neg_seq": feat.get("h_e_neg_seq", torch.zeros(_EV_K, 768)).float(),
            "graph_adj":   feat.get("graph_adj",   torch.zeros(1, 1)).float(),
        }


class WildFakeDataset_EaMRD(WildFakeDataset):
    def __init__(self, fold, split: str, encoder_type: str = "clip",
                 feature_cache: Optional[str] = None, **kwargs):
        super().__init__()
        self.split = split
        self.encoder_type = encoder_type
        self.feature_cache = feature_cache   # e.g. "WildFake_FilteredEv"
        self.data = self._get_data(fold, split)
        self.data["video_id"] = self.data["video_id"].astype(str)

        enc = _encoded_vids("WildFake", encoder_type, feature_cache)
        before = len(self.data)
        if enc:
            self.data = self.data[self.data["video_id"].isin(enc)]
        after = len(self.data)
        cache_tag = feature_cache or encoder_type
        print(f"[WildFake/{split}] cache={cache_tag} {before} -> {after} (filtered by encoded features)")
        self.data = self.data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid  = str(item["video_id"])
        label = torch.tensor(int(item["label"]), dtype=torch.long)
        feat = _load_feat("WildFake", vid, self.encoder_type, self.feature_cache)
        return {
            "vid":         vid,
            "label":       label,
            "h_v":         feat["h_v"].float(),
            "h_c":         feat["h_c"].float(),
            "h_e_pos":     feat["h_e_pos"].float(),
            "h_e_neg":     feat["h_e_neg"].float(),
            "h_r":         feat["h_r"].float(),
            "h_c_seq":     feat.get("h_c_seq",     torch.zeros(1, 768)).float(),
            "h_e_pos_seq": feat.get("h_e_pos_seq", torch.zeros(_EV_K, 768)).float(),
            "h_e_neg_seq": feat.get("h_e_neg_seq", torch.zeros(_EV_K, 768)).float(),
            "graph_adj":   feat.get("graph_adj",   torch.zeros(1, 1)).float(),
        }


class EaMRDCollator:
    """Stack variable-dim feature tensors into batches.
    Pads h_c_seq and graph_adj to the max N_c within the batch.
    """

    def __call__(self, batch: list[dict]) -> dict:
        # Fixed-dim tensors
        out = {
            "vids":        [item["vid"] for item in batch],
            "labels":      torch.stack([item["label"]       for item in batch]),
            "h_v":         torch.stack([item["h_v"]         for item in batch]),
            "h_c":         torch.stack([item["h_c"]         for item in batch]),
            "h_e_pos":     torch.stack([item["h_e_pos"]     for item in batch]),
            "h_e_neg":     torch.stack([item["h_e_neg"]     for item in batch]),
            "h_r":         torch.stack([item["h_r"]         for item in batch]),
            "h_e_pos_seq": torch.stack([item["h_e_pos_seq"] for item in batch]),  # [B, K, 768]
            "h_e_neg_seq": torch.stack([item["h_e_neg_seq"] for item in batch]),  # [B, K, 768]
        }

        # Variable-length: pad h_c_seq and graph_adj to max N_c in batch
        max_nc = max(item["h_c_seq"].size(0) for item in batch)
        D = batch[0]["h_c_seq"].size(1)
        padded_c_seq = torch.zeros(len(batch), max_nc, D)
        padded_adj   = torch.zeros(len(batch), max_nc, max_nc)
        claim_mask   = torch.zeros(len(batch), max_nc, dtype=torch.bool)

        for i, item in enumerate(batch):
            n = item["h_c_seq"].size(0)
            padded_c_seq[i, :n] = item["h_c_seq"]
            padded_adj[i, :n, :n] = item["graph_adj"][:n, :n]
            claim_mask[i, :n] = True

        out["h_c_seq"]   = padded_c_seq   # [B, max_nc, 768]
        out["graph_adj"] = padded_adj     # [B, max_nc, max_nc]
        out["claim_mask"] = claim_mask    # [B, max_nc]
        return out


FakeSVCollator_EaMRD = EaMRDCollator
FakeTTCollator_EaMRD = EaMRDCollator
WildFakeCollator_EaMRD = EaMRDCollator
