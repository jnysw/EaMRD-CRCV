"""
Step 0: Migrate existing visual features from vit_tensor.pt to per-video cache files.

Source: F:\\Bishe_workspace\\fakesv-qwen3\\vit_tensor.pt
  Format: {vid: tensor([16, 1024])}  — Chinese-CLIP ViT-L/14 vision features, 16 frames

Target: data/cache/FakeSV/features/clip_{vid}.pt
  Format: {"h_v": tensor([1024])}    — mean-pooled, ready for model

This avoids loading the full 5495-video tensor file at every training epoch.
Only needs to be run ONCE.

Usage:
    python preprocess/00_migrate_features.py
"""

import os
import sys

import torch
from tqdm import tqdm

SRC = r"F:\Bishe_workspace\fakesv-qwen3\vit_tensor.pt"
CACHE_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
DST_DIR = os.path.join(CACHE_ROOT, "FakeSV", "features")


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    print(f"Loading {SRC} ...")
    all_feats = torch.load(SRC, weights_only=True)  # {vid: [16, 1024]}
    print(f"Loaded {len(all_feats)} videos, feature shape: {next(iter(all_feats.values())).shape}")

    already_done = {f[len("clip_"):-3] for f in os.listdir(DST_DIR) if f.startswith("clip_")}
    todo = {k: v for k, v in all_feats.items() if k not in already_done}
    print(f"Already done: {len(already_done)}  Todo: {len(todo)}")

    for vid, feat in tqdm(todo.items(), desc="Migrating h_v"):
        # feat: [16, 1024]  → mean pool → [1024]
        h_v = feat.float().mean(dim=0)  # [1024]
        out = {"h_v": h_v}
        torch.save(out, os.path.join(DST_DIR, f"clip_{vid}.pt"))

    print(f"Done. {len(todo)} files written to {DST_DIR}")
    print("NOTE: h_c, h_e_pos, h_e_neg will be added by 07_encode_features.py after claim/evidence pipeline.")


if __name__ == "__main__":
    main()
