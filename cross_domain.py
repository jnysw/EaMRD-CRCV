"""
Cross-domain generalization experiment: Train on FakeSV → Test on FakeTT

Usage:
    python cross_domain.py --checkpoint logs/MMDD-HHMMSS/best_model.pth
"""

import argparse
import sys

import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader

from evaluate import evaluate
from train import build_dataset_and_collator, build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="FakeSV trained checkpoint")
    parser.add_argument("--config", default="configs/fakesv_v2/ablation/full.yaml")
    parser.add_argument("--target-feature-cache", default="FakeTT_FilteredEv_v3")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    if cfg["dataset"] != "FakeSV" or cfg["type"] != "temporal":
        raise ValueError("Expected a temporal FakeSV source configuration")
    target_cfg = dict(cfg, dataset="FakeTT", feature_cache=args.target_feature_cache)
    test_ds, collator = build_dataset_and_collator(target_cfg, "temporal", "test")
    loader = DataLoader(test_ds, batch_size=64, collate_fn=collator, shuffle=False)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source_dim = state["proj_v.net.0.weight"].shape[1]
    target_dim = test_ds[0]["h_v"].numel()
    if source_dim != target_dim:
        raise ValueError(
            f"Incompatible visual feature spaces: checkpoint expects {source_dim} dimensions, "
            f"but FakeTT cache provides {target_dim}. Re-encode both datasets with the same "
            "frozen visual encoder before a zero-shot cross-domain evaluation."
        )
    model = build_model(cfg)
    model.load_state_dict(state)
    model.to(device)

    metrics = evaluate(model, loader, device)
    logger.info("=== Cross-domain: FakeSV → FakeTT ===")
    logger.info(f"Acc={metrics['acc']:.4f}  F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}")


if __name__ == "__main__":
    main()
