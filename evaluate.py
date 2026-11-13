"""
Evaluate a trained EaMRD-CRCV checkpoint.
Usage:
    python evaluate.py --config configs/default.yaml --checkpoint logs/MMDD-HHMMSS/best_model.pth
"""

import argparse
import sys

import torch
import yaml
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import build_dataset_and_collator, build_model


def evaluate(model, loader, device) -> dict:
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            h_v = batch["h_v"].to(device)
            h_c = batch["h_c"].to(device)
            h_e_pos = batch["h_e_pos"].to(device)
            h_r = batch["h_r"].to(device)
            labels = batch["labels"]

            to_device = lambda name: batch[name].to(device) if name in batch else None
            out = model(
                h_v, h_c, h_e_pos, h_r,
                h_e_neg=to_device("h_e_neg") if model.contrastive_mode == "hard_neg" else None,
                h_c_seq=to_device("h_c_seq"),
                h_e_pos_seq=to_device("h_e_pos_seq"),
                h_e_neg_seq=to_device("h_e_neg_seq"),
                graph_adj=to_device("graph_adj"),
                claim_mask=to_device("claim_mask"),
            )
            probs = torch.softmax(out["pred"], dim=-1)[:, 1].cpu().numpy()
            preds = out["pred"].argmax(dim=-1).cpu().numpy()

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0

    return {"acc": acc, "f1": f1, "auc": auc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    fold = "temporal" if cfg["type"] == "temporal" else 1
    dataset, collator = build_dataset_and_collator(cfg, fold, args.split)
    loader = DataLoader(dataset, batch_size=64, collate_fn=collator, shuffle=False)

    model = build_model(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False))
    model.to(device)

    metrics = evaluate(model, loader, device)
    logger.info(f"[{cfg['dataset']}/{args.split}] Acc={metrics['acc']:.4f}  F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}")


if __name__ == "__main__":
    main()
