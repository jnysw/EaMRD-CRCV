"""
EaMRD-CRCV Training Script
Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/ablation/wo_contrastive.yaml
"""

import importlib
import math
import os
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import colorama
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from colorama import Fore
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.core_utils import (
    BinaryClassificationMetric,
    EarlyStopping,
    WarmUpStepLR,
    set_seed,
    set_worker_seed,
)

colorama.init()

import argparse as _ap
_pre = _ap.ArgumentParser(add_help=False)
_pre.add_argument("--log-dir", default=None)
_pre_args, _ = _pre.parse_known_args()
LOG_DIR = Path(_pre_args.log_dir) if _pre_args.log_dir else Path("logs") / datetime.now().strftime("%m%d-%H%M%S")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def build_model(cfg: dict) -> nn.Module:
    from model.eamrd_crcv import EaMRDCRCV
    return EaMRDCRCV(
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
        temperature=cfg["model"]["temperature"],
        lambda_contrastive=cfg["model"]["lambda_contrastive"],
        use_contrastive=cfg["model"].get("use_contrastive", True),
        contrastive_mode=cfg["model"].get("contrastive_mode", "in_batch"),
        use_gat=cfg["model"].get("use_gat", False),
        use_ceca=cfg["model"].get("use_ceca", False),
        gat_hidden=cfg["model"].get("gat_hidden", 128),
        gat_heads=cfg["model"].get("gat_heads", 4),
        gat_dropout=cfg["model"].get("gat_dropout", 0.3),
        ceca_heads=cfg["model"].get("ceca_heads", 4),
        ceca_dropout=cfg["model"].get("ceca_dropout", 0.1),
        # Ablation flags
        use_claim=cfg["model"].get("use_claim", True),
        use_evidence=cfg["model"].get("use_evidence", True),
        use_reasoning=cfg["model"].get("use_reasoning", True),
        use_conflict=cfg["model"].get("use_conflict", True),
        # Final fusion (defaults preserve original concat behaviour)
        fusion_type=cfg["model"].get("fusion_type", "concat"),
        use_ce_interaction=cfg["model"].get("use_ce_interaction", True),
        use_evidence_gate=cfg["model"].get("use_evidence_gate", True),
        use_reasoning_gate=cfg["model"].get("use_reasoning_gate", True),
        fusion_dropout=cfg["model"].get("fusion_dropout", 0.1),
        return_fusion_gates=cfg["model"].get("return_fusion_gates", False),
        # E1 fixes (defaults reproduce the original Full model)
        contrastive_valid_mask=cfg["model"].get("contrastive_valid_mask", False),
        gat_keep_self_loop=cfg["model"].get("gat_keep_self_loop", False),
        ceca_mask_padding=cfg["model"].get("ceca_mask_padding", False),
    )


def build_dataset_and_collator(cfg: dict, fold, split: str):
    dataset_name = cfg["dataset"]
    encoder_type  = cfg.get("encoder_type", "clip")      # "clip" | "siglip2"
    feature_cache = cfg.get("feature_cache", None)        # e.g. "FakeSV_FilteredEv"
    module = importlib.import_module("data_utils.eamrd_dataset")
    dataset_cls = getattr(module, f"{dataset_name}Dataset_EaMRD")
    collator_cls = getattr(module, f"{dataset_name}Collator_EaMRD")
    dataset = dataset_cls(fold=fold, split=split,
                          encoder_type=encoder_type, feature_cache=feature_cache)
    collator = collator_cls()
    return dataset, collator


class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.evaluator = BinaryClassificationMetric(self.device)
        set_seed(cfg["seed"])
        self.generator = torch.Generator().manual_seed(cfg["seed"])
        self.class_weight = None   # set in _reset() if use_class_weight=true

        if cfg["type"] == "temporal":
            self.folds = ["temporal"]
        elif cfg["type"] == "5-fold":
            self.folds = [1, 2, 3, 4, 5]
        else:
            raise ValueError(f"Unknown experiment type: {cfg['type']}")

    def _make_loader(self, dataset, collator, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.cfg["batch_size"],
            collate_fn=collator,
            shuffle=shuffle,
            num_workers=0,
            generator=self.generator,
        )

    def _reset(self, fold):
        train_ds, collator = build_dataset_and_collator(self.cfg, fold, "train")
        test_ds, _ = build_dataset_and_collator(self.cfg, fold, "test")
        self.train_loader = self._make_loader(train_ds, collator, shuffle=True)
        self.test_loader = self._make_loader(test_ds, collator, shuffle=False)

        # Compute class weights (inverse-frequency) to handle distribution shift
        use_cw = self.cfg.get("use_class_weight", False)
        self.class_weight = None
        if use_cw:
            import numpy as np
            labels = torch.tensor([int(train_ds[i]["label"]) for i in range(len(train_ds))])
            counts = torch.bincount(labels, minlength=2).float()
            w = counts.sum() / (2.0 * counts)          # balanced inverse-freq
            self.class_weight = w.to(self.device)
            logger.info(f"Class weights: real={w[0]:.3f}, fake={w[1]:.3f}")

        if self.cfg["type"] == "temporal":
            val_ds, _ = build_dataset_and_collator(self.cfg, fold, "valid")
            self.val_loader = self._make_loader(val_ds, collator, shuffle=False)

        steps_per_epoch = math.ceil(len(train_ds) / self.cfg["batch_size"])
        self.model = build_model(self.cfg).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg["optimizer"]["lr"],
            weight_decay=self.cfg["optimizer"]["weight_decay"],
        )
        total_steps = steps_per_epoch * self.cfg["num_epoch"]
        self.scheduler = WarmUpStepLR(
            self.optimizer,
            warmup_total_steps=total_steps,
            warmup_rate=self.cfg["scheduler"].get("warmup_rate", 0.05),
        )
        self.early_stopper = EarlyStopping(
            patience=self.cfg.get("patience", 5),
            path=str(LOG_DIR / "best_model.pth"),
        )

    def _forward(self, batch):
        dev = self.device
        h_v     = batch["h_v"].to(dev)
        h_c     = batch["h_c"].to(dev)
        h_e_pos = batch["h_e_pos"].to(dev)
        h_e_neg = batch["h_e_neg"].to(dev)
        h_r     = batch["h_r"].to(dev)
        labels  = batch["labels"].to(dev)

        # Optional sequence inputs (present when use_gat or use_ceca)
        h_c_seq     = batch.get("h_c_seq")
        h_e_pos_seq = batch.get("h_e_pos_seq")
        h_e_neg_seq = batch.get("h_e_neg_seq")
        graph_adj   = batch.get("graph_adj")
        claim_mask  = batch.get("claim_mask")

        to_dev = lambda t: t.to(dev) if t is not None else None

        out = self.model(
            h_v, h_c, h_e_pos, h_r,
            h_e_neg=h_e_neg if self.cfg["model"].get("contrastive_mode") == "hard_neg" else None,
            h_c_seq=to_dev(h_c_seq),
            h_e_pos_seq=to_dev(h_e_pos_seq),
            h_e_neg_seq=to_dev(h_e_neg_seq),
            graph_adj=to_dev(graph_adj),
            claim_mask=to_dev(claim_mask),
        )
        loss = self.model.compute_loss(out["pred"], labels, out["contrastive"],
                                       class_weight=self.class_weight)
        return out["pred"], labels, loss

    def _train_epoch(self):
        self.model.train()
        losses = []
        for batch in tqdm(self.train_loader, bar_format=f"{Fore.BLUE}{{l_bar}}{{bar}}{{r_bar}}"):
            pred, labels, loss = self._forward(batch)
            _, preds = torch.max(pred, 1)
            self.evaluator.update(preds, labels)
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
        metrics = self.evaluator.compute()
        logger.info(f"{Fore.BLUE}Train Loss={np.mean(losses):.4f} Acc={metrics['acc']:.4f} F1={metrics['f1']:.4f}")

    @torch.no_grad()
    def _eval(self, loader, split_name: str, color=Fore.RED, use_earlystop=False,
              save_preds: bool = False) -> dict:
        self.model.eval()
        losses, all_probs, all_labels, all_vids = [], [], [], []
        for batch in tqdm(loader, bar_format=f"{color}{{l_bar}}{{bar}}{{r_bar}}"):
            pred, labels, loss = self._forward(batch)
            _, preds = torch.max(pred, 1)
            self.evaluator.update(preds, labels)
            losses.append(loss.item())
            if save_preds:
                probs = torch.softmax(pred, dim=1)[:, 1]  # P(fake)
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())
                vids = batch.get("vids", batch.get("vid"))
                if vids is not None:
                    all_vids.extend(vids)
        metrics = self.evaluator.compute()
        logger.info(f"{color}{split_name} Loss={np.mean(losses):.4f} Acc={metrics['acc']:.4f} F1={metrics['f1']:.4f} AUC≈-")
        if use_earlystop:
            self.early_stopper(metrics["acc"], self.model)
        if save_preds and all_probs:
            import json
            probs_cat = torch.cat(all_probs).numpy().tolist()
            labels_cat = torch.cat(all_labels).numpy().tolist()
            pred_data = {"probs": probs_cat, "labels": labels_cat, "vids": all_vids}
            pred_path = LOG_DIR / f"preds_{split_name.lower()}.json"
            with open(pred_path, "w") as f:
                json.dump(pred_data, f)
            logger.info(f"Saved {len(probs_cat)} predictions -> {pred_path}")
        return metrics

    def run(self):
        # Fixed decision rule: argmax / P(fake) >= 0.5. No threshold search on val/test.
        acc_list, f1_list = [], []
        for fold in self.folds:
            logger.info(f"=== Fold: {fold} ===")
            self._reset(fold)
            for epoch in range(self.cfg["num_epoch"]):
                logger.info(f"Epoch {epoch+1}/{self.cfg['num_epoch']}")
                self._train_epoch()
                if self.cfg["type"] == "temporal":
                    self._eval(self.val_loader, "Val", color=Fore.YELLOW, use_earlystop=True)
                else:
                    self._eval(self.test_loader, "Test", use_earlystop=True)
                if self.early_stopper.early_stop:
                    logger.info("Early stopping.")
                    break

            self.model.load_state_dict(torch.load(str(LOG_DIR / "best_model.pth"), weights_only=False))
            metrics = self._eval(self.test_loader, "BestTest", save_preds=True)
            acc_list.append(metrics["acc"])
            f1_list.append(metrics["f1"])

        logger.info(f"Final Acc={np.mean(acc_list):.4f} \u00b1 {np.std(acc_list):.4f}")
        logger.info(f"Final F1 ={np.mean(f1_list):.4f} \u00b1 {np.std(f1_list):.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--log-dir", default=None)   # handled by pre-parser above
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed in config (for multi-seed experiments)")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["seed"] = args.seed
        logger.info(f"[CLI] seed overridden to {args.seed}")

    logger.remove()
    logger.add(LOG_DIR / "train.log", retention="30 days", level="DEBUG")
    logger.add(sys.stdout, level="INFO")
    logger.info(f"Config: {cfg}")

    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()

