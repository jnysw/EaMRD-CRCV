"""
Ablation Experiment Runner — FakeSV V2

Runs each ablation variant with 5 seeds and collects Acc / F1 statistics.

Usage:
    python run_ablation.py                     # run all variants
    python run_ablation.py --variants full base # run specific variants
    python run_ablation.py --variants wo_cmr    # single variant

Variants:
    full      — EaMRD-CRCV (complete model)
    base      — Base model (no evidence, no reasoning)
    wo_cmr    — w/o Claim-centric Retrieval
    wo_ev     — w/o Evidence Verification
    score_ev  — Score-based Evidence Selection
"""

import argparse
import json
import os
import re
import subprocess
import sys
import numpy as np
from pathlib import Path

PYTHON   = sys.executable
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
SEEDS    = [123, 130, 2024, 456, 728]

VARIANTS = {
    "full":     "configs/fakesv_v2/ablation/full.yaml",
    "base":     "configs/fakesv_v2/ablation/base.yaml",
    "wo_cmr":   "configs/fakesv_v2/ablation/wo_cmr.yaml",
    "wo_ev":    "configs/fakesv_v2/ablation/wo_ev.yaml",
    "score_ev": "configs/fakesv_v2/ablation/score_ev.yaml",
    "wo_egr":   "configs/fakesv_v2/ablation/wo_egr.yaml",
    "wo_cmc":   "configs/fakesv_v2/ablation/wo_cmc.yaml",
}

VARIANT_LABELS = {
    "full":     "EaMRD-CRCV (Full)",
    "base":     "Base Model",
    "wo_cmr":   "w/o CMR",
    "wo_ev":    "w/o EV",
    "score_ev": "Score EV",
    "wo_egr":   "w/o EGR",
    "wo_cmc":   "w/o CMC",
}


def run_one(variant: str, config: str, seed: int) -> tuple:
    log_dir = os.path.join(WORK_DIR, "logs", "ablation", variant, f"seed{seed}")
    os.makedirs(log_dir, exist_ok=True)
    stdout_path = os.path.join(log_dir, "stdout.log")

    # Skip if already done
    if os.path.exists(stdout_path):
        with open(stdout_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        m = re.search(r"BestTest.*?Acc=([\d.]+).*?F1=([\d.]+)", text)
        if m:
            print(f"  [SKIP] {variant}/seed{seed}  Acc={m.group(1)}  F1={m.group(2)}")
            return float(m.group(1)), float(m.group(2))

    with open(stdout_path, "w", encoding="utf-8") as fout:
        proc = subprocess.run(
            [PYTHON, "train.py",
             "--config",  config,
             "--seed",    str(seed),
             "--log-dir", log_dir],
            stdout=fout, stderr=fout,
            cwd=WORK_DIR,
        )

    with open(stdout_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    m = re.search(r"BestTest.*?Acc=([\d.]+).*?F1=([\d.]+)", text)
    if m:
        acc, f1 = float(m.group(1)), float(m.group(2))
        print(f"  {variant}/seed{seed}  Acc={acc:.4f}  F1={f1:.4f}  exit={proc.returncode}")
        return acc, f1
    else:
        print(f"  {variant}/seed{seed}  NO RESULT (exit={proc.returncode})")
        lines = text.strip().split("\n")
        for ln in lines[-10:]:
            print(f"    | {ln}")
        return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()),
                        choices=list(VARIANTS.keys()),
                        help="Which variants to run (default: all)")
    args = parser.parse_args()

    all_results = {}

    for variant in args.variants:
        config = VARIANTS[variant]
        label  = VARIANT_LABELS[variant]
        print(f"\n{'='*65}")
        print(f"  {label}  ({variant})")
        print(f"  Config: {config}")
        print(f"{'='*65}")

        accs, f1s = [], []
        for seed in SEEDS:
            acc, f1 = run_one(variant, config, seed)
            if acc is not None:
                accs.append(acc)
                f1s.append(f1)

        all_results[variant] = {"accs": accs, "f1s": f1s}

        if accs:
            print(f"\n  {label}: Acc={np.mean(accs):.4f}±{np.std(accs):.4f}  "
                  f"F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}  ({len(accs)}/{len(SEEDS)} seeds)")

    # ── Final summary table ────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  ABLATION RESULTS  (FakeSV, 5-seed)")
    print(f"{'='*70}")
    print(f"  {'Variant':<30} {'Acc':>10} {'F1':>10}  {'#seeds':>6}")
    print(f"  {'-'*58}")

    order = ["base", "wo_cmr", "wo_ev", "score_ev", "wo_egr", "wo_cmc", "full"]
    for v in order:
        if v not in all_results:
            continue
        r = all_results[v]
        accs, f1s = r["accs"], r["f1s"]
        label = VARIANT_LABELS[v]
        if accs:
            print(f"  {label:<30} {np.mean(accs):>6.4f}±{np.std(accs):.4f}  "
                  f"{np.mean(f1s):>6.4f}±{np.std(f1s):.4f}  {len(accs):>3}/{len(SEEDS)}")
        else:
            print(f"  {label:<30} {'N/A':>10}  {'N/A':>10}  {'0':>3}/{len(SEEDS)}")

    # Save JSON summary
    summary_path = os.path.join(WORK_DIR, "logs", "ablation", "summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({v: {"acc_mean": float(np.mean(r["accs"])) if r["accs"] else None,
                       "acc_std":  float(np.std(r["accs"]))  if r["accs"] else None,
                       "f1_mean":  float(np.mean(r["f1s"]))  if r["f1s"] else None,
                       "f1_std":   float(np.std(r["f1s"]))   if r["f1s"] else None,
                       "n_seeds":  len(r["accs"])}
                   for v, r in all_results.items()},
                  f, indent=2)
    print(f"\n  Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
