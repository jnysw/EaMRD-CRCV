"""
Run multi-seed experiment for any config.
Saves results to logs/{tag}_seed{seed}/ and prints final summary.

Usage:
    python run_multiseed.py                                         # FakeSV v2 best (default)
    python run_multiseed.py --config configs/fakett_v2/full.yaml --tag fakett_v2
    python run_multiseed.py --config configs/fakett_v2/full.yaml --tag fakett_v2 --seeds 123 130 2024 456 728
"""
import argparse
import os
import re
import subprocess
import sys
import numpy as np
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  default="configs/fakesv_v2/best.yaml")
    ap.add_argument("--seeds",   nargs="+", type=int, default=[123, 130, 2024, 456, 728])
    ap.add_argument("--tag",     default=None,
                    help="Log dir tag (default: derived from config path)")
    return ap.parse_args()

args    = parse_args()
PYTHON  = sys.executable
CONFIG  = args.config
SEEDS   = args.seeds
WORK_DIR = os.path.dirname(os.path.abspath(__file__))

# Derive tag from config path if not specified: configs/fakett_v2/full.yaml -> fakett_v2
if args.tag:
    TAG = args.tag
else:
    parts = Path(CONFIG).parts
    TAG = "_".join(parts[-2:]).replace(".yaml", "") if len(parts) >= 2 else Path(CONFIG).stem

accs, f1s = [], []

for seed in SEEDS:
    log_dir = os.path.join(WORK_DIR, "logs", f"{TAG}_seed{seed}")
    os.makedirs(log_dir, exist_ok=True)   # create dir BEFORE starting process
    stdout_path = os.path.join(log_dir, "stdout.log")

    print(f"\n{'='*60}")
    print(f"  seed={seed}   log_dir={log_dir}")
    print(f"{'='*60}", flush=True)

    with open(stdout_path, "w", encoding="utf-8") as fout:
        proc = subprocess.run(
            [PYTHON, "train.py",
             "--config",  CONFIG,
             "--seed",    str(seed),
             "--log-dir", log_dir],
            stdout=fout,
            stderr=fout,
            cwd=WORK_DIR,
        )

    # Parse result
    with open(stdout_path, encoding="utf-8", errors="replace") as f:
        text = f.read()

    m = re.search(r"BestTest.*?Acc=([\d.]+).*?F1=([\d.]+)", text)
    if m:
        acc, f1 = float(m.group(1)), float(m.group(2))
        accs.append(acc)
        f1s.append(f1)
        print(f"  seed={seed}  Acc={acc:.4f}  F1={f1:.4f}  exit={proc.returncode}", flush=True)
    else:
        print(f"  seed={seed}  NO RESULT (exit={proc.returncode})", flush=True)
        # Print last 20 lines for debugging
        lines = text.strip().split("\n")
        for ln in lines[-20:]:
            print("   |", ln, flush=True)

# Summary
print(f"\n{'='*60}")
print(f"  FINAL SUMMARY  ({len(accs)}/{len(SEEDS)} seeds completed)")
print(f"{'='*60}")
print(f"  {'Seed':<8} {'Acc':>7} {'F1':>7}")
print(f"  {'-'*24}")
for i, seed in enumerate(SEEDS):
    if i < len(accs):
        print(f"  {seed:<8} {accs[i]:>7.4f} {f1s[i]:>7.4f}")
    else:
        print(f"  {seed:<8}  failed")

if accs:
    print(f"  {'-'*24}")
    print(f"  {'Mean':<8} {np.mean(accs):>7.4f} {np.mean(f1s):>7.4f}")
    print(f"  {'Std':<8} {np.std(accs):>7.4f} {np.std(f1s):>7.4f}")
    print(f"\n  Acc = {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  F1  = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
