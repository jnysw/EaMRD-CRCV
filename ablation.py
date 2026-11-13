"""
Ablation Experiment Runner
Runs all ablation variants and prints a summary table.

Usage:
    python ablation.py --dataset FakeSV
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


ABLATION_CONFIGS = [
    ("w/o Claim",        "configs/ablation/wo_claim.yaml"),
    ("w/o Contradiction","configs/ablation/wo_contradiction.yaml"),
    ("w/o NLI",          "configs/ablation/wo_nli.yaml"),
    ("w/o Reasoning",    "configs/ablation/wo_reasoning.yaml"),
    ("w/o Contrastive",  "configs/ablation/wo_contrastive.yaml"),
    ("Hard-neg",         "configs/ablation/hard_neg.yaml"),
    ("Full (default)",   "configs/default.yaml"),
]


def run_variant(config_path: str, log_suffix: str) -> int:
    cmd = [sys.executable, "train.py", "--config", config_path]
    print(f"\n{'='*60}")
    print(f"Running: {config_path}")
    print(f"{'='*60}")
    proc = subprocess.run(cmd, text=True)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="FakeSV")
    parser.add_argument("--variants", nargs="*", help="Run specific variants only")
    args = parser.parse_args()

    results = {}
    for name, config in ABLATION_CONFIGS:
        if args.variants and name not in args.variants:
            continue
        if not Path(config).exists():
            print(f"[SKIP] Config not found: {config}")
            continue
        rc = run_variant(config, name.replace("/", "_").replace(" ", "_"))
        results[name] = "OK" if rc == 0 else f"FAILED(rc={rc})"

    print("\n=== Ablation Summary ===")
    for name, status in results.items():
        print(f"  {name:<25}: {status}")


if __name__ == "__main__":
    main()
