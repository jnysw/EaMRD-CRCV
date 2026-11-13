"""Summarize paired WWW baselines and archived Full runs without test selection."""

import csv
import json
import math
import statistics
from pathlib import Path

from run_www_baselines import OUT, ROOT, SEEDS, DATASETS, VARIANTS, read_metrics


def main() -> None:
    records = []
    for dataset in DATASETS:
        by_variant = {}
        for variant in (*VARIANTS, "full"):
            per_seed = {}
            for seed in SEEDS:
                run = (ROOT / "experiments" / "e1_infonce" / dataset / "full" / f"seed_{seed}") if variant == "full" else (OUT / dataset / variant / f"seed_{seed}")
                per_seed[seed] = read_metrics(run / "train.log")
            by_variant[variant] = per_seed
        for variant, per_seed in by_variant.items():
            acc = [per_seed[s]["Acc"] for s in SEEDS]
            f1 = [per_seed[s]["F1"] for s in SEEDS]
            full = by_variant["full"]
            deltas_acc = [full[s]["Acc"] - per_seed[s]["Acc"] for s in SEEDS]
            deltas_f1 = [full[s]["F1"] - per_seed[s]["F1"] for s in SEEDS]
            records.append({
                "dataset": dataset, "variant": variant, "n": len(SEEDS),
                "acc_mean": statistics.mean(acc), "acc_std": statistics.pstdev(acc),
                "f1_mean": statistics.mean(f1), "f1_std": statistics.pstdev(f1),
                "full_minus_variant_acc_mean": statistics.mean(deltas_acc),
                "full_minus_variant_f1_mean": statistics.mean(deltas_f1),
                "full_acc_wins": sum(v > 0 for v in deltas_acc),
            })
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print((OUT / "summary.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
