"""Independently verify saved predictions and paired test IDs for WWW table."""

import json
from pathlib import Path

from run_www_baselines import OUT, ROOT, DATASETS, SEEDS, VARIANTS, read_metrics


def metrics_from_predictions(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = [int(v) for v in data["labels"]]
    preds = [int(p >= 0.5) for p in data["probs"]]
    assert len(labels) == len(preds) == len(data["vids"])
    assert len(set(data["vids"])) == len(labels)
    acc = sum(a == b for a, b in zip(labels, preds)) / len(labels)
    f1 = []
    for cls in (0, 1):
        tp = sum(a == cls and b == cls for a, b in zip(labels, preds))
        fp = sum(a != cls and b == cls for a, b in zip(labels, preds))
        fn = sum(a == cls and b != cls for a, b in zip(labels, preds))
        f1.append(2 * tp / (2 * tp + fp + fn))
    return data["vids"], labels, acc, sum(f1) / 2


def main():
    checked = 0
    for dataset in DATASETS:
        reference = None
        for variant in (*VARIANTS, "full"):
            for seed in SEEDS:
                run = (ROOT / "experiments" / "e1_infonce" / dataset / "full" / f"seed_{seed}") if variant == "full" else (OUT / dataset / variant / f"seed_{seed}")
                vids, labels, acc, f1 = metrics_from_predictions(run / "preds_besttest.json")
                logged = read_metrics(run / "train.log")
                assert abs(acc - logged["Acc"]) < 0.00006, (dataset, variant, seed, "acc")
                assert abs(f1 - logged["F1"]) < 0.00006, (dataset, variant, seed, "f1")
                if reference is None:
                    reference = (vids, labels)
                else:
                    assert (vids, labels) == reference, (dataset, variant, seed, "test examples differ")
                checked += 1
        print(f"{dataset}: {len(reference[0])} common test videos")
    print(f"Verified {checked} runs: accuracy, macro-F1, and ordered test examples agree.")


if __name__ == "__main__":
    main()
