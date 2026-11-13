"""Run prespecified, paired internal baselines for the WWW audit paper.

Full checkpoints are in experiments/e1_infonce; this script trains reduced-input
models on exactly those splits, caches, seeds, and training hyperparameters.
It never chooses a model using test results.
"""

import copy
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "www_baselines_v1"
SEEDS = (123, 130, 456, 728, 2024)
DATASETS = ("FakeSV", "FakeTT")
VARIANTS = {
    "visual_only": {
        "use_claim": False,
        "use_evidence": False,
        "use_reasoning": False,
        "use_conflict": False,
        "use_gat": False,
        "use_ceca": False,
        "use_contrastive": False,
    },
    "visual_claim": {
        "use_claim": True,
        "use_evidence": False,
        "use_reasoning": False,
        "use_conflict": False,
        "use_gat": False,
        "use_ceca": False,
        "use_contrastive": False,
    },
    "flat_fusion": {
        "use_claim": True,
        "use_evidence": True,
        "use_reasoning": True,
        "use_conflict": False,
        "use_gat": False,
        "use_ceca": False,
        "use_contrastive": False,
    },
}
FINAL_RE = re.compile(r"Final (Acc|F1)\s*=([0-9.]+)")


def read_metrics(log_path: Path) -> dict[str, float]:
    pairs = FINAL_RE.findall(log_path.read_text(encoding="utf-8"))
    result = {k: float(v) for k, v in pairs}
    if set(result) != {"Acc", "F1"}:
        raise RuntimeError(f"Missing final metrics: {log_path}")
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in DATASETS:
        template_path = ROOT / "experiments" / "e1_infonce" / dataset / "full" / "seed_123" / "config.yaml"
        template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        for variant, switches in VARIANTS.items():
            for seed in SEEDS:
                run_dir = OUT / dataset / variant / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                config = copy.deepcopy(template)
                config["seed"] = seed
                config["model"].update(switches)
                config_path = run_dir / "config.yaml"
                config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
                log_path = run_dir / "train.log"
                if not log_path.exists() or "Final Acc=" not in log_path.read_text(encoding="utf-8"):
                    cmd = [sys.executable, "train.py", "--config", str(config_path), "--log-dir", str(run_dir)]
                    print(f"RUN {dataset} {variant} seed={seed}", flush=True)
                    with (run_dir / "stdout.txt").open("w", encoding="utf-8") as stdout:
                        subprocess.run(cmd, cwd=ROOT, stdout=stdout, stderr=subprocess.STDOUT, check=True)
                metrics = read_metrics(log_path)
                rows.append({"dataset": dataset, "variant": variant, "seed": seed,
                             "accuracy": metrics["Acc"], "macro_f1": metrics["F1"]})
                with (OUT / "results.csv").open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"DONE {dataset} {variant} seed={seed} Acc={metrics['Acc']:.4f} F1={metrics['F1']:.4f}", flush=True)
    manifest = {"source_full": "experiments/e1_infonce", "seeds": SEEDS,
                "datasets": DATASETS, "variants": VARIANTS,
                "selection": "fixed variants and seeds; checkpoint selected by validation accuracy"}
    (OUT / "protocol.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
