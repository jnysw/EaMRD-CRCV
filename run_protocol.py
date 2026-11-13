"""
Pre-registered multi-seed experiment runner (E0 protocol).

    python run_protocol.py --exp configs/experiments/e1_infonce.yaml               # run everything
    python run_protocol.py --exp ... --dataset FakeTT --variants full,e1b_all      # subset
    python run_protocol.py --exp ... --dry-run                                     # list runs
    python run_protocol.py --exp ... --aggregate-only                              # tables only

Layout:  experiments/<name>/<dataset>/<variant>/seed_<s>/{config.yaml, stdout.txt, train.log,
         best_model.pth, preds_besttest.json}
Tables:  experiments/<name>/runs.csv             one row per run
         experiments/<name>/seed_comparison.csv  per seed, paired difference vs reference
         experiments/<name>/summary.csv          per dataset x variant, ranked by mean VAL acc

Rules enforced here:
  - seeds and the experiment file hash are written to protocol_lock.json on the first run;
    a later run with different seeds is refused (pass --relock to start a new protocol);
  - selection metric = validation acc at the early-stopping epoch; test acc/F1 are only
    reported (never used for ranking);
  - finished runs (BestTest present in stdout.txt) are skipped, so the script is resumable.
"""

import argparse
import copy
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
ANSI = re.compile(r"\x1b\[[0-9;]*m")
RE_VAL = re.compile(r"Val Loss=([\d.]+) Acc=([\d.]+) F1=([\d.]+)")
RE_TRAIN = re.compile(r"Train Loss=([\d.]+) Acc=([\d.]+) F1=([\d.]+)")
RE_TEST = re.compile(r"BestTest Loss=([\d.]+) Acc=([\d.]+) F1=([\d.]+)")


def apply_overrides(base: dict, overrides: dict) -> dict:
    cfg = copy.deepcopy(base)
    for dotted, val in (overrides or {}).items():
        node, keys = cfg, dotted.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = val
    return cfg


def parse_run(stdout: Path) -> dict | None:
    if not stdout.exists():
        return None
    t = ANSI.sub("", stdout.read_text(encoding="utf-8", errors="replace"))
    test = RE_TEST.search(t)
    if not test:
        return None
    vals = [float(m.group(2)) for m in RE_VAL.finditer(t)]
    val_f1 = [float(m.group(3)) for m in RE_VAL.finditer(t)]
    trains = [float(m.group(2)) for m in RE_TRAIN.finditer(t)]
    be = int(np.argmax(vals)) if vals else None   # EarlyStopping keeps the first strict max
    return {"epochs_run": len(vals), "best_epoch": be + 1 if be is not None else None,
            "val_acc": vals[be] if vals else None, "val_f1": val_f1[be] if vals else None,
            "train_acc_at_best": trains[be] if be is not None and be < len(trains) else None,
            "test_acc": float(test.group(2)), "test_f1": float(test.group(3))}


def check_lock(out_dir: Path, exp_path: Path, seeds: list, relock: bool):
    lock = out_dir / "protocol_lock.json"
    cur = {"seeds": seeds, "exp_file": str(exp_path),
           "exp_sha256": hashlib.sha256(exp_path.read_bytes()).hexdigest(),
           "locked_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if lock.exists() and not relock:
        old = json.loads(lock.read_text(encoding="utf-8"))
        if old["seeds"] != seeds:
            raise SystemExit(f"Seeds changed after lock {old['seeds']} -> {seeds}. "
                             f"Seeds are pre-registered; use a new experiment name or --relock.")
        if old["exp_sha256"] != cur["exp_sha256"]:
            print("  [note] experiment file changed since lock (variants added/edited); seeds unchanged.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps(cur, indent=2), encoding="utf-8")
    print(f"  [lock] seeds {seeds} registered -> {lock}")


def write_csv(path: Path, rows: list):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def aggregate(out_dir: Path, exp: dict, datasets: list):
    seeds, ref = exp["seeds"], exp["reference"]
    runs, comp, summ = [], [], []
    for ds in datasets:
        res = {}
        for var in exp["variants"]:
            for s in seeds:
                r = parse_run(out_dir / ds / var / f"seed_{s}" / "stdout.txt")
                if r:
                    res[(var, s)] = r
                    runs.append({"dataset": ds, "variant": var, "seed": s, **r})
        for var in exp["variants"]:
            done = [s for s in seeds if (var, s) in res]
            if not done:
                continue
            paired = [s for s in done if (ref, s) in res]
            for s in done:
                row = {"dataset": ds, "variant": var, "seed": s,
                       "val_acc": res[(var, s)]["val_acc"], "test_acc": res[(var, s)]["test_acc"],
                       "test_f1": res[(var, s)]["test_f1"]}
                if (ref, s) in res:
                    for k in ("val_acc", "test_acc", "test_f1"):
                        row[f"d_{k}_vs_{ref}"] = round(res[(var, s)][k] - res[(ref, s)][k], 4)
                comp.append(row)
            g = lambda k, ss=done: np.array([res[(var, s)][k] for s in ss])  # noqa: E731
            d = lambda k: np.array([res[(var, s)][k] - res[(ref, s)][k] for s in paired])  # noqa: E731
            summ.append({
                "dataset": ds, "variant": var, "n_seeds": len(done),
                "val_acc_mean": round(g("val_acc").mean(), 4), "val_acc_std": round(g("val_acc").std(), 4),
                "test_acc_mean": round(g("test_acc").mean(), 4), "test_acc_std": round(g("test_acc").std(), 4),
                "test_f1_mean": round(g("test_f1").mean(), 4), "test_f1_std": round(g("test_f1").std(), 4),
                "best_epoch_median": float(np.median(g("best_epoch"))),
                "n_paired": len(paired),
                f"d_val_acc_mean": round(d("val_acc").mean(), 4) if paired else None,
                f"d_test_acc_mean": round(d("test_acc").mean(), 4) if paired else None,
                f"d_test_acc_std": round(d("test_acc").std(), 4) if paired else None,
                f"d_test_f1_mean": round(d("test_f1").mean(), 4) if paired else None,
                "test_wins_vs_ref": int((d("test_acc") > 0).sum()) if paired else None,
                "complete": len(done) == len(seeds),
            })
    summ.sort(key=lambda r: (r["dataset"], -r["val_acc_mean"]))
    for r in summ:
        r["val_rank"] = 1 + sum(1 for x in summ if x["dataset"] == r["dataset"] and x["val_acc_mean"] > r["val_acc_mean"])
    write_csv(out_dir / "runs.csv", runs)
    write_csv(out_dir / "seed_comparison.csv", comp)
    write_csv(out_dir / "summary.csv", summ)

    print(f"\n{'=' * 104}\n  {exp['name']}  (ranked by mean VAL acc; test reported only; d* = paired vs '{ref}')\n{'=' * 104}")
    print(f"{'dataset':<7} {'variant':<18} {'n':>2} {'val':>15} {'test acc':>15} {'test F1':>15} "
          f"{'d_test':>8} {'wins':>5} {'rank':>4}")
    for r in summ:
        print(f"{r['dataset']:<7} {r['variant']:<18} {r['n_seeds']:>2} "
              f"{r['val_acc_mean']:.4f}+-{r['val_acc_std']:.4f} {r['test_acc_mean']:.4f}+-{r['test_acc_std']:.4f} "
              f"{r['test_f1_mean']:.4f}+-{r['test_f1_std']:.4f} "
              f"{(r['d_test_acc_mean'] if r['d_test_acc_mean'] is not None else float('nan')):>+8.4f} "
              f"{str(r['test_wins_vs_ref']):>5} {r['val_rank']:>4}")
    print(f"\n  tables: {out_dir / 'summary.csv'}, {out_dir / 'seed_comparison.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--dataset", default="all", help="FakeSV | FakeTT | all")
    ap.add_argument("--variants", default="all", help="comma-separated variant names or 'all'")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--relock", action="store_true", help="overwrite protocol_lock.json (new protocol)")
    args = ap.parse_args()

    exp_path = (ROOT / args.exp).resolve()
    exp = yaml.safe_load(exp_path.read_text(encoding="utf-8"))
    out_dir = ROOT / "experiments" / exp["name"]
    datasets = list(exp["base_configs"]) if args.dataset == "all" else [args.dataset]
    variants = list(exp["variants"]) if args.variants == "all" else args.variants.split(",")
    unknown = [v for v in variants if v not in exp["variants"]]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")

    if not args.aggregate_only:
        if not args.dry_run:
            check_lock(out_dir, exp_path, exp["seeds"], args.relock)
        todo = [(ds, v, s) for ds in datasets for v in variants for s in exp["seeds"]]
        print(f"  {len(todo)} runs: {datasets} x {variants} x seeds {exp['seeds']}")
        for i, (ds, var, seed) in enumerate(todo, 1):
            run_dir = out_dir / ds / var / f"seed_{seed}"
            stdout = run_dir / "stdout.txt"
            done = parse_run(stdout)
            if done:
                print(f"  [{i}/{len(todo)}] SKIP {ds}/{var}/seed_{seed}  val={done['val_acc']:.4f}")
                continue
            base = yaml.safe_load((ROOT / exp["base_configs"][ds]).read_text(encoding="utf-8"))
            cfg = apply_overrides(base, exp["variants"][var])
            cfg["seed"] = seed
            if args.dry_run:
                print(f"  [DRY] {ds}/{var}/seed_{seed}  overrides={exp['variants'][var]}")
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            cfg_path = run_dir / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
            print(f"  [{i}/{len(todo)}] RUN  {ds}/{var}/seed_{seed} ...", flush=True)
            t0 = time.time()
            proc = subprocess.run([PYTHON, "train.py", "--config", str(cfg_path), "--seed", str(seed),
                                   "--log-dir", str(run_dir)],
                                  capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
            stdout.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
            r = parse_run(stdout)
            if r is None:
                print(f"        FAILED (exit {proc.returncode}); see {stdout}")
            else:
                print(f"        val={r['val_acc']:.4f} test={r['test_acc']:.4f} "
                      f"best_ep={r['best_epoch']} ({time.time() - t0:.0f}s)")

    if not args.dry_run:
        aggregate(out_dir, exp, datasets)


if __name__ == "__main__":
    main()
