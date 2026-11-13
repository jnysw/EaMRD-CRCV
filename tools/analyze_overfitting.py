"""
Phase 1.3: overfitting / selection-noise diagnosis from EXISTING logs (no training).

Run groups (official Full, concat, same feature cache):
  seed_search_<ds>                  50 seeds, the only unbiased multi-seed sample
  ablation_removeone_20260825/full  the 3 seeds reported in the summary
  fusion_<ds>_concat_s2024          reference checkpoint used by later tools

Outputs (OUT_DIR):
  metrics_by_dataset_split.csv   per run: best epoch, train/val/test acc, gaps
  overfit_summary.csv            per group aggregate + test-set binomial SE + majority baseline
  label_distribution.csv
  overfit_curves/<ds>_seed_search.png
"""

import re

import numpy as np
import pandas as pd

from _common import OUT_DIR, ROOT, SPLITS, load_split, write_csv

ANSI = re.compile(r"\x1b\[[0-9;]*m")
RE = {k: re.compile(rf"{k} Loss=([\d.]+) Acc=([\d.]+) F1=([\d.]+)") for k in ("Train", "Val", "BestTest")}
GROUPS = {
    "FakeSV": {"seed_search": "logs/seed_search_fakesv",
               "reported_3seeds": "logs/ablation_removeone_20260825/fakesv_v2/full",
               "ref_s2024": "logs/fusion_fakesv_concat_s2024"},
    "FakeTT": {"seed_search": "logs/seed_search_fakett",
               "reported_3seeds": "logs/ablation_removeone_20260825/fakett_v2/full",
               "ref_s2024": "logs/fusion_fakett_concat_s2024"},
}


def curves(path):
    t = ANSI.sub("", path.read_text(encoding="utf-8", errors="replace"))
    c = {k: [(float(m.group(1)), float(m.group(2)), float(m.group(3))) for m in r.finditer(t)] for k, r in RE.items()}
    return c


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    (OUT_DIR / "overfit_curves").mkdir(exist_ok=True)
    runs, summ, curve_rows = [], [], []
    for ds, groups in GROUPS.items():
        for g, rel in groups.items():
            logs = sorted((ROOT / rel).rglob("train.log"))
            all_tr, all_va = [], []
            for p in logs:
                c = curves(p)
                if not c["BestTest"] or not c["Val"]:
                    continue
                va = [x[1] for x in c["Val"]]
                tr = [x[1] for x in c["Train"]]
                be = int(np.argmax(va))  # first max == strict-improvement EarlyStopping
                runs.append({"dataset": ds, "group": g, "log": str(p.relative_to(ROOT)),
                             "epochs_run": len(va), "best_epoch": be + 1,
                             "train_acc_best": tr[be], "val_acc_best": va[be],
                             "test_acc": c["BestTest"][0][1], "test_f1": c["BestTest"][0][2],
                             "train_loss_best": c["Train"][be][0], "val_loss_best": c["Val"][be][0],
                             "final_train_acc": tr[-1], "max_train_acc": max(tr),
                             "train_val_gap_best": round(tr[be] - va[be], 4),
                             "val_test_gap": round(va[be] - c["BestTest"][0][1], 4),
                             "val_loss_min_epoch": int(np.argmin([x[0] for x in c["Val"]])) + 1})
                all_tr.append(tr)
                all_va.append(va)
            if g == "seed_search" and all_tr:
                L = max(map(len, all_tr))
                pad = lambda a: np.array([x + [np.nan] * (L - len(x)) for x in a])  # noqa: E731
                T, V = pad(all_tr), pad(all_va)
                ep = np.arange(1, L + 1)
                for e in range(L):
                    curve_rows.append({"dataset": ds, "epoch": e + 1,
                                       "n_runs_alive": int((~np.isnan(V[:, e])).sum()),
                                       "train_acc_mean": round(float(np.nanmean(T[:, e])), 4),
                                       "train_acc_std": round(float(np.nanstd(T[:, e])), 4),
                                       "val_acc_mean": round(float(np.nanmean(V[:, e])), 4),
                                       "val_acc_std": round(float(np.nanstd(V[:, e])), 4)})
                if plt is None:
                    continue
                plt.figure(figsize=(6, 4))
                for M, name in ((T, "train acc"), (V, "val acc")):
                    mu, sd = np.nanmean(M, 0), np.nanstd(M, 0)
                    plt.plot(ep, mu, label=name)
                    plt.fill_between(ep, mu - sd, mu + sd, alpha=0.2)
                be = [r["best_epoch"] for r in runs if r["dataset"] == ds and r["group"] == g]
                plt.axvline(np.median(be), ls="--", c="gray", label=f"median best epoch={np.median(be):.0f}")
                plt.xlabel("epoch"); plt.ylabel("acc"); plt.title(f"{ds} Full, {len(T)} seeds (seed_search)")
                plt.legend(); plt.tight_layout()
                plt.savefig(OUT_DIR / "overfit_curves" / f"{ds}_seed_search.png", dpi=120)
                plt.close()

    write_csv(OUT_DIR / "overfit_curves" / "seed_search_epoch_curves.csv", curve_rows)
    r = pd.DataFrame(runs)
    write_csv(OUT_DIR / "metrics_by_dataset_split.csv", r.round(4).to_dict("records"))

    lab = []
    for ds in GROUPS:
        for sp in SPLITS:
            df = load_split(ds, sp)
            n, nf = len(df), int(df.label.sum())
            lab.append({"dataset": ds, "split": sp, "n": n, "n_fake": nf, "frac_fake": round(nf / n, 4),
                        "majority_acc": round(max(nf, n - nf) / n, 4)})
    write_csv(OUT_DIR / "label_distribution.csv", lab)
    labd = {(x["dataset"], x["split"]): x for x in lab}

    for (ds, g), x in r.groupby(["dataset", "group"]):
        n_test = labd[(ds, "test")]["n"]
        p = x.test_acc.mean()
        summ.append({"dataset": ds, "group": g, "n_runs": len(x),
                     "test_acc_mean": x.test_acc.mean(), "test_acc_std": x.test_acc.std(ddof=0),
                     "test_acc_min": x.test_acc.min(), "test_acc_max": x.test_acc.max(),
                     "test_f1_mean": x.test_f1.mean(), "test_f1_std": x.test_f1.std(ddof=0),
                     "val_acc_mean": x.val_acc_best.mean(),
                     "train_acc_at_best_mean": x.train_acc_best.mean(),
                     "max_train_acc_mean": x.max_train_acc.mean(),
                     "best_epoch_median": x.best_epoch.median(),
                     "epochs_run_median": x.epochs_run.median(),
                     "train_val_gap_mean": x.train_val_gap_best.mean(),
                     "corr_val_test": x[["val_acc_best", "test_acc"]].corr().iloc[0, 1] if len(x) > 2 else None,
                     "test_binomial_se": np.sqrt(p * (1 - p) / n_test),
                     "one_test_sample_acc": 1 / n_test,
                     "test_majority_acc": labd[(ds, "test")]["majority_acc"]})
    s = pd.DataFrame(summ).round(4)
    write_csv(OUT_DIR / "overfit_summary.csv", s.to_dict("records"))
    pd.set_option("display.width", 250)
    print(s.T.to_string())
    print(pd.DataFrame(lab).to_string())


if __name__ == "__main__":
    main()
