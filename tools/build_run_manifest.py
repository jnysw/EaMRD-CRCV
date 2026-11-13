"""
Phase 0.2: build run_manifest.csv from every existing train.log (read-only).

For each log: dataset, feature_cache, seed, config hash, model flags, split sizes
seen at train time, whether those sizes match the CURRENT split files, best epoch
(epoch whose val Acc was a strict new max -> the checkpoint EarlyStopping saved),
val Acc at that epoch, BestTest Acc/F1, epochs run, log mtime.

Also writes trace_fakett_full.csv: every FakeTT run with the official Full
feature cache, to trace the reported 0.8384+-0.0016 vs seed-2024 0.8161.
"""

import ast
import datetime as dt
import json
import re

from _common import OUT_DIR, ROOT, SPLITS, load_split, sha256_text, split_hash, write_csv

RE_CFG = re.compile(r"Config: (\{.*\})\s*$")
RE_SIZE = re.compile(r"\[(\w+)/(train|valid|test)\] cache=(\S+) (\d+) -> (\d+)")
RE_VAL = re.compile(r"Val Loss=([\d.]+) Acc=([\d.]+) F1=([\d.]+)")
RE_TEST = re.compile(r"BestTest Loss=([\d.]+) Acc=([\d.]+) F1=([\d.]+)")
RE_TRAIN = re.compile(r"Train Loss=([\d.]+) Acc=([\d.]+) F1=([\d.]+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")

MODEL_KEYS = ["hidden_dim", "dropout", "lambda_contrastive", "contrastive_mode", "use_gat",
              "use_ceca", "use_claim", "use_evidence", "use_reasoning", "use_conflict",
              "fusion_type", "use_evidence_gate", "use_ce_interaction"]


def current_split_sizes():
    out = {}
    for ds in ("FakeSV", "FakeTT"):
        for sp in SPLITS:
            df = load_split(ds, sp)
            out[(ds, sp)] = (len(df), split_hash(df))
    return out


def parse_log(path, cur_sizes):
    text = ANSI.sub("", path.read_text(encoding="utf-8", errors="replace"))
    lines = text.splitlines()
    cfg = None
    for ln in lines:
        m = RE_CFG.search(ln)
        if m:
            try:
                cfg = ast.literal_eval(m.group(1))
            except Exception:  # noqa: BLE001
                cfg = None
            break
    sizes = {}
    for m in RE_SIZE.finditer(text):
        sizes[m.group(2)] = int(m.group(5))
    vals = [float(m.group(2)) for m in RE_VAL.finditer(text)]
    trains = [float(m.group(2)) for m in RE_TRAIN.finditer(text)]
    test = RE_TEST.search(text)

    best_epoch, best_val = None, None
    for i, v in enumerate(vals):
        if best_val is None or v > best_val:   # EarlyStopping saves on strict improvement
            best_val, best_epoch = v, i + 1

    row = {"log_path": str(path.relative_to(ROOT)),
           "mtime": dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")}
    if cfg:
        m = cfg.get("model", {})
        ds = cfg.get("dataset")
        row.update({"dataset": ds, "feature_cache": cfg.get("feature_cache") or ds,
                    "seed": cfg.get("seed"), "type": cfg.get("type"),
                    "lr": cfg.get("optimizer", {}).get("lr"),
                    "weight_decay": cfg.get("optimizer", {}).get("weight_decay"),
                    "batch_size": cfg.get("batch_size"),
                    "use_class_weight": cfg.get("use_class_weight", False),
                    "config_hash": sha256_text(json.dumps(cfg, sort_keys=True, default=str))[:16]})
        for k in MODEL_KEYS:
            row[f"m_{k}"] = m.get(k)
        match = []
        for sp in SPLITS:
            if sp in sizes and (ds, sp) in cur_sizes:
                match.append(sizes[sp] == cur_sizes[(ds, sp)][0])
        row["split_size_match_current"] = all(match) if match else None
        if all(match) and match:
            row["split_hash_current"] = "|".join(cur_sizes[(ds, sp)][1][:8] for sp in SPLITS)
    row.update({"train_size": sizes.get("train"), "val_size": sizes.get("valid"),
                "test_size": sizes.get("test"), "epochs_run": len(vals),
                "best_epoch": best_epoch, "selection_metric": "val_acc",
                "best_val_acc": best_val,
                "train_acc_at_best": trains[best_epoch - 1] if best_epoch and len(trains) >= best_epoch else None,
                "final_train_acc": trains[-1] if trains else None,
                "acc": float(test.group(2)) if test else None,
                "f1": float(test.group(3)) if test else None})
    ckpt = path.parent / "best_model.pth"
    row["checkpoint"] = str(ckpt.relative_to(ROOT)) if ckpt.exists() else ""
    return row


def main():
    cur = current_split_sizes()
    logs = sorted(p for p in ROOT.rglob("train.log")
                  if "experiments" not in p.parts and "__pycache__" not in p.parts)
    rows = [parse_log(p, cur) for p in logs]
    write_csv(OUT_DIR / "run_manifest.csv", rows)
    done = [r for r in rows if r.get("acc") is not None]
    print(f"logs={len(rows)} with BestTest={len(done)}")

    # Trace FakeTT official Full feature cache runs
    trace = [r for r in done if r.get("dataset") == "FakeTT"
             and r.get("feature_cache") == "FakeTT_FilteredEv_v3"
             and r.get("m_use_gat") and r.get("m_use_ceca")
             and r.get("m_use_claim", True) and r.get("m_use_evidence", True)
             and r.get("m_use_reasoning", True)
             and (r.get("m_fusion_type") in (None, "concat"))]
    trace.sort(key=lambda r: (str(r.get("config_hash")), str(r.get("seed"))))
    write_csv(OUT_DIR / "trace_fakett_full.csv", trace)
    print(f"FakeTT Full (FakeTT_FilteredEv_v3, concat, GAT+CECA) runs: {len(trace)}")
    for r in trace:
        print(f"  seed={r['seed']} lr={r['lr']} bs={r['batch_size']} do={r.get('m_dropout')} "
              f"cw={r['use_class_weight']} acc={r['acc']} f1={r['f1']} cfg={r['config_hash']} "
              f"{r['mtime']} {r['log_path']}")

    trace_sv = [r for r in done if r.get("dataset") == "FakeSV"
                and r.get("feature_cache") == "FakeSV_FilteredEv"
                and r.get("m_use_gat") and r.get("m_use_ceca")
                and r.get("m_use_claim", True) and r.get("m_use_evidence", True)
                and r.get("m_use_reasoning", True)
                and (r.get("m_fusion_type") in (None, "concat"))]
    write_csv(OUT_DIR / "trace_fakesv_full.csv", trace_sv)
    print(f"FakeSV Full (FakeSV_FilteredEv, concat, GAT+CECA) runs: {len(trace_sv)}")


if __name__ == "__main__":
    main()
