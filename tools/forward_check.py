"""
Phase 0.2: minimal forward with the official Full configs (no training, read-only).

1. Build the model from the official yaml, run one real val batch, check shapes/finite.
2. Count trainable parameters per sub-module (after LazyLinear materialises).
3. Diff the official yaml against the Config line of a reference checkpoint's log,
   load that checkpoint (strict) and re-score val/test to confirm reproducibility.

Writes param_counts.csv and forward_check.json into OUT_DIR. Never writes into logs/.
"""

import ast
import json
import sys

import torch
import yaml

from _common import FULL_CFG, OUT_DIR, ROOT, write_csv

sys.argv = [sys.argv[0], "--log-dir", str(OUT_DIR / "_train_import_logdir")]
import train as T  # noqa: E402  (LOG_DIR redirected above; nothing is written there)

REF_RUN = {"FakeSV": "logs/fusion_fakesv_concat_s2024",
           "FakeTT": "logs/fusion_fakett_concat_s2024"}


def log_cfg(run_dir):
    for ln in (ROOT / run_dir / "train.log").read_text(encoding="utf-8", errors="replace").splitlines():
        if "Config: {" in ln:
            return ast.literal_eval(ln[ln.index("Config: ") + 8:].strip())
    return None


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{p}{k}."))
        else:
            out[f"{p}{k}"] = v
    return out


def make_loader(cfg, split):
    ds, col = T.build_dataset_and_collator(cfg, "temporal", split)
    return torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], collate_fn=col,
                                       shuffle=False, num_workers=0)


def forward(model, batch, cfg, dev):
    g = lambda k: batch[k].to(dev) if batch.get(k) is not None else None  # noqa: E731
    return model(g("h_v"), g("h_c"), g("h_e_pos"), g("h_r"),
                 h_e_neg=g("h_e_neg") if cfg["model"].get("contrastive_mode") == "hard_neg" else None,
                 h_c_seq=g("h_c_seq"), h_e_pos_seq=g("h_e_pos_seq"), h_e_neg_seq=g("h_e_neg_seq"),
                 graph_adj=g("graph_adj"), claim_mask=g("claim_mask"))


@torch.no_grad()
def score(model, loader, cfg, dev):
    correct = n = 0
    tp = fp = fn = tn = 0
    for b in loader:
        pred = forward(model, b, cfg, dev)["pred"].argmax(1).cpu()
        y = b["labels"]
        correct += (pred == y).sum().item()
        n += len(y)
        tp += ((pred == 1) & (y == 1)).sum().item(); fp += ((pred == 1) & (y == 0)).sum().item()
        fn += ((pred == 0) & (y == 1)).sum().item(); tn += ((pred == 0) & (y == 0)).sum().item()
    f1p = 2 * tp / max(2 * tp + fp + fn, 1)
    f1n = 2 * tn / max(2 * tn + fn + fp, 1)
    return {"acc": round(correct / n, 4), "macro_f1": round((f1p + f1n) / 2, 4), "n": n}


def count_params(model):
    rows, total = [], 0
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters()
                if p.requires_grad and not isinstance(p, torch.nn.parameter.UninitializedParameter))
        total += n
        rows.append({"module": name, "type": type(mod).__name__, "trainable_params": n})
    own = sum(p.numel() for n_, p in model.named_parameters(recurse=False) if p.requires_grad)
    if own:
        rows.append({"module": "<own>", "type": "", "trainable_params": own})
        total += own
    return rows, total


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    report, prow = {}, []
    for ds, cfg_rel in FULL_CFG.items():
        cfg = yaml.safe_load((ROOT / cfg_rel).read_text(encoding="utf-8"))
        T.set_seed(cfg["seed"])
        model = T.build_model(cfg).to(dev).eval()
        vl = make_loader(cfg, "valid")
        batch = next(iter(vl))
        with torch.no_grad():
            out = forward(model, batch, cfg, dev)
        shapes = {k: list(v.shape) for k, v in out.items() if torch.is_tensor(v)}
        finite = all(torch.isfinite(v).all().item() for v in out.values() if torch.is_tensor(v))
        rows, total = count_params(model)
        for r in rows:
            prow.append({"dataset": ds, **r, "share": round(r["trainable_params"] / total, 4)})
        n_train = len(make_loader(cfg, "train").dataset)

        ref = REF_RUN[ds]
        lc = log_cfg(ref)
        diff = {}
        if lc:
            a, b = flat(cfg), flat(lc)
            for k in sorted(set(a) | set(b)):
                if a.get(k) != b.get(k):
                    diff[k] = {"official_yaml": a.get(k), "ref_log": b.get(k)}
        ref_model = T.build_model(lc or cfg).to(dev).eval()
        with torch.no_grad():
            forward(ref_model, batch, lc or cfg, dev)   # materialise LazyLinear
        sd = torch.load(ROOT / ref / "best_model.pth", map_location=dev, weights_only=False)
        missing, unexpected = ref_model.load_state_dict(sd, strict=False)
        rescored = {sp: score(ref_model, make_loader(lc or cfg, sp), lc or cfg, dev)
                    for sp in ("valid", "test")}
        report[ds] = {"official_cfg": cfg_rel, "batch_keys": sorted(batch.keys()),
                      "out_shapes": shapes, "finite": finite,
                      "trainable_params_total": total, "n_train": n_train,
                      "params_per_train_sample": round(total / n_train, 1),
                      "ref_checkpoint": f"{ref}/best_model.pth",
                      "cfg_diff_official_vs_ref": diff,
                      "ckpt_missing_keys": list(missing), "ckpt_unexpected_keys": list(unexpected),
                      "ref_rescored": rescored}
        print(ds, json.dumps(report[ds], indent=1, default=str))
    write_csv(OUT_DIR / "param_counts.csv", prow)
    (OUT_DIR / "forward_check.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
