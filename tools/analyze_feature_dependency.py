"""
Phase 1.4: perturbation sensitivity of an existing Full checkpoint (read-only, no training).

For each branch representation entering the concat fusion (pv, pc, pe, pr) replace it by
its TRAIN-set mean and re-run only the classifier. Report dAcc / dF1 / mean |d logit|
on val and test, overall and on evidence subsets (has_refute / support_only / no_evidence).
This is perturbation sensitivity, NOT causal contribution.

Also:
  - shortcut probes: logistic regression on evidence-availability metadata only
    (n_support, n_refute, n_neutral, no_items) and on the LLM decision only, fit on train,
    evaluated once on val/test;
  - per-sample predictions with vids (the logs' preds_besttest.json has empty vids);
  - error cases (high-confidence and borderline misclassifications) for 03_case_analysis.

Outputs: feature_dependency.csv, shortcut_probes.csv, preds_ref_<ds>.csv, error_cases.csv
"""

import sys

import numpy as np
import pandas as pd
import torch
import yaml

from _common import FULL_CFG, OUT_DIR, ROOT, CACHE_ROOT, RAW_CACHE, load_json, sha256_text, write_csv

sys.argv = [sys.argv[0], "--log-dir", str(OUT_DIR / "_train_import_logdir")]
import train as T  # noqa: E402
from forward_check import REF_RUN, make_loader  # noqa: E402

BRANCHES = ("pv", "pc", "pe", "pr")


@torch.no_grad()
def branch_reprs(model, loader, dev):
    out = {k: [] for k in BRANCHES}
    out["y"], out["vid"] = [], []
    for b in loader:
        g = lambda k: b[k].to(dev)  # noqa: E731
        pv = model.proj_v(g("h_v"))
        pc = model._encode_claims(g("h_c"), g("h_c_seq"), g("graph_adj"), g("claim_mask"))
        pe, _, _, _ = model._encode_evidence(g("h_e_pos"), torch.zeros_like(g("h_e_pos")), g("h_c_seq"),
                                             g("h_e_pos_seq"), None, g("claim_mask"))
        pr = model.proj_r(g("h_r"))
        for k, v in zip(BRANCHES, (pv, pc, pe, pr)):
            out[k].append(v)
        out["y"].append(b["labels"])
        out["vid"] += list(b["vids"])
    return {k: (torch.cat(v) if k != "vid" else v) for k, v in out.items()}


def metrics(logits, y):
    p = logits.argmax(1).cpu()
    acc = (p == y).float().mean().item()
    f1s = []
    for c in (0, 1):
        tp = ((p == c) & (y == c)).sum().item()
        f1s.append(2 * tp / max(2 * tp + ((p == c) & (y != c)).sum().item() + ((p != c) & (y == c)).sum().item(), 1))
    return acc, float(np.mean(f1s))


@torch.no_grad()
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = pd.read_csv(OUT_DIR / "evidence_per_video.csv", dtype={"vid": str})
    ev["subset"] = np.where((ev.n_support == 0) & (ev.n_refute == 0), "no_evidence",
                            np.where(ev.n_refute > 0, "has_refute", "support_only"))
    rows, probe_rows, cases = [], [], []
    for ds, cfg_rel in FULL_CFG.items():
        cfg = yaml.safe_load(open(cfg_rel, encoding="utf-8"))
        T.set_seed(cfg["seed"])
        model = T.build_model(cfg).to(dev).eval()
        tr_loader = make_loader(cfg, "train")
        with torch.no_grad():
            b0 = next(iter(tr_loader))
            model(*(b0[k].to(dev) for k in ("h_v", "h_c", "h_e_pos", "h_r")), h_c_seq=b0["h_c_seq"].to(dev),
                  h_e_pos_seq=b0["h_e_pos_seq"].to(dev), h_e_neg_seq=b0["h_e_neg_seq"].to(dev),
                  graph_adj=b0["graph_adj"].to(dev), claim_mask=b0["claim_mask"].to(dev))
        model.load_state_dict(torch.load(ROOT / REF_RUN[ds] / "best_model.pth", map_location=dev, weights_only=False))
        model.eval()
        R = {sp: branch_reprs(model, make_loader(cfg, sp) if sp != "train" else tr_loader, dev)
             for sp in ("train", "valid", "test")}
        means = {k: R["train"][k].mean(0, keepdim=True) for k in BRANCHES}
        evd = ev[ev.dataset == ds].set_index("vid")

        for sp in ("valid", "test"):
            r = R[sp]
            base = model.classifier(torch.cat([r[k] for k in BRANCHES], -1))
            prob = torch.softmax(base, 1)[:, 1].cpu().numpy()
            subset = evd.loc[r["vid"], "subset"].values
            pd.DataFrame({"vid": r["vid"], "label": r["y"].numpy(), "p_fake": prob.round(4), "subset": subset}) \
                .to_csv(OUT_DIR / f"preds_ref_{ds}_{sp}.csv", index=False)
            for sub in ("all", "has_refute", "support_only", "no_evidence"):
                m = np.ones(len(subset), bool) if sub == "all" else (subset == sub)
                if m.sum() == 0:
                    continue
                mt = torch.from_numpy(m)
                a0, f0 = metrics(base[mt.to(base.device)], r["y"][mt])
                row = {"dataset": ds, "split": sp, "subset": sub, "n": int(m.sum()),
                       "frac_fake": round(r["y"][mt].float().mean().item(), 4),
                       "acc": round(a0, 4), "f1": round(f0, 4),
                       "mean_p_fake": round(float(prob[m].mean()), 4)}
                for k in BRANCHES:
                    parts = [means[j].expand_as(r[j]) if j == k else r[j] for j in BRANCHES]
                    lg = model.classifier(torch.cat(parts, -1))
                    a, f = metrics(lg[mt.to(lg.device)], r["y"][mt])
                    dl = (lg[:, 1] - lg[:, 0] - (base[:, 1] - base[:, 0])).abs()[mt.to(lg.device)].mean().item()
                    row.update({f"dAcc_{k}": round(a - a0, 4), f"dF1_{k}": round(f - f0, 4), f"dLogit_{k}": round(dl, 3)})
                rows.append(row)

            if sp == "test":
                y = r["y"].numpy()
                wrong = (prob >= 0.5).astype(int) != y
                conf = np.abs(prob - 0.5)
                for tag, idx in (("high_conf_error", np.where(wrong & (conf >= 0.4))[0]),
                                 ("borderline_error", np.where(wrong & (conf < 0.1))[0])):
                    for i in idx[np.argsort(-conf[idx])][:8]:
                        vid = r["vid"][i]
                        e = evd.loc[vid]
                        rs = load_json(CACHE_ROOT / RAW_CACHE[ds] / "reasoning_v2" / f"{vid}.json") or {}
                        ke = rs.get("key_evidence_used") or []
                        cases.append({"dataset": ds, "type": tag, "case_id": sha256_text(vid)[:10],
                                      "label": int(y[i]), "p_fake": round(float(prob[i]), 3),
                                      "subset": e.subset, "n_claims": int(e.n_claims_total),
                                      "n_support": int(e.n_support), "n_refute": int(e.n_refute),
                                      "n_neutral": int(e.n_neutral),
                                      "llm_assessment": rs.get("overall_assessment"), "llm_decision": rs.get("decision"),
                                      "llm_conf": rs.get("confidence"),
                                      "key_evidence": ";".join(f"{k.get('claim_id')}:{k.get('relation')}" for k in ke if isinstance(k, dict)),
                                      "reason_excerpt": str(rs.get("reason", ""))[:160].replace("\n", " ")})

        # Shortcut probes (fit on train only, evaluate once on val/test)
        from sklearn.linear_model import LogisticRegression
        d = ev[ev.dataset == ds].copy()
        d["no_items"] = (d.n_items == 0).astype(int)
        d["llm_fake"] = d.llm_pred_fake.astype(float).fillna(0.5)
        probes = {"evidence_availability_only": ["n_support", "n_refute", "n_neutral", "no_items"],
                  "llm_decision_only": ["llm_fake"],
                  "availability+llm_decision": ["n_support", "n_refute", "n_neutral", "no_items", "llm_fake"]}
        for name, cols in probes.items():
            tr = d[d.split == "train"]
            clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(tr[cols], tr.label)
            for sp in ("valid", "test"):
                te = d[d.split == sp]
                p = clf.predict(te[cols])
                f1 = np.mean([2 * ((p == c) & (te.label == c)).sum() / max((p == c).sum() + (te.label == c).sum(), 1) for c in (0, 1)])
                probe_rows.append({"dataset": ds, "probe": name, "split": sp, "n": len(te),
                                   "acc": round(float((p == te.label).mean()), 4), "macro_f1": round(float(f1), 4)})

    df = pd.DataFrame(rows)
    write_csv(OUT_DIR / "feature_dependency.csv", df.to_dict("records"))
    write_csv(OUT_DIR / "shortcut_probes.csv", probe_rows)
    write_csv(OUT_DIR / "error_cases.csv", cases)
    pd.set_option("display.width", 250)
    print(df[["dataset", "split", "subset", "n", "frac_fake", "acc", "mean_p_fake"] + [f"dAcc_{k}" for k in BRANCHES] + [f"dLogit_{k}" for k in BRANCHES]].to_string())
    print(pd.DataFrame(probe_rows).to_string())
    print(pd.DataFrame(cases).groupby(["dataset", "type", "subset"]).size().to_string())


if __name__ == "__main__":
    main()
