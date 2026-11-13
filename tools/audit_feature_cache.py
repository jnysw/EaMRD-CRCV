"""
Phase 1.1/1.2: read-only audit of the official Full feature caches.

Per sample (feature_per_sample.csv): number of claims / edges / non-padding
pos-neg-neu evidence rows, norms, NaN/Inf, shape consistency, within-video
near-duplicate claims & evidence, h_r zero.
Aggregates:
  feature_statistics.csv     dataset x split x label
  redundancy_statistics.csv  cross-sample identical vectors, no-evidence key
                             collisions, cross-split near-duplicate videos, event overlap
"""

import collections

import numpy as np
import torch
import torch.nn.functional as F

from _common import FEATURE_CACHE, OUT_DIR, SPLITS, load_feat, load_split, write_csv

DUP_COS = 0.95      # within-video near-duplicate threshold
XSPLIT_COS = 0.98   # cross-split near-duplicate threshold (video / claim mean)


def nz_rows(x):
    return int((x.abs().sum(-1) > 1e-8).sum().item()) if x is not None else 0


def max_offdiag_cos(x):
    """Return (#pairs with cos>DUP_COS, max off-diagonal cos) among non-zero rows."""
    if x is None:
        return 0, None
    x = x[x.abs().sum(-1) > 1e-8]
    if x.size(0) < 2:
        return 0, None
    s = F.normalize(x.float(), dim=-1) @ F.normalize(x.float(), dim=-1).T
    iu = torch.triu_indices(s.size(0), s.size(0), 1)
    v = s[iu[0], iu[1]]
    return int((v > DUP_COS).sum().item()), round(v.max().item(), 4)


def bucket(n):
    return "0" if n == 0 else "1" if n == 1 else "2-3" if n <= 3 else "4+"


def main():
    rows, vecs = [], collections.defaultdict(dict)
    for ds, cache in FEATURE_CACHE.items():
        for sp in SPLITS:
            df = load_split(ds, sp)
            ev_col = "event" if "event" in df.columns else None
            for _, r in df.iterrows():
                vid = r["video_id"]
                f = load_feat(cache, vid)
                row = {"dataset": ds, "split": sp, "vid": vid, "label": int(r["label"]),
                       "event": r[ev_col] if ev_col else None, "has_cache": f is not None}
                if f is None:
                    rows.append(row)
                    continue
                hcs, adj = f.get("h_c_seq"), f.get("graph_adj")
                n_c = nz_rows(hcs)
                n_nodes = hcs.size(0) if hcs is not None else 0
                if adj is not None and adj.numel() > 1:
                    off = adj - torch.diag(torch.diag(adj))
                    n_edges = int((off > 0).sum().item()) // 2
                    diag_ok = bool((torch.diag(adj)[:n_nodes] > 0).all().item())
                    adj_vals = off[off > 0]
                else:
                    n_edges, diag_ok, adj_vals = 0, bool(adj is not None and adj.numel() == 1 and adj.item() > 0), torch.tensor([])
                n_pos, n_neg = nz_rows(f.get("h_e_pos_seq")), nz_rows(f.get("h_e_neg_seq"))
                n_neu = nz_rows(f.get("h_e_neu_seq"))
                dup_c, max_c = max_offdiag_cos(hcs)
                ev_all = torch.cat([t for t in (f.get("h_e_pos_seq"), f.get("h_e_neg_seq")) if t is not None])
                dup_e, max_e = max_offdiag_cos(ev_all)
                allt = [v for v in f.values() if torch.is_tensor(v) and v.is_floating_point()]
                row.update({
                    "n_claims": n_c, "claim_bucket": bucket(n_c), "n_nodes": n_nodes,
                    "adj_dim": adj.size(0) if adj is not None else None,
                    "shape_ok": adj is not None and hcs is not None and adj.size(0) == hcs.size(0),
                    "diag_selfloop_ok": diag_ok, "n_edges": n_edges,
                    "density": round(2 * n_edges / (n_nodes * (n_nodes - 1)), 4) if n_nodes > 1 else None,
                    "mean_edge_w": round(adj_vals.mean().item(), 4) if adj_vals.numel() else None,
                    "n_pos": n_pos, "n_neg": n_neg, "n_neu": n_neu,
                    "no_pos": n_pos == 0, "no_pos_no_neg": n_pos == 0 and n_neg == 0,
                    "has_refute": n_neg > 0,
                    "h_r_zero": float(f["h_r"].abs().sum()) < 1e-8,
                    "h_v_zero": float(f["h_v"].abs().sum()) < 1e-8,
                    "nan_inf": any((~torch.isfinite(t)).any().item() for t in allt),
                    "norm_h_v": round(f["h_v"].norm().item(), 3),
                    "norm_h_c": round(f["h_c"].norm().item(), 3),
                    "norm_h_e_pos": round(f["h_e_pos"].norm().item(), 3),
                    "norm_h_r": round(f["h_r"].norm().item(), 3),
                    "dup_claim_pairs": dup_c, "max_claim_cos": max_c,
                    "dup_evidence_pairs": dup_e, "max_evidence_cos": max_e,
                    "dim_h_v": f["h_v"].numel(), "keys": "|".join(sorted(f.keys())),
                })
                rows.append(row)
                for k in ("h_v", "h_c", "h_r", "h_e_pos"):
                    vecs[(ds, k)][(sp, vid)] = f[k].float().flatten()
    write_csv(OUT_DIR / "feature_per_sample.csv", rows)

    import pandas as pd
    d = pd.DataFrame(rows)
    c = d[d.has_cache == True]  # noqa: E712
    agg = c.groupby(["dataset", "split", "label"]).agg(
        n=("vid", "size"),
        claims_mean=("n_claims", "mean"),
        claims_0=("claim_bucket", lambda s: (s == "0").mean()),
        claims_1=("claim_bucket", lambda s: (s == "1").mean()),
        claims_2_3=("claim_bucket", lambda s: (s == "2-3").mean()),
        claims_4p=("claim_bucket", lambda s: (s == "4+").mean()),
        claims_eq6=("n_claims", lambda s: (s >= 6).mean()),
        edges_mean=("n_edges", "mean"),
        frac_no_edges=("n_edges", lambda s: (s == 0).mean()),
        density_mean=("density", "mean"),
        pos_mean=("n_pos", "mean"), neg_mean=("n_neg", "mean"), neu_mean=("n_neu", "mean"),
        frac_no_pos=("no_pos", "mean"), frac_no_pos_no_neg=("no_pos_no_neg", "mean"),
        frac_has_refute=("has_refute", "mean"),
        frac_pos_eq5=("n_pos", lambda s: (s >= 5).mean()),
        frac_h_r_zero=("h_r_zero", "mean"), frac_nan_inf=("nan_inf", "mean"),
        frac_shape_bad=("shape_ok", lambda s: (~s.astype(bool)).mean()),
        frac_selfloop_bad=("diag_selfloop_ok", lambda s: (~s.astype(bool)).mean()),
        frac_dup_claim=("dup_claim_pairs", lambda s: (s > 0).mean()),
        frac_dup_evidence=("dup_evidence_pairs", lambda s: (s > 0).mean()),
        norm_h_v=("norm_h_v", "mean"), norm_h_c=("norm_h_c", "mean"),
        norm_h_e_pos=("norm_h_e_pos", "mean"), norm_h_r=("norm_h_r", "mean"),
    ).round(4).reset_index()
    miss = d.groupby(["dataset", "split", "label"])["has_cache"].apply(lambda s: int((~s.astype(bool)).sum())).reset_index(name="n_missing_cache")
    agg = agg.merge(miss, on=["dataset", "split", "label"], how="left")
    write_csv(OUT_DIR / "feature_statistics.csv", agg.to_dict("records"))
    print(agg.to_string())

    red = []
    for ds in FEATURE_CACHE:
        cd = c[c.dataset == ds]
        for k in ("h_v", "h_c", "h_r", "h_e_pos"):
            store = vecs[(ds, k)]
            keys = list(store)
            M = torch.stack([store[x] for x in keys])
            h = collections.Counter(tuple(np.round(v.numpy(), 5).tolist()).__hash__() for v in M)
            n_in_dup = sum(cnt for cnt in h.values() if cnt > 1)
            red.append({"dataset": ds, "check": f"identical_{k}_across_samples",
                        "value": n_in_dup, "of": len(keys),
                        "note": "samples sharing an exactly identical vector with >=1 other sample"})
        red.append({"dataset": ds, "check": "infonce_duplicate_pe_keys_no_pos",
                    "value": int(cd.no_pos.sum()), "of": len(cd),
                    "note": "no-pos samples -> CECA returns zeros -> identical pe key in InfoNCE"})
        tr = [x for x in vecs[(ds, "h_v")] if x[0] == "train"]
        Mtr = F.normalize(torch.stack([vecs[(ds, "h_v")][x] for x in tr]), dim=-1)
        Mtrc = F.normalize(torch.stack([vecs[(ds, "h_c")][x] for x in tr]), dim=-1)
        for sp in ("valid", "test"):
            ks = [x for x in vecs[(ds, "h_v")] if x[0] == sp]
            Mv = F.normalize(torch.stack([vecs[(ds, "h_v")][x] for x in ks]), dim=-1)
            mx = (Mv @ Mtr.T).max(1).values
            Mc = F.normalize(torch.stack([vecs[(ds, "h_c")][x] for x in ks]), dim=-1)
            mxc = (Mc @ Mtrc.T).max(1).values
            red.append({"dataset": ds, "check": f"{sp}_videos_near_dup_train_h_v(cos>{XSPLIT_COS})",
                        "value": int((mx > XSPLIT_COS).sum()), "of": len(ks),
                        "note": f"median max-cos={mx.median():.3f}"})
            red.append({"dataset": ds, "check": f"{sp}_videos_near_dup_train_h_c(cos>{XSPLIT_COS})",
                        "value": int((mxc > XSPLIT_COS).sum()), "of": len(ks),
                        "note": f"median max-cos={mxc.median():.3f}"})
            ev_tr = set(d[(d.dataset == ds) & (d.split == "train")].event.dropna())
            ev_sp = d[(d.dataset == ds) & (d.split == sp)].event
            red.append({"dataset": ds, "check": f"{sp}_samples_whose_event_in_train",
                        "value": int(ev_sp.isin(ev_tr).sum()), "of": len(ev_sp),
                        "note": "event id overlap (temporal split, events may legitimately recur)"})
    write_csv(OUT_DIR / "redundancy_statistics.csv", red)
    for r in red:
        print(r)


if __name__ == "__main__":
    main()
