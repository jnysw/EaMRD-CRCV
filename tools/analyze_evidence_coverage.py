"""
Phase 1.1: Claim-Evidence lineage, polarity semantics, provenance and reasoning audit
(read-only; raw json/jsonl only, no API calls).

Outputs (OUT_DIR):
  evidence_coverage.csv        per dataset x split x label aggregate
  evidence_relation_query.csv  relation x query_type x label counts
  reasoning_vs_label.csv       overall_assessment / decision x label
  refute_sites.csv             top siteName for refute evidence by label (site names only)
  evidence_per_video.csv       per-video flags used by later tools
"""

import collections

import pandas as pd

from _common import (CACHE_ROOT, OUT_DIR, RAW_CACHE, SPLITS, VERIFIED_EV, claim_list,
                     load_json, load_jsonl_by_vid, load_split, write_csv)

MAX_CLAIMS, EV_K = 6, 5
FAKE_DECISIONS = {"false", "fake", "misleading"}


def to_ts(x):
    if not x:
        return pd.NaT
    t = pd.to_datetime(x, errors="coerce", utc=True)
    return t


def main():
    per_video, rq, rs, sites = [], collections.Counter(), collections.Counter(), collections.Counter()
    for ds in ("FakeSV", "FakeTT"):
        ev = load_jsonl_by_vid(VERIFIED_EV[ds])
        raw = CACHE_ROOT / RAW_CACHE[ds]
        for sp in SPLITS:
            df = load_split(ds, sp)
            for _, r in df.iterrows():
                vid, y = r["video_id"], int(r["label"])
                claims = claim_list(load_json(raw / "claims" / f"{vid}.json"))
                cid_rank = {c.get("id"): i for i, c in enumerate(claims)}
                kept_ids = {c.get("id") for c in claims[:MAX_CLAIMS]}
                rec = ev.get(vid) or {}
                items = rec.get("verified_evidence") or []
                sup = [e for e in items if e.get("relation") == "support"]
                ref = [e for e in items if e.get("relation") == "refute"]
                neu = [e for e in items if e.get("relation") == "neutral"]
                enc = sup[:EV_K] + ref[:EV_K]
                vt = to_ts(r.get("publish_time")) if "publish_time" in r else pd.NaT
                n_after = n_dated = 0
                for e in items:
                    rq[(ds, sp, y, e.get("relation"), e.get("query_type"))] += 1
                    if e.get("relation") == "refute":
                        sites[(ds, y, (e.get("siteName") or "?")[:40])] += 1
                    t = to_ts(e.get("datePublished"))
                    if pd.notna(t):
                        n_dated += 1
                        if pd.notna(vt) and t > (vt.tz_localize("UTC") if vt.tzinfo is None else vt):
                            n_after += 1
                rsn = load_json(raw / "reasoning_v2" / f"{vid}.json") or {}
                ke = rsn.get("key_evidence_used") or []
                ke_ids = [k.get("claim_id") for k in ke if isinstance(k, dict)]
                dec = str(rsn.get("decision", "")).strip().lower()
                per_video.append({
                    "dataset": ds, "split": sp, "vid": vid, "label": y,
                    "n_claims_total": len(claims), "n_claims_dropped": max(len(claims) - MAX_CLAIMS, 0),
                    "in_verified_jsonl": vid in ev, "n_retrieved": rec.get("n_retrieved"),
                    "n_items": len(items), "n_support": len(sup), "n_refute": len(ref), "n_neutral": len(neu),
                    "n_support_truncated": max(len(sup) - EV_K, 0), "n_refute_truncated": max(len(ref) - EV_K, 0),
                    "enc_items": len(enc),
                    "enc_items_with_claim_id": sum(1 for e in enc if e.get("claim_id")),
                    "enc_items_claim_id_in_claims": sum(1 for e in enc if e.get("claim_id") in cid_rank),
                    "enc_items_claim_dropped_by_trunc": sum(1 for e in enc if e.get("claim_id") in cid_rank and e.get("claim_id") not in kept_ids),
                    "enc_distinct_claims": len({e.get("claim_id") for e in enc if e.get("claim_id")}),
                    "items_with_url": sum(1 for e in items if e.get("url")),
                    "items_dated": n_dated, "items_after_video": n_after,
                    "has_video_time": pd.notna(vt),
                    "reasoning_present": bool(rsn),
                    "assessment": rsn.get("overall_assessment"), "decision": rsn.get("decision"),
                    "llm_pred_fake": (dec in FAKE_DECISIONS) if dec else None,
                    "llm_confidence": rsn.get("confidence"),
                    "n_key_evidence": len(ke),
                    "key_ev_bad_claim_id": sum(1 for k in ke_ids if k and k not in cid_rank),
                    "key_ev_without_any_verified": len(ke) > 0 and len(items) == 0,
                    "key_ev_refute_claims": sum(1 for k in ke if isinstance(k, dict) and k.get("relation") == "refute"),
                    "n_conflicts": len(rsn.get("cross_modal_conflicts") or []),
                })
                rs[(ds, sp, y, "assessment", str(rsn.get("overall_assessment")))] += 1
                rs[(ds, sp, y, "decision", str(rsn.get("decision")))] += 1

    d = pd.DataFrame(per_video)
    write_csv(OUT_DIR / "evidence_per_video.csv", d.to_dict("records"))

    def agg(g):
        enc = g.enc_items.sum()
        has_t = g[g.has_video_time]
        llm = g[g.llm_pred_fake.notna()]
        return pd.Series({
            "n": len(g),
            "frac_in_verified_jsonl": g.in_verified_jsonl.mean(),
            "frac_no_items": (g.n_items == 0).mean(),
            "frac_no_support": (g.n_support == 0).mean(),
            "frac_has_refute": (g.n_refute > 0).mean(),
            "frac_refute_only": ((g.n_refute > 0) & (g.n_support == 0)).mean(),
            "frac_neutral_only": ((g.n_neutral > 0) & (g.n_support == 0) & (g.n_refute == 0)).mean(),
            "mean_support": g.n_support.mean(), "mean_refute": g.n_refute.mean(), "mean_neutral": g.n_neutral.mean(),
            "frac_videos_support_truncated": (g.n_support_truncated > 0).mean(),
            "frac_videos_claims_truncated": (g.n_claims_dropped > 0).mean(),
            "enc_items_claim_id_rate": g.enc_items_with_claim_id.sum() / max(enc, 1),
            "enc_items_claim_id_match_rate": g.enc_items_claim_id_in_claims.sum() / max(enc, 1),
            "enc_items_for_dropped_claim": g.enc_items_claim_dropped_by_trunc.sum() / max(enc, 1),
            "mean_distinct_claims_in_enc_evidence": g[g.enc_items > 0].enc_distinct_claims.mean(),
            "url_rate": g.items_with_url.sum() / max(g.n_items.sum(), 1),
            "dated_rate": g.items_dated.sum() / max(g.n_items.sum(), 1),
            "after_video_rate_of_dated": (has_t.items_after_video.sum() / max(has_t.items_dated.sum(), 1)) if len(has_t) else None,
            "frac_videos_with_post_video_evidence": (has_t.items_after_video > 0).mean() if len(has_t) else None,
            "reasoning_present": g.reasoning_present.mean(),
            "llm_decision_acc": ((llm.llm_pred_fake.astype(bool)).astype(int) == llm.label).mean() if len(llm) else None,
            "frac_unverifiable": (g.assessment == "Unverifiable").mean(),
            "frac_unverifiable_if_no_support": (g[g.n_support == 0].assessment == "Unverifiable").mean() if (g.n_support == 0).any() else None,
            "llm_pred_fake_if_no_items": g[g.n_items == 0].llm_pred_fake.astype(float).mean() if (g.n_items == 0).any() else None,
            "frac_key_ev_bad_claim_id": (g.key_ev_bad_claim_id > 0).mean(),
            "frac_key_ev_without_verified": g.key_ev_without_any_verified.mean(),
        })

    cov = d.groupby(["dataset", "split", "label"]).apply(agg).round(4).reset_index()
    write_csv(OUT_DIR / "evidence_coverage.csv", cov.to_dict("records"))
    pd.set_option("display.width", 250)
    print(cov.T.to_string())

    write_csv(OUT_DIR / "evidence_relation_query.csv",
              [{"dataset": k[0], "split": k[1], "label": k[2], "relation": k[3], "query_type": k[4], "count": v}
               for k, v in sorted(rq.items(), key=lambda kv: str(kv[0]))])
    write_csv(OUT_DIR / "reasoning_vs_label.csv",
              [{"dataset": k[0], "split": k[1], "label": k[2], "field": k[3], "value": k[4], "count": v}
               for k, v in sorted(rs.items(), key=lambda kv: str(kv[0]))])
    top = []
    for ds in ("FakeSV", "FakeTT"):
        for y in (0, 1):
            items = sorted(((k[2], v) for k, v in sites.items() if k[0] == ds and k[1] == y), key=lambda x: -x[1])[:15]
            top += [{"dataset": ds, "label": y, "siteName": s, "refute_count": c} for s, c in items]
    write_csv(OUT_DIR / "refute_sites.csv", top)

    rq_df = pd.DataFrame([{"dataset": k[0], "label": k[2], "relation": k[3], "query_type": k[4], "count": v} for k, v in rq.items()])
    print(rq_df.pivot_table(index=["dataset", "relation", "query_type"], columns="label", values="count", aggfunc="sum").to_string())
    rs_df = pd.DataFrame([{"dataset": k[0], "label": k[2], "field": k[3], "value": k[4], "count": v} for k, v in rs.items()])
    print(rs_df[rs_df.field == "assessment"].pivot_table(index=["dataset", "value"], columns="label", values="count", aggfunc="sum").to_string())
    print(rs_df[rs_df.field == "decision"].pivot_table(index=["dataset", "value"], columns="label", values="count", aggfunc="sum").to_string())


if __name__ == "__main__":
    main()
