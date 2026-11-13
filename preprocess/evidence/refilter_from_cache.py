"""
Phase 1: Re-apply per-modality top-k to old JSONL entries (no API calls).
Reads verified_evidence_FakeSV.jsonl (old, global top-k) and the LLM score
cache, re-runs filter_vid() for each vid in the old file so the new
per-modality top-k logic is applied.  All scores come from cache → zero API.

Usage:
    python preprocess/evidence/refilter_from_cache.py
"""

import io, json, os, sys, traceback
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dotenv import load_dotenv
load_dotenv(override=True)

from tqdm import tqdm
from preprocess.evidence.evidence_filter import EventAwareEvidenceFilter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OLD_JSONL   = os.path.join(PROJECT_ROOT, "preprocess", "output", "verified_evidence_FakeSV.jsonl")
EVENT_JSONL = os.path.join(PROJECT_ROOT, "preprocess", "output", "event_summary_FakeSV.jsonl")
CLAIMS_DIR  = os.path.join(PROJECT_ROOT, "data", "cache", "FakeSV", "claims")
EVIDENCE_DIR= os.path.join(PROJECT_ROOT, "data", "cache", "FakeSV", "evidence", "raw")
CACHE_PATH  = os.path.join(PROJECT_ROOT, "data", "cache", "FakeSV", "evidence_filter_cache_v2.json")
OUT_V2      = os.path.join(PROJECT_ROOT, "preprocess", "output", "verified_evidence_FakeSV_v2.jsonl")

TOP_K     = 3
THRESHOLD = 0.4

# ── Load helpers ─────────────────────────────────────────────────────────────

def load_event_summaries(path):
    result = {}
    if not os.path.exists(path): return result
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
                vid = rec.get("vid") or rec.get("video_id")
                if vid: result[str(vid)] = rec
            except: pass
    return result

def load_old_vids(path):
    """Return ordered list of unique vids from old JSONL (last occurrence wins)."""
    vid_order, seen = [], set()
    if not os.path.exists(path): return []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                vid = json.loads(line)["vid"]
                if vid not in seen:
                    seen.add(vid)
                    vid_order.append(vid)
            except: pass
    return vid_order

def load_claims(vid):
    path = os.path.join(CLAIMS_DIR, f"{vid}.json")
    if not os.path.exists(path): return []
    try:
        d = json.load(open(path, encoding="utf-8", errors="replace"))
        claims = d.get("all_claims", [])
        return claims if isinstance(claims, list) else []
    except: return []

def load_evidences(vid):
    path = os.path.join(EVIDENCE_DIR, f"{vid}.json")
    if not os.path.exists(path): return []
    try:
        d = json.load(open(path, encoding="utf-8", errors="replace"))
        if isinstance(d, list): return d
        return d.get("evidence", d.get("evidences", d.get("results", [])))
    except: return []

def load_already_done_v2(path):
    done = set()
    if not os.path.exists(path): return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try: done.add(json.loads(line)["vid"])
            except: pass
    return done

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    api_key  = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://runapi.co/v1")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not set"); sys.exit(1)

    print("[Phase-1] Re-filtering old JSONL with per-modality top-k (from cache)...")
    event_summary = load_event_summaries(EVENT_JSONL)
    old_vids      = load_old_vids(OLD_JSONL)
    already_done  = load_already_done_v2(OUT_V2)

    todo = [v for v in old_vids if v not in already_done]
    print(f"  Old JSONL unique vids : {len(old_vids)}")
    print(f"  Already in v2         : {len(already_done)}")
    print(f"  To re-filter          : {len(todo)}", flush=True)

    filt = EventAwareEvidenceFilter(
        api_key=api_key, base_url=base_url,
        event_match_threshold=THRESHOLD, top_k=TOP_K,
        cache_path=CACHE_PATH,
    )

    os.makedirs(os.path.dirname(OUT_V2), exist_ok=True)
    success = fail = 0

    with open(OUT_V2, "a", encoding="utf-8") as out_f:
        for vid in tqdm(todo, desc="Re-filter (cache)"):
            ev_info   = event_summary.get(vid)
            if not ev_info:
                continue
            claims    = load_claims(vid)
            evidences = load_evidences(vid)
            if not claims:
                continue
            try:
                result = filt.filter_vid(
                    vid=vid, claims=claims, event_info=ev_info,
                    raw_evidences=evidences,
                )
                out_rec = {k: v for k, v in result.items()
                           if k not in ("all_scored", "claim_results")}
                out_rec["claim_summary"] = [
                    {"claim_id": cr["claim_id"], "modality": cr["modality"],
                     "n_evidence": cr["n_evidence"]}
                    for cr in result.get("claim_results", [])
                ]
                out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                out_f.flush()
                success += 1
            except Exception as exc:
                print(f"\n[WARN] {vid}: {exc}", flush=True)
                fail += 1

    filt.cache.flush()
    print(f"\n[Phase-1 DONE] success={success}  fail={fail}")
    print(f"  Output: {OUT_V2}", flush=True)


if __name__ == "__main__":
    main()
