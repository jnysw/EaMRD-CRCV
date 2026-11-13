"""
Merge old JSONL (re-apply per-modality top-k in pure Python) +
Run API calls only for truly new vids.

Phase A: Old 2579 vids → re-filter existing evidence list in Python (0 API calls)
Phase B: New ~1045 vids → call API via EventAwareEvidenceFilter

Usage:
    python preprocess/evidence/merge_and_complete.py
"""

import io, json, os, sys, time, traceback
from collections import defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dotenv import load_dotenv
load_dotenv(override=True)

from tqdm import tqdm
from preprocess.evidence.evidence_filter import EventAwareEvidenceFilter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OLD_JSONL    = os.path.join(PROJECT_ROOT, "preprocess", "output", "verified_evidence_FakeSV.jsonl")
EVENT_JSONL  = os.path.join(PROJECT_ROOT, "preprocess", "output", "event_summary_FakeSV.jsonl")
CLAIMS_DIR   = os.path.join(PROJECT_ROOT, "data", "cache", "FakeSV", "claims")
EVIDENCE_DIR = os.path.join(PROJECT_ROOT, "data", "cache", "FakeSV", "evidence", "raw")
CACHE_PATH   = os.path.join(PROJECT_ROOT, "data", "cache", "FakeSV", "evidence_filter_cache_v2.json")
SPLIT_TXT    = os.path.join(PROJECT_ROOT, "data", "cache", "FakeSV", "Split_target.txt")
OUT_V2       = os.path.join(PROJECT_ROOT, "preprocess", "output", "verified_evidence_FakeSV_v2.jsonl")

TOP_K     = 3
THRESHOLD = 0.4

# ── Helpers ───────────────────────────────────────────────────────────────────

def reapply_per_modality_topk(evidence_list, top_k=TOP_K, threshold=THRESHOLD):
    """Re-apply per-modality top-k to an already-scored evidence list."""
    passed = [e for e in evidence_list if e.get("event_match_score", 0) >= threshold]
    passed.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    modality_counts: dict = defaultdict(int)
    seen_urls: set = set()
    verified = []
    for ev in passed:
        modality = ev.get("claim_modality", "TTL")
        url = ev.get("url", "")
        if url in seen_urls:
            continue
        if modality_counts[modality] < top_k:
            seen_urls.add(url)
            verified.append(ev)
            modality_counts[modality] += 1
    return verified

def load_old_records(path):
    """Load old JSONL → {vid: last_record}."""
    recs = {}
    if not os.path.exists(path):
        return recs
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                recs[obj["vid"]] = obj
            except:
                pass
    return recs

def load_event_summaries(path):
    result = {}
    if not os.path.exists(path):
        return result
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

def load_done_v2(path):
    done = set()
    if not os.path.exists(path): return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try: done.add(json.loads(line)["vid"])
            except: pass
    return done

def load_split_vids(path):
    return [l.strip() for l in open(path, encoding="utf-8") if l.strip()]

def load_claims(vid):
    path = os.path.join(CLAIMS_DIR, f"{vid}.json")
    if not os.path.exists(path): return []
    try:
        d = json.load(open(path, encoding="utf-8", errors="replace"))
        c = d.get("all_claims", [])
        return c if isinstance(c, list) else []
    except: return []

def load_evidences(vid):
    path = os.path.join(EVIDENCE_DIR, f"{vid}.json")
    if not os.path.exists(path): return []
    try:
        d = json.load(open(path, encoding="utf-8", errors="replace"))
        if isinstance(d, list): return d
        return d.get("evidence", d.get("evidences", d.get("results", [])))
    except: return []

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    api_key  = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://runapi.co/v1")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not set"); sys.exit(1)

    print("Loading data...", flush=True)
    old_records  = load_old_records(OLD_JSONL)       # {vid: rec}
    event_summary= load_event_summaries(EVENT_JSONL)
    split_vids   = load_split_vids(SPLIT_TXT)
    done_v2      = load_done_v2(OUT_V2)

    old_vids_set = set(old_records.keys())
    split_set    = set(split_vids)

    # Phase A: old vids that are in split and not yet in v2
    phase_a = [v for v in split_vids if v in old_vids_set and v not in done_v2]
    # Phase B: new vids (in split, NOT in old, not in v2, have summary & evidence)
    phase_b = [
        v for v in split_vids
        if v not in old_vids_set and v not in done_v2
        and v in event_summary
        and os.path.exists(os.path.join(EVIDENCE_DIR, f"{v}.json"))
    ]

    print(f"Old JSONL unique vids: {len(old_records)}")
    print(f"Split vids:            {len(split_vids)}")
    print(f"Already in v2:         {len(done_v2)}")
    print(f"Phase A (re-filter):   {len(phase_a)}  [0 API calls, instant]")
    print(f"Phase B (new API):     {len(phase_b)}  [API calls needed]", flush=True)

    os.makedirs(os.path.dirname(OUT_V2), exist_ok=True)

    # ── Phase A ────────────────────────────────────────────────────────────────
    print("\n[Phase A] Re-applying per-modality top-k to old records...", flush=True)
    a_written = 0
    with open(OUT_V2, "a", encoding="utf-8") as out_f:
        for vid in tqdm(phase_a, desc="Phase A"):
            rec = dict(old_records[vid])
            old_ev = rec.get("verified_evidence", [])
            new_ev = reapply_per_modality_topk(old_ev, TOP_K, THRESHOLD)
            rec["verified_evidence"] = new_ev
            rec["n_verified"] = len(new_ev)
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            a_written += 1
    print(f"  Phase A done: {a_written} records written", flush=True)

    # ── Phase B ────────────────────────────────────────────────────────────────
    if not phase_b:
        print("\n[Phase B] Nothing new to process!")
    else:
        print(f"\n[Phase B] Processing {len(phase_b)} new vids via API...", flush=True)
        filt = EventAwareEvidenceFilter(
            api_key=api_key, base_url=base_url,
            event_match_threshold=THRESHOLD, top_k=TOP_K,
            cache_path=CACHE_PATH,
        )
        b_success = b_fail = 0
        start = time.time()
        with open(OUT_V2, "a", encoding="utf-8") as out_f:
            for vid in tqdm(phase_b, desc="Phase B (API)"):
                ev_info   = event_summary.get(vid)
                if not ev_info: continue
                claims    = load_claims(vid)
                evidences = load_evidences(vid)
                if not claims: continue
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
                    b_success += 1
                except Exception as exc:
                    print(f"\n[WARN] {vid}: {exc}", flush=True)
                    b_fail += 1
        filt.cache.flush()
        elapsed = time.time() - start
        print(f"\n  Phase B done: success={b_success} fail={b_fail} "
              f"time={elapsed/60:.1f}min", flush=True)

    # ── Summary ────────────────────────────────────────────────────────────────
    total = sum(1 for _ in open(OUT_V2, encoding="utf-8"))
    print(f"\n[COMPLETE] v2 JSONL total lines: {total}")
    print(f"  Output: {OUT_V2}", flush=True)


if __name__ == "__main__":
    main()
