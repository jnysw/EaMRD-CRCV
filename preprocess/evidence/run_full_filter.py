"""
Full-run: Event-aware Evidence Filtering for all Split_target videos
====================================================================
- Reads vid list from Split_target.txt
- Skips vids already written to output JSONL (resumable)
- Appends results to verified_evidence_FakeSV.jsonl
- Logs progress to logs/run_full_filter.log

Usage:
    python preprocess/evidence/run_full_filter.py
    python preprocess/evidence/run_full_filter.py --dataset FakeSV --workers 1
"""

import argparse
import io
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dotenv import load_dotenv
load_dotenv(override=True)

from tqdm import tqdm

from preprocess.evidence.evidence_filter import EventAwareEvidenceFilter

# ─── Paths ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATHS = {
    "FakeSV": {
        "split_target":        os.path.join(PROJECT_ROOT, "data", "cache", "FakeSV", "Split_target.txt"),
        "evidence_dir":        os.path.join(PROJECT_ROOT, "data", "cache", "FakeSV", "evidence", "raw"),
        "claims_dir":          os.path.join(PROJECT_ROOT, "data", "cache", "FakeSV", "claims"),
        "event_summary_jsonl": os.path.join(PROJECT_ROOT, "preprocess", "output", "event_summary_FakeSV.jsonl"),
        "cache_path":          os.path.join(PROJECT_ROOT, "data", "cache", "FakeSV", "evidence_filter_cache_v2.json"),
        "output_jsonl":        os.path.join(PROJECT_ROOT, "preprocess", "output", "verified_evidence_FakeSV.jsonl"),
    },
    "FakeTT": {
        "FakeTT_data":        r"F:\Bishe_workspace\FakeTT\data.json",
        "evidence_dir":        os.path.join(PROJECT_ROOT, "data", "cache", "FakeTT", "evidence", "raw"),
        "claims_dir":          os.path.join(PROJECT_ROOT, "data", "cache", "FakeTT", "claims"),
        "event_summary_jsonl": os.path.join(PROJECT_ROOT, "preprocess", "output", "event_summary_FakeTT.jsonl"),
        "cache_path":          os.path.join(PROJECT_ROOT, "data", "cache", "FakeTT", "evidence_filter_cache_v2_en.json"),
        "output_jsonl":        os.path.join(PROJECT_ROOT, "preprocess", "output", "verified_evidence_FakeTT_v3.jsonl"),
    },
    "WildFake": {
        "FakeTT_data":        r"F:\Bishe_workspace\WildFake\data_zh.jsonl",
        "evidence_dir":        os.path.join(PROJECT_ROOT, "data", "cache", "WildFake", "evidence", "raw"),
        "claims_dir":          os.path.join(PROJECT_ROOT, "data", "cache", "WildFake", "claims"),
        "event_summary_jsonl": os.path.join(PROJECT_ROOT, "preprocess", "output", "event_summary_WildFake.jsonl"),
        "cache_path":          os.path.join(PROJECT_ROOT, "data", "cache", "WildFake", "evidence_filter_cache_zh.json"),
        "output_jsonl":        os.path.join(PROJECT_ROOT, "preprocess", "output", "verified_evidence_WildFake.jsonl"),
    },
}

# ─── Loaders ─────────────────────────────────────────────────────────────────

def load_split_vids(path, columns=None):
    """Load video IDs from a plain text file OR a JSONL file.

    For JSONL files (e.g. FakeTT data.json), pass columns=["video_id"] to
    specify which key to extract; falls back to "vid" if the key is absent.
    Plain text files (one vid per line) are handled regardless of `columns`.
    """
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
    # Detect JSONL
    try:
        rec = json.loads(first.strip())
        if isinstance(rec, dict):
            key = (columns[0] if columns else None) or "video_id"
            vids = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        vid = r.get(key) or r.get("vid") or r.get("video_id")
                        if vid:
                            vids.append(str(vid))
                    except Exception:
                        pass
            return vids
    except Exception:
        pass
    # Plain text
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def load_event_summaries(jsonl_path):
    result = {}
    if not os.path.exists(jsonl_path):
        return result
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                vid = rec.get("vid") or rec.get("video_id")
                if vid:
                    result[str(vid)] = rec
            except Exception:
                pass
    return result

def load_done_vids(output_jsonl):
    done = set()
    if not os.path.exists(output_jsonl):
        return done
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                vid = json.loads(line).get("vid", "")
                if vid:
                    done.add(str(vid))
            except Exception:
                pass
    return done

def load_claims(claims_dir, vid):
    path = os.path.join(claims_dir, f"{vid}.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            d = json.load(f)
        claims = d.get("all_claims", [])
        return claims if isinstance(claims, list) else []
    except Exception:
        return []

def load_evidences(evidence_dir, vid):
    path = os.path.join(evidence_dir, f"{vid}.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            d = json.load(f)
        if isinstance(d, list):
            return d
        return d.get("evidence", d.get("evidences", d.get("results", [])))
    except Exception:
        return []

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",   default="FakeSV", choices=["FakeSV","FakeTT","WildFake"])
    ap.add_argument("--top-k",     type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--max-ev",    type=int, default=10,
                    help="Max evidence per (claim, modality) batch")
    ap.add_argument("--workers",   type=int, default=1,
                    help="Parallel video workers (default=1)")
    ap.add_argument("--output",    type=str, default=None,
                    help="Override output JSONL path")
    ap.add_argument("--limit",     type=int, default=0,
                    help="Process only first N videos (0 = all)")
    args = ap.parse_args()

    paths    = dict(PATHS[args.dataset])  # copy so we can override
    if args.output:
        paths["output_jsonl"] = args.output
    api_key  = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://runapi.co/v1")

    if not api_key:
        print("[ERROR] OPENAI_API_KEY not set"); sys.exit(1)

    # Load data
    print(f"[INFO] Loading {args.dataset}...")
    if args.dataset in ("FakeTT", "WildFake"):
        split_vids = load_split_vids(paths["FakeTT_data"], columns=["video_id"])
    else:
        split_vids    = load_split_vids(paths["split_target"])
    event_summary = load_event_summaries(paths["event_summary_jsonl"])
    done_vids     = load_done_vids(paths["output_jsonl"])

    print(f"[INFO] Split total:      {len(split_vids)}")
    print(f"[INFO] Event summaries:  {len(event_summary)}")
    print(f"[INFO] Already done:     {len(done_vids)}")

    # Build todo list
    todo = [
        v for v in split_vids
        if v not in done_vids
        and v in event_summary
        and os.path.exists(os.path.join(paths["evidence_dir"], f"{v}.json"))
    ]
    skipped_no_summary = sum(
        1 for v in split_vids
        if v not in done_vids and v not in event_summary
    )
    skipped_no_evidence = sum(
        1 for v in split_vids
        if v not in done_vids and v in event_summary
        and not os.path.exists(os.path.join(paths["evidence_dir"], f"{v}.json"))
    )
    print(f"[INFO] To process:       {len(todo)}")
    print(f"[INFO] Skip (no summary):{skipped_no_summary}")
    print(f"[INFO] Skip (no evidence):{skipped_no_evidence}")
    print(f"[INFO] Output: {paths['output_jsonl']}", flush=True)

    if args.limit > 0:
        todo = todo[:args.limit]
        print(f"[INFO] Limited to first {args.limit} videos")

    if not todo:
        print("[INFO] Nothing to do!")
        return

    # Build filter
    lang = "en" if args.dataset in ("FakeTT",) else "zh"
    filt = EventAwareEvidenceFilter(
        api_key=api_key,
        base_url=base_url,
        event_match_threshold=args.threshold,
        top_k=args.top_k,
        max_evidence_per_modality=args.max_ev,
        cache_path=paths["cache_path"],
        lang=lang,
    )

    os.makedirs(os.path.dirname(paths["output_jsonl"]), exist_ok=True)

    stats = {"success": 0, "failed": 0, "total_retrieved": 0,
             "total_passed": 0, "total_kept": 0}
    start_time = time.time()
    import threading
    _write_lock = threading.Lock()

    def process_one_vid(vid):
        try:
            ev_info   = event_summary[vid]
            claims    = load_claims(paths["claims_dir"], vid)
            evidences = load_evidences(paths["evidence_dir"], vid)

            if not claims:
                return vid, {"vid": vid, "n_claims": 0, "n_retrieved": 0,
                             "n_passed_threshold": 0, "n_verified": 0,
                             "verified_evidence": [], "skip_reason": "no_claims"}, False

            result = filt.filter_vid(
                vid=vid,
                claims=claims,
                event_info=ev_info,
                raw_evidences=evidences,
            )
            out_rec = {k: v for k, v in result.items()
                       if k not in ("all_scored", "claim_results")}
            out_rec["claim_summary"] = [
                {"claim_id": cr["claim_id"], "modality": cr["modality"],
                 "n_evidence": cr["n_evidence"]}
                for cr in result.get("claim_results", [])
            ]
            return vid, out_rec, True
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"\n[WARN] {vid}: {exc}\n{tb}", flush=True)
            return vid, None, False

    with open(paths["output_jsonl"], "a", encoding="utf-8") as out_f:
        if args.workers == 1:
            for vid in tqdm(todo, desc=f"FilterEvidence [{args.dataset}]"):
                vid, out_rec, ok = process_one_vid(vid)
                if out_rec:
                    out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                if ok:
                    stats["success"] += 1
                else:
                    stats["failed"] += 1
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(process_one_vid, v): v for v in todo}
                for fut in tqdm(as_completed(futures), total=len(todo),
                                desc=f"FilterEvidence [{args.dataset}]"):
                    vid, out_rec, ok = fut.result()
                    if out_rec:
                        with _write_lock:
                            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                            out_f.flush()
                    if ok:
                        stats["success"] += 1
                    else:
                        stats["failed"] += 1

    filt.cache.flush()

    elapsed = time.time() - start_time
    n = stats["success"]
    print(f"\n[DONE] success={n}  failed={stats['failed']}  "
          f"time={elapsed/60:.1f}min  avg={elapsed/max(n,1):.1f}s/vid", flush=True)
    if n > 0:
        print(f"  Avg retrieved: {stats['total_retrieved']/n:.1f}")
        print(f"  Avg passed:    {stats['total_passed']/n:.1f}")
        print(f"  Avg kept:      {stats['total_kept']/n:.1f}")
        print(f"  Pass rate:     {100*stats['total_passed']/max(stats['total_retrieved'],1):.1f}%")
    print(f"  Output: {paths['output_jsonl']}", flush=True)


if __name__ == "__main__":
    main()
