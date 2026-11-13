"""
Step 5: Evidence Credibility Modeling
  Score each evidence item on three dimensions:
    1. Source Reliability  — domain whitelist (authoritative_sources.py)
    2. NLI Alignment       — DeBERTa cross-encoder (support / refute / neutral)
    3. Temporal Consistency — penalty for stale evidence

  Score(eⱼ) = 0.4×source + 0.4×NLI_align + 0.2×time

  Output: data/cache/{dataset}/evidence/scored/{vid}.json
          Fields used by 06_run_reasoning.py:
            - top_semantic:     [{source_domain, date, score, summary, url, ...}]
            - top_contradiction: same
            - top_k:            merged list (up to 2K)

Usage:
    python preprocess/05_score_evidence.py --dataset FakeSV --top_k 5 --limit 50
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from utils.authoritative_sources import ALL_AUTHORITY_DOMAINS

CACHE_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
TODAY = datetime.now(tz=timezone.utc)

# Domain → reliability tier score (0–1)
DOMAIN_TIERS: dict[str, float] = {
    "piyao.org.cn": 1.0,
    "news.cn": 0.95,
    "people.com.cn": 0.95,
    "news.cctv.com": 0.95,
    "chinanews.com.cn": 0.9,
    "factpaper.cn": 1.0,
    "fact.qq.com": 1.0,
    "nhc.gov.cn": 0.9,
    "mps.gov.cn": 0.9,
    "mem.gov.cn": 0.9,
    "cma.gov.cn": 0.85,
    "ceic.ac.cn": 0.85,
    "ndcpa.gov.cn": 0.9,
    "kepuchina.cn": 0.8,
    "jyb.cn": 0.75,
    "12377.cn": 0.85,
}
DEFAULT_DOMAIN_SCORE = 0.4


def get_domain_score(url: str) -> float:
    for domain, score in DOMAIN_TIERS.items():
        if domain in url:
            return score
    return DEFAULT_DOMAIN_SCORE


def get_temporal_score(date_str: str, days_threshold: int = 365 * 3) -> float:
    """Return 1.0 for recent evidence, exponential decay for older."""
    if not date_str:
        return 0.5  # unknown date, neutral
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            pub = datetime.strptime(date_str[:19], fmt[:len(date_str[:19])])
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            age_days = (TODAY - pub).days
            if age_days < 0:
                age_days = 0
            # Score decays from 1.0 → 0.2 over threshold
            score = max(0.2, 1.0 - 0.8 * (age_days / days_threshold))
            return score
        except ValueError:
            continue
    return 0.5


def load_nli_model():
    """Load DeBERTa NLI cross-encoder (lazy import to avoid startup cost)."""
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(
        "cross-encoder/nli-deberta-v3-base",
        num_labels=3,
        device="cuda",
    )
    return model


def get_nli_score(model, claim: str, evidence_text: str) -> dict[str, float]:
    """Returns {support, refute, neutral} probabilities."""
    import torch
    import torch.nn.functional as F

    score = model.predict([(claim, evidence_text)], convert_to_numpy=False)
    probs = F.softmax(score[0].detach().float(), dim=-1).tolist()
    # DeBERTa NLI label order: contradiction(0), entailment(1), neutral(2)
    return {"refute": probs[0], "support": probs[1], "neutral": probs[2]}


def score_evidence_item(item: dict, nli_model) -> float:
    """Compute weighted credibility score for a single evidence item."""
    src = get_domain_score(item.get("url", ""))
    temporal = get_temporal_score(item.get("datePublished", ""))
    text = (item.get("summary") or item.get("snippet") or "")[:512]
    claim = item.get("claim", "")

    if text and claim:
        nli = get_nli_score(nli_model, claim, text)
        nli_val = nli["support"] - nli["refute"] * 0.5  # support bonus, refute penalty
    else:
        nli_val = 0.0
        nli = {"support": 0.0, "refute": 0.0, "neutral": 1.0}

    score = 0.4 * src + 0.4 * max(0.0, nli_val) + 0.2 * temporal
    item["_score"] = score
    item["_nli"] = nli
    item["_src_score"] = src
    item["_temporal_score"] = temporal
    return score


def extract_domain(url: str) -> str:
    """Extract base domain from URL string."""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc
        return host.replace("www.", "")
    except Exception:
        return url[:50]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["FakeSV", "FakeTT"], default="FakeSV")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    raw_dir = os.path.join(CACHE_ROOT, args.dataset, "evidence", "raw")
    scored_dir = os.path.join(CACHE_ROOT, args.dataset, "evidence", "scored")
    os.makedirs(scored_dir, exist_ok=True)

    done = {os.path.splitext(f)[0] for f in os.listdir(scored_dir) if f.endswith(".json")}
    raw_files = [f for f in os.listdir(raw_dir) if f.endswith(".json")]
    todo = [f for f in raw_files if os.path.splitext(f)[0] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"Raw evidence: {len(raw_files)}, Done: {len(done)}, Todo: {len(todo)}")

    if not todo:
        print("All done.")
        return

    nli_model = load_nli_model()

    for fname in tqdm(todo, desc="Scoring evidence"):
        vid = os.path.splitext(fname)[0]
        raw_path = os.path.join(raw_dir, fname)

        with open(raw_path, encoding="utf-8") as f:
            data = json.load(f)

        evidence = data.get("evidence", [])
        for item in evidence:
            # Normalise fields expected by downstream steps
            item.setdefault("source_domain", extract_domain(item.get("url", "")))
            item.setdefault("date", item.get("datePublished", "unknown"))
            item.setdefault("summary", item.get("snippet", item.get("description", "")))
            item.setdefault("claim", item.get("claim_text", ""))
            score_evidence_item(item, nli_model)

        # Sort by score descending
        evidence.sort(key=lambda x: x.get("_score", 0.0), reverse=True)

        # Separate semantic and contradiction for balanced top-K
        semantic = [e for e in evidence if e.get("query_type") == "semantic"][:args.top_k]
        contradiction = [e for e in evidence if e.get("query_type") == "contradiction"][:args.top_k]
        top_k = (semantic + contradiction)[:args.top_k * 2]  # Keep up to 2K for hard-neg

        # Build compact output that downstream scripts and feature encoder need
        def slim(ev_list):
            keep = ["claim_id", "claim_text", "query_type", "source_domain", "date",
                    "summary", "url", "_score", "_nli"]
            return [{k: e[k] for k in keep if k in e} for e in ev_list]

        out_path = os.path.join(scored_dir, f"{vid}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "vid": vid,
                "top_semantic":      slim(semantic),
                "top_contradiction": slim(contradiction),
                "top_k":             slim(top_k),
            }, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
