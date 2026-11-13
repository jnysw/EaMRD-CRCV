"""
Step 4: M2 — Adaptive Multi-view Evidence Retrieval

For each claim, generates:
  - Semantic Query:      positive evidence search
  - Contradiction Query: debunking / refutation search

Prioritizes claims by: priority = conflict_score × confidence
High-priority claims get up to 2 Bocha search calls; others get 1.

** OPTIMIZED: 1 LLM call per video (batch all claims), not 1 per claim **

Output: data/cache/{dataset}/evidence/raw/{vid}.json

Usage:
    python preprocess/04_retrieve_evidence.py --dataset FakeSV --limit 50
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from tqdm import tqdm

from utils.bocha_search import bocha_authoritative_search
from utils.serper_search import serper_search_multi
from utils.chat_llm import ChatLLM
from utils.cot_common import safe_read_jsonl

load_dotenv(override=True)

CACHE_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "cache")

# ──────────────────────────────────────────────
# Batch prompts: all claims for ONE video in a single LLM call
# ──────────────────────────────────────────────
SYSTEM_PROMPT_BATCH = """\
You are a fact-checking query generator for a rumor detection system.

**QUERY LANGUAGE: {lang_instruction}**

Given a list of factual claims extracted from a single short video, generate two types of search queries for EACH claim:

1. SEMANTIC QUERY: A concise web search query to find factual information that directly relates to or verifies the claim. Focus on key entities, events, locations, and dates.

2. CONTRADICTION QUERY: A web search query specifically designed to find evidence that contradicts, debunks, or fact-checks the claim. Include debunking terms in the query language.

Rules:
- Queries must be concise (5–10 words ideal).
- Include specific named entities, locations, dates if present in the claim.
- The contradiction query should actively surface debunking content.
- The contradiction_query MUST be meaningfully different from the semantic_query.
- Output ONLY a valid JSON array — one object per claim. No explanation, no markdown."""

USER_PROMPT_BATCH = """\
Generate search queries for ALL the following claims from the same video.

Claims:
{claims_block}

Output a JSON array with one object per claim, in the same order:
[
  {{
    "claim_id": "<id>",
    "semantic_query": "...",
    "contradiction_query": "...",
    "priority": <float>
  }},
  ...
]"""

LANG_INSTRUCTIONS_QUERY = {
    "FakeSV":   "查询语言为中文。Contradiction query 必须包含辟谣/谣言/虚假/不实/假新闻等中文辟谣词汇。",
    "FakeTT":   "Generate queries in English. Contradiction query must include terms like: fake / false / hoax / rumor / debunked / fact-check.",
    "WildFake": "查询语言为中文。Contradiction query 必须包含辟谣/谣言/虚假/不实/假新闻等中文辟谣词汇。",
}


def get_batch_system_prompt(dataset: str) -> str:
    return SYSTEM_PROMPT_BATCH.format(
        lang_instruction=LANG_INSTRUCTIONS_QUERY.get(dataset, "Use the same language as the claim.")
    )


def load_claims(vid: str, claims_dir: str) -> list[dict]:
    path = os.path.join(claims_dir, f"{vid}.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("all_claims", [])


def load_claim_graph(vid: str, graph_dir: str) -> dict:
    path = os.path.join(graph_dir, f"{vid}.json")
    if not os.path.exists(path):
        return {"edges": [], "max_conflict_score": 0.0}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_claim_conflict_score(claim_id: str, graph: dict) -> float:
    """Max contradiction score for a given claim across all its edges."""
    edges = graph.get("edges", [])
    scores = [
        e["conflict_score"]
        for e in edges
        if (e["source_id"] == claim_id or e["target_id"] == claim_id)
    ]
    return max(scores) if scores else 0.0


def build_claims_block(claim_requests: list[dict]) -> str:
    """Format all claims into a numbered list for the batch prompt."""
    lines = []
    for i, req in enumerate(claim_requests, 1):
        lines.append(
            f"{i}. [{req['claim_id']}] ({req['source_modality']}, "
            f"conf={req['_claim_obj'].get('confidence', 0.5):.2f}, "
            f"conflict={req['conflict_score']:.2f}, "
            f"priority={req['priority_score']:.3f})\n"
            f"   Claim: {req['claim_text']}"
        )
    return "\n".join(lines)


def parse_batch_output(raw: str, claim_requests: list[dict]) -> list[dict]:
    """Parse LLM batch output (JSON array). Falls back per-claim on failure."""
    raw = (raw or "").strip()
    # strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    def make_fallback(req: dict) -> dict:
        claim_text = req.get("claim_text", "")
        is_chinese = bool(claim_text) and "\u4e00" <= claim_text[0] <= "\u9fff"
        return {
            "claim_id": req["claim_id"],
            "semantic_query": claim_text[:200],
            "contradiction_query": claim_text[:150] + (" 谣言辟谣" if is_chinese else " fake debunked"),
            "priority": req["priority_score"],
        }

    # Try parsing as JSON array
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and len(parsed) == len(claim_requests):
            # merge each item with its fallback to fill missing keys
            result = []
            for item, req in zip(parsed, claim_requests):
                fallback = make_fallback(req)
                result.append({**fallback, **item})
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting first JSON array via regex
    m = re.search(r'\[[\s\S]*\]', raw)
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                result = []
                # match by claim_id if possible
                id_map = {r["claim_id"]: r for r in claim_requests}
                for item in parsed:
                    cid = item.get("claim_id", "")
                    req = id_map.get(cid, claim_requests[0] if claim_requests else {})
                    fallback = make_fallback(req)
                    result.append({**fallback, **item})
                # pad with fallbacks if count mismatch
                parsed_ids = {i.get("claim_id") for i in parsed}
                for req in claim_requests:
                    if req["claim_id"] not in parsed_ids:
                        result.append(make_fallback(req))
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # Full fallback
    return [make_fallback(req) for req in claim_requests]


def search_for_claim(query_data: dict, count: int = 5, use_serper: bool = False) -> list[dict]:
    """Run both Semantic and Contradiction queries, tag results.
    FakeTT uses Serper (Google, English); FakeSV uses Bocha (Chinese)."""
    results = []
    for qtype in ("semantic", "contradiction"):
        q = query_data.get(f"{qtype}_query", "")
        if not q:
            continue
        if use_serper:
            raw = serper_search_multi([q], count=count)
        else:
            raw, _ = bocha_authoritative_search(query=q, count=count)
        for item in raw:
            item["query_type"] = qtype
            item["claim_id"] = query_data.get("claim_id", "")
            item["claim_text"] = ""  # filled by caller
        results.extend(raw)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["FakeSV", "FakeTT", "WildFake"], default="FakeSV")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--search-api", choices=["auto", "bocha", "serper"], default="auto",
                        help="Search backend: auto=bocha for FakeSV, serper for FakeTT/WildFake")
    args = parser.parse_args()

    claims_dir = os.path.join(CACHE_ROOT, args.dataset, "claims")
    graph_dir  = os.path.join(CACHE_ROOT, args.dataset, "claim_graphs")
    raw_dir    = os.path.join(CACHE_ROOT, args.dataset, "evidence", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    done       = {os.path.splitext(f)[0] for f in os.listdir(raw_dir) if f.endswith(".json")}
    claim_vids = [os.path.splitext(f)[0] for f in os.listdir(claims_dir) if f.endswith(".json")]
    todo       = [v for v in claim_vids if v not in done]
    if args.limit:
        todo = todo[:args.limit]
    # Determine search backend
    if args.search_api == "auto":
        use_serper = (args.dataset in ("FakeTT",))
    else:
        use_serper = (args.search_api == "serper")
    backend = "Serper (Google)" if use_serper else "Bocha"
    print(f"Claims: {len(claim_vids)}  Done: {len(done)}  Todo: {len(todo)}")
    print(f"Search backend: {backend}")

    system_prompt = get_batch_system_prompt(args.dataset)
    # ChatLLM is used with a raw prompt string; we override per-video below
    client = ChatLLM(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions"),
        key=os.getenv("OPENAI_API_KEY", ""),
        prompt="",   # overridden per call via direct API
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.3,
        max_tokens=1024,   # more tokens needed for multi-claim array
    )

    for vid in tqdm(todo, desc="Retrieving evidence"):
        claims = load_claims(vid, claims_dir)
        if not claims:
            with open(os.path.join(raw_dir, f"{vid}.json"), "w", encoding="utf-8") as f:
                json.dump({"vid": vid, "evidence": []}, f)
            continue

        graph = load_claim_graph(vid, graph_dir)

        # Build priority for each claim
        claim_requests = []
        for c in claims:
            cid      = c.get("id", f"C_{claims.index(c)}")
            conf     = c.get("confidence", 0.5)
            conflict = get_claim_conflict_score(cid, graph)
            priority = conflict * conf
            claim_requests.append({
                "claim_text":      c.get("claim", "")[:300],
                "claim_id":        cid,
                "source_modality": c.get("source_modality", "unknown"),
                "conflict_score":  conflict,
                "priority_score":  round(priority, 3),
                "_claim_obj":      c,
            })

        # ── ONE LLM call per video ──────────────────────────────────────
        claims_block = build_claims_block(claim_requests)
        user_prompt  = USER_PROMPT_BATCH.format(claims_block=claims_block)
        try:
            raw_output = client.chat_single(
                system=system_prompt,
                user=user_prompt,
            )
        except Exception as exc:
            print(f"[{vid}] Query gen failed: {exc}")
            raw_output = ""

        query_list = parse_batch_output(raw_output, claim_requests)

        all_evidence = []
        for req, query_data in zip(claim_requests, query_list):
            evidence = search_for_claim(query_data, count=args.count, use_serper=use_serper)
            for ev in evidence:
                ev["claim_text"] = req["claim_text"]
            all_evidence.extend(evidence)

        with open(os.path.join(raw_dir, f"{vid}.json"), "w", encoding="utf-8") as f:
            json.dump({"vid": vid, "evidence": all_evidence}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
