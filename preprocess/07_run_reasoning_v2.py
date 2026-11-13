"""
Step 7: Evidence-guided Reasoning (v2)
=======================================
Key changes from 06_run_reasoning.py:
  - Evidence source: verified_evidence_FakeSV_v2.jsonl (Filtering output)
    instead of 05_score_evidence.py (DeBERTa NLI scores) -- which is now REMOVED
  - Evidence grouped by claim_id in the prompt
  - Empty evidence handled gracefully (fallback to multimodal reasoning)
  - Claim relationship taxonomy: contradiction / complementary / independent
  - New output fields: evidence_availability, reasoning_basis
  - API calls via requests (not aiohttp) to avoid SSL/proxy issues

Output: data/cache/{dataset}/reasoning_v2/{vid}.json
  {
    "vid": "...",
    "overall_assessment": "True"|"False"|"Misleading"|"Unverifiable",
    "decision": "True"|"False",
    "confidence": float,
    "key_evidence_used": [{"claim_id":"","evidence":"","relation":""}],
    "evidence_availability": {"total_claims":N,"claims_with_evidence":M,"claims_without_evidence":K},
    "reasoning_basis": ["external_evidence"|"cross_modal_consistency"|"world_knowledge"|...],
    "cross_modal_conflicts": [...],
    "reason": "...",
    "uncertain_aspects": "..."
  }

Usage:
    python preprocess/07_run_reasoning_v2.py --dataset FakeSV --limit 20
    python preprocess/07_run_reasoning_v2.py --dataset FakeSV
"""

import argparse
import io
import json
import os
import re
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import requests
import urllib3
from dotenv import load_dotenv
from tqdm import tqdm

from utils.cot_common import safe_read_jsonl
from utils.quad_utils import get_quad_paths, build_image_content

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv(override=True)

CACHE_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "cache")

# ─── Language instructions ────────────────────────────────────────────────────

LANG_INSTRUCTIONS = {
    "FakeSV": (
        "所有自由文本字段（reason, uncertain_aspects, cross_modal_conflicts, "
        "key_evidence_used 中的 evidence 字段）必须用**中文**书写。"
        "JSON 键名保持英文，overall_assessment/decision/reasoning_basis 的枚举值保持英文。"
    ),
    "FakeTT":   "Write all free-text fields in English.",
    "WildFake": (
        "所有自由文本字段（reason, uncertain_aspects, cross_modal_conflicts, "
        "key_evidence_used 中的 evidence 字段）必须用**中文**书写。"
        "JSON 键名保持英文，overall_assessment/decision/reasoning_basis 的枚举值保持英文。"
    ),
}

LANG_REMINDERS = {
    "FakeSV":   "所有 reason、uncertain_aspects、cross_modal_conflicts 字段必须用中文，不得使用英文。",
    "FakeTT":   "All free-text fields must be in English.",
    "WildFake": "所有 reason、uncertain_aspects、cross_modal_conflicts 字段必须用中文，不得使用英文。",
}

# ─── Data paths ───────────────────────────────────────────────────────────────

DATA_PATHS = {
    "FakeSV": {
        "metadata":          r"F:\Bishe_workspace\ExMRD-master\data\FakeSV\data.jsonl",
        "ocr":               r"F:\Bishe_workspace\fakesv-qwen3\ocr.jsonl",
        "asr":               r"F:\Bishe_workspace\fakesv-qwen3\transcript.jsonl",
        "verified_evidence": r"F:\Bishe_workspace\EaMRD-CRCV\preprocess\output\verified_evidence_FakeSV_v2.jsonl",
    },
    "FakeTT": {
        "metadata":          r"F:\Bishe_workspace\FakeTT\data.json",
        "ocr":               r"F:\Bishe_workspace\FakeTT\ocr.jsonl",
        "asr":               r"F:\Bishe_workspace\FakeTT\transcript.jsonl",
        "verified_evidence": r"F:\Bishe_workspace\EaMRD-CRCV\preprocess\output\verified_evidence_FakeTT_v2.jsonl",
    },
    "WildFake": {
        "metadata":          r"F:\Bishe_workspace\WildFake\data_zh.jsonl",
        "ocr":               r"F:\Bishe_workspace\WildFake\ocr.jsonl",
        "asr":               r"F:\Bishe_workspace\WildFake\transcript.jsonl",
        "verified_evidence": r"F:\Bishe_workspace\EaMRD-CRCV\preprocess\output\verified_evidence_WildFake.jsonl",
    },
}

# ─── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert fact-checker analyzing short videos for misinformation and rumors.

**OUTPUT LANGUAGE: {lang_instruction}**

You will be provided with:
1. Claims extracted from multiple modalities of a short video (OCR, ASR, Vision, Title)
2. Cross-modal claim relationships
3. Filtered external evidence, grouped by claim
4. The original video content (OCR text, ASR text, title) and key frames

Important context:
- External evidence has already been filtered for event consistency, evidence usefulness, and temporal consistency.
- Only verification-relevant evidence is provided. The model should still judge whether it supports or refutes the claim.
- Rejected retrieval results are NOT shown.

---

## ① What counts as GENUINE CONTRADICTION (the ONLY suspicious signal)
Flag as contradiction ONLY when:
- Two modalities assert **the same fact** with **logically incompatible** values simultaneously.
  ✓ Title: "event happened in Beijing" + OCR: "event happened in Shanghai" → contradiction
  ✓ Title: "no casualties" + ASR: "10 people died in this event" → contradiction

## ② What does NOT count as contradiction — DO NOT flag these
1. **Temporal progression**: "suspect fled" then "suspect was captured" → sequential stages of same event (complementary)
2. **Same-domain terminology**: "mobile hospital" / "mobile laboratory" → different names for same entity (complementary)
3. **Context + outcome**: "bridge limit 55 tons" + "vehicle overloaded" → causal explanation (complementary)
4. **Different numerical aspects**: "4 people in vehicle" vs "3 died" → capacity ≠ casualties (different dimensions)
5. **Pre-event vs post-event perspectives**: "last moments before crash" + "157 killed" → different time-points of same event
6. **Degree/scale differences**: "bridge flooded" vs "water reached bridge surface" → intensity variation, NOT contradiction
7. **Measurement uncertainty**: wind force "12 级" vs "10 级"; count "2 people" vs "3 people" for same group → normal variation
8. **Visual frame absence**: X not visible in sampled frames does NOT mean X did not occur in the video

## ③ Visual frame limitation
The quad images cover only a small sample of frames from the video (frames 0–15 of many more).
- Do NOT conclude "the video does not show X" solely because X is absent from these frames.
- Critical evidence often appears in later parts of the video not captured.
- Absence of visual evidence ≠ evidence of absence.

## ④ External evidence direction
If external evidence PRIMARILY SUPPORTS the video content:
- Minor discrepancies in secondary details (exact count, timing, specific wording) do **NOT** flip the decision to False.
- Judge the **overall direction** of evidence, not individual mismatches.
- A real event with slightly inaccurate secondary details ≠ fabricated misinformation.

## ⑤ Decision threshold — False requires POSITIVE proof
**To decide "False", you need:**
- Strong external evidence that DIRECTLY REFUTES a key claim, OR
- Clear, logically incompatible cross-modal contradiction (see ① above)

**When uncertain — default to "Unverifiable", NOT "False":**
- No strong refuting evidence → "Unverifiable"
- Confidence < 0.65 + no direct refutation → lean "Unverifiable"
- "False" must be earned by positive evidence of falseness, not by absence of proof

**Decision priority:**
1. Strong external evidence refutes key claim → **False** (high confidence)
2. Strong external evidence supports key claim → **True**
3. Genuine logical contradiction between modalities (see ①) → **False** (medium confidence)
4. Unclear, complementary, or degree differences only → **Unverifiable** → decision **"False"**
5. No evidence AND claims are plausible/internally consistent → **True** (requires positive trust signal)

**CRITICAL on Unverifiable**: "Unverifiable" = you cannot confirm the claim is true. In misinformation detection, inability to verify is a suspicious signal. Always set decision = **"False"** when overall_assessment is "Unverifiable". Only set decision = "True" when evidence or strong cross-modal consistency ACTIVELY supports the claims.

## ⑥ Confidence calibration
- > 0.8: multiple strong signals clearly agree
- 0.6–0.8: partial evidence or some uncertainty
- < 0.6: evidence unavailable, weak, or only minor inconsistencies

Output ONLY a valid JSON object with the exact schema shown. No markdown, no extra text."""

USER_PROMPT_TEMPLATE = """\
Analyze the following short video for potential misinformation.

=== VIDEO METADATA ===
Title: {title}

=== EXTRACTED CLAIMS (by modality) ===
Title Claims:  {title_claims_text}
OCR Claims:    {ocr_claims_text}
ASR Claims:    {asr_claims_text}
Vision Claims: {vision_claims_text}

=== CROSS-MODAL CLAIM RELATIONSHIPS ===
{consistency_summary}

=== FILTERED EXTERNAL EVIDENCE (grouped by claim) ===
{evidence_text}

=== ORIGINAL VIDEO CONTENT ===
OCR Text:  {ocr_text}
ASR Text:  {asr_text}

[KEY FRAMES]
(Quad images follow — each covers 4 consecutive video frames)

=== TASK ===
Based on the above information, determine if this video contains false or misleading information.

Output the following JSON object exactly:
{{
  "overall_assessment": "True" | "False" | "Misleading" | "Unverifiable",
  "decision": "True" | "False",
  "confidence": <float 0.0-1.0>,
  "key_evidence_used": [
    {{"claim_id": "<claim_id>", "evidence": "<brief description>", "relation": "support|refute"}}
  ],
  "evidence_availability": {{
    "total_claims": <integer>,
    "claims_with_evidence": <integer>,
    "claims_without_evidence": <integer>
  }},
  "reasoning_basis": ["external_evidence", "cross_modal_consistency", "world_knowledge", "visual_content", "textual_content"],
  "cross_modal_conflicts": ["<description of genuine contradictions>"],
  "reason": "<2-4 sentence explanation citing specific evidence>",
  "uncertain_aspects": "<aspects that could not be verified>"
}}

Decision rules (CRITICAL — follow strictly):
- "False" ONLY when you have strong direct evidence of falseness. When in doubt, output "Unverifiable".
- "decision" must be "True" (not a rumor) or "False" (is a rumor/misleading). When overall_assessment is "Unverifiable", set decision to **"False"** (unverifiable = suspicious).
- "reasoning_basis": only include methods actually used.
- "cross_modal_conflicts": ONLY genuine logical incompatibilities (see ② in system prompt). Degree differences, temporal sequences, and terminology variants are NOT conflicts.
- **LANGUAGE REMINDER: {lang_reminder}**"""

# ─── Helpers ──────────────────────────────────────────────────────────────────

def format_claims_text(claims: list | None) -> str:
    if not claims:
        return "(none)"
    return "; ".join(
        f'[{c["id"]}] {c["claim"]} (conf={c.get("confidence", 0):.2f})'
        for c in claims[:4]
    )


def build_consistency_summary(graph: dict) -> str:
    """Convert claim graph edges → natural language, using new taxonomy."""
    lines = []
    for edge in graph.get("edges", []):
        rel = edge.get("relation", "neutral")
        if rel == "neutral":
            continue
        if rel == "contradiction":
            rel_word = "CONTRADICTS"
        elif rel == "support":
            rel_word = "COMPLEMENTS"
        else:
            rel_word = rel.upper()
        line = (
            f'[{edge["source_id"]}] ({edge["source_modality"]}) '
            f'{rel_word} '
            f'[{edge["target_id"]}] ({edge["target_modality"]})'
        )
        expl = edge.get("explanation", "")
        if expl:
            line += f" — {expl[:120]}"
        lines.append(line)
    return "\n".join(lines) if lines else "No significant cross-modal relationships detected."


def build_evidence_text_v2(ev_record: dict | None) -> tuple[str, dict]:
    """
    Build evidence text grouped by claim_id.
    Returns (prompt_text, evidence_availability_dict).
    """
    if not ev_record:
        return "(no evidence available)", {"total_claims": 0, "claims_with_evidence": 0, "claims_without_evidence": 0}

    verified = ev_record.get("verified_evidence", [])

    # Group by claim_id
    claim_groups: dict[str, list] = {}
    for ev in verified:
        cid = ev.get("claim_id", "UNK")
        claim_groups.setdefault(cid, []).append(ev)

    # Also build a set of claim_ids from claim_summary
    claim_summary = ev_record.get("claim_summary", ev_record.get("claim_results_summary", []))
    all_claim_ids = [cs["claim_id"] for cs in claim_summary] if claim_summary else list(claim_groups.keys())

    claims_with_evidence = len(claim_groups)
    claims_without_evidence = len([c for c in all_claim_ids if c not in claim_groups])
    total_claims = len(all_claim_ids) if all_claim_ids else claims_with_evidence

    availability = {
        "total_claims": total_claims,
        "claims_with_evidence": claims_with_evidence,
        "claims_without_evidence": claims_without_evidence,
    }

    if not verified:
        return "(no verified evidence found for any claim)", availability

    lines = []
    for cid in all_claim_ids:
        evs = claim_groups.get(cid, [])
        # Get claim text from first evidence item or claim_summary
        claim_text = ""
        if evs:
            claim_text = evs[0].get("claim_text", "")
        else:
            for cs in claim_summary:
                if cs["claim_id"] == cid:
                    claim_text = cs.get("claim", cs.get("claim_text", ""))
                    break

        header = f"[{cid}] {claim_text[:80]}"
        lines.append(header)
        if not evs:
            lines.append("  (no verified evidence)")
        else:
            for i, ev in enumerate(evs[:3], 1):
                rel    = ev.get("relation", "unknown")
                src    = ev.get("siteName", ev.get("source_domain", "?"))[:30]
                date   = (ev.get("datePublished", "") or "")[:10]
                em     = ev.get("event_match_score", 0)
                us     = ev.get("evidence_usefulness_score", 0)
                content = (ev.get("summary") or ev.get("snippet") or "")[:200]
                lines.append(
                    f"  [{i}] [{rel.upper()}] {src} ({date})"
                    f" event_match={em:.1f} usefulness={us:.1f}"
                    f"\n      {content}"
                )
        lines.append("")

    return "\n".join(lines).strip(), availability


# ─── API call ─────────────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=2, pool_maxsize=4, max_retries=0)
    session.mount("http://",  adapter)
    session.mount("https://", adapter)
    return session


def call_api(session: requests.Session, url: str, api_key: str, model: str,
             content: list, max_tokens: int = 1500, temperature: float = 0.3) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(4):
        try:
            resp = session.post(url, headers=headers, data=json.dumps(payload),
                                timeout=120, verify=False)
            resp.raise_for_status()
            time.sleep(0.3)
            return resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.ProxyError:
            wait = 15 * (attempt + 1)
            print(f"  [ProxyError] attempt {attempt+1}, rebuilding session, sleep {wait}s", flush=True)
            try: session.close()
            except: pass
            session = _make_session()
            time.sleep(wait)
        except requests.exceptions.ConnectionError as e:
            wait = 10 * (attempt + 1)
            print(f"  [ConnError] attempt {attempt+1}: {e}, sleep {wait}s", flush=True)
            time.sleep(wait)
        except Exception as e:
            print(f"  [APIError] attempt {attempt+1}: {e}", flush=True)
            if attempt == 3:
                return ""
            time.sleep(5)
    return ""


# ─── Output parsing ───────────────────────────────────────────────────────────

def get_default_result(vid: str) -> dict:
    return {
        "vid": vid,
        "overall_assessment": "Unverifiable",
        "decision": "False",
        "confidence": 0.5,
        "key_evidence_used": [],
        "evidence_availability": {"total_claims": 0, "claims_with_evidence": 0, "claims_without_evidence": 0},
        "reasoning_basis": ["world_knowledge"],
        "cross_modal_conflicts": [],
        "reason": "Reasoning failed to parse.",
        "uncertain_aspects": "Output format error.",
    }


def parse_reasoning(raw: str, vid: str, ev_availability: dict) -> dict:
    raw = (raw or "").strip()
    matches = re.findall(r'\{[\s\S]*\}', raw)
    if not matches:
        r = get_default_result(vid)
        r["evidence_availability"] = ev_availability   # always computed
        return r
    try:
        result = json.loads(max(matches, key=len))
        assert "decision" in result and result["decision"] in ["True", "False"]
        assert 0.0 <= float(result.get("confidence", 0)) <= 1.0
        result["vid"] = vid
        # Always use programmatically computed availability (LLM counts are unreliable)
        result["evidence_availability"] = ev_availability
        if "reasoning_basis" not in result:
            result["reasoning_basis"] = []
        return result
    except (json.JSONDecodeError, AssertionError, KeyError) as e:
        print(f"[WARN] Parse error for {vid}: {e}")
        r = get_default_result(vid)
        r["evidence_availability"] = ev_availability
        return r


# ─── Main ─────────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    if not os.path.exists(path): return {}
    with open(path, encoding="utf-8") as f: return json.load(f)

def get_text(df, vid: str, *cols) -> str:
    if df is None or df.empty: return ""
    rows = df[df.iloc[:, 0].astype(str) == str(vid)]
    if rows.empty: return ""
    row = rows.iloc[0]
    for c in cols:
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c])[:800]
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["FakeSV", "FakeTT", "WildFake"], default="FakeSV")
    parser.add_argument("--limit",   type=int, default=0)
    # Default to Split_target.txt so bare invocation always processes 3624 split vids
    parser.add_argument("--vids-file", type=str, default=None,
                        help="Text file with one vid per line (default: Split_target.txt)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verified-ev", type=str, default=None,
                        help="Override path to verified_evidence JSONL (default: from DATA_PATHS)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Override output directory (default: data/cache/{dataset}/reasoning_v2/)")
    args = parser.parse_args()

    claims_dir = os.path.join(CACHE_ROOT, args.dataset, "claims")
    graph_dir  = os.path.join(CACHE_ROOT, args.dataset, "claim_graphs")
    out_dir    = args.out_dir if args.out_dir else os.path.join(CACHE_ROOT, args.dataset, "reasoning_v2")
    os.makedirs(out_dir, exist_ok=True)

    # Load filtered evidence
    ev_path = args.verified_ev if args.verified_ev else DATA_PATHS[args.dataset]["verified_evidence"]
    print(f"Loading filtered evidence from {ev_path} ...", flush=True)
    evidence_map: dict[str, dict] = {}
    if os.path.exists(ev_path):
        for line in open(ev_path, encoding="utf-8"):
            try:
                obj = json.loads(line)
                evidence_map[str(obj["vid"])] = obj
            except: pass
    print(f"  Loaded {len(evidence_map)} evidence records", flush=True)

    # Build todo list — default to dataset Split_target when no --vids-file given
    if args.vids_file:
        vids_file = args.vids_file
    else:
        vids_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "cache", args.dataset, "Split_target.txt"
        )
        if not os.path.exists(vids_file):
            vids_file = None

    if vids_file:
        todo_raw = [l.strip() for l in open(vids_file, encoding="utf-8") if l.strip()]
        print(f"Using vids-file: {vids_file}  ({len(todo_raw)} vids)", flush=True)
    else:
        todo_raw = [os.path.splitext(f)[0] for f in os.listdir(claims_dir) if f.endswith(".json")]
        print(f"No vids-file found, reading all {len(todo_raw)} vids from claims_dir", flush=True)

    done = set() if args.overwrite else {
        os.path.splitext(f)[0] for f in os.listdir(out_dir) if f.endswith(".json")
    }
    todo = [v for v in todo_raw if v not in done and os.path.exists(os.path.join(claims_dir, f"{v}.json"))]
    if args.limit:
        todo = todo[:args.limit]
    print(f"Todo: {len(todo)}  Done: {len(done)}  Evidence available: "
          f"{sum(1 for v in todo if v in evidence_map)}", flush=True)

    # Load metadata + OCR/ASR
    meta_df = pd.read_json(DATA_PATHS[args.dataset]["metadata"], lines=True, dtype={"video_id": str})
    meta_df["video_id"] = meta_df["video_id"].astype(str)
    meta_map = meta_df.set_index("video_id").to_dict(orient="index")
    ocr_df = safe_read_jsonl(DATA_PATHS[args.dataset]["ocr"])
    asr_df = safe_read_jsonl(DATA_PATHS[args.dataset]["asr"])

    api_key  = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://runapi.co/v1").rstrip("/") + "/chat/completions"
    model    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not set"); sys.exit(1)

    session  = _make_session()
    sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        lang_instruction=LANG_INSTRUCTIONS.get(args.dataset, "Use the video's language.")
    )

    for vid in tqdm(todo, desc=f"Reasoning-v2 [{args.dataset}]"):
        claims_data = load_json(os.path.join(claims_dir, f"{vid}.json"))
        graph       = load_json(os.path.join(graph_dir,  f"{vid}.json"))
        ev_record   = evidence_map.get(vid)

        if not claims_data:
            continue

        row   = meta_map.get(vid, {})
        title = str(row.get("keywords", row.get("description", row.get("title", "")))).strip()[:256]
        ocr   = get_text(ocr_df, vid, "description", "ocr_text", "text", "ret") or "(no OCR)"
        asr   = get_text(asr_df, vid, "transcript", "text", "ret") or "(no transcript)"

        evidence_text, ev_availability = build_evidence_text_v2(ev_record)

        user_msg = (
            sys_prompt + "\n\n"
            + USER_PROMPT_TEMPLATE.format(
                title=title or "(no title)",
                title_claims_text = format_claims_text(claims_data.get("title_claims")),
                ocr_claims_text   = format_claims_text(claims_data.get("ocr_claims")),
                asr_claims_text   = format_claims_text(claims_data.get("asr_claims")),
                vision_claims_text= format_claims_text(claims_data.get("vision_claims")),
                consistency_summary=build_consistency_summary(graph),
                evidence_text=evidence_text,
                ocr_text=ocr,
                asr_text=asr,
                lang_reminder=LANG_REMINDERS.get(args.dataset, ""),
            )
        )

        quad_paths  = get_quad_paths(vid, args.dataset)
        image_parts = build_image_content(quad_paths)
        content     = [{"type": "text", "text": user_msg}] + image_parts

        raw    = call_api(session, base_url, api_key, model, content)
        result = parse_reasoning(raw, vid, ev_availability)

        out_path = os.path.join(out_dir, f"{vid}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    try: session.close()
    except: pass
    print(f"\n[DONE] Results in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
