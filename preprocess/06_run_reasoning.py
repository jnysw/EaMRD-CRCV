"""
Step 6: M4 — MLLM Structured Reasoning

Inputs per video:
  - All claims (from Step 3) + claim consistency graph (from Step 3b)
  - Top-K scored evidence (from Step 5)
  - OCR text, ASR text
  - 4 × Quads4 images → injected as multi-image into GPT-4o-mini

Output: data/cache/{dataset}/reasoning/{vid}.json
  {
    "vid": "...",
    "overall_assessment": "True"|"False"|"Misleading"|"Unverifiable",
    "decision": "True"|"False",
    "confidence": float,
    "key_evidence_used": [...],
    "cross_modal_conflicts": [...],
    "reason": "...",
    "uncertain_aspects": "..."
  }

Usage:
    python preprocess/06_run_reasoning.py --dataset FakeSV --limit 50
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from utils.chat_llm import ChatLLM
from utils.cot_common import safe_read_jsonl
from utils.quad_utils import get_quad_paths, build_image_content

load_dotenv(override=True)

CACHE_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "cache")

LANG_INSTRUCTIONS_REASONING = {
    "FakeSV": (
        "所有自由文本字段（reason, key_evidence_used, cross_modal_conflicts, uncertain_aspects）"
        "必须用中文（Chinese）书写。JSON 键名保持英文，overall_assessment 和 decision 的值保持英文字符串。"
    ),
    "FakeTT": "Write all free-text fields in English.",
}

DATA_PATHS = {
    "FakeSV": {
        "metadata": r"F:\Bishe_workspace\ExMRD-master\data\FakeSV\data.jsonl",
        "ocr":      r"F:\Bishe_workspace\fakesv-qwen3\ocr.jsonl",
        "asr":      r"F:\Bishe_workspace\fakesv-qwen3\transcript.jsonl",
    },
    "FakeTT": {
        "metadata": r"F:\Bishe_workspace\FakeTT\data.json",  # JSONL, one record per line
        "ocr":      r"F:\Bishe_workspace\FakeTT\ocr.jsonl",
        "asr":      r"F:\Bishe_workspace\FakeTT\transcript.jsonl",
    },
}

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert fact-checker analyzing short videos for misinformation and rumors.

**OUTPUT LANGUAGE: {lang_instruction}**

You will be provided with:
1. Claims extracted from different modalities of a short video (OCR, ASR, visual, title)
2. Cross-modal consistency analysis (which claims support or contradict each other)
3. External evidence retrieved from the web
4. The original video content (OCR text, ASR text, title, and key frames)

Your task is to reason about whether the video contains false or misleading information.

Reasoning guidelines:
- Cross-modal contradictions (e.g., visual shows X but text claims Y) are suspicious signals, not definitive proof of falsehood.
- Evaluate external evidence carefully: consider source credibility and relevance.
- A claim supported by credible external evidence is more likely to be true.
- A claim contradicted by multiple credible sources is more likely to be false.
- Partial falsehood counts: if one key claim is false, the video may be misleading even if other claims are true.

IMPORTANT: Cross-modal contradictions (e.g., visual content contradicting text claims) are strong suspicious signals. When cross-modal conflicts exist AND external evidence supports the contradiction, weight this heavily in your final decision.

Calibration reminder:
- Use confidence < 0.6 when evidence is weak, conflicting, or unavailable.
- Use confidence 0.6–0.8 when evidence is moderate or partially conflicting.
- Use confidence > 0.8 ONLY when multiple credible sources clearly support your decision.

Output ONLY a valid JSON object. No explanation outside the JSON. No markdown."""


def get_reasoning_system_prompt(dataset: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        lang_instruction=LANG_INSTRUCTIONS_REASONING.get(
            dataset, "Use the same language as the video content."
        )
    )


USER_PROMPT_TEMPLATE = """\
Analyze the following short video for potential misinformation.

=== VIDEO METADATA ===
Title: {title}

=== EXTRACTED CLAIMS ===
OCR Claims:    {ocr_claims_text}
ASR Claims:    {asr_claims_text}
Vision Claims: {vision_claims_text}
Title Claims:  {title_claims_text}

=== CROSS-MODAL CONSISTENCY ===
{consistency_summary}
(Format: "Claim A [source] CONTRADICTS/SUPPORTS Claim B [source]: brief explanation")

=== EXTERNAL EVIDENCE (Top-{k} by credibility score) ===
{evidence_text}
(Each evidence entry: [Source] [Date] [Credibility: H/M/L] Content summary)

=== ORIGINAL CONTENT ===
OCR Text:  {ocr_text}
ASR Text:  {asr_text}

[KEY FRAMES]
(4 quad images follow — each covers 4 consecutive frames)

=== TASK ===
Based on all the above information, determine if this video contains false or misleading information.

Output:
{{
  "overall_assessment": "True" | "False" | "Misleading" | "Unverifiable",
  "decision": "True" | "False",
  "confidence": <float between 0.0 and 1.0>,
  "key_evidence_used": ["brief description of most important evidence pieces"],
  "cross_modal_conflicts": ["description of detected conflicts"],
  "reason": "A concise explanation (2-4 sentences) of your decision, citing specific evidence.",
  "uncertain_aspects": "Any aspects you could not verify or that remain ambiguous."
}}

Notes:
- "decision" must be binary: "True" (not a rumor) or "False" (is a rumor/misleading).
- "overall_assessment" can be more nuanced.
- "confidence" reflects your certainty: use < 0.6 if evidence is weak or conflicting."""


def format_claims_text(claims: list | None) -> str:
    if not claims:
        return "(none)"
    return "; ".join(f'[{c["id"]}] {c["claim"]} (conf={c.get("confidence",0):.2f})'
                     for c in claims[:3])


def build_consistency_summary(graph: dict) -> str:
    """Convert claim graph edges to natural language summary."""
    lines = []
    for edge in graph.get("edges", []):
        rel = edge.get("relation", "neutral")
        if rel == "neutral":
            continue
        rel_word = "CONTRADICTS" if rel == "contradiction" else "SUPPORTS"
        line = (f'Claim [{edge["source_id"]}] ({edge["source_modality"]}) '
                f'{rel_word} '
                f'Claim [{edge["target_id"]}] ({edge["target_modality"]})')
        lines.append(line)
    return "\n".join(lines) if lines else "No significant cross-modal conflicts detected."


def build_evidence_text(scored_evidence: list, top_k: int = 5) -> str:
    lines = []
    for i, ev in enumerate(scored_evidence[:top_k]):
        score = ev.get("_score", 0.0)
        cred = "H" if score > 0.7 else ("M" if score > 0.4 else "L")
        domain = ev.get("source_domain", "?")
        date = ev.get("date", "?")
        summary = (ev.get("summary") or "")[:300]
        lines.append(f"[{i+1}] [{domain}] [{date}] [Credibility: {cred}]\n    {summary}")
    return "\n".join(lines) if lines else "(no evidence available)"


def get_default_reasoning(vid: str) -> dict:
    return {
        "vid": vid,
        "overall_assessment": "Unverifiable",
        "decision": "False",
        "confidence": 0.5,
        "key_evidence_used": [],
        "cross_modal_conflicts": [],
        "reason": "Reasoning failed to parse.",
        "uncertain_aspects": "Output format error.",
    }


def parse_reasoning(raw: str, vid: str) -> dict:
    raw = (raw or "").strip()
    matches = re.findall(r'\{[\s\S]*\}', raw)
    if not matches:
        return get_default_reasoning(vid)
    try:
        result = json.loads(max(matches, key=len))
        assert "decision" in result and result["decision"] in ["True", "False"]
        assert 0.0 <= float(result.get("confidence", 0)) <= 1.0
        result["vid"] = vid
        return result
    except (json.JSONDecodeError, AssertionError, KeyError) as e:
        print(f"[WARN] Reasoning parse error for {vid}: {e}")
        return get_default_reasoning(vid)


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_text(df: pd.DataFrame, vid: str, *cols) -> str:
    if df is None or df.empty:
        return ""
    rows = df[df.iloc[:, 0].astype(str) == str(vid)]
    if rows.empty:
        return ""
    row = rows.iloc[0]
    for c in cols:
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c])[:800]
    return ""


def call_api_with_content(client: ChatLLM, content_list: list) -> list[str]:
    import asyncio
    import json as _json
    import aiohttp

    async def _single(session, content):
        payload = {
            "messages": [{"role": "user", "content": content}],
            "model": client.model,
            "stream": False,
            "temperature": client.temperature,
            "max_tokens": client.max_tokens or 1024,
        }
        for attempt in range(3):
            try:
                async with session.post(
                    url=client.base_url,
                    headers={"Authorization": f"Bearer {client.key}", "Content-Type": "application/json"},
                    data=_json.dumps(payload),
                    timeout=aiohttp.ClientTimeout(total=180),
                ) as resp:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(20)
                else:
                    print(f"API error: {e}")
                    return ""

    async def _batch(items):
        async with aiohttp.ClientSession() as session:
            return list(await asyncio.gather(*[_single(session, c) for c in items]))

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_batch(content_list))
    finally:
        loop.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["FakeSV", "FakeTT"], default="FakeSV")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    claims_dir  = os.path.join(CACHE_ROOT, args.dataset, "claims")
    graph_dir   = os.path.join(CACHE_ROOT, args.dataset, "claim_graphs")
    scored_dir  = os.path.join(CACHE_ROOT, args.dataset, "evidence", "scored")
    out_dir     = os.path.join(CACHE_ROOT, args.dataset, "reasoning")
    os.makedirs(out_dir, exist_ok=True)

    done       = {os.path.splitext(f)[0] for f in os.listdir(out_dir) if f.endswith(".json")}
    claim_vids = [os.path.splitext(f)[0] for f in os.listdir(claims_dir) if f.endswith(".json")]
    todo       = [v for v in claim_vids if v not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"Claims: {len(claim_vids)}  Done: {len(done)}  Todo: {len(todo)}")

    meta_df = pd.read_json(DATA_PATHS[args.dataset]["metadata"], lines=True, dtype={"video_id": str})
    meta_df["video_id"] = meta_df["video_id"].astype(str)
    meta_map = meta_df.set_index("video_id").to_dict(orient="index")

    ocr_df = safe_read_jsonl(DATA_PATHS[args.dataset]["ocr"])
    asr_df = safe_read_jsonl(DATA_PATHS[args.dataset]["asr"])

    client = ChatLLM(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions"),
        key=os.getenv("OPENAI_API_KEY", ""),
        prompt="",
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.3,
        max_tokens=1024,
    )

    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]

    for batch_vids in tqdm(batches, desc=f"Reasoning [{args.dataset}]"):
        contents, vids_in_batch = [], []

        for vid in batch_vids:
            claims_data = load_json(os.path.join(claims_dir, f"{vid}.json"))
            graph       = load_json(os.path.join(graph_dir, f"{vid}.json"))
            scored      = load_json(os.path.join(scored_dir, f"{vid}.json"))

            if not claims_data:
                continue

            row   = meta_map.get(vid, {})
            title = str(row.get("keywords", row.get("description", row.get("title", "")))).strip()[:256]
            ocr   = get_text(ocr_df,  vid, "description", "ocr_text", "text", "ret")
            asr   = get_text(asr_df,  vid, "transcript", "text", "ret")

            top_evidence = scored.get("top_k", [])[:args.top_k]

            prompt_text = (
                get_reasoning_system_prompt(args.dataset) + "\n\n"
                + USER_PROMPT_TEMPLATE.format(
                    title=title or "(no title)",
                    ocr_claims_text   = format_claims_text(claims_data.get("ocr_claims")),
                    asr_claims_text   = format_claims_text(claims_data.get("asr_claims")),
                    vision_claims_text= format_claims_text(claims_data.get("vision_claims")),
                    title_claims_text = format_claims_text(claims_data.get("title_claims")),
                    consistency_summary=build_consistency_summary(graph),
                    k=len(top_evidence),
                    evidence_text=build_evidence_text(top_evidence, args.top_k),
                    ocr_text=ocr or "(no OCR)",
                    asr_text=asr or "(no transcript)",
                )
            )

            quad_paths  = get_quad_paths(vid, args.dataset)
            image_parts = build_image_content(quad_paths)

            content = [{"type": "text", "text": prompt_text}] + image_parts
            contents.append(content)
            vids_in_batch.append(vid)

        if not contents:
            continue

        try:
            outputs = call_api_with_content(client, contents)
        except Exception as exc:
            print(f"API error: {exc}")
            outputs = [""] * len(contents)

        for vid, raw in zip(vids_in_batch, outputs):
            result = parse_reasoning(raw, vid)
            with open(os.path.join(out_dir, f"{vid}.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
