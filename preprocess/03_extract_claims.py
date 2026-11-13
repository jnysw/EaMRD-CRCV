"""
Step 3: M1 — Claim-centric Multimodal Fact Decomposition

Input:  4 × Quads4 images + OCR text + ASR text + Title (keywords)
Model:  GPT-4o-mini (multimodal)
Output: data/cache/{dataset}/claims/{vid}.json
        Structured claims per modality:
        {
          "vid": "...",
          "ocr_claims":    [{"id": "OCR_1", "claim": "...", "confidence": 0.9}, ...],
          "asr_claims":    [...],
          "vision_claims": [...],   ← filtered at confidence ≥ 0.8
          "title_claims":  [...],
          "all_claims":    [...]    ← flat list merged across modalities (top-3 per modality)
        }

Usage:
    python preprocess/03_extract_claims.py --dataset FakeSV --limit 50
    python preprocess/03_extract_claims.py --dataset FakeSV
"""

import argparse
import json
import os
import re
import sys

# Allow imports from project root (utils/, data_utils/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from utils.chat_llm import ChatLLM
from utils.cot_common import safe_read_jsonl
from utils.quad_utils import get_quad_paths, build_image_content, has_quads

load_dotenv(override=True)

CACHE_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "cache")

DATA_PATHS = {
    "FakeSV": {
        "metadata": r"F:\Bishe_workspace\ExMRD-master\data\FakeSV\data.jsonl",
        "ocr":      r"F:\Bishe_workspace\fakesv-qwen3\ocr.jsonl",
        "asr":      r"F:\Bishe_workspace\fakesv-qwen3\transcript.jsonl",
    },
    "FakeTT": {
        "metadata": r"F:\Bishe_workspace\FakeTT\data.json",
        "ocr":      r"F:\Bishe_workspace\FakeTT\ocr.jsonl",
        "asr":      r"F:\Bishe_workspace\FakeTT\transcript.jsonl",
    },
    "WildFake": {
        "metadata": r"F:\Bishe_workspace\WildFake\data_zh.jsonl",
        "ocr":      r"F:\Bishe_workspace\WildFake\ocr.jsonl",
        "asr":      r"F:\Bishe_workspace\WildFake\transcript.jsonl",
    },
}

VISION_CONFIDENCE_THRESHOLD = 0.8
MAX_CLAIMS_PER_MODALITY = 3

SYSTEM_PROMPT_TEMPLATE = """\
You are a fact-checking assistant specialized in analyzing short videos for rumor detection.

**OUTPUT LANGUAGE: {lang_instruction}**

Your task is to extract verifiable atomic claims from different modalities of a short video.

Rules for claim extraction:
1. Each claim must be a single, atomic, independently verifiable factual statement.
2. Claims must be specific enough to be searched on the internet (avoid vague statements).
3. Extract claims ONLY from the given modality — do not infer or combine across modalities.
4. If a modality contains no verifiable factual claims, output null for that modality.
5. Assign a confidence score (0.0–1.0) based on clarity and verifiability of the claim.
6. For vision_claims, apply a stricter standard: only include claims about clearly observable facts in the video frames (e.g., visible location signs, identifiable objects, observable events). Confidence threshold should be ≥ 0.8.
7. Do NOT include opinions, emotions, or non-factual statements.
8. Output ONLY valid JSON. Do not include any explanation or markdown formatting.

BAD vision claim: "Video shows people in a crowd"  (not verifiable)
GOOD vision claim: "Video frames show a sign reading '武汉中心医院'"  (verifiable location)
BAD vision claim: "The scene looks chaotic"  (subjective)
GOOD vision claim: "Video frames show vehicles labeled '119消防'"  (verifiable)"""

LANG_INSTRUCTIONS = {
    "FakeSV":   "所有 claim 文本必须用中文（Chinese）书写。JSON 的键名保持英文不变。",
    "FakeTT":   "Write all claim texts in English.",
    "WildFake": "所有 claim 文本必须用中文（Chinese）书写。JSON 的键名保持英文不变。",
}


def get_system_prompt(dataset: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        lang_instruction=LANG_INSTRUCTIONS.get(dataset, "Write claims in the same language as the video content.")
    )

USER_PROMPT_TEMPLATE = """\
Analyze the following short video content and extract atomic verifiable claims from each modality separately.

[TITLE]
{title}

[OCR TEXT] (text visible in the video frames)
{ocr_text}

[ASR TEXT] (spoken words transcribed from audio)
{asr_text}

[VIDEO FRAMES]
(4 quad images follow — each quad covers 4 consecutive frames)

Output a JSON object with exactly this structure:
{{
  "ocr_claims": [
    {{"id": "OCR_1", "claim": "...", "confidence": 0.9}},
    ...
  ],
  "asr_claims": [
    {{"id": "ASR_1", "claim": "...", "confidence": 0.85}},
    ...
  ],
  "vision_claims": [
    {{"id": "VIS_1", "claim": "...", "confidence": 0.82}},
    ...
  ],
  "title_claims": [
    {{"id": "TTL_1", "claim": "...", "confidence": 0.95}},
    ...
  ]
}}

If a modality has no verifiable claims, use null:
  "vision_claims": null"""


def parse_claims_response(raw: str, vid: str) -> dict:
    """Parse LLM JSON response; return safe default on failure."""
    raw = (raw or "").strip().strip("```json").strip("```").strip()
    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Extract JSON block
    matches = re.findall(r'\{[\s\S]*\}', raw)
    if matches:
        try:
            return json.loads(max(matches, key=len))
        except json.JSONDecodeError:
            pass
    print(f"[WARN] JSON parse failed for {vid}")
    return {"ocr_claims": None, "asr_claims": None, "vision_claims": None, "title_claims": None}


def filter_and_top_k(claims: list | None, threshold: float = 0.0, k: int = MAX_CLAIMS_PER_MODALITY) -> list | None:
    if claims is None:
        return None
    filtered = [c for c in claims if isinstance(c, dict) and c.get("confidence", 0) >= threshold]
    return sorted(filtered, key=lambda x: x.get("confidence", 0), reverse=True)[:k]


def merge_all_claims(parsed: dict) -> list:
    """Flatten all modalities into a single claim list for retrieval."""
    flat = []
    for modality, key in [("ocr", "ocr_claims"), ("asr", "asr_claims"),
                           ("vision", "vision_claims"), ("title", "title_claims")]:
        items = parsed.get(key) or []
        for c in items:
            flat.append({**c, "source_modality": modality})
    # Deduplicate by claim text
    seen = set()
    unique = []
    for c in flat:
        if c.get("claim", "") not in seen:
            seen.add(c["claim"])
            unique.append(c)
    return sorted(unique, key=lambda x: x.get("confidence", 0), reverse=True)


def load_metadata(dataset: str) -> pd.DataFrame:
    path = DATA_PATHS[dataset]["metadata"]
    df = pd.read_json(path, lines=True, dtype={"video_id": str})
    if "video_id" not in df.columns and "vid" in df.columns:
        df = df.rename(columns={"vid": "video_id"})
    df["video_id"] = df["video_id"].astype(str)
    return df


def get_text(df: pd.DataFrame, vid: str, *cols) -> str:
    rows = df[df["vid"].astype(str) == str(vid)]
    if rows.empty:
        return ""
    row = rows.iloc[0]
    for c in cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c])[:800]
    return ""


def build_request(vid: str, title: str, ocr: str, asr: str, dataset: str) -> dict:
    """Build a single API request dict including image content."""
    text_part = USER_PROMPT_TEMPLATE.format(
        title=title or "(no title)",
        ocr_text=ocr or "(no OCR)",
        asr_text=asr or "(no transcript)",
    )
    quad_paths = get_quad_paths(vid, dataset)
    image_content = build_image_content(quad_paths)

    system_prompt = get_system_prompt(dataset)
    content = [{"type": "text", "text": system_prompt + "\n\n" + text_part}] + image_content
    return {"_vid": vid, "_content": content}


def call_api_with_content(client: ChatLLM, content_list: list) -> str:
    """
    Send a pre-built content list to the API.
    We bypass the template format of ChatLLM by passing a special key.
    """
    import asyncio, json as _json, aiohttp

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
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(20)
                else:
                    return ""

    async def _batch(items):
        async with aiohttp.ClientSession() as session:
            return await asyncio.gather(*[_single(session, c) for c in items])

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_batch(content_list))
    finally:
        loop.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["FakeSV", "FakeTT", "WildFake"], default="FakeSV")
    parser.add_argument("--limit", type=int, default=0, help="Limit to N videos (0=all)")
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    cache_dir = os.path.join(CACHE_ROOT, args.dataset, "claims")
    os.makedirs(cache_dir, exist_ok=True)

    done_vids = {os.path.splitext(f)[0] for f in os.listdir(cache_dir) if f.endswith(".json")}
    meta_df = load_metadata(args.dataset)
    ocr_df  = safe_read_jsonl(DATA_PATHS[args.dataset]["ocr"])
    asr_df  = safe_read_jsonl(DATA_PATHS[args.dataset]["asr"])

    todo_vids = [v for v in meta_df["video_id"].tolist() if v not in done_vids]
    if args.limit:
        todo_vids = todo_vids[:args.limit]
    print(f"[{args.dataset}] Done: {len(done_vids)}  Todo: {len(todo_vids)}")

    client = ChatLLM(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions"),
        key=os.getenv("OPENAI_API_KEY", ""),
        prompt="",  # not used — we build content manually
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.3,
        max_tokens=1024,
    )

    # Build keyword → vid map for title lookup (FakeSV)
    meta_map = meta_df.set_index("video_id").to_dict(orient="index")

    batches = [todo_vids[i:i + args.batch] for i in range(0, len(todo_vids), args.batch)]

    for batch_vids in tqdm(batches, desc=f"Claim extraction [{args.dataset}]"):
        contents = []
        for vid in batch_vids:
            row = meta_map.get(vid, {})
            title = str(row.get("keywords", "")).strip()[:256]  # FakeSV has keywords, FakeTT has description
            if not title:
                title = str(row.get("description", row.get("title", ""))).strip()[:256]
            ocr = get_text(ocr_df, vid, "description", "ocr_text", "text", "ret")
            asr = get_text(asr_df, vid, "transcript", "text", "ret")
            req = build_request(vid, title, ocr, asr, args.dataset)
            contents.append(req["_content"])

        try:
            outputs = call_api_with_content(client, contents)
        except Exception as exc:
            print(f"API error: {exc}")
            outputs = [""] * len(batch_vids)

        for vid, raw in zip(batch_vids, outputs):
            parsed = parse_claims_response(raw, vid)

            # Filter and top-K per modality
            parsed["ocr_claims"]    = filter_and_top_k(parsed.get("ocr_claims"),    threshold=0.0)
            parsed["asr_claims"]    = filter_and_top_k(parsed.get("asr_claims"),    threshold=0.0)
            parsed["vision_claims"] = filter_and_top_k(parsed.get("vision_claims"), threshold=VISION_CONFIDENCE_THRESHOLD)
            parsed["title_claims"]  = filter_and_top_k(parsed.get("title_claims"),  threshold=0.0)
            parsed["all_claims"]    = merge_all_claims(parsed)
            parsed["vid"] = vid
            parsed["_raw"] = raw[:500] if raw else ""

            with open(os.path.join(cache_dir, f"{vid}.json"), "w", encoding="utf-8") as f:
                json.dump(parsed, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
