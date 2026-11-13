"""
Event-level Video Summary Extraction — Prompt definitions.

Purpose: Extract fact-checking-oriented event context from a video to constrain
open-domain evidence retrieval and remove irrelevant web results.
"""

# Language instructions per dataset
LANG_INSTRUCTIONS = {
    "FakeSV": (
        "所有自由文本字段（event_summary, entities, time_information, location_information, "
        "verification_targets）必须用中文（Chinese）书写。"
        "JSON 键名保持英文，event_type 的值保持英文枚举。"
        "即使 claims 是英文，也必须用中文输出所有字段内容。"
        "event_type must be one of: "
        "natural_disaster / public_event / social_issue / political / crime / health / other"
    ),
    "FakeTT": (
        "All free-text fields must be written in ENGLISH. "
        "JSON keys remain in English. "
        "event_type must be one of: "
        "natural_disaster / public_event / social_issue / political / crime / health / other"
    ),
    "WildFake": (
        "所有自由文本字段（event_summary, entities, time_information, location_information, "
        "verification_targets）必须用中文（Chinese）书写。"
        "JSON 键名保持英文，event_type 的值保持英文枚举。"
        "即使 claims 是英文，也必须用中文输出所有字段内容。"
        "event_type must be one of: "
        "natural_disaster / public_event / social_issue / political / crime / health / other"
    ),
}

SYSTEM_PROMPT_TEMPLATE = """\
You are a professional fact-checking assistant specializing in short-video misinformation analysis.

**OUTPUT LANGUAGE: {lang_instruction}**

Your task is to extract event-level context from a short video.
This context will be used to retrieve and filter external evidence for fact-checking.

The goal is NOT to judge the video's veracity. Only extract factual event context.

You will be given:
1. Key frames of the video (as quad images, each covering 4 consecutive frames)
2. OCR text (text visible in the video)
3. ASR transcript (spoken words)
4. Pre-extracted claims (to guide understanding — do not simply copy them)

Extraction guidelines:
- event_summary: Describe the REAL-WORLD EVENT shown or claimed in the video.
  Be specific. Include WHO, WHAT, WHERE, WHEN if visible.
  Do NOT just describe the video appearance; describe the event being documented.
- entities: Extract specific named entities (people, organizations, locations, objects).
  Prefer specific names over general categories.
- event_type: Classify the event into one of the allowed categories.
- time_information: Any temporal clues (year, date, season, "recently", etc.).
  Use "unknown" if no time information is present.
- location_information: Any location clues (city, country, building, region).
  Use "unknown" if no location information is present.
- verification_targets: 2-4 specific factual statements from the video that REQUIRE
  external verification. These will become evidence retrieval queries.
  Each target must be independently verifiable via web search.

Output ONLY a valid JSON object. No explanation outside the JSON. No markdown."""


USER_PROMPT_TEMPLATE = """\
Analyze the following short video and extract event-level context for fact-checking.

=== OCR TEXT (visible text in video) ===
{ocr_text}

=== ASR TRANSCRIPT (spoken content) ===
{asr_text}

=== PRE-EXTRACTED CLAIMS (for context only) ===
{claims_text}

[KEY FRAMES FOLLOW]
(Quad images: each covers 4 consecutive video frames)

=== TASK ===
Extract event-level information for evidence retrieval.

Return ONLY this JSON:
{{
  "event_summary": "One paragraph describing the real-world event shown/claimed in the video",
  "entities": ["person/org/location/object 1", "person/org/location/object 2"],
  "event_type": "one of: natural_disaster/public_event/social_issue/political/crime/health/other",
  "time_information": "temporal clues or 'unknown'",
  "location_information": "location clues or 'unknown'",
  "verification_targets": [
    "Specific factual claim 1 that needs external verification",
    "Specific factual claim 2 that needs external verification"
  ]
}}

⚠️ FINAL REMINDER: {lang_reminder}"""

LANG_REMINDERS = {
    "FakeSV":   "所有字段内容必须用中文书写，即使原始 claims 是英文。JSON 键保持英文。",
    "FakeTT":   "All field values must be written in English.",
    "WildFake": "所有字段内容必须用中文书写，即使原始 claims 是英文。JSON 键保持英文。",
}


def get_system_prompt(dataset: str) -> str:
    lang = LANG_INSTRUCTIONS.get(dataset, "Use the same language as the video content.")
    return SYSTEM_PROMPT_TEMPLATE.format(lang_instruction=lang)


def format_claims_text(claims_data: dict | None) -> str:
    """Format pre-extracted claims for injection into the prompt."""
    if not claims_data:
        return "(no claims available)"
    all_claims = claims_data.get("all_claims", [])
    if not all_claims:
        return "(no claims available)"
    lines = []
    for c in all_claims[:6]:
        cid = c.get("id", "?")
        txt = c.get("claim", "")
        conf = c.get("confidence", 0.0)
        lines.append(f"  [{cid}] {txt} (conf={conf:.2f})")
    return "\n".join(lines) if lines else "(no claims available)"
