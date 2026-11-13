"""
Evidence Verifier (Modality-aware, Batch Evaluation)
=====================================================
Key changes from v1:
- Prompt now includes claim_modality (OCR/ASR/VIS/TTL) + explanation
- One LLM call evaluates ALL evidence candidates for a (claim, modality) pair
- Returns a list of per-evidence scores (one JSON object per evidence)
- Falls back to per-evidence calls if batch JSON parse fails

Output per evidence:
{
    "event_match_score":       0-1,
    "evidence_usefulness_score": 0-1,
    "relation":                "support|refute|neutral|unknown",
    "time_consistency":        0-1,
    "reason":                  "..."
}
"""

import json
import re
import time
import warnings

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# ─── Prompts ─────────────────────────────────────────────────────────────────
# Language-aware prompt selection
# Use SYSTEM_PROMPT_EN + USER_PROMPT_TEMPLATE_EN for English content (FakeTT)
# Use SYSTEM_PROMPT    + USER_PROMPT_TEMPLATE     for Chinese content (FakeSV)

SYSTEM_PROMPT = (
    "You are an evidence retrieval evaluator for video misinformation detection. "
    "Your goal is NOT to judge whether the claim is true. "
    "Your goal is to determine whether retrieved documents provide useful external evidence "
    "for verifying the video claim. "
    "The evidence is retrieved based on modality-specific claims. "
    "You must evaluate evidence relevance with respect to the given claim and its modality. "
    "Always respond with valid JSON only — no markdown fences, no extra text."
)

SYSTEM_PROMPT_EN = (
    "You are an expert evidence evaluator for detecting misinformation in English-language "
    "short videos (TikTok, social media). "
    "Your task: evaluate whether retrieved web documents provide USEFUL evidence for "
    "verifying a specific video claim. "
    "\n\n"
    "CRITICAL: Many fake videos on social media use REAL events but present them in a "
    "MISLEADING context (wrong date, wrong location, wrong cause, exaggerated claim, "
    "satire presented as fact, or out-of-context footage). "
    "A document that confirms the surface-level topic is NOT necessarily useful evidence — "
    "it must confirm or refute the SPECIFIC CLAIM as presented in the video. "
    "\n\n"
    "Scoring guidance:\n"
    "event_match_score: Does this document describe the SAME specific incident?\n"
    "  1.0 = same event, same time, same place, same people\n"
    "  0.7 = same event but missing some specifics\n"
    "  0.4 = related topic but likely a DIFFERENT incident or time period\n"
    "  0.1 = only shares keywords; clearly a different event\n"
    "\n"
    "evidence_usefulness_score: Can this document help verify the EXACT claim?\n"
    "  1.0 = directly confirms or refutes the specific factual assertion\n"
    "  0.7 = relevant context that partially addresses the claim\n"
    "  0.3 = background/topical info; does NOT address the claim's specific assertion\n"
    "  0.0 = irrelevant, logo pages, disambiguation pages, unrelated content\n"
    "\n"
    "relation:\n"
    "  support  = evidence confirms the specific claim as stated\n"
    "  refute   = evidence contradicts or debunks the claim\n"
    "  neutral  = evidence is relevant but neither confirms nor refutes\n"
    "  unknown  = cannot determine relevance\n"
    "\n"
    "IMPORTANT: If evidence supports a general topic but the VIDEO CLAIM adds specific "
    "details (numbers, names, dates, locations, causation) that are NOT confirmed by the "
    "evidence, score evidence_usefulness_score ≤ 0.5 and relation=neutral. "
    "Avoid giving 1.0 scores unless the document DIRECTLY addresses the specific claim. "
    "\n\n"
    "Always respond with valid JSON only — no markdown fences, no extra text."
)

USER_PROMPT_TEMPLATE = """\
Video event description:
{event_description}

Claim modality:
{claim_modality}

Possible modalities:
- OCR: text extracted from video frames
- ASR: speech transcript from audio
- VIS: visual content description from video frames
- TTL: video title or metadata text

Verification targets:
{verification_targets}

Claim:
{claim}

Retrieved evidence candidates from this modality:
{evidence_list}

Evaluate EACH retrieved document:

1. Event consistency:
Does this document refer to the same real-world event?
Consider: entities, location, event type, temporal information, event context.

2. Evidence usefulness:
Can this document help verify the given claim?
Consider:
- whether it provides factual information related to the claim
- whether it supports or contradicts the claim
- whether it is only topically similar but refers to another event

Important:
- Do not assume similar keywords indicate the same event.
- A document about a different but similar event should receive a low event_match_score.
- Evidence absence is acceptable if no reliable document is found.

Output a JSON ARRAY with one object per evidence candidate (same order as input).
Each object must have exactly these keys:
[
  {{
    "evidence_id": 1,
    "event_match_score": <float 0-1>,
    "evidence_usefulness_score": <float 0-1>,
    "relation": "<support|refute|neutral|unknown>",
    "time_consistency": <float 0-1>,
    "reason": "<one sentence>"
  }},
  ...
]
Output ONLY the JSON array, no markdown, no extra text."""

USER_PROMPT_TEMPLATE_EN = """\
Video event description:
{event_description}

Claim modality: {claim_modality}
Modality definitions:
- OCR: on-screen text / captions extracted from video frames
- ASR: spoken words from audio track
- VIS: description of what is visually depicted in video frames
- TTL: video title or hashtag metadata

Specific things to verify:
{verification_targets}

Claim from video ({claim_modality}):
{claim}

Retrieved web documents to evaluate:
{evidence_list}

For EACH document, answer:
(a) Event match: Does this document describe the EXACT SAME real-world incident as the video claim?
    — Same specific event at the same time and place = high score
    — Same topic but different incident or time period = low score (0.1–0.3)
    — Background/wiki/general article about the same topic but not the specific event = 0.2–0.4

(b) Evidence usefulness: Does this document help verify the SPECIFIC assertion in the claim?
    — Directly confirms or refutes the specific fact claimed = high (0.7–1.0)
    — Provides relevant context but doesn't address the specific claim detail = medium (0.3–0.6)
    — Only confirms the general topic exists, not the specific claim = low (0.1–0.3)
    — Logo pages, disambiguation pages, social media profiles = 0.0

(c) Relation: Does the document support, refute, or stay neutral on the specific claim?
    — Use "refute" only if the document EXPLICITLY contradicts what the claim asserts
    — Use "support" only if it DIRECTLY confirms the specific claim as stated
    — Otherwise use "neutral"

(d) Time consistency: Do the document's date/timeframe match the video's implied timeframe?

(e) reason: One sentence explaining your evaluation.

CRITICAL CALIBRATION:
- Do NOT give 1.0 scores unless the document directly addresses the SPECIFIC claim
- A news article confirming a general event (e.g. "this person exists" or "this event happened")
  that does NOT address the video's specific assertion should score usefulness ≤ 0.5
- Political commentary, analysis, and opinion pieces have lower usefulness than primary news reports
- Fact-check articles that EXPLICITLY debunk a claim = high refute score
- Generic background Wikipedia articles = event_match ≤ 0.4, usefulness ≤ 0.3

Output a JSON ARRAY with one object per evidence candidate (same order as input).
Each object must have exactly these keys:
[
  {{
    "evidence_id": 1,
    "event_match_score": <float 0-1>,
    "evidence_usefulness_score": <float 0-1>,
    "relation": "<support|refute|neutral|unknown>",
    "time_consistency": <float 0-1>,
    "reason": "<one sentence>"
  }},
  ...
]
Output ONLY the JSON array, no markdown, no extra text."""

# ─── Modality helpers ────────────────────────────────────────────────────────

MODALITY_ALIASES = {
    "title":  "TTL",
    "ocr":    "OCR",
    "asr":    "ASR",
    "vision": "VIS",
    "vis":    "VIS",
    "ttl":    "TTL",
}

def normalise_modality(raw: str) -> str:
    """Convert source_modality / claim_id prefix to canonical OCR/ASR/VIS/TTL."""
    if not raw:
        return "TTL"
    raw = raw.lower().strip()
    return MODALITY_ALIASES.get(raw, raw.upper()[:3])

def modality_from_claim_id(claim_id: str) -> str:
    """Extract modality from claim_id like 'OCR_1', 'TTL_2', 'ASR_3'."""
    prefix = claim_id.split("_")[0].upper() if claim_id else ""
    return normalise_modality(prefix)

# ─── JSON parsing ────────────────────────────────────────────────────────────

FALLBACK_RESULT = {
    "event_match_score":         0.0,
    "evidence_usefulness_score": 0.0,
    "relation":                  "unknown",
    "time_consistency":          0.0,
    "reason":                    "LLM evaluation failed",
}

def _clamp_result(result: dict) -> dict:
    out = dict(FALLBACK_RESULT)
    out.update(result)
    for k in ("event_match_score", "evidence_usefulness_score", "time_consistency"):
        try:
            out[k] = max(0.0, min(1.0, float(out[k])))
        except (TypeError, ValueError):
            out[k] = 0.0
    valid_relations = {"support", "refute", "neutral", "unknown"}
    rel = str(out.get("relation", "")).lower()
    out["relation"] = rel if rel in valid_relations else "unknown"
    return out

def _strip_fences(text: str) -> str:
    """Remove markdown code fences."""
    return re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

def _parse_array(raw: str) -> list:
    """Try to parse a JSON array from LLM output."""
    text = _strip_fences(raw)
    # Direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    # Find first [...] block
    m = re.search(r"\[[\s\S]*?\]", text)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    return []

def _parse_single(raw: str) -> dict:
    """Try to parse a single JSON object from LLM output."""
    text = _strip_fences(raw)
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*?\}", text)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {}

# ─── Evidence list formatter ─────────────────────────────────────────────────

def _format_evidence_list(evidences: list) -> str:
    """Format a list of evidence dicts into structured prompt text."""
    parts = []
    for i, ev in enumerate(evidences, 1):
        title   = (ev.get("title") or "")[:200]
        url     = (ev.get("url") or "")[:150]
        content = (
            ev.get("summary") or ev.get("snippet") or ev.get("content") or ""
        )[:400]
        parts.append(
            f"[Evidence_{i}]\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content: {content}"
        )
    return "\n\n".join(parts)

# ─── Main class ──────────────────────────────────────────────────────────────

class EvidenceVerifier:
    """
    Modality-aware LLM evidence verifier.

    Evaluates a batch of evidence candidates for a single (claim, modality) pair
    in one LLM call, returning one score dict per evidence.

    Args:
        lang: "zh" for Chinese (FakeSV) or "en" for English (FakeTT).
              Selects the appropriate system prompt and user prompt template.
    """

    def __init__(
        self,
        base_url: str = "https://runapi.co/v1",
        api_key: str = "",
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: int = 90,
        lang: str = "zh",
    ):
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/chat/completions"):
            base_url += "/chat/completions"
        self.url         = base_url
        self.api_key     = api_key
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.timeout     = timeout
        self.lang        = lang.lower().strip()
        # Select prompts based on language
        if self.lang == "en":
            self._system_prompt   = SYSTEM_PROMPT_EN
            self._user_template   = USER_PROMPT_TEMPLATE_EN
        else:
            self._system_prompt   = SYSTEM_PROMPT
            self._user_template   = USER_PROMPT_TEMPLATE
        self._session    = self._make_session()

    def _make_session(self) -> requests.Session:
        """Create a Session with limited connection pool to avoid WinError 10055."""
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=2,
            pool_maxsize=4,
            max_retries=0,   # we do our own retry logic
        )
        session.mount("http://",  adapter)
        session.mount("https://", adapter)
        return session

    # ------------------------------------------------------------------
    def _call_llm(self, user_msg: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user",   "content": user_msg},
            ],
            "stream": False,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(4):
            try:
                resp = self._session.post(
                    self.url, headers=headers,
                    data=json.dumps(payload),
                    timeout=self.timeout,
                    verify=False,
                )
                resp.raise_for_status()
                time.sleep(0.3)   # gentle pacing to avoid socket exhaustion
                return resp.json()["choices"][0]["message"]["content"]
            except requests.exceptions.ProxyError as e:
                wait = 15 * (attempt + 1)
                print(f"[verifier] ProxyError on attempt {attempt+1}, "
                      f"rebuilding session, sleeping {wait}s ...", flush=True)
                try: self._session.close()
                except: pass
                self._session = self._make_session()
                time.sleep(wait)
            except requests.exceptions.ConnectionError as e:
                wait = 10 * (attempt + 1)
                print(f"[verifier] ConnectionError on attempt {attempt+1}, "
                      f"sleeping {wait}s ...", flush=True)
                time.sleep(wait)
            except Exception:
                raise
        raise RuntimeError("_call_llm failed after 4 attempts")

    # ------------------------------------------------------------------
    def evaluate_batch(
        self,
        claim: str,
        claim_modality: str,
        event_description: str,
        verification_targets: list,
        evidences: list,
    ) -> list:
        """
        Evaluate a list of evidence items in ONE LLM call.

        Returns list[dict] with one score dict per evidence (same order).
        Falls back to per-evidence calls if batch parsing fails.
        """
        if not evidences:
            return []

        vt_text = "\n".join(f"- {t}" for t in (verification_targets or []))
        ev_text = _format_evidence_list(evidences)

        user_msg = self._user_template.format(
            event_description=event_description[:500],
            claim_modality=claim_modality,
            verification_targets=vt_text or "(none)",
            claim=claim[:300],
            evidence_list=ev_text,
        )

        last_exc = None
        for attempt in range(2):
            try:
                raw = self._call_llm(user_msg)
                parsed = _parse_array(raw)
                if len(parsed) == len(evidences):
                    return [_clamp_result(r) for r in parsed]
                # Length mismatch → retry
                last_exc = ValueError(
                    f"Expected {len(evidences)} results, got {len(parsed)}. raw={raw[:200]}"
                )
            except Exception as exc:
                last_exc = exc
            if attempt == 0:
                time.sleep(3)

        # --- Fallback: per-evidence single calls ---
        print(f"[WARN] Batch parse failed ({last_exc}), falling back to per-evidence calls",
              flush=True)
        return [self._evaluate_single(claim, claim_modality, event_description,
                                      verification_targets, ev)
                for ev in evidences]

    # ------------------------------------------------------------------
    def _evaluate_single(
        self,
        claim: str,
        claim_modality: str,
        event_description: str,
        verification_targets: list,
        evidence: dict,
    ) -> dict:
        """Single-evidence fallback using same prompt (evidence_list = one item)."""
        vt_text = "\n".join(f"- {t}" for t in (verification_targets or []))
        ev_text = _format_evidence_list([evidence])

        user_msg = self._user_template.format(
            event_description=event_description[:500],
            claim_modality=claim_modality,
            verification_targets=vt_text or "(none)",
            claim=claim[:300],
            evidence_list=ev_text,
        )

        # Change output instruction to single object for single evidence
        user_msg = user_msg.replace(
            "Output a JSON ARRAY with one object per evidence candidate (same order as input).\n"
            "Each object must have exactly these keys:",
            "Output a single JSON object with exactly these keys:"
        ).replace(
            "[\n  {{\n    \"evidence_id\": 1,",
            "{{\n    \"evidence_id\": 1,"
        ).replace(
            "  }},\n  ...\n]",
            "}}"
        )

        last_exc = None
        for attempt in range(2):
            try:
                raw = self._call_llm(user_msg)
                # Try array first, then single object
                parsed_list = _parse_array(raw)
                if parsed_list:
                    return _clamp_result(parsed_list[0])
                parsed = _parse_single(raw)
                if parsed and "event_match_score" in parsed:
                    return _clamp_result(parsed)
                last_exc = ValueError(f"Parse failed: {raw[:200]}")
            except Exception as exc:
                last_exc = exc
            if attempt == 0:
                time.sleep(3)

        fallback = dict(FALLBACK_RESULT)
        fallback["reason"] = f"LLM failed: {str(last_exc)[:100]}"
        return fallback

    # ------------------------------------------------------------------
    def evaluate(
        self,
        claim: str,
        event_description: str,
        verification_targets: list,
        evidence: dict,
        claim_modality: str = "TTL",
    ) -> dict:
        """Single-evidence convenience wrapper (for backward compat)."""
        results = self.evaluate_batch(
            claim=claim,
            claim_modality=claim_modality,
            event_description=event_description,
            verification_targets=verification_targets,
            evidences=[evidence],
        )
        return results[0] if results else dict(FALLBACK_RESULT)
