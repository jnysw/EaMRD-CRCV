"""
Event Summary utilities: JSON parsing, cache I/O, output helpers.
"""

import json
import os
import re
from pathlib import Path
from typing import Optional


VALID_EVENT_TYPES = {
    "natural_disaster", "public_event", "social_issue",
    "political", "crime", "health", "other",
}


def parse_event_summary_json(raw: str, vid: str) -> Optional[dict]:
    """
    Parse LLM response into an event summary dict.
    Returns None if parsing fails entirely.
    """
    if not raw:
        return None

    # Strip markdown code fences
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()

    # Try full parse
    try:
        data = json.loads(raw)
        return _validate_and_fix(data, vid)
    except json.JSONDecodeError:
        pass

    # Try extracting the first {...} block
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(raw[start:end + 1])
            return _validate_and_fix(data, vid)
        except json.JSONDecodeError:
            pass

    return None


def _validate_and_fix(data: dict, vid: str) -> dict:
    """Ensure all required fields are present with sensible defaults."""
    result = {
        "vid": vid,
        "event_summary": str(data.get("event_summary", "")).strip() or "unknown",
        "entities": _ensure_list(data.get("entities", [])),
        "event_type": _fix_event_type(data.get("event_type", "other")),
        "time_information": str(data.get("time_information", "unknown")).strip() or "unknown",
        "location_information": str(data.get("location_information", "unknown")).strip() or "unknown",
        "verification_targets": _ensure_list(data.get("verification_targets", [])),
    }
    return result


def _ensure_list(val) -> list:
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str) and val.strip():
        return [val.strip()]
    return []


def _fix_event_type(raw: str) -> str:
    raw = str(raw).lower().strip().replace(" ", "_").replace("-", "_")
    return raw if raw in VALID_EVENT_TYPES else "other"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cache(cache_path: str) -> dict:
    """Load {vid: result} cache from JSON file."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_cache(cache: dict, cache_path: str) -> None:
    """Save cache dict to JSON file atomically."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, cache_path)


# ---------------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------------

def append_to_jsonl(record: dict, jsonl_path: str) -> None:
    """Append a single record to the output JSONL file."""
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_done_vids_from_jsonl(jsonl_path: str) -> set:
    """Return set of already-processed video IDs from existing JSONL output."""
    done = set()
    if not os.path.exists(jsonl_path):
        return done
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    vid = rec.get("vid") or rec.get("video_id")
                    if vid:
                        done.add(str(vid))
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass
    return done


# ---------------------------------------------------------------------------
# Pretty print for inspection
# ---------------------------------------------------------------------------

def print_event_summary(record: dict, claim_text: str = "") -> None:
    print(f"\n{'='*60}")
    print(f"Video ID : {record.get('vid', '?')}")
    if claim_text:
        print(f"Claim    : {claim_text[:120]}")
    print(f"Type     : {record.get('event_type', '?')}")
    print(f"Time     : {record.get('time_information', '?')}")
    print(f"Location : {record.get('location_information', '?')}")
    print(f"Summary  :\n  {record.get('event_summary', '?')}")
    ents = record.get("entities", [])
    if ents:
        print(f"Entities : {', '.join(ents[:8])}")
    targets = record.get("verification_targets", [])
    if targets:
        print("Verification Targets:")
        for i, t in enumerate(targets, 1):
            print(f"  {i}. {t}")
    print("="*60)
