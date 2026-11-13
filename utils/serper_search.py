"""Serper Web Search API wrapper (Google Search via serper.dev).

For English datasets (FakeTT) where Bocha has limited English coverage.
API docs: https://serper.dev/
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

DEFAULT_SEARCH_URL = "https://google.serper.dev/search"
DEFAULT_RESULT_COUNT = 5
DEFAULT_MIN_INTERVAL = 0.3   # 300ms between requests

_last_request_time = 0.0


def _get_api_key() -> str:
    # Re-read .env every time so key can be swapped without restarting the process
    load_dotenv(override=True)
    key = os.getenv("SERPER_API_KEY", "")
    if not key:
        raise ValueError("SERPER_API_KEY is not set. Add it to .env.")
    return key


def _rate_limit(min_interval: float = DEFAULT_MIN_INTERVAL) -> None:
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_request_time = time.time()


def _parse_result_item(item: dict) -> dict:
    """Normalize Serper result to match Bocha's output schema."""
    return {
        "title":         item.get("title", ""),
        "url":           item.get("link", ""),
        "snippet":       item.get("snippet", ""),
        "summary":       item.get("snippet", ""),   # Serper has no separate summary
        "datePublished": item.get("date", ""),
        "siteName":      item.get("domain", ""),
    }


def serper_search(
    query: str,
    count: int | None = None,
    gl: str = "us",
    hl: str = "en",
    max_retries: int = 3,
) -> list[dict]:
    """Call Serper Google Search API; return normalized result list.

    Args:
        query: Search query string.
        count: Number of results (default 5, max 10).
        gl:    Country code for Google results (default 'us').
        hl:    Language code (default 'en').
    """
    if not query or not query.strip():
        return []

    api_key = _get_api_key()
    url = os.getenv("SERPER_SEARCH_URL", DEFAULT_SEARCH_URL)
    count = count or int(os.getenv("SERPER_RESULT_COUNT", DEFAULT_RESULT_COUNT))
    count = min(max(1, count), 10)

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "q":   query.strip(),
        "gl":  gl,
        "hl":  hl,
        "num": count,
    }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            _rate_limit()
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            # Key exhausted / unauthorized — stop immediately
            if response.status_code in (402, 403, 429):
                msg = f"[serper_search] API key exhausted or unauthorized (HTTP {response.status_code}). Update SERPER_API_KEY in .env and restart."
                print(msg)
                raise RuntimeError(msg)
            response.raise_for_status()
            data = response.json()
            organic = data.get("organic", [])
            return [_parse_result_item(item) for item in organic]
        except RuntimeError:
            raise   # propagate key-exhaustion immediately
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"[serper_search] Failed query={query!r}: {exc}")

    return []


def serper_search_multi(
    queries: list[str],
    count: int | None = None,
) -> list[dict]:
    """Search multiple queries and deduplicate by URL."""
    seen_urls: set[str] = set()
    merged: list[dict] = []

    for query in queries:
        results = serper_search(query=query, count=count)
        for item in results:
            url = item.get("url", "")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            merged.append(item)

    return merged
