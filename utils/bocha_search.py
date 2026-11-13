"""Bocha Web Search API wrapper for R³CoT evidence verification."""

import json
import os
import time
from typing import Any

import requests
from dotenv import load_dotenv

try:
    # When imported as part of the utils package
    from utils.authoritative_sources import (
        ALL_AUTHORITY_DOMAINS,
        domains_to_include,
        get_tier_include,
    )
except ImportError:
    # Fallback: same-directory import (e.g. running bocha_search.py directly)
    from authoritative_sources import (  # type: ignore
        ALL_AUTHORITY_DOMAINS,
        domains_to_include,
        get_tier_include,
    )

load_dotenv(override=True)

DEFAULT_SEARCH_URL = "https://api.bochaai.com/v1/web-search"
DEFAULT_FRESHNESS = "noLimit"  # 官方推荐；oneYear 容易漏掉旧闻
DEFAULT_RESULT_COUNT = 5
DEFAULT_MIN_INTERVAL = 0.5

_last_request_time = 0.0


def _get_api_key() -> str:
    key = os.getenv("BOCHA_API_KEY", "")
    if not key:
        raise ValueError("BOCHA_API_KEY is not set. Add it to .env or environment variables.")
    return key


def _env_bool(name: str, default: bool = True) -> bool:
    val = os.getenv(name, str(default)).lower()
    return val in ("1", "true", "yes", "on")


def _default_include() -> str | None:
    """Default authority domains from env or built-in list."""
    if not _env_bool("BOCHA_USE_AUTHORITY", True):
        return None
    custom = os.getenv("BOCHA_INCLUDE_DOMAINS", "").strip()
    if custom:
        return custom
    tier = os.getenv("BOCHA_AUTHORITY_TIER", "all_authority")
    include = get_tier_include(tier)
    if include:
        return include
    return domains_to_include(ALL_AUTHORITY_DOMAINS)


def _rate_limit(min_interval: float = DEFAULT_MIN_INTERVAL) -> None:
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_request_time = time.time()


def _parse_result_item(item: dict) -> dict:
    return {
        "title": item.get("name") or item.get("title", ""),
        "url": item.get("url", ""),
        "snippet": item.get("snippet", ""),
        "summary": item.get("summary", ""),
        "datePublished": item.get("datePublished") or item.get("dateLastCrawled", ""),
        "siteName": item.get("siteName", ""),
    }


def _extract_web_results(data: dict) -> list[dict]:
    """Parse Bocha API response; results live under data.webPages.value."""
    payload = data.get("data", data)
    if isinstance(payload, dict):
        web_pages = payload.get("webPages", {})
        if isinstance(web_pages, dict):
            values = web_pages.get("value", [])
            if isinstance(values, list):
                return [v for v in values if isinstance(v, dict)]
    # fallback: top-level webPages (older/alternate format)
    web_pages = data.get("webPages", {})
    if isinstance(web_pages, dict):
        values = web_pages.get("value", [])
        if isinstance(values, list):
            return [v for v in values if isinstance(v, dict)]
    return []


def bocha_web_search(
    query: str,
    count: int | None = None,
    freshness: str | None = None,
    summary: bool = True,
    include: str | None = None,
    exclude: str | None = None,
    use_default_include: bool = False,
    max_retries: int = 3,
) -> list[dict]:
    """Call Bocha Web Search API and return normalized result list.

    Args:
        query: Search keywords.
        include: Restrict to domains, e.g. "piyao.org.cn|news.cn" (Bocha API param).
        exclude: Exclude domains.
    """
    if not query or not query.strip():
        return []

    api_key = _get_api_key()
    url = os.getenv("BOCHA_SEARCH_URL", DEFAULT_SEARCH_URL)
    count = count or int(os.getenv("BOCHA_RESULT_COUNT", DEFAULT_RESULT_COUNT))
    freshness = freshness or os.getenv("BOCHA_FRESHNESS", DEFAULT_FRESHNESS)
    count = min(max(1, count), 8)

    if include is None and use_default_include:
        include = _default_include()

    payload: dict[str, Any] = {
        "query": query.strip(),
        "freshness": freshness,
        "summary": summary,
        "count": count,
    }
    if include:
        payload["include"] = include
    if exclude:
        payload["exclude"] = exclude

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            _rate_limit()
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()

            values = _extract_web_results(data)
            return [_parse_result_item(item) for item in values]
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"[bocha_search] Failed query={query!r} include={include!r}: {exc}")

    if last_error:
        return []
    return []


def bocha_authoritative_search(
    query: str,
    count: int | None = None,
    freshness: str | None = None,
    summary: bool = True,
    fallback_open: bool = True,
) -> tuple[list[dict], str]:
    """Tiered search: rumor/fact-check -> gov -> all authority -> open web.

    Returns (results, source_label) where source_label indicates which tier matched.
    """
    tiers: list[tuple[str, str | None]] = [
        ("rumor_factcheck", get_tier_include("rumor_factcheck")),
        ("gov", get_tier_include("gov")),
        ("all_authority", get_tier_include("all_authority")),
    ]

    for label, include in tiers:
        if not include:
            continue
        results = bocha_web_search(
            query=query,
            count=count,
            freshness=freshness,
            summary=summary,
            include=include,
        )
        if results:
            return results, label

    if fallback_open:
        results = bocha_web_search(
            query=query,
            count=count,
            freshness=freshness,
            summary=summary,
            include=None,
            use_default_include=False,
        )
        if results:
            return results, "open_web"

    return [], "none"


def bocha_web_search_multi(
    queries: list[str],
    count: int | None = None,
    freshness: str | None = None,
    summary: bool = True,
    include: str | None = None,
    authoritative: bool | None = None,
) -> list[dict]:
    """Search multiple queries and deduplicate by URL."""
    if authoritative is None:
        authoritative = _env_bool("BOCHA_USE_AUTHORITY", True)

    seen_urls: set[str] = set()
    merged: list[dict] = []

    for query in queries:
        if authoritative:
            results, _ = bocha_authoritative_search(
                query=query,
                count=count,
                freshness=freshness,
                summary=summary,
            )
        else:
            results = bocha_web_search(
                query=query,
                count=count,
                freshness=freshness,
                summary=summary,
                include=include,
            )
        for item in results:
            url = item.get("url", "")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            merged.append(item)

    return merged


def format_search_results(results: list[dict], max_chars: int = 3000) -> str:
    """Format search results into LLM-readable evidence context."""
    if not results:
        return "（未检索到相关网页结果）"

    blocks: list[str] = []
    total_len = 0

    for idx, item in enumerate(results, 1):
        title = item.get("title") or "无标题"
        url = item.get("url") or ""
        snippet = item.get("summary") or item.get("snippet") or ""
        date = item.get("datePublished") or ""
        site = item.get("siteName") or ""

        block = (
            f"[{idx}] 标题: {title}\n"
            f"来源: {site}\n"
            f"链接: {url}\n"
            f"时间: {date}\n"
            f"摘要: {snippet}\n"
        )
        if total_len + len(block) > max_chars:
            break
        blocks.append(block)
        total_len += len(block)

    return "\n".join(blocks)


def search_and_format(
    queries: list[str],
    count: int | None = None,
    freshness: str | None = None,
    max_chars: int = 3000,
    authoritative: bool | None = None,
) -> tuple[str, list[dict]]:
    """Run multi-query search and return formatted context plus raw results."""
    results = bocha_web_search_multi(
        queries,
        count=count,
        freshness=freshness,
        authoritative=authoritative,
    )
    return format_search_results(results, max_chars=max_chars), results
