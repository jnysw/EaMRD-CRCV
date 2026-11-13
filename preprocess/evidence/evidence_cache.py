"""
Evidence Filter Cache
Cache key: SHA256(claim + event_summary + url)
Storage:   data/cache/{dataset}/evidence_filter_cache.json
"""

import hashlib
import json
import os
import threading

_lock = threading.Lock()


def make_cache_key(claim: str, event_summary: str, url: str) -> str:
    """SHA-256 key for a (claim, event_summary, url) triple."""
    raw = (claim or "") + "|||" + (event_summary or "") + "|||" + (url or "")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cache(cache_path: str) -> dict:
    """Load the entire cache dict from disk."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_cache(cache: dict, cache_path: str) -> None:
    """Atomically save cache dict to disk."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    os.replace(tmp, cache_path)


class EvidenceFilterCache:
    """Thread-safe in-memory + on-disk cache for evidence filter LLM results."""

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self._cache = load_cache(cache_path)
        self._dirty = 0          # number of unsaved writes
        self._save_every = 20    # flush to disk every N writes

    def get(self, claim: str, event_summary: str, url: str):
        key = make_cache_key(claim, event_summary, url)
        with _lock:
            return self._cache.get(key)

    def set(self, claim: str, event_summary: str, url: str, result: dict) -> None:
        key = make_cache_key(claim, event_summary, url)
        with _lock:
            self._cache[key] = result
            self._dirty += 1
            if self._dirty >= self._save_every:
                save_cache(self._cache, self.cache_path)
                self._dirty = 0

    def flush(self) -> None:
        with _lock:
            if self._dirty > 0:
                save_cache(self._cache, self.cache_path)
                self._dirty = 0

    def __len__(self):
        return len(self._cache)
