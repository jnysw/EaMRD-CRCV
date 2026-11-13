"""
Event-aware Evidence Filter (Modality-aware, v2)
=================================================
Changes from v1:
- Evidence is grouped by claim modality (OCR/ASR/VIS/TTL)
- Each (claim, modality) group is sent to LLM in ONE batch call (up to 10 evidences)
- Cache key now includes modality
- Output records include claim_id and claim_modality for downstream reasoning

Pipeline:
    For each claim in all_claims:
        1. Find evidence items linked to this claim_id
        2. Determine modality from claim_id prefix (OCR/ASR/VIS/TTL)
        3. Take up to max_evidence_per_modality items
        4. Batch-evaluate with LLM (one call)
        5. Score each evidence: 0.4*E + 0.4*U + 0.2*T
    Combine all scored evidence across claims
        6. Filter: event_match_score >= threshold
        7. Sort by final_score desc
        8. Return top_k

Usage:
    from preprocess.evidence.evidence_filter import EventAwareEvidenceFilter
    filt = EventAwareEvidenceFilter(api_key=..., base_url=...)
    result = filt.filter_vid(vid, claims, event_info, raw_evidences)
"""

import hashlib
import json
import os

from .evidence_cache import EvidenceFilterCache
from .evidence_scorer import compute_final_score
from .evidence_verifier import (
    EvidenceVerifier,
    modality_from_claim_id,
    normalise_modality,
    FALLBACK_RESULT,
)

_DEFAULT_CACHE_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "cache",
    "evidence_filter_cache_v2.json"   # v2 to avoid mixing with old cache
))


# ─── Cache key ────────────────────────────────────────────────────────────────

def _batch_cache_key(claim: str, modality: str, event_summary: str, urls: list) -> str:
    """Cache key for a (claim, modality, evidence_batch) triple."""
    url_str = "|".join(sorted(urls))
    raw = (claim or "") + "|||" + modality + "|||" + (event_summary or "") + "|||" + url_str
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _build_event_description(event_info: dict) -> str:
    parts = []
    if event_info.get("event_summary"):
        parts.append(event_info["event_summary"])
    if event_info.get("entities"):
        parts.append("Key entities: " + ", ".join(event_info["entities"][:5]))
    if event_info.get("location_information") and event_info["location_information"] != "unknown":
        parts.append("Location: " + event_info["location_information"])
    if event_info.get("time_information") and event_info["time_information"] != "unknown":
        parts.append("Time: " + event_info["time_information"])
    if event_info.get("event_type"):
        parts.append("Event type: " + event_info["event_type"])
    return "\n".join(parts)


def _group_evidence_by_claim(raw_evidences: list) -> dict:
    """
    Group evidence items by claim_id.
    Returns dict: claim_id -> list[evidence_dict]
    """
    groups = {}
    for ev in raw_evidences:
        cid = ev.get("claim_id", "TTL_0")
        groups.setdefault(cid, []).append(ev)
    return groups


# ─── Main class ──────────────────────────────────────────────────────────────

class EventAwareEvidenceFilter:
    """
    Modality-aware, batch evidence filter using LLM scoring.

    Parameters
    ----------
    api_key : str
    base_url : str
    model : str
    event_match_threshold : float  — evidences below this are discarded
    top_k : int  — final top evidences kept
    max_evidence_per_modality : int  — max per (claim, modality) batch
    cache_path : str
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://runapi.co/v1",
        model: str = "gpt-4o-mini",
        event_match_threshold: float = 0.4,
        top_k: int = 3,
        max_evidence_per_modality: int = 10,
        cache_path: str = _DEFAULT_CACHE_PATH,
        lang: str = "zh",
    ):
        self.threshold  = event_match_threshold
        self.top_k      = top_k
        self.max_ev     = max_evidence_per_modality
        self.verifier   = EvidenceVerifier(base_url=base_url, api_key=api_key, model=model, lang=lang)
        self.cache      = EvidenceFilterCache(cache_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter_vid(
        self,
        vid: str,
        claims: list,
        event_info: dict,
        raw_evidences: list,
        top_k: int = None,
        event_match_threshold: float = None,
    ) -> dict:
        """
        Filter evidence for all claims of a video.

        Parameters
        ----------
        vid : str
        claims : list[dict]  — from claims cache, each has {id, claim, source_modality, ...}
        event_info : dict    — from event_summary_FakeSV.jsonl
        raw_evidences : list[dict]  — all raw evidence for this vid
        top_k, event_match_threshold : override defaults

        Returns
        -------
        dict:
            vid, n_claims, n_retrieved, n_passed_threshold, n_verified,
            claim_results (per-claim details),
            verified_evidence (final top_k across all claims)
        """
        if top_k is None:
            top_k = self.top_k
        if event_match_threshold is None:
            event_match_threshold = self.threshold

        event_summary      = event_info.get("event_summary", "")
        verification_targets = event_info.get("verification_targets", [])
        event_description  = _build_event_description(event_info)

        # Group raw evidence by claim_id
        ev_by_claim = _group_evidence_by_claim(raw_evidences)

        all_scored = []         # all scored evidence across all claims
        claim_results = []      # per-claim breakdown

        for claim_obj in claims:
            claim_id  = claim_obj.get("id", "TTL_0")
            claim_txt = claim_obj.get("claim", "")
            modality  = normalise_modality(
                claim_obj.get("source_modality", modality_from_claim_id(claim_id))
            )

            # Find evidence for this claim_id
            claim_evs = ev_by_claim.get(claim_id, [])
            if not claim_evs:
                claim_results.append({
                    "claim_id": claim_id, "claim": claim_txt, "modality": modality,
                    "n_evidence": 0, "scored_evidence": [],
                })
                continue

            # Limit to max_ev; prefer unique URLs
            seen_urls = set()
            unique_evs = []
            for ev in claim_evs:
                url = ev.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    unique_evs.append(ev)
            batch = unique_evs[:self.max_ev]

            # --- Cache lookup (per batch) ---
            cache_key = _batch_cache_key(claim_txt, modality, event_summary,
                                          [e.get("url","") for e in batch])
            cached_results = self.cache._cache.get(cache_key)

            if cached_results is not None and len(cached_results) == len(batch):
                eval_results = cached_results
            else:
                eval_results = self.verifier.evaluate_batch(
                    claim=claim_txt,
                    claim_modality=modality,
                    event_description=event_description,
                    verification_targets=verification_targets,
                    evidences=batch,
                )
                # Save to cache
                import threading
                with threading.Lock():
                    self.cache._cache[cache_key] = eval_results
                    self.cache._dirty += 1
                    if self.cache._dirty >= self.cache._save_every:
                        from .evidence_cache import save_cache
                        save_cache(self.cache._cache, self.cache.cache_path)
                        self.cache._dirty = 0

            # Merge eval results with evidence items
            scored_for_claim = []
            for ev, er in zip(batch, eval_results):
                credibility_boost = ev.get("credibility_boost", 0.3)
                final = compute_final_score(
                    er.get("event_match_score", 0.0),
                    er.get("evidence_usefulness_score", 0.0),
                    er.get("time_consistency", 0.0),
                    credibility_boost=credibility_boost,
                )
                scored_for_claim.append({
                    **ev,
                    "claim_id":                  claim_id,
                    "claim_modality":             modality,
                    "event_match_score":          er.get("event_match_score", 0.0),
                    "evidence_usefulness_score":  er.get("evidence_usefulness_score", 0.0),
                    "relation":                   er.get("relation", "unknown"),
                    "time_consistency":           er.get("time_consistency", 0.0),
                    "reason":                     er.get("reason", ""),
                    "credibility_boost":          credibility_boost,
                    "final_score":                final,
                })

            all_scored.extend(scored_for_claim)
            claim_results.append({
                "claim_id":       claim_id,
                "claim":          claim_txt,
                "modality":       modality,
                "n_evidence":     len(batch),
                "scored_evidence": scored_for_claim,
            })

        self.cache.flush()

        # Step 1: filter by event_match_score
        passed = [e for e in all_scored if e["event_match_score"] >= event_match_threshold]

        # Step 2: sort by final_score (descending)
        passed.sort(key=lambda x: x["final_score"], reverse=True)

        # Step 3: per-modality top-k + global URL dedup
        # Each modality (OCR/ASR/VIS/TTL) keeps at most top_k evidences.
        # URLs are deduplicated globally across modalities.
        from collections import defaultdict
        modality_counts: dict = defaultdict(int)
        seen_urls: set = set()
        verified = []
        for ev in passed:   # already sorted by final_score desc
            modality = ev.get("claim_modality", "TTL")
            url = ev.get("url", "")
            if url in seen_urls:
                continue
            if modality_counts[modality] < top_k:
                seen_urls.add(url)
                verified.append(ev)
                modality_counts[modality] += 1

        return {
            "vid":                vid,
            "n_claims":           len(claims),
            "n_retrieved":        len(raw_evidences),
            "n_passed_threshold": len(passed),
            "n_verified":         len(verified),
            "claim_results":      claim_results,
            "all_scored":         all_scored,
            "verified_evidence":  verified,
        }

    # ------------------------------------------------------------------
    # Backward-compat wrapper (single claim, flat evidence list)
    # ------------------------------------------------------------------

    def filter(
        self,
        vid: str,
        claim: str,
        event_info: dict,
        evidences: list,
        top_k: int = None,
        event_match_threshold: float = None,
        claim_modality: str = "TTL",
    ) -> dict:
        """
        Single-claim convenience method (backward compatible with v1).
        Wraps filter_vid().
        """
        fake_claim_obj = {"id": f"{claim_modality}_0", "claim": claim,
                          "source_modality": claim_modality.lower()}
        for ev in evidences:
            if "claim_id" not in ev:
                ev["claim_id"] = f"{claim_modality}_0"
        return self.filter_vid(
            vid=vid,
            claims=[fake_claim_obj],
            event_info=event_info,
            raw_evidences=evidences,
            top_k=top_k,
            event_match_threshold=event_match_threshold,
        )
