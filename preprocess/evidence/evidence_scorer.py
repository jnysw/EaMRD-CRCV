"""
Evidence Scorer
Score = 0.30*E + 0.30*U + 0.10*T + 0.30*C

C (credibility_boost) is derived from source site credibility:
  - Tier 1 (fact-check: snopes.com, factcheck.org, politifact.com): C = 1.0
  - Tier 2 (authority news: reuters.com, bbc.com, apnews.com, etc.): C = 0.6~0.8
  - General web: C = 0.3
  - Social media (youtube, facebook, tiktok, reddit): C = 0.0
"""


def compute_final_score(
    event_match_score: float,
    evidence_usefulness_score: float,
    time_consistency: float,
    credibility_boost: float = 0.3,
) -> float:
    """
    Weighted combination of four dimensions.

    E (event_match_score):          Does this document refer to the same real-world event?
    U (evidence_usefulness_score):  Can this document help verify the claim?
    T (time_consistency):           Is the temporal context consistent?
    C (credibility_boost):          Source credibility (fact-check > news > general > social)

    Score = 0.30*E + 0.30*U + 0.10*T + 0.30*C
    """
    e = max(0.0, min(1.0, float(event_match_score)))
    u = max(0.0, min(1.0, float(evidence_usefulness_score)))
    t = max(0.0, min(1.0, float(time_consistency)))
    c = max(0.0, min(1.0, float(credibility_boost)))
    return round(0.30 * e + 0.30 * u + 0.10 * t + 0.30 * c, 4)


def score_evidence_item(eval_result: dict, credibility_boost: float = 0.3) -> float:
    """Compute final score from a verifier eval dict."""
    return compute_final_score(
        eval_result.get("event_match_score", 0.0),
        eval_result.get("evidence_usefulness_score", 0.0),
        eval_result.get("time_consistency", 0.0),
        credibility_boost=credibility_boost,
    )