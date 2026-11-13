"""Shared read-only helpers for the audit tools (no writes outside OUT_DIR)."""

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "experiments" / "audit_20261015"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_ROOT = ROOT / "data" / "cache"

# Official Full configs (per AGENT_QUICKSTART.md)
FULL_CFG = {
    "FakeSV": ROOT / "configs" / "fakesv_v2" / "best.yaml",
    "FakeTT": ROOT / "configs" / "fakett_v2" / "full.yaml",
}
FEATURE_CACHE = {"FakeSV": "FakeSV_FilteredEv", "FakeTT": "FakeTT_FilteredEv_v3"}
RAW_CACHE = {"FakeSV": "FakeSV", "FakeTT": "FakeTT"}
VERIFIED_EV = {
    "FakeSV": ROOT / "preprocess" / "output" / "verified_evidence_FakeSV_v2.jsonl",
    "FakeTT": ROOT / "preprocess" / "output" / "verified_evidence_FakeTT_v3.jsonl",
}
SPLITS = ("train", "valid", "test")


def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_split(dataset: str, split: str):
    """Return the split dataframe exactly as train.py builds it (before cache filtering)."""
    from data_utils.baseline_data import FakeSVDataset, FakeTTDataset
    cls = {"FakeSV": FakeSVDataset, "FakeTT": FakeTTDataset}[dataset]
    df = cls()._get_data("temporal", split)
    df = df.copy()
    df["video_id"] = df["video_id"].astype(str)
    return df.reset_index(drop=True)


def split_hash(df) -> str:
    return sha256_text("\n".join(sorted(df["video_id"].tolist())))


def feat_path(cache: str, vid: str) -> Path:
    return CACHE_ROOT / cache / "features" / f"feat_{vid}.pt"


def load_feat(cache: str, vid: str):
    import torch
    p = feat_path(cache, vid)
    if not p.exists():
        return None
    return torch.load(p, weights_only=True)


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        return json.load(f)


def load_jsonl_by_vid(path) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out[str(d["vid"])] = d
    return out


def claim_list(claims_data: dict) -> list:
    """All claims in the order used by 08_encode_features_v2.py (the encoder keeps only [:6])."""
    out = []
    for key in ("ocr_claims", "asr_claims", "vision_claims", "title_claims"):
        for c in (claims_data or {}).get(key) or []:
            if c.get("claim"):
                out.append(c)
    return out


def write_csv(path, rows: list):
    import csv
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
