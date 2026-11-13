"""
Step 08d: Feature Encoding — WildFake (Chinese subset) with Filtered Evidence & Reasoning V2

Differences from 08_encode_features_v2.py (FakeSV):
  - h_v: extracted on-the-fly from frames_16/ using Chinese-CLIP ViT-L/14 (1024-dim)
  - No vit_tensor.pt pre-cache needed
  - Target VIDs loaded from data_zh.jsonl (1826 Chinese videos)
  - Output: data/cache/WildFake_FilteredEv/features/feat_{vid}.pt

Output schema:
  h_v          [1024]         — Chinese-CLIP vision (mean over 16 frames)
  h_c          [768]          — mean-pooled claim text (Chinese-CLIP text)
  h_e_pos      [768]          — support evidence (mean-pooled)
  h_e_neg      [768]          — refute evidence (mean-pooled)
  h_r          [768]          — SBERT reasoning v2 encoding
  h_c_seq      [N_c, 768]     — per-claim encodings
  h_e_pos_seq  [K, 768]       — per-support-evidence encodings
  h_e_neg_seq  [K, 768]       — per-refute-evidence encodings
  graph_adj    [N_c, N_c]     — claim graph adjacency

Usage:
    python preprocess/08d_encode_features_wildfake.py
    python preprocess/08d_encode_features_wildfake.py --limit 20 --device cuda
"""

import argparse
import json
import os
import sys
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_ROOT      = os.path.join(PROJECT_ROOT, "data", "cache")

CLIP_MODEL_PATH = "OFA-Sys/chinese-clip-vit-large-patch14"   # HF cache
SBERT_MODEL     = "sentence-transformers/all-mpnet-base-v2"

FRAMES_ROOT     = r"F:\Bishe_workspace\WildFake\frames_16"
DATA_ZH_PATH    = r"F:\Bishe_workspace\WildFake\data_zh.jsonl"

CLAIMS_DIR      = os.path.join(CACHE_ROOT, "WildFake", "claims")
GRAPH_DIR       = os.path.join(CACHE_ROOT, "WildFake", "claim_graphs")
REASONING_DIR   = os.path.join(CACHE_ROOT, "WildFake", "reasoning_v2")
VERIFIED_EV_PATH = os.path.join(PROJECT_ROOT, "preprocess", "output", "verified_evidence_WildFake.jsonl")

DST_CACHE       = os.path.join(CACHE_ROOT, "WildFake_FilteredEv")
FEAT_DIR        = os.path.join(DST_CACHE, "features")

EV_K = 5   # evidence slots per video

# ── Model loaders ─────────────────────────────────────────────────────────────

def load_chinese_clip(model_path: str, device: str):
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor
    processor = ChineseCLIPProcessor.from_pretrained(model_path)
    model = ChineseCLIPModel.from_pretrained(model_path).to(device).eval()
    return model, processor


def load_sbert(device: str):
    return SentenceTransformer(SBERT_MODEL, device=device)

# ── Visual features ───────────────────────────────────────────────────────────

@torch.no_grad()
def get_h_v(vid: str, clip_model, clip_proc, device: str) -> torch.Tensor:
    from PIL import Image
    vid_dir = os.path.join(FRAMES_ROOT, vid)
    frame_paths = sorted(glob(os.path.join(vid_dir, "frame_*.jpg")))
    if not frame_paths:
        return torch.zeros(1024)
    feats = []
    for fp in frame_paths:
        try:
            img = Image.open(fp).convert("RGB")
            inputs = clip_proc(images=img, return_tensors="pt").to(device)
            feat = clip_model.get_image_features(**inputs).float()
            feat = F.normalize(feat, dim=-1)
            feats.append(feat.cpu().squeeze(0))
        except Exception:
            pass
    if not feats:
        return torch.zeros(1024)
    return torch.stack(feats).mean(dim=0)   # [1024]

# ── Text features (Chinese-CLIP) ──────────────────────────────────────────────

@torch.no_grad()
def encode_texts_mean(texts: list, clip_model, clip_proc, device: str) -> torch.Tensor:
    if not texts:
        return torch.zeros(768)
    inputs = clip_proc(text=texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=52).to(device)
    raw = clip_model.get_text_features(**inputs)
    if torch.is_tensor(raw):
        features = raw.float()
    elif hasattr(raw, "pooler_output"):
        features = raw.pooler_output.float()
    else:
        features = raw.last_hidden_state[:, 0].float()
    features = F.normalize(features, dim=-1)
    return features.cpu().mean(dim=0)   # [768]


@torch.no_grad()
def encode_texts_seq(texts: list, clip_model, clip_proc, device: str,
                     fixed_k: int | None = None) -> torch.Tensor:
    if not texts:
        n = fixed_k if fixed_k else 1
        return torch.zeros(n, 768)
    inputs = clip_proc(text=texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=52).to(device)
    raw = clip_model.get_text_features(**inputs)
    if torch.is_tensor(raw):
        features = raw.float()
    elif hasattr(raw, "pooler_output"):
        features = raw.pooler_output.float()
    else:
        features = raw.last_hidden_state[:, 0].float()
    features = F.normalize(features, dim=-1).cpu()
    if fixed_k is not None:
        if features.size(0) < fixed_k:
            pad = torch.zeros(fixed_k - features.size(0), 768)
            features = torch.cat([features, pad], dim=0)
        else:
            features = features[:fixed_k]
    return features   # [fixed_k, 768]

# ── Reasoning (SBERT) ─────────────────────────────────────────────────────────

@torch.no_grad()
def encode_reasoning_v2(sbert, reasoning_data: dict) -> torch.Tensor:
    parts = []
    assessment = reasoning_data.get("overall_assessment", "")
    if assessment:
        parts.append(f"Assessment: {assessment}")
    if reasoning_data.get("reason"):
        parts.append(reasoning_data["reason"])
    for kev in reasoning_data.get("key_evidence_used", [])[:2]:
        txt = kev.get("evidence") or kev.get("claim_id", "") if isinstance(kev, dict) else str(kev)
        if txt:
            parts.append(txt)
    for conf in reasoning_data.get("cross_modal_conflicts", [])[:2]:
        if isinstance(conf, str) and conf:
            parts.append(conf)
    if not parts:
        return torch.zeros(768)
    text = " ".join(parts)[:512]
    emb = sbert.encode([text], convert_to_tensor=True, normalize_embeddings=True)
    return emb.cpu().squeeze(0)   # [768]

# ── Claim / graph helpers ─────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8", errors="replace") as f:
        try:
            return json.load(f)
        except Exception:
            return {}


def extract_claim_texts(claims_data: dict) -> list:
    out = []
    for key in ("ocr_claims", "asr_claims", "vision_claims", "title_claims"):
        for c in claims_data.get(key) or []:
            if c.get("claim"):
                out.append(c["claim"])
    return out[:6]


def extract_claim_ids(claims_data: dict) -> list:
    out = []
    for key in ("ocr_claims", "asr_claims", "vision_claims", "title_claims"):
        for c in claims_data.get(key) or []:
            if c.get("claim"):
                out.append(c.get("id", ""))
    return out[:6]


def build_graph_adj(claims_data: dict, graph_data: dict) -> torch.Tensor:
    claim_ids = extract_claim_ids(claims_data)
    n = len(claim_ids)
    if n == 0:
        return torch.zeros(1, 1)
    id2idx = {cid: i for i, cid in enumerate(claim_ids)}
    adj = torch.zeros(n, n)
    for edge in graph_data.get("edges", []):
        src = edge.get("source_id", "")
        tgt = edge.get("target_id", "")
        score = float(edge.get("conflict_score", 0.0))
        if src in id2idx and tgt in id2idx:
            i, j = id2idx[src], id2idx[tgt]
            adj[i, j] = score
            adj[j, i] = score
    return adj + torch.eye(n)

# ── Evidence loader ────────────────────────────────────────────────────────────

def load_filtered_evidence(path: str) -> dict:
    lookup = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            vid = d["vid"]
            lookup[vid] = d.get("verified_evidence", [])
    return lookup


def get_ev_texts(ev_list: list, relation: str, top_k: int = EV_K) -> list:
    out = []
    for ev in ev_list:
        if ev.get("relation") == relation:
            text = ev.get("summary") or ev.get("snippet") or ev.get("title") or ""
            if text:
                out.append(text)
        if len(out) >= top_k:
            break
    return out

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",  type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--regen",  action="store_true")
    args = parser.parse_args()

    os.makedirs(FEAT_DIR, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load target VID list from data_zh.jsonl
    with open(DATA_ZH_PATH, encoding="utf-8") as f:
        target_vids = {json.loads(l)["video_id"] for l in f if l.strip()}
    print(f"Target vids (Chinese subset): {len(target_vids)}")

    done_vids = {f[5:-3] for f in os.listdir(FEAT_DIR) if f.startswith("feat_")}
    todo = sorted(target_vids - (done_vids if not args.regen else set()))
    if args.limit:
        todo = todo[:args.limit]
    print(f"Already encoded : {len(done_vids)}")
    print(f"To process      : {len(todo)}")

    if not todo:
        print("All done.")
        return

    print("Loading SBERT...")
    sbert = load_sbert(device)

    print(f"Loading Chinese-CLIP from {CLIP_MODEL_PATH}...")
    clip_model, clip_proc = load_chinese_clip(CLIP_MODEL_PATH, device)

    print(f"Loading filtered evidence from {VERIFIED_EV_PATH}...")
    ev_lookup = load_filtered_evidence(VERIFIED_EV_PATH)
    print(f"  Loaded evidence for {len(ev_lookup)} videos.")

    null_h_v = torch.zeros(1024)
    null_hce = torch.zeros(768)
    no_ev    = 0

    for vid in tqdm(todo, desc="Encoding [WildFake_FilteredEv]"):
        feat_path = os.path.join(FEAT_DIR, f"feat_{vid}.pt")

        if args.regen and os.path.exists(feat_path):
            feat = torch.load(feat_path, weights_only=True)
        else:
            feat = {}

        # Visual features (from frames)
        if not args.regen:
            feat["h_v"] = get_h_v(vid, clip_model, clip_proc, device)

        # Claim features
        claims_data = load_json(os.path.join(CLAIMS_DIR, f"{vid}.json"))
        graph_data  = load_json(os.path.join(GRAPH_DIR,  f"{vid}.json"))
        claim_texts = extract_claim_texts(claims_data)

        if not args.regen:
            feat["h_c"] = encode_texts_mean(claim_texts, clip_model, clip_proc, device) if claim_texts else null_hce

        # Evidence features (filtered)
        ev_list   = ev_lookup.get(vid, [])
        if not ev_list:
            no_ev += 1
        pos_texts = get_ev_texts(ev_list, "support", top_k=EV_K)
        neg_texts = get_ev_texts(ev_list, "refute",  top_k=EV_K)

        if not args.regen:
            feat["h_e_pos"] = encode_texts_mean(pos_texts, clip_model, clip_proc, device) if pos_texts else null_hce
            feat["h_e_neg"] = encode_texts_mean(neg_texts, clip_model, clip_proc, device) if neg_texts else null_hce

        # Reasoning features (SBERT)
        reasoning_data = load_json(os.path.join(REASONING_DIR, f"{vid}.json"))
        feat["h_r"] = encode_reasoning_v2(sbert, reasoning_data).cpu()

        # Sequence features (for GAT + CECA)
        feat["h_c_seq"]     = encode_texts_seq(claim_texts, clip_model, clip_proc, device).cpu() if claim_texts else torch.zeros(1, 768)
        feat["h_e_pos_seq"] = encode_texts_seq(pos_texts, clip_model, clip_proc, device, fixed_k=EV_K).cpu()
        feat["h_e_neg_seq"] = encode_texts_seq(neg_texts, clip_model, clip_proc, device, fixed_k=EV_K).cpu()
        feat["graph_adj"]   = build_graph_adj(claims_data, graph_data).cpu()

        torch.save(feat, feat_path)

    print(f"\nDone. NoEvidence={no_ev}/{len(todo)}")
    print(f"Features saved to: {FEAT_DIR}")


if __name__ == "__main__":
    main()
