"""
Step 8: Feature Encoding V2 — with Filtered Evidence & Reasoning V2

Reads:
  - verified_evidence_FakeSV_v2.jsonl  (Event-aware filtered evidence)
  - data/cache/FakeSV/reasoning_v2/    (Reasoning V2 structured output)
  - data/cache/FakeSV/claims/          (Claim texts, same as before)
  - data/cache/FakeSV/claim_graphs/    (Claim graphs, same as before)
  - vit_tensor.pt                      (Pre-extracted visual features)

Writes:
  data/cache/FakeSV_FilteredEv/features/feat_{vid}.pt
  {
    "h_v":          tensor([1024])     — visual (Chinese-CLIP ViT-L/14, unchanged)
    "h_c":          tensor([768])      — claim text (mean-pooled, unchanged)
    "h_e_pos":      tensor([768])      — support evidence (mean-pooled, NEW source)
    "h_e_neg":      tensor([768])      — refute  evidence (mean-pooled, NEW source)
    "h_r":          tensor([768])      — reasoning V2 (SBERT, enhanced encoding)
    "h_c_seq":      tensor([N_c, 768]) — per-claim encodings (unchanged)
    "h_e_pos_seq":  tensor([K, 768])   — per-support-evidence encodings (NEW)
    "h_e_neg_seq":  tensor([K, 768])   — per-refute-evidence encodings (NEW)
    "graph_adj":    tensor([N_c, N_c]) — claim graph (unchanged)
  }

Usage:
    python preprocess/08_encode_features_v2.py [--limit N] [--device cuda]
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CACHE_ROOT     = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
VIT_TENSOR_PATH = r"F:\Bishe_workspace\fakesv-qwen3\vit_tensor.pt"
CLIP_MODEL_PATH = r"F:\Bishe_workspace\ExMRD-master\preprocess\model\chinese-clip-vit-large-patch14"
SBERT_MODEL     = "sentence-transformers/all-mpnet-base-v2"

VERIFIED_EV_PATH = os.path.join(
    os.path.dirname(__file__), "output", "verified_evidence_FakeSV_v2.jsonl"
)

# Only encode split-target videos (train+val+test temporal split, excl. 辟谣)
SPLIT_TARGET_PATH = os.path.join(CACHE_ROOT, "FakeSV", "Split_target.txt")

# Input dirs (same dataset, different sources)
SRC_CACHE    = os.path.join(CACHE_ROOT, "FakeSV")
CLAIMS_DIR   = os.path.join(SRC_CACHE, "claims")
REASONING_DIR = os.path.join(SRC_CACHE, "reasoning_v2")
GRAPH_DIR    = os.path.join(SRC_CACHE, "claim_graphs")

# Output dir (new cache directory)
DST_CACHE = os.path.join(CACHE_ROOT, "FakeSV_FilteredEv")
FEAT_DIR  = os.path.join(DST_CACHE, "features")

EV_K = 5   # fixed number of evidence slots per video

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

def load_chinese_clip(model_path: str, device: str):
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor
    processor = ChineseCLIPProcessor.from_pretrained(model_path)
    model = ChineseCLIPModel.from_pretrained(model_path).to(device).eval()
    return model, processor


def load_sbert(device: str):
    return SentenceTransformer(SBERT_MODEL, device=device)


# ---------------------------------------------------------------------------
# Filtered evidence loader
# ---------------------------------------------------------------------------

def load_filtered_evidence(path: str) -> dict:
    """Load verified_evidence JSONL → {vid: [ev_items]}."""
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


def get_ev_texts(ev_list: list, relation: str, top_k: int = EV_K) -> list[str]:
    """Extract text from evidence items filtered by relation (support/refute/neutral)."""
    out = []
    for ev in ev_list:
        if ev.get("relation") == relation:
            text = ev.get("summary") or ev.get("snippet") or ev.get("title") or ""
            if text:
                out.append(text)
        if len(out) >= top_k:
            break
    return out


# ---------------------------------------------------------------------------
# Text encoding helpers (same as 07_encode_features.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_texts_mean(texts: list[str], model, processor, device: str) -> torch.Tensor:
    """Mean-pool Chinese-CLIP text features → [768]."""
    if not texts:
        return torch.zeros(768)
    inputs = processor(text=texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=52).to(device)
    raw = model.get_text_features(**inputs)
    if torch.is_tensor(raw):
        features = raw.float()
    elif hasattr(raw, "pooler_output"):
        features = raw.pooler_output.float()
    else:
        features = raw.last_hidden_state[:, 0].float()
    features = F.normalize(features, dim=-1)
    return features.cpu().mean(dim=0)   # [768]


@torch.no_grad()
def encode_texts_seq(texts: list[str], model, processor, device: str,
                     fixed_k: int | None = None) -> torch.Tensor:
    """Per-item Chinese-CLIP text features → [N, 768], optionally padded to fixed_k."""
    if not texts:
        n = fixed_k if fixed_k else 1
        return torch.zeros(n, 768)
    inputs = processor(text=texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=52).to(device)
    raw = model.get_text_features(**inputs)
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


@torch.no_grad()
def encode_reasoning_v2(sbert, reasoning_data: dict) -> torch.Tensor:
    """Encode reasoning V2 output with SBERT → [768].

    Includes richer context than V1:
      - reason (main analysis text)
      - key_evidence_used (evidence summaries used)
      - cross_modal_conflicts (detected conflicts)
      - overall_assessment signal prefix
    """
    parts = []

    # Overall assessment as a prefix signal
    assessment = reasoning_data.get("overall_assessment", "")
    if assessment:
        parts.append(f"Assessment: {assessment}")

    # Main reasoning text
    if reasoning_data.get("reason"):
        parts.append(reasoning_data["reason"])

    # Key evidence used (up to 2)
    for kev in reasoning_data.get("key_evidence_used", [])[:2]:
        if isinstance(kev, dict):
            txt = kev.get("evidence") or kev.get("claim_id", "")
        else:
            txt = str(kev)
        if txt:
            parts.append(txt)

    # Cross-modal conflicts (up to 2)
    for conf in reasoning_data.get("cross_modal_conflicts", [])[:2]:
        if isinstance(conf, str) and conf:
            parts.append(conf)

    if not parts:
        return torch.zeros(768)

    text = " ".join(parts)[:512]
    emb = sbert.encode([text], convert_to_tensor=True, normalize_embeddings=True)
    return emb.cpu().squeeze(0)   # [768]


# ---------------------------------------------------------------------------
# Claim & graph helpers (copied from 07_encode_features.py)
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8", errors="replace") as f:
        return json.load(f)


def extract_claim_texts(claims_data: dict) -> list[str]:
    out = []
    for key in ("ocr_claims", "asr_claims", "vision_claims", "title_claims"):
        for c in claims_data.get(key) or []:
            if c.get("claim"):
                out.append(c["claim"])
    return out[:6]


def extract_claim_ids(claims_data: dict) -> list[str]:
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
    adj = adj + torch.eye(n)
    return adj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Encode features V2 (filtered evidence)")
    parser.add_argument("--limit",  type=int, default=0, help="0 = process all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--regen", action="store_true",
                        help="Regen existing feat files (update h_r/evidence only)")
    parser.add_argument("--regen-ev", action="store_true",
                        help="Re-encode only evidence features (h_e_*) from updated verified_evidence; "
                             "reuses existing h_v/h_c/h_r/graph_adj to save time")
    args = parser.parse_args()

    os.makedirs(FEAT_DIR, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Determine videos to process ----
    # Master list: split-target vids (3624 = train+val+test temporal, excl. 辟谣)
    if os.path.exists(SPLIT_TARGET_PATH):
        with open(SPLIT_TARGET_PATH, encoding="utf-8") as f:
            target_vids = set(f.read().strip().split())
        print(f"Split target vids: {len(target_vids)}")
    else:
        # Fallback: all claim vids
        target_vids = {os.path.splitext(f)[0] for f in os.listdir(CLAIMS_DIR) if f.endswith(".json")}
        print(f"No split target file found; using all claim vids: {len(target_vids)}")

    regen_all = args.regen or args.regen_ev
    done_vids = {f[5:-3] for f in os.listdir(FEAT_DIR) if f.startswith("feat_")}
    todo = sorted(target_vids - (done_vids if not regen_all else set()))

    if args.limit:
        todo = todo[:args.limit]
    mode = "regen-ev" if args.regen_ev else ("regen" if args.regen else "full")
    print(f"Mode: {mode}")
    print(f"Already encoded:  {len(done_vids)}")
    print(f"To process:       {len(todo)}")

    if not todo:
        print("All done.")
        return

    # ---- Load models ----
    print("Loading SBERT...")
    sbert = load_sbert(device)

    print(f"Loading Chinese-CLIP from {CLIP_MODEL_PATH}...")
    clip_model, clip_proc = load_chinese_clip(CLIP_MODEL_PATH, device)

    print(f"Loading vit_tensor.pt from {VIT_TENSOR_PATH}...")
    vit_cache = torch.load(VIT_TENSOR_PATH, weights_only=True)
    print(f"  Loaded {len(vit_cache)} entries.")

    # ---- Load filtered evidence ----
    print(f"Loading filtered evidence from {VERIFIED_EV_PATH}...")
    ev_lookup = load_filtered_evidence(VERIFIED_EV_PATH)
    print(f"  Loaded evidence for {len(ev_lookup)} videos.")

    null_h_v  = torch.zeros(1024)
    null_hce  = torch.zeros(768)
    skipped   = 0
    no_ev     = 0

    for vid in tqdm(todo, desc=f"Encoding [FakeSV_FilteredEv {mode}]"):
        feat_path = os.path.join(FEAT_DIR, f"feat_{vid}.pt")

        if regen_all and os.path.exists(feat_path):
            feat = torch.load(feat_path, weights_only=True)
        else:
            feat = {}

        # # ---- Visual features (unchanged) ----
        # if not args.regen and not args.regen_ev:
        #     vit_feat = vit_cache.get(vid)
        #     if vit_feat is not None:
        #         h_v = vit_feat.float().mean(dim=0)
        #         h_v = F.normalize(h_v, dim=-1)
        #     else:
        #         h_v = null_h_v
        #     feat["h_v"] = h_v.cpu()
        # ---- Visual features ----
        if not args.regen and not args.regen_ev:
            vit_feat = vit_cache.get(vid)

            if vit_feat is not None:
                vit_feat = vit_feat.float()

                # 保留逐帧视觉特征，供训练阶段 Vision Self-Attention 使用
                # Expected:
                #   [T, 1024]
                h_v_seq = F.normalize(
                    vit_feat,
                    dim=-1,
                )

                feat["h_v_seq"] = h_v_seq.cpu()

                # 保留原始 mean-pooled h_v，保证旧模型/旧实验兼容
                h_v = h_v_seq.mean(dim=0)
                h_v = F.normalize(
                    h_v,
                    dim=-1,
                )

            else:
                h_v_seq = torch.zeros(1, 1024)
                h_v = null_h_v

            feat["h_v_seq"] = h_v_seq.cpu()
            feat["h_v"] = h_v.cpu()
        # ---- Claim features (unchanged) ----
        claims_data = load_json(os.path.join(CLAIMS_DIR, f"{vid}.json"))
        graph_data  = load_json(os.path.join(GRAPH_DIR,  f"{vid}.json"))
        claim_texts = extract_claim_texts(claims_data)

        if not args.regen and not args.regen_ev:
            feat["h_c"] = encode_texts_mean(claim_texts, clip_model, clip_proc, device) if claim_texts else null_hce

        # ---- Evidence features (NEW: from filtered evidence) ----
        ev_list  = ev_lookup.get(vid, [])
        if not ev_list:
            no_ev += 1

        pos_texts = get_ev_texts(ev_list, "support", top_k=EV_K)
        neg_texts = get_ev_texts(ev_list, "refute",  top_k=EV_K)

        if not args.regen:
            feat["h_e_pos"] = encode_texts_mean(pos_texts, clip_model, clip_proc, device) if pos_texts else null_hce
            feat["h_e_neg"] = encode_texts_mean(neg_texts, clip_model, clip_proc, device) if neg_texts else null_hce

        # ---- Reasoning features (NEW: from reasoning_v2) ----
        if args.regen or not args.regen_ev:
            reasoning_data = load_json(os.path.join(REASONING_DIR, f"{vid}.json"))
            feat["h_r"] = encode_reasoning_v2(sbert, reasoning_data).cpu()

        # ---- Sequence features (for GAT + CECA) ----
        if not args.regen:
            if claim_texts:
                feat["h_c_seq"] = encode_texts_seq(claim_texts, clip_model, clip_proc, device).cpu()
            else:
                feat["h_c_seq"] = torch.zeros(1, 768)

            feat["h_e_pos_seq"] = encode_texts_seq(pos_texts, clip_model, clip_proc, device, fixed_k=EV_K).cpu()
            feat["h_e_neg_seq"] = encode_texts_seq(neg_texts, clip_model, clip_proc, device, fixed_k=EV_K).cpu()
        if not args.regen and not args.regen_ev:
            feat["graph_adj"] = build_graph_adj(claims_data, graph_data).cpu()

        torch.save(feat, feat_path)

    print(f"\nDone. Skipped={skipped}, NoEvidence={no_ev}/{len(todo)}")
    print(f"Features saved to: {FEAT_DIR}")


if __name__ == "__main__":
    main()
