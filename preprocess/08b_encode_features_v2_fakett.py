"""
Step 08b: Feature Encoding V2 — FakeTT with Filtered Evidence & Reasoning V2

Differences from 08_encode_features_v2.py (FakeSV):
  - h_v: encoded from frames_16/ using openai CLIP ViT-L/14 (768-dim, not 1024)
  - h_c / h_e: openai CLIP text encoder (768-dim)
  - No split-target filter — process all 1992 FakeTT videos
  - Output: data/cache/FakeTT_FilteredEv/features/feat_{vid}.pt

Output schema:
  h_v          [768]          — OpenAI CLIP vision (mean over 16 frames)
  h_c          [768]          — mean-pooled claim text
  h_e_pos      [768]          — support evidence (mean-pooled)
  h_e_neg      [768]          — refute  evidence (mean-pooled)
  h_r          [768]          — SBERT reasoning v2 encoding
  h_c_seq      [N_c, 768]     — per-claim encodings
  h_e_pos_seq  [K, 768]       — per-support-evidence encodings
  h_e_neg_seq  [K, 768]       — per-refute-evidence encodings
  graph_adj    [N_c, N_c]     — claim graph adjacency

Usage:
    python preprocess/08b_encode_features_v2_fakett.py
    python preprocess/08b_encode_features_v2_fakett.py --limit 10 --device cuda
"""

import argparse, json, os, sys
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_ROOT     = os.path.join(PROJECT_ROOT, "data", "cache")
SBERT_MODEL    = "sentence-transformers/all-mpnet-base-v2"

FRAMES_ROOT    = r"F:\Bishe_workspace\FakeTT\frames_16"
CLAIMS_DIR     = os.path.join(CACHE_ROOT, "FakeTT", "claims")
GRAPH_DIR      = os.path.join(CACHE_ROOT, "FakeTT", "claim_graphs")
REASONING_DIR  = os.path.join(CACHE_ROOT, "FakeTT", "reasoning_v2")
VERIFIED_EV_PATH = os.path.join(PROJECT_ROOT, "preprocess", "output", "verified_evidence_FakeTT_v3.jsonl")

DST_CACHE      = os.path.join(CACHE_ROOT, "FakeTT_FilteredEv_v3")
FEAT_DIR       = os.path.join(DST_CACHE, "features")

EV_K = 5   # evidence slots per video

# ── Model loaders ─────────────────────────────────────────────────────────────

def load_openai_clip(device: str):
    import clip
    model, preprocess = clip.load("ViT-L/14", device=device)
    model = model.float().eval()
    return model, preprocess

def load_sbert(device: str):
    return SentenceTransformer(SBERT_MODEL, device=device)

# ── Visual features ───────────────────────────────────────────────────────────

@torch.no_grad()
def get_h_v(vid: str, clip_model, preprocess, device: str) -> torch.Tensor:
    from PIL import Image
    vid_dir = os.path.join(FRAMES_ROOT, vid)
    frame_paths = sorted(glob(os.path.join(vid_dir, "frame_*.jpg")))
    if not frame_paths:
        return torch.zeros(768)
    feats = []
    for fp in frame_paths:
        try:
            img = preprocess(Image.open(fp).convert("RGB")).unsqueeze(0).to(device)
            feat = clip_model.encode_image(img).float()
            feat = F.normalize(feat, dim=-1)
            feats.append(feat.cpu())
        except Exception:
            pass
    if not feats:
        return torch.zeros(768)
    return torch.stack(feats).mean(dim=0).squeeze(0)   # [768]

# ── Text features (openai CLIP) ───────────────────────────────────────────────

@torch.no_grad()
def encode_texts_mean(texts: list, clip_model, device: str) -> torch.Tensor:
    import clip as openai_clip
    if not texts:
        return torch.zeros(768)
    tokens = openai_clip.tokenize(texts, truncate=True).to(device)
    features = clip_model.encode_text(tokens).float()
    features = F.normalize(features, dim=-1)
    return features.cpu().mean(dim=0)   # [768]

@torch.no_grad()
def encode_texts_seq(texts: list, clip_model, device: str, fixed_k: int | None = None) -> torch.Tensor:
    import clip as openai_clip
    if not texts:
        n = fixed_k if fixed_k else 1
        return torch.zeros(n, 768)
    tokens = openai_clip.tokenize(texts, truncate=True).to(device)
    features = clip_model.encode_text(tokens).float()
    features = F.normalize(features, dim=-1).cpu()
    if fixed_k is not None:
        if features.size(0) < fixed_k:
            pad = torch.zeros(fixed_k - features.size(0), 768)
            features = torch.cat([features, pad], dim=0)
        else:
            features = features[:fixed_k]
    return features   # [fixed_k, 768] or [N, 768]

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
        txt = kev.get("evidence", "") if isinstance(kev, dict) else str(kev)
        if txt:
            parts.append(txt)
    for conf in reasoning_data.get("cross_modal_conflicts", [])[:2]:
        if isinstance(conf, str) and conf:
            parts.append(conf)
    if not parts:
        return torch.zeros(768)
    text = " ".join(parts)[:512]
    emb = sbert.encode([text], convert_to_tensor=True, normalize_embeddings=True)
    return emb.cpu().squeeze(0)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8", errors="replace") as f:
        return json.load(f)

def extract_claim_texts(claims_data: dict) -> list:
    out = []
    for key in ("ocr_claims", "asr_claims", "vision_claims", "title_claims"):
        for c in (claims_data.get(key) or []):
            if c.get("claim"):
                out.append(c["claim"])
    return out[:6]

def extract_claim_ids(claims_data: dict) -> list:
    out = []
    for key in ("ocr_claims", "asr_claims", "vision_claims", "title_claims"):
        for c in (claims_data.get(key) or []):
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

def load_filtered_evidence(path: str) -> dict:
    lookup = {}
    if not os.path.exists(path):
        return lookup
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            lookup[str(d["vid"])] = d.get("verified_evidence", [])
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
    parser.add_argument("--limit",        type=int, default=0)
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--regen",        action="store_true", help="Re-encode existing feat files (h_r only)")
    parser.add_argument("--regen-ev",     action="store_true",
                        help="Re-encode only evidence features (h_e_*) from updated verified_evidence; "
                             "reuses existing h_v/h_c/h_r/graph_adj to save time")
    parser.add_argument("--reasoning-dir", type=str, default=None,
                        help="Override reasoning cache directory (default: data/cache/FakeTT/reasoning_v2)")
    parser.add_argument("--output-dir",   type=str, default=None,
                        help="Override output feature directory (default: FakeTT_FilteredEv_v3)")
    args = parser.parse_args()

    # Allow overriding reasoning dir
    reasoning_dir = args.reasoning_dir if args.reasoning_dir else REASONING_DIR

    feat_dir = args.output_dir if args.output_dir else FEAT_DIR
    os.makedirs(feat_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  OutputDir: {feat_dir}")

    # Build todo list (all 1992 FakeTT videos)
    all_vids = {os.path.splitext(f)[0] for f in os.listdir(CLAIMS_DIR) if f.endswith(".json")}
    done_vids = {f[5:-3] for f in os.listdir(feat_dir) if f.startswith("feat_")}
    regen_all = args.regen or args.regen_ev
    todo = sorted(all_vids - (done_vids if not regen_all else set()))
    if args.limit:
        todo = todo[:args.limit]
    mode = "regen-ev" if args.regen_ev else ("regen" if args.regen else "full")
    print(f"Mode: {mode}  Total: {len(all_vids)}  Done: {len(done_vids)}  To process: {len(todo)}")

    if not todo:
        print("All done.")
        return

    # Load models
    print("Loading SBERT...")
    sbert = load_sbert(device)

    clip_model = clip_preprocess = None
    if not args.regen_ev:
        # regen-ev only needs CLIP for evidence text encoding, not visual
        print("Loading OpenAI CLIP ViT-L/14...")
        clip_model, clip_preprocess = load_openai_clip(device)
    else:
        # regen-ev: still needs CLIP for text encoding of evidence
        print("Loading OpenAI CLIP ViT-L/14 (text encoder only)...")
        clip_model, _ = load_openai_clip(device)

    # Load filtered evidence
    print(f"Loading filtered evidence from {VERIFIED_EV_PATH}...")
    ev_lookup = load_filtered_evidence(VERIFIED_EV_PATH)
    print(f"  Loaded evidence for {len(ev_lookup)} videos.")

    null_h = torch.zeros(768)
    no_ev = 0

    for vid in tqdm(todo, desc=f"Encoding [FakeTT {mode}]"):
        feat_path = os.path.join(feat_dir, f"feat_{vid}.pt")

        if args.regen_ev:
            # Load existing feat, only update evidence-related fields
            if os.path.exists(feat_path):
                feat = torch.load(feat_path, weights_only=True)
            else:
                feat = {}
        elif args.regen:
            feat = torch.load(feat_path, weights_only=True) if os.path.exists(feat_path) else {}
        else:
            feat = {}

        claims_data = load_json(os.path.join(CLAIMS_DIR, f"{vid}.json"))
        graph_data  = load_json(os.path.join(GRAPH_DIR,  f"{vid}.json"))
        claim_texts = extract_claim_texts(claims_data)

        # Visual and claim features (skip in regen-ev and regen modes)
        if not args.regen and not args.regen_ev:
            feat["h_v"] = get_h_v(vid, clip_model, clip_preprocess, device).cpu()
            feat["h_c"] = encode_texts_mean(claim_texts, clip_model, device) if claim_texts else null_h.clone()

        # Evidence features (always update in regen-ev; skip in regen mode)
        ev_list   = ev_lookup.get(vid, [])
        if not ev_list:
            no_ev += 1
        pos_texts = get_ev_texts(ev_list, "support")
        neg_texts = get_ev_texts(ev_list, "refute")

        if not args.regen:
            feat["h_e_pos"] = encode_texts_mean(pos_texts, clip_model, device) if pos_texts else null_h.clone()
            feat["h_e_neg"] = encode_texts_mean(neg_texts, clip_model, device) if neg_texts else null_h.clone()

        # Reasoning features (always update in regen; skip in regen-ev unless h_r missing)
        if args.regen or not args.regen_ev:
            reasoning_data = load_json(os.path.join(reasoning_dir, f"{vid}.json"))
            feat["h_r"] = encode_reasoning_v2(sbert, reasoning_data).cpu()

        # Sequence features (skip in regen mode; update in full and regen-ev)
        if not args.regen:
            feat["h_c_seq"]     = encode_texts_seq(claim_texts, clip_model, device).cpu() if claim_texts else torch.zeros(1, 768)
            feat["h_e_pos_seq"] = encode_texts_seq(pos_texts,   clip_model, device, fixed_k=EV_K).cpu()
            feat["h_e_neg_seq"] = encode_texts_seq(neg_texts,   clip_model, device, fixed_k=EV_K).cpu()
        if not args.regen and not args.regen_ev:
            feat["graph_adj"]   = build_graph_adj(claims_data, graph_data).cpu()

        torch.save(feat, feat_path)

    print(f"\nDone. NoEvidence={no_ev}/{len(todo)}")
    print(f"Features saved to: {feat_dir}")


if __name__ == "__main__":
    main()
