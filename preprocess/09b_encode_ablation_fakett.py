"""
Step 09b: Ablation Feature Encoding — FakeTT

Mirror of 09_encode_ablation.py but for FakeTT:
  - OpenAI CLIP ViT-L/14 (768-dim) instead of Chinese-CLIP
  - Uses FakeTT paths (raw/scored evidence, event_summary_FakeTT.jsonl)
  - No Split_target.txt — processes all 1992 FakeTT videos
  - For woEGR / woCMC: fast path (copy h_v/h_c/h_e from FakeTT_FilteredEv, only re-encode h_r)

Variants:
  woCMR   — w/o Claim-centric Retrieval
              h_c: single global CLIP encoding of event_summary text
              h_e: old scored evidence (evidence/scored/top_semantic, not per-claim filtered)

  woEV    — w/o Evidence Verification
              h_c: per-claim CLIP (unchanged)
              h_e: raw retrieved evidence (evidence/raw/, no scoring/filtering)

  scoreEV — Score-based Evidence Selection (traditional scoring, no LLM verifier)
              h_c: per-claim CLIP (unchanged)
              h_e: old NLI+time scored evidence (top_semantic)

  woEGR   — w/o Evidence-Guided Reasoning
              h_c / h_e / h_v: identical to FakeTT_FilteredEv
              h_r: SBERT encoding of reasoning_v2_woEGR/ (evidence hidden from LLM)

  woCMC   — w/o Cross-Modal Conflict in Reasoning
              h_c / h_e / h_v: identical to FakeTT_FilteredEv
              h_r: SBERT encoding of reasoning_v2_woCMC/ (conflict analysis hidden)

Usage:
  python preprocess/09b_encode_ablation_fakett.py --variant woCMR
  python preprocess/09b_encode_ablation_fakett.py --variant woEV
  python preprocess/09b_encode_ablation_fakett.py --variant scoreEV
  python preprocess/09b_encode_ablation_fakett.py --variant woEGR
  python preprocess/09b_encode_ablation_fakett.py --variant woCMC
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_ROOT    = os.path.join(PROJECT_ROOT, "data", "cache")
SBERT_MODEL   = "sentence-transformers/all-mpnet-base-v2"
FRAMES_ROOT   = r"F:\Bishe_workspace\FakeTT\frames_16"

SRC_CACHE     = os.path.join(CACHE_ROOT, "FakeTT")
CLAIMS_DIR    = os.path.join(SRC_CACHE, "claims")
SCORED_DIR    = os.path.join(SRC_CACHE, "evidence", "scored")
RAW_DIR       = os.path.join(SRC_CACHE, "evidence", "raw")
REASONING_DIR = os.path.join(SRC_CACHE, "reasoning_v2")
GRAPH_DIR     = os.path.join(SRC_CACHE, "claim_graphs")

EVENT_SUMMARY_PATH = os.path.join(
    PROJECT_ROOT, "preprocess", "output", "event_summary_FakeTT.jsonl"
)
FILTERD_EV_CACHE = os.path.join(CACHE_ROOT, "FakeTT_FilteredEv_v3", "features")

EV_K = 5   # fixed number of evidence slots

VARIANT_CACHE = {
    "woCMR":   "FakeTT_woCMR",
    "woEV":    "FakeTT_woEV",
    "scoreEV": "FakeTT_scoreEV",
    "woEGR":   "FakeTT_woEGR",
    "woCMC":   "FakeTT_woCMC",
}

REASONING_DIRS = {
    "woEGR": os.path.join(SRC_CACHE, "reasoning_v2_woEGR"),
    "woCMC": os.path.join(SRC_CACHE, "reasoning_v2_woCMC"),
}

# ── Model loaders ─────────────────────────────────────────────────────────────

def load_openai_clip(device: str):
    import clip
    model, preprocess = clip.load("ViT-L/14", device=device)
    model = model.float().eval()
    return model, preprocess

def load_sbert(device: str):
    return SentenceTransformer(SBERT_MODEL, device=device)

# ── CLIP encoding helpers ──────────────────────────────────────────────────────

@torch.no_grad()
def encode_texts_mean(texts: list, clip_model, device: str) -> torch.Tensor:
    import clip as openai_clip
    if not texts:
        return torch.zeros(768)
    tokens = openai_clip.tokenize(texts, truncate=True).to(device)
    features = clip_model.encode_text(tokens).float()
    features = F.normalize(features, dim=-1)
    return features.cpu().mean(dim=0)

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
    return features

# ── SBERT reasoning encoding ───────────────────────────────────────────────────

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

# ── JSON helpers ───────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read().strip()
        if not content:
            return {}
        return json.loads(content)
    except Exception:
        return {}

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

def load_event_summaries() -> dict:
    lookup = {}
    if not os.path.exists(EVENT_SUMMARY_PATH):
        return lookup
    with open(EVENT_SUMMARY_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                lookup[str(obj["vid"])] = obj.get("event_summary", "")
            except Exception:
                pass
    return lookup

def get_raw_ev_texts(vid: str, top_k: int = EV_K) -> list:
    """Get text from raw (unscored) evidence."""
    path = os.path.join(RAW_DIR, f"{vid}.json")
    if not os.path.exists(path):
        return []
    data = load_json(path)
    ev_list = data.get("evidence", [])
    texts = []
    for ev in ev_list:
        text = ev.get("summary") or ev.get("snippet") or ev.get("title") or ""
        if text:
            texts.append(text)
        if len(texts) >= top_k:
            break
    return texts

def get_scored_ev_texts(vid: str, top_k: int = EV_K) -> list:
    """Get text from NLI+time scored evidence (top_semantic)."""
    path = os.path.join(SCORED_DIR, f"{vid}.json")
    if not os.path.exists(path):
        return []
    data = load_json(path)
    items = data.get("top_semantic", [])
    texts = []
    for ev in items:
        text = ev.get("summary") or ev.get("snippet") or ev.get("title") or ""
        if text:
            texts.append(text)
        if len(texts) >= top_k:
            break
    return texts

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                        choices=["woCMR", "woEV", "scoreEV", "woEGR", "woCMC"])
    parser.add_argument("--limit",  type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--regen",  action="store_true", help="Re-encode all (ignore done)")
    args = parser.parse_args()

    variant = args.variant
    dst_cache_name = VARIANT_CACHE[variant]
    dst_feat_dir = os.path.join(CACHE_ROOT, dst_cache_name, "features")
    os.makedirs(dst_feat_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Variant: {variant}  Device: {device}  Dst: {dst_cache_name}")

    # Build todo list (all FakeTT videos)
    all_vids = sorted(
        os.path.splitext(f)[0] for f in os.listdir(CLAIMS_DIR) if f.endswith(".json")
    )
    done_vids = {f[5:-3] for f in os.listdir(dst_feat_dir) if f.startswith("feat_")}
    todo = [v for v in all_vids if v not in done_vids] if not args.regen else all_vids
    if args.limit:
        todo = todo[:args.limit]
    print(f"Total: {len(all_vids)}  Done: {len(done_vids)}  To process: {len(todo)}")

    if not todo:
        print("All done.")
        return

    # ── Fast path for woEGR / woCMC ─────────────────────────────────────────
    if variant in ("woEGR", "woCMC"):
        ablation_reasoning_dir = REASONING_DIRS[variant]
        if not os.path.exists(ablation_reasoning_dir):
            print(f"[ERROR] Reasoning dir missing: {ablation_reasoning_dir}")
            sys.exit(1)

        print("Loading SBERT (for h_r only)...")
        sbert = load_sbert(device)
        null_h = torch.zeros(768)

        for vid in tqdm(todo, desc=f"Encoding [{dst_cache_name}] (h_r only)"):
            src_feat_path = os.path.join(FILTERD_EV_CACHE, f"feat_{vid}.pt")
            dst_feat_path = os.path.join(dst_feat_dir, f"feat_{vid}.pt")

            # Load source features (copy h_v, h_c, h_e_*, h_c_seq, h_e_*_seq, graph_adj)
            if os.path.exists(src_feat_path):
                feat = torch.load(src_feat_path, weights_only=True)
            else:
                feat = {}

            # Re-encode h_r from ablation reasoning dir
            reasoning_data = load_json(os.path.join(ablation_reasoning_dir, f"{vid}.json"))
            feat["h_r"] = encode_reasoning_v2(sbert, reasoning_data).cpu()

            torch.save(feat, dst_feat_path)

        print(f"Done. Features saved to: {dst_feat_dir}")
        return

    # ── Full encoding path for woCMR / woEV / scoreEV ────────────────────────

    print("Loading SBERT...")
    sbert = load_sbert(device)
    print("Loading OpenAI CLIP ViT-L/14...")
    clip_model, _ = load_openai_clip(device)

    # Load event summaries (needed for woCMR)
    event_summaries: dict = {}
    if variant == "woCMR":
        print("Loading event summaries...")
        event_summaries = load_event_summaries()
        print(f"  Loaded {len(event_summaries)} event summaries.")

    null_h = torch.zeros(768)

    for vid in tqdm(todo, desc=f"Encoding [{dst_cache_name}]"):
        # Always start from FakeTT_FilteredEv features (copy base; we'll overwrite some)
        src_feat_path = os.path.join(FILTERD_EV_CACHE, f"feat_{vid}.pt")
        feat = torch.load(src_feat_path, weights_only=True) if os.path.exists(src_feat_path) else {}

        claims_data = load_json(os.path.join(CLAIMS_DIR, f"{vid}.json"))
        graph_data  = load_json(os.path.join(GRAPH_DIR,  f"{vid}.json"))

        if variant == "woCMR":
            # h_c: single global event_summary encoding (not per-claim)
            summary_text = event_summaries.get(vid, "")
            feat["h_c"] = encode_texts_mean([summary_text], clip_model, device) if summary_text else null_h.clone()
            # h_c_seq: repeat global h_c for each claim slot
            claim_texts = extract_claim_texts(claims_data)
            n_claims = max(1, len(claim_texts))
            feat["h_c_seq"] = feat["h_c"].unsqueeze(0).expand(n_claims, -1).clone()

            # h_e: scored evidence (not per-claim)
            ev_texts = get_scored_ev_texts(vid)
            feat["h_e_pos"] = encode_texts_mean(ev_texts, clip_model, device) if ev_texts else null_h.clone()
            feat["h_e_neg"] = null_h.clone()
            feat["h_e_pos_seq"] = encode_texts_seq(ev_texts, clip_model, device, fixed_k=EV_K)
            feat["h_e_neg_seq"] = torch.zeros(EV_K, 768)

        elif variant in ("woEV", "scoreEV"):
            # h_c: per-claim CLIP (same as FilteredEv — already copied from src_feat_path)
            if variant == "woEV":
                ev_texts = get_raw_ev_texts(vid)
            else:
                ev_texts = get_scored_ev_texts(vid)

            feat["h_e_pos"] = encode_texts_mean(ev_texts, clip_model, device) if ev_texts else null_h.clone()
            feat["h_e_neg"] = null_h.clone()
            feat["h_e_pos_seq"] = encode_texts_seq(ev_texts, clip_model, device, fixed_k=EV_K)
            feat["h_e_neg_seq"] = torch.zeros(EV_K, 768)

        # h_r: from standard reasoning_v2 (same as FilteredEv)
        reasoning_data = load_json(os.path.join(REASONING_DIR, f"{vid}.json"))
        feat["h_r"] = encode_reasoning_v2(sbert, reasoning_data).cpu()

        # graph_adj: always recompute from claims + graph (same structure)
        feat["graph_adj"] = build_graph_adj(claims_data, graph_data)

        torch.save(feat, os.path.join(dst_feat_dir, f"feat_{vid}.pt"))

    print(f"\nDone. Features saved to: {dst_feat_dir}")


if __name__ == "__main__":
    main()
