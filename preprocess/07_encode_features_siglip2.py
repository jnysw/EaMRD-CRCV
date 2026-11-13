"""
Step 7 (SigLIP2 variant): Offline Feature Encoding with SigLIP2

Replaces Chinese-CLIP + SBERT with SigLIP2 for vision & text features.
SBERT reasoning encoding is kept (h_r is task-specific; SigLIP2 text
encoder is not better suited for free-form reasoning text).

Output cache:  data/cache/FakeSV_SigLIP2/features/feat_{vid}.pt
  {
    "h_v":         [768]      -- SigLIP2 vision, mean over 16 frames
    "h_c":         [768]      -- SigLIP2 text, mean-pooled claims
    "h_e_pos":     [768]      -- SigLIP2 text, mean-pooled semantic evidence
    "h_e_neg":     [768]      -- SigLIP2 text, mean-pooled contradiction evidence
    "h_r":         [768]      -- SBERT reasoning (unchanged)
    "h_c_seq":     [N_c, 768] -- SigLIP2 text, per-claim
    "h_e_pos_seq": [5, 768]   -- SigLIP2 text, per-semantic-evidence
    "h_e_neg_seq": [5, 768]   -- SigLIP2 text, per-contradiction-evidence
    "graph_adj":   [N_c, N_c] -- Claim graph (unchanged, from 07)
  }

Usage:
    python preprocess/07_encode_features_siglip2.py --dataset FakeSV
    python preprocess/07_encode_features_siglip2.py --dataset FakeSV --limit 50
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

CACHE_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "cache")

FRAMES_ROOTS = {
    "FakeSV": r"F:\Bishe_workspace\fakesv-qwen3\frames_16",
    "FakeTT": r"F:\Bishe_workspace\FakeTT\frames_16",
}

SIGLIP2_MODEL = "google/siglip2-base-patch16-224"
SIGLIP2_LOCAL = r"F:\Bishe_workspace\hf_cache"   # local cache dir for snapshot_download

SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"

EV_K = 5  # fixed number of evidence items


# ---------------------------------------------------------------------------
# Helpers shared with 07_encode_features.py
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
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


def extract_evidence_texts(scored: dict, query_type: str, top_k: int = EV_K) -> list[str]:
    key = "top_semantic" if query_type == "semantic" else "top_contradiction"
    items = scored.get(key, [])[:top_k]
    return [ev.get("summary", "") for ev in items if ev.get("summary")]


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


@torch.no_grad()
def encode_reasoning(sbert, reasoning_data: dict) -> torch.Tensor:
    parts = []
    if reasoning_data.get("reason"):
        parts.append(reasoning_data["reason"])
    parts.extend(reasoning_data.get("key_evidence_used", [])[:2])
    if not parts:
        return torch.zeros(768)
    text = " ".join(parts)[:512]
    emb = sbert.encode([text], convert_to_tensor=True, normalize_embeddings=True)
    return emb.cpu().squeeze(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["FakeSV", "FakeTT"], default="FakeSV")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default=SIGLIP2_MODEL,
                        help="SigLIP2 model name or local path")
    args = parser.parse_args()

    # Output to a separate cache dir so we don't overwrite Chinese-CLIP features
    cache_key = f"{args.dataset}_SigLIP2"
    feat_dir = os.path.join(CACHE_ROOT, cache_key, "features")
    os.makedirs(feat_dir, exist_ok=True)

    # Source dirs (claims, evidence, graph) from original FakeSV cache
    src_dir = os.path.join(CACHE_ROOT, args.dataset)
    claims_dir    = os.path.join(src_dir, "claims")
    scored_dir    = os.path.join(src_dir, "evidence", "scored")
    reasoning_dir = os.path.join(src_dir, "reasoning")
    graph_dir     = os.path.join(src_dir, "claim_graphs")

    # Find videos to encode
    claim_vids = [os.path.splitext(f)[0] for f in os.listdir(claims_dir) if f.endswith(".json")]
    done = {os.path.splitext(f[5:])[0] for f in os.listdir(feat_dir) if f.startswith("feat_")}
    todo = [v for v in claim_vids if v not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"Cache dir : {feat_dir}")
    print(f"Claims    : {len(claim_vids)}")
    print(f"Done      : {len(done)}")
    print(f"Todo      : {len(todo)}")

    if not todo:
        print("All done.")
        return

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device    : {device}")

    # Load models
    print("\nLoading SigLIP2 encoder...")
    # Resolve local path: prefer flat dir if it contains model.safetensors,
    # then try HF snapshot cache, then fall back to args.model
    siglip_path = args.model
    flat_dir = os.path.join(SIGLIP2_LOCAL, "siglip2-base-patch16-224")
    hf_snapshots = os.path.join(SIGLIP2_LOCAL, "models--google--siglip2-base-patch16-224",
                                "snapshots")
    if os.path.isfile(os.path.join(flat_dir, "model.safetensors")):
        siglip_path = flat_dir
        print(f"  Using flat local dir: {siglip_path}")
    elif os.path.isdir(hf_snapshots):
        snapshots = [s for s in os.listdir(hf_snapshots)
                     if os.path.isfile(os.path.join(hf_snapshots, s, "model.safetensors"))]
        if snapshots:
            siglip_path = os.path.join(hf_snapshots, snapshots[0])
            print(f"  Using local snapshot: {siglip_path}")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.siglip2_encoder import SigLIP2Encoder
    encoder = SigLIP2Encoder(siglip_path, device=device)
    DIM = encoder.dim

    print("\nLoading SBERT (for reasoning)...")
    sbert = SentenceTransformer(SBERT_MODEL, device=device)

    frames_root = FRAMES_ROOTS[args.dataset]
    null_hv  = torch.zeros(DIM)
    null_hce = torch.zeros(DIM)

    for vid in tqdm(todo, desc=f"Encoding [{cache_key}]"):
        feat_path = os.path.join(feat_dir, f"feat_{vid}.pt")

        claims_data = load_json(os.path.join(claims_dir, f"{vid}.json"))
        scored      = load_json(os.path.join(scored_dir, f"{vid}.json"))
        graph_data  = load_json(os.path.join(graph_dir, f"{vid}.json"))

        claim_texts = extract_claim_texts(claims_data)
        pos_texts   = extract_evidence_texts(scored, "semantic")
        neg_texts   = extract_evidence_texts(scored, "contradiction")

        feat = {}

        # --- h_v: encode video frames with SigLIP2 vision encoder ---
        vid_dir = os.path.join(frames_root, vid)
        frame_paths = sorted(glob(os.path.join(vid_dir, "frame_*.jpg")))
        _h_v = encoder.encode_image_from_paths(frame_paths)
        feat["h_v"] = (_h_v if _h_v is not None else null_hv).cpu()

        # --- mean-pooled text features ---
        feat["h_c"]     = (encoder.encode_texts(claim_texts) if claim_texts else null_hce).cpu()
        feat["h_e_pos"] = (encoder.encode_texts(pos_texts)   if pos_texts   else null_hce).cpu()
        feat["h_e_neg"] = (encoder.encode_texts(neg_texts)   if neg_texts   else null_hce).cpu()

        # --- SBERT reasoning (unchanged) ---
        feat["h_r"] = encode_reasoning(
            sbert, load_json(os.path.join(reasoning_dir, f"{vid}.json"))
        ).cpu()

        # --- sequence features for GAT + CECA ---
        n_claims = len(claim_texts)
        feat["h_c_seq"] = (
            encoder.encode_texts_seq(claim_texts) if n_claims > 0
            else torch.zeros(1, DIM)
        ).cpu()
        feat["h_e_pos_seq"] = encoder.encode_texts_seq(pos_texts,  fixed_k=EV_K).cpu()
        feat["h_e_neg_seq"] = encoder.encode_texts_seq(neg_texts,  fixed_k=EV_K).cpu()
        feat["graph_adj"]   = build_graph_adj(claims_data, graph_data).cpu()

        torch.save(feat, feat_path)

    print(f"\nDone. Features saved to: {feat_dir}")


if __name__ == "__main__":
    main()
