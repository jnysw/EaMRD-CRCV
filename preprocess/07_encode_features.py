"""
Step 7: Offline Feature Encoding

For each video, encodes and saves all training features as a single .pt file:
  data/cache/{dataset}/features/feat_{vid}.pt
  {
    "h_v":     tensor([1024])  — visual (mean-pooled CLIP vision, 1024 for FakeSV / 768 for FakeTT)
    "h_c":     tensor([512])   — claim text (CLIP text encoder, same space as h_v)
    "h_e_pos": tensor([512])   — top-semantic evidence text
    "h_e_neg": tensor([512])   — top-contradiction evidence text
    "h_r":     tensor([768])   — structured reasoning (SBERT all-mpnet-base-v2)
  }

FakeSV: uses Chinese-CLIP (OFA-Sys/chinese-clip-vit-large-patch14)
  - h_v loaded from pre-extracted vit_tensor.pt (1024-dim, mean over 16 frames)
  - h_c / h_e: Chinese-CLIP text encoder (512-dim projected)
FakeTT: uses openai/clip-vit-large-patch14
  - h_v extracted from raw frames (768-dim, mean over 16 frames)
  - h_c / h_e: openai CLIP text encoder (768-dim)

Usage:
    python preprocess/07_encode_features.py --dataset FakeSV --limit 50
    python preprocess/07_encode_features.py --dataset FakeSV
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

VIT_TENSOR_PATH = r"F:\Bishe_workspace\fakesv-qwen3\vit_tensor.pt"

FRAMES_ROOTS = {
    "FakeSV": r"F:\Bishe_workspace\fakesv-qwen3\frames_16",
    "FakeTT": r"F:\Bishe_workspace\FakeTT\frames_16",
}

CLIP_MODELS = {
    "FakeSV": r"F:\Bishe_workspace\ExMRD-master\preprocess\model\chinese-clip-vit-large-patch14",
    "FakeTT": "openai/clip-vit-large-patch14",
}

SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_chinese_clip(model_path: str, device: str):
    """Load Chinese-CLIP model and processor for text encoding."""
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor
    processor = ChineseCLIPProcessor.from_pretrained(model_path)
    model = ChineseCLIPModel.from_pretrained(model_path).to(device).eval()
    return model, processor


def load_openai_clip(device: str):
    """Load openai CLIP ViT-L/14 for both vision and text encoding."""
    import clip
    model, preprocess = clip.load("ViT-L/14", device=device)
    model = model.float().eval()
    return model, preprocess


def load_sbert(device: str):
    return SentenceTransformer(SBERT_MODEL, device=device)


# ---------------------------------------------------------------------------
# Visual feature helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_h_v_fakesv(vid: str, vit_cache: dict) -> torch.Tensor | None:
    """Return 1024-dim h_v from pre-extracted Chinese-CLIP features."""
    feat = vit_cache.get(vid)
    if feat is None:
        return None
    h_v = feat.float().mean(dim=0)  # [16, 1024] → [1024]
    return F.normalize(h_v, dim=-1)


@torch.no_grad()
def get_h_v_fakeTT(vid: str, frames_root: str, clip_model, preprocess, device: str) -> torch.Tensor | None:
    """Encode FakeTT frames with openai CLIP; return 768-dim h_v."""
    from PIL import Image
    vid_dir = os.path.join(frames_root, vid)
    frame_paths = sorted(glob(os.path.join(vid_dir, "frame_*.jpg")))
    if not frame_paths:
        return None
    feats = []
    for fp in frame_paths:
        img = preprocess(Image.open(fp).convert("RGB")).unsqueeze(0).to(device)
        feat = clip_model.encode_image(img).float()
        feat = F.normalize(feat, dim=-1)
        feats.append(feat.cpu())
    if not feats:
        return None
    return torch.stack(feats).mean(dim=0).squeeze(0)  # [768]


# ---------------------------------------------------------------------------
# Text feature helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_texts_chinese_clip(texts: list[str], model, processor, device: str) -> torch.Tensor:
    """Encode a list of texts with Chinese-CLIP; returns mean-pooled [768] tensor."""
    if not texts:
        return torch.zeros(768)
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=52).to(device)
    raw = model.get_text_features(**inputs)
    # transformers >= 5.x returns BaseModelOutputWithPooling instead of a tensor
    if torch.is_tensor(raw):
        features = raw.float()
    elif hasattr(raw, "pooler_output"):
        features = raw.pooler_output.float()
    else:
        features = raw.last_hidden_state[:, 0].float()
    features = F.normalize(features, dim=-1)
    return features.cpu().mean(dim=0)  # [768]


@torch.no_grad()
def encode_texts_openai_clip(texts: list[str], model, device: str) -> torch.Tensor:
    """Encode texts with openai CLIP; returns mean-pooled [768] tensor."""
    import clip
    if not texts:
        return torch.zeros(768)
    tokens = clip.tokenize(texts, truncate=True).to(device)
    features = model.encode_text(tokens).float()
    features = F.normalize(features, dim=-1)
    return features.cpu().mean(dim=0)  # [768]


# ---------------------------------------------------------------------------
# Reasoning text
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_reasoning(sbert, reasoning_data: dict) -> torch.Tensor:
    """Encode MLLM reasoning output with SBERT; returns [768]."""
    parts = []
    if reasoning_data.get("reason"):
        parts.append(reasoning_data["reason"])
    parts.extend(reasoning_data.get("key_evidence_used", [])[:2])
    if not parts:
        return torch.zeros(768)
    text = " ".join(parts)[:512]
    emb = sbert.encode([text], convert_to_tensor=True, normalize_embeddings=True)
    return emb.cpu().squeeze(0)  # [768]


# ---------------------------------------------------------------------------
# Load evidence/claims helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_claim_texts(claims_data: dict) -> list[str]:
    """Flatten all modality claims into text list."""
    out = []
    for key in ("ocr_claims", "asr_claims", "vision_claims", "title_claims"):
        for c in claims_data.get(key) or []:
            if c.get("claim"):
                out.append(c["claim"])
    return out[:6]  # cap to avoid very long token sequences


def extract_evidence_texts(scored: dict, query_type: str, top_k: int = 5) -> list[str]:
    key = "top_semantic" if query_type == "semantic" else "top_contradiction"
    items = scored.get(key, [])[:top_k]
    return [ev.get("summary", "") for ev in items if ev.get("summary")]


def extract_claim_ids(claims_data: dict) -> list[str]:
    """Return ordered list of claim IDs matching extract_claim_texts order."""
    out = []
    for key in ("ocr_claims", "asr_claims", "vision_claims", "title_claims"):
        for c in claims_data.get(key) or []:
            if c.get("claim"):
                out.append(c.get("id", ""))
    return out[:6]


@torch.no_grad()
def encode_texts_seq_chinese_clip(
    texts: list[str], model, processor, device: str, fixed_k: int | None = None
) -> torch.Tensor:
    """Encode each text individually with Chinese-CLIP; returns [N, 768].
    If fixed_k is set, pads/truncates output to [fixed_k, 768] with zero rows."""
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
    return features


@torch.no_grad()
def encode_texts_seq_openai_clip(
    texts: list[str], model, device: str, fixed_k: int | None = None
) -> torch.Tensor:
    """Encode each text individually with openai CLIP; returns [N, 768]."""
    import clip
    if not texts:
        n = fixed_k if fixed_k else 1
        return torch.zeros(n, 768)
    tokens = clip.tokenize(texts, truncate=True).to(device)
    features = model.encode_text(tokens).float()
    features = F.normalize(features, dim=-1).cpu()
    if fixed_k is not None:
        if features.size(0) < fixed_k:
            pad = torch.zeros(fixed_k - features.size(0), 768)
            features = torch.cat([features, pad], dim=0)
        else:
            features = features[:fixed_k]
    return features


def build_graph_adj(claims_data: dict, graph_data: dict, max_nodes: int = 6) -> torch.Tensor:
    """Build claim graph adjacency matrix [N_c, N_c] with conflict scores.
    Rows/cols correspond to claim order from extract_claim_texts."""
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
            adj[j, i] = score  # undirected
    # Add self-loops
    adj = adj + torch.eye(n)
    return adj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["FakeSV", "FakeTT"], default="FakeSV")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--regen-seq", action="store_true",
                        help="Re-encode existing feat files to add h_c_seq/h_e_*_seq/graph_adj")
    args = parser.parse_args()

    feat_dir = os.path.join(CACHE_ROOT, args.dataset, "features")
    os.makedirs(feat_dir, exist_ok=True)

    claims_dir    = os.path.join(CACHE_ROOT, args.dataset, "claims")
    scored_dir    = os.path.join(CACHE_ROOT, args.dataset, "evidence", "scored")
    reasoning_dir = os.path.join(CACHE_ROOT, args.dataset, "reasoning")
    graph_dir     = os.path.join(CACHE_ROOT, args.dataset, "claim_graphs")

    all_feat_files = {os.path.splitext(f[5:])[0]: os.path.join(feat_dir, f)
                      for f in os.listdir(feat_dir) if f.startswith("feat_")}
    claim_vids = [os.path.splitext(f)[0] for f in os.listdir(claims_dir) if f.endswith(".json")]

    if args.regen_seq:
        # Re-encode existing feat files that lack sequence features
        todo = [v for v in claim_vids if v in all_feat_files and
                "h_c_seq" not in torch.load(all_feat_files[v], weights_only=True)]
        print(f"Regen-seq mode: {len(todo)} files missing sequence features")
    else:
        done = set(all_feat_files.keys())
        todo = [v for v in claim_vids if v not in done]

    if args.limit:
        todo = todo[:args.limit]
    print(f"Claims: {len(claim_vids)}  Todo: {len(todo)}")

    if not todo:
        print("All done.")
        return

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load models
    print("Loading SBERT...")
    sbert = load_sbert(device)

    vit_cache = None
    clip_model = clip_proc = clip_preprocess = None

    if args.dataset == "FakeSV":
        print("Loading Chinese-CLIP...")
        clip_model, clip_proc = load_chinese_clip(CLIP_MODELS["FakeSV"], device)
        if not args.regen_seq:
            print(f"Loading vit_tensor.pt from {VIT_TENSOR_PATH} ...")
            vit_cache = torch.load(VIT_TENSOR_PATH, weights_only=True)
            print(f"  Loaded {len(vit_cache)} entries.")
    else:
        print("Loading openai CLIP ViT-L/14...")
        clip_model, clip_preprocess = load_openai_clip(device)
        if not args.regen_seq:
            pass

    if not args.regen_seq:
        vit_cache = torch.load(VIT_TENSOR_PATH, weights_only=True) if args.dataset == "FakeSV" and vit_cache is None else vit_cache

    null_h_v = torch.zeros(1024 if args.dataset == "FakeSV" else 768)
    null_hce = torch.zeros(768)
    EV_K = 5  # fixed number of evidence items to store per video

    for vid in tqdm(todo, desc=f"Encoding [{args.dataset}]"):
        feat_path = os.path.join(feat_dir, f"feat_{vid}.pt")

        # Load existing feat if regen mode
        if args.regen_seq and os.path.exists(feat_path):
            feat = torch.load(feat_path, weights_only=True)
        else:
            feat = {}

        claims_data = load_json(os.path.join(claims_dir, f"{vid}.json"))
        scored      = load_json(os.path.join(scored_dir, f"{vid}.json"))
        graph_data  = load_json(os.path.join(graph_dir, f"{vid}.json"))

        claim_texts = extract_claim_texts(claims_data)
        pos_texts   = extract_evidence_texts(scored, "semantic")
        neg_texts   = extract_evidence_texts(scored, "contradiction")

        if args.dataset == "FakeSV":
            encode_fn     = lambda texts: encode_texts_chinese_clip(texts, clip_model, clip_proc, device)
            encode_fn_seq = lambda texts, fixed_k=None: encode_texts_seq_chinese_clip(texts, clip_model, clip_proc, device, fixed_k=fixed_k)
        else:
            encode_fn     = lambda texts: encode_texts_openai_clip(texts, clip_model, device)
            encode_fn_seq = lambda texts, fixed_k=None: encode_texts_seq_openai_clip(texts, clip_model, device, fixed_k=fixed_k)

        if not args.regen_seq:
            # --- h_v (only in full encode mode) ---
            if args.dataset == "FakeSV":
                _h_v = get_h_v_fakesv(vid, vit_cache)
            else:
                _h_v = get_h_v_fakeTT(vid, FRAMES_ROOTS["FakeTT"], clip_model, clip_preprocess, device)
            feat["h_v"] = (_h_v if _h_v is not None else null_h_v).cpu()

            # --- mean-pooled (backward compat) ---
            feat["h_c"]     = (encode_fn(claim_texts) if claim_texts else null_hce).cpu()
            feat["h_e_pos"] = (encode_fn(pos_texts)   if pos_texts   else null_hce).cpu()
            feat["h_e_neg"] = (encode_fn(neg_texts)   if neg_texts   else null_hce).cpu()
            feat["h_r"]     = encode_reasoning(sbert, load_json(os.path.join(reasoning_dir, f"{vid}.json"))).cpu()

        # --- sequence features (new, for GAT + CECA) ---
        n_claims = len(claim_texts)
        if n_claims > 0:
            feat["h_c_seq"] = encode_fn_seq(claim_texts, fixed_k=None).cpu()  # [N_c, 768]
        else:
            feat["h_c_seq"] = torch.zeros(1, 768)

        feat["h_e_pos_seq"] = encode_fn_seq(pos_texts,  fixed_k=EV_K).cpu()  # [EV_K, 768]
        feat["h_e_neg_seq"] = encode_fn_seq(neg_texts,  fixed_k=EV_K).cpu()  # [EV_K, 768]
        feat["graph_adj"]   = build_graph_adj(claims_data, graph_data).cpu()  # [N_c, N_c]

        torch.save(feat, feat_path)

    print("Done.")


if __name__ == "__main__":
    main()
