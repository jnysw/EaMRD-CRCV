"""
Step 9: Ablation Feature Encoding

Creates feature caches for ablation study variants.
All variants keep h_v, h_r unchanged (same as FakeSV_FilteredEv).
Only h_c and h_e differ to isolate each module's contribution.

Variants:
  woCMR   — w/o Claim-centric Retrieval
              h_c: single global CLIP encoding of event_summary text
              h_e: old scored evidence (evidence/scored/, not per-claim filtered)

  woEV    — w/o Evidence Verification  
              h_c: per-claim CLIP (unchanged)
              h_e: raw retrieved evidence (evidence/raw/, no scoring/filtering)

  scoreEV — Score-based Evidence Selection (traditional scoring, no LLM verifier)
              h_c: per-claim CLIP (unchanged)
              h_e: old NLI+time scored evidence (evidence/scored/top_semantic)

  woEGR   — w/o Evidence-Guided Reasoning
              h_c / h_e / h_v: identical to FakeSV_FilteredEv
              h_r: SBERT encoding of reasoning_v2_woEGR/ (evidence hidden from LLM)

  woCMC   — w/o Cross-Modal Conflict in Reasoning
              h_c / h_e / h_v: identical to FakeSV_FilteredEv
              h_r: SBERT encoding of reasoning_v2_woCMC/ (conflict analysis hidden)

Usage:
  python preprocess/09_encode_ablation.py --variant woCMR
  python preprocess/09_encode_ablation.py --variant woEV
  python preprocess/09_encode_ablation.py --variant scoreEV
  python preprocess/09_encode_ablation.py --variant woEGR
  python preprocess/09_encode_ablation.py --variant woCMC
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CACHE_ROOT      = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
CLIP_MODEL_PATH = r"F:\Bishe_workspace\ExMRD-master\preprocess\model\chinese-clip-vit-large-patch14"
SBERT_MODEL     = "sentence-transformers/all-mpnet-base-v2"
VIT_TENSOR_PATH = r"F:\Bishe_workspace\fakesv-qwen3\vit_tensor.pt"

SRC_CACHE     = os.path.join(CACHE_ROOT, "FakeSV")
CLAIMS_DIR    = os.path.join(SRC_CACHE, "claims")
SCORED_DIR    = os.path.join(SRC_CACHE, "evidence", "scored")
RAW_DIR       = os.path.join(SRC_CACHE, "evidence", "raw")
REASONING_DIR = os.path.join(SRC_CACHE, "reasoning_v2")
GRAPH_DIR     = os.path.join(SRC_CACHE, "claim_graphs")
SPLIT_TARGET  = os.path.join(SRC_CACHE, "Split_target.txt")

EVENT_SUMMARY_PATH = os.path.join(
    os.path.dirname(__file__), "output", "event_summary_FakeSV.jsonl"
)

EV_K = 5   # fixed number of evidence slots

VARIANT_CACHE = {
    "woCMR":   "FakeSV_woCMR",
    "woEV":    "FakeSV_woEV",
    "scoreEV": "FakeSV_scoreEV",
    "woEGR":   "FakeSV_woEGR",   # same h_c/h_e/h_v as FilteredEv, only h_r differs
    "woCMC":   "FakeSV_woCMC",   # same h_c/h_e/h_v as FilteredEv, only h_r differs
}

# For woEGR / woCMC: source for h_r (different reasoning output dirs)
VARIANT_REASONING_DIR = {
    "woEGR": os.path.join(SRC_CACHE, "reasoning_v2_woEGR"),
    "woCMC": os.path.join(SRC_CACHE, "reasoning_v2_woCMC"),
}

# For woEGR / woCMC: source for h_c / h_e / h_v (copy from FilteredEv)
FILTEREDEV_FEAT_DIR = os.path.join(CACHE_ROOT, "FakeSV_FilteredEv", "features")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_chinese_clip(model_path: str, device: str):
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor
    processor = ChineseCLIPProcessor.from_pretrained(model_path)
    model = ChineseCLIPModel.from_pretrained(model_path).to(device).eval()
    return model, processor


def load_sbert(device: str):
    return SentenceTransformer(SBERT_MODEL, device=device)

# ---------------------------------------------------------------------------
# Encoding helpers (reused from 08_encode_features_v2.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_texts_mean(texts, model, processor, device):
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
    return features.cpu().mean(dim=0)


@torch.no_grad()
def encode_texts_seq(texts, model, processor, device, fixed_k=None):
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
def encode_reasoning_v2(sbert, data: dict) -> torch.Tensor:
    parts = []
    if data.get("overall_assessment"):
        parts.append(f"Assessment: {data['overall_assessment']}")
    if data.get("reason"):
        parts.append(data["reason"])
    for kev in data.get("key_evidence_used", [])[:2]:
        txt = kev.get("evidence") or kev.get("claim_id", "") if isinstance(kev, dict) else str(kev)
        if txt:
            parts.append(txt)
    for conf in data.get("cross_modal_conflicts", [])[:2]:
        if isinstance(conf, str) and conf:
            parts.append(conf)
    if not parts:
        return torch.zeros(768)
    text = " ".join(parts)[:512]
    emb = sbert.encode([text], convert_to_tensor=True, normalize_embeddings=True)
    return emb.cpu().squeeze(0)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8", errors="replace") as f:
        return json.load(f)


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
        src, tgt = edge.get("source_id", ""), edge.get("target_id", "")
        score = float(edge.get("conflict_score", 0.0))
        if src in id2idx and tgt in id2idx:
            i, j = id2idx[src], id2idx[tgt]
            adj[i, j] = score
            adj[j, i] = score
    return adj + torch.eye(n)


def load_event_summaries(path: str) -> dict:
    """Load event_summary_FakeSV.jsonl → {vid: data}."""
    lookup = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                lookup[d["vid"]] = d
    return lookup


def event_summary_text(ev: dict) -> str:
    """Build a single global description text from an event summary entry."""
    parts = []
    if ev.get("event_summary"):
        parts.append(ev["event_summary"])
    entities = ev.get("entities", [])
    if entities:
        parts.append("涉及实体：" + "，".join(str(e) for e in entities))
    vt = ev.get("verification_targets", [])
    if vt:
        parts.append("核实目标：" + "，".join(str(v) for v in vt))
    return "；".join(parts)[:256]


def get_scored_texts(scored: dict, query_type: str, top_k: int = EV_K) -> list:
    key = "top_semantic" if query_type == "semantic" else "top_contradiction"
    return [ev.get("summary", "") for ev in scored.get(key, [])[:top_k] if ev.get("summary")]


def get_raw_texts(raw: dict, query_type: str, top_k: int = EV_K) -> list:
    """Get evidence texts from raw evidence by retrieval order."""
    ev_list = raw.get("evidence", [])
    # Filter by query_type (semantic = general; contradiction = NLI-based)
    # In raw evidence, query_type is stored as a field
    typed = [e for e in ev_list if e.get("query_type") == query_type]
    if not typed:
        typed = ev_list   # fallback: use all
    texts = []
    for ev in typed[:top_k]:
        text = ev.get("summary") or ev.get("snippet") or ev.get("title") or ""
        if text:
            texts.append(text)
    return texts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["woCMR", "woEV", "scoreEV", "woEGR", "woCMC"], required=True)
    parser.add_argument("--limit",  type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cache_name = VARIANT_CACHE[args.variant]
    feat_dir   = os.path.join(CACHE_ROOT, cache_name, "features")
    os.makedirs(feat_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Variant: {args.variant}  →  {cache_name}")
    print(f"Device: {device}")

    # --- Load split target ---
    with open(SPLIT_TARGET, encoding="utf-8") as f:
        target_vids = set(f.read().strip().split())
    done_vids = {f[5:-3] for f in os.listdir(feat_dir) if f.startswith("feat_")}
    todo = sorted(target_vids - done_vids)
    if args.limit:
        todo = todo[:args.limit]
    print(f"Target: {len(target_vids)}  Done: {len(done_vids)}  Todo: {len(todo)}")
    if not todo:
        print("All done.")
        return

    # ── Fast path for woEGR / woCMC: copy FilteredEv features, only update h_r ──
    if args.variant in ("woEGR", "woCMC"):
        reasoning_dir_abl = VARIANT_REASONING_DIR[args.variant]
        if not os.path.isdir(reasoning_dir_abl):
            print(f"[ERROR] Reasoning dir not found: {reasoning_dir_abl}")
            print(f"  Run: python preprocess/10_run_reasoning_ablation.py --ablation {args.variant}")
            sys.exit(1)

        print("Loading SBERT (only h_r will be re-encoded)...")
        sbert = load_sbert(device)

        missing_r = 0
        for vid in tqdm(todo, desc=f"Encoding [{cache_name}] (h_r only)"):
            src_path = os.path.join(FILTEREDEV_FEAT_DIR, f"feat_{vid}.pt")
            dst_path = os.path.join(feat_dir, f"feat_{vid}.pt")

            if not os.path.exists(src_path):
                missing_r += 1
                continue

            feat = torch.load(src_path, weights_only=True)
            # Re-encode h_r from ablation-specific reasoning dir
            reasoning_data = load_json(os.path.join(reasoning_dir_abl, f"{vid}.json"))
            feat["h_r"] = encode_reasoning_v2(sbert, reasoning_data).cpu()
            torch.save(feat, dst_path)

        print(f"\nDone. Missing FilteredEv source: {missing_r}/{len(todo)}")
        print(f"Features saved to: {feat_dir}")
        return

    # --- Load models ---
    print("Loading SBERT...")
    sbert = load_sbert(device)
    print(f"Loading Chinese-CLIP...")
    clip_model, clip_proc = load_chinese_clip(CLIP_MODEL_PATH, device)
    print(f"Loading vit_tensor.pt...")
    vit_cache = torch.load(VIT_TENSOR_PATH, weights_only=True)

    # --- Load auxiliary data ---
    ev_summaries = {}
    if args.variant == "woCMR":
        print("Loading event summaries...")
        ev_summaries = load_event_summaries(EVENT_SUMMARY_PATH)
        print(f"  Loaded {len(ev_summaries)} event summaries.")

    null_hv  = torch.zeros(1024)
    null_hce = torch.zeros(768)
    missing_ev = 0

    for vid in tqdm(todo, desc=f"Encoding [{cache_name}]"):
        feat_path = os.path.join(feat_dir, f"feat_{vid}.pt")
        feat = {}

        # ── Visual features (always same) ────────────────────────────────
        vit_feat = vit_cache.get(vid)
        if vit_feat is not None:
            h_v = F.normalize(vit_feat.float().mean(dim=0), dim=-1)
        else:
            h_v = null_hv
        feat["h_v"] = h_v.cpu()

        # ── Reasoning features (always same: reasoning_v2) ───────────────
        reasoning_data = load_json(os.path.join(REASONING_DIR, f"{vid}.json"))
        feat["h_r"] = encode_reasoning_v2(sbert, reasoning_data).cpu()

        # ── Claim features ────────────────────────────────────────────────
        claims_data = load_json(os.path.join(CLAIMS_DIR, f"{vid}.json"))
        graph_data  = load_json(os.path.join(GRAPH_DIR,  f"{vid}.json"))
        claim_texts = extract_claim_texts(claims_data)

        if args.variant == "woCMR":
            # Single global description from event_summary
            ev_sum = ev_summaries.get(vid, {})
            global_text = event_summary_text(ev_sum) if ev_sum else " ".join(claim_texts)
            feat["h_c"] = encode_texts_mean([global_text], clip_model, clip_proc, device) if global_text else null_hce
            feat["h_c_seq"] = encode_texts_seq([global_text], clip_model, clip_proc, device).cpu()  # [1, 768]
            feat["graph_adj"] = torch.zeros(1, 1)   # no claim graph for single global claim
        else:
            # Per-claim (unchanged from full model)
            feat["h_c"] = encode_texts_mean(claim_texts, clip_model, clip_proc, device) if claim_texts else null_hce
            feat["h_c_seq"] = encode_texts_seq(claim_texts, clip_model, clip_proc, device).cpu() if claim_texts else torch.zeros(1, 768)
            feat["graph_adj"] = build_graph_adj(claims_data, graph_data).cpu()

        # ── Evidence features ─────────────────────────────────────────────
        if args.variant == "woCMR":
            # Old scored evidence (non-claim-specific general evidence)
            scored = load_json(os.path.join(SCORED_DIR, f"{vid}.json"))
            pos_texts = get_scored_texts(scored, "semantic")
            neg_texts = get_scored_texts(scored, "contradiction")

        elif args.variant == "woEV":
            # Raw retrieved evidence — no scoring, no LLM verification
            raw = load_json(os.path.join(RAW_DIR, f"{vid}.json"))
            pos_texts = get_raw_texts(raw, "semantic")
            neg_texts = get_raw_texts(raw, "contradiction")

        elif args.variant == "scoreEV":
            # Traditionally scored evidence (NLI+time score, no LLM verification)
            scored = load_json(os.path.join(SCORED_DIR, f"{vid}.json"))
            pos_texts = get_scored_texts(scored, "semantic")
            neg_texts = get_scored_texts(scored, "contradiction")

        if not pos_texts:
            missing_ev += 1

        feat["h_e_pos"] = encode_texts_mean(pos_texts, clip_model, clip_proc, device) if pos_texts else null_hce
        feat["h_e_neg"] = encode_texts_mean(neg_texts, clip_model, clip_proc, device) if neg_texts else null_hce
        feat["h_e_pos_seq"] = encode_texts_seq(pos_texts, clip_model, clip_proc, device, fixed_k=EV_K).cpu()
        feat["h_e_neg_seq"] = encode_texts_seq(neg_texts, clip_model, clip_proc, device, fixed_k=EV_K).cpu()

        torch.save(feat, feat_path)

    print(f"\nDone. Missing evidence: {missing_ev}/{len(todo)}")
    print(f"Features saved to: {feat_dir}")


if __name__ == "__main__":
    main()
