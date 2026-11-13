"""
Step 3b: Build Cross-modal Claim Consistency Graph

For each video, compute NLI (contradiction / support / neutral) between all pairs
of claims from different modalities. This graph is used in:
  - Step 4: priority = conflict_score × confidence (API budget allocation)
  - Step 6: consistency_summary injected into MLLM reasoning prompt

Output: data/cache/{dataset}/claim_graphs/{vid}.json
  {
    "vid": "...",
    "nodes": [{"id": "OCR_1", "claim": "...", "modality": "ocr", "confidence": 0.9}, ...],
    "edges": [
      {"source_id": "OCR_1", "source_modality": "ocr",
       "target_id": "VIS_1", "target_modality": "vision",
       "relation": "contradiction",   // "contradiction" | "support" | "neutral"
       "nli_scores": {"entailment": 0.1, "contradiction": 0.82, "neutral": 0.08},
       "explanation": "OCR claims X, but vision shows Y"}
    ],
    "max_conflict_score": 0.82  // highest inter-modality conflict score
  }

Usage:
    python preprocess/03b_build_claim_graph.py --dataset FakeSV --limit 50
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from sentence_transformers import CrossEncoder
from tqdm import tqdm

CACHE_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "cache")


def load_nli_model(device: str = "cuda"):
    return CrossEncoder("cross-encoder/nli-deberta-v3-base", num_labels=3, device=device)


def get_nli(model: CrossEncoder, premise: str, hypothesis: str) -> dict:
    """DeBERTa NLI label order: contradiction(0), entailment(1), neutral(2)"""
    raw = model.predict([(premise, hypothesis)], convert_to_numpy=False)
    probs = F.softmax(raw[0].detach().float(), dim=-1).tolist()
    return {"contradiction": probs[0], "entailment": probs[1], "neutral": probs[2]}


def classify_relation(scores: dict, threshold: float = 0.5) -> str:
    if scores["contradiction"] > threshold:
        return "contradiction"
    if scores["entailment"] > threshold:
        return "support"
    return "neutral"


def build_graph(vid: str, claims_data: dict, nli_model) -> dict:
    """Build cross-modal claim graph for one video."""
    all_claims = claims_data.get("all_claims", [])
    if not all_claims:
        return {"vid": vid, "nodes": [], "edges": [], "max_conflict_score": 0.0}

    nodes = []
    for c in all_claims:
        nodes.append({
            "id": c.get("id", f"{c.get('source_modality','?').upper()}_?"),
            "claim": c.get("claim", ""),
            "modality": c.get("source_modality", "unknown"),
            "confidence": c.get("confidence", 0.5),
        })

    pairs = []
    pair_nodes = []

    # Batch NLI calls for all cross-modal pairs in this video.
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            ni, nj = nodes[i], nodes[j]
            if ni["modality"] == nj["modality"]:
                continue  # skip same-modality pairs
            if not ni["claim"].strip() or not nj["claim"].strip():
                continue
            pairs.append((ni["claim"], nj["claim"]))
            pair_nodes.append((ni, nj))

    edges = []
    max_conflict = 0.0
    if pairs:
        raw_scores = nli_model.predict(pairs, convert_to_numpy=False, batch_size=32)
        raw_tensor = (torch.stack(raw_scores) if isinstance(raw_scores, list)
                      and raw_scores and isinstance(raw_scores[0], torch.Tensor)
                      else torch.as_tensor(raw_scores))
        probabilities = F.softmax(raw_tensor.float(), dim=-1).tolist()
    else:
        probabilities = []

    for (ni, nj), probs in zip(pair_nodes, probabilities):
        scores = {"contradiction": probs[0], "entailment": probs[1], "neutral": probs[2]}
        relation = classify_relation(scores)
        conflict_score = scores["contradiction"]

        if conflict_score > max_conflict:
            max_conflict = conflict_score

        edges.append({
            "source_id": ni["id"],
            "source_modality": ni["modality"],
            "target_id": nj["id"],
            "target_modality": nj["modality"],
            "relation": relation,
            "nli_scores": scores,
            "conflict_score": conflict_score,
        })

    return {"vid": vid, "nodes": nodes, "edges": edges, "max_conflict_score": round(max_conflict, 4)}


def graph_matches_claims(graph: dict, claims_data: dict) -> bool:
    """Check node identity and text, not just whether a graph file exists."""
    expected = [
        (c.get("id", f"{c.get('source_modality','?').upper()}_?"),
         c.get("claim", ""), c.get("source_modality", "unknown"))
        for c in claims_data.get("all_claims", [])
    ]
    actual = [
        (n.get("id"), n.get("claim"), n.get("modality"))
        for n in graph.get("nodes", [])
    ]
    return actual == expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["FakeSV", "FakeTT", "WildFake"], default="FakeSV")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--refresh-stale", action="store_true",
                        help="Rebuild graphs whose nodes differ from current claims")
    parser.add_argument("--output-dir", default=None,
                        help="Write graphs to a separate directory; existing originals are copied unchanged")
    args = parser.parse_args()
    if args.refresh_stale and not args.output_dir:
        parser.error("--refresh-stale requires --output-dir to preserve the original graph cache")

    claims_dir = os.path.join(CACHE_ROOT, args.dataset, "claims")
    source_dir = os.path.join(CACHE_ROOT, args.dataset, "claim_graphs")
    graph_dir = args.output_dir or source_dir
    os.makedirs(graph_dir, exist_ok=True)

    if os.path.abspath(graph_dir) != os.path.abspath(source_dir):
        for name in os.listdir(source_dir):
            if name.endswith(".json") and not os.path.exists(os.path.join(graph_dir, name)):
                shutil.copy2(os.path.join(source_dir, name), os.path.join(graph_dir, name))

    done = {os.path.splitext(f)[0] for f in os.listdir(graph_dir) if f.endswith(".json")}
    claim_vids = [os.path.splitext(f)[0] for f in os.listdir(claims_dir) if f.endswith(".json")]
    todo = [v for v in claim_vids if v not in done]
    if args.refresh_stale:
        for vid in claim_vids:
            if vid not in done:
                continue
            with open(os.path.join(claims_dir, f"{vid}.json"), encoding="utf-8") as f:
                claims_data = json.load(f)
            try:
                with open(os.path.join(graph_dir, f"{vid}.json"), encoding="utf-8") as f:
                    graph_data = json.load(f)
                fresh = graph_matches_claims(graph_data, claims_data)
            except (OSError, ValueError):
                fresh = False
            if not fresh:
                todo.append(vid)
    if args.limit:
        todo = todo[:args.limit]
    print(f"Claims: {len(claim_vids)}  Done: {len(done)}  Todo: {len(todo)}")

    if not todo:
        print("All done.")
        return

    print("Loading DeBERTa NLI model...")
    nli_model = load_nli_model(args.device)

    for vid in tqdm(todo, desc="Building claim graphs"):
        with open(os.path.join(claims_dir, f"{vid}.json"), encoding="utf-8") as f:
            claims_data = json.load(f)

        graph = build_graph(vid, claims_data, nli_model)

        with open(os.path.join(graph_dir, f"{vid}.json"), "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
