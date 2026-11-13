"""Create a new FakeTT feature cache with only graph_adj refreshed."""

import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
CLAIMS = CACHE / "FakeTT" / "claims"
GRAPHS = CACHE / "FakeTT" / "claim_graphs_rebuilt"
SOURCE = CACHE / "FakeTT_FilteredEv_v3" / "features"
DEST = CACHE / "FakeTT_FilteredEv_v3_g2" / "features"


def claim_ids(data):
    ids = []
    for key in ("ocr_claims", "asr_claims", "vision_claims", "title_claims"):
        for claim in data.get(key) or []:
            if claim.get("claim"):
                ids.append(claim.get("id", ""))
    return ids[:6]


def graph_adj(data, graph):
    ids = claim_ids(data)
    if not ids:
        return torch.zeros(1, 1)
    index = {cid: i for i, cid in enumerate(ids)}
    adj = torch.eye(len(ids))
    for edge in graph.get("edges", []):
        src, dst = edge.get("source_id"), edge.get("target_id")
        if src in index and dst in index:
            i, j = index[src], index[dst]
            score = float(edge.get("conflict_score", 0.0))
            adj[i, j] = adj[j, i] = score
    return adj


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    files = sorted(SOURCE.glob("feat_*.pt"))
    changed = 0
    for source in files:
        vid = source.stem[5:]
        claims = json.loads((CLAIMS / f"{vid}.json").read_text(encoding="utf-8"))
        graph = json.loads((GRAPHS / f"{vid}.json").read_text(encoding="utf-8"))
        expected = [
            (c.get("id"), c.get("claim"), c.get("source_modality"))
            for c in claims.get("all_claims", [])
        ]
        actual = [(n.get("id"), n.get("claim"), n.get("modality"))
                  for n in graph.get("nodes", [])]
        if expected != actual:
            raise ValueError(f"stale graph: {vid}")
        feature = torch.load(source, map_location="cpu", weights_only=True)
        updated = graph_adj(claims, graph)
        if updated.shape != feature["graph_adj"].shape:
            raise ValueError(f"graph shape mismatch: {vid}")
        changed += not torch.equal(updated, feature["graph_adj"])
        feature["graph_adj"] = updated
        torch.save(feature, DEST / source.name)
    print(f"features={len(files)} graph_adj_changed={changed} output={DEST}")


if __name__ == "__main__":
    main()
