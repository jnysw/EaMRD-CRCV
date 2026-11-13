"""Re-encode FakeSV reasoning only, preserving all other cached features."""

import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
REASONING = CACHE / "FakeSV" / "reasoning_v2"
SOURCE = CACHE / "FakeSV_FilteredEv" / "features"
DEST = CACHE / "FakeSV_FilteredEv_hr2" / "features"
MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def reasoning_text(data):
    """Keep the text construction identical to Step 08's existing h_r path."""
    parts = []
    assessment = data.get("overall_assessment", "")
    if assessment:
        parts.append(f"Assessment: {assessment}")
    if data.get("reason"):
        parts.append(data["reason"])
    for item in data.get("key_evidence_used", [])[:2]:
        value = (item.get("evidence") or item.get("claim_id", "")) if isinstance(item, dict) else str(item)
        if value:
            parts.append(value)
    for conflict in data.get("cross_modal_conflicts", [])[:2]:
        if isinstance(conflict, str) and conflict:
            parts.append(conflict)
    return " ".join(parts)[:512]


def main():
    files = sorted(SOURCE.glob("feat_*.pt"))
    records = []
    for source in files:
        vid = source.stem[5:]
        data = json.loads((REASONING / f"{vid}.json").read_text(encoding="utf-8"))
        records.append((source, reasoning_text(data)))
    print(f"model={MODEL} samples={len(records)} nonempty={sum(bool(t) for _, t in records)}", flush=True)
    encoder = SentenceTransformer(MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    texts = [text for _, text in records if text]
    encoded = encoder.encode(texts, batch_size=64, convert_to_tensor=True,
                             normalize_embeddings=True, show_progress_bar=True).cpu()
    if encoded.ndim != 2 or encoded.shape[1] != 768:
        raise ValueError(f"expected 768-dimensional reasoning features, got {tuple(encoded.shape)}")
    DEST.mkdir(parents=True, exist_ok=True)
    index = 0
    for source, text in records:
        feature = torch.load(source, map_location="cpu", weights_only=True)
        feature["h_r"] = encoded[index].clone() if text else torch.zeros(768)
        index += bool(text)
        torch.save(feature, DEST / source.name)
    print(f"saved={len(records)} output={DEST}", flush=True)


if __name__ == "__main__":
    main()
