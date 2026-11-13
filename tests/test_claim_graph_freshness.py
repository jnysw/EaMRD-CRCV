"""Regression checks for claim graph cache freshness."""

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "preprocess" / "03b_build_claim_graph.py"
    spec = importlib.util.spec_from_file_location("claim_graph_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_graph_freshness_checks_text_and_order():
    matches = _module().graph_matches_claims
    claims = {"all_claims": [
        {"id": "OCR_1", "claim": "A", "source_modality": "ocr"},
        {"id": "VIS_1", "claim": "B", "source_modality": "vision"},
    ]}
    graph = {"nodes": [
        {"id": "OCR_1", "claim": "A", "modality": "ocr"},
        {"id": "VIS_1", "claim": "B", "modality": "vision"},
    ]}
    assert matches(graph, claims)
    graph["nodes"][1]["claim"] = "old B"
    assert not matches(graph, claims)
    graph["nodes"][1]["claim"] = "B"
    graph["nodes"].reverse()
    assert not matches(graph, claims)
