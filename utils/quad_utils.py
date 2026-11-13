"""
Quads4 utilities for FakeSV/FakeTT preprocessing.

Quads4 naming convention:
  FakeSV: F:\\Bishe_workspace\\fakesv-qwen3\\quads_4\\{vid}_quad_{N}.jpg  (N=0,1,2,3)
  FakeTT: F:\\Bishe_workspace\\FakeTT\\quads_4\\{vid}_quad_{N}.jpg

Each quad is a 2×2 grid of 4 consecutive frames:
  quad_0 → frames 00-03   quad_1 → frames 04-07
  quad_2 → frames 08-11   quad_3 → frames 12-15
"""

import base64
import os
from glob import glob


QUADS_ROOTS = {
    "FakeSV": r"F:\Bishe_workspace\fakesv-qwen3\quads_4",
    "FakeTT": r"F:\Bishe_workspace\FakeTT\quads_4",
}

FRAMES_ROOTS = {
    "FakeSV": r"F:\Bishe_workspace\fakesv-qwen3\frames_16",
    "FakeTT": r"F:\Bishe_workspace\FakeTT\frames_16",
}


def get_quad_paths(video_id: str, dataset: str) -> list[str]:
    """
    Return 4 Quads4 image paths for a video.
    Falls back to up to 4 raw frames if quads are missing.
    """
    quads_root = QUADS_ROOTS.get(dataset, "")
    paths = [os.path.join(quads_root, f"{video_id}_quad_{i}.jpg") for i in range(4)]
    existing = [p for p in paths if os.path.exists(p)]
    if len(existing) == 4:
        return existing

    # Fallback: use first 4 raw frames
    frames_root = FRAMES_ROOTS.get(dataset, "")
    frame_dir = os.path.join(frames_root, video_id)
    fallback = sorted(glob(os.path.join(frame_dir, "frame_*.jpg")))[:4]
    return fallback


def encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_image_content(quad_paths: list[str]) -> list[dict]:
    """
    Build GPT-4o-mini multi-image content blocks for Quads4.
    Each quad gets a text label + image_url block.
    Uses detail='low' to save tokens (Quads4 already compresses temporal info).
    """
    content = []
    for i, path in enumerate(quad_paths):
        content.append({"type": "text", "text": f"[Quad {i+1}: frames {i*4+1}-{i*4+4}]"})
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image_b64(path)}",
                "detail": "low",
            },
        })
    return content


def has_quads(video_id: str, dataset: str) -> bool:
    quads_root = QUADS_ROOTS.get(dataset, "")
    return all(
        os.path.exists(os.path.join(quads_root, f"{video_id}_quad_{i}.jpg"))
        for i in range(4)
    )
