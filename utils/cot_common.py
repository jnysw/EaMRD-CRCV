"""Shared helpers for CoT preprocessing scripts."""

import io
import json
import os
from typing import Any

import pandas as pd


def safe_read_jsonl(file_path: str, default_columns: list[str] | None = None) -> pd.DataFrame:
    """Read jsonl safely; return empty DataFrame if file missing or empty."""
    cols = default_columns or ["vid", "ret"]
    if not os.path.exists(file_path):
        print(f"[警告] 文件不存在: {file_path}")
        return pd.DataFrame(columns=cols)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        if not content.strip():
            print(f"[警告] 文件为空: {file_path}")
            return pd.DataFrame(columns=cols)

        df = pd.read_json(io.StringIO(content), lines=True, dtype={"vid": str})
        if "video_id" in df.columns and "vid" not in df.columns:
            df = df.rename(columns={"video_id": "vid"})
        return df
    except Exception as exc:
        print(f"[错误] 读取失败: {file_path}, 错误: {exc}")
        return pd.DataFrame(columns=cols)


def get_cot_text(df: pd.DataFrame, vid: str, text_cols: tuple[str, ...] = ("ret", "text")) -> str:
    """Get CoT text from a row, trying ret then text column."""
    rows = df[df["vid"].astype(str) == str(vid)]
    if rows.empty:
        return ""
    row = rows.iloc[0]
    for col in text_cols:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col])
    return ""


def parse_query_json(raw: str, fallback_text: str = "") -> list[str]:
    """Parse LLM JSON output containing search queries."""
    raw = (raw or "").strip()
    if not raw:
        return _fallback_queries(fallback_text)

    # Try direct JSON parse
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "queries" in data:
            queries = data["queries"]
        elif isinstance(data, list):
            queries = data
        else:
            queries = []
        queries = [str(q).strip() for q in queries if str(q).strip()]
        if queries:
            return queries[:3]
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block from markdown
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
            queries = data.get("queries", [])
            queries = [str(q).strip() for q in queries if str(q).strip()]
            if queries:
                return queries[:3]
        except json.JSONDecodeError:
            pass

    return _fallback_queries(fallback_text)


def _fallback_queries(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    # Use first 80 chars of title/description as fallback query
    return [text[:80].strip()]
