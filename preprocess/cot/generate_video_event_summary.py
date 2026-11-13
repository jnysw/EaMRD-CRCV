"""
Event-level Video Summary Extraction  (Step 3.5 in pipeline)

Reads from already-existing preprocessing outputs (frames/OCR/ASR/Claims).
Calls an MLLM (GPT-4o-mini by default via runapi.co) to generate structured
event-level context for each video.

Output:
  - Per-video cache:  data/cache/{dataset}/event_summaries/{vid}.json
  - Aggregated JSONL: preprocess/output/event_summary_{dataset}.jsonl

Usage:
    python preprocess/cot/generate_video_event_summary.py --dataset FakeSV --limit 20
    python preprocess/cot/generate_video_event_summary.py --dataset FakeTT
    python preprocess/cot/generate_video_event_summary.py --dataset FakeSV --workers 4
"""

import argparse
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from utils.chat_llm import ChatLLM
from utils.cot_common import safe_read_jsonl, get_cot_text
from utils.quad_utils import get_quad_paths, build_image_content, has_quads

from preprocess.cot.event_summary_prompt import (
    get_system_prompt,
    USER_PROMPT_TEMPLATE,
    LANG_REMINDERS,
    format_claims_text,
)
from preprocess.cot.event_summary_utils import (
    parse_event_summary_json,
    load_cache,
    save_cache,
    append_to_jsonl,
    load_done_vids_from_jsonl,
)

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CACHE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

DATA_PATHS = {
    "FakeSV": {
        "metadata": r"F:\Bishe_workspace\ExMRD-master\data\FakeSV\data.jsonl",
        "ocr":      r"F:\Bishe_workspace\fakesv-qwen3\ocr.jsonl",
        "asr":      r"F:\Bishe_workspace\fakesv-qwen3\transcript.jsonl",
    },
    "FakeTT": {
        "metadata": r"F:\Bishe_workspace\FakeTT\data.json",
        "ocr":      r"F:\Bishe_workspace\FakeTT\ocr.jsonl",
        "asr":      r"F:\Bishe_workspace\FakeTT\transcript.jsonl",
    },
    "WildFake": {
        "metadata": r"F:\Bishe_workspace\WildFake\data_zh.jsonl",
        "ocr":      r"F:\Bishe_workspace\WildFake\ocr.jsonl",
        "asr":      r"F:\Bishe_workspace\WildFake\transcript.jsonl",
    },
}

DEFAULT_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# VideoEventSummarizer — swappable MLLM interface
# ---------------------------------------------------------------------------

class VideoEventSummarizer:
    """
    Multimodal MLLM wrapper for event summary extraction.

    Swap the underlying model by passing a different model name or base_url.
    Compatible models: GPT-4o, GPT-4o-mini, Qwen-VL, InternVL (via OpenAI-compat API).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = "https://runapi.co/v1",
        api_key: str = "",
        dataset: str = "FakeSV",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ):
        self.dataset = dataset
        self.system_prompt = get_system_prompt(dataset)
        self._llm = ChatLLM(
            base_url=base_url,
            key=api_key,
            prompt="",            # not used in chat_single mode
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def summarize(
        self,
        vid: str,
        ocr: str,
        asr: str,
        claims_data: Optional[dict],
        quad_paths: list[str],
    ) -> Optional[dict]:
        """
        Call the MLLM and return a parsed event summary dict, or None on failure.
        """
        claims_text = format_claims_text(claims_data)
        lang_reminder = LANG_REMINDERS.get(self.dataset, "Use appropriate language.")
        user_text = USER_PROMPT_TEMPLATE.format(
            ocr_text=ocr[:800] or "(none)",
            asr_text=asr[:800] or "(none)",
            claims_text=claims_text,
            lang_reminder=lang_reminder,
        )

        # Build multimodal user message (text + image blocks)
        import requests as _requests
        import urllib3 as _urllib3
        _urllib3.disable_warnings(_urllib3.exceptions.InsecureRequestWarning)

        image_blocks = build_image_content(quad_paths) if quad_paths else []
        content = [{"type": "text", "text": user_text}] + image_blocks

        payload = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": content},
            ],
            "model": self._llm.model,
            "stream": False,
            "temperature": self._llm.temperature,
            "max_tokens": self._llm.max_tokens,
        }

        url = self._llm.base_url
        headers = {
            "Authorization": f"Bearer {self._llm.key}",
            "Content-Type": "application/json",
        }

        raw = None
        for attempt in range(3):
            resp_json = None
            try:
                resp = _requests.post(
                    url=url, headers=headers,
                    data=json.dumps(payload),
                    timeout=120,
                    verify=False,   # bypass SSL cert hostname mismatch
                )
                resp_json = resp.json()
                raw = resp_json["choices"][0]["message"]["content"]
                break
            except Exception as exc:
                if attempt < 2:
                    time.sleep(5)
                else:
                    raise Exception(resp_json or str(exc))

        if raw is None:
            raise Exception("No response after retries")

        result = parse_event_summary_json(raw, vid)
        return result


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_metadata(dataset: str) -> pd.DataFrame:
    paths = DATA_PATHS[dataset]
    meta_path = paths["metadata"]
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    try:
        df = pd.read_json(meta_path, lines=True, dtype={"video_id": str})
    except Exception:
        with open(meta_path, "r", encoding="utf-8") as f:
            records = [json.loads(l) for l in f if l.strip()]
        df = pd.DataFrame(records)
    if "vid" not in df.columns and "video_id" in df.columns:
        df = df.rename(columns={"video_id": "vid"})
    df["vid"] = df["vid"].astype(str)
    return df


def load_claims(vid: str, dataset: str) -> Optional[dict]:
    path = os.path.join(CACHE_ROOT, dataset, "claims", f"{vid}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def process_one(
    vid: str,
    summarizer: VideoEventSummarizer,
    ocr_df: pd.DataFrame,
    asr_df: pd.DataFrame,
    dataset: str,
    cache_dir: str,
) -> Optional[dict]:
    """Process a single video: check cache → call MLLM → save."""

    # Per-video cache
    cache_path = os.path.join(cache_dir, f"{vid}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    ocr_text = get_cot_text(ocr_df, vid, ("ret", "ocr", "text"))
    asr_text = get_cot_text(asr_df, vid, ("text", "ret"))
    claims_data = load_claims(vid, dataset)
    quad_paths = get_quad_paths(vid, dataset)

    try:
        result = summarizer.summarize(
            vid=vid,
            ocr=ocr_text,
            asr=asr_text,
            claims_data=claims_data,
            quad_paths=quad_paths,
        )
    except Exception as exc:
        print(f"\n[WARN] {vid}: MLLM call failed — {exc}", flush=True)
        return None

    if result is None:
        print(f"\n[WARN] {vid}: JSON parse failed", flush=True)
        return None

    # Save per-video cache
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate event-level video summaries for evidence-aware retrieval."
    )
    ap.add_argument("--dataset",   default="FakeSV", choices=["FakeSV", "FakeTT", "WildFake"])
    ap.add_argument("--limit",     type=int, default=None,
                    help="Process only N videos (for testing)")
    ap.add_argument("--workers",   type=int, default=1,
                    help="Parallel workers (default=1 to avoid rate limits)")
    ap.add_argument("--model",     default=DEFAULT_MODEL,
                    help=f"MLLM model name (default: {DEFAULT_MODEL})")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-process videos that already have cached results")
    ap.add_argument("--split-dir", default=None,
                    help="Only process videos listed in split .txt files under this directory "
                         "(e.g. ExMRD-master/data/FakeSV/vids/). "
                         "Reads *_temporal.txt files; skips videos not in any split.")
    args = ap.parse_args()

    dataset = args.dataset
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://runapi.co/v1")

    if not api_key:
        print("[ERROR] OPENAI_API_KEY not set in environment / .env file")
        sys.exit(1)

    # Paths
    cache_dir   = os.path.join(CACHE_ROOT, dataset, "event_summaries")
    jsonl_path  = os.path.join(OUTPUT_DIR, f"event_summary_{dataset}.jsonl")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print(f"[INFO] Dataset: {dataset}")
    paths = DATA_PATHS[dataset]
    meta_df = load_metadata(dataset)
    ocr_df  = safe_read_jsonl(paths["ocr"],  default_columns=["vid", "ret"])
    asr_df  = safe_read_jsonl(paths["asr"],  default_columns=["vid", "text"])
    print(f"[INFO] Metadata: {len(meta_df)} videos")

    # Build video list — optionally restrict to split files only
    all_vids = meta_df["vid"].tolist()

    if args.split_dir:
        split_dir = args.split_dir
        split_vids: set = set()
        import glob as _glob
        txt_files = _glob.glob(os.path.join(split_dir, "*_temporal.txt"))
        if not txt_files:
            print(f"[WARN] No .txt files found in --split-dir: {split_dir}")
        for txt in sorted(txt_files):
            with open(txt, "r", encoding="utf-8") as _f:
                vids_in_file = [l.strip() for l in _f if l.strip()]
            split_vids.update(vids_in_file)
            print(f"[INFO]   {os.path.basename(txt)}: {len(vids_in_file)} videos")
        all_vids = [v for v in all_vids if str(v) in split_vids]
        # Also include vids that are in split but not in metadata (edge case)
        meta_set = set(str(v) for v in all_vids)
        extra = [v for v in split_vids if v not in meta_set]
        if extra:
            all_vids += extra
        print(f"[INFO] After split filter: {len(all_vids)} videos")

    if args.limit:
        all_vids = all_vids[:args.limit]

    if not args.overwrite:
        done_jsonl  = load_done_vids_from_jsonl(jsonl_path)
        done_cache  = {f.stem for f in pathlib.Path(cache_dir).glob("*.json")} \
                      if pathlib.Path(cache_dir).exists() else set()
        done = done_jsonl | done_cache
        todo = [v for v in all_vids if str(v) not in done]
        print(f"[INFO] Done(JSONL)={len(done_jsonl)} Done(cache)={len(done_cache)} "
              f"Combined={len(done)}  |  To process: {len(todo)}")
    else:
        todo = all_vids
        print(f"[INFO] Overwrite mode — processing all {len(todo)} videos")

    if not todo:
        print("[INFO] Nothing to do. Use --overwrite to reprocess.")
        return

    # Initialize summarizer
    summarizer = VideoEventSummarizer(
        model=args.model,
        base_url=base_url,
        api_key=api_key,
        dataset=dataset,
    )

    # Process
    success = 0; failed = 0
    start_time = time.time()

    if args.workers == 1:
        # Sequential (safer for rate limits)
        for vid in tqdm(todo, desc=f"Event Summary [{dataset}]"):
            result = process_one(vid, summarizer, ocr_df, asr_df, dataset, cache_dir)
            if result:
                append_to_jsonl(result, jsonl_path)
                success += 1
            else:
                failed += 1
    else:
        # Parallel
        def _worker(vid):
            return vid, process_one(vid, summarizer, ocr_df, asr_df, dataset, cache_dir)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, v): v for v in todo}
            for fut in tqdm(as_completed(futures), total=len(todo),
                            desc=f"Event Summary [{dataset}]"):
                vid, result = fut.result()
                if result:
                    append_to_jsonl(result, jsonl_path)
                    success += 1
                else:
                    failed += 1

    elapsed = time.time() - start_time
    print(f"\n[DONE] success={success}  failed={failed}  "
          f"time={elapsed:.1f}s  avg={elapsed/max(success,1):.1f}s/video", flush=True)
    print(f"[OUT]  {jsonl_path}", flush=True)
    print(f"[CACHE] {cache_dir}/  ({success} files)", flush=True)


if __name__ == "__main__":
    main()
