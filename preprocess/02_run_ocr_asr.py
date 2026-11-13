"""
Step 2: Run OCR (PaddleOCR) and ASR (Whisper) on videos.

已完成状态:
  - FakeSV OCR → F:\Bishe_workspace\fakesv-qwen3\ocr.jsonl
  - FakeSV ASR → F:\Bishe_workspace\fakesv-qwen3\transcript.jsonl
  - FakeTT OCR → F:\Bishe_workspace\FakeTT\ocr.jsonl  (English, lang='en')
  - FakeTT ASR → F:\Bishe_workspace\FakeTT\transcript.jsonl (English, language='en')

本脚本在未完成的情况下补充运行，或用于新数据集。

Usage:
    python preprocess/02_run_ocr_asr.py --dataset FakeSV --mode ocr
    python preprocess/02_run_ocr_asr.py --dataset FakeSV --mode asr
    python preprocess/02_run_ocr_asr.py --dataset FakeTT --mode ocr
    python preprocess/02_run_ocr_asr.py --dataset FakeTT --mode asr
    python preprocess/02_run_ocr_asr.py --dataset FakeTT --mode quads
"""

import argparse
import json
import os

import pandas as pd
from tqdm import tqdm

# 文件路径配置
COMPLETED_FILES = {
    "FakeSV": {
        "ocr": r"F:\Bishe_workspace\fakesv-qwen3\ocr.jsonl",
        "asr": r"F:\Bishe_workspace\fakesv-qwen3\transcript.jsonl",
        "frames": r"F:\Bishe_workspace\fakesv-qwen3\frames_16",
        "videos": r"F:\Bishe_workspace\fakesv-qwen3\videos",
        "quads": r"F:\Bishe_workspace\fakesv-qwen3\quads_4",
    },
    "FakeTT": {
        "ocr": r"F:\Bishe_workspace\FakeTT\ocr.jsonl",
        "asr": r"F:\Bishe_workspace\FakeTT\transcript.jsonl",
        "frames": r"F:\Bishe_workspace\FakeTT\frames_16",
        "videos": r"F:\Bishe_workspace\FakeTT\videos",
        "quads": r"F:\Bishe_workspace\FakeTT\quads_4",
    },
    "WildFake": {
        "ocr": r"F:\Bishe_workspace\WildFake\ocr.jsonl",
        "asr": r"F:\Bishe_workspace\WildFake\transcript.jsonl",
        "frames": r"F:\Bishe_workspace\WildFake\frames_16",
        "videos": r"F:\Bishe_workspace\WildFake\HMVD",
        "quads": r"F:\Bishe_workspace\WildFake\quads_4",
    },
}


def _load_vid_filter(vids_file: str) -> set:
    """Load a set of video IDs from a JSONL file (field: video_id). Returns None if not specified."""
    if not vids_file:
        return None
    import json as _json
    vids = set()
    with open(vids_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = _json.loads(line)
                vid = r.get("video_id") or r.get("vid")
                if vid:
                    vids.add(str(vid))
            except Exception:
                pass
    print(f"[vid_filter] Loaded {len(vids)} video IDs from {vids_file}")
    return vids


def check_status():
    print("\n=== OCR / ASR 完成状态检查 ===")
    for dataset, files in COMPLETED_FILES.items():
        for key, path in files.items():
            if key in ("videos", "quads"):
                continue
            exists = os.path.exists(path)
            count = "?"
            if exists and path.endswith(".jsonl"):
                try:
                    df = pd.read_json(path, lines=True)
                    count = str(len(df))
                except Exception:
                    count = "read_error"
            elif exists and os.path.isdir(path):
                count = f"{sum(1 for _ in os.scandir(path))} dirs"
            print(f"  [{dataset}][{key}] {'✓' if exists else '✗'}  {path}  ({count}条)")


def run_ocr_fakesv(output_path: str):
    """Run PaddleOCR on FakeSV frames."""
    from paddleocr import PaddleOCR

    frames_root = COMPLETED_FILES["FakeSV"]["frames"]
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True)

    results = []
    vid_dirs = [d for d in os.listdir(frames_root) if os.path.isdir(os.path.join(frames_root, d))]
    for vid in tqdm(vid_dirs, desc="OCR"):
        frame_dir = os.path.join(frames_root, vid)
        frames = sorted(os.listdir(frame_dir))
        texts = []
        for fname in frames[:4]:  # use first 4 frames for OCR efficiency
            fpath = os.path.join(frame_dir, fname)
            try:
                res = ocr.ocr(fpath, cls=True)
                if res and res[0]:
                    texts.extend([line[1][0] for line in res[0] if line[1]])
            except Exception:
                pass
        results.append({"vid": vid, "ocr_text": " ".join(texts)})

    df = pd.DataFrame(results)
    df.to_json(output_path, lines=True, orient="records", force_ascii=False)
    print(f"OCR done: {len(df)} videos → {output_path}")


def run_asr_fakesv(output_path: str):
    """Run Whisper ASR on FakeSV audio."""
    import whisper

    audio_root = r"F:\Bishe_workspace\fakesv-qwen3"
    model = whisper.load_model("large-v3")

    wav_files = [f for f in os.listdir(audio_root) if f.endswith(".wav")]
    results = []
    for wav in tqdm(wav_files, desc="ASR"):
        vid = os.path.splitext(wav)[0]
        wav_path = os.path.join(audio_root, wav)
        try:
            result = model.transcribe(wav_path, language="zh")
            transcript = result.get("text", "")
        except Exception as e:
            transcript = ""
        results.append({"vid": vid, "transcript": transcript})

    df = pd.DataFrame(results)
    df.to_json(output_path, lines=True, orient="records", force_ascii=False)
    print(f"ASR done: {len(df)} videos → {output_path}")


def run_ocr_fakett(output_path: str, frames_root: str, vid_filter: set = None):
    """Run PaddleOCR (English) on FakeTT/WildFake frames.
    vid_filter: if provided, only process videos whose ID is in this set.
    """
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=True, lang="en")

    vid_dirs = sorted([d for d in os.listdir(frames_root) if os.path.isdir(os.path.join(frames_root, d))])
    if vid_filter:
        vid_dirs = [v for v in vid_dirs if v in vid_filter]
        print(f"vid_filter applied: {len(vid_dirs)} videos to process")
    # Resume: skip already-done vids
    done = set()
    if os.path.exists(output_path):
        existing = pd.read_json(output_path, lines=True, dtype={"vid": str})
        done = set(existing["vid"].tolist())
        print(f"Resuming OCR: {len(done)} already done, {len(vid_dirs) - len(done)} remaining")

    results = []
    for vid in tqdm(vid_dirs, desc="OCR"):
        if vid in done:
            continue
        frame_dir = os.path.join(frames_root, vid)
        frames = sorted(os.listdir(frame_dir))
        texts = []
        for fname in frames[:4]:  # first 4 frames sufficient for OCR
            fpath = os.path.join(frame_dir, fname)
            try:
                res = ocr.ocr(fpath, cls=True)
                if res and res[0]:
                    texts.extend([line[1][0] for line in res[0] if line[1]])
            except Exception:
                pass
        results.append({"vid": vid, "ocr_text": " ".join(texts)})

    # Append to file
    mode = "a" if done else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"OCR done: {len(results)} new videos → {output_path}")


def run_asr_fakett(output_path: str, videos_root: str, model_name: str = "medium", vid_filter: set = None):
    """Run faster-whisper (English) ASR on .mp4 videos.
    vid_filter: if provided, only process videos whose ID is in this set.
    """
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    use_faster = False

    try:
        from faster_whisper import WhisperModel
        print(f"Loading faster-whisper model: {model_name} on {device} ({compute_type})")
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        use_faster = True
        print("  faster-whisper loaded OK")
    except Exception as e:
        print(f"  faster-whisper failed ({e}), falling back to openai-whisper...")
        try:
            import whisper as _whisper
            print(f"Loading openai-whisper model: {model_name}")
            model = _whisper.load_model(model_name)
            print("  openai-whisper loaded OK")
        except Exception as e2:
            print(f"ERROR: both ASR backends failed: {e2}")
            return

    mp4_files = sorted([f for f in os.listdir(videos_root) if f.endswith(".mp4")])
    if vid_filter:
        mp4_files = [f for f in mp4_files if os.path.splitext(f)[0] in vid_filter]
        print(f"vid_filter applied: {len(mp4_files)} videos to process")
    # Resume: skip already-done vids
    done = set()
    if os.path.exists(output_path):
        try:
            existing = pd.read_json(output_path, lines=True, dtype={"vid": str})
            done = set(existing["vid"].tolist())
            print(f"Resuming ASR: {len(done)} already done, {len(mp4_files) - len(done)} remaining")
        except Exception:
            pass

    n_new = 0
    write_mode = "a" if done else "w"
    with open(output_path, write_mode, encoding="utf-8") as out_f:
        for fname in tqdm(mp4_files, desc="ASR"):
            vid = os.path.splitext(fname)[0]
            if vid in done:
                continue
            video_path = os.path.join(videos_root, fname)
            try:
                if use_faster:
                    segments, _ = model.transcribe(video_path, language="en", beam_size=5)
                    transcript = " ".join(seg.text for seg in segments).strip()
                else:
                    result = model.transcribe(video_path, language="en")
                    transcript = result.get("text", "")
            except Exception as e:
                print(f"  [WARN] ASR failed for {vid}: {e}")
                transcript = ""
            out_f.write(json.dumps({"vid": vid, "transcript": transcript}, ensure_ascii=False) + "\n")
            out_f.flush()
            n_new += 1

    print(f"ASR done: {n_new} new videos → {output_path}")


def build_quads(dataset: str):
    """Build 2×2 quad images from 16 extracted frames (needed for MLLM Steps 03/06)."""
    import glob as globmod
    from PIL import Image

    paths = COMPLETED_FILES[dataset]
    frames_root = paths["frames"]
    quads_root  = paths["quads"]

    if not os.path.exists(frames_root):
        print(f"ERROR: frames dir not found: {frames_root}")
        return

    os.makedirs(quads_root, exist_ok=True)
    vid_dirs = sorted([d for d in os.listdir(frames_root)
                       if os.path.isdir(os.path.join(frames_root, d))])

    done = 0
    for vid in tqdm(vid_dirs, desc=f"Build quads [{dataset}]"):
        # Check if all 4 quads already exist
        quad_paths = [os.path.join(quads_root, f"{vid}_quad_{i}.jpg") for i in range(4)]
        if all(os.path.exists(p) for p in quad_paths):
            done += 1
            continue

        frame_files = sorted(globmod.glob(os.path.join(frames_root, vid, "frame_*.jpg")))
        if len(frame_files) < 4:
            continue

        # Build 4 quads: each quad = 4 consecutive frames in 2×2 grid
        for q_idx in range(4):
            q_frames = frame_files[q_idx * 4: q_idx * 4 + 4]
            if len(q_frames) < 4:
                break
            imgs = [Image.open(p).convert("RGB") for p in q_frames]
            w, h = imgs[0].size
            grid = Image.new("RGB", (w * 2, h * 2))
            for i, img in enumerate(imgs):
                grid.paste(img, ((i % 2) * w, (i // 2) * h))
            grid.save(quad_paths[q_idx], quality=90)

    print(f"Quads done: {len(vid_dirs)} vids → {quads_root}  (skipped {done} already done)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["FakeSV", "FakeTT", "WildFake"], default="FakeTT")
    parser.add_argument("--mode", choices=["ocr", "asr", "check", "quads"], default="check")
    parser.add_argument("--whisper-model", default="medium",
                        help="Whisper model size: tiny/base/small/medium/large-v3 (default: medium)")
    parser.add_argument("--vids-file", type=str, default=None,
                        help="Optional JSONL with video_id list to limit processing (e.g. data_3k.jsonl)")
    args = parser.parse_args()

    if args.mode == "check":
        check_status()
        return

    paths = COMPLETED_FILES[args.dataset]

    if args.mode == "quads":
        build_quads(args.dataset)
        return

    if args.mode == "ocr":
        if os.path.exists(paths["ocr"]):
            df = pd.read_json(paths["ocr"], lines=True)
            print(f"OCR already at {paths['ocr']}: {len(df)} rows")
            if len(df) >= 1900:
                return
            print("  → incomplete, resuming...")
        vid_filter = _load_vid_filter(args.vids_file)
        if args.dataset == "FakeSV":
            run_ocr_fakesv(paths["ocr"])
        else:
            run_ocr_fakett(paths["ocr"], paths["frames"], vid_filter=vid_filter)

    elif args.mode == "asr":
        if os.path.exists(paths["asr"]):
            df = pd.read_json(paths["asr"], lines=True)
            print(f"ASR already at {paths['asr']}: {len(df)} rows")
            if len(df) >= 1900:
                return
            print("  → incomplete, resuming...")
        vid_filter = _load_vid_filter(args.vids_file)
        if args.dataset == "FakeSV":
            run_asr_fakesv(paths["asr"])
        else:
            run_asr_fakett(paths["asr"], paths["videos"], model_name=args.whisper_model, vid_filter=vid_filter)


if __name__ == "__main__":
    main()
