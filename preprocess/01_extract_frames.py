"""
Step 1: Extract keyframes from videos using ffmpeg (16 frames per video).

已完成状态: FakeSV 的帧已经在 fakesv-qwen3/frames_16/ 下完成提取。
本脚本主要用于 FakeTT 补充提取，或重新提取时使用。

Usage:
    python preprocess/01_extract_frames.py --dataset FakeSV
    python preprocess/01_extract_frames.py --dataset FakeTT
"""

import argparse
import glob
import logging
import os
import shutil
import subprocess

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── 路径配置（与 configs/paths.yaml 保持一致）────────────────────────────────
DATASET_VIDEO_ROOTS = {
    "FakeSV": r"F:\Bishe_workspace\fakesv-qwen3\videos",
    "FakeTT": r"F:\Bishe_workspace\FakeTT\videos",
    "WildFake": r"F:\Bishe_workspace\WildFake\HMVD",
}
DATASET_FRAME_ROOTS = {
    "FakeSV": r"F:\Bishe_workspace\fakesv-qwen3\frames_16",
    "FakeTT": r"F:\Bishe_workspace\FakeTT\frames_16",
    "WildFake": r"F:\Bishe_workspace\WildFake\frames_16",
}
NUM_FRAMES = 16


def get_video_duration(video_path: str) -> float | None:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def extract_frames(video_path: str, output_dir: str, num_frames: int = 16) -> bool:
    duration = get_video_duration(video_path)
    if not duration or duration <= 0:
        logging.warning(f"Skipping invalid video: {video_path}")
        return False

    vid = os.path.splitext(os.path.basename(video_path))[0]
    out = os.path.join(output_dir, vid)

    if os.path.exists(out):
        existing = glob.glob(os.path.join(out, "frame_*.jpg"))
        if len(existing) == num_frames:
            return True
        shutil.rmtree(out)

    os.makedirs(out, exist_ok=True)
    interval = duration / num_frames
    for i in range(num_frames):
        ts = i * interval
        outfile = os.path.join(out, f"frame_{i:03d}.jpg")
        try:
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-ss", str(ts), "-i", video_path,
                 "-frames:v", "1", "-q:v", "2", outfile],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as e:
            logging.error(f"ffmpeg error at frame {i} for {vid}: {e.stderr}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["FakeSV", "FakeTT", "WildFake"], default="FakeTT")
    parser.add_argument("--vids-file", type=str, default=None,
                        help="Optional JSONL file with video_id list to limit extraction (e.g. data_3k.jsonl)")
    args = parser.parse_args()

    video_root = DATASET_VIDEO_ROOTS[args.dataset]
    frame_root = DATASET_FRAME_ROOTS[args.dataset]

    if not os.path.exists(video_root):
        logging.error(f"Video root not found: {video_root}")
        return

    os.makedirs(frame_root, exist_ok=True)

    if args.vids_file:
        import json
        with open(args.vids_file, encoding="utf-8") as f:
            vid_set = {json.loads(l)["video_id"] for l in f if l.strip()}
        videos = [os.path.join(video_root, v + ".mp4") for v in vid_set
                  if os.path.exists(os.path.join(video_root, v + ".mp4"))]
        logging.info(f"Using vids-file: {args.vids_file} → {len(videos)} videos")
    else:
        videos = glob.glob(os.path.join(video_root, "*.mp4"))
    logging.info(f"Found {len(videos)} videos in {video_root}")

    ok = sum(extract_frames(v, frame_root, NUM_FRAMES) for v in tqdm(videos, desc="Extracting frames"))
    logging.info(f"Done: {ok}/{len(videos)} videos processed.")


if __name__ == "__main__":
    main()
