"""
Phase 0.2 baseline lock (read-only). The repo is NOT a git repository, so the
code snapshot is a sha256 manifest of source/config files instead of a commit SHA.

Writes (experiments/audit_20261015/):
    code_manifest.csv   path, bytes, sha256   (*.py, *.yaml under repo, excl. caches/logs)
    env.txt             python / torch / cuda / GPU / pip freeze
    split_manifest.csv  dataset, split, n, n_fake, n_real, n_events, sha256(vid list)
    cache_manifest.csv  cache dir, n_files, total_bytes, sha256(name+size listing),
                        sha256 over all file contents
    baseline_lock.json  summary of the above
"""

import json
import platform
import subprocess
import sys

from _common import (CACHE_ROOT, FEATURE_CACHE, OUT_DIR, ROOT, SPLITS, load_split,
                     sha256_file, sha256_text, split_hash, write_csv)

SKIP_DIRS = {"data", "logs", "__pycache__", "experiments", "outputs", "multiseed",
             "multiseed_new_split", "articles"}


def code_manifest():
    rows = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.suffix not in (".py", ".yaml", ".yml"):
            continue
        rel = p.relative_to(ROOT)
        top = rel.parts[0]
        if top in SKIP_DIRS or top.startswith(("ablation_", "hparam_")):
            continue
        rows.append({"path": str(rel), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    write_csv(OUT_DIR / "code_manifest.csv", rows)
    return rows


def env_info():
    lines = [f"python: {sys.version}", f"platform: {platform.platform()}"]
    try:
        import torch
        lines += [f"torch: {torch.__version__}", f"cuda available: {torch.cuda.is_available()}",
                  f"cuda: {torch.version.cuda}", f"cudnn: {torch.backends.cudnn.version()}"]
        if torch.cuda.is_available():
            lines.append(f"gpu: {torch.cuda.get_device_name(0)}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"torch import failed: {e}")
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                text=True, timeout=120).stdout
    except Exception as e:  # noqa: BLE001
        freeze = f"pip freeze failed: {e}"
    text = "\n".join(lines) + "\n\n# pip freeze\n" + freeze
    (OUT_DIR / "env.txt").write_text(text, encoding="utf-8")
    return lines


def split_manifest():
    rows = []
    for ds in ("FakeSV", "FakeTT"):
        for sp in SPLITS:
            df = load_split(ds, sp)
            rows.append({"dataset": ds, "split": sp, "n": len(df),
                         "n_fake": int((df["label"] == 1).sum()),
                         "n_real": int((df["label"] == 0).sum()),
                         "n_events": int(df["event"].nunique()),
                         "vid_list_sha256": split_hash(df)})
    write_csv(OUT_DIR / "split_manifest.csv", rows)
    return rows


def cache_manifest():
    rows = []
    for ds, cache in FEATURE_CACHE.items():
        fdir = CACHE_ROOT / cache / "features"
        files = sorted(fdir.glob("feat_*.pt"))
        listing = "\n".join(f"{f.name}\t{f.stat().st_size}" for f in files)
        content = sha256_text("\n".join(f"{f.name}:{sha256_file(f)}" for f in files))
        rows.append({"dataset": ds, "cache": cache, "n_files": len(files),
                     "total_bytes": sum(f.stat().st_size for f in files),
                     "listing_sha256": sha256_text(listing), "content_sha256": content})
    write_csv(OUT_DIR / "cache_manifest.csv", rows)
    return rows


def main():
    code = code_manifest()
    env = env_info()
    splits = split_manifest()
    caches = cache_manifest()
    lock = {"git": "NOT A GIT REPOSITORY - code identified by code_manifest.csv",
            "code_files": len(code),
            "code_manifest_sha256": sha256_file(OUT_DIR / "code_manifest.csv"),
            "env": env, "splits": splits, "caches": caches}
    (OUT_DIR / "baseline_lock.json").write_text(json.dumps(lock, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    print(json.dumps({k: v for k, v in lock.items() if k != "env"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
