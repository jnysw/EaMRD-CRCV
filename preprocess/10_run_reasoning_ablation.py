"""
Step 10: Reasoning Ablation Runs

Variants:
  woEGR  — w/o Evidence-Guided Reasoning
           Run reasoning WITHOUT showing evidence to the LLM.
           Output: data/cache/FakeSV/reasoning_v2_woEGR/

  woCMC  — w/o Cross-Modal Conflict analysis
           Run reasoning WITHOUT cross-modal consistency summary.
           Output: data/cache/FakeSV/reasoning_v2_woCMC/

All other settings identical to 07_run_reasoning_v2.py (same prompt, same model).

Usage:
    python preprocess/10_run_reasoning_ablation.py --ablation woEGR
    python preprocess/10_run_reasoning_ablation.py --ablation woCMC
    python preprocess/10_run_reasoning_ablation.py --ablation woEGR --limit 5
"""

import argparse, io, json, os, re, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import requests
import urllib3
from dotenv import load_dotenv
from tqdm import tqdm

# Re-use helpers from 07_run_reasoning_v2 (import as module)
import importlib.util, pathlib
_spec = importlib.util.spec_from_file_location(
    "reasoning_v2",
    pathlib.Path(__file__).parent / "07_run_reasoning_v2.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Pull needed symbols
SYSTEM_PROMPT_TEMPLATE  = _mod.SYSTEM_PROMPT_TEMPLATE
USER_PROMPT_TEMPLATE    = _mod.USER_PROMPT_TEMPLATE
LANG_INSTRUCTIONS       = _mod.LANG_INSTRUCTIONS
LANG_REMINDERS          = _mod.LANG_REMINDERS
DATA_PATHS              = _mod.DATA_PATHS
CACHE_ROOT              = _mod.CACHE_ROOT
format_claims_text      = _mod.format_claims_text
build_consistency_summary = _mod.build_consistency_summary
build_evidence_text_v2  = _mod.build_evidence_text_v2
call_api                = _mod.call_api
parse_reasoning         = _mod.parse_reasoning
_make_session           = _mod._make_session
get_text                = _mod.get_text
load_json               = _mod.load_json

from utils.cot_common import safe_read_jsonl
from utils.quad_utils  import get_quad_paths, build_image_content

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv(override=True)

# ─── Ablation-specific overrides ─────────────────────────────────────────────

ABLATION_SUFFIX = {
    "woEGR": "reasoning_v2_woEGR",   # evidence replaced with placeholder
    "woCMC": "reasoning_v2_woCMC",   # cross-modal consistency replaced with placeholder
}

NO_EVIDENCE_TEXT = (
    "[ABLATION: evidence not provided — reason using only video content and world knowledge]"
)

NO_CMC_TEXT = (
    "[ABLATION: cross-modal consistency analysis not provided — focus on evidence and content]"
)


def build_user_msg(sys_prompt: str, ablation: str, title: str,
                   claims_data: dict, graph: dict, ev_record,
                   ocr: str, asr: str, dataset: str) -> tuple[str, dict]:
    """Build user message with ablation-specific modifications."""

    title_claims  = format_claims_text(claims_data.get("title_claims"))
    ocr_claims    = format_claims_text(claims_data.get("ocr_claims"))
    asr_claims    = format_claims_text(claims_data.get("asr_claims"))
    vision_claims = format_claims_text(claims_data.get("vision_claims"))

    # Evidence
    evidence_text, ev_availability = build_evidence_text_v2(ev_record)
    if ablation == "woEGR":
        evidence_text = NO_EVIDENCE_TEXT   # hide evidence from LLM

    # Cross-modal consistency
    consistency_summary = build_consistency_summary(graph)
    if ablation == "woCMC":
        consistency_summary = NO_CMC_TEXT  # hide conflict analysis from LLM

    user_msg = (
        sys_prompt + "\n\n"
        + USER_PROMPT_TEMPLATE.format(
            title=title or "(no title)",
            title_claims_text=title_claims,
            ocr_claims_text=ocr_claims,
            asr_claims_text=asr_claims,
            vision_claims_text=vision_claims,
            consistency_summary=consistency_summary,
            evidence_text=evidence_text,
            ocr_text=ocr,
            asr_text=asr,
            lang_reminder=LANG_REMINDERS.get(dataset, ""),
        )
    )
    return user_msg, ev_availability


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", choices=["woEGR", "woCMC"], required=True)
    parser.add_argument("--dataset",  choices=["FakeSV", "FakeTT"], default="FakeSV")
    parser.add_argument("--limit",    type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset    = args.dataset
    claims_dir = os.path.join(CACHE_ROOT, dataset, "claims")
    graph_dir  = os.path.join(CACHE_ROOT, dataset, "claim_graphs")
    out_dir    = os.path.join(CACHE_ROOT, dataset, ABLATION_SUFFIX[args.ablation])
    os.makedirs(out_dir, exist_ok=True)

    # Load filtered evidence (we still use it for ev_availability tracking;
    # only woEGR hides it from the prompt)
    ev_path = DATA_PATHS[dataset]["verified_evidence"]
    evidence_map: dict = {}
    if os.path.exists(ev_path):
        for line in open(ev_path, encoding="utf-8"):
            try:
                obj = json.loads(line)
                evidence_map[str(obj["vid"])] = obj
            except:
                pass
    print(f"Loaded {len(evidence_map)} evidence records", flush=True)

    # Build todo list (same split target as full pipeline)
    split_target = os.path.join(CACHE_ROOT, dataset, "Split_target.txt")
    if os.path.exists(split_target):
        todo_raw = [l.strip() for l in open(split_target, encoding="utf-8") if l.strip()]
    else:
        todo_raw = [os.path.splitext(f)[0] for f in os.listdir(claims_dir) if f.endswith(".json")]

    done = set() if args.overwrite else {
        os.path.splitext(f)[0] for f in os.listdir(out_dir) if f.endswith(".json")
    }
    todo = [v for v in todo_raw if v not in done
            and os.path.exists(os.path.join(claims_dir, f"{v}.json"))]
    if args.limit:
        todo = todo[:args.limit]
    print(f"Ablation: {args.ablation}  Todo: {len(todo)}  Done: {len(done)}", flush=True)

    if not todo:
        print("All done.")
        return

    # Load metadata / OCR / ASR
    meta_df = pd.read_json(DATA_PATHS[dataset]["metadata"], lines=True, dtype={"video_id": str})
    meta_df["video_id"] = meta_df["video_id"].astype(str)
    meta_map = meta_df.set_index("video_id").to_dict(orient="index")
    ocr_df   = safe_read_jsonl(DATA_PATHS[dataset]["ocr"])
    asr_df   = safe_read_jsonl(DATA_PATHS[dataset]["asr"])

    api_key  = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://runapi.co/v1").rstrip("/") + "/chat/completions"
    model    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not set"); sys.exit(1)

    sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        lang_instruction=LANG_INSTRUCTIONS.get(dataset, "Use the video's language.")
    )
    session = _make_session()

    for vid in tqdm(todo, desc=f"Reasoning [{args.ablation}]"):
        claims_data = load_json(os.path.join(claims_dir, f"{vid}.json"))
        graph       = load_json(os.path.join(graph_dir,  f"{vid}.json"))
        ev_record   = evidence_map.get(vid)

        if not claims_data:
            continue

        row   = meta_map.get(vid, {})
        title = str(row.get("keywords", row.get("description", row.get("title", "")))).strip()[:256]
        ocr   = get_text(ocr_df, vid, "description", "ocr_text", "text", "ret") or "(no OCR)"
        asr   = get_text(asr_df, vid, "transcript", "text", "ret") or "(no transcript)"

        user_msg, ev_availability = build_user_msg(
            sys_prompt, args.ablation, title, claims_data, graph,
            ev_record, ocr, asr, dataset
        )

        quad_paths  = get_quad_paths(vid, dataset)
        image_parts = build_image_content(quad_paths)
        content     = [{"type": "text", "text": user_msg}] + image_parts

        raw    = call_api(session, base_url, api_key, model, content)
        result = parse_reasoning(raw, vid, ev_availability)
        result["ablation"] = args.ablation   # tag for traceability

        with open(os.path.join(out_dir, f"{vid}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    try:
        session.close()
    except:
        pass
    print(f"\n[DONE] {args.ablation} results saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
