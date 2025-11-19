#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
annotate_reasoning_shifts.py
────────────────────────────
Detects PASS-1 "shift in reasoning" (explicit rethinking) in math results and appends
LLM-adjudicated flags to each row.

• Input: result JSONL files under a results_root (e.g., GRPO-1.5B/step50/step0050_test.jsonl)
  Each line should have either:
    Schema A (flat):    { problem, gold_answer, step, split, sample_idx, output, ... }
    Schema B (two-pass):{ ..., pass1:{ output: "<think>...</think><answer>...</answer>", ... }, ... }

• Output: for each input JSONL, writes sibling file with suffix *_shifted.jsonl
  Each row gains the following PASS-1 fields:
    - shift_cand_hits:  [pattern names matched by regex prefilter]
    - shift_llm:        bool (LLM-confirmed shift)
    - shift_phrase:     short cue phrase selected by LLM (or first regex hit)
    - shift_justification: brief reason from LLM (≤50 chars)
    - shift_decided_by: "llm" | "regex-none"     (regex-none when no cue; LLM not called)
    - shift_model:      deployment name used (e.g., "gpt-4o")

• Calls DeepSeek via Princeton AI-Sandbox (Azure-style). We use a sticky fallback that
  never re-tries /responses after a 404 and defaults to Chat Completions.

Usage:
  python annotate_reasoning_shifts.py \
      results/GRPO-1.5B \
      --split test \
      --seed 123 \
      --always_llm         # (optional) ask LLM even without regex cues
      --overwrite          # (optional) re-annotate rows that already have fields

Notes:
  - We ONLY inspect PASS-1 (the think text if present).
  - We’re conservative: a shift requires an explicit cue (e.g., "wait", "on second thought").
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root on sys.path when executed from src/scripts/annotate
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.annotate.config import load_sandbox_config
from src.annotate.llm_client import build_chat_client
from src.annotate.prefilter import extract_think, find_cue_hits
from src.annotate.prompts import SHIFT_PROMPT

# ─────────────────── AzureOpenAI (Princeton Sandbox) ───────────────────
_sandbox = load_sandbox_config()
SANDBOX_API_KEY  = _sandbox["api_key"]
SANDBOX_ENDPOINT = _sandbox["endpoint"]
SANDBOX_API_VER  = _sandbox["api_version"]
DEPLOYMENT_NAME  = _sandbox["deployment"]   # <<< use the deployed name

_client = build_chat_client(
    endpoint=SANDBOX_ENDPOINT,
    api_key=SANDBOX_API_KEY,
    api_version=SANDBOX_API_VER,
)

_RESPONSES_AVAILABLE = None  # sticky probe (we'll effectively disable for Princeton)

def ds_call(messages, max_tokens: int, temperature: float = 0.0) -> str:
    """
    Prefer Azure Chat Completions on the Princeton sandbox.
    Try /responses at most once; after 404, stick to chat.completions.
    """
    global _RESPONSES_AVAILABLE

    # Hard-disable Responses for known Princeton host (avoids the 404 spam entirely)
    if _RESPONSES_AVAILABLE is None:
        _RESPONSES_AVAILABLE = (
            ("api-ai-sandbox.princeton.edu" not in SANDBOX_ENDPOINT)
        )

    if _RESPONSES_AVAILABLE:
        try:
            # Not generally available on Princeton; kept for portability.
            resp = _client.responses.create(
                model=DEPLOYMENT_NAME,
                input=[{"role": "user", "content": messages[-1]["content"]}],
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            try:
                return resp.output_text
            except Exception:
                chunks = []
                for out in getattr(resp, "output", []):
                    for blk in getattr(out, "content", []):
                        if getattr(blk, "type", "") == "output_text":
                            chunks.append(getattr(blk, "text", ""))
                return "".join(chunks).strip()
        except Exception as e:
            logging.info("Responses API not available (%s). Falling back to chat.completions.", e)
            _RESPONSES_AVAILABLE = False

    # Stable path on Princeton sandbox
    resp = _client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
    )
    return resp.choices[0].message.content.strip()

# ────────────────────────── Helpers ──────────────────────────
def extract_pass1_text(rec: Dict[str, Any]) -> Optional[str]:
    """Get the pass-1 full text; prefer the <think> block if present."""
    txt = None
    if isinstance(rec.get("pass1"), dict):
        txt = rec["pass1"].get("output") or rec["pass1"].get("full_text") or rec["pass1"].get("raw")
    txt = txt or rec.get("output") or rec.get("full_text") or rec.get("raw")
    if not txt:
        return None
    think = extract_think(txt)
    return think if think is not None else txt.strip()

def llm_decide_shift(problem: str, think_text: str, model_name: str) -> Tuple[bool, str, str]:
    """Call LLM once; parse a strict JSON object."""
    msg = SHIFT_PROMPT.format(problem=problem[:2000], think=think_text[:4000])
    raw = ds_call(
        messages=[
            {"role": "system", "content": "You are concise and conservative."},
            {"role": "user",   "content": msg},
        ],
        max_tokens=140,
        temperature=0.0,
    )
    # Strip code fences if any
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:-1]).strip()
    try:
        obj = json.loads(raw)
    except Exception:
        logging.warning("Non-JSON LLM output; treating as NO. Raw: %s", raw[:200])
        return False, "—", "parse-fail"
    shift = str(obj.get("shift", "NO")).upper() == "YES"
    cue = str(obj.get("cue", "—")).strip() or "—"
    just = str(obj.get("justification", "")).strip()[:50]
    return shift, cue, just or ("explicit cue" if shift else "no explicit cue")

def should_skip(rec: Dict[str, Any], overwrite: bool) -> bool:
    if overwrite:
        return False
    # If we already annotated, skip
    return all(k in rec for k in ("shift_llm", "shift_phrase", "shift_justification"))

def scan_files(root: Path, split: Optional[str]) -> List[Path]:
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if not fn.endswith(".jsonl"):
                continue
            # Avoid re-processing already annotated outputs
            if fn.endswith("_shifted.jsonl"):
                continue
            if split and split not in fn:
                continue
            files.append(Path(dp) / fn)
    files.sort()
    return files

# ────────────────────────── Main ──────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root", help="Directory containing step*/.../*.jsonl")
    ap.add_argument("--split", default="test", help="Substring filter for filenames (e.g. 'test')")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--always_llm", action="store_true",
                    help="Call LLM even when no regex cue hits (defaults to NO for speed).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-annotate rows even if fields already present.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    random.seed(args.seed)

    root = Path(args.results_root)
    files = scan_files(root, args.split)
    if not files:
        logging.error("No JSONL files found under %s (split filter: %r).", root, args.split)
        sys.exit(1)

    logging.info("Found %d files to annotate.", len(files))

    for path in files:
        logging.info("Annotating: %s", path)
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue

        # Create randomized processing order (but keep original order in output)
        order = list(range(len(rows)))
        random.shuffle(order)

        # Work on a copy of rows; write annotations back into 'rows'
        for idx in order:
            rec = rows[idx]
            if should_skip(rec, args.overwrite):
                continue

            think_text = extract_pass1_text(rec) or ""
            problem = rec.get("problem", "") or rec.get("clue", "")

            cand_hits = find_cue_hits(think_text)
            shift_llm = False
            shift_phrase = "—"
            shift_just = "no explicit cue"
            decided_by = "regex-none"

            # If we see candidate cues OR user insists, ask the LLM to confirm.
            if cand_hits or args.always_llm:
                shift_llm, shift_phrase, shift_just = llm_decide_shift(problem, think_text, DEPLOYMENT_NAME)
                decided_by = "llm"

            rec["shift_cand_hits"]     = cand_hits
            rec["shift_llm"]           = bool(shift_llm)
            rec["shift_phrase"]        = shift_phrase
            rec["shift_justification"] = shift_just
            rec["shift_decided_by"]    = decided_by
            rec["shift_model"]         = DEPLOYMENT_NAME

        out = path.with_name(path.stem + "_shifted.jsonl")
        with out.open("w", encoding="utf-8") as w:
            for r in rows:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
        logging.info("Wrote %s", out)

if __name__ == "__main__":
    main()
