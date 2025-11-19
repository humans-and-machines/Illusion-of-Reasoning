#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Annotate pass-1 'shift in reasoning' for inference JSONL outputs.

Policy (strict / rare):
- Only annotate TRUE when there is BOTH:
  (A) an explicit cue inside <think> like “wait”, “hold on”, “on second thought”,
      “actually,” “scratch that,” “I misread…”, “re-check”, “doesn’t fit/match”, etc.
  AND
  (B) a material revision: the author rejects/corrects an earlier idea
      (new candidate, corrected derivation, contradiction resolved).
- We first do a lexical prefilter. If it hits, we ask the LLM to verify.
- If the LLM is uncertain or returns invalid JSON, default to FALSE.
- Idempotent: lines already containing 'shift_in_reasoning_v1' in pass1 are skipped.
- Processes candidates in random order; rewrites files atomically.

Usage:
  python annotate_shift_pass1.py /path/to/results_root --split test
"""

import os
import re
import json
import time
import sys
import argparse
import random
import logging
import hashlib
from pathlib import Path
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
from json import JSONDecodeError
from dataclasses import dataclass

from src.annotate.config import load_azure_config
from src.annotate.llm_client import build_preferred_client
from src.annotate.prefilter import (
    extract_think as _extract_think,
    find_shift_cues as _find_shift_cues,
)
from src.annotate.prompts import (
    SHIFT_JUDGE_SYSTEM_PROMPT as PROMPT_SYSTEM,
    SHIFT_JUDGE_USER_TEMPLATE as PROMPT_USER_TEMPLATE,
)

try:
    from openai import OpenAIError
except ImportError:  # pragma: no cover - optional dependency
    class OpenAIError(Exception):  # type: ignore
        """Fallback OpenAI error when openai package is absent."""

# Ensure repo root is on sys.path when executed from src/scripts/annotate
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ───────────────────── Azure OpenAI config (env or CLI) ─────────────────────
# Preferred: v1 base-URL style (Responses API). Fallback: legacy AzureOpenAI.
_cfg = load_azure_config()
DEFAULT_ENDPOINT = _cfg["endpoint"]
DEFAULT_DEPLOYMENT = _cfg["deployment"]  # deployment name
DEFAULT_API_VERSION = _cfg["api_version"]
DEFAULT_USE_V1 = int(_cfg["use_v1"])

# ───────────────────── Helper utilities ─────────────────────────────
@contextmanager
def timed(label: str):
    """Context manager to time a block and log at DEBUG."""
    start = time.time()
    try:
        yield
    finally:
        logging.debug("[TIMING] %s: %.2fs", label, time.time() - start)

MAX_FIELD_LEN = 4096
def _clamp(txt: str, lim: int = MAX_FIELD_LEN) -> str:
    txt = txt or ""
    return txt[-lim:] if len(txt) > lim else txt

def _dump_filtered(prompt: str):
    ts = time.strftime("%Y%m%d_%H%M%S")
    dg = hashlib.md5(prompt.encode()).hexdigest()[:8]
    fn = f"filtered_prompt_{ts}_{dg}.txt"
    with open(fn, "w", encoding="utf-8") as f:
        f.write(prompt)
    logging.warning("LLM filtered/failed; saved prompt to %s", fn)

def _json_from_text(s: str) -> Optional[Dict[str, Any]]:
    """Extract the first top-level JSON object from arbitrary text."""
    s = (s or "").strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except JSONDecodeError:
            return None
    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            return json.loads(s[i:j+1])
        except JSONDecodeError:
            return None
    return None

# ───────────────────── Client factory (v1 preferred) ─────────────────────
_CLIENT_STATE = {"client": None, "uses_v1": False}

def _client_lazy(client_cfg: Dict[str, Any]) -> None:
    """Create and cache an Azure OpenAI client (Responses preferred)."""
    if _CLIENT_STATE["client"] is not None:
        return

    endpoint = client_cfg["endpoint"].rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing AZURE_OPENAI_API_KEY (set in env or .env).")

    client, uses_v1 = build_preferred_client(
        endpoint=endpoint,
        api_key=api_key,
        api_version=client_cfg["api_version"],
        use_v1=bool(client_cfg["use_v1"]),
    )
    _CLIENT_STATE["client"] = client
    _CLIENT_STATE["uses_v1"] = uses_v1

# ───────────────────── LLM call ─────────────────────────────
def llm_judge_shift(
    client_cfg: Dict[str, Any],
    deployment: str,
    example: Dict[str, Any],
) -> Dict[str, Any]:
    """Call the LLM judge for a single example."""
    _client_lazy(client_cfg)  # ensure client exists
    problem = example["problem"]
    think = example["think"]
    cue_names = example["cues"]
    pos = example["pos"]
    user_content = PROMPT_USER_TEMPLATE.format(
        problem=_clamp(problem or "(unknown)"),
        think=_clamp(think or ""),
        cues=", ".join(cue_names) if cue_names else "(none)",
        pos=(-1 if pos is None else pos),
    )

    # Prefer Responses API if available; fallback to Chat Completions.
    try:
        if _CLIENT_STATE["uses_v1"] and hasattr(_CLIENT_STATE["client"], "responses"):
            resp = _CLIENT_STATE["client"].responses.create(
                model=deployment,
                instructions=PROMPT_SYSTEM,
                input=[{"role": "user", "content": user_content}],
                temperature=0.0,
                max_output_tokens=500,
            )
            content = getattr(resp, "output_text", None) or ""
        else:
            resp = _CLIENT_STATE["client"].chat.completions.create(
                model=deployment,
                temperature=0.0,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": PROMPT_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
            )
            content = resp.choices[0].message.content if resp and resp.choices else ""
    except OpenAIError as e:
        _dump_filtered(user_content + "\n\n[ERROR] " + repr(e))
        return {
            "shift_in_reasoning": False,
            "confidence": "low",
            "markers_found": [],
            "first_marker_index": -1 if pos is None else int(pos),
            "before_excerpt": "",
            "after_excerpt": "",
            "explanation_short": "Model call failed; defaulting to FALSE.",
        }

    obj = _json_from_text(content or "")
    if not isinstance(obj, dict):
        _dump_filtered(user_content + "\n\n[UNPARSEABLE]\n" + (content or ""))
        return {
            "shift_in_reasoning": False,
            "confidence": "low",
            "markers_found": [],
            "first_marker_index": -1 if pos is None else int(pos),
            "before_excerpt": "",
            "after_excerpt": "",
            "explanation_short": "Unparseable response; default FALSE.",
        }

    # Guardrail: must include an explicit cue (prefilter or model markers)
    markers = obj.get("markers_found") or []
    has_explicit = bool(cue_names or markers)
    if not has_explicit:
        obj["shift_in_reasoning"] = False
        obj.setdefault("confidence", "low")
        obj.setdefault("explanation_short", "No explicit cue; conservative FALSE.")
    return obj

# ───────────────────── File scanning / update ─────────────────────
def nat_step_from_path(path: str) -> Optional[int]:
    """Extract numeric step from a filename like step00050_test.jsonl."""
    m = re.search(r"step(\d+)", path)
    return int(m.group(1)) if m else None

def scan_jsonl(root: str, split: Optional[str]) -> List[str]:
    """Recursively list JSONL files (sorted by step desc), optionally filtered by split."""
    files: List[str] = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if not fn.endswith(".jsonl"):
                continue
            if split and split not in fn:
                continue
            files.append(os.path.join(dp, fn))
    files.sort(key=lambda p: (nat_step_from_path(p) or 0, p), reverse=True)
    return files

def record_id_for_logs(obj: Dict[str, Any]) -> str:
    """Human-readable identifier for logging."""
    return (
        obj.get("row_key")
        or obj.get("problem")
        or obj.get("clue")
        or f"idx={obj.get('dataset_index','?')}"
    )

def _load_records(path: str) -> List[Dict[str, Any]]:
    """Load JSONL lines; keep raw lines for those that fail to parse."""
    with open(path, "r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = []
        for ln in f:
            try:
                records.append(json.loads(ln))
            except JSONDecodeError:
                records.append({"__raw__": ln})
    return records


def _prefilter_records(records: List[Dict[str, Any]]) -> List[int]:
    """Apply prefilter to find candidates and stamp obvious FALSE cases."""
    todo: List[int] = []
    for i, rec in enumerate(records):
        if "__raw__" in rec:
            continue
        p1 = rec.get("pass1") or {}
        if "shift_in_reasoning_v1" in p1:
            continue

        out = p1.get("output")
        if not out:
            continue

        think = _extract_think(out)
        if not think:
            p1["shift_in_reasoning_v1"] = False
            rec["pass1"] = p1
            continue

        cues, pos = _find_shift_cues(think)
        p1["_shift_prefilter_markers"] = cues
        p1["_shift_prefilter_pos"] = pos
        rec["pass1"] = p1
        todo.append(i)
    return todo


def _mark_no_cue(p1: Dict[str, Any], deployment: str) -> None:
    """Set conservative FALSE annotation when no cue is present."""
    p1["shift_in_reasoning_v1"] = False
    p1["shift_markers_v1"] = []
    p1["shift_first_marker_char"] = -1
    p1["shift_before_excerpt"] = ""
    p1["shift_after_excerpt"] = ""
    p1["shift_rationale_gpt"] = "No explicit cue; conservative FALSE."
    p1["shift_rationale_gpt_model"] = deployment
    p1["shift_rationale_gpt_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class AnnotateOpts:
    """Container for annotation options."""
    seed: int
    max_calls: Optional[int]
    dry_run: bool
    jitter: float
    deployment: str
    client_cfg: Dict[str, Any]


def _annotate_record(rec: Dict[str, Any], opts: AnnotateOpts, rng: random.Random) -> bool:
    """Annotate a single record; returns True if an LLM call was made."""
    p1 = rec.get("pass1") or {}
    out = p1.get("output") or ""
    think = _extract_think(out) or ""
    problem = rec.get("problem") or rec.get("clue") or ""
    cues = p1.get("_shift_prefilter_markers") or []
    pos = p1.get("_shift_prefilter_pos")

    if not cues:
        _mark_no_cue(p1, opts.deployment)
        rec["pass1"] = p1
        return False

    if opts.dry_run:
        logging.info("DRY-RUN would annotate id=%s", record_id_for_logs(rec))
        return False

    with timed(f"llm_call id={record_id_for_logs(rec)}"):
        result = llm_judge_shift(
            opts.client_cfg,
            opts.deployment,
            {
                "problem": problem,
                "think": think,
                "cues": cues,
                "pos": pos,
            },
        )

    p1["shift_in_reasoning_v1"] = bool(result.get("shift_in_reasoning", False))
    p1["shift_markers_v1"] = list(result.get("markers_found", []) or cues)
    p1["shift_first_marker_char"] = int(
        result.get("first_marker_index", -1 if pos is None else pos)
    )
    p1["shift_before_excerpt"] = _clamp(result.get("before_excerpt", ""), 240)
    p1["shift_after_excerpt"] = _clamp(result.get("after_excerpt", ""), 280)
    p1["shift_rationale_gpt"] = _clamp(result.get("explanation_short", ""), 300)
    p1["shift_rationale_gpt_model"] = opts.deployment
    p1["shift_rationale_gpt_time"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    rec["pass1"] = p1
    if opts.jitter > 0:
        time.sleep(rng.uniform(0.0, opts.jitter))
    return True


def annotate_file(path: str, opts: AnnotateOpts):
    """Annotate a single JSONL file in-place."""
    logging.info("Annotating: %s", path)
    records = _load_records(path)

    todo_idxs = _prefilter_records(records)
    rng = random.Random(opts.seed)
    rng.shuffle(todo_idxs)
    if opts.max_calls is not None:
        todo_idxs = todo_idxs[:opts.max_calls]

    calls = 0
    for idx in todo_idxs:
        if _annotate_record(records[idx], opts, rng):
            calls += 1

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            if "__raw__" in rec:
                f.write(rec["__raw__"])
            else:
                json.dump(rec, f, ensure_ascii=False)
                f.write("\n")
    os.replace(tmp, path)
    logging.info("Updated %s (LLM calls: %d)", path, calls)

# ───────────────────── CLI ─────────────────────
def build_argparser():
    """CLI argument builder."""
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root", help="Directory containing step*/.../*.jsonl")
    ap.add_argument(
        "--split",
        default=None,
        help="Filter filenames that contain this substring (e.g., 'test').",
    )

    ap.add_argument(
        "--seed", type=int, default=1234, help="Shuffle seed for random processing order."
    )
    ap.add_argument(
        "--max_calls", type=int, default=None, help="Optional cap on model calls."
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Discover candidates but do not call the model or write changes.",
    )
    ap.add_argument(
        "--jitter",
        type=float,
        default=0.25,
        help="Max random sleep (seconds) between calls; 0 to disable.",
    )
    ap.add_argument("--loglevel", default="INFO")

    # Azure OpenAI specifics
    ap.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Azure OpenAI endpoint (e.g., https://<res>.openai.azure.com/).",
    )
    ap.add_argument(
        "--deployment",
        default=DEFAULT_DEPLOYMENT,
        help="Azure OpenAI deployment name (e.g., 'gpt-4o').",
    )
    ap.add_argument(
        "--api_version",
        default=DEFAULT_API_VERSION,
        help="Azure API version (legacy client only).",
    )
    ap.add_argument(
        "--use_v1",
        type=int,
        default=DEFAULT_USE_V1,
        help="1=prefer v1 Responses API, 0=legacy client.",
    )
    return ap

def main():
    """Entrypoint."""
    ap = build_argparser()
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    files = scan_jsonl(args.results_root, args.split)
    if not files:
        print("No JSONL files found; check path/split.", file=sys.stderr)
        sys.exit(1)

    opts = AnnotateOpts(
        seed=args.seed,
        max_calls=args.max_calls,
        dry_run=args.dry_run,
        jitter=args.jitter,
        deployment=args.deployment,
        client_cfg={
            "endpoint": args.endpoint,
            "api_version": args.api_version,
            "use_v1": args.use_v1,
        },
    )

    for path in files:
        annotate_file(path, opts)

if __name__ == "__main__":
    main()
