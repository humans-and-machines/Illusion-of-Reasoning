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
from src.annotate.clean_failed_shift_labels import clean_root as _clean_failed_root

try:
    from openai import OpenAIError as _OpenAIError
except ImportError:  # pragma: no cover - optional dependency
    class _OpenAIError(Exception):
        """Fallback OpenAI error when openai package is absent."""

OpenAIError = _OpenAIError

try:
    import httpx
    HTTPError = httpx.HTTPError
except ImportError:  # pragma: no cover - optional dependency
    class HTTPError(Exception):  # type: ignore[too-many-ancestors]
        """Fallback HTTP error when httpx is unavailable."""

try:
    from portkey_ai import Portkey  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional dependency
    Portkey = None  # type: ignore[assignment]

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
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    digest = hashlib.md5(prompt.encode()).hexdigest()[:8]
    filename = f"filtered_prompt_{timestamp}_{digest}.txt"
    with open(filename, "w", encoding="utf-8") as file_handle:
        file_handle.write(prompt)
    logging.warning("LLM filtered/failed; saved prompt to %s", filename)


def _sanitize_jsonish(text: str) -> str:
    """
    Best-effort cleanup for JSON-like text that may contain invalid escape
    sequences produced by LLMs (e.g., LaTeX-style \\( ... \\)).

    This is intentionally conservative: we strip a small set of known-bad
    escapes that commonly appear in math explanations but are illegal in
    strict JSON (\\(, \\), \\[, \\]).
    """
    if not text:
        return text
    return (
        text.replace("\\(", "(")
        .replace("\\)", ")")
        .replace("\\[", "[")
        .replace("\\]", "]")
    )


def _json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first top-level JSON object from arbitrary text."""
    text = (text or "").strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(_sanitize_jsonish(text))
        except JSONDecodeError:
            return None
    i = text.find("{")
    j = text.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            return json.loads(_sanitize_jsonish(text[i:j+1]))
        except JSONDecodeError:
            return None
    return None

# ───────────────────── Client factory (Azure / Portkey) ─────────────────────
_CLIENT_STATE = {"client": None, "uses_v1": False}

def _client_lazy(client_cfg: Dict[str, Any]) -> None:
    """Create and cache an LLM client (Azure or Portkey)."""
    if _CLIENT_STATE["client"] is not None:
        return

    backend = client_cfg.get("backend", "azure")

    if backend == "portkey":
        # Portkey AI Gateway / Princeton AI Sandbox path.
        if Portkey is None:
            raise RuntimeError(
                "backend='portkey' requires the portkey-ai package: pip install portkey-ai"
            )
        api_key = os.getenv("AI_SANDBOX_KEY") or os.getenv("PORTKEY_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "backend='portkey' expects AI_SANDBOX_KEY (or PORTKEY_API_KEY) in the environment."
            )

        client = Portkey(api_key=api_key)
        _CLIENT_STATE["client"] = client
        # Portkey client exposes only chat.completions in this flow.
        _CLIENT_STATE["uses_v1"] = False
        return

    # Default: Azure OpenAI via openai>=1.x helpers.
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
                max_output_tokens=1000,
            )
            content = getattr(resp, "output_text", None) or ""
        else:
            resp = _CLIENT_STATE["client"].chat.completions.create(
                model=deployment,
                temperature=0.0,
                max_tokens=1000,
                messages=[
                    {"role": "system", "content": PROMPT_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
            )
            content = resp.choices[0].message.content if resp and resp.choices else ""
    except (OpenAIError, HTTPError) as error:
        _dump_filtered(user_content + "\n\n[ERROR] " + repr(error))
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
    match = re.search(r"step(\d+)", path)
    return int(match.group(1)) if match else None

def scan_jsonl(root: str, split: Optional[str]) -> List[str]:
    """Recursively list JSONL files (sorted by step desc), optionally filtered by split."""
    files: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if not filename.endswith(".jsonl"):
                continue
            if split and split not in filename:
                continue
            files.append(os.path.join(dirpath, filename))
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
    with open(path, "r", encoding="utf-8") as file_handle:
        records: List[Dict[str, Any]] = []
        for line in file_handle:
            try:
                records.append(json.loads(line))
            except JSONDecodeError:
                records.append({"__raw__": line})
    return records


def _prefilter_records_for_pass(
    records: List[Dict[str, Any]],
    pass_key: str,
    force_relabel: bool = False,
) -> List[int]:
    """Apply prefilter for a specific pass section and stamp obvious FALSE cases."""
    todo: List[int] = []
    for i, rec in enumerate(records):
        if "__raw__" in rec:
            continue
        section = rec.get(pass_key) or {}
        if not isinstance(section, dict):
            continue
        if "shift_in_reasoning_v1" in section and not force_relabel:
            continue

        out = section.get("output") or ""
        if not out.strip():
            # No text to inspect; conservatively mark FALSE and skip.
            section["shift_in_reasoning_v1"] = False
            rec[pass_key] = section
            continue

        # If there is no explicit <think> block (e.g., some OpenRouter runs),
        # fall back to using the full output as the "think" text for cue search.
        think = _extract_think(out) or out
        cues, pos = _find_shift_cues(think)
        section["_shift_prefilter_markers"] = cues
        section["_shift_prefilter_pos"] = pos
        rec[pass_key] = section
        todo.append(i)
    return todo


def _mark_no_cue(pass_section: Dict[str, Any], deployment: str) -> None:
    """Set conservative FALSE annotation when no cue is present."""
    pass_section["shift_in_reasoning_v1"] = False
    pass_section["shift_markers_v1"] = []
    pass_section["shift_first_marker_char"] = -1
    pass_section["shift_before_excerpt"] = ""
    pass_section["shift_after_excerpt"] = ""
    pass_section["shift_rationale_gpt"] = "No explicit cue; conservative FALSE."
    pass_section["shift_rationale_gpt_model"] = deployment
    pass_section["shift_rationale_gpt_time"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )


@dataclass
class AnnotateOpts:
    """Container for annotation options."""
    seed: int
    max_calls: Optional[int]
    dry_run: bool
    jitter: float
    force_relabel: bool
    client_cfg: Dict[str, Any]
    passes: List[str]


def _annotate_record_for_pass(
    rec: Dict[str, Any],
    pass_key: str,
    opts: AnnotateOpts,
    deployment: str,
    rng: random.Random,
) -> bool:
    """Annotate a single pass section within a record; returns True if an LLM call was made."""
    pass_section = rec.get(pass_key) or {}
    if not isinstance(pass_section, dict):
        return False

    out = pass_section.get("output") or ""
    # Fall back to the full output when <think> tags are absent.
    think = _extract_think(out) or out or ""
    problem = rec.get("problem") or rec.get("clue") or ""
    cues = pass_section.get("_shift_prefilter_markers") or []
    pos = pass_section.get("_shift_prefilter_pos")

    # If there is literally no text, mark FALSE without calling the LLM.
    if not think.strip():
        _mark_no_cue(pass_section, deployment)
        rec[pass_key] = pass_section
        return False

    if opts.dry_run:
        logging.info(
            "DRY-RUN would annotate %s for id=%s",
            pass_key,
            record_id_for_logs(rec),
        )
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

    pass_section["shift_in_reasoning_v1"] = bool(result.get("shift_in_reasoning", False))
    pass_section["shift_markers_v1"] = list(result.get("markers_found", []) or cues)
    pass_section["shift_first_marker_char"] = int(
        result.get("first_marker_index", -1 if pos is None else pos)
    )
    pass_section["shift_before_excerpt"] = _clamp(result.get("before_excerpt", ""), 240)
    pass_section["shift_after_excerpt"] = _clamp(result.get("after_excerpt", ""), 280)
    pass_section["shift_rationale_gpt"] = _clamp(result.get("explanation_short", ""), 300)
    pass_section["shift_rationale_gpt_model"] = deployment
    pass_section["shift_rationale_gpt_time"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    rec[pass_key] = pass_section
    if opts.jitter > 0:
        time.sleep(rng.uniform(0.0, opts.jitter))
    return True


def annotate_file(path: str, opts: AnnotateOpts):
    """Annotate a single JSONL file in-place."""
    logging.info("Annotating: %s", path)
    records = _load_records(path)

    calls = 0
    passes = opts.passes or ["pass1"]
    for pass_key in passes:
        logging.info("Prefiltering pass section %s", pass_key)
        todo_idxs = _prefilter_records_for_pass(
            records,
            pass_key,
            force_relabel=opts.force_relabel,
        )
        rng = random.Random(opts.seed)
        rng.shuffle(todo_idxs)
        if opts.max_calls is not None:
            remaining = max(opts.max_calls - calls, 0)
            todo_idxs = todo_idxs[:remaining]
        for idx in todo_idxs:
            if opts.max_calls is not None and calls >= opts.max_calls:
                break
            if _annotate_record_for_pass(
                records[idx],
                pass_key,
                opts,
                opts.client_cfg.get("deployment", ""),
                rng,
            ):
                calls += 1
        if opts.max_calls is not None and calls >= opts.max_calls:
            break

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as tmp_file:
        for rec in records:
            if "__raw__" in rec:
                tmp_file.write(rec["__raw__"])
            else:
                json.dump(rec, tmp_file, ensure_ascii=False)
                tmp_file.write("\n")
    os.replace(tmp, path)
    logging.info("Updated %s (LLM calls: %d)", path, calls)

# ───────────────────── CLI ─────────────────────
def build_argparser():
    """CLI argument builder."""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("results_root", help="Directory containing step*/.../*.jsonl")
    arg_parser.add_argument(
        "--split",
        default=None,
        help="Filter filenames that contain this substring (e.g., 'test').",
    )

    arg_parser.add_argument(
        "--seed", type=int, default=1234, help="Shuffle seed for random processing order."
    )
    arg_parser.add_argument(
        "--max_calls", type=int, default=None, help="Optional cap on model calls."
    )
    arg_parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Discover candidates but do not call the model or write changes.",
    )
    arg_parser.add_argument(
        "--jitter",
        type=float,
        default=0.25,
        help="Max random sleep (seconds) between calls; 0 to disable.",
    )
    arg_parser.add_argument("--loglevel", default="INFO")

    arg_parser.add_argument(
        "--force_relabel",
        action="store_true",
        help="Re-annotate records even if shift_in_reasoning_v1 already exists.",
    )

    arg_parser.add_argument(
        "--clean_failed_first",
        action="store_true",
        help=(
            "Run clean_failed_shift_labels on results_root before annotation "
            "(strips fallback FALSE shift labels from prior failed judge calls)."
        ),
    )

    arg_parser.add_argument(
        "--passes",
        default="pass1",
        help=(
            "Comma-separated pass keys to annotate "
            "(e.g., 'pass1', 'pass1,pass2,pass2a,pass2b,pass2c'). "
            "Defaults to 'pass1' for backwards compatibility."
        ),
    )

    arg_parser.add_argument(
        "--backend",
        choices=["azure", "portkey"],
        default="azure",
        help="LLM backend: 'azure' (default) or 'portkey' (AI Sandbox via portkey-ai).",
    )

    # Azure OpenAI specifics
    arg_parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Azure OpenAI endpoint (e.g., https://<res>.openai.azure.com/).",
    )
    arg_parser.add_argument(
        "--deployment",
        default=DEFAULT_DEPLOYMENT,
        help="Azure OpenAI deployment name (e.g., 'gpt-4o').",
    )
    arg_parser.add_argument(
        "--api_version",
        default=DEFAULT_API_VERSION,
        help="Azure API version (legacy client only).",
    )
    arg_parser.add_argument(
        "--use_v1",
        type=int,
        default=DEFAULT_USE_V1,
        help="1=prefer v1 Responses API, 0=legacy client.",
    )
    return arg_parser

def main():
    """Entrypoint."""
    arg_parser = build_argparser()
    args = arg_parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.clean_failed_first:
        logging.info(
            "Cleaning prior fallback shift labels under %s before annotation.",
            os.path.abspath(args.results_root),
        )
        _clean_failed_root(args.results_root)

    files = scan_jsonl(args.results_root, args.split)
    if not files:
        print("No JSONL files found; check path/split.", file=sys.stderr)
        sys.exit(1)

    opts = AnnotateOpts(
        seed=args.seed,
        max_calls=args.max_calls,
        dry_run=args.dry_run,
        jitter=args.jitter,
        force_relabel=args.force_relabel,
        client_cfg={
            "backend": args.backend,
            "endpoint": args.endpoint,
            "api_version": args.api_version,
            "use_v1": args.use_v1,
        },
        passes=[
            p.strip()
            for p in (args.passes or "").split(",")
            if p.strip()
        ],
    )

    for path in files:
        annotate_file(path, opts)

if __name__ == "__main__":
    main()
