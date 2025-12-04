#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core shift-in-reasoning annotation utilities (no CLI).

This module contains the pure functions and data structures used to:
- scan JSONL result files
- prefilter candidate examples via lexical cues
- call an LLM judge
- write ``shift_in_reasoning_v1`` labels and metadata back to disk

CLI entrypoints (argparse / logging / clean-up wiring) live in
``src.annotate.cli.shift_cli`` and import from here.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import random
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Dict, Iterable, List, Optional, Set

from ...common.jsonl_utils import scan_jsonl_files
from ..infra.config import load_azure_config
from ..infra.llm_client import build_preferred_client
from .prefilter import extract_think as _extract_think
from .prefilter import find_shift_cues as _find_shift_cues
from .prompts import SHIFT_JUDGE_SYSTEM_PROMPT as PROMPT_SYSTEM
from .prompts import SHIFT_JUDGE_USER_TEMPLATE as PROMPT_USER_TEMPLATE


try:  # pragma: no cover - optional dependency
    from openai import OpenAIError as _OpenAIError
except ImportError:  # pragma: no cover - optional dependency

    class _OpenAIError(Exception):
        """Fallback OpenAI error when openai package is absent."""


OpenAIError = _OpenAIError

try:  # pragma: no cover - optional dependency
    import httpx

    HTTPError = httpx.HTTPError
except ImportError:  # pragma: no cover - optional dependency

    class HTTPError(Exception):  # type: ignore[too-many-ancestors]
        """Fallback HTTP error when httpx is unavailable."""


try:  # pragma: no cover - optional dependency
    from portkey_ai import Portkey  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional dependency
    Portkey = None  # type: ignore[assignment]


# ───────────────────── Azure OpenAI config (env or CLI) ─────────────────────
_cfg = load_azure_config()
DEFAULT_ENDPOINT = _cfg["endpoint"]
DEFAULT_DEPLOYMENT = _cfg["deployment"]
DEFAULT_API_VERSION = _cfg["api_version"]
DEFAULT_USE_V1 = int(_cfg["use_v1"])


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
    """Persist problematic prompts for later inspection."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    digest = json.dumps({"t": timestamp, "p": prompt})
    filename = f"filtered_prompt_{timestamp}_{abs(hash(digest)) & 0xFFFFFFFF:08x}.txt"
    with open(filename, "w", encoding="utf-8") as file_handle:
        file_handle.write(prompt)
    # Logging is optional; handled via the root logger configuration.
    logging.warning("LLM filtered/failed; saved prompt to %s", filename)


def _sanitize_jsonish(text: str) -> str:
    """
    Best-effort cleanup for JSON-like text that may contain invalid escape
    sequences produced by LLMs (e.g., LaTeX-style ``\\( ... \\)``).

    When possible, this parses and re-dumps JSON so that string values are
    normalized; on parse failure it falls back to a simple textual replacement
    that strips a small set of known-bad escapes (``\\(``, ``\\)``, ``\\[``,
    ``\\]``) that are illegal in strict JSON.
    """
    if not text:
        return text

    try:
        obj = json.loads(text)
    except (JSONDecodeError, TypeError, ValueError):
        # Fall back to a simple textual replacement.
        return text.replace("\\(", "(").replace("\\)", ")").replace("\\[", "[").replace("\\]", "]")

    def _fix(val: Any) -> Any:
        if isinstance(val, str):
            return val.replace("\\(", "(").replace("\\)", ")").replace("\\[", "[").replace("\\]", "]")
        if isinstance(val, list):
            return [_fix(v) for v in val]
        if isinstance(val, dict):
            return {k: _fix(v) for k, v in val.items()}
        return val

    cleaned_obj = _fix(obj)
    return json.dumps(cleaned_obj, ensure_ascii=False)


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
            return json.loads(_sanitize_jsonish(text[i : j + 1]))
        except JSONDecodeError:
            return None
    return None


_CLIENT_STATE: Dict[str, Any] = {"client": None, "uses_v1": False}


def _client_lazy(client_cfg: Dict[str, Any]) -> None:
    """Create and cache an LLM client (Azure or Portkey)."""
    if _CLIENT_STATE["client"] is not None:
        return

    backend = client_cfg.get("backend", "azure")

    if backend == "portkey":
        if Portkey is None:
            raise RuntimeError(
                "backend='portkey' requires the portkey-ai package: pip install portkey-ai",
            )
        api_key = os.getenv("AI_SANDBOX_KEY") or os.getenv("PORTKEY_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "backend='portkey' expects AI_SANDBOX_KEY (or PORTKEY_API_KEY) in the environment.",
            )

        client = Portkey(api_key=api_key)
        _CLIENT_STATE["client"] = client
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


def llm_judge_shift(
    client_cfg: Dict[str, Any],
    deployment: str,
    example: Dict[str, Any],
) -> Dict[str, Any]:
    """Call the LLM judge for a single example."""
    _client_lazy(client_cfg)
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

    markers = obj.get("markers_found") or []
    has_explicit = bool(cue_names or markers)
    if not has_explicit:
        obj["shift_in_reasoning"] = False
        obj.setdefault("confidence", "low")
        obj.setdefault("explanation_short", "No explicit cue; conservative FALSE.")
    return obj


def nat_step_from_path(path: str) -> Optional[int]:
    """Extract numeric step from a filename like ``step00050_test.jsonl``."""
    match = re.search(r"step(\d+)", path)
    return int(match.group(1)) if match else None


def scan_jsonl(root: str, split: Optional[str]) -> List[str]:
    """
    Recursively list JSONL files, sorted by step descending.

    :param root: Root directory containing results.
    :param split: Optional substring to filter filenames (e.g. ``\"test\"``).
    """
    files = scan_jsonl_files(root, split_substr=split)
    files.sort(key=lambda p: (nat_step_from_path(p) or 0, p), reverse=True)
    return files


def record_id_for_logs(obj: Dict[str, Any]) -> str:
    """Human-readable identifier for logging."""
    return obj.get("row_key") or obj.get("problem") or obj.get("clue") or f"idx={obj.get('dataset_index', '?')}"


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


@contextmanager
def _exclusive_file_lock(path: str):
    """Acquire an exclusive lock for the given path."""
    lock_path = path + ".lock"
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _write_records_to_disk(
    path: str,
    records: List[Dict[str, Any]],
    dirty_idxs: Optional[Iterable[int]] = None,
) -> None:
    """Persist the in-memory records to disk via atomic replace.

    When ``dirty_idxs`` is provided, we merge only the touched rows with the
    latest on-disk version so that concurrent workers do not clobber each other.
    """
    tmp = path + ".tmp"
    dirty_set: Set[int] = set(idx for idx in (dirty_idxs or ()) if 0 <= idx < len(records))
    with _exclusive_file_lock(path):
        try:
            disk_records = _load_records(path)
        except FileNotFoundError:
            disk_records = []

        if dirty_set and len(disk_records) == len(records):
            merged_records = list(disk_records)
            for idx in dirty_set:
                merged_records[idx] = records[idx]
            write_records = merged_records
        else:
            write_records = records

        with open(tmp, "w", encoding="utf-8") as tmp_file:
            for rec in write_records:
                if "__raw__" in rec:
                    tmp_file.write(rec["__raw__"])
                else:
                    json.dump(rec, tmp_file, ensure_ascii=False)
                    tmp_file.write("\n")
        os.replace(tmp, path)


def _prefilter_records_for_pass(
    records: List[Dict[str, Any]],
    pass_key: str,
    force_relabel: bool = False,
    dirty_idxs: Optional[Set[int]] = None,
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
            section["shift_in_reasoning_v1"] = False
            rec[pass_key] = section
            if dirty_idxs is not None:
                dirty_idxs.add(i)
            continue

        think = _extract_think(out) or out
        cues, pos = _find_shift_cues(think)
        section["_shift_prefilter_markers"] = cues
        section["_shift_prefilter_pos"] = pos
        rec[pass_key] = section
        if dirty_idxs is not None:
            dirty_idxs.add(i)
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
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(),
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


@dataclass
class PassRunContext:
    """Metadata for annotating a specific record/pass combination."""

    rng: random.Random
    record_index: int
    dirty_idxs: Optional[Set[int]] = None

    def mark_dirty(self) -> None:
        """Record that the current row was modified."""
        if self.dirty_idxs is not None:
            self.dirty_idxs.add(self.record_index)


def _annotate_record_for_pass(
    rec: Dict[str, Any],
    pass_key: str,
    opts: AnnotateOpts,
    ctx: PassRunContext,
) -> bool:
    """
    Annotate a single pass section within a record.

    Returns True if an LLM call was made.
    """
    pass_section = rec.get(pass_key) or {}
    if not isinstance(pass_section, dict):
        return False

    out = pass_section.get("output") or ""
    think = _extract_think(out) or out or ""
    problem = rec.get("problem") or rec.get("clue") or ""
    cues = pass_section.get("_shift_prefilter_markers") or []
    pos = pass_section.get("_shift_prefilter_pos")
    deployment = opts.client_cfg.get("deployment", "")

    if not think.strip():
        _mark_no_cue(pass_section, deployment)
        rec[pass_key] = pass_section
        ctx.mark_dirty()
        return False

    if opts.dry_run:
        logging.info(
            "DRY-RUN would annotate %s for id=%s",
            pass_key,
            record_id_for_logs(rec),
        )
        return False

    if opts.jitter > 0:
        time.sleep(ctx.rng.random() * opts.jitter)

    with timed(f"llm_call id={record_id_for_logs(rec)}"):
        result = llm_judge_shift(
            opts.client_cfg,
            deployment,
            {
                "problem": problem,
                "think": think,
                "cues": cues,
                "pos": pos,
            },
        )

    pass_section["shift_in_reasoning_v1"] = bool(result.get("shift_in_reasoning", False))
    pass_section["shift_markers_v1"] = result.get("markers_found") or []
    pass_section["shift_first_marker_char"] = int(result.get("first_marker_index", -1))
    pass_section["shift_before_excerpt"] = result.get("before_excerpt") or ""
    pass_section["shift_after_excerpt"] = result.get("after_excerpt") or ""
    pass_section["shift_rationale_gpt"] = result.get("explanation_short") or ""
    pass_section["shift_rationale_gpt_model"] = opts.client_cfg.get("deployment", "")
    pass_section["shift_rationale_gpt_time"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(),
    )
    rec[pass_key] = pass_section
    ctx.mark_dirty()
    return True


def annotate_file(
    path: str,
    opts: AnnotateOpts,
    hook=_annotate_record_for_pass,
) -> None:
    """
    Annotate all configured passes in a single JSONL file.

    The ``hook`` parameter is primarily for testing; callers normally rely on
    the default :func:`_annotate_record_for_pass`.
    """
    logging.info("Annotating %s", path)
    records = _load_records(path)
    rng = random.Random(opts.seed)

    dirty_idxs: Set[int] = set()
    calls = 0
    for pass_key in opts.passes:
        todo_idxs = _prefilter_records_for_pass(
            records,
            pass_key,
            force_relabel=opts.force_relabel,
            dirty_idxs=dirty_idxs,
        )
        rng.shuffle(todo_idxs)
        if opts.max_calls is not None:
            remaining = max(opts.max_calls - calls, 0)
            todo_idxs = todo_idxs[:remaining]
        for idx in todo_idxs:
            if opts.max_calls is not None and calls >= opts.max_calls:
                break
            ctx = PassRunContext(rng=rng, record_index=idx, dirty_idxs=dirty_idxs)
            if hook(
                records[idx],
                pass_key,
                opts,
                ctx,
            ):
                calls += 1
                _write_records_to_disk(path, records, dirty_idxs={idx})
        if opts.max_calls is not None and calls >= opts.max_calls:
            break

    if dirty_idxs:
        _write_records_to_disk(path, records, dirty_idxs=dirty_idxs)
    logging.info("Updated %s (LLM calls: %d)", path, calls)


__all__ = [
    "AnnotateOpts",
    "DEFAULT_API_VERSION",
    "DEFAULT_DEPLOYMENT",
    "DEFAULT_ENDPOINT",
    "DEFAULT_USE_V1",
    "annotate_file",
    "llm_judge_shift",
    "nat_step_from_path",
    "record_id_for_logs",
    "scan_jsonl",
    "_annotate_record_for_pass",
    "_json_from_text",
    "_sanitize_jsonish",
]
