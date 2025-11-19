#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared inference helpers used across math/carpark/crossword entrypoints.
The goal is to keep canonicalization, tag parsing, and small torch utilities
in one place so the task-specific scripts stay focused on their domain logic.
"""

from __future__ import annotations

import json
import math
import re
from typing import Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:  # Avoid hard dependency at import time.
    from datasets import Dataset

# Regular expressions reused by multiple scripts.
RE_THINK = re.compile(r"(?si)<think>(.*?)</think>")
RE_ANSWER = re.compile(r"(?si)<answer>(.*?)</answer>")


# ---------------------------------------------------------------------------
# Text + tag helpers
# ---------------------------------------------------------------------------
def extract_blocks(txt: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (<think>..., <answer>...) contents (whitespace stripped)."""
    think = ans = None
    m = RE_THINK.search(txt)
    if m:
        think = m.group(1).strip()
    m = RE_ANSWER.search(txt)
    if m:
        ans = m.group(1).strip()
    return think, ans


def valid_tag_structure(full_text: str) -> bool:
    """Require exactly one <think>…</think> before <answer>…</answer>."""
    opens_think = len(re.findall(r"(?i)<think>", full_text))
    closes_think = len(re.findall(r"(?i)</think>", full_text))
    opens_ans = len(re.findall(r"(?i)<answer>", full_text))
    closes_ans = len(re.findall(r"(?i)</answer>", full_text))
    if not (opens_think == closes_think == 1 and opens_ans == closes_ans == 1):
        return False
    try:
        a = re.search(r"(?i)<think>", full_text).start()
        b = re.search(r"(?i)</think>", full_text).start()
        c = re.search(r"(?i)<answer>", full_text).start()
        d = re.search(r"(?i)</answer>", full_text).start()
        return a < b < c < d
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Torch helpers
# ---------------------------------------------------------------------------
def move_inputs_to_device(inputs: dict, device: Optional[torch.device] = None) -> tuple[dict, torch.Tensor]:
    """Move a HuggingFace-style inputs dict to CUDA (or provided device) and return input_lengths.

    This dedups the common pattern of computing attention_mask sums and moving tensors.
    """
    input_lengths = inputs["attention_mask"].sum(dim=1)
    target_device = device
    if target_device is None and torch.cuda.is_available():
        target_device = torch.device("cuda")
    if target_device is not None:
        for k in inputs:
            inputs[k] = inputs[k].to(target_device)
        input_lengths = input_lengths.to(inputs["input_ids"].device)
    return inputs, input_lengths


def find_markers_and_context(
    think_text: Optional[str],
    prompt_text: str,
    patterns: Sequence[Tuple[str, re.Pattern]],
    *,
    skip_prefix_chars: int = 0,
):
    """
    Scan think_text for the earliest match among patterns.
    Returns (markers, earliest_pos, context_prefix, excerpt).
    """
    if not think_text:
        return [], None, None, None
    search_text = think_text[skip_prefix_chars:] if skip_prefix_chars > 0 else think_text
    earliest_pos = None
    markers = []
    for name, pat in patterns:
        m = pat.search(search_text)
        if m:
            markers.append(name)
            pos_global = (skip_prefix_chars + m.start()) if skip_prefix_chars > 0 else m.start()
            if earliest_pos is None or pos_global < earliest_pos:
                earliest_pos = pos_global
    if not markers:
        return [], None, None, None
    prefix = think_text[:earliest_pos] if earliest_pos is not None else think_text
    context = f"{prompt_text}\n\n{prefix}"
    lo = max(0, (earliest_pos or 0) - 60)
    hi = min(len(think_text), (earliest_pos or 0) + 60)
    excerpt = think_text[lo:hi]
    return markers, earliest_pos, context, excerpt


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------
RE_LATEX_FRAC = re.compile(r"\\frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}", re.I)
RE_LATEX_CMDS = re.compile(r"\\(left|right|,|;|!|:)", re.I)
RE_SPACES = re.compile(r"\s+")
RE_BRACES = re.compile(r"[{}]")
RE_PARENS_COMMAs = re.compile(r"[()\[\],]")


def canon_math(x: Optional[str]) -> Optional[str]:
    """
    Permissive canonicalizer for math answers. Lowercases, removes spacing/punctuation,
    simplifies common LaTeX forms, and normalizes pi.
    """
    if x is None:
        return None
    s = x.strip()
    s = (s.replace("–", "-").replace("—", "-").replace("−", "-")
           .replace("π", "pi").replace("\\pi", "pi"))
    s = RE_LATEX_CMDS.sub("", s)
    s = RE_LATEX_FRAC.sub(r"\1/\2", s)
    s = RE_BRACES.sub("", s)
    s = RE_SPACES.sub("", s)
    s = RE_PARENS_COMMAs.sub("", s)
    s = s.replace("\\boxed", "").replace("$", "")
    s = s.lower().rstrip(".")
    s = re.sub(r"/{2,}", "/", s)
    s = re.sub(r"\+{2,}", "+", s)
    s = re.sub(r"-{2,}", "-", s)
    if s.startswith("+"):
        s = s[1:]
    return s


def contains_canon(hay: Optional[str], needle: Optional[str]) -> bool:
    """Substring check after both sides are canonicalized."""
    return bool(hay and needle and (needle in hay))


# ---------------------------------------------------------------------------
# Torch utilities
# ---------------------------------------------------------------------------
def finite_mean(vals: Iterable[float]) -> Optional[float]:
    vs = [float(v) for v in vals if v == v and math.isfinite(float(v))]
    return (sum(vs) / len(vs)) if vs else None


def first_eos_any(token_ids: torch.Tensor, eos_id_list: Optional[Sequence[int]]) -> int:
    if not eos_id_list:
        return token_ids.numel()
    hits = []
    for eid in eos_id_list:
        pos = (token_ids == eid).nonzero(as_tuple=False)
        if pos.numel() > 0:
            hits.append(pos[0].item())
    return min(hits) if hits else token_ids.numel()


def entropy_from_start_index(model, seq_ids: torch.Tensor, start_idx: int) -> List[float]:
    """
    Compute token-wise entropy starting at position start_idx (inclusive).
    Safe for NaNs thanks to re-centering.
    """
    device = next(model.parameters()).device
    seq_ids = seq_ids.to(device)
    ents: List[float] = []
    with torch.inference_mode():
        out = model(input_ids=seq_ids[:, :start_idx + 1], use_cache=True)
        past = out.past_key_values
        L = seq_ids.shape[1]
        for t in range(start_idx, L - 1):
            out = model(input_ids=seq_ids[:, t:t + 1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()
            logp = torch.nn.functional.log_softmax(logits, dim=-1)
            p = logp.exp()
            h = float(-(p * logp).sum().item())
            if not math.isfinite(h):
                logits = (logits - logits.max()).float()
                logp = torch.nn.functional.log_softmax(logits, dim=-1)
                p = logp.exp()
                h = float(-(p * logp).sum().item())
            ents.append(h)
    return ents


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------
def load_local_json_dataset(path: str) -> "Dataset":
    """
    Read a JSONL-like local file into a datasets.Dataset.
    Lines that are empty or not JSON objects are skipped.
    """
    from datasets import Dataset  # Local import keeps optional dependency light.

    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line) if line.startswith("{") else None
            if obj is None:
                continue
            records.append(obj)
    return Dataset.from_list(records)
