#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared inference helpers used across math/carpark/crossword entrypoints.
The goal is to keep canonicalization, tag parsing, and small torch utilities
in one place so the task-specific scripts stay focused on their domain logic.
"""
# pylint: disable=too-many-lines

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from importlib import import_module
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import torch
    from transformers import StoppingCriteria
except ImportError:  # pragma: no cover - type-check / lint environments
    class _TorchStub:
        """Stub object that raises if torch-dependent utilities are used."""

        def __getattr__(self, _name: str) -> Any:
            msg = "torch is required for inference utilities in inference.common."
            raise ImportError(msg)

        def is_available(self) -> bool:
            """Return False to mirror torch.cuda.is_available-style probes."""
            return False

        def device(self) -> str:
            """Placeholder device accessor to satisfy style checks."""
            return "cpu"

    torch = _TorchStub()  # type: ignore[assignment]

    class StoppingCriteria:  # type: ignore[too-many-ancestors]
        """Stub StoppingCriteria base when transformers is unavailable."""

        def __call__(self, *args: Any, **kwargs: Any) -> bool:  # noqa: ARG002
            msg = "transformers is required for StoppingCriteria in inference.common."
            raise ImportError(msg)

        def clone(self) -> "StoppingCriteria":
            """Return self; provided solely to satisfy minimal API expectations."""
            return self

        def has_stops(self) -> bool:
            """Placeholder method for compatibility with StopOnSubstrings."""
            return False


# Regular expressions reused by multiple scripts.
RE_THINK = re.compile(r"(?si)<think>(.*?)</think>")
RE_ANSWER = re.compile(r"(?si)<answer>(.*?)</answer>")

# Optional reconsider-cue detectors shared across math/crossword runners.
RECONSIDER_PATTERNS: Sequence[Tuple[str, re.Pattern]] = [
    ("wait_line", re.compile(r"(?im)^\s*wait[,\.\-–—… ]", re.I)),
    ("wait_reconsider", re.compile(r"\bwait\b.*\breconsider\b", re.I | re.S)),
    ("reconsider_exact", re.compile(r"\bwait[,!\.\s]*let me reconsider\b", re.I)),
    ("step_by_step", re.compile(r"\blet'?s take (this|it) step[-\s]?by[-\s]?step\b", re.I)),
    ("step_by_step_alt", re.compile(r"\bstep[-\s]?by[-\s]?step\b", re.I)),
    ("recheck", re.compile(r"\bre[-\s]?check(ing)?\b", re.I)),
]


# ---------------------------------------------------------------------------
# Text + tag helpers
# ---------------------------------------------------------------------------
def extract_blocks(txt: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (<think>..., <answer>...) contents (whitespace stripped)."""
    think = ans = None
    think_match = RE_THINK.search(txt)
    if think_match:
        think = think_match.group(1).strip()
    answer_match = RE_ANSWER.search(txt)
    if answer_match:
        ans = answer_match.group(1).strip()
    return think, ans


def valid_tag_structure(full_text: str) -> bool:
    """Require exactly one <think>…</think> before <answer>…</answer>."""
    opens_think = len(re.findall(r"(?i)<think>", full_text))
    closes_think = len(re.findall(r"(?i)</think>", full_text))
    opens_ans = len(re.findall(r"(?i)<answer>", full_text))
    closes_ans = len(re.findall(r"(?i)</answer>", full_text))
    if not (opens_think == closes_think == 1 and opens_ans == closes_ans == 1):
        return False
    think_open_match = re.search(r"(?i)<think>", full_text)
    think_close_match = re.search(r"(?i)</think>", full_text)
    answer_open_match = re.search(r"(?i)<answer>", full_text)
    answer_close_match = re.search(r"(?i)</answer>", full_text)
    if not all(
        [think_open_match, think_close_match, answer_open_match, answer_close_match],
    ):
        return False
    think_open_pos = think_open_match.start()  # type: ignore[union-attr]
    think_close_pos = think_close_match.start()  # type: ignore[union-attr]
    answer_open_pos = answer_open_match.start()  # type: ignore[union-attr]
    answer_close_pos = answer_close_match.start()  # type: ignore[union-attr]
    return think_open_pos < think_close_pos < answer_open_pos < answer_close_pos


# ---------------------------------------------------------------------------
# Torch helpers
# ---------------------------------------------------------------------------
def move_inputs_to_device(
    inputs: dict,
    device: Optional[torch.device] = None,
) -> tuple[dict, torch.Tensor]:
    """Move a HuggingFace-style inputs dict to CUDA (or provided device).

    Returns the computed attention-mask lengths alongside the moved inputs.
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


def tokenize_prefixes_for_generate(
    tokenizer,
    prefixes: Sequence[str],
    *,
    max_length: int = 4096,
    device: Optional[torch.device] = None,
) -> tuple[dict, torch.Tensor]:
    """
    Tokenize a list of string prefixes for generation and move to device.

    This centralizes the common pattern used across math/carpark/crossword
    inference loops.
    """
    inputs = tokenizer(
        prefixes,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return move_inputs_to_device(inputs, device=device)


def build_generate_kwargs(
    *,
    cap: int,
    pad_token_id: int,
    eos_ids,
    entropy_mode: str,
    temperature: Optional[float],
    top_p: Optional[float],
    synced_gpus: bool = False,
) -> Dict[str, Any]:
    """
    Build generate() kwargs for a given token cap and sampling configuration.

    If temperature <= 0 → greedy (do_sample=False) and omit temperature/top_p.
    Else → sampling with provided temperature/top_p.
    """
    do_sample = temperature is not None and float(temperature) > 0.0
    kwargs: Dict[str, Any] = {
        "max_new_tokens": cap,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_ids,
        "do_sample": do_sample,
        "return_dict_in_generate": True,
        "output_scores": entropy_mode != "none",
        "num_return_sequences": 1,
    }
    if synced_gpus and hasattr(torch, "distributed"):
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                kwargs["synced_gpus"] = True
        except Exception:  # pragma: no cover - defensive
            pass
    if do_sample:
        kwargs["temperature"] = float(temperature)
        if top_p is not None:
            kwargs["top_p"] = float(top_p)
    return kwargs


def decode_generated_row(tokenizer, seqs: torch.Tensor, input_lengths: torch.Tensor, row_i: int,
                         *, skip_special_tokens: bool = True) -> Tuple[torch.Tensor, str, int]:
    """
    Given batched generation outputs, return (gen_ids, decoded_text, start_tok_idx)
    for a single row. This de-duplicates the common indexing/decoding pattern.
    """
    start_tok_idx = int(input_lengths[row_i].item())
    gen_ids = seqs[row_i, start_tok_idx:]
    raw_txt = tokenizer.decode(gen_ids, skip_special_tokens=skip_special_tokens)
    return gen_ids, raw_txt, start_tok_idx


def _trim_and_classify(
    gen_ids: torch.Tensor,
    raw_text: str,
    stop_strings: Sequence[str],
    cap: int,
    eos_ids: Optional[Sequence[int]],
) -> Tuple[str, str]:
    """
    Helper to trim on stop strings and classify stop reason for a single row.
    """
    active_stop_strings = list(stop_strings) or []
    found_stop = any(stop_str in raw_text for stop_str in active_stop_strings)
    has_eos = False
    if eos_ids:
        for eos_token_id in eos_ids:
            if (gen_ids == eos_token_id).any():
                has_eos = True
                break
    hit_max = len(gen_ids) >= cap
    stop_reason = classify_stop_reason(found_stop, has_eos, hit_max)

    trimmed = raw_text
    for stop_str in active_stop_strings:
        if stop_str in trimmed:
            trimmed = trimmed.split(stop_str, 1)[0]
            break
    return trimmed.strip(), stop_reason


def _row_entropy_from_scores(
    scores,
    sequences: torch.Tensor,
    row_index: int,
    start_tok_idx: int,
    eos_ids: Optional[Sequence[int]],
    model,
) -> List[float]:
    """
    Compute per-token entropies for a single row, with fallback to
    entropy_from_start_index on NaNs/Infs.
    """
    gen_ids = sequences[row_index, start_tok_idx:]
    scores_len = len(scores)
    eos_limit = first_eos_any(gen_ids, eos_ids) if eos_ids else gen_ids.shape[0]
    t_stop = min(eos_limit, scores_len)
    token_entropies: List[float] = []
    bad = False
    for score_index in range(t_stop):
        logits = scores[score_index][row_index].float()
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            bad = True
            break
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
            bad = True
            break
        probabilities = log_probs.exp()
        entropy_val = float(-(probabilities * log_probs).sum().item())
        if not math.isfinite(entropy_val):
            bad = True
            break
        token_entropies.append(entropy_val)

    if bad or not token_entropies:
        start_index = start_tok_idx - 1
        token_entropies = entropy_from_start_index(
            model,
            sequences[row_index : row_index + 1],
            start_index,
        ) or []
    return token_entropies


def decode_and_score_batch(
    *,
    tokenizer,
    sequences: torch.Tensor,
    scores,
    input_lengths: torch.Tensor,
    stop_strings: Sequence[str],
    cap: int,
    eos_ids: Optional[Sequence[int]],
    entropy_mode: str,
    model,
) -> Tuple[List[str], List[List[float]], List[str]]:
    """
    Shared row-wise decode + entropy loop used by math/carpark/crossword _gen_batch
    helpers. Returns (decoded_texts, entropy_series, stop_reasons).
    """
    total_rows = sequences.shape[0]
    decoded_texts: List[str] = []
    entropy_series: List[List[float]] = []
    stop_reasons: List[str] = []

    for row_index in range(total_rows):
        start_tok_idx = int(input_lengths[row_index].item())
        gen_ids = sequences[row_index, start_tok_idx:]
        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        trimmed, stop_reason = _trim_and_classify(
            gen_ids,
            raw_text,
            stop_strings,
            cap,
            eos_ids,
        )
        decoded_texts.append(trimmed)
        stop_reasons.append(stop_reason)

        if entropy_mode == "none":
            entropy_series.append([])
            continue

        entropies = _row_entropy_from_scores(
            scores,
            sequences,
            row_index,
            start_tok_idx,
            eos_ids,
            model,
        )
        entropy_series.append(entropies)

    return decoded_texts, entropy_series, stop_reasons


def classify_stop_reason(found_stop: bool, has_eos: bool, hit_max: bool) -> str:
    """
    Map boolean stop conditions into a standardized stop_reason string.
    """
    if found_stop:
        return "stop_token"
    if has_eos:
        return "eos"
    if hit_max:
        return "max_new_tokens"
    return "other"


class StopOnSubstrings(StoppingCriteria):
    """Stop generation when any of the provided substrings is seen."""

    def __init__(self, tokenizer, stops: List[str]):
        self.stop_ids = [
            tokenizer.encode(text, add_special_tokens=False) for text in stops
        ]

    @staticmethod
    def _endswith(sequence: torch.Tensor, suffix_ids: List[int]) -> bool:
        return (
            len(sequence) >= len(suffix_ids)
            and sequence[-len(suffix_ids) :].tolist() == suffix_ids
        )

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,  # noqa: ARG002 - interface requirement
        **kwargs: Any,  # noqa: ARG002 - interface requirement
    ) -> bool:
        for row in input_ids:
            for stop_id_seq in self.stop_ids:
                if stop_id_seq and self._endswith(row, stop_id_seq):
                    return True
        return False

    def has_stops(self) -> bool:
        """Return True if any stop sequences are configured."""
        return bool(self.stop_ids)


def build_entropy_pass_base(
    *,
    prev_output: Optional[str],
    full_text: str,
    pred_answer_text: str,
    pred_canon: Optional[str],
    entropy_overall: Optional[float],
    entropy_think: Optional[float],
    entropy_answer: Optional[float],
) -> Dict[str, Any]:
    """
    Common core for per-pass result dicts with entropy stats and prediction fields.
    """
    return {
        "prev_output": prev_output,
        "output": full_text,
        "pred_answer": pred_answer_text,
        "pred_answer_canon": pred_canon,
        "entropy": entropy_overall,
        "entropy_think": entropy_think,
        "entropy_answer": entropy_answer,
    }


def add_token_and_tag_fields(
    base: Dict[str, Any],
    *,
    tokens_total: int,
    tokens_think: int,
    tokens_answer: int,
    full_text: str,
) -> Dict[str, Any]:
    """
    Add shared token-count and tag-structure fields to a per-pass result dict.
    """
    base.update(
        {
            "tokens_total": tokens_total,
            "tokens_end_think": tokens_think,
            "tokens_think": tokens_think,
            "tokens_answer": tokens_answer,
            "valid_tag_structure": valid_tag_structure(full_text),
        },
    )
    return base


def find_markers_and_context(
    think_text: Optional[str],
    prompt_text: str,
    patterns: Sequence[Tuple[str, re.Pattern]],
    *,
    skip_prefix_chars: int = 0,
):  # pylint: disable=too-many-locals
    """
    Scan think_text for the earliest match among patterns.
    Returns (markers, earliest_pos, context_prefix, excerpt).
    """
    if not think_text:
        return [], None, None, None
    search_text = think_text[skip_prefix_chars:] if skip_prefix_chars > 0 else think_text
    earliest_pos = None
    markers: List[str] = []
    for name, pattern in patterns:
        match = pattern.search(search_text)
        if match:
            markers.append(name)
            pos_global = (
                skip_prefix_chars + match.start()
                if skip_prefix_chars > 0
                else match.start()
            )
            if earliest_pos is None or pos_global < earliest_pos:
                earliest_pos = pos_global
    if not markers:
        return [], None, None, None
    prefix = think_text[:earliest_pos] if earliest_pos is not None else think_text
    context = f"{prompt_text}\n\n{prefix}"
    window_start = max(0, (earliest_pos or 0) - 60)
    window_end = min(len(think_text), (earliest_pos or 0) + 60)
    excerpt = think_text[window_start:window_end]
    return markers, earliest_pos, context, excerpt


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------
RE_LATEX_FRAC = re.compile(r"\\frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}", re.I)
RE_LATEX_CMDS = re.compile(r"\\(left|right|,|;|!|:)", re.I)
RE_SPACES = re.compile(r"\s+")
RE_BRACES = re.compile(r"[{}]")
RE_PARENS_COMMAs = re.compile(r"[()\[\],]")


def canon_math(value: Optional[str]) -> Optional[str]:
    """
    Permissive canonicalizer for math answers. Lowercases, removes spacing/punctuation,
    simplifies common LaTeX forms, and normalizes pi.
    """
    if value is None:
        return None
    canonical = value.strip()
    canonical = (
        canonical.replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .replace("π", "pi")
        .replace("\\pi", "pi")
    )
    canonical = RE_LATEX_CMDS.sub("", canonical)
    canonical = RE_LATEX_FRAC.sub(r"\1/\2", canonical)
    canonical = RE_BRACES.sub("", canonical)
    canonical = RE_SPACES.sub("", canonical)
    canonical = RE_PARENS_COMMAs.sub("", canonical)
    canonical = canonical.replace("\\boxed", "").replace("$", "")
    canonical = canonical.lower().rstrip(".")
    canonical = re.sub(r"/{2,}", "/", canonical)
    canonical = re.sub(r"\+{2,}", "+", canonical)
    canonical = re.sub(r"-{2,}", "-", canonical)
    if canonical.startswith("+"):
        canonical = canonical[1:]
    return canonical


def contains_canon(hay: Optional[str], needle: Optional[str]) -> bool:
    """Substring check after both sides are canonicalized."""
    return bool(hay and needle and (needle in hay))


# ---------------------------------------------------------------------------
# Torch utilities
# ---------------------------------------------------------------------------
def finite_mean(values: Iterable[float]) -> Optional[float]:
    """Return the mean of finite values, ignoring NaNs."""
    finite_values = [
        float(value)
        for value in values
        if not math.isnan(float(value)) and math.isfinite(float(value))
    ]
    return (sum(finite_values) / len(finite_values)) if finite_values else None


def first_eos_any(token_ids: torch.Tensor, eos_id_list: Optional[Sequence[int]]) -> int:
    """Return the first EOS position in a sequence, or full length if absent."""
    if not eos_id_list:
        return token_ids.numel()
    hit_positions: List[int] = []
    for eos_token_id in eos_id_list:
        positions = (token_ids == eos_token_id).nonzero(as_tuple=False)
        if positions.numel() > 0:
            hit_positions.append(positions[0].item())
    return min(hit_positions) if hit_positions else token_ids.numel()


def entropy_from_start_index(model, seq_ids: torch.Tensor, start_idx: int) -> List[float]:
    """
    Compute token-wise entropy starting at position start_idx (inclusive).
    Safe for NaNs thanks to re-centering.
    """
    device = next(model.parameters()).device
    seq_ids = seq_ids.to(device)
    entropies: List[float] = []
    with torch.inference_mode():
        out = model(input_ids=seq_ids[:, : start_idx + 1], use_cache=True)
        past_key_values = out.past_key_values
        sequence_length = seq_ids.shape[1]
        for time_index in range(start_idx, sequence_length - 1):
            out = model(
                input_ids=seq_ids[:, time_index : time_index + 1],
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :].float()
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            probabilities = log_probs.exp()
            entropy_value = float(-(probabilities * log_probs).sum().item())
            if not math.isfinite(entropy_value):
                logits = (logits - logits.max()).float()
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                probabilities = log_probs.exp()
                entropy_value = float(-(probabilities * log_probs).sum().item())
            entropies.append(entropy_value)
    return entropies


# ---------------------------------------------------------------------------
# HF cache helpers
# ---------------------------------------------------------------------------
def setup_hf_cache_dir_env(base_dir: str = "./.hf_cache") -> str:
    """
    Initialize HuggingFace cache directory and environment variables.

    Returns the absolute HF cache directory path so callers can pass it to
    transformers / datasets loaders.
    """
    hf_cache_dir = os.path.abspath(base_dir)
    os.environ.update(
        HF_HOME=hf_cache_dir,
        TRANSFORMERS_CACHE=os.path.join(hf_cache_dir, "transformers"),
        HF_HUB_CACHE=os.path.join(hf_cache_dir, "hub"),
    )
    return hf_cache_dir


def setup_script_logger(name: str) -> logging.Logger:
    """
    Configure a basic process-wide logger using LOGLEVEL env and return a module logger.

    This mirrors the common pattern used in the inference entrypoints.
    """
    loglevel = os.getenv("LOGLEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, loglevel, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------
def load_local_json_dataset(path: str) -> "Dataset":
    """
    Read a JSONL-like local file into a datasets.Dataset.
    Lines that are empty or not JSON objects are skipped.
    """
    dataset_cls, _ = require_datasets()

    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            line = line.strip()
            if not line:
                continue
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(obj)
    return dataset_cls.from_list(records)


# ---------------------------------------------------------------------------
# JSONL + dataset field helpers
# ---------------------------------------------------------------------------
def append_jsonl_row(path: str, row: Dict[str, Any]) -> None:
    """
    Append a single JSON-serializable row to a JSONL file, creating parent
    directories as needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        json.dump(row, handle, ensure_ascii=False)
        handle.write("\n")


def scan_existing_problem_samples(path: str) -> Dict[str, set]:
    """
    Scan an existing JSONL results file and return a mapping
    {problem -> {sample_idx,...}}.
    """
    if not os.path.exists(path):
        return {}
    existing: Dict[str, set] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            problem = obj.get("problem")
            sample_idx = obj.get("sample_idx")
            if problem is None or sample_idx is None:
                continue
            existing.setdefault(problem, set()).add(int(sample_idx))
    return existing


def scan_existing_pass1_results(
    path: str,
) -> tuple[DefaultDict[str, set], Dict[tuple, str]]:
    """
    Scan a JSONL results file and recover seen problems and pass-1 outputs.

    Returns:
      existing_samples: problem -> set(sample_idx) that already exist
      existing_pass1: (problem, sample_idx) -> pass1['output'] text (if available)
    """
    existing_samples: DefaultDict[str, set] = defaultdict(set)
    existing_pass1: Dict[tuple, str] = {}
    for obj in iter_jsonl_objects(path):
        prob = obj.get("problem")
        sample_idx = obj.get("sample_idx")
        if prob is None or sample_idx is None:
            continue
        existing_samples[prob].add(int(sample_idx))
        pass1_section = obj.get("pass1") or {}
        pass1_output = pass1_section.get("output")
        if isinstance(pass1_output, str):
            existing_pass1[(prob, int(sample_idx))] = pass1_output
    return existing_samples, existing_pass1


def extract_problem_and_answer(example: Dict[str, Any]) -> Tuple[Optional[str], Any]:
    """
    Extract a unified (problem, answer) pair from a heterogeneous example dict.
    """
    problem = (
        example.get("problem")
        or example.get("question")
        or example.get("prompt")
        or example.get("instruction")
        or example.get("query")
    )
    answer = (
        example.get("answer")
        or example.get("solution")
        or example.get("final_answer")
        or example.get("boxed_answer")
        or example.get("target")
    )
    return problem, answer


def build_math_gateway_row_base(
    *,
    problem: str,
    gold_answer: Any,
    gold_answer_canon: Any,
    split: str,
    step: int,
    sample_idx: int,
) -> Dict[str, Any]:
    """
    Common prefix for single-pass MATH JSONL rows used by gateway scripts.
    """
    return {
        "problem": problem,
        "gold_answer": gold_answer,
        "gold_answer_canon": gold_answer_canon,
        "split": split,
        "step": step,
        "sample_idx": sample_idx,
    }


def build_usage_dict(usage: Any) -> Dict[str, Any]:
    """
    Build a usage dict from an OpenAI/Portkey-style usage object, tolerating
    missing attributes.
    """
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def iter_jsonl_objects(path: str) -> Iterable[dict]:
    """
    Yield JSON objects from a JSONL file, skipping empty or invalid lines.
    """
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield obj


def build_two_pass_row_base(
    *,
    step: int,
    split_name: str,
    sample_idx: int,
    pass1: Dict[str, Any],
    pass2: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Shared core fields for two-pass rows (carpark/math-llama style).
    """
    return {
        "step": step,
        "split": split_name,
        "sample_idx": sample_idx,
        "pass1": pass1,
        "pass2": pass2,
    }


@dataclass
class PassOutputs:
    """Container for per-pass outputs used when writing rows."""

    full_texts: List[str]
    ent_think: List[List[float]]
    ent_answer: List[List[float]]
    stop_reason_think: List[str]
    stop_reason_answer: List[str]


def limit_dataset_examples(dataset, num_examples: Optional[int]):
    """
    If num_examples is set and positive, return a sliced dataset with at most that
    many rows; otherwise return the dataset unchanged.
    """
    if num_examples is not None and num_examples > 0:
        return dataset.select(range(min(num_examples, len(dataset))))
    return dataset


def prepare_math_gateway_dataset(
    *,
    dataset_id: str,
    split: str,
    seed: int,
    num_examples: Optional[int],
    dataset_path: Optional[str],
    outpath: str,
    logger: logging.Logger,
    load_math500_fn,
    load_remote_dataset_fn,
    cache_dir: Optional[str] = None,
):
    """
    Load a MATH-style dataset (local MATH-500 or remote HF path), optionally cap
    the number of examples, shuffle, scan existing results, and log summary
    stats. Returns (dataset, existing_problem_samples, dataset_name_for_log).
    """
    hf_cache_dir = cache_dir or os.path.abspath("./.hf_cache")
    if dataset_id.upper() == "MATH-500":
        dataset = load_math500_fn(hf_cache_dir, split, seed, dataset_path=dataset_path)
        dataset_name_for_log = "MATH-500"
    else:
        dataset = load_remote_dataset_fn(dataset_id, split, hf_cache_dir)
        dataset_name_for_log = dataset_id

    dataset = limit_dataset_examples(dataset, num_examples)
    dataset = dataset.shuffle(seed=seed)
    existing = scan_existing_problem_samples(outpath)
    logger.info(
        "Dataset: %s split=%s | N=%d | existing=%d",
        dataset_name_for_log,
        split,
        len(dataset),
        len(existing),
    )
    logger.info("Output: %s", outpath)
    return dataset, existing, dataset_name_for_log


def prepare_math_gateway_dataset_from_args(
    args,
    *,
    outpath: str,
    logger: logging.Logger,
    load_math500_fn,
    load_remote_dataset_fn,
    cache_dir: Optional[str] = None,
):
    """
    Convenience wrapper mapping common CLI args into prepare_math_gateway_dataset.
    """
    return prepare_math_gateway_dataset(
        dataset_id=args.dataset_id,
        split=args.split,
        seed=args.seed,
        num_examples=args.num_examples,
        dataset_path=args.dataset_path,
        outpath=outpath,
        logger=logger,
        load_math500_fn=load_math500_fn,
        load_remote_dataset_fn=load_remote_dataset_fn,
        cache_dir=cache_dir,
    )


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def add_basic_runner_args(arg_parser, *, default_dtype: str = "float16") -> None:
    """
    Attach common dataset/decoding/budget/system flags used by unified runners.
    """
    # Data selection
    arg_parser.add_argument("--split", default="test")
    arg_parser.add_argument("--num_examples", type=int, default=None)

    # Decoding + sampling
    arg_parser.add_argument("--batch_size", type=int, default=8)
    arg_parser.add_argument("--num_samples", type=int, default=1)
    arg_parser.add_argument("--temperature", type=float, default=0.0)
    arg_parser.add_argument("--top_p", type=float, default=0.95)

    # Budgets (per pass)
    arg_parser.add_argument("--think_cap", type=int, default=750)
    arg_parser.add_argument("--answer_cap", type=int, default=50)

    # System/runtime
    arg_parser.add_argument("--dtype", default=default_dtype, choices=["float16", "bfloat16"])


def add_model_and_output_args(arg_parser) -> None:
    """Attach model + output_dir arguments shared by unified runners."""
    arg_parser.add_argument("--model_name_or_path", required=True)
    arg_parser.add_argument("--revision")
    arg_parser.add_argument("--output_dir", required=True)


def add_math_gateway_dataset_args(arg_parser) -> None:
    """
    Attach dataset selection arguments shared by simple math gateway scripts
    (Azure/OpenRouter/Portkey-style single-pass runners).
    """
    arg_parser.add_argument(
        "--dataset_id",
        default="MATH-500",
        help="Use 'MATH-500' or a HF dataset path.",
    )
    arg_parser.add_argument(
        "--dataset_path",
        default=None,
        help="Optional local JSONL for MATH-500-style records.",
    )
    arg_parser.add_argument("--split", default="test")
    arg_parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Optional cap (<500).",
    )
    arg_parser.add_argument("--num_samples", type=int, default=1)


def build_math_gateway_arg_parser(
    *,
    default_temperature: float,
    description: Optional[str] = None,
) -> argparse.ArgumentParser:
    """
    Construct an ArgumentParser with shared math gateway arguments.

    This includes output_dir, dataset selection, sampling/budget knobs,
    and basic seed/step controls. Caller should attach backend-specific
    arguments (Azure/OpenRouter/Portkey) on top.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Root directory for JSONL outputs.",
    )
    add_math_gateway_dataset_args(parser)
    add_math_gateway_sampling_args(parser, default_temperature=default_temperature)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step", type=int, default=0)
    return parser


def configure_unified_runner_common(arg_parser, *, default_dtype: str) -> None:
    """
    Attach the shared runtime/entropy/two-pass flags used by unified runners.
    """
    add_basic_runner_args(arg_parser, default_dtype=default_dtype)
    arg_parser.add_argument("--step", type=int, default=0)
    arg_parser.add_argument("--tokenizer_path", default=None)
    arg_parser.add_argument("--seed", type=int, default=42)
    arg_parser.add_argument(
        "--entropy_mode",
        choices=["full", "reconsider", "none"],
        default="reconsider",
    )
    arg_parser.add_argument(
        "--attn_implementation",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
    )
    add_two_pass_args(arg_parser)


@dataclass
class MathPassMeta:
    """Metadata required to pack a single math pass result."""

    problem: str
    canon_gold: Optional[str]
    injected_cue: bool
    prev_output: Optional[str]
    cue_prefix_str: str
    stop_reason_think: Optional[str]
    stop_reason_answer: Optional[str]


@dataclass
class MathTokenStats:
    """Token-count summary for a single math pass."""

    tokens_think: int
    tokens_answer: int
    tokens_total: int


@dataclass
class MathEntropySummary:
    """Entropy statistics for a single math pass."""

    overall: Optional[float]
    think: Optional[float]
    answer: Optional[float]
    pre_cue: Optional[float]
    reconsider_think: Optional[float]
    reconsider_full: Optional[float]


@dataclass
class MathReconsiderationInfo:
    """Reconsideration markers and positions for a math pass."""

    markers: List[str]
    pos_in_think: Optional[int]
    context: Optional[str]
    excerpt: Optional[str]
    t_cue: Optional[int]


def _compute_math_token_stats(
    ent_think: List[float],
    ent_answer: List[float],
) -> Tuple[List[float], MathTokenStats]:
    """Return combined entropy series and basic token counts."""
    tok_ents_all = (ent_think or []) + (ent_answer or [])
    tokens_think = len(ent_think or [])
    tokens_answer = len(ent_answer or [])
    tokens_total = len(tok_ents_all)
    return tok_ents_all, MathTokenStats(
        tokens_think=tokens_think,
        tokens_answer=tokens_answer,
        tokens_total=tokens_total,
    )


def _compute_math_reconsideration_info(
    think_text: str,
    meta: MathPassMeta,
    tokens_think: int,
) -> MathReconsiderationInfo:
    """Derive reconsideration markers, context, and cue index from metadata."""
    skip_chars = len(meta.cue_prefix_str) if meta.injected_cue else 0
    markers, pos_in_think, reconsider_context, reconsider_excerpt = find_markers_and_context(
        think_text,
        f"Problem: {meta.problem}",
        RECONSIDER_PATTERNS,
        skip_prefix_chars=skip_chars,
    )
    if meta.injected_cue:
        markers = ["injected_cue"] + (markers or [])

    t_cue = 0 if meta.injected_cue else None
    if (not meta.injected_cue) and (pos_in_think is not None):
        t_cue = max(0, min(pos_in_think, tokens_think))

    return MathReconsiderationInfo(
        markers=markers or [],
        pos_in_think=pos_in_think,
        context=reconsider_context,
        excerpt=reconsider_excerpt,
        t_cue=t_cue,
    )


def _summarize_math_entropies(
    tok_ents_all: List[float],
    ent_think: List[float],
    ent_answer: List[float],
    tokens_think: int,
    tokens_total: int,
    t_cue: Optional[int],
) -> MathEntropySummary:
    """Compute overall and segment-wise entropy summaries for a math pass."""
    entropy_overall = finite_mean(tok_ents_all) if tok_ents_all else None
    entropy_think = finite_mean(ent_think) if ent_think else None
    entropy_answer = finite_mean(ent_answer) if ent_answer else None
    entropy_pre_cue = None
    entropy_reconsider_think = None
    entropy_reconsider_full = None

    if t_cue is not None:
        if tokens_total > t_cue:
            entropy_reconsider_full = finite_mean(tok_ents_all[t_cue:])
        if tokens_think > t_cue:
            entropy_reconsider_think = finite_mean(tok_ents_all[t_cue:tokens_think])

    return MathEntropySummary(
        overall=entropy_overall,
        think=entropy_think,
        answer=entropy_answer,
        pre_cue=entropy_pre_cue,
        reconsider_think=entropy_reconsider_think,
        reconsider_full=entropy_reconsider_full,
    )


def build_math_pass_meta(
    *,
    problem: str,
    canon_gold: Optional[str],
    injected_cue: bool,
    prev_output: Optional[str],
    cue_prefix_str: str,
    stop_reason_think: Optional[str],
    stop_reason_answer: Optional[str],
) -> MathPassMeta:
    """Helper to construct MathPassMeta consistently across math inference cores."""
    return MathPassMeta(
        problem=problem,
        canon_gold=canon_gold,
        injected_cue=injected_cue,
        prev_output=prev_output,
        cue_prefix_str=cue_prefix_str,
        stop_reason_think=stop_reason_think,
        stop_reason_answer=stop_reason_answer,
    )


def pack_math_pass_result(
    full_text: str,
    ent_think: List[float],
    ent_answer: List[float],
    meta: MathPassMeta,
) -> Dict[str, Any]:
    """
    Assemble per-pass math result dict with entropy, reconsideration markers,
    and token/tag statistics.
    """
    tok_ents_all, token_stats = _compute_math_token_stats(ent_think, ent_answer)
    think, answer = extract_blocks(full_text)
    think_text = think or ""
    pred_answer_text = answer or ""

    reconsider_info = _compute_math_reconsideration_info(
        think_text,
        meta,
        token_stats.tokens_think,
    )
    entropy_summary = _summarize_math_entropies(
        tok_ents_all=tok_ents_all,
        ent_think=ent_think,
        ent_answer=ent_answer,
        tokens_think=token_stats.tokens_think,
        tokens_total=token_stats.tokens_total,
        t_cue=reconsider_info.t_cue,
    )

    pred_canon = canon_math(pred_answer_text)
    is_correct_pred = contains_canon(pred_canon, meta.canon_gold)

    base = build_entropy_pass_base(
        prev_output=meta.prev_output,
        full_text=full_text,
        pred_answer_text=pred_answer_text,
        pred_canon=pred_canon,
        entropy_overall=entropy_summary.overall,
        entropy_think=entropy_summary.think,
        entropy_answer=entropy_summary.answer,
    )
    base.update(
        {
            "entropy_pre_cue": entropy_summary.pre_cue,
            "entropy_reconsider_think": entropy_summary.reconsider_think,
            "entropy_reconsider_full": entropy_summary.reconsider_full,
            "stop_reason_think": meta.stop_reason_think,
            "stop_reason_answer": meta.stop_reason_answer,
            "has_reconsider_cue": bool(reconsider_info.markers),
            "reconsider_markers": reconsider_info.markers,
            "reconsider_pos": reconsider_info.pos_in_think,
            "reconsider_context": reconsider_info.context,
            "reconsider_excerpt": reconsider_info.excerpt,
            "is_correct_pred": is_correct_pred,
            "is_correct_after_reconsideration": bool(reconsider_info.markers)
            and bool(is_correct_pred),
        },
    )
    return add_token_and_tag_fields(
        base,
        tokens_total=token_stats.tokens_total,
        tokens_think=token_stats.tokens_think,
        tokens_answer=token_stats.tokens_answer,
        full_text=full_text,
    )


def build_math_inference_config_kwargs(
    *,
    batch_size: int,
    num_samples: int,
    temperature: float,
    top_p: float,
    entropy_mode: str,
    eos_ids,
    two_pass: bool,
    second_pass_phrase: str,
    second_pass_use_sample_idx: int,
    think_cap: int,
    answer_cap: int,
) -> Dict[str, Any]:
    """
    Build the common kwargs dict for math-style inference configs / loops.
    """
    return {
        "batch_size": batch_size,
        "num_samples": num_samples,
        "temperature": temperature,
        "top_p": top_p,
        "entropy_mode": entropy_mode,
        "eos_ids": eos_ids,
        "two_pass": two_pass,
        "second_pass_phrase": second_pass_phrase,
        "second_pass_use_sample_idx": second_pass_use_sample_idx,
        "think_cap": think_cap,
        "answer_cap": answer_cap,
    }


def build_math_inference_config_kwargs_from_args(args, eos_ids):
    """
    Map common CLI args into kwargs for math-style inference loops / configs.
    """
    return build_math_inference_config_kwargs(
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        entropy_mode=args.entropy_mode,
        eos_ids=eos_ids,
        two_pass=args.two_pass,
        second_pass_phrase=args.second_pass_phrase,
        second_pass_use_sample_idx=args.second_pass_use_sample_idx,
        think_cap=args.think_cap,
        answer_cap=args.answer_cap,
    )


def add_two_pass_args(arg_parser) -> None:
    """
    Attach the common two-pass control flags used by math/carpark/crossword runners.
    """
    arg_parser.add_argument("--two_pass", action="store_true")
    arg_parser.add_argument(
        "--second_pass_phrase",
        default="Wait, we need to reconsider. Let's think this through step by step.",
    )
    arg_parser.add_argument("--second_pass_use_sample_idx", type=int, default=0)


def require_datasets():
    """
    Import and return (Dataset, load_dataset), raising with a consistent message if unavailable.
    """
    try:
        datasets_mod = import_module("datasets")
    except ImportError:
        print("datasets is required: pip install datasets", file=sys.stderr)
        raise
    dataset_cls = getattr(datasets_mod, "Dataset")
    load_dataset_fn = getattr(datasets_mod, "load_dataset")
    return dataset_cls, load_dataset_fn


def build_eos_ids_from_tokenizer(tokenizer, extra_tokens: Sequence[str]) -> Optional[List[int]]:
    """
    Build a sorted list of EOS token IDs from a tokenizer, including its native
    eos_token_id and any additional tokens provided.
    """
    eos_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    for tok in extra_tokens:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.pad_token_id:
            eos_ids.add(int(tid))
    return sorted(eos_ids) if eos_ids else None


def configure_tokenizer_and_eos(
    tokenizer,
    *,
    extra_tokens: Sequence[str],
) -> Optional[List[int]]:
    """
    Apply standard padding/truncation settings and build an EOS-ID list.
    """
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.truncation_side = "left"
    return build_eos_ids_from_tokenizer(tokenizer, extra_tokens)


def init_unified_backend_and_eos(
    *,
    backend_cls,
    model_name_or_path: str,
    revision: Optional[str],
    cache_dir: str,
    dtype: str,
    device_map: str,
    attn_implementation: Optional[str],
    tokenizer_path: Optional[str],
):
    """
    Initialize a backend and a standard EOS-ID list for unified runners.

    `backend_cls` is typically HFBackend, but is passed explicitly so tests
    can monkeypatch it on the caller module.
    """
    backend = backend_cls.from_pretrained(
        model_name_or_path,
        revision=revision,
        cache_dir=cache_dir,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        tokenizer_path=tokenizer_path,
    )
    eos_ids = build_eos_ids_from_tokenizer(
        backend.tokenizer,
        extra_tokens=("<|im_end|>", "<|endoftext|>"),
    )
    return backend, eos_ids


def call_with_retries(
    func,
    *,
    max_retries: int,
    retry_backoff: float,
    logger: logging.Logger,
    sample_idx: int,
    problem_snippet: str,
    min_sleep: Optional[float] = None,
    exception_types: Sequence[type[BaseException]] = (Exception,),
):
    """
    Call `func()` with simple retry-on-exception semantics shared by math gateways.
    """
    attempt = 0
    while True:
        try:
            return func()
        except tuple(exception_types) as exc:
            attempt += 1
            if attempt > max_retries:
                logger.error(
                    "Failed after %d retries on sample_idx=%d | prob snippet=%.60s | err=%r",
                    attempt - 1,
                    sample_idx,
                    problem_snippet,
                    exc,
                )
                raise
            sleep_dur = retry_backoff * attempt
            if min_sleep is not None:
                sleep_dur = max(min_sleep, sleep_dur)
            logger.warning(
                "Retry %d/%d for sample_idx=%d after error: %r (sleep %.1fs)",
                attempt,
                max_retries,
                sample_idx,
                exc,
                sleep_dur,
            )
            time.sleep(sleep_dur)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
OPENR1_PROMPT_TEMPLATE = (
    "You are a helpful AI assistant. First think in the <think> block, then write "
    "ONLY the final answer in the <answer> block. Do NOT add anything after "
    "</answer>.\n\n"
    "Problem: {problem}\n\n"
    "<think>\n"
    "</think>\n\n"
    "<answer>\n"
    "</answer>"
)


def build_math_gateway_messages(system_prompt: str, problem: str) -> List[Dict[str, str]]:
    """
    Standard two-message chat for math gateway calls: system + user(problem).
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]


def iter_math_gateway_samples(
    dataset,
    num_samples: int,
    existing: Dict[str, set[int]],
) -> Iterable[Tuple[str, Any, int]]:
    """
    Yield (problem, gold_answer, sample_idx) triples for samples that still
    need generation, shared by math gateway scripts.
    """
    for example in dataset:
        problem, gold_answer = extract_problem_and_answer(example)
        if not problem or gold_answer is None:
            continue
        generated_indices = existing.get(problem, set())
        for sample_idx in range(num_samples):
            if sample_idx in generated_indices:
                continue
            yield problem, gold_answer, sample_idx


def parse_openai_chat_response(resp: Any) -> tuple[str, Any, Any]:
    """
    Extract (text, finish_reason, usage) from an OpenAI/OpenRouter/Portkey-style
    chat completion response object.
    """
    text = ""
    finish_reason = None
    if getattr(resp, "choices", None):
        choice = resp.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        message = getattr(choice, "message", None)
        text = getattr(message, "content", "") if message is not None else ""
    usage = getattr(resp, "usage", None)
    return text, finish_reason, usage


def add_math_gateway_sampling_args(
    parser,
    *,
    default_temperature: float,
) -> None:
    """
    Attach the shared sampling/budget args used by math gateway runners
    (OpenRouter/Portkey/Azure) on top of dataset args.
    """
    parser.add_argument("--temperature", type=float, default=default_temperature)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_output_tokens", type=int, default=900)
    parser.add_argument("--request_timeout", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step", type=int, default=0)
