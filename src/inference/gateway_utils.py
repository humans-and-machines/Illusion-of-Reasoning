#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset / JSONL / CLI / gateway helpers shared by inference scripts.

These utilities were previously part of ``inference.common`` and are now split
out to keep that module below lint size limits. Public functions are still
re-exported from ``inference.common`` for backwards compatibility.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from importlib import import_module
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple


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
        datefmt="%-Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Dataset + JSONL helpers
# ---------------------------------------------------------------------------
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


def append_jsonl_row(path: str, row: Dict[str, Any]) -> None:
    """
    Append a single JSON-serializable row to a JSONL file, creating parent
    directories as needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        json.dump(row, handle, ensure_ascii=False)
        handle.write("\n")


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
) -> Tuple[DefaultDict[str, set], Dict[Tuple[Any, Any], str]]:
    """
    Scan a JSONL results file and recover seen problems and pass-1 outputs.

    Returns:
      existing_samples: problem -> set(sample_idx) that already exist
      existing_pass1: (problem, sample_idx) -> pass1['output'] text (if available)
    """
    existing_samples: DefaultDict[str, set] = defaultdict(set)
    existing_pass1: Dict[Tuple[Any, Any], str] = {}
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
        dataset = load_remote_dataset_fn(
            dataset_id,
            split=split,
            cache_dir=hf_cache_dir,
        )
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


def load_remote_dataset_default(dataset_id: str, split: str, cache_dir: str):
    """Default HF datasets loader used by math gateway scripts."""
    _, load_dataset_fn = require_datasets()
    return load_dataset_fn(dataset_id, split=split, cache_dir=cache_dir)


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


# ---------------------------------------------------------------------------
# Tokenizer / backend helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Prompt templates and gateway helpers
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


def parse_openai_chat_response(resp: Any) -> Tuple[str, Any, Any]:
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


def call_with_gateway_retries(
    func,
    *,
    args: argparse.Namespace,
    logger: logging.Logger,
    sample_idx: int,
    problem_snippet: str,
    min_sleep: Optional[float] = None,
):
    """
    Convenience wrapper around ``call_with_retries`` using standard CLI args.

    This deduplicates the common pattern used by math gateway runners.
    """
    return call_with_retries(
        func,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
        logger=logger,
        sample_idx=sample_idx,
        problem_snippet=problem_snippet,
        min_sleep=min_sleep,
    )


__all__ = [
    "setup_hf_cache_dir_env",
    "setup_script_logger",
    "require_datasets",
    "load_local_json_dataset",
    "append_jsonl_row",
    "iter_jsonl_objects",
    "scan_existing_problem_samples",
    "scan_existing_pass1_results",
    "extract_problem_and_answer",
    "build_math_gateway_row_base",
    "build_usage_dict",
    "build_two_pass_row_base",
    "PassOutputs",
    "limit_dataset_examples",
    "prepare_math_gateway_dataset",
    "prepare_math_gateway_dataset_from_args",
    "load_remote_dataset_default",
    "add_basic_runner_args",
    "add_model_and_output_args",
    "add_math_gateway_dataset_args",
    "add_two_pass_args",
    "build_math_gateway_arg_parser",
    "configure_unified_runner_common",
    "build_eos_ids_from_tokenizer",
    "configure_tokenizer_and_eos",
    "init_unified_backend_and_eos",
    "OPENR1_PROMPT_TEMPLATE",
    "build_math_gateway_messages",
    "iter_math_gateway_samples",
    "parse_openai_chat_response",
    "add_math_gateway_sampling_args",
    "call_with_retries",
    "call_with_gateway_retries",
]
