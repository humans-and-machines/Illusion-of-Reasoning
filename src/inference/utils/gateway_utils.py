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
import contextlib
import fcntl
import json
import logging
import os
import sys
import time
from collections import defaultdict
from importlib import import_module
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from src.inference.utils.math_pass_utils import DEFAULT_SECOND_PASS_PHRASE
from src.inference.utils.gateway_dataset_config import (
    MathGatewayDatasetConfig,
    MathGatewayDatasetLimits,
    MathGatewayDatasetSource,
    PassOutputs,
)

# ---------------------------------------------------------------------------
def setup_hf_cache_dir_env(base_dir: str = "./.hf_cache") -> str:
    """
    Initialize HuggingFace cache directory and related environment variables.

    :param base_dir: Base directory to use for the HF cache (directories such
        as ``HF_HOME`` and ``TRANSFORMERS_CACHE`` will be created under this).
    :returns: Absolute path to the HF cache directory so callers can pass it
        to transformers / datasets loaders.
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

    :param name: Logger name (typically ``__name__`` of the calling module).
    :returns: Configured :class:`logging.Logger` instance.
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
    Import and return ``(Dataset, load_dataset)``, raising with a consistent message if unavailable.

    :returns: Tuple ``(DatasetCls, load_dataset_fn)`` from :mod:`datasets`.
    :raises ImportError: If the :mod:`datasets` package is not installed.
    """
    try:
        datasets_mod = import_module("datasets")
    except ImportError:
        print("datasets is required: pip install datasets", file=sys.stderr)
        raise
    dataset_cls = getattr(datasets_mod, "Dataset")
    load_dataset_fn = getattr(datasets_mod, "load_dataset")
    return dataset_cls, load_dataset_fn


@contextlib.contextmanager
def locked_file(path: str, mode: str, *, lock_type: int = fcntl.LOCK_SH):
    """
    Open a file with an fcntl lock, waiting until the peer releases it.
    """
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, mode, encoding="utf-8") as handle:
        fcntl.flock(handle, lock_type)
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def load_local_json_dataset(path: str) -> "Dataset":
    """
    Read a JSONL-like local file into a :class:`datasets.Dataset`.

    Lines that are empty or not JSON objects are skipped.

    :param path: Path to a JSONL (or JSONL-like) file on disk.
    :returns: Dataset constructed from the list of parsed JSON objects.
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

    :param path: Path to the JSONL file on disk.
    :param row: Mapping that can be serialized by :mod:`json`.
    :returns: ``None``. The row is appended to ``path``.
    """
    with locked_file(path, "a", lock_type=fcntl.LOCK_EX) as handle:
        json.dump(row, handle, ensure_ascii=False)
        handle.write("\n")


def iter_jsonl_objects(path: str) -> Iterable[dict]:
    """
    Yield JSON objects from a JSONL file, skipping empty or invalid lines.

    :param path: Path to a JSONL file on disk.
    :returns: Iterator over successfully parsed JSON objects.
    """
    if not os.path.exists(path):
        return
    with locked_file(path, "r", lock_type=fcntl.LOCK_SH) as handle:
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
    Scan an existing JSONL results file and return a mapping of seen samples.

    :param path: Path to a JSONL results file on disk.
    :returns: Mapping ``{problem -> {sample_idx, ...}}`` capturing existing rows.
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

    :param path: Path to a JSONL results file on disk.
    :returns: Tuple ``(existing_samples, existing_pass1)`` where
        ``existing_samples`` maps problems to the set of seen sample indices
        and ``existing_pass1`` maps ``(problem, sample_idx)`` pairs to the
        stored pass-1 output text.
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

    :param example: Example record with task-specific keys.
    :returns: Tuple ``(problem, answer)`` where either element may be ``None``.
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

    :param problem: Normalized problem text.
    :param gold_answer: Gold answer value (raw).
    :param gold_answer_canon: Canonicalized gold answer, if available.
    :param split: Dataset split name (for example, ``\"test\"``).
    :param step: Training or checkpoint step identifier.
    :param sample_idx: Sample index for this generation.
    :returns: Dictionary containing shared row fields.
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

    :param usage: Usage object exposing token-count attributes, or ``None``.
    :returns: Dictionary with ``prompt_tokens``, ``completion_tokens``, and
        ``total_tokens`` fields (possibly ``None``).
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

    :param step: Training or checkpoint step identifier.
    :param split_name: Dataset split name (for example, ``\"test\"``).
    :param sample_idx: Sample index for this generation.
    :param pass1: Packed pass-1 result dictionary.
    :param pass2: Packed pass-2 result dictionary, or ``None``.
    :returns: Dictionary with shared two-pass row fields.
    """
    return {
        "step": step,
        "split": split_name,
        "sample_idx": sample_idx,
        "pass1": pass1,
        "pass2": pass2,
    }


def limit_dataset_examples(
    dataset,
    num_examples: Optional[int],
    *,
    from_end: bool = False,
    start: Optional[int] = None,
):
    """
    If num_examples is set and positive, return a sliced dataset with at most that
    many rows; otherwise return the dataset unchanged.

    The slice location is determined as follows:
      - If ``start`` is not ``None``, take ``num_examples`` rows starting at that
        zero-based index (clamped to the dataset length).
      - Else if ``from_end`` is ``True``, take rows from the end of the dataset.
      - Otherwise, take rows from the beginning.

    :param dataset: Dataset object supporting ``select`` and ``len``.
    :param num_examples: Maximum number of examples to keep, or ``None``.
    :param from_end: If ``True`` and ``start`` is ``None``, select rows from the
        end instead of the start.
    :param start: Optional zero-based starting index for the slice.
    :returns: Original dataset or a truncated view.
    """
    if num_examples is None or num_examples <= 0:
        return dataset

    total = len(dataset)
    if total <= 0:
        return dataset

    count = min(num_examples, total)

    if start is not None:
        # Clamp start into [0, total]
        safe_start = max(min(start, total), 0)
        # Adjust count if the window would overflow.
        if safe_start + count > total:
            count = max(total - safe_start, 0)
        indices = range(safe_start, safe_start + count)
    elif from_end:
        safe_start = max(total - count, 0)
        indices = range(safe_start, safe_start + count)
    else:
        indices = range(count)

    return dataset.select(indices)


def limit_dataset_for_args(dataset, args):
    """
    Convenience wrapper around :func:`limit_dataset_examples` using runner args.

    This reads ``num_examples``, ``examples_from_end``, and ``dataset_start``
    attributes from ``args`` (using safe defaults when absent) and applies the
    corresponding slice to the dataset.

    :param dataset: Dataset object supporting ``select`` and ``len``.
    :param args: Namespace-like object with dataset-limiting attributes.
    :returns: Original dataset or a truncated view.
    """
    return limit_dataset_examples(
        dataset,
        getattr(args, "num_examples", None),
        from_end=getattr(args, "examples_from_end", False),
        start=getattr(args, "dataset_start", 0),
    )


def prepare_math_gateway_dataset(
    config: Optional[MathGatewayDatasetConfig] = None,
    **legacy_kwargs: Any,
):
    """
    Load a MATH-style dataset (local MATH-500 or remote HF path), optionally cap
    the number of examples, shuffle, scan existing results, and log summary
    stats. Returns (dataset, existing_problem_samples, dataset_name_for_log).

    This function accepts either a :class:`MathGatewayDatasetConfig` instance
    or the legacy keyword arguments used by older callers.

    :param config: Optional dataset configuration object.
    :param legacy_kwargs: Legacy keyword arguments (for backwards-compatibility)
        corresponding to the fields of :class:`MathGatewayDatasetConfig`.
    :returns: Tuple ``(dataset, existing, dataset_name_for_log)``.
    """
    if config is None:
        required_keys = [
            "dataset_id",
            "split",
            "seed",
            "num_examples",
            "dataset_path",
            "outpath",
            "logger",
            "load_math500_fn",
            "load_remote_dataset_fn",
        ]
        missing = [name for name in required_keys if name not in legacy_kwargs]
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise TypeError(
                f"prepare_math_gateway_dataset missing required arguments: "
                f"{missing_str}",
            )
        config = MathGatewayDatasetConfig(
            source=MathGatewayDatasetSource(
                dataset_id=legacy_kwargs["dataset_id"],
                split=legacy_kwargs["split"],
                dataset_path=legacy_kwargs["dataset_path"],
            ),
            limits=MathGatewayDatasetLimits(
                seed=legacy_kwargs["seed"],
                num_examples=legacy_kwargs["num_examples"],
                cache_dir=legacy_kwargs.get("cache_dir"),
                from_end=legacy_kwargs.get("from_end", False),
                start=legacy_kwargs.get("start", 0),
            ),
            outpath=legacy_kwargs["outpath"],
            logger=legacy_kwargs["logger"],
            load_math500_fn=legacy_kwargs["load_math500_fn"],
            load_remote_dataset_fn=legacy_kwargs["load_remote_dataset_fn"],
        )

    cache_dir = config.cache_dir or os.path.abspath("./.hf_cache")
    if config.dataset_id.upper() == "MATH-500":
        dataset = config.load_math500_fn(
            cache_dir,
            config.split,
            config.seed,
            dataset_path=config.dataset_path,
        )
        dataset_name_for_log = "MATH-500"
    else:
        dataset = config.load_remote_dataset_fn(
            config.dataset_id,
            split=config.split,
            cache_dir=cache_dir,
        )
        dataset_name_for_log = config.dataset_id

    dataset = limit_dataset_examples(
        dataset,
        config.num_examples,
        from_end=config.from_end,
        start=config.start if config.start is not None else 0,
    )
    dataset = dataset.shuffle(seed=config.seed)
    existing = scan_existing_problem_samples(config.outpath)
    config.logger.info(
        "Dataset: %s split=%s | N=%d | existing=%d",
        dataset_name_for_log,
        config.split,
        len(dataset),
        len(existing),
    )
    config.logger.info("Output: %s", config.outpath)
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
    Convenience wrapper mapping common CLI args into :func:`prepare_math_gateway_dataset`.

    :param args: Argument namespace with dataset-related fields.
    :param outpath: Path to output JSONL file (used to scan existing rows).
    :param logger: Logger used for informational messages.
    :param load_math500_fn: Callable that loads the MATH-500 dataset.
    :param load_remote_dataset_fn: Callable that loads a remote dataset.
    :param cache_dir: Optional HF cache directory; a default is used if ``None``.
    :returns: Tuple ``(dataset, existing, dataset_name_for_log)``.
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
        from_end=getattr(args, "examples_from_end", False),
        start=getattr(args, "dataset_start", 0),
    )


def load_remote_dataset_default(dataset_id: str, split: str, cache_dir: str):
    """
    Default HF datasets loader used by math gateway scripts.

    :param dataset_id: Hugging Face dataset identifier.
    :param split: Dataset split name (for example, ``\"test\"``).
    :param cache_dir: Directory to use as a datasets cache.
    :returns: Dataset object returned by :func:`datasets.load_dataset`.
    """
    _, load_dataset_fn = require_datasets()
    return load_dataset_fn(dataset_id, split=split, cache_dir=cache_dir)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def add_basic_runner_args(arg_parser, *, default_dtype: str = "float16") -> None:
    """
    Attach common dataset/decoding/budget/system flags used by unified runners.

    :param arg_parser: Argument parser to which options are added.
    :param default_dtype: Default dtype string (for example, ``\"float16\"``).
    :returns: ``None``. Arguments are added to ``arg_parser`` in place.
    """
    # Data selection
    arg_parser.add_argument("--split", default="test")
    arg_parser.add_argument("--num_examples", type=int, default=None)
    arg_parser.add_argument(
        "--dataset_start",
        type=int,
        default=0,
        help=(
            "Zero-based starting index within the split from which to take "
            "examples (used together with num_examples)."
        ),
    )
    arg_parser.add_argument(
        "--examples_from_end",
        action="store_true",
        help=(
            "If set together with num_examples, take examples from the end "
            "of the split instead of the beginning."
        ),
    )

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
    """
    Attach ``model_name_or_path`` and ``output_dir`` arguments shared by unified runners.

    :param arg_parser: Argument parser to which options are added.
    :returns: ``None``. Arguments are added to ``arg_parser`` in place.
    """
    arg_parser.add_argument("--model_name_or_path", required=True)
    arg_parser.add_argument("--revision")
    arg_parser.add_argument("--output_dir", required=True)


def add_math_gateway_dataset_args(arg_parser) -> None:
    """
    Attach dataset selection arguments shared by simple math gateway scripts
    (Azure/OpenRouter/Portkey-style single-pass runners).

    :param arg_parser: Argument parser to which dataset arguments are added.
    :returns: ``None``. Arguments are added to ``arg_parser`` in place.
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
    arg_parser.add_argument(
        "--dataset_start",
        type=int,
        default=0,
        help=(
            "Zero-based starting index within the split from which to take "
            "examples (used together with num_examples)."
        ),
    )
    arg_parser.add_argument(
        "--examples_from_end",
        action="store_true",
        help=(
            "If set together with num_examples, take examples from the end "
            "of the split instead of the beginning."
        ),
    )
    arg_parser.add_argument("--num_samples", type=int, default=1)


def add_two_pass_args(arg_parser) -> None:
    """
    Attach the common two-pass control flags used by math/carpark/crossword runners.

    :param arg_parser: Argument parser to which two-pass arguments are added.
    :returns: ``None``. Arguments are added to ``arg_parser`` in place.
    """
    arg_parser.add_argument("--two_pass", action="store_true")
    arg_parser.add_argument(
        "--second_pass_phrase",
        default=DEFAULT_SECOND_PASS_PHRASE,
    )
    arg_parser.add_argument("--second_pass_use_sample_idx", type=int, default=0)


def build_math_gateway_arg_parser(
    *,
    default_temperature: float,
    description: Optional[str] = None,
) -> argparse.ArgumentParser:
    """
    Construct an ArgumentParser with shared math gateway arguments.

    This includes ``output_dir``, dataset selection, sampling/budget knobs,
    and basic seed/step controls. Caller should attach backend-specific
    arguments (Azure/OpenRouter/Portkey) on top.

    :param default_temperature: Default temperature for sampling options.
    :param description: Optional help text for the parser.
    :returns: Configured :class:`argparse.ArgumentParser` instance.
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

    :param arg_parser: Argument parser to which arguments are added.
    :param default_dtype: Default dtype string (for example, ``\"float16\"``).
    :returns: ``None``. Arguments are added to ``arg_parser`` in place.
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

    :param tokenizer: Tokenizer providing ``eos_token_id``, ``pad_token_id``,
        and ``convert_tokens_to_ids``.
    :param extra_tokens: Additional string tokens to treat as EOS.
    :returns: Sorted list of EOS token IDs, or ``None`` if none are found.
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

    :param tokenizer: Tokenizer to configure for left padding/truncation.
    :param extra_tokens: Additional string tokens to treat as EOS.
    :returns: Sorted list of EOS token IDs, or ``None`` if none are found.
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

    ``backend_cls`` is typically :class:`HFBackend`, but is passed explicitly
    so tests can monkeypatch it on the caller module.

    :param backend_cls: Backend class exposing ``from_pretrained``.
    :param model_name_or_path: Model identifier or path to load.
    :param revision: Optional model revision or commit hash.
    :param cache_dir: Directory to use as a HF cache.
    :param dtype: String alias for desired torch dtype (for example, ``\"float16\"``).
    :param device_map: Device mapping string passed to ``from_pretrained``.
    :param attn_implementation: Optional attention implementation override.
    :param tokenizer_path: Optional separate tokenizer identifier/path.
    :returns: Tuple ``(backend, eos_ids)`` where ``eos_ids`` may be ``None``.
    """
    backend = backend_cls.from_pretrained(
        model_name_or_path=model_name_or_path,
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

    :param system_prompt: System prompt string to use for the chat.
    :param problem: Problem text to inject into the user message.
    :returns: List of role/content message dictionaries.
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

    :param dataset: Dataset object containing math examples.
    :param num_samples: Number of samples to generate per problem.
    :param existing: Mapping from problem to the set of existing sample indices.
    :returns: Iterator over ``(problem, gold_answer, sample_idx)`` tuples.
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

    :param resp: Response object returned by an OpenAI-compatible client.
    :returns: Tuple ``(text, finish_reason, usage)`` extracted from the response.
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

    :param parser: Argument parser to which sampling arguments are added.
    :param default_temperature: Default temperature for the ``--temperature`` option.
    :returns: ``None``. Arguments are added to ``parser`` in place.
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
    Call ``func()`` with simple retry-on-exception semantics shared by math gateways.

    :param func: Zero-argument callable to execute.
    :param max_retries: Maximum number of retries before giving up.
    :param retry_backoff: Base backoff in seconds (multiplied by attempt index).
    :param logger: Logger used for warnings and errors.
    :param sample_idx: Sample index used for logging context.
    :param problem_snippet: Short snippet of the problem text for logging.
    :param min_sleep: Optional minimum sleep duration between retries.
    :param exception_types: Sequence of exception types that trigger retries.
    :returns: Result of ``func()`` if it eventually succeeds.
    :raises Exception: Re-raises the last exception if retries are exhausted.
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

    :param func: Zero-argument callable to execute.
    :param args: Parsed CLI arguments providing retry settings.
    :param logger: Logger used for warnings and errors.
    :param sample_idx: Sample index used for logging context.
    :param problem_snippet: Short snippet of the problem text for logging.
    :param min_sleep: Optional minimum sleep duration between retries.
    :returns: Result of ``func()`` if it eventually succeeds.
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
    "locked_file",
    "iter_jsonl_objects",
    "scan_existing_problem_samples",
    "scan_existing_pass1_results",
    "extract_problem_and_answer",
    "build_math_gateway_row_base",
    "build_usage_dict",
    "build_two_pass_row_base",
    "PassOutputs",
    "limit_dataset_examples",
    "limit_dataset_for_args",
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
