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
import inspect
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Iterable, List, Optional, Sequence, Tuple

import src.inference.utils.gateway_dataset_utils as _gateway_dataset_utils
import src.inference.utils.gateway_retry as _gateway_retry
from src.inference.utils.gateway_dataset_config import (
    MathGatewayDatasetConfig,
    MathGatewayDatasetLimits,
    MathGatewayDatasetSource,
    PassOutputs,
)
from src.inference.utils.gateway_dataset_utils import (
    DatasetPrepConfig,
    MathGatewayMeta,
    append_jsonl_row,
    build_math_gateway_row_base,
    build_two_pass_row_base,
    build_usage_dict,
    extract_problem_and_answer,
    iter_jsonl_objects,
    limit_dataset_examples,
    limit_dataset_for_args,
    locked_file,
    prepare_math_gateway_dataset,
    prepare_math_gateway_dataset_from_args,
    scan_existing_pass1_results,
    scan_existing_problem_samples,
)
from src.inference.utils.gateway_logging import setup_hf_cache_dir_env, setup_script_logger
from src.inference.utils.gateway_retry import RetryContext, RetrySettings, call_with_retries
from src.inference.utils.math_pass_utils import DEFAULT_SECOND_PASS_PHRASE


if TYPE_CHECKING:  # pragma: no cover
    from datasets import Dataset


def require_datasets():
    """
    Import and return ``(Dataset, load_dataset)`` via ``gateway_dataset_utils``.

    This keeps a stable public hook for callers such as
    :mod:`src.inference.utils.common` without introducing the recursive
    monkeypatching that previously caused ``RecursionError``.
    """
    return _gateway_dataset_utils.require_datasets()


# Expose the retry module's time reference so tests can monkeypatch it.
time = _gateway_retry.time


@dataclass(frozen=True)
class GatewayCallParams:
    """Shared generation parameters used by gateway providers."""

    temperature: float
    top_p: float
    max_output_tokens: int
    request_timeout: int


def build_retry_context(
    *,
    logger: Any,
    sample_idx: int,
    problem_snippet: str,
    min_sleep: float = 10.0,
) -> RetryContext:
    """Construct a ``RetryContext`` with consistent defaults."""
    return RetryContext(
        logger=logger,
        sample_idx=sample_idx,
        problem_snippet=problem_snippet,
        min_sleep=min_sleep,
    )


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def load_local_json_dataset(path: str) -> "Dataset":
    """
    Thin proxy to :func:`gateway_dataset_utils.load_local_json_dataset`.

    The original implementation temporarily monkeypatched
    :func:`gateway_dataset_utils.require_datasets` to honor local test
    overrides. That indirection introduced a recursion when both modules
    delegated to each other. Since production callers only need the core
    behavior, we now call straight through.
    """
    return _gateway_dataset_utils.load_local_json_dataset(path)


def load_remote_dataset_default(dataset_id: str, split: str, cache_dir: str):
    """
    Thin proxy to :func:`gateway_dataset_utils.load_remote_dataset_default`.
    """
    return _gateway_dataset_utils.load_remote_dataset_default(
        dataset_id,
        split,
        cache_dir,
    )


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
            "Zero-based starting index within the split from which to take examples (used together with num_examples)."
        ),
    )
    arg_parser.add_argument(
        "--examples_from_end",
        action="store_true",
        help=("If set together with num_examples, take examples from the end of the split instead of the beginning."),
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
            "Zero-based starting index within the split from which to take examples (used together with num_examples)."
        ),
    )
    arg_parser.add_argument(
        "--examples_from_end",
        action="store_true",
        help=("If set together with num_examples, take examples from the end of the split instead of the beginning."),
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
    config: Optional[BackendInitConfig] = None,
    **kwargs,
):
    """
    Initialize a backend and a standard EOS-ID list for unified runners.

    ``backend_cls`` is typically :class:`HFBackend`, but is passed explicitly
    so tests can monkeypatch it on the caller module.

    :param backend_cls: Backend class exposing ``from_pretrained``.
    :param config: Backend/init configuration bundle. For backwards
        compatibility callers may still pass legacy keyword arguments
        (``model_name_or_path``, ``revision``, ``cache_dir``, ``dtype``,
        ``device_map``, ``attn_implementation``, ``tokenizer_path``) which
        will be converted into a :class:`BackendInitConfig`.
    :returns: Tuple ``(backend, eos_ids)`` where ``eos_ids`` may be ``None``.
    """
    if config is None:
        try:
            config = BackendInitConfig(
                model_name_or_path=kwargs.pop("model_name_or_path"),
                revision=kwargs.pop("revision", None),
                cache_dir=kwargs.pop("cache_dir"),
                dtype=kwargs.pop("dtype"),
                device_map=kwargs.pop("device_map"),
                attn_implementation=kwargs.pop("attn_implementation", None),
                tokenizer_path=kwargs.pop("tokenizer_path", None),
            )
        except KeyError as exc:
            raise TypeError(
                "model_name_or_path cache_dir dtype device_map required for backend init",
            ) from exc
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")

    backend = backend_cls.from_pretrained(
        model_name_or_path=config.model_name_or_path,
        revision=config.revision,
        cache_dir=config.cache_dir,
        dtype=config.dtype,
        device_map=config.device_map,
        attn_implementation=config.attn_implementation,
        tokenizer_path=config.tokenizer_path,
    )
    eos_ids = build_eos_ids_from_tokenizer(
        backend.tokenizer,
        extra_tokens=config.extra_tokens,
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


def call_with_gateway_retries(
    func,
    args: argparse.Namespace,
    context: Optional[RetryContext] = None,
    **legacy_context: object,
):
    """
    Convenience wrapper around ``call_with_retries`` using standard CLI args.
    """
    _gateway_retry.call_with_retries = call_with_retries
    return _gateway_retry.call_with_gateway_retries(
        func,
        args=args,
        context=context,
        **legacy_context,
    )


def call_with_gateway_retries_compat(
    call_fn,
    func,
    args: argparse.Namespace,
    context: RetryContext,
):
    """
    Invoke ``call_with_gateway_retries`` while tolerating legacy signatures.

    The helper inspects the provided ``call_fn`` (which may be monkeypatched in
    tests) and falls back to positional arguments when the keyword-only
    ``context`` parameter is not present.
    """
    try:
        sig = inspect.signature(call_fn)
        if "context" in sig.parameters:
            return call_fn(
                func,
                args=args,
                context=context,
            )
    except (ValueError, TypeError):  # pragma: no cover - defensive
        sig = None

    expects_min_sleep = bool(sig and ("min_sleep" in sig.parameters or len(sig.parameters) >= 6))
    min_sleep = getattr(context, "min_sleep", None)
    if expects_min_sleep:
        return call_fn(
            func,
            args=args,
            logger=context.logger,
            sample_idx=context.sample_idx,
            problem_snippet=context.problem_snippet,
            min_sleep=min_sleep,
        )
    return call_fn(
        func,
        args=args,
        logger=context.logger,
        sample_idx=context.sample_idx,
        problem_snippet=context.problem_snippet,
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
    "MathGatewayMeta",
    "MathGatewayDatasetConfig",
    "MathGatewayDatasetLimits",
    "MathGatewayDatasetSource",
    "build_math_gateway_row_base",
    "GatewayCallParams",
    "build_usage_dict",
    "build_two_pass_row_base",
    "PassOutputs",
    "limit_dataset_examples",
    "limit_dataset_for_args",
    "prepare_math_gateway_dataset",
    "prepare_math_gateway_dataset_from_args",
    "DatasetPrepConfig",
    "BackendInitConfig",
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
    "build_retry_context",
    "RetryContext",
    "RetrySettings",
    "call_with_retries",
    "call_with_gateway_retries",
    "call_with_gateway_retries_compat",
]


@dataclass(frozen=True)
class BackendInitConfig:
    """
    Configuration for initializing a backend and tokenizer.
    """

    model_name_or_path: str
    cache_dir: str
    dtype: str
    device_map: str
    revision: Optional[str] = None
    attn_implementation: Optional[str] = None
    tokenizer_path: Optional[str] = None
    extra_tokens: ClassVar[Sequence[str]] = ("<|im_end|>", "<|endoftext|>")
