#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared inference helpers used across math/carpark/crossword entrypoints.
The goal is to keep canonicalization, tag parsing, and small torch utilities
in one place so the task-specific scripts stay focused on their domain logic.

Note: lightweight, text-only helpers have been moved to ``text_utils`` and
torch/generation helpers live in ``generation`` to keep this module smaller.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, List

from src.inference.utils import gateway_utils as _gateway_utils
from src.inference.utils import generation as _generation_utils
from src.inference.utils import math_pass_utils as _math_pass_utils


# Explicitly bind selected gateway_utils helpers so that static analyzers see
# them on this module without treating them as unused imports.
DEFAULT_SECOND_PASS_PHRASE = _math_pass_utils.DEFAULT_SECOND_PASS_PHRASE
build_second_pass_cue_strings = _math_pass_utils.build_second_pass_cue_strings
OPENR1_PROMPT_TEMPLATE = _gateway_utils.OPENR1_PROMPT_TEMPLATE
PassOutputs = _gateway_utils.PassOutputs
append_jsonl_row = _gateway_utils.append_jsonl_row
locked_file = _gateway_utils.locked_file
build_eos_ids_from_tokenizer = _gateway_utils.build_eos_ids_from_tokenizer
build_math_gateway_arg_parser = _gateway_utils.build_math_gateway_arg_parser
build_math_gateway_messages = _gateway_utils.build_math_gateway_messages
build_math_gateway_row_base = _gateway_utils.build_math_gateway_row_base
build_usage_dict = _gateway_utils.build_usage_dict
build_two_pass_row_base = _gateway_utils.build_two_pass_row_base
GatewayCallParams = _gateway_utils.GatewayCallParams
call_with_gateway_retries = _gateway_utils.call_with_gateway_retries
call_with_retries = _gateway_utils.call_with_retries
configure_tokenizer_and_eos = _gateway_utils.configure_tokenizer_and_eos
configure_unified_runner_common = _gateway_utils.configure_unified_runner_common
extract_problem_and_answer = _gateway_utils.extract_problem_and_answer
iter_jsonl_objects = _gateway_utils.iter_jsonl_objects
iter_math_gateway_samples = _gateway_utils.iter_math_gateway_samples
limit_dataset_examples = _gateway_utils.limit_dataset_examples
load_local_json_dataset = _gateway_utils.load_local_json_dataset
load_remote_dataset_default = _gateway_utils.load_remote_dataset_default
parse_openai_chat_response = _gateway_utils.parse_openai_chat_response
prepare_math_gateway_dataset = _gateway_utils.prepare_math_gateway_dataset
prepare_math_gateway_dataset_from_args = _gateway_utils.prepare_math_gateway_dataset_from_args
require_datasets = _gateway_utils.require_datasets
scan_existing_pass1_results = _gateway_utils.scan_existing_pass1_results
scan_existing_problem_samples = _gateway_utils.scan_existing_problem_samples
setup_hf_cache_dir_env = _gateway_utils.setup_hf_cache_dir_env
setup_script_logger = _gateway_utils.setup_script_logger

# Explicitly bind selected math_pass_utils helpers as well.
RECONSIDER_PATTERNS = _math_pass_utils.RECONSIDER_PATTERNS
add_token_and_tag_fields = _math_pass_utils.add_token_and_tag_fields
build_entropy_pass_base = _math_pass_utils.build_entropy_pass_base
build_math_pass_meta = _math_pass_utils.build_math_pass_meta
canon_math = _math_pass_utils.canon_math
contains_canon = _math_pass_utils.contains_canon
extract_blocks = _math_pass_utils.extract_blocks
finite_mean = _math_pass_utils.finite_mean
pack_math_pass_result = _math_pass_utils.pack_math_pass_result
valid_tag_structure = _math_pass_utils.valid_tag_structure

# Retain dynamic re-exports so that any additions to __all__ in the helper
# modules are still reflected here at runtime.
for _mod in (_math_pass_utils, _gateway_utils):
    for _name in getattr(_mod, "__all__", []):
        globals()[_name] = getattr(_mod, _name)

_GENERATION_EXPORTS = list(_generation_utils.__all__)

for _name in _GENERATION_EXPORTS:
    globals()[_name] = getattr(_generation_utils, _name)

# Make static aliases for lint/static-analysis visibility.
GenerationLimits = _generation_utils.GenerationLimits
SamplingConfigBase = _generation_utils.SamplingConfigBase
StopOnSubstrings = _generation_utils.StopOnSubstrings
classify_stop_reason = _generation_utils.classify_stop_reason
decode_generated_row = _generation_utils.decode_generated_row
move_inputs_to_device = _generation_utils.move_inputs_to_device
build_second_pass_think_prefixes = _generation_utils.build_second_pass_think_prefixes
empty_pass_outputs = _generation_utils.empty_pass_outputs
run_generate_batch = _generation_utils.run_generate_batch

_CORE_EXPORTS = _GENERATION_EXPORTS + [
    # Explicitly re-export selected helpers so static analyzers
    # can see them on this module without relying on dynamic globals().
    "build_second_pass_cue_strings",
    "build_math_inference_config_kwargs",
    "build_math_inference_config_kwargs_from_args",
    "require_torch",
    "require_transformers",
    "DEFAULT_SECOND_PASS_PHRASE",
    "OPENR1_PROMPT_TEMPLATE",
    "extract_problem_and_answer",
    "load_local_json_dataset",
    "require_datasets",
    "scan_existing_pass1_results",
    "setup_script_logger",
    "build_math_pass_meta",
    "canon_math",
    "finite_mean",
    "pack_math_pass_result",
    "GatewayCallParams",
]

_HELPER_EXPORTS: List[str] = [
    helper_name
    for _mod in (_math_pass_utils, _gateway_utils)
    for helper_name in (getattr(_mod, "__all__", None) or [])
]

__all__ = _CORE_EXPORTS + _HELPER_EXPORTS


@dataclass(frozen=True)
class _MathInferenceLimits:
    """Batch/sample/token caps for math inference."""

    batch_size: int
    num_samples: int
    think_cap: int
    answer_cap: int


@dataclass(frozen=True)
class _MathInferenceSampling:
    """Sampling/two-pass knobs for math inference."""

    temperature: float
    top_p: float
    two_pass: bool
    second_pass_phrase: str
    second_pass_use_sample_idx: int


@dataclass(frozen=True)
class MathInferenceKwargs:
    """Container for common math inference kwargs to keep signatures short."""

    limits: _MathInferenceLimits
    sampling: _MathInferenceSampling
    entropy_mode: str
    eos_ids: Any

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "MathInferenceKwargs":
        """
        Build a :class:`MathInferenceKwargs` grouping limits and sampling config.
        """
        required = [
            "batch_size",
            "num_samples",
            "temperature",
            "top_p",
            "entropy_mode",
            "eos_ids",
            "two_pass",
            "second_pass_phrase",
            "second_pass_use_sample_idx",
            "think_cap",
            "answer_cap",
        ]
        missing = [name for name in required if name not in kwargs]
        if missing:
            raise KeyError(f"Missing required config fields: {missing}")
        limits = _MathInferenceLimits(
            batch_size=kwargs["batch_size"],
            num_samples=kwargs["num_samples"],
            think_cap=kwargs["think_cap"],
            answer_cap=kwargs["answer_cap"],
        )
        sampling = _MathInferenceSampling(
            temperature=kwargs["temperature"],
            top_p=kwargs["top_p"],
            two_pass=kwargs["two_pass"],
            second_pass_phrase=kwargs["second_pass_phrase"],
            second_pass_use_sample_idx=kwargs["second_pass_use_sample_idx"],
        )
        return cls(
            limits=limits,
            sampling=sampling,
            entropy_mode=kwargs["entropy_mode"],
            eos_ids=kwargs.get("eos_ids"),
        )


def build_math_inference_config_kwargs(
    config: MathInferenceKwargs | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Build the common kwargs dict for math-style inference configs / loops.

    :param config: Optional :class:`MathInferenceKwargs` container. When
        omitted, ``**kwargs`` is consumed for backwards-compatible flat fields
        (batch_size, num_samples, temperature, etc.).
    :param kwargs: Flat config fields; used when ``config`` is not provided.
    :returns: Keyword-argument dictionary suitable for :class:`MathInferenceConfig`.
    """
    cfg = config or MathInferenceKwargs.from_kwargs(**kwargs)
    return {
        "batch_size": cfg.limits.batch_size,
        "num_samples": cfg.limits.num_samples,
        "temperature": cfg.sampling.temperature,
        "top_p": cfg.sampling.top_p,
        "entropy_mode": cfg.entropy_mode,
        "eos_ids": cfg.eos_ids,
        "two_pass": cfg.sampling.two_pass,
        "second_pass_phrase": cfg.sampling.second_pass_phrase,
        "second_pass_use_sample_idx": cfg.sampling.second_pass_use_sample_idx,
        "think_cap": cfg.limits.think_cap,
        "answer_cap": cfg.limits.answer_cap,
    }


def build_math_inference_config_kwargs_from_args(args, eos_ids):
    """
    Map common CLI args into kwargs for math-style inference loops and configs.

    :param args: Argument namespace exposing common math-inference options.
    :param eos_ids: EOS token ID or IDs derived from the tokenizer.
    :returns: Keyword-argument dictionary suitable for :class:`MathInferenceConfig`.
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


def require_torch(caller: str):
    """
    Import and return the ``torch`` module.

    A user-friendly :class:`RuntimeError` is raised if the dependency is missing.

    :param caller: Human-readable name of the caller used in error messages.
    :returns: Imported :mod:`torch` module.
    :raises RuntimeError: If the ``torch`` package is not available.
    """
    try:
        torch_mod = import_module("torch")
        if not hasattr(torch_mod, "inference_mode"):

            @contextmanager
            def _no_grad():
                yield

            torch_mod.inference_mode = _no_grad  # type: ignore[attr-defined]
        return torch_mod
    except ImportError as exc:  # pragma: no cover - hard dependency
        # Raise ImportError so callers (and tests using importorskip) can
        # treat the missing dependency as an optional feature.
        raise ImportError(
            f"{caller} requires 'torch'; install the 'torch' package.",
        ) from exc


def require_transformers(caller: str):
    """
    Import and return the ``transformers`` module.

    A user-friendly :class:`RuntimeError` is raised if the dependency is missing.

    :param caller: Human-readable name of the caller used in error messages.
    :returns: Imported :mod:`transformers` module.
    :raises RuntimeError: If the ``transformers`` package is not available.
    """
    try:
        return import_module("transformers")
    except ImportError as exc:  # pragma: no cover - hard dependency
        raise ImportError(
            f"{caller} requires 'transformers'; install it to use this script.",
        ) from exc
