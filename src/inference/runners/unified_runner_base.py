#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared helpers for unified inference runners (math / carpark / crossword).

This module centralizes argument parsing and backend/dataset wiring so that
the thin CLI entrypoint modules (``cli.unified_math``, ``cli.unified_carpark``,
``cli.unified_crossword``) can stay tiny. This keeps Pylint's
duplicate-code (R0801) noise low while preserving testability.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Iterable, List, Optional, Sequence

from src.inference.domains.math import math_core
from src.inference.utils.common import (
    GenerationLimits,
    _MathInferenceSampling,
    build_math_inference_config_kwargs,
    build_math_inference_config_kwargs_from_args,
    require_datasets,
)
from src.inference.utils.gateway_utils import (
    add_model_and_output_args,
    configure_unified_runner_common,
    init_unified_backend_and_eos,
    limit_dataset_for_args,
    setup_hf_cache_dir_env,
)


# ---------------------------------------------------------------------------
# Math unified runner
# ---------------------------------------------------------------------------


def parse_math_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Construct and parse CLI arguments for the unified math runner.

    :param argv: Optional sequence of argument strings; defaults to :data:`sys.argv`.
    :returns: Parsed :class:`argparse.Namespace` containing math runner options.
    """
    parser = argparse.ArgumentParser()
    add_model_and_output_args(parser)
    parser.add_argument(
        "--dataset_id",
        default="MATH-500",
        help="Use 'MATH-500' or a HF dataset path.",
    )
    parser.add_argument(
        "--dataset_path",
        default=None,
        help="Optional local JSONL for MATH-500-style records.",
    )
    configure_unified_runner_common(parser, default_dtype="float16")
    return parser.parse_args(argv)


def _load_generic_dataset(dataset_id: str, split: str, cache_dir: str):
    """
    Load a generic HF dataset lazily via ``datasets.load_dataset``.

    :param dataset_id: Hugging Face dataset identifier.
    :param split: Dataset split name (for example, ``\"test\"``).
    :param cache_dir: Directory to use as a datasets cache.
    :returns: Dataset object returned by :func:`datasets.load_dataset`.
    """
    datasets_mod = import_module("datasets")
    return datasets_mod.load_dataset(dataset_id, split=split, cache_dir=cache_dir)


def run_math_main(backend_cls, argv: Optional[Sequence[str]] = None) -> None:
    """
    Entry point logic for unified math inference using :mod:`math_core`.

    :param backend_cls: Backend class (typically :class:`HFBackend`) to construct.
    :param argv: Optional sequence of argument strings; defaults to :data:`sys.argv`.
    :returns: ``None``. The function parses arguments and runs inference.
    """
    args = parse_math_args(argv)
    hf_cache_dir = setup_hf_cache_dir_env("./.hf_cache")

    backend, eos_ids = init_unified_backend_and_eos(
        backend_cls=backend_cls,
        model_name_or_path=args.model_name_or_path,
        revision=args.revision,
        cache_dir=hf_cache_dir,
        dtype=args.dtype,
        device_map="auto",
        attn_implementation=args.attn_implementation,
        tokenizer_path=args.tokenizer_path,
    )

    if args.dataset_id == "MATH-500" and args.dataset_path:
        dataset = math_core.load_math500(
            hf_cache_dir,
            args.split,
            args.seed,
            dataset_path=args.dataset_path,
        )
    elif args.dataset_id == "MATH-500":
        dataset = math_core.load_math500(hf_cache_dir, args.split, args.seed)
    else:
        dataset = _load_generic_dataset(args.dataset_id, args.split, hf_cache_dir)
    dataset = limit_dataset_for_args(dataset, args)

    cfg = math_core.MathInferenceConfig(
        split_name=args.split,
        output_dir=args.output_dir,
        step=args.step,
        **build_math_inference_config_kwargs_from_args(args, eos_ids),
    )

    math_core.run_inference_on_split(
        examples=dataset,
        tokenizer=backend.tokenizer,
        model=backend.model,
        config=cfg,
    )
    print(f"All inference complete → {args.output_dir}")


# ---------------------------------------------------------------------------
# Simple math-inference helper for tests
# ---------------------------------------------------------------------------


class _SimpleListDataset:
    """
    Minimal Dataset-like wrapper over a list of dicts for testing.

    :param records: Iterable of mapping-like records.
    """

    def __init__(self, records: Iterable[dict]):
        """
        Initialize the dataset wrapper from an iterable of records.

        :param records: Iterable of mapping-like records.
        """
        self._records: List[dict] = list(records)

    def __len__(self) -> int:
        """
        Return the number of records in the dataset.

        :returns: Length of the underlying records list.
        """
        return len(self._records)

    def select(self, indices: Iterable[int]) -> "_SimpleListDataset":
        """
        Return a new :class:`_SimpleListDataset` containing the selected indices.

        :param indices: Iterable of integer indices to keep.
        :returns: New dataset instance with only the selected records.
        """
        idx_list = list(indices)
        return _SimpleListDataset([self._records[i] for i in idx_list])

    def __iter__(self):
        """
        Iterate over the underlying records.

        :returns: Iterator over record dictionaries.
        """
        return iter(self._records)


# Reuse shared limits container from src.inference.utils.common for tests.
MathTestLimits = GenerationLimits


MathTestSampling = _MathInferenceSampling


@dataclass(init=False)
class MathTestConfig:
    """
    Lightweight configuration container for test-only math inference.

    Mirrors the key fields of MathInferenceConfig without requiring datasets,
    while grouping related settings into nested containers to keep instance
    attributes small for linting and readability.

    :param dataset: Iterable of dicts representing math examples.
    :param output_dir: Directory where JSONL results will be written.
    :param step: Training or checkpoint step identifier.
    :param eos_ids: EOS token ID or IDs used to terminate generation.
    :param batch_size: Number of examples to process per batch (kwarg).
    :param num_samples: Number of samples to generate per problem (kwarg).
    :param temperature: Sampling temperature for generation (kwarg).
    :param top_p: Nucleus-sampling parameter for generation (kwarg).
    :param think_cap: Token cap for ``<think>`` generations (kwarg).
    :param answer_cap: Token cap for ``<answer>`` generations (kwarg).
    :param two_pass: Whether to enable the reconsideration second pass (kwarg).
    :param second_pass_phrase: Cue phrase injected into second-pass prompts (kwarg).
    :param second_pass_use_sample_idx: Preferred sample index to feed pass 2 (kwarg).
    """

    dataset: Iterable[dict]
    output_dir: str
    step: int
    limits: MathTestLimits
    sampling: MathTestSampling
    eos_ids: Optional[List[int]] = None

    def __init__(self, *args, **kwargs) -> None:
        if len(args) > 3:
            raise TypeError("MathTestConfig accepts at most 3 positional arguments")

        if args:
            dataset = args[0]
        else:
            dataset = kwargs.pop("dataset")
        if len(args) >= 2:
            output_dir = args[1]
        else:
            output_dir = kwargs.pop("output_dir")
        if len(args) >= 3:
            step = args[2]
        else:
            step = kwargs.pop("step")

        def _pop_config(key: str, default: Any) -> Any:
            return kwargs.pop(key, default)

        limits_config = {
            "batch_size": _pop_config("batch_size", 8),
            "num_samples": _pop_config("num_samples", 1),
            "think_cap": _pop_config("think_cap", 750),
            "answer_cap": _pop_config("answer_cap", 50),
        }
        sampling_config = {
            "temperature": _pop_config("temperature", 0.0),
            "top_p": _pop_config("top_p", 0.95),
            "two_pass": _pop_config("two_pass", False),
            "second_pass_phrase": _pop_config("second_pass_phrase", "cue"),
            "second_pass_use_sample_idx": _pop_config(
                "second_pass_use_sample_idx",
                0,
            ),
        }
        eos_ids = kwargs.pop("eos_ids", None)

        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        self.dataset = dataset
        self.output_dir = output_dir
        self.step = step
        self.limits = MathTestLimits(**limits_config)
        self.sampling = MathTestSampling(**sampling_config)
        self.eos_ids = eos_ids


def run_math_inference(
    *,
    backend,
    config: MathTestConfig,
) -> None:
    """
    Lightweight math inference helper used by tests.

    This wraps :func:`math_core.run_inference_on_split` with a minimal
    Dataset-like adapter so tests don't depend on :class:`datasets.Dataset`.

    :param backend: Backend instance exposing ``tokenizer`` and ``model`` attributes.
    :param config: Test configuration describing dataset and sampling parameters.
    :returns: ``None``. Results are written to ``config.output_dir``.
    """
    examples = _SimpleListDataset(config.dataset)
    cfg = math_core.MathInferenceConfig(
        split_name="test",
        output_dir=config.output_dir,
        step=config.step,
        **build_math_inference_config_kwargs(
            batch_size=config.limits.batch_size,
            num_samples=config.limits.num_samples,
            temperature=config.sampling.temperature,
            top_p=config.sampling.top_p,
            entropy_mode="reconsider",
            eos_ids=config.eos_ids,
            two_pass=config.sampling.two_pass,
            second_pass_phrase=config.sampling.second_pass_phrase,
            second_pass_use_sample_idx=config.sampling.second_pass_use_sample_idx,
            think_cap=config.limits.think_cap,
            answer_cap=config.limits.answer_cap,
        ),
    )
    math_core.run_inference_on_split(
        examples=examples,
        tokenizer=backend.tokenizer,
        model=backend.model,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Carpark unified runner
# ---------------------------------------------------------------------------


def parse_carpark_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Construct and parse CLI arguments for the unified carpark runner.

    :param argv: Optional sequence of argument strings; defaults to :data:`sys.argv`.
    :returns: Parsed :class:`argparse.Namespace` containing carpark options.
    """
    parser = argparse.ArgumentParser()
    add_model_and_output_args(parser)

    parser.add_argument("--dataset_id", default="od2961/rush4-5-6-balanced")
    parser.add_argument("--dataset_prompt_column", default="messages")
    parser.add_argument("--dataset_solution_column", default="solution")
    configure_unified_runner_common(parser, default_dtype="bfloat16")
    return parser.parse_args(argv)


def run_carpark_main(
    load_module,
    backend_cls,
    argv: Optional[Sequence[str]] = None,
) -> None:
    """
    Entry point logic for unified carpark (Rush Hour) inference.

    :param load_module: Callable returning the carpark module to use.
    :param backend_cls: Backend class (typically :class:`HFBackend`) to construct.
    :param argv: Optional sequence of argument strings; defaults to :data:`sys.argv`.
    :returns: ``None``. The function parses arguments and runs inference.
    """
    args = parse_carpark_args(argv)
    hf_cache_dir = os.path.abspath("./.hf_cache")

    backend, eos_ids = init_unified_backend_and_eos(
        backend_cls=backend_cls,
        model_name_or_path=args.model_name_or_path,
        revision=args.revision,
        cache_dir=hf_cache_dir,
        dtype=args.dtype,
        device_map="auto",
        attn_implementation=args.attn_implementation,
        tokenizer_path=args.tokenizer_path,
    )

    carpark_mod = load_module()
    dataset = carpark_mod.load_rush_dataset(
        dataset_id=args.dataset_id,
        split=args.split,
        cache_dir=hf_cache_dir,
        prompt_col=args.dataset_prompt_column,
        solution_col=args.dataset_solution_column,
    )
    dataset = limit_dataset_for_args(dataset, args)

    config_kwargs = build_math_inference_config_kwargs_from_args(args, eos_ids)
    carpark_mod.run_inference_on_split(
        split_name=args.split,
        examples=dataset,
        tokenizer=backend.tokenizer,
        model=backend.model,
        step=args.step,
        outdir=args.output_dir,
        prompt_col=args.dataset_prompt_column,
        solution_col=args.dataset_solution_column,
        **config_kwargs,
    )
    print(f"All inference complete → {args.output_dir}")


# ---------------------------------------------------------------------------
# Crossword unified runner
# ---------------------------------------------------------------------------


def parse_crossword_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Construct and parse CLI arguments for the unified crossword runner.

    :param argv: Optional sequence of argument strings; defaults to :data:`sys.argv`.
    :returns: Parsed :class:`argparse.Namespace` containing crossword options.
    """
    parser = argparse.ArgumentParser()
    add_model_and_output_args(parser)
    parser.add_argument(
        "--dataset_id",
        default="CROSSWORD-LOCAL",
        help="Use 'CROSSWORD-LOCAL' for local JSONL, or a HF path if available.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        help="Path to local JSONL with clue/answer/enumeration.",
    )
    configure_unified_runner_common(parser, default_dtype="float16")
    return parser.parse_args(argv)


def _has_crossword_configs(cw_mod) -> bool:
    """Return True if the crossword module exposes the required config classes."""
    required_cfg_attrs = (
        "CrosswordCapsConfig",
        "CrosswordSamplingConfig",
        "CrosswordTwoPassConfig",
        "CrosswordInferenceConfig",
    )
    return all(hasattr(cw_mod, attr) for attr in required_cfg_attrs)


def _build_crossword_run_kwargs(cw_mod, args, backend, eos_ids, examples):
    """Construct kwargs for running crossword inference, using configs when available."""
    if _has_crossword_configs(cw_mod):
        caps_cfg = cw_mod.CrosswordCapsConfig(
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            think_cap=args.think_cap,
            answer_cap=args.answer_cap,
        )
        sampling_cfg = cw_mod.CrosswordSamplingConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            entropy_mode=args.entropy_mode,
        )
        two_pass_cfg = cw_mod.CrosswordTwoPassConfig(
            enabled=args.two_pass,
            phrase=args.second_pass_phrase,
            sample_index=args.second_pass_use_sample_idx,
        )
        cfg = cw_mod.CrosswordInferenceConfig(
            split_name=args.split,
            output_dir=args.output_dir,
            step=args.step,
            eos_ids=eos_ids,
            caps=caps_cfg,
            sampling=sampling_cfg,
            two_pass=two_pass_cfg,
        )
        return {
            "examples": examples,
            "tokenizer": backend.tokenizer,
            "model": backend.model,
            "config": cfg,
        }

    return {
        "split_name": args.split,
        "examples": examples,
        "tokenizer": backend.tokenizer,
        "model": backend.model,
        "batch_size": args.batch_size,
        "num_samples": args.num_samples,
        "think_cap": args.think_cap,
        "answer_cap": args.answer_cap,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "entropy_mode": args.entropy_mode,
        "two_pass": args.two_pass,
        "second_pass_phrase": args.second_pass_phrase,
        "second_pass_use_sample_idx": args.second_pass_use_sample_idx,
        "eos_ids": eos_ids,
        "step": args.step,
        "output_dir": args.output_dir,
    }


def run_crossword_main(
    load_module,
    backend_cls,
    argv: Optional[Sequence[str]] = None,
) -> None:
    """
    Run cryptic crossword inference with a unified HF backend.

    :param load_module: Callable returning the crossword module to use.
    :param backend_cls: Backend class (typically :class:`HFBackend`) to construct.
    :param argv: Optional sequence of argument strings; defaults to :data:`sys.argv`.
    :returns: ``None``. The function parses arguments and runs inference.
    """
    args = parse_crossword_args(argv)
    hf_cache_dir = os.path.abspath("./.hf_cache")

    backend, eos_ids = init_unified_backend_and_eos(
        backend_cls=backend_cls,
        model_name_or_path=args.model_name_or_path,
        revision=args.revision,
        cache_dir=hf_cache_dir,
        dtype=args.dtype,
        device_map="auto",
        attn_implementation=args.attn_implementation,
        tokenizer_path=args.tokenizer_path,
    )

    cw_mod = load_module()

    if args.dataset_id.upper() == "CROSSWORD-LOCAL":
        if not args.dataset_path:
            raise ValueError("--dataset_path is required when dataset_id=CROSSWORD-LOCAL")
        examples = cw_mod.load_crossword_local(args.dataset_path)
    else:
        _, load_dataset = require_datasets()
        examples = load_dataset(args.dataset_id, split=args.split, cache_dir=hf_cache_dir)

    examples = limit_dataset_for_args(examples, args)

    run_kwargs = _build_crossword_run_kwargs(cw_mod, args, backend, eos_ids, examples)
    cw_mod.run_inference_on_split(**run_kwargs)
    print(f"All inference complete → {args.output_dir}")
