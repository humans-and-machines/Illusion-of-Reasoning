#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared helpers for unified inference runners (math / carpark / crossword).

This module centralizes argument parsing and backend/dataset wiring so that
the thin entrypoint modules (unified_math_runner, unified_carpark_runner,
unified_crossword_runner) can be 2–3 line wrappers. This keeps Pylint's
duplicate-code (R0801) noise low while preserving testability.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from importlib import import_module
from typing import Iterable, List, Optional, Sequence
from src.inference import math_core
from src.inference.common import (
    GenerationLimits,
    build_math_inference_config_kwargs,
    build_math_inference_config_kwargs_from_args,
    require_datasets,
)
from src.inference.gateway_utils import (
    add_model_and_output_args,
    configure_unified_runner_common,
    init_unified_backend_and_eos,
    setup_hf_cache_dir_env,
)


# ---------------------------------------------------------------------------
# Math unified runner
# ---------------------------------------------------------------------------

def parse_math_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Construct and parse CLI arguments for the unified math runner."""
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
    """Load a generic HF dataset lazily via datasets.load_dataset."""
    datasets_mod = import_module("datasets")
    return datasets_mod.load_dataset(dataset_id, split=split, cache_dir=cache_dir)


def run_math_main(backend_cls, argv: Optional[Sequence[str]] = None) -> None:
    """Entry point logic for unified math inference using math_core."""
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
    if args.num_examples is not None and args.num_examples > 0:
        dataset = dataset.select(range(min(args.num_examples, len(dataset))))

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
    """Minimal Dataset-like wrapper over a list of dicts for testing."""

    def __init__(self, records: Iterable[dict]):
        self._records: List[dict] = list(records)

    def __len__(self) -> int:
        return len(self._records)

    def select(self, indices: Iterable[int]) -> "_SimpleListDataset":
        """Return a new _SimpleListDataset containing the selected indices."""
        idx_list = list(indices)
        return _SimpleListDataset([self._records[i] for i in idx_list])

    def __iter__(self):
        return iter(self._records)


# Reuse shared limits container from src.inference.common for tests.
MathTestLimits = GenerationLimits


@dataclass
class MathTestSampling:
    """Sampling and two-pass configuration for test-only math inference."""

    temperature: float
    top_p: float
    two_pass: bool
    second_pass_phrase: str
    second_pass_use_sample_idx: int


@dataclass
class MathTestConfig:
    """
    Lightweight configuration container for test-only math inference.

    Mirrors the key fields of MathInferenceConfig without requiring datasets
    while grouping related settings to keep instance attributes small.
    """

    dataset: Iterable[dict]
    output_dir: str
    step: int
    limits: MathTestLimits
    sampling: MathTestSampling
    eos_ids: Optional[List[int]]


def run_math_inference(
    *,
    backend,
    config: MathTestConfig,
) -> None:
    """
    Lightweight math inference helper used by tests.

    This wraps math_core.run_inference_on_split with a minimal Dataset-like
    adapter so tests don't depend on datasets.Dataset.
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
    """Construct and parse CLI arguments for the unified carpark runner."""
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
    """Entry point logic for unified carpark (Rush Hour) inference."""
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
    if args.num_examples is not None and args.num_examples > 0:
        dataset = dataset.select(range(min(args.num_examples, len(dataset))))

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
    """Construct and parse CLI arguments for the unified crossword runner."""
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


def run_crossword_main(
    load_module,
    backend_cls,
    argv: Optional[Sequence[str]] = None,
) -> None:
    """Run cryptic crossword inference with a unified HF backend."""
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

    if args.num_examples is not None and args.num_examples > 0:
        examples = examples.select(range(min(args.num_examples, len(examples))))

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

    cw_mod.run_inference_on_split(
        examples=examples,
        tokenizer=backend.tokenizer,
        model=backend.model,
        config=cfg,
    )
    print(f"All inference complete → {args.output_dir}")
