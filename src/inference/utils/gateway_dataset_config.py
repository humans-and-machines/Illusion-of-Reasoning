#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight container types used by :mod:`src.inference.utils.gateway_utils`.

These dataclasses are split out so that the main gateway utilities module
stays below lint size and attribute-count thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class PassOutputs:
    """
    Container for per-pass outputs used when writing rows.

    :param full_texts: Full text outputs for each row.
    :param ent_think: Token-level entropy series for think phases.
    :param ent_answer: Token-level entropy series for answer phases.
    :param stop_reason_think: Stop reasons for think phases.
    :param stop_reason_answer: Stop reasons for answer phases.
    """

    full_texts: List[str]
    ent_think: List[List[float]]
    ent_answer: List[List[float]]
    stop_reason_think: List[str]
    stop_reason_answer: List[str]


@dataclass
class MathGatewayDatasetSource:
    """
    Source-related configuration for a math gateway dataset.

    :param dataset_id: Identifier such as ``\"MATH-500\"`` or a HF dataset path.
    :param split: Dataset split name (for example, ``\"test\"``).
    :param dataset_path: Optional local JSONL path for MATH-500-style records.
    """

    dataset_id: str
    split: str
    dataset_path: Optional[str]


@dataclass
class MathGatewayDatasetLimits:
    """
    Limiting and caching configuration for a math gateway dataset.

    :param seed: Random seed for shuffling.
    :param num_examples: Optional cap on number of examples.
    :param cache_dir: Optional HF cache directory; a default is used if ``None``.
    :param from_end: Whether to select examples from the end of the dataset.
    :param start: Zero-based starting index for slicing.
    """

    seed: int
    num_examples: Optional[int]
    cache_dir: Optional[str] = None
    from_end: bool = False
    start: int = 0


@dataclass
class MathGatewayDatasetConfig:
    """
    Configuration for loading and limiting a math gateway dataset.

    This class groups related settings into ``source`` and ``limits`` objects
    to keep the number of instance attributes small while preserving a
    backwards-compatible attribute interface.

    :param source: Dataset identity and path information.
    :param limits: Limiting and caching configuration.
    :param outpath: Path to output JSONL file (used to scan existing rows).
    :param logger: Logger used for informational messages.
    :param load_math500_fn: Callable that loads the MATH-500 dataset.
    :param load_remote_dataset_fn: Callable that loads a remote dataset.
    """

    source: MathGatewayDatasetSource
    limits: MathGatewayDatasetLimits
    outpath: str
    logger: logging.Logger
    load_math500_fn: Any
    load_remote_dataset_fn: Any

    # Backwards-compatible flat properties ---------------------------------
    @property
    def dataset_id(self) -> str:
        """Dataset identifier (for example, 'MATH-500' or a HF path)."""
        return self.source.dataset_id

    @property
    def split(self) -> str:
        """Dataset split name (for example, 'train' or 'test')."""
        return self.source.split

    @property
    def dataset_path(self) -> Optional[str]:
        """Optional local JSONL path for MATH-500-style records."""
        return self.source.dataset_path

    @property
    def seed(self) -> int:
        """Random seed used when shuffling or subselecting examples."""
        return self.limits.seed

    @property
    def num_examples(self) -> Optional[int]:
        """Optional cap on the number of examples loaded from the dataset."""
        return self.limits.num_examples

    @property
    def cache_dir(self) -> Optional[str]:
        """Optional Hugging Face cache directory used when loading datasets."""
        return self.limits.cache_dir

    @property
    def from_end(self) -> bool:
        """Whether examples should be drawn from the end of the dataset."""
        return self.limits.from_end

    @property
    def start(self) -> int:
        """Zero-based starting index used when slicing the dataset."""
        return self.limits.start


__all__ = [
    "PassOutputs",
    "MathGatewayDatasetSource",
    "MathGatewayDatasetLimits",
    "MathGatewayDatasetConfig",
]
