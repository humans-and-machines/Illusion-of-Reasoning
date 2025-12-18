#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset loading and record helpers for Rush Hour (car-park) inference.

This module hosts the pieces that turn raw dataset rows into the normalized
structures consumed by :mod:`src.inference.domains.carpark.carpark_solver`.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

from src.inference.utils.common import iter_jsonl_objects, require_datasets
from src.inference.utils.task_registry import CARPARK_SYSTEM_PROMPT

from .carpark_board import _canon_rush_generic, _canon_rush_gold


SYSTEM_PROMPT = CARPARK_SYSTEM_PROMPT


def _ensure_messages(obj: Any) -> List[Dict[str, str]]:
    """Dataset may store messages as JSON string or as a Python list."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return []
    raise ValueError("messages field is neither list nor JSON-encoded list")


def norm_fields(
    example: Dict[str, Any],
    prompt_col: str,
    solution_col: str,
) -> tuple[List[Dict[str, str]], Any]:
    """
    Normalize raw record fields into (messages, solution).
    """
    messages = example.get(prompt_col)
    solution = example.get(solution_col)
    try:
        messages = _ensure_messages(messages)
    except ValueError:
        problem = example.get("problem") or example.get("board") or example.get("prompt") or ""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(problem)},
        ]
    return messages, solution


def load_existing_example_index(outpath: str) -> Dict[str, set[int]]:
    """
    Scan an existing JSONL file and return a mapping
    ``{example_id -> {sample_idx, ...}}``.
    """
    existing_by_example: Dict[str, set[int]] = defaultdict(set)
    if not os.path.exists(outpath):
        return existing_by_example
    for record in iter_jsonl_objects(outpath):
        example_id = record.get("example_id")
        sample_idx = record.get("sample_idx")
        if example_id is None or not isinstance(sample_idx, int):
            continue
        existing_by_example[str(example_id)].add(sample_idx)
    return existing_by_example


@dataclass
class BatchRangeConfig:
    """Configuration describing how to build batch items for a dataset slice."""

    prompt_col: str
    solution_col: str
    num_samples: int
    existing_by_example: Dict[str, set[int]]


def build_batch_items_for_range(
    examples,
    start_idx: int,
    batch_size: int,
    config: BatchRangeConfig,
) -> List[Dict[str, Any]]:
    """
    Construct ``batch_items`` for a contiguous range of examples.
    """
    batch_items: List[Dict[str, Any]] = []
    for offset, raw_example in enumerate(
        examples.select(
            range(start_idx, min(start_idx + batch_size, len(examples))),
        ),
    ):
        messages_and_solution = norm_fields(
            raw_example,
            config.prompt_col,
            config.solution_col,
        )
        example_id = str(raw_example.get("id", f"idx_{start_idx + offset}"))
        missing = [
            sample_idx
            for sample_idx in range(config.num_samples)
            if sample_idx not in config.existing_by_example.get(example_id, set())
        ]
        if not missing:
            continue
        batch_items.append(
            {
                "id": example_id,
                "messages": messages_and_solution[0],
                "solution": messages_and_solution[1],
                "missing_indices": missing,
            },
        )
    return batch_items


def load_rush_dataset(
    dataset_id: str,
    split: str,
    cache_dir: str,
    prompt_col: str = "messages",
    solution_col: str = "solution",
) -> "Dataset":
    """
    Load a Rush Hour dataset and ensure required columns are present.

    This helper also enforces a stable ``id`` field so that resume logic can
    reliably identify examples across multiple jobs and dataset slices. When
    the underlying dataset does not expose an ``id`` column, we synthesize one
    based on the global row index (``id='idx_{i}'``).

    :param dataset_id: Hugging Face dataset identifier or local path.
    :param split: Dataset split name (for example, ``\"train\"`` or ``\"test\"``).
    :param cache_dir: Directory to use as a datasets cache.
    :param prompt_col: Column name containing chat messages or prompts.
    :param solution_col: Column name containing gold solutions.
    :returns: A :class:`datasets.Dataset` exposing ``column_names`` and ``select``.
    :raises ValueError: If the requested prompt or solution columns are missing.
    """
    dataset_cls, load_dataset = require_datasets()
    dataset = load_dataset(
        dataset_id,
        split=split,
        cache_dir=cache_dir,
    )
    columns = set(dataset.column_names)
    if prompt_col not in columns or solution_col not in columns:
        raise ValueError(
            f"Dataset missing required columns: {prompt_col}, {solution_col}. Found: {sorted(columns)}",
        )

    # Ensure a stable example identifier so that resuming across multiple
    # jobs and dataset slices can correctly detect already-completed examples.
    if "id" not in columns:
        dataset = dataset.map(
            lambda _example, idx: {"id": f"idx_{idx}"},
            with_indices=True,
        )

    return dataset


__all__ = [
    "SYSTEM_PROMPT",
    "norm_fields",
    "load_existing_example_index",
    "build_batch_items_for_range",
    "load_rush_dataset",
    "_canon_rush_generic",
    "_canon_rush_gold",
]
