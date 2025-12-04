#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset/JSONL helpers and dataset preparation utilities for math gateways.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, DefaultDict, Dict, Iterable, List, Optional, Tuple

from src.common.jsonl_utils import iter_jsonl_lines
from src.inference.utils.gateway_dataset_config import (
    MathGatewayDatasetConfig,
    MathGatewayDatasetLimits,
    MathGatewayDatasetSource,
)


if TYPE_CHECKING:  # pragma: no cover
    from datasets import Dataset


__all__ = [
    "MathGatewayMeta",
    "DatasetPrepConfig",
    "require_datasets",
    "locked_file",
    "load_local_json_dataset",
    "append_jsonl_row",
    "iter_jsonl_objects",
    "scan_existing_problem_samples",
    "scan_existing_pass1_results",
    "extract_problem_and_answer",
    "build_math_gateway_row_base",
    "build_usage_dict",
    "build_two_pass_row_base",
    "limit_dataset_examples",
    "limit_dataset_for_args",
    "prepare_math_gateway_dataset",
    "prepare_math_gateway_dataset_from_args",
    "load_remote_dataset_default",
]


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
        filtered_lines = (line for line in file_handle if line.lstrip().startswith("{"))
        for obj in iter_jsonl_lines(filtered_lines):
            if isinstance(obj, dict):
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
        yield from iter_jsonl_lines(handle)


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
        for obj in iter_jsonl_lines(handle):
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


@dataclass(frozen=True)
class MathGatewayMeta:
    """
    Core identifiers for a single math gateway sample.
    """

    split: str
    step: int
    sample_idx: int


@dataclass(frozen=True)
class DatasetPrepConfig:
    """
    Bundle the arguments required to prepare the math gateway dataset.
    """

    outpath: str
    logger: Any
    load_math500_fn: Any
    load_remote_dataset_fn: Any
    cache_dir: Optional[str] = None


def build_math_gateway_row_base(
    *,
    problem: str,
    gold_answer: Any,
    gold_answer_canon: Any,
    meta: Optional[MathGatewayMeta] = None,
    **legacy_ids,
) -> Dict[str, Any]:
    """
    Common prefix for single-pass MATH JSONL rows used by gateway scripts.

    :param problem: Normalized problem text.
    :param gold_answer: Gold answer value (raw).
    :param gold_answer_canon: Canonicalized gold answer, if available.
    :param meta: Identifiers for the sample (split, step, sample_idx). For
        backwards compatibility callers may still pass ``split``, ``step``,
        and ``sample_idx`` directly.
    :returns: Dictionary containing shared row fields.
    """
    if meta is None:
        try:
            meta = MathGatewayMeta(
                split=legacy_ids["split"],
                step=legacy_ids["step"],
                sample_idx=legacy_ids["sample_idx"],
            )
        except KeyError as exc:
            missing = ("split", "step", "sample_idx")
            raise ValueError(f"Missing required identifiers {missing}") from exc

    return {
        "problem": problem,
        "gold_answer": gold_answer,
        "gold_answer_canon": gold_answer_canon,
        "split": meta.split,
        "step": meta.step,
        "sample_idx": meta.sample_idx,
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
        safe_start = max(min(start, total), 0)
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
                f"prepare_math_gateway_dataset missing required arguments: {missing_str}",
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
    config: Optional[DatasetPrepConfig] = None,
    **kwargs,
):
    """
    Convenience wrapper mapping common CLI args into :func:`prepare_math_gateway_dataset`.
    """
    if config is None:
        try:
            config = DatasetPrepConfig(
                outpath=kwargs.pop("outpath"),
                logger=kwargs.pop("logger"),
                load_math500_fn=kwargs.pop("load_math500_fn"),
                load_remote_dataset_fn=kwargs.pop("load_remote_dataset_fn"),
                cache_dir=kwargs.pop("cache_dir", None),
            )
        except KeyError as exc:
            raise TypeError(
                "outpath, logger, load_math500_fn, and load_remote_dataset_fn are required",
            ) from exc
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")

    return prepare_math_gateway_dataset(
        dataset_id=args.dataset_id,
        split=args.split,
        seed=args.seed,
        num_examples=args.num_examples,
        dataset_path=args.dataset_path,
        outpath=config.outpath,
        logger=config.logger,
        load_math500_fn=config.load_math500_fn,
        load_remote_dataset_fn=config.load_remote_dataset_fn,
        cache_dir=config.cache_dir,
        from_end=getattr(args, "examples_from_end", False),
        start=getattr(args, "dataset_start", 0),
    )


def load_remote_dataset_default(dataset_id: str, split: str, cache_dir: str):
    """
    Default HF datasets loader used by math gateway scripts.
    """
    _, load_dataset_fn = require_datasets()
    try:
        return load_dataset_fn(dataset_id, split=split, cache_dir=cache_dir)
    except TypeError:
        # Some stubs accept positional args only.
        return load_dataset_fn(dataset_id, split, cache_dir)
