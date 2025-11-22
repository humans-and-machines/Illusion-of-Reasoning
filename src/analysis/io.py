#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File and record loading utilities for analysis scripts.

These helpers centralize the common patterns used across the ad-hoc analysis
scripts (JSON/JSONL/JSONL.GZ reading and "step-aware" directory scanning).
"""

from __future__ import annotations

import gzip
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .utils import (
    extract_pass1_and_step,
    nat_step_from_path,
    step_within_bounds,
)

# ---------------------------------------------------------------------------
# Scanning patterns
# ---------------------------------------------------------------------------

# Directories that almost never contain user data of interest
SKIP_DIR_DEFAULT = {"compare-1shot", "1shot", "hf_cache", "__pycache__"}

# Directory names like ".../step-01234/"
STEP_DIR_PAT = re.compile(r"(?:^|/)(step[-_]?\d{1,5})(?:/|$)", re.I)

# Filenames like "step100.jsonl", "global_step-100.jsonl", "checkpoint_100.jsonl"
STEP_FILE_PAT = re.compile(r"^(?:step|global[_-]?step|checkpoint)[-_]?\d{1,5}", re.I)


# ---------------------------------------------------------------------------
# Scanning helpers
# ---------------------------------------------------------------------------

def _normalize_skip(skip_substrings: Optional[Iterable[str]]) -> List[str]:
    if skip_substrings is None:
        return sorted(SKIP_DIR_DEFAULT)
    return [str(s).lower() for s in skip_substrings]


def scan_jsonl_files(root: str, split_substr: Optional[str] = None) -> List[str]:
    """
    Recursively collect ``.jsonl`` files under a root directory.

    :param root: Root directory under which to search.
    :param split_substr: Optional substring that must appear in the filename
        (for example, ``\"train\"`` or ``\"test\"``).
    :returns: Sorted list of matching ``.jsonl`` file paths.
    """
    out: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if not filename.endswith(".jsonl"):
                continue
            if split_substr and split_substr not in filename:
                continue
            out.append(os.path.join(dirpath, filename))
    out.sort()
    return out


def scan_files_step_only(
    root: str,
    split_substr: Optional[str] = None,
    skip_substrings: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Scan for JSON/JSONL/JSONL.GZ files that are associated with a training
    step, either via the directory name or filename.

    This mirrors the “step-aware” scanning used in several scripts (for example,
    ``temperature_effects``, ``temp_graph``, ``entropy_bin_regression``,
    ``final-plot``).

    :param root: Root directory under which to search.
    :param split_substr: Optional substring that must appear in the filename.
    :param skip_substrings: Optional collection of directory-name substrings to skip.
    :returns: Sorted list of JSON/JSONL/JSONL.GZ files associated with steps.
    """
    skip = _normalize_skip(skip_substrings)
    out: List[str] = []

    for dirpath, _, filenames in os.walk(root):
        dirpath_normalized = dirpath.replace("\\", "/").lower()
        if any(substr in dirpath_normalized for substr in skip):
            continue
        dir_has_step = STEP_DIR_PAT.search(dirpath_normalized) is not None
        for filename in filenames:
            low = filename.lower()
            if not (low.endswith(".jsonl") or low.endswith(".jsonl.gz") or low.endswith(".json")):
                continue
            if split_substr and split_substr not in filename:
                continue
            file_has_step = STEP_FILE_PAT.search(filename) is not None
            if not (dir_has_step or file_has_step):
                continue
            out.append(os.path.join(dirpath, filename))

    out.sort()
    return out


def build_files_by_domain(
    domain_map: Dict[str, str],
    domains: Iterable[str],
    split_substr: Optional[str],
    skip_substrings: Iterable[str],
) -> Dict[str, List[str]]:
    """
    Helper to construct ``files_by_domain`` mappings from a domain → root map.

    This captures the common pattern used in ``temp_graph`` and
    ``temperature_effects`` when turning discovered roots into per-domain
    file lists.
    """
    files_by_domain: Dict[str, List[str]] = {}
    for dom in domains:
        path = domain_map.get(dom)
        if not path:
            continue
        files = scan_files_step_only(path, split_substr, skip_substrings)
        if files:
            files_by_domain[dom] = files
    return files_by_domain


def build_standard_domain_roots(
    root_crossword: Optional[str],
    root_math: Optional[str],
    root_math2: Optional[str],
    root_carpark: Optional[str],
) -> Dict[str, Optional[str]]:
    """
    Construct a standard domain→root mapping for Crossword/Math/Math2/Carpark.
    """
    return {
        "Crossword": root_crossword,
        "Math": root_math,
        "Math2": root_math2,
        "Carpark": root_carpark,
    }


def build_jsonl_files_by_domain(
    domain_roots: Dict[str, Optional[str]],
    split_substr: Optional[str],
) -> tuple[Dict[str, List[str]], Optional[str]]:
    """
    Helper to construct ``files_by_domain`` mappings for plain ``.jsonl`` roots.

    This captures the common pattern used in several scripts that take
    ``--root_*`` flags (for example, heatmap_1, table_1).

    :param domain_roots: Mapping from domain label to root directory (or None).
    :param split_substr: Optional substring filter on filenames.
    :returns: (files_by_domain, first_root) where first_root is the first
              non-empty root encountered, or None if none yielded files.
    """
    files_by_domain: Dict[str, List[str]] = {}
    first_root: Optional[str] = None
    for domain, root in domain_roots.items():
        if not root:
            continue
        files = scan_jsonl_files(root, split_substr)
        if not files:
            continue
        files_by_domain[domain] = files
        if first_root is None:
            first_root = root
    return files_by_domain, first_root




def collect_jsonl_files_for_domains(
    domain_roots: Dict[str, Optional[str]],
    split_substr: Optional[str],
    results_root: Optional[str],
    *,
    all_domain_label: str = "All",
) -> Tuple[Dict[str, List[str]], str]:
    """
    Construct a ``files_by_domain`` mapping for plain ``.jsonl`` roots with an
    optional fallback ``results_root``.

    This captures the common pattern used by several scripts that accept
    ``--root_*`` flags plus an optional positional ``results_root`` fallback.

    :param domain_roots: Mapping from domain label to root directory (or None).
    :param split_substr: Optional substring filter on filenames.
    :param results_root: Optional fallback root used when no per-domain roots
        yield any files.
    :param all_domain_label: Domain label to use when only ``results_root`` is
        provided; defaults to ``\"All\"``.
    :returns: ``(files_by_domain, first_root)`` where ``first_root`` is the
        first non-empty root encountered.
    :raises SystemExit: If no files are found or no valid root is available.
    """
    files_by_domain, first_root = build_jsonl_files_by_domain(
        domain_roots,
        split_substr,
    )
    if not files_by_domain:
        if not results_root:
            raise SystemExit("Provide --root_* folders or a fallback results_root.")
        files_by_domain[all_domain_label] = scan_jsonl_files(
            results_root,
            split_substr,
        )
        first_root = results_root

    total_files = sum(len(values) for values in files_by_domain.values())
    if total_files == 0:
        raise SystemExit("No JSONL files found. Check roots/--split.")

    if first_root is None:
        raise SystemExit("No valid results root was found.")

    return files_by_domain, first_root


def build_files_by_domain_for_args(args: Any) -> Tuple[Dict[str, List[str]], str]:
    """
    Convenience wrapper to build ``files_by_domain``/``first_root`` from CLI args.

    This captures the common pattern used in graph-style scripts that accept
    ``--root_*``, ``--split`` and a positional ``results_root``.
    """
    domain_roots = build_standard_domain_roots(
        root_crossword=getattr(args, "root_crossword", None),
        root_math=getattr(args, "root_math", None),
        root_math2=getattr(args, "root_math2", None),
        root_carpark=getattr(args, "root_carpark", None),
    )
    return collect_jsonl_files_for_domains(
        domain_roots,
        split_substr=getattr(args, "split", None),
        results_root=getattr(args, "results_root", None),
    )


def scan_files_with_steps_or_meta(
    root: str,
    split_substr: Optional[str] = None,
    skip_substrings: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Similar to :func:`scan_files_step_only`, but also keep files where we can
    infer a step from record contents (via :func:`nat_step_from_path`).

    This is useful when some logs do not encode the step in filenames or
    directories but do provide it inside the JSON records.

    :param root: Root directory under which to search.
    :param split_substr: Optional substring that must appear in the filename.
    :param skip_substrings: Optional collection of directory-name substrings to skip.
    :returns: Sorted list of JSON/JSONL/JSONL.GZ files considered step-aware.
    """
    files = scan_files_step_only(root, split_substr=split_substr, skip_substrings=skip_substrings)
    if files:
        return files

    # Fallback: be more permissive, still skipping common non-data dirs.
    skip = _normalize_skip(skip_substrings)
    out: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        dirpath_normalized = dirpath.replace("\\", "/").lower()
        if any(substr in dirpath_normalized for substr in skip):
            continue
        for filename in filenames:
            low = filename.lower()
            if not (
                low.endswith(".jsonl")
                or low.endswith(".jsonl.gz")
                or low.endswith(".json")
            ):
                continue
            if split_substr and split_substr not in filename:
                continue
            full_path = os.path.join(dirpath, filename)
            if nat_step_from_path(full_path) is not None:
                out.append(full_path)

    out.sort()
    return out


# ---------------------------------------------------------------------------
# Record iteration
# ---------------------------------------------------------------------------

def _iter_json_from_text(text: str) -> Iterable[Dict[str, Any]]:
    """
    Yield JSON objects from a string that may contain either a JSON value
    (list or dict) or newline-delimited JSON records.
    """
    text = text.strip()
    if not text:
        return
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: treat as JSONL-style text.
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record
        return

    if isinstance(obj, list):
        for record in obj:
            if isinstance(record, dict):
                yield record
    elif isinstance(obj, dict):
        yield obj


def _iter_json_lines(handle) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a line-delimited JSON file handle."""
    for line in handle:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            yield record


def iter_records_from_file(path: str) -> Iterable[Dict[str, Any]]:
    """
    Yield JSON records from a ``.jsonl``, ``.jsonl.gz``, or ``.json`` file.

    This mirrors the most robust variant used in entropy-driven analysis
    scripts: it tolerates either newline-delimited JSON or a single JSON
    list/dict in ``.json`` files.

    :param path: Path to a JSON/JSONL/JSONL.GZ file on disk.
    :returns: Iterator over JSON objects (dictionaries) parsed from the file.
    """
    if path.endswith(".jsonl.gz"):
        opener = gzip.open
        mode = "rt"
    else:
        opener = open
        mode = "r"

    try:
        with opener(path, mode, encoding="utf-8") as file_handle:  # type: ignore[arg-type]
            if path.endswith(".json"):
                text = file_handle.read()
                yield from _iter_json_from_text(text)
            else:
                yield from _iter_json_lines(file_handle)
    except OSError:
        # Silently skip unreadable files.
        pass


def iter_pass1_samples_by_domain(
    files_by_domain: Dict[str, List[str]],
    min_step: Optional[int] = None,
    max_step: Optional[int] = None,
) -> Iterable[tuple[str, Dict[str, Any], int, Dict[str, Any]]]:
    """
    Iterate PASS-1-style samples grouped by domain.

    Yields ``(domain, pass1_dict, step_int, record)`` tuples, applying
    ``nat_step_from_path`` and :func:`extract_pass1_and_step` plus the
    optional ``[min_step, max_step]`` bounds.
    """
    for domain, files in files_by_domain.items():
        for path in files:
            step_from_name = nat_step_from_path(path)
            for rec in iter_records_from_file(path):
                pass1, step = extract_pass1_and_step(rec, step_from_name)
                if not pass1 or step is None:
                    continue
                if not step_within_bounds(step, min_step, max_step):
                    continue
                yield str(domain), pass1, int(step), rec
