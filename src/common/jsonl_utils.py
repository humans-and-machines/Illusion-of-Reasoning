"""Lightweight JSONL scanning and parsing helpers shared across modules."""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Iterator, List, Optional


def scan_jsonl_files(root: str, split_substr: Optional[str] = None) -> List[str]:
    """
    Recursively collect ``.jsonl`` files under a root directory.

    :param root: Root directory under which to search.
    :param split_substr: Optional substring that must appear in the filename
        (for example, ``"train"`` or ``"test"``).
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


def iter_jsonl_lines(lines: Iterable[str], *, strict: bool = False) -> Iterator[Any]:
    """
    Yield parsed JSON objects from an iterable of JSONL lines.

    Blank lines and lines that fail JSON decoding are skipped.
    """
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            yield json.loads(stripped)
        except json.JSONDecodeError:
            if strict:
                raise
            continue
