#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared helpers for deriving stable problem identifiers across analysis scripts.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def resolve_problem_identifier(
    record: Dict[str, Any],
    fallback: Optional[str] = None,
) -> str:
    """
    Return the best-effort textual identifier for a problem row.

    Preference order:
      1. Explicit problem/question/clue/row_key fields
      2. dataset_index (formatted as ``idx:<n>``)
      3. Provided fallback string
      4. Literal ``\"unknown\"``
    """
    problem = record.get("problem") or record.get("question") or record.get("clue") or record.get("row_key")
    if problem is not None:
        return str(problem)

    dataset_index = record.get("dataset_index")
    if dataset_index is not None:
        return f"idx:{dataset_index}"

    if fallback is not None:
        return str(fallback)
    return "unknown"


__all__ = ["resolve_problem_identifier"]
