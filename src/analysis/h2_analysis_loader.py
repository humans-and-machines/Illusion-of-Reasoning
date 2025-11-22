#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Utilities for loading PASS-1 rows and extracting uncertainty/Aha labels."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from src.analysis.core.h2_uncertainty_helpers import get_aha_gpt_flag
from src.analysis.utils import coerce_bool, nat_step_from_path

TEMP_PATS = [re.compile(r"temp(?:erature)?[_-]?([0-9]*\.?[0-9]+)", re.I)]


def maybe_temp_from_path(path: str) -> Optional[float]:
    """Extract a temperature token from the path (e.g., ``temp-0.3``)."""
    for pattern in TEMP_PATS:
        match = pattern.search(path)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None

def _get_aha_native(pass1_dict: Dict[str, Any]) -> Optional[int]:
    """Native reconsider cue, ignoring injected cues on pass-1."""
    aha_raw = coerce_bool(pass1_dict.get("has_reconsider_cue"))
    markers = pass1_dict.get("reconsider_markers") or []
    if isinstance(markers, list) and ("injected_cue" in markers):
        return 0
    return 0 if aha_raw is None else int(aha_raw)


def _choose_uncertainty(
    pass1_dict: Dict[str, Any],
    preference: str = "answer",
) -> Optional[float]:
    """Pick an entropy value for the requested ``preference``."""
    if preference == "answer":
        search_order = ["entropy_answer", "entropy", "entropy_think"]
    elif preference == "overall":
        search_order = ["entropy", "entropy_answer", "entropy_think"]
    elif preference == "think":
        search_order = ["entropy_think", "entropy", "entropy_answer"]
    else:
        search_order = []

    for key in search_order:
        candidate = pass1_dict.get(key)
        if candidate is not None:
            return float(candidate)
    return None


@dataclass(frozen=True)
class Pass1RowContext:
    """Context needed to construct PASS-1 rows while iterating over files."""

    path: str
    step_from_name: Optional[int]
    temp_from_path: Optional[float]
    unc_field: str
    aha_source: str


def _build_pass1_row(
    rec: Dict[str, Any],
    context: Pass1RowContext,
) -> Optional[Dict[str, Any]]:
    """Construct a single PASS-1 row dict or return ``None`` if invalid."""
    pass1_payload = rec.get("pass1") or {}
    if not pass1_payload:
        return None

    problem = rec.get("problem") or rec.get("clue") or rec.get("row_key")
    if problem is None:
        dataset_index = rec.get("dataset_index")
        problem = f"idx:{dataset_index}" if dataset_index is not None else "unknown"
    default_step = context.step_from_name if context.step_from_name is not None else None
    step_value = rec.get("step", default_step)
    if step_value is None:
        return None

    correct = coerce_bool(pass1_payload.get("is_correct_pred"))
    if correct is None:
        return None
    aha_gpt = get_aha_gpt_flag(pass1_payload, rec)
    aha_native = _get_aha_native(pass1_payload)
    if context.aha_source == "gpt":
        aha_label = aha_gpt if aha_gpt is not None else aha_native
    else:
        aha_label = aha_native if aha_native is not None else aha_gpt
    if aha_label is None:
        return None

    uncertainty = _choose_uncertainty(pass1_payload, context.unc_field)
    if uncertainty is None:
        return None

    temp_value = (
        rec.get("temperature")
        or pass1_payload.get("temperature")
        or rec.get("config", {}).get("temperature")
        or context.temp_from_path
    )

    return {
        "problem": str(problem),
        "step": int(step_value),
        "sample_idx": rec.get("sample_idx"),
        "correct": int(correct),
        "aha": int(aha_label),
        "uncertainty": float(uncertainty),
        "temperature": None if temp_value is None else float(temp_value),
        "source_file": context.path,
    }


def load_pass1_rows(files: List[str], unc_field: str, aha_source: str) -> pd.DataFrame:
    """
    Load pass-1 rows across all files while capturing metadata and uncertainty.
    """
    rows: List[Dict[str, Any]] = []
    for path in files:
        context = Pass1RowContext(
            path=path,
            step_from_name=nat_step_from_path(path),
            temp_from_path=maybe_temp_from_path(path),
            unc_field=unc_field,
            aha_source=aha_source,
        )
        with open(path, "r", encoding="utf-8") as file_handle:
            for raw_line in file_handle:
                stripped_line = raw_line.strip()
                if not stripped_line:
                    continue
                try:
                    record = json.loads(stripped_line)
                except (json.JSONDecodeError, ValueError):
                    continue
                row = _build_pass1_row(
                    rec=record,
                    context=context,
                )
                if row:
                    rows.append(row)
    pass1_df = pd.DataFrame(rows)
    if pass1_df.empty:
        raise RuntimeError("No usable PASS-1 rows found (missing aha and/or uncertainty).")
    return pass1_df


__all__ = [
    "coerce_bool",
    "load_pass1_rows",
    "nat_step_from_path",
    "maybe_temp_from_path",
]
