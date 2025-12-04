#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities to flatten multi–cue math inference outputs into a per–cue JSONL.

This is a small helper for experiments where a single JSONL file contains
multiple second–pass reconsideration variants (e.g., ``pass2a``, ``pass2b``,
``pass2c``) corresponding to different lexical cues.  It produces a new
JSONL file with one row per ``(problem, sample_idx, cue_variant)`` and
adds convenience fields for downstream analysis:

    - ``cue_variant``: ``"baseline"`` | ``"C1"`` | ``"C2"`` | ``"C3"``
    - ``baseline_correct``: correctness of ``pass1.is_correct_pred``
    - ``intervention_correct``: correctness for the given cue/pass
    - ``entropy``: pass–level mean token entropy
    - ``entropy_quartile``: 1–4, based on baseline entropies

Typical usage (from the repo root):

    python -m src.annotate.tasks.math_cue_variants \\
      artifacts/results/GRPO-1.5B-math-temp-0.05-3/step950/step0950_test.jsonl

This will write ``step0950_test_flat_cues.jsonl`` next to the input file.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from ...common.jsonl_utils import iter_jsonl_lines


def _load_records(path: str) -> List[Dict]:
    """Load a JSONL file into a list of Python dicts."""
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as file_handle:
        for obj in iter_jsonl_lines(file_handle, strict=True):
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _percentile(sorted_values: List[float], percentile_value: float) -> Optional[float]:
    """Return a simple index-based percentile from a sorted list."""
    if not sorted_values:
        return None
    if percentile_value <= 0.0:
        return sorted_values[0]
    if percentile_value >= 1.0:
        return sorted_values[-1]
    index = int(round(percentile_value * (len(sorted_values) - 1)))
    return sorted_values[index]


def _compute_entropy_quartile_boundaries(
    records: Iterable[Dict],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute Q1, Q2, Q3 boundaries using ``pass1.entropy`` as the baseline.
    """
    baseline_entropies: List[float] = []
    for record in records:
        pass1 = record.get("pass1") or {}
        entropy = pass1.get("entropy")
        if isinstance(entropy, (int, float)):
            baseline_entropies.append(float(entropy))

    baseline_entropies.sort()
    if not baseline_entropies:
        return None, None, None

    quartile_1 = _percentile(baseline_entropies, 0.25)
    quartile_2 = _percentile(baseline_entropies, 0.50)
    quartile_3 = _percentile(baseline_entropies, 0.75)
    return quartile_1, quartile_2, quartile_3


def _entropy_quartile(
    entropy: Optional[float],
    quartile_1: Optional[float],
    quartile_2: Optional[float],
    quartile_3: Optional[float],
) -> Optional[int]:
    """Map an entropy value to quartile index 1..4, or None if undefined."""
    if entropy is None or quartile_1 is None or quartile_2 is None or quartile_3 is None:
        return None

    value = float(entropy)
    if value <= quartile_1:
        return 1
    if value <= quartile_2:
        return 2
    if value <= quartile_3:
        return 3
    return 4


_CUE_PASS_PAIRS = [
    ("C1", "pass2a"),
    ("C2", "pass2b"),
    ("C3", "pass2c"),
]

_TOKEN_FIELDS = ("tokens_total", "tokens_think", "tokens_answer")


def _extract_shift_fields(section: Dict) -> Dict:
    """
    Extract any existing shift-related annotation fields from a pass section.

    This lets downstream analysis on the flattened JSONL see the same
    ``shift_in_reasoning_v1`` and related metadata that the main annotator
    writes into the original JSONL.
    """
    if not isinstance(section, dict):
        return {}

    out: Dict = {}
    for key, value in section.items():
        if key.startswith("shift_") or key.startswith("_shift_prefilter_"):
            out[key] = value
    return out


def _section_weights(section: Dict) -> Dict[str, int]:
    """Capture output/token-length metadata for a pass section."""
    weights: Dict[str, int] = {}
    output_text = section.get("output") or ""
    if isinstance(output_text, str):
        weights["output_len"] = len(output_text)
    else:
        weights["output_len"] = len(str(output_text))
    for field in _TOKEN_FIELDS:
        value = section.get(field)
        if isinstance(value, (int, float)):
            weights[field] = int(value)
    return weights


@dataclass(frozen=True)
class _RowContext:
    base_meta: Dict
    baseline_correct: bool
    quartiles: Tuple[Optional[float], Optional[float], Optional[float]]


def _infer_output_path(input_path: str, output_path: Optional[str]) -> str:
    """
    Infer a default flattened-output path based on the input path.

    When ``output_path`` is provided, it is returned unchanged.
    """
    if output_path is not None:
        return output_path
    directory, filename = os.path.split(os.path.abspath(input_path))
    stem, ext = os.path.splitext(filename)
    if not ext:
        ext = ".jsonl"
    return os.path.join(directory, f"{stem}_flat_cues{ext}")


def _write_rows_for_record(
    record: Dict,
    quartile_1: Optional[float],
    quartile_2: Optional[float],
    quartile_3: Optional[float],
    out_file,
) -> None:
    """Write baseline and cue-variant rows for a single record."""
    pass1 = record.get("pass1") or {}
    baseline_correct = bool(pass1.get("is_correct_pred"))
    baseline_entropy = pass1.get("entropy")

    context = _RowContext(
        base_meta=_build_base_meta(record),
        baseline_correct=baseline_correct,
        quartiles=(quartile_1, quartile_2, quartile_3),
    )

    baseline_row = _build_flattened_row(
        context,
        pass1,
        "pass1",
        "baseline",
        baseline_entropy,
    )
    out_file.write(json.dumps(baseline_row, ensure_ascii=False) + "\n")

    for cue_label, pass_key in _CUE_PASS_PAIRS:
        section = record.get(pass_key)
        if not isinstance(section, dict):
            continue
        entropy_value = section.get("entropy")
        row = _build_flattened_row(
            context,
            section,
            pass_key,
            cue_label,
            entropy_value,
        )
        out_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_base_meta(record: Dict) -> Dict:
    """Construct a reusable base metadata payload for a record."""
    return {
        "problem": record.get("problem"),
        "gold_answer": record.get("gold_answer"),
        "gold_answer_canon": record.get("gold_answer_canon"),
        "step": record.get("step"),
        "split": record.get("split"),
        "sample_idx": record.get("sample_idx"),
        "domain": record.get("domain")
        or record.get("dataset")
        or record.get("task")
        or record.get("split")
        or "unknown",
    }


def _build_flattened_row(
    context: _RowContext,
    section: Dict,
    pass_key: str,
    cue_variant: str,
    entropy_value: Optional[float],
) -> Dict:
    """Compose a single flattened row and include weight metadata."""
    intervention_correct = bool(section.get("is_correct_pred"))
    quartile_1, quartile_2, quartile_3 = context.quartiles
    row = {
        **context.base_meta,
        "pass": pass_key,
        "cue_variant": cue_variant,
        "baseline_correct": context.baseline_correct,
        "intervention_correct": intervention_correct,
        "entropy": entropy_value,
        "entropy_quartile": _entropy_quartile(
            entropy_value,
            quartile_1,
            quartile_2,
            quartile_3,
        ),
        **_extract_shift_fields(section),
    }
    row.update(_section_weights(section))
    return row


def flatten_math_cue_variants(
    input_path: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Flatten a single multi–cue math JSONL file into a per–cue JSONL.

    :param input_path: Path to the original ``stepXXXX_split.jsonl`` file.
    :param output_path: Optional output path.  If omitted, a sibling file
        named ``<stem>_flat_cues.jsonl`` will be created.
    :returns: The output path used.
    """
    records = _load_records(input_path)
    quartile_1, quartile_2, quartile_3 = _compute_entropy_quartile_boundaries(
        records,
    )
    output_path = _infer_output_path(input_path, output_path)

    # We may have iterated once to compute quartiles, so iterate again here.
    with open(output_path, "w", encoding="utf-8") as out_file:
        for record in records:
            _write_rows_for_record(
                record,
                quartile_1,
                quartile_2,
                quartile_3,
                out_file,
            )

    return output_path


def _build_argparser() -> argparse.ArgumentParser:
    """CLI for flattening a single multi–cue JSONL file."""
    parser = argparse.ArgumentParser(
        description=(
            "Flatten a multi–cue math inference JSONL (pass1, pass2a/b/c) "
            "into a per–cue JSONL with convenience fields."
        ),
    )
    parser.add_argument(
        "input_path",
        help="Path to stepXXXX_split.jsonl (e.g., step0950_test.jsonl).",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Optional output JSONL path (defaults to <stem>_flat_cues.jsonl).",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = _build_argparser()
    args = parser.parse_args()
    output_path = flatten_math_cue_variants(args.input_path, args.output_path)
    print(f"Wrote flattened cue file to {output_path}")


if __name__ == "__main__":
    main()
