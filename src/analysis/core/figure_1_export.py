"""Export helpers for Figure 1 analyses."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..common.pass_extraction import extract_pass_answer
from ..labels import aha_gpt_for_rec
from .figure_1_data import nat_step_from_path


@dataclass(frozen=True)
class FormalThresholds:
    """Delta thresholds plus min-step requirement."""

    delta1: float
    delta2: float
    delta3: Optional[float]
    min_prior_steps: int


@dataclass(frozen=True)
class GptFilterConfig:
    """Settings for GPT-based gating."""

    keys: List[str]
    subset_native: bool


@dataclass(frozen=True)
class ExportDestinations:
    """On-disk export paths."""

    out_dir: str
    slug: str


@dataclass(frozen=True)
class FormalExportConfig:
    """Settings for exporting Formal Aha events."""

    dataset: str
    model: str
    thresholds: FormalThresholds
    gpt_filter: GptFilterConfig
    destinations: ExportDestinations
    max_chars: int = 4000


@dataclass(frozen=True)
class PassRecord:
    """Thin wrapper for a domain/problem/step record."""

    domain: str
    problem: str
    step: int
    payload: Dict[str, Any]


def export_formal_aha_json_with_text(
    problem_step_df: pd.DataFrame,
    files_by_domain: Dict[str, List[str]],
    config: FormalExportConfig,
) -> Tuple[str, str, int]:
    """Export Formal Aha events (with optional raw answers) to JSON/JSONL."""
    targets, problem_index = _build_target_index(problem_step_df)
    if not targets:
        return _write_empty_exports(
            config.destinations.out_dir,
            config.destinations.slug,
        )

    events, remaining = _collect_events_for_targets(
        files_by_domain,
        problem_index,
        config,
        targets,
    )
    if remaining:
        events.extend(_events_for_missing(remaining, problem_index, config))
    return _write_event_outputs(events, config)


def _build_target_index(
    problem_step_df: pd.DataFrame,
) -> Tuple[set[Tuple[str, str, int]], Dict[Tuple[str, str, int], pd.Series]]:
    """Return the target key set and a lookup table for problem rows."""
    if "domain" in problem_step_df.columns:
        targets = {
            (str(row["domain"]), str(row["problem"]), int(row["step"]))
            for _, row in problem_step_df.loc[
                problem_step_df["aha_formal"] == 1, ["domain", "problem", "step"]
            ].iterrows()
        }
        index = {
            (str(row["domain"]), str(row["problem"]), int(row["step"])): row
            for _, row in problem_step_df.iterrows()
        }
    else:
        targets = {
            ("All", str(row["problem"]), int(row["step"]))
            for _, row in problem_step_df.loc[
                problem_step_df["aha_formal"] == 1, ["problem", "step"]
            ].iterrows()
        }
        dup = problem_step_df.copy()
        dup["domain"] = "All"
        index = {
            (str(row["domain"]), str(row["problem"]), int(row["step"])): row
            for _, row in dup.iterrows()
        }
    return targets, index


def _collect_events_for_targets(
    files_by_domain: Dict[str, List[str]],
    problem_index: Dict[Tuple[str, str, int], pd.Series],
    config: FormalExportConfig,
    targets: set[Tuple[str, str, int]],
) -> Tuple[List[Dict[str, Any]], set[Tuple[str, str, int]]]:
    """Collect event dicts for Aha targets found in pass1 files."""
    events: List[Dict[str, Any]] = []
    remaining = set(targets)
    for domain, files in files_by_domain.items():
        if not files:
            continue
        for record in _iter_domain_records(domain, files):
            if not remaining:
                break
            result = _event_from_pass_record(record, problem_index, config)
            if result is None:
                continue
            key, event = result
            if key not in remaining:
                continue
            events.append(event)
            remaining.discard(key)
    return events, remaining


def _iter_domain_records(domain: str, files: List[str]) -> Iterator[PassRecord]:
    """Yield decoded JSONL records for the provided domain."""
    for path in files:
        step_from_name = nat_step_from_path(path)
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = payload.get(
                    "step",
                    step_from_name if step_from_name is not None else None,
                )
                if step is None:
                    continue
                problem_value = (
                    payload.get("problem")
                    or payload.get("clue")
                    or payload.get("row_key")
                )
                if problem_value is None:
                    dataset_index = payload.get("dataset_index")
                    problem_value = (
                        f"idx:{dataset_index}" if dataset_index is not None else "unknown"
                    )
                yield PassRecord(
                    domain=str(domain),
                    problem=str(problem_value),
                    step=int(step),
                    payload=payload,
                )


def _event_from_pass_record(
    record: PassRecord,
    problem_index: Dict[Tuple[str, str, int], pd.Series],
    config: FormalExportConfig,
) -> Optional[Tuple[Tuple[str, str, int], Dict[str, Any]]]:
    """Return (key, event_dict) when the record meets the export criteria."""
    pass1_payload = record.payload.get("pass1") or {}
    if not isinstance(pass1_payload, dict):
        return None
    aha_gpt_now = aha_gpt_for_rec(
        pass1_payload,
        record.payload,
        config.gpt_filter.subset_native,
        config.gpt_filter.keys,
        record.domain,
    )
    if aha_gpt_now != 1:
        return None
    key = (record.domain, record.problem, record.step)
    row = problem_index.get(key)
    if row is None:
        return None
    answer = _truncate_answer(
        extract_pass_answer(pass1_payload),
        config.max_chars,
    )
    event = _build_event_dict(
        record,
        row,
        config,
        question=record.problem,
        answer=answer,
    )
    return key, event


def _truncate_answer(answer: Optional[str], max_chars: int) -> Optional[str]:
    """Clamp answers to ``max_chars`` when needed."""
    if answer and max_chars and len(answer) > max_chars:
        return answer[:max_chars] + " …[truncated]"
    return answer


def _build_event_dict(
    record: PassRecord,
    row: pd.Series,
    config: FormalExportConfig,
    question: str,
    answer: Optional[str],
) -> Dict[str, Any]:
    """Helper for building event dicts with consistent typing."""
    p_plus_raw = row["p_correct_given_shift"]
    p_plus = float(p_plus_raw) if np.isfinite(p_plus_raw) else None
    p_base = float(row["freq_correct"])
    delta_gain = (p_plus - p_base) if p_plus is not None else None
    return {
        "domain": record.domain,
        "dataset": config.dataset,
        "model": config.model,
        "problem": record.problem,
        "step": int(record.step),
        "n_samples": int(row["n_samples"]),
        "p_correct": p_base,
        "p_shift_rate": float(row["aha_rate_gpt"]),
        "p_correct_given_shift": (float(p_plus) if p_plus is not None else None),
        "delta_gain_at_shift": (float(delta_gain) if delta_gain is not None else None),
        "thresholds": {
            "delta1": float(config.thresholds.delta1),
            "delta2": float(config.thresholds.delta2),
            "delta3": (
                float(config.thresholds.delta3)
                if config.thresholds.delta3 is not None
                else None
            ),
            "min_prior_steps": int(config.thresholds.min_prior_steps),
        },
        "question": question,
        "answer": answer,
    }


def _write_empty_exports(out_dir: str, slug: str) -> Tuple[str, str, int]:
    """Write empty JSON/JSONL files when no events exist."""
    json_path = os.path.join(out_dir, f"formal_aha_events__{slug}.json")
    jsonl_path = os.path.join(out_dir, f"formal_aha_events__{slug}.jsonl")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump([], handle)
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        handle.write("")
    return json_path, jsonl_path, 0


def _events_for_missing(
    remaining: set[Tuple[str, str, int]],
    problem_index: Dict[Tuple[str, str, int], pd.Series],
    config: FormalExportConfig,
) -> List[Dict[str, Any]]:
    """Generate placeholder events when GPT positives are missing."""
    extras: List[Dict[str, Any]] = []
    for domain, problem, step in sorted(remaining):
        row = problem_index[(domain, problem, step)]
        placeholder = PassRecord(
            domain=str(domain),
            problem=str(problem),
            step=int(step),
            payload={},
        )
        extras.append(
            _build_event_dict(
                placeholder,
                row,
                config,
                question=problem,
                answer=None,
            )
        )
    return extras


def _write_event_outputs(
    events: List[Dict[str, Any]],
    config: FormalExportConfig,
) -> Tuple[str, str, int]:
    """Persist JSON/JSONL outputs and return their locations."""
    out_dir = config.destinations.out_dir
    slug = config.destinations.slug
    json_path = os.path.join(out_dir, f"formal_aha_events__{slug}.json")
    jsonl_path = os.path.join(out_dir, f"formal_aha_events__{slug}.jsonl")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(events, handle, ensure_ascii=False, indent=2)
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return json_path, jsonl_path, len(events)


__all__ = [
    "FormalExportConfig",
    "FormalThresholds",
    "GptFilterConfig",
    "ExportDestinations",
    "export_formal_aha_json_with_text",
]
