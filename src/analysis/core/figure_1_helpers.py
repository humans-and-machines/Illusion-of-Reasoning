#!/usr/bin/env python3
"""Support utilities shared by the Figure 1 pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from ..common.pass_extraction import extract_pass_answer
from ..io import iter_records_from_file
from ..labels import aha_gpt_for_rec
from ..utils import FormalThresholds, formal_flags_with_gain, nat_step_from_path
from . import FORMAL_REQUIRED_COLUMNS, build_formal_thresholds_from_args
from . import make_formal_thresholds as _make_formal_thresholds


ExportKey = Tuple[str, str, int]
EXPORT_REQUIRED_COLUMNS = FORMAL_REQUIRED_COLUMNS | {"aha_formal", "n_samples"}

# Legacy alias for tests importing directly from this module.
make_formal_thresholds = _make_formal_thresholds


def _formal_flags_for_group(
    problem_step_group: pd.DataFrame,
    thresholds: FormalThresholds,
) -> np.ndarray:
    freq_correct = problem_step_group["freq_correct"].to_numpy(float)
    aha_rate = problem_step_group["aha_rate_gpt"].to_numpy(float)
    aha_any = problem_step_group["aha_any_gpt"].to_numpy(int)
    p_correct_given_shift = problem_step_group["p_correct_given_shift"].to_numpy(float)
    return formal_flags_with_gain(
        freq=freq_correct,
        rate=aha_rate,
        shift=aha_any,
        p_plus=p_correct_given_shift,
        thresholds=thresholds,
    )


def mark_formal_pairs(
    problem_step_df: pd.DataFrame,
    delta1: float = 0.20,
    delta2: float = 0.20,
    min_prior_steps: int = 2,
    delta3: Optional[float] = None,
) -> pd.DataFrame:
    """Return a copy of ``problem_step_df`` annotated with formal AHA flags."""
    if not FORMAL_REQUIRED_COLUMNS.issubset(problem_step_df.columns):
        missing = FORMAL_REQUIRED_COLUMNS - set(problem_step_df.columns)
        raise ValueError(f"mark_formal_pairs: missing columns {missing}")

    group_keys = ["problem"]
    if "domain" in problem_step_df.columns:
        group_keys = ["domain"] + group_keys
    sorted_frame = problem_step_df.sort_values(group_keys + ["step"]).reset_index(drop=True).copy()
    flags = np.zeros(len(sorted_frame), dtype=int)

    thresholds = build_formal_thresholds_from_args(
        delta1=delta1,
        delta2=delta2,
        min_prior_steps=min_prior_steps,
        delta3=delta3,
    )
    for _, sub_frame in sorted_frame.groupby(group_keys, sort=False):
        sub_flags = _formal_flags_for_group(sub_frame, thresholds)
        flags[sub_frame.index.to_numpy()] = sub_flags

    sorted_frame["aha_formal"] = flags
    return sorted_frame


def _bootstrap_ratio_for_values(
    indicator_values: np.ndarray,
    num_bootstrap_samples: int,
    rng: np.random.Generator,
) -> Tuple[int, int, float, float, float]:
    num_samples = int(indicator_values.size)
    if num_samples == 0:
        return 0, 0, np.nan, np.nan, np.nan
    num_successes = int(indicator_values.sum())
    mean_ratio = float(indicator_values.mean())
    if num_bootstrap_samples <= 0 or num_samples == 1:
        return num_successes, num_samples, mean_ratio, np.nan, np.nan
    bootstrap_indices = rng.integers(0, num_samples, size=(num_bootstrap_samples, num_samples))
    bootstrap_means = indicator_values[bootstrap_indices].mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return num_successes, num_samples, mean_ratio, float(lower), float(upper)


def bootstrap_problem_ratio(
    problem_step_df: pd.DataFrame,
    col: str,
    num_bootstrap_samples: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute bootstrapped per-step ratios for a boolean column."""
    if col not in problem_step_df.columns:
        raise KeyError(f"bootstrap_problem_ratio: column '{col}' not found in problem_step_df.")
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for step, sub_frame in problem_step_df.groupby("step"):
        indicator_values = sub_frame[col].astype(int).to_numpy()
        (
            num_successes,
            num_samples,
            mean_ratio,
            lower,
            upper,
        ) = _bootstrap_ratio_for_values(indicator_values, num_bootstrap_samples, rng)
        rows.append(
            {
                "step": int(step),
                "k": num_successes,
                "n": num_samples,
                "ratio": mean_ratio,
                "lo": lower,
                "hi": upper,
            },
        )
    return pd.DataFrame(rows).sort_values("step")


@dataclass
class FormalExportMeta:
    """Metadata describing the dataset/model associated with an export."""

    dataset: str
    model: str


@dataclass
class FormalAhaExportConfig:
    """Configuration describing how formal AHA events should be exported."""

    meta: FormalExportMeta
    thresholds: FormalThresholds
    gpt_keys: List[str]
    gpt_subset_native: bool
    out_dir: str
    slug: str
    max_chars: int = 4000


@dataclass
class _FormalExportContext:
    """Mutable state used while constructing formal AHA exports."""

    config: FormalAhaExportConfig
    remaining: Set[ExportKey]
    index_map: Dict[ExportKey, pd.Series]


def _build_export_index(
    problem_step_df: pd.DataFrame,
) -> Tuple[Set[ExportKey], Dict[ExportKey, pd.Series]]:
    if "domain" in problem_step_df.columns:
        frame_with_domain = problem_step_df
    else:
        frame_with_domain = problem_step_df.copy()
        frame_with_domain["domain"] = "All"
    target_rows = frame_with_domain.loc[
        frame_with_domain["aha_formal"] == 1,
        ["domain", "problem", "step"],
    ]
    targets = {(str(row["domain"]), str(row["problem"]), int(row["step"])) for _, row in target_rows.iterrows()}
    index_map = {
        (str(row["domain"]), str(row["problem"]), int(row["step"])): row for _, row in frame_with_domain.iterrows()
    }
    return targets, index_map


def _truncate_answer_text(answer: Optional[str], max_chars: int) -> Optional[str]:
    if answer is None or max_chars <= 0:
        return answer
    if len(answer) <= max_chars:
        return answer
    return answer[:max_chars] + " …[truncated]"


def _problem_id_from_record(record: Dict[str, Any]) -> str:
    prob_raw = record.get("problem") or record.get("clue") or record.get("row_key")
    if prob_raw is not None:
        return str(prob_raw)
    dataset_index = record.get("dataset_index")
    return f"idx:{dataset_index}" if dataset_index is not None else "unknown"


def _build_event_dict(
    key: ExportKey,
    row: pd.Series,
    question: str,
    answer: Optional[str],
    config: FormalAhaExportConfig,
) -> Dict[str, Any]:
    domain, problem, step = key
    p_plus = float(row["p_correct_given_shift"]) if np.isfinite(row["p_correct_given_shift"]) else None
    p_base = float(row["freq_correct"])
    delta_gain = p_plus - p_base if p_plus is not None else None
    thresholds = config.thresholds
    return {
        "domain": domain,
        "dataset": config.meta.dataset,
        "model": config.meta.model,
        "problem": problem,
        "step": int(step),
        "n_samples": int(row["n_samples"]),
        "p_correct": p_base,
        "p_shift_rate": float(row["aha_rate_gpt"]),
        "p_correct_given_shift": float(p_plus) if p_plus is not None else None,
        "delta_gain_at_shift": float(delta_gain) if delta_gain is not None else None,
        "thresholds": {
            "delta1": float(thresholds.delta1),
            "delta2": float(thresholds.delta2),
            "delta3": float(thresholds.delta3) if thresholds.delta3 is not None else None,
            "min_prior_steps": int(thresholds.min_prior_steps),
        },
        "question": question,
        "answer": answer,
    }


def _write_empty_export(config: FormalAhaExportConfig) -> Tuple[str, str, int]:
    json_path = os.path.join(
        config.out_dir,
        f"formal_aha_events__{config.slug}.json",
    )
    jsonl_path = os.path.join(
        config.out_dir,
        f"formal_aha_events__{config.slug}.jsonl",
    )
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump([], json_file)
    with open(jsonl_path, "w", encoding="utf-8"):
        pass
    return json_path, jsonl_path, 0


def _write_export_events(
    events: List[Dict[str, Any]],
    config: FormalAhaExportConfig,
) -> Tuple[str, str, int]:
    json_path = os.path.join(
        config.out_dir,
        f"formal_aha_events__{config.slug}.json",
    )
    jsonl_path = os.path.join(
        config.out_dir,
        f"formal_aha_events__{config.slug}.jsonl",
    )
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(events, json_file, ensure_ascii=False, indent=2)
    with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for event in events:
            jsonl_file.write(json.dumps(event, ensure_ascii=False) + "\n")
    return json_path, jsonl_path, len(events)


def _maybe_build_event_for_record(
    domain: str,
    record: Dict[str, Any],
    step_from_name: Optional[int],
    context: _FormalExportContext,
) -> Optional[Dict[str, Any]]:
    step_value = record.get("step", step_from_name)
    if step_value is None:
        return None
    step = int(step_value)
    domain_key = str(domain)
    problem = _problem_id_from_record(record)
    key = (domain_key, problem, step)
    if key not in context.remaining:
        return None
    pass1_dict = record.get("pass1") or {}
    if not isinstance(pass1_dict, dict):
        return None
    aha_flag = aha_gpt_for_rec(
        pass1_dict,
        record,
        context.config.gpt_subset_native,
        context.config.gpt_keys,
        domain_key,
    )
    if aha_flag != 1:
        return None
    row = context.index_map.get(key)
    if row is None:
        return None
    answer = _truncate_answer_text(extract_pass_answer(pass1_dict), context.config.max_chars)
    context.remaining.discard(key)
    return _build_event_dict(
        key=key,
        row=row,
        question=problem,
        answer=answer,
        config=context.config,
    )


def _collect_events_for_domain(
    domain: str,
    files: List[str],
    context: _FormalExportContext,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if not files or not context.remaining:
        return events
    for path in files:
        if not context.remaining:
            break
        step_from_name = nat_step_from_path(path)
        for record in iter_records_from_file(path):
            if not context.remaining:
                break
            event = _maybe_build_event_for_record(
                domain=domain,
                record=record,
                step_from_name=step_from_name,
                context=context,
            )
            if event is not None:
                events.append(event)
    return events


def export_formal_aha_json_with_text(
    problem_step_df: pd.DataFrame,
    files_by_domain: Dict[str, List[str]],
    config: FormalAhaExportConfig,
) -> Tuple[str, str, int]:
    """Export the formal AHA events alongside truncated answer text."""
    if not EXPORT_REQUIRED_COLUMNS.issubset(problem_step_df.columns):
        missing = EXPORT_REQUIRED_COLUMNS - set(problem_step_df.columns)
        raise ValueError(
            f"export_formal_aha_json_with_text: missing columns: {missing}",
        )
    targets, index_map = _build_export_index(problem_step_df)
    if not targets:
        return _write_empty_export(config)
    context = _FormalExportContext(
        config=config,
        remaining=set(targets),
        index_map=index_map,
    )
    events: List[Dict[str, Any]] = []
    for domain_files in files_by_domain.items():
        if not context.remaining:
            break
        domain, files = domain_files
        events.extend(_collect_events_for_domain(domain, files, context))
    if context.remaining:
        for domain, problem, step in sorted(context.remaining):
            row = context.index_map[(domain, problem, step)]
            events.append(
                _build_event_dict(
                    key=(domain, problem, step),
                    row=row,
                    question=problem,
                    answer=None,
                    config=config,
                ),
            )
    return _write_export_events(events, config)


def build_positive_delta_flags(
    problem_step_df: pd.DataFrame,
) -> Dict[str, Dict[int, bool]]:
    """
    Return per-domain/step flags indicating whether formal shifts show positive gain.

    True means the mean gain-at-shift exceeds 0.12 among rows with ``aha_any_gpt==1``.
    """
    flags: Dict[str, Dict[int, bool]] = {}
    if "domain" in problem_step_df.columns:
        groups = problem_step_df.groupby(["domain", "step"], sort=False)
    else:
        frame = problem_step_df.copy()
        frame["domain"] = "All"
        groups = frame.groupby(["domain", "step"], sort=False)
    for (domain, step), sub in groups:
        mask = np.isfinite(sub["p_correct_given_shift"].to_numpy())
        mask &= sub["aha_any_gpt"].to_numpy() == 1
        if mask.any():
            delta = sub.loc[mask, "p_correct_given_shift"].to_numpy() - sub.loc[mask, "freq_correct"].to_numpy()
            flag = bool(np.nanmean(delta) > 0.12)
        else:
            flag = False
        flags.setdefault(str(domain), {})[int(step)] = flag
    return flags


__all__ = [
    "FormalExportMeta",
    "FormalAhaExportConfig",
    "make_formal_thresholds",
    "build_formal_thresholds_from_args",
    "mark_formal_pairs",
    "bootstrap_problem_ratio",
    "export_formal_aha_json_with_text",
    "build_positive_delta_flags",
]
