"""Core helpers and module re-exports for RQ1–RQ3 analyses."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..io import iter_pass1_samples_by_domain
from ..labels import aha_gpt_for_rec
from ..utils import (
    FormalThresholds,
    build_roots_by_temp_from_templates,
    coerce_bool,
    coerce_float,
    formal_flags_with_gain,
    formal_prior_ok,
    nat_step_from_path,
    parse_temp_from_dir,
)


FORMAL_REQUIRED_COLUMNS = {
    "step",
    "problem",
    "freq_correct",
    "aha_rate_gpt",
    "aha_any_gpt",
    "p_correct_given_shift",
}


def make_formal_thresholds(
    delta1: float,
    delta2: float,
    min_prior_steps: int,
    delta3: Optional[float],
) -> FormalThresholds:
    """
    Convenience constructor for :class:`FormalThresholds` used by multiple scripts.
    """
    return FormalThresholds(
        delta1=float(delta1),
        delta2=float(delta2),
        min_prior_steps=int(min_prior_steps),
        delta3=None if delta3 is None else float(delta3),
    )


def build_formal_thresholds_from_args(
    delta1: float,
    delta2: float,
    min_prior_steps: int,
    delta3: Optional[float],
) -> FormalThresholds:
    """
    Shared wrapper to construct :class:`FormalThresholds` from raw args.
    """
    return make_formal_thresholds(
        delta1=delta1,
        delta2=delta2,
        min_prior_steps=min_prior_steps,
        delta3=delta3,
    )


@dataclass
class LoadRowsConfig:
    """Configuration for loading per-record/sample rows with GPT and Carpark gating."""

    gpt_keys: List[str]
    gpt_subset_native: bool
    min_step: Optional[int]
    max_step: Optional[int]
    carpark_success_fn: Callable[[Any], Optional[int]]


@dataclass
class FormalFlagConfig:
    """Bundle of thresholds and output settings for formal-aha flag computation."""

    thresholds: FormalThresholds
    out_column: str = "aha_formal_ps"
    formal_prior_ok_fn: Callable[
        [np.ndarray, np.ndarray, int, FormalThresholds],
        bool,
    ] = formal_prior_ok
    group_keys: Optional[List[str]] = None


def compute_correct_and_shift(
    domain: str,
    pass1_data: Dict[str, Any],
    record: Dict[str, Any],
    *,
    config: Optional[LoadRowsConfig] = None,
    **legacy_gating: Any,
) -> Optional[Tuple[int, int]]:
    """
    Compute (correct, shift) flags for a single PASS-1-style sample.

    The ``carpark_success_fn`` is only used for Carpark-style domains.
    Returns ``None`` if correctness cannot be determined.
    """
    if config is None:
        try:
            config = LoadRowsConfig(
                gpt_keys=legacy_gating.pop("gpt_keys"),
                gpt_subset_native=legacy_gating.pop("gpt_subset_native"),
                min_step=legacy_gating.pop("min_step", None),
                max_step=legacy_gating.pop("max_step", None),
                carpark_success_fn=legacy_gating.pop("carpark_success_fn"),
            )
        except KeyError as exc:
            raise TypeError(
                "compute_correct_and_shift requires a LoadRowsConfig or legacy gating kwargs "
                "(gpt_keys, gpt_subset_native, carpark_success_fn)"
            ) from exc
        if legacy_gating:
            raise TypeError(f"Unexpected gating arguments: {sorted(legacy_gating)}")

    dom_lower = str(domain).lower()
    if dom_lower.startswith("carpark"):
        soft_reward = coerce_float(
            record.get("soft_reward", pass1_data.get("soft_reward")),
        )
        success_flag = config.carpark_success_fn(soft_reward)
        if success_flag is None:
            return None
        correct = int(success_flag)
    else:
        success = coerce_bool(pass1_data.get("is_correct_pred"))
        if success is None:
            return None
        correct = int(success)

    shift = aha_gpt_for_rec(
        pass1_data,
        record,
        config.gpt_subset_native,
        config.gpt_keys,
        domain,
    )
    return correct, int(shift)


def aha_series_for_group(sub: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract sorted series used in formal-aha scans.

    Returns freq_correct, aha_rate_gpt, aha_any_gpt arrays sorted by step.
    """
    sub = sub.sort_values("step")
    freq = sub["freq_correct"].to_numpy(float)
    rate = sub["aha_rate_gpt"].to_numpy(float)
    shift = sub["aha_any_gpt"].to_numpy(int)
    return freq, rate, shift


def compute_formal_flags_for_group(
    sub: pd.DataFrame,
    thresholds: FormalThresholds,
    formal_prior_ok_fn: Callable[[np.ndarray, np.ndarray, int, FormalThresholds], bool],
) -> np.ndarray:
    """
    Compute formal-aha flags for a grouped problem-step slice.

    The returned array has length ``len(sub)`` and uses the provided
    :func:`formal_prior_ok_fn` together with :func:`aha_series_for_group`.
    """
    freq, rate, shift = aha_series_for_group(sub)
    flags = np.zeros(len(sub), dtype=int)
    for idx in range(len(sub)):
        prior_ok = formal_prior_ok_fn(freq, rate, idx, thresholds)
        flags[idx] = int(prior_ok and (shift[idx] == 1))
    return flags


def add_standard_formal_flags(
    frame: pd.DataFrame,
    group_keys: List[str],
    *,
    config: Optional[FormalFlagConfig] = None,
    **legacy_thresholds: Any,
) -> pd.DataFrame:
    """
    Convenience wrapper around :func:`add_formal_flags_column` using
    the shared :func:`formal_prior_ok` predicate.
    """
    if config is not None:
        if config.group_keys is None:
            config.group_keys = list(group_keys)
        return add_formal_flags_column(frame, config=config)

    legacy_thresholds.setdefault("formal_prior_ok_fn", formal_prior_ok)
    return add_formal_flags_column(frame, group_keys=group_keys, **legacy_thresholds)


def add_formal_flags_column(
    frame: pd.DataFrame,
    group_keys: Optional[List[str]] = None,
    *,
    config: Optional[FormalFlagConfig] = None,
    **legacy_thresholds: Any,
) -> pd.DataFrame:
    """
    Add a per-row Formal Aha flag column grouped by ``group_keys``.

    The input ``frame`` must contain ``step``, ``freq_correct``,
    ``aha_rate_gpt`` and ``aha_any_gpt`` columns. The returned DataFrame
    is a sorted copy of ``frame`` with an added integer ``out_column``.
    """
    required_columns = {"step", "freq_correct", "aha_rate_gpt", "aha_any_gpt"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError("add_formal_flags_column: missing columns: " + ", ".join(sorted(missing_columns)))

    if group_keys is None and config is not None:
        group_keys = config.group_keys
    if not group_keys:
        raise ValueError("add_formal_flags_column requires non-empty group_keys")

    if config is None:
        try:
            thresholds = build_formal_thresholds_from_args(
                delta1=legacy_thresholds.pop("delta1"),
                delta2=legacy_thresholds.pop("delta2"),
                min_prior_steps=legacy_thresholds.pop("min_prior_steps"),
                delta3=legacy_thresholds.pop("delta3", None),
            )
        except KeyError as exc:
            raise TypeError(
                "FormalFlagConfig or legacy thresholds (delta1, delta2, min_prior_steps) are required"
            ) from exc
        config = FormalFlagConfig(
            thresholds=thresholds,
            out_column=legacy_thresholds.pop("out_column", "aha_formal_ps"),
            formal_prior_ok_fn=legacy_thresholds.pop("formal_prior_ok_fn", formal_prior_ok),
        )
        if legacy_thresholds:
            raise TypeError(f"Unexpected formal flag arguments: {sorted(legacy_thresholds)}")

    sort_keys = list(group_keys) + ["step"]
    grouped_frame = frame.sort_values(sort_keys).reset_index(drop=True).copy()
    flags = np.zeros(len(grouped_frame), dtype=int)

    for _, sub in grouped_frame.groupby(group_keys, sort=False):
        sub_flags = compute_formal_flags_for_group(
            sub,
            thresholds=config.thresholds,
            formal_prior_ok_fn=config.formal_prior_ok_fn,
        )
        flags[sub.index.to_numpy()] = sub_flags

    grouped_frame[config.out_column] = flags
    return grouped_frame


def iter_pass1_records(
    files: Iterable[str],
) -> Iterable[Tuple[str, Optional[int], Dict[str, object]]]:
    """
    Yield ``(path, step_from_name, rec)`` tuples for PASS-1-style JSONL logs.

    This mirrors the common pattern of scanning files, inferring step from the
    filename, and iterating JSONL records.
    """
    for path in files:
        step_from_name = nat_step_from_path(path)
        with open(path, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                try:
                    record = json.loads(stripped_line)
                except json.JSONDecodeError:
                    continue
                yield path, step_from_name, record


def build_problem_step_from_samples(
    samples_df: pd.DataFrame,
    include_native: bool = False,
    native_col: str = "aha_native",
) -> pd.DataFrame:
    """
    Aggregate sample-level rows to per-(problem, step) summaries.

    When ``include_native`` is True, native aha columns are aggregated as well.
    """
    group_keys = ["domain", "step", "problem"] if "domain" in samples_df.columns else ["step", "problem"]

    aggregation_spec: Dict[str, Tuple[str, str]] = {
        "n_samples": ("correct", "size"),
        "freq_correct": ("correct", "mean"),
        "aha_any_gpt": ("aha_gpt", "max"),
        "aha_rate_gpt": ("aha_gpt", "mean"),
    }
    if include_native:
        aggregation_spec.update(
            {
                "aha_any_native": (native_col, "max"),
                "aha_rate_native": (native_col, "mean"),
            },
        )

    base = (
        samples_df.groupby(group_keys, as_index=False)
        .agg(**aggregation_spec)
        .sort_values(group_keys)
        .reset_index(drop=True)
    )

    def _pcs_row(group_df: pd.DataFrame) -> pd.Series:
        mask = group_df["aha_gpt"] == 1
        if mask.any():
            return pd.Series(
                {
                    "p_correct_given_shift": float(
                        group_df.loc[mask, "correct"].mean(),
                    ),
                },
            )
        return pd.Series({"p_correct_given_shift": np.nan})

    try:
        pcs = samples_df.groupby(group_keys).apply(_pcs_row, include_groups=False)
    except TypeError:  # pragma: no cover - older pandas
        pcs = samples_df.groupby(group_keys).apply(_pcs_row)
    pcs = pcs.reset_index()
    problem_step_df = base.merge(pcs, on=group_keys, how="left")

    for col in ("n_samples", "aha_any_gpt"):
        problem_step_df[col] = problem_step_df[col].astype(int)
    if include_native:
        problem_step_df["aha_any_native"] = problem_step_df["aha_any_native"].astype(int)
    for col in ("freq_correct", "aha_rate_gpt", "p_correct_given_shift"):
        problem_step_df[col] = problem_step_df[col].astype(float)
    if include_native:
        problem_step_df["aha_rate_native"] = problem_step_df["aha_rate_native"].astype(float)

    return problem_step_df


def build_problem_step_for_formal(d_samples: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience wrapper for Formal Aha analyses.

    Aggregates sample-level rows to per-(problem, step) summaries and also
    includes native Aha statistics, treating ``aha_words`` as the native
    column.
    """
    return build_problem_step_from_samples(
        d_samples,
        include_native=True,
        native_col="aha_words",
    )


def mark_formal_pairs_with_gain(
    problem_step_df: pd.DataFrame,
    thresholds: FormalThresholds,
) -> pd.DataFrame:
    """
    Mark (problem, step) pairs that satisfy the Formal Aha definition with a
    gain-at-shift threshold.

    Expects the columns listed in :data:`FORMAL_REQUIRED_COLUMNS` and returns
    a copy of the input with an ``aha_formal_pair`` indicator column.
    """
    missing_columns = FORMAL_REQUIRED_COLUMNS - set(problem_step_df.columns)
    if missing_columns:
        raise ValueError(f"Formal marking missing columns: {missing_columns}")

    sorted_df = problem_step_df.sort_values(
        ["problem", "step"],
    ).reset_index(drop=True)
    flags = np.zeros(len(sorted_df), dtype=int)

    for _, group_df in sorted_df.groupby("problem", sort=False):
        freq, rate, shift = aha_series_for_group(group_df)
        p_plus = group_df["p_correct_given_shift"].to_numpy(float)
        group_flags = formal_flags_with_gain(
            freq,
            rate,
            shift,
            p_plus,
            thresholds,
        )
        flags[group_df.index.to_numpy()] = group_flags

    result_df = sorted_df.copy()
    result_df["aha_formal_pair"] = flags
    return result_df


def iter_correct_and_shift_samples(
    files_by_domain: Dict[str, List[str]],
    *,
    config: Optional[LoadRowsConfig] = None,
    **legacy_gating: Any,
) -> Iterable[Tuple[str, int, Dict[str, Any], int, int]]:
    """
    Iterate PASS-1 samples and yield correctness and shift flags.

    Yields ``(domain, step, record, correct, shift)`` tuples.
    """
    if config is None:
        try:
            config = LoadRowsConfig(
                gpt_keys=legacy_gating.pop("gpt_keys"),
                gpt_subset_native=legacy_gating.pop("gpt_subset_native"),
                min_step=legacy_gating.pop("min_step", None),
                max_step=legacy_gating.pop("max_step", None),
                carpark_success_fn=legacy_gating.pop("carpark_success_fn"),
            )
        except KeyError as exc:
            raise TypeError(
                "iter_correct_and_shift_samples requires a LoadRowsConfig or legacy gating kwargs "
                "(gpt_keys, gpt_subset_native, carpark_success_fn)"
            ) from exc
        if legacy_gating:
            raise TypeError(f"Unexpected gating arguments: {sorted(legacy_gating)}")

    for domain_name, pass1_data, step, record in iter_pass1_samples_by_domain(
        files_by_domain,
        min_step=config.min_step,
        max_step=config.max_step,
    ):
        result = compute_correct_and_shift(
            domain_name,
            pass1_data,
            record,
            config=config,
        )
        if result is None:
            continue
        correct_flag, shift_flag = result
        yield (
            str(domain_name),
            int(step),
            record,
            int(correct_flag),
            int(shift_flag),
        )


def iter_correct_and_shift_samples_for_config(
    files_by_domain: Dict[str, List[str]],
    config: LoadRowsConfig,
) -> Iterable[Tuple[str, int, Dict[str, Any], int, int]]:
    """
    Convenience wrapper around :func:`iter_correct_and_shift_samples` that
    accepts a :class:`LoadRowsConfig` instance.
    """
    return iter_correct_and_shift_samples(files_by_domain, config=config)


def _classify_domain_from_dir(dirname_lower: str) -> Optional[str]:
    """
    Map a directory name to a canonical domain key for temperature scans.
    """
    if "xword" in dirname_lower:
        return "Crossword"
    if "carpark" in dirname_lower:
        return "Carpark"
    if "math" in dirname_lower:
        if "7b" in dirname_lower:
            return "Math2"
        return "Math"
    if ("low-temp" in dirname_lower or "low_temp" in dirname_lower) and "1.5b" in dirname_lower:
        return "Math"
    return None


def _rank_math_path(path: str) -> Tuple[int, int]:
    """
    Heuristic ranking for Math directories: prefer 1.5B, then Qwen, then Llama.
    """
    basename = os.path.basename(path).lower()
    score = 0
    if "1.5b" in basename:
        score += 300
    if "qwen" in basename:
        score += 200
    if "llama" in basename:
        score += 100
    if "low-temp" in basename or "low_temp" in basename:
        score += 10
    return score, len(path)


def _prefer_math_path(new_path: str, prev_path: Optional[str]) -> bool:
    """
    Decide whether to replace ``prev_path`` with ``new_path`` for Math roots.
    """
    if prev_path is None:
        return True
    return _rank_math_path(new_path) > _rank_math_path(prev_path)


def _log_discovered_roots(mapping: Dict[float, Dict[str, str]]) -> None:
    """
    Log a summary of discovered roots grouped by temperature.
    """
    print("[info] discovered roots by temperature:")
    for temp_value in sorted(mapping):
        domain_map = mapping[temp_value]
        items = ", ".join(f"{key}→{value}" for key, value in domain_map.items())
        print(f"  T={temp_value}: {items}")


def _update_mapping_for_dir(
    mapping: Dict[float, Dict[str, str]],
    root: str,
    dirname: str,
    domain: str,
    temp_value: float,
) -> None:
    """
    Update the discovered-roots mapping for a single directory.
    """
    full_path = os.path.join(root, dirname)
    domain_map = mapping.setdefault(temp_value, {})
    previous_path = domain_map.get(domain)
    if domain == "Math":
        if _prefer_math_path(full_path, previous_path):
            domain_map[domain] = full_path
    else:
        if previous_path is None or len(full_path) > len(previous_path):
            domain_map[domain] = full_path


def discover_roots_by_temp(
    scan_root: str,
    temps: List[float],
    low_alias: float,
    skip_substrings: set,
) -> Dict[float, Dict[str, str]]:
    """
    Shared directory discovery used by temperature_effects and temp_graph.

    Walks ``scan_root`` and discovers per-temperature roots by domain, using
    :func:`parse_temp_from_dir` and simple heuristics to classify domains.
    """
    temps_set = {float(temp) for temp in temps}
    skip = {str(substr).lower() for substr in skip_substrings}
    mapping: Dict[float, Dict[str, str]] = {}

    for root, dirs, _ in os.walk(scan_root):
        for dirname in dirs:
            if any(substr in dirname.lower() for substr in skip):
                continue
            temp_value = parse_temp_from_dir(dirname, low_alias)
            if temp_value is None:
                continue
            if temp_value not in temps_set:
                continue

            domain = _classify_domain_from_dir(dirname.lower())
            if domain is None:
                continue

            _update_mapping_for_dir(
                mapping,
                root,
                dirname,
                domain,
                temp_value,
            )

    if mapping:
        _log_discovered_roots(mapping)
    return mapping


def discover_roots_for_temp_args(
    args: argparse.Namespace,
    skip_set: set,
    include_math3: bool,
) -> Dict[float, Dict[str, str]]:
    """
    Shared helper used by temperature_effects and temp_graph to construct
    the mapping ``{temp -> domain -> root_dir}`` from CLI arguments.
    """
    if args.scan_root:
        return discover_roots_by_temp(
            args.scan_root,
            args.temps,
            args.low_alias,
            skip_set,
        )

    temps = [float(temp) for temp in args.temps]
    math3_tpl = args.math3_tpl if include_math3 else None
    return build_roots_by_temp_from_templates(
        temps=temps,
        crossword_tpl=args.crossword_tpl,
        math_tpl=args.math_tpl,
        math2_tpl=args.math2_tpl,
        math3_tpl=math3_tpl,
        carpark_tpl=args.carpark_tpl,
    )


__all__ = [
    "compute_correct_and_shift",
    "aha_series_for_group",
    "compute_formal_flags_for_group",
    "build_formal_thresholds_from_args",
    "add_formal_flags_column",
    "iter_pass1_records",
    "build_problem_step_from_samples",
    "build_problem_step_for_formal",
    "discover_roots_by_temp",
    "discover_roots_for_temp_args",
]
