"""
Helpers for loading pass1/pass2 rows and computing Aha/uncertainty features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.analysis.common.problem_utils import resolve_problem_identifier
from src.analysis.core import iter_pass1_records
from src.analysis.utils import (
    coerce_bool,
    extract_pass1_and_step,
    gpt_keys_for_mode,
)


@dataclass(frozen=True)
class AhaContext:
    """Behavioral knobs for deriving aha labels on a pass row."""

    gpt_fn: Callable[[Dict[str, Any], Dict[str, Any]], int]
    gate_by_words: bool
    include_forced_flag: bool


@dataclass(frozen=True)
class UncertaintyContext:
    """Fields/metrics describing how to extract uncertainty values."""

    field: str
    measure: str


@dataclass(frozen=True)
class PassRowContext:
    """Reusable metadata/behavior needed to build per-pass rows."""

    pair_id: int
    pass_id: int
    problem: str
    step: int
    aha: AhaContext
    uncertainty: UncertaintyContext


@dataclass(frozen=True)
class RowBuilderContext:
    """Bundle of shared per-run row-building configuration."""

    pass1_aha: AhaContext
    pass2_aha: AhaContext
    uncertainty: UncertaintyContext


def _aha_words(record: Dict[str, Any]) -> int:
    flag = coerce_bool(record.get("has_reconsider_cue"))
    markers = record.get("reconsider_markers") or []
    if isinstance(markers, list) and (
        "injected_cue" in markers or "forced_insight" in markers
    ):
        return 0
    return 0 if flag is None else int(flag)


def _aha_gpt_from_mode(
    pass_obj: Dict[str, Any],
    rec: Dict[str, Any],
    mode: str,
) -> int:
    for key in gpt_keys_for_mode(mode):
        value = pass_obj.get(key, rec.get(key, None))
        if value is not None and coerce_bool(value) == 1:
            return 1
    return 0


def _aha_gpt_canonical(pass_payload: Dict[str, Any], rec: Dict[str, Any]) -> int:
    return _aha_gpt_from_mode(pass_payload, rec, "canonical")


def _aha_gpt_broad(pass_payload: Dict[str, Any], rec: Dict[str, Any]) -> int:
    return _aha_gpt_from_mode(pass_payload, rec, "broad")


def _first_num(record: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for key in keys:
        value = record.get(key)
        if isinstance(value, (int, float, np.floating)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except (TypeError, ValueError):  # pragma: no cover - defensive parsing
                continue
    return None


def _any_key_contains(
    record: Dict[str, Any],
    must: List[str],
    any_of: List[str],
) -> Optional[float]:
    for key, value in record.items():
        key_lower = key.lower()
        if all(token in key_lower for token in must) and any(
            token in key_lower for token in any_of
        ):
            if isinstance(value, (int, float, np.floating)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except (TypeError, ValueError):  # pragma: no cover - defensive parsing
                    continue
    return None


def extract_uncertainty_or_ppx(
    pass_dict: Dict[str, Any],
    unc_field: str,
    measure: str,
) -> Optional[float]:
    """
    Extract an entropy/perplexity signal from ``pass_dict``.
    """
    if pass_dict is None:
        return None
    normalized = {str(k): v for k, v in pass_dict.items()}

    if unc_field == "answer":
        ppx = _any_key_contains(normalized, ["perplexity"], ["answer"]) or _first_num(
            normalized,
            ["answer_perplexity", "perplexity_answer"],
        )
        entropy_nats = _first_num(
            normalized,
            ["answer_entropy_nats", "entropy_answer_nats"],
        ) or _any_key_contains(normalized, ["entropy", "nats"], ["answer"])
        entropy = _first_num(
            normalized,
            ["entropy_answer", "answer_entropy"],
        ) or _any_key_contains(normalized, ["entropy"], ["answer"])
    elif unc_field == "think":
        ppx = _any_key_contains(normalized, ["perplexity"], ["think"])
        entropy_nats = _first_num(normalized, ["think_entropy_nats"])
        entropy = _first_num(
            normalized,
            ["entropy_think", "think_entropy"],
        ) or _any_key_contains(normalized, ["entropy"], ["think"])
    else:
        ppx = _first_num(normalized, ["perplexity"]) or _any_key_contains(
            normalized,
            ["perplexity"],
            [""],
        )
        entropy_nats = _first_num(
            normalized,
            ["overall_entropy_nats", "entropy_nats"],
        )
        entropy = _first_num(normalized, ["overall_entropy", "entropy"])

    return _measure_value(measure, normalized, ppx, entropy_nats, entropy)


def _measure_value(
    measure: str,
    value_map: Dict[str, Any],
    ppx: Optional[float],
    ent_n: Optional[float],
    ent: Optional[float],
) -> Optional[float]:
    if measure == "perplexity":
        return _perplexity_value(value_map, ppx, ent_n, ent)
    return _entropy_value(value_map, ppx, ent_n, ent)


def _perplexity_value(
    value_map: Dict[str, Any],
    ppx: Optional[float],
    ent_n: Optional[float],
    ent: Optional[float],
) -> Optional[float]:
    value: Optional[float] = None
    if ppx is not None:
        value = float(ppx)
    elif ent_n is not None:
        value = float(np.exp(ent_n))
    elif ent is not None:
        value = float(np.exp(ent))
    else:
        ent_any = _any_key_contains(value_map, ["entropy"], [""])
        if ent_any is not None:
            value = float(np.exp(ent_any))
    return value


def _entropy_value(
    value_map: Dict[str, Any],
    ppx: Optional[float],
    ent_n: Optional[float],
    ent: Optional[float],
) -> Optional[float]:
    value: Optional[float] = None
    if ent_n is not None:
        value = float(ent_n)
    elif ent is not None:
        value = float(ent)
    elif ppx is not None and ppx > 0:
        value = float(np.log(ppx))
    else:
        ent_any = _any_key_contains(value_map, ["entropy"], [""])
        if ent_any is not None:
            value = float(ent_any)
    return value


def _extract_prompt_key(pass_dict: Dict[str, Any], rec: Dict[str, Any]) -> str:
    for key in (
        "prompt_id",
        "prompt_key",
        "prompt_variant",
        "template_id",
        "prompt_name",
        "prompt_version",
    ):
        value = (pass_dict or {}).get(key, rec.get(key, None))
        if value not in (None, ""):
            return str(value)
    text_value = (pass_dict or {}).get("prompt") or rec.get("prompt")
    if isinstance(text_value, str) and text_value:
        return f"textlen{len(text_value)}"
    return "unknown"


def _detect_forced_insight(pass_dict: Dict[str, Any], rec: Dict[str, Any]) -> int:
    pass_markers = pass_dict.get("reconsider_markers") or []
    record_markers = rec.get("reconsider_markers") or []
    markers = set(pass_markers + record_markers)
    prompt_variant_value = (
        pass_dict.get("prompt_variant") or rec.get("prompt_variant") or ""
    )
    variant_lower = str(prompt_variant_value).lower()
    forced = (
        ("forced_insight" in markers)
        or ("injected_cue" in markers)
        or ("force" in variant_lower and "insight" in variant_lower)
        or ("reconsider" in variant_lower and "force" in variant_lower)
    )
    return int(bool(forced))


def _build_pass_row(
    pass_payload: Dict[str, Any],
    record: Dict[str, Any],
    context: PassRowContext,
) -> Optional[Dict[str, Any]]:
    correctness = coerce_bool(pass_payload.get("is_correct_pred"))
    if correctness is None:
        return None
    words_flag = _aha_words(pass_payload)
    gpt_raw = context.aha.gpt_fn(pass_payload, record)
    gpt_flag = (
        int(gpt_raw and words_flag)
        if context.aha.gate_by_words
        else int(gpt_raw)
    )
    uncertainty = extract_uncertainty_or_ppx(
        pass_payload,
        context.uncertainty.field,
        context.uncertainty.measure,
    )
    row = {
        "pair_id": context.pair_id,
        "pass_id": context.pass_id,
        "problem": context.problem,
        "step": context.step,
        "prompt_key": _extract_prompt_key(pass_payload, record),
        "correct": int(correctness),
        "aha_words": int(words_flag),
        "aha_gpt": int(gpt_flag),
        "unc": None if uncertainty is None else float(uncertainty),
    }
    if context.aha.include_forced_flag:
        row["forced_insight"] = int(_detect_forced_insight(pass_payload, record))
    return row


def _rows_for_record(
    record: Dict[str, Any],
    step_from_name: Optional[int],
    pair_id: int,
    builder_ctx: RowBuilderContext,
) -> List[Dict[str, Any]]:
    """Build pass-level rows for a single record."""
    pass1_payload, step_value = extract_pass1_and_step(record, step_from_name)
    if not pass1_payload or step_value is None:
        return []

    base_problem = str(resolve_problem_identifier(record, fallback=f"unk:{pair_id}"))
    step_int = int(step_value)
    rows: List[Dict[str, Any]] = []
    for pass_id, payload, aha_ctx in _iter_pass_payloads(record, pass1_payload, builder_ctx):
        pass_context = PassRowContext(
            pair_id=pair_id,
            pass_id=pass_id,
            problem=base_problem,
            step=step_int,
            aha=aha_ctx,
            uncertainty=builder_ctx.uncertainty,
        )
        row = _build_pass_row(payload, record, pass_context)
        if row is not None:
            rows.append(row)
    return rows


def _iter_pass_payloads(
    record: Dict[str, Any],
    pass1_payload: Dict[str, Any],
    builder_ctx: RowBuilderContext,
):
    """Yield (pass_id, payload, aha_ctx) tuples for available passes."""
    yield (1, pass1_payload, builder_ctx.pass1_aha)
    pass2_payload = record.get("pass2")
    if isinstance(pass2_payload, dict):
        yield (2, pass2_payload, builder_ctx.pass2_aha)


def load_rows(
    files: List[str],
    gpt_mode: str = "canonical",
    gate_gpt_by_words: bool = True,
    unc_field: str = "answer",
    measure: str = "entropy",
) -> pd.DataFrame:
    """
    Load pass1/pass2 samples needed for the uncertainty-bucket analysis.
    """
    rows: List[Dict[str, Any]] = []
    gpt_fn = _aha_gpt_canonical if gpt_mode == "canonical" else _aha_gpt_broad
    unc_context = UncertaintyContext(field=unc_field, measure=measure)
    aha_pass1 = AhaContext(
        gpt_fn=gpt_fn,
        gate_by_words=gate_gpt_by_words,
        include_forced_flag=False,
    )
    aha_pass2 = AhaContext(
        gpt_fn=gpt_fn,
        gate_by_words=gate_gpt_by_words,
        include_forced_flag=True,
    )

    builder_ctx = RowBuilderContext(
        pass1_aha=aha_pass1,
        pass2_aha=aha_pass2,
        uncertainty=unc_context,
    )

    for pair_id, (_, step_from_name, record) in enumerate(iter_pass1_records(files)):
        rows.extend(
            _rows_for_record(
                record=record,
                step_from_name=step_from_name,
                pair_id=pair_id,
                builder_ctx=builder_ctx,
            ),
        )

    rows_df = pd.DataFrame(rows)
    if rows_df.empty:
        raise SystemExit("No usable rows (pass1/pass2) found.")
    return rows_df


def _prepare_bucket_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a filtered copy containing only rows with defined uncertainty."""
    bucket_df = frame.copy()
    if "unc" not in bucket_df.columns or bucket_df["unc"].dropna().empty:
        raise SystemExit(
            "No pass1 rows with uncertainty/entropy. Ensure your records carry "
            "entropy fields (e.g., entropy, entropy_answer, entropy_think).",
        )
    return bucket_df[~bucket_df["unc"].isna()].copy()


def _assign_fixed_perplexity_buckets(
    bucket_df: pd.DataFrame,
    custom_edges: Optional[List[float]],
) -> pd.DataFrame:
    """Apply fixed-edge perplexity buckets."""
    if not custom_edges:
        raise ValueError(
            "fixed bucketing requires --bucket_edges like 0.2,0.5,1,1.5,2,3",
        )
    bucket_edges = sorted(set(custom_edges))
    result = bucket_df.copy()
    result["perplexity_bucket"] = pd.cut(
        result["unc"],
        bins=bucket_edges,
        include_lowest=True,
    )
    return result


def _assign_quantile_perplexity_buckets(
    bucket_df: pd.DataFrame,
    n_buckets: int,
) -> pd.DataFrame:
    """Apply quantile-based perplexity buckets."""
    result = bucket_df.copy()
    result["perplexity_bucket"] = pd.qcut(
        result["unc"],
        q=n_buckets,
        duplicates="drop",
    )
    return result


def add_perplexity_buckets(
    df_pass1: pd.DataFrame,
    n_buckets: int,
    method: str,
    custom_edges: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Annotate ``df_pass1`` with quantile- or edge-based uncertainty buckets.
    """
    bucket_df = _prepare_bucket_frame(df_pass1)
    if method == "fixed":
        bucket_df = _assign_fixed_perplexity_buckets(bucket_df, custom_edges)
    else:
        bucket_df = _assign_quantile_perplexity_buckets(bucket_df, n_buckets)
    bucket_df["perplexity_bucket"] = bucket_df["perplexity_bucket"].astype(str)
    return bucket_df
