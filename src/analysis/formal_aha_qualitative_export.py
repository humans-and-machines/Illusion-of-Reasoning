#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export a small qualitative appendix of (Formal) Aha examples.

This is meant to support the paper appendix section that manually inspects a
handful of Formal-Aha-like events. It:
  - scans PASS-1 JSONL outputs under provided roots,
  - aggregates to (problem, step) summaries,
  - marks Formal Aha pairs under configurable thresholds,
  - selects the top-K pairs per domain,
  - extracts a representative shifted trace (prefer correct shifted traces),
  - writes a LaTeX snippet with `promptbox` blocks.

Notes / limitations:
  - For the RHour/Carpark domain the stored evaluation logs are keyed by
    `example_id` (e.g., "idx_42") and often do not embed the full board prompt.
    The export therefore uses `example_id` + gold move list as the "problem"
    descriptor.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.analysis.io import scan_jsonl_files
from src.analysis.labels import aha_gpt_for_rec
from src.analysis.common.pass_extraction import extract_pass_answer
from src.analysis.utils import FormalThresholds, formal_flags_with_gain, gpt_keys_for_mode


_TEMP_RE = re.compile(r"(?:^|[-_])temp[-_](?P<t>[0-9]+(?:\.[0-9]+)?)$", re.I)


def _infer_temp_from_root(root: str) -> Optional[float]:
    name = os.path.basename(os.path.normpath(root))
    match = _TEMP_RE.search(name)
    if not match:
        return None
    try:
        return float(match.group("t"))
    except (TypeError, ValueError):
        return None


def _problem_id(domain: str, rec: Dict[str, Any]) -> Optional[str]:
    if domain == "RHour":
        val = rec.get("example_id")
        return None if val is None else str(val)
    val = rec.get("problem") or rec.get("clue") or rec.get("row_key")
    return None if val is None else str(val)


def _truthy_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(bool(value))
    return None


def _xword_entry(text: Any) -> str:
    """
    Canonicalize a crossword entry for strict comparison.

    Keeps only A–Z and uppercases. (E.g., "ICE-BLUE" -> "ICEBLUE".)
    """
    if text is None:
        return ""
    raw = str(text).strip()
    if not raw:
        return ""
    return "".join(ch for ch in raw.upper() if "A" <= ch <= "Z")


def _xword_correct(pass1: Dict[str, Any], rec: Dict[str, Any]) -> Optional[int]:
    """
    Strict Xword correctness: extracted <answer> must match gold entry.

    This avoids relying on potentially noisy `is_correct_pred` fields when tag
    structure is malformed.
    """
    if not isinstance(pass1, dict) or not isinstance(rec, dict):
        return None
    pred = extract_pass_answer(pass1) or pass1.get("pred_answer")
    pred_entry = _xword_entry(pred)
    gold = rec.get("gold_answer") or rec.get("gold_answer_canon") or rec.get("gold")
    gold_entry = _xword_entry(gold)
    if not pred_entry or not gold_entry:
        return None
    return int(pred_entry == gold_entry)


@dataclass(frozen=True)
class DomainThresholds:
    delta1: float
    delta2: float
    delta3: Optional[float]
    min_prior_steps: int = 2


@dataclass(frozen=True)
class ExampleSpec:
    domain: str
    root: str
    temp: Optional[float]
    problem: str
    step: int
    n_samples: int
    freq_correct: float
    aha_rate: float
    p_correct_given_shift: float
    gain: float
    record: Dict[str, Any]


def _load_samples_for_root(domain: str, root: str, gpt_keys: List[str]) -> pd.DataFrame:
    files = scan_jsonl_files(root, split_substr="test")
    rows: List[Dict[str, Any]] = []
    for path in files:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pass1 = rec.get("pass1") or {}
                if not isinstance(pass1, dict):
                    continue
                step_raw = rec.get("step")
                if step_raw is None:
                    continue
                try:
                    step = int(step_raw)
                except (TypeError, ValueError):
                    continue
                prob = _problem_id(domain, rec)
                if prob is None:
                    continue
                if domain == "Xword":
                    correct = _xword_correct(pass1, rec)
                else:
                    correct_raw = pass1.get("is_correct_pred")
                    correct = _truthy_int(correct_raw)
                if correct is None:
                    continue
                shift = int(
                    aha_gpt_for_rec(
                        pass1,
                        rec,
                        gpt_subset_native=False,
                        gpt_keys=gpt_keys,
                        domain=("Carpark" if domain == "RHour" else domain),
                    ),
                )
                rows.append(
                    {
                        "problem": prob,
                        "step": step,
                        "correct": int(correct),
                        "aha_gpt": int(shift),
                    },
                )
    return pd.DataFrame(rows)


def _build_problem_step(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    group = df.groupby(["problem", "step"], as_index=False)
    base = group.agg(
        n_samples=("correct", "size"),
        freq_correct=("correct", "mean"),
        aha_any_gpt=("aha_gpt", "max"),
        aha_rate_gpt=("aha_gpt", "mean"),
    )

    def _p_correct_shift(group_df: pd.DataFrame) -> float:
        mask = group_df["aha_gpt"] == 1
        if mask.any():
            return float(group_df.loc[mask, "correct"].mean())
        return float("nan")

    try:
        pcs = df.groupby(["problem", "step"]).apply(
            lambda g: pd.Series({"p_correct_given_shift": _p_correct_shift(g)}),
            include_groups=False,
        )
    except TypeError:  # pragma: no cover - older pandas
        pcs = df.groupby(["problem", "step"]).apply(
            lambda g: pd.Series({"p_correct_given_shift": _p_correct_shift(g)}),
        )
    pcs = pcs.reset_index()
    out = base.merge(pcs, on=["problem", "step"], how="left")
    out["gain"] = out["p_correct_given_shift"] - out["freq_correct"]
    return out.sort_values(["problem", "step"]).reset_index(drop=True)


def _mark_formal(problem_step_df: pd.DataFrame, thresholds: DomainThresholds) -> pd.DataFrame:
    if problem_step_df.empty:
        return problem_step_df
    thr = FormalThresholds(
        delta1=float(thresholds.delta1),
        delta2=float(thresholds.delta2),
        min_prior_steps=int(thresholds.min_prior_steps),
        delta3=None if thresholds.delta3 is None else float(thresholds.delta3),
    )
    flags = np.zeros(len(problem_step_df), dtype=int)
    for _, sub in problem_step_df.groupby("problem", sort=False):
        sub = sub.sort_values("step")
        freq = sub["freq_correct"].to_numpy(float)
        rate = sub["aha_rate_gpt"].to_numpy(float)
        shift = sub["aha_any_gpt"].to_numpy(int)
        p_plus = sub["p_correct_given_shift"].to_numpy(float)
        local = formal_flags_with_gain(freq, rate, shift, p_plus, thr)
        flags[sub.index.to_numpy()] = local
    out = problem_step_df.copy()
    out["aha_formal"] = flags
    return out


def _iter_records(root: str) -> Iterable[Tuple[str, Dict[str, Any]]]:
    files = scan_jsonl_files(root, split_substr="test")
    for path in files:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield path, json.loads(line)
                except json.JSONDecodeError:
                    continue


def _pick_representative_record(
    domain: str,
    root: str,
    problem: str,
    step: int,
    gpt_keys: List[str],
) -> Optional[Dict[str, Any]]:
    candidate_shift: Optional[Dict[str, Any]] = None
    candidate_shift_correct: Optional[Dict[str, Any]] = None
    for _, rec in _iter_records(root):
        if int(rec.get("step", -1)) != int(step):
            continue
        if _problem_id(domain, rec) != problem:
            continue
        pass1 = rec.get("pass1") or {}
        if not isinstance(pass1, dict):
            continue
        shift = int(
            aha_gpt_for_rec(
                pass1,
                rec,
                gpt_subset_native=False,
                gpt_keys=gpt_keys,
                domain=("Carpark" if domain == "RHour" else domain),
            ),
        )
        if shift != 1:
            continue
        if domain == "Xword":
            correct = _xword_correct(pass1, rec)
        else:
            correct = _truthy_int(pass1.get("is_correct_pred"))
        if correct == 1 and candidate_shift_correct is None:
            candidate_shift_correct = rec
        if candidate_shift is None:
            candidate_shift = rec
        if candidate_shift_correct is not None:
            break
    return candidate_shift_correct or candidate_shift


def _format_float(value: Optional[float], digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    fmt = f"{{:.{digits}f}}"
    return fmt.format(float(value))


def _format_pp(value: Optional[float], digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    fmt = f"{{:+.{digits}f}}"
    return fmt.format(float(value) * 100.0)


def _record_excerpt(pass1: Dict[str, Any], max_chars: int) -> str:
    before = pass1.get("shift_before_excerpt") or ""
    after = pass1.get("shift_after_excerpt") or ""
    markers = pass1.get("shift_markers_v1") or []
    marker_text = ", ".join(markers) if isinstance(markers, list) else str(markers)
    if before or after:
        text = f"markers: {marker_text}\n…{before}{after}"
    else:
        text = pass1.get("output") or ""
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + " …[truncated]"
    return text


_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*(?:</answer>|$)", re.I | re.S)


def _clean_answer_text(raw: Any, max_chars: int = 240) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    match = _ANSWER_TAG_RE.search(text)
    if match:
        text = match.group(1).strip()
    # Drop obvious tag remnants when outputs are malformed.
    text = re.sub(r"</?(?:think|answer)[^>]*>", "", text, flags=re.I).strip()
    # Prefer a short, last-line answer if possible.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        if 0 < len(last) <= 200:
            text = last
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + " …[truncated]"
    return text


def _describe_problem(domain: str, rec: Dict[str, Any]) -> str:
    if domain == "RHour":
        example_id = rec.get("example_id", "unknown")
        gold = rec.get("gold_answer") or rec.get("gold_answer_canon_set") or []
        return f"example_id: {example_id}\\nGold: {gold}"
    if domain == "Xword":
        clue = rec.get("problem") or rec.get("clue") or rec.get("row_key") or "unknown"
        gold = rec.get("gold_answer") or rec.get("gold_answer_canon") or "unknown"
        return f"Clue: {clue}\\nGold: {_xword_entry(gold)}"
    # Math/Xword store the full prompt in `problem`.
    return str(rec.get("problem") or rec.get("clue") or rec.get("row_key") or "unknown")


def _write_tex(
    out_path: str,
    examples_by_domain: Dict[str, List[ExampleSpec]],
    thresholds_by_domain: Dict[str, DomainThresholds],
    *,
    max_excerpt_chars: int,
) -> None:
    lines: List[str] = []
    lines.append(r"\subsection{Qualitive review of formal ``Aha!'' Moments}")
    lines.append(r"\label{app:qualitative-formal-aha}")
    lines.append("")
    lines.append(
        "We qualitatively inspect a small set of (Formal) ``Aha!'' detections from our stored Qwen2.5--1.5B evaluation outputs. "
        "For each domain we apply the Formal criteria at the problem--checkpoint level and then show representative shifted traces."
    )
    lines.append("")

    for domain in ["Math", "Xword", "RHour"]:
        domain_examples = examples_by_domain.get(domain, [])
        thr = thresholds_by_domain[domain]
        d3 = "None" if thr.delta3 is None else _format_float(thr.delta3, digits=3)
        lines.append(rf"\paragraph{{{domain}.}}")
        lines.append(
            rf"We use $(\delta_1={_format_float(thr.delta1,3)},\,\delta_2={_format_float(thr.delta2,3)},\,"
            rf"\delta_3={d3})$ with \texttt{{min\_prior\_steps}}={thr.min_prior_steps}."
        )
        if domain == "RHour":
            lines.append(
                "Because RHour accuracies are near zero in these stored outputs, we found too few events satisfying a positive gain constraint; "
                "we therefore omit the gain threshold for this qualitative inspection."
            )
        lines.append("")

        if not domain_examples:
            lines.append(r"\emph{No examples found under these settings.}")
            lines.append("")
            continue

        for idx, spec in enumerate(domain_examples, start=1):
            t_str = "--" if spec.temp is None else f"{spec.temp:g}"
            title = f"{domain} example {idx} (T={t_str}, step={spec.step})"
            lines.append(rf"\begin{{promptbox}}{{{title}}}")
            lines.append(
                f"root: {os.path.basename(os.path.normpath(spec.root))}\n"
                f"problem: {spec.problem}\n"
                f"step: {spec.step}\n"
                f"n_samples: {spec.n_samples}\n"
                f"shift_rate: {_format_float(spec.aha_rate,3)}\n"
                f"freq_correct: {_format_float(spec.freq_correct,3)}\n"
                f"p_correct_given_shift: {_format_float(spec.p_correct_given_shift,3)}\n"
                f"gain: {_format_pp(spec.gain)} pp\n"
                "\n"
                "Shift excerpt (PASS-1):\n"
            )
            pass1 = (spec.record.get("pass1") or {}) if isinstance(spec.record.get("pass1"), dict) else {}
            lines.append(_record_excerpt(pass1, max_excerpt_chars))
            lines.append("\n\nPASS-1 <answer>:\n")
            answer = extract_pass_answer(pass1) or pass1.get("pred_answer") or ""
            lines.append(_clean_answer_text(answer, max_chars=260))
            lines.append(r"\end{promptbox}")
            lines.append("")

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def _xword_quality_for_record(record: Dict[str, Any]) -> float:
    """Heuristic quality score for Xword qualitative examples."""
    pass1 = record.get("pass1") or {}
    if not isinstance(pass1, dict):
        return -1e9
    before = str(pass1.get("shift_before_excerpt") or "")
    after = str(pass1.get("shift_after_excerpt") or "")
    excerpt_len = float(len(before + after))
    markers = pass1.get("shift_markers_v1") or []
    if not isinstance(markers, list):
        markers = [str(markers)]
    markers_lower = {str(m).lower() for m in markers}
    marker_bonus = 0.0
    if markers_lower.intersection({"rethink", "re-evaluate", "start over", "restart", "wait", "actually", "instead"}):
        marker_bonus += 2000.0
    if markers_lower.intersection({"doesn't fit", "does not fit", "mismatch"}):
        marker_bonus += 500.0
    answer = extract_pass_answer(pass1) or pass1.get("pred_answer") or ""
    answer_len = float(len(_xword_entry(answer)))
    valid_bonus = 200.0 if bool(pass1.get("valid_tag_structure")) else 0.0
    clue = str(record.get("problem") or "")
    clue_lower = clue.lower()
    abbrev_penalty = 0.0
    if "initial" in clue_lower or "abbr" in clue_lower:
        abbrev_penalty -= 800.0
    if answer_len <= 3:
        abbrev_penalty -= 400.0
    return marker_bonus + excerpt_len + valid_bonus + 15.0 * answer_len + abbrev_penalty


def _load_xword_cache(
    root: str,
    gpt_keys: List[str],
) -> Tuple[pd.DataFrame, Dict[Tuple[str, int], Dict[str, Any]]]:
    """
    Load Xword sample rows and a representative shifted-record index for a root.

    The record index maps (problem, step) -> a representative PASS-1 record that
    has a detected shift; correct shifted traces are preferred when available.
    """
    files = scan_jsonl_files(root, split_substr="test")
    rows: List[Dict[str, Any]] = []
    rep_correct: Dict[Tuple[str, int], Dict[str, Any]] = {}

    for path in files:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pass1 = rec.get("pass1") or {}
                if not isinstance(pass1, dict):
                    continue
                step_raw = rec.get("step")
                if step_raw is None:
                    continue
                try:
                    step = int(step_raw)
                except (TypeError, ValueError):
                    continue
                problem = rec.get("problem") or rec.get("clue") or rec.get("row_key")
                if problem is None:
                    continue
                problem_str = str(problem)

                correct = _xword_correct(pass1, rec)
                if correct is None:
                    continue
                shift = int(
                    aha_gpt_for_rec(
                        pass1,
                        rec,
                        gpt_subset_native=False,
                        gpt_keys=gpt_keys,
                        domain="Crossword",
                    ),
                )
                rows.append(
                    {
                        "problem": problem_str,
                        "step": step,
                        "correct": int(correct),
                        "aha_gpt": int(shift),
                    },
                )
                if shift != 1:
                    continue
                key = (problem_str, step)
                if int(correct) == 1:
                    rep_correct.setdefault(key, rec)

    df = pd.DataFrame(rows)
    problem_step_df = _build_problem_step(df) if not df.empty else df
    # Only keep correct shifted traces for qualitative examples.
    return problem_step_df, dict(rep_correct)


def _select_xword_examples_from_cache(
    caches: Sequence[Tuple[str, pd.DataFrame, Dict[Tuple[str, int], Dict[str, Any]]]],
    thresholds: DomainThresholds,
    *,
    top_k: int,
) -> List[ExampleSpec]:
    candidates: List[Tuple[float, float, str, pd.Series, Dict[str, Any]]] = []
    for root, ps_df, rep_map in caches:
        if ps_df is None or getattr(ps_df, "empty", True):
            continue
        marked = _mark_formal(ps_df, thresholds)
        flagged = marked.loc[marked["aha_formal"] == 1].copy()
        if flagged.empty:
            continue
        flagged["gain_sort"] = flagged["gain"].where(np.isfinite(flagged["gain"]), -1e9)
        for _, row in flagged.iterrows():
            key = (str(row["problem"]), int(row["step"]))
            record = rep_map.get(key)
            if record is None:
                continue
            quality = _xword_quality_for_record(record)
            primary = float(row["gain_sort"]) * 1000.0 + float(quality)
            candidates.append((primary, quality, root, row, record))

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    picked: List[ExampleSpec] = []
    seen_clues = set()
    for _, _, root, row, record in candidates:
        if len(picked) >= int(top_k):
            break
        clue_key = str(record.get("problem") or row["problem"])
        if clue_key in seen_clues:
            continue
        seen_clues.add(clue_key)
        prob_desc = _describe_problem("Xword", record)
        temp = _infer_temp_from_root(root)
        gain = float(row["gain"]) if np.isfinite(row["gain"]) else float("nan")
        picked.append(
            ExampleSpec(
                domain="Xword",
                root=root,
                temp=temp,
                problem=prob_desc,
                step=int(row["step"]),
                n_samples=int(row["n_samples"]),
                freq_correct=float(row["freq_correct"]),
                aha_rate=float(row["aha_rate_gpt"]),
                p_correct_given_shift=float(row["p_correct_given_shift"])
                if np.isfinite(row["p_correct_given_shift"])
                else float("nan"),
                gain=gain,
                record=record,
            ),
        )
    return picked


def _select_examples(
    domain: str,
    roots: Sequence[str],
    thresholds: DomainThresholds,
    *,
    top_k: int,
) -> List[ExampleSpec]:
    gpt_keys = gpt_keys_for_mode("canonical")
    candidates: List[Tuple[float, float, str, pd.Series, Optional[Dict[str, Any]]]] = []
    for root in roots:
        df = _load_samples_for_root(domain, root, gpt_keys)
        if df.empty:
            continue
        ps = _build_problem_step(df)
        ps = _mark_formal(ps, thresholds)
        flagged = ps.loc[ps["aha_formal"] == 1].copy()
        if flagged.empty:
            continue
        # Prefer higher gain; when gain is NaN (no correct shifted traces), treat as very low.
        flagged["gain_sort"] = flagged["gain"].where(np.isfinite(flagged["gain"]), -1e9)
        for _, row in flagged.iterrows():
            quality = 0.0
            record: Optional[Dict[str, Any]] = None

            # Xword is very sparse; spend extra effort to rank events by the
            # clarity of the shifted trace (excerpt length, marker strength).
            if domain == "Xword":
                problem = str(row["problem"])
                step = int(row["step"])
                record = _pick_representative_record(domain, root, problem, step, gpt_keys)
                if record is None:
                    continue
                pass1 = record.get("pass1") or {}
                before = str(pass1.get("shift_before_excerpt") or "")
                after = str(pass1.get("shift_after_excerpt") or "")
                excerpt_len = float(len(before + after))

                markers = pass1.get("shift_markers_v1") or []
                if not isinstance(markers, list):
                    markers = [str(markers)]
                markers_lower = {str(m).lower() for m in markers}
                marker_bonus = 0.0
                if markers_lower.intersection({"rethink", "re-evaluate", "start over", "restart", "wait", "actually", "instead"}):
                    marker_bonus += 2000.0
                if markers_lower.intersection({"doesn't fit", "does not fit", "mismatch"}):
                    marker_bonus += 500.0

                answer = extract_pass_answer(pass1) or pass1.get("pred_answer") or ""
                answer_len = float(len(_xword_entry(answer)))
                valid_bonus = 200.0 if bool(pass1.get("valid_tag_structure")) else 0.0

                clue = str(record.get("problem") or "")
                clue_lower = clue.lower()
                abbrev_penalty = 0.0
                if "initial" in clue_lower or "abbr" in clue_lower:
                    abbrev_penalty -= 800.0
                if answer_len <= 3:
                    abbrev_penalty -= 400.0

                quality = marker_bonus + excerpt_len + valid_bonus + 15.0 * answer_len + abbrev_penalty

            primary = float(row["gain_sort"])
            if domain == "Xword":
                # Prefer clear qualitative shifts even if the mean gain is smaller,
                # since many high-gain Xword events are trivial abbreviations.
                primary = primary * 1000.0 + float(quality)

            candidates.append((float(primary), float(quality), root, row, record))

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    picked: List[ExampleSpec] = []
    seen = set()
    for _, _, root, row, record in candidates:
        if len(picked) >= int(top_k):
            break
        problem = str(row["problem"])
        step = int(row["step"])
        key = (problem, step)
        if key in seen:
            continue
        seen.add(key)
        if record is None:
            record = _pick_representative_record(domain, root, problem, step, gpt_keys)
        if record is None:
            continue
        prob_desc = _describe_problem(domain, record)
        temp = _infer_temp_from_root(root)
        gain = float(row["gain"]) if np.isfinite(row["gain"]) else float("nan")
        picked.append(
            ExampleSpec(
                domain=domain,
                root=root,
                temp=temp,
                problem=prob_desc,
                step=step,
                n_samples=int(row["n_samples"]),
                freq_correct=float(row["freq_correct"]),
                aha_rate=float(row["aha_rate_gpt"]),
                p_correct_given_shift=float(row["p_correct_given_shift"])
                if np.isfinite(row["p_correct_given_shift"])
                else float("nan"),
                gain=gain,
                record=record,
            ),
        )
    return picked


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math_roots", nargs="+", default=[], help="Result roots for Math (Qwen2.5-1.5B)")
    parser.add_argument("--xword_roots", nargs="+", default=[], help="Result roots for Xword (Qwen2.5-1.5B)")
    parser.add_argument("--rhour_roots", nargs="+", default=[], help="Result roots for RHour/Carpark (Qwen2.5-1.5B)")
    parser.add_argument("--out_tex", required=True, help="Output LaTeX file path")
    parser.add_argument("--top_k", type=int, default=5, help="Examples per domain")
    parser.add_argument("--max_excerpt_chars", type=int, default=900, help="Max excerpt length per example")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)

    thresholds_by_domain = {
        # Math needs slightly looser prior-failure/stability thresholds to yield examples.
        "Math": DomainThresholds(delta1=0.25, delta2=0.25, delta3=0.0, min_prior_steps=2),
        # Xword is sparse in stored outputs; use permissive thresholds but require positive gain.
        "Xword": DomainThresholds(delta1=0.5, delta2=0.5, delta3=0.0, min_prior_steps=2),
        # RHour accuracies are near-zero; positive-gain formal events are extremely rare.
        "RHour": DomainThresholds(delta1=0.25, delta2=0.25, delta3=None, min_prior_steps=2),
    }

    examples_by_domain: Dict[str, List[ExampleSpec]] = {}
    examples_by_domain["Math"] = _select_examples(
        "Math",
        args.math_roots,
        thresholds_by_domain["Math"],
        top_k=args.top_k,
    )

    # Xword: build a cache once (avoid repeatedly re-scanning JSONLs while trying
    # multiple threshold settings).
    gpt_keys = gpt_keys_for_mode("canonical")
    xword_caches: List[Tuple[str, pd.DataFrame, Dict[Tuple[str, int], Dict[str, Any]]]] = []
    for root in args.xword_roots:
        ps_df, rep_map = _load_xword_cache(root, gpt_keys)
        xword_caches.append((root, ps_df, rep_map))

    xword_threshold_candidates = [
        thresholds_by_domain["Xword"],
        DomainThresholds(delta1=1.0, delta2=0.5, delta3=0.0, min_prior_steps=2),
        DomainThresholds(delta1=1.0, delta2=1.0, delta3=0.0, min_prior_steps=2),
    ]
    best_thr = thresholds_by_domain["Xword"]
    best_examples: List[ExampleSpec] = []
    best_score = float("-inf")
    for thr in xword_threshold_candidates:
        examples = _select_xword_examples_from_cache(xword_caches, thr, top_k=args.top_k)
        if len(examples) < int(args.top_k):
            continue
        score = float(sum(_xword_quality_for_record(ex.record) for ex in examples))
        if score > best_score:
            best_score = score
            best_thr = thr
            best_examples = examples
    thresholds_by_domain["Xword"] = best_thr
    examples_by_domain["Xword"] = best_examples
    examples_by_domain["RHour"] = _select_examples(
        "RHour",
        args.rhour_roots,
        thresholds_by_domain["RHour"],
        top_k=args.top_k,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_tex)), exist_ok=True)
    _write_tex(
        args.out_tex,
        examples_by_domain,
        thresholds_by_domain,
        max_excerpt_chars=int(args.max_excerpt_chars),
    )
    counts = {k: len(v) for k, v in examples_by_domain.items()}
    print(f"[info] wrote {args.out_tex} (examples: {counts})")


if __name__ == "__main__":
    main()
