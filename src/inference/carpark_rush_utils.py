#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rush Hour canonicalization and soft-match reward helpers shared by carpark_core.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# Single-move token: piece letter + direction + steps (dir accepts v or V)
TOKEN_RE = re.compile(r"^\s*([A-Za-z])([<>^vV])(\d+)\s*$")


def _toklist(seq: str) -> List[str]:
    """'Bv2,A>1' -> ['Bv2','A>1'] (assumes canonicalized input)."""
    if not seq:
        return []
    return [token for token in seq.split(",") if token]


def _split_token(token: str) -> Tuple[str, str, int]:
    """Split canonical token into (piece, direction, steps)."""
    piece = token[0]
    direction = token[1]
    steps = int(token[2:])
    return piece, direction, steps


def _lcs_len(seq_a: List[str], seq_b: List[str]) -> int:
    """Token-level LCS (small sequences -> O(nm) is fine)."""
    n_rows, n_cols = len(seq_a), len(seq_b)
    lcs_row = [0] * (n_cols + 1)
    for row_index in range(1, n_rows + 1):
        prev = 0
        for col_index in range(1, n_cols + 1):
            current = lcs_row[col_index]
            if seq_a[row_index - 1] == seq_b[col_index - 1]:
                lcs_row[col_index] = prev + 1
            else:
                lcs_row[col_index] = max(lcs_row[col_index], lcs_row[col_index - 1])
            prev = current
    return lcs_row[n_cols]


def _multiset_overlap_ratio(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Return Jaccard-style overlap between two token multisets."""
    counts_a, counts_b = Counter(tokens_a), Counter(tokens_b)
    all_tokens = set(counts_a) | set(counts_b)
    intersection = sum(min(counts_a[token], counts_b[token]) for token in all_tokens)
    union = sum(max(counts_a[token], counts_b[token]) for token in all_tokens)
    return (intersection / union) if union else 0.0


def _piece_dir(token: str) -> Tuple[str, str]:
    piece, direction, _ = _split_token(token)
    return piece, direction


def _canon_move(token: str) -> Optional[str]:
    """Canonicalize one move: piece upper, dir in {<,>,^,v}, steps numeric."""
    match = TOKEN_RE.match(token)
    if not match:
        return None
    piece, direction, steps = match.groups()
    piece = piece.upper()
    direction = direction.lower() if direction in ("v", "V") else direction
    return f"{piece}{direction}{steps}"


def _canon_join(tokens: List[str]) -> Optional[str]:
    output_tokens: List[str] = []
    for token in tokens:
        canon = _canon_move(token)
        if canon is None:
            return None
        output_tokens.append(canon)
    return ",".join(output_tokens)


def _canon_rush_string(text: str) -> Optional[str]:
    """Canonicalize a comma-separated move string."""
    text = text.replace("\n", " ").strip()
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        return None
    return _canon_join(parts)


def _canon_rush_generic(value: Any) -> Optional[str]:
    """Canonicalize either a string sequence or a list[str] of tokens."""
    if value is None:
        return None
    if isinstance(value, str):
        return _canon_rush_string(value)
    if isinstance(value, list):
        if all(isinstance(token, str) for token in value):
            return _canon_join(value)
        return None
    return None


def _is_valid_rush(text: Optional[str]) -> bool:
    """Valid iff canonicalization succeeds."""
    return text is not None


def _canon_rush_gold(gold: Any) -> set[str]:
    """
    Return a set of canonical, valid gold answers.

    Accepts:
      - str: 'Bv2,A>1'
      - list[str]: ['Bv2','A>1']
      - list[list[str] | str]: multiple valid alternatives
    """
    output: set[str] = set()
    if gold is None:
        return output
    if isinstance(gold, str) or (
        isinstance(gold, list) and all(isinstance(token, str) for token in gold)
    ):
        canon = _canon_rush_generic(gold)
        if _is_valid_rush(canon):
            output.add(canon)
        return output
    if isinstance(gold, list):
        for alternative in gold:
            canon = _canon_rush_generic(alternative)
            if _is_valid_rush(canon):
                output.add(canon)
        return output
    return output


def _compute_prefix_matches(pred_tokens: List[str], gold_tokens: List[str]) -> int:
    """Return the number of matching prefix tokens."""
    limit = min(len(pred_tokens), len(gold_tokens))
    matches = 0
    for index in range(limit):
        if pred_tokens[index] == gold_tokens[index]:
            matches += 1
        else:
            break
    return matches


def _compute_pos_exact(pred_tokens: List[str], gold_tokens: List[str], gold_len: int) -> float:
    """Return the fraction of positions with exact token matches."""
    limit = min(len(pred_tokens), len(gold_tokens))
    if gold_len == 0 or limit == 0:
        return 0.0
    exact_matches = sum(
        1 for index in range(limit) if pred_tokens[index] == gold_tokens[index]
    )
    return exact_matches / gold_len


def _piece_dir_match_and_step_score(
    pred_token: str,
    gold_token: str,
) -> Tuple[bool, float]:
    """Return (piece_dir_match, step_score) for a token pair."""
    pred_piece, pred_dir = _piece_dir(pred_token)
    gold_piece, gold_dir = _piece_dir(gold_token)
    if (pred_piece, pred_dir) != (gold_piece, gold_dir):
        return False, 0.0
    _, _, pred_steps = _split_token(pred_token)
    _, _, gold_steps = _split_token(gold_token)
    denom = max(gold_steps, 1)
    step_score = max(0.0, 1.0 - abs(pred_steps - gold_steps) / denom)
    return True, step_score


def _compute_piece_dir_and_step_close(
    pred_tokens: List[str],
    gold_tokens: List[str],
    gold_len: int,
) -> Tuple[float, float]:
    """Return (piece_dir, step_close) components for a single gold sequence."""
    piece_dir_matches = 0
    step_closeness_vals: List[float] = []
    limit = min(len(pred_tokens), len(gold_tokens))
    for index in range(limit):
        matches_piece_dir, step_score = _piece_dir_match_and_step_score(
            pred_tokens[index],
            gold_tokens[index],
        )
        if matches_piece_dir:
            piece_dir_matches += 1
        step_closeness_vals.append(step_score)
    piece_dir = piece_dir_matches / gold_len if gold_len > 0 else 0.0
    step_close = (sum(step_closeness_vals) / gold_len) if gold_len > 0 else 0.0
    return piece_dir, step_close


def _score_rush_pair(
    pred_tokens: List[str],
    gold_canon: str,
    *,
    weights: Dict[str, float],
    length_penalty_base: float,
) -> Tuple[float, Dict[str, Any]]:
    """Compute score and component breakdown for a single (pred, gold) pair."""
    gold_tokens = _toklist(gold_canon)
    gold_len = max(1, len(gold_tokens))

    components: Dict[str, Any] = {
        "prefix": _compute_prefix_matches(pred_tokens, gold_tokens) / gold_len,
        "pos_exact": _compute_pos_exact(pred_tokens, gold_tokens, gold_len),
        "lcs": _lcs_len(pred_tokens, gold_tokens) / gold_len,
        "bag_overlap": _multiset_overlap_ratio(pred_tokens, gold_tokens),
    }
    piece_dir, step_close = _compute_piece_dir_and_step_close(
        pred_tokens,
        gold_tokens,
        gold_len,
    )
    components["piece_dir"] = piece_dir
    components["step_close"] = step_close

    length_delta = abs(len(pred_tokens) - len(gold_tokens))
    components["length_delta"] = length_delta
    components["length_penalty"] = length_penalty_base ** length_delta

    base_score = (
        weights["prefix"] * components["prefix"]
        + weights["pos_exact"] * components["pos_exact"]
        + weights["piece_dir"] * components["piece_dir"]
        + weights["step_close"] * components["step_close"]
        + weights["lcs"] * components["lcs"]
        + weights["bag_overlap"] * components["bag_overlap"]
    )
    score = max(0.0, min(1.0, base_score * components["length_penalty"]))
    return float(score), components



def rush_soft_match_reward(
    pred_answer_text: str,
    gold_answer_any: Any,
    *,
    weights: Optional[Dict[str, float]] = None,
    length_penalty_base: float = 0.92,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute a soft match reward in ``[0, 1]`` using only gold sequence(s).

    The reward combines several overlap and alignment components between the
    predicted Rush Hour move sequence and one or more canonicalized gold
    sequences.

    :param pred_answer_text: Raw predicted answer text describing moves.
    :param gold_answer_any: Gold answer representation (string or list of sequences).
    :param weights: Optional mapping of component names to scalar weights.
    :param length_penalty_base: Base used for the exponential length penalty term.
    :returns: Tuple ``(score, detail)`` where ``score`` is a float in ``[0, 1]``
        and ``detail`` contains canonical forms, chosen gold sequence, and
        per-component scores.
    """
    if weights is None:
        weights = {
            "prefix": 0.35,
            "pos_exact": 0.25,
            "piece_dir": 0.15,
            "step_close": 0.10,
            "lcs": 0.10,
            "bag_overlap": 0.05,
        }

    pred_canon = _canon_rush_generic(pred_answer_text)
    gold_set = _canon_rush_gold(gold_answer_any)

    detail: Dict[str, Any] = {
        "pred_canon": pred_canon,
        "gold_canon_options": sorted(list(gold_set)),
        "components": {},
        "picked_gold": None,
    }

    if (not pred_canon) or (not gold_set):
        return 0.0, detail

    pred_tokens = _toklist(pred_canon)

    best_score = 0.0
    best_components: Optional[Dict[str, Any]] = None
    best_gold: Optional[str] = None

    for gold_canon in gold_set:
        if pred_canon == gold_canon:
            components = {
                "prefix": 1.0,
                "pos_exact": 1.0,
                "piece_dir": 1.0,
                "step_close": 1.0,
                "lcs": 1.0,
                "bag_overlap": 1.0,
                "length_penalty": 1.0,
                "length_delta": 0,
            }
            score = 1.0
        else:
            score, components = _score_rush_pair(
                pred_tokens,
                gold_canon,
                weights=weights,
                length_penalty_base=length_penalty_base,
            )

        if score > best_score:
            best_score = score
            best_components = components
            best_gold = gold_canon

    detail["components"] = best_components or {}
    detail["picked_gold"] = best_gold
    return float(best_score), detail
