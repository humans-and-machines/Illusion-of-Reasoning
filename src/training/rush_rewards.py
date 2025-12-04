# -*- coding: utf-8 -*-
"""
rush_rewards.py — Dense Rush Hour reward helpers for GRPO training.

This module was factored out of rewards_core to keep that module within
Pylint's size limits while preserving the public Rush rewards API:

    - rush_solution_shaped
    - rush_solution_exact

The implementation mirrors the original rush section from rewards_core.py.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class _RushRewardInputs:
    """Bundle Rush reward inputs to keep function signatures small."""

    answer: Any | None = None
    gold: Any | None = None
    board_str: str | None = None
    board_size: int | None = None
    gold_moves: int | None = None

    @classmethod
    def from_kwargs(
        cls,
        overrides: Dict[str, Any],
        base: "_RushRewardInputs | None" = None,
    ) -> "_RushRewardInputs":
        """Merge overrides with an optional base inputs object."""
        base_inputs = base or cls()
        return cls(
            answer=overrides.get("answer", base_inputs.answer),
            gold=overrides.get("gold", base_inputs.gold),
            board_str=overrides.get("board_str", base_inputs.board_str),
            board_size=overrides.get("board_size", overrides.get("N", base_inputs.board_size)),
            gold_moves=overrides.get("gold_moves", base_inputs.gold_moves),
        )


# ---------- Token utilities ----------

# Directions allowed; accept uppercase "V" too.
_TOKEN_RE = re.compile(r"^[A-Z][<>^vV]\d+$")


def _parse_and_normalize_token(token: str) -> Optional[Tuple[str, str, int]]:
    """
    Parse a single token (e.g., "Bv3", "A>2", "C^1") and normalize to:
      (piece_upper, dir_norm, steps_int)
    where dir_norm ∈ {"<", ">", "^", "v"}.
    """
    if not isinstance(token, str):
        return None
    token = token.strip().replace(" ", "")
    if not _TOKEN_RE.match(token):
        return None

    piece = token[0].upper()
    direction_raw = token[1]
    direction = "v" if direction_raw in ("v", "V") else direction_raw
    try:
        steps = int(token[2:])
    except (TypeError, ValueError):
        return None
    if steps <= 0:
        return None
    return (piece, direction, steps)


def _looks_like_token_list(text: str) -> bool:
    """Return True if text can be parsed as a comma-separated list of tokens."""
    normalized = (text or "").strip().replace(" ", "")
    if not normalized:
        return False
    parts = normalized.split(",")
    for raw_token in parts:
        token_tuple = _parse_and_normalize_token(raw_token)
        if token_tuple is None:
            return False
    return True


def _canon_seq(seq: Any) -> Optional[List[str]]:
    """
    Canonicalize a move sequence (string like "Bv1,A>1" or list of tokens)
    into a list of normalized tokens ["Bv1", "A>1"], merging consecutive
    duplicates of the same (piece, dir) by summing steps.

    Returns None if invalid.
    """
    if isinstance(seq, (list, tuple)):
        parts = [str(element) for element in seq]
    elif isinstance(seq, str):
        seq_str = seq.strip().replace(" ", "")
        if not seq_str:
            return None
        parts = seq_str.split(",")
    else:
        return None

    tokens: List[Tuple[str, str, int]] = []
    for raw_token in parts:
        token_tuple = _parse_and_normalize_token(raw_token)
        if token_tuple is None:
            return None
        tokens.append(token_tuple)

    # Merge consecutive same (piece, dir).
    merged: List[Tuple[str, str, int]] = []
    for piece, direction, steps in tokens:
        if merged and merged[-1][0] == piece and merged[-1][1] == direction:
            merged[-1] = (piece, direction, merged[-1][2] + steps)
        else:
            merged.append((piece, direction, steps))
    return [f"{piece}{direction}{steps}" for (piece, direction, steps) in merged]


def _extract_answer_block(text: str) -> Optional[str]:
    """Prefer <answer>...</answer>; otherwise find the first token list."""
    if not isinstance(text, str):
        return None
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.S | re.I)
    if match:
        candidate = match.group(1).strip()
        if _looks_like_token_list(candidate):
            return candidate
    # Fallback: tolerant regex for tokens.
    match_tokens = re.search(
        r"([A-Za-z][<>^vV]\d+(?:\s*,\s*[A-Za-z][<>^vV]\d+)*)",
        text,
    )
    return match_tokens.group(1) if match_tokens else None


# Accept commas, whitespace, newlines, or semicolons between tokens.
_TOKEN_SCAN = re.compile(r"([A-Za-z])\s*([><\^vVUDLR])\s*([0-9]+)")


def _canon_seq_from_text(text: Any) -> Optional[List[str]]:
    """Extract tokens anywhere in the prediction (favoring <answer>...</answer> if present)."""
    if isinstance(text, (list, tuple)):
        raw_text = ",".join(str(token) for token in text)
    else:
        raw_text = str(text or "")

    # Prefer the <answer> block if present.
    match = re.search(r"(?is)<answer>\s*(.*?)\s*</answer>", raw_text)
    scan_text = match.group(1) if match else raw_text

    moves: List[Tuple[str, str, int]] = []
    for match_obj in _TOKEN_SCAN.finditer(scan_text):
        piece = match_obj.group(1).upper()
        direction_raw = match_obj.group(2)
        # Map symbols as-is; map letters case-insensitively to canonical symbols.
        if direction_raw in ("<", ">", "^", "v"):
            direction = direction_raw
        else:
            direction = {
                "U": "^",
                "D": "v",
                "L": "<",
                "R": ">",
                "V": "v",
            }.get(direction_raw.upper(), direction_raw)

        steps = int(match_obj.group(3))
        if steps <= 0:
            continue
        moves.append((piece, direction, steps))

    if not moves:
        return None

    # Merge consecutive same (piece, dir).
    merged_moves: List[Tuple[str, str, int]] = []
    for piece, direction, steps in moves:
        if merged_moves and merged_moves[-1][0] == piece and merged_moves[-1][1] == direction:
            merged_moves[-1] = (
                piece,
                direction,
                merged_moves[-1][2] + steps,
            )
        else:
            merged_moves.append((piece, direction, steps))
    return [f"{piece}{direction}{steps}" for (piece, direction, steps) in merged_moves]


def _len_tokens(tokens: Optional[List[str]]) -> int:
    return 0 if not tokens else len(tokens)


# ---------- Prompt extraction ----------


def _stringify_prompt(prompts: Any) -> str:
    """
    Robustly produce a single string from common prompt representations:
    - str
    - list[str]
    - list[dict] with "content" fields
    - dict with "prompt" / "content" fields
    """
    if isinstance(prompts, str):
        return prompts
    if isinstance(prompts, dict):
        for key in ("content", "prompt", "text"):
            if key in prompts and isinstance(prompts[key], str):
                return prompts[key]
    if isinstance(prompts, (list, tuple)):
        chunks: List[str] = []
        for item in prompts:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                # OpenAI-style chat message: {"role": "user", "content": "..."}.
                val = item.get("content")
                if isinstance(val, str):
                    chunks.append(val)
                elif isinstance(val, (list, tuple)):
                    # Some SDKs split content into parts.
                    chunks.extend([str(x) for x in val])
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(prompts)


def _extract_puzzle_from_prompt(
    prompt: str,
) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """Extract (board_string, size, gold_moves) from a natural-language prompt."""
    size = None
    match_size = re.search(r"Board\s*size\s*:\s*(\d+)\s*[xX×]\s*(\d+)", prompt)
    if match_size:
        rows = int(match_size.group(1))
        cols = int(match_size.group(2))
        if rows == cols:
            size = rows

    # Accept board on next line; allow whitespace inside, then strip it.
    match_board = re.search(r"(?is)Board\s*:\s*([A-Za-zx\s]+)", prompt)
    board = None
    if match_board:
        board_raw = match_board.group(1)
        board = re.sub(r"\s+", "", board_raw)

    # If we know N and captured more than needed, crop to N*N.
    if size is not None and board:
        need = size * size
        if len(board) >= need:
            board = board[:need]
        else:
            # Try to find a contiguous run of length N*N anywhere (fallback).
            pattern = re.compile(rf"\b([A-Za-zx]{{{need}}})\b")
            match_contig = pattern.search(prompt)
            if match_contig:
                board = match_contig.group(1)

    # If N unknown but board exists, infer from perfect square length.
    if size is None and board:
        root = int(round(math.sqrt(len(board))))
        if root * root == len(board):
            size = root

    match_moves = re.search(
        r"(?:Minimal|Optimal)\s+(?:length\s*\(moves\)|moves?)\s*:\s*(\d+)",
        prompt,
        flags=re.I,
    )
    gold_moves = int(match_moves.group(1)) if match_moves else None
    return board, size, gold_moves


# ---------- Board & simulation ----------


class Board:
    """
    Rush Hour board:
      - grid: NxN row-major string
          "o" = empty, "x" = wall, "A".."Z" = cars (length 2 or 3 typically)
      - "A" is the target car (assumed horizontal); goal is A's rightmost cell at col N-1.
    """

    def __init__(self, grid_str: str, size: int):
        """Initialise a board from a flat ``grid_str`` and side length."""
        self.size = size
        grid = (grid_str or "").strip()
        if len(grid) != size * size:
            raise ValueError(f"Board string length {len(grid)} != N*N ({size * size})")
        self.grid: List[List[str]] = [list(grid[row * size : (row + 1) * size]) for row in range(size)]
        self.cars: Dict[str, List[Tuple[int, int]]] = self._index_cars()
        self.orient: Dict[str, str] = self._orientations()  # "H" or "V"

    def clone(self) -> "Board":
        """Return a deep copy of this board."""
        clone_board = Board("o" * (self.size * self.size), self.size)
        clone_board.grid = [row[:] for row in self.grid]
        clone_board.cars = {key: value[:] for key, value in self.cars.items()}
        clone_board.orient = dict(self.orient)
        return clone_board

    def _index_cars(self) -> Dict[str, List[Tuple[int, int]]]:
        cars: Dict[str, List[Tuple[int, int]]] = {}
        for row_idx in range(self.size):
            for col_idx in range(self.size):
                char = self.grid[row_idx][col_idx]
                if char not in ("o", "x"):
                    cars.setdefault(char, []).append((row_idx, col_idx))
        for positions in cars.values():
            positions.sort()
        return cars

    def _orientations(self) -> Dict[str, str]:
        orient: Dict[str, str] = {}
        for car, cells in self.cars.items():
            if len(cells) == 1:
                orient[car] = "H"  # default for single-cells
                continue
            rows = {row for (row, _) in cells}
            cols = {col for (_, col) in cells}
            if len(rows) == 1:
                orient[car] = "H"
            elif len(cols) == 1:
                orient[car] = "V"
            else:
                # Malformed; pick H to avoid crashing.
                orient[car] = "H"
        return orient

    def is_solved(self) -> bool:
        """Return True if car 'A' has reached the right edge."""
        if "A" not in self.cars:
            return False
        max_col = max(col for (_, col) in self.cars["A"])
        return max_col == self.size - 1

    def a_row_and_rightmost(self) -> Tuple[int, int]:
        """Return (row_index, rightmost_col) for car 'A', or (-1, -1)."""
        if "A" not in self.cars or not self.cars["A"]:
            return (-1, -1)
        row = self.cars["A"][0][0]
        rightmost = max(col for (_, col) in self.cars["A"])
        return row, rightmost

    def blockers_and_distance(self) -> Tuple[int, int]:
        """
        Heuristic components:
          - blockers: count of cars/walls in A's row strictly to A's right
          - distance: (N - 1 - rightmost_A)
        """
        row, rightmost = self.a_row_and_rightmost()
        if row < 0:
            return (0, 0)
        distance = (self.size - 1) - rightmost
        blockers = 0
        for col in range(rightmost + 1, self.size):
            char = self.grid[row][col]
            if char == "x":
                blockers += 1
            elif char != "o":
                blockers += 1
        return blockers, distance

    def _cells_for(self, car: str) -> List[Tuple[int, int]]:
        return self.cars.get(car, [])

    def _clear_cells(self, cells: Iterable[Tuple[int, int]]) -> None:
        for row, col in cells:
            self.grid[row][col] = "o"

    def _occupy_cells(self, car: str, cells: Iterable[Tuple[int, int]]) -> None:
        cells = list(cells)
        for row, col in cells:
            self.grid[row][col] = car
        self.cars[car] = sorted(cells)

    def _step_move(self, car: str, direction: str) -> bool:
        """
        Attempt to move ``car`` one cell in ``direction``.

        Returns True if the move is legal and applied, otherwise False.
        """
        cells = self._cells_for(car)
        if not cells:
            return False

        orientation = self.orient.get(car, "H")
        if direction in "<>" and orientation != "H":
            return False
        if direction in "^v" and orientation != "V":
            return False

        blocked = False
        new_cells: List[Tuple[int, int]]

        if direction == "<":
            leftmost = min(col for (_, col) in cells)
            row = cells[0][0]
            next_col = leftmost - 1
            blocked = next_col < 0 or self.grid[row][next_col] != "o"
            new_cells = [(row, col - 1) for (row, col) in cells]
        elif direction == ">":
            rightmost = max(col for (_, col) in cells)
            row = cells[0][0]
            next_col = rightmost + 1
            blocked = next_col >= self.size or self.grid[row][next_col] != "o"
            new_cells = [(row, col + 1) for (row, col) in cells]
        elif direction == "^":
            top = min(row for (row, _) in cells)
            col = cells[0][1]
            next_row = top - 1
            blocked = next_row < 0 or self.grid[next_row][col] != "o"
            new_cells = [(row - 1, col) for (row, col) in cells]
        else:  # "v"
            bottom = max(row for (row, _) in cells)
            col = cells[0][1]
            next_row = bottom + 1
            blocked = next_row >= self.size or self.grid[next_row][col] != "o"
            new_cells = [(row + 1, col) for (row, col) in cells]

        if blocked:
            return False

        self._clear_cells(cells)
        self._occupy_cells(car, new_cells)
        return True

    def apply_token(self, token: str) -> bool:
        """
        Apply a compact move token like ``A>2`` to the board.

        Returns True if all steps are legal and applied, otherwise False.
        """
        car = token[0]
        direction = token[1]
        steps = int(token[2:])
        for _ in range(steps):
            if not self._step_move(car, direction):
                return False
        return True


def _simulate_prefix(board: Board, tokens: List[str]) -> Tuple[int, bool, Board]:
    """
    Apply tokens in order until one fails or goal is reached.

    Returns (valid_prefix_len, solved, final_board_state).
    """
    board_copy = board.clone()
    valid = 0
    for token in tokens:
        token_ok = board_copy.apply_token(token)
        if not token_ok:
            return valid, False, board_copy
        valid += 1
        if board_copy.is_solved():
            return valid, True, board_copy
    return valid, board_copy.is_solved(), board_copy


# ---------- Gold handling ----------


def _canon_gold_candidates(gold: Any) -> List[List[str]]:
    if gold is None:
        return []
    # Unwrap ["A>2,Bv5"] → "A>2,Bv5".
    if isinstance(gold, (list, tuple)) and len(gold) == 1 and isinstance(gold[0], str):
        gold = gold[0]
    if isinstance(gold, str) or (isinstance(gold, (list, tuple)) and (not gold or isinstance(gold[0], str))):
        seq = _canon_seq(gold)
        return [seq] if seq is not None else []

    candidates: List[List[str]] = []
    if isinstance(gold, (list, tuple)):
        for gold_entry in gold:
            seq = _canon_seq(gold_entry)
            if seq is not None:
                candidates.append(seq)
    return candidates


# ---------- Rewards ----------

# --- Think-length helpers (bonus for longer reasoning) ---
_THINK_BLOCK = re.compile(r"(?is)<think>\s*(.*?)\s*</think>")


def _count_think_tokens(text: str) -> int:
    match = _THINK_BLOCK.search(text or "")
    if not match:
        return 0
    # Whitespace-delimited proxy for token count (keeps it fast & robust).
    return len(re.findall(r"\S+", match.group(1)))


def _rush_formatting_bonus(
    pred_text: str,
    think_min_tokens: int,
    think_full_tokens: int,
    think_bonus_cap: float,
) -> float:
    """Tiny formatting bonus when we spot token patterns and sufficient thinking."""
    fmt_bonus = 0.0
    think_bonus = 0.0

    try:
        num_tokens = len(list(_TOKEN_SCAN.finditer(pred_text)))
    except re.error:
        num_tokens = 0
    if num_tokens > 0:
        fmt_bonus = 0.01

    num_think_tokens = _count_think_tokens(pred_text)
    if num_think_tokens > think_min_tokens:
        ramp = (num_think_tokens - think_min_tokens) / max(
            1,
            (think_full_tokens - think_min_tokens),
        )
        think_bonus = think_bonus_cap * min(1.0, max(0.0, ramp))

    return min(0.01, fmt_bonus + think_bonus)


def _rush_build_board_and_moves(
    prompts: Any,
    board_str: str | None,
    size: int | None,
) -> Tuple[Board | None, int | None]:
    """Construct a Board from explicit params or from the prompt, plus gold moves."""
    board: Board | None = None
    gold_moves_from_prompt: int | None = None

    if board_str is not None and size is not None:
        try:
            board = Board(board_str, int(size))
        except (TypeError, ValueError):
            board = None
        return board, gold_moves_from_prompt

    prompt_text = _stringify_prompt(prompts)
    prompt_board_str, prompt_size, gold_moves_from_prompt = _extract_puzzle_from_prompt(
        prompt_text,
    )
    if prompt_board_str and prompt_size:
        try:
            board = Board(prompt_board_str, int(prompt_size))
        except (TypeError, ValueError):
            board = None

    return board, gold_moves_from_prompt


def _rush_prefix_credit(pred_can: List[str], gold_cands: List[List[str]]) -> float:
    """Fractional credit based on longest common prefix against any gold sequence."""
    if not gold_cands or not pred_can:
        return 0.0

    best_lcp = 0
    denom = 1
    for gold_candidate in gold_cands:
        if not gold_candidate:
            continue
        lcp = 0
        for pred_token, gold_token in zip(pred_can, gold_candidate):
            if pred_token != gold_token:
                break
            lcp += 1
        best_lcp = max(best_lcp, lcp)
        denom = max(denom, _len_tokens(gold_candidate))

    return best_lcp / max(1, denom)


def _solve_term_for_target(sequence_len: int, target: int) -> float:
    """Shaping term that prefers shorter sequences up to a target length."""
    if sequence_len <= target:
        return 1.0
    return 1.0 / (1.0 + (sequence_len - target))


def _phi_progress(board: Board, final_board: Board, solved: bool) -> float:
    """Heuristic Φ term based on change in blockers + distance."""
    blockers_start, dist_start = board.blockers_and_distance()
    blockers_end, dist_end = final_board.blockers_and_distance()
    heuristic_start = blockers_start + dist_start
    heuristic_end = blockers_end + dist_end
    if heuristic_start > 0:
        return max(0.0, (heuristic_start - heuristic_end) / heuristic_start)
    return 1.0 if solved else 0.0


def _rush_solve_terms(
    pred_can: List[str],
    board: Board | None,
    target_moves: int | None,
) -> Tuple[float, float, float]:
    """
    Compute (prefix_fraction, solve_term, phi_term) for a predicted sequence.

    prefix_fraction is the legal-prefix fraction if a board is available, else 0.
    """
    prefix_fraction = 0.0
    solve_term = 0.0
    phi_term = 0.0

    if board is None:
        if target_moves is not None:
            sequence_len = _len_tokens(pred_can)
            solve_term = _solve_term_for_target(sequence_len, int(target_moves))
        return prefix_fraction, solve_term, phi_term

    valid_steps, solved, final_board = _simulate_prefix(board, pred_can)
    prefix_fraction = valid_steps / max(1, _len_tokens(pred_can))

    if solved:
        sequence_len = _len_tokens(pred_can)
        target = target_moves if target_moves is not None else sequence_len
        solve_term = _solve_term_for_target(sequence_len, int(target))

    phi_term = _phi_progress(board, final_board, solved)
    return prefix_fraction, solve_term, phi_term


def rush_solution_shaped(
    *,
    prompts,
    completions,
    reward_inputs: _RushRewardInputs | None = None,
    **kwargs,
) -> List[float]:
    """
    Dense, shaped reward in ``[0, 1]`` for Rush Hour that works with or without a board.

    When a board is provided (``board_str`` + ``N``), legality and Φ-progress are used.
    When no board is provided, the reward still gives:

    - exact match against gold candidate sequences
    - prefix / longest-common-prefix credit vs gold
    - gold-only "solve optimality": shorter sequences are better based on ``gold_moves``

    ``reward_inputs`` can pre-bundle gold/board hints; explicit keyword overrides like
    ``gold_moves`` or ``board_str`` still take precedence.
    """
    inputs = _RushRewardInputs.from_kwargs(kwargs, reward_inputs)
    ctx: Dict[str, Any] = {}
    ctx["weights"] = {
        "exact": float(kwargs.get("w_exact", 0.65)),
        "solve": float(kwargs.get("w_solve", 0.20)),
        "prefix": float(kwargs.get("w_prefix", 0.10)),
        "phi": float(kwargs.get("w_phi", 0.05)),
    }

    if isinstance(completions, str):
        completions = [completions]
    else:
        completions = list(completions)
    ctx["completions"] = completions

    ctx["gold_cands"] = _canon_gold_candidates(
        inputs.gold or inputs.answer or kwargs.get("answers") or kwargs.get("gold_answers"),
    )
    ctx["board"], ctx["board_target_moves"] = _rush_build_board_and_moves(
        prompts,
        inputs.board_str,
        inputs.board_size,
    )

    if inputs.gold_moves is not None:
        ctx["target_moves"] = inputs.gold_moves
    elif ctx["board_target_moves"] is not None:
        ctx["target_moves"] = ctx["board_target_moves"]
    else:
        ctx["target_moves"] = min(
            (_len_tokens(candidate) for candidate in ctx["gold_cands"] if candidate is not None),
            default=None,
        )

    ctx["think_min_tokens"] = int(kwargs.get("think_min_tokens", 25))
    ctx["think_full_tokens"] = int(kwargs.get("think_full_tokens", 100))
    ctx["think_bonus_cap"] = float(kwargs.get("think_bonus_cap", 0.02))

    out_scores: List[float] = []
    for pred in completions:
        pred_can = _canon_seq_from_text(str(pred or ""))

        # 0) Formatting bonus when we see at least one valid token but parser fails.
        if pred_can is None:
            out_scores.append(
                _rush_formatting_bonus(
                    str(pred or ""),
                    ctx["think_min_tokens"],
                    ctx["think_full_tokens"],
                    ctx["think_bonus_cap"],
                )
            )
            continue

        solve_terms = _rush_solve_terms(
            pred_can,
            ctx["board"],
            ctx["target_moves"],
        )

        out_scores.append(
            float(
                max(
                    0.0,
                    min(
                        1.0,
                        ctx["weights"]["exact"]
                        * (
                            1.0
                            if any(pred_can == candidate for candidate in ctx["gold_cands"] if candidate is not None)
                            else 0.0
                        )
                        + ctx["weights"]["solve"] * solve_terms[1]
                        + ctx["weights"]["prefix"]
                        * max(
                            _rush_prefix_credit(pred_can, ctx["gold_cands"]),
                            solve_terms[0],
                        )
                        + ctx["weights"]["phi"] * solve_terms[2],
                    ),
                )
            )
        )

    return out_scores


def rush_solution_exact(
    *,
    prompts,
    completions,
    answer=None,
    gold=None,
    **kwargs,
) -> List[float]:
    """
    Exact-match Rush Hour reward with a small formatting bonus.

    Returns 1.0 for canonical sequence matches, otherwise 0.0, with a tiny bonus
    when the parser fails but the format shows some valid token structure.
    """
    # prompts accepted for API parity; not used directly in this reward.
    _ = prompts
    if isinstance(completions, str):
        completions = [completions]
    else:
        completions = list(completions)

    gold_cands = _canon_gold_candidates(
        gold or answer or kwargs.get("answers") or kwargs.get("gold_answers"),
    )
    think_min_tokens = int(kwargs.get("think_min_tokens", 25))
    think_full_tokens = int(kwargs.get("think_full_tokens", 100))
    think_bonus_cap = float(kwargs.get("think_bonus_cap", 0.02))

    scores: List[float] = []
    for pred in completions:
        pred_text = str(pred or "")
        pred_can = _canon_seq_from_text(pred_text)

        # 0) Formatting bonus when we see at least one valid token but parser fails.
        if pred_can is None:
            bonus = _rush_formatting_bonus(
                pred_text,
                think_min_tokens,
                think_full_tokens,
                think_bonus_cap,
            )
            scores.append(bonus)
            continue

        exact_match = any(pred_can == candidate for candidate in gold_cands if candidate is not None)
        scores.append(1.0 if exact_match else 0.0)

    return scores


__all__ = ["rush_solution_shaped", "rush_solution_exact"]
