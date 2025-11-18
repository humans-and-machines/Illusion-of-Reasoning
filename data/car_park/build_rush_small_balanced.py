#!/usr/bin/env python3
"""
Build a balanced Rush Hour dataset for 4x4 / 5x5 / 6x6.

Two generation modes per size:
1) Enumeration (fast for 4x4/5x5): enumerate canonical boards, then solve all.
2) Sampling ("build-and-add"): repeatedly generate random canonical boards,
   solve them, and keep only solvable ones until hitting a per-size target.

Why sampling? Enumeration for 6x6 explodes. Sampling lets you stop as soon
as you have enough *solvable* rows, instead of enumerating huge numbers
you'll later discard.

Features:
- Per-size constraints on max pieces, min empties.
- Enforce presence of A (at least one H2 that becomes 'A'); optional fixed row.
- BFS optimal solver with per-size node cap.
- Solvable-only dataset by default; optionally exclude already-solved starts.
- Difficulty labeling:
    * fixed thresholds (easy<3, medium<5, hard>=5), or
    * quantiles (legacy).
- Optional balancing: per-size 1/3 easy, 1/3 medium, 1/3 hard.
- Stratified split by (size, difficulty).
- Detailed timing & progress bars.
- Optional push to Hugging Face Hub.

Copyright (c) 2024 The Illusion-of-Reasoning contributors.
Licensed under the Apache License, Version 2.0. See LICENSE for details.
"""

from __future__ import annotations
import argparse
import os
import json
import time
import random
import multiprocessing as mp
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import partial
from typing import List, Tuple, Dict, Optional, Set

from tqdm import tqdm
from datasets import Dataset, DatasetDict, Features, Value, Sequence
from huggingface_hub import HfApi, HfFolder

DEFAULT_MAX_PIECES = {4: 7, 5: 9, 6: 11}
DEFAULT_MIN_EMPTIES = {4: 2, 5: 3, 6: 4}

# -------------------------
# Core helpers (size-agnostic)
# -------------------------

def rc_idx(row_idx: int, col_idx: int, size: int) -> int:
    """Convert row/col to linear index."""
    return row_idx * size + col_idx

def idx_rc(index: int, size: int) -> Tuple[int,int]:
    """Convert linear index to (row, col)."""
    return divmod(index, size)

@dataclass
class Piece:
    """
    Representation of a Rush Hour piece on the grid.

    :ivar pid: Unique piece identifier.
    :vartype pid: int
    :ivar cells: Linear indices occupied by the piece.
    :vartype cells: List[int]
    :ivar axis: Orientation of the piece (``"H"`` or ``"V"``).
    :vartype axis: str
    :ivar length: Number of cells the piece spans.
    :vartype length: int
    """
    pid: int
    cells: List[int]
    axis: str  # 'H' or 'V'
    length: int

@dataclass
class BoardPlacement:
    """Mutable grid plus size for piece placement helpers."""
    grid_pid: List[Optional[int]]
    size: int

@dataclass
class GenerationParams:
    """Parameters controlling random board generation."""
    max_pieces: int
    min_empties: int

def _place_h(state: BoardPlacement, row_idx: int, col_start: int, length: int, pid: int):
    """Place a horizontal piece if space is available."""
    grid_pid = state.grid_pid
    size = state.size
    if col_start + length > size:
        return None
    cells = [rc_idx(row_idx, col_start + d, size) for d in range(length)]
    if any(grid_pid[x] is not None for x in cells):
        return None
    for x in cells:
        grid_pid[x] = pid
    return cells

def _place_v(state: BoardPlacement, row_start: int, col_idx: int, length: int, pid: int):
    """Place a vertical piece if space is available."""
    grid_pid = state.grid_pid
    size = state.size
    if row_start + length > size:
        return None
    cells = [rc_idx(row_start + d, col_idx, size) for d in range(length)]
    if any(grid_pid[x] is not None for x in cells):
        return None
    for x in cells:
        grid_pid[x] = pid
    return cells

def _clear(grid_pid, cells):
    """
    Clear piece ids from the provided cells.

    :param grid_pid: Mutable grid of piece ids.
    :type grid_pid: List[Optional[int]]
    :param cells: Cell indices to clear.
    :type cells: Iterable[int]
    """
    for x in cells:
        grid_pid[x] = None

def _canonicalize(grid_pid, pieces: Dict[int, 'Piece'], size: int) -> Optional[str]:
    """Assign 'A' to earliest H2 piece; map other pieces to B,C,... by first appearance."""
    earliest_pid, earliest_idx = None, 10**9
    for p in pieces.values():
        if p.axis == 'H' and p.length == 2:
            first = min(p.cells)
            if first < earliest_idx:
                earliest_idx = first
                earliest_pid = p.pid
    if earliest_pid is None:
        return None  # enforce A presence

    mapping, nxt = {}, ord('B')
    seen = set()
    for pid in grid_pid:
        if pid is None: continue
        if pid in seen: continue
        seen.add(pid)
        mapping[pid] = 'A' if pid == earliest_pid else chr(nxt)
        if pid != earliest_pid:
            nxt += 1

    out = []
    for pid in grid_pid:
        out.append('o' if pid is None else mapping[pid])
    return "".join(out)

# -------------------------
# Enumeration (exact) for small sizes
# -------------------------

def enumerate_boards_for_size(
    size: int,
    max_pieces: int,
    min_empties: int,
    seed: int,
    limit: int,
) -> List[str]:
    """Enumerate canonical boards for small sizes via backtracking."""
    total = size * size
    grid_pid = [None] * total
    state = BoardPlacement(grid_pid, size)
    pieces: Dict[int, Piece] = {}
    results: Set[str] = set()
    pid0 = 1

    def backtrack(start_i: int, next_pid: int, placed: int):
        if placed > max_pieces:
            return
        i = start_i
        while i < total and grid_pid[i] is not None:
            i += 1
        if i >= total:
            if grid_pid.count(None) < min_empties:
                return
            board_str = _canonicalize(grid_pid, pieces, size)
            if board_str is not None:
                results.add(board_str)
            return

        # leave empty
        backtrack(i + 1, next_pid, placed)

        row_idx, col_idx = idx_rc(i, size)

        # H2
        cells = _place_h(state, row_idx, col_idx, 2, next_pid)
        if cells:
            pieces[next_pid] = Piece(next_pid, cells, 'H', 2)
            backtrack(i + 1, next_pid + 1, placed + 1)
            _clear(grid_pid, cells)
            del pieces[next_pid]

        # H3
        cells = _place_h(state, row_idx, col_idx, 3, next_pid)
        if cells:
            pieces[next_pid] = Piece(next_pid, cells, 'H', 3)
            backtrack(i + 1, next_pid + 1, placed + 1)
            _clear(grid_pid, cells)
            del pieces[next_pid]

        # V2
        cells = _place_v(state, row_idx, col_idx, 2, next_pid)
        if cells:
            pieces[next_pid] = Piece(next_pid, cells, 'V', 2)
            backtrack(i + 1, next_pid + 1, placed + 1)
            _clear(grid_pid, cells)
            del pieces[next_pid]

        # V3
        cells = _place_v(state, row_idx, col_idx, 3, next_pid)
        if cells:
            pieces[next_pid] = Piece(next_pid, cells, 'V', 3)
            backtrack(i + 1, next_pid + 1, placed + 1)
            _clear(grid_pid, cells)
            del pieces[next_pid]

    backtrack(0, pid0, 0)
    boards = list(results)
    rng = random.Random(seed)
    rng.shuffle(boards)
    if limit > 0:
        boards = boards[:limit]
    return boards

# -------------------------
# Sampling (build-and-add) for big sizes
# -------------------------

def sample_boards_for_size(size: int,
                           seed: int,
                           params: GenerationParams,
                           batch: int = 5000,
                           a_row: Optional[int] = None) -> List[str]:
    """
    Randomly generate up to `batch` candidate canonical boards for a given size.
    Enforces: at least one H2 that becomes 'A', optional A-row pin, + min_empties.
    """
    rng = random.Random(seed ^ (size * 7919 + batch))
    total = size * size
    out: Set[str] = set()

    for _ in range(batch * 4):  # multiple attempts for robustness
        grid_pid = [None] * total
        state = BoardPlacement(grid_pid, size)
        pieces: Dict[int, Piece] = {}
        pid = 1

        # 1) place the A-piece (H2) first
        if a_row is not None:
            row_idx = a_row % size
            col_start = rng.randrange(0, size - 1)
            cells = _place_h(state, row_idx, col_start, 2, pid)
            if not cells:
                continue
            pieces[pid] = Piece(pid, cells, 'H', 2)
            pid += 1
        else:
            placed_a = False
            for _ in range(8):
                row_idx = rng.randrange(size)
                col_start = rng.randrange(0, size - 1)
                cells = _place_h(state, row_idx, col_start, 2, pid)
                if cells:
                    pieces[pid] = Piece(pid, cells, 'H', 2)
                    pid += 1
                    placed_a = True
                    break
            if not placed_a:
                continue

        # 2) place random other pieces up to max_pieces
        kinds = [('H', 2), ('H', 3), ('V', 2), ('V', 3)]
        for _ in range(params.max_pieces - 1):  # -1 because A is placed
            if rng.random() < 0.25:  # early stop to keep variety/sparsity
                break
            axis, length = kinds[rng.randrange(len(kinds))]
            placed = False
            for _ in range(12):
                row_idx = rng.randrange(size)
                col_idx = rng.randrange(size)
                if axis == 'H':
                    if col_idx > size - length:
                        col_idx = size - length
                    cells = _place_h(state, row_idx, col_idx, length, pid)
                else:
                    if row_idx > size - length:
                        row_idx = size - length
                    cells = _place_v(state, row_idx, col_idx, length, pid)
                if cells:
                    pieces[pid] = Piece(pid, cells, axis, length)
                    pid += 1
                    placed = True
                    break
            if not placed:
                continue

        # 3) enforce min empties
        empties = sum(1 for cell in grid_pid if cell is None)
        if empties < params.min_empties:
            continue

        # ensure A exists
        if not any(piece.axis == 'H' and piece.length == 2 for piece in pieces.values()):
            continue

        board = _canonicalize(grid_pid, pieces, size)
        if board is None:
            continue
        # drop already-solved
        if _is_goal(board, size):
            continue

        a_idxs = [i for i, ch in enumerate(board) if ch == 'A']
        if not a_idxs:
            continue
        a_r = a_idxs[0] // size
        if all(board[rc_idx(a_r, cc, size)] != 'o' for cc in range(size)):
            continue

        out.add(board)
        if len(out) >= batch:
            break

    return list(out)

# -------------------------
# Solver (BFS)
# -------------------------

def _to_grid(board: str, size: int):
    """Convert board string to 2D grid."""
    board_iter = iter(board)
    return [[next(board_iter) for _ in range(size)] for _ in range(size)]

def _grid_to_board(grid):
    """
    Flatten a 2D grid of characters back into a board string.

    :param grid: 2D list of board characters.
    :type grid: List[List[str]]
    :returns: Board encoded as a single string.
    :rtype: str
    """
    return "".join("".join(row) for row in grid)

def _parse_pieces(board: str, size: int) -> Dict[str, dict]:
    """Return piece metadata (axis, length, cells) keyed by label."""
    cells_by_piece = defaultdict(list)
    for idx, char in enumerate(board):
        if char == 'o':
            continue
        row_idx, col_idx = idx_rc(idx, size)
        cells_by_piece[char].append((row_idx, col_idx))

    pieces = {}
    for label, positions in cells_by_piece.items():
        row_set = {row for row, _ in positions}
        col_set = {col for _, col in positions}
        if len(row_set) == 1:
            axis = 'H'
            length = max(col for _, col in positions) - min(col for _, col in positions) + 1
        elif len(col_set) == 1:
            axis = 'V'
            length = max(row for row, _ in positions) - min(row for row, _ in positions) + 1
        else:
            continue
        pieces[label] = {"axis": axis, "len": length, "cells": sorted(positions)}
    return pieces

def _is_goal(board: str, size: int) -> bool:
    """Return True if car A already touches the exit on the right edge."""
    indices_a = [i for i, char in enumerate(board) if char == 'A']
    return bool(indices_a) and max(i % size for i in indices_a) == size - 1

def _horizontal_moves(piece_label: str, cells: List[Tuple[int, int]], grid_state, length: int, size: int):
    """
    Yield horizontal moves for a piece.

    :param piece_label: Piece identifier.
    :type piece_label: str
    :param cells: Current cell coordinates for the piece.
    :type cells: List[Tuple[int, int]]
    :param grid_state: Current board grid.
    :type grid_state: List[List[str]]
    :param length: Piece length.
    :type length: int
    :param size: Board size.
    :type size: int
    :yields: Tuples of (board string, move string).
    """
    row_idx = cells[0][0]
    min_col = min(col for _, col in cells)
    max_col = max(col for _, col in cells)
    step = 1
    while min_col - step >= 0 and all(
        grid_state[row_idx][col] == 'o'
        for col in range(min_col - step, min_col)
    ):
        next_grid = [row[:] for row in grid_state]
        for _, col in cells:
            next_grid[row_idx][col] = 'o'
        for col in range(min_col - step, min_col - step + length):
            next_grid[row_idx][col] = piece_label
        yield _grid_to_board(next_grid), f"{piece_label}<{step}"
        step += 1
    step = 1
    while max_col + step < size and all(
        grid_state[row_idx][col] == 'o'
        for col in range(max_col + 1, max_col + step + 1)
    ):
        next_grid = [row[:] for row in grid_state]
        for _, col in cells:
            next_grid[row_idx][col] = 'o'
        start_col = max_col + step - length + 1
        for col in range(start_col, start_col + length):
            next_grid[row_idx][col] = piece_label
        yield _grid_to_board(next_grid), f"{piece_label}>{step}"
        step += 1


def _vertical_moves(piece_label: str, cells: List[Tuple[int, int]], grid_state, length: int, size: int):
    """
    Yield vertical moves for a piece.

    :param piece_label: Piece identifier.
    :type piece_label: str
    :param cells: Current cell coordinates for the piece.
    :type cells: List[Tuple[int, int]]
    :param grid_state: Current board grid.
    :type grid_state: List[List[str]]
    :param length: Piece length.
    :type length: int
    :param size: Board size.
    :type size: int
    :yields: Tuples of (board string, move string).
    """
    col_idx = cells[0][1]
    min_row = min(row for row, _ in cells)
    max_row = max(row for row, _ in cells)
    step = 1
    while min_row - step >= 0 and all(
        grid_state[row_idx][col_idx] == 'o'
        for row_idx in range(min_row - step, min_row)
    ):
        next_grid = [row[:] for row in grid_state]
        for row_idx, _ in cells:
            next_grid[row_idx][col_idx] = 'o'
        for row_idx in range(min_row - step, min_row - step + length):
            next_grid[row_idx][col_idx] = piece_label
        yield _grid_to_board(next_grid), f"{piece_label}^{step}"
        step += 1
    step = 1
    while max_row + step < size and all(
        grid_state[row_idx][col_idx] == 'o'
        for row_idx in range(max_row + 1, max_row + step + 1)
    ):
        next_grid = [row[:] for row in grid_state]
        for row_idx, _ in cells:
            next_grid[row_idx][col_idx] = 'o'
        start_row = max_row + step - length + 1
        for row_idx in range(start_row, start_row + length):
            next_grid[row_idx][col_idx] = piece_label
        yield _grid_to_board(next_grid), f"{piece_label}v{step}"
        step += 1


def _neighbors(board: str, size: int):
    """Yield reachable boards with move strings from a given board."""
    grid_state = _to_grid(board, size)
    pieces = _parse_pieces(board, size)
    for piece_label, info in pieces.items():
        axis = info["axis"]
        length = info["len"]
        cells = info["cells"]
        if axis == 'H':
            yield from _horizontal_moves(piece_label, cells, grid_state, length, size)
        else:
            yield from _vertical_moves(piece_label, cells, grid_state, length, size)

def _canon_labels(board: str) -> str:
    """Relabel pieces to canonical letters (A fixed, others B,C,...) for hashing."""
    mapping = {}
    next_code = ord('B')
    canonical_chars = []
    for char in board:
        if char in ('o', 'A'):
            canonical_chars.append(char)
            continue
        if char not in mapping:
            mapping[char] = chr(next_code)
            next_code += 1
        canonical_chars.append(mapping[char])
    return "".join(canonical_chars)

def bfs_solve(board: str, size: int, max_nodes: int):
    """BFS to find an optimal solution; returns solvable flag, move count, path."""
    if _is_goal(board, size):
        return True, 0, []
    start = _canon_labels(board)
    queue = deque([start])
    seen = {start}
    parent = {}
    nodes = 0
    while queue:
        state = queue.popleft()
        nodes += 1
        if nodes > max_nodes:
            break
        for next_board, move in _neighbors(state, size):
            canonical = _canon_labels(next_board)
            if canonical in seen:
                continue
            seen.add(canonical)
            parent[canonical] = (state, move)
            if _is_goal(canonical, size):
                path = _reconstruct_path(start, canonical, parent)
                return True, len(path), path
            queue.append(canonical)
    return False, None, None


def _reconstruct_path(start: str, goal: str, parent: Dict[str, Tuple[str, str]]) -> List[str]:
    """Rebuild the move sequence from BFS parent pointers."""
    path: List[str] = []
    current = goal
    while current != start:
        prev_state, last_move = parent[current]
        path.append(last_move)
        current = prev_state
    path.reverse()
    return path

def _solve_one(board: str, size: int, max_nodes: int):
    """Solve a board with BFS and return enriched tuple."""
    solvable, moves, seq = bfs_solve(board, size=size, max_nodes=max_nodes)
    return board, solvable, (moves if moves is not None else -1), (seq if seq else [])

# -------------------------
# Dataset builders
# -------------------------

def build_messages(size: int, board: str, moves: int, solution: List[str]) -> List[Dict[str,str]]:
    """Return chat-style messages describing the puzzle and optimal path."""
    sys_p = (
        f"You are an expert Rush Hour ({size}×{size}) solver.\n\n"
        f"Input: {size*size}-char row-major board; 'o' empty, 'A' red 2-long horizontal, "
        "others B..Z (len 2/3; H/V).\n"
        "Output exactly ONE optimal solution wrapped in tags:\n"
        "<think>\n(brief internal reasoning)\n</think>\n<answer>\ncomma,separated,move,list\n</answer>\n"
        "Token format: <PIECE><DIR><STEPS> (A>2,B<1,Cv3). One move = slide any positive distance along axis.\n"
        "Goal: right end of 'A' reaches the right edge.\n"
    )
    user_p = (
        f"Board size: {size}x{size}\n"
        f"Board: {board}\n"
        f"Optimal length (moves): {moves}\n"
        "Return only the <think> and <answer> blocks."
    )
    assistant = "<think>\nShortest path found.\n</think>\n<answer>\n" + ",".join(solution) + "\n</answer>"
    return [
        {"role":"system","content":sys_p},
        {"role":"user","content":user_p},
        {"role":"assistant","content":assistant},
    ]

def to_dataset(rows: List[dict]) -> Dataset:
    """
    Convert rows to a Hugging Face Dataset with the expected schema.

    :param rows: List of row dictionaries with board metadata.
    :type rows: List[dict]
    :returns: Hugging Face Dataset constructed from the rows.
    :rtype: Dataset
    """
    feats = Features({
        "board": Value("string"),
        "size": Value("int32"),
        "moves": Value("int32"),
        "solution": Sequence(Value("string")),
        "difficulty": Value("string"),
        "messages": Sequence({"role": Value("string"), "content": Value("string")}),
    })
    return Dataset.from_dict({
        "board": [r["board"] for r in rows],
        "size":  [r["size"]  for r in rows],
        "moves": [r["moves"] for r in rows],
        "solution": [r["solution"] for r in rows],
        "difficulty": [r["difficulty"] for r in rows],
        "messages": [r["messages"] for r in rows],
    }, features=feats)

def _split_counts(bucket_size: int, train_frac: float, val_frac: float, test_frac: float):
    """Compute bucket counts with rounding guards."""
    n_tr = int(round(train_frac * bucket_size))
    n_va = int(round(val_frac * bucket_size))
    n_te = int(round(test_frac * bucket_size))
    total = n_tr + n_va + n_te
    if total != bucket_size:
        n_te = max(0, bucket_size - n_tr - n_va)
    if bucket_size >= 3 and (n_va == 0 or n_te == 0):
        n_tr = max(1, int(bucket_size * train_frac))
        n_va = max(1, int(bucket_size * val_frac))
        n_te = max(1, bucket_size - n_tr - n_va)

    total = n_tr + n_va + n_te
    if total > bucket_size:
        n_te = max(0, n_te - (total - bucket_size))
    elif total < bucket_size:
        n_te += bucket_size - total
    return n_tr, n_va, n_te


def stratified_split(rows: List[dict], split=(0.8, 0.1, 0.1), seed: int = 42):
    """
    Stratify by (size, difficulty) so each bucket is split into train/val/test
    with proportions in `split`. Deterministic via `seed`.

    :param rows: Full list of labeled rows.
    :type rows: List[dict]
    :param split: Fractions for train, validation, and test.
    :type split: Tuple[float, float, float]
    :param seed: RNG seed for deterministic shuffles.
    :type seed: int
    :returns: Train, validation, and test row lists.
    :rtype: Tuple[List[dict], List[dict], List[dict]]
    """
    rng = random.Random(seed)
    by_key: Dict[Tuple[int, str], List[dict]] = defaultdict(list)
    for row in rows:
        # expect r to have keys: "size" (int) and "difficulty" in {"easy","medium","hard"}
        by_key[(int(row["size"]), str(row["difficulty"]))].append(row)

    train_frac, val_frac, test_frac = split
    train, val, test = [], [], []

    for bucket in by_key.values():
        if not bucket:
            continue
        rng.shuffle(bucket)
        n_tr, n_va, n_te = _split_counts(len(bucket), train_frac, val_frac, test_frac)

        train += bucket[:n_tr]
        val   += bucket[n_tr:n_tr + n_va]
        test  += bucket[n_tr + n_va:n_tr + n_va + n_te]

    return train, val, test


def quantile_buckets_per_size(rows: List[dict], quantile_1: float = 0.33, quantile_2: float = 0.66):
    """Return per-size move thresholds for easy/medium buckets based on quantiles."""
    by_size = defaultdict(list)
    for row in rows:
        by_size[row["size"]].append(row["moves"])
    thresholds = {}
    for size, moves_by_size in by_size.items():
        moves_sorted = sorted(moves_by_size)
        if not moves_sorted:
            thresholds[size] = (0, 0)
            continue

        def pick(prob, sorted_moves=moves_sorted):
            idx = max(0, min(len(sorted_moves) - 1, int(prob * (len(sorted_moves) - 1))))
            return sorted_moves[idx]

        thresholds[size] = (pick(quantile_1), pick(quantile_2))
    return thresholds


def label_difficulty_quantile(size: int, moves: int, thresholds: Dict[int, Tuple[int,int]]) -> str:
    """Label a puzzle using per-size quantile thresholds."""
    threshold_easy, threshold_medium = thresholds.get(size, (moves, moves))
    if moves <= threshold_easy:
        return "easy"
    if moves <= threshold_medium:
        return "medium"
    return "hard"

def label_difficulty_fixed(moves: int, easy_lt: int = 4, med_lt: int = 6) -> str:
    """
    Fixed thresholds:
      easy   if moves < easy_lt      (default: 2–3)
      medium if moves < med_lt       (default: 4–5)
      hard   otherwise               (default: ≥6)
    NOTE: 1-move puzzles are filtered out upstream.
    """
    if moves < easy_lt:
        return "easy"
    if moves < med_lt:
        return "medium"
    return "hard"


def dedupe_boards(candidates: List[str], seen: Set[str]) -> Tuple[List[str], int]:
    """Drop already-seen boards and return deduped list plus removed count."""
    deduped = [board for board in candidates if board not in seen]
    for board in deduped:
        seen.add(board)
    return deduped, len(candidates) - len(deduped)


def filter_already_solved(boards: List[str], size: int) -> Tuple[List[str], int]:
    """Remove boards where A already reaches exit; returns filtered list and removed count."""
    def start_is_goal(board_str, board_size=size):
        indices_a = [i for i, ch in enumerate(board_str) if ch == 'A']
        return bool(indices_a) and max(i % board_size for i in indices_a) == board_size - 1
    filtered = [board for board in boards if not start_is_goal(board)]
    return filtered, len(boards) - len(filtered)


def solve_boards(boards: List[str], size: int, nodes_cap: int, num_workers: int):
    """Solve boards, returning results and elapsed seconds."""
    worker = partial(_solve_one, size=size, max_nodes=nodes_cap)
    solve_start = time.time()
    if num_workers > 1:
        with mp.Pool(processes=num_workers) as pool:
            results = list(
                tqdm(pool.imap(worker, boards, chunksize=256),
                     total=len(boards), desc=f"{size}x{size} solve", unit="b")
            )
    else:
        results = list(map(worker, boards))
    return results, time.time() - solve_start


def append_sampled_rows(
    results,
    size: int,
    size_rows: List[dict],
    target_rows: int,
    solvable_only: bool,
) -> int:
    """Append solved rows, respecting solvable-only and target cap. Return kept count."""
    kept = 0
    for board, solvable, moves, seq in results:
        if solvable_only and not solvable:
            continue
        move_count = moves if moves is not None else -1
        if move_count == 1:
            continue  # drop length-1 puzzles
        size_rows.append({"board": board, "size": size, "moves": move_count, "solution": seq or []})
        kept += 1
        if len(size_rows) >= target_rows:
            break
    return kept

def balance_by_size_and_difficulty(rows: List[dict], sizes: List[int], seed: int = 42) -> List[dict]:
    """
    Downsample so that for each size, the counts of {easy, medium, hard} are equal
    (the minimum among the three). Deterministic via `seed`.
    """
    rng = random.Random(seed)
    buckets: Dict[Tuple[int, str], List[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["size"], row["difficulty"])].append(row)

    balanced: List[dict] = []
    for size in sizes:
        easy_rows = buckets.get((size, "easy"),   [])
        medium_rows = buckets.get((size, "medium"), [])
        hard_rows = buckets.get((size, "hard"),   [])
        if not (easy_rows and medium_rows and hard_rows):
            # If any bucket is empty, skip balancing for this size.
            balanced.extend(easy_rows + medium_rows + hard_rows)
            continue
        keep = min(len(easy_rows), len(medium_rows), len(hard_rows))
        rng.shuffle(easy_rows)
        rng.shuffle(medium_rows)
        rng.shuffle(hard_rows)
        balanced.extend(easy_rows[:keep] + medium_rows[:keep] + hard_rows[:keep])
    return balanced

# -------------------------
# Utilities
# -------------------------

def parse_int_map(mapping_spec: str) -> Dict[int, int]:
    """Parse maps like '4:7,5:9,6:11' into {size: value} dicts."""
    if not mapping_spec:
        return {}
    parsed = {}
    for part in mapping_spec.split(","):
        size_str, value_str = part.split(":")
        parsed[int(size_str.strip())] = int(value_str.strip())
    return parsed


@dataclass
class Thresholds:
    """Fixed and quantile thresholds for difficulty labeling."""
    easy_lt: int
    med_lt: int
    quantile_1: float
    quantile_2: float


@dataclass
class RunConfig:
    """Pre-parsed, validated configuration derived from CLI args."""
    per_size_limits: Dict[str, Dict[int, int]]
    sample_sizes: Set[int]
    target_map: Dict[int, int]
    a_row_map: Dict[int, int]
    thresholds: Thresholds


@dataclass
class SamplingState:
    """
    Rolling state during sampling loops.

    :ivar seen: Set of boards already processed.
    :vartype seen: Set[str]
    :ivar rows: Accumulated solvable rows.
    :vartype rows: List[dict]
    :ivar batch_size: Current generation batch size.
    :vartype batch_size: int
    :ivar rounds: Number of generation rounds completed.
    :vartype rounds: int
    :ivar start_time: Wall-clock start time of sampling loop.
    :vartype start_time: float
    """
    seen: Set[str]
    rows: List[dict]
    batch_size: int = 20000
    rounds: int = 0
    start_time: float = 0.0


def build_config(args) -> RunConfig:
    """
    Translate CLI args into typed, validated configuration values.

    :param args: Parsed CLI arguments.
    :type args: argparse.Namespace
    :returns: Populated runtime configuration object.
    :rtype: RunConfig
    """
    try:
        threshold_parts = args.fixed_thresholds.split(",") if args.fixed_thresholds else ["3", "5"]
        easy_lt = int(threshold_parts[0])
        med_lt = int(threshold_parts[1])
    except (ValueError, TypeError, IndexError):
        easy_lt, med_lt = 3, 5

    try:
        quantile_1, quantile_2 = map(float, args.difficulty_quantiles.split(","))
    except (ValueError, TypeError):
        quantile_1, quantile_2 = 0.33, 0.66

    max_pieces_map = parse_int_map(args.max_pieces_per_size)
    min_empties_map = parse_int_map(args.min_empties_per_size)
    try:
        max_nodes_map = parse_int_map(args.max_nodes_per_size)
    except ValueError:
        max_nodes_map = {}

    sample_sizes = (
        {int(size_str) for size_str in args.sample_sizes.split(",") if size_str.strip()}
        if args.sample_sizes
        else set()
    )
    target_map = parse_int_map(args.target_per_size)
    a_row_map = parse_int_map(args.a_row_per_size or "")

    thresholds = Thresholds(
        easy_lt=easy_lt,
        med_lt=med_lt,
        quantile_1=quantile_1,
        quantile_2=quantile_2,
    )

    per_size_limits = {
        "max_pieces": max_pieces_map,
        "min_empties": min_empties_map,
        "max_nodes": max_nodes_map,
    }

    return RunConfig(
        per_size_limits=per_size_limits,
        sample_sizes=sample_sizes,
        target_map=target_map,
        a_row_map=a_row_map,
        thresholds=thresholds,
    )


def sample_rows_for_size(size: int, args, config: RunConfig, a_row: Optional[int]) -> List[dict]:
    """
    Generate solvable rows via sampling for a specific board size.

    :param size: Board dimension.
    :type size: int
    :param args: Parsed CLI arguments.
    :type args: argparse.Namespace
    :param config: Precomputed run configuration.
    :type config: RunConfig
    :param a_row: Optional fixed row index for the A car.
    :type a_row: Optional[int]
    :returns: List of sampled, solved rows.
    :rtype: List[dict]
    """
    max_p = config.per_size_limits["max_pieces"].get(size, DEFAULT_MAX_PIECES.get(size, 9))
    min_e = config.per_size_limits["min_empties"].get(size, DEFAULT_MIN_EMPTIES.get(size, 3))
    nodes_cap = config.per_size_limits["max_nodes"].get(size, args.max_nodes)
    target_rows = config.target_map.get(size, args.limit_per_size if args.limit_per_size > 0 else 50000)
    gen_params = GenerationParams(max_pieces=max_p, min_empties=min_e)
    print(f"[sample] {size}x{size}: target solvable rows = {target_rows} (a_row={a_row})")

    state = SamplingState(seen=set(), rows=[], start_time=time.time())
    progress = tqdm(total=target_rows, desc=f"{size}x{size} sample-keep", unit="row")
    while len(state.rows) < target_rows:
        state.rounds += 1
        gen_start = time.time()
        cand = sample_boards_for_size(
            size=size,
            seed=args.seed + state.rounds,
            params=gen_params,
            batch=state.batch_size,
            a_row=a_row,
        )
        before_dedup = len(cand)
        deduped, removed = dedupe_boards(cand, state.seen)
        filtered = deduped
        if args.exclude_already_solved:
            filtered, removed_solved = filter_already_solved(deduped, size)
            removed += removed_solved

        results, solve_secs = solve_boards(filtered, size, nodes_cap, args.num_workers)
        kept = append_sampled_rows(results, size, state.rows, target_rows, args.solvable_only)
        progress.update(kept)
        print(
            f"[sample] {size}x{size}: generated {before_dedup}, kept {kept}, removed {removed}, seen={len(state.seen)} "
            f"(gen {time.time()-gen_start:.2f}s, solve {solve_secs:.2f}s)"
        )

        keep_rate = kept / max(1, len(cand))
        if keep_rate < 0.05 and state.batch_size < 80000:
            state.batch_size *= 2
        elif keep_rate > 0.20 and state.batch_size > 10000:
            state.batch_size = max(10000, state.batch_size // 2)

    progress.close()
    mins = (time.time() - state.start_time) / 60.0
    print(f"[done] {size}x{size}: built {len(state.rows)} solvable rows in {mins:.2f} min via sampling.")
    return state.rows


def enumerate_rows_for_size(size: int, args, config: RunConfig) -> List[dict]:
    """
    Enumerate canonical boards then solve them for a specific size.

    :param size: Board dimension.
    :type size: int
    :param args: Parsed CLI arguments.
    :type args: argparse.Namespace
    :param config: Precomputed run configuration.
    :type config: RunConfig
    :returns: List of solved rows from enumeration.
    :rtype: List[dict]
    """
    max_p = config.per_size_limits["max_pieces"].get(size, DEFAULT_MAX_PIECES.get(size, 9))
    min_e = config.per_size_limits["min_empties"].get(size, DEFAULT_MIN_EMPTIES.get(size, 3))
    nodes_cap = config.per_size_limits["max_nodes"].get(size, args.max_nodes)
    start_time = time.time()
    boards = enumerate_boards_for_size(
        size=size,
        max_pieces=max_p,
        min_empties=min_e,
        seed=args.seed,
        limit=args.limit_per_size,
    )
    enum_duration = time.time() - start_time
    rate = (len(boards) / enum_duration) if enum_duration > 0 else 0.0
    print(f"[enum] {size}x{size}: {len(boards)} canonical boards. ({enum_duration:.2f}s, {rate:.0f}/s)")

    if args.exclude_already_solved:
        filter_start = time.time()
        boards, removed = filter_already_solved(boards, size)
        print(f"[filter] {size}x{size}: removed {removed} already-solved starts. ({time.time()-filter_start:.2f}s)")

    print(f"[solve] {size}x{size}: solving with {args.num_workers} workers, max_nodes={nodes_cap} …")
    results, solve_secs = solve_boards(boards, size, nodes_cap, args.num_workers)
    print(f"[done] {size}x{size}: solved {len(boards)} boards in {solve_secs/60.0:.2f} min")

    size_rows = []
    for board, solvable, moves, seq in results:
        if args.solvable_only and not solvable:
            continue
        size_rows.append({
            "board": board,
            "size": size,
            "moves": moves if moves is not None else -1,
            "solution": seq or [],
        })

    print(f"[keep] {size}x{size}: {len(size_rows)} rows after solvable-only filter.")
    return size_rows

# -------------------------
# Main
# -------------------------

def parse_args():
    """Build and parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Rush Hour 4x4/5x5/6x6 dataset (solvable-only, fixed thresholds, optional balancing).")
    parser.add_argument("--sizes", nargs="+", type=int, default=[4,5,6], help="Board sizes to include (4, 5, 6).")
    parser.add_argument("--out_dir", type=str, default="./rush-balanced", help="Output directory.")
    parser.add_argument("--split", type=str, default="0.8,0.1,0.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_per_size", type=int, default=0, help="Max canonical boards per size for enumeration (0=all).")

    # BFS caps
    parser.add_argument("--max_nodes", type=int, default=300000, help="Default BFS node cap per puzzle.")
    parser.add_argument("--max_nodes_per_size", type=str, default="4:300000,5:600000,6:1200000",
                    help="Per-size BFS caps, e.g. '4:300000,5:600000,6:1200000'. Overrides --max_nodes for listed sizes.")

    # Workers
    parser.add_argument("--num_workers", type=int, default=max(1, mp.cpu_count()//2))

    # Hub
    parser.add_argument("--dataset_id", type=str, default=None)
    parser.add_argument("--push_to_hub", action="store_true")

    # Enumeration constraints
    parser.add_argument("--max_pieces_per_size", type=str, default="4:7,5:9,6:11", help="e.g., '4:7,5:9,6:11'")
    parser.add_argument(
        "--min_empties_per_size",
        type=str,
        default="4:2,5:3,6:4",
        help="e.g., '4:2,5:3,6:4'",
    )

    # Policy toggles
    parser.add_argument(
        "--solvable_only", action="store_true", default=True,
        help="Keep only solvable boards.",
    )
    parser.add_argument(
        "--exclude_already_solved", action="store_true", default=True,
        help="Drop boards where A is already at the exit.",
    )
    parser.add_argument(
        "--difficulty_quantiles", type=str, default="0.33,0.66",
        help="For quantiles: q1,q2 per size.",
    )
    parser.add_argument(
        "--balance_per_size", action="store_true",
        help="Downsample so each size has equal counts across {easy,medium,hard}.",
    )

    # Sampling controls
    parser.add_argument(
        "--sample_sizes", type=str, default="",
        help='Sizes to *sample* rather than enumerate, e.g. "6" or "5,6".',
    )
    parser.add_argument(
        "--target_per_size",
        type=str,
        default="",
        help="Per-size *final solvable rows* target (for sample sizes), e.g. '6:50000'.",
    )
    parser.add_argument(
        "--a_row_per_size",
        type=str,
        default=None,
        help="Optional fixed row for A by size (0-based), e.g. '6:2' to pin A on row 2 for 6×6.",
    )

    parser.add_argument(
        "--difficulty_scheme",
        type=str,
        default="fixed",
        choices=["fixed", "quantile"],
        help=(
            "Use fixed thresholds (easy<4, medium<6, hard>=6; i.e., 2–3 easy, 4–5 medium) "
            "or quantiles."
        ),
    )

    parser.add_argument(
        "--fixed_thresholds",
        type=str,
        default="4,6",
        help=(
            "For --difficulty_scheme fixed: 'easy_lt,medium_lt'. "
            "Default 4,6 → 2–3=easy, 4–5=medium, ≥6=hard."
        ),
    )

    return parser.parse_args()


def rows_for_size(size: int, args, config: RunConfig) -> List[dict]:
    """Build solvable rows for one board size, sampling or enumerating."""
    if size not in (4, 5, 6):
        print(f"[warn] size {size} not supported; skipping.")
        return []

    a_row = config.a_row_map.get(size, None)
    max_p = config.per_size_limits["max_pieces"].get(size, DEFAULT_MAX_PIECES.get(size, 9))
    min_e = config.per_size_limits["min_empties"].get(size, DEFAULT_MIN_EMPTIES.get(size, 3))
    print(f"[enum] {size}x{size}: max_pieces={max_p}, min_empties={min_e}")

    if size in config.sample_sizes:
        return sample_rows_for_size(size, args, config, a_row)
    return enumerate_rows_for_size(size, args, config)


def label_rows(all_rows: List[dict], args, config: RunConfig) -> List[dict]:
    """Attach difficulty labels and chat messages to rows."""
    rows_labeled = []
    if args.difficulty_scheme == "quantile":
        print("[diff] computing quantile thresholds per size …")
        thresholds = quantile_buckets_per_size(
            all_rows,
            quantile_1=config.thresholds.quantile_1,
            quantile_2=config.thresholds.quantile_2,
        )
        for size_key, (threshold_easy, threshold_medium) in thresholds.items():
            print(
                f"  size {size_key}x{size_key}: easy ≤ {threshold_easy}, "
                f"medium ≤ {threshold_medium}, hard > {threshold_medium}"
            )
        for row in all_rows:
            diff = label_difficulty_quantile(row["size"], row["moves"], thresholds)
            rows_labeled.append({
                **row,
                "difficulty": diff,
                "messages": build_messages(row["size"], row["board"], row["moves"], row["solution"]),
            })
        return rows_labeled

    print(
        f"[diff] using fixed thresholds: easy < {config.thresholds.easy_lt}, "
        f"medium < {config.thresholds.med_lt}, hard ≥ {config.thresholds.med_lt} (dropping 1-move puzzles)"
    )
    for row in all_rows:
        diff = label_difficulty_fixed(row["moves"], config.thresholds.easy_lt, config.thresholds.med_lt)
        rows_labeled.append({
            **row,
            "difficulty": diff,
            "messages": build_messages(row["size"], row["board"], row["moves"], row["solution"]),
        })
    return rows_labeled


def maybe_balance(rows_labeled: List[dict], args) -> List[dict]:
    """Optionally downsample to even out easy/medium/hard per size."""
    if not args.balance_per_size:
        return rows_labeled
    before = len(rows_labeled)
    rows_balanced = balance_by_size_and_difficulty(rows_labeled, sizes=args.sizes, seed=args.seed)
    after = len(rows_balanced)
    print(
        f"[balance] per size: downsampled {before} → {after} "
        "to get 1/3 easy/medium/hard each."
    )
    return rows_balanced


def build_dataset(rows_labeled: List[dict], args):
    """Split labeled rows into HF DatasetDict."""
    train_frac, val_frac, test_frac = map(float, args.split.split(","))
    train, val, test = stratified_split(
        rows_labeled, split=(train_frac, val_frac, test_frac), seed=args.seed
    )
    return DatasetDict({
        "train": to_dataset(train),
        "validation": to_dataset(val),
        "test": to_dataset(test),
    })


def write_outputs(dsd, args) -> None:
    """Write jsonl splits and README to disk."""
    for split_name in ["train", "validation", "test"]:
        path = os.path.join(args.out_dir, f"{split_name}.jsonl")
        with open(path, "w", encoding="utf-8") as file_obj:
            for ex in dsd[split_name]:
                file_obj.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"[write] {path}  ({len(dsd[split_name])} rows)")

    sizes_str = ", ".join(str(s) for s in sorted(set(args.sizes)))
    readme = f"""# Rush Hour 4×4, 5×5 & 6×6 — Balanced (solvable-only, fixed thresholds)
Enumerates boards (4×4/5×5) and/or samples for selected sizes (e.g., 6×6), BFS-solves to optimal sequences, and stratifies by difficulty.

## Policies
- sizes: {sizes_str}
- solvable_only: {args.solvable_only}
- exclude_already_solved: {args.exclude_already_solved}
- difficulty_scheme: {args.difficulty_scheme}
- fixed_thresholds: {args.fixed_thresholds}  # easy < t1, medium < t2, hard ≥ t2
- balance_per_size: {args.balance_per_size}

## Schema
- board (str) | size (int) | moves (int) | solution (list[str]) | difficulty (easy/medium/hard) | messages (chat)
"""
    with open(os.path.join(args.out_dir, "README.md"), "w", encoding="utf-8") as file_obj:
        file_obj.write(readme)


def maybe_push_to_hub(dsd, args) -> None:
    """Push dataset to the hub when requested."""
    if not args.push_to_hub:
        return
    if not args.dataset_id:
        raise SystemExit("--push_to_hub requires --dataset_id (e.g., user/rush-balanced-4-6)")
    token = os.environ.get("HF_TOKEN") or HfFolder.get_token()
    if not token:
        raise SystemExit("Missing HF token. Set HF_TOKEN or run `huggingface-cli login`.")
    print(f"[hub] pushing to {args.dataset_id} …")
    dsd.push_to_hub(args.dataset_id, private=False)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=os.path.join(args.out_dir, "README.md"),
        path_in_repo="README.md",
        repo_id=args.dataset_id,
        repo_type="dataset",
    )
    print("[hub] done.")


def log_summary(dsd) -> None:
    """Print split counts."""
    total = sum(len(dsd[s]) for s in dsd.keys())
    print(f"[done] dataset built with {total} examples.")
    for split_name in dsd.keys():
        print(f"  {split_name}: {len(dsd[split_name])}")


def main():
    """Parse CLI arguments, generate balanced datasets, and write/push outputs."""
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    config = build_config(args)
    all_rows: List[dict] = []

    for size in args.sizes:
        all_rows.extend(rows_for_size(size, args, config))

    rows_labeled = label_rows(all_rows, args, config)
    rows_balanced = maybe_balance(rows_labeled, args)
    dsd = build_dataset(rows_balanced, args)
    write_outputs(dsd, args)
    maybe_push_to_hub(dsd, args)
    log_summary(dsd)

if __name__ == "__main__":
    # More stable on some clusters than 'fork'
    try:
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
