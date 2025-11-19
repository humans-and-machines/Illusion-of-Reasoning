from __future__ import annotations
# -*- coding: utf-8 -*-
"""
cryptic_rewards.py  —  Dense-but-precise rewards for GRPO on cryptic crosswords.

Rewarding scheme (all rewards are per-sample and clipped to [0, 1]):

  TAGS & TAG FACTOR
  -----------------
  We look for four tags in the model's completion:
      <think>, </think>, <answer>, </answer>
  Let present_tags be how many of these appear (0..4). We define:
      tag_factor = present_tags / 4.0          # 0.00, 0.25, 0.50, 0.75, 1.00
  This tag_factor multiplies the tiny "contains" bonus and the crossword
  accuracy reward (shaping). The exact-match term remains a strict 0/1.

  COMPONENTS
  ----------
  1) Exact match inside <answer>…</answer>  (binary, NOT scaled by tag_factor)
     - Extract inner <answer> text; canonicalize to LETTERS-ONLY A–Z string.
     - Compare to canonicalized gold. If equal → base = 1.0, else 0.0.
     - Optional enumeration check: reject if lengths mismatch (see kwargs).

  2) Tiny "contains-anywhere" bonus  (scaled by tag_factor)
     - If the gold string appears ANYWHERE in the whole completion as a
       stand-alone word (not touching letters on either side), we add:
           contains_bonus * tag_factor    (default contains_bonus = 0.02)
     - "Stand-alone word" means the character immediately before/after the
       match is NOT a letter (Unicode-aware). Implemented with lookarounds:
           (?<!LETTER) gold (?!LETTER), LETTER = [^\\W\\d_]
       (so punctuation, quotes, spaces, parentheses, etc. are valid separators;
        glued forms like 'foosuite' or 'suitefoo' do not trigger the bonus.)

  3) Crossword accuracy reward (shaping)  (scaled by tag_factor)
     - Canonicalize completion and gold as in (2), and check if the gold
       appears ANYWHERE. If yes, we compute a length factor:
           0.0 at <= min_tokens, → 1.0 by >= max_full_len (linear in between)
       and multiply by tag_factor. This shaping is added as:
           + 0.25 * crossword_accuracy_reward
       Defaults: min_tokens=25, max_full_len=80 (override via kwargs).

  FINAL
  -----
  pure_accuracy_reward(...) returns:
      clip_0_1( base + contains_bonus * tag_factor + 0.25 * crossword_accuracy )
  crossword_accuracy_reward(...) returns the shaping term itself (already
  tag-scaled and length-scaled), useful when combining rewards explicitly.

Kwargs you may pass:
  - contains_bonus: float (default 0.02) tiny bonus magnitude
  - expected_len / n_letters / length / lengths: enumeration checks for exact match
  - min_tokens (int, default 25), max_full_len (int, default 80)

Notes:
  • Lookbehind/ahead assertions are fixed-width and zero-width in Python’s `re`;
    we use 1-char letter lookarounds to ensure separate-word matches robustly.
  • We purposely avoid \\b “word boundary” because \\b treats digits/_ as “word”
    characters, which isn’t the desired notion for crossword “words”.

"""


# ----------------------------- config ------------------------------
import math
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ----------------------------- config ------------------------------

MIN_TOKENS_DEFAULT = 25     # crossword shaping: 0 at <= this many tokens
MAX_FULL_LEN_DEFAULT = 80   # crossword shaping: 1 at >= this many tokens

# Treat “letter” as Unicode letters (no digits/underscore) for word isolation.
LETTER = r"[^\W\d_]"  # letters only (Unicode)
# ----------------------------- regexes -----------------------------

# Accept the tag pair anywhere in the string (not anchored).
_format_pat = re.compile(r"(?si)<think>.*?</think>.*?<answer>.*?</answer>")

# Extract the <answer> inner text; allow newlines/spaces.
_answer_pat = re.compile(r"(?si)<answer>\s*(.*?)\s*</answer>")

# ---------------------------- helpers ------------------------------

def _extract_content(comp: Any) -> str:
    """
    Accepts:
      • a plain string
      • a chat-style list like [{'role': 'assistant', 'content': '...'}]
      • nested sequences from chat structures
    Returns assistant text as a string.
    """
    if comp is None:
        return ""
    if isinstance(comp, str):
        return comp
    if isinstance(comp, Sequence) and comp:
        first = comp[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
        if isinstance(first, Sequence):
            return _extract_content(first)
    return str(comp)


def _canon_crossword_letters(s: str) -> str:
    """
    Canonicalize a crossword answer to LETTERS-ONLY A–Z:
      - Unicode NFKD + strip combining marks (remove accents)
      - Normalize curly quotes/dashes → straight variants
      - Map '&' → 'AND'
      - Uppercase
      - Keep only A–Z
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                                   "—": "-", "–": "-", "−": "-"}))
    s = s.replace("&", "AND").upper()
    return "".join(ch for ch in s if "A" <= ch <= "Z")


def _count_tokens_whitespace(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))

# --------------------------- rewards -------------------------------


def _normalize_for_word_match(s: str) -> str:
    """
    Normalize without removing separators so regex word boundaries work.
    - Strip accents
    - Normalize curly quotes/dashes
    - Map '&' → 'AND'
    (Case-insensitive regex handles case differences.)
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                                   "—": "-", "–": "-", "−": "-"}))
    return s.replace("&", "AND")

# --------------------------- rewards -------------------------------

def crossword_accuracy_reward(
    completions: List[Any],
    answer:      List[str],
    **kw,
) -> List[float]:
    """
    "Contains-anywhere" shaping, scaled by tag_factor and length:

      reward = has_gold * length_factor * tag_factor

    where:
      - has_gold:
          contains_mode = "any"        → (default) gold letters appear anywhere (after canon)
                        = "contiguous" → gold letters appear within a single contiguous LETTER+ run
                        = "word"       → gold letters appear as a standalone LETTER+ token
      - length_factor:  0 at <= min_tokens; 1 at >= max_full_len; linear ramp.
      - tag_factor:     (#present_tags / 4.0), tags ∈ {<think>, </think>, <answer>, </answer>}.

    Kwargs:
      - min_tokens (int, default 25)
      - max_full_len (int, default 80)
      - contains_mode (str, default "any"): "any" | "contiguous" | "word"
    """
    outs: List[float] = []

    min_tokens = int(kw.get("min_tokens", MIN_TOKENS_DEFAULT))
    max_full   = int(kw.get("max_full_len", MAX_FULL_LEN_DEFAULT))
    contains_mode = str(kw.get("contains_mode", "any")).lower()
    scaled = bool(kw.get("scaled", False))  # default: no scaling (tests expect full credit)

    TAGS = ("<think>", "</think>", "<answer>", "</answer>")

    for gold, comp in zip(answer, completions):
        raw = _extract_content(comp) or ""
        txt_lower = raw.lower()

        present_tags = sum(1 for t in TAGS if t in txt_lower)
        tag_factor   = present_tags / 4.0 if scaled else 1.0

        gold_can = _canon_crossword_letters(gold)

        # --- has_gold according to contains_mode ---
        if contains_mode == "any":
            raw_can  = _canon_crossword_letters(raw)
            has_gold = bool(gold_can) and (gold_can in raw_can)
        else:
            raw_norm = _normalize_for_word_match(raw)
            tokens = [_canon_crossword_letters(t)
                      for t in re.findall(fr"{LETTER}+", raw_norm, flags=re.UNICODE)]
            if contains_mode == "word":
                has_gold = bool(gold_can) and any(tok == gold_can for tok in tokens)
            else:  # "contiguous"
                has_gold = bool(gold_can) and any(gold_can in tok for tok in tokens)

        if not scaled:
            outs.append(1.0 if has_gold else 0.0)
            continue

        # Length ramp (whitespace-token proxy)
        n_tokens = _count_tokens_whitespace(raw)
        if n_tokens <= min_tokens:
            length_factor = 0.0
        elif n_tokens >= max_full:
            length_factor = 1.0
        else:
            length_factor = (n_tokens - min_tokens) / float(max_full - min_tokens)

        outs.append( (1.0 if has_gold else 0.0) * length_factor * tag_factor )

    return outs

def pure_accuracy_reward(
    completions: List[Any],
    answer:      List[str],
    **kw,
) -> List[float]:
    """
    Exact-match + tiny separate-word bonus + crossword shaping (clipped to 1.0):

      base = 1.0 iff canonicalized <answer>…</answer> == canonicalized gold, else 0.0

      tiny_bonus = contains_bonus * tag_factor  if the canonicalized letters of the
                   gold appear as a standalone LETTER+ word in the (normalized)
                   completion (no spaces/punct inside the match); else 0.0.
         • Regex (case-insensitive):
              (?<!LETTER)  GOLD_LETTERS  (?!LETTER)
           where LETTER = [^\\W\\d_], and GOLD_LETTERS has no spaces/punct.

      shaping = 0.25 * crossword_accuracy_reward(…)  # already tag-/length-scaled

      return min(1.0, base + tiny_bonus + 0.25 * shaping)
    """
    outs: List[float] = []
    contains_bonus = float(kw.get("contains_bonus", 0.02))
    TAGS = ("<think>", "</think>", "<answer>", "</answer>")

    for comp, gold in zip(completions, answer):
        txt = _extract_content(comp) or ""
        txt_lower = txt.lower()
        g   = (gold or "").strip()

        # tag factor (¼ per present tag; applies to tiny bonus)
        present_tags = sum(1 for t in TAGS if t in txt_lower)
        tag_factor   = present_tags / 4.0

        # ---- WORD-LEVEL partial credit (no cross-space) ----
        # Use canonicalized letters for the gold and a normalized text that preserves separators.
        gold_letters = _canon_crossword_letters(g)        # A–Z only, no spaces/hyphens
        raw_norm     = _normalize_for_word_match(txt)     # keeps separators for boundaries

        # Standalone word of letters-only (prevents hits like 'fooa nsfoo' or 'XSUITEX')
        sep_word_pat = re.compile(
            rf"(?i)(?<!{LETTER}){re.escape(gold_letters)}(?!{LETTER})",
            re.UNICODE
        )
        has_sep_word = bool(gold_letters) and bool(sep_word_pat.search(raw_norm))
        tiny_bonus   = contains_bonus * tag_factor if has_sep_word else 0.0

        # ---- Exact match inside <answer>…</answer> (binary) ----
        m = _answer_pat.search(txt)
        if not m:
            outs.append(tiny_bonus)      # no answer tag → only tiny bonus (if any)
            continue

        pred_c = _canon_crossword_letters(m.group(1))
        gold_c = _canon_crossword_letters(g)

        # Optional strict enumeration enforcement
        expected_len = kw.get("expected_len") or kw.get("n_letters") or kw.get("length")
        if expected_len is None and "lengths" in kw:
            try:
                expected_len = sum(int(x) for x in kw["lengths"])
            except Exception:
                expected_len = None
        if expected_len is not None:
            try:
                if len(pred_c) != int(expected_len):
                    outs.append(tiny_bonus)  # enumeration mismatch → only tiny bonus
                    continue
            except Exception:
                pass

        base = 1.0 if (pred_c == gold_c) else 0.0
        outs.append(base + tiny_bonus)

    # Add 0.25 * crossword shaping (optionally make contains strict via contains_mode kw)
    shaping = crossword_accuracy_reward(completions, answer, **kw)
    return [min(1.0, base + 0.25 * s) for base, s in zip(outs, shaping)]


def _canon_crossword_letters(text: str) -> str:
    """
    Canonicalize a crossword answer to LETTERS-ONLY A–Z string:
      - Unicode NFKD + strip combining marks (remove accents)
      - Normalize curly quotes/dashes; map '&' -> 'AND'
      - Uppercase
      - Drop everything that's not A–Z
    """
    if text is None:
        return ""

    # Normalize unicode + strip accents
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    # Normalize common punctuation variants
    text = text.translate(
        str.maketrans(
            {"’": "'", "‘": "'", "“": '"', "”": '"', "—": "-", "–": "-", "−": "-"},
        ),
    )

    # Map symbols that semantically carry letters
    text = text.replace("&", "AND")

    # Uppercase, then keep letters only
    text = text.upper()
    return "".join(ch for ch in text if "A" <= ch <= "Z")


# ── helpers you referenced ──────────────────────────────────────────────
MOVE_RE = re.compile(r"^[A-Z][<>^v]\d+$")
TOK_LIST = re.compile(r"\s*,\s*")
ANS_TAG = re.compile(r"(?is)<answer>(.*?)</answer>")


def _extract_answer_text(text: str) -> str:
    """Return inner text of <answer>…</answer> if present; else the whole string."""
    match = ANS_TAG.search(text or "")
    return (match.group(1) if match else (text or "")).strip()


def _canon_token(tok: str) -> Optional[str]:
    """Uppercase, strip spaces, ensure it looks like PIECE+DIR+STEPS."""
    token_str = (tok or "").strip().upper().replace(" ", "")
    return token_str if MOVE_RE.match(token_str) else None

# ---------- Parsing helpers ----------

def _extract_answer_block(text: str) -> Optional[str]:
    """Prefer <answer>...</answer>; otherwise find the first token list in text."""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.S | re.I)
    if m:
        cand = m.group(1).strip()
        if _looks_like_token_list(cand):
            return cand
    # fallback: any token list looking like A>2,B<1,...
    m2 = re.search(r"([A-Za-z][<>^v]\d+(?:\s*,\s*[A-Za-z][<>^v]\d+)*)", text)
    return m2.group(1) if m2 else None

def _looks_like_token_list(s: str) -> bool:
    s = s.strip().upper().replace(" ", "")
    if not s:
        return False
    parts = s.split(",")
    return all(TOKEN_RE.match(p) for p in parts)

def _canon_seq(seq: str) -> Optional[List[str]]:
    """
    Canonicalize a move sequence string to a list of tokens:
      - Uppercase letters
      - Remove spaces
      - Validate each token
      - Merge consecutive identical (car,dir) tokens by summing steps
    Returns None if invalid.
    """
    if not isinstance(seq, str):
        return None
    s = seq.strip().upper().replace(" ", "")
    if not s:
        return None
    parts = s.split(",")
    toks = []
    for p in parts:
        if not TOKEN_RE.match(p):
            return None
        car = p[0]
        d = p[1]
        n = int(p[2:])
        if n <= 0:
            return None
        toks.append((car, d, n))

    # merge consecutive same (car,dir)
    merged = []
    for car, d, n in toks:
        if merged and merged[-1][0] == car and merged[-1][1] == d:
            merged[-1] = (car, d, merged[-1][2] + n)
        else:
            merged.append((car, d, n))
    return [f"{c}{d}{k}" for (c, d, k) in merged]


def _len_tokens(tokens: Optional[List[str]]) -> int:
    return 0 if not tokens else len(tokens)

# rush_reward.py
# Complete Rush Hour reward utilities:
# - Robust token parsing & canonicalization
# - Prompt/board extraction (handles '4x4' or '4×4', 'Board:', 'Minimal/Optimal moves')
# - Board simulator with walls 'x' and empty 'o'
# - Potential-based shaping: Φ = blockers + distance_to_exit (decrease is rewarded)
# - Rewards:
#     * rush_solution_shaped: dense [0,1], preserves optimal solution ordering
#     * rush_solution_exact:  1 iff exact canonical match; else 0
#
# Usage (TRL-style):
#   reward_fn: rush_reward:rush_solution_shaped
#
# Notes:
# - Tokens look like:  A>2,B<1,Cv3
#   piece ∈ 'A'..'Z'; dir ∈ {<, >, ^, v}; steps ∈ {1,2,...}
# - We accept 'V' and normalize to 'v'.
# - We prefer answers inside <answer>...</answer>, but will fall back to the first token list.

# ---------- Token utilities ----------

# Directions allowed; accept uppercase 'V' too
_TOKEN_RE = re.compile(r"^[A-Z][<>^vV]\d+$")


def _parse_and_normalize_token(token: str) -> Optional[Tuple[str, str, int]]:
    """
    Parse a single token (e.g., 'Bv3', 'A>2', 'C^1') and normalize to:
      (piece_upper, dir_norm, steps_int)
    dir_norm ∈ {'<','>','^','v'}
    """
    if not isinstance(token, str):
        return None
    token = token.strip().replace(" ", "")
    if not _TOKEN_RE.match(token):
        return None

    piece = token[0].upper()
    direction = token[1]
    direction = "v" if direction in ("v", "V") else direction
    try:
        steps = int(token[2:])
    except (TypeError, ValueError):
        return None
    if steps <= 0:
        return None
    return (piece, direction, steps)


def _looks_like_token_list(text: str) -> bool:
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
    into a list of normalized tokens ["Bv1","A>1"], merging consecutive
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

    # Merge consecutive same (piece, dir)
    merged: List[Tuple[str, str, int]] = []
    for (piece, direction, steps) in tokens:
        if merged and merged[-1][0] == piece and merged[-1][1] == direction:
            merged[-1] = (piece, direction, merged[-1][2] + steps)
        else:
            merged.append((piece, direction, steps))
    return [f"{piece}{direction}{steps}" for (piece, direction, steps) in merged]

def _extract_answer_block(text: str) -> Optional[str]:
    """Prefer <answer>...</answer>; otherwise find the first token list."""
    if not isinstance(text, str):
        return None
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.S | re.I)
    if m:
        cand = m.group(1).strip()
        if _looks_like_token_list(cand):
            return cand
    # fallback: tolerant regex for tokens
    m2 = re.search(r"([A-Za-z][<>^vV]\d+(?:\s*,\s*[A-Za-z][<>^vV]\d+)*)", text)
    return m2.group(1) if m2 else None

# Accept commas, whitespace, newlines, or semicolons between tokens.
_TOKEN_SCAN = re.compile(r"([A-Za-z])\s*([><\^vVUDLR])\s*([0-9]+)")

def _canon_seq_from_text(text: Any) -> Optional[List[str]]:
    """Extract tokens anywhere in the prediction (favoring <answer>...</answer> if present)."""
    if isinstance(text, (list, tuple)):
        raw_text = ",".join(str(token) for token in text)
    else:
        raw_text = str(text or "")
    # Prefer the <answer> block if present
    match = re.search(r"(?is)<answer>\s*(.*?)\s*</answer>", raw_text)
    scan_text = match.group(1) if match else raw_text

    moves: List[Tuple[str, str, int]] = []
    for match_obj in _TOKEN_SCAN.finditer(scan_text):
        piece = match_obj.group(1).upper()
        direction_raw = match_obj.group(2)  # one of <,>,^,v or U/D/L/R (sometimes V)
        # Map symbols as-is; map letters case-insensitively to canonical symbols
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

    # merge consecutive same (piece,dir)
    merged_moves: List[Tuple[str, str, int]] = []
    for (piece, direction, steps) in moves:
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
    - list[dict] with 'content' fields
    - dict with 'prompt' / 'content' fields
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
                # OpenAI-style chat message: {'role': 'user', 'content': '...'}
                val = item.get("content")
                if isinstance(val, str):
                    chunks.append(val)
                elif isinstance(val, (list, tuple)):
                    # some SDKs split content into parts
                    chunks.extend([str(x) for x in val])
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(prompts)

def _extract_puzzle_from_prompt(prompt: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    N = None
    mN = re.search(r"Board\s*size\s*:\s*(\d+)\s*[xX×]\s*(\d+)", prompt)
    if mN:
        rN, cN = int(mN.group(1)), int(mN.group(2))
        if rN == cN:
            N = rN

    # NEW: accept board on next line; allow whitespace inside, then strip it
    mB = re.search(r"(?is)Board\s*:\s*([A-Za-zx\s]+)", prompt)
    board = None
    if mB:
        board_raw = mB.group(1)
        board = re.sub(r"\s+", "", board_raw)  # remove newlines/spaces

    # If we know N and captured more than needed, crop to N*N
    if N is not None and board:
        need = N * N
        if len(board) >= need:
            board = board[:need]
        else:
            # try to find a contiguous run of length N*N anywhere (fallback)
            patt = re.compile(rf"\b([A-Za-zx]{{{need}}})\b")
            m = patt.search(prompt)
            if m:
                board = m.group(1)

    # If N unknown but board exists, infer from perfect square length
    if N is None and board:
        root = int(round(math.sqrt(len(board))))
        if root * root == len(board):
            N = root

    mg = re.search(r"(?:Minimal|Optimal)\s+(?:length\s*\(moves\)|moves?)\s*:\s*(\d+)", prompt, flags=re.I)
    gold_moves = int(mg.group(1)) if mg else None
    return board, N, gold_moves

# ---------- Board & simulation ----------

class Board:
    """
    Rush Hour board:
      - grid: NxN row-major string
          'o' = empty, 'x' = wall, 'A'..'Z' = cars (length 2 or 3 typically)
      - 'A' is the target car (assumed horizontal); goal is A's rightmost cell at col N-1.
    """

    def __init__(self, grid_str: str, N: int):
        self.N = N
        s = (grid_str or "").strip()
        if len(s) != N * N:
            raise ValueError(f"Board string length {len(s)} != N*N ({N*N})")
        self.grid: List[List[str]] = [list(s[r * N:(r + 1) * N]) for r in range(N)]
        self.cars: Dict[str, List[Tuple[int, int]]] = self._index_cars()
        self.orient: Dict[str, str] = self._orientations()  # 'H' or 'V'

    def clone(self) -> "Board":
        b = Board("o" * (self.N * self.N), self.N)
        b.grid = [row[:] for row in self.grid]
        b.cars = {k: v[:] for k, v in self.cars.items()}
        b.orient = dict(self.orient)
        return b

    def _index_cars(self) -> Dict[str, List[Tuple[int, int]]]:
        cars: Dict[str, List[Tuple[int, int]]] = {}
        for r in range(self.N):
            for c in range(self.N):
                ch = self.grid[r][c]
                if ch != 'o' and ch != 'x':
                    cars.setdefault(ch, []).append((r, c))
        for k in cars:
            cars[k].sort()
        return cars

    def _orientations(self) -> Dict[str, str]:
        orient: Dict[str, str] = {}
        for car, cells in self.cars.items():
            if len(cells) == 1:
                orient[car] = 'H'  # default for single-cells
                continue
            rows = {r for (r, _) in cells}
            cols = {c for (_, c) in cells}
            if len(rows) == 1:
                orient[car] = 'H'
            elif len(cols) == 1:
                orient[car] = 'V'
            else:
                # malformed; pick H to avoid crashing
                orient[car] = 'H'
        return orient

    def is_solved(self) -> bool:
        if 'A' not in self.cars:
            return False
        max_c = max(c for (_, c) in self.cars['A'])
        return max_c == self.N - 1

    def a_row_and_rightmost(self) -> Tuple[int, int]:
        if 'A' not in self.cars or not self.cars['A']:
            return (-1, -1)
        row = self.cars['A'][0][0]
        rightmost = max(c for (_, c) in self.cars['A'])
        return row, rightmost

    def blockers_and_distance(self) -> Tuple[int, int]:
        """
        Heuristic pieces:
          - blockers: count of cars/walls in A's row strictly to A's right
          - distance: (N - 1 - rightmost_A)
        """
        row, rightmost = self.a_row_and_rightmost()
        if row < 0:
            return (0, 0)
        dist = (self.N - 1) - rightmost
        blockers = 0
        for c in range(rightmost + 1, self.N):
            ch = self.grid[row][c]
            if ch == 'x':
                blockers += 1
            elif ch != 'o':
                blockers += 1
        return blockers, dist

    def _cells_for(self, car: str) -> List[Tuple[int, int]]:
        return self.cars.get(car, [])

    def _clear_cells(self, cells: Iterable[Tuple[int, int]]) -> None:
        for r, c in cells:
            self.grid[r][c] = 'o'

    def _occupy_cells(self, car: str, cells: Iterable[Tuple[int, int]]) -> None:
        cells = list(cells)
        for r, c in cells:
            self.grid[r][c] = car
        self.cars[car] = sorted(cells)

    def _step_move(self, car: str, direction: str) -> bool:
        """
        Attempt to move ``car`` one cell in ``direction``.

        Returns ``True`` if the move is legal and applied, otherwise ``False``.
        """
        cells = self._cells_for(car)
        if not cells:
            return False

        ori = self.orient.get(car, 'H')
        if direction in '<>' and ori != 'H':
            return False
        if direction in '^v' and ori != 'V':
            return False

        blocked = False
        new_cells: List[Tuple[int, int]]

        if direction == '<':
            leftmost = min(c for (_, c) in cells)
            row = cells[0][0]
            next_col = leftmost - 1
            blocked = next_col < 0 or self.grid[row][next_col] != 'o'
            new_cells = [(row, col - 1) for (row, col) in cells]
        elif direction == '>':
            rightmost = max(c for (_, c) in cells)
            row = cells[0][0]
            next_col = rightmost + 1
            blocked = next_col >= self.N or self.grid[row][next_col] != 'o'
            new_cells = [(row, col + 1) for (row, col) in cells]
        elif direction == '^':
            top = min(r for (r, _) in cells)
            col = cells[0][1]
            next_row = top - 1
            blocked = next_row < 0 or self.grid[next_row][col] != 'o'
            new_cells = [(row - 1, col) for (row, col) in cells]
        else:  # 'v'
            bottom = max(r for (r, _) in cells)
            col = cells[0][1]
            next_row = bottom + 1
            blocked = next_row >= self.N or self.grid[next_row][col] != 'o'
            new_cells = [(row + 1, col) for (row, col) in cells]

        if blocked:
            return False

        self._clear_cells(cells)
        self._occupy_cells(car, new_cells)
        return True

    def apply_token(self, token: str) -> bool:
        """
        Apply a compact move token like ``A>2`` to the board.

        Returns ``True`` if all steps are legal and applied, otherwise ``False``.
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
    # unwrap ["A>2,Bv5"] → "A>2,Bv5"
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
    # whitespace-delimited proxy for token count (keeps it fast & robust)
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
            1, (think_full_tokens - think_min_tokens),
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


def _rush_solve_terms(
    pred_can: List[str],
    board: Board | None,
    target_moves: int | None,
) -> Tuple[float, float, float]:
    """
    Compute (prefix_fraction, solve_term, phi_term) for a predicted sequence.

    prefix_fraction is the legal-prefix fraction if a board is available, else 0.
    """
    solve_term = 0.0
    phi_term = 0.0

    if board is None:
        if target_moves is not None:
            sequence_len = _len_tokens(pred_can)
            target = int(target_moves)
            solve_term = (
                1.0
                if sequence_len <= target
                else 1.0 / (1.0 + (sequence_len - target))
            )
        return 0.0, solve_term, phi_term

    valid_k, solved, final_board = _simulate_prefix(board, pred_can)
    prefix_fraction = valid_k / max(1, _len_tokens(pred_can))

    if solved:
        sequence_len = _len_tokens(pred_can)
        target = target_moves if target_moves is not None else sequence_len
        solve_term = (
            1.0
            if sequence_len <= target
            else 1.0 / (1.0 + (sequence_len - target))
        )

    blockers_start, dist_start = board.blockers_and_distance()
    blockers_end, dist_end = final_board.blockers_and_distance()
    heuristic_start = blockers_start + dist_start
    heuristic_end = blockers_end + dist_end
    if heuristic_start > 0:
        phi_term = max(0.0, (heuristic_start - heuristic_end) / heuristic_start)
    else:
        phi_term = 1.0 if solved else 0.0

    return prefix_fraction, solve_term, phi_term


def rush_solution_shaped(  # pylint: disable=too-many-locals
    *, 
    prompts,
    completions,
    answer=None,
    gold=None,
    # NEW: allow bypassing prompt parsing entirely:
    board_str: str | None = None,
    N: int | None = None,
    gold_moves: int | None = None,
    **kw,
) -> List[float]:
    """
    Dense, shaped reward in [0,1] for Rush Hour that works WITH or WITHOUT a board.

    When a board is provided (board_str+N), legality and Φ-progress are used.
    When no board is provided, the reward still gives:
      - exact match (vs gold candidates)
      - prefix/LCP credit vs gold
      - gold-only "solve optimality": shorter sequences are better based on gold_moves
    """
    # weights (sum ≤ 1 to keep final score in [0,1]; tune if desired)
    w_exact  = float(kw.get("w_exact", 0.65))
    w_solve  = float(kw.get("w_solve", 0.20))
    w_prefix = float(kw.get("w_prefix", 0.10))
    w_phi    = float(kw.get("w_phi", 0.05))

    # Normalize inputs
    if isinstance(completions, str):
        completions = [completions]
    else:
        completions = list(completions)

    # Canonicalize gold candidates
    gold_cands = _canon_gold_candidates(gold or answer or kw.get("answers") or kw.get("gold_answers"))
    gold_min_len = min((_len_tokens(gc) for gc in gold_cands if gc is not None), default=None)

    # Prefer explicit board/moves if provided; else try parsing prompt for convenience
    board, gm_from_prompt = _rush_build_board_and_moves(prompts, board_str, N)

    # Choose a gold_moves target for boardless solve shaping
    if gold_moves is not None:
        target_moves = gold_moves
    elif gm_from_prompt is not None:
        target_moves = gm_from_prompt
    else:
        target_moves = gold_min_len

    think_min_tokens = int(kw.get("think_min_tokens", 25))
    think_full_tokens = int(kw.get("think_full_tokens", 100))
    think_bonus_cap = float(kw.get("think_bonus_cap", 0.02))

    out_scores: List[float] = []
    for pred in completions:
        pred_text = str(pred or "")
        pred_can  = _canon_seq_from_text(pred_text)

        # 0) Formatting bonus when we see at least one valid token but parser fails
        if pred_can is None:
            bonus = _rush_formatting_bonus(
                pred_text,
                think_min_tokens,
                think_full_tokens,
                think_bonus_cap,
            )
            out_scores.append(bonus)
            continue

        # 1) Exact match
        exact = 1.0 if any(pred_can == gc for gc in gold_cands if gc is not None) else 0.0

        # 2) Prefix credit (LCP vs best gold candidate)
        prefix_from_gold = _rush_prefix_credit(pred_can, gold_cands)

        # 3) Solve & Φ terms (and legal-prefix fraction if a board is present)
        legal_prefix, solve_term, phi_term = _rush_solve_terms(
            pred_can,
            board,
            target_moves,
        )
        prefix = max(prefix_from_gold, legal_prefix)

        score = (
            w_exact  * exact +
            w_solve  * solve_term +
            w_prefix * prefix +
            w_phi    * phi_term
        )
        out_scores.append(float(max(0.0, min(1.0, score))))

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
    # prompts accepted for API parity; not used directly in this reward
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

        # 0) Formatting bonus when we see at least one valid token but parser fails
        if pred_can is None:
            bonus = _rush_formatting_bonus(
                pred_text,
                think_min_tokens,
                think_full_tokens,
                think_bonus_cap,
            )
            scores.append(bonus)
            continue

        exact_match = any(pred_can == gc for gc in gold_cands if gc is not None)
        scores.append(1.0 if exact_match else 0.0)

    return scores


def _canon_math(text: str) -> str:
    """
    Canonicalize math answers:
    - strip leading/trailing whitespace
    - remove LaTeX spacing commands (\\ ), $...$, and curly braces around single tokens
    - normalize minus-zero to zero
    - drop trailing .0 from integers
    - remove spaces inside the expression unless inside LaTeX commands
    - unify parentheses usage for single-number expressions
    """
    text = text.strip()

    # Remove LaTeX math mode markers
    text = text.replace("$", "")

    # Remove LaTeX spacing commands
    text = re.sub(r"\\\s+", "", text)

    # Remove outer braces around a single token
    if re.fullmatch(r"\{[^{}]+\}", text):
        text = text[1:-1]

    # Drop surrounding parentheses if they enclose just a number/frac/root
    if re.fullmatch(r"\([^()]+\)", text):
        inner = text[1:-1].strip()
        # only drop if it doesn't change grouping meaning
        if re.match(r"^[\d\.\-\\sqrt]+$", inner):
            text = inner

    # Strip spaces
    text = text.replace(" ", "")

    # Convert -0 to 0
    if text in ("-0", "+0"):
        text = "0"

    # Remove trailing .0 from integers
    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".")[0]

    return text

def pure_accuracy_reward_math(
    completions: List[Any],
    answer:      List[str],
    **_unused_kwargs,
) -> List[float]:
    """
    Pure exact-match for math problems with format requirement:
      • Output must match <think> … </think><answer> … </answer> (any spacing/newlines).
      • The <answer> content must exactly equal the gold (canonicalized math form).
    """
    outs: List[float] = []

    for comp, gold in zip(completions, answer):
        txt = _extract_content(comp)

        # Must satisfy the full tag template
        if not _format_pat.match(txt):
            outs.append(0.0)
            continue

        match = _answer_pat.search(txt)
        if not match:
            outs.append(0.0)
            continue

        pred = match.group(1)
        is_correct_match = (_canon_math(pred) == _canon_math(gold))
        outs.append(1.0 if is_correct_match else 0.0)

    return outs


# --------------------------- crossword-format helpers -------------------------------

def _extract_answer_letters(comp: Any) -> str:
    """Pull the inner <answer>…</answer> text and canon to LETTERS-ONLY A–Z."""
    raw = _extract_content(comp)
    match = _answer_pat.search(raw or "")
    if not match:
        return ""
    return _canon_crossword_letters(match.group(1))


def crossword_format_reward(
    completions: List[Any],
    answer: List[str],
    prompt: List[Any] | None = None,  # unused but kept for API parity
) -> List[float]:
    """
    Formatting bonus (crosswords):
    - 0 if the answer is correct.
    - 1 if the answer is wrong but:
        * tags for <think>…</think><answer>…</answer> are present, and
        * the submitted answer is all-uppercase (no lowercase letters).
    Otherwise 0.
    """
    _ = prompt  # kept for API parity; not used in scoring
    outs: List[float] = []
    for comp, gold in zip(completions, answer):
        raw = _extract_content(comp) or ""
        gold_can = _canon_crossword_letters(gold)
        ans_can = _extract_answer_letters(comp)

        # no bonus for correct answers
        if ans_can == gold_can and gold_can:
            outs.append(0.0)
            continue

        has_tags = bool(_format_pat.search(raw))
        ans_match = _answer_pat.search(raw or "")
        ans_text = ans_match.group(1) if ans_match else ""
        has_lower = any(ch.isalpha() and ch.islower() for ch in ans_text)
        outs.append(1.0 if has_tags and ans_text and not has_lower else 0.0)
    return outs


def crossword_length_reward(
    completions: List[Any],
    prompt: List[Any],
) -> List[float]:
    """
    Reward 1.0 if the answer length matches the enumeration in the prompt, else 0.0.
    Prompts are expected to include "(N)" where N is the length.
    """
    outs: List[float] = []
    for comp, prm in zip(completions, prompt):
        user_txt = ""
        if isinstance(prm, list) and prm:
            # expect a chat-style list of messages; pick user content
            for msg in prm:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_txt = str(msg.get("content", ""))
                    break
        match = re.search(r"\((\d+)\)", user_txt)
        if not match:
            outs.append(0.0)
            continue
        expected_len = int(match.group(1))
        ans_len = len(_extract_answer_letters(comp))
        outs.append(1.0 if ans_len == expected_len else 0.0)
    return outs


def formating(completions, **_unused_kwargs):
    """
    Check that the reasoning/answer are wrapped in <think>…</think><answer>…</answer>
    with newlines; used as a formatting bonus in tests.
    """
    pattern = re.compile(r"^<think>\s*.*?\s*</think>\s*<answer>\s*.*?\s*</answer>\s*$", re.DOTALL)
    outs = []
    for comp in completions:
        raw = _extract_content(comp) or ""
        outs.append(1.0 if pattern.match(raw) else 0.0)
    return outs
