#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-pass batch inference for Rush Hour (car-parking) using Qwen2.5-style chat LMs
that produce: <think> ... </think><answer> ... </answer>

This version adds resume/fill behavior:
 - If the output JSONL already contains some rows for an example_id, we detect
   which sample_idx values are present and write ONLY the missing ones.
 - Net effect: after this run, every example will have EXACTLY --num_samples rows.
"""

import os
import re
import json
import math
import sys
import logging
import argparse
from typing import Optional, List, Tuple, Dict, Any, Union
from collections import Counter, defaultdict  # [FILL] defaultdict import added

import torch
from torch.nn import functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

# ---- Soft, gold-only matching reward for Rush Hour ----
# Drop this below your canonicalization helpers (so _canon_rush_generic/_canon_rush_gold/TOKEN_RE exist).

from typing import Any, Dict, List, Tuple

def _toklist(seq: str) -> List[str]:
    """'Bv2,A>1' -> ['Bv2','A>1'] (assumes canonicalized input)"""
    if not seq:
        return []
    return [t for t in seq.split(",") if t]

def _split_token(tok: str) -> Tuple[str, str, int]:
    # Assumes canonical form already validated by TOKEN_RE.
    piece = tok[0]
    dir_ = tok[1]
    steps = int(tok[2:])
    return piece, dir_, steps

def _lcs_len(a: List[str], b: List[str]) -> int:
    # Token-level LCS (small sequences -> O(nm) is fine)
    n, m = len(a), len(b)
    dp = [0]*(m+1)
    for i in range(1, n+1):
        prev = 0
        for j in range(1, m+1):
            cur = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = cur
    return dp[m]

def _multiset_overlap_ratio(a: List[str], b: List[str]) -> float:
    ca, cb = Counter(a), Counter(b)
    inter = sum(min(ca[t], cb[t]) for t in set(ca) | set(cb))
    union = sum(max(ca[t], cb[t]) for t in set(ca) | set(cb))
    return (inter / union) if union else 0.0

def _piece_dir(tok: str) -> Tuple[str, str]:
    p, d, _ = _split_token(tok)
    return p, d

def rush_soft_match_reward(
    pred_answer_text: str,
    gold_answer_any: Any,
    *,
    weights: Dict[str, float] = None,
    length_penalty_base: float = 0.92,   # 0.92^|Δlen| (mild penalty)
) -> Tuple[float, Dict[str, Any]]:
    """
    Returns (reward in [0,1], breakdown dict). Uses only gold sequence(s).
    Reward = best over gold alternatives.

    Assumes _canon_rush_generic, _canon_rush_gold, and TOKEN_RE exist in scope.
    """
    if weights is None:
        weights = dict(
            prefix=0.35,
            pos_exact=0.25,
            piece_dir=0.15,
            step_close=0.10,
            lcs=0.10,
            bag_overlap=0.05,
        )
    # 1) Canonicalize
    pred_canon = _canon_rush_generic(pred_answer_text)
    gold_set   = _canon_rush_gold(gold_answer_any)

    detail = {
        "pred_canon": pred_canon,
        "gold_canon_options": sorted(list(gold_set)),
        "components": {},
        "picked_gold": None,
    }

    if (not pred_canon) or (not gold_set):
        # Invalid pred or missing gold -> 0
        return 0.0, detail

    # 2) Tokenize pred once
    pred_tokens = _toklist(pred_canon)

    best_score = 0.0
    best_components = None
    best_gold = None

    for gold_canon in gold_set:
        gold_tokens = _toklist(gold_canon)
        Lg = max(1, len(gold_tokens))  # guard

        # a) Exact match short-circuit
        if pred_canon == gold_canon:
            comp = dict(
                prefix=1.0, pos_exact=1.0, piece_dir=1.0, step_close=1.0,
                lcs=1.0, bag_overlap=1.0, length_penalty=1.0
            )
            score = 1.0
            if score > best_score:
                best_score, best_components, best_gold = score, comp, gold_canon
            continue

        # b) Components
        # prefix exact
        pfx = 0
        for i in range(min(len(pred_tokens), len(gold_tokens))):
            if pred_tokens[i] == gold_tokens[i]:
                pfx += 1
            else:
                break
        prefix = pfx / Lg

        # position-wise exact (overlap region, normalized by gold length)
        pos_exact = sum(
            1 for i in range(min(len(pred_tokens), len(gold_tokens)))
            if pred_tokens[i] == gold_tokens[i]
        ) / Lg

        # piece+dir match (ignore steps), position-wise
        pd_matches = 0
        step_closeness_vals = []
        for i in range(min(len(pred_tokens), len(gold_tokens))):
            pp, pd = _piece_dir(pred_tokens[i])
            gp, gd = _piece_dir(gold_tokens[i])
            if (pp, pd) == (gp, gd):
                pd_matches += 1
                _, _, ps = _split_token(pred_tokens[i])
                _, _, gs = _split_token(gold_tokens[i])
                # closeness = 1 - normalized abs error (clip to [0,1])
                denom = max(gs, 1)
                step_closeness_vals.append(max(0.0, 1.0 - abs(ps - gs) / denom))
            else:
                step_closeness_vals.append(0.0)
        piece_dir = pd_matches / Lg
        step_close = (sum(step_closeness_vals) / Lg) if Lg > 0 else 0.0

        # LCS ratio (token-level)
        lcs = _lcs_len(pred_tokens, gold_tokens) / Lg

        # multiset overlap (Jaccard over token multisets)
        bag_overlap = _multiset_overlap_ratio(pred_tokens, gold_tokens)

        # length penalty
        dlen = abs(len(pred_tokens) - len(gold_tokens))
        length_penalty = (length_penalty_base ** dlen)

        # weighted sum
        base = (
            weights["prefix"]     * prefix
          + weights["pos_exact"]  * pos_exact
          + weights["piece_dir"]  * piece_dir
          + weights["step_close"] * step_close
          + weights["lcs"]        * lcs
          + weights["bag_overlap"]* bag_overlap
        )
        score = max(0.0, min(1.0, base * length_penalty))

        if score > best_score:
            best_score = score
            best_components = dict(
                prefix=prefix,
                pos_exact=pos_exact,
                piece_dir=piece_dir,
                step_close=step_close,
                lcs=lcs,
                bag_overlap=bag_overlap,
                length_penalty=length_penalty,
                dlen=dlen,
            )
            best_gold = gold_canon

    detail["components"] = best_components or {}
    detail["picked_gold"] = best_gold
    return float(best_score), detail


# ───────────────────────── System prompt (from training) ─────────────────────────
SYSTEM_PROMPT = (
    "You are an expert Rush Hour ({N}×{N}) solver.\n"
    "INPUTS\n"
    "• Board (row-major string with 'o','A','B'..'Z', optional 'x')\n"
    "• Board size (e.g., 4×4/5×5/6×6)\n"
    "• Optimum moves {moves}\n"
    "OUTPUT\n"
    "• Exactly ONE optimal sequence in <answer> only.\n"
    "• Token = <PIECE><DIR><STEPS> (e.g., A>2,B<1,Cv3)\n"
    "• DIR: '<' left, '>' right, '^' up, 'v' down\n"
    "• No spaces/prose/extra lines in <answer>.\n"
    "GOAL\n"
    "• Right end of 'A' reaches the right edge.\n"
    "OPTIMALITY & TIE-BREAK\n"
    "• Use exactly {moves} tokens.\n"
    "• If multiple optimal, choose lexicographically smallest ASCII comma-list.\n"
    "VALIDATION\n"
    "• REGEX: ^[A-Z][<>^v]\\d+(,[A-Z][<>^v]\\d+)*$\n"
    "• AXES: A is 2-long horizontal; others length 2/3, fixed H or V.\n"
    "• COLLISION: no overlaps; 'x' is immovable.\n"
    "• GOAL: applying all tokens reaches the goal.\n"
    "• LENGTH: #tokens = {moves} (when provided).\n"
    "RETHINK\n"
    "1) In <think>, propose S1 and run all VALIDATION checks.\n"
    "2) If any fail, name the failure, propose S2, re-check.\n"
    "3) If any fail, propose S3, re-check. Repeat as needed.\n"
    "4) Put ONLY the final passing sequence in <answer>.\n"
    "EXAMPLE (guidance only)\n"
    "• Board 4×4: oAABCooBCoooDDoo, {moves}=2\n"
    "<think>\n"
    "S1: A>1 → GOAL✗ (blocked by B). Wait, hang on\n"
    "S2: Bv2,A>1 → all ✓.\n"
    "</think>\n"
    "<answer>\n"
    "Bv2,A>1\n"
    "</answer>\n"
)

# ───────────────────────── Regex + tags ─────────────────────────
RE_THINK  = re.compile(r"(?si)<think>(.*?)</think>")
RE_ANSWER = re.compile(r"(?si)<answer>(.*?)</answer>")

# Single-move token: piece letter + direction + steps (dir accepts v or V)
TOKEN_RE  = re.compile(r"^\s*([A-Za-z])([<>^vV])(\d+)\s*$")

# Optional reconsideration markers (analytics)
_RECONSIDER_PATTERNS = [
    ("wait_line",        re.compile(r"(?im)^\s*wait[,\.\-–—… ]", re.I)),
    ("wait_reconsider",  re.compile(r"\bwait\b.*\breconsider\b", re.I | re.S)),
    ("step_by_step",     re.compile(r"\bstep[-\s]?by[-\s]?step\b", re.I)),
    ("recheck",          re.compile(r"\bre[-\s]?check(ing)?\b", re.I)),
]

# ───────────────────────── Utils ─────────────────────────
def _finite_mean(vals: List[float]) -> Optional[float]:
    vs = [float(v) for v in vals if v == v and math.isfinite(float(v))]
    return (sum(vs) / len(vs)) if vs else None

def _extract_blocks(txt: str) -> Tuple[Optional[str], Optional[str]]:
    m = RE_THINK.search(txt); think = m.group(1).strip() if m else None
    m = RE_ANSWER.search(txt); ans   = m.group(1).strip() if m else None
    return think, ans

def _valid_tag_structure(full_text: str) -> bool:
    opens_think  = len(re.findall(r"(?i)<think>", full_text))
    closes_think = len(re.findall(r"(?i)</think>", full_text))
    opens_ans    = len(re.findall(r"(?i)<answer>", full_text))
    closes_ans   = len(re.findall(r"(?i)</answer>", full_text))
    if not (opens_think == closes_think == 1 and opens_ans == closes_ans == 1):
        return False
    try:
        a = re.search(r"(?i)<think>", full_text).start()
        b = re.search(r"(?i)</think>", full_text).start()
        c = re.search(r"(?i)<answer>", full_text).start()
        d = re.search(r"(?i)</answer>", full_text).start()
        return a < b < c < d
    except Exception:
        return False

# ---- Canonicalization for Rush Hour sequences --------------------------------
def _canon_move(token: str) -> Optional[str]:
    """Canonicalize one move: piece upper, dir in {<,>,^,v}, steps numeric."""
    m = TOKEN_RE.match(token)
    if not m:
        return None
    piece, direction, steps = m.groups()
    piece = piece.upper()
    direction = direction.lower() if direction in ("v", "V") else direction  # normalize v/V → v
    return f"{piece}{direction}{steps}"

def _canon_join(tokens: List[str]) -> Optional[str]:
    out: List[str] = []
    for t in tokens:
        ct = _canon_move(t)
        if ct is None:
            return None
        out.append(ct)
    return ",".join(out)

def _canon_rush_string(s: str) -> Optional[str]:
    # Remove whitespace/newlines around commas, keep commas as separators
    s = s.replace("\n", " ").strip()
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if not parts:
        return None
    return _canon_join(parts)

def _canon_rush_generic(x: Any) -> Optional[str]:
    """Canonicalize either a string sequence or a list[str] of tokens."""
    if x is None:
        return None
    if isinstance(x, str):
        return _canon_rush_string(x)
    if isinstance(x, list):
        # Flat list of tokens
        if all(isinstance(t, str) for t in x):
            return _canon_join(x)
        # List of alternatives: ignore here (handled by _canon_rush_gold)
        return None
    return None

def _is_valid_rush(s: Optional[str]) -> bool:
    """Valid iff canonicalization succeeds."""
    return s is not None

def _canon_rush_gold(gold: Any) -> "set[str]":
    """
    Return a set of canonical, valid gold answers.
    Accepts:
      - str: 'Bv2,A>1'
      - list[str]: ['Bv2','A>1']
      - list[list[str] | str]: multiple valid alternatives
    """
    out: set[str] = set()
    if gold is None:
        return out
    if isinstance(gold, str) or (isinstance(gold, list) and all(isinstance(t, str) for t in gold)):
        s = _canon_rush_generic(gold)
        if _is_valid_rush(s):
            out.add(s)  # single canonical sequence
        return out
    if isinstance(gold, list):
        for alt in gold:
            s = _canon_rush_generic(alt)
            if _is_valid_rush(s):
                out.add(s)
        return out
    return out

# ───────────────────────── Stopping on substrings ─────────────────────────
class StopOnSubstrings(StoppingCriteria):
    def __init__(self, tokenizer: AutoTokenizer, stops: List[str]):
        self.stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stops]
    @staticmethod
    def _endswith(a: torch.Tensor, b: List[int]) -> bool:
        return len(a) >= len(b) and a[-len(b):].tolist() == b
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        for row in input_ids:
            for s in self.stop_ids:
                if s and self._endswith(row, s):
                    return True
        return False

# ───────────────────────── Logging ─────────────────────────
logging.basicConfig(
    level=getattr(logging, os.getenv("LOGLEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)
logger.info("Starting %s", os.path.basename(__file__))

# ───────────────────────── Chat builders ─────────────────────────
def _ensure_messages(obj: Union[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """Dataset may store messages as JSON string or as a Python list."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    raise ValueError("messages field is neither list nor JSON-encoded list")

def chat_base_for_pass1_from_messages(tokenizer, messages: List[Dict[str, str]]) -> str:
    msgs = list(messages)
    has_sys = any(m.get("role") == "system" for m in msgs)
    if not has_sys:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def chat_base_for_pass2_from_messages(tokenizer, messages: List[Dict[str, str]], prev_output: str, cue: str) -> str:
    msgs = list(messages)
    has_sys = any(m.get("role") == "system" for m in msgs)
    if not has_sys:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
    msgs = msgs + [
        {"role": "assistant", "content": prev_output},
        {"role": "user", "content": cue},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

# ───────────────────── Inference Loop (two-phase per pass) ─────────────────────
def run_inference_on_split(
    split_name: str,
    examples,  # datasets.Dataset
    tokenizer,
    model,
    step: int,
    outdir: str,
    prompt_col: str = "messages",
    solution_col: str = "solution",
    batch_size: int = 8,
    num_samples: int = 1,
    temperature: float = 0.0,
    top_p: float = 0.95,
    entropy_mode: str = "reconsider",
    eos_ids: Optional[List[int]] = None,
    two_pass: bool = False,
    second_pass_phrase: str = "Wait, we need to reconsider. Let's think this through step by step.",
    second_pass_use_sample_idx: int = 0,
    think_cap: int = 750,
    answer_cap: int = 50,
):
    def _gen_batch(prefixes: List[str], cap: int, stop_strs: List[str]) -> Tuple[
        List[str], List[List[float]], torch.Tensor, torch.Tensor, List[str]
    ]:
        inputs = tokenizer(prefixes, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        input_lengths = inputs["attention_mask"].sum(dim=1)
        if torch.cuda.is_available():
            for k in inputs:
                inputs[k] = inputs[k].to("cuda")
            input_lengths = input_lengths.to(inputs["input_ids"].device)

        stop = StoppingCriteriaList([StopOnSubstrings(tokenizer, stop_strs)]) if stop_strs else None

        # ✅ Temperature=0 guard: NEVER sample when temp <= 0.0
        temp_val = float(temperature) if temperature is not None else 0.0
        use_sampling = temp_val > 0.0  # sample iff temperature > 0; otherwise greedy

        gen_kwargs = dict(
            max_new_tokens=cap,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=eos_ids,
            return_dict_in_generate=True,
            output_scores=(entropy_mode != "none"),
            num_return_sequences=1,
        )
        if use_sampling:
            gen_kwargs.update(dict(
                do_sample=True,
                temperature=temp_val,
                top_p=float(top_p),
            ))
        else:
            gen_kwargs.update(dict(do_sample=False))

        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs, stopping_criteria=stop)

        total_rows = out.sequences.shape[0]
        seqs = out.sequences
        decs: List[str] = []
        ent_series: List[List[float]] = []
        stop_reasons: List[str] = []

        for row_i in range(total_rows):
            start_tok_idx = int(input_lengths[row_i].item())
            gen_ids = seqs[row_i, start_tok_idx:]
            raw_txt = tokenizer.decode(gen_ids, skip_special_tokens=True)

            found_stop = any(s in raw_txt for s in (stop_strs or []))
            has_eos = bool(eos_ids and any((gen_ids == eid).any() for eid in eos_ids))
            hit_max = len(gen_ids) >= cap

            stop_reasons.append(
                "stop_token" if found_stop else
                "eos" if has_eos else
                "max_new_tokens" if hit_max else
                "other"
            )

            txt = raw_txt
            for s in (stop_strs or []):
                if s in txt:
                    txt = txt.split(s, 1)[0]
                    break
            decs.append(txt.strip())

            if entropy_mode == "none":
                ent_series.append([])
                continue

            scores_T = len(out.scores)
            t_stop = min(gen_ids.shape[0], scores_T)
            tok_ents = []
            bad = False
            for t in range(t_stop):
                logits = out.scores[t][row_i].float()
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    bad = True; break
                logp = F.log_softmax(logits, dim=-1)
                if torch.isnan(logp).any() or torch.isinf(logp).any():
                    bad = True; break
                p = logp.exp()
                h = float(-(p * logp).sum().item())
                if not math.isfinite(h):
                    bad = True; break
                tok_ents.append(h)

            if bad or len(tok_ents) == 0:
                from math import isfinite
                tok_ents = []
                with torch.inference_mode():
                    seq = seqs[row_i:row_i+1]
                    out2 = model(input_ids=seq[:, :start_tok_idx+1], use_cache=True)
                    past = out2.past_key_values
                    L = seq.shape[1]
                    for t in range(start_tok_idx, L-1):
                        o = model(input_ids=seq[:, t:t+1], past_key_values=past, use_cache=True)
                        past = o.past_key_values
                        logits = o.logits[:, -1, :].float()
                        logp = F.log_softmax(logits, dim=-1)
                        p = logp.exp()
                        h = float(-(p * logp).sum().item())
                        if not isfinite(h):
                            break
                        tok_ents.append(h)

            ent_series.append(tok_ents)

        return decs, ent_series, input_lengths, seqs, stop_reasons

    def _repeat_for_samples(xs: List[str], S: int) -> List[str]:
        return [x for x in xs for _ in range(S)]

    def _norm_fields(ex: dict, prompt_col: str, sol_col: str):
        msgs = ex.get(prompt_col)
        sol  = ex.get(sol_col)
        try:
            msgs = _ensure_messages(msgs)
        except Exception:
            problem = ex.get("problem") or ex.get("board") or ex.get("prompt") or ""
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(problem)},
            ]
        return msgs, sol

    def _pack_pass_result(
        problem_text: str,
        full_text: str,
        ent_think: List[float],
        ent_answer: List[float],
        injected_cue: bool,
        prev_output: Optional[str] = None,
        cue_prefix_str: str = "",
        stop_reason_think: Optional[str] = None,
        stop_reason_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        tok_ents_all = (ent_think or []) + (ent_answer or [])
        Tthink = len(ent_think or [])
        Tans   = len(ent_answer or [])

        think, answer = _extract_blocks(full_text)
        think_text = think or ""
        pred_answer_text = (answer or "").strip()

        # Reconsideration markers (ignore injected cue at prefix)
        skip_chars = len(cue_prefix_str) if injected_cue else 0
        markers = []
        if think_text:
            search_text = think_text[skip_chars:] if skip_chars > 0 else think_text
            for name, pat in _RECONSIDER_PATTERNS:
                if pat.search(search_text):
                    markers.append(name)
                    break
        if injected_cue:
            markers = ["injected_cue"] + markers

        entropy_overall = _finite_mean(tok_ents_all) if tok_ents_all else None
        entropy_think   = _finite_mean(ent_think)     if ent_think else None
        entropy_answer  = _finite_mean(ent_answer)    if ent_answer else None

        pred_canon = _canon_rush_generic(pred_answer_text)
        is_valid = _is_valid_rush(pred_canon)

        return dict(
            prev_output=prev_output,
            output=full_text,
            pred_answer=pred_answer_text,
            pred_answer_canon=pred_canon,
            entropy=entropy_overall,
            entropy_think=entropy_think,
            entropy_answer=entropy_answer,
            stop_reason_think=stop_reason_think,
            stop_reason_answer=stop_reason_answer,
            has_reconsider_cue=bool(markers),
            reconsider_markers=markers,
            is_valid_pred=is_valid,
            is_correct_pred=False,  # set after gold comparison
            tokens_total=len(tok_ents_all),
            tokens_end_think=Tthink,
            tokens_think=Tthink,
            tokens_answer=Tans,
            valid_tag_structure=_valid_tag_structure(full_text),
        )

    outpath = os.path.join(outdir, f"step{step:04d}_{split_name}.jsonl")

    # [FILL] Read existing rows -> which sample_idx we already have per example_id
    existing_by_example: Dict[str, set[int]] = defaultdict(set)
    if os.path.exists(outpath):
        with open(outpath, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    exid = rec.get("example_id")
                    k = rec.get("sample_idx")
                    if exid is not None and isinstance(k, int):
                        existing_by_example[exid].add(k)
                except Exception:
                    pass

    logger.info("→ %s | %d examples | resume: loaded %d existing example IDs",
                split_name, len(examples), len(existing_by_example))
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    BATCH = batch_size
    S = int(num_samples)  # target samples per example
    idx_global = 0

    for i in range(0, len(examples), BATCH):
        idx_lo, idx_hi = i, min(i + BATCH, len(examples))
        batch_ds = examples.select(range(idx_lo, idx_hi))

        # Build a batch of examples that are NOT fully complete, carrying which ks are missing
        batch = []
        for j, ex in enumerate(batch_ds):
            msgs, sol = _norm_fields(ex, prompt_col, solution_col)
            ex_id = str(ex.get("id", f"idx_{idx_lo + j}"))  # stable fallback ID
            have = existing_by_example.get(ex_id, set())
            missing_k = sorted([k for k in range(S) if k not in have])
            if not missing_k:
                continue  # this example already has all S samples
            batch.append({"id": ex_id, "messages": msgs, "solution": sol, "missing_k": missing_k})

        if not batch:
            continue  # this chunk is fully complete

        B = len(batch)

        # ===== PASS 1 =====
        base1 = [chat_base_for_pass1_from_messages(tokenizer, ex["messages"]) for ex in batch]
        pre1_think = _repeat_for_samples([b + "<think>\n" for b in base1], S)
        think1_texts, think1_ents, _, _, think1_stop = _gen_batch(pre1_think, think_cap, ["</think>"])

        pre1_answer = []
        for row_i in range(B * S):
            pre = pre1_think[row_i] + think1_texts[row_i] + "</think>\n<answer>\n"
            pre1_answer.append(pre)
        answer1_texts, answer1_ents, _, _, answer1_stop = _gen_batch(pre1_answer, answer_cap, ["</answer>"])

        pass1_full = [
            f"<think>{think1_texts[row_i]}</think>\n<answer>{answer1_texts[row_i]}</answer>"
            for row_i in range(B * S)
        ]

        # Which sample feeds pass-2
        firstpass_choice = []
        for bi in range(B):
            k_choice = max(0, min(second_pass_use_sample_idx, S - 1))
            row_i = bi * S + k_choice
            firstpass_choice.append(pass1_full[row_i])

        # ===== PASS 2 (optional) =====
        pass2_full = [""] * (B * S)
        think2_ents: List[List[float]] = [[] for _ in range(B * S)]
        answer2_ents: List[List[float]] = [[] for _ in range(B * S)]
        think2_stop: List[str] = [""] * (B * S)
        answer2_stop: List[str] = [""] * (B * S)

        cue_str = second_pass_phrase.strip() + " "

        if two_pass:
            base2 = [
                chat_base_for_pass2_from_messages(
                    tokenizer,
                    ex["messages"],
                    firstpass_choice[bi],
                    second_pass_phrase.strip(),
                )
                for bi, ex in enumerate(batch)
            ]
            pre2_think = _repeat_for_samples([b + "<think>\n" + cue_str for b in base2], S)
            think2_texts_only_new, think2_ents, _, _, think2_stop = _gen_batch(pre2_think, think_cap, ["</think>"])
            think2_texts = [cue_str + t for t in think2_texts_only_new]

            pre2_answer = []
            for row_i in range(B * S):
                pre = pre2_think[row_i] + think2_texts_only_new[row_i] + "</think>\n<answer>\n"
                pre2_answer.append(pre)
            answer2_texts, answer2_ents, _, _, answer2_stop = _gen_batch(pre2_answer, answer_cap, ["</answer>"])

            pass2_full = [
                f"<think>{think2_texts[row_i]}</think>\n<answer>{answer2_texts[row_i]}</answer>"
                for row_i in range(B * S)
            ]

        # ===== WRITE JSON (only missing ks) =====
        with open(outpath, "a", encoding="utf-8") as f:
            for bi, ex in enumerate(batch):
                gold_set = _canon_rush_gold(ex["solution"])
                gold_list_sorted = sorted(list(gold_set))

                # Only write rows for sample_idx we are missing for this example
                for k in ex["missing_k"]:
                    row_i = bi * S + k

                    p1 = _pack_pass_result(
                        problem_text=str(ex["messages"]),
                        full_text=pass1_full[row_i],
                        ent_think=think1_ents[row_i],
                        ent_answer=answer1_ents[row_i],
                        injected_cue=False,
                        prev_output=None,
                        cue_prefix_str="",
                        stop_reason_think=think1_stop[row_i],
                        stop_reason_answer=answer1_stop[row_i],
                    )
                    pred1 = p1.get("pred_answer_canon")
                    p1["is_correct_pred"] = bool(pred1 and pred1 in gold_set)
                    p1_reward, p1_breakdown = rush_soft_match_reward(p1["pred_answer"], ex["solution"])
                    p1["soft_reward"] = p1_reward
                    p1["soft_reward_detail"] = p1_breakdown

                    p2 = None
                    if two_pass:
                        p2 = _pack_pass_result(
                            problem_text=str(ex["messages"]),
                            full_text=pass2_full[row_i],
                            ent_think=think2_ents[row_i],
                            ent_answer=answer2_ents[row_i],
                            injected_cue=True,
                            prev_output=firstpass_choice[bi],
                            cue_prefix_str=cue_str,
                            stop_reason_think=think2_stop[row_i],
                            stop_reason_answer=answer2_stop[row_i],
                        )
                        pred2 = p2.get("pred_answer_canon")
                        p2["is_correct_pred"] = bool(pred2 and pred2 in gold_set)
                        p2_reward, p2_breakdown = rush_soft_match_reward(p2["pred_answer"], ex["solution"])
                        p2["soft_reward"] = p2_reward
                        p2["soft_reward_detail"] = p2_breakdown
                        p2["improved_over_pass1"] = bool(p2["is_correct_pred"]) and not bool(p1["is_correct_pred"])

                    row = {
                        "example_id": ex["id"],
                        "gold_answer": ex["solution"],
                        "gold_answer_canon_set": gold_list_sorted,
                        "step": step,
                        "split": split_name,
                        "sample_idx": k,
                        "pass1": p1,
                        "pass2": p2,
                    }
                    json.dump(row, f, ensure_ascii=False)
                    f.write("\n")

                    # [FILL] mark this k as done so later batches won't try again
                    existing_by_example[ex["id"]].add(k)

                idx_global += 1

# ─────────────────────────── Dataset loader ───────────────────────────
def load_rush_dataset(
    dataset_id: str,
    split: str,
    cache_dir: str,
    prompt_col: str = "messages",
    solution_col: str = "solution",
):
    from datasets import load_dataset
    ds = load_dataset(dataset_id, split=split, cache_dir=cache_dir)
    cols = set(ds.column_names)
    if prompt_col not in cols or solution_col not in cols:
        raise ValueError(f"Dataset missing required columns: {prompt_col}, {solution_col}. Found: {sorted(cols)}")
    return ds

# ─────────────────────────── Main ───────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--output_dir", required=True)

    # Data
    ap.add_argument("--dataset_id", default="od2961/rush4-5-6-balanced")
    ap.add_argument("--dataset_prompt_column", default="messages")
    ap.add_argument("--dataset_solution_column", default="solution")
    ap.add_argument("--split", default="test")
    ap.add_argument("--num_examples", type=int, default=None)

    # Decoding
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.95)

    # Budgets
    ap.add_argument("--think_cap", type=int, default=750)
    ap.add_argument("--answer_cap", type=int, default=50)

    # System/runtime
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--tokenizer_path", default=None)
    ap.add_argument("--seed", type=int, default=42)

    # Entropy + attention impl
    ap.add_argument("--entropy_mode", choices=["full","reconsider","none"], default="reconsider")
    ap.add_argument("--attn_implementation", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])

    # Two-pass controls
    ap.add_argument("--two_pass", action="store_true")
    ap.add_argument("--second_pass_phrase", default="Wait, we need to reconsider. Let's think this through step by step.")
    ap.add_argument("--second_pass_use_sample_idx", type=int, default=0)

    args = ap.parse_args()

    # Tokenizer
    HF_CACHE_DIR = os.path.abspath("./.hf_cache")
    tok_src = args.tokenizer_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tok_src,
        revision=args.revision,
        trust_remote_code=True,
        cache_dir=HF_CACHE_DIR,
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.truncation_side = "left"

    # EOS IDs
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.pad_token_id:
            eos_ids.add(int(tid))
    eos_ids = sorted(eos_ids) if eos_ids else None

    # Model
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        revision=args.revision,
        trust_remote_code=True,
        cache_dir=HF_CACHE_DIR,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation=args.attn_implementation,
    ).eval()

    # Dataset
    ds = load_rush_dataset(
        dataset_id=args.dataset_id,
        split=args.split,
        cache_dir=HF_CACHE_DIR,
        prompt_col=args.dataset_prompt_column,
        solution_col=args.dataset_solution_column,
    )
    if args.num_examples is not None and args.num_examples > 0:
        ds = ds.select(range(min(args.num_examples, len(ds))))

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Model: %s @ %s | dtype=%s", args.model_name_or_path, args.revision, dtype)
    logger.info("Dataset: %s split=%s | N=%d", args.dataset_id, args.split, len(ds))
    logger.info("Output dir: %s", args.output_dir)

    run_inference_on_split(
        split_name=args.split,
        examples=ds,
        tokenizer=tokenizer,
        model=model,
        step=args.step,
        outdir=args.output_dir,
        prompt_col=args.dataset_prompt_column,
        solution_col=args.dataset_solution_column,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        entropy_mode=args.entropy_mode,
        eos_ids=eos_ids,
        two_pass=args.two_pass,
        second_pass_phrase=args.second_pass_phrase,
        second_pass_use_sample_idx=args.second_pass_use_sample_idx,
        think_cap=args.think_cap,
        answer_cap=args.answer_cap,
    )

    logger.info("All inference complete.")

if __name__ == "__main__":
    main()
