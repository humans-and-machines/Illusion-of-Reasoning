#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, math, sys, logging, argparse
from typing import Optional, List, Tuple, Dict, Any

import torch
from torch.nn import functional as F
from packaging import version
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

# ───────────────────────── System prompt (CROSSWORD) ─────────────────────────
SYSTEM_PROMPT = """  You are an expert *cryptic-crossword solver*.

  Do this (repeat until fully consistent):

  A) DEVICE TRIAGE
    • List plausible devices from {anagram, container, reversal, hidden, charade,
      deletion, homophone, double def, &lit, letter selection, substitution, abbr}.
    • For each, quote the indicator word(s). Reject with a reason.

  B) PARSE
    • Mark the **definition** (start or end).
    • Mark the **wordplay** with exact fodder + operations.

  C) CHECKS
    • Enumeration must match exactly.
    • Letter accounting must be exact (anagram multiset or stepwise build).

  D) DECIDE
    • Pick the candidate best matching definition, indicator(s), and enumeration.
    • Do NOT assume anagram without a clear anagrind and fully used fodder.

  E) RECONSIDER (if any check fails)
    • Begin the next <think> with: "Wait, we need to reconsider. Let's think this through step by step."
    • Say why it failed, then re-run A–D with an alternative device/parse.

  FORMAT (no deviations):
    • Reasoning only in <think>…</think>
    • Final entry ONLY (UPPER-CASE) in <answer>…</answer>
  ------------------------------------------------------------
  HIDDEN
  Clue: Close, as seen in plaNET EARly (4)
  <think>Device: HIDDEN; indicator “as seen in”.
  Def: “Close”. Wordplay: hidden in “plaNET EARly” → NEAR.
  Enumeration: (4) OK.</think>
  <answer> NEAR </answer>

  Clue: Mix TEA for a hot drink (3)
  <think>Device: ANAGRAM; indicator “Mix”. Fodder TEA → TEA.
  Def: “a hot drink”. Accounting exact; (3) OK.</think>
  <answer> TEA </answer>

  Clue: Pet, when back, is a god (3)
  <think>Device: REVERSAL; indicator “when back”.
  Wordplay: GOD ← → DOG. Def: “Pet”. (3) OK.</think>
  <answer> DOG </answer>

  Clue: Animal, by the sound of “dear” (4)
  <think>Device triage: {homophone ✓ (“by the sound of”), hidden ✗, anagram ✗, …}
  Def: “Animal”. Wordplay: “dear” (sounds like) → DEER. Enumeration (4) OK.</think>
  <answer>DEER</answer>

  Clue: Shoe liner at home on fish (6)
  <think>Device triage: {hidden ? (“on” is not a hidden indicator), anagram ✗ (no anagrind),
  charade ✓ (“at home”=IN, “on”=next to), homophone ✗, …}
  Attempt (HIDDEN) rejected: no indicator; also hidden spans don’t give (6).
  Candidate attempt (wrong path): — fails enumeration/indicator, so we must rethink. 
  Re-evaluate as CHARADES: IN (“at home”) + SOLE (“fish”) → INSOLE.
  Accounting: INSOLE letters: I N S O L E (6). Definition “Shoe liner” fits. Enumeration (6) OK.</think>
  <answer>INSOLE</answer>
"""

# ───────────────────────── Regex helpers ─────────────────────────
RE_THINK  = re.compile(r"(?si)<think>(.*?)</think>")
RE_ANSWER = re.compile(r"(?si)<answer>(.*?)</answer>")

# Optional cue detectors you already use
_RECONSIDER_PATTERNS = [
    ("wait_line",        re.compile(r"(?im)^\s*wait[,\.\-–—… ]", re.I)),
    ("wait_reconsider",  re.compile(r"\bwait\b.*\breconsider\b", re.I | re.S)),
    ("reconsider_exact", re.compile(r"\bwait[,!\.\s]*let me reconsider\b", re.I)),
    ("step_by_step",     re.compile(r"\blet'?s take (this|it) step[-\s]?by[-\s]?step\b", re.I)),
    ("step_by_step_alt", re.compile(r"\bstep[-\s]?by[-\s]?step\b", re.I)),
    ("recheck",          re.compile(r"\bre[-\s]?check(ing)?\b", re.I)),
]

# ───────────────────────── Utilities ─────────────────────────
def _finite_mean(vals: List[float]) -> Optional[float]:
    vs = [float(v) for v in vals if v == v and math.isfinite(float(v))]
    return (sum(vs) / len(vs)) if vs else None

def _first_eos_any(token_ids: torch.Tensor, eos_id_list: Optional[List[int]]) -> int:
    if not eos_id_list:
        return token_ids.numel()
    hits = []
    for eid in eos_id_list:
        pos = (token_ids == eid).nonzero(as_tuple=False)
        if pos.numel() > 0:
            hits.append(pos[0].item())
    return min(hits) if hits else token_ids.numel()

def _entropy_from_start_index(model, seq_ids: torch.Tensor, start_idx: int) -> List[float]:
    device = next(model.parameters()).device
    seq_ids = seq_ids.to(device)
    ents: List[float] = []
    with torch.inference_mode():
        out = model(input_ids=seq_ids[:, :start_idx+1], use_cache=True)
        past = out.past_key_values
        L = seq_ids.shape[1]
        for t in range(start_idx, L-1):
            out = model(input_ids=seq_ids[:, t:t+1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()
            logp = torch.nn.functional.log_softmax(logits, dim=-1)
            p = logp.exp()
            h = float(-(p * logp).sum().item())
            if not math.isfinite(h):
                logits = (logits - logits.max()).float()
                logp = torch.nn.functional.log_softmax(logits, dim=-1)
                p = logp.exp()
                h = float(-(p * logp).sum().item())
            ents.append(h)
    return ents


def _extract_blocks(txt: str) -> Tuple[Optional[str], Optional[str]]:
    think = ans = None
    m = RE_THINK.search(txt)
    if m: think = m.group(1).strip()
    m = RE_ANSWER.search(txt)
    if m: ans = m.group(1).strip()
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

# Crossword-friendly canon: casefold; strip spaces, hyphens, punctuation.
RE_PUNCT = re.compile(r"[^a-z0-9]", re.I)
def _canon_cross(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = x.strip().lower()
    # normalize common punctuation/spacing/hyphens
    s = s.replace("–","-").replace("—","-")
    s = RE_PUNCT.sub("", s)
    return s

def _contains_canon(hay: Optional[str], needle: Optional[str]) -> bool:
    if not hay or not needle:
        return False
    return needle in hay

def _find_markers_and_context(think_text: Optional[str], clue_text: str, skip_prefix_chars: int = 0):
    if not think_text:
        return [], None, None, None
    search_text = think_text[skip_prefix_chars:] if skip_prefix_chars > 0 else think_text
    earliest_pos = None
    markers = []
    for name, pat in _RECONSIDER_PATTERNS:
        m = pat.search(search_text)
        if m:
            markers.append(name)
            pos_global = (skip_prefix_chars + m.start()) if skip_prefix_chars > 0 else m.start()
            if earliest_pos is None or pos_global < earliest_pos:
                earliest_pos = pos_global
    if not markers:
        return [], None, None, None
    prefix = think_text[:earliest_pos] if earliest_pos is not None else think_text
    reconsider_context = f"Clue: {clue_text}\n\n{prefix}"
    lo = max(0, (earliest_pos or 0) - 60)
    hi = min(len(think_text), (earliest_pos or 0) + 60)
    reconsider_excerpt = think_text[lo:hi]
    return markers, earliest_pos, reconsider_context, reconsider_excerpt

# ───────────────────────── Stopping on substrings ─────────────────────────
class StopOnSubstrings(StoppingCriteria):
    def __init__(self, tokenizer, stops: List[str]):
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

# ───────────────────────── Logging & DS patch ─────────────────────────
logging.basicConfig(
    level=getattr(logging, os.getenv("LOGLEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)
logger.info("Starting %s", os.path.basename(__file__))

try:
    import torch
    if version.parse(torch.__version__) >= version.parse("2.6.0"):
        from torch.serialization import add_safe_globals  # type: ignore
        from deepspeed.runtime.zero.config import ZeroStageEnum  # type: ignore
        from deepspeed.runtime.fp16.loss_scaler import LossScaler  # type: ignore
        add_safe_globals([ZeroStageEnum, LossScaler])
        logger.info("DeepSpeed ZeRO patch enabled")
except Exception as e:
    logger.warning("DeepSpeed patch failed: %r", e)

# ───────────────────────── Prompt builders ─────────────────────────
def chat_base_for_pass1(tokenizer, clue: str, enumeration: Optional[str]) -> str:
    enum_text = f" ({enumeration})" if enumeration else ""
    user_text = f"Clue: {clue}{enum_text}"
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True
    )

def chat_base_for_pass2(tokenizer, clue: str, enumeration: Optional[str], prev_output: str, cue: str) -> str:
    enum_text = f" ({enumeration})" if enumeration else ""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Clue: {clue}{enum_text}"},
            {"role": "assistant", "content": prev_output},
            {"role": "user", "content": cue},
        ],
        tokenize=False, add_generation_prompt=True
    )

# ───────────────────── Inference Loop ─────────────────────
def run_inference_on_split(
    split_name: str,
    examples,  # datasets.Dataset
    tokenizer,
    model,
    step: int,
    outdir: str,
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
    import torch
    from torch.nn import functional as F

    def _make_gen_kwargs(cap: int) -> dict:
        """
        If temperature <= 0 → greedy (do_sample=False) and omit temperature/top_p.
        Else → sampling with provided temperature/top_p.
        """
        do_sample = (temperature is not None) and (float(temperature) > 0.0)
        kwargs = dict(
            max_new_tokens=cap,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=eos_ids,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=(entropy_mode != "none"),
            num_return_sequences=1,   # we replicate prompts for num_samples elsewhere
        )
        if do_sample:
            kwargs["temperature"] = float(temperature)
            if top_p is not None:
                kwargs["top_p"] = float(top_p)
        return kwargs

    def _gen_batch(prefixes: List[str], cap: int, stop_strs: List[str]):
        inputs = tokenizer(prefixes, return_tensors="pt", padding=True, truncation=True, max_length=5500)
        input_lengths = inputs["attention_mask"].sum(dim=1)
        if torch.cuda.is_available():
            for k in inputs:
                inputs[k] = inputs[k].to("cuda")
            input_lengths = input_lengths.to(inputs["input_ids"].device)

        stop = StoppingCriteriaList([StopOnSubstrings(tokenizer, stop_strs)]) if stop_strs else None
        gen_kwargs = _make_gen_kwargs(cap)

        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs, stopping_criteria=stop)

        total_rows = out.sequences.shape[0]
        seqs = out.sequences
        decs, ent_series, stop_reasons = [], [], []
        for row_i in range(total_rows):
            start_tok_idx = int(input_lengths[row_i].item())
            gen_ids = seqs[row_i, start_tok_idx:]
            raw_txt = tokenizer.decode(gen_ids, skip_special_tokens=True)

            found_stop = any(s in raw_txt for s in (stop_strs or []))
            has_eos = False
            if eos_ids:
                for eid in eos_ids:
                    if (gen_ids == eid).any():
                        has_eos = True; break
            hit_max = len(gen_ids) >= cap
            stop_reasons.append("stop_token" if found_stop else "eos" if has_eos else "max_new_tokens" if hit_max else "other")

            txt = raw_txt
            for s in (stop_strs or []):
                if s in txt:
                    txt = txt.split(s, 1)[0]
                    break
            decs.append(txt.strip())

            if entropy_mode == "none":
                ent_series.append([]); continue

            # replace with:
            scores_T = len(out.scores)
            t_stop = min(_first_eos_any(gen_ids, eos_ids) if eos_ids else gen_ids.shape[0], scores_T)
            tok_ents, bad = [], False
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
                start_idx = start_tok_idx - 1
                tok_ents = _entropy_from_start_index(model, seqs[row_i:row_i+1], start_idx) or []

            ent_series.append(tok_ents)


        return decs, ent_series, input_lengths, seqs, stop_reasons

    def _repeat_for_samples(xs: List[str], S: int) -> List[str]:
        return [x for x in xs for _ in range(S)]

    def _norm_fields(ex: dict):
        # Expecting JSONL like from your set: clue / answer / enumeration (optional)
        clue = (ex.get("clue") or ex.get("problem") or ex.get("question") or ex.get("prompt"))
        gold = (ex.get("answer") or ex.get("target") or ex.get("solution"))
        enum = (ex.get("enumeration") or ex.get("enum") or ex.get("lengths"))
        if isinstance(enum, (list, tuple)):
            enum = " ".join(str(x) for x in enum)
        return clue, gold, enum

    def _pack_pass_result(
        clue: str,
        enumeration: Optional[str],
        full_text: str,
        ent_think: List[float],
        ent_answer: List[float],
        injected_cue: bool,
        canon_gold: Optional[str],
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
        pred_answer_text = answer or ""

        skip_chars = len(cue_prefix_str) if injected_cue else 0
        markers, pos_in_think, reconsider_context, reconsider_excerpt = _find_markers_and_context(
            think_text, clue, skip_prefix_chars=skip_chars
        )
        if injected_cue:
            markers = ["injected_cue"] + (markers or [])

        entropy_overall = _finite_mean(tok_ents_all) if tok_ents_all else None
        entropy_think   = _finite_mean(ent_think)     if ent_think else None
        entropy_answer  = _finite_mean(ent_answer)    if ent_answer else None

        pred_canon = _canon_cross(pred_answer_text)
        is_correct_pred = _contains_canon(pred_canon, canon_gold)

        return dict(
            enumeration=enumeration,
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
            reconsider_markers=markers or [],
            reconsider_pos=pos_in_think,
            reconsider_context=reconsider_context,
            reconsider_excerpt=reconsider_excerpt,

            is_correct_pred=is_correct_pred,
            is_correct_after_reconsideration=bool(markers) and bool(is_correct_pred),

            tokens_total=len(tok_ents_all),
            tokens_end_think=Tthink,
            tokens_think=Tthink,
            tokens_answer=Tans,

            valid_tag_structure=_valid_tag_structure(full_text),
        )

    # ---------- resume ----------
    outpath = os.path.join(outdir, f"step{step:04d}_{split_name}.jsonl")
    seen: set[str] = set()
    if os.path.exists(outpath):
        with open(outpath, encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["problem"])
                except Exception:
                    pass

    logger.info("→ %s | %d examples (skipping %d already done)", split_name, len(examples), len(seen))

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    BATCH = batch_size

    for i in range(0, len(examples), BATCH):
        idx_lo, idx_hi = i, min(i + BATCH, len(examples))
        batch_ds = examples.select(range(idx_lo, idx_hi))

        batch = []
        for ex in batch_ds:
            clue, gold, enum = _norm_fields(ex)
            if not clue or clue in seen:
                continue
            ex = dict(ex)
            ex["_clue"] = clue
            ex["_gold"] = gold
            ex["_enum"] = enum
            batch.append(ex)
        if not batch:
            continue

        B, S = len(batch), num_samples

        # ===== PASS 1 =====
        base1 = [chat_base_for_pass1(tokenizer, ex["_clue"], ex["_enum"]) for ex in batch]
        pre1_think = _repeat_for_samples([b + "<think>\n" for b in base1], S)
        think1_texts, think1_ents, _, _, think1_stop = _gen_batch(pre1_think, think_cap, ["</think>"])

        pre1_answer = []
        for row_i in range(B * S):
            pre = pre1_think[row_i] + think1_texts[row_i] + "</think>\n<answer>\n"
            pre1_answer.append(pre)
        answer1_texts, answer1_ents, _, _, answer1_stop = _gen_batch(pre1_answer, answer_cap, ["</answer>"])
        pass1_full = [f"<think>{think1_texts[r]}</think>\n<answer>{answer1_texts[r]}</answer>" for r in range(B*S)]

        # pick sample for pass-2 context
        firstpass_choice = []
        for bi in range(B):
            k_choice = max(0, min(second_pass_use_sample_idx, S - 1))
            row_i = bi * S + k_choice
            firstpass_choice.append(pass1_full[row_i])

        # ===== PASS 2 =====
        pass2_full = [""] * (B * S)
        think2_ents = [[] for _ in range(B * S)]
        answer2_ents = [[] for _ in range(B * S)]
        think2_stop = [""] * (B * S)
        answer2_stop = [""] * (B * S)
        cue_str = second_pass_phrase.strip() + " "

        if two_pass:
            base2 = [
                chat_base_for_pass2(tokenizer, ex["_clue"], ex["_enum"], firstpass_choice[bi], second_pass_phrase.strip())
                for bi, ex in enumerate(batch)
            ]
            pre2_think = _repeat_for_samples([b + "<think>\n" + cue_str for b in base2], S)
            think2_texts_only, think2_ents, _, _, think2_stop = _gen_batch(pre2_think, think_cap, ["</think>"])
            think2_texts = [cue_str + t for t in think2_texts_only]

            pre2_answer = []
            for row_i in range(B * S):
                pre = pre2_think[row_i] + think2_texts_only[row_i] + "</think>\n<answer>\n"
                pre2_answer.append(pre)
            answer2_texts, answer2_ents, _, _, answer2_stop = _gen_batch(pre2_answer, answer_cap, ["</answer>"])
            pass2_full = [f"<think>{think2_texts[r]}</think>\n<answer>{answer2_texts[r]}</answer>" for r in range(B*S)]

        # ===== WRITE JSON =====
        for bi, ex in enumerate(batch):
            canon_gold = _canon_cross(ex["_gold"])
            for k in range(S):
                row_i = bi * S + k
                p1 = _pack_pass_result(
                    clue=ex["_clue"], enumeration=ex["_enum"], full_text=pass1_full[row_i],
                    ent_think=think1_ents[row_i], ent_answer=answer1_ents[row_i],
                    injected_cue=False, canon_gold=canon_gold, prev_output=None, cue_prefix_str="",
                    stop_reason_think=think1_stop[row_i], stop_reason_answer=answer1_stop[row_i],
                )
                p2 = None
                if two_pass:
                    p2 = _pack_pass_result(
                        clue=ex["_clue"], enumeration=ex["_enum"], full_text=pass2_full[row_i],
                        ent_think=think2_ents[row_i], ent_answer=answer2_ents[row_i],
                        injected_cue=True, canon_gold=canon_gold, prev_output=firstpass_choice[bi],
                        cue_prefix_str=cue_str, stop_reason_think=think2_stop[row_i], stop_reason_answer=answer2_stop[row_i],
                    )
                    p2["improved_over_pass1"] = bool(p2.get("is_correct_pred")) and not bool(p1.get("is_correct_pred"))

                row = {
                    "problem": ex["_clue"],                 # keep field name for downstream
                    "gold_answer": ex["_gold"],
                    "gold_answer_canon": canon_gold,
                    "enumeration": ex["_enum"],
                    "step": step,
                    "split": split_name,
                    "sample_idx": k,
                    "pass1": p1,
                    "pass2": p2,
                }
                with open(outpath, "a", encoding="utf-8") as f:
                    json.dump(row, f, ensure_ascii=False)
                    f.write("\n")
            seen.add(ex["_clue"])

# ───────────────────────── Dataset helpers ─────────────────────────
def _load_local_json_dataset(path: str):
    from datasets import Dataset
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(obj)
    return Dataset.from_list(records)

def load_crossword_local(dataset_path: str):
    logger.info("Loading CROSSWORD JSONL from: %s", dataset_path)
    return _load_local_json_dataset(dataset_path)

# ───────────────────────── Main ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--output_dir", required=True)

    # Data
    ap.add_argument("--dataset_id", default="CROSSWORD-LOCAL",
                    help="Use 'CROSSWORD-LOCAL' for local JSONL, or a HF path if you have one.")
    ap.add_argument("--dataset_path", type=str, required=False,
                    help="Path to local JSONL with fields: clue, answer, enumeration (optional).")
    ap.add_argument("--split", default="test")
    ap.add_argument("--num_examples", type=int, default=None)

    # Decoding + sampling
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.95)

    # Budgets
    ap.add_argument("--think_cap", type=int, default=750)
    ap.add_argument("--answer_cap", type=int, default=50)

    # System/runtime
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--tokenizer_path", default=None)
    ap.add_argument("--seed", type=int, default=42)

    # Entropy + attention
    ap.add_argument("--entropy_mode", choices=["full","reconsider","none"], default="reconsider")
    ap.add_argument("--attn_implementation", default="sdpa", choices=["sdpa","eager","flash_attention_2"])

    # Two-pass
    ap.add_argument("--two_pass", action="store_true")
    ap.add_argument("--second_pass_phrase", default="Wait, we need to reconsider. Let's think this through step by step.")
    ap.add_argument("--second_pass_use_sample_idx", type=int, default=0)

    args = ap.parse_args()

    HF_CACHE_DIR = os.path.abspath("./.hf_cache")
    tok_src = args.tokenizer_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_src, revision=args.revision, trust_remote_code=True, cache_dir=HF_CACHE_DIR)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.truncation_side = "left"

    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.pad_token_id:
            eos_ids.add(int(tid))
    eos_ids = sorted(eos_ids) if eos_ids else None

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
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
    if args.dataset_id.upper() == "CROSSWORD-LOCAL":
        if not args.dataset_path:
            raise ValueError("--dataset_path is required when dataset_id=CROSSWORD-LOCAL")
        ds = load_crossword_local(args.dataset_path)
        dataset_name_for_log = f"CROSSWORD-LOCAL:{os.path.basename(args.dataset_path)}"
    else:
        from datasets import load_dataset
        ds = load_dataset(args.dataset_id, split=args.split, cache_dir=HF_CACHE_DIR)
        dataset_name_for_log = args.dataset_id

    if args.num_examples is not None and args.num_examples > 0:
        ds = ds.select(range(min(args.num_examples, len(ds))))

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Model: %s @ %s | dtype=%s", args.model_name_or_path, args.revision, dtype)
    logger.info("Dataset: %s split=%s | N=%d", dataset_name_for_log, args.split, len(ds))
    logger.info("Output dir: %s", args.output_dir)

    run_inference_on_split(
        split_name=args.split,
        examples=ds,
        tokenizer=tokenizer,
        model=model,
        step=args.step,
        outdir=args.output_dir,
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
    logging.getLogger(__name__).info("All inference complete.")

if __name__ == "__main__":
    main()
