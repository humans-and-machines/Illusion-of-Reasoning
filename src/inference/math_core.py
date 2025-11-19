#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-pass batch inference for Qwen2.5-style chat LMs that produce:
  <think> ... </think><answer> ... </answer>

Now with resume/fill:
- If a results file exists (step####_<split>.jsonl), the script checks how many
  samples exist per problem and ONLY generates the missing ones up to
  --num_samples (e.g., 8). It preserves prior rows and appends the new ones.

Other features (unchanged)
--------------------------
- Uses your system prompt for BOTH passes (first message in the chat).
- Two-phase generation per pass (think → answer) with explicit stops.
- Per-pass hard caps: think <= 750, answer <= 50 new tokens.
- Pass 2: shows Pass-1 output as an assistant turn, user supplies a cue,
  and the new <think> starts with the cue (we prefill the cue inside <think>).
- Correctness: gold counted correct if it appears ANYWHERE inside <answer>
  after canonicalization (substring check).
- Cue-robust analytics: injected cue at the start of pass-2 <think> does NOT
  trigger “reconsider” detection.
- EOS includes <|im_end|> and <|endoftext|>.
- NaN-safe token entropy; optional "reconsider" slicing.
"""

import os
import re
import json
import math
import sys
import logging
import argparse
from typing import Optional, List, Tuple, Dict, Any, DefaultDict
from collections import defaultdict

import torch
from torch.nn import functional as F
from packaging import version
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)
from src.inference.common import (
    canon_math as _canon_math,
    contains_canon as _contains_canon,
    entropy_from_start_index as _entropy_from_start_index,
    extract_blocks as _extract_blocks,
    find_markers_and_context as _find_markers_and_context,
    finite_mean as _finite_mean,
    first_eos_any as _first_eos_any,
    load_local_json_dataset,
    move_inputs_to_device,
    valid_tag_structure as _valid_tag_structure,
)

# ───────────────────────── System prompt ─────────────────────────
SYSTEM_PROMPT = """You are an expert *mathematics problem-solver*.

  Every time you receive a problem you must:
  • Analyse it thoroughly.  
    – Pinpoint the **goal** (what quantity/set/form is requested).  
    – Pinpoint the **givens/constraints** (domains, integrality, non-negativity, geometric conditions).  
    – Choose the **methods** to apply (algebraic manipulation, factorization, inequalities, counting, modular arithmetic, geometry, calculus, etc.).  
    – Write out the full derivation that leads to the final result.

  • Check that the result satisfies all original constraints (no extraneous roots, correct domain, simplified form, exact arithmetic).

  • Respond in **exactly** the tag-based format shown below – no greeting, no commentary outside the tags.  
    – The final answer goes inside `<answer>` **only**.  
    – Use **exact** math (fractions, radicals, π, e). Avoid unnecessary decimals.  
    – Canonical forms: integers as plain numbers; reduced fractions a/b with b>0; simplified radicals; rationalized denominators; sets/tuples with standard notation; intervals in standard notation.  
    – If there is **no solution**, write `NO SOLUTION`. If the problem is **underdetermined**, write `I DON'T KNOW`.

  • You have a hard cap of **750 output tokens**. Be concise but complete.

  ------------------------------------------------------------
  TAG TEMPLATE (copy this shape for every problem)
  <think>
  YOUR reasoning process goes here:  
  1. quote the relevant bits of the problem  
  2. name the mathematical tool(s) you apply  
  3. show each intermediate step until the result is reached  
     
  If you spot an error or an unmet constraint, iterate, repeating steps 1–3 as many
  times as necessary until you are confident in your result. Finish by verifying the
  result satisfies the original conditions exactly (substitution/checks).
  </think>
  <answer>
  THEANSWER
  </answer>
  """

# Optional "aha"/reconsider cue detectors (for analytics)
_RECONSIDER_PATTERNS = [
    ("wait_line",        re.compile(r"(?im)^\s*wait[,\.\-–—… ]", re.I)),
    ("wait_reconsider",  re.compile(r"\bwait\b.*\breconsider\b", re.I | re.S)),
    ("reconsider_exact", re.compile(r"\bwait[,!\.\s]*let me reconsider\b", re.I)),
    ("step_by_step",     re.compile(r"\blet'?s take (this|it) step[-\s]?by[-\s]?step\b", re.I)),
    ("step_by_step_alt", re.compile(r"\bstep[-\s]?by[-\s]?step\b", re.I)),
    ("recheck",          re.compile(r"\bre[-\s]?check(ing)?\b", re.I)),
]

# ───────────────────────── Utilities ─────────────────────────
_fmean = _finite_mean

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

# ───── PyTorch 2.6 DeepSpeed un-pickle patch (safe no-op if absent) ─────
try:
    if version.parse(torch.__version__) >= version.parse("2.6.0"):
        from torch.serialization import add_safe_globals  # type: ignore
        from deepspeed.runtime.zero.config import ZeroStageEnum  # type: ignore
        from deepspeed.runtime.fp16.loss_scaler import LossScaler  # type: ignore
        add_safe_globals([ZeroStageEnum, LossScaler])
        logger.info("DeepSpeed ZeRO patch enabled")
except Exception as e:  # noqa: BLE001
    logger.warning("DeepSpeed patch failed: %r", e)

# ───────────────────────── Prompt builders (WITH system msg) ─────────────────────────
def chat_base_for_pass1(tokenizer, problem: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem: {problem}"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

def chat_base_for_pass2(tokenizer, problem: str, prev_output: str, cue: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem: {problem}"},
            {"role": "assistant", "content": prev_output},
            {"role": "user", "content": cue},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

# ─────────────────────────── Results scanning ───────────────────────────
# >>> RESUME/FILL LOGIC <<<
def _scan_existing_results(results_path: str) -> tuple[DefaultDict[str, set], Dict[tuple, str]]:
    """
    Return:
      existing_samples: problem -> set(sample_idx) that already exist
      existing_pass1: (problem, sample_idx) -> pass1['output'] text (if available)
    """
    existing_samples: DefaultDict[str, set] = defaultdict(set)
    existing_pass1: Dict[tuple, str] = {}
    if not os.path.exists(results_path):
        return existing_samples, existing_pass1
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            prob = obj.get("problem")
            k = obj.get("sample_idx")
            if prob is None or k is None:
                continue
            existing_samples[prob].add(int(k))
            p1 = (obj.get("pass1") or {})
            p1_out = p1.get("output")
            if isinstance(p1_out, str):
                existing_pass1[(prob, int(k))] = p1_out
    return existing_samples, existing_pass1

# ───────────────────── Inference Loop (two-phase per pass) ─────────────────────
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
    """
    Respects existing results and fills missing sample indices per problem up to `num_samples`.
    """

    # warn if greedy + multiple samples
    if (temperature is None or float(temperature) == 0.0) and num_samples > 1:
        logger.warning("temperature=0 with num_samples=%d → all samples will be identical (greedy).", num_samples)

    # ---------- helper: build generation kwargs ----------
    def _make_gen_kwargs(cap: int) -> dict:
        do_sample = (temperature is not None) and (float(temperature) > 0.0)
        kwargs = dict(
            max_new_tokens=cap,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=eos_ids,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=(entropy_mode != "none"),
            num_return_sequences=1,
        )
        if do_sample:
            kwargs["temperature"] = float(temperature)
            if top_p is not None:
                kwargs["top_p"] = float(top_p)
        return kwargs

    def _gen_batch(prefixes: List[str], cap: int, stop_strs: List[str]) -> Tuple[
        List[str], List[List[float]], torch.Tensor, torch.Tensor, List[str]
    ]:
        inputs = tokenizer(prefixes, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        inputs, input_lengths = move_inputs_to_device(inputs)
        stop = StoppingCriteriaList([StopOnSubstrings(tokenizer, stop_strs)]) if stop_strs else None
        gen_kwargs = _make_gen_kwargs(cap)
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
            has_eos = False
            if eos_ids:
                for eid in eos_ids:
                    if (gen_ids == eid).any():
                        has_eos = True
                        break
            hit_max = len(gen_ids) >= cap
            if found_stop: stop_reasons.append("stop_token")
            elif has_eos:  stop_reasons.append("eos")
            elif hit_max:  stop_reasons.append("max_new_tokens")
            else:          stop_reasons.append("other")

            txt = raw_txt
            for s in (stop_strs or []):
                if s in txt:
                    txt = txt.split(s, 1)[0]
                    break
            decs.append(txt.strip())

            if entropy_mode == "none":
                ent_series.append([]); continue

            scores_T = len(out.scores)
            t_stop = min(_first_eos_any(gen_ids, eos_ids) if eos_ids else gen_ids.shape[0], scores_T)
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
                start_idx = start_tok_idx - 1
                tok_ents = _entropy_from_start_index(model, seqs[row_i:row_i+1], start_idx) or []
            ent_series.append(tok_ents)

        return decs, ent_series, input_lengths, seqs, stop_reasons

    def _norm_fields(ex: dict):
        problem = (ex.get("problem") or ex.get("question") or ex.get("query") or ex.get("prompt") or ex.get("instruction"))
        gold    = (ex.get("answer")  or ex.get("final_answer") or ex.get("target") or ex.get("boxed_answer") or ex.get("solution"))
        if gold and not any(k in ex for k in ("answer","final_answer","target","boxed_answer","solution")):
            m = re.search(r"\\boxed\{([^}]*)\}", str(gold))
            if not m:
                m = re.search(r"\\boxed\(([^)]*)\)", str(gold))
            if m:
                gold = m.group(1)
        return problem, gold

    def _pack_pass_result(
        problem: str,
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
        T      = len(tok_ents_all)
        think, answer = _extract_blocks(full_text)
        think_text = think or ""
        pred_answer_text = answer or ""
        skip_chars = len(cue_prefix_str) if injected_cue else 0
        markers, pos_in_think, reconsider_context, reconsider_excerpt = _find_markers_and_context(
            think_text,
            f"Problem: {problem}",
            _RECONSIDER_PATTERNS,
            skip_prefix_chars=skip_chars,
        )
        if injected_cue:
            markers = ["injected_cue"] + (markers or [])
        t_cue = 0 if injected_cue else None
        if (not injected_cue) and (pos_in_think is not None):
            t_cue = max(0, min(pos_in_think, Tthink))
        entropy_overall = _finite_mean(tok_ents_all) if tok_ents_all else None
        entropy_think   = _finite_mean(ent_think)     if ent_think else None
        entropy_answer  = _finite_mean(ent_answer)    if ent_answer else None
        entropy_pre_cue = None
        entropy_reconsider_think = None
        entropy_reconsider_full = None
        if t_cue is not None:
            if T > t_cue:
                entropy_reconsider_full = _finite_mean(tok_ents_all[t_cue:])
            if Tthink > t_cue:
                entropy_reconsider_think = _finite_mean(tok_ents_all[t_cue:Tthink])

        pred_canon = _canon_math(pred_answer_text)
        is_correct_pred = _contains_canon(pred_canon, canon_gold)

        return dict(
            prev_output=prev_output,
            output=full_text,
            pred_answer=pred_answer_text,
            pred_answer_canon=pred_canon,
            entropy=entropy_overall,
            entropy_think=entropy_think,
            entropy_answer=entropy_answer,
            entropy_pre_cue=entropy_pre_cue,
            entropy_reconsider_think=entropy_reconsider_think,
            entropy_reconsider_full=entropy_reconsider_full,
            stop_reason_think=stop_reason_think,
            stop_reason_answer=stop_reason_answer,
            has_reconsider_cue=bool(markers),
            reconsider_markers=markers or [],
            reconsider_pos=pos_in_think,
            reconsider_context=reconsider_context,
            reconsider_excerpt=reconsider_excerpt,
            is_correct_pred=is_correct_pred,
            is_correct_after_reconsideration=bool(markers) and bool(is_correct_pred),
            tokens_total=T,
            tokens_end_think=Tthink,
            tokens_think=Tthink,
            tokens_answer=Tans,
            valid_tag_structure=_valid_tag_structure(full_text),
        )

    # ---------- results path & scan existing ----------
    outpath = os.path.join(outdir, f"step{step:04d}_{split_name}.jsonl")
    existing_samples, existing_pass1 = _scan_existing_results(outpath)
    logger.info("Resume scan: %d problems already present", len(existing_samples))

    # ---------- main loop ----------
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    for i in range(0, len(examples), batch_size):
        idx_lo, idx_hi = i, min(i + batch_size, len(examples))
        slice_ds = examples.select(range(idx_lo, idx_hi))

        # Build a working batch of ONLY problems that still need samples
        work_items = []
        for ex in slice_ds:
            prob, gold = _norm_fields(ex)
            if not prob:
                continue
            have = existing_samples.get(prob, set())
            if len(have) >= num_samples:
                continue
            todo = [k for k in range(num_samples) if k not in have]
            if not todo:
                continue
            ex = dict(ex)
            ex["_normalized_problem"] = prob
            ex["_normalized_gold"] = gold
            ex["_todo_samples"] = todo
            work_items.append(ex)

        if not work_items:
            continue

        B = len(work_items)

        # ===== PASS 1 for missing samples =====
        base1 = [chat_base_for_pass1(tokenizer, ex["_normalized_problem"]) for ex in work_items]

        pre1_think: List[str] = []
        row_to_ex_idx: List[int] = []
        row_target_sample_idx: List[int] = []

        for ex_idx, ex in enumerate(work_items):
            for k in ex["_todo_samples"]:
                pre1_think.append(base1[ex_idx] + "<think>\n")
                row_to_ex_idx.append(ex_idx)
                row_target_sample_idx.append(k)

        think1_texts, think1_ents, _, _, think1_stop = _gen_batch(pre1_think, think_cap, ["</think>"])

        pre1_answer = []
        for row_i in range(len(pre1_think)):
            pre = pre1_think[row_i] + think1_texts[row_i] + "</think>\n<answer>\n"
            pre1_answer.append(pre)
        answer1_texts, answer1_ents, _, _, answer1_stop = _gen_batch(pre1_answer, answer_cap, ["</answer>"])

        pass1_full_rows = [
            f"<think>{think1_texts[row_i]}</think>\n<answer>{answer1_texts[row_i]}</answer>"
            for row_i in range(len(pre1_think))
        ]

        # Build per-example mapping of (newly generated) sample_idx -> pass1 text
        new_pass1_by_ex_and_k: Dict[tuple, str] = {}
        for row_i in range(len(pass1_full_rows)):
            ex_idx = row_to_ex_idx[row_i]
            k = row_target_sample_idx[row_i]
            new_pass1_by_ex_and_k[(ex_idx, k)] = pass1_full_rows[row_i]

        # ===== Decide Pass-2 "prev_output" per example =====
        cue_str = second_pass_phrase.strip() + " "
        firstpass_choice_text_per_ex: List[str] = []

        if two_pass:
            for ex_idx, ex in enumerate(work_items):
                prob = ex["_normalized_problem"]
                k_choice = max(0, min(second_pass_use_sample_idx, num_samples - 1))

                # Prefer an existing pass1 for the chosen k if available
                prev = existing_pass1.get((prob, k_choice))
                if prev is None:
                    # else prefer the pass1 we just generated for that k (if in this mini-batch)
                    prev = new_pass1_by_ex_and_k.get((ex_idx, k_choice))
                    if prev is None:
                        # else fallback: smallest existing k, else smallest new k
                        have_sorted = sorted(existing_samples.get(prob, set()))
                        if have_sorted:
                            prev = existing_pass1.get((prob, have_sorted[0]))
                        else:
                            new_ks = sorted([k for (eidx, k) in new_pass1_by_ex_and_k.keys() if eidx == ex_idx])
                            if new_ks:
                                prev = new_pass1_by_ex_and_k[(ex_idx, new_ks[0])]
                if prev is None:
                    # As an absolute fallback, use the first generated row for this ex in this batch
                    # (should be rare)
                    for row_i in range(len(pass1_full_rows)):
                        if row_to_ex_idx[row_i] == ex_idx:
                            prev = pass1_full_rows[row_i]; break
                firstpass_choice_text_per_ex.append(prev or "")
        else:
            firstpass_choice_text_per_ex = [""] * B  # unused

        # ===== PASS 2 for the SAME rows (one per missing sample) =====
        if two_pass:
            base2_per_ex = [
                chat_base_for_pass2(
                    tokenizer,
                    ex["_normalized_problem"],
                    firstpass_choice_text_per_ex[ex_idx],
                    second_pass_phrase.strip(),
                )
                for ex_idx, ex in enumerate(work_items)
            ]

            pre2_think: List[str] = []
            for row_i in range(len(pre1_think)):
                ex_idx = row_to_ex_idx[row_i]
                base2 = base2_per_ex[ex_idx]
                pre2_think.append(base2 + "<think>\n" + cue_str)

            think2_texts_only_new, think2_ents, _, _, think2_stop = _gen_batch(pre2_think, think_cap, ["</think>"])
            think2_texts = [cue_str + t for t in think2_texts_only_new]

            pre2_answer = []
            for row_i in range(len(pre2_think)):
                pre = pre2_think[row_i] + think2_texts_only_new[row_i] + "</think>\n<answer>\n"
                pre2_answer.append(pre)
            answer2_texts, answer2_ents, _, _, answer2_stop = _gen_batch(pre2_answer, answer_cap, ["</answer>"])

            pass2_full_rows = [
                f"<think>{think2_texts[row_i]}</think>\n<answer>{answer2_texts[row_i]}</answer>"
                for row_i in range(len(pre2_think))
            ]
        else:
            think2_ents = [[] for _ in range(len(pre1_think))]
            answer2_ents = [[] for _ in range(len(pre1_think))]
            think2_stop = [""] * len(pre1_think)
            answer2_stop = [""] * len(pre1_think)
            pass2_full_rows = [""] * len(pre1_think)

        # ===== WRITE JSON (append) =====
        with open(outpath, "a", encoding="utf-8") as f:
            for row_i in range(len(pass1_full_rows)):
                ex_idx = row_to_ex_idx[row_i]
                ex = work_items[ex_idx]
                prob = ex["_normalized_problem"]
                gold = ex["_normalized_gold"]
                canon_gold = _canon_math(gold)
                k = row_target_sample_idx[row_i]

                p1 = _pack_pass_result(
                    problem=prob,
                    full_text=pass1_full_rows[row_i],
                    ent_think=think1_ents[row_i],
                    ent_answer=answer1_ents[row_i],
                    canon_gold=canon_gold,
                    injected_cue=False,
                    prev_output=None,
                    cue_prefix_str="",
                    stop_reason_think=think1_stop[row_i],
                    stop_reason_answer=answer1_stop[row_i],
                )

                p2 = None
                if two_pass:
                    p2 = _pack_pass_result(
                        problem=prob,
                        full_text=pass2_full_rows[row_i],
                        ent_think=think2_ents[row_i],
                        ent_answer=answer2_ents[row_i],
                        canon_gold=canon_gold,
                        injected_cue=True,
                        prev_output=firstpass_choice_text_per_ex[ex_idx],
                        cue_prefix_str=cue_str,
                        stop_reason_think=think2_stop[row_i],
                        stop_reason_answer=answer2_stop[row_i],
                    )
                    p2["improved_over_pass1"] = bool(p2.get("is_correct_pred")) and not bool(p1.get("is_correct_pred"))

                row = {
                    "problem": prob,
                    "gold_answer": gold,
                    "gold_answer_canon": canon_gold,
                    "step": step,
                    "split": split_name,
                    "sample_idx": k,
                    "pass1": p1,
                    "pass2": p2,
                }
                json.dump(row, f, ensure_ascii=False); f.write("\n")

                # Update in-memory resume maps so later batches see the new rows
                existing_samples[prob].add(k)
                existing_pass1[(prob, k)] = p1["output"]

        # logging
        filled = sum(len(ex["_todo_samples"]) for ex in work_items)
        logger.info("Filled %d missing samples across %d problems in this batch.", filled, B)

def load_math500(cache_dir: str, split: str, seed: int, dataset_path: Optional[str] = None):
    from datasets import load_dataset
    if dataset_path:
        logger.info("Loading MATH-500 from local file: %s", dataset_path)
        return load_local_json_dataset(dataset_path)
    candidates = [
        "HuggingFaceH4/MATH-500",
        "AI-MO/MATH-500",
        "lighteval/MATH-500",
        "openai/math-500",
        "TIGER-Lab/MATH-500",
    ]
    for repo in candidates:
        try:
            logger.info("Trying remote MATH-500 candidate: %s", repo)
            ds_full = load_dataset(repo, split="test", cache_dir=cache_dir)
            colnames = set(ds_full.column_names)
            def _norm(ex):
                problem = (ex.get("problem") or ex.get("question") or
                           ex.get("prompt") or ex.get("instruction") or ex.get("query"))
                ans = (ex.get("answer") or ex.get("solution") or
                       ex.get("final_answer") or ex.get("boxed_answer") or ex.get("target"))
                return {"problem": problem, "answer": ans}
            ds = ds_full.map(_norm, remove_columns=list(colnames))
            ds = ds.filter(lambda ex: ex["problem"] is not None and ex["answer"] is not None)
            if len(ds) == 0:
                raise ValueError(f"{repo} contained no usable (problem,answer) pairs")
            logger.info("Loaded MATH-500 from %s | N=%d", repo, len(ds))
            return ds
        except Exception as e:
            logger.warning("Skipping %s (%r)", repo, e)
    try:
        ds_full = load_dataset("hendrycks/competition_math", split="test", cache_dir=cache_dir)
        n = min(500, len(ds_full))
        return ds_full.shuffle(seed=seed).select(range(n))
    except Exception as e:
        raise RuntimeError(f"Could not load MATH-500 or fallback dataset: {e}")

# ─────────────────────────── Main ───────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--output_dir", required=True)

    # Data selection
    ap.add_argument("--dataset_id", default="MATH-500", help="Use 'MATH-500' (default) or a HF dataset path.")
    ap.add_argument("--split", default="test", help="Split name to run on.")
    ap.add_argument("--num_examples", type=int, default=None, help="Optional cap if you want fewer than 500.")

    # Decoding + sampling
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.95)

    # Budgets (per pass)
    ap.add_argument("--think_cap", type=int, default=750)
    ap.add_argument("--answer_cap", type=int, default=50)

    # System/runtime
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--tokenizer_path", default=None)
    ap.add_argument("--seed", type=int, default=42)

    # Entropy + attention impl
    ap.add_argument("--entropy_mode", choices=["full","reconsider","none"], default="reconsider")
    ap.add_argument("--attn_implementation", default="sdpa",
                    choices=["sdpa", "eager", "flash_attention_2"])

    # Two-pass controls
    ap.add_argument("--two_pass", action="store_true")
    ap.add_argument("--second_pass_phrase", default="Wait, we need to reconsider. Let's think this through step by step.")
    ap.add_argument("--second_pass_use_sample_idx", type=int, default=0)

    args = ap.parse_args()

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

    # EOS set
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

    from datasets import load_dataset
    if args.dataset_id.upper() == "MATH-500":
        ds = load_math500(HF_CACHE_DIR, args.split, args.seed)
        dataset_name_for_log = "MATH-500"
    else:
        ds = load_dataset(args.dataset_id, split=args.split, cache_dir=HF_CACHE_DIR)
        dataset_name_for_log = args.dataset_id

    if args.num_examples is not None and args.num_examples > 0:
        ds = ds.select(range(min(args.num_examples, len(ds))))

    ds = ds.shuffle(seed=int.from_bytes(os.urandom(4), "little"))

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

    logger.info("All inference complete.")

if __name__ == "__main__":
    main()
