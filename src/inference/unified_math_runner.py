#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified math inference runner using the shared task registry and backends.

Behavior mirrors math-inference.py:
 - resume/fill JSONL (writes missing sample_idx only)
 - optional two-pass with cue injection
 - entropy logging (NaN-safe fallback)
 - correctness via canonicalized substring match

This runner uses HFBackend for model/tokenizer setup but still calls
model.generate directly to compute token-level entropies.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import torch
from packaging import version
from torch.nn import functional as F
from transformers import StoppingCriteria, StoppingCriteriaList

from src.inference.backends import HFBackend, StopOnSubstrings
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
from src.inference.task_registry import TASK_REGISTRY

logging.basicConfig(
    level=getattr(logging, os.getenv("LOGLEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)
logger.info("Starting %s", os.path.basename(__file__))

# PyTorch 2.6 DeepSpeed un-pickle patch (safe no-op if absent)
try:
    if version.parse(torch.__version__) >= version.parse("2.6.0"):
        from torch.serialization import add_safe_globals  # type: ignore
        from deepspeed.runtime.zero.config import ZeroStageEnum  # type: ignore
        from deepspeed.runtime.fp16.loss_scaler import LossScaler  # type: ignore

        add_safe_globals([ZeroStageEnum, LossScaler])
        logger.info("DeepSpeed ZeRO patch enabled")
except Exception as e:  # noqa: BLE001
    logger.warning("DeepSpeed patch failed: %r", e)

# Optional reconsider markers (analytics)
_RECONSIDER_PATTERNS = [
    ("wait_line", re.compile(r"(?im)^\s*wait[,\.\-–—… ]", re.I)),
    ("wait_reconsider", re.compile(r"\bwait\b.*\breconsider\b", re.I | re.S)),
    ("reconsider_exact", re.compile(r"\bwait[,!\.\s]*let me reconsider\b", re.I)),
    ("step_by_step", re.compile(r"\blet'?s take (this|it) step[-\s]?by[-\s]?step\b", re.I)),
    ("step_by_step_alt", re.compile(r"\bstep[-\s]?by[-\s]?step\b", re.I)),
    ("recheck", re.compile(r"\bre[-\s]?check(ing)?\b", re.I)),
]


# -------------------- resume/fill helpers --------------------
def _scan_existing_results(results_path: str) -> tuple[DefaultDict[str, set], Dict[tuple, str]]:
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
            p1 = obj.get("pass1") or {}
            p1_out = p1.get("output")
            if isinstance(p1_out, str):
                existing_pass1[(prob, int(k))] = p1_out
    return existing_samples, existing_pass1


# -------------------- generation helpers --------------------
def _make_gen_kwargs(cap: int, temperature: float, top_p: Optional[float], eos_ids: Optional[List[int]]) -> dict:
    do_sample = (temperature is not None) and (float(temperature) > 0.0)
    kwargs = dict(
        max_new_tokens=cap,
        pad_token_id=None,
        eos_token_id=eos_ids,
        do_sample=do_sample,
        return_dict_in_generate=True,
        output_scores=True,
        num_return_sequences=1,
    )
    if do_sample:
        kwargs["temperature"] = float(temperature)
        if top_p is not None:
            kwargs["top_p"] = float(top_p)
    return kwargs


def _gen_batch(
    tokenizer,
    model,
    prefixes: List[str],
    cap: int,
    stop_strs: List[str],
    temperature: float,
    top_p: Optional[float],
    eos_ids: Optional[List[int]],
    max_length: int = 4096,
):
    inputs = tokenizer(
        prefixes, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    inputs, input_lengths = move_inputs_to_device(inputs)
    stop = StoppingCriteriaList([StopOnSubstrings(tokenizer, stop_strs)]) if stop_strs else None
    gen_kwargs = _make_gen_kwargs(cap, temperature, top_p, eos_ids)
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
        if found_stop:
            stop_reasons.append("stop_token")
        elif has_eos:
            stop_reasons.append("eos")
        elif hit_max:
            stop_reasons.append("max_new_tokens")
        else:
            stop_reasons.append("other")

        decs.append(raw_txt)

        tok_ents: List[float] = []
        prompt_len = start_tok_idx
        scores_T = len(out.scores)
        seq_len = seqs.shape[1]
        for t in range(prompt_len, min(prompt_len + scores_T, seq_len - 1)):
            logits = out.scores[t - prompt_len][row_i : row_i + 1, :].float()
            logp = F.log_softmax(logits, dim=-1)
            if torch.isnan(logp).any() or torch.isinf(logp).any():
                tok_ents = []
                break
            p = logp.exp()
            h = float(-(p * logp).sum().item())
            if not torch.isfinite(torch.tensor(h)):
                tok_ents = []
                break
            tok_ents.append(h)
        if not tok_ents:
            start_idx = start_tok_idx - 1
            tok_ents = _entropy_from_start_index(model, seqs[row_i : row_i + 1], start_idx) or []
        ent_series.append(tok_ents)

    return decs, ent_series, input_lengths, seqs, stop_reasons


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
    Tans = len(ent_answer or [])
    T = len(tok_ents_all)
    think, answer = _extract_blocks(full_text)
    think_text = think or ""
    pred_answer_text = answer or ""
    skip_chars = len(cue_prefix_str) if injected_cue else 0
    markers, pos_in_think, reconsider_context, reconsider_excerpt = _find_markers_and_context(
        think_text, f"Problem: {problem}", _RECONSIDER_PATTERNS, skip_prefix_chars=skip_chars
    )
    if injected_cue:
        markers = ["injected_cue"] + (markers or [])
    t_cue = 0 if injected_cue else None
    if (not injected_cue) and (pos_in_think is not None):
        t_cue = max(0, min(pos_in_think, Tthink))
    entropy_overall = _finite_mean(tok_ents_all) if tok_ents_all else None
    entropy_think = _finite_mean(ent_think) if ent_think else None
    entropy_answer = _finite_mean(ent_answer) if ent_answer else None
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
        entropy_pre_cue=None,
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
        valid_tag_structure=_valid_tag_structure(full_text),
    )


# -------------------- Dataset helper --------------------
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
                problem = (
                    ex.get("problem")
                    or ex.get("question")
                    or ex.get("prompt")
                    or ex.get("instruction")
                    or ex.get("query")
                )
                ans = (
                    ex.get("answer")
                    or ex.get("solution")
                    or ex.get("final_answer")
                    or ex.get("boxed_answer")
                    or ex.get("target")
                )
                return {"problem": problem, "answer": ans}

            ds = ds_full.map(_norm, remove_columns=list(colnames))
            ds = ds.filter(lambda ex: ex["problem"] is not None and ex["answer"] is not None)
            if len(ds) == 0:
                raise ValueError(f"{repo} contained no usable (problem,answer) pairs")
            logger.info("Loaded MATH-500 from %s | N=%d", repo, len(ds))
            return ds
        except Exception as e:  # noqa: BLE001
            logger.warning("Skipping %s (%r)", repo, e)
    try:
        ds_full = load_dataset("hendrycks/competition_math", split="test", cache_dir=cache_dir)
        n = min(500, len(ds_full))
        return ds_full.shuffle(seed=seed).select(range(n))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Could not load MATH-500 or fallback dataset: {e}") from e


# -------------------- Main inference loop --------------------
def run_math_inference(
    backend: HFBackend,
    dataset,
    output_dir: str,
    *,
    step: int,
    batch_size: int,
    num_samples: int,
    temperature: float,
    top_p: float,
    think_cap: int,
    answer_cap: int,
    two_pass: bool,
    second_pass_phrase: str,
    second_pass_use_sample_idx: int,
    eos_ids: Optional[List[int]],
):
    tokenizer = backend.tokenizer
    model = backend.model

    outpath = os.path.join(output_dir, f"step{step:04d}_test.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    existing_samples, existing_pass1 = _scan_existing_results(outpath)

    # normalize dataset rows
    normed = []
    for ex in dataset:
        problem = (
            ex.get("problem")
            or ex.get("question")
            or ex.get("query")
            or ex.get("prompt")
            or ex.get("instruction")
        )
        gold = (
            ex.get("answer")
            or ex.get("final_answer")
            or ex.get("target")
            or ex.get("boxed_answer")
            or ex.get("solution")
        )
        if not problem:
            continue
        normed.append(
            {
                "_normalized_problem": problem,
                "_normalized_gold": gold,
                "_todo_samples": [],
            }
        )
    logger.info("→ test | %d examples", len(normed))

    # mark missing sample indices for each example
    for ex in normed:
        prob = ex["_normalized_problem"]
        have = existing_samples.get(prob, set())
        ex["_todo_samples"] = [k for k in range(num_samples) if k not in have]

    # mini-batches
    for i in range(0, len(normed), batch_size):
        batch = normed[i : i + batch_size]
        work_items = [ex for ex in batch if ex["_todo_samples"]]
        if not work_items:
            continue
        B = len(work_items)

        # PASS 1
        sys_prompt = TASK_REGISTRY["math-qwen"].system_prompt
        base1 = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"Problem: {ex['_normalized_problem']}"},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for ex in work_items
        ]

        pre1_think: List[str] = []
        row_to_ex_idx: List[int] = []
        row_target_sample_idx: List[int] = []
        for ex_idx, ex in enumerate(work_items):
            for k in ex["_todo_samples"]:
                pre1_think.append(base1[ex_idx] + "<think>\n")
                row_to_ex_idx.append(ex_idx)
                row_target_sample_idx.append(k)
        think1_texts, think1_ents, _, _, think1_stop = _gen_batch(
            tokenizer,
            model,
            pre1_think,
            think_cap,
            ["</think>"],
            temperature,
            top_p,
            eos_ids,
        )

        pre1_answer = []
        for row_i in range(len(pre1_think)):
            pre = pre1_think[row_i] + think1_texts[row_i] + "</think>\n<answer>\n"
            pre1_answer.append(pre)
        answer1_texts, answer1_ents, _, _, answer1_stop = _gen_batch(
            tokenizer,
            model,
            pre1_answer,
            answer_cap,
            ["</answer>"],
            temperature,
            top_p,
            eos_ids,
        )

        pass1_full_rows = [
            f"<think>{think1_texts[row_i]}</think>\n<answer>{answer1_texts[row_i]}</answer>"
            for row_i in range(len(pre1_think))
        ]

        new_pass1_by_ex_and_k: Dict[tuple, str] = {}
        for row_i in range(len(pass1_full_rows)):
            ex_idx = row_to_ex_idx[row_i]
            k = row_target_sample_idx[row_i]
            new_pass1_by_ex_and_k[(ex_idx, k)] = pass1_full_rows[row_i]

        # Select pass1 text for cue
        cue_str = second_pass_phrase.strip() + " "
        firstpass_choice_text_per_ex: List[str] = []
        if two_pass:
            for ex_idx, ex in enumerate(work_items):
                prob = ex["_normalized_problem"]
                k_choice = max(0, min(second_pass_use_sample_idx, num_samples - 1))
                prev = existing_pass1.get((prob, k_choice))
                if prev is None:
                    prev = new_pass1_by_ex_and_k.get((ex_idx, k_choice))
                    if prev is None:
                        have_sorted = sorted(existing_samples.get(prob, set()))
                        if have_sorted:
                            prev = existing_pass1.get((prob, have_sorted[0]))
                        else:
                            new_ks = sorted(
                                [k for (eidx, k) in new_pass1_by_ex_and_k.keys() if eidx == ex_idx]
                            )
                            if new_ks:
                                prev = new_pass1_by_ex_and_k[(ex_idx, new_ks[0])]
                if prev is None:
                    for row_i in range(len(pass1_full_rows)):
                        if row_to_ex_idx[row_i] == ex_idx:
                            prev = pass1_full_rows[row_i]
                            break
                firstpass_choice_text_per_ex.append(prev or "")
        else:
            firstpass_choice_text_per_ex = [""] * B

        # PASS 2 (one per missing sample)
        if two_pass:
            base2_per_ex = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": f"Problem: {ex['_normalized_problem']}"},
                        {"role": "assistant", "content": firstpass_choice_text_per_ex[ex_idx]},
                        {"role": "user", "content": second_pass_phrase.strip()},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for ex_idx, ex in enumerate(work_items)
            ]

            pre2_think: List[str] = []
            for row_i in range(len(pre1_think)):
                ex_idx = row_to_ex_idx[row_i]
                base2 = base2_per_ex[ex_idx]
                pre2_think.append(base2 + "<think>\n" + cue_str)

            think2_texts_only_new, think2_ents, _, _, think2_stop = _gen_batch(
                tokenizer,
                model,
                pre2_think,
                think_cap,
                ["</think>"],
                temperature,
                top_p,
                eos_ids,
            )
            think2_texts = [cue_str + t for t in think2_texts_only_new]

            pre2_answer = []
            for row_i in range(len(pre1_think)):
                pre = pre2_think[row_i] + think2_texts_only_new[row_i] + "</think>\n<answer>\n"
                pre2_answer.append(pre)

            answer2_texts, answer2_ents, _, _, answer2_stop = _gen_batch(
                tokenizer,
                model,
                pre2_answer,
                answer_cap,
                ["</answer>"],
                temperature,
                top_p,
                eos_ids,
            )

        else:
            think2_ents = [[] for _ in range(len(pre1_think))]
            answer2_ents = [[] for _ in range(len(pre1_think))]
            think2_stop = [""] * len(pre1_think)
            answer2_stop = [""] * len(pre1_think)
            think2_texts = [""] * len(pre1_think)
            answer2_texts = [""] * len(pre1_think)

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
                    injected_cue=False,
                    canon_gold=canon_gold,
                    prev_output=None,
                    cue_prefix_str="",
                    stop_reason_think=think1_stop[row_i],
                    stop_reason_answer=answer1_stop[row_i],
                )

                if two_pass:
                    p2 = _pack_pass_result(
                        problem=prob,
                        full_text=f"<think>{think2_texts[row_i]}</think>\n<answer>{answer2_texts[row_i]}</answer>",
                        ent_think=think2_ents[row_i],
                        ent_answer=answer2_ents[row_i],
                        injected_cue=True,
                        canon_gold=canon_gold,
                        prev_output=p1["output"],
                        cue_prefix_str=cue_str,
                        stop_reason_think=think2_stop[row_i],
                        stop_reason_answer=answer2_stop[row_i],
                    )
                    p2["improved_over_pass1"] = bool(p2.get("is_correct_pred")) and not bool(
                        p1.get("is_correct_pred")
                    )
                else:
                    p2 = None

                row = {
                    "problem": prob,
                    "gold_answer": gold,
                    "gold_answer_canon": canon_gold,
                    "step": step,
                    "split": "test",
                    "sample_idx": k,
                    "pass1": p1,
                    "pass2": p2,
                }
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")
            # mark seen for resume tracking
            for ex in work_items:
                existing_samples[ex["_normalized_problem"]].update(ex["_todo_samples"])
    logger.info("All inference complete → %s", outpath)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--dataset_id", default="MATH-500", help="Use 'MATH-500' or a HF dataset path.")
    ap.add_argument("--dataset_path", default=None, help="Optional local JSONL for MATH-500-style records.")
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
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--tokenizer_path", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attn_implementation", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])

    # Two-pass controls
    ap.add_argument("--two_pass", action="store_true")
    ap.add_argument(
        "--second_pass_phrase",
        default="Wait, we need to reconsider. Let's think this through step by step.",
    )
    ap.add_argument("--second_pass_use_sample_idx", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    HF_CACHE_DIR = os.path.abspath("./.hf_cache")

    backend = HFBackend.from_pretrained(
        args.model_name_or_path,
        revision=args.revision,
        cache_dir=HF_CACHE_DIR,
        dtype=args.dtype,
        device_map="auto",
        attn_implementation=args.attn_implementation,
        tokenizer_path=args.tokenizer_path,
    )

    # EOS IDs
    eos_ids = set()
    if backend.tokenizer.eos_token_id is not None:
        eos_ids.add(int(backend.tokenizer.eos_token_id))
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tid = backend.tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != backend.tokenizer.pad_token_id:
            eos_ids.add(int(tid))
    eos_ids = sorted(eos_ids) if eos_ids else None

    ds = load_math500(
        cache_dir=HF_CACHE_DIR,
        split=args.split,
        seed=args.seed,
        dataset_path=args.dataset_path,
    )
    if args.num_examples is not None and args.num_examples > 0:
        ds = ds.select(range(min(args.num_examples, len(ds))))

    run_math_inference(
        backend=backend,
        dataset=ds,
        output_dir=args.output_dir,
        step=args.step,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        think_cap=args.think_cap,
        answer_cap=args.answer_cap,
        two_pass=args.two_pass,
        second_pass_phrase=args.second_pass_phrase,
        second_pass_use_sample_idx=args.second_pass_use_sample_idx,
        eos_ids=eos_ids,
    )


if __name__ == "__main__":
    main()
