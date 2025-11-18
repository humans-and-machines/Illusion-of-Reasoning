#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-pass batch inference for Llama-8B-style chat LMs that produce:
  <think> ... </think><answer> ... </answer>

DeepSpeed (ZeRO-3) edition:
- Loads ZeRO-3 shards directly from checkpoint-XXX/global_stepXXXX (no merge).
- Uses deepspeed.initialize + engine.load_checkpoint.
- Generation goes through engine.module.generate(...).

Resume/fill:
- If step####_<split>.jsonl exists, append only missing samples per problem
  up to --num_samples (preserves prior rows).

Other behaviors preserved:
- Two-phase per pass (think -> answer) with explicit `</think>` / `</answer>` stops.
- Per-pass caps: think <= 750 new tokens, answer <= 50.
- Pass 2 injects a cue at start of <think>, but that cue is excluded from
  reconsider-cue analytics.
- EOS set includes Llama-3 tokens <|eot_id|>, <|end_of_text|>.
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
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
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

# ───────────────────────── Regex helpers ─────────────────────────
RE_THINK  = re.compile(r"(?si)<think>(.*?)</think>")
RE_ANSWER = re.compile(r"(?si)<answer>(.*?)</answer>")

# Optional reconsider-cue detectors (analytics)
_RECON_PATTERNS = [
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

RE_LATEX_FRAC = re.compile(r"\\frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}", re.I)
RE_LATEX_CMDS = re.compile(r"\\(left|right|,|;|!|:)", re.I)
RE_SPACES = re.compile(r"\s+")
RE_BRACES = re.compile(r"[{}]")
RE_PARENS_COMMAs = re.compile(r"[()\[\],]")

def _canon_math(x: Optional[str]) -> Optional[str]:
    if x is None: return None
    s = x.strip()
    s = (s.replace("–","-").replace("—","-").replace("−","-")
           .replace("π", "pi").replace("\\pi", "pi"))
    s = RE_LATEX_CMDS.sub("", s)
    s = RE_LATEX_FRAC.sub(r"\1/\2", s)
    s = RE_BRACES.sub("", s)
    s = RE_SPACES.sub("", s)
    s = RE_PARENS_COMMAs.sub("", s)
    s = s.replace("\\boxed", "").replace("$", "").lower().rstrip(".")
    s = re.sub(r"/{2,}", "/", s)
    s = re.sub(r"\+{2,}", "+", s)
    s = re.sub(r"-{2,}", "-", s)
    if s.startswith("+"): s = s[1:]
    return s

def _contains_canon(hay: Optional[str], needle: Optional[str]) -> bool:
    return bool(hay and needle and (needle in hay))

def _extract_blocks(txt: str) -> Tuple[Optional[str], Optional[str]]:
    m1, m2 = RE_THINK.search(txt), RE_ANSWER.search(txt)
    return (m1.group(1).strip() if m1 else None,
            m2.group(1).strip() if m2 else None)

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

def _find_markers_and_context(think_text: Optional[str], problem_text: str, skip_prefix_chars: int = 0):
    if not think_text: return [], None, None, None
    search_text = think_text[skip_prefix_chars:] if skip_prefix_chars > 0 else think_text
    earliest_pos, markers = None, []
    for name, pat in _RECON_PATTERNS:
        m = pat.search(search_text)
        if m:
            markers.append(name)
            pos_global = (skip_prefix_chars + m.start()) if skip_prefix_chars > 0 else m.start()
            if earliest_pos is None or pos_global < earliest_pos:
                earliest_pos = pos_global
    if not markers:
        return [], None, None, None
    prefix = think_text[:earliest_pos] if earliest_pos is not None else think_text
    reconsider_context = f"Problem: {problem_text}\n\n{prefix}"
    lo = max(0, (earliest_pos or 0) - 60)
    hi = min(len(think_text), (earliest_pos or 0) + 60)
    reconsider_excerpt = think_text[lo:hi]
    return markers, earliest_pos, reconsider_context, reconsider_excerpt

def _first_eos_any(token_ids: torch.Tensor, eos_id_list: Optional[List[int]]) -> int:
    if not eos_id_list: return token_ids.numel()
    hits = []
    for eid in eos_id_list:
        pos = (token_ids == eid).nonzero(as_tuple=False)
        if pos.numel() > 0: hits.append(pos[0].item())
    return min(hits) if hits else token_ids.numel()

def _entropy_from_start_index(model, seq_ids: torch.Tensor, start_idx: int) -> List[float]:
    try:
        device = next(model.parameters()).device  # works for DS wrapper's .module too
    except Exception:
        device = seq_ids.device
    seq_ids = seq_ids.to(device)
    ents: List[float] = []
    with torch.inference_mode():
        out = model(input_ids=seq_ids[:, :start_idx+1], use_cache=True)
        past = out.past_key_values
        L = seq_ids.shape[1]
        for t in range(start_idx, L-1):
            out = model(input_ids=seq_ids[:, t:t+1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.log_logits if hasattr(out, "log_logits") else out.logits
            logits = logits[:, -1, :].float()
            logp = F.log_softmax(logits, dim=-1)
            p = logp.exp()
            h = float(-(p * logp).sum().item())
            if not math.isfinite(h):
                logits = (logits - logits.max()).float()
                logp = F.log_softmax(logits, dim=-1); p = logp.exp()
                h = float(-(p * logp).sum().item())
            ents.append(h)
    return ents

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

# ───── PyTorch 2.6 DeepSpeed unpickle patch (no-op if absent) ─────
try:
    if version.parse(torch.__version__) >= version.parse("2.6.0"):
        from torch.serialization import add_safe_globals  # type: ignore
        from deepspeed.runtime.zero.config import ZeroStageEnum  # type: ignore
        from deepspeed.runtime.fp16.loss_scaler import LossScaler  # type: ignore
        add_safe_globals([ZeroStageEnum, LossScaler])
        logger.info("DeepSpeed ZeRO patch enabled")
except Exception as e:  # noqa: BLE001
    logger.warning("DeepSpeed patch failed: %r", e)

# ─────────────────────────── DeepSpeed wrapper ───────────────────────────
class DSModelWrapper:
    """Adapter so code can call `.generate` and `.parameters()` on a DeepSpeed engine."""
    def __init__(self, engine):
        self.engine = engine
        self.module = engine.module
        self.device = getattr(engine, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.config = getattr(self.module, "config", None)
    def __getattr__(self, name):
        if hasattr(self.module, name):
            return getattr(self.module, name)
        raise AttributeError(name)
    def parameters(self):
        return self.module.parameters()
    def generate(self, *args, **kwargs):
        return self.module.generate(*args, **kwargs)
    def eval(self):
        self.module.eval()
        return self

# ───────────────────────── Prompt builders (Llama chat template) ─────────────────────────
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
def _scan_existing_results(results_path: str) -> tuple[DefaultDict[str, set], Dict[tuple, str]]:
    existing_samples: DefaultDict[str, set] = defaultdict(set)
    existing_pass1: Dict[tuple, str] = {}
    if not os.path.exists(results_path):
        return existing_samples, existing_pass1
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            prob = obj.get("problem"); k = obj.get("sample_idx")
            if prob is None or k is None: continue
            existing_samples[prob].add(int(k))
            p1_out = (obj.get("pass1") or {}).get("output")
            if isinstance(p1_out, str):
                existing_pass1[(prob, int(k))] = p1_out
    return existing_samples, existing_pass1

# ───────────────────── Inference Loop (two-phase per pass) ─────────────────────
def run_inference_on_split(
    split_name: str,
    examples,  # datasets.Dataset
    tokenizer,
    model,     # DSModelWrapper or plain HF model
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
    if (temperature is None or float(temperature) == 0.0) and num_samples > 1:
        logging.warning("temperature=0 with num_samples=%d → all samples will be identical (greedy).", num_samples)

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
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            kwargs["synced_gpus"] = True
        if do_sample:
            kwargs["temperature"] = float(temperature)
            if top_p is not None:
                kwargs["top_p"] = float(top_p)
        return kwargs

    def _model_device():
        try:
            return model.device
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _gen_batch(prefixes: List[str], cap: int, stop_strs: List[str]) -> Tuple[
        List[str], List[List[float]], torch.Tensor, torch.Tensor, List[str]
    ]:
        inputs = tokenizer(prefixes, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        input_lengths = inputs["attention_mask"].sum(dim=1)
        dev = _model_device()
        for k in inputs:
            inputs[k] = inputs[k].to(dev)

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
                        has_eos = True; break
            hit_max = len(gen_ids) >= cap
            stop_reasons.append("stop_token" if found_stop else "eos" if has_eos else "max_new_tokens" if hit_max else "other")

            txt = raw_txt
            for s in (stop_strs or []):
                if s in txt:
                    txt = txt.split(s, 1)[0]; break
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

    outpath = os.path.join(outdir, f"step{step:04d}_{split_name}.jsonl")
    existing_samples, existing_pass1 = _scan_existing_results(outpath)
    logger.info("Resume scan: %d problems already present", len(existing_samples))

    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    for i in range(0, len(examples), batch_size):
        idx_lo, idx_hi = i, min(i + batch_size, len(examples))
        slice_ds = examples.select(range(idx_lo, idx_hi))

        work_items = []
        for ex in slice_ds:
            prob = (ex.get("problem") or ex.get("question") or ex.get("query") or ex.get("prompt") or ex.get("instruction"))
            gold = (ex.get("answer")  or ex.get("final_answer") or ex.get("target") or ex.get("boxed_answer") or ex.get("solution"))
            if gold and not any(k in ex for k in ("answer","final_answer","target","boxed_answer","solution")):
                m = re.search(r"\\boxed\{([^}]*)\}", str(gold)) or re.search(r"\\boxed\(([^)]*)\)", str(gold))
                if m: gold = m.group(1)
            if not prob: continue
            have = existing_samples.get(prob, set())
            if len(have) >= num_samples: continue
            todo = [k for k in range(num_samples) if k not in have]
            if not todo: continue
            ex = dict(ex)
            ex["_normalized_problem"] = prob
            ex["_normalized_gold"] = gold
            ex["_todo_samples"] = todo
            work_items.append(ex)

        if not work_items: continue

        base1 = [chat_base_for_pass1(tokenizer, ex["_normalized_problem"]) for ex in work_items]
        pre1_think, row_to_ex_idx, row_target_sample_idx = [], [], []
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

        new_pass1_by_ex_and_k: Dict[tuple, str] = {}
        for row_i in range(len(pass1_full_rows)):
            ex_idx = row_to_ex_idx[row_i]; k = row_target_sample_idx[row_i]
            new_pass1_by_ex_and_k[(ex_idx, k)] = pass1_full_rows[row_i]

        cue_str = second_pass_phrase.strip() + " "
        firstpass_choice_text_per_ex: List[str] = []

        two_pass = bool(two_pass)
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
                            new_ks = sorted([k for (eidx, k) in new_pass1_by_ex_and_k if eidx == ex_idx])
                            if new_ks: prev = new_pass1_by_ex_and_k[(ex_idx, new_ks[0])]
                if prev is None:
                    for row_i in range(len(pass1_full_rows)):
                        if row_to_ex_idx[row_i] == ex_idx:
                            prev = pass1_full_rows[row_i]; break
                firstpass_choice_text_per_ex.append(prev or "")
        else:
            firstpass_choice_text_per_ex = [""] * len(work_items)

        if two_pass:
            base2_per_ex = [
                chat_base_for_pass2(
                    tokenizer,
                    ex["_normalized_problem"],
                    firstpass_choice_text_per_ex[ex_idx],
                    second_pass_phrase.strip(),
                ) for ex_idx, ex in enumerate(work_items)
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

        with open(outpath, "a", encoding="utf-8") as f:
            for row_i in range(len(pass1_full_rows)):
                ex_idx = row_to_ex_idx[row_i]
                ex = work_items[ex_idx]
                prob = ex["_normalized_problem"]
                gold = ex["_normalized_gold"]
                canon_gold = _canon_math(gold)
                k = row_target_sample_idx[row_i]

                def _pack(full_text: str, ent_t: List[float], ent_a: List[float],
                          injected: bool, prev_output=None, cue_prefix="",
                          stop_t=None, stop_a=None):
                    tok_ents_all = (ent_t or []) + (ent_a or [])
                    Tthink, Tans, T = len(ent_t or []), len(ent_a or []), len(tok_ents_all)
                    think, answ = _extract_blocks(full_text)
                    think_text = think or ""
                    pred_answer_text = answ or ""
                    skip_chars = len(cue_prefix) if injected else 0
                    markers, pos_in_think, recon_ctx, recon_x = _find_markers_and_context(
                        think_text, prob, skip_prefix_chars=skip_chars
                    )
                    if injected: markers = ["injected_cue"] + (markers or [])
                    t_cue = 0 if injected else (max(0, min(pos_in_think, Tthink)) if (pos_in_think is not None) else None)
                    entropy_overall = _finite_mean(tok_ents_all) if tok_ents_all else None
                    entropy_think   = _finite_mean(ent_t) if ent_t else None
                    entropy_answer  = _finite_mean(ent_a) if ent_a else None
                    entropy_reconsider_think = None
                    entropy_reconsider_full = None
                    if t_cue is not None:
                        if T > t_cue:      entropy_reconsider_full  = _finite_mean(tok_ents_all[t_cue:])
                        if Tthink > t_cue: entropy_reconsider_think = _finite_mean(tok_ents_all[t_cue:Tthink])
                    pred_canon = _canon_math(pred_answer_text)
                    is_corr = _contains_canon(pred_canon, canon_gold)
                    return dict(
                        prev_output=prev_output, output=full_text,
                        pred_answer=pred_answer_text, pred_answer_canon=pred_canon,
                        entropy=entropy_overall, entropy_think=entropy_think, entropy_answer=entropy_answer,
                        entropy_pre_cue=None, entropy_reconsider_think=entropy_reconsider_think,
                        entropy_reconsider_full=entropy_reconsider_full,
                        stop_reason_think=stop_t, stop_reason_answer=stop_a,
                        has_reconsider_cue=bool(markers), reconsider_markers=markers or [],
                        reconsider_pos=pos_in_think, reconsider_context=recon_ctx, reconsider_excerpt=recon_x,
                        is_correct_pred=is_corr,
                        is_correct_after_reconsideration=bool(markers) and bool(is_corr),
                        tokens_total=T, tokens_end_think=Tthink, tokens_think=Tthink, tokens_answer=Tans,
                        valid_tag_structure=_valid_tag_structure(full_text),
                    )

                p1 = _pack(pass1_full_rows[row_i], think1_ents[row_i], answer1_ents[row_i],
                           injected=False, prev_output=None, cue_prefix="", stop_t=think1_stop[row_i], stop_a=answer1_stop[row_i])

                p2 = None
                if two_pass:
                    p2 = _pack(pass2_full_rows[row_i], think2_ents[row_i], answer2_ents[row_i],
                               injected=True, prev_output=None, cue_prefix=cue_str,
                               stop_t=think2_stop[row_i], stop_a=answer2_stop[row_i])
                    p2["improved_over_pass1"] = bool(p2.get("is_correct_pred")) and not bool(p1.get("is_correct_pred"))

                json.dump({
                    "problem": prob,
                    "gold_answer": gold,
                    "gold_answer_canon": canon_gold,
                    "step": step,
                    "split": split_name,
                    "sample_idx": k,
                    "pass1": p1,
                    "pass2": p2,
                }, f, ensure_ascii=False); f.write("\n")

    logger.info("Finished split=%s; wrote %s", split_name, outpath)

# ─────────────────────────── Dataset helpers ───────────────────────────
def _load_local_json_dataset(path: str):
    from datasets import Dataset
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line) if line.startswith("{") else None
            if obj is None: continue
            records.append(obj)
    return Dataset.from_list(records)

def load_math500(cache_dir: str, split: str, seed: int, dataset_path: Optional[str] = None):
    from datasets import load_dataset
    if dataset_path:
        return _load_local_json_dataset(dataset_path)
    candidates = [
        "HuggingFaceH4/MATH-500", "AI-MO/MATH-500", "lighteval/MATH-500",
        "openai/math-500", "TIGER-Lab/MATH-500",
    ]
    for repo in candidates:
        try:
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
            if len(ds) == 0: raise ValueError("No usable pairs")
            return ds
        except Exception:
            continue
    ds_full = load_dataset("hendrycks/competition_math", split="test", cache_dir=cache_dir)
    n = min(500, len(ds_full))
    return ds_full.shuffle(seed=seed).select(range(n))

# ─────────────────────────── Main ───────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True,
                    help="Path to checkpoint-XXX directory (contains global_stepXXXX/)")
    ap.add_argument("--revision")
    ap.add_argument("--output_dir", required=True)

    # Data selection
    ap.add_argument("--dataset_id", default="MATH-500", help="Use 'MATH-500' or a HF dataset path")
    ap.add_argument("--split", default="test")
    ap.add_argument("--num_examples", type=int, default=None)

    # Decoding + sampling
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.95)

    # Budgets (per pass)
    ap.add_argument("--think_cap", type=int, default=750)
    ap.add_argument("--answer_cap", type=int, default=50)

    # System/runtime
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    ap.add_argument("--step", type=int, default=0,
                    help="Derives default --ds_tag=global_step{step} if not set.")
    ap.add_argument("--tokenizer_path", default=None)
    ap.add_argument("--seed", type=int, default=42)

    # Entropy + attention impl
    ap.add_argument("--entropy_mode", choices=["full","reconsider","none"], default="reconsider")
    ap.add_argument("--attn_implementation", default="sdpa",
                    choices=["sdpa", "eager", "flash_attention_2"])

    # Two-pass
    ap.add_argument("--two_pass", action="store_true")
    ap.add_argument("--second_pass_phrase", default="Wait, we need to reconsider. Let's think this through step by step.")
    ap.add_argument("--second_pass_use_sample_idx", type=int, default=0)

    # DeepSpeed controls
    ap.add_argument("--ds_config", required=True, help="Path to DeepSpeed ZeRO-3 inference JSON")
    ap.add_argument("--ds_tag", default=None,
                    help="Checkpoint tag folder (e.g., 'global_step400'). "
                         "Defaults to global_step{--step} if set, else 'latest'.")
    args = ap.parse_args()

    # Tokenizer
    HF_CACHE_DIR = os.path.abspath("./.hf_cache")
    tok_src = args.tokenizer_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tok_src, revision=args.revision, trust_remote_code=True, cache_dir=HF_CACHE_DIR,
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.truncation_side = "left"

    # EOS set (Llama-3/8B style)
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    for tok in ("<|eot_id|>", "<|end_of_text|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.pad_token_id:
            eos_ids.add(int(tid))
    eos_ids = sorted(eos_ids) if eos_ids else None

    # DeepSpeed init and checkpoint load
    import deepspeed
    cfg = AutoConfig.from_pretrained(
        args.model_name_or_path, revision=args.revision, trust_remote_code=True, cache_dir=HF_CACHE_DIR,
    )
    try: cfg.attn_implementation = args.attn_implementation
    except Exception: pass

    model_hf = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    # dtype hint
    try:
        model_hf.to(torch.bfloat16 if args.dtype == "bfloat16" else torch.float16)
    except Exception:
        pass

    engine, _, _, _ = deepspeed.initialize(
        model=model_hf,
        config=args.ds_config,
        model_parameters=[p for p in model_hf.parameters() if p.requires_grad],
    )
    engine.module.eval()

    ds_tag = args.ds_tag or (f"global_step{args.step}" if args.step else "latest")
    logger.info("Loading ZeRO checkpoint: dir=%s | tag=%s", args.model_name_or_path, ds_tag)
    engine.load_checkpoint(
        args.model_name_or_path, tag=ds_tag,
        load_optimizer_states=False, load_lr_scheduler_states=False, load_module_strict=False,
    )

    model = DSModelWrapper(engine).eval()

    # Dataset
    from datasets import load_dataset
    if args.dataset_id.upper() == "MATH-500":
        ds = load_math500(HF_CACHE_DIR, args.split, args.seed)
        dataset_name_for_log = "MATH-500"
    else:
        ds = load_dataset(args.dataset_id, split=args.split, cache_dir=HF_CACHE_DIR)
        dataset_name_for_log = args.dataset_id

    if args.num_examples is not None and args.num_examples > 0:
        ds = ds.select(range(min(args.num_examples, len(ds))))

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Model: %s | dtype-hint=%s | DS tag=%s", args.model_name_or_path, args.dtype, ds_tag)
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
