#!/usr/bin/env python3
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Deterministic evaluation for Qwen2.5-1.5B Open-R1 GRPO crossword checkpoint on
Cryptonite ‘validation’. Mirrors the training message format and extracts the
last <answer>…</answer> from the generated **assistant** text.

New:
• --num_samples=N → N attempts per clue (accuracy counts as correct if ANY sample is correct).
• Per-sample average token entropy recorded.

Key stability features
──────────────────────
• SDPA attention; disable sliding_window everywhere.
• Greedy/beam primary path; optional sampling when num_samples>1.
• Optional assistant anchor injected as **prompt tokens** (not generated).
• Optional EOS ban for first N decode steps to avoid instant <|im_end|>.
• Two-stage “force answer” patch (kept for single-sample mode).
• Batch-safe stopper that looks only at the **generated span**.

Usage
-----
python check.py \
  --model_path /n/fs/similarity/.../checkpoint-1000 \
  --dataset_name od2961/Guardian-cryptonite-official-split \
  --split validation \
  --out cryptonite_val_preds.jsonl \
  --batch_size 6 \
  --num_samples 8 \
  --max_new_tokens 300 \
  --anchor_think \
  --ban_eos_steps 16 \
  --temperature 0.7 --top_p 0.9
"""

from __future__ import annotations

import argparse, json, logging, os, re, string, sys, zipfile, itertools
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria, StoppingCriteriaList, LogitsProcessor
)

logger = logging.getLogger(__name__)

# ─────────────────────────── Normalization & extraction ─────────────────────────── #

_PUNCT_RE = re.compile(rf"[{re.escape(string.punctuation)}]")

def normalise(ans: str) -> str:
    return _PUNCT_RE.sub("", ans).replace(" ", "").upper()

# training tag patterns
_ANS_PAT = re.compile(r"(?si)<answer>\s*([^<\n]+?)\s*</answer>")
_CAPS_FALLBACK = re.compile(r"\b[A-Z]{2,}\b")

def extract_answer_last(txt: str) -> Optional[str]:
    last = None
    for m in _ANS_PAT.finditer(txt or ""):
        last = m
    return last.group(1).strip() if last else None

def extract_answer_with_fallback(txt: str) -> Optional[str]:
    m = extract_answer_last(txt)
    if m: return m
    caps = _CAPS_FALLBACK.findall(txt or "")
    return caps[-1] if caps else None

# ─────────────────────────── Dataset loading (HF or local zip) ─────────────────────────── #

def _rows_from_jsonl_bytes(buf: bytes) -> List[dict]:
    rows = []
    for line in buf.splitlines():
        s = line.decode("utf-8").strip()
        if not s: continue
        obj = json.loads(s)
        raw_ans = obj.get("answer") or obj.get("solution") or obj.get("soln")
        if raw_ans is None:
            raise ValueError(f"Missing answer key in jsonl row: {list(obj.keys())}")
        rows.append({"problem": f"{obj['clue']}  \n<think>", "answer": normalise(raw_ans)})
    return rows

def load_cryptonite_split(dataset_name: Optional[str], split: str, cryptonite_zip: Optional[Path]) -> Dataset:
    if cryptonite_zip:
        z = Path(cryptonite_zip)
        if not z.exists(): raise FileNotFoundError(z)
        with zipfile.ZipFile(z) as zf:
            members = [m for m in zf.namelist() if m.endswith(".jsonl")]
            if not members: raise ValueError("ZIP contains no .jsonl files")
            parts: Dict[str, List[dict]] = {}
            for m in members:
                blob = zf.read(m)
                if "train" in m: parts["train"] = _rows_from_jsonl_bytes(blob)
                elif "valid" in m or "val" in m: parts["validation"] = _rows_from_jsonl_bytes(blob)
                elif "test" in m: parts["test"] = _rows_from_jsonl_bytes(blob)
            if "validation" not in parts and "train" in parts:
                rows = parts["train"]; nval = max(1, int(0.10 * len(rows)))
                parts["validation"] = rows[-nval:]; parts["train"] = rows[:-nval]
            ds = DatasetDict({k: Dataset.from_list(v) for k, v in parts.items()})
            return ds[split]
    if not dataset_name:
        raise ValueError("Either --cryptonite_zip or --dataset_name must be provided.")
    return load_dataset(dataset_name, split=split)

# ─────────────────────────── Prompt building (training-identical) ─────────────────────────── #

TRAIN_SYSTEM_PROMPT = """You are an expert *cryptic-crossword solver*.

Every time you receive a clue you must:
• Analyse it thoroughly.  
  – Pinpoint the **definition**.  
  – Pinpoint the **word-play** (anagram, container, reversal, homophone, charade, etc.).  
  – Write out the full derivation that turns the word-play into the answer.  

• Check that the answer’s length matches the enumeration in brackets.

• Respond in **exactly** the tag-based format shown below – no greeting, no commentary outside the tags.  
  – The final answer goes inside `<answer>` in UPPER-CASE.  
  – Never reveal the answer elsewhere.

------------------------------------------------------------
TAG TEMPLATE (copy this shape for every clue)
<think>
YOUR reasoning process goes here:  
1. quote the relevant bits of the clue  
2. name the cryptic device(s) you apply  
3. show each intermediate step until the answer is reached  

If you spot an error with the answer reached, iterate, repeating steps 1-3 as many
times as necessary until you are confident in your answer.
</think>
<answer>
THEANSWER        ← all-caps; length must match enumeration
</answer>"""

def build_messages(clue_with_think: str, gold_answer_norm: str) -> List[dict]:
    enum = len(gold_answer_norm)
    return [
        {"role": "system", "content": TRAIN_SYSTEM_PROMPT},
        {"role": "user", "content": f"{clue_with_think.strip()} ({enum})"},
    ]

# ─────────────────────────── Stopping / logits processors ─────────────────────────── #

class StopOnGeneratedSubstring(StoppingCriteria):
    """
    Batch-safe stopper that only scans **generated** tokens.
    Accepts per-row prompt lengths (already including anchor) and repeats are allowed.
    """
    def __init__(self, tok, prompt_lens: List[int], substr: str):
        self.tok = tok
        self.substr = substr
        self.prompt_lens = list(prompt_lens)  # length must equal current batch dimension
        self.done = [False] * len(self.prompt_lens)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor, **kwargs) -> bool:
        B = input_ids.size(0)
        for i in range(B):
            if self.done[i]:
                continue
            L0 = int(self.prompt_lens[i])
            if input_ids.size(1) <= L0:
                continue
            gen_ids = input_ids[i, L0:]
            text = self.tok.decode(gen_ids.tolist(), skip_special_tokens=True)
            if self.substr in text:
                self.done[i] = True
        return all(self.done)

class BanEosForSteps(LogitsProcessor):
    """Mask EOS ids for the first N steps of generation (batch-wide)."""
    def __init__(self, eos_token_ids: Union[int, List[int]], ban_steps: int):
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        self.eos_ids = [e for e in eos_token_ids if e is not None and e >= 0]
        self.ban_steps = max(0, int(ban_steps))
        self.step = 0
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.step < self.ban_steps and self.eos_ids:
            for e in self.eos_ids:
                if e < scores.size(-1):
                    scores[:, e] = -float("inf")
        self.step += 1
        return scores

# ─────────────────────────── Generation helpers ─────────────────────────── #

def _sanitize_generation_config(model):
    gc = model.generation_config
    gc.do_sample = False
    gc.temperature = None
    gc.top_p = None
    gc.top_k = None
    if hasattr(model.config, "sliding_window"):
        model.config.sliding_window = None
    if hasattr(gc, "sliding_window"):
        gc.sliding_window = None

def _concat_anchor(tok, input_ids, attention_mask, anchor_text: Optional[str]):
    """Append an assistant anchor as **prompt tokens** (not generated)."""
    if not anchor_text:
        return input_ids, attention_mask
    anchor_ids = tok(anchor_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(input_ids.device)
    B = input_ids.size(0)
    anc = anchor_ids.expand(B, -1)
    new_ids = torch.cat([input_ids, anc], dim=1)
    new_mask = torch.cat([attention_mask, torch.ones_like(anc)], dim=1)
    return new_ids, new_mask

def _generate(
    model, tok, input_ids, attention_mask,
    max_new_tokens: int,
    num_beams: int,
    min_new_tokens: int,
    stop_criteria: Optional[StoppingCriteriaList],
    eos_token_ids: Union[int, List[int]],
    num_return_sequences: int = 1,
    ban_eos_steps: int = 0,
    do_sample: bool = False,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
):
    logits_proc = []
    if ban_eos_steps:
        logits_proc.append(BanEosForSteps(eos_token_ids, ban_eos_steps))
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens if min_new_tokens and min_new_tokens > 0 else None,
            num_beams=num_beams if (num_beams and num_beams > 1 and not do_sample) else 1,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if (do_sample and top_p not in (None, 0)) else None,
            top_k=top_k if (do_sample and top_k not in (None, 0)) else None,
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tok.pad_token_id,
            eos_token_id=eos_token_ids,
            use_cache=True,
            stopping_criteria=stop_criteria,
            logits_processor=logits_proc if logits_proc else None,
        )
    # HF duplicates the batch internally for num_return_sequences; mirror that for lens
    input_lens = attention_mask.sum(dim=-1)
    if num_return_sequences and num_return_sequences > 1:
        input_lens = input_lens.repeat_interleave(num_return_sequences)
    return out.sequences, out.scores, input_lens

def _decode_generated_only(tok, sequences, input_lens) -> List[str]:
    texts = []
    for i in range(sequences.size(0)):
        L0 = int(input_lens[i].item())
        gen_ids = sequences[i, L0:]
        texts.append(tok.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    return texts

def _first_eos_pos(tensor_row: torch.Tensor, eos_ids: Optional[List[int]]) -> Optional[int]:
    if eos_ids is None or tensor_row.numel() == 0: return None
    pos = None
    for e in eos_ids:
        hits = (tensor_row == e).nonzero(as_tuple=False)
        if hits.numel() > 0:
            p = int(hits[0].item()) + 1  # include eos token
            pos = p if pos is None else min(pos, p)
    return pos

def _token_logprobs_stream(scores, sequences, input_lens, eos_ids: Optional[List[int]] = None):
    import torch.nn.functional as F
    B = sequences.size(0)
    T = len(scores)
    sums = [0.0]*B; lens = [0]*B
    if T == 0:
        return sums, [0.0]*B, lens
    # effective lengths (truncate on first EOS inside generated span)
    L_eff = []
    for i in range(B):
        L0 = int(input_lens[i].item())
        gen = sequences[i, L0:]
        Li = min(gen.size(0), T)
        if Li > 0:
            p = _first_eos_pos(gen[:Li], eos_ids)
            if p is not None:
                Li = min(Li, p)
        L_eff.append(Li)
    maxL = max(L_eff) if L_eff else 0
    for t in range(maxL):
        active = [i for i, L in enumerate(L_eff) if t < L]
        if not active: break
        tok_ids = []
        for i in active:
            L0 = int(input_lens[i].item())
            tok_ids.append(int(sequences[i, L0+t].item()))
        tok_ids = torch.tensor(tok_ids, device=scores[t].device, dtype=torch.long)
        step = torch.log_softmax(scores[t][active].float(), dim=-1)
        picked = step.gather(1, tok_ids.view(-1,1)).squeeze(1)
        for j, i in enumerate(active):
            val = float(picked[j].item())
            if val == val and val != float("-inf"):  # finite
                sums[i] += val
                lens[i] += 1
        del tok_ids, step, picked
    avgs = [(s/l if l>0 else 0.0) for s,l in zip(sums, lens)]
    return sums, avgs, lens

def _token_entropy_stream(scores, sequences, input_lens, eos_ids: Optional[List[int]] = None):
    """Average token entropy per sequence, truncated at EOS."""
    import torch.nn.functional as F
    B = sequences.size(0)
    T = len(scores)
    sums = [0.0]*B; lens = [0]*B
    if T == 0:
        return [0.0]*B, lens
    # effective lengths (truncate on first EOS inside generated span)
    L_eff = []
    for i in range(B):
        L0 = int(input_lens[i].item())
        gen = sequences[i, L0:]
        Li = min(gen.size(0), T)
        if Li > 0:
            p = _first_eos_pos(gen[:Li], eos_ids)
            if p is not None:
                Li = min(Li, p)
        L_eff.append(Li)
    maxL = max(L_eff) if L_eff else 0
    for t in range(maxL):
        active = [i for i, L in enumerate(L_eff) if t < L]
        if not active: break
        step_logits = scores[t][active].float()
        probs = torch.softmax(step_logits, dim=-1)
        ent = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=-1)  # [len(active)]
        for j, i in enumerate(active):
            v = float(ent[j].item())
            if v == v:  # not NaN
                sums[i] += v
                lens[i] += 1
        del step_logits, probs, ent
    avgs = [(s/l if l>0 else 0.0) for s,l in zip(sums, lens)]
    return avgs, lens

# ─────────────────────────── Main ─────────────────────────── #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=Path, required=True)
    ap.add_argument("--dataset_name", default="od2961/Guardian-cryptonite-official-split")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--cryptonite_zip", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("cryptonite_val_preds.jsonl"))
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--num_samples", type=int, default=1, help="Attempts per clue; uses sampling when >1")
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--min_new_tokens", type=int, default=0)
    ap.add_argument("--max_prompt_tokens", type=int, default=768)  # unused with chat template; kept for compat
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--length_penalty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)

    # Stability / guidance
    ap.add_argument("--anchor_think", action="store_true", help="Prepend assistant with '<think>\\n' as tokens.")
    ap.add_argument("--ban_eos_steps", type=int, default=0, help="Disallow EOS for first N steps.")
    ap.add_argument("--force_answer", action="store_true", help="Second-stage patch if no </answer> (only used when num_samples==1).")

    # Sampling knobs (used when num_samples>1, or in fallback sampling)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--fallback_sampling", action="store_true", help="For single-sample mode only.")

    args = ap.parse_args()

    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    # Repro/perf
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass

    # Dtype & device map
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    device_map = "auto" if torch.cuda.is_available() else None

    # Tokenizer / model (local only to avoid Hub repo-id confusion)
    logger.info("Loading tokenizer from %s …", args.model_path)
    tok = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True, local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"

    logger.info("Loading model from %s …", args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
        device_map=device_map,
        local_files_only=True,
    )
    if hasattr(model.config, "sliding_window"): model.config.sliding_window = None
    _sanitize_generation_config(model)
    model.config.forced_eos_token_id = None
    model.config.forced_bos_token_id = None
    model.eval()

    # Make EOS robust for Qwen chat (<|im_end|>)
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [tok.eos_token_id] if (im_end_id is None or im_end_id == -1) else [tok.eos_token_id, im_end_id]

    # Dataset
    logger.info("Loading dataset …")
    ds = load_cryptonite_split(args.dataset_name, args.split, args.cryptonite_zip)
    n_total = len(ds)
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, n_total)))
        n_total = len(ds)
    logger.info("Loaded split '%s' with %d examples.", args.split, n_total)

    # Output
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = open(out_path, "a", encoding="utf-8", buffering=1)

    # Stats
    correct = seen = 0
    n_no_answer = n_nonmatch = 0
    printed_examples = 0
    R = max(1, int(args.num_samples))

    # Main loop
    for start in range(0, n_total, args.batch_size):
        batch = ds.select(range(start, min(n_total, start + args.batch_size)))
        B = len(batch)

        # Build dialogs and tokenize via chat template (IDs, not strings)
        dialogs = [build_messages(r["problem"], r["answer"]) for r in batch]
        enc = tok.apply_chat_template(
            dialogs,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
        )

        # Support both BatchEncoding and bare Tensor returns
        if hasattr(enc, "input_ids"):
            input_ids = enc["input_ids"].to(model.device)
            attention = enc.get("attention_mask", torch.ones_like(input_ids)).to(model.device)
        elif isinstance(enc, torch.Tensor):
            input_ids = enc.to(model.device)
            attention = (input_ids != tok.pad_token_id).long() if tok.pad_token_id is not None else torch.ones_like(input_ids)
        else:
            raise TypeError(f"Unexpected encoding type: {type(enc)}")

        # Optional assistant anchor as PROMPT tokens
        anchor_text = "<think>\n" if args.anchor_think else None
        input_ids, attention = _concat_anchor(tok, input_ids, attention, anchor_text)

        # Prompt lengths AFTER anchor (per original row), then repeat for samples
        prompt_lens_per_row = attention.sum(dim=-1).tolist()   # [B]
        prompt_lens = list(itertools.chain.from_iterable([[L]*R for L in prompt_lens_per_row]))  # [B*R]

        # Batch-safe stopper (generated span only), compatible with repeated samples
        stops = StoppingCriteriaList([StopOnGeneratedSubstring(tok, prompt_lens, "</answer>")])

        # Decide sampling vs greedy:
        do_sample = (R > 1)
        num_beams = 1 if do_sample else args.num_beams  # keep simple when sampling

        # Primary generation (possibly multi-sample)
        sequences, scores, in_lens = _generate(
            model, tok, input_ids, attention,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            num_beams=num_beams,
            stop_criteria=stops,
            eos_token_ids=eos_ids,
            num_return_sequences=R,
            ban_eos_steps=args.ban_eos_steps,
            do_sample=do_sample,
            temperature=(args.temperature if do_sample else None),
            top_p=(args.top_p if (do_sample and args.top_p not in (None, 0)) else None),
            top_k=(args.top_k if (do_sample and args.top_k not in (None, 0)) else None),
        )
        texts = _decode_generated_only(tok, sequences, in_lens)
        logsum, logavg, genlens = _token_logprobs_stream(scores, sequences, in_lens, eos_ids=eos_ids)
        entavg, _ = _token_entropy_stream(scores, sequences, in_lens, eos_ids=eos_ids)

        # Group by original row
        assert len(texts) == B * R, (len(texts), B, R)
        grouped = [texts[i*R:(i+1)*R] for i in range(B)]
        grouped_logsum = [logsum[i*R:(i+1)*R] for i in range(B)]
        grouped_logavg = [logavg[i*R:(i+1)*R] for i in range(B)]
        grouped_genlen = [genlens[i*R:(i+1)*R] for i in range(B)]
        grouped_entavg = [entavg[i*R:(i+1)*R] for i in range(B)]

        # Single-sample fallbacks (only if R==1)
        # If you really want to re-try subsets when R>1, add a loop here; typically unnecessary.
        if R == 1:
            need_retry = []
            for i, t in enumerate(texts):
                if extract_answer_last(t): continue
                need_retry.append(i)

            # Fallback 1: sampling
            if need_retry and args.fallback_sampling:
                ids_sub = input_ids[need_retry]
                att_sub = attention[need_retry]
                # For stopper with repeated samples=1, lens is just per row
                pl_sub = att_sub.sum(dim=-1).tolist()
                stops_sub = StoppingCriteriaList([StopOnGeneratedSubstring(tok, pl_sub, "</answer>")])

                seq2, sc2, il2 = _generate(
                    model, tok,
                    ids_sub, att_sub,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=max(args.min_new_tokens, 12),
                    num_beams=1,
                    stop_criteria=stops_sub,
                    eos_token_ids=eos_ids,
                    num_return_sequences=1,
                    ban_eos_steps=max(0, args.ban_eos_steps // 2),
                    do_sample=True,
                    temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                )
                txt2 = _decode_generated_only(tok, seq2, il2)
                lsum2, lavg2, glen2 = _token_logprobs_stream(sc2, seq2, il2, eos_ids=eos_ids)
                ent2, _ = _token_entropy_stream(sc2, seq2, il2, eos_ids=eos_ids)
                # splice back
                for j, bi in enumerate(need_retry):
                    grouped[bi] = [txt2[j]]
                    grouped_logsum[bi] = [lsum2[j]]
                    grouped_logavg[bi] = [lavg2[j]]
                    grouped_genlen[bi] = [glen2[j]]
                    grouped_entavg[bi] = [ent2[j]]

            # Fallback 2: force answer scaffold
            need_retry = [i for i in range(B) if not extract_answer_last(grouped[i][0])]
            if need_retry and args.force_answer:
                scaffold = "\n</think>\n<answer>\n"
                dialogs_sub = [dialogs[i] for i in need_retry]
                enc2 = tok.apply_chat_template(
                    dialogs_sub, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt", padding=True,
                )
                if hasattr(enc2, "input_ids"):
                    ids2 = enc2["input_ids"].to(model.device)
                    att2 = enc2.get("attention_mask", torch.ones_like(ids2)).to(model.device)
                else:
                    ids2 = enc2.to(model.device)
                    att2 = (ids2 != tok.pad_token_id).long() if tok.pad_token_id is not None else torch.ones_like(ids2)

                ids2, att2 = _concat_anchor(tok, ids2, att2, anchor_text)
                ids2, att2 = _concat_anchor(tok, ids2, att2, scaffold)

                pl2 = att2.sum(dim=-1).tolist()
                stops2 = StoppingCriteriaList([StopOnGeneratedSubstring(tok, pl2, "</answer>")])

                seq3, sc3, il3 = _generate(
                    model, tok, ids2, att2,
                    max_new_tokens=max(32, min(128, args.max_new_tokens//3)),
                    min_new_tokens=8,
                    num_beams=1,
                    stop_criteria=stops2,
                    eos_token_ids=eos_ids,
                    num_return_sequences=1,
                    ban_eos_steps=4,
                    do_sample=False
                )
                txt3 = _decode_generated_only(tok, seq3, il3)
                lsum3, lavg3, glen3 = _token_logprobs_stream(sc3, seq3, il3, eos_ids=eos_ids)
                ent3, _ = _token_entropy_stream(sc3, seq3, il3, eos_ids=eos_ids)
                for j, bi in enumerate(need_retry):
                    grouped[bi] = [txt3[j]]
                    grouped_logsum[bi] = [lsum3[j]]
                    grouped_logavg[bi] = [lavg3[j]]
                    grouped_genlen[bi] = [glen3[j]]
                    grouped_entavg[bi] = [ent3[j]]

        # Per-row scoring + write
        for i, row in enumerate(batch):
            gold_norm = row["answer"]
            samples = []
            n_answered = 0
            n_correct_here = 0

            for k in range(len(grouped[i])):  # R or 1
                text = grouped[i][k]
                raw_pred = extract_answer_with_fallback(text) or ""
                pred_norm = normalise(raw_pred) if raw_pred else ""
                ok = (pred_norm == gold_norm)
                if raw_pred: n_answered += 1
                if ok: n_correct_here += 1

                samples.append({
                    "k": k,
                    "pred_text": text,
                    "pred_answer_raw": raw_pred,
                    "pred_answer_norm": pred_norm,
                    "is_correct": bool(ok),
                    "gen_len": grouped_genlen[i][k],
                    "logprob_sum": grouped_logsum[i][k],
                    "logprob_avg": grouped_logavg[i][k],
                    "entropy_avg": grouped_entavg[i][k],
                })

            seen += 1
            any_correct = (n_correct_here > 0)
            if n_answered == 0: n_no_answer += 1
            elif not any_correct: n_nonmatch += 1
            else: correct += 1

            payload = {
                "idx": start + i,
                "clue": row["problem"].split("\n<think>")[0].strip(),
                "enum": len(gold_norm),
                "gold_answer": gold_norm,
                "num_samples": len(samples),
                "n_answered": n_answered,
                "n_correct": n_correct_here,
                "any_correct": bool(any_correct),
                "samples": samples,
            }
            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")

            if (n_answered == 0 or not any_correct) and printed_examples < 100:
                tail = (samples[0]["pred_text"] or "")[-300:]
                print("\n--- DEBUG SAMPLE ---")
                print("CLUE:", payload["clue"])
                print("ENUM:", payload["enum"], "GOLD:", gold_norm)
                print("GEN TAIL (s0):", tail if tail else "<empty>")
                print("EXTRACTED (s0):", samples[0]["pred_answer_raw"] if samples else "None")
                print("--------------------\n")
                printed_examples += 1

        # flush/fsync per batch
        fout.flush()
        try: os.fsync(fout.fileno())
        except Exception: pass

        # Aggregate batch stats
        # Average logprob across all sequences this batch
        flat_logavg = list(itertools.chain.from_iterable(grouped_logavg))
        batch_avg = (sum(a for a in flat_logavg if a == a and a != float("-inf")) /  # finite
                     max(1, sum(1 for a in flat_logavg if a == a and a != float("-inf"))))

        acc = 100.0 * correct / max(1, seen)
        print(f"[{seen:5d}/{n_total}] acc={acc:5.2f}%  (last batch avg logp={batch_avg:.3f})  "
              f"[no-ans: {n_no_answer}  nonmatch: {n_nonmatch}]   (R={R})",
              flush=True)

    fout.close()
    print(f"Done. Wrote {seen} rows → {out_path}  (final acc={100.0*correct/max(1,seen):.2f}%).")

if __name__ == "__main__":
    main()
