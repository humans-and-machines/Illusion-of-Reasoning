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

import argparse
import itertools
import json
import logging
import os
import re
import string
import sys
import zipfile
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    # Package execution: src.utils.check
    from .check_helpers import (
        BanEosForSteps,
        StopOnGeneratedSubstring,
        _concat_anchor,
        _decode_generated_only,
        _generate,
        _sanitize_generation_config,
        _token_entropy_stream,
        _token_logprobs_stream,
    )
except ImportError:  # pragma: no cover - running as a stand-alone script
    # Script execution: python check.py from src/utils directory
    from check_helpers import (  # type: ignore[import-error]
        BanEosForSteps,
        StopOnGeneratedSubstring,
        _concat_anchor,
        _decode_generated_only,
        _generate,
        _sanitize_generation_config,
        _token_entropy_stream,
        _token_logprobs_stream,
    )

logger = logging.getLogger(__name__)

# ─────────────────────────── Normalization & extraction ─────────────────────────── #

_PUNCT_RE = re.compile(rf"[{re.escape(string.punctuation)}]")


def normalise(ans: str) -> str:
    """Normalise an answer string for comparison."""
    return _PUNCT_RE.sub("", ans).replace(" ", "").upper()


# training tag patterns
_ANS_PAT = re.compile(r"(?si)<answer>\s*([^<\n]+?)\s*</answer>")
_CAPS_FALLBACK = re.compile(r"\b[A-Z]{2,}\b")


def extract_answer_last(txt: str) -> Optional[str]:
    """Extract the last <answer>...</answer> span from text."""
    last_match = None
    for match in _ANS_PAT.finditer(txt or ""):
        last_match = match
    return last_match.group(1).strip() if last_match else None


def extract_answer_with_fallback(txt: str) -> Optional[str]:
    """Extract answer span, falling back to final all-caps token."""
    match = extract_answer_last(txt)
    if match:
        return match
    caps = _CAPS_FALLBACK.findall(txt or "")
    return caps[-1] if caps else None


# ─────────────────────────── Dataset loading (HF or local zip) ─────────────────────────── #


def _rows_from_jsonl_bytes(buf: bytes) -> List[dict]:
    """Parse a JSONL byte buffer into rows with normalised answers."""
    rows: List[dict] = []
    for line in buf.splitlines():
        text = line.decode("utf-8").strip()
        if not text:
            continue
        obj = json.loads(text)
        raw_ans = obj.get("answer") or obj.get("solution") or obj.get("soln")
        if raw_ans is None:
            raise ValueError(
                f"Missing answer key in jsonl row: {list(obj.keys())}"
            )
        rows.append(
            {
                "problem": f"{obj['clue']}  \n<think>",
                "answer": normalise(raw_ans),
            }
        )
    return rows


def _import_datasets_module():
    """Import the optional 'datasets' package at runtime."""
    try:
        return import_module("datasets")
    except ImportError as exc:  # pragma: no cover - runtime-only dependency
        msg = (
            "The 'datasets' package is required to load the Cryptonite dataset. "
            "Install it with `pip install datasets`."
        )
        raise RuntimeError(msg) from exc


def _build_dataset_from_parts(
    datasets_mod: Any,
    parts: Dict[str, List[dict]],
    split: str,
) -> Any:
    """Construct a DatasetDict from pre-parsed JSONL parts and return the desired split."""
    dataset_dict = datasets_mod.DatasetDict(
        {
            key: datasets_mod.Dataset.from_list(value)
            for key, value in parts.items()
        }
    )
    return dataset_dict[split]


def load_cryptonite_split(
    dataset_name: Optional[str],
    split: str,
    cryptonite_zip: Optional[Path],
) -> Any:
    """Load the requested split either from HF datasets or a local ZIP."""
    datasets_mod = _import_datasets_module()

    if cryptonite_zip:
        zip_path = Path(cryptonite_zip)
        if not zip_path.exists():
            raise FileNotFoundError(zip_path)

        with zipfile.ZipFile(zip_path) as zip_file:
            members = [m for m in zip_file.namelist() if m.endswith(".jsonl")]
            if not members:
                raise ValueError("ZIP contains no .jsonl files")

            parts: Dict[str, List[dict]] = {}
            for member in members:
                blob = zip_file.read(member)
                if "train" in member:
                    parts["train"] = _rows_from_jsonl_bytes(blob)
                elif "valid" in member or "val" in member:
                    parts["validation"] = _rows_from_jsonl_bytes(blob)
                elif "test" in member:
                    parts["test"] = _rows_from_jsonl_bytes(blob)

            if "validation" not in parts and "train" in parts:
                train_rows = parts["train"]
                nval = max(1, int(0.10 * len(train_rows)))
                parts["validation"] = train_rows[-nval:]
                parts["train"] = train_rows[:-nval]

            return _build_dataset_from_parts(datasets_mod, parts, split)

    if not dataset_name:
        msg = "Either --cryptonite_zip or --dataset_name must be provided."
        raise ValueError(msg)

    return datasets_mod.load_dataset(dataset_name, split=split)

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
    """Build training-format messages for a single crossword clue."""
    enumeration = len(gold_answer_norm)
    return [
        {"role": "system", "content": TRAIN_SYSTEM_PROMPT},
        {"role": "user", "content": f"{clue_with_think.strip()} ({enumeration})"},
    ]

# ─────────────────────────── Main ─────────────────────────── #


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for crossword evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument(
        "--dataset_name",
        default="od2961/Guardian-cryptonite-official-split",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--cryptonite_zip", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cryptonite_val_preds.jsonl"),
    )
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Attempts per clue; uses sampling when >1",
    )
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--min_new_tokens", type=int, default=0)
    # Unused with chat template; kept for compatibility with older runs.
    parser.add_argument("--max_prompt_tokens", type=int, default=768)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)

    # Stability / guidance
    parser.add_argument(
        "--anchor_think",
        action="store_true",
        help="Prepend assistant with '<think>\\n' as tokens.",
    )
    parser.add_argument(
        "--ban_eos_steps",
        type=int,
        default=0,
        help="Disallow EOS for first N steps.",
    )
    parser.add_argument(
        "--force_answer",
        action="store_true",
        help=(
            "Second-stage patch if no </answer> "
            "(only used when num_samples==1)."
        ),
    )

    # Sampling knobs (used when num_samples>1, or in fallback sampling)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument(
        "--fallback_sampling",
        action="store_true",
        help="For single-sample mode only.",
    )
    return parser.parse_args()


def _setup_logging() -> None:
    """Configure basic logging to stdout."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)


def _load_libraries() -> tuple[Any, Any, Any, Any]:
    """Dynamically import torch and transformers classes."""
    torch_mod = import_module("torch")
    transformers_mod = import_module("transformers")

    auto_tokenizer_cls = getattr(transformers_mod, "AutoTokenizer")
    auto_model_cls = getattr(transformers_mod, "AutoModelForCausalLM")
    stopping_criteria_list_cls = getattr(transformers_mod, "StoppingCriteriaList")

    return torch_mod, auto_tokenizer_cls, auto_model_cls, stopping_criteria_list_cls


def main() -> None:
    """Entry point for crossword evaluation."""
    args = _parse_args()
    _setup_logging()
    torch_mod, auto_tokenizer_cls, auto_model_cls, stopping_criteria_list_cls = _load_libraries()

    # Repro/perf
    torch_mod.manual_seed(args.seed)
    torch_mod.backends.cuda.matmul.allow_tf32 = True
    try:
        torch_mod.set_float32_matmul_precision("high")
    except (AttributeError, TypeError, RuntimeError):
        # Older torch versions or CPU-only builds may not support this.
        pass

    # Dtype & device map
    dtype_map = {
        "bfloat16": torch_mod.bfloat16,
        "float16": torch_mod.float16,
        "float32": torch_mod.float32,
    }
    dtype = dtype_map[args.dtype]
    device_map = "auto" if torch_mod.cuda.is_available() else None

    # Tokenizer / model (local only to avoid Hub repo-id confusion)
    logger.info("Loading tokenizer from %s …", args.model_path)
    tok = auto_tokenizer_cls.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"

    logger.info("Loading model from %s …", args.model_path)
    model = auto_model_cls.from_pretrained(
        str(args.model_path),
        torch_dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
        device_map=device_map,
        local_files_only=True,
    )
    if hasattr(model.config, "sliding_window"):
        model.config.sliding_window = None
    _sanitize_generation_config(model)
    model.config.forced_eos_token_id = None
    model.config.forced_bos_token_id = None
    model.eval()

    # Make EOS robust for Qwen chat (<|im_end|>)
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [tok.eos_token_id]
    if im_end_id is not None and im_end_id != -1:
        eos_ids.append(im_end_id)

    # Dataset
    logger.info("Loading dataset …")
    dataset = load_cryptonite_split(
        args.dataset_name,
        args.split,
        args.cryptonite_zip,
    )
    n_total = len(dataset)
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, n_total)))
        n_total = len(dataset)
    logger.info("Loaded split '%s' with %d examples.", args.split, n_total)

    # Output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Stats
    correct = 0
    seen = 0
    n_no_answer = 0
    n_nonmatch = 0
    printed_examples = 0
    num_samples = max(1, int(args.num_samples))

    # Main loop
    with out_path.open("a", encoding="utf-8", buffering=1) as fout:
        for start in range(0, n_total, args.batch_size):
            batch = dataset.select(range(start, min(n_total, start + args.batch_size)))
            batch_size = len(batch)

            # Build dialogs and tokenize via chat template (IDs, not strings)
            dialogs = [
                build_messages(row["problem"], row["answer"]) for row in batch
            ]
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
                attention = enc.get(
                    "attention_mask",
                    torch_mod.ones_like(input_ids),
                ).to(model.device)
            elif isinstance(enc, torch_mod.Tensor):
                input_ids = enc.to(model.device)
                if tok.pad_token_id is not None:
                    attention = (input_ids != tok.pad_token_id).long()
                else:
                    attention = torch_mod.ones_like(input_ids)
            else:
                raise TypeError(f"Unexpected encoding type: {type(enc)}")

            # Optional assistant anchor as PROMPT tokens
            anchor_text = "<think>\n" if args.anchor_think else None
            input_ids, attention = _concat_anchor(
                torch_mod,
                tok,
                input_ids,
                attention,
                anchor_text,
            )

            # Prompt lengths AFTER anchor (per original row), then repeat for samples
            prompt_lens_per_row = attention.sum(dim=-1).tolist()  # [batch_size]
            prompt_lens: List[int] = []
            for prompt_length in prompt_lens_per_row:
                prompt_lens.extend([prompt_length] * num_samples)

            # Batch-safe stopper (generated span only), compatible with repeated samples
            stops = stopping_criteria_list_cls(
                [StopOnGeneratedSubstring(tok, prompt_lens, "</answer>")]
            )

            # Decide sampling vs greedy:
            do_sample = num_samples > 1
            num_beams = (
                1 if do_sample else args.num_beams
            )  # keep simple when sampling

            # Primary generation (possibly multi-sample)
            sequences, scores, in_lens = _generate(
                torch_mod,
                model,
                tok,
                input_ids,
                attention,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                num_beams=num_beams,
                stop_criteria=stops,
                eos_token_ids=eos_ids,
                num_return_sequences=num_samples,
                ban_eos_steps=args.ban_eos_steps,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                top_p=(
                    args.top_p
                    if (do_sample and args.top_p not in (None, 0))
                    else None
                ),
                top_k=(
                    args.top_k
                    if (do_sample and args.top_k not in (None, 0))
                    else None
                ),
            )
            texts = _decode_generated_only(tok, sequences, in_lens)
            logsum, logavg, genlens = _token_logprobs_stream(
                torch_mod,
                scores,
                sequences,
                in_lens,
                eos_ids=eos_ids,
            )
            entavg, _ = _token_entropy_stream(
                torch_mod,
                scores,
                sequences,
                in_lens,
                eos_ids=eos_ids,
            )

            # Group by original row
            assert len(texts) == batch_size * num_samples, (
                len(texts),
                batch_size,
                num_samples,
            )
            grouped = [
                texts[index * num_samples : (index + 1) * num_samples]
                for index in range(batch_size)
            ]
            grouped_logsum = [
                logsum[index * num_samples : (index + 1) * num_samples]
                for index in range(batch_size)
            ]
            grouped_logavg = [
                logavg[index * num_samples : (index + 1) * num_samples]
                for index in range(batch_size)
            ]
            grouped_genlen = [
                genlens[index * num_samples : (index + 1) * num_samples]
                for index in range(batch_size)
            ]
            grouped_entavg = [
                entavg[index * num_samples : (index + 1) * num_samples]
                for index in range(batch_size)
            ]

            # Single-sample fallbacks (only if num_samples == 1)
            # If you really want to re-try subsets when num_samples>1,
            # add a loop here; typically unnecessary.
            if num_samples == 1:
                need_retry = []
                for index, text in enumerate(texts):
                    if extract_answer_last(text):
                        continue
                    need_retry.append(index)

                # Fallback 1: sampling
                if need_retry and args.fallback_sampling:
                    ids_sub = input_ids[need_retry]
                    att_sub = attention[need_retry]
                    # For stopper with repeated samples=1, lens is just per row
                    pl_sub = att_sub.sum(dim=-1).tolist()
                    stops_sub = stopping_criteria_list_cls(
                        [StopOnGeneratedSubstring(tok, pl_sub, "</answer>")]
                    )

                    seq2, sc2, il2 = _generate(
                        torch_mod,
                        model,
                        tok,
                        ids_sub,
                        att_sub,
                        max_new_tokens=args.max_new_tokens,
                        min_new_tokens=max(args.min_new_tokens, 12),
                        num_beams=1,
                        stop_criteria=stops_sub,
                        eos_token_ids=eos_ids,
                        num_return_sequences=1,
                        ban_eos_steps=max(0, args.ban_eos_steps // 2),
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )
                    txt2 = _decode_generated_only(tok, seq2, il2)
                    lsum2, lavg2, glen2 = _token_logprobs_stream(
                        torch_mod,
                        sc2,
                        seq2,
                        il2,
                        eos_ids=eos_ids,
                    )
                    ent2, _ = _token_entropy_stream(
                        torch_mod,
                        sc2,
                        seq2,
                        il2,
                        eos_ids=eos_ids,
                    )
                    # splice back
                    for local_index, batch_index in enumerate(need_retry):
                        grouped[batch_index] = [txt2[local_index]]
                        grouped_logsum[batch_index] = [lsum2[local_index]]
                        grouped_logavg[batch_index] = [lavg2[local_index]]
                        grouped_genlen[batch_index] = [glen2[local_index]]
                        grouped_entavg[batch_index] = [ent2[local_index]]

                # Fallback 2: force answer scaffold
                need_retry = [
                    batch_index
                    for batch_index in range(batch_size)
                    if not extract_answer_last(grouped[batch_index][0])
                ]
                if need_retry and args.force_answer:
                    scaffold = "\n</think>\n<answer>\n"
                    dialogs_sub = [dialogs[index] for index in need_retry]
                    enc2 = tok.apply_chat_template(
                        dialogs_sub,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        padding=True,
                    )
                    if hasattr(enc2, "input_ids"):
                        ids2 = enc2["input_ids"].to(model.device)
                        att2 = enc2.get(
                            "attention_mask",
                            torch_mod.ones_like(ids2),
                        ).to(model.device)
                    else:
                        ids2 = enc2.to(model.device)
                        if tok.pad_token_id is not None:
                            att2 = (ids2 != tok.pad_token_id).long()
                        else:
                            att2 = torch_mod.ones_like(ids2)

                    ids2, att2 = _concat_anchor(torch_mod, tok, ids2, att2, anchor_text)
                    ids2, att2 = _concat_anchor(torch_mod, tok, ids2, att2, scaffold)

                    pl2 = att2.sum(dim=-1).tolist()
                    stops2 = stopping_criteria_list_cls(
                        [StopOnGeneratedSubstring(tok, pl2, "</answer>")]
                    )

                    seq3, sc3, il3 = _generate(
                        torch_mod,
                        model,
                        tok,
                        ids2,
                        att2,
                        max_new_tokens=max(
                            32,
                            min(128, args.max_new_tokens // 3),
                        ),
                        min_new_tokens=8,
                        num_beams=1,
                        stop_criteria=stops2,
                        eos_token_ids=eos_ids,
                        num_return_sequences=1,
                        ban_eos_steps=4,
                        do_sample=False,
                    )
                    txt3 = _decode_generated_only(tok, seq3, il3)
                    lsum3, lavg3, glen3 = _token_logprobs_stream(
                        torch_mod,
                        sc3,
                        seq3,
                        il3,
                        eos_ids=eos_ids,
                    )
                    ent3, _ = _token_entropy_stream(
                        torch_mod,
                        sc3,
                        seq3,
                        il3,
                        eos_ids=eos_ids,
                    )
                    for local_index, batch_index in enumerate(need_retry):
                        grouped[batch_index] = [txt3[local_index]]
                        grouped_logsum[batch_index] = [lsum3[local_index]]
                        grouped_logavg[batch_index] = [lavg3[local_index]]
                        grouped_genlen[batch_index] = [glen3[local_index]]
                        grouped_entavg[batch_index] = [ent3[local_index]]

            # Per-row scoring + write
            for row_index, row in enumerate(batch):
                gold_norm = row["answer"]
                samples: List[Dict[str, Union[int, str, float, bool]]] = []
                n_answered = 0
                n_correct_here = 0

                for sample_index in range(len(grouped[row_index])):  # num_samples or 1
                    text = grouped[row_index][sample_index]
                    raw_pred = extract_answer_with_fallback(text) or ""
                    pred_norm = normalise(raw_pred) if raw_pred else ""
                    is_correct = pred_norm == gold_norm
                    if raw_pred:
                        n_answered += 1
                    if is_correct:
                        n_correct_here += 1

                    samples.append(
                        {
                            "k": sample_index,
                            "pred_text": text,
                            "pred_answer_raw": raw_pred,
                            "pred_answer_norm": pred_norm,
                            "is_correct": bool(is_correct),
                            "gen_len": grouped_genlen[row_index][sample_index],
                            "logprob_sum": grouped_logsum[row_index][sample_index],
                            "logprob_avg": grouped_logavg[row_index][sample_index],
                            "entropy_avg": grouped_entavg[row_index][sample_index],
                        }
                    )

                seen += 1
                any_correct = n_correct_here > 0
                if n_answered == 0:
                    n_no_answer += 1
                elif not any_correct:
                    n_nonmatch += 1
                else:
                    correct += 1

                payload = {
                    "idx": start + row_index,
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
                    print(
                        "EXTRACTED (s0):",
                        samples[0]["pred_answer_raw"] if samples else "None",
                    )
                    print("--------------------\n")
                    printed_examples += 1

            # flush/fsync per batch
            fout.flush()
            try:
                os.fsync(fout.fileno())
            except OSError:
                # Best-effort; not critical for correctness.
                pass

            # Aggregate batch stats
            # Average logprob across all sequences this batch
            flat_logavg = list(itertools.chain.from_iterable(grouped_logavg))
            finite_logavg = [
                value
                for value in flat_logavg
                if not torch_mod.isnan(torch_mod.tensor(value))
                and value != float("-inf")
            ]
            if finite_logavg:
                batch_avg = sum(finite_logavg) / len(finite_logavg)
            else:
                batch_avg = 0.0

            accuracy = 100.0 * correct / max(1, seen)
            print(
                f"[{seen:5d}/{n_total}] acc={accuracy:5.2f}%  "
                f"(last batch avg logp={batch_avg:.3f})  "
                f"[no-ans: {n_no_answer}  nonmatch: {n_nonmatch}]   "
                f"(R={num_samples})",
                flush=True,
            )

    final_accuracy = 100.0 * correct / max(1, seen)
    print(
        f"Done. Wrote {seen} rows → {out_path}  "
        f"(final acc={final_accuracy:.2f}%).",
    )


if __name__ == "__main__":
    main()
