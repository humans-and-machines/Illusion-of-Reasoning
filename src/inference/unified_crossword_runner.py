#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified launcher for the cryptic crossword inference loop.

This reuses the existing crossword-inference implementation but standardizes
model/tokenizer loading through the shared HF backend.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

from src.inference.backends import HFBackend


def _load_crossword_module():
    path = Path(__file__).with_name("crossword-inference.py")
    spec = importlib.util.spec_from_file_location("crossword_inference_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load crossword-inference module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--output_dir", required=True)

    # Data
    ap.add_argument(
        "--dataset_id",
        default="CROSSWORD-LOCAL",
        help="Use 'CROSSWORD-LOCAL' for local JSONL, or a HF path if available.",
    )
    ap.add_argument("--dataset_path", type=str, required=False, help="Path to local JSONL with clue/answer/enumeration.")
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

    # Entropy + attention impl
    ap.add_argument("--entropy_mode", choices=["full", "reconsider", "none"], default="reconsider")

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
    cw = _load_crossword_module()
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

    if args.dataset_id.upper() == "CROSSWORD-LOCAL":
        if not args.dataset_path:
            raise ValueError("--dataset_path is required when dataset_id=CROSSWORD-LOCAL")
        ds = cw.load_crossword_local(args.dataset_path)
        dataset_name_for_log = f"CROSSWORD-LOCAL:{os.path.basename(args.dataset_path)}"
    else:
        from datasets import load_dataset

        ds = load_dataset(args.dataset_id, split=args.split, cache_dir=HF_CACHE_DIR)
        dataset_name_for_log = args.dataset_id

    if args.num_examples is not None and args.num_examples > 0:
        ds = ds.select(range(min(args.num_examples, len(ds))))

    print(
        f"Model: {args.model_name_or_path} @ {args.revision} | dtype={args.dtype}\n"
        f"Dataset: {dataset_name_for_log} split={args.split} | N={len(ds)}\n"
        f"Output dir: {args.output_dir}"
    )

    cw.run_inference_on_split(
        split_name=args.split,
        examples=ds,
        tokenizer=backend.tokenizer,
        model=backend.model,
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
    print(f"All inference complete → {args.output_dir}")


if __name__ == "__main__":
    main()
