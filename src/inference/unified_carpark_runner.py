#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified launcher for the Rush Hour (car-park) inference loop.

This wraps the existing carpark-inference implementation but uses the shared
HF backend for model/tokenizer creation so all entrypoints can standardize on
one loading path. Resume/fill, two-pass, and analytics remain unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

from src.inference.backends import HFBackend


def _load_carpark_module():
    """Dynamically load the legacy carpark-inference.py as a module."""
    path = Path(__file__).with_name("carpark-inference.py")
    spec = importlib.util.spec_from_file_location("carpark_inference_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load carpark-inference module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod


def parse_args():
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
    ap.add_argument("--entropy_mode", choices=["full", "reconsider", "none"], default="reconsider")
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
    carpark = _load_carpark_module()
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

    # EOS IDs (reuse carpark logic)
    eos_ids = set()
    if backend.tokenizer.eos_token_id is not None:
        eos_ids.add(int(backend.tokenizer.eos_token_id))
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tid = backend.tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != backend.tokenizer.pad_token_id:
            eos_ids.add(int(tid))
    eos_ids = sorted(eos_ids) if eos_ids else None

    ds = carpark.load_rush_dataset(
        dataset_id=args.dataset_id,
        split=args.split,
        cache_dir=HF_CACHE_DIR,
        prompt_col=args.dataset_prompt_column,
        solution_col=args.dataset_solution_column,
    )
    if args.num_examples is not None and args.num_examples > 0:
        ds = ds.select(range(min(args.num_examples, len(ds))))

    carpark.run_inference_on_split(
        split_name=args.split,
        examples=ds,
        tokenizer=backend.tokenizer,
        model=backend.model,
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

    print(f"All inference complete → {args.output_dir}")


if __name__ == "__main__":
    main()
