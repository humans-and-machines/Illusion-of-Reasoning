#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference helper for Azure-hosted DeepSeek-R1 (open-source) on MATH-500.

Behavior:
- Single-pass generation with the same math system prompt used in GRPO runs.
- Writes JSONL to: {output_dir}/step{step:04d}_{split}.jsonl
- Resumable: if a JSONL already exists, only missing sample_idx entries are generated.
- Uses Azure OpenAI (Responses API if available; falls back to Chat Completions).

Usage (env-based auth):
  export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
  export AZURE_OPENAI_API_KEY="***"
  export AZURE_OPENAI_DEPLOYMENT="deepseek-r1"  # or your deployment name
  python -m src.inference.math-deepseek-azure --output_dir results/deepseek-r1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from src.annotate.config import load_azure_config
from src.annotate.llm_client import build_preferred_client
from src.inference.common import (
    canon_math as _canon_math,
    extract_blocks as _extract_blocks,
    load_local_json_dataset,
    valid_tag_structure as _valid_tag_structure,
)
from src.inference.task_registry import TASK_REGISTRY

try:
    from datasets import Dataset, load_dataset
except Exception as e:  # noqa: BLE001
    print("datasets is required: pip install datasets", file=sys.stderr)
    raise


LOGLEVEL = os.getenv("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOGLEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Registry hook (no behavior change; helps keep metadata centralized)
TASK_SPEC = TASK_REGISTRY.get("math-azure")

# ----------------------- Prompt + regex helpers -----------------------
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
  </think>
  <answer>
  THEANSWER
  </answer>
"""

# ----------------------- Dataset helpers -----------------------
def load_math500(cache_dir: str, split: str, seed: int, dataset_path: Optional[str] = None) -> Dataset:
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
            ds_full = load_dataset(repo, split=split, cache_dir=cache_dir)
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
            continue

    try:
        ds_full = load_dataset("hendrycks/competition_math", split="test", cache_dir=cache_dir)
        n = min(500, len(ds_full))
        return ds_full.shuffle(seed=seed).select(range(n))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Could not load MATH-500 or fallback dataset: {e}") from e


# ----------------------- I/O helpers -----------------------
def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False)
        f.write("\n")


def _scan_existing(path: str) -> Dict[str, set]:
    if not os.path.exists(path):
        return {}
    existing: Dict[str, set] = {}
    with open(path, "r", encoding="utf-8") as f:
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
            existing.setdefault(prob, set()).add(int(k))
    return existing


# ----------------------- Azure client + call -----------------------
def _make_client(args):
    cfg = load_azure_config()
    endpoint = (args.endpoint or cfg["endpoint"]).rstrip("/")
    deployment = args.deployment or cfg["deployment"]
    api_version = args.api_version or cfg["api_version"]
    api_key = args.api_key or os.getenv("AZURE_OPENAI_API_KEY", cfg.get("api_key", ""))
    if not api_key:
        raise RuntimeError("AZURE_OPENAI_API_KEY is required (env or --api_key).")
    client, uses_v1 = build_preferred_client(
        endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        use_v1=bool(args.use_v1),
    )
    return client, uses_v1, endpoint, deployment, api_version


def _call_model(
    client,
    uses_v1: bool,
    deployment: str,
    problem: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    request_timeout: int,
):
    if uses_v1 and hasattr(client, "responses"):
        resp = client.responses.create(
            model=deployment,
            instructions=SYSTEM_PROMPT,
            input=[{"role": "user", "content": problem}],
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            timeout=request_timeout,
        )
        text = getattr(resp, "output_text", None) or ""
        finish_reason = getattr(getattr(resp, "output", None), "finish_reason", None)
        usage = getattr(resp, "usage", None)
        return text, finish_reason, usage

    # Legacy Chat Completions
    resp = client.chat.completions.create(
        model=deployment,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_output_tokens,
        timeout=request_timeout,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ],
    )
    finish_reason = None
    if resp and getattr(resp, "choices", None):
        finish_reason = getattr(resp.choices[0], "finish_reason", None)
        text = resp.choices[0].message.content or ""
    else:
        text = ""
    usage = getattr(resp, "usage", None)
    return text, finish_reason, usage


# ----------------------- Main loop -----------------------
def main():
    default_cfg = load_azure_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True, help="Root directory for JSONL outputs.")
    ap.add_argument("--dataset_id", default="MATH-500", help="Use 'MATH-500' or a HF dataset path.")
    ap.add_argument("--dataset_path", default=None, help="Optional local JSONL for MATH-500-style records.")
    ap.add_argument("--split", default="test")
    ap.add_argument("--num_examples", type=int, default=None, help="Optional cap (<500).")
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_output_tokens", type=int, default=900)
    ap.add_argument("--request_timeout", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--step", type=int, default=0)

    # Azure params (env defaults from configs/azure.yml)
    ap.add_argument("--endpoint", default=default_cfg.get("endpoint"))
    ap.add_argument("--deployment", default=None, help="Azure deployment name (e.g., deepseek-r1).")
    ap.add_argument("--api_version", default=default_cfg.get("api_version"))
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--use_v1", type=int, default=int(default_cfg.get("use_v1", 1)),
                    help="1 → prefer Responses API (v1); 0 → force Chat Completions.")

    # Retry controls
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--retry_backoff", type=float, default=2.0)

    args = ap.parse_args()

    random.seed(args.seed)
    outpath = os.path.join(args.output_dir, f"step{args.step:04d}_{args.split}.jsonl")

    client, uses_v1, endpoint, deployment, api_version = _make_client(args)
    logger.info("Client ready | uses_v1=%s | endpoint=%s | deployment=%s", uses_v1, endpoint, deployment)

    HF_CACHE_DIR = os.path.abspath("./.hf_cache")
    if args.dataset_id.upper() == "MATH-500":
        ds = load_math500(HF_CACHE_DIR, args.split, args.seed, dataset_path=args.dataset_path)
        dataset_name_for_log = "MATH-500"
    else:
        ds = load_dataset(args.dataset_id, split=args.split, cache_dir=HF_CACHE_DIR)
        dataset_name_for_log = args.dataset_id

    if args.num_examples is not None and args.num_examples > 0:
        ds = ds.select(range(min(args.num_examples, len(ds))))

    ds = ds.shuffle(seed=args.seed)
    existing = _scan_existing(outpath)
    logger.info("Dataset: %s split=%s | N=%d | existing=%d", dataset_name_for_log, args.split, len(ds), len(existing))
    logger.info("Output: %s", outpath)

    total_new = 0
    for ex in ds:
        prob = ex.get("problem") or ex.get("question") or ex.get("prompt") or ex.get("instruction") or ex.get("query")
        gold = ex.get("answer") or ex.get("solution") or ex.get("final_answer") or ex.get("boxed_answer") or ex.get("target")
        if not prob or gold is None:
            continue
        canon_gold = _canon_math(gold)

        have = existing.get(prob, set())
        todo = [k for k in range(args.num_samples) if k not in have]
        if not todo:
            continue

        for k in todo:
            # retry loop
            attempt = 0
            while True:
                try:
                    text, finish_reason, usage = _call_model(
                        client=client,
                        uses_v1=uses_v1,
                        deployment=deployment,
                        problem=prob,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_output_tokens=args.max_output_tokens,
                        request_timeout=args.request_timeout,
                    )
                    break
                except Exception as e:  # noqa: BLE001
                    attempt += 1
                    if attempt > args.max_retries:
                        logger.error("Failed after %d retries on sample_idx=%d | prob snippet=%.60s | err=%r",
                                     attempt - 1, k, prob, e)
                        raise
                    sleep_dur = args.retry_backoff * attempt
                    logger.warning("Retry %d/%d for sample_idx=%d after error: %r (sleep %.1fs)",
                                   attempt, args.max_retries, k, e, sleep_dur)
                    time.sleep(sleep_dur)

            think, ans = _extract_blocks(text)
            pred_canon = _canon_math(ans)
            is_correct = bool(pred_canon and canon_gold and canon_gold in pred_canon)

            row = {
                "problem": prob,
                "gold_answer": gold,
                "gold_answer_canon": canon_gold,
                "split": args.split,
                "step": args.step,
                "sample_idx": k,
                "endpoint": endpoint,
                "deployment": deployment,
                "api_version": api_version,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "pass1": {
                    "output": text.strip(),
                    "pred_answer": ans,
                    "pred_answer_canon": pred_canon,
                    "is_correct_pred": is_correct,
                    "valid_tag_structure": _valid_tag_structure(text),
                    "finish_reason": finish_reason,
                },
            }

            if usage:
                try:
                    row["usage"] = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "total_tokens": getattr(usage, "total_tokens", None),
                    }
                except Exception:
                    pass

            _append_jsonl(outpath, row)
            total_new += 1
            existing.setdefault(prob, set()).add(k)

    logger.info("All done. Wrote %d new samples → %s", total_new, outpath)


if __name__ == "__main__":
    main()
