#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch inference for Qwen-2.5-7B checkpoints, emitting
  <think> … </think><answer> … </answer>
blocks that are easy to grade.

This revision **removes the bespoke StopOnTag stopping-criteria**.  We now let
 the model run until it either hits `max_new_tokens` or its own internal EOS
token.  The full chain-of-thought and answer tags are preserved in the output
JSON — no post-processing truncation.
"""

import os, sys, json, time, logging, argparse, re
from packaging import version
import torch

# —————————————————— logging ———————————————————
logging.basicConfig(
    level=getattr(logging, os.getenv("LOGLEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)
logger.info("Starting %s", os.path.basename(__file__))

# ——— PyTorch 2.6 DeepSpeed un-pickle patch ———
try:
    if version.parse(torch.__version__) >= version.parse("2.6.0"):
        from torch.serialization import add_safe_globals
        from deepspeed.runtime.zero.config import ZeroStageEnum
        from deepspeed.runtime.fp16.loss_scaler import LossScaler

        add_safe_globals([ZeroStageEnum, LossScaler])
        logger.info("DeepSpeed ZeRO patch enabled")
except Exception as e:  # noqa: BLE001
    logger.warning("DeepSpeed patch failed: %r", e)

# ————— cache dirs —————
HF_CACHE_DIR = os.path.abspath("./.hf_cache")
os.environ.update(
    HF_HOME=HF_CACHE_DIR,
    TRANSFORMERS_CACHE=os.path.join(HF_CACHE_DIR, "transformers"),
    HF_HUB_CACHE=os.path.join(HF_CACHE_DIR, "hub"),
)

# ————— prompt —————
PROMPT_TEMPLATE = (
    "You are a helpful AI assistant. First think in the <think> block, then write "
    "ONLY the final answer in the <answer> block. Do NOT add anything after "
    "</answer>.\n\n"
    "Problem: {problem}\n\n"
    "<think>\n"
    "</think>\n\n"
    "<answer>\n"
    "</answer>"
)
MAX_TOKENS = 1224  # generous upper-bound so the model can finish naturally

# —————————————————— helpers ———————————————————

def append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False)
        f.write("\n")

# —————————————————— inference loop ———————————————————

from torch.nn import functional as F

def run_inference_on_split(
    split_name: str,
    examples,
    tokenizer,
    model,
    step: int,
    outdir: str,
    batch_size: int = 16,
    num_samples: int = 1,
    temperature: float = 0.7,
):
    """Generate completions, compute avg token-entropy, and save JSONL."""
    outpath = os.path.join(outdir, f"step{step:04d}_{split_name}.jsonl")
    seen = set()
    if os.path.exists(outpath):
        with open(outpath, encoding="utf-8") as f:
            seen = {json.loads(l)["problem"] for l in f}

    logger.info("→ %s | %d examples", split_name, len(examples))
    t0 = time.time()

    for i in range(0, len(examples), batch_size):
        batch_ds = examples.select(range(i, min(i + batch_size, len(examples))))
        batch = [ex for ex in batch_ds if ex["problem"] not in seen]
        if not batch:
            continue

        prompts = [PROMPT_TEMPLATE.format(problem=ex["problem"]) for ex in batch]
        inputs  = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k,v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=MAX_TOKENS,
            pad_token_id = tokenizer.eos_token_id,
            eos_token_id = tokenizer.eos_token_id,
            do_sample    = (num_samples > 1),
            temperature  = temperature if num_samples > 1 else 0.0,
            num_return_sequences = num_samples,

            # **new for entropy**  
            output_scores=True,
            return_dict_in_generate=True,
        )

        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)

        # out.sequences: [B×S × num_samples, prompt_len+new_tokens]
        # out.scores: tuple of length new_tokens, each [B×S*num_samples, V]
        prompt_len = inputs["input_ids"].shape[-1]

        # Compute avg token entropy for each generated sequence
        # token_scores[k] has shape [batch_size*num_samples, vocab_size]
        entropies = []
        for batch_idx in range(out.sequences.shape[0]):
            # gather entropies over each new token
            tok_ent = []
            for logits in out.scores:
                # logits: [B×num_samples, V]
                probs = F.softmax(logits[batch_idx : batch_idx+1, :], dim=-1)
                ent   = -(probs * probs.log()).sum(dim=-1)  # [1]
                tok_ent.append(ent.item())
            entropies.append(sum(tok_ent) / len(tok_ent))

        # decode the new tokens for output
        decs = tokenizer.batch_decode(
            out.sequences[:, prompt_len:], skip_special_tokens=False
        )

        # now save each sample along with its entropy
        for bi, ex in enumerate(batch):
            for k in range(num_samples):
                idx = bi * num_samples + k
                txt = decs[idx]
                avg_ent = entropies[idx]
                ans_match = re.search(r"<answer>(.*?)</answer>", txt, re.I|re.S)
                if not ans_match:
                    logger.warning(
                        "❗Missing <answer> for problem '%s' (sample %d). Saved anyway.",
                        ex["problem"][:50], k,
                    )
                row = {
                    "problem": ex["problem"],
                    "gold_answer": ex.get("answer"),
                    "step": step,
                    "split": split_name,
                    "sample_idx": k,
                    "output": txt.strip(),
                    "entropy": avg_ent,       # ← new field
                }
                append_jsonl(outpath, row)

            seen.add(ex["problem"])

    logger.info("✓ %s done in %.1fs → %s", split_name, time.time()-t0, outpath)

# —————————————————————— main ——————————————————————
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_examples", type=int, default=500)
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        revision=args.revision,
        trust_remote_code=True,
        cache_dir=HF_CACHE_DIR,
    )
    tok.padding_side = "left"
    tok.pad_token = tok.pad_token or tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        revision=args.revision,
        trust_remote_code=True,
        cache_dir=HF_CACHE_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()

    from datasets import load_dataset
    ds = load_dataset("open-r1/OpenR1-Math-220k", cache_dir=HF_CACHE_DIR)["train"]
    ds = ds.shuffle(seed=42).select(range(args.num_examples))

    os.makedirs(args.output_dir, exist_ok=True)
    run_inference_on_split(
        "train",
        ds,
        tok,
        model,
        step=0,
        outdir=args.output_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        temperature=args.temperature,
    )
    logger.info("All inference complete.")