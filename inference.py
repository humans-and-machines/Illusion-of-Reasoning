#!/usr/bin/env python
import sys
print(f"→ START {__file__}", file=sys.stderr, flush=True)

import os
print("→ [DEBUG] imported os", file=sys.stderr, flush=True)
import json
print("→ [DEBUG] imported json", file=sys.stderr, flush=True)
import torch
print("→ [DEBUG] imported torch", file=sys.stderr, flush=True)
import argparse
print("→ [DEBUG] imported argparse", file=sys.stderr, flush=True)
import gc
print("→ [DEBUG] imported gc", file=sys.stderr, flush=True)
import torch._dynamo
print("→ [DEBUG] imported torch._dynamo", file=sys.stderr, flush=True)
from packaging import version
print("→ [DEBUG] imported packaging.version", file=sys.stderr, flush=True)

# —— Patch for PyTorch 2.6+ DeepSpeed ZeRO unpickle support ——
try:
    if version.parse(torch.__version__) >= version.parse("2.6.0"):
        print("→ [DEBUG] torch>=2.6 detected, attempting DeepSpeed imports…", file=sys.stderr, flush=True)
        from torch.serialization import add_safe_globals
        from deepspeed.runtime.zero.config import ZeroStageEnum
        from deepspeed.runtime.fp16.loss_scaler import LossScaler

        trusted_classes = [ZeroStageEnum, LossScaler]
        add_safe_globals([cls for cls in trusted_classes if isinstance(cls, type)])
        print("✅ DeepSpeed ZeRO unpickle support enabled", file=sys.stderr, flush=True)
    else:
        print("→ [DEBUG] torch<2.6 — skipping DeepSpeed patch", file=sys.stderr, flush=True)
except Exception as e:
    print(f"⚠️ DeepSpeed patch block failed at import: {e!r}", file=sys.stderr, flush=True)

# ——— cache setup ———
HF_CACHE_DIR = os.path.abspath("./.hf_cache")
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_DIR, "transformers")
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")
print("→ [DEBUG] cache setup complete", file=sys.stderr, flush=True)

# ——— model & checkpoints ———
MODEL = "od2961/Qwen2.5-1.5B-Instruct-SFT"

REVISIONS = [
    "7d3da18",  # step 2550
    "d2157c5",  # step 2500
    "49ab2ba",  # step 2450
    "f2878b3",  # step 2400
    "4c4fef1",  # step 2350
    "68a9eef",  # step 2300
    "46bc986",  # step 2250
    "fd081d7",  # step 2200
    "d9d72fd",  # step 2150
    "f9b2384",  # step 2100
    "bce3b11",  # step 2050
    "69acec6",  # step 2000
    "eb508a2",  # step 1950
    "cf2265b",  # step 1900
    "de6b7b5",  # step 1850
    "96f738d",  # step 1800
    "054b90b",  # step 1750
    "e47265a",  # step 1700
    "09d1b6b",  # step 1650
]

SHA_TO_STEP = {
    "7d3da18": 2550,
    "d2157c5": 2500,
    "49ab2ba": 2450,
    "f2878b3": 2400,
    "4c4fef1": 2350,
    "68a9eef": 2300,
    "46bc986": 2250,
    "fd081d7": 2200,
    "d9d72fd": 2150,
    "f9b2384": 2100,
    "bce3b11": 2050,
    "69acec6": 2000,
    "eb508a2": 1950,
    "cf2265b": 1900,
    "de6b7b5": 1850,
    "96f738d": 1800,
    "054b90b": 1750,
    "e47265a": 1700,
    "09d1b6b": 1650,
}

# ——— prompt template ———
PROMPT_TEMPLATE = (
    "You are a helpful AI Assistant that provides well‑reasoned but short responses.\n"
    "You first briefly think about the reasoning process as an internal monologue and then provide the user with the answer.\n"
    "Respond in the following format:\n\n"
    "Problem: {problem}\n\n"
    "<think>\n"
    "</think>\n\n"
    "<answer>\n"
    "</answer>\n"
)

THINK_STOP = "</think>"
ANSWER_STOP = "</answer>"
THINK_MAX_TOKENS = 750
ANSWER_MAX_TOKENS = 250

def load_ckpt(revision: str):
    print(f"→ [DEBUG] load_ckpt: start loading rev={revision}", file=sys.stderr, flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(
        MODEL, revision=revision, trust_remote_code=True,
        cache_dir=os.environ["TRANSFORMERS_CACHE"]
    )
    print(f"→ [DEBUG] tokenizer loaded for {revision}", file=sys.stderr, flush=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=revision, trust_remote_code=True,
        torch_dtype=torch.float16,
        cache_dir=os.environ["TRANSFORMERS_CACHE"]
    )
    print(f"→ [DEBUG] model.from_pretrained done for {revision}", file=sys.stderr, flush=True)

    if not getattr(load_ckpt, "compiled_once", False):
        print(f"→ [DEBUG] compiling model for {revision}", file=sys.stderr, flush=True)
        mdl = torch.compile(mdl)
        load_ckpt.compiled_once = True
        print(f"→ [DEBUG] compile complete for {revision}", file=sys.stderr, flush=True)
    else:
        torch._dynamo.reset()
        print(f"→ [DEBUG] Dynamo cache reset for reuse {revision}", file=sys.stderr, flush=True)

    if hasattr(mdl.config, "attn_implementation"):
        mdl.config.attn_implementation = "flash_attention_2"
        print("✅ FlashAttention 2 enabled via model config.", file=sys.stderr, flush=True)
    else:
        print("⚠️ Model config does not support FlashAttention 2.", file=sys.stderr, flush=True)

    return tok, mdl


def append_jsonl(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_inference_on_split(split_name, examples, tokenizer, model, rev, output_dir, batch_size=16):
    print(f"→ [DEBUG] run_inference_on_split: start {split_name} @ rev={rev}", file=sys.stderr, flush=True)
    from transformers import StoppingCriteria, StoppingCriteriaList

    class StopOnTag(StoppingCriteria):
        def __init__(self, tokenizer, stop_str: str):
            self.stop_ids = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(stop_str, add_special_tokens=False)
            )
        def __call__(self, input_ids, scores, **kwargs):
            seq = input_ids[0].tolist()
            return len(seq) >= len(self.stop_ids) and seq[-len(self.stop_ids):] == self.stop_ids

    stop_think = StoppingCriteriaList([StopOnTag(tokenizer, THINK_STOP)])
    stop_answer = StoppingCriteriaList([StopOnTag(tokenizer, ANSWER_STOP)])

    step = SHA_TO_STEP.get(rev)
    if step is None:
        raise ValueError(f"No training step found for revision {rev}")
    filename = f"{MODEL.split('/')[-1]}_step{step:04d}_{split_name}.jsonl"
    outpath = os.path.join(output_dir, filename)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    completed = set()
    if os.path.exists(outpath):
        with open(outpath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    completed.add(data["problem"])
                except json.JSONDecodeError:
                    continue
    print(f"[{rev}][{split_name}] skipping {len(completed)} already-completed examples", file=sys.stderr, flush=True)

    total = len(examples)
    for i in range(0, total, batch_size):
        print(f"→ [DEBUG] processing batch {i//batch_size+1} of {((total-1)//batch_size)+1}", file=sys.stderr, flush=True)
        batch_raw = examples.select(range(i, min(i+batch_size, total)))
        batch = [ex for ex in batch_raw if ex["problem"] not in completed]
        if not batch:
            continue

        prompts = [PROMPT_TEMPLATE.format(problem=ex["problem"]) for ex in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        print(f"→ [DEBUG] generating <think> for batch {i//batch_size+1}", file=sys.stderr, flush=True)
        with torch.inference_mode():
            think_ids = model.generate(
                **inputs,
                max_new_tokens=THINK_MAX_TOKENS,
                do_sample=False,
                stopping_criteria=stop_think,
                pad_token_id=tokenizer.eos_token_id,
            )
        think_texts = tokenizer.batch_decode(think_ids, skip_special_tokens=True)

        answer_prompts = [t.strip() + "\n\n<answer>\n" for t in think_texts]
        ans_inputs = tokenizer(answer_prompts, return_tensors="pt", padding=True, truncation=True)
        if torch.cuda.is_available():
            ans_inputs = {k: v.to("cuda") for k, v in ans_inputs.items()}

        print(f"→ [DEBUG] generating <answer> for batch {i//batch_size+1}", file=sys.stderr, flush=True)
        with torch.inference_mode():
            ans_ids = model.generate(
                **ans_inputs,
                max_new_tokens=ANSWER_MAX_TOKENS,
                do_sample=False,
                stopping_criteria=stop_answer,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer_texts = tokenizer.batch_decode(ans_ids, skip_special_tokens=True)

        for j, ex in enumerate(batch):
            result = {
                "problem": ex["problem"],
                "gold_answer": ex["answer"],
                "step": rev,
                "split": split_name,
                "output": think_texts[j].strip() + "\n\n" + answer_texts[j].strip()
            }
            append_jsonl(outpath, result)
            print(f"[{rev}][{split_name} {i+j+1}/{total}] saved", file=sys.stderr, flush=True)
            completed.add(ex["problem"])

    print(f"→ [DEBUG] run_inference_on_split complete for {split_name} @ rev={rev}", file=sys.stderr, flush=True)


def main():
    print("→ [DEBUG] entered main()", file=sys.stderr, flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    print("→ [DEBUG] about to import datasets…", file=sys.stderr, flush=True)
    from datasets import load_dataset
    print("→ [DEBUG] import datasets done", file=sys.stderr, flush=True)

    print("→ [DEBUG] about to load_dataset…", file=sys.stderr, flush=True)
    ds = load_dataset("open-r1/OpenR1-Math-220k", "default", cache_dir=HF_CACHE_DIR)
    print("→ [DEBUG] load_dataset complete", file=sys.stderr, flush=True)

    train_examples = ds["train"].shuffle(seed=42).select(range(500))

    for rev in REVISIONS:
        print(f"\n=== Running revision {rev} ===", file=sys.stderr, flush=True)
        try:
            tokenizer, model = load_ckpt(rev)
        except Exception as e:
            print(f"[ERROR] Failed to load rev {rev}: {e}", file=sys.stderr, flush=True)
            continue

        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")

        print(f"→ [DEBUG] warm-up generate for {rev}", file=sys.stderr, flush=True)
        _ = model.generate(
            **tokenizer("warmup", return_tensors="pt").to("cuda"),
            max_new_tokens=1
        )
        print(f"[{rev}] warm-up done", file=sys.stderr, flush=True)

        run_inference_on_split("train", train_examples, tokenizer, model, rev, args.output_dir)

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"→ [DEBUG] after cleanup for rev={rev}", file=sys.stderr, flush=True)

    print("\nAll revisions completed — JSONL files saved.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()