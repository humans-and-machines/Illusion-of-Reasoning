#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Legacy OpenR1-style batch inference for Qwen-2.5-7B checkpoints.

Emits ``<think> … </think><answer> … </answer>`` blocks that are easy to grade
on OpenR1-Math-style data. This script is kept for backwards compatibility and
specialized experiments; for **canonical MATH-500 / HF evaluation**, prefer
``src.inference.runners.unified_math_runner``.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from importlib import import_module

from packaging import version

from src.inference.backends import HFBackend
from src.inference.utils.common import OPENR1_PROMPT_TEMPLATE, require_datasets

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
    torch_module_for_patch = import_module("torch")
    if version.parse(torch_module_for_patch.__version__) >= version.parse("2.6.0"):
        torch_serialization = import_module("torch.serialization")
        deepspeed_zero = import_module("deepspeed.runtime.zero.config")
        deepspeed_loss_scaler = import_module("deepspeed.runtime.fp16.loss_scaler")

        add_safe_globals = torch_serialization.add_safe_globals
        ZeroStageEnum = deepspeed_zero.ZeroStageEnum
        LossScaler = deepspeed_loss_scaler.LossScaler

        add_safe_globals([ZeroStageEnum, LossScaler])
        logger.info("DeepSpeed ZeRO patch enabled")
except ImportError as deepspeed_import_exc:
    logger.info("DeepSpeed patch skipped (optional deps missing): %r", deepspeed_import_exc)

# ————— cache dirs —————
HF_CACHE_DIR = os.path.abspath("./.hf_cache")
os.environ.update(
    HF_HOME=HF_CACHE_DIR,
    TRANSFORMERS_CACHE=os.path.join(HF_CACHE_DIR, "transformers"),
    HF_HUB_CACHE=os.path.join(HF_CACHE_DIR, "hub"),
)

# ————— prompt —————
PROMPT_TEMPLATE = OPENR1_PROMPT_TEMPLATE
MAX_TOKENS = 1224  # generous upper-bound so the model can finish naturally

# —————————————————— helpers ———————————————————


def _require_torch_modules():
    """
    Import ``torch`` and ``torch.nn.functional`` dynamically, raising if missing.

    :returns: A tuple ``(torch_module, functional_module)`` with the imported modules.
    :raises ImportError: If PyTorch is not installed in the environment.
    """
    try:
        torch_module = import_module("torch")
        functional_module = import_module("torch.nn.functional")
    except ImportError as torch_import_exc:  # pragma: no cover - optional runtime dependency
        print("torch is required: pip install torch", file=sys.stderr)
        raise torch_import_exc
    return torch_module, functional_module


def append_jsonl(path: str, row: dict) -> None:
    """
    Append a JSON-serializable row to a JSONL file, creating parent directories.

    :param path: Path to the JSONL file to append to.
    :param row: Mapping or object that can be serialized by :mod:`json`.
    :returns: ``None``. The row is appended to ``path`` on disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as jsonl_file:
        json.dump(row, jsonl_file, ensure_ascii=False)
        jsonl_file.write("\n")


def _load_seen_problems(output_path: str) -> set:
    """
    Load the set of problems already present in an existing JSONL file.

    :param output_path: Path to the JSONL file containing prior results.
    :returns: Set of problem strings that already appear in the file.
    """
    if not os.path.exists(output_path):
        return set()
    seen_problems: set = set()
    with open(output_path, encoding="utf-8") as existing_file:
        for line in existing_file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            problem = record.get("problem")
            if problem:
                seen_problems.add(problem)
    return seen_problems


def _build_generation_kwargs(tokenizer_obj, num_samples: int, temperature: float) -> dict:
    """
    Construct generation keyword arguments for the language model.

    :param tokenizer_obj: Tokenizer instance providing an ``eos_token_id`` attribute.
    :param num_samples: Number of samples to generate per prompt.
    :param temperature: Sampling temperature; ignored when ``num_samples == 1``.
    :returns: Dictionary of keyword arguments suitable for ``generate`` calls.
    """
    do_sample = num_samples > 1
    return {
        "max_new_tokens": MAX_TOKENS,
        "pad_token_id": tokenizer_obj.eos_token_id,
        "eos_token_id": tokenizer_obj.eos_token_id,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else 0.0,
        "num_return_sequences": num_samples,
        "output_scores": True,
        "return_dict_in_generate": True,
    }


def _compute_entropies(scores, functional_module) -> list[float]:
    """
    Compute average token entropy for each generated sequence.

    :param scores: Sequence of logits tensors returned by ``generate``.
    :param functional_module: ``torch.nn.functional`` module providing ``softmax``.
    :returns: List of mean token entropies, one per generated sequence.
    """
    if not scores:
        return []
    num_sequences = scores[0].shape[0]
    entropies: list[float] = []
    for sequence_index in range(num_sequences):
        token_entropies = []
        for logits in scores:
            probabilities = functional_module.softmax(
                logits[sequence_index : sequence_index + 1, :],
                dim=-1,
            )
            entropy_tensor = -(probabilities * probabilities.log()).sum(dim=-1)
            token_entropies.append(float(entropy_tensor.item()))
        entropies.append(sum(token_entropies) / len(token_entropies))
    return entropies


def _decode_new_tokens(tokenizer_obj, sequences, input_ids):
    """
    Decode only the newly generated tokens for each sequence.

    :param tokenizer_obj: Tokenizer used to decode token IDs into text.
    :param sequences: Tensor of full generated sequences including the prompt.
    :param input_ids: Tensor of input IDs that seeded the generation.
    :returns: List of decoded strings corresponding to the new tokens only.
    """
    prompt_length = input_ids.shape[-1]
    return tokenizer_obj.batch_decode(
        sequences[:, prompt_length:],
        skip_special_tokens=False,
    )


def _write_batch_rows(
    batch,
    decoded_sequences,
    entropies,
    context,
) -> None:
    """
    Write JSONL rows for a single batch of generated samples.

    :param batch: Iterable of dataset examples containing at least
        ``\"problem\"`` and ``\"answer\"``.
    :param decoded_sequences: List of decoded outputs, one per sample.
    :param entropies: List of average token entropies aligned with ``decoded_sequences``.
    :param context: Batch-level context carrying configuration and output path.
    :returns: ``None``. Rows are appended to the configured JSONL file.
    """
    for batch_index, example in enumerate(batch):
        for sample_idx in range(context.config.num_samples):
            flat_index = batch_index * context.config.num_samples + sample_idx
            text = decoded_sequences[flat_index]
            avg_entropy = entropies[flat_index]
            answer_match = re.search(
                r"<answer>(.*?)</answer>",
                text,
                re.I | re.S,
            )
            if not answer_match:
                logger.warning(
                    "❗Missing <answer> for problem '%s' (sample %d). Saved anyway.",
                    example["problem"][:50],
                    sample_idx,
                )
            row = {
                "problem": example["problem"],
                "gold_answer": example.get("answer"),
                "step": context.config.step,
                "split": context.config.split_name,
                "sample_idx": sample_idx,
                "output": text.strip(),
                "entropy": avg_entropy,
            }
            append_jsonl(context.output_path, row)

        context.seen_problems.add(example["problem"])


@dataclass
class InferenceConfig:
    """
    Configuration bundle for a single inference run.

    :param split_name: Name of the dataset split being processed.
    :param step: Training or checkpoint step identifier for logging and filenames.
    :param output_dir: Directory where JSONL results will be written.
    :param batch_size: Number of examples to process per batch.
    :param num_samples: Number of samples to generate per problem.
    :param temperature: Sampling temperature for generation.
    :param examples: Dataset object providing ``select`` and iteration.
    """

    split_name: str
    step: int
    output_dir: str
    batch_size: int
    num_samples: int
    temperature: float
    examples: object


@dataclass
class BatchWriteContext:
    """
    Context object passed to :func:`_write_batch_rows`.

    :param config: Inference configuration used for the run.
    :param output_path: Path to the JSONL file being written.
    :param seen_problems: Set of problems that have already been written.
    """

    config: InferenceConfig
    output_path: str
    seen_problems: set

# —————————————————— inference loop ———————————————————

def run_inference_on_split(config: InferenceConfig) -> None:
    """
    Convenience wrapper for backward compatibility.

    This entry point exists for older call sites and simply indicates that
    the caller must provide a tokenizer and model via
    :func:`run_inference_on_split_with_model`.

    :param config: Inference configuration for the run.
    :returns: ``None``. The function always raises a :class:`RuntimeError`.
    :raises RuntimeError: Always; callers should use ``run_inference_on_split_with_model``.
    """
    raise RuntimeError(
        "run_inference_on_split must be called with tokenizer and lm_model",
    )  # pragma: no cover


def run_inference_on_split_with_model(
    config: InferenceConfig,
    *,
    tokenizer,
    lm_model,
) -> None:
    """
    Generate completions, compute average token entropy, and save JSONL rows.

    :param config: Inference configuration specifying dataset and generation parameters.
    :param tokenizer: Tokenizer used to encode prompts and decode outputs.
    :param lm_model: Hugging Face-style model exposing a ``generate`` method.
    :returns: ``None``. Results are appended to the output JSONL file.
    """
    torch_module, functional_module = _require_torch_modules()

    output_path = os.path.join(
        config.output_dir,
        f"step{config.step:04d}_{config.split_name}.jsonl",
    )
    seen_problems = _load_seen_problems(output_path)

    logger.info("→ %s | %d examples", config.split_name, len(config.examples))
    start_time = time.time()

    for batch_start in range(0, len(config.examples), config.batch_size):
        batch_dataset = config.examples.select(
            range(
                batch_start,
                min(batch_start + config.batch_size, len(config.examples)),
            ),
        )
        batch = [
            example
            for example in batch_dataset
            if example["problem"] not in seen_problems
        ]
        if not batch:
            continue

        prompts = [
            PROMPT_TEMPLATE.format(problem=example["problem"])
            for example in batch
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        if torch_module.cuda.is_available():
            inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}

        with torch_module.inference_mode():
            output = lm_model.generate(
                **inputs,
                **_build_generation_kwargs(
                    tokenizer_obj=tokenizer,
                    num_samples=config.num_samples,
                    temperature=config.temperature,
                ),
            )

        entropies = _compute_entropies(output.scores, functional_module)
        _write_batch_rows(
            batch=batch,
            decoded_sequences=_decode_new_tokens(
                tokenizer_obj=tokenizer,
                sequences=output.sequences,
                input_ids=inputs["input_ids"],
            ),
            entropies=entropies,
            context=BatchWriteContext(
                config=config,
                output_path=output_path,
                seen_problems=seen_problems,
            ),
        )

    logger.info(
        "✓ %s done in %.1fs → %s",
        config.split_name,
        time.time() - start_time,
        output_path,
    )

def _load_openr1_math(cache_dir: str, num_examples: int):
    """
    Load the OpenR1-Math-220k training split via :mod:`datasets`.

    :param cache_dir: Directory to use as a datasets cache.
    :param num_examples: Maximum number of examples to keep from the training split.
    :returns: A dataset object representing the shuffled subset.
    """
    _, load_dataset_fn = require_datasets()
    dataset = load_dataset_fn("open-r1/OpenR1-Math-220k", cache_dir=cache_dir)["train"]
    return dataset.shuffle(seed=42).select(range(num_examples))


# —————————————————————— main ——————————————————————
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_examples", type=int, default=500)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    backend = HFBackend.from_pretrained(
        args.model_name_or_path,
        revision=args.revision,
        cache_dir=HF_CACHE_DIR,
        dtype="float16",
        device_map="auto",
        attn_implementation=None,
    )
    hf_tokenizer = backend.tokenizer
    language_model = backend.model

    examples = _load_openr1_math(HF_CACHE_DIR, args.num_examples)

    os.makedirs(args.output_dir, exist_ok=True)
    inference_run_config = InferenceConfig(
        split_name="train",
        step=0,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        temperature=args.temperature,
        examples=examples,
    )
    run_inference_on_split_with_model(
        inference_run_config,
        tokenizer=hf_tokenizer,
        lm_model=language_model,
    )
    logger.info("All inference complete.")
