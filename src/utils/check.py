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
import logging
import sys
from pathlib import Path


try:
    # Package execution: src.utils.check
    from . import check_eval
except ImportError:  # pragma: no cover - running as a stand-alone script
    # Script execution: python check.py from src/utils directory
    import check_eval  # type: ignore[import-error]

logger = logging.getLogger(__name__)

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
        help=("Second-stage patch if no </answer> (only used when num_samples==1)."),
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


def main() -> None:
    """Entry point for crossword evaluation."""
    args = _parse_args()
    _setup_logging()
    stats, out_path = check_eval.run(args)
    final_accuracy = 100.0 * stats.correct / max(1, stats.seen)
    print(
        f"Done. Wrote {stats.seen} rows → {out_path}  (final acc={final_accuracy:.2f}%).",
    )


if __name__ == "__main__":
    main()
