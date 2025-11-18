
#!/usr/bin/env python3
"""
Convert a simple crossword CSV (Clue, Answer, Length) into a Hugging Face
DatasetDict with splits (train/validation/test) that your Qwen pipeline can consume.

Copyright (c) 2024 The Illusion-of-Reasoning contributors.
Licensed under the Apache License, Version 2.0. See LICENSE for details.

Schema (per split)
------------------
- problem:  "Clue: {clue} ({length})\n<think>"
- answer:   canonicalized answer (punct/space stripped, UPPERCASE)

Usage
-----
python csv_to_hf_crossword_dataset.py \
  --in_csv path/to/clues.csv \
  --hub_user od2961 \
  --name mini-crosswords \
  --split 0.8,0.1,0.1 \
  --private              # default; use --public to publish
  # --skip_push          # save to disk only
  # --seed 13            # for reproducible split
  # --max_len 0          # keep all lengths (0 disables)
  # --enum canonical     # 'provided' (CSV Length) or 'canonical' (from normalized answer)

Requires: pip install datasets pandas
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import string
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Tuple

from data.crossword.hf_dataset_utils import dataset_classes

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict

PUNCT_RE = re.compile(rf"[{re.escape(string.punctuation)}\s]+")


def canonicalize(ans: str) -> str:
    """
    Remove punctuation and whitespace then uppercase the answer.

    :param ans: Raw answer string from the CSV row.
    :type ans: str
    :returns: Canonicalized answer string.
    :rtype: str
    """
    return PUNCT_RE.sub("", (ans or "")).upper()


def parse_split_spec(spec: str) -> Tuple[float, float, float]:
    """
    Parse a CSV split string like ``'0.8,0.1,0.1'`` into normalized fractions.

    :param spec: Split string.
    :type spec: str
    :returns: Tuple of train, validation, test fractions that sum to 1.0.
    :rtype: Tuple[float, float, float]
    :raises argparse.ArgumentTypeError: If the string is malformed.
    """
    try:
        parts = [float(val) for val in spec.split(",")]
        if len(parts) != 3:
            raise ValueError
        sum_parts = sum(parts)
        if sum_parts <= 0:
            raise ValueError
        return tuple(part / sum_parts for part in parts)  # normalize to sum=1
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            "Split must be like '0.8,0.1,0.1'"
        ) from exc


def read_csv_rows(path: Path) -> Iterable[dict]:
    """
    Yield raw CSV rows as dictionaries.

    :param path: Path to the input CSV.
    :type path: Path
    :yields: Row dictionaries from the CSV file.
    """
    with path.open(newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        yield from reader


def build_examples(rows: Iterable[dict], enum_mode: str, max_len: int) -> List[dict]:
    """
    Transform raw CSV rows into canonicalized examples.

    :param rows: Iterable of raw CSV row dicts.
    :type rows: Iterable[dict]
    :param enum_mode: Enumeration mode, either ``"provided"`` or ``"canonical"``.
    :type enum_mode: str
    :param max_len: Maximum allowed answer length (0 to disable).
    :type max_len: int
    :returns: List of processed example dictionaries.
    :rtype: List[dict]
    """
    examples = []
    warn_mismatches = 0
    for row in rows:
        clue = (row.get("Clue") or row.get("clue") or "").strip()
        raw_ans = (row.get("Answer") or row.get("answer") or "").strip()
        length_raw = (row.get("Length") or row.get("length") or "").strip()
        if not clue or not raw_ans:
            continue

        norm = canonicalize(raw_ans)
        can_len = len(norm)
        # resolve enumeration
        prov_len = None
        if length_raw:
            try:
                prov_len = int(length_raw)
            except ValueError:
                prov_len = None

        if enum_mode == "provided" and prov_len is not None:
            enum = prov_len
            if enum != can_len:
                warn_mismatches += 1
        else:
            enum = can_len  # canonical

        if max_len and can_len > max_len:
            continue

        examples.append({
            "problem": f"Clue: {clue} ({enum})\n<think>",
            "answer": norm,
        })
    if warn_mismatches:
        print(
            f"[warn] {warn_mismatches} rows had provided Length != canonical length",
            file=sys.stderr,
        )
    return examples


def to_datasetdict(
    examples: List[dict],
    split_fracs: Tuple[float, float, float],
    seed: int,
) -> "DatasetDict":
    """
    Split examples into an HF DatasetDict with deterministic shuffling.

    :param examples: Processed example dictionaries.
    :type examples: List[dict]
    :param split_fracs: Fractions for train/validation/test.
    :type split_fracs: Tuple[float, float, float]
    :param seed: RNG seed for deterministic shuffling.
    :type seed: int
    :returns: DatasetDict containing train, validation, and test splits.
    :rtype: DatasetDict
    :raises ValueError: If no examples are provided.
    """
    dataset_cls, dataset_dict_cls = dataset_classes()
    num_examples = len(examples)
    if num_examples == 0:
        raise ValueError("No usable rows parsed from CSV.")
    # deterministic split
    rng = random.Random(seed)
    idx = list(range(num_examples))
    rng.shuffle(idx)
    n_train = int(split_fracs[0] * num_examples)
    n_val = int(split_fracs[1] * num_examples)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    def select(idxs):
        return [examples[i] for i in idxs]

    return dataset_dict_cls({
        "train": dataset_cls.from_list(select(train_idx)),
        "validation": dataset_cls.from_list(select(val_idx)),
        "test": dataset_cls.from_list(select(test_idx)),
    })


def main() -> None:
    """
    Parse CLI args, build datasets, and optionally push to the Hugging Face hub.

    :raises ValueError: If no usable rows are parsed from the CSV.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_csv",
        required=True,
        type=Path,
        help="CSV with columns: Clue, Answer, Length",
    )
    parser.add_argument(
        "--hub_user",
        required=True,
        help="HF username/org (e.g., od2961)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="HF repo name (default from CSV filename)",
    )
    parser.add_argument(
        "--split",
        type=parse_split_spec,
        default="0.8,0.1,0.1",
        help="train,val,test fractions (normalize to 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="RNG seed for split shuffling",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=0,
        help="Keep answers <= this length (0 disables)",
    )
    parser.add_argument(
        "--enum",
        choices=["provided", "canonical"],
        default="provided",
        help="Use provided Length or compute from canonicalized answer",
    )
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument(
        "--private",
        action="store_true",
        help="Push as private (default)",
    )
    visibility.add_argument(
        "--public",
        action="store_true",
        help="Push as public",
    )
    parser.add_argument(
        "--skip_push",
        action="store_true",
        help="Save to disk only (no push to hub)",
    )

    args = parser.parse_args()

    repo_name = args.name or args.in_csv.stem.replace("_", "-")
    repo_id = f"{args.hub_user}/{repo_name}"

    rows = list(read_csv_rows(args.in_csv))
    examples = build_examples(rows, enum_mode=args.enum, max_len=args.max_len)
    dataset_dict = to_datasetdict(examples, split_fracs=args.split, seed=args.seed)

    if args.skip_push:
        out_dir = Path("hf_datasets") / repo_name
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        dataset_dict.save_to_disk(out_dir)
        print(f"Saved to {out_dir}")
    else:
        dataset_dict.push_to_hub(repo_id, private=not args.public)
        print(f"Pushed to hub: {repo_id} (private={not args.public})")


if __name__ == "__main__":
    main()
