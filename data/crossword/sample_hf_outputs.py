#!/usr/bin/env python3
"""
Sample a few rows from a Hugging Face crossword dataset (local or Hub-hosted).

Copyright (c) 2024 The Illusion-of-Reasoning contributors.
Licensed under the Apache License, Version 2.0. See LICENSE for details.
"""

from __future__ import annotations

import argparse
import importlib
import pprint
import random
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from datasets import DatasetDict, IterableDatasetDict


def dataset_loaders() -> Tuple:
    """
    Import dataset loader functions lazily.

    :returns: Tuple of (load_dataset, load_from_disk) callables.
    :rtype: Tuple
    :raises ImportError: If the `datasets` package is missing.
    """
    try:
        datasets_module = importlib.import_module("datasets")
    except ImportError as exc:
        raise ImportError("Install `datasets` to inspect HF outputs.") from exc
    return datasets_module.load_dataset, datasets_module.load_from_disk


def main() -> None:
    """
    Load a HF dataset and print a few random examples per split.

    :raises FileNotFoundError: If a local dataset path is provided but missing.
    """
    parser = argparse.ArgumentParser(
        description="Load a HF dataset and print a few random examples per split."
    )
    parser.add_argument(
        "--repo",
        required=True,
        help=(
            "Either the HF repo ID (e.g. od2961/Guardian-disjoint_word_init) "
            "or a local path (e.g. hf_datasets/disjoint_word_init)."
        ),
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of samples to show per split (default: 5).",
    )
    args = parser.parse_args()

    load_dataset, load_from_disk = dataset_loaders()

    # try loading from disk first, then fall back to Hub
    try:
        dataset_dict: "DatasetDict | IterableDatasetDict" = load_from_disk(args.repo)
    except (FileNotFoundError, ValueError, OSError):
        dataset_dict = load_dataset(args.repo)

    pretty_printer = pprint.PrettyPrinter(indent=2)

    for split_name in dataset_dict.keys():
        print(f"\n=== {split_name.upper()} ({args.n} random examples) ===")
        # shuffle indices reproducibly
        indices = list(range(len(dataset_dict[split_name])))
        random.seed(42)
        random.shuffle(indices)
        for idx in indices[: args.n]:
            example = dataset_dict[split_name][idx]
            pretty_printer.pprint(example)


if __name__ == "__main__":
    main()
