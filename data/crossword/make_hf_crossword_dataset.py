#!/usr/bin/env python
"""
Convert a Dryad or Cryptonite cryptic-crossword bundle into a Hugging Face
DatasetDict (train / validation / test) that mirrors the Decrypt loaders.

Examples
--------
# Cryptonite official split (two .jsonl files inside a .zip)
python make_hf_crossword_dataset.py \
    --in_file cryptonite-official-split.zip \
    --hub_user od2961

# Dryad bundle (.json)
python make_hf_crossword_dataset.py \
    --in_file disjoint.json \
    --hub_user od2961 \
    --max_len 0         # keep all answer lengths

Copyright (c) 2024 The Illusion-of-Reasoning contributors.
Licensed under the Apache License, Version 2.0. See LICENSE for details.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import string
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

from data.crossword.hf_dataset_utils import dataset_classes

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
PUNCT_RE = re.compile(rf"[{re.escape(string.punctuation)}]")

def normalise(ans: str) -> str:
    """
    Strip punctuation and uppercase the answer so matching is exact.

    :param ans: Raw answer string from the bundle.
    :type ans: str
    :returns: Canonicalized answer string.
    :rtype: str
    """
    return PUNCT_RE.sub("", ans).upper()

def load_bundle(path: Path) -> Dict[str, List[dict]]:
    """
    Return a Python dict representing the bundle, unzipping if needed.

    Supported layouts
    -----------------
    • Dryad JSON file            : *.json, with keys train/val/test or clues
    • Dryad ZIP                  : *.zip containing one *.json (same layouts)
    • Cryptonite official split  : *.zip containing
                                   cryptonite-{train,test}.jsonl (+optional valid)

    :param path: Path to a JSON file or ZIP archive containing crossword data.
    :type path: Path
    :returns: Mapping of split name to list of clue dictionaries.
    :rtype: Dict[str, List[dict]]
    :raises ValueError: If the archive layout or schema is unrecognized.
    """
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zip_file:
            members = [
                member
                for member in zip_file.namelist()
                if member.endswith((".json", ".jsonl"))
            ]
            if not members:
                raise ValueError("ZIP contains no .json/.jsonl files")

            # -------- Cryptonite official split (.jsonl inside .zip) -------- #
            if all(
                member.startswith("cryptonite-") and member.endswith(".jsonl")
                for member in members
            ):
                bundle: Dict[str, List[dict]] = {}
                for member in members:
                    split = (
                        "train"      if "train" in member
                        else "validation" if ("valid" in member or "val" in member)
                        else "test"       if "test"  in member
                        else None
                    )
                    if split is None:
                        continue
                    with zip_file.open(member) as file_pointer:
                        bundle[split] = [json.loads(line) for line in file_pointer]
                # If no validation split, carve 10 % of train for dev
                if "validation" not in bundle:
                    rows = bundle["train"]
                    frac = max(1, int(0.10 * len(rows)))
                    bundle["validation"] = rows[-frac:]
                    bundle["train"]      = rows[:-frac]
                return bundle

            # ---------------- One-file bundle (.json inside .zip) ----------- #
            if len(members) == 1 and members[0].endswith(".json"):
                with zip_file.open(members[0]) as file_pointer:
                    return json.load(file_pointer)

            raise ValueError("Unrecognised ZIP layout")

    # --------------- Plain JSON file on disk ------------------------------- #
    with path.open() as file_obj:
        return json.load(file_obj)


def rows_from(split: List[dict], max_len: int) -> "Dataset":
    """
    Convert raw clue rows into a Hugging Face Dataset with optional max length.

    :param split: List of raw clue dictionaries.
    :type split: List[dict]
    :param max_len: Maximum allowed normalized answer length (0 to keep all).
    :type max_len: int
    :returns: Dataset containing question/answer pairs.
    :rtype: Dataset
    :raises KeyError: If a required answer field is missing.
    """
    dataset_cls, _ = dataset_classes()
    rows = []
    for item in split:
        raw_ans = item.get("answer") or item.get("solution") or item.get("soln")
        if raw_ans is None:
            raise KeyError(f"No answer key in JSON object: {list(item.keys())}")
        norm_ans = normalise(raw_ans)
        if max_len and len(norm_ans) > max_len:
            continue
        rows.append({
            "problem": f"{item['clue']}  \n<think>",
            "answer": norm_ans,
        })
    return dataset_cls.from_list(rows)


def make_hf_dataset(path: Path, max_len: int) -> "DatasetDict":
    """
    Build a DatasetDict with train/validation/test splits from a crossword bundle.

    :param path: Path to a JSON or ZIP bundle.
    :type path: Path
    :param max_len: Maximum allowed normalized answer length (0 to keep all).
    :type max_len: int
    :returns: DatasetDict with train, validation, and test splits.
    :rtype: DatasetDict
    :raises ValueError: If the bundle schema is not recognized.
    """
    _, dataset_dict_cls = dataset_classes()
    bundle = load_bundle(path)

    # Splits already provided
    if {"train", "validation", "test"} <= bundle.keys() \
       or {"train", "val", "test"} <= bundle.keys():
        key_val = "val" if "val" in bundle else "validation"
        return dataset_dict_cls({
            "train":      rows_from(bundle["train"], max_len),
            "validation": rows_from(bundle[key_val],  max_len),
            "test":       rows_from(bundle["test"],  max_len),
        })

    # Dryad 2020 dump: everything under 'clues' (make 80/10/10 split)
    if "clues" in bundle:
        ds_all = rows_from(bundle["clues"], max_len)
        frac = int(0.10 * len(ds_all))
        return dataset_dict_cls({
            "train":      ds_all.select(range(len(ds_all) - 2 * frac)),
            "validation": ds_all.select(range(len(ds_all) - 2 * frac, len(ds_all) - frac)),
            "test":       ds_all.select(range(len(ds_all) - frac, len(ds_all))),
        })

    raise ValueError("Unrecognised bundle schema")

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    """
    Parse CLI args, build the dataset, and optionally push to the HF hub.

    :raises ValueError: If the bundle schema is unrecognized.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", required=True, type=Path,
                        help="*.json or *.zip bundle")
    parser.add_argument("--hub_user", required=True,
                        help="HF username/org to push under, e.g. od2961")
    parser.add_argument("--name", default=None,
                        help="override HF repo name (default from file stem)")
    parser.add_argument("--max_len", type=int, default=5,
                        help="keep answers ≤ this length (0 disables)")
    parser.add_argument("--skip_push", action="store_true",
                        help="create datasets locally only (no push)")
    args = parser.parse_args()

    name = args.name or args.in_file.stem.replace("_", "-")
    repo = f"{args.hub_user}/Guardian-{name}"
    print(f"➜  Converting {args.in_file.name}  →  {repo}")

    dataset_dict = make_hf_dataset(args.in_file, args.max_len)

    if args.skip_push:
        tmp = Path("hf_datasets") / name
        if tmp.exists():
            shutil.rmtree(tmp)
        dataset_dict.save_to_disk(tmp)
        print(f"  ↳ saved to {tmp}")
    else:
        dataset_dict.push_to_hub(repo, private=True)   # set private=False if you want it public
        print(f"  ↳ pushed to hub as {repo}")

if __name__ == "__main__":
    main()
