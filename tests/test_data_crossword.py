import argparse
import importlib
import json
import zipfile

import pytest

from data.crossword import csv_to_hf_crossword_dataset as csvmod
from data.crossword import hf_dataset_utils as utils
from data.crossword import make_hf_crossword_dataset as mk
from data.crossword import sample_hf_outputs as sampler


def test_dataset_classes_returns_stub_classes():
    dataset_cls, dataset_dict_cls = utils.dataset_classes()
    dataset = dataset_cls.from_list([{"a": 1}])
    assert len(dataset) == 1
    dsd = dataset_dict_cls({"train": dataset})
    assert dsd["train"] is dataset


def test_dataset_classes_raises_when_missing(monkeypatch):
    def _raise(_):
        raise ImportError("missing")

    monkeypatch.setattr(importlib, "import_module", lambda name: _raise(name))
    with pytest.raises(ImportError):
        utils.dataset_classes()


def test_normalise_strips_punctuation_and_uppercases():
    assert mk.normalise("A.b! c?") == "AB C"


def test_load_bundle_handles_cryptonite_zip_and_derives_validation(tmp_path):
    zip_path = tmp_path / "cryptonite.zip"
    train_rows = [{"clue": "c1", "answer": "ans1"}, {"clue": "c2", "answer": "ans2"}]
    test_rows = [{"clue": "t1", "answer": "tans"}]
    with zipfile.ZipFile(zip_path, mode="w") as zf:
        zf.writestr("cryptonite-train.jsonl", "\n".join(json.dumps(r) for r in train_rows))
        zf.writestr("cryptonite-test.jsonl", "\n".join(json.dumps(r) for r in test_rows))

    bundle = mk.load_bundle(zip_path)
    assert set(bundle.keys()) == {"train", "validation", "test"}
    assert len(bundle["train"]) == 1  # 10% of train carved out for validation
    assert len(bundle["validation"]) == 1
    assert bundle["validation"][0]["answer"] == "ans2"
    assert bundle["test"][0]["clue"] == "t1"


def test_rows_from_respects_max_len_and_requires_answers():
    rows = [
        {"clue": "One", "answer": "abc!"},
        {"clue": "Two", "solution": "waytoolong"},
    ]
    dataset = mk.rows_from(rows, max_len=3)
    assert len(dataset) == 1
    assert dataset[0]["answer"] == "ABC"

    with pytest.raises(KeyError):
        mk.rows_from([{"clue": "No answer here"}], max_len=0)


def test_make_hf_dataset_splits_dryad_schema(tmp_path):
    clues = [{"clue": f"r{i}", "answer": "ans"} for i in range(10)]
    bundle_path = tmp_path / "dryad.json"
    bundle_path.write_text(json.dumps({"clues": clues}))

    dsd = mk.make_hf_dataset(bundle_path, max_len=5)
    assert set(dsd.keys()) == {"train", "validation", "test"}
    assert len(dsd["train"]) == 8
    assert len(dsd["validation"]) == 1
    assert len(dsd["test"]) == 1


def test_parse_split_spec_normalizes_and_validates():
    assert csvmod.parse_split_spec("1,1,2") == (0.25, 0.25, 0.5)
    with pytest.raises(argparse.ArgumentTypeError):
        csvmod.parse_split_spec("bad-spec")


def test_canonicalize_removes_spaces_and_punct():
    assert csvmod.canonicalize("A b-c") == "ABC"


def test_build_examples_filters_and_respects_enum_mode():
    rows = [
        {"Clue": "One", "Answer": "A B", "Length": "3"},
        {"clue": "Two", "answer": "TooLong", "Length": "7"},
        {"Clue": "", "Answer": "skip"},
    ]
    examples = csvmod.build_examples(rows, enum_mode="provided", max_len=3)
    assert len(examples) == 1
    assert examples[0]["problem"].startswith("Clue: One (3)")
    assert examples[0]["answer"] == "AB"


def test_to_datasetdict_deterministic_split():
    examples = [
        {"problem": "p1", "answer": "A"},
        {"problem": "p2", "answer": "B"},
        {"problem": "p3", "answer": "C"},
        {"problem": "p4", "answer": "D"},
        {"problem": "p5", "answer": "E"},
    ]
    dsd = csvmod.to_datasetdict(examples, split_fracs=(0.6, 0.2, 0.2), seed=0)
    assert set(dsd.keys()) == {"train", "validation", "test"}
    assert len(dsd["train"]) == 3
    assert len(dsd["validation"]) == 1
    assert len(dsd["test"]) == 1
    assert {row["problem"] for row in dsd["train"].rows} == {"p1", "p2", "p3"}


def test_dataset_loaders_import_error(monkeypatch):
    def _raise(_):
        raise ImportError("no datasets")

    monkeypatch.setattr(importlib, "import_module", lambda name: _raise(name))
    with pytest.raises(ImportError):
        sampler.dataset_loaders()


def test_dataset_loaders_return_callables():
    load_dataset, load_from_disk = sampler.dataset_loaders()
    assert callable(load_dataset)
    assert callable(load_from_disk)
