#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest


configs = pytest.importorskip("src.training.configs")


def test_script_arguments_requires_dataset_or_mixture():
    with pytest.raises(ValueError):
        configs.ScriptArguments(dataset_name=None, dataset_mixture=None)

    args = configs.ScriptArguments(dataset_name="ds-name", dataset_mixture=None)
    assert args.dataset_name == "ds-name"
    assert args.dataset_mixture is None


def test_script_arguments_converts_dataset_mixture_to_dataclass():
    mixture_dict = {
        "datasets": [
            {
                "id": "ds1",
                "config": "cfg1",
                "split": "train",
                "columns": ["col1", "col2"],
                "weight": 0.5,
            },
            {
                "id": "ds2",
                "config": "cfg2",
                "split": "train",
                "columns": ["col1", "col2"],
                "weight": 0.5,
            },
        ],
        "seed": 123,
        "test_split_size": 0.1,
    }

    args = configs.ScriptArguments(dataset_name=None, dataset_mixture=mixture_dict)
    mixture = args.dataset_mixture

    assert isinstance(mixture, configs.DatasetMixtureConfig)
    assert mixture.seed == 123
    assert mixture.test_split_size == 0.1
    assert len(mixture.datasets) == 2
    assert isinstance(mixture.datasets[0], configs.DatasetConfig)
    assert mixture.datasets[0].dataset_id == "ds1"
    assert mixture.datasets[0].columns == ["col1", "col2"]
    assert mixture.datasets[0].weight == 0.5


def test_script_arguments_dataset_mixture_validates_structure():
    with pytest.raises(ValueError):
        configs.ScriptArguments(dataset_name=None, dataset_mixture={"foo": "bar"})

    with pytest.raises(ValueError):
        configs.ScriptArguments(
            dataset_name=None,
            dataset_mixture={"datasets": "not-a-list"},
        )


def test_script_arguments_dataset_mixture_enforces_column_consistency():
    mixture_dict = {
        "datasets": [
            {"id": "d1", "columns": ["a", "b"]},
            {"id": "d2", "columns": ["a", "c"]},
        ],
    }
    with pytest.raises(ValueError):
        configs.ScriptArguments(dataset_name=None, dataset_mixture=mixture_dict)


def test_merge_dataclass_attributes_overwrites_and_flattens():
    @dataclass
    class A:
        a: int
        shared: int

    @dataclass
    class B:
        b: int
        shared: int

    target = SimpleNamespace()
    cfg_a = A(a=1, shared=10)
    cfg_b = B(b=2, shared=20)

    out = configs.merge_dataclass_attributes(target, cfg_a, cfg_b)
    assert out is target
    assert target.a == 1
    assert target.b == 2
    # Later configs overwrite earlier ones for shared keys.
    assert target.shared == 20


def test_merge_dataclass_attributes_ignores_none():
    @dataclass
    class C:
        c: int = 3

    target = SimpleNamespace()
    out = configs.merge_dataclass_attributes(target, None, C())
    assert out is target
    assert target.c == 3
