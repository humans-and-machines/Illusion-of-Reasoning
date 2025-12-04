import importlib
import sys
from pathlib import Path

import pytest

from src.analysis import aha_utils


def test_analysis_init_exports_expected_symbols():
    analysis = importlib.reload(importlib.import_module("src.analysis"))
    expected = [
        "rq1_analysis",
        "rq2_analysis",
        "rq3_analysis",
        "core",
        "common",
        "figures",
        "io",
        "labels",
        "metrics",
    ]
    for name in expected:
        assert name in analysis.__all__
        assert hasattr(analysis, name)


def test_script_init_adds_repo_root(monkeypatch):
    module = importlib.import_module("src.analysis._script_init")
    repo_root = Path(module.__file__).resolve().parents[2]
    stripped_path = [entry for entry in sys.path if Path(entry).resolve() != repo_root]
    monkeypatch.setattr(sys, "path", stripped_path)
    module.ensure_repo_root_on_path()
    assert str(repo_root) in sys.path
    module.ensure_repo_root_on_path()
    assert sys.path.count(str(repo_root)) == 1


def test_any_keys_true_prefers_pass1_and_falls_back():
    pass1_fields = {"foo": "yes"}
    record = {"foo": False, "bar": True}
    assert aha_utils.any_keys_true(pass1_fields, record, ["foo", "bar"]) == 1

    pass1_fields = {"foo": None}
    record = {"foo": None, "bar": 0}
    assert aha_utils.any_keys_true(pass1_fields, record, ["foo", "bar"]) == 0

    pass1_fields = {}
    record = {"bar": 1}
    assert aha_utils.any_keys_true(pass1_fields, record, ["foo", "bar"]) == 1


def test_aha_native_blocks_injected_markers():
    assert (
        aha_utils.aha_native(
            {"has_reconsider_cue": True, "reconsider_markers": []},
            {},
        )
        == 1
    )
    assert (
        aha_utils.aha_native(
            {"has_reconsider_cue": True, "reconsider_markers": ["injected_cue"]},
            {},
        )
        == 0
    )
    assert (
        aha_utils.aha_native(
            {"has_reconsider_cue": None, "reconsider_markers": []},
            {},
        )
        == 0
    )


@pytest.mark.parametrize(
    "pass1_fields,record,domain,allow_judge,expected",
    [
        ({"has_reconsider_cue": True, "reconsider_markers": []}, {}, None, False, 1),
        ({"has_reconsider_cue": True, "reconsider_markers": ["injected_cue"]}, {}, None, False, 0),
        ({}, {"_shift_prefilter_markers": ["prefilter"]}, "math", False, 1),
        ({}, {"shift_markers_v1": ["judge"]}, "crossword", False, 1),
        ({}, {"shift_markers_v1": ["judge"]}, None, True, 1),
    ],
)
def test_cue_gate_for_llm(pass1_fields, record, domain, allow_judge, expected):
    assert aha_utils.cue_gate_for_llm(pass1_fields, record, domain, allow_judge) == expected
