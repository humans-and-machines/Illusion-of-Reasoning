import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import src.analysis.core.h2_uncertainty_helpers as h2


def test_aha_gpt_eff_respects_gate_by_words():
    pass1 = {"shift_in_reasoning_v1": 1, "has_reconsider_cue": 0}
    rec = {}
    gpt, words = h2._aha_gpt_eff(pass1, rec, gate_by_words=False)
    assert gpt == 1 and words == 0
    gpt_gate, _ = h2._aha_gpt_eff(pass1, rec, gate_by_words=True)
    assert gpt_gate == 0  # gated off by words


def test_build_bucket_row_filters_invalid_and_extracts_fields():
    rec = {
        "pass1": {
            "is_correct_pred": 1,
            "entropy_answer": 0.4,
            "shift_in_reasoning_v1": 1,
            "has_reconsider_cue": 1,
            "reconsider_markers": [],
        },
        "problem": "p1",
        "step": 5,
    }
    row = h2._build_bucket_row(step_from_name=None, rec=rec, unc_field="answer", gate_gpt_by_words=False)
    assert row is not None
    assert row["problem"] == "p1"
    assert row["correct"] == 1
    assert row["aha_gpt"] == 1
    assert row["uncertainty"] == pytest.approx(0.4)

    # Missing pass1 returns None
    assert (
        h2._build_bucket_row(step_from_name=None, rec={"problem": "p1"}, unc_field="entropy", gate_gpt_by_words=False)
        is None
    )

    # Gate GPT by words blocks aha_gpt
    rec2 = copy.deepcopy(rec)
    rec2["pass1"]["shift_in_reasoning_v1"] = 0
    row2 = h2._build_bucket_row(step_from_name=None, rec=rec2, unc_field="answer", gate_gpt_by_words=True)
    assert row2["aha_gpt"] == 0

    # Missing uncertainty returns None
    rec3 = {"pass1": {"is_correct_pred": 1}, "problem": "p3", "step": 1}
    assert h2._build_bucket_row(step_from_name=None, rec=rec3, unc_field="answer", gate_gpt_by_words=False) is None


def test_build_bucket_row_returns_none_when_correct_missing():
    rec = {
        "pass1": {
            "is_correct_pred": None,
            "entropy_answer": 0.2,
        },
        "problem": "p1",
        "step": 1,
    }
    assert h2._build_bucket_row(step_from_name=None, rec=rec, unc_field="answer", gate_gpt_by_words=False) is None


def test_load_all_defs_for_buckets_reads_jsonl(tmp_path):
    rec = {"pass1": {"is_correct_pred": 1, "entropy_answer": 0.2, "shift_in_reasoning_v1": 0}, "problem": "p1"}
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(h2, "iter_records_from_file", lambda path: [rec])
    monkeypatch.setattr(h2, "nat_step_from_path", lambda path: 1)
    df = h2.load_all_defs_for_buckets(["step0001.jsonl"], unc_field="answer", gate_gpt_by_words=False)
    monkeypatch.undo()
    assert len(df) == 1
    assert df.iloc[0]["step"] == 1  # pulled from filename


def test_label_formal_samples_marks_only_matching_pairs():
    samples = pd.DataFrame(
        {"problem": ["p1", "p2"], "step": [1, 1], "aha_gpt": [1, 1], "aha_words": [1, 0], "correct": [1, 0]}
    )
    formal_step = pd.DataFrame({"problem": ["p1"], "step": [1], "aha_formal_pair": [1]})
    labeled = h2.label_formal_samples(samples, formal_step)
    assert labeled.loc[labeled["problem"] == "p1", "aha_formal"].iloc[0] == 1
    assert labeled.loc[labeled["problem"] == "p2", "aha_formal"].iloc[0] == 0


def test_make_uncertainty_buckets_and_standardize():
    samples = pd.DataFrame(
        {
            "problem": ["p1", "p2", "p3"],
            "step": [1, 1, 1],
            "correct": [1, 0, 1],
            "aha_gpt": [1, 0, 1],
            "uncertainty": [0.1, 0.2, 0.3],
        }
    )
    bucket_df = h2._make_uncertainty_buckets(samples, n_buckets=2)
    assert "uncertainty_std" in bucket_df.columns
    assert bucket_df["unc_bucket"].nunique() >= 3


def test_make_uncertainty_buckets_falls_back_to_cut(monkeypatch):
    # Force qcut to collapse categories and ensure the pd.cut fallback is invoked.
    samples = pd.DataFrame(
        {
            "problem": ["p1", "p2", "p3"],
            "step": [1, 1, 1],
            "correct": [1, 0, 1],
            "aha_gpt": [1, 0, 1],
            "uncertainty": [0.1, 0.1, 0.1],
        }
    )

    fake_qcut_return = pd.Series(pd.Categorical([0, 0, 0], categories=[0]))
    monkeypatch.setattr(pd, "qcut", lambda *a, **k: fake_qcut_return)

    cut_called = {}

    def fake_cut(*args, **kwargs):
        cut_called["args"] = args
        return pd.Series(pd.Categorical([0, 1, 2], categories=[0, 1, 2]))

    monkeypatch.setattr(pd, "cut", fake_cut)

    bucket_df = h2._make_uncertainty_buckets(samples, n_buckets=4)
    # pd.cut should have been used to produce three buckets.
    assert cut_called
    assert sorted(bucket_df["bucket_id"].unique()) == [0, 1, 2]


def test_make_all3_uncertainty_buckets_handles_paths(monkeypatch, tmp_path):
    jsonl = tmp_path / "step0001.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "problem": "p1",
                    "pass1": {
                        "is_correct_pred": 1,
                        "entropy_answer": 0.1,
                        "shift_in_reasoning_v1": 1,
                        "has_reconsider_cue": 1,
                    },
                }
            )
            + "\n"
        )

    args = SimpleNamespace(
        unc_buckets=3,
        dataset_name="DS",
        model_name="Model",
        unc_field="answer",
        gpt_gate_by_words=False,
        delta1=0.1,
        delta2=0.1,
        delta3=None,
        min_prior_steps=1,
        split="test",
        step=1,
        num_samples=1,
        output_dir=str(tmp_path),
    )
    calls = {}

    def fake_plot(d_words, d_gpt, d_formal, out_png, title_suffix=""):
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        Path(out_png).touch()
        calls["plot"] = True

    monkeypatch.setattr(h2, "plot_uncertainty_buckets_three", fake_plot)
    out_dir = tmp_path
    out_png, out_csv, d_all, problem_step_df = h2.make_all3_uncertainty_buckets_figure(
        files=[str(jsonl)],
        out_dir=str(out_dir),
        args=args,
    )
    assert Path(out_png).exists()
    assert Path(out_csv).exists()
    assert calls["plot"] is True


def test_style_ame_axis_applies_labels():
    class AxisStub:
        def __init__(self):
            self.calls = []

        def set_xlabel(self, val):
            self.calls.append(("xlabel", val))

        def set_ylabel(self, val):
            self.calls.append(("ylabel", val))

        def set_title(self, val):
            self.calls.append(("title", val))

        def grid(self, *args, **kwargs):
            self.calls.append(("grid", args, kwargs))

        def legend(self, *args, **kwargs):
            self.calls.append(("legend", args, kwargs))

    axis = AxisStub()
    h2.style_ame_axis(axis)
    assert ("xlabel", "Training step") in axis.calls
    assert ("ylabel", "AME(aha)") in axis.calls
    assert any(call[0] == "legend" for call in axis.calls)
