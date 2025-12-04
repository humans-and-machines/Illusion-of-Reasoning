import argparse

import numpy as np
import pandas as pd
import pytest

import src.analysis.figure_2_data as f2


def test_any_keys_true_prefers_pass1():
    pass1 = {"flag": "yes"}
    record = {"flag": False, "other": True}
    assert f2._any_keys_true(pass1, record, ["flag", "other"]) == 1
    assert f2._any_keys_true({}, record, ["missing"]) == 0


def test_build_pass1_row_success_and_gate(monkeypatch):
    monkeypatch.setattr(f2, "problem_key_from_record", lambda rec, missing_default=None: "prob1")
    monkeypatch.setattr(f2, "choose_uncertainty", lambda pass1, pref=None: 0.7)
    monkeypatch.setattr(f2, "aha_words", lambda pass1: 1)
    record = {"pass1": {"is_correct_pred": True, "foo": "bar"}, "step": 2, "shift_in_reasoning_v1": True}
    row = f2._build_pass1_row(record, None, "entropy", ["shift_in_reasoning_v1"], gate_gpt_by_words=False)
    assert row["problem"] == "prob1"
    assert row["aha_gpt"] == 1

    # Gate GPT by words: with aha_words returning False, GPT flag is forced to 0
    monkeypatch.setattr(f2, "aha_words", lambda pass1: 0)
    gated = f2._build_pass1_row(record, None, "entropy", ["shift_in_reasoning_v1"], gate_gpt_by_words=True)
    assert gated["aha_gpt"] == 0


def test_build_pass1_row_rejects_missing(monkeypatch):
    monkeypatch.setattr(f2, "choose_uncertainty", lambda *args, **kwargs: None)
    record = {"pass1": {"is_correct_pred": True}, "step": 1}
    assert f2._build_pass1_row(record, None, "entropy", [], False) is None


def test_build_pass1_row_handles_absent_pass1():
    record = {"pass1": {}, "step": 3}
    assert f2._build_pass1_row(record, None, "entropy", [], False) is None


def test_build_pass1_row_rejects_uncastable_correct(monkeypatch):
    monkeypatch.setattr(f2, "coerce_bool", lambda value: None)
    record = {"pass1": {"is_correct_pred": "maybe", "entropy": 0.2}, "step": 4}
    assert f2._build_pass1_row(record, None, "entropy", [], False) is None


def test_load_pass1_rows_and_empty(monkeypatch):
    # Good rows collected
    monkeypatch.setattr(
        f2, "iter_pass1_records", lambda files: [("p", 1, {"pass1": {"is_correct_pred": True, "entropy": 0.1}})]
    )
    monkeypatch.setattr(
        f2,
        "_build_pass1_row",
        lambda **kwargs: {"problem": "p", "step": 1, "correct": 1, "uncertainty": 0.1, "aha_words": 0, "aha_gpt": 0},
    )
    df = f2.load_pass1_rows(["p"], "entropy", [], False)
    assert len(df) == 1

    # No usable rows → RuntimeError
    monkeypatch.setattr(f2, "_build_pass1_row", lambda **kwargs: None)
    with pytest.raises(RuntimeError):
        f2.load_pass1_rows(["p"], "entropy", [], False)


def test_wilson_ci_edge_and_mid():
    lo, hi = f2.wilson_ci(0, 0)
    assert np.isnan(lo) and np.isnan(hi)
    lo2, hi2 = f2.wilson_ci(5, 10)
    assert 0 <= lo2 <= hi2 <= 1


def test_make_edges_and_density_and_correct_hist():
    std_vals = np.array([0.0, 1.0, 2.0, 3.0])
    edges = f2.make_edges_from_std(std_vals, bins=5, xlim=(0.0, 1.0))
    assert len(edges) == 11  # max(10, bins)+1

    centers, density = f2.density_from_hist(np.array([]), edges)
    assert np.all(density == 0)
    assert len(centers) == len(edges) - 1

    centers2, density2 = f2.density_from_hist(np.array([0.2, 0.8, 0.9]), np.array([0.0, 0.5, 1.0]), smooth_k=4)
    assert len(centers2) == 2
    assert len(density2) >= len(centers2)

    correct_hist = f2.compute_correct_hist(np.array([0.2, 0.8]), np.array([1, 0]), np.array([0.0, 0.5, 1.0]))
    assert correct_hist.tolist() == [1, 0]


def test_make_edges_uses_percentiles_when_xlim_missing():
    values = np.array([-5.0, -1.0, 0.0, 2.0, 10.0])
    edges = f2.make_edges_from_std(values, bins=3)
    p1, p99 = np.nanpercentile(values, [1, 99])
    assert edges[0] == pytest.approx(p1)
    assert edges[-1] == pytest.approx(p99)


def test_standardize_uncertainty_proxy(monkeypatch):
    called = {}

    def fake_std(df):
        called["ok"] = True
        return df.assign(uncertainty_std=1.0)

    monkeypatch.setattr(f2, "standardize_uncertainty", fake_std)
    out = f2._standardize_uncertainty(pd.DataFrame({"uncertainty": [0.1]}))
    assert called["ok"] is True
    assert "uncertainty_std" in out.columns


def test_maybe_run_rq2(monkeypatch):
    called = {}

    def fake_import(name):
        called["imported"] = name

        class _Mod:
            def main(self):
                called["ran"] = True

        return _Mod()

    monkeypatch.setattr(f2, "importlib", argparse.Namespace(import_module=fake_import))
    monkeypatch.setattr(f2, "build_results_root_argv", lambda root, split: ["argv"])
    monkeypatch.setattr(f2, "run_module_main_with_argv", lambda main, argv, prog=None: main())

    args = argparse.Namespace(run_rq2=True, results_root="root", split=None)
    f2._maybe_run_rq2(args)
    assert called["ran"] is True
    assert "imported" in called

    # No run when flag is False
    called.clear()
    args_no = argparse.Namespace(run_rq2=False)
    f2._maybe_run_rq2(args_no)
    assert called == {}


def test_load_all_samples_from_csv(monkeypatch, tmp_path):
    csv_path = tmp_path / "rq2" / "h2_analysis" / "h2_all3_pass1_samples.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    monkeypatch.setattr(f2.os.path, "isfile", lambda p: str(p) == str(csv_path))
    monkeypatch.setattr(f2, "_maybe_run_rq2", lambda args: None)
    monkeypatch.setattr(f2.pd, "read_csv", lambda path: pd.DataFrame({"a": [1], "b": [2]}))

    args = argparse.Namespace(
        results_root=str(tmp_path),
        rq2_dir=None,
        split=None,
        unc_field="entropy",
        gpt_mode="canonical",
        gpt_gate_by_words=False,
        delta1=0.2,
        delta2=0.2,
        delta3=None,
        min_prior_steps=2,
    )
    df = f2._load_all_samples(args)
    assert list(df.columns) == ["a", "b"]


def test_load_all_samples_from_raw(monkeypatch):
    args = argparse.Namespace(
        results_root="root",
        rq2_dir=None,
        split="test",
        unc_field="entropy",
        gpt_mode="canonical",
        gpt_gate_by_words=False,
        delta1=0.2,
        delta2=0.2,
        delta3=None,
        min_prior_steps=2,
    )
    monkeypatch.setattr(f2, "_maybe_run_rq2", lambda a: None)
    monkeypatch.setattr(
        f2, "os", argparse.Namespace(path=argparse.Namespace(isfile=lambda p: False, join=f2.os.path.join))
    )
    monkeypatch.setattr(f2, "scan_jsonl_files", lambda root, split_substr=None: ["f1.jsonl"])
    monkeypatch.setattr(f2, "gpt_keys_for_mode", lambda mode: ["k1"])
    monkeypatch.setattr(
        f2,
        "load_pass1_rows",
        lambda *args, **kwargs: pd.DataFrame(
            {"problem": ["p1"], "step": [1], "correct": [1], "uncertainty": [0.1], "aha_words": [0], "aha_gpt": [1]}
        ),
    )
    monkeypatch.setattr(
        f2,
        "build_problem_step_from_samples",
        lambda df, include_native=True, native_col="aha_words": df.assign(
            freq_correct=0.5,
            aha_rate_gpt=0.5,
            aha_any_gpt=1,
            p_correct_given_shift=0.6,
        ),
    )
    monkeypatch.setattr(f2, "make_formal_thresholds", lambda delta1, delta2, min_prior_steps, delta3: "thr")
    monkeypatch.setattr(f2, "mark_formal_pairs", lambda df, thresholds: df.assign(aha_formal=1))
    monkeypatch.setattr(f2, "label_formal_samples", lambda samples, ps: samples.assign(aha_formal=ps["aha_formal"]))

    df = f2._load_all_samples(args)
    assert "aha_formal" in df.columns

    # No files → SystemExit
    monkeypatch.setattr(f2, "scan_jsonl_files", lambda *a, **k: [])
    with pytest.raises(SystemExit):
        f2._load_all_samples(args)
