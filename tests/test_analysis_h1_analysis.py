#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import runpy
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import src.analysis.h1_analysis as h1


def test_build_sample_row_gates_gpt(monkeypatch):
    record = {"pass1": {"is_correct_pred": True}}
    monkeypatch.setattr(h1, "extract_pass1_and_step", lambda rec, step: (rec["pass1"], 2))
    monkeypatch.setattr(h1, "problem_key_from_record", lambda rec, missing_default=None: "p1")
    monkeypatch.setattr(h1, "coerce_bool", lambda value: 1)
    # Words returns 1, canonical GPT returns True, broad returns False (to check branch)
    monkeypatch.setattr(h1, "aha_words", lambda pass1: True)
    monkeypatch.setattr(h1, "aha_gpt_canonical", lambda pass1, record: True)
    monkeypatch.setattr(h1, "aha_gpt_broad", lambda pass1, record: False)

    row_gated = h1._build_sample_row(record, 1, gpt_mode="canonical", gate_gpt_by_words=True)
    assert row_gated["aha_gpt"] == 1  # gated keeps True & words True
    row_ungated = h1._build_sample_row(record, 1, gpt_mode="broad", gate_gpt_by_words=False)
    assert row_ungated["aha_gpt"] == 0  # broad stub returns False


def test_build_sample_row_skips_when_missing(monkeypatch):
    # No pass1 or step -> None
    monkeypatch.setattr(h1, "extract_pass1_and_step", lambda rec, step: (None, None))
    assert h1._build_sample_row({}, None, "canonical", True) is None
    # Missing correctness -> None
    monkeypatch.setattr(h1, "extract_pass1_and_step", lambda rec, step: ({"is_correct_pred": None}, 0))
    monkeypatch.setattr(h1, "coerce_bool", lambda v: None)
    assert h1._build_sample_row({}, None, "canonical", True) is None


def test_load_samples_and_empty(monkeypatch):
    monkeypatch.setattr(h1, "nat_step_from_path", lambda path: 3)
    monkeypatch.setattr(
        h1,
        "iter_records_from_file",
        lambda path: [
            {"pass1": {"is_correct_pred": True}, "problem": "p"},
            {"pass1": {"is_correct_pred": False}, "problem": "p"},
        ],
    )
    monkeypatch.setattr(
        h1,
        "_build_sample_row",
        lambda rec, step, gpt_mode, gate_gpt_by_words: {
            "problem": "p",
            "step": step,
            "correct": 1,
            "aha_words": 1,
            "aha_gpt": 0,
        },
    )
    df = h1.load_samples(["file1"], gpt_mode="canonical", gate_gpt_by_words=True)
    assert not df.empty
    assert set(df.columns) >= {"problem", "step", "correct", "aha_words", "aha_gpt"}

    # No rows -> SystemExit
    monkeypatch.setattr(h1, "_build_sample_row", lambda *args, **kwargs: None)
    with pytest.raises(SystemExit):
        h1.load_samples(["file1"], gpt_mode="canonical", gate_gpt_by_words=True)


def test_compute_glm_metrics_handles_missing_group(monkeypatch):
    class DummyRes:
        def predict(self, data):
            return np.ones(len(data)) * 0.5

    glm_data = pd.DataFrame({"correct": [1, 1], "aha_words": [1, 1], "step": [0, 1]})
    metrics = h1._compute_glm_metrics(glm_data, DummyRes(), "aha_words")
    assert np.isnan(metrics["delta_acc"])


def test_fit_glm_raises_without_statsmodels(monkeypatch, tmp_path):
    glm_data = pd.DataFrame({"correct": [1], "step": [0], "aha_words": [1]})
    monkeypatch.setattr(h1, "sm", None)
    monkeypatch.setattr(h1, "smf", None)
    with pytest.raises(RuntimeError):
        h1.fit_glm(glm_data, "aha_words", out_txt=str(tmp_path / "x.txt"))

    # Restore minimal stubs and trigger missing column error
    monkeypatch.setattr(h1, "sm", SimpleNamespace(families=SimpleNamespace(Binomial=lambda: None)))
    monkeypatch.setattr(h1, "smf", SimpleNamespace(glm=lambda *a, **k: None))
    with pytest.raises(ValueError):
        h1.fit_glm(pd.DataFrame({"correct": [1], "step": [0]}), "aha_words", out_txt=str(tmp_path / "x.txt"))


def test_build_problem_step_and_mark_formal(monkeypatch):
    samples_df = pd.DataFrame(
        [
            {"problem": "p1", "step": 0, "correct": 0, "aha_gpt": 0},
            {"problem": "p1", "step": 1, "correct": 1, "aha_gpt": 1},
            {"problem": "p2", "step": 0, "correct": 1, "aha_gpt": 0},
        ],
    )
    problem_step = h1.build_problem_step(samples_df)
    assert problem_step.loc[0, "n_samples"] == 1
    assert problem_step["aha_any_gpt"].max() == 1

    marked = h1.mark_formal(problem_step, delta1=0.5, delta2=0.5, min_prior_steps=1)
    assert "aha_formal_ps" in marked.columns
    # First step cannot be formal because of min_prior_steps
    assert marked.loc[marked["step"] == 0, "aha_formal_ps"].eq(0).all()

    with pytest.raises(ValueError):
        h1.mark_formal(pd.DataFrame([{"problem": "p"}]), 0.1, 0.1, 0)


def test_compute_glm_metrics_and_fit(monkeypatch, tmp_path):
    # Dummy glm_result that returns deterministic predictions
    class DummyResult:
        def __init__(self):
            self.params = {"aha_words": 0.5}
            self.bse = {"aha_words": 0.25}
            self.pvalues = {"aha_words": 0.1}

        def predict(self, data):
            # Return a simple function of the aha column to vary with input
            return np.array([0.2 + 0.3 * a for a in data["aha_words"]])

    dummy_result = DummyResult()
    glm_data = pd.DataFrame({"correct": [0, 1], "aha_words": [0, 1], "step": [0, 1]})
    metrics = h1._compute_glm_metrics(glm_data, dummy_result, "aha_words")
    assert metrics["ame"] == pytest.approx(0.3, rel=1e-5)
    assert metrics["acc_overall"] == pytest.approx(0.5)
    assert metrics["acc_aha1"] == pytest.approx(1.0)

    # Stub out statsmodels + covariance helpers for fit_glm
    class DummyModel:
        pass

    monkeypatch.setattr(h1, "sm", SimpleNamespace(families=SimpleNamespace(Binomial=lambda: "binom")))
    monkeypatch.setattr(h1, "smf", SimpleNamespace(glm=lambda formula, data, family: DummyModel()))
    monkeypatch.setattr(
        h1,
        "glm_fit_with_covariance",
        lambda model, data, cluster_by: (dummy_result, "cluster", {"group": cluster_by}),
    )
    monkeypatch.setattr(h1, "write_glm_summary_header", lambda path, res, cov_type, cov_kwds: None)

    out_txt = tmp_path / "glm.txt"
    result = h1.fit_glm(glm_data, aha_col="aha_words", out_txt=str(out_txt), cluster_by="problem")
    assert result["N"] == 2
    assert result["coef"] == 0.5
    assert result["z"] == 2.0
    assert out_txt.exists()


def test_group_accuracy_tables(tmp_path):
    samples = pd.DataFrame(
        [
            {"variant": "v", "problem": "p", "step": 0, "correct": 1, "aha_words": 1, "aha_gpt": 0, "aha_formal": 1},
            {"variant": "v", "problem": "p", "step": 0, "correct": 0, "aha_words": 0, "aha_gpt": 0, "aha_formal": 0},
        ],
    )
    acc_csv, delta_csv, step_csv = h1.compute_group_accuracy_tables(samples, str(tmp_path))
    assert os.path.exists(acc_csv)
    assert os.path.exists(delta_csv)
    assert os.path.exists(step_csv)

    acc_df = pd.read_csv(acc_csv)
    assert set(acc_df["variant"]) == {"words", "gpt", "formal"}


def test_group_accuracy_tables_skips_missing(tmp_path):
    samples = pd.DataFrame([{"problem": "p", "step": 0, "correct": 1, "aha_words": 1}])
    acc_csv, delta_csv, step_csv = h1.compute_group_accuracy_tables(samples, str(tmp_path))
    delta_df = pd.read_csv(delta_csv)
    assert set(delta_df["variant"]) == {"words"}
    by_step_df = pd.read_csv(step_csv)
    assert set(by_step_df["variant"]) == {"words"}


def test_format_and_draw_tables():
    assert h1._fmt_float_or_str(1.23456) == "1.2346"
    assert h1._fmt_float_or_str("x") == "x"

    fig, axis = plt.subplots()
    summary_df = pd.DataFrame(
        [
            {
                "variant": "words",
                "N": 2,
                "share_aha": 0.5,
                "acc_aha1": 1.0,
                "acc_aha0": 0.0,
                "delta_acc": 0.5,
                "AME": 0.1,
                "coef": 0.2,
                "p": 0.05,
            },
        ],
    )
    y_pos = h1._draw_glm_table(axis, summary_df)
    assert y_pos < 1.0

    acc_delta_df = pd.DataFrame(
        [
            {
                "variant": "words",
                "acc_aha1": 1.0,
                "acc_aha0": 0.0,
                "delta_acc": 0.5,
                "n_aha1": 1,
                "n_aha0": 1,
            },
        ],
    )
    h1._draw_accuracy_delta_table(axis, acc_delta_df, start_y=y_pos)
    plt.close(fig)


def test_write_a4_summary_pdf(monkeypatch, tmp_path):
    called = {}
    monkeypatch.setattr(h1, "set_global_fonts", lambda *args, **kwargs: called.setdefault("fonts", True))
    monkeypatch.setattr(h1, "create_a4_figure", lambda dpi=300: plt.figure())

    summary_df = pd.DataFrame(
        [
            {
                "variant": "words",
                "N": 1,
                "share_aha": 1.0,
                "acc_aha1": 1.0,
                "acc_aha0": 0.0,
                "delta_acc": 1.0,
                "AME": 0.1,
                "coef": 0.2,
                "p": 0.05,
            }
        ]
    )
    acc_delta_df = pd.DataFrame(
        [{"variant": "words", "acc_aha1": 1.0, "acc_aha0": 0.0, "delta_acc": 1.0, "n_aha1": 1, "n_aha0": 1}]
    )
    args = SimpleNamespace(dataset_name="d", model_name="m", font_family="Times", font_size=12)
    out_pdf = tmp_path / "out.pdf"
    h1.write_a4_summary_pdf(summary_df, acc_delta_df, str(out_pdf), args)
    assert out_pdf.exists()
    assert called["fonts"] is True


def test_load_and_prepare_samples(monkeypatch):
    args = SimpleNamespace(
        results_root="root",
        split=None,
        gpt_mode="canonical",
        no_gate_gpt_by_words=False,
        delta1=0.5,
        delta2=0.5,
        min_prior_steps=1,
    )

    monkeypatch.setattr(h1, "scan_files", lambda root, split: ["f1"])
    # Fake samples + formal data
    monkeypatch.setattr(
        h1,
        "load_samples",
        lambda files, gpt_mode, gate_gpt_by_words: pd.DataFrame(
            [
                {"problem": "p", "step": 0, "correct": 1, "aha_words": 1, "aha_gpt": 1},
                {"problem": "p", "step": 1, "correct": 0, "aha_words": 0, "aha_gpt": 0},
            ],
        ),
    )
    monkeypatch.setattr(
        h1,
        "build_problem_step",
        lambda samples_df: pd.DataFrame(
            [{"step": 0, "problem": "p", "aha_formal_ps": 0}, {"step": 1, "problem": "p", "aha_formal_ps": 1}],
        ),
    )
    monkeypatch.setattr(h1, "mark_formal", lambda ps_df, delta1, delta2, min_prior_steps: ps_df)

    df = h1._load_and_prepare_samples(args)
    assert "aha_formal" in df.columns
    assert (df["aha_formal"] <= df["aha_gpt"]).all()


def test_load_and_prepare_samples_no_files(monkeypatch):
    args = SimpleNamespace(
        results_root="root",
        split=None,
        gpt_mode="canonical",
        no_gate_gpt_by_words=False,
        delta1=0.5,
        delta2=0.5,
        min_prior_steps=1,
    )
    monkeypatch.setattr(h1, "scan_files", lambda root, split: [])
    with pytest.raises(SystemExit):
        h1._load_and_prepare_samples(args)


def test_load_and_prepare_samples_warns(monkeypatch, capsys):
    args = SimpleNamespace(
        results_root="root",
        split=None,
        gpt_mode="canonical",
        no_gate_gpt_by_words=True,  # disable gating so aha_gpt can exceed words
        delta1=0.5,
        delta2=0.5,
        min_prior_steps=0,
    )
    monkeypatch.setattr(h1, "scan_files", lambda root, split: ["f1"])
    monkeypatch.setattr(
        h1,
        "load_samples",
        lambda files, gpt_mode, gate_gpt_by_words: pd.DataFrame(
            [{"problem": "p", "step": 0, "correct": 1, "aha_words": 0, "aha_gpt": 1}],
        ),
    )
    monkeypatch.setattr(
        h1,
        "build_problem_step",
        lambda samples_df: pd.DataFrame([{"step": 0, "problem": "p", "aha_formal_ps": 1}]),
    )
    monkeypatch.setattr(h1, "mark_formal", lambda ps_df, delta1, delta2, min_prior_steps: ps_df)
    h1._load_and_prepare_samples(args)
    out = capsys.readouterr().out
    assert "[warn]" in out


def test_fit_glms_and_write_summary(monkeypatch, tmp_path):
    args = SimpleNamespace(cluster_by="problem", dataset_name="d", model_name="m")
    samples_df = pd.DataFrame(
        [{"problem": "p", "step": 0, "correct": 1, "aha_words": 1, "aha_gpt": 0, "aha_formal": 1}],
    )

    def fake_fit(samples_df, aha_col, out_txt, cluster_by):
        return {
            "N": len(samples_df),
            "aha": aha_col,
            "share_aha": 1.0,
            "acc_overall": 1.0,
            "acc_aha1": 1.0,
            "acc_aha0": 0.0,
            "delta_acc": 1.0,
            "coef": 0.1,
            "se": 0.1,
            "z": 1.0,
            "p": 0.05,
            "AME": 0.2,
            "summary_path": out_txt,
        }

    monkeypatch.setattr(h1, "fit_glm", fake_fit)

    summary_df, csv_path = h1._fit_glms_and_write_summary(args, samples_df, str(tmp_path))
    assert csv_path.endswith("h1_glm_ame_summary.csv")
    assert set(summary_df["variant"]) == {"words", "gpt", "formal"}
    assert os.path.exists(csv_path)


def test_write_latex_table(tmp_path):
    summary_df = pd.DataFrame(
        [
            {
                "variant": "words",
                "N": 1,
                "share_aha": 1.0,
                "acc_aha1": 1.0,
                "acc_aha0": 0.0,
                "delta_acc": 0.1,
                "AME": 0.2,
                "coef": 0.3,
                "p": 0.04,
            },
        ],
    )
    tex_path = tmp_path / "table.tex"
    h1._write_latex_table(summary_df, str(tex_path))
    assert tex_path.exists()
    assert "Variant" in tex_path.read_text()

    # Non-finite values should render as strings without raising
    summary_nan = pd.DataFrame(
        [
            {
                "variant": "nanrow",
                "N": 0,
                "share_aha": np.nan,
                "acc_aha1": np.nan,
                "acc_aha0": np.nan,
                "delta_acc": np.nan,
                "AME": np.nan,
                "coef": np.nan,
                "p": np.nan,
            },
        ],
    )
    tex_path2 = tmp_path / "table2.tex"
    h1._write_latex_table(summary_nan, str(tex_path2))
    assert tex_path2.exists()


def test_write_pdf_summary_and_print_console(monkeypatch, tmp_path, capsys):
    args = SimpleNamespace(dataset_name="d", model_name="m", font_family="Times", font_size=12)
    summary_df = pd.DataFrame(
        [
            {
                "variant": "words",
                "N": 1,
                "share_aha": 1.0,
                "acc_overall": 1.0,
                "acc_aha1": 1.0,
                "acc_aha0": 0.0,
                "delta_acc": 0.1,
                "AME": 0.2,
                "coef": 0.3,
                "se": 0.1,
                "z": 3.0,
                "p": 0.04,
                "summary_path": str(tmp_path / "glm.txt"),
            },
        ],
    )
    delta_csv = tmp_path / "delta.csv"
    pd.DataFrame(
        [{"variant": "words", "acc_aha1": 1.0, "acc_aha0": 0.0, "delta_acc": 0.1, "n_aha1": 1, "n_aha0": 1}],
    ).to_csv(delta_csv, index=False)

    called = {}
    monkeypatch.setattr(
        h1, "write_a4_summary_pdf", lambda summary_df, acc_delta_df, out_pdf, args: called.setdefault("pdf", out_pdf)
    )

    h1._write_pdf_summary(args, str(tmp_path), summary_df, str(delta_csv))
    assert "h1_glm_summary.pdf" in called["pdf"]

    h1._print_console_summary(summary_df, "summary.csv", "acc.csv", "delta.csv", "step.csv")
    captured = capsys.readouterr()
    assert "summary.csv" in captured.out
    assert "Delta" in captured.out or "Δ" in captured.out

    # Non-finite delta branch
    summary_nan = summary_df.copy()
    summary_nan.loc[:, "delta_acc"] = np.nan
    h1._print_console_summary(summary_nan, "summary.csv", "acc.csv", "delta.csv", "step.csv")


def test_main_happy_path(monkeypatch, tmp_path):
    args = SimpleNamespace(
        results_root=str(tmp_path),
        out_dir=str(tmp_path / "out"),
        split=None,
        gpt_mode="canonical",
        no_gate_gpt_by_words=False,
        delta1=0.5,
        delta2=0.5,
        min_prior_steps=1,
        cluster_by="problem",
        tex_path=None,
        make_pdf=False,
        font_family="Times",
        font_size=12,
        dataset_name="d",
        model_name="m",
    )
    monkeypatch.setattr(h1.argparse.ArgumentParser, "parse_args", lambda self: args)
    monkeypatch.setattr(
        h1,
        "_load_and_prepare_samples",
        lambda args: pd.DataFrame(
            [{"problem": "p", "step": 0, "correct": 1, "aha_words": 1, "aha_gpt": 1, "aha_formal": 1}],
        ),
    )
    monkeypatch.setattr(
        h1,
        "_fit_glms_and_write_summary",
        lambda args, df, out_dir: (
            pd.DataFrame(
                [
                    {
                        "variant": "words",
                        "N": 1,
                        "share_aha": 1.0,
                        "acc_aha1": 1.0,
                        "acc_aha0": 0.0,
                        "delta_acc": 0.1,
                        "AME": 0.2,
                        "coef": 0.3,
                        "se": 0.1,
                        "z": 3.0,
                        "p": 0.04,
                        "summary_path": "x",
                    }
                ]
            ),
            out_dir + "/h1_glm_ame_summary.csv",
        ),
    )
    monkeypatch.setattr(
        h1,
        "compute_group_accuracy_tables",
        lambda df, out_dir: (out_dir + "/acc.csv", out_dir + "/delta.csv", out_dir + "/step.csv"),
    )
    monkeypatch.setattr(h1, "_write_pdf_summary", lambda args, out_dir, summary_df, delta_csv: None)
    monkeypatch.setattr(h1, "_print_console_summary", lambda *args, **kwargs: None)

    h1.main()
    assert os.path.isdir(args.out_dir)


def test_main_with_tex_and_pdf(monkeypatch, tmp_path):
    out_dir = tmp_path / "out_pdf"
    tex_path = tmp_path / "table.tex"
    args = SimpleNamespace(
        results_root=str(tmp_path),
        out_dir=str(out_dir),
        split=None,
        gpt_mode="canonical",
        no_gate_gpt_by_words=False,
        delta1=0.5,
        delta2=0.5,
        min_prior_steps=1,
        cluster_by="problem",
        tex_path=str(tex_path),
        make_pdf=True,
        font_family="Times",
        font_size=12,
        dataset_name="d",
        model_name="m",
    )
    monkeypatch.setattr(h1.argparse.ArgumentParser, "parse_args", lambda self: args)
    samples_df = pd.DataFrame(
        [{"problem": "p", "step": 0, "correct": 1, "aha_words": 1, "aha_gpt": 1, "aha_formal": 1}],
    )
    summary_df = pd.DataFrame(
        [
            {
                "variant": "words",
                "N": 1,
                "share_aha": 1.0,
                "acc_overall": 1.0,
                "acc_aha1": 1.0,
                "acc_aha0": 0.0,
                "delta_acc": 0.1,
                "AME": 0.2,
                "coef": 0.3,
                "se": 0.1,
                "z": 3.0,
                "p": 0.04,
                "summary_path": "x",
            },
        ],
    )
    monkeypatch.setattr(h1, "_load_and_prepare_samples", lambda args: samples_df)
    monkeypatch.setattr(
        h1,
        "_fit_glms_and_write_summary",
        lambda args, df, out_dir: (summary_df, os.path.join(str(out_dir), "summary.csv")),
    )
    monkeypatch.setattr(
        h1,
        "compute_group_accuracy_tables",
        lambda df, out_dir: (
            os.path.join(str(out_dir), "acc.csv"),
            os.path.join(str(out_dir), "delta.csv"),
            os.path.join(str(out_dir), "step.csv"),
        ),
    )

    calls = {"tex": None, "pdf": None, "print": False}

    def _stub_latex(summary_df, path):
        calls["tex"] = path
        tex_path.write_text("stub")

    def _stub_pdf(args, out_dir, summary_df, delta_csv):
        calls["pdf"] = delta_csv

    monkeypatch.setattr(h1, "_write_latex_table", _stub_latex)
    monkeypatch.setattr(h1, "_write_pdf_summary", _stub_pdf)
    monkeypatch.setattr(h1, "_print_console_summary", lambda *a, **k: calls.__setitem__("print", True))

    h1.main()

    assert calls["tex"] == str(tex_path)
    assert calls["pdf"].endswith("delta.csv")
    assert calls["print"] is True
    assert tex_path.exists()
    assert out_dir.is_dir()


def test_module_entrypoint_runs_with_help(monkeypatch):
    monkeypatch.delitem(sys.modules, "src.analysis.h1_analysis", raising=False)
    monkeypatch.setattr(sys, "argv", ["h1_analysis.py", "--help"])
    with pytest.raises(SystemExit):
        runpy.run_module("src.analysis.h1_analysis", run_name="__main__")
