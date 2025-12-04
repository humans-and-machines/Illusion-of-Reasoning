import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.analysis.cue_entropy_regression as cue_reg


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_load_flat_df_normalizes_and_drops_baseline(tmp_path):
    path = tmp_path / "flat.jsonl"
    _write_jsonl(
        path,
        [
            {
                "cue_variant": "baseline",
                "intervention_correct": True,
                "baseline_correct": None,
                "entropy": 0.5,
                "problem": "p0",
            },
            {
                "cue_variant": "cueA",
                "intervention_correct": None,
                "entropy": 1.0,
                "problem": "p1",
            },
        ],
    )
    df = cue_reg._load_flat_df(path)
    assert set(df["cue_variant"]) == {"cueA"}
    assert df["intervention_correct"].tolist() == [0]
    assert df["baseline_correct"].tolist() == [0]


def test_load_flat_df_errors_on_missing_entropy(tmp_path):
    path = tmp_path / "flat_missing.jsonl"
    _write_jsonl(path, [{"cue_variant": "cueA", "intervention_correct": True}])
    with pytest.raises(RuntimeError):
        cue_reg._load_flat_df(path)


def test_build_feature_matrix_and_scale():
    df = pd.DataFrame(
        {
            "entropy": [0.1, 0.2],
            "baseline_correct": [1, 0],
            "problem": ["a", "b"],
        }
    )
    mat = cue_reg._build_feature_matrix(df)
    assert mat.shape[1] >= 3  # entropy, baseline, at least one dummy
    scaled, sd = cue_reg._scale_features(mat)
    assert scaled.shape == mat.shape
    assert sd is not None and sd > 0


def test_train_penalized_model_success_and_failure():
    features = np.array([[0.0, 0.0], [1.0, 1.0]])
    target = np.array([0, 1])
    result, error = cue_reg._train_penalized_model(features, target, 1.0, 100)
    assert error is None
    coef, probs = result
    assert isinstance(coef, float) and probs.shape == (2,)

    bad_features = np.array([[np.nan, 0.0], [np.nan, 1.0]])
    result, error = cue_reg._train_penalized_model(bad_features, target, 1.0, 100)
    assert result is None and "logit failed" in error


def test_is_degenerate_outcome_flags_all_zero_and_one():
    size, degenerate = cue_reg._is_degenerate_outcome(np.array([0, 0, 0]))
    assert size == 3 and degenerate is True
    size, degenerate = cue_reg._is_degenerate_outcome(np.array([0, 1, 0]))
    assert size == 3 and degenerate is False


def test_wald_stats_and_ci_bounds():
    feature_matrix = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    probs = np.array([0.4, 0.6, 0.5])
    stats = cue_reg._wald_stats(feature_matrix, probs, coef_entropy=0.5)
    assert stats is not None
    ci = cue_reg._ci_bounds(0.0, 1.0)
    assert ci == (-1.96, 1.96)


def test_wald_stats_handles_singular():
    feature_matrix = np.array([[0.0, 0.0], [0.0, 0.0]])
    probs = np.array([0.0, 0.0])
    assert cue_reg._wald_stats(feature_matrix, probs, coef_entropy=1.0) is None


def test_build_summary_maps_entropy_sd():
    summary = cue_reg._build_summary(
        cue="cueA",
        sample_size=4,
        coef_entropy_std=0.2,
        wald={"se": 0.1, "z": 2.0, "p": 0.05},
        entropy_sd=2.0,
    )
    assert summary["entropy_odds_ratio"] == pytest.approx(np.exp(0.1))
    assert summary["sample_size"] == 4


def test_fit_logit_handles_degenerate_and_success(tmp_path):
    df = pd.DataFrame(
        {
            "cue_variant": ["cueA"] * 3,
            "intervention_correct": [1, 1, 1],
            "baseline_correct": [0, 1, 0],
            "entropy": [0.1, 0.2, 0.3],
            "problem": ["p1", "p1", "p2"],
        }
    )
    out = cue_reg._fit_logit(df, "cueA", regularization_strength=1.0, max_iter=100)
    assert "degenerate outcome" in out["error"]

    df = pd.DataFrame(
        {
            "cue_variant": ["cueA"] * 6,
            "intervention_correct": [1, 0, 1, 0, 1, 0],
            "baseline_correct": [0, 1, 0, 1, 0, 1],
            "entropy": [0.1, 0.2, 0.3, 0.4, 0.15, 0.35],
            "problem": ["p1", "p1", "p2", "p2", "p3", "p3"],
        }
    )
    out = cue_reg._fit_logit(df, "cueA", regularization_strength=1.0, max_iter=200)
    assert out["cue"] == "cueA"
    assert "entropy_coef" in out


def test_fit_logit_reports_wald_failure(monkeypatch):
    df = pd.DataFrame(
        {
            "cue_variant": ["cueA"] * 2,
            "intervention_correct": [1, 0],
            "baseline_correct": [0, 1],
            "entropy": [0.1, 0.2],
            "problem": ["p1", "p2"],
        }
    )
    monkeypatch.setattr(cue_reg, "_wald_stats", lambda *a, **k: None)
    out = cue_reg._fit_logit(df, "cueA", regularization_strength=1.0, max_iter=10)
    assert "wald failed" in out["error"]


def test_formatters_and_print_summary_line(capsys):
    assert cue_reg._format_scalar(None, ".2f") == "n/a"
    assert cue_reg._format_ci(None, None, ".2f") == "[n/a, n/a]"
    cue_reg._print_summary_line(
        "cueA",
        {
            "entropy_coef": 0.1,
            "entropy_se": 0.05,
            "entropy_ci_low": 0.0,
            "entropy_ci_high": 0.2,
            "entropy_odds_ratio": 1.1,
            "entropy_or_unit_ci_low": 1.0,
            "entropy_or_unit_ci_high": 1.2,
            "entropy_odds_ratio_1sd": 1.1,
            "entropy_or_1sd_ci_low": 1.0,
            "entropy_or_1sd_ci_high": 1.2,
            "entropy_sd": 0.5,
            "entropy_pvalue": 0.01,
            "error": None,
        },
    )
    out = capsys.readouterr().out
    assert "cueA" in out and "coef_std" in out
    cue_reg._print_summary_line("cueB", {"error": "oops"})
    out_err = capsys.readouterr().out
    assert "oops" in out_err


def test_ensure_flat_path_uses_flat_or_temp(monkeypatch, tmp_path):
    explicit_args = argparse.Namespace(flat_jsonl=tmp_path / "f.jsonl", input_jsonl=None)
    flat_path, cleanup = cue_reg._ensure_flat_path(explicit_args)
    assert flat_path == explicit_args.flat_jsonl and cleanup is None

    created = tmp_path / "input.jsonl"
    _write_jsonl(created, [{"cue_variant": "c", "entropy": 0.1}])

    def fake_flatten(input_path, output_path):
        assert input_path == str(created)
        Path(output_path).write_text(Path(input_path).read_text(), encoding="utf-8")
        return output_path

    monkeypatch.setattr(cue_reg, "flatten_math_cue_variants", fake_flatten)
    args = argparse.Namespace(flat_jsonl=None, input_jsonl=created)
    flat_path, cleanup_path = cue_reg._ensure_flat_path(args)
    assert flat_path.exists() and cleanup_path == flat_path
    cleanup_path.unlink()


def test_cleanup_temp_path_fallback_reimport(monkeypatch, tmp_path):
    # Force import path where cleanup_temp_path is missing to trigger fallback stub.
    original_mod = sys.modules.get("src.analysis.cue_entropy_regression")
    monkeypatch.delitem(sys.modules, "src.analysis.cue_entropy_regression", raising=False)
    stub_delta = types.ModuleType("src.analysis.cue_delta_accuracy")
    monkeypatch.setitem(sys.modules, "src.analysis.cue_delta_accuracy", stub_delta)
    mod = importlib.import_module("src.analysis.cue_entropy_regression")
    temp_file = tmp_path / "temp.txt"
    temp_file.write_text("x", encoding="utf-8")
    mod._cleanup_temp_path(temp_file)
    assert not temp_file.exists()
    if original_mod is not None:
        sys.modules["src.analysis.cue_entropy_regression"] = original_mod


def test_main_cleanup_with_temp(monkeypatch, tmp_path, capsys):
    input_path = tmp_path / "input.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "cue_variant": "cueA",
                "intervention_correct": True,
                "baseline_correct": False,
                "entropy": 0.2,
                "problem": "p1",
            },
            {
                "cue_variant": "cueA",
                "intervention_correct": False,
                "baseline_correct": True,
                "entropy": 0.4,
                "problem": "p2",
            },
        ],
    )

    def fake_flatten(inp, out):
        Path(out).write_text(Path(inp).read_text(), encoding="utf-8")
        return out

    cleanup_calls = []
    monkeypatch.setattr(cue_reg, "flatten_math_cue_variants", fake_flatten)
    monkeypatch.setattr(cue_reg, "_cleanup_temp_path", lambda path: cleanup_calls.append(path))
    args = argparse.Namespace(
        flat_jsonl=None,
        input_jsonl=input_path,
        csv=None,
        C=1.0,
        max_iter=20,
        solver="lbfgs",
    )
    monkeypatch.setattr(cue_reg, "parse_args", lambda: args)

    cue_reg.main()
    out = capsys.readouterr().out
    assert "cueA" in out
    assert cleanup_calls and cleanup_calls[0] is not None


def test_main_runs_and_writes_csv(monkeypatch, tmp_path, capsys):
    flat_path = tmp_path / "flat.jsonl"
    _write_jsonl(
        flat_path,
        [
            {
                "cue_variant": "cueA",
                "intervention_correct": True,
                "baseline_correct": False,
                "entropy": 0.2,
                "problem": "p1",
            },
            {
                "cue_variant": "cueA",
                "intervention_correct": False,
                "baseline_correct": True,
                "entropy": 0.4,
                "problem": "p2",
            },
        ],
    )
    args = argparse.Namespace(
        flat_jsonl=flat_path,
        input_jsonl=None,
        csv=tmp_path / "out.csv",
        C=1.0,
        max_iter=100,
        solver="lbfgs",
    )
    monkeypatch.setattr(cue_reg, "parse_args", lambda: args)
    cleanup_calls = []
    monkeypatch.setattr(cue_reg, "_cleanup_temp_path", lambda path: cleanup_calls.append(path))

    cue_reg.main()
    out = capsys.readouterr().out
    assert "cueA" in out
    assert args.csv.exists()
    assert cleanup_calls == [None]
