import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.analysis.cue_entropy_regression as cue_reg


def _write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_load_flat_df_normalizes_and_filters(tmp_path):
    data_path = _write_jsonl(
        tmp_path / "flat.jsonl",
        [
            {"cue_variant": "v1", "intervention_correct": True, "entropy": 0.5, "problem": "p1"},
            {"cue_variant": "baseline", "intervention_correct": False, "entropy": 0.6, "problem": "p1"},
            {"cue_variant": "v1", "baseline_correct": None, "entropy": 0.7, "problem": "p2"},
        ],
    )
    df = cue_reg._load_flat_df(data_path)
    assert set(df["cue_variant"]) == {"v1"}
    assert df["intervention_correct"].tolist() == [1, 0]
    assert df["baseline_correct"].tolist() == [0, 0]


def test_load_flat_df_missing_entropy_errors(tmp_path):
    bad_path = _write_jsonl(tmp_path / "missing.jsonl", [{"cue_variant": "v1"}])
    with pytest.raises(RuntimeError):
        cue_reg._load_flat_df(bad_path)

    null_entropy = _write_jsonl(
        tmp_path / "null.jsonl",
        [{"cue_variant": "v1", "entropy": None, "intervention_correct": 1}],
    )
    with pytest.raises(RuntimeError):
        cue_reg._load_flat_df(null_entropy)


def test_feature_matrix_and_scaling_and_degeneracy():
    subset = pd.DataFrame(
        {
            "entropy": [0.5, 1.0, 1.5],
            "baseline_correct": [0, 1, 0],
            "problem": ["p1", "p1", "p2"],
        }
    )
    design = cue_reg._build_feature_matrix(subset)
    assert design.shape[1] == 3  # entropy, baseline_correct, one-hot

    scaled, sd = cue_reg._scale_features(design)
    assert scaled.shape == design.shape
    assert sd is not None and sd > 0

    sample_size, degenerate = cue_reg._is_degenerate_outcome(np.array([0, 0, 0]))
    assert sample_size == 3 and degenerate is True


def test_train_penalized_model_and_wald_stats():
    features = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    target = np.array([0, 0, 1])
    scaled, _ = cue_reg._scale_features(features)
    model_result, err = cue_reg._train_penalized_model(scaled, target, regularization_strength=1.0, max_iter=200)
    assert err is None
    coef, probs = model_result
    assert isinstance(coef, float)
    assert probs.shape == (3,)

    stats = cue_reg._wald_stats(scaled, probs, coef)
    assert stats and {"se", "z", "p"} <= set(stats)


def test_build_summary_and_fit_logit_roundtrip(monkeypatch):
    monkeypatch.setattr(
        cue_reg,
        "_wald_stats",
        lambda feature_matrix, probs, coef: {"se": 0.1, "z": coef / 0.1, "p": 0.05},
    )
    df = pd.DataFrame(
        {
            "cue_variant": ["v1"] * 6,
            "intervention_correct": [1, 0, 1, 0, 1, 0],
            "baseline_correct": [0, 1, 0, 1, 0, 1],
            "entropy": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
            "problem": ["p1", "p1", "p2", "p2", "p3", "p3"],
        }
    )
    summary = cue_reg._fit_logit(df, "v1", regularization_strength=1.0, max_iter=200)
    assert summary["cue"] == "v1"
    assert "entropy_coef" in summary
    assert "entropy_odds_ratio" in summary

    built = cue_reg._build_summary(
        cue="c",
        sample_size=3,
        coef_entropy_std=0.5,
        wald={"se": 0.1, "z": 5.0, "p": 0.001},
        entropy_sd=2.0,
    )
    assert built["entropy_or_unit_ci_low"] < built["entropy_or_unit_ci_high"]
    assert built["entropy_odds_ratio_1sd"] == pytest.approx(np.exp(0.5))


def test_format_helpers():
    assert cue_reg._format_scalar(1.2345, ".2f") == "1.23"
    assert cue_reg._format_scalar(None, ".2f") == "n/a"
    assert cue_reg._format_ci(0.1, 0.2, ".1f") == "[0.1, 0.2]"
    assert cue_reg._format_ci(None, 0.2, ".1f") == "[n/a, n/a]"


def test_load_flat_df_empty_raises(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError):
        cue_reg._load_flat_df(empty)


def test_wald_stats_zero_std_error(monkeypatch):
    feature_matrix = np.array([[0.0]])
    probs = np.array([0.5])
    monkeypatch.setattr(
        cue_reg.np.linalg,
        "inv",
        lambda m: np.array([[1.0, 0.0], [0.0, 0.0]]),  # forces std_error == 0
    )
    assert cue_reg._wald_stats(feature_matrix, probs, coef_entropy=0.0) is None


def test_fit_logit_handles_empty_and_model_error(monkeypatch):
    df = pd.DataFrame(
        {
            "cue_variant": ["c1", "c1"],
            "intervention_correct": [1, 0],
            "baseline_correct": [0, 1],
            "entropy": [0.1, 0.2],
            "problem": ["p1", "p2"],
        }
    )
    assert cue_reg._fit_logit(df, "missing", 1.0, 10) == {}

    class FailingLogit:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, feature_scaled, target):
            raise ValueError("boom")

        def predict_proba(self, feature_scaled):
            return np.array([[0.5, 0.5]])

        @property
        def coef_(self):
            return np.array([[0.0]])

    monkeypatch.setattr(cue_reg, "LogisticRegression", FailingLogit)
    result = cue_reg._fit_logit(df, "c1", 1.0, 10)
    assert result["error"].startswith("logit failed")


def test_build_summary_without_entropy_sd():
    summary = cue_reg._build_summary(
        cue="c",
        sample_size=2,
        coef_entropy_std=0.4,
        wald={"se": 0.1, "z": 4.0, "p": 0.05},
        entropy_sd=float("nan"),
    )
    assert summary["entropy_sd"] is None
    assert summary["entropy_odds_ratio"] == pytest.approx(np.exp(0.4))


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog"])
    args = cue_reg.parse_args()
    assert args.C == 1.0
    assert args.max_iter == 2500


def test_ensure_flat_path_errors_and_cleans(monkeypatch, tmp_path):
    args_missing = SimpleNamespace(flat_jsonl=None, input_jsonl=None)
    with pytest.raises(RuntimeError):
        cue_reg._ensure_flat_path(args_missing)

    args_direct = SimpleNamespace(flat_jsonl=tmp_path / "flat.jsonl", input_jsonl=None)
    path, cleanup = cue_reg._ensure_flat_path(args_direct)
    assert path == args_direct.flat_jsonl and cleanup is None

    # When flattening fails, temp file should be removed.
    args_needs_flat = SimpleNamespace(flat_jsonl=None, input_jsonl=tmp_path / "input.jsonl")
    args_needs_flat.input_jsonl.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cue_reg, "flatten_math_cue_variants", lambda *a, **k: (_ for _ in ()).throw(ValueError("x")))
    with pytest.raises(ValueError):
        cue_reg._ensure_flat_path(args_needs_flat)
    leftovers = {p for p in tmp_path.glob("*.jsonl") if p != args_needs_flat.input_jsonl}
    assert not leftovers  # temp file unlinked


def test_cleanup_temp_path_removes(tmp_path):
    tmp_file = tmp_path / "temp.jsonl"
    tmp_file.write_text("x", encoding="utf-8")
    cue_reg.cleanup_temp_path(tmp_file)
    assert not tmp_file.exists()


def test_main_uses_stubbed_helpers(monkeypatch, tmp_path, capsys):
    flat = tmp_path / "flat.jsonl"
    flat.write_text("", encoding="utf-8")

    monkeypatch.setattr(cue_reg, "_ensure_flat_path", lambda args: (flat, None))
    monkeypatch.setattr(
        cue_reg,
        "_load_flat_df",
        lambda path: pd.DataFrame(
            {
                "cue_variant": ["c1"],
                "intervention_correct": [1],
                "baseline_correct": [0],
                "entropy": [0.1],
                "problem": ["p1"],
            }
        ),
    )
    calls = {}
    monkeypatch.setattr(cue_reg, "_fit_logit", lambda *a, **k: {"cue": "c1", "entropy_coef": 0.0})
    monkeypatch.setattr(cue_reg, "_print_summary_line", lambda cue, summary: calls.setdefault("printed", True))
    monkeypatch.setattr("sys.argv", ["prog", "--flat-jsonl", str(flat)])

    cue_reg.main()
    assert calls.get("printed")


def test_main_runs_and_cleans(monkeypatch, tmp_path):
    calls = {}
    args = SimpleNamespace(flat_jsonl=tmp_path / "flat.jsonl", input_jsonl=None, C=1.0, max_iter=5, csv=None)
    monkeypatch.setattr(cue_reg, "parse_args", lambda: args)
    monkeypatch.setattr(cue_reg, "_ensure_flat_path", lambda a: (args.flat_jsonl, "CLEANUP"))
    monkeypatch.setattr(
        cue_reg,
        "_load_flat_df",
        lambda path: pd.DataFrame(
            {
                "cue_variant": ["c1", "c2"],
                "intervention_correct": [1, 0],
                "baseline_correct": [0, 1],
                "entropy": [0.1, 0.2],
                "problem": ["p1", "p2"],
            }
        ),
    )
    monkeypatch.setattr(
        cue_reg,
        "_fit_logit",
        lambda df, cue, regularization_strength, max_iter: {
            "cue": cue,
            "sample_size": len(df),
            "n": len(df),
            "entropy_coef": 0.1,
            "entropy_se": 0.1,
            "entropy_pvalue": 0.5,
            "entropy_ci_low": -0.1,
            "entropy_ci_high": 0.3,
            "entropy_sd": 1.0,
            "entropy_odds_ratio": 1.1,
            "entropy_or_unit_ci_low": 1.0,
            "entropy_or_unit_ci_high": 1.2,
            "entropy_odds_ratio_1sd": 1.1,
            "entropy_or_1sd_ci_low": 0.9,
            "entropy_or_1sd_ci_high": 1.3,
        },
    )
    monkeypatch.setattr(
        cue_reg, "_print_summary_line", lambda cue, summary: calls.setdefault("printed", []).append(cue)
    )
    monkeypatch.setattr(cue_reg, "_cleanup_temp_path", lambda path: calls.setdefault("cleanup", path))

    cue_reg.main()

    assert calls["cleanup"] == "CLEANUP"
    assert set(calls["printed"]) == {"c1", "c2"}


def test_main_exec_runpy(monkeypatch, tmp_path):
    import runpy
    import sys

    # Patch upstream helpers used inside the module before exec.
    monkeypatch.setattr(
        "src.analysis.io.iter_records_from_file",
        lambda path: [
            {
                "cue_variant": "v1",
                "intervention_correct": True,
                "baseline_correct": False,
                "entropy": 0.1,
                "problem": "p1",
            },
            {
                "cue_variant": "v1",
                "intervention_correct": False,
                "baseline_correct": True,
                "entropy": 0.2,
                "problem": "p2",
            },
        ],
        raising=False,
    )
    monkeypatch.setattr("src.analysis.cue_delta_accuracy._cleanup_temp_path", lambda path: None)

    flat_path = tmp_path / "flat.jsonl"
    flat_path.write_text("", encoding="utf-8")
    sys.argv = ["prog", "--flat-jsonl", str(flat_path)]

    # Execute module as __main__ to hit the guard at the bottom.
    sys.modules.pop("src.analysis.cue_entropy_regression", None)
    result = runpy.run_module("src.analysis.cue_entropy_regression", run_name="__main__")
    # run_module returns globals; ensure main defined
    assert "main" in result


def test_main_skips_empty_summaries(monkeypatch, tmp_path, capsys):
    dummy_path = tmp_path / "flat.jsonl"
    dummy_path.write_text('{"cue_variant": "c1", "entropy": 0.1, "intervention_correct": 1, "problem": "p1"}\n')
    monkeypatch.setattr(
        cue_reg,
        "parse_args",
        lambda: SimpleNamespace(flat_jsonl=dummy_path, input_jsonl=None, csv=None, C=1.0, max_iter=10),
    )
    monkeypatch.setattr(cue_reg, "_ensure_flat_path", lambda args: (args.flat_jsonl, None))
    monkeypatch.setattr(cue_reg, "_load_flat_df", lambda path: pd.DataFrame({"cue_variant": ["c1"]}))
    monkeypatch.setattr(cue_reg, "_fit_logit", lambda *a, **k: {})
    closed = {}
    monkeypatch.setattr(cue_reg, "_cleanup_temp_path", lambda p: closed.setdefault("done", True))
    cue_reg.main()
    out = capsys.readouterr().out
    assert "Cue regression summaries" in out
    assert closed.get("done")


def test_print_summary_line_and_main_writes_csv(monkeypatch, tmp_path, capsys):
    summary = {
        "entropy_coef_std": 0.5,
        "entropy_or_unit": 1.1,
        "entropy_or_unit_ci_low": 0.9,
        "entropy_or_unit_ci_high": 1.3,
        "entropy_odds_ratio_1sd": 1.4,
        "entropy_or_1sd_ci_low": 1.2,
        "entropy_or_1sd_ci_high": 1.6,
        "entropy_pvalue": 0.0123,
        "entropy_sd": 0.7,
        "entropy_se": 0.05,
    }
    cue_reg._print_summary_line("cueA", summary)
    out = capsys.readouterr().out
    assert "cueA" in out and "OR_unit" in out

    csv_path = tmp_path / "out.csv"
    dummy_path = tmp_path / "flat2.jsonl"
    dummy_path.write_text('{"cue_variant": "c1", "entropy": 0.1, "intervention_correct": 1, "problem": "p1"}\n')
    monkeypatch.setattr(
        cue_reg,
        "parse_args",
        lambda: SimpleNamespace(flat_jsonl=dummy_path, input_jsonl=None, csv=csv_path, C=1.0, max_iter=10),
    )
    monkeypatch.setattr(cue_reg, "_ensure_flat_path", lambda args: (args.flat_jsonl, None))
    monkeypatch.setattr(cue_reg, "_load_flat_df", lambda path: pd.DataFrame({"cue_variant": ["c1"]}))
    monkeypatch.setattr(cue_reg, "_fit_logit", lambda *a, **k: dict(summary, cue="c1"))
    cue_reg.main()
    assert csv_path.exists()
    written = csv_path.read_text()
    assert "entropy_coef_std" in written and "c1" in written
