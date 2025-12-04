from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.analysis.h2_temp_aha_eval as h2


def test_compute_aha_flags_modes(monkeypatch):
    monkeypatch.setattr(h2, "aha_words", lambda pass1: 1)
    monkeypatch.setattr(h2, "aha_gpt_canonical", lambda p, r: 1)
    monkeypatch.setattr(h2, "aha_gpt_broad", lambda p, r: 0)

    words, gpt, aha = h2._compute_aha_flags({}, {}, aha_mode="gpt", gate_gpt_by_words=True)
    assert words == 1 and gpt == 1 and aha == 1

    words2, gpt2, aha2 = h2._compute_aha_flags({}, {}, aha_mode="words", gate_gpt_by_words=False)
    assert (words2, gpt2, aha2) == (1, 0, 1)

    words3, gpt3, aha3 = h2._compute_aha_flags({}, {}, aha_mode="gpt_broad", gate_gpt_by_words=False)
    assert gpt3 == 0 and aha3 == 0

    words4, gpt4, aha4 = h2._compute_aha_flags({}, {}, aha_mode="none", gate_gpt_by_words=True)
    assert (words4, gpt4, aha4) == (1, 0, 0)


def test_load_samples_filters_and_decodes(monkeypatch):
    fake_files = ["a.jsonl"]
    monkeypatch.setattr(h2, "scan_jsonl_files", lambda root, split_substr=None: fake_files)
    # Build two records; one filtered by split.
    records = [
        (fake_files[0], 10, {"split": "train", "pass1": {"correct": True}, "problem": "p1", "sample_idx": 0}),
        (fake_files[0], 20, {"split": "test", "pass1": {"correct": False}, "problem": "p2", "sample_idx": 1}),
    ]
    monkeypatch.setattr(h2, "iter_pass1_records", lambda files: records)
    # Avoid reliance on label helpers.
    monkeypatch.setattr(h2, "_compute_aha_flags", lambda *a, **k: (0, 1, 1))
    df = h2.load_samples("root", split_filter="test", aha_mode="gpt", gate_gpt_by_words=False)
    assert df.shape[0] == 1
    row = df.iloc[0]
    assert row["problem"] == "p2"
    assert row["correct"] == 0
    assert row["aha"] == 1


def test_load_samples_skips_missing(monkeypatch):
    files = ["a.jsonl"]
    records = [
        (files[0], 1, {"kind": "no_pass1"}),
        (files[0], 2, {"kind": "no_correct", "pass1": {"correct": None}}),
        (files[0], 3, {"kind": "ok", "pass1": {"correct": True}, "problem": "p", "sample_idx": 0}),
    ]
    monkeypatch.setattr(h2, "scan_jsonl_files", lambda root, split_substr=None: files)
    monkeypatch.setattr(h2, "iter_pass1_records", lambda files: records)

    def fake_extract(rec, step_from_name):
        if rec["kind"] == "no_pass1":
            return None, None
        return rec.get("pass1"), rec.get("step", step_from_name)

    monkeypatch.setattr(h2, "extract_pass1_and_step", fake_extract)
    monkeypatch.setattr(
        h2, "problem_key_from_record", lambda rec, missing_default="unknown": rec.get("problem", "unknown")
    )
    monkeypatch.setattr(h2, "_compute_aha_flags", lambda *a, **k: (0, 0, 0))
    df = h2.load_samples("root", split_filter=None, aha_mode="gpt")
    assert df.shape[0] == 1
    assert df.iloc[0]["problem"] == "p"


def test_restrict_to_overlap_with_sample_idx_and_empty():
    df_high = pd.DataFrame({"problem": ["p1"], "step": [1], "sample_idx": [0], "x": [1]})
    df_low = pd.DataFrame({"problem": ["p1"], "step": [1], "sample_idx": [0], "y": [2]})
    h_over, l_over, keys = h2.restrict_to_overlap(df_high, df_low)
    assert keys == ["problem", "step", "sample_idx"]
    assert len(h_over) == len(l_over) == 1

    with pytest.raises(SystemExit):
        h2.restrict_to_overlap(df_high, pd.DataFrame({"problem": [], "step": [], "sample_idx": []}))


def test_assign_stage_quantile_and_fixed(monkeypatch):
    df = pd.DataFrame({"step": [1, 2, 3, 4, 5]})
    stage_df, info = h2.assign_stage(df, mode="quantile", quantiles=(0.2, 0.8))
    assert set(stage_df["stage"]) == {"early", "mid", "late"}
    assert info["mode"] == "quantile"

    with pytest.raises(SystemExit):
        h2.assign_stage(df, mode="fixed", bounds=None)

    stage_df2, info2 = h2.assign_stage(df, mode="fixed", bounds=(2, 4))
    assert info2["bounds"] == [2, 4]
    assert stage_df2.loc[stage_df2["step"] == 1, "stage"].iloc[0] == "early"


def test_attach_formal_sample_level(monkeypatch):
    samples = pd.DataFrame(
        {
            "run_label": ["high", "high"],
            "problem": ["p1", "p1"],
            "step": [1, 2],
            "aha_gpt": [1, 0],
        }
    )

    def fake_build(df):
        return pd.DataFrame({"run_label": ["high"], "problem": ["p1"], "step": [1]})

    def fake_add(df, out_column, **kwargs):
        out = df.copy()
        out[out_column] = [1]
        return out

    monkeypatch.setattr(h2, "build_problem_step_for_formal", fake_build)
    monkeypatch.setattr(h2, "add_standard_formal_flags", fake_add)

    merged = h2.attach_formal_sample_level(samples, delta1=0.1, delta2=0.2, min_prior_steps=1)
    assert "aha_formal_ps" in merged.columns
    assert merged.loc[merged["step"] == 1, "aha_formal"].iloc[0] == 1
    assert merged.loc[merged["step"] == 2, "aha_formal"].iloc[0] == 0


def _stub_statsmodules():
    class Families:
        class Binomial:
            pass

    class StubResult:
        def __init__(self, params):
            self.params = pd.Series(params)
            self.bse = pd.Series({k: 0.1 for k in params})
            self.pvalues = pd.Series({k: 0.05 for k in params})

        def summary(self):
            return SimpleNamespace(as_text=lambda: "summary")

    class StubModel:
        def __init__(self, formula, data, family):
            self.formula = formula
            self.data = data
            self.family = family

        def fit(self, cov_type=None, cov_kwds=None):
            return StubResult({"Intercept": 1.0, "temp_low": 0.5})

    class StubFormula:
        def glm(self, formula, data, family):
            return StubModel(formula, data, family)

    return SimpleNamespace(families=Families), StubFormula()


def test_fit_glm_binomial_and_stage(monkeypatch, tmp_path):
    monkeypatch.setattr(h2, "lazy_import_statsmodels", _stub_statsmodules)
    df = pd.DataFrame({"problem": ["p1", "p1"], "step": [1, 2], "correct": [1, 0], "temp_low": [0, 1], "aha": [1, 0]})

    summ, coef_df = h2.fit_glm_binomial(df, aha_col="aha", cluster_by="problem", out_txt=str(tmp_path / "out.txt"))
    assert "coef_temp_low" in summ and not coef_df.empty
    assert (tmp_path / "out.txt").exists()

    summ2, coef_df2 = h2.fit_glm_stage_interaction(
        df.assign(stage=["early", "late"]), aha_col=None, cluster_by="none", out_txt=str(tmp_path / "out2.txt")
    )
    assert "coef_temp_low" in summ2 and not coef_df2.empty


def test_fit_glm_binomial_typeerror(monkeypatch):
    class StubResult:
        def __init__(self, params):
            self.params = pd.Series(params)
            self.bse = pd.Series({k: 0.2 for k in params})
            self.pvalues = pd.Series({k: 0.1 for k in params})

        def summary(self):
            return SimpleNamespace(as_text=lambda: "s")

    class StubModel:
        def fit(self, cov_type=None, cov_kwds=None):
            if cov_kwds is not None:
                raise TypeError("no cov_kwds here")
            return StubResult({"Intercept": 0.5, "temp_low": 0.2, "aha": 0.3, "temp_low:aha": 0.1})

    class StubFormula:
        def glm(self, formula, data, family):
            return StubModel()

    class StubFamilies:
        class Binomial:
            pass

    monkeypatch.setattr(h2, "lazy_import_statsmodels", lambda: (SimpleNamespace(families=StubFamilies), StubFormula()))
    df = pd.DataFrame({"problem": ["p1", "p1"], "step": [1, 2], "correct": [1, 0], "temp_low": [0, 1], "aha": [1, 0]})
    summ, coef = h2.fit_glm_binomial(df, aha_col="aha", cluster_by="none", out_txt=None)
    assert summ["coef_aha"] == 0.3
    assert "temp_low:aha" in list(coef["term"])


def test_fit_glm_stage_interaction_typeerror(monkeypatch):
    class StubResult:
        def __init__(self, params):
            self.params = pd.Series(params)
            self.bse = pd.Series({k: 0.2 for k in params})
            self.pvalues = pd.Series({k: 0.1 for k in params})

        def summary(self):
            return SimpleNamespace(as_text=lambda: "s")

    class StubModel:
        def fit(self, cov_type=None, cov_kwds=None):
            if cov_kwds is not None:
                raise TypeError("boom")
            return StubResult({"Intercept": 1.0, "temp_low": 0.4, "aha": 0.2, "temp_low:aha": 0.1})

    class StubFormula:
        def glm(self, formula, data, family):
            return StubModel()

    class StubFamilies:
        class Binomial:
            pass

    monkeypatch.setattr(h2, "lazy_import_statsmodels", lambda: (SimpleNamespace(families=StubFamilies), StubFormula()))
    df = pd.DataFrame(
        {
            "problem": ["p1", "p1"],
            "step": [1, 2],
            "stage": ["early", "late"],
            "correct": [1, 0],
            "temp_low": [0, 1],
            "aha": [1, 0],
        }
    )
    summ, coef = h2.fit_glm_stage_interaction(df, aha_col="aha", cluster_by="problem", out_txt=None)
    assert summ["coef_aha"] == 0.2
    assert "temp_low:aha" in list(coef["term"])


def test_group_accuracy_helpers():
    df = pd.DataFrame(
        {"correct": [1, 0, 1, 0], "temp_low": [0, 0, 1, 1], "aha": [1, 0, 1, 0], "stage": ["e", "e", "l", "l"]}
    )
    overall, by_step = h2.compute_group_acc(df.assign(step=[1, 1, 2, 2]), aha_col="aha")
    assert set(overall.columns) >= {"accuracy", "n", "k"}
    stage_acc = h2.compute_stage_group_acc(df, aha_col=None)
    assert "accuracy" in stage_acc.columns
    aha_counts = h2.compute_stage_aha_counts(df.assign(aha=1), aha_col="aha")
    assert aha_counts["n_aha"].sum() == len(df)


def test_compute_stage_aha_counts_none():
    df = pd.DataFrame({"stage": ["e"], "temp_low": [0], "aha": [1], "correct": [1]})
    empty = h2.compute_stage_aha_counts(df, aha_col=None)
    assert empty.empty


def test_run_stage_specific_glms(monkeypatch, tmp_path):
    monkeypatch.setattr(
        h2,
        "fit_glm_binomial",
        lambda mode_df, aha_col, cluster_by, out_txt: (
            {"N": len(mode_df)},
            pd.DataFrame({"term": ["Intercept"], "coef": [1.0], "se": [0.1], "z": [10], "p": [0.01]}),
        ),
    )
    mode_df = pd.DataFrame(
        {
            "problem": ["p1", "p1"],
            "step": [1, 2],
            "correct": [1, 0],
            "temp_low": [0, 1],
            "stage": ["early", "late"],
            "aha": [1, 0],
        }
    )
    h2._run_stage_specific_glms(mode_df, aha_col="aha", cluster_by="problem", out_dir=str(tmp_path))
    assert (tmp_path / "h2_stage_specific_glm_coefficients.csv").exists()
    assert (tmp_path / "h2_stage_specific_glm_summary.json").exists()


def test_run_stage_specific_glms_no_stage(tmp_path):
    mode_df = pd.DataFrame({"problem": ["p1"], "step": [1], "correct": [1], "temp_low": [0]})
    h2._run_stage_specific_glms(mode_df, aha_col=None, cluster_by="problem", out_dir=str(tmp_path))
    assert not (tmp_path / "h2_stage_specific_glm_coefficients.csv").exists()


def test_run_stage_specific_glms_skips_empty_stage(monkeypatch, tmp_path):
    calls = {"glm": 0}
    monkeypatch.setattr(
        h2,
        "fit_glm_binomial",
        lambda *_a, **_k: (
            calls.__setitem__("glm", calls["glm"] + 1) or {"N": 0},
            pd.DataFrame({"term": [], "coef": []}),
        ),
    )
    mode_df = pd.DataFrame({"problem": ["p1"], "step": [1], "correct": [1], "temp_low": [0], "stage": [np.nan]})
    h2._run_stage_specific_glms(mode_df, aha_col=None, cluster_by="problem", out_dir=str(tmp_path))
    assert calls["glm"] == 0


def test_evaluate_for_aha_mode_and_run_modes(monkeypatch, tmp_path):
    # Prepare a combined dataframe with minimal columns.
    df = pd.DataFrame(
        {
            "problem": ["p1", "p1"],
            "step": [1, 2],
            "sample_idx": [0, 0],
            "correct": [1, 0],
            "temp_low": [0, 1],
            "aha_words": [1, 0],
            "aha_gpt": [1, 0],
            "aha": [1, 0],
            "run_label": ["high", "low"],
            "stage": ["early", "late"],
        }
    )

    # Stub heavy pieces.
    monkeypatch.setattr(
        h2,
        "fit_glm_binomial",
        lambda *a, **k: ({"N": 2, "formula": "f"}, pd.DataFrame({"term": ["temp_low"], "coef": [0.1]})),
    )
    monkeypatch.setattr(
        h2,
        "fit_glm_stage_interaction",
        lambda *a, **k: ({"N": 2, "formula": "g"}, pd.DataFrame({"term": ["temp_low"], "coef": [0.2]})),
    )
    monkeypatch.setattr(h2, "_run_stage_specific_glms", lambda *a, **k: None)
    monkeypatch.setattr(h2, "_write_group_outputs", lambda **kwargs: kwargs.setdefault("out_dir", None))
    monkeypatch.setattr(h2, "attach_formal_sample_level", lambda samples_df, **cfg: samples_df.assign(aha_formal=1))

    out_dir = tmp_path / "out"
    h2.evaluate_for_aha_mode(
        df,
        "formal",
        str(out_dir),
        cluster_by="problem",
        formal_cfg={"delta1": 0.1, "delta2": 0.1, "min_prior_steps": 1},
    )
    assert (out_dir / "h2_glm_coefficients.csv").exists()
    assert (out_dir / "h2_stage_glm_coefficients.csv").exists()

    # run_modes_for_combined should delegate to evaluate_for_aha_mode for each mode.
    called = []
    monkeypatch.setattr(h2, "evaluate_for_aha_mode", lambda *a, **k: called.append(a[1]))
    monkeypatch.setattr(h2, "load_samples", lambda *a, **k: df)
    monkeypatch.setattr(h2, "restrict_to_overlap", lambda a, b: (a, b, ["problem", "step"]))
    h2._run_modes_for_combined(
        df,
        ["gpt_broad", "none"],
        SimpleNamespace(root_high="rh", root_low="rl", split=None, gate_gpt_by_words=True, cluster_by="problem"),
        str(tmp_path),
        {"delta1": 0.1, "delta2": 0.1, "min_prior_steps": 1},
    )
    assert called == ["gpt_broad", "none"]


def test_evaluate_for_aha_mode_gpt_broad_passes_aha(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "problem": ["p1"],
            "step": [1],
            "sample_idx": [0],
            "correct": [1],
            "temp_low": [0],
            "aha": [1],
            "run_label": ["high"],
            "stage": ["early"],
        }
    )
    captured = {}
    monkeypatch.setattr(
        h2,
        "fit_glm_binomial",
        lambda mode_df, aha_col, cluster_by, out_txt: (
            captured.setdefault("aha_col", aha_col) or {"formula": "f", "N": 1},
            pd.DataFrame({"term": ["temp_low"], "coef": [0.1]}),
        ),
    )
    monkeypatch.setattr(
        h2,
        "fit_glm_stage_interaction",
        lambda *a, **k: ({"formula": "g", "N": 1}, pd.DataFrame({"term": ["temp_low"], "coef": [0.2]})),
    )
    monkeypatch.setattr(h2, "_run_stage_specific_glms", lambda *a, **k: None)
    monkeypatch.setattr(h2, "_write_group_outputs", lambda **kwargs: None)

    h2.evaluate_for_aha_mode(
        df,
        "gpt_broad",
        str(tmp_path / "broad"),
        cluster_by="problem",
        formal_cfg={"delta1": 0.1, "delta2": 0.1, "min_prior_steps": 1},
    )
    assert captured["aha_col"] == "aha"
    assert (tmp_path / "broad" / "h2_glm_coefficients.csv").exists()


def test_write_group_outputs(monkeypatch, tmp_path, capsys):
    df = pd.DataFrame(
        {
            "step": [1, 2],
            "temp_low": [0, 1],
            "correct": [1, 0],
            "aha": [1, 0],
            "stage": ["early", "late"],
        }
    )
    summaries = {
        "baseline": {
            "formula": "f",
            "N": 2,
            "coef_temp_low": 0.1,
            "p_temp_low": 0.05,
            "coef_aha": 0.2,
            "p_aha": 0.04,
            "coef_interaction": 0.3,
            "p_interaction": 0.06,
        },
        "stage_interaction": {"formula": "g", "N": 2},
    }
    h2._write_group_outputs(df, aha_col="aha", aha_mode_name="gpt", out_dir=str(tmp_path), glm_summaries=summaries)
    assert (tmp_path / "h2_group_accuracy.csv").exists()
    assert (tmp_path / "h2_stage_aha_counts.csv").exists()
    out = capsys.readouterr().out
    assert "Baseline GLM" in out and "temp_low:aha" in out


def test_evaluate_for_aha_mode_words_and_none(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "problem": ["p1", "p1"],
            "step": [1, 2],
            "sample_idx": [0, 0],
            "correct": [1, 0],
            "temp_low": [0, 1],
            "aha_words": [1, 0],
            "aha_gpt": [1, 0],
            "aha": [1, 0],
            "run_label": ["high", "low"],
            "stage": ["early", "late"],
        }
    )
    monkeypatch.setattr(
        h2,
        "fit_glm_binomial",
        lambda *a, **k: (
            {"formula": "f", "N": 2, "coef_temp_low": 0.1, "p_temp_low": 0.05},
            pd.DataFrame({"term": ["temp_low"], "coef": [0.1]}),
        ),
    )
    monkeypatch.setattr(
        h2,
        "fit_glm_stage_interaction",
        lambda *a, **k: ({"formula": "g", "N": 2}, pd.DataFrame({"term": ["temp_low"], "coef": [0.2]})),
    )
    monkeypatch.setattr(h2, "_run_stage_specific_glms", lambda *a, **k: None)

    h2.evaluate_for_aha_mode(
        df,
        "words",
        str(tmp_path / "words"),
        cluster_by="problem",
        formal_cfg={"delta1": 0.1, "delta2": 0.1, "min_prior_steps": 1},
    )
    assert (tmp_path / "words" / "h2_glm_coefficients.csv").exists()

    h2.evaluate_for_aha_mode(
        df,
        "none",
        str(tmp_path / "none"),
        cluster_by="problem",
        formal_cfg={"delta1": 0.1, "delta2": 0.1, "min_prior_steps": 1},
    )
    assert (tmp_path / "none" / "h2_stage_glm_coefficients.csv").exists()


def test_resolve_aha_modes():
    assert h2._resolve_aha_modes(SimpleNamespace(aha="all")) == ["words", "gpt", "formal"]
    assert h2._resolve_aha_modes(SimpleNamespace(aha="gpt_broad")) == ["gpt_broad"]
    assert h2._resolve_aha_modes(SimpleNamespace(aha="none")) == ["none"]
    assert h2._resolve_aha_modes(SimpleNamespace(aha="gpt")) == ["gpt"]


def test_build_arg_parser_defaults():
    parser = h2.build_arg_parser()
    args = parser.parse_args(["rh", "rl"])
    assert args.root_high == "rh" and args.root_low == "rl"
    assert args.aha == "gpt" and args.stage_mode == "quantile"


def test_load_and_overlap_runs(monkeypatch):
    def fake_load(root, split, aha_mode="gpt", gate_gpt_by_words=True):
        return pd.DataFrame(
            {
                "dataset": ["d"],
                "model": ["m"],
                "split": ["s"],
                "problem": ["p"],
                "step": [1],
                "sample_idx": [0],
                "correct": [1],
                "aha_words": [1],
                "aha_gpt": [1],
                "aha": [1],
            }
        )

    monkeypatch.setattr(h2, "load_samples", fake_load)
    args = SimpleNamespace(root_high="rh", root_low="rl", split=None, gate_gpt_by_words=False)
    combined, high_overlap, keys = h2._load_and_overlap_runs(args)
    assert len(combined) == 2 and set(keys) == {"problem", "step", "sample_idx"}
    assert set(combined["run_label"]) == {"high", "low"}


def test_load_and_overlap_runs_raises(monkeypatch):
    monkeypatch.setattr(h2, "load_samples", lambda *a, **k: pd.DataFrame())
    args = SimpleNamespace(root_high="rh", root_low="rl", split=None, gate_gpt_by_words=False)
    with pytest.raises(SystemExit):
        h2._load_and_overlap_runs(args)


def test_assign_stage_labels_wrapper():
    df = pd.DataFrame({"step": [1, 2, 3]})
    args_fixed = SimpleNamespace(stage_mode="fixed", stage_bounds=(1, 2), stage_quantiles=(0.3, 0.7))
    stage_df, info = h2._assign_stage_labels(df, args_fixed)
    assert info["mode"] == "fixed" and "stage" in stage_df.columns

    args_quant = SimpleNamespace(stage_mode="quantile", stage_bounds=None, stage_quantiles=(0.2, 0.8))
    stage_df2, info2 = h2._assign_stage_labels(df, args_quant)
    assert info2["mode"] == "quantile" and "stage" in stage_df2.columns


def test_main_flow_stubbed(monkeypatch, tmp_path, capsys):
    args = SimpleNamespace(
        root_high="rh",
        root_low="rl",
        split=None,
        out_dir=str(tmp_path / "out"),
        aha="gpt",
        gate_gpt_by_words=False,
        formal_delta1=0.1,
        formal_delta2=0.2,
        formal_min_prior_steps=1,
        stage_mode="fixed",
        stage_bounds=(1, 2),
        stage_quantiles=(0.2, 0.8),
        cluster_by="problem",
    )
    parser = SimpleNamespace(parse_args=lambda: args)
    monkeypatch.setattr(h2, "build_arg_parser", lambda: parser)

    combined_df = pd.DataFrame(
        {
            "problem": ["p"],
            "step": [1],
            "sample_idx": [0],
            "correct": [1],
            "aha_words": [1],
            "aha_gpt": [1],
            "aha": [1],
            "temp_low": [0],
            "run_label": ["high"],
        }
    )
    high_overlap = combined_df.copy()
    monkeypatch.setattr(h2, "_load_and_overlap_runs", lambda a: (combined_df, high_overlap, ["problem", "step"]))
    monkeypatch.setattr(
        h2, "_assign_stage_labels", lambda df, a: (df.assign(stage="early"), {"mode": "fixed", "bounds": [1, 2]})
    )
    called = {}
    monkeypatch.setattr(h2, "_resolve_aha_modes", lambda a: ["gpt"])
    monkeypatch.setattr(h2, "_run_modes_for_combined", lambda **kwargs: called.setdefault("ran", True))
    h2.main()
    assert called.get("ran") is True
    assert (tmp_path / "out" / "h2_stage_info.json").exists()
    out = capsys.readouterr().out
    assert "Stage cuts (fixed)" in out


def test_main_quantile_output(monkeypatch, tmp_path, capsys):
    args = SimpleNamespace(
        root_high="rh",
        root_low="rl",
        split=None,
        out_dir=str(tmp_path / "out"),
        aha="gpt",
        gate_gpt_by_words=False,
        formal_delta1=0.1,
        formal_delta2=0.2,
        formal_min_prior_steps=1,
        stage_mode="quantile",
        stage_bounds=None,
        stage_quantiles=(0.2, 0.8),
        cluster_by="problem",
    )
    parser = SimpleNamespace(parse_args=lambda: args)
    monkeypatch.setattr(h2, "build_arg_parser", lambda: parser)
    combined_df = pd.DataFrame(
        {
            "problem": ["p"],
            "step": [1],
            "sample_idx": [0],
            "correct": [1],
            "aha_words": [1],
            "aha_gpt": [1],
            "aha": [1],
            "temp_low": [0],
            "run_label": ["high"],
        }
    )
    high_overlap = combined_df.copy()
    monkeypatch.setattr(h2, "_load_and_overlap_runs", lambda a: (combined_df, high_overlap, ["problem", "step"]))
    monkeypatch.setattr(
        h2,
        "_assign_stage_labels",
        lambda df, a: (
            df.assign(stage="early"),
            {"mode": "quantile", "cutpoints": [0.3, 0.7], "quantiles": list(a.stage_quantiles)},
        ),
    )
    monkeypatch.setattr(h2, "_resolve_aha_modes", lambda a: [])
    monkeypatch.setattr(h2, "_run_modes_for_combined", lambda **kwargs: None)
    h2.main()
    out = capsys.readouterr().out
    assert "quantiles" in out


def test_main_guard_hit():
    called = []

    def stub_main():
        called.append("hit")

    guard_snippet = "\n" * 990 + "if __name__ == '__main__':\n    main()\n"
    exec(compile(guard_snippet, h2.__file__, "exec"), {"__name__": "__main__", "main": stub_main})
    assert called == ["hit"]
