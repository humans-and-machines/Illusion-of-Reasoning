import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.analysis.figure_2_regression as reg


class _AxisStub:
    def __init__(self):
        self.calls = []
        self.visible = True

    def plot(self, *args, **kwargs):
        self.calls.append(("plot", args, kwargs))
        return None

    def fill_between(self, *args, **kwargs):
        self.calls.append(("fill_between", args, kwargs))
        return None

    def set_xticks(self, *args, **kwargs):
        self.calls.append(("xticks", args, kwargs))

    def set_xticklabels(self, *args, **kwargs):
        self.calls.append(("xticklabels", args, kwargs))

    def set_xlabel(self, *args, **kwargs):
        self.calls.append(("xlabel", args, kwargs))

    def set_title(self, *args, **kwargs):
        self.calls.append(("title", args, kwargs))

    def grid(self, *args, **kwargs):
        self.calls.append(("grid", args, kwargs))

    def set_ylabel(self, *args, **kwargs):
        self.calls.append(("ylabel", args, kwargs))

    def set_visible(self, flag):
        self.visible = flag


def test_fit_glm_interaction_uses_regularized_on_failure(monkeypatch):
    _FakeFamilies = types.SimpleNamespace(families=types.SimpleNamespace(Binomial=lambda: None))

    class _FakeRes:
        params = np.array([1.0])

    class _FakeModel:
        def fit(self, cov_type=None):
            raise ValueError("fail fit")

        def fit_regularized(self, alpha, L1_wt):
            return _FakeRes()

    class _FakeFormula:
        def glm(self, *args, **kwargs):
            return _FakeModel()

    monkeypatch.setattr(reg, "lazy_import_statsmodels", lambda: (_FakeFamilies, _FakeFormula()))

    res, model, used_cov = reg.fit_glm_interaction(pd.DataFrame({"problem": [], "step": []}), "aha", "bucket")
    assert isinstance(res, _FakeRes)
    assert used_cov == "ridge"
    assert isinstance(model, _FakeModel)


def test_fit_glm_interaction_hc1_when_stable(monkeypatch):
    _FakeFamilies = types.SimpleNamespace(families=types.SimpleNamespace(Binomial=lambda: None))

    class _FakeRes:
        params = np.array([1.0, 0.0])

    class _FakeModel:
        def fit(self, cov_type=None):
            assert cov_type == "HC1"
            return _FakeRes()

    class _FakeFormula:
        def glm(self, *args, **kwargs):
            return _FakeModel()

    monkeypatch.setattr(reg, "lazy_import_statsmodels", lambda: (_FakeFamilies, _FakeFormula()))
    res, _, used_cov = reg.fit_glm_interaction(pd.DataFrame({"problem": [], "step": []}), "aha", "bucket")
    assert isinstance(res, _FakeRes)
    assert used_cov == "none"


def test_fit_glm_interaction_switches_on_nan_params(monkeypatch):
    _FakeFamilies = types.SimpleNamespace(families=types.SimpleNamespace(Binomial=lambda: None))

    class _FirstRes:
        params = np.array([np.nan])

    class _SecondRes:
        params = np.array([0.5])

    class _FakeModel:
        def __init__(self):
            self.regularized_called = False

        def fit(self, cov_type=None):
            # Returns NaN params to trigger RuntimeError fallback.
            return _FirstRes()

        def fit_regularized(self, alpha, L1_wt):
            self.regularized_called = True
            self.alpha = alpha
            self.L1_wt = L1_wt
            return _SecondRes()

    fake_model = _FakeModel()

    class _FakeFormula:
        def glm(self, *args, **kwargs):
            return fake_model

    monkeypatch.setattr(reg, "lazy_import_statsmodels", lambda: (_FakeFamilies, _FakeFormula()))
    res, model, used_cov = reg.fit_glm_interaction(
        pd.DataFrame({"problem": [1], "step": [1], "aha": [1], "bucket": ["x"]}),
        "aha",
        "bucket",
        ridge_alpha=0.7,
    )
    assert model is fake_model
    assert isinstance(res, _SecondRes)
    assert used_cov == "ridge"
    assert fake_model.regularized_called
    assert fake_model.alpha == 0.7
    assert fake_model.L1_wt == 0.0


def test_bootstrap_bucket_means_returns_nan_when_no_bootstrap(monkeypatch):
    ctx = reg.BucketRegressionContext(
        res=None,
        model=None,
        base_df=pd.DataFrame({"x": [1, 2]}),
        aha_column="aha",
        bucket_column="bucket",
        bucket_levels=["b1"],
    )
    monkeypatch.setattr(reg, "predict_formula", lambda *a, **k: np.array([0.5, 0.5]))
    mean, lo, hi = reg._bootstrap_bucket_means(
        context=ctx,
        level_value="b1",
        aha_value=1,
        num_bootstrap_samples=0,
        rng=np.random.default_rng(0),
    )
    assert mean == pytest.approx(0.5)
    assert np.isnan(lo) and np.isnan(hi)


def test_bootstrap_bucket_means_bootstraps(monkeypatch):
    base_df = pd.DataFrame({"value": [0.0, 1.0, 2.0]})
    ctx = reg.BucketRegressionContext(
        res=None,
        model=None,
        base_df=base_df,
        aha_column="aha",
        bucket_column="bucket",
        bucket_levels=["b"],
    )
    monkeypatch.setattr(reg, "predict_formula", lambda *args: args[2]["value"].to_numpy())
    rng_seed = 7
    rng = np.random.default_rng(rng_seed)
    rng_expected = np.random.default_rng(rng_seed)
    expected_bootstrap = []
    for _ in range(3):
        sampled = rng_expected.choice(len(base_df), size=len(base_df), replace=True)
        expected_bootstrap.append(np.mean(base_df["value"].to_numpy()[sampled]))
    mean, lo, hi = reg._bootstrap_bucket_means(
        context=ctx,
        level_value="b",
        aha_value=0,
        num_bootstrap_samples=3,
        rng=rng,
    )
    assert mean == pytest.approx(np.mean(base_df["value"]))
    assert lo == pytest.approx(np.nanpercentile(expected_bootstrap, 2.5))
    assert hi == pytest.approx(np.nanpercentile(expected_bootstrap, 97.5))


def test_predict_margins_by_bucket_collects_arrays(monkeypatch):
    calls = []

    def fake_bootstrap(context, level_value, aha_value, num_bootstrap_samples, rng):
        calls.append((level_value, aha_value, num_bootstrap_samples, isinstance(rng, np.random.Generator)))
        return level_value + aha_value, level_value - 0.1, level_value + 0.1

    ctx = reg.BucketRegressionContext(
        res=None,
        model=None,
        base_df=pd.DataFrame(),
        aha_column="aha",
        bucket_column="bucket",
        bucket_levels=[0, 1],
    )
    monkeypatch.setattr(reg, "_bootstrap_bucket_means", fake_bootstrap)
    means, lows, highs = reg.predict_margins_by_bucket(ctx, aha_value=2, num_bootstrap_samples=5, rng_seed=11)
    assert calls == [
        (0, 2, 5, True),
        (1, 2, 5, True),
    ]
    assert means.tolist() == [2, 3]
    assert lows.tolist() == [-0.1, 0.9]
    assert highs.tolist() == [0.1, 1.1]


def test_predict_curves_for_context_delegates(monkeypatch):
    def fake_predict(context, aha_value, num_bootstrap_samples, rng_seed):
        return (
            np.array([aha_value], dtype=float),
            np.array([rng_seed], dtype=float),
            np.array([num_bootstrap_samples], dtype=float),
        )

    monkeypatch.setattr(reg, "predict_margins_by_bucket", fake_predict)
    ctx = reg.BucketRegressionContext(
        res=None,
        model=None,
        base_df=pd.DataFrame(),
        aha_column="aha",
        bucket_column="bucket",
        bucket_levels=[0],
    )
    curves = reg._predict_curves_for_context(ctx, num_bootstrap_samples=7)
    assert curves["pred_noaha"][0] == 0
    assert curves["low_noaha"][0] == 0
    assert curves["high_noaha"][0] == 7
    assert curves["pred_aha"][0] == 1
    assert curves["low_aha"][0] == 1
    assert curves["high_aha"][0] == 7


def test_plot_variant_axis_skips_when_column_missing():
    frame = pd.DataFrame({"other": [1]})
    axis = _AxisStub()
    rows = []
    ctx = reg.VariantPlotContext(
        prepared_frame=frame,
        bucket_levels=[1, 2],
        bucket_labels=["1", "2"],
        config=reg.RegressionPlotConfig(
            frame=frame,
            ppx_bucket_column="bucket",
            dataset="D",
            model_name="M",
            num_bootstrap_samples=0,
            output=reg.RegressionOutputConfig(
                out_png="out.png",
                out_pdf="out.pdf",
                title_suffix="t",
                a4_pdf=False,
                a4_orientation="landscape",
            ),
        ),
        rows_for_csv=rows,
    )
    reg._plot_variant_axis(axis, aha_column="missing", title="t", plot_context=ctx)
    assert axis.visible is False
    assert rows == []


def test_plot_variant_axis_draws_and_extends_rows(monkeypatch):
    # Prepare a tiny frame with the needed aha column.
    frame = pd.DataFrame({"bucket": ["a", "b"], "aha_words": [0, 1]})
    rows = []
    axis = _AxisStub()
    ctx = reg.VariantPlotContext(
        prepared_frame=frame,
        bucket_levels=["a", "b"],
        bucket_labels=["A", "B"],
        config=reg.RegressionPlotConfig(
            frame=frame,
            ppx_bucket_column="bucket",
            dataset="DS",
            model_name="Model",
            num_bootstrap_samples=0,
            output=reg.RegressionOutputConfig(
                out_png="out.png",
                out_pdf="out.pdf",
                title_suffix="t",
                a4_pdf=False,
                a4_orientation="landscape",
            ),
        ),
        rows_for_csv=rows,
    )

    monkeypatch.setattr(reg, "_build_regression_context", lambda *a, **k: None)
    monkeypatch.setattr(
        reg,
        "_predict_curves_for_context",
        lambda *a, **k: {
            "pred_noaha": np.array([0.1, 0.2]),
            "low_noaha": np.array([0.0, 0.1]),
            "high_noaha": np.array([0.2, 0.3]),
            "pred_aha": np.array([0.3, 0.4]),
            "low_aha": np.array([0.2, 0.3]),
            "high_aha": np.array([0.4, 0.5]),
        },
    )

    reg._plot_variant_axis(axis, "aha_words", "Words", ctx)
    # Two bucket levels -> four rows (aha=0/1)
    assert len(rows) == 4
    assert any(call[0] == "plot" for call in axis.calls)
    assert any(call[0] == "fill_between" for call in axis.calls)


def test_build_regression_context_coerces_ints(monkeypatch):
    recorded = {}

    def fake_fit(df, aha_column, bucket_column, ridge_alpha):
        recorded["df"] = df.copy()
        recorded["aha_column"] = aha_column
        recorded["bucket_column"] = bucket_column
        recorded["ridge_alpha"] = ridge_alpha
        return "res", "model", "cov"

    monkeypatch.setattr(reg, "fit_glm_interaction", fake_fit)
    base_df = pd.DataFrame({"aha": ["0", "1"], "bucket": ["a", "b"]})
    ctx = reg._build_regression_context(base_df, "aha", "bucket", ["a", "b"])

    assert recorded["aha_column"] == "aha"
    assert recorded["bucket_column"] == "bucket"
    assert recorded["ridge_alpha"] == 0.5
    assert ctx.res == "res"
    assert ctx.model == "model"
    assert ctx.bucket_levels == ["a", "b"]
    assert ctx.base_df["aha"].dtype.kind in "iu"
    assert list(ctx.base_df["aha"]) == [0, 1]
    # Ensure we did not mutate the original.
    assert base_df["aha"].dtype == object


def test_plot_regression_curves_writes_csv(monkeypatch, tmp_path):
    frame = pd.DataFrame({"bucket": ["x", "y"], "aha_words": [0, 1]})
    out_png = tmp_path / "plot.png"
    out_pdf = tmp_path / "plot.pdf"

    config = reg.RegressionPlotConfig(
        frame=frame,
        ppx_bucket_column="bucket",
        dataset="DS",
        model_name="Model",
        num_bootstrap_samples=0,
        output=reg.RegressionOutputConfig(
            out_png=str(out_png),
            out_pdf=str(out_pdf),
            title_suffix="t",
            a4_pdf=False,
            a4_orientation="portrait",
        ),
    )

    # Stub plotting pieces.
    axes = [_AxisStub(), _AxisStub(), _AxisStub()]
    fig_stub = types.SimpleNamespace(
        legend=lambda *a, **k: None,
        tight_layout=lambda *a, **k: None,
    )
    monkeypatch.setattr(reg.plt, "subplots", lambda *a, **k: (fig_stub, axes))
    monkeypatch.setattr(reg, "add_lower_center_legend", lambda *a, **k: None)
    monkeypatch.setattr(reg, "save_figure_outputs", lambda *a, **k: None)
    monkeypatch.setattr(
        reg,
        "_plot_variant_axis",
        lambda axis, aha_column, title, plot_context: plot_context.rows_for_csv.append(
            {"variant": title, "bucket": "b", "aha": 0, "mean": 0.1, "lo": 0.0, "hi": 0.2}
        ),
    )

    out_csv = reg.plot_regression_curves(config)
    assert Path(out_csv).exists()
    df = pd.read_csv(out_csv)
    assert not df.empty
    assert set(df.columns) == {"variant", "bucket", "aha", "mean", "lo", "hi"}
