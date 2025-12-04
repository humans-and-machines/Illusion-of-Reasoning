import runpy
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import src.analysis.h2_analysis as h2


def test_configure_matplotlib_sets_agg_backend(monkeypatch):
    used = {}

    class FakeMatplotlib:
        def use(self, backend):
            used["backend"] = backend

    monkeypatch.setattr(h2.importlib, "import_module", lambda name: FakeMatplotlib() if name == "matplotlib" else None)
    h2._configure_matplotlib()
    assert used["backend"] == "Agg"


def test_subset_by_step_filters_range():
    df = pd.DataFrame({"step": [0, 1, 2, 3]})
    filtered = h2._subset_by_step(df, min_step=1, max_step=2)
    assert filtered["step"].tolist() == [1, 2]


def test_diagnostic_plots_invokes_plotting(monkeypatch):
    calls = {"diag": 0, "lines": 0, "ame": 0}
    real_import = h2.importlib.import_module

    def fake_import(name):
        if name == "src.analysis.h2_plotting":
            return SimpleNamespace(
                plot_diag_panel=lambda *_a, **_k: calls.__setitem__("diag", calls["diag"] + 1),
                lineplot=lambda *a, **k: calls.__setitem__("lines", calls["lines"] + 1),
                plot_ame_with_ci=lambda *_a, **_k: calls.__setitem__("ame", calls["ame"] + 1),
            )
        return real_import(name)

    monkeypatch.setattr(h2.importlib, "import_module", fake_import)
    pass1_df = pd.DataFrame({"step": [1], "aha": [0]})
    reg_df = pd.DataFrame({"step": [1], "aha_coef": [0.1], "aha_ame": [0.2], "unc_coef": [0.3], "naive_delta": [0.4]})
    h2._diagnostic_plots(pass1_df, reg_df, out_dir=".")
    assert calls["diag"] == 1 and calls["lines"] == 4 and calls["ame"] == 1


def test_diagnostic_plots_returns_early_on_empty_reg_df(monkeypatch):
    calls = {"diag": 0, "line": 0, "ame": 0}

    def fake_import(name):
        assert name == "src.analysis.h2_plotting"
        return SimpleNamespace(
            plot_diag_panel=lambda *_a, **_k: calls.__setitem__("diag", calls["diag"] + 1),
            lineplot=lambda *_a, **_k: calls.__setitem__("line", calls["line"] + 1),
            plot_ame_with_ci=lambda *_a, **_k: calls.__setitem__("ame", calls["ame"] + 1),
        )

    monkeypatch.setattr(h2.importlib, "import_module", fake_import)
    pass1_df = pd.DataFrame({"step": [1]})
    empty_reg_df = pd.DataFrame()
    h2._diagnostic_plots(pass1_df, empty_reg_df, out_dir=".")
    assert calls == {"diag": 1, "line": 0, "ame": 0}


def test_plot_pooled_effects_uses_stub_matplotlib(monkeypatch, tmp_path):
    # Provide fake matplotlib to avoid real backend use
    class FakeAxis:
        def __init__(self):
            self.plotted = False

        def plot(self, *a, **k):
            self.plotted = True

        def set_xlabel(self, *_): ...
        def set_ylabel(self, *_): ...
        def set_title(self, *_): ...
        def grid(self, *_, **__): ...

    class FakeFig:
        def __init__(self, owner):
            self.saved = None
            self.axis = FakeAxis()
            self.owner = owner

        def tight_layout(self): ...
        def savefig(self, path):
            self.saved = path
            self.owner.saved_path = path

    class FakePlt:
        def __init__(self):
            self.saved_path = None

        def subplots(self, *a, **k):
            fig = FakeFig(self)
            return fig, fig.axis

        def close(self, *_): ...

    import types

    fake_plt = FakePlt()
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_matplotlib.use = lambda *_: None
    fake_matplotlib.__path__ = []  # indicate package for submodule import
    plt_module = types.ModuleType("matplotlib.pyplot")
    plt_module.subplots = fake_plt.subplots
    plt_module.close = fake_plt.close
    sys.modules["matplotlib"] = fake_matplotlib
    sys.modules["matplotlib.pyplot"] = plt_module
    monkeypatch.setattr(h2, "_configure_matplotlib", lambda: None)
    pooled_df = pd.DataFrame({"step": [1, 2], "aha_effect": [0.1, 0.2]})

    h2._plot_pooled_effects(pooled_df, out_dir=str(tmp_path))
    assert fake_plt.saved_path is not None


def test_plot_pooled_effects_noop_when_empty(monkeypatch, tmp_path):
    called = {"configure": 0, "csv": 0}
    empty_df = pd.DataFrame()
    monkeypatch.setattr(h2, "_configure_matplotlib", lambda: called.__setitem__("configure", called["configure"] + 1))
    monkeypatch.setattr(empty_df, "to_csv", lambda *_a, **_k: called.__setitem__("csv", called["csv"] + 1))
    h2._plot_pooled_effects(empty_df, out_dir=str(tmp_path))
    assert called == {"configure": 0, "csv": 0}


def test_run_uncertainty_figures_prints_paths(monkeypatch, capsys):
    def fake_import(name):
        assert name == "src.analysis.core.h2_uncertainty_helpers"
        return SimpleNamespace(
            make_all3_uncertainty_buckets_figure=lambda **_k: (
                "buckets.png",
                "buckets.csv",
                {"d": "all"},
                {"ps": "formal"},
            ),
            plot_uncertainty_hist_100bins=lambda **_k: ("hist.png", "hist.csv"),
        )

    monkeypatch.setattr(h2.importlib, "import_module", fake_import)
    h2._run_uncertainty_figures(files=["one"], out_dir="out", args=SimpleNamespace())
    captured = capsys.readouterr().out
    assert "Buckets figure: buckets.png" in captured
    assert "Histogram (100 bins): hist.png" in captured


def test_summarize_outputs_lists_expected_files(capsys, tmp_path):
    out_dir = tmp_path / "h2_out"
    out_dir.mkdir()
    samples_csv = out_dir / "samples.csv"
    h2._summarize_outputs(str(out_dir), str(samples_csv))
    out = capsys.readouterr().out
    assert "Wrote samples CSV" in out
    assert "h2_uncertainty_hist_100bins.csv" in out


def test_run_pipeline_happy_path(monkeypatch, tmp_path):
    calls = {"diag": 0, "unc_figs": 0, "pooled": 0}
    monkeypatch.setattr(h2, "scan_jsonl_files", lambda root, split: ["dummy.jsonl"])
    monkeypatch.setattr(h2, "_configure_matplotlib", lambda: None)
    pass1_df = pd.DataFrame({"step": [1, 2], "uncertainty": [0.5, 0.6], "aha": [0, 1]})
    monkeypatch.setattr(h2, "load_pass1_rows", lambda files, unc, src: pass1_df.copy())
    monkeypatch.setattr(
        h2,
        "standardize_uncertainty_with_stats",
        lambda df: (df.assign(uncertainty_std=0.0), None, None),
    )
    reg_df = pd.DataFrame({"step": [1], "aha_coef": [0.1], "aha_ame": [0.2], "unc_coef": [0.3], "naive_delta": [0.4]})
    monkeypatch.setattr(h2, "fit_stepwise_glms", lambda df, cfg: reg_df)
    monkeypatch.setattr(h2, "_diagnostic_plots", lambda *a, **k: calls.__setitem__("diag", calls["diag"] + 1))
    monkeypatch.setattr(
        h2, "_run_uncertainty_figures", lambda *a, **k: calls.__setitem__("unc_figs", calls["unc_figs"] + 1)
    )
    monkeypatch.setattr(
        h2,
        "compute_pooled_step_effects",
        lambda df, ridge_l2: pd.DataFrame({"step": [1], "aha_effect": [0.1]}),
    )
    monkeypatch.setattr(h2, "_plot_pooled_effects", lambda *a, **k: calls.__setitem__("pooled", calls["pooled"] + 1))
    monkeypatch.setattr(h2, "_summarize_outputs", lambda *a, **k: None)

    parser = h2.build_arg_parser()
    args = parser.parse_args([str(tmp_path)])
    args.out_dir = str(tmp_path / "out")

    h2.run_pipeline(args)

    assert calls["diag"] == 1 and calls["unc_figs"] == 1 and calls["pooled"] == 1
    samples_csv = tmp_path / "out" / "h2_pass1_samples.csv"
    assert samples_csv.exists()


def test_run_pipeline_raises_when_no_files(monkeypatch, tmp_path):
    monkeypatch.setattr(h2, "_configure_matplotlib", lambda: None)
    monkeypatch.setattr(h2, "scan_jsonl_files", lambda *_a, **_k: [])
    parser = h2.build_arg_parser()
    args = parser.parse_args([str(tmp_path)])
    with pytest.raises(SystemExit):
        h2.run_pipeline(args)


def test_run_pipeline_raises_when_no_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(h2, "_configure_matplotlib", lambda: None)
    monkeypatch.setattr(h2, "scan_jsonl_files", lambda *_a, **_k: ["one"])
    monkeypatch.setattr(h2, "load_pass1_rows", lambda *_a, **_k: pd.DataFrame())
    parser = h2.build_arg_parser()
    args = parser.parse_args([str(tmp_path)])
    with pytest.raises(SystemExit):
        h2.run_pipeline(args)


def test_run_pipeline_writes_native_compare(monkeypatch, tmp_path):
    monkeypatch.setattr(h2, "_configure_matplotlib", lambda: None)
    monkeypatch.setattr(h2, "scan_jsonl_files", lambda *_a, **_k: ["one"])
    calls = {"load_sources": []}

    def fake_load(files, unc_field, source):
        calls["load_sources"].append(source)
        return pd.DataFrame({"step": [1], "uncertainty": [0.1], "aha": [1]})

    monkeypatch.setattr(h2, "load_pass1_rows", fake_load)
    monkeypatch.setattr(h2, "standardize_uncertainty_with_stats", lambda df: (df, None, None))
    monkeypatch.setattr(
        h2,
        "fit_stepwise_glms",
        lambda *_a, **_k: pd.DataFrame(
            {"step": [1], "aha_coef": [0.1], "aha_ame": [0.2], "unc_coef": [0.3], "naive_delta": [0.4]}
        ),
    )
    monkeypatch.setattr(h2, "_diagnostic_plots", lambda *_a, **_k: None)
    monkeypatch.setattr(h2, "_run_uncertainty_figures", lambda *_a, **_k: None)
    monkeypatch.setattr(
        h2, "compute_pooled_step_effects", lambda *_a, **_k: pd.DataFrame({"step": [1], "aha_effect": [0.5]})
    )
    monkeypatch.setattr(h2, "_plot_pooled_effects", lambda *_a, **_k: None)
    monkeypatch.setattr(h2, "_summarize_outputs", lambda *_a, **_k: None)

    parser = h2.build_arg_parser()
    args = parser.parse_args([str(tmp_path), "--compare_native"])
    args.out_dir = str(tmp_path / "out")

    h2.run_pipeline(args)

    native_csv = tmp_path / "out" / "h2_pass1_samples_native.csv"
    assert native_csv.exists()
    assert calls["load_sources"] == ["gpt", "native"]


def test_main_invokes_run_pipeline(monkeypatch):
    args = SimpleNamespace(mock="arg")
    parser = SimpleNamespace(parse_args=lambda: args)
    called = {}
    monkeypatch.setattr(h2, "build_arg_parser", lambda: parser)
    monkeypatch.setattr(h2, "run_pipeline", lambda parsed: called.setdefault("args", parsed))
    h2.main()
    assert called["args"] is args


def test_module_entrypoint_runs_with_help(monkeypatch):
    monkeypatch.delitem(sys.modules, "src.analysis.h2_analysis", raising=False)
    monkeypatch.setattr(sys, "argv", ["h2_analysis.py", "--help"])
    with pytest.raises(SystemExit):
        runpy.run_module("src.analysis.h2_analysis", run_name="__main__")
