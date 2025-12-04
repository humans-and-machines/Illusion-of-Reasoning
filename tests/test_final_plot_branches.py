import argparse
import builtins
import importlib
import importlib.util
import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest

import src.analysis.final_plot as fp


def test_optional_imports_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        target_list = fromlist or ()
        if name == "src.analysis" and "rq3_analysis" in target_list:
            raise ImportError("rq3 missing")
        if name == "matplotlib.pyplot":
            raise ImportError("plt missing")
        if name == "src.analysis.rq3_analysis":
            raise ImportError("rq3 missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delattr(sys.modules.get("src.analysis"), "rq3_analysis", raising=False)
    monkeypatch.delitem(sys.modules, "src.analysis.rq3_analysis", raising=False)
    monkeypatch.delitem(sys.modules, "matplotlib.pyplot", raising=False)
    spec = importlib.util.spec_from_file_location("temp_final_plot_missing", fp.__file__)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module._rq3_analysis_module is None
    assert module.plt is None


def test_build_row_filters_invalid_pass1_and_max_step():
    cfg = fp.CarparkEvalConfig("ge", 0.0, min_step=None, max_step=1)
    assert fp._build_row_for_record("p", {"step": 0, "pass1": "not-a-dict"}, cfg) is None

    cfg_max = fp.CarparkEvalConfig("ge", 0.0, min_step=None, max_step=3)
    rec = {"step": 5, "pass1": {"entropy": 0.2}}
    assert fp._build_row_for_record("p", rec, cfg_max) is None


def test_build_row_uses_carpark_fallbacks(monkeypatch):
    monkeypatch.setattr(fp, "extract_correct", lambda data, rec: None)

    calls = []

    def fake_carpark(rec, data, op, thr):
        calls.append(data.get("tag"))
        return 1 if data.get("tag") == "p1" else 0

    monkeypatch.setattr(fp, "carpark_success_from_soft_reward", fake_carpark)
    rec = {
        "step": 1,
        "example_id": "ex",
        "pass1": {"entropy": 0.4, "shift_in_reasoning_v1": False, "tag": "p1"},
        "pass2": {"has_reconsider_cue": True, "reconsider_markers": ["injected_cue"], "tag": "p2"},
    }
    cfg = fp.CarparkEvalConfig("ge", 0.0, min_step=None, max_step=None)

    row = fp._build_row_for_record("p", rec, cfg)
    assert row is not None
    assert row["p1_correct"] == 1
    assert row["p2_correct"] == 0
    assert calls == ["p1", "p2"]


def test_summarize_for_figure_includes_empty_bin(monkeypatch):
    df = pd.DataFrame(
        {
            "group_id": ["g1"],
            "entropy": [0.1],
            "p1_correct": [1],
            "p2_triggered": [1],
            "p2_correct": [1],
            "p1_shift": [1],
        },
    )

    def fake_apply(self, func, *args, **kwargs):
        empty_result = func(pd.DataFrame())
        category = getattr(self.obj["_bin"].dtype, "categories", [pd.Interval(0, 1)])[0]
        empty_result.name = (category, "baseline 0/8")
        frame = empty_result.to_frame().T
        frame.index = pd.MultiIndex.from_tuples([empty_result.name], names=["_bin", "_stratum"])
        return frame

    monkeypatch.setattr(pd.core.groupby.generic.DataFrameGroupBy, "apply", fake_apply)
    _, bot = fp.summarize_for_figure(df, bins=[0, 1])
    assert int(bot.iloc[0]["N"]) == 0
    assert pd.isna(bot.iloc[0]["rate"])


def test_plot_figure_requires_matplotlib(monkeypatch, tmp_path):
    monkeypatch.setattr(fp, "plt", None)
    monkeypatch.setattr(fp, "_PLT_IMPORT_ERROR", ImportError("missing"))
    intervals = pd.IntervalIndex.from_breaks([0, 1])
    df_top = pd.DataFrame({"_bin": intervals, "N": [1], "shift_share": [0.5]})
    df_bot = pd.DataFrame(
        {
            "_bin": intervals,
            "_stratum": ["baseline ≥1/8"],
            "N": [1],
            "rate": [0.5],
            "se": [0.1],
            "lo": [0.4],
            "hi": [0.6],
        },
    )
    cfg = fp.FigureOutputConfig(
        out_png=str(tmp_path / "p.png"),
        out_pdf=str(tmp_path / "p.pdf"),
        title_suffix="t",
        dpi=100,
    )
    with pytest.raises(SystemExit):
        fp.plot_figure(df_top, df_bot, [0, 1], cfg)


def test_maybe_run_rq3_early_return(monkeypatch):
    class Boom:
        def main(self):
            raise AssertionError("should not run")

    monkeypatch.setattr(fp, "_rq3_analysis_module", Boom())
    args = argparse.Namespace(run_rq3=False)
    fp._maybe_run_rq3(args)


def test_maybe_run_rq3_with_split(monkeypatch):
    captured = {}

    class Dummy:
        def main(self):
            captured["argv"] = list(sys.argv)

    monkeypatch.setattr(fp, "_rq3_analysis_module", Dummy())
    monkeypatch.setattr(fp, "_RQ3_IMPORT_ERROR", None)
    args = argparse.Namespace(run_rq3=True, scan_root="root", split="dev")
    old = list(sys.argv)
    fp._maybe_run_rq3(args)
    assert captured["argv"][0] == "rq3_analysis.py"
    assert "--split" in captured["argv"]
    assert "dev" in captured["argv"]
    assert sys.argv == old


def test_save_outputs_with_plot(monkeypatch, tmp_path, capsys):
    args = argparse.Namespace(
        dataset_name="DS",
        model_name="Model",
        out_dir=str(tmp_path),
        scan_root="root",
        make_plot=True,
        dpi=50,
    )

    def fake_plot_figure(df_top, df_bot, bins, config):
        Path(config.out_png).write_text("png", encoding="utf-8")
        Path(config.out_pdf).write_text("pdf", encoding="utf-8")

    monkeypatch.setattr(fp, "plot_figure", fake_plot_figure)
    grp_top = pd.DataFrame({"x": [1]})
    df_bot = pd.DataFrame({"y": [2]})
    fp._save_outputs(args, grp_top, df_bot, [0, 1])
    out = capsys.readouterr().out

    assert (tmp_path / "uncertainty_gated_effect__DS__Model.png").exists()
    assert "[saved]" in out


def test_main_entrypoint_exits_when_no_files(monkeypatch, tmp_path):
    import src.analysis.io as analysis_io

    args = argparse.Namespace(
        scan_root=str(tmp_path),
        split=None,
        bins=[0, 1],
        min_step=None,
        max_step=None,
        carpark_success_op="ge",
        carpark_soft_threshold=0.0,
        out_dir=str(tmp_path),
        dataset_name="DS",
        model_name="Model",
        make_plot=False,
        dpi=100,
    )

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda self: args)
    monkeypatch.setattr(analysis_io, "scan_files_step_only", lambda root, split_substr=None, skip_substrings=None: [])

    with pytest.raises(SystemExit):
        runpy.run_path(fp.__file__, run_name="__main__")


def test_main_exits_when_dataframe_empty(monkeypatch, tmp_path):
    args = argparse.Namespace(
        scan_root=str(tmp_path),
        split=None,
        bins=[0, 1],
        min_step=None,
        max_step=None,
        carpark_success_op="ge",
        carpark_soft_threshold=0.0,
        out_dir=str(tmp_path),
        dataset_name="DS",
        model_name="Model",
        make_plot=False,
        dpi=100,
    )

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda self: args)
    monkeypatch.setattr(fp, "scan_files_step_only", lambda root, split_substr=None, skip_substrings=None: ["file1"])
    monkeypatch.setattr(fp, "_maybe_run_rq3", lambda args: None)
    monkeypatch.setattr(fp, "load_dataframe", lambda **kwargs: pd.DataFrame())

    with pytest.raises(SystemExit):
        fp.main()
