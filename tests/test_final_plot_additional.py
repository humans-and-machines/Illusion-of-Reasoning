import argparse

import pandas as pd
import pytest

import src.analysis.final_plot as fp


def test_load_dataframe_with_filters(monkeypatch):
    records = [
        {
            "pass1": {"is_correct_pred": True, "entropy": 0.1, "shift_in_reasoning_v1": True},
            "pass2": {"has_reconsider_cue": True, "reconsider_markers": ["injected_cue"], "is_correct_pred": True},
            "step": 1,
            "example_id": "ex",
        },
        {
            "pass1": {"is_correct_pred": False, "entropy": None},
            "step": 2,
        },
    ]
    monkeypatch.setattr(fp, "iter_records_from_file", lambda path: records)
    df = fp.load_dataframe(["p1"], carpark_op="ge", carpark_thr=0.0, min_step=None, max_step=None)
    assert len(df) == 1
    assert df.iloc[0]["p2_triggered"] == 1
    assert df.iloc[0]["p2_correct"] == 1


def test_summarize_for_figure_handles_empty_stratum():
    df = pd.DataFrame(
        {
            "group_id": ["g1", "g2"],
            "entropy": [0.1, 0.2],
            "p1_correct": [1, 0],
            "p2_triggered": [1, 0],
            "p2_correct": [1, 0],
            "p1_shift": [1, 0],
        },
    )
    bins = [0.0, 0.5]
    top, bot = fp.summarize_for_figure(df, bins)
    assert "shift_share" in top.columns
    # Empty stratum should yield NaNs and zeros safely
    empty_rows = bot[bot["_stratum"] == "baseline 0/8"]
    assert empty_rows.empty or empty_rows["N"].fillna(0).iloc[0] == 0


def test_plotting_helpers_produce_paths(monkeypatch, tmp_path):
    layout = fp._build_interval_index([0.0, 1.0])
    assert layout.tick_labels == ["[0,1)"]


def test_plot_figure_saves(monkeypatch, tmp_path):
    df_top = pd.DataFrame({"_bin": pd.IntervalIndex.from_breaks([0, 1]), "N": [1], "shift_share": [0.5]})
    df_bot = pd.DataFrame(
        {
            "_bin": pd.IntervalIndex.from_breaks([0, 1]),
            "_stratum": ["baseline ≥1/8"],
            "N": [1],
            "rate": [0.5],
            "se": [0.1],
            "lo": [0.3],
            "hi": [0.7],
        },
    )
    bins = [0.0, 1.0]
    # use real plt but ensure files land in temp dir
    cfg = fp.FigureOutputConfig(
        out_png=str(tmp_path / "plot.png"), out_pdf=str(tmp_path / "plot.pdf"), title_suffix="title", dpi=100
    )
    fp.plot_figure(df_top, df_bot, bins, cfg)
    assert (tmp_path / "plot.png").exists()
    assert (tmp_path / "plot.pdf").exists()


def test_maybe_run_rq3_no_module(monkeypatch):
    monkeypatch.setattr(fp, "_rq3_analysis_module", None)
    monkeypatch.setattr(fp, "_RQ3_IMPORT_ERROR", ImportError("x"))
    args = argparse.Namespace(run_rq3=True, scan_root="root", split=None)
    fp._maybe_run_rq3(args)  # should no-op


def test_save_outputs_without_plot(monkeypatch, tmp_path, capsys):
    args = argparse.Namespace(
        dataset_name="DS",
        model_name="Model",
        out_dir=str(tmp_path),
        scan_root="root",
        make_plot=False,
    )
    grp_top = pd.DataFrame({"a": [1]})
    df_bot = pd.DataFrame({"b": [2]})
    bins = [0, 1]
    monkeypatch.setattr(fp, "plot_figure", lambda **kwargs: None)
    fp._save_outputs(args, grp_top, df_bot, bins)
    out = capsys.readouterr().out
    assert "saved" in out
    from pathlib import Path

    assert (Path(args.out_dir) / f"uncertainty_shift_prevalence__{args.dataset_name}__{args.model_name}.csv").exists()


def test_main_scans_and_exits(monkeypatch, tmp_path):
    # create fake file
    file_path = tmp_path / "step-1.jsonl"
    file_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        fp, "scan_files_step_only", lambda root, split_substr=None, skip_substrings=None: [str(file_path)]
    )
    monkeypatch.setattr(fp, "_maybe_run_rq3", lambda args: None)
    monkeypatch.setattr(
        fp,
        "load_dataframe",
        lambda files, carpark_op, carpark_thr, min_step, max_step: pd.DataFrame(
            {
                "group_id": ["g1"],
                "entropy": [0.1],
                "p1_correct": [1],
                "p2_triggered": [1],
                "p2_correct": [1],
                "p1_shift": [1],
            }
        ),
    )
    monkeypatch.setattr(
        fp,
        "summarize_for_figure",
        lambda df, bins: (pd.DataFrame({"_bin": [], "N": [], "shift_share": []}), pd.DataFrame()),
    )
    monkeypatch.setattr(fp, "_save_outputs", lambda args, grp_top, df_bot, bins: None)
    monkeypatch.setattr(fp, "add_carpark_threshold_args", lambda parser: None)
    monkeypatch.setattr(fp, "add_common_plot_args", lambda parser: None)
    monkeypatch.setattr(fp, "plt", None)

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
    )
    # patch parser.parse_args to return our args
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda self: args)
    fp.main()


def test_main_runs_rq3_and_exits_on_empty(monkeypatch, tmp_path):
    file_path = tmp_path / "step-1.jsonl"
    file_path.write_text("{}", encoding="utf-8")
    calls = {"rq3": 0, "load": 0}
    monkeypatch.setattr(
        fp, "scan_files_step_only", lambda root, split_substr=None, skip_substrings=None: [str(file_path)]
    )
    monkeypatch.setattr(fp, "_maybe_run_rq3", lambda args: calls.__setitem__("rq3", calls["rq3"] + 1))

    def fake_load(files, carpark_op, carpark_thr, min_step, max_step):
        calls["load"] += 1
        return pd.DataFrame()

    monkeypatch.setattr(fp, "load_dataframe", fake_load)
    monkeypatch.setattr(fp, "summarize_for_figure", lambda df, bins: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(fp, "_save_outputs", lambda *a, **k: None)
    monkeypatch.setattr(fp, "add_carpark_threshold_args", lambda parser: None)
    monkeypatch.setattr(fp, "add_common_plot_args", lambda parser: None)
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
        run_rq3=True,
    )
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda self: args)
    with pytest.raises(SystemExit):
        fp.main()
    assert calls["rq3"] == 1 and calls["load"] == 1
