import runpy
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import src.analysis.heatmap_1 as heatmap


def test_frac8_label_handles_floats():
    assert heatmap.frac8_label(0.123) == "0.123"


def test_set_rendered_width_returns_false_when_too_small():
    fig, _ = plt.subplots()
    try:
        success = heatmap.set_rendered_width(fig, target_width_in=0.01, dpi=200)
        assert success is False
    finally:
        plt.close(fig)


def test_write_latex_helper_no_matching_row(tmp_path):
    args = types.SimpleNamespace(model_name="M", dataset_name="D")
    overall_grid = pd.DataFrame(
        {
            "delta1": [0.0],
            "delta2": [0.0],
            "n_events": [0],
            "n_pairs": [0],
            "pct": [np.nan],
        }
    )
    heatmap._write_latex_helper(
        args=args,
        overall_grid=overall_grid,
        delta_values=[0.25],
        out_dir=str(tmp_path),
        slug="s",
    )
    # Early return: no file should be written.
    assert not list(tmp_path.iterdir())


def test_main_executes_with_stubbed_dependencies(monkeypatch, tmp_path):
    # Stub out helper dependencies used inside main to avoid heavy I/O.
    monkeypatch.setattr(
        "src.analysis.core.iter_correct_and_shift_samples_for_config",
        lambda files_by_domain, config: [
            ("Crossword", 0, {"problem": "p1"}, True, False),
            ("Crossword", 1, {"problem": "p1"}, False, False),
        ],
    )
    monkeypatch.setattr(
        "src.analysis.core.LoadRowsConfig",
        lambda **kwargs: types.SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        "src.analysis.io.build_jsonl_files_by_domain", lambda roots, split: ({"Crossword": ["f"]}, str(tmp_path))
    )
    monkeypatch.setattr("src.analysis.metrics.make_carpark_success_fn", lambda *args, **kwargs: lambda *_a, **_k: True)
    monkeypatch.setattr("src.analysis.utils.gpt_keys_for_mode", lambda mode: ["gpt-key"])
    monkeypatch.setattr("src.analysis.utils.get_problem_id", lambda rec: rec.get("problem"))

    def _add_carpark_threshold_args(parser):
        parser.add_argument("--carpark_success_op", default="ge")
        parser.add_argument("--carpark_soft_threshold", type=float, default=0.0)
        return None

    monkeypatch.setattr("src.analysis.utils.add_carpark_threshold_args", _add_carpark_threshold_args)
    monkeypatch.setattr(
        "src.analysis.core.plotting_helpers.compute_effective_max_step", lambda args, hard_max_step=None: 2
    )
    monkeypatch.setattr("src.analysis.plotting.apply_default_style", lambda: None)

    # Run the module as a script to exercise the __main__ guard and per-domain branch.
    argv_before = sys.argv
    sys.argv = ["heatmap_1.py", "--root_crossword", str(tmp_path)]
    try:
        runpy.run_path("src/analysis/heatmap_1.py", run_name="__main__")
    finally:
        sys.argv = argv_before


def test_get_rendered_size_and_set_rendered_width_success(monkeypatch):
    class DummyFig:
        def __init__(self):
            self.size = [2.0, 1.0]
            self.saved = []

        def savefig(self, path, bbox_inches=None, dpi=None):
            self.saved.append((path, dpi))

        def get_size_inches(self):
            return self.size

        def set_size_inches(self, new_size):
            self.size = list(new_size)

    monkeypatch.setattr(heatmap, "imread", lambda path: np.zeros((100, 200, 3)))
    fig = DummyFig()
    ok = heatmap.set_rendered_width(fig, target_width_in=1.0, dpi=200)
    assert ok
    width, height = heatmap.get_rendered_size(fig, dpi=200)
    assert pytest.approx(width, rel=1e-6) == 1.0
    assert pytest.approx(height, rel=1e-6) == 0.5


def test_write_latex_helper_writes_file(tmp_path):
    args = types.SimpleNamespace(model_name="M", dataset_name="D")
    overall_grid = pd.DataFrame(
        {
            "delta1": [0.125],
            "delta2": [0.125],
            "n_events": [3],
            "n_pairs": [6],
            "pct": [50.0],
        }
    )
    heatmap._write_latex_helper(
        args=args,
        overall_grid=overall_grid,
        delta_values=[0.125, 0.25],
        out_dir=str(tmp_path),
        slug="s",
    )
    out_tex = tmp_path / "aha_heatmap_summary__s.tex"
    assert out_tex.exists()
    assert "50.00" in out_tex.read_text()


def test_main_per_domain_branch(monkeypatch, tmp_path):
    calls = {"per_domain": 0, "group": 0}
    monkeypatch.setattr(
        heatmap,
        "_build_arg_parser",
        lambda: types.SimpleNamespace(
            parse_args=lambda: types.SimpleNamespace(
                root_crossword="r1",
                root_math=None,
                root_math2=None,
                root_math3=None,
                root_carpark=None,
                split=None,
                out_dir=str(tmp_path),
                dataset_name="D",
                model_name="M",
                per_domain=True,
                make_15b_overall=False,
                domains_15b="Crossword",
                delta_values=[0.0],
                gpt_mode="canonical",
                no_gpt_subset_native=False,
                min_step=None,
                max_step=None,
                carpark_success_op="gt",
                carpark_soft_threshold=0.0,
                cmap="YlGnBu",
                title_overall="T",
                title_15b="T15",
            )
        ),
    )
    monkeypatch.setattr(heatmap, "_build_label_map", lambda args: {"Crossword": "CW"})
    monkeypatch.setattr(heatmap, "_collect_files_and_out_dir", lambda args: ({"Crossword": ["f1"]}, str(tmp_path)))
    monkeypatch.setattr(heatmap, "_build_load_config", lambda args: "cfg")
    monkeypatch.setattr(
        heatmap,
        "_load_step_level_data",
        lambda files_by_domain, load_config: pd.DataFrame(
            {
                "domain_key": ["Crossword"],
                "problem_id": ["p1"],
                "step": [0],
                "acc_frac": [0.5],
                "shift_frac": [0.0],
            }
        ),
    )
    monkeypatch.setattr(
        heatmap,
        "_build_overall_grid_and_rows",
        lambda step_df, delta_values: (
            pd.DataFrame({"delta1": [0.0], "delta2": [0.0], "n_events": [1], "n_pairs": [1], "pct": [100.0]}),
            [],
        ),
    )
    monkeypatch.setattr(heatmap, "plot_heatmap", lambda *a, **k: None)
    monkeypatch.setattr(
        heatmap,
        "_add_per_domain_grids_and_plots",
        lambda *a, **k: calls.__setitem__("per_domain", calls["per_domain"] + 1),
    )
    monkeypatch.setattr(
        heatmap, "_add_group_15b_grid_and_plot", lambda *a, **k: calls.__setitem__("group", calls["group"] + 1)
    )
    monkeypatch.setattr(heatmap, "_write_output_table", lambda *a, **k: "out.csv")
    monkeypatch.setattr(heatmap, "_write_latex_helper", lambda *a, **k: None)
    monkeypatch.setattr(heatmap, "_print_summary", lambda *a, **k: None)

    heatmap.main()
    assert calls["per_domain"] == 1


def test_iter_sample_rows_skips_missing_problem(monkeypatch):
    monkeypatch.setattr(
        heatmap,
        "iter_correct_and_shift_samples_for_config",
        lambda files_by_domain, config: [("D", 0, {"raw": "rec"}, True, False)],
    )
    monkeypatch.setattr(heatmap, "get_problem_id", lambda rec: None)
    rows = list(heatmap._iter_sample_rows({"D": ["f"]}, config=None))
    assert rows == []


def test_set_rendered_width_missing_attrs_and_attribute_error(monkeypatch):
    assert heatmap.set_rendered_width(object(), target_width_in=1.0) is False

    class BadFig:
        def get_size_inches(self):
            raise AttributeError("boom")

        def set_size_inches(self, _size):
            pass

    assert heatmap.set_rendered_width(BadFig(), target_width_in=1.0) is False

    # Force the max-iter fallthrough path (never converges, never too small)
    calls = {"get_rendered_size": 0}

    class StableFig:
        def __init__(self):
            self.size = [1.0, 1.0]

        def get_size_inches(self):
            return self.size

        def set_size_inches(self, new_size):
            self.size = list(new_size)

    monkeypatch.setattr(
        heatmap,
        "get_rendered_size",
        lambda fig, dpi=200: calls.__setitem__("get_rendered_size", calls["get_rendered_size"] + 1) or (1.2, 1.0),
    )
    assert heatmap.set_rendered_width(StableFig(), target_width_in=1.0, dpi=200, max_iter=2) is False
    assert calls["get_rendered_size"] == 2


def test_plot_heatmap_fallback_cmap(monkeypatch, tmp_path):
    class DummyCmap:
        def __call__(self, _value):
            return (1.0, 1.0, 1.0, 1.0)

    class DummyLinear:
        called = []

        @classmethod
        def from_list(cls, name, colors):
            cls.called.append((name, colors))
            return DummyCmap()

    monkeypatch.setattr(heatmap, "cm", types.SimpleNamespace())
    monkeypatch.setattr(
        heatmap,
        "mcolors",
        types.SimpleNamespace(
            LinearSegmentedColormap=DummyLinear,
            Normalize=heatmap.mcolors.Normalize,
        ),
    )

    class DummyAxes:
        def __init__(self):
            self.calls = []

        def imshow(self, *a, **k):
            self.calls.append("imshow")

        def set_xticks(self, *a, **k):
            pass

        def set_yticks(self, *a, **k):
            pass

        def set_xticklabels(self, *a, **k):
            pass

        def set_yticklabels(self, *a, **k):
            pass

        def set_xlabel(self, *a, **k):
            pass

        def set_ylabel(self, *a, **k):
            pass

        def set_title(self, *a, **k):
            pass

        def text(self, *a, **k):
            pass

    class DummyFig:
        def __init__(self):
            self.saved = []

        def tight_layout(self):
            pass

        def savefig(self, path, *a, **k):
            path = Path(path)
            path.write_bytes(b"ok")
            self.saved.append(path)

        def set_size_inches(self, *_args):
            pass

        def get_size_inches(self):
            return (1.0, 1.0)

    dummy_axes = DummyAxes()
    monkeypatch.setattr(heatmap.plt, "subplots", lambda figsize=None: (DummyFig(), dummy_axes))
    monkeypatch.setattr(heatmap, "set_rendered_width", lambda *a, **k: True)
    monkeypatch.setattr(heatmap.plt, "close", lambda fig: None)

    df = pd.DataFrame(
        {
            "delta1": [0.0],
            "delta2": [0.0],
            "n_events": [1],
            "n_pairs": [1],
            "pct": [100.0],
        }
    )
    out_png = tmp_path / "h.png"
    heatmap.plot_heatmap(df, "T", str(out_png), cmap_name="custom")
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    assert DummyLinear.called[0][0] == "custom"


def test_plot_heatmap_uses_mcolors_get_cmap(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr(heatmap, "cm", types.SimpleNamespace())  # no get_cmap

    class DummyCmap:
        def __call__(self, _value):
            return (0.0, 0.0, 0.0, 1.0)

    def fake_get_cmap(name):
        calls["cmap"] = name
        return DummyCmap()

    monkeypatch.setattr(
        heatmap,
        "mcolors",
        types.SimpleNamespace(
            get_cmap=fake_get_cmap,
            Normalize=heatmap.mcolors.Normalize,
        ),
    )

    class DummyAxes:
        def imshow(self, *a, **k):
            pass

        def set_xticks(self, *a, **k):
            pass

        def set_yticks(self, *a, **k):
            pass

        def set_xticklabels(self, *a, **k):
            pass

        def set_yticklabels(self, *a, **k):
            pass

        def set_xlabel(self, *a, **k):
            pass

        def set_ylabel(self, *a, **k):
            pass

        def set_title(self, *a, **k):
            pass

        def text(self, *a, **k):
            pass

    class DummyFig:
        def tight_layout(self):
            pass

        def savefig(self, path, *a, **k):
            Path(path).write_bytes(b"ok")

        def set_size_inches(self, *_a):
            pass

        def get_size_inches(self):
            return (1.0, 1.0)

    monkeypatch.setattr(heatmap.plt, "subplots", lambda figsize=None: (DummyFig(), DummyAxes()))
    monkeypatch.setattr(heatmap, "set_rendered_width", lambda *a, **k: True)
    monkeypatch.setattr(heatmap.plt, "close", lambda fig: None)

    df = pd.DataFrame({"delta1": [0.0], "delta2": [0.0], "n_events": [1], "n_pairs": [1], "pct": [50.0]})
    out_png = tmp_path / "mcolors.png"
    heatmap.plot_heatmap(df, "T", str(out_png), cmap_name="MC")
    assert calls["cmap"] == "MC"
    assert out_png.exists()


def test_collect_files_and_out_dir_errors(monkeypatch):
    args = types.SimpleNamespace(
        split=None,
        out_dir=None,
        root_crossword=None,
        root_math=None,
        root_math2=None,
        root_math3=None,
        root_carpark=None,
    )
    monkeypatch.setattr(heatmap, "_build_domain_roots", lambda a: {})
    monkeypatch.setattr(heatmap, "build_jsonl_files_by_domain", lambda roots, split: ({}, None))
    with pytest.raises(SystemExit, match="Provide at least one --root_\\* folder."):
        heatmap._collect_files_and_out_dir(args)

    monkeypatch.setattr(heatmap, "build_jsonl_files_by_domain", lambda roots, split: ({"Math": []}, "/tmp/root"))
    with pytest.raises(SystemExit, match="No JSONL files found"):
        heatmap._collect_files_and_out_dir(args)


def test_add_group_15b_grid_and_plot_branches(monkeypatch, capsys, tmp_path):
    args = types.SimpleNamespace(
        make_15b_overall=False,
        domains_15b="Crossword",
        delta_values=[0.0],
        title_15b="T15",
        cmap="YlGnBu",
    )
    long_rows: list[pd.DataFrame] = []
    heatmap._add_group_15b_grid_and_plot(pd.DataFrame(), args, long_rows, str(tmp_path))
    assert long_rows == []

    args.make_15b_overall = True
    step_df = pd.DataFrame({"domain_key": ["Other"], "delta": [0.0]})
    heatmap._add_group_15b_grid_and_plot(step_df, args, long_rows, str(tmp_path))
    captured = capsys.readouterr()
    assert "requested domains are present" in captured.out
    assert long_rows == []


def test_write_output_table_concatenates(tmp_path):
    df1 = pd.DataFrame({"a": [1]})
    df2 = pd.DataFrame({"a": [2]})
    out = heatmap._write_output_table([df1, df2], str(tmp_path), "slug")
    out_path = Path(out)
    assert out_path.exists()
    merged = pd.read_csv(out_path)
    assert merged["a"].tolist() == [1, 2]


def test_print_summary_outputs(capsys):
    args = types.SimpleNamespace(per_domain=True, make_15b_overall=True)
    overall_grid = pd.DataFrame(
        {
            "delta1": [0.0, 0.5],
            "delta2": [0.0, 0.5],
            "pct": [100.0, np.nan],
        }
    )
    heatmap._print_summary(
        args=args,
        overall_grid=overall_grid,
        out_csv="out.csv",
        out_png_overall="out.png",
        out_dir="/tmp/out",
    )
    out = capsys.readouterr().out
    assert "Overall Aha! prevalence grid" in out
    assert "NaN" in out
