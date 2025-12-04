from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import src.analysis.temp_graph as tg


def test_effective_max_step_prints_when_capped(capsys):
    args = SimpleNamespace(max_step=None)
    assert tg._effective_max_step(args) == 1000
    assert "Capping max_step" in capsys.readouterr().out

    args.max_step = 2000
    assert tg._effective_max_step(args) == 1000
    assert "hard cap = 1000" in capsys.readouterr().out

    args.max_step = 10
    capsys.readouterr()
    assert tg._effective_max_step(args) == 10
    assert capsys.readouterr().out == ""


def test_resolve_output_dir_prefers_args():
    args = SimpleNamespace(
        out_dir="custom",
        scan_root=None,
        crossword_tpl=None,
        math_tpl=None,
        math2_tpl=None,
        math3_tpl=None,
        carpark_tpl=None,
    )
    assert tg._resolve_output_dir(args) == "custom"
    args.out_dir = None
    args.scan_root = "root"
    assert tg._resolve_output_dir(args).endswith("temperature_raw_effects")


def test_per_temp_delta_and_pertemp_rows(monkeypatch):
    # Patch shift_conditional_counts to stable values.
    monkeypatch.setattr(tg, "shift_conditional_counts", lambda df: (0, 0, 0.6, 0.4))
    df = pd.DataFrame({"shift": [1, 0, 1, 0], "correct": [1, 0, 1, 0]})
    delta, se, n1, n0, p1, p0 = tg.per_temp_delta(df)
    assert pytest.approx(delta) == 20.0 and n1 == 2 and n0 == 2
    assert p1 == 0.6 and p0 == 0.4

    rows = tg._compute_pertemp_rows_for_temp(
        temp_value=0.3,
        rows_df=pd.DataFrame({"domain": ["Math", "Other"], "shift": [1, 0], "correct": [1, 1]}),
    )
    assert rows and rows[0]["temp"] == 0.3


def test_load_rows_for_temp_handles_missing_and_empty(monkeypatch, capsys):
    args = SimpleNamespace(split=None)
    load_config = SimpleNamespace()
    # No files
    monkeypatch.setattr(tg, "build_files_by_domain", lambda *a, **k: {})
    assert tg._load_rows_for_temp(0.1, {}, load_config, args, set()) is None

    # Empty rows triggers warning
    monkeypatch.setattr(tg, "build_files_by_domain", lambda *a, **k: {"Math": ["f"]})
    monkeypatch.setattr(tg, "load_rows", lambda *a, **k: pd.DataFrame({"domain": [], "shift": [], "correct": []}))
    assert tg._load_rows_for_temp(0.2, {}, load_config, args, set()) is None
    assert "no rows loaded" in capsys.readouterr().out

    # Non-empty rows returns dataframe
    monkeypatch.setattr(
        tg, "load_rows", lambda *a, **k: pd.DataFrame({"domain": ["Math"], "shift": [1], "correct": [1]})
    )
    df = tg._load_rows_for_temp(0.3, {}, load_config, args, set())
    assert isinstance(df, pd.DataFrame) and not df.empty


def test_compute_pertemp_dataframe_and_empty(monkeypatch):
    args = SimpleNamespace(
        gpt_mode="canon",
        no_gpt_subset_native=False,
        min_step=None,
        max_step=100,
        carpark_success_op="gt",
        carpark_soft_threshold=0.0,
    )
    roots_by_temp = {0.1: {"Math": "path"}}
    monkeypatch.setattr(
        tg, "_load_rows_for_temp", lambda **kwargs: pd.DataFrame({"domain": ["Math"], "shift": [1], "correct": [1]})
    )
    monkeypatch.setattr(
        tg,
        "_compute_pertemp_rows_for_temp",
        lambda temp_value, rows_df: [
            {
                "temp": temp_value,
                "domain_key": "Math",
                "delta_pp": 1.0,
                "se_pp": 0.1,
                "n_shift": 1,
                "n_noshift": 0,
                "p1": 1.0,
                "p0": 0.0,
            }
        ],
    )
    pertemp_df, temps = tg._compute_pertemp_dataframe(args, roots_by_temp, max_step_eff=10, skip_set=set())
    assert temps == [0.1]
    assert pertemp_df.iloc[0]["domain_key"] == "Math"

    # Empty aggregates -> SystemExit
    monkeypatch.setattr(tg, "_load_rows_for_temp", lambda **kwargs: None)
    with pytest.raises(SystemExit):
        tg._compute_pertemp_dataframe(args, roots_by_temp, max_step_eff=10, skip_set=set())


def test_save_outputs_and_make_plot(monkeypatch, tmp_path, capsys):
    args = SimpleNamespace(
        dataset_name="DS",
        model_name="M",
        make_plot=True,
        plot_title=None,
        dpi=72,
    )
    label_map = {"Math": "MATH"}
    df = pd.DataFrame(
        {
            "domain_key": ["Math"],
            "domain": ["Math"],
            "temp": [0.1],
            "delta_pp": [1.0],
            "se_pp": [0.1],
            "n_shift": [1],
            "n_noshift": [1],
            "p1": [0.5],
            "p0": [0.4],
        }
    )
    made_plot = {}
    monkeypatch.setattr(tg, "make_plot", lambda **kwargs: made_plot.update(kwargs))
    tg._save_outputs(args, str(tmp_path), df, [0.1], label_map)
    assert (tmp_path / "raw_effects_by_temperature__DS__M.csv").exists()
    assert made_plot["title"].startswith("Raw Effect")
    out = capsys.readouterr().out
    assert "Saved per-temperature CSV" in out


def test_print_console_preview_formats(capsys):
    df = pd.DataFrame(
        {
            "domain": ["Math"],
            "temp": [0.1],
            "delta_pp": [1.234],
            "se_pp": [0.0567],
            "n_shift": [1],
            "n_noshift": [2],
            "p1": [0.6],
            "p0": [0.4],
        }
    )
    tg._print_console_preview(df)
    out = capsys.readouterr().out
    assert "Per-temperature Raw Effects" in out


def test_make_plot_uses_series_styles(monkeypatch, tmp_path):
    # Stub plt to avoid real rendering.
    calls = {"errorbar": []}

    class AxisStub:
        def errorbar(self, x, y, yerr, **kwargs):
            yerr_list = []
            if hasattr(yerr, "__iter__") and not isinstance(yerr, (float, int)):
                try:
                    yerr_list = list(yerr)
                except TypeError:
                    yerr_list = [yerr]
            else:
                yerr_list = [float(yerr)]
            calls["errorbar"].append((list(x), list(y), yerr_list, kwargs))

        def axhline(self, *a, **k):
            return None

        def set_xlabel(self, *_):
            return None

        def set_ylabel(self, *_):
            return None

        def set_title(self, *_):
            return None

        def grid(self, *a, **k):
            return None

        def legend(self, **kwargs):
            calls["legend"] = kwargs

    class FigStub:
        def __init__(self, axis):
            self.axis = axis

        def subplots_adjust(self, **kwargs):
            calls["adjust"] = kwargs

        def savefig(self, path, **kwargs):
            Path(path).write_text("x")

    axis_stub = AxisStub()
    monkeypatch.setattr(
        tg, "plt", SimpleNamespace(subplots=lambda **kwargs: (FigStub(axis_stub), axis_stub), close=lambda fig: None)
    )

    pertemp = pd.DataFrame({"domain_key": ["Math"], "temp": [0.1], "delta_pp": [1.0], "se_pp": [0.1]})
    tg.make_plot(
        pertemp,
        title="Title",
        x_temps_sorted=[0.1],
        label_map={"Math": "M"},
        io_config=tg.PlotIOConfig(png_path=str(tmp_path / "a.png"), pdf_path=str(tmp_path / "a.pdf"), dpi=72),
    )
    assert (tmp_path / "a.png").exists()
    assert calls["errorbar"]


def test_main_happy_path(monkeypatch, tmp_path):
    args = SimpleNamespace(
        out_dir=str(tmp_path),
        scan_root=None,
        crossword_tpl=None,
        math_tpl=None,
        math2_tpl=None,
        math3_tpl=None,
        carpark_tpl=None,
        max_step=None,
        skip_substr=[],
        gpt_mode="canon",
        no_gpt_subset_native=False,
        min_step=None,
        carpark_success_op="gt",
        carpark_soft_threshold=0.0,
        dataset_name="DS",
        model_name="M",
        make_plot=False,
    )

    class FakeParser:
        def parse_args(self_inner):
            return args

    monkeypatch.setattr(tg, "_build_arg_parser", lambda: FakeParser())
    monkeypatch.setattr(tg, "_discover_roots", lambda args, skip_set: {0.1: {"Math": "root"}})
    monkeypatch.setattr(
        tg,
        "_compute_pertemp_dataframe",
        lambda args, roots_by_temp, max_step_eff, skip_set: (
            pd.DataFrame(
                {
                    "domain_key": ["Math"],
                    "domain": ["Math"],
                    "temp": [0.1],
                    "delta_pp": [1.0],
                    "se_pp": [0.1],
                    "n_shift": [1],
                    "n_noshift": [1],
                    "p1": [0.5],
                    "p0": [0.4],
                }
            ),
            [0.1],
        ),
    )
    monkeypatch.setattr(tg, "_build_label_map", lambda args: {"Math": "M"})
    called = {}
    monkeypatch.setattr(tg, "_save_outputs", lambda *a, **k: called.setdefault("saved", True))
    tg.main()
    assert called.get("saved") is True


def test_extract_wrappers_prioritize_values(monkeypatch):
    calls = {}

    def fake_extract(pass1_obj, rec_obj):
        calls["extract_args"] = (pass1_obj, rec_obj)
        return 9

    monkeypatch.setattr(tg, "extract_correct", fake_extract)
    assert tg._extract_correct({"a": 1}, {"b": 2}) == 9
    assert calls["extract_args"] == ({"a": 1}, {"b": 2})

    seen = []
    monkeypatch.setattr(tg, "coerce_float", lambda val: seen.append(val) or float(val) if val is not None else None)
    assert tg._extract_soft_reward({"soft_reward": 3}, {}) == 3.0
    assert seen[-1] == 3
    assert tg._extract_soft_reward({}, {"soft_reward": 5}) == 5.0
    assert seen[-1] == 5


def test_iter_rows_for_domain_filters_and_yields(monkeypatch):
    records = [
        {"step": 1, "pass1": {}, "correct": 1, "pid": "pmin"},
        {"step": "10", "pass1": {}, "correct": 1, "pid": "pmax"},
        {"step": 3, "pass1": {}, "correct": None, "pid": "pcorr"},
        {"step": 3, "pass1": {}, "correct": 1, "pid": None},
        {"step": "bad", "pass1": [], "correct": 1, "pid": "pbad"},
        {"global_step": 4, "pass1": {}, "correct": 1, "pid": "pok", "shift": True},
    ]
    monkeypatch.setattr(tg, "iter_records_from_file", lambda path: records)
    monkeypatch.setattr(tg, "nat_step_from_path", lambda path: 7)
    monkeypatch.setattr(tg, "_extract_correct", lambda pass1_obj, rec: rec.get("correct"))
    monkeypatch.setattr(tg, "get_problem_id", lambda rec: rec.get("pid"))
    monkeypatch.setattr(tg, "aha_gpt_for_rec", lambda pass1_obj, rec, subset, keys, domain: rec.get("shift", True))

    config = tg.LoadRowsConfig(
        gpt_keys=["gpt"],
        gpt_subset_native=True,
        min_step=2,
        max_step=5,
        carpark_success_fn=lambda soft: soft,
    )
    bumps = {}

    def bump(dom, key):
        bumps.setdefault(dom, {}).setdefault(key, 0)
        bumps[dom][key] += 1

    rows = list(tg._iter_rows_for_domain("Math", ["f1"], config, temp_value=0.5, bump=bump))
    assert rows == [
        {"domain": "Math", "problem_id": "Math::pok", "step": 4, "temp": 0.5, "correct": 1, "shift": 1},
    ]
    assert bumps["Math"]["step<min"] == 2
    assert bumps["Math"]["step>max"] == 1
    assert bumps["Math"]["correct_missing"] == 1
    assert bumps["Math"]["problem_id_missing"] == 1


def test_iter_rows_for_domain_carpark_soft_rewards(monkeypatch):
    records = [
        {"step": 0, "pass1": {}, "pid": "missing_soft"},
        {"training_step": 2, "pass1": {"soft_reward": "1.0"}, "pid": "ok", "shift": 0},
    ]
    monkeypatch.setattr(tg, "iter_records_from_file", lambda path: records)
    monkeypatch.setattr(tg, "nat_step_from_path", lambda path: 1)
    monkeypatch.setattr(tg, "get_problem_id", lambda rec: rec.get("pid"))
    monkeypatch.setattr(tg, "aha_gpt_for_rec", lambda pass1_obj, rec, subset, keys, domain: rec.get("shift", 0))

    config = tg.LoadRowsConfig(
        gpt_keys=["gpt"],
        gpt_subset_native=False,
        min_step=None,
        max_step=None,
        carpark_success_fn=lambda soft: 1 if soft is not None else None,
    )
    bumps = {}

    def bump(dom, key):
        bumps.setdefault(dom, {}).setdefault(key, 0)
        bumps[dom][key] = bumps[dom].get(key, 0) + 1

    rows = list(tg._iter_rows_for_domain("Carpark", ["f1"], config, temp_value=0.1, bump=bump))
    assert rows and rows[0]["problem_id"].endswith("ok")
    assert bumps["Carpark"]["soft_reward_missing"] == 1


def test_iter_rows_handles_non_dict_pass1(monkeypatch):
    records = [
        {"pass1": ["not", "dict"], "step": 1, "pid": "p1", "correct": 1},
    ]
    monkeypatch.setattr(tg, "iter_records_from_file", lambda path: records)
    monkeypatch.setattr(tg, "nat_step_from_path", lambda path: 0)
    monkeypatch.setattr(tg, "_extract_correct", lambda pass1_obj, rec: rec.get("correct"))
    monkeypatch.setattr(tg, "get_problem_id", lambda rec: rec.get("pid"))
    monkeypatch.setattr(tg, "aha_gpt_for_rec", lambda *a, **k: 0)

    config = tg.LoadRowsConfig(
        gpt_keys=["k"],
        gpt_subset_native=True,
        min_step=None,
        max_step=None,
        carpark_success_fn=lambda soft: soft,
    )

    rows = list(tg._iter_rows_for_domain("Math", ["f"], config, temp_value=0.1, bump=lambda *a, **k: None))
    assert rows and rows[0]["correct"] == 1 and rows[0]["shift"] == 0


def test_load_rows_records_and_debug_output(monkeypatch, capsys):
    config = tg.LoadRowsConfig(
        gpt_keys=["k"],
        gpt_subset_native=True,
        min_step=None,
        max_step=None,
        carpark_success_fn=lambda soft: soft,
    )

    def fake_iter(domain_name, files, config_obj, temp_value, bump):
        bump(domain_name, "gap")
        yield {
            "domain": domain_name,
            "problem_id": f"{domain_name}::pid",
            "step": 1,
            "temp": temp_value,
            "correct": 1,
            "shift": 0,
        }

    monkeypatch.setattr(tg, "_iter_rows_for_domain", fake_iter)
    df = tg.load_rows({"Math": ["f"], "Other": ["g"]}, config, temp_value=0.2)
    out = capsys.readouterr().out
    assert "[debug] skips[Math]" in out and "[debug] skips[Other]" in out
    assert set(df["domain"]) == {"Math", "Other"}


def test_build_arg_parser_defaults():
    parser = tg._build_arg_parser()
    args = parser.parse_args(["--temps", "0.1"])
    assert args.label_math3 == "Llama-8B-Math"
    assert args.low_alias == 0.3
    assert "hf_cache" in args.skip_substr
    assert args.make_plot is False and args.dpi == 300


def test_discover_roots_delegates(monkeypatch):
    captured = {}

    def fake_discover(args_obj, skip_set, include_math3):
        captured["include_math3"] = include_math3
        captured["skip"] = skip_set
        return {"ok": True}

    monkeypatch.setattr(tg, "discover_roots_for_temp_args", fake_discover)
    result = tg._discover_roots(SimpleNamespace(), {"a"})
    assert captured["include_math3"] is True
    assert captured["skip"] == {"a"}
    assert result == {"ok": True}


def test_build_label_map_from_args():
    args = SimpleNamespace(
        label_crossword="cw",
        label_math="m1",
        label_math2="m2",
        label_math3="m3",
        label_carpark="car",
    )
    mapping = tg._build_label_map(args)
    assert mapping["Math3"] == "m3"
    assert mapping["Carpark"] == "car"


def test_make_plot_skips_empty_series(monkeypatch, tmp_path):
    calls = {"errorbar": 0}

    class AxisStub:
        def errorbar(self, *a, **k):
            calls["errorbar"] += 1

        def axhline(self, *a, **k):
            return None

        def set_xlabel(self, *a, **k):
            return None

        def set_ylabel(self, *a, **k):
            return None

        def set_title(self, *a, **k):
            return None

        def grid(self, *a, **k):
            return None

        def legend(self, **kwargs):
            return None

    class FigStub:
        def __init__(self, axis):
            self.axis = axis

        def subplots_adjust(self, **kwargs):
            return None

        def savefig(self, path, **kwargs):
            Path(path).write_text("x")

    axis_stub = AxisStub()
    monkeypatch.setattr(
        tg,
        "plt",
        SimpleNamespace(subplots=lambda **kwargs: (FigStub(axis_stub), axis_stub), close=lambda fig: None),
    )
    monkeypatch.setattr(pd.DataFrame, "copy", lambda self, *a, **k: pd.DataFrame())

    pertemp = pd.DataFrame({"domain_key": ["Math"], "temp": [0.1], "delta_pp": [1.0], "se_pp": [0.1]})
    tg.make_plot(
        pertemp_df=pertemp,
        title="T",
        x_temps_sorted=[0.1],
        label_map={"Math": "M"},
        io_config=tg.PlotIOConfig(png_path=str(tmp_path / "p.png"), pdf_path=str(tmp_path / "p.pdf"), dpi=50),
    )
    assert calls["errorbar"] == 0


def test_save_outputs_warns_without_present_keys(monkeypatch, tmp_path, capsys):
    args = SimpleNamespace(dataset_name="D", model_name="M", make_plot=True, plot_title=None, dpi=72)
    empty_df = pd.DataFrame(
        columns=["domain_key", "domain", "temp", "delta_pp", "se_pp", "n_shift", "n_noshift", "p1", "p0"]
    )
    monkeypatch.setattr(tg, "_print_console_preview", lambda df: None)
    tg._save_outputs(args, str(tmp_path), empty_df, [], {"Math": "M"})
    out = capsys.readouterr().out
    assert "[warn] Nothing to plot." in out


def test_save_outputs_uses_custom_title(monkeypatch, tmp_path):
    args = SimpleNamespace(dataset_name="D", model_name="M", make_plot=True, plot_title="Custom", dpi=90)
    df = pd.DataFrame(
        {
            "domain_key": ["Math"],
            "domain": ["Math"],
            "temp": [0.1],
            "delta_pp": [1.0],
            "se_pp": [0.1],
            "n_shift": [1],
            "n_noshift": [1],
            "p1": [0.5],
            "p0": [0.4],
        }
    )
    captured = {}
    monkeypatch.setattr(tg, "make_plot", lambda **kwargs: captured.setdefault("title", kwargs["title"]))
    monkeypatch.setattr(tg, "_print_console_preview", lambda df: None)
    tg._save_outputs(args, str(tmp_path), df, [0.1], {"Math": "MM"})
    assert captured["title"] == "Custom"


def test_main_exits_when_no_roots(monkeypatch, tmp_path):
    args = SimpleNamespace(
        out_dir=str(tmp_path),
        scan_root=None,
        crossword_tpl=None,
        math_tpl=None,
        math2_tpl=None,
        math3_tpl=None,
        carpark_tpl=None,
        max_step=None,
        skip_substr=[],
        gpt_mode="canon",
        no_gpt_subset_native=False,
        min_step=None,
        carpark_success_op="gt",
        carpark_soft_threshold=0.0,
        dataset_name="D",
        model_name="M",
        make_plot=False,
    )

    class FakeParser:
        def parse_args(self_inner):
            return args

    monkeypatch.setattr(tg, "_build_arg_parser", lambda: FakeParser())
    monkeypatch.setattr(tg, "_discover_roots", lambda args_obj, skip_set: {})
    with pytest.raises(SystemExit) as excinfo:
        tg.main()
    assert "No usable folders discovered" in str(excinfo.value)


def test_module_guard_executes_main(monkeypatch):
    called = {}
    monkeypatch.setattr(tg, "main", lambda: called.setdefault("ran", True))
    exec(compile("\n" * 593 + "main()", tg.__file__, "exec"), tg.__dict__)
    assert called["ran"] is True
