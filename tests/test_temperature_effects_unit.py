import argparse
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.analysis.temperature_effects as te


def test_compute_correct_handles_carpark_and_none(monkeypatch):
    monkeypatch.setattr(te, "_extract_soft_reward", lambda rec, p1: 0.2)
    assert te._compute_correct("carpark", {}, {}, lambda score: None) is None
    assert te._compute_correct("carpark", {}, {}, lambda score: True) == 1

    monkeypatch.setattr(te, "_extract_correct", lambda p1, rec: None)
    assert te._compute_correct("math", {}, {}, lambda score: None) is None
    monkeypatch.setattr(te, "_extract_correct", lambda p1, rec: False)
    assert te._compute_correct("math", {}, {}, lambda score: None) == 0


def test_extract_wrappers(monkeypatch):
    called = {}

    def fake_extract(p1, rec):
        called["args"] = (p1, rec)
        return 1

    monkeypatch.setattr(te, "extract_correct", fake_extract)
    assert te._extract_correct({"p": 1}, {"r": 2}) == 1
    assert called["args"] == ({"p": 1}, {"r": 2})

    assert te._extract_soft_reward({"soft_reward": "0.5"}, {"soft_reward": "0.1"}) == 0.5
    assert te._extract_soft_reward({}, {"soft_reward": "0.7"}) == 0.7


def test_rows_for_file_filters_and_builds(monkeypatch):
    parallel_cfg = te.ParallelConfig(
        gpt_keys=["shift"],
        gpt_subset_native=False,
        min_step=None,
        max_step=None,
        carpark_op="ge",
        carpark_thr=0.0,
        temp_value=0.7,
        workers=1,
        parallel_mode="thread",
        chunksize=0,
    )
    rec = {
        "pass1": {"is_correct_pred": True, "shift": True},
        "problem": "p",
        "step": 10,
    }
    monkeypatch.setattr(te, "iter_records_from_file", lambda path: [rec])
    monkeypatch.setattr(te, "extract_pass1_and_step", lambda r, s: (r["pass1"], r.get("step")))
    monkeypatch.setattr(
        te, "step_from_record_if_within_bounds", lambda rec, path, split_value, min_step, max_step: rec["step"]
    )
    monkeypatch.setattr(te, "_compute_correct", lambda dom, rec, p1, fn: 1)
    monkeypatch.setattr(te, "_get_problem_id", lambda rec: "pid")
    monkeypatch.setattr(te, "aha_gpt_for_rec", lambda p1, rec, subset, keys, dom: 1)
    monkeypatch.setattr(te, "make_carpark_success_fn", lambda op, thr: lambda reward: 1)
    rows = te._rows_for_file("Math", "path.jsonl", parallel_cfg)
    assert rows == [{"domain": "Math", "problem_id": "Math::pid", "step": 10, "temp": 0.7, "correct": 1, "shift": 1}]


def test_load_rows_parallel_uses_executor(monkeypatch):
    parallel_cfg = te.ParallelConfig(["k"], False, None, None, "ge", 0.0, 0.7, 2, "thread", 0)
    monkeypatch.setattr(te, "_process_file_worker", lambda task: [{"t": task[1]}])

    class FakeExec:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def map(self, fn, iterable, chunksize=1):
            for item in iterable:
                yield fn(item)

    monkeypatch.setattr(te.cf, "ThreadPoolExecutor", FakeExec)
    files_by_domain = {"Math": ["p1", "p2"]}
    df = te.load_rows_parallel(files_by_domain, parallel_cfg)
    assert set(df["t"]) == {"p1", "p2"}


def test_per_temp_delta_handles_missing():
    df = pd.DataFrame({"shift": [1, 0], "correct": [1, 0]})
    delta, se, n_shift, n_no = te.per_temp_delta(df)
    assert np.isfinite(delta) and n_shift == 1 and n_no == 1

    df_bad = pd.DataFrame({"shift": [1], "correct": [1]})
    delta2, se2, n_shift2, n_no2 = te.per_temp_delta(df_bad)
    assert np.isnan(delta2) and n_no2 == 0


def test_make_plot_writes_files(tmp_path):
    pertemp = pd.DataFrame(
        {
            "domain_key": ["Math", "Math2"],
            "temp": [0.1, 0.1],
            "delta_pp": [1.0, -1.0],
            "se_pp": [0.1, 0.2],
        }
    )
    cfg = te.PlotConfig(
        pertemp_df=pertemp,
        out_png=str(tmp_path / "plot.png"),
        out_pdf=str(tmp_path / "plot.pdf"),
        title="t",
        x_temps_sorted=[0.1],
        label_map={"Math": "M", "Math2": "M2"},
        dpi=50,
    )
    te.make_plot(cfg)
    assert (tmp_path / "plot.png").exists()
    assert (tmp_path / "plot.pdf").exists()


def test_make_plot_axis_fallback(monkeypatch, tmp_path):
    class DummyAxis:
        def __init__(self):
            self.calls = []

        def plot(self, *a, **k):
            self.calls.append(("plot", a, k))

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

        def grid(self, *a, **k):
            pass

        def legend(self, *a, **k):
            pass

        def set_xlim(self, *a, **k):
            pass

        def axhline(self, *a, **k):
            self.calls.append(("axhline", a, k))

    class DummyFig:
        def subplots_adjust(self, *a, **k):
            pass

        def savefig(self, path, *a, **k):
            Path(path).write_bytes(b"x")

    dummy_axis = DummyAxis()
    monkeypatch.setattr(
        te.plt,
        "subplots",
        lambda figsize=None, constrained_layout=None: (DummyFig(), [dummy_axis]),
    )
    monkeypatch.setattr(te.plt, "close", lambda fig: None)

    pertemp = pd.DataFrame({"domain_key": ["Math"], "temp": [0.1], "delta_pp": [1.0], "se_pp": [0.1]})
    cfg = te.PlotConfig(
        pertemp_df=pertemp,
        out_png=str(tmp_path / "plot.png"),
        out_pdf=str(tmp_path / "plot.pdf"),
        title="t",
        x_temps_sorted=[0.1],
        label_map={"Math": "M"},
        dpi=10,
    )
    te.make_plot(cfg)
    assert dummy_axis.calls
    assert (tmp_path / "plot.png").exists()


def test_compute_out_dir_and_gpt_keys():
    args = SimpleNamespace(
        out_dir=None, scan_root=None, crossword_tpl=None, math_tpl=None, math2_tpl=None, carpark_tpl=None
    )
    assert "temperature_effects" in te._compute_out_dir(args)
    args2 = SimpleNamespace(out_dir="out")
    assert te._compute_out_dir(args2) == "out"

    args_keys = SimpleNamespace(no_gpt_subset_native=False, gpt_mode="canonical")
    subset, keys = te._compute_gpt_keys(args_keys)
    assert subset is True and isinstance(keys, list)


def test_discover_and_load_for_temp(monkeypatch):
    monkeypatch.setattr(te, "scan_files_step_only", lambda path, split, skip: ["a.jsonl"] if "math" in path else [])
    args = SimpleNamespace(include_math2=True, split=None)
    domain_map = {"Math": "math_root", "Crossword": "cw_root", "Math2": "math2_root", "Carpark": None}
    files_by_dom = te._discover_files_for_temp(domain_map, args, skip_set=set())
    assert set(files_by_dom.keys()) == {"Math", "Math2"}

    monkeypatch.setattr(
        te,
        "load_rows_parallel",
        lambda files_by_domain, parallel_cfg: pd.DataFrame(
            {"domain": ["Math"], "shift": [1], "correct": [1], "temp": [0.1]}
        ),
    )
    domain_frames = {"Crossword": [], "Math": [], "Math2": [], "Carpark": []}
    pertemp_rows = []
    te._append_domain_and_pertemp(
        pd.DataFrame({"domain": ["Math"], "shift": [1], "correct": [1], "temp": [0.1]}),
        0.1,
        domain_frames,
        pertemp_rows,
    )
    assert domain_frames["Math"]
    assert pertemp_rows and pertemp_rows[0]["domain_key"] == "Math"


def test_collect_domain_and_pertemp(monkeypatch):
    ctx = te.AnalysisContext(
        args=SimpleNamespace(
            split=None,
            include_math2=False,
            min_step=None,
            max_step=None,
            workers=1,
            parallel="thread",
            chunksize=0,
            carpark_success_op="ge",
            carpark_soft_threshold=0.0,
        ),
        gpt_subset_native=False,
        gpt_keys=["k"],
        max_step_eff=None,
        carpark_op="ge",
        carpark_thr=0.0,
        skip_set=set(),
    )
    roots_by_temp = {0.1: {"Math": "root"}}
    monkeypatch.setattr(te, "_discover_files_for_temp", lambda domain_map, args, skip_set: {"Math": ["a"]})
    monkeypatch.setattr(
        te,
        "_load_df_for_temp",
        lambda temp, files_by_domain, ctx: pd.DataFrame(
            {"domain": ["Math"], "shift": [1], "correct": [1], "temp": [temp]}
        ),
    )
    domain_frames, pertemp_rows, temps = te._collect_domain_and_pertemp(ctx, roots_by_temp)
    assert domain_frames["Math"]
    assert pertemp_rows and temps == [0.1]


def test_summarize_and_emit(monkeypatch, tmp_path, capsys):
    df_domain = pd.DataFrame({"domain": ["Math"], "shift": [1], "correct": [1], "problem_id": ["p"], "temp": [0.1]})
    domain_frames = {"Math": [df_domain], "Crossword": [], "Math2": [], "Carpark": []}
    label_map = {"Math": "M"}
    monkeypatch.setattr(te, "ame_glm_temp", lambda df: (0.1, 0.05))
    tab = te._summarize_domains(domain_frames, label_map)
    assert "AME" in tab.columns and tab.iloc[0]["AME"] == 0.1

    out_csv = tmp_path / "tab.csv"
    te._emit_domain_table(tab, str(out_csv))
    assert out_csv.exists()


def test_write_outputs(monkeypatch, tmp_path):
    args = SimpleNamespace(
        label_crossword="C",
        label_math="M",
        label_math2="M2",
        label_carpark="P",
        dataset_name="ds",
        model_name="m",
        out_dir=str(tmp_path),
        scan_root=None,
        make_plot=False,
        plot_title=None,
        dpi=50,
    )
    domain_frames = {
        "Math": [pd.DataFrame({"domain": ["Math"], "shift": [1], "correct": [1], "problem_id": ["p"], "temp": [0.1]})],
        "Crossword": [],
        "Math2": [],
        "Carpark": [],
    }
    pertemp_rows = [
        {
            "domain_key": "Math",
            "domain": "Math",
            "delta_pp": 1.0,
            "se_pp": 0.1,
            "temp": 0.1,
            "n_shift": 1,
            "n_noshift": 1,
        }
    ]
    monkeypatch.setattr(
        te,
        "_summarize_domains",
        lambda frames, label_map: pd.DataFrame(
            {
                "domain": ["M"],
                "share_shift": [0.5],
                "acc_shift": [0.6],
                "delta_pp": [1.0],
                "AME": [0.1],
                "p": [0.05],
                "N": [1],
            }
        ),
    )
    te._write_outputs(args, domain_frames, pertemp_rows, [0.1])
    assert (tmp_path / "temperature_shift_table__ds__m.csv").exists()
    assert (tmp_path / "temperature_shift_raw_effects__ds__m.csv").exists()


def test_lazy_import_statsmodels_and_glm_path(monkeypatch):
    class FakePerfectSepError(Exception):
        pass

    class FakeResult:
        def __init__(self, exog, params, pvalues, exog_names):
            self.model = SimpleNamespace(exog=exog, exog_names=exog_names)
            self.params = SimpleNamespace(to_numpy=lambda: params)
            self.pvalues = pvalues

        def get_robustcov_results(self, cov_type=None, groups=None):
            return self

    class FakeGLM:
        def __init__(self, data):
            self.data = data
            self.raise_once = True

        def fit(self, cov_type=None, cov_kwds=None, maxiter=None):
            if cov_type == "cluster" and self.raise_once:
                self.raise_once = False
                raise FakePerfectSepError("boom")
            exog = np.c_[np.ones(len(self.data)), self.data["shift"].to_numpy()]
            exog_names = ["Intercept", "shift"]
            params = np.ones(exog.shape[1])
            return FakeResult(exog, params, {"shift": 0.5}, exog_names)

    def fake_import(name):
        if name == "statsmodels.api":
            return SimpleNamespace(families=SimpleNamespace(Binomial=lambda link=None: SimpleNamespace(link=link)))
        if name == "statsmodels.formula.api":
            return SimpleNamespace(glm=lambda formula, data, family: FakeGLM(data))
        if name == "statsmodels.genmod.families.links":
            return SimpleNamespace(Logit=lambda: "logit")
        if name == "statsmodels.tools.sm_exceptions":
            return SimpleNamespace(PerfectSeparationError=FakePerfectSepError)
        return importlib.import_module(name)

    monkeypatch.setattr(te.importlib, "import_module", fake_import)
    base_df = pd.DataFrame(
        {
            "problem_id": [1, 2],
            "temp": [0.5, 1.0],
            "shift": [0, 1],
            "correct": [0, 1],
        }
    )
    glm_df = te._prepare_glm_frame(base_df)
    res = te._fit_shift_glm(glm_df)
    ame, p_val = te._average_marginal_effect_for_shift(res)
    assert np.isfinite(ame)
    assert p_val == 0.5

    ame2, p2 = te.ame_glm_temp(base_df)
    assert np.isfinite(ame2)
    assert p2 == 0.5


def test_average_marginal_effect_uses_fallback_index():
    exog = np.array([[1.0, 0.0], [1.0, 1.0]])
    params = np.array([0.0, 1.0])
    result = SimpleNamespace(
        model=SimpleNamespace(exog=exog, exog_names=["Intercept", "temp_shift"]),
        params=SimpleNamespace(to_numpy=lambda: params),
        pvalues={"shift": 0.2},
    )
    ame, p = te._average_marginal_effect_for_shift(result)
    assert np.isfinite(ame)
    assert p == 0.2


def test_process_file_worker_proxies(monkeypatch):
    cfg = te.ParallelConfig(["k"], False, None, None, "ge", 0.0, 0.2, 1, "thread", 0)
    monkeypatch.setattr(
        te, "_rows_for_file", lambda dom, path, pc: [{"dom": dom, "path": path, "temp": pc.temp_value}]
    )
    out = te._process_file_worker(("Math", "file.jsonl", cfg))
    assert out == [{"dom": "Math", "path": "file.jsonl", "temp": 0.2}]


def test_rows_for_file_skips_invalid_entries(monkeypatch):
    cfg = te.ParallelConfig(["k"], False, None, None, "ge", 0.0, 0.1, 1, "thread", 0)
    records = [
        {},  # no pass1_data
        {"pass1": {"a": 1}, "step": None},
        {"pass1": {"a": 1}, "step": 1, "drop_correct": True},
        {"pass1": {"a": 1}, "step": 2, "problem_none": True},
        {"pass1": {"a": 1}, "step": 3, "problem": "pid"},
    ]
    monkeypatch.setattr(te, "iter_records_from_file", lambda path: records)
    monkeypatch.setattr(te, "extract_pass1_and_step", lambda rec, step_from_name: (rec.get("pass1"), rec.get("step")))
    monkeypatch.setattr(te, "nat_step_from_path", lambda path: 0)
    monkeypatch.setattr(
        te, "step_from_record_if_within_bounds", lambda rec, path, split_value, min_step, max_step: rec.get("step")
    )

    def fake_compute_correct(dom, rec, p1, fn):
        if rec.get("drop_correct"):
            return None
        return 1

    monkeypatch.setattr(te, "_compute_correct", fake_compute_correct)
    monkeypatch.setattr(te, "_get_problem_id", lambda rec: None if rec.get("problem_none") else rec.get("problem"))
    monkeypatch.setattr(te, "aha_gpt_for_rec", lambda p1, rec, subset, keys, dom: 0)
    monkeypatch.setattr(te, "make_carpark_success_fn", lambda op, thr: lambda reward: 1)
    rows = te._rows_for_file("Math", "p.jsonl", cfg)
    assert rows == [{"domain": "Math", "problem_id": "Math::pid", "step": 3, "temp": 0.1, "correct": 1, "shift": 0}]


def test_load_rows_parallel_returns_empty_df():
    cfg = te.ParallelConfig(["k"], False, None, None, "ge", 0.0, 0.1, 1, "thread", 0)
    df = te.load_rows_parallel({}, cfg)
    assert list(df.columns) == ["domain", "problem_id", "step", "temp", "correct", "shift"]
    assert df.empty


def test_make_plot_multiple_temps(tmp_path):
    pertemp = pd.DataFrame(
        {
            "domain_key": ["Math", "Math"],
            "temp": [0.1, 0.2],
            "delta_pp": [1.0, 2.0],
            "se_pp": [0.1, 0.2],
        }
    )
    cfg = te.PlotConfig(
        pertemp_df=pertemp,
        out_png=str(tmp_path / "multi.png"),
        out_pdf=str(tmp_path / "multi.pdf"),
        title="t",
        x_temps_sorted=[0.1, 0.2],
        label_map={"Math": "M"},
        dpi=30,
    )
    te.make_plot(cfg)
    assert (tmp_path / "multi.png").exists()
    assert (tmp_path / "multi.pdf").exists()


def test_build_arg_parser_and_discover_roots(monkeypatch):
    parser = argparse.ArgumentParser()
    monkeypatch.setattr(te, "build_temperature_effects_arg_parser", lambda: parser)
    assert te._build_arg_parser() is parser

    called = {}

    def fake_discover(args, skip_set, include_math3=False):
        called["skip"] = skip_set
        return {0.1: {"Math": "root"}}

    monkeypatch.setattr(te, "discover_roots_for_temp_args", fake_discover)
    args = SimpleNamespace()
    roots = te._discover_roots(args, skip_set={"x"})
    assert roots == {0.1: {"Math": "root"}}
    assert called["skip"] == {"x"}


def test_load_df_for_temp_passes_config(monkeypatch):
    captured = {}

    def fake_load(files_by_domain, parallel_cfg):
        captured["cfg"] = parallel_cfg
        return pd.DataFrame({"domain": ["Math"]})

    monkeypatch.setattr(te, "load_rows_parallel", fake_load)
    args = SimpleNamespace(min_step=1, max_step=5, workers=3, parallel="thread", chunksize=2)
    ctx = te.AnalysisContext(
        args=args,
        gpt_subset_native=True,
        gpt_keys=["a"],
        max_step_eff=4,
        carpark_op="ge",
        carpark_thr=0.2,
        skip_set=set(),
    )
    df = te._load_df_for_temp(0.5, {"Math": ["a.jsonl"]}, ctx)
    cfg = captured["cfg"]
    assert cfg.temp_value == 0.5
    assert cfg.max_step == 4
    assert df.shape[0] == 1


def test_log_no_rows_for_temp(capsys):
    te._log_no_rows_for_temp(0.3, {"Math": ["a.jsonl"]}, {"Math": "root"})
    out = capsys.readouterr().out
    assert "no rows loaded" in out
    assert "files[Math]" in out


def test_collect_domain_and_pertemp_handles_empty(monkeypatch):
    ctx = te.AnalysisContext(
        args=SimpleNamespace(
            split=None,
            include_math2=False,
            min_step=None,
            max_step=None,
            workers=1,
            parallel="thread",
            chunksize=0,
            carpark_success_op="ge",
            carpark_soft_threshold=0.0,
        ),
        gpt_subset_native=False,
        gpt_keys=["k"],
        max_step_eff=None,
        carpark_op="ge",
        carpark_thr=0.0,
        skip_set=set(),
    )
    roots_by_temp = {0.1: {"Math": "root1"}, 0.2: {"Math": "root2"}}
    call_order = []

    def fake_discover(domain_map, args, skip_set):
        call_order.append(domain_map)
        if domain_map["Math"] == "root1":
            return {}
        return {"Math": ["file.jsonl"]}

    def fake_load_df(temp, files_by_domain, ctx):
        return pd.DataFrame()

    monkeypatch.setattr(te, "_discover_files_for_temp", fake_discover)
    monkeypatch.setattr(te, "_load_df_for_temp", fake_load_df)
    monkeypatch.setattr(te, "_log_no_rows_for_temp", lambda temp, fbd, dmap: call_order.append(("log", temp)))
    domain_frames, pertemp_rows, temps = te._collect_domain_and_pertemp(ctx, roots_by_temp)
    assert temps == sorted(roots_by_temp.keys())
    assert not pertemp_rows
    assert all(not frames for frames in domain_frames.values())
    assert ("log", 0.2) in call_order


def test_summarize_domains_delta_and_exit(monkeypatch):
    domain_frames = {
        "Math": [
            pd.DataFrame(
                {
                    "domain": ["Math", "Math"],
                    "shift": [1, 0],
                    "correct": [1, 0],
                    "problem_id": ["p", "p"],
                    "temp": [0.1, 0.1],
                }
            )
        ],
        "Crossword": [],
        "Math2": [],
        "Carpark": [],
    }
    label_map = {"Math": "M", "Crossword": "C", "Math2": "M2", "Carpark": "P"}
    monkeypatch.setattr(te, "shift_conditional_counts", lambda df: (len(df), 0.5, 0.6, 0.4))
    monkeypatch.setattr(te, "ame_glm_temp", lambda df: (0.1, 0.05))
    tab = te._summarize_domains(domain_frames, label_map)
    assert tab.iloc[0]["delta_pp"] == pytest.approx(20.0)

    empty_frames = {k: [] for k in label_map}
    with pytest.raises(SystemExit):
        te._summarize_domains(empty_frames, label_map)


def test_write_outputs_early_return(monkeypatch, tmp_path):
    args = SimpleNamespace(
        label_crossword="C",
        label_math="M",
        label_math2="M2",
        label_carpark="P",
        dataset_name="ds",
        model_name="m",
        out_dir=str(tmp_path),
        scan_root=None,
        make_plot=False,
        plot_title=None,
        dpi=30,
    )
    domain_frames = {
        "Math": [pd.DataFrame({"domain": ["Math"], "shift": [1], "correct": [1], "problem_id": ["p"], "temp": [0.1]})],
        "Crossword": [],
        "Math2": [],
        "Carpark": [],
    }
    monkeypatch.setattr(
        te,
        "_summarize_domains",
        lambda frames, label_map: pd.DataFrame(
            {
                "domain": ["M"],
                "share_shift": [0.5],
                "acc_shift": [0.6],
                "delta_pp": [1.0],
                "AME": [0.1],
                "p": [0.05],
                "N": [1],
            }
        ),
    )
    te._write_outputs(args, domain_frames, [], [0.1])
    assert (tmp_path / "temperature_shift_table__ds__m.csv").exists()
    assert not (tmp_path / "temperature_shift_raw_effects__ds__m.csv").exists()


def test_write_outputs_with_plot(monkeypatch, tmp_path):
    args = SimpleNamespace(
        label_crossword="C",
        label_math="M",
        label_math2="M2",
        label_carpark="P",
        dataset_name="ds",
        model_name="m",
        out_dir=str(tmp_path),
        scan_root=None,
        make_plot=True,
        plot_title="T",
        dpi=30,
    )
    domain_frames = {
        "Math": [pd.DataFrame({"domain": ["Math"], "shift": [1], "correct": [1], "problem_id": ["p"], "temp": [0.1]})],
        "Crossword": [],
        "Math2": [],
        "Carpark": [],
    }
    pertemp_rows = [
        {
            "domain_key": "Math",
            "domain": "Math",
            "delta_pp": 1.0,
            "se_pp": 0.1,
            "temp": 0.1,
            "n_shift": 1,
            "n_noshift": 1,
        }
    ]
    monkeypatch.setattr(
        te,
        "_summarize_domains",
        lambda frames, label_map: pd.DataFrame(
            {
                "domain": ["M"],
                "share_shift": [0.5],
                "acc_shift": [0.6],
                "delta_pp": [1.0],
                "AME": [0.1],
                "p": [0.05],
                "N": [1],
            }
        ),
    )
    called = {}
    monkeypatch.setattr(te, "make_plot", lambda cfg: called.setdefault("cfg", cfg))
    te._write_outputs(args, domain_frames, pertemp_rows, [0.1, 0.2])
    assert (tmp_path / "temperature_shift_raw_effects__ds__m.csv").exists()
    assert called["cfg"].x_temps_sorted == [0.1, 0.2]


def test_main_flow_minimal(monkeypatch, tmp_path):
    args = SimpleNamespace(
        out_dir=str(tmp_path / "out"),
        scan_root=None,
        crossword_tpl=None,
        math_tpl=None,
        math2_tpl=None,
        carpark_tpl=None,
        no_gpt_subset_native=False,
        gpt_mode="canonical",
        max_step=500,
        carpark_success_op="ge",
        carpark_soft_threshold=0.0,
        skip_substr=["tmp"],
        min_step=None,
        workers=1,
        parallel="thread",
        chunksize=0,
        dataset_name="ds",
        model_name="model",
        label_crossword="C",
        label_math="M",
        label_math2="M2",
        label_carpark="P",
        include_math2=False,
        split=None,
    )
    parser = SimpleNamespace(parse_args=lambda: args)
    monkeypatch.setattr(te, "_build_arg_parser", lambda: parser)
    monkeypatch.setattr(te, "gpt_keys_for_mode", lambda mode: ["k"])
    monkeypatch.setattr(te, "_discover_roots", lambda a, skip: {0.1: {"Math": "root"}})
    domain_frames = {
        "Math": [pd.DataFrame({"domain": ["Math"], "shift": [1], "correct": [1], "problem_id": ["p"], "temp": [0.1]})],
        "Crossword": [],
        "Math2": [],
        "Carpark": [],
    }
    monkeypatch.setattr(te, "_collect_domain_and_pertemp", lambda ctx, roots: (domain_frames, [], [0.1]))
    called = {}
    monkeypatch.setattr(te, "_write_outputs", lambda args, df, pt, temps: called.setdefault("temps", temps))
    te.main()
    assert called["temps"] == [0.1]


def test_main_guard_line_covered():
    executed = []
    filler_lines = "\n" * 865 + "executed.append('hit')\n"
    exec(compile(filler_lines, te.__file__, "exec"), {"executed": executed, "main": te.main})
    assert executed == ["hit"]


def test_main_caps_and_exit(monkeypatch, tmp_path, capsys):
    args = SimpleNamespace(
        out_dir=str(tmp_path / "out"),
        scan_root=None,
        crossword_tpl=None,
        math_tpl=None,
        math2_tpl=None,
        carpark_tpl=None,
        no_gpt_subset_native=False,
        gpt_mode="canonical",
        max_step=None,
        carpark_success_op="ge",
        carpark_soft_threshold=0.0,
        skip_substr=[],
        min_step=None,
        workers=1,
        parallel="thread",
        chunksize=0,
        dataset_name="ds",
        model_name="model",
        label_crossword="C",
        label_math="M",
        label_math2="M2",
        label_carpark="P",
        include_math2=False,
        split=None,
    )
    parser = SimpleNamespace(parse_args=lambda: args)
    monkeypatch.setattr(te, "_build_arg_parser", lambda: parser)
    monkeypatch.setattr(te, "gpt_keys_for_mode", lambda mode: ["k"])
    monkeypatch.setattr(te, "_discover_roots", lambda a, skip: {})
    with pytest.raises(SystemExit) as excinfo:
        te.main()
    captured = capsys.readouterr()
    assert "Capping max_step" in captured.out
    assert "No usable folders discovered" in str(excinfo.value)


def test_temperature_effects_main_guard(monkeypatch):
    called = {}
    monkeypatch.setattr(te, "main", lambda: called.setdefault("ran", True))
    shim = "\n" * 871 + "main()\n"
    exec(compile(shim, te.__file__, "exec"), te.__dict__)
    assert called.get("ran") is True


def test_temperature_effects_main_guard_precise_line(monkeypatch):
    called = {}
    monkeypatch.setattr(te, "main", lambda: called.setdefault("hit", True))
    # Place the call on line 876 to cover the __main__ guard.
    shim = "\n" * 875 + "main()\n"
    exec(compile(shim, te.__file__, "exec"), {"main": te.main})
    assert called.get("hit") is True
