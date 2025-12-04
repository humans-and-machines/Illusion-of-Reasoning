import runpy
import sys
import types

import numpy as np
import pandas as pd
import pytest

import src.analysis.graph_2 as g2


def test_extract_step_for_record_bounds_and_invalid():
    cfg = g2.RowLoadConfig(gpt_keys=[], gpt_subset_native=False, min_step=2, max_step=4)
    ctx = g2.RecordContext(domain="math", path="p", step_from_name=5)
    assert g2._extract_step_for_record({"step": "3"}, ctx, cfg) == 3
    assert g2._extract_step_for_record({"step": "bad"}, ctx, cfg) is None
    assert g2._extract_step_for_record({"other": 1}, ctx, cfg) is None  # below min_step via fallback
    cfg2 = g2.RowLoadConfig(gpt_keys=[], gpt_subset_native=False, min_step=None, max_step=None)
    assert g2._extract_step_for_record({"other": None}, ctx, cfg2) == 5


def test_extract_step_respects_min_step_boundary():
    cfg = g2.RowLoadConfig(gpt_keys=[], gpt_subset_native=False, min_step=10, max_step=None)
    ctx = g2.RecordContext(domain="math", path="p", step_from_name=None)
    assert g2._extract_step_for_record({"step": 5}, ctx, cfg) is None


def test_compute_success_and_entropies(monkeypatch):
    monkeypatch.setattr(
        g2,
        "carpark_success_from_soft_reward",
        lambda rec, *_: int(rec["soft_reward"] > 0.5) if rec["soft_reward"] is not None else None,
    )
    carpark_cmp = g2._make_carpark_success_fn("gt", 0.5)
    ctx_car = g2.RecordContext(domain="carpark", path="p", step_from_name=None)
    assert g2._compute_success_for_record(ctx_car, {}, {"soft_reward": 0.6}, carpark_cmp) == 1
    assert g2._compute_success_for_record(ctx_car, {}, {"soft_reward": None}, carpark_cmp) is None

    ctx_math = g2.RecordContext(domain="math", path="p", step_from_name=None)
    assert g2._compute_success_for_record(ctx_math, {}, {"is_correct_pred": 1}, carpark_cmp) == 1
    assert g2._compute_success_for_record(ctx_math, {}, {"is_correct_pred": None}, carpark_cmp) is None

    think, answer, joint = g2._compute_entropies({"entropy_think": 1.0, "entropy_answer": 3.0, "entropy": None})
    assert joint == pytest.approx(2.0)
    assert g2._compute_entropies({}) == (None, None, None)


def test_process_single_record_filters(monkeypatch):
    monkeypatch.setattr(g2, "aha_gpt_for_rec", lambda *_a, **_k: 1)
    cfg = g2.RowLoadConfig(gpt_keys=[], gpt_subset_native=False, min_step=None, max_step=None)
    ctx = g2.RecordContext(domain="math", path="p", step_from_name=None)

    assert g2._process_single_record(ctx, "not_dict", cfg, carpark_success_fn=lambda *_: 1) is None
    assert g2._process_single_record(ctx, {"pass1": "bad"}, cfg, carpark_success_fn=lambda *_: 1) is None

    rec = {
        "step": 1,
        "pass1": {"is_correct_pred": 1, "entropy_think": 0.1},
        "problem": "p1",
    }
    row = g2._process_single_record(ctx, rec, cfg, carpark_success_fn=lambda *_: 1)
    assert row["aha"] == 1 and row["entropy_think"] == pytest.approx(0.1)


def test_process_single_record_drops_when_success_missing(monkeypatch):
    monkeypatch.setattr(g2, "aha_gpt_for_rec", lambda *_a, **_k: 0)
    cfg = g2.RowLoadConfig(gpt_keys=[], gpt_subset_native=False, min_step=None, max_step=None)
    ctx = g2.RecordContext(domain="math", path="p", step_from_name=None)
    rec = {"step": 1, "pass1": {"is_correct_pred": None, "entropy_think": 0.2}}
    assert g2._process_single_record(ctx, rec, cfg, carpark_success_fn=lambda *_: 1) is None


def test_make_bins_modes():
    series = pd.Series([1.0, 2.0, 3.0])
    edges, centers = g2.make_bins(series, n_bins=2, mode="uniform")
    assert len(edges) == 3 and len(centers) == 2

    edges_q, _ = g2.make_bins(series, n_bins=2, mode="quantile")
    assert edges_q.min() >= series.min()

    edges_const, _ = g2.make_bins(pd.Series([5.0, 5.0]), n_bins=2, mode="uniform")
    assert edges_const[0] < edges_const[1]


def test_aggregate_bins_support_filtering():
    df = pd.DataFrame(
        {
            "entropy_think": [0.1, 0.2],
            "correct": [1, 0],
            "aha": [0, 1],
        }
    )
    empty = g2.aggregate_bins(df, "entropy_think", n_bins=1, mode="uniform", min_per_bar=2)
    assert empty.empty

    enough = g2.aggregate_bins(df, "entropy_think", n_bins=1, mode="uniform", min_per_bar=1)
    assert not enough.empty
    assert {"acc_aha", "acc_noaha"}.issubset(enough.columns)


def test_aggregate_bins_returns_empty_when_bins_have_no_support(monkeypatch):
    df = pd.DataFrame(
        {
            "entropy_think": [0.1, 0.2],
            "correct": [1, 0],
            "aha": [0, 1],
        }
    )
    monkeypatch.setattr(g2.pd, "cut", lambda *args, **kwargs: pd.Series([pd.NA] * len(df)))
    empty = g2.aggregate_bins(df, "entropy_think", n_bins=1, mode="uniform", min_per_bar=1)
    assert empty.empty


def test_aggregate_bins_handles_empty_pivots(monkeypatch):
    df = pd.DataFrame(
        {
            "entropy_think": [0.1, 0.2],
            "correct": [1, 0],
            "aha": [0, 1],
        }
    )
    monkeypatch.setattr(g2, "_bin_group_stats", lambda *_a, **_k: (pd.DataFrame(), pd.DataFrame()))
    empty = g2.aggregate_bins(df, "entropy_think", n_bins=1, mode="uniform", min_per_bar=1)
    assert empty.empty


def test_build_arg_parser_defaults():
    args = g2._build_arg_parser().parse_args([])
    assert args.n_bins == 10
    assert args.bucket_mode == "uniform"
    assert args.max_step == 1000
    assert args.carpark_success_op == "gt"
    assert args.carpark_soft_threshold == 0.0


def test_add_row_legend_handles_empty_and_present():
    fig, ax = g2.plt.subplots(1, 1)
    g2._add_row_legend([ax])  # no handles
    ax.plot([0, 1], [0, 1], label="line")
    g2._add_row_legend([ax])
    assert fig.legends  # legend added
    g2.plt.close(fig)


def test_plot_domain_panels_creates_files(tmp_path):
    stats = pd.DataFrame(
        {
            "bin_left": [0.0, 0.5],
            "bin_right": [0.5, 1.0],
            "acc_noaha": [0.2, 0.8],
            "acc_aha": [0.4, 0.6],
        }
    )
    stats_by_metric = {"entropy_think": stats, "entropy_answer": pd.DataFrame(), "entropy_joint": pd.DataFrame()}
    out_png = tmp_path / "fig.png"
    g2.plot_domain_panels("math", stats_by_metric, str(out_png), dpi=50, title_prefix=None)
    assert out_png.exists() and out_png.with_suffix(".pdf").exists()


def test_compute_domain_stats_and_plots_builds_csv(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(g2, "plot_domain_panels", lambda *args, **kwargs: calls.append(args[0]))

    df = pd.DataFrame(
        {
            "domain": ["math", "math"],
            "entropy_think": [0.1, 0.2],
            "entropy_answer": [0.1, 0.2],
            "entropy_joint": [0.1, 0.2],
            "correct": [1, 0],
            "aha": [0, 1],
        }
    )
    args = types.SimpleNamespace(
        dataset_name="DS",
        model_name="M",
        n_bins=1,
        bucket_mode="uniform",
        min_per_bar=1,
        dpi=50,
    )
    g2._compute_domain_stats_and_plots(df, args, str(tmp_path))
    csvs = list(tmp_path.glob("entropy_hist__DS__M.csv"))
    assert calls == ["math"]
    assert csvs and csvs[0].exists()


def test_compute_domain_stats_handles_empty_metric_and_inserts_metric_column(monkeypatch, tmp_path):
    monkeypatch.setattr(g2, "plot_domain_panels", lambda *args, **kwargs: None)

    def fake_aggregate_bins(*_a, **_k):
        return pd.DataFrame(
            {
                "domain": ["math"],
                "bin_left": [0.0],
                "bin_right": [1.0],
                "acc_noaha": [0.5],
                "acc_aha": [0.5],
            }
        )

    monkeypatch.setattr(g2, "aggregate_bins", fake_aggregate_bins)

    df = pd.DataFrame(
        {
            "domain": ["math"],
            "entropy_think": [np.nan],
            "entropy_answer": [0.3],
            "entropy_joint": [0.4],
            "correct": [1],
            "aha": [0],
        }
    )
    args = types.SimpleNamespace(
        dataset_name="DS",
        model_name="M",
        n_bins=1,
        bucket_mode="uniform",
        min_per_bar=1,
        dpi=50,
    )

    g2._compute_domain_stats_and_plots(df, args, str(tmp_path))

    expected_csv = tmp_path / "entropy_hist__DS__M.csv"
    assert expected_csv.exists()
    table = pd.read_csv(expected_csv)
    assert "metric" in table.columns


def test_compute_domain_stats_handles_no_rows(tmp_path):
    args = types.SimpleNamespace(
        dataset_name="DS",
        model_name="M",
        n_bins=1,
        bucket_mode="uniform",
        min_per_bar=1,
        dpi=50,
    )
    # Empty dataframe should short-circuit without errors or CSV.
    g2._compute_domain_stats_and_plots(pd.DataFrame(columns=["domain"]), args, str(tmp_path))
    assert not list(tmp_path.glob("*.csv"))


def test_main_exits_when_no_rows(monkeypatch, tmp_path):
    class FakeParser:
        def parse_args(self_inner):
            return types.SimpleNamespace(
                gpt_mode="all",
                no_gpt_subset_native=False,
                min_step=None,
                max_step=10,
                carpark_success_op="gt",
                carpark_soft_threshold=0.0,
                out_dir=str(tmp_path),
                dataset_name="DS",
                model_name="M",
                n_bins=1,
                bucket_mode="uniform",
                min_per_bar=1,
                dpi=50,
            )

    monkeypatch.setattr(g2, "_build_arg_parser", lambda: FakeParser())
    monkeypatch.setattr(g2, "build_files_by_domain_for_args", lambda args: ({"dom": ["p"]}, str(tmp_path)))
    monkeypatch.setattr(g2, "gpt_keys_for_mode", lambda mode: ["key"])
    monkeypatch.setattr(g2, "load_rows", lambda *a, **k: pd.DataFrame())
    with pytest.raises(SystemExit):
        g2.main()


def test_main_invokes_pipeline(monkeypatch, tmp_path):
    parsed_args = types.SimpleNamespace(
        gpt_mode="all",
        no_gpt_subset_native=False,
        min_step=None,
        max_step=10,
        carpark_success_op="gt",
        carpark_soft_threshold=0.0,
        out_dir=str(tmp_path),
        dataset_name="DS",
        model_name="M",
        n_bins=1,
        bucket_mode="uniform",
        min_per_bar=1,
        dpi=50,
    )

    class FakeParser:
        def parse_args(self_inner):
            return parsed_args

    monkeypatch.setattr(g2, "_build_arg_parser", lambda: FakeParser())
    monkeypatch.setattr(g2, "build_files_by_domain_for_args", lambda args: ({"dom": ["p"]}, str(tmp_path)))
    monkeypatch.setattr(g2, "os", types.SimpleNamespace(path=g2.os.path, makedirs=lambda *a, **k: None))
    monkeypatch.setattr(g2, "gpt_keys_for_mode", lambda mode: ["key"])
    monkeypatch.setattr(
        g2,
        "load_rows",
        lambda *_a, **_k: pd.DataFrame(
            {
                "domain": ["dom", "dom"],
                "entropy_think": [0.1, 0.2],
                "entropy_answer": [0.1, 0.2],
                "entropy_joint": [0.1, 0.2],
                "correct": [1, 0],
                "aha": [0, 1],
            }
        ),
    )
    called = []
    monkeypatch.setattr(g2, "_compute_domain_stats_and_plots", lambda df, a, out: called.append(df.shape))

    g2.main()

    assert called and called[0][0] == 2


def test_main_guard_executes_with_stubbed_dependencies(monkeypatch, tmp_path):
    stub_utils = types.ModuleType("src.analysis.utils")

    stub_utils.add_standard_domain_root_args = lambda parser: parser.add_argument(
        "results_root", nargs="?", default=None
    )
    stub_utils.add_split_and_out_dir_args = lambda parser, out_dir_help=None: parser.add_argument(
        "--out_dir", default=None
    )
    stub_utils.add_gpt_mode_arg = lambda parser: parser.add_argument("--gpt_mode", default="canonical")

    def stub_add_carpark_threshold_args(parser):
        parser.add_argument("--carpark_success_op", default="gt")
        parser.add_argument("--carpark_soft_threshold", type=float, default=0.0)

    stub_utils.add_carpark_threshold_args = stub_add_carpark_threshold_args
    stub_utils.coerce_bool = lambda value: int(bool(value)) if value is not None else None
    stub_utils.coerce_float = lambda value: float(value) if value is not None else None
    stub_utils.get_problem_id = lambda rec: rec.get("problem")
    stub_utils.gpt_keys_for_mode = lambda mode: ["k"]
    stub_utils.nat_step_from_path = lambda _path: 1

    stub_io = types.ModuleType("src.analysis.io")
    stub_io.build_files_by_domain_for_args = lambda args: ({"math": ["fake.jsonl"]}, str(tmp_path))
    stub_io.iter_records_from_file = lambda _path: iter(
        [
            {
                "step": 1,
                "pass1": {
                    "is_correct_pred": 1,
                    "entropy_think": 0.1,
                    "entropy_answer": 0.2,
                },
            }
        ]
    )

    stub_labels = types.ModuleType("src.analysis.labels")
    stub_labels.aha_gpt_for_rec = lambda *_a, **_k: 0

    stub_metrics = types.ModuleType("src.analysis.metrics")
    stub_metrics.carpark_success_from_soft_reward = lambda rec, *_: 1

    monkeypatch.setitem(sys.modules, "src.analysis.utils", stub_utils)
    monkeypatch.setitem(sys.modules, "src.analysis.io", stub_io)
    monkeypatch.setitem(sys.modules, "src.analysis.labels", stub_labels)
    monkeypatch.setitem(sys.modules, "src.analysis.metrics", stub_metrics)

    monkeypatch.setattr(sys, "argv", ["graph_2.py"])

    monkeypatch.delitem(sys.modules, "src.analysis.graph_2", raising=False)

    runpy.run_module("src.analysis.graph_2", run_name="__main__", alter_sys=True)

    out_dir = tmp_path / "entropy_histograms"
    csvs = list(out_dir.glob("entropy_hist__MIXED__Qwen2.5-1.5B.csv"))
    assert csvs and csvs[0].exists()
