import sys
import types

import pandas as pd


# Prefer real matplotlib; fall back to a minimal stub if missing.
try:  # pragma: no cover
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover - minimal stub

    class _StubAxis:
        def __init__(self):
            self.texts = []

        def set_title(self, *_args, **_kwargs):
            pass

        def set_xlabel(self, *_args, **_kwargs):
            pass

        def set_ylabel(self, *_args, **_kwargs):
            pass

        def set_ylim(self, *_args, **_kwargs):
            pass

        def text(self, *args, **kwargs):
            self.texts.append((args, kwargs))

        def bar(self, *args, **kwargs):
            return []

        def set_xticks(self, *args, **kwargs):
            pass

        def grid(self, *args, **kwargs):
            pass

        def get_legend_handles_labels(self):
            return [], []

        def plot(self, *args, **kwargs):
            return []

        def errorbar(self, *args, **kwargs):
            return []

    class _StubFig:
        def add_subplot(self, *_args, **_kwargs):
            return _StubAxis()

        def tight_layout(self):
            pass

        def savefig(self, *args, **kwargs):
            pass

    class _StubPlt:
        def subplots(self, *args, **kwargs):
            nrows = kwargs.get("nrows", None) or (args[0] if args else 1)
            ncols = kwargs.get("ncols", None) or (args[1] if len(args) > 1 else 1)
            single = nrows == 1 and ncols == 1
            axes = _StubAxis()
            if not single:
                axes = tuple(_StubAxis() for _ in range(int(nrows * ncols)))
            return _StubFig(), axes

        def close(self, *_args, **_kwargs):
            pass

        def figure(self, *args, **kwargs):
            return _StubFig()

    plt = _StubPlt()  # type: ignore
    sys.modules["matplotlib"] = types.SimpleNamespace(use=lambda *a, **k: None, rcParams={})
    sys.modules["matplotlib.pyplot"] = plt

import src.analysis.graph_2 as g2


def test_make_bins_quantile_deduplicates_edges():
    series = pd.Series([1, 1, 1, 1])
    edges, centers = g2.make_bins(series, n_bins=3, mode="quantile")
    assert len(edges) >= 2
    assert centers.size == len(edges) - 1


def test_process_single_record_filters(monkeypatch):
    cfg = g2.RowLoadConfig(gpt_keys=["shift"], gpt_subset_native=True, min_step=None, max_step=None)
    ctx = g2.RecordContext(domain="Math", path="/tmp/p.jsonl", step_from_name=None)
    monkeypatch.setattr(g2, "aha_gpt_for_rec", lambda pass1, rec, subset, keys, domain: 1)
    monkeypatch.setattr(g2, "get_problem_id", lambda rec: "pid")

    assert g2._process_single_record(ctx, {}, cfg, carpark_success_fn=lambda *_a, **_k: None) is None

    rec = {"step": 0, "pass1": {"is_correct_pred": True, "entropy_think": 0.1, "entropy_answer": 0.2}}
    row = g2._process_single_record(ctx, rec, cfg, carpark_success_fn=lambda *_a, **_k: 1)
    assert row["correct"] == 1 and row["aha"] == 1


def test_aggregate_bins_empty_and_filter():
    empty = pd.DataFrame({"entropy_think": []})
    out = g2.aggregate_bins(empty, "entropy_think", n_bins=3, mode="uniform", min_per_bar=1)
    assert out.empty

    df = pd.DataFrame({"entropy_think": [0.0, 0.1, 0.2, 0.3], "correct": [1, 1, 0, 0], "aha": [0, 0, 1, 1]})
    out2 = g2.aggregate_bins(df, "entropy_think", n_bins=2, mode="uniform", min_per_bar=3)
    assert out2.empty


def test_compute_domain_stats_and_plots(monkeypatch, tmp_path):
    monkeypatch.setattr(g2, "plot_domain_panels", lambda *a, **k: None)
    df = pd.DataFrame(
        {
            "domain": ["Crossword", "Crossword"],
            "entropy_think": [0.1, 0.2],
            "entropy_answer": [0.1, 0.3],
            "entropy_joint": [0.2, 0.4],
            "correct": [1, 0],
            "aha": [0, 1],
        }
    )
    args = types.SimpleNamespace(
        dataset_name="D", model_name="M", n_bins=2, bucket_mode="uniform", min_per_bar=1, dpi=100
    )
    g2._compute_domain_stats_and_plots(df, args, str(tmp_path))
    csvs = list(tmp_path.glob("*.csv"))
    assert csvs


def test_main_executes_with_stubs(monkeypatch, tmp_path):
    monkeypatch.setattr(g2, "build_files_by_domain_for_args", lambda args: ({"Math": ["f1"]}, str(tmp_path)))
    monkeypatch.setattr(
        g2,
        "load_rows",
        lambda files_by_domain, cfg, carpark_success_fn: pd.DataFrame(
            {
                "domain": ["Math"],
                "entropy_think": [0.1],
                "entropy_answer": [0.2],
                "entropy_joint": [0.3],
                "correct": [1],
                "aha": [0],
            }
        ),
    )
    monkeypatch.setattr(g2, "plot_domain_panels", lambda *a, **k: None)
    monkeypatch.setattr(g2, "_compute_domain_stats_and_plots", lambda df, args, out_dir: None)
    args = types.SimpleNamespace(
        out_dir=str(tmp_path),
        dataset_name="D",
        model_name="M",
        gpt_mode="canonical",
        no_gpt_subset_native=False,
        min_step=None,
        max_step=1000,
        carpark_success_op="gt",
        carpark_soft_threshold=0.0,
        n_bins=2,
        bucket_mode="uniform",
        min_per_bar=1,
        dpi=100,
    )
    monkeypatch.setattr(g2, "_build_arg_parser", lambda: types.SimpleNamespace(parse_args=lambda: args))
    g2.main()
