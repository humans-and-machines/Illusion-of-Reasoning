from pathlib import Path
from types import SimpleNamespace

import numpy as np

import src.analysis.graph_3_impl as g3


def test_temp_match_numeric_and_string_paths():
    assert g3.temp_match("/runs/temp-0.30/file.jsonl", ["0.3"])
    assert g3.temp_match("/runs/temp-foo/file.jsonl", ["foo"])
    assert not g3.temp_match("/runs/no-temp/file.jsonl", ["0.3"])


def test_build_per_domain_accuracy_and_edges_global():
    rows = [
        {"domain": "Carpark", "entropy": 0.1, "aha": 1, "correct": 0},
        {"domain": "Math", "entropy": 0.2, "aha": 0, "correct": 1},
    ]
    per = g3._build_per_domain_accuracy(rows, g3.DOMAINS)
    assert per["Carpark"]["entropy"].tolist() == [0.1]
    assert per["Math"]["correct"].tolist() == [1]

    args = SimpleNamespace(bins=2, binning="fixed", entropy_min=None, entropy_max=None, share_bins="global")
    edges_by, overall = g3._compute_edges_for_accuracy(per, ["Carpark", "Math"], args)
    assert np.allclose(edges_by["Carpark"], overall)
    assert np.allclose(edges_by["Math"], overall)


def test_minimal_axes_and_save_all_formats(tmp_path, capsys):
    class AxisStub:
        def __init__(self):
            self.spines = {"top": SimpleNamespace(visible=True), "right": SimpleNamespace(visible=True)}
            self.grids = []

        def grid(self, *a, **k):
            self.grids.append((a, k))

        # Allow set_visible to toggle a flag.

    for spine in ["top", "right"]:
        AxisStub.spines = None

    axis = AxisStub()
    # Patch set_visible to record.
    for spine in axis.spines.values():
        spine.set_visible = lambda flag, s=spine: setattr(s, "visible", flag)

    g3.minimal_axes(axis)
    assert axis.spines["top"].visible is False and axis.spines["right"].visible is False
    assert axis.grids

    saved = []

    class FigStub:
        def savefig(self, path, **kwargs):
            saved.append(Path(path))
            Path(path).write_text("x")

    g3.save_all_formats(FigStub(), str(tmp_path / "out"), dpi=123)
    out = capsys.readouterr().out
    assert any(p.suffix == ".png" for p in saved)
    assert "[ok] wrote" in out


def test_render_counts_and_accuracy_panels_empty_and_nonempty():
    class AxisStub:
        def __init__(self):
            self.texts = []
            self.bars = []
            self.ylims = []
            self.spines = {
                "top": SimpleNamespace(set_visible=lambda flag: setattr(self, "top_vis", flag)),
                "right": SimpleNamespace(set_visible=lambda flag: setattr(self, "right_vis", flag)),
            }

        def text(self, *a, **k):
            self.texts.append((a, k))

        def bar(self, *a, **k):
            self.bars.append((a, k))

        def set_ylim(self, *a, **k):
            self.ylims.append((a, k))

        def set_ylabel(self, *_):
            return SimpleNamespace(set_multialignment=lambda *_a: None)

        def set_title(self, *_, **__):
            return None

        def set_xticks(self, *_, **__):
            return None

        def set_xticklabels(self, *_, **__):
            return None

        def grid(self, *_, **__):
            return None

    # Counts panel with no data should emit a text placeholder.
    empty_per = {"Carpark": {"entropy": np.array([]), "aha": np.array([])}}
    counts_cfg = g3.CountsFigureConfig(
        per_domain=empty_per,
        domains=["Carpark"],
        edges_by_domain={"Carpark": np.array([0.0, 1.0])},
        x_label="x",
        args=SimpleNamespace(title=None, width_in=1.0, height_in=1.0),
        color_aha="red",
    )
    axis_empty = AxisStub()
    g3._render_counts_panel(axis_empty, "Carpark", counts_cfg)
    assert axis_empty.texts

    # Accuracy panel with data should produce bars and an ylim.
    per_dom = {
        "Math": {
            "entropy": np.array([0.1, 0.2]),
            "aha": np.array([0, 1]),
            "correct": np.array([1, 0]),
        }
    }
    acc_cfg = g3.AccuracyFigureConfig(
        per_domain=per_dom,
        domains=["Math"],
        edges_by_domain={"Math": np.array([0.0, 0.5, 1.0])},
        x_label="x",
        args=SimpleNamespace(y_pad=5, title=None, width_in=1.0, height_in=1.0),
        color_noaha="blue",
        color_aha="green",
    )
    axis_data = AxisStub()
    g3._render_accuracy_panel(axis_data, "Math", acc_cfg)
    assert axis_data.bars and axis_data.ylims


def test_render_metric_accuracy_and_counts_pipeline(monkeypatch, tmp_path):
    args = SimpleNamespace(
        outfile_tag=None,
        outdir=str(tmp_path),
        bins=2,
        binning="fixed",
        entropy_min=None,
        entropy_max=None,
        share_bins="global",
        dpi=50,
        width_in=2.0,
        height_in=2.0,
        title=None,
        y_pad=1,
        roots_carpark=[],
        roots_crossword=[],
        roots_math=[],
        carpark_success_op="ge",
        carpark_soft_threshold=0.0,
        split=None,
        min_step=0,
        max_step=10,
        gpt_mode="canonical",
    )

    # Accuracy: first call returns False when no rows.
    monkeypatch.setattr(g3, "_load_metric_rows", lambda *a, **k: [])
    assert g3.render_metric_accuracy(args, "answer", "a", "b") is False

    rows = [
        {"domain": "Carpark", "entropy": 0.1, "aha": 1, "correct": 1},
        {"domain": "Crossword", "entropy": 0.2, "aha": 0, "correct": 0},
        {"domain": "Math", "entropy": 0.3, "aha": 1, "correct": 1},
    ]
    monkeypatch.setattr(g3, "_load_metric_rows", lambda *a, **k: rows)
    monkeypatch.setattr(
        g3,
        "_build_per_domain_accuracy",
        lambda r, d: {
            dom: {"entropy": np.array([0.1]), "aha": np.array([1]), "correct": np.array([1])} for dom in g3.DOMAINS
        },
    )
    monkeypatch.setattr(
        g3,
        "_compute_edges_for_accuracy",
        lambda per, doms, _args: ({dom: np.array([0.0, 1.0]) for dom in doms}, np.array([0.0, 1.0])),
    )
    monkeypatch.setattr(g3, "_metric_labels", lambda metric: ("X", "tag"))
    table_calls = []
    monkeypatch.setattr(g3, "_write_accuracy_tables", lambda cfg: table_calls.append(cfg.metric))
    monkeypatch.setattr(g3, "_build_accuracy_figure", lambda cfg: SimpleNamespace(saved=True))
    saved_bases = []
    monkeypatch.setattr(g3, "save_all_formats", lambda fig, out_base, dpi: saved_bases.append(out_base))

    assert g3.render_metric_accuracy(args, "answer", "c1", "c2") is True
    assert table_calls == ["answer"]
    assert any("graph_3_pass1_bins_tag_combined" in base for base in saved_bases)

    # Counts pipeline
    monkeypatch.setattr(g3, "_load_metric_rows", lambda *a, **k: rows)
    monkeypatch.setattr(
        g3,
        "_build_per_domain_counts",
        lambda r, d: {dom: {"entropy": np.array([0.1]), "aha": np.array([1])} for dom in g3.DOMAINS},
    )
    monkeypatch.setattr(
        g3, "_compute_edges_for_counts", lambda per, doms, _args: {dom: np.array([0.0, 1.0]) for dom in doms}
    )
    monkeypatch.setattr(g3, "_metric_labels", lambda metric: ("X", "tag"))
    monkeypatch.setattr(g3, "_build_counts_figure", lambda cfg: SimpleNamespace(saved=True))
    saved_counts = []
    monkeypatch.setattr(g3, "save_all_formats", lambda fig, out_base, dpi: saved_counts.append(out_base))
    assert g3.render_metric_counts(args, "think", "cA") is True
    assert any("graph_3_pass1_counts_tag_combined" in base for base in saved_counts)
