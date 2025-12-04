from src.analysis.core import plotting_helpers


def test_set_global_fonts_invokes_apply(monkeypatch):
    called = {}

    def fake_apply(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(plotting_helpers, "apply_paper_font_style", fake_apply)
    plotting_helpers.set_global_fonts(font_family="Foo", font_size=13)

    assert called["font_family"] == "Foo"
    assert called["font_size"] == 13
    assert called["mathtext_fontset"] == "dejavuserif"


def test_compute_effective_max_step_caps(capsys):
    args = type("Args", (), {"max_step": None})()
    val = plotting_helpers.compute_effective_max_step(args, hard_max_step=100)
    out = capsys.readouterr().out
    assert val == 100
    assert "Capping max_step" in out

    args.max_step = 200
    val = plotting_helpers.compute_effective_max_step(args, hard_max_step=150)
    out2 = capsys.readouterr().out
    assert val == 150
    assert "hard cap = 150" in out2

    args.max_step = 50
    out3 = capsys.readouterr().out
    assert plotting_helpers.compute_effective_max_step(args, hard_max_step=60) == 50
    assert out3 == ""


class _LineStub:
    def __init__(self, color, label):
        self._color = color
        self._label = label

    def get_color(self):
        return self._color

    def get_label(self):
        return self._label


class _AxisStub:
    def __init__(self, lines):
        self.lines = lines


def test_aha_histogram_legend_handles():
    axis = _AxisStub([_LineStub("red", "Aha=0"), _LineStub("blue", "Aha=1")])
    handles = plotting_helpers.aha_histogram_legend_handles(axis)
    assert len(handles) == 3
    assert handles[0].get_label() == "Total (bin count)"
    labels = [h.get_label() for h in handles[1:]]
    assert labels == ["Aha=0", "Aha=1"]


def test_create_a4_figure_uses_page_size_and_dpi(monkeypatch):
    called = {}

    def fake_a4(orientation):
        called["orientation"] = orientation
        return (8, 9)

    monkeypatch.setattr(plotting_helpers, "a4_size_inches", fake_a4)
    monkeypatch.setattr(plotting_helpers.plt, "figure", lambda **kwargs: called.update(kwargs) or "FIG")

    fig = plotting_helpers.create_a4_figure(orientation="landscape", dpi=123)
    assert fig == "FIG"
    assert called["orientation"] == "landscape"
    assert called["figsize"] == (8, 9)
    assert called["dpi"] == 123
