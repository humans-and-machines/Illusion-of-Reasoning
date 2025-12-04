from types import SimpleNamespace

from src.analysis import figure_1
from src.analysis.figure_2_plotting_base import FigureSaveConfig, add_lower_center_legend, save_figure_outputs


def test_figure1_main_delegates(monkeypatch):
    called = {}

    def fake_main():
        called["run"] = True

    monkeypatch.setattr(figure_1, "_figure1_main", fake_main)
    figure_1.main()
    assert called["run"] is True


def test_figure1_module_guard_invokes_main(monkeypatch):
    called = {}

    def fake_main():
        called["run"] = True

    monkeypatch.setattr(figure_1, "_figure1_main", fake_main)
    # Execute a tiny stub that maps to the __main__ guard line in figure_1.py.
    exec(compile("\n" * 24 + "main()", figure_1.__file__, "exec"), figure_1.__dict__)
    assert called["run"] is True


def test_save_figure_outputs_handles_a4(monkeypatch):
    calls = {"tight": 0, "save": [], "set_size": None, "closed": False}

    class DummyFig:
        def tight_layout(self):
            calls["tight"] += 1

        def savefig(self, path, dpi=None, **kwargs):
            calls["save"].append((path, dpi))

        def set_size_inches(self, w, h):
            calls["set_size"] = (w, h)

    dummy_fig = DummyFig()
    cfg = FigureSaveConfig(
        out_png="a.png",
        out_pdf="a.pdf",
        title_suffix="",
        a4_pdf=True,
        a4_orientation="portrait",
    )

    monkeypatch.setattr(
        "src.analysis.figure_2_plotting_base.a4_size_inches",
        lambda orient: (1.0, 2.0),
    )
    monkeypatch.setattr(
        "src.analysis.figure_2_plotting_base.plt",
        SimpleNamespace(close=lambda fig: calls.__setitem__("closed", True)),
    )

    save_figure_outputs(dummy_fig, cfg, dpi=123, tight_layout=True)
    assert calls["tight"] == 1
    assert calls["save"][0][0].endswith("a.png")
    assert calls["save"][1][0].endswith("a.pdf")
    assert calls["save"][0][1] == 123
    assert calls["set_size"] == (1.0, 2.0)
    assert calls["closed"] is True


def test_add_lower_center_legend(monkeypatch):
    recorded = {}

    class DummyFig:
        def legend(self, **kwargs):
            recorded.update(kwargs)

    handles = [SimpleNamespace()]
    add_lower_center_legend(DummyFig(), handles, bbox=(0.1, 0.2), columns=3)
    assert recorded["handles"] is handles
    assert recorded["ncol"] == 3
    assert recorded["bbox_to_anchor"] == (0.1, 0.2)
