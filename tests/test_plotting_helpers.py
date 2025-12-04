import matplotlib
import matplotlib.pyplot as plt

import src.analysis.plotting as plotting


def test_apply_default_and_entropy_styles(monkeypatch):
    # Ensure no error updating rcParams and overrides apply.
    plotting.apply_default_style(extra={"axes.titlesize": 10})
    assert matplotlib.rcParams["axes.titlesize"] == 10

    plotting.apply_entropy_plot_style(extra={"legend.fontsize": 15})
    assert matplotlib.rcParams["legend.fontsize"] == 15
    # pdf.fonttype should be set by entropy style
    assert matplotlib.rcParams["pdf.fonttype"] == 42


def test_apply_paper_font_style_sets_params():
    plotting.apply_paper_font_style(font_family="TestFamily", font_size=8, mathtext_fontset="stix")
    assert matplotlib.rcParams["font.family"][0] == "serif"
    assert matplotlib.rcParams["font.serif"][0] == "TestFamily"
    assert matplotlib.rcParams["font.size"] == 8
    assert matplotlib.rcParams["mathtext.fontset"] == "stix"


def test_a4_size_inches_orientation():
    assert plotting.a4_size_inches("portrait") == plotting.A4_PORTRAIT
    assert plotting.a4_size_inches("landscape") == plotting.A4_LANDSCAPE
    assert plotting.a4_size_inches("P") == plotting.A4_PORTRAIT


def test_save_figure_creates_dirs_and_formats(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    outbase = tmp_path / "nested" / "fig"
    plotting.save_figure(fig, str(outbase), dpi=50, formats=("png",))
    assert (tmp_path / "nested" / "fig.png").exists()
