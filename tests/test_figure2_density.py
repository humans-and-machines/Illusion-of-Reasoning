import numpy as np
import pandas as pd

import src.analysis.figure_2_density as dens


def test_plot_four_correct_hists_handles_empty_panels(monkeypatch):
    # Avoid file outputs during test.
    monkeypatch.setattr(dens, "save_figure_outputs", lambda *args, **kwargs: None)
    # If compute_correct_hist were called this would raise; empty panels should skip it.
    monkeypatch.setattr(dens, "compute_correct_hist", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError))

    df_empty = pd.DataFrame(
        {
            "uncertainty": [],
            "correct": [],
            "aha_words": [],
            "aha_gpt": [],
        }
    )
    edges = np.array([0.0, 1.0])
    cfg = dens.FourHistConfig(
        out_png="unused.png",
        out_pdf="unused.pdf",
        title_suffix="",
        a4_pdf=False,
        a4_orientation="landscape",
        edges=edges,
    )
    dens.plot_four_correct_hists(df_empty, cfg)
