import numpy as np

import src.analysis.figure_2_data as f2d


def test_density_from_hist_handles_non_finite_hist(monkeypatch):
    edges = np.array([0.0, 0.5, 1.0])
    values = np.array([0.25, 0.75])

    def fake_histogram(vals, bins, density):
        assert density is True
        assert np.all(vals == values)
        assert np.all(bins == edges)
        return np.array([np.inf, 1.0]), bins

    monkeypatch.setattr(f2d.np, "histogram", fake_histogram)

    centers, hist = f2d.density_from_hist(values, edges)
    assert np.allclose(centers, [0.25, 0.75])
    assert np.all(hist == 0)
