import numpy as np
import pandas as pd

import src.analysis.core.figure_1_plotting as f1p


def test_panel_auto_ylim_handles_constant_and_empty():
    empty = f1p._panel_auto_ylim({"A": pd.DataFrame()}, pad=0.1)
    assert empty == (0.0, 1.0)
    df = pd.DataFrame({"ratio": [0.5, 0.5], "lo": [0.5, 0.5], "hi": [0.5, 0.5]})
    lo, hi = f1p._panel_auto_ylim({"A": df}, pad=0.0)
    assert hi > lo and np.isclose(hi - lo, 0.05)
