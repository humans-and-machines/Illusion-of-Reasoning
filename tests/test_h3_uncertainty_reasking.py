import pandas as pd

from src.analysis.h3_uncertainty import reasking


def test_build_forced_condition_table_skips_empty_groups(monkeypatch):
    class _FakeDF:
        columns = ["forced_insight2"]

        def groupby(self, column):
            assert column == "forced_insight2"
            return [(1, pd.DataFrame())]

    df = _FakeDF()
    out = reasking._build_forced_condition_table(df, lambda **_: {"ok": True})
    assert isinstance(out, pd.DataFrame)
    assert out.empty
