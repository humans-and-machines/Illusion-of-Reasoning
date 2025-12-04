import pandas as pd

import src.analysis.core as core


def _sample_frame():
    return pd.DataFrame(
        {
            "problem": ["p1", "p1"],
            "step": [1, 2],
            "freq_correct": [0.1, 0.2],
            "aha_rate_gpt": [0.0, 0.5],
            "aha_any_gpt": [0, 1],
        }
    )


def test_add_standard_formal_flags_prefers_config(monkeypatch):
    frame = _sample_frame()
    cfg = core.FormalFlagConfig(thresholds=core.make_formal_thresholds(0.1, 0.2, 1, None))

    called = {}

    def fake_add(frame_arg, *, config=None, **kwargs):
        called["frame"] = frame_arg
        called["config"] = config
        called["kwargs"] = kwargs
        return "sentinel"

    monkeypatch.setattr(core, "add_formal_flags_column", fake_add)
    result = core.add_standard_formal_flags(frame, ["problem"], config=cfg)

    assert result == "sentinel"
    assert called["frame"] is frame
    assert called["config"] is cfg
    assert called["kwargs"] == {}


def test_add_standard_formal_flags_injects_default_prior_ok(monkeypatch):
    frame = _sample_frame()

    captured = {}

    def fake_add(_frame, *, config=None, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs
        return _frame

    monkeypatch.setattr(core, "add_formal_flags_column", fake_add)

    result = core.add_standard_formal_flags(
        frame,
        ["problem"],
        delta1=0.1,
        delta2=0.2,
        min_prior_steps=1,
        delta3=0.3,
    )

    assert captured["config"] is None
    assert captured["kwargs"]["formal_prior_ok_fn"] is core.formal_prior_ok
    # ensure thresholds were forwarded via **legacy_thresholds
    assert result.equals(frame)
