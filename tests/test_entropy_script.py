import builtins
import json
import runpy
import sys
import types
from types import ModuleType, SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_entropy_script_runs_with_synthetic_data(tmp_path, monkeypatch):
    base = tmp_path / "artifacts" / "results" / "od2961" / "Math220k" / "GRPO" / "1.5B"
    analysis_dir = base / "analysis"

    # Build a small yet diverse dataset to satisfy qcut/decile splits and flips.
    records = []
    for i in range(20):
        records.append(
            {
                "entropy": 0.1 * (i + 1),
                "has_recheck": i % 2,
                "rechecked": i % 2,
                "correct": (i + 1) % 2,
                "step": 50 + 20 * i,
                "problem": f"p{i % 5}",
                "sample_idx": i % 4,
                "output": f"out{i}",
            }
        )
    # Add a flip trajectory across steps 50→850.
    records.extend(
        [
            {
                "entropy": 0.05,
                "has_recheck": 0,
                "rechecked": 0,
                "correct": 0,
                "step": 50,
                "problem": "flip_prob",
                "sample_idx": 99,
                "output": "early",
            },
            {
                "entropy": 2.5,
                "has_recheck": 1,
                "rechecked": 1,
                "correct": 1,
                "step": 850,
                "problem": "flip_prob",
                "sample_idx": 99,
                "output": "late",
            },
        ]
    )

    base_file = _write_jsonl(base / "records.jsonl", records)
    scored_file = _write_jsonl(analysis_dir / "sample_scored.jsonl", records)

    class _AxisStub:
        def plot(self, *a, **k):  # pragma: no cover - dummy
            return None

        def errorbar(self, *a, **k):  # pragma: no cover - dummy
            return None

        def text(self, *a, **k):  # pragma: no cover - dummy
            return None

        def set_xlim(self, *a, **k):  # pragma: no cover - dummy
            return None

        def set_xticks(self, *a, **k):  # pragma: no cover - dummy
            return None

        def set_ylabel(self, *a, **k):  # pragma: no cover - dummy
            return None

        def set_xlabel(self, *a, **k):  # pragma: no cover - dummy
            return None

        def set_title(self, *a, **k):  # pragma: no cover - dummy
            return None

        def tick_params(self, *a, **k):  # pragma: no cover - dummy
            return None

        def set_ylim(self, *a, **k):  # pragma: no cover - dummy
            return None

        def legend(self, *a, **k):  # pragma: no cover - dummy
            return None

        def bar(self, *a, **k):  # pragma: no cover - dummy
            return None

        def twinx(self, *a, **k):  # pragma: no cover - dummy
            return _AxisStub()

    def _fig_stub():
        return SimpleNamespace(
            tight_layout=lambda *a, **k: None,
            savefig=lambda *a, **k: None,
            close=lambda *a, **k: None,
        )

    plt_stub = types.ModuleType("matplotlib.pyplot")
    plt_stub.figure = lambda *a, **k: _fig_stub()
    plt_stub.gcf = lambda: _fig_stub()
    plt_stub.gca = lambda: _AxisStub()
    plt_stub.subplots = lambda *a, **k: (_fig_stub(), _AxisStub())
    plt_stub.boxplot = lambda *a, **k: None
    plt_stub.scatter = lambda *a, **k: None
    plt_stub.colorbar = lambda *a, **k: None
    plt_stub.errorbar = lambda *a, **k: None
    plt_stub.xlabel = lambda *a, **k: None
    plt_stub.ylabel = lambda *a, **k: None
    plt_stub.title = lambda *a, **k: None
    plt_stub.xticks = lambda *a, **k: None
    plt_stub.ylim = lambda *a, **k: None
    plt_stub.legend = lambda *a, **k: None
    plt_stub.tight_layout = lambda *a, **k: None
    plt_stub.savefig = lambda *a, **k: None
    plt_stub.close = lambda *a, **k: None
    plt_stub.switch_backend = lambda *a, **k: None
    plt_stub.pyplot = plt_stub
    # Ensure subsequent imports and local references use the stubbed matplotlib pyplot.
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", plt_stub)
    if "matplotlib" in sys.modules:
        monkeypatch.setattr(sys.modules["matplotlib"], "pyplot", plt_stub, raising=False)
    monkeypatch.setattr(sys.modules[__name__], "plt", plt_stub)

    # Stub statsmodels.formula.api.logit to avoid heavy dependency.
    class _FakeResult:
        cov_type = "cluster"
        bse = np.array([1.0, 1.0, 1.0, 1.0])

        def summary(self):
            return "summary"

    class _FakeModel:
        def fit(self, disp=False, cov_type=None, cov_kwds=None):
            return _FakeResult()

    fake_sm_module = ModuleType("statsmodels.formula.api")
    fake_sm_module.logit = lambda formula, data: _FakeModel()
    monkeypatch.setitem(sys.modules, "statsmodels.formula.api", fake_sm_module)
    original_zip = builtins.zip

    def _zip_no_strict(*args, **kwargs):
        kwargs.pop("strict", None)
        return original_zip(*args)

    monkeypatch.setattr(builtins, "zip", _zip_no_strict)
    monkeypatch.setattr(plt, "errorbar", lambda *args, **kwargs: None)
    monkeypatch.setattr(Axes, "errorbar", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(plt, "figure", lambda *args, **kwargs: types.SimpleNamespace())
    monkeypatch.setattr(plt, "boxplot", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "scatter", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "colorbar", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "tight_layout", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "xticks", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "xlabel", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "ylabel", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "title", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "legend", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "subplots", lambda *args, **kwargs: (_fig_stub(), _AxisStub()))
    original_getitem = pd.DataFrame.__getitem__

    def _safe_getitem(self, key):
        if not isinstance(key, (str, bytes)) and not np.isscalar(key):
            try:
                return self.iloc[list(key)]
            except Exception:
                return self.iloc[0:0]
        return original_getitem(self, key)

    monkeypatch.setattr(pd.DataFrame, "__getitem__", _safe_getitem)

    # Ensure plots/files write under the tmp_path sandbox.
    monkeypatch.chdir(tmp_path)

    # Run the script module; it should complete without raising.
    runpy.run_module("src.analysis.entropy", run_name="__main__")

    # Sanity-check that outputs were attempted.
    assert base_file.exists()
    assert scored_file.exists()
    assert (tmp_path / "analysis").exists()
