import runpy
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


def _patch_common(monkeypatch, tmp_path):
    # Avoid writing to real FS and avoid plotting side effects.
    monkeypatch.setattr("pathlib.Path.mkdir", lambda self, *a, **k: None)

    # Provide a lightweight matplotlib.pyplot stub so imports succeed without touching a backend.
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
        return types.SimpleNamespace(
            tight_layout=lambda *a, **k: None,
            savefig=lambda *a, **k: None,
            close=lambda *a, **k: None,
        )

    axis_stub = _AxisStub()
    fig_stub = _fig_stub()

    plt_stub = types.ModuleType("matplotlib.pyplot")
    plt_stub.figure = lambda *a, **k: fig_stub
    plt_stub.gcf = lambda: fig_stub
    plt_stub.gca = lambda: axis_stub
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
    plt_stub.pyplot = plt_stub  # guard against any accidental plt.pyplot access
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", plt_stub)
    if "matplotlib" in sys.modules:
        monkeypatch.setattr(sys.modules["matplotlib"], "pyplot", plt_stub, raising=False)

    def fake_read_json(path, lines=True):
        # Default frame used for the initial summary/plots.
        return pd.DataFrame(
            {
                "has_recheck": [0, 1],
                "entropy": [0.1, 0.2],
                "step": [1, 2],
                "rechecked": [0, 1],
                "correct": [0, 1],
            },
        )

    monkeypatch.setattr("pandas.read_json", fake_read_json)

    def glob_factory(analysis_files):
        def fake_glob(self, pattern):
            if "analysis" in str(self):
                return analysis_files
            # Root glob for initial concat
            return [tmp_path / "root.jsonl"]

        return fake_glob

    return glob_factory


def test_entropy_module_raises_when_no_scored(monkeypatch, tmp_path):
    sys.modules.pop("src.analysis.entropy", None)
    glob_factory = _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr("pathlib.Path.glob", glob_factory([]))

    with pytest.raises(SystemExit):
        runpy.run_module("src.analysis.entropy", run_name="__main__")


def test_entropy_module_keyerror_on_missing_columns(monkeypatch, tmp_path):
    sys.modules.pop("src.analysis.entropy", None)
    glob_factory = _patch_common(monkeypatch, tmp_path)
    analysis_file = tmp_path / "analysis.jsonl"
    monkeypatch.setattr("pathlib.Path.glob", glob_factory([analysis_file]))

    def fake_read_json(path, lines=True):
        if Path(path) == analysis_file:
            # Missing 'correct' and 'rechecked' triggers KeyError check
            return pd.DataFrame({"entropy": [0.5]})
        return pd.DataFrame(
            {
                "has_recheck": [0, 1],
                "entropy": [0.1, 0.2],
                "step": [1, 2],
                "rechecked": [0, 1],
                "correct": [0, 1],
            },
        )

    monkeypatch.setattr("pandas.read_json", fake_read_json)

    with pytest.raises(KeyError):
        runpy.run_module("src.analysis.entropy", run_name="__main__")
