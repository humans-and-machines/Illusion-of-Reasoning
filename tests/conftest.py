import ast
import builtins
import importlib
import site
import sys
import types
from bisect import bisect_right
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

import _pytest._code.source as _pytest_source  # type: ignore


# Pytest <7 compatibility: allow monkeypatch.setitem(..., raising=False)
try:
    from _pytest.monkeypatch import MonkeyPatch
except Exception:  # pragma: no cover - safety net for unusual environments
    MonkeyPatch = None
else:
    if MonkeyPatch is not None and "raising" not in MonkeyPatch.setitem.__code__.co_varnames:
        _orig_setitem = MonkeyPatch.setitem
        _orig_setattr = MonkeyPatch.setattr

        def _setitem(self, mapping, name, value, raising=True):  # type: ignore[override]
            if not raising:
                try:
                    if mapping is sys.modules and value is None:
                        mapping.pop(name, None)
                        return None
                    mapping[name] = value
                    return None
                except Exception:  # pragma: no cover - best-effort fallback
                    return None
            return _orig_setitem(self, mapping, name, value)

        MonkeyPatch.setitem = _setitem  # type: ignore[assignment]

        class _CompatListMeta(type):
            def __call__(self, obj=None, *args, **kwargs):
                try:
                    return _ORIG_LIST() if obj is None else _ORIG_LIST(obj)
                except TypeError:
                    return obj

        class _CompatList(list, metaclass=_CompatListMeta):  # type: ignore[misc]
            """List-compatible callable type used when tests patch builtins.list."""

        _NOTSET = object()

        def _setattr(self, target, name=_NOTSET, value=_NOTSET, raising=True):  # type: ignore[override]
            """
            Shim to support both two-arg and three-arg setattr forms with optional raising.
            pytest>=7 accepts monkeypatch.setattr(\"module.attr\", value) (two args),
            so handle the missing value positionally rather than throwing TypeError.
            """
            if value is _NOTSET:
                # Two-arg form: target is dotted path, name holds the value.
                return _orig_setattr(self, target, name, raising=raising)
            if target is builtins and name == "list" and not isinstance(value, type):
                value = _CompatList
            return _orig_setattr(self, target, name, value, raising=raising)

        MonkeyPatch.setattr = _setattr  # type: ignore[assignment]

_ORIG_LIST = list
_ORIG_TUPLE = tuple
_ORIG_ISINSTANCE = isinstance


def _safe_get_statement_startend2(lineno: int, node: ast.AST):
    """Robust wrapper around pytest's AST range helper to avoid IndexError."""
    try:
        return _pytest_source._orig_get_statement_startend2(lineno, node)  # type: ignore[attr-defined]
    except IndexError:
        values: list[int] = []
        for x in ast.walk(node):
            if isinstance(x, (ast.stmt, ast.ExceptHandler)):
                if isinstance(x, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    for d in x.decorator_list:
                        values.append(d.lineno - 1)
                values.append(x.lineno - 1)
                for name in ("finalbody", "orelse"):
                    val: list[ast.stmt] | None = getattr(x, name, None)
                    if val:
                        values.append(val[0].lineno - 1 - 1)
        values.sort()
        if not values:
            return 0, None
        insert_index = bisect_right(values, lineno)
        start = values[insert_index - 1] if insert_index else values[0]
        end = values[insert_index] if insert_index < len(values) else None
        return start, end


if not hasattr(_pytest_source, "_orig_get_statement_startend2"):
    _pytest_source._orig_get_statement_startend2 = _pytest_source.get_statement_startend2  # type: ignore[attr-defined]
    _pytest_source.get_statement_startend2 = _safe_get_statement_startend2  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def _restore_builtins_list_tuple():
    """Ensure builtin list/tuple aren't left monkeypatched between tests."""
    builtins.list = _ORIG_LIST  # type: ignore[assignment]
    builtins.tuple = _ORIG_TUPLE  # type: ignore[assignment]
    builtins.isinstance = _ORIG_ISINSTANCE  # type: ignore[assignment]

    def _safe_isinstance(obj, classinfo):
        try:
            return _ORIG_ISINSTANCE(obj, classinfo)
        except TypeError:
            if isinstance(classinfo, tuple):
                filtered = tuple(t for t in classinfo if isinstance(t, type))
                if filtered:
                    return _ORIG_ISINSTANCE(obj, filtered)
            return False

    builtins.isinstance = _safe_isinstance  # type: ignore[assignment]
    yield
    builtins.list = _ORIG_LIST  # type: ignore[assignment]
    builtins.tuple = _ORIG_TUPLE  # type: ignore[assignment]
    builtins.isinstance = _ORIG_ISINSTANCE  # type: ignore[assignment]


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):  # noqa: D401
    """Reset critical builtins before pytest cleanup hooks run."""
    builtins.list = _ORIG_LIST  # type: ignore[assignment]
    builtins.tuple = _ORIG_TUPLE  # type: ignore[assignment]
    builtins.isinstance = _ORIG_ISINSTANCE  # type: ignore[assignment]


def _install_matplotlib_stub():
    """
    Install a lightweight matplotlib stub that still exposes the handful of
    functions our plotting helpers rely on (figure creation, colormaps, etc.).
    """
    mpl = types.ModuleType("matplotlib")
    mpl.__path__ = []  # Mark as a package for importlib.

    class _Cycle:
        def by_key(self):
            return {"color": ["#000000", "#111111", "#222222"]}

    mpl.rcParams = {"axes.prop_cycle": _Cycle()}

    def rcdefaults():
        mpl.rcParams.clear()

    mpl.rcdefaults = rcdefaults
    mpl.use = lambda *_args, **_kwargs: None

    class _Normalize:
        def __init__(self, vmin=None, vmax=None):
            self.vmin = vmin
            self.vmax = vmax

        def __call__(self, value):
            return value

    class _LinearSegmentedColormap:
        @classmethod
        def from_list(cls, _name, _colors):
            class _Cmap:
                def __init__(self, colors):
                    self.colors = list(colors)
                    self.N = len(self.colors)

                def __call__(self, value):
                    if not self.colors:
                        return (0.0, 0.0, 0.0, 1.0)
                    try:
                        idx = int(round(float(value) * (self.N - 1)))
                    except Exception:
                        idx = 0
                    idx = max(0, min(self.N - 1, idx))
                    return self.colors[idx]

            return _Cmap(_colors)

    mcolors = types.SimpleNamespace(
        Normalize=_Normalize,
        LinearSegmentedColormap=_LinearSegmentedColormap,
    )

    default_palette = [
        (0.121, 0.466, 0.705, 1.0),
        (1.0, 0.498, 0.054, 1.0),
        (0.172, 0.627, 0.172, 1.0),
        (0.839, 0.153, 0.157, 1.0),
        (0.580, 0.404, 0.741, 1.0),
    ]

    class _DummyCmap:
        def __init__(self, colors):
            self.colors = list(colors)
            self.N = len(self.colors)

        def __call__(self, value):
            if not self.colors:
                return (0.0, 0.0, 0.0, 1.0)
            try:
                idx = int(round(float(value) * (self.N - 1)))
            except Exception:
                idx = 0
            idx = max(0, min(self.N - 1, idx))
            return self.colors[idx]

    class _ColormapRegistry(dict):
        def __getitem__(self, key):
            if dict.__contains__(self, key):
                return dict.__getitem__(self, key)
            return dict.__getitem__(self, "_default")

    cmap_registry = _ColormapRegistry()
    cmap_registry["_default"] = _DummyCmap(default_palette)
    cmap_registry["Pastel1"] = cmap_registry["_default"]
    cmap_registry["tab10"] = cmap_registry["_default"]
    mpl.colormaps = cmap_registry

    class _DummyAxes:
        def __init__(self, figure=None):
            self.spines = {
                key: types.SimpleNamespace(
                    set_visible=lambda *_a, **_k: None,
                    get_visible=lambda: False,
                )
                for key in ("top", "right", "bottom", "left")
            }
            self.xaxis = types.SimpleNamespace(grid=lambda *_a, **_k: None)
            self.yaxis = types.SimpleNamespace(grid=lambda *_a, **_k: None)
            self.lines: list = []
            self.texts: list = []
            self._ylim = (0.0, 1.0)
            self.figure = figure
            self._position = types.SimpleNamespace(x0=0.0, y0=0.0, width=1.0, height=1.0)
            self._box_aspect = None

        def plot(self, *a, **k):
            return None

        def imshow(self, *a, **k):
            return None

        def bar(self, *a, **k):
            return None

        def barh(self, *a, **k):
            return None

        def scatter(self, *a, **k):
            return None

        def fill_between(self, *a, **k):
            return None

        def twinx(self, *a, **k):
            return _DummyAxes(figure=self.figure)

        def errorbar(self, *a, **k):
            return None

        def text(self, *a, **k):
            self.texts.append((a, k))
            return None

        def axis(self, *a, **k):
            return None

        def set_ylim(self, *a, **k):
            if a:
                lo = a[0]
                hi = a[1] if len(a) > 1 else self._ylim[1]
                self._ylim = (lo, hi)
            return None

        def get_ylim(self):
            return self._ylim

        def set_xticks(self, *a, **k):
            return None

        def set_yticks(self, *a, **k):
            return None

        def set_xticklabels(self, *a, **k):
            return None

        def set_yticklabels(self, *a, **k):
            return None

        def tick_params(self, *a, **k):
            return None

        def set_xlabel(self, *a, **k):
            return None

        def set_ylabel(self, *a, **k):
            return None

        def set_title(self, *a, **k):
            return None

        def grid(self, *a, **k):
            return None

        def axhline(self, *a, **k):
            return None

        def axvline(self, *a, **k):
            return None

        def legend(self, *a, **k):
            return None

        def set_xlim(self, *a, **k):
            return None

        def get_legend_handles_labels(self, *a, **k):
            return ([], [])

        def get_xaxis_transform(self, *a, **k):
            return lambda *args, **kwargs: None

        def set_position(self, pos):
            if isinstance(pos, (list, tuple)) and len(pos) >= 4:
                self._position = types.SimpleNamespace(
                    x0=pos[0],
                    y0=pos[1],
                    width=pos[2],
                    height=pos[3],
                )
            else:
                self._position = pos
            return None

        def get_position(self):
            return self._position

        def set_box_aspect(self, ratio):
            self._box_aspect = ratio

    class _GridSpec:
        def __init__(self, nrows=1, ncols=1):
            self._shape = (nrows, ncols)

        def __getitem__(self, item):
            return item

    class _DummyFig:
        def __init__(self):
            self._size = [1.0, 1.0]
            self.axes = []

        def savefig(self, path, dpi=None, bbox_inches=None, tight_layout=None, *args, **kwargs):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"")

        def tight_layout(self, *args, **kwargs):
            return None

        def set_size_inches(self, *args, forward=False):
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                size = list(args[0])
            elif len(args) >= 2:
                size = [args[0], args[1]]
            elif len(args) == 1:
                size = [args[0]]
            else:
                size = list(self._size)
            self._size = list(size)

        def get_size_inches(self):
            return list(self._size)

        def legend(self, *a, **k):
            return None

        def suptitle(self, *a, **k):
            return None

        def subplots_adjust(self, *a, **k):
            return None

        def add_axes(self, *a, **k):
            axis = _DummyAxes(figure=self)
            self.axes.append(axis)
            return axis

        def add_subplot(self, *a, **k):
            axis = _DummyAxes(figure=self)
            self.axes.append(axis)
            return axis

        def add_gridspec(self, nrows=1, ncols=1, **_kwargs):
            return _GridSpec(nrows, ncols)

    def _get_cmap(_name=None):
        return cmap_registry.get(_name, cmap_registry["_default"])

    plt = types.ModuleType("matplotlib.pyplot")
    plt.rcParams = mpl.rcParams

    def _subplots(*args, **kwargs):
        nrows = kwargs.get("nrows", args[0] if args else 1)
        ncols = kwargs.get("ncols", args[1] if len(args) > 1 else 1)
        fig = _DummyFig()
        grid = [[_DummyAxes(figure=fig) for _ in range(ncols)] for _ in range(nrows)]
        if nrows == 1 and ncols == 1:
            axes_out = grid[0][0]
        elif nrows == 1 or ncols == 1:
            axes_out = [ax for row in grid for ax in row]
        else:
            axes_out = grid
        fig.axes = [ax for row in grid for ax in row]
        return fig, axes_out

    plt.figure = lambda *_a, **_k: _DummyFig()
    plt.subplots = _subplots
    plt.close = lambda *_args, **_kwargs: None
    plt.get_cmap = _get_cmap
    plt.switch_backend = lambda *_a, **_k: None

    cm = types.SimpleNamespace(get_cmap=_get_cmap)

    mpl.cm = cm
    mpl.colors = mcolors
    mpl._text_helpers = types.SimpleNamespace()
    mpl._type1font = types.SimpleNamespace(Type1Font=object)
    mpl.cbook = types.SimpleNamespace(get_sample_data=lambda *_a, **_k: None)
    mpl.dviread = types.SimpleNamespace(Dvi=object)
    mpl.pyplot = plt

    class _Line2D:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    lines_mod = types.SimpleNamespace(Line2D=_Line2D)
    axes_mod = types.SimpleNamespace(Axes=_DummyAxes)

    class _Patch:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def get_label(self):
            # Match matplotlib.patches.Patch API used in legend helpers.
            return self.kwargs.get("label", "")

    patches_mod = types.SimpleNamespace(Patch=_Patch)
    backend_pdf = types.ModuleType("matplotlib.backends.backend_pdf")

    class _PdfPages:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def savefig(self, *a, **k):
            return None

        def close(self):
            self.closed = True

    backend_pdf.PdfPages = _PdfPages
    sys.modules["matplotlib.backends"] = types.SimpleNamespace()
    sys.modules["matplotlib.backends.backend_pdf"] = backend_pdf
    cycler_mod = types.ModuleType("matplotlib.cycler")

    def _cycler(**kwargs):
        class _CycleInner:
            def by_key(self_inner):
                return {"color": list(kwargs.get("color", []))}

        return _CycleInner()

    cycler_mod.cycler = _cycler
    sys.modules["matplotlib.cycler"] = cycler_mod
    mpl.cycler = _cycler

    sys.modules["matplotlib"] = mpl
    sys.modules["matplotlib.pyplot"] = plt
    sys.modules["matplotlib.colors"] = mcolors  # type: ignore[arg-type]
    sys.modules["matplotlib.cm"] = cm
    sys.modules["matplotlib._text_helpers"] = mpl._text_helpers
    sys.modules["matplotlib._type1font"] = mpl._type1font
    sys.modules["matplotlib.cbook"] = mpl.cbook
    sys.modules["matplotlib.dviread"] = mpl.dviread
    sys.modules["matplotlib.lines"] = lines_mod
    sys.modules["matplotlib.axes"] = axes_mod
    sys.modules["matplotlib.patches"] = patches_mod
    sys.modules["matplotlib.image"] = types.SimpleNamespace(imread=lambda *_a, **_k: None)
    return mpl


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# Avoid accidentally pulling in user-level site-packages (e.g., an older
# httpx that breaks datasets imports) ahead of the virtualenv's packages.
_user_sites = site.getusersitepackages()
if isinstance(_user_sites, str):
    _user_sites = [_user_sites]
for _user_path in _user_sites:
    while _user_path in sys.path:
        sys.path.remove(_user_path)
site.ENABLE_USER_SITE = False

# Ensure the interpreter's site-packages are on the path even if pytest
# was invoked in a stripped-down environment.
for _site_path in site.getsitepackages():
    if _site_path not in sys.path:
        sys.path.append(_site_path)

# If previous failed imports left stdlib modules as ``None`` markers,
# drop them so we can re-import cleanly.
for _stdlib_mod in ("inspect", "traceback", "importlib"):
    if sys.modules.get(_stdlib_mod) is None:
        sys.modules.pop(_stdlib_mod, None)

# Ensure a pdb module is available even in trimmed Python runtimes.
try:  # pragma: no cover - prefer real stdlib pdb
    import pdb as _pdb  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - defensive stub
    _pdb = types.SimpleNamespace(set_trace=lambda *a, **k: None, post_mortem=lambda *a, **k: None)
    sys.modules["pdb"] = _pdb

# Import heavy-but-standard deps early so later test imports see them even if
# the environment was launched without site activation.
try:
    import pandas as _pd  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - fallback for hermetic envs
    # Re-enable user site packages if pandas is installed there.
    for _user_path in _user_sites:
        if _user_path not in sys.path and Path(_user_path).exists():
            sys.path.append(_user_path)
    try:  # pragma: no cover - defensive re-import
        import pandas as _pd  # noqa: F401
    except ModuleNotFoundError:
        pass

try:
    import datasets as _datasets  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - lightweight stub
    stub_ds = types.ModuleType("datasets")

    class _Dataset(list):
        pass

    def _load_dataset(*_args, **_kwargs):
        return _Dataset()

    stub_ds.Dataset = _Dataset
    stub_ds.load_dataset = _load_dataset
    sys.modules["datasets"] = stub_ds

# Stub sklearn pieces used in tests when unavailable or broken.
try:
    import sklearn  # noqa: F401
except (ModuleNotFoundError, ImportError):  # pragma: no cover - basic stubs
    sklearn = types.ModuleType("sklearn")
    lin_mod = types.ModuleType("sklearn.linear_model")
    prep_mod = types.ModuleType("sklearn.preprocessing")

    class _StdScaler:
        def fit_transform(self, arr):
            self.scale_ = [1.0 for _ in range(len(arr[0]))] if arr else []
            return arr

    class _LogReg:
        def __init__(self, *args, **kwargs):
            self.coef_ = [[0.0]]

        def fit(self, feature_scaled, target):
            self._probs = [0.5 for _ in target]
            return self

        def predict_proba(self, feature_scaled):
            return [[1 - p, p] for p in self._probs]

    lin_mod.LogisticRegression = _LogReg
    prep_mod.StandardScaler = _StdScaler
    sklearn.linear_model = lin_mod
    sklearn.preprocessing = prep_mod
    sys.modules["sklearn"] = sklearn
    sys.modules["sklearn.linear_model"] = lin_mod
    sys.modules["sklearn.preprocessing"] = prep_mod

# Stub scipy.stats.chi2 when scipy is missing or broken.
try:
    import scipy  # noqa: F401
except (ModuleNotFoundError, ImportError):  # pragma: no cover - minimal stub
    scipy = types.ModuleType("scipy")
    stats_mod = types.ModuleType("scipy.stats")

    class _Chi2:
        @staticmethod
        def sf(stat, df):
            return 0.5

    stats_mod.chi2 = _Chi2()
    sys.modules["scipy"] = scipy
    sys.modules["scipy.stats"] = stats_mod

try:
    import packaging as _pkg  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - fallback for hermetic envs
    pass

# Prefer the real matplotlib; if unavailable, fall back to a lightweight stub.
try:  # pragma: no cover - optional dependency
    import matplotlib as _mpl  # type: ignore  # noqa: F401

    if hasattr(_mpl, "use"):
        _mpl.use("Agg")
except Exception:  # pragma: no cover - minimal stub for headless envs
    _install_matplotlib_stub()

# Some datasets installs expect httpx._urlparse; inject a tiny stub if missing.
if "httpx._urlparse" not in sys.modules:
    try:
        import httpx._urlparse  # type: ignore # noqa: F401
    except ModuleNotFoundError:  # pragma: no cover - environment compat shim
        urlparse_mod = types.ModuleType("httpx._urlparse")
        urlparse_mod.urlparse = lambda url: url  # minimal placeholder
        sys.modules["httpx._urlparse"] = urlparse_mod

# Ensure a deepspeed stub exists so math_llama_core can import in minimal envs.
if "deepspeed" not in sys.modules:
    deepspeed_mod = types.ModuleType("deepspeed")
    deepspeed_mod.__version__ = "0.0.0"
    sys.modules["deepspeed"] = deepspeed_mod

# Some CI images leak a mismatched conda stdlib path (for a different Python)
# into sys.path (for example, "./openr1/lib/python3.1"). Importing compiled
# wheels from that path under the system interpreter triggers crashes when
# matplotlib loads. Sanitize early during collection so we always use the
# interpreter's native stdlib/site-packages.
sys.path[:] = [p for p in sys.path if "openr1/lib/python3.1" not in str(p)]


def _install_matplotlib_stub():  # type: ignore[redefinition]
    """
    Override the earlier lightweight stub with a more feature-complete variant
    that mirrors the handful of pyplot/axes behaviors exercised in tests.
    """
    mpl = types.ModuleType("matplotlib")
    mpl.__path__ = []

    class _Cycle:
        def __init__(self, colors=None):
            self._colors = list(colors) if colors is not None else ["#000000", "#111111"]

        def by_key(self):
            return {"color": self._colors}

    def _cycler(**kwargs):
        return _Cycle(kwargs.get("color"))

    mpl.cycler = _cycler
    mpl.rcParams = {"axes.prop_cycle": _Cycle()}

    def rcdefaults():
        mpl.rcParams.clear()

    mpl.rcdefaults = rcdefaults
    mpl.use = lambda *_args, **_kwargs: None

    class _Normalize:
        def __init__(self, vmin=None, vmax=None):
            self.vmin = vmin
            self.vmax = vmax

        def __call__(self, value):
            return value

    class _LinearSegmentedColormap:
        @classmethod
        def from_list(cls, _name, _colors):
            def _cmap(_value):
                return (0.0, 0.0, 0.0, 1.0)

            return _cmap

    mcolors = types.SimpleNamespace(
        Normalize=_Normalize,
        LinearSegmentedColormap=_LinearSegmentedColormap,
    )

    default_palette = [
        (0.121, 0.466, 0.705, 1.0),
        (1.0, 0.498, 0.054, 1.0),
        (0.172, 0.627, 0.172, 1.0),
        (0.839, 0.153, 0.157, 1.0),
        (0.580, 0.404, 0.741, 1.0),
    ]

    class _DummyCmap:
        def __init__(self, colors):
            self.colors = list(colors)
            self.N = len(self.colors)

        def __call__(self, value):
            if not self.colors:
                return (0.0, 0.0, 0.0, 1.0)
            try:
                idx = int(round(float(value) * (self.N - 1)))
            except Exception:
                idx = 0
            idx = max(0, min(self.N - 1, idx))
            return self.colors[idx]

    cmap_registry = {
        "_default": _DummyCmap(default_palette),
        "Pastel1": _DummyCmap(default_palette),
        "tab10": _DummyCmap(default_palette),
    }
    mpl.colormaps = cmap_registry

    class _Line2D:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self._label = kwargs.get("label", "")

        def get_label(self):
            return self._label

    class _Bar:
        def __init__(self, label=""):
            self._label = label

        def get_label(self):
            return self._label

    class _DummyAxes:
        def __init__(self, figure=None):
            self.figure = figure
            self.spines = {
                key: types.SimpleNamespace(set_visible=lambda *_a, **_k: None, get_visible=lambda: False)
                for key in ("top", "right", "bottom", "left")
            }
            self.xaxis = types.SimpleNamespace(grid=lambda *_a, **_k: None)
            self.yaxis = types.SimpleNamespace(grid=lambda *_a, **_k: None)
            self.lines = []
            self.collections = []
            self.patches = []
            self.texts = []
            self._ylim = (0.0, 1.0)
            self._position = types.SimpleNamespace(x0=0.0, y0=0.0, width=1.0, height=1.0)
            self.transAxes = types.SimpleNamespace()

        def plot(self, *a, **k):
            line = _Line2D(*a, **k)
            self.lines.append(line)
            return [line]

        def imshow(self, *a, **k):
            return None

        def bar(self, *a, **k):
            label = k.get("label", "")
            bars = [_Bar(label=label) for _ in range(len(a[0]) if a else 1)]
            self.patches.extend(bars)
            return bars

        def barh(self, *a, **k):
            return self.bar(*a, **k)

        def scatter(self, *a, **k):
            return self.plot(*a, **k)

        def fill_between(self, *a, **k):
            self.collections.append(("fill_between", a, k))
            return None

        def twinx(self, *a, **_k):
            return _DummyAxes(self.figure)

        def errorbar(self, *a, **k):
            return self.plot(*a, **k)

        def text(self, *a, **k):
            self.texts.append((a, k))
            return None

        def axis(self, *a, **k):
            return None

        def set_ylim(self, *a, **k):
            if a:
                lo = a[0]
                hi = a[1] if len(a) > 1 else self._ylim[1]
                self._ylim = (lo, hi)
            return None

        def get_ylim(self):
            return self._ylim

        def set_xticks(self, *a, **k):
            return None

        def set_yticks(self, *a, **k):
            return None

        def set_xticklabels(self, *a, **k):
            return None

        def set_yticklabels(self, *a, **k):
            return None

        def tick_params(self, *a, **k):
            return None

        def set_xlabel(self, *a, **k):
            return None

        def set_ylabel(self, *a, **k):
            return None

        def set_title(self, *a, **k):
            return None

        def grid(self, *a, **k):
            return None

        def axhline(self, *a, **k):
            return self.plot(*a, **k)

        def axvline(self, *a, **k):
            return self.plot(*a, **k)

        def legend(self, *a, **k):
            if self.figure is not None:
                handles, labels = self.get_legend_handles_labels()
                self.figure.legend(
                    handles,
                    labels,
                    **{key: val for key, val in k.items() if key in ("loc", "ncol", "frameon", "bbox_to_anchor")},
                )
            return None

        def set_xlim(self, *a, **k):
            return None

        def get_legend_handles_labels(self, *a, **k):
            handles = list(self.lines) + list(self.patches)
            labels = [getattr(h, "get_label", lambda: "")() for h in handles]
            return (handles, labels)

        def get_xaxis_transform(self, *a, **k):
            return lambda *args, **kwargs: None

        def set_position(self, pos):
            if isinstance(pos, (list, tuple)) and len(pos) >= 4:
                self._position = types.SimpleNamespace(x0=pos[0], y0=pos[1], width=pos[2], height=pos[3])
            else:
                self._position = pos
            return None

        def get_position(self):
            return self._position

    class _GridSpec:
        def __init__(self, nrows=1, ncols=1):
            self._shape = (nrows, ncols)

        def __getitem__(self, item):
            return item

    class _DummyFig:
        def __init__(self):
            self._size = [1.0, 1.0]
            self.axes = []
            self.legends = []

        def savefig(self, path, dpi=None, bbox_inches=None, tight_layout=None, *args, **kwargs):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"")

        def tight_layout(self, *args, **kwargs):
            return None

        def set_size_inches(self, *args, forward=False):
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                size = list(args[0])
            elif len(args) >= 2:
                size = [args[0], args[1]]
            elif len(args) == 1:
                size = [args[0]]
            else:
                size = list(self._size)
            self._size = list(size)

        def get_size_inches(self):
            return list(self._size)

        def legend(self, handles=None, labels=None, *a, **k):
            legend_obj = types.SimpleNamespace(handles=handles or [], labels=labels or [])
            self.legends.append(legend_obj)
            return legend_obj

        def suptitle(self, *a, **k):
            return None

        def subplots_adjust(self, *a, **k):
            return None

        def add_axes(self, *a, **k):
            axis = _DummyAxes(self)
            self.axes.append(axis)
            return axis

        def add_subplot(self, *a, **k):
            axis = _DummyAxes(self)
            self.axes.append(axis)
            return axis

        def add_gridspec(self, nrows=1, ncols=1, **_kwargs):
            return _GridSpec(nrows, ncols)

    def _get_cmap(name=None):
        return cmap_registry.get(name, cmap_registry["_default"])

    plt = types.ModuleType("matplotlib.pyplot")
    plt.rcParams = mpl.rcParams
    plt.cycler = _cycler
    plt.rcdefaults = rcdefaults

    def _subplots(*args, **kwargs):
        nrows = kwargs.get("nrows", args[0] if args else 1)
        ncols = kwargs.get("ncols", args[1] if len(args) > 1 else 1)
        fig = _DummyFig()
        grid = [[_DummyAxes(fig) for _ in range(ncols)] for _ in range(nrows)]
        if nrows == 1 and ncols == 1:
            axes_out = grid[0][0]
        elif nrows == 1 or ncols == 1:
            axes_out = [ax for row in grid for ax in row]
        else:
            axes_out = grid
        fig.axes = [ax for row in grid for ax in row]
        return fig, axes_out

    plt.figure = lambda *_a, **_k: _DummyFig()
    plt.subplots = _subplots
    plt.close = lambda *_args, **_kwargs: None
    plt.get_cmap = _get_cmap
    plt.switch_backend = lambda *_a, **_k: None

    cm = types.SimpleNamespace(get_cmap=_get_cmap)
    mpl.cm = cm
    mpl.colors = mcolors
    mpl._text_helpers = types.SimpleNamespace()
    mpl._type1font = types.SimpleNamespace(Type1Font=object)
    mpl.cbook = types.SimpleNamespace(get_sample_data=lambda *_a, **_k: None)
    mpl.dviread = types.SimpleNamespace(Dvi=object)
    mpl.pyplot = plt

    lines_mod = types.SimpleNamespace(Line2D=_Line2D)
    axes_mod = types.SimpleNamespace(Axes=_DummyAxes)

    class _Patch:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def get_label(self):
            return self.kwargs.get("label", "")

    patches_mod = types.SimpleNamespace(Patch=_Patch)
    backend_pdf = types.ModuleType("matplotlib.backends.backend_pdf")

    class _PdfPages:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def savefig(self, *a, **k):
            return None

        def close(self):
            self.closed = True

    backend_pdf.PdfPages = _PdfPages
    sys.modules["matplotlib.backends"] = types.SimpleNamespace()
    sys.modules["matplotlib.backends.backend_pdf"] = backend_pdf
    cycler_mod = types.ModuleType("matplotlib.cycler")
    cycler_mod.cycler = _cycler
    sys.modules["matplotlib.cycler"] = cycler_mod
    sys.modules["matplotlib"] = mpl
    sys.modules["matplotlib.pyplot"] = plt
    sys.modules["matplotlib.colors"] = mcolors  # type: ignore[arg-type]
    sys.modules["matplotlib.cm"] = cm
    sys.modules["matplotlib._text_helpers"] = mpl._text_helpers
    sys.modules["matplotlib._type1font"] = mpl._type1font
    sys.modules["matplotlib.cbook"] = mpl.cbook
    sys.modules["matplotlib.dviread"] = mpl.dviread
    sys.modules["matplotlib.lines"] = lines_mod
    sys.modules["matplotlib.axes"] = axes_mod
    sys.modules["matplotlib.patches"] = patches_mod
    sys.modules["matplotlib.image"] = types.SimpleNamespace(imread=lambda *_a, **_k: None)
    return mpl


def _ensure_matplotlib():
    """
    Import matplotlib in a way that avoids incompatible vendored envs.

    If importing fails (for example, due to a stray conda env on sys.path),
    install a tiny stub that satisfies the handful of attributes used in tests.
    """
    sanitized = [p for p in sys.path if "openr1/lib/python3.1" not in str(p)]
    if sanitized != sys.path:
        sys.path[:] = sanitized
    try:
        mpl = importlib.import_module("matplotlib")
        plt_mod = importlib.import_module("matplotlib.pyplot")
        if not hasattr(plt_mod, "switch_backend"):
            plt_mod.switch_backend = lambda *_a, **_k: None
        needs_stub = not all(
            [
                hasattr(plt_mod, "figure"),
                hasattr(plt_mod, "subplots"),
                hasattr(plt_mod, "cycler"),
                hasattr(mpl, "colormaps"),
                hasattr(mpl, "cycler"),
            ]
        )
        if needs_stub:
            mpl = _install_matplotlib_stub()
            plt_mod = importlib.import_module("matplotlib.pyplot")
    except Exception:  # pragma: no cover - defensive fallback for broken envs
        mpl = _install_matplotlib_stub()
    return mpl


_ensure_matplotlib()

# Eager torch/accelerate/transformers stubs so imports during collection succeed
try:  # pragma: no cover - prefer real torch if available
    import torch as _real_torch  # type: ignore

    # Backfill minimal attributes if the installed torch is a lightweight stub.
    if not hasattr(_real_torch, "inference_mode"):

        @contextmanager
        def _no_grad_real():
            yield

        _real_torch.inference_mode = _no_grad_real  # type: ignore[attr-defined]
    if not hasattr(_real_torch, "no_grad"):

        @contextmanager
        def _no_grad_real():
            yield

        _real_torch.no_grad = _no_grad_real  # type: ignore[attr-defined]
    if not hasattr(_real_torch, "ones"):
        _real_torch.ones = lambda shape=None, dtype=None, device=None: _real_torch.tensor(1).expand(shape)  # type: ignore[attr-defined]
    if not hasattr(_real_torch, "zeros"):
        _real_torch.zeros = lambda shape=None, dtype=None, device=None: _real_torch.tensor(0).expand(shape)  # type: ignore[attr-defined]
    if not hasattr(_real_torch, "full"):
        _real_torch.full = (
            lambda shape, fill_value=0, dtype=None, device=None: _real_torch.ones(shape, dtype=dtype, device=device)
            * fill_value
        )  # type: ignore[attr-defined]
    if not hasattr(_real_torch, "SymBool"):
        _real_torch.SymBool = type("SymBool", (), {})
    if not hasattr(_real_torch, "SymFloat"):
        _real_torch.SymFloat = type("SymFloat", (), {})
    if not hasattr(_real_torch, "utils"):
        _real_torch.utils = types.SimpleNamespace(data=types.SimpleNamespace())
    if not hasattr(_real_torch, "tensor"):
        _real_torch.tensor = lambda data=None, dtype=None, device=None: _real_torch.Tensor(data)  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - exercised in minimal envs

    class _Tensor:
        def __init__(self, data=None, dtype=None, device=None):
            arr = np.array([]) if data is None else np.array(data)
            self.data = arr
            self.dtype = dtype or arr.dtype
            self.device = device or "cpu"

        @property
        def shape(self):
            return getattr(self.data, "shape", ())

        def size(self, dim=None):
            if dim is None:
                return self.data.size
            return self.data.shape[dim]

        def detach(self):
            return self

        def cpu(self):
            return self

        def float(self):
            return _Tensor(self.data.astype(float), dtype=float, device=self.device)

        def int(self):
            return _Tensor(self.data.astype(int), dtype=int, device=self.device)

        def __int__(self):
            return int(self.data)

        def tolist(self):
            return self.data.tolist()

        def __iter__(self):
            for item in self.data:
                yield _Tensor(item, dtype=self.dtype, device=self.device)

        def __array__(self, dtype=None):
            return np.array(self.data, dtype=dtype)

        def __bool__(self):
            return bool(self.data.any() if hasattr(self.data, "any") else self.data)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, item):
            return _Tensor(self.data[item], dtype=self.dtype, device=self.device)

        def __setitem__(self, key, value):
            self.data[key] = value.data if isinstance(value, _Tensor) else value

        def __add__(self, other):
            return _Tensor(
                self.data + (other.data if isinstance(other, _Tensor) else other), dtype=self.dtype, device=self.device
            )

        def __sub__(self, other):
            return _Tensor(
                self.data - (other.data if isinstance(other, _Tensor) else other), dtype=self.dtype, device=self.device
            )

        def __mul__(self, other):
            return _Tensor(
                self.data * (other.data if isinstance(other, _Tensor) else other), dtype=self.dtype, device=self.device
            )

        def __truediv__(self, other):
            return _Tensor(
                self.data / (other.data if isinstance(other, _Tensor) else other), dtype=self.dtype, device=self.device
            )

        def __le__(self, other):
            other_data = other.data if isinstance(other, _Tensor) else other
            return _Tensor(self.data <= other_data, dtype=bool, device=self.device)

        def __invert__(self):
            return _Tensor(~self.data, dtype=self.dtype, device=self.device)

        def __eq__(self, other):
            return self.eq(other)

        def nonzero(self, as_tuple=False):
            idx = np.argwhere(self.data)
            if as_tuple:
                return tuple(idx.T)
            return _Tensor(idx, dtype=int, device=self.device)

        def eq(self, other):
            other_data = other.data if isinstance(other, _Tensor) else other
            return _Tensor(self.data == other_data, dtype=bool, device=self.device)

        def all(self):
            return bool(self.data.all())

        def any(self, dim=None):
            return _Tensor(self.data.any(axis=dim), dtype=bool, device=self.device)

        def numel(self):
            return int(self.data.size)

        def exp(self):
            return _Tensor(np.exp(self.data), dtype=float, device=self.device)

        def argmax(self, dim=None):
            return _Tensor(self.data.argmax(axis=dim), dtype=int, device=self.device)

        def sum(self, dim=None):
            return _Tensor(self.data.sum(axis=dim), dtype=self.dtype, device=self.device)

        def nansum(self, dim=None):
            return _Tensor(np.nansum(self.data, axis=dim), dtype=self.dtype, device=self.device)

        def mean(self, dim=None):
            return _Tensor(self.data.mean(axis=dim), dtype=float, device=self.device)

        def std(self, dim=None, ddof=0):
            return _Tensor(self.data.std(axis=dim, ddof=ddof), dtype=float, device=self.device)

        def unsqueeze(self, dim):
            return _Tensor(np.expand_dims(self.data, axis=dim), dtype=self.dtype, device=self.device)

        def expand_as(self, other):
            target_shape = getattr(other, "shape", None)
            return _Tensor(np.broadcast_to(self.data, target_shape), dtype=self.dtype, device=self.device)

        def view(self, *shape):
            return _Tensor(self.data.reshape(*shape), dtype=self.dtype, device=self.device)

        def repeat_interleave(self, repeats, dim=0):
            return _Tensor(np.repeat(self.data, repeats, axis=dim), dtype=self.dtype, device=self.device)

        def clone(self):
            return _Tensor(self.data.copy(), dtype=self.dtype, device=self.device)

        def item(self):
            return self.data.item()

    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = _Tensor
    torch_stub.tensor = lambda data=None, dtype=None, device=None: _Tensor(data, dtype=dtype, device=device)
    torch_stub.zeros_like = lambda t, dtype=None, device=None: _Tensor(
        np.zeros_like(np.array(t)), dtype=dtype or getattr(t, "dtype", np.int64), device=device
    )
    torch_stub.ones_like = lambda t, dtype=None, device=None: _Tensor(
        np.ones_like(np.array(t)), dtype=dtype or getattr(t, "dtype", np.int64), device=device
    )
    torch_stub.full = lambda shape, fill_value=0, dtype=None, device=None: _Tensor(
        np.full(shape, fill_value, dtype=dtype or np.int64), dtype=dtype, device=device
    )
    torch_stub.zeros = lambda shape, dtype=None, device=None: _Tensor(
        np.zeros(shape, dtype=dtype or np.int64), dtype=dtype, device=device
    )
    torch_stub.ones = lambda shape, dtype=None, device=None: _Tensor(
        np.ones(shape, dtype=dtype or np.int64), dtype=dtype, device=device
    )
    torch_stub.isnan = lambda x: _Tensor(np.isnan(np.array(x)), dtype=bool)
    torch_stub.isinf = lambda x: _Tensor(np.isinf(np.array(x)), dtype=bool)
    torch_stub.arange = lambda n, device=None, dtype=None: _Tensor(
        np.arange(n, dtype=dtype), dtype=dtype, device=device
    )
    torch_stub.cat = lambda tensors, dim=0: _Tensor(np.concatenate([np.array(t) for t in tensors], axis=dim))
    torch_stub.isclose = lambda a, b, rtol=1e-05, atol=1e-08: _Tensor(
        np.isclose(np.array(a), np.array(b), rtol=rtol, atol=atol)
    )
    torch_stub.long = np.int64
    torch_stub.bool = np.bool_
    torch_stub.SymBool = _Tensor
    torch_stub.SymFloat = _Tensor
    torch_stub.device = lambda name=None: types.SimpleNamespace(type=str(name or "cpu"))
    nn_functional = types.SimpleNamespace(
        log_softmax=lambda logits, dim=-1: _Tensor(np.log(np.ones_like(np.array(logits))), dtype=float)
    )
    torch_stub.nn = types.SimpleNamespace(functional=nn_functional)
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    torch_stub.__spec__ = importlib.machinery.ModuleSpec("torch", None)

    @contextmanager
    def _no_grad():
        yield

    torch_stub.no_grad = _no_grad
    torch_stub.inference_mode = _no_grad
    sys.modules["torch"] = torch_stub

    # accelerate / distributed
    acc_pkg = types.ModuleType("accelerate")
    acc_pkg.__path__ = []
    acc_utils = types.ModuleType("accelerate.utils")
    acc_utils.broadcast_object_list = lambda objs, *_, **__: objs
    acc_utils.gather_object = lambda obj, *_, **__: obj
    sys.modules["accelerate"] = acc_pkg
    sys.modules["accelerate.utils"] = acc_utils
    dist_pkg = types.ModuleType("torch.distributed")
    fsdp_pkg = types.ModuleType("torch.distributed.fsdp")
    fsdp_pkg.FullyShardedDataParallel = type("FSDP", (), {})
    dist_pkg.fsdp = fsdp_pkg
    sys.modules["torch.distributed"] = dist_pkg
    sys.modules["torch.distributed.fsdp"] = fsdp_pkg

    # torch.nn utils.rnn.pad_sequence
    nn_pkg = types.ModuleType("torch.nn")
    utils_pkg = types.ModuleType("torch.nn.utils")
    rnn_pkg = types.ModuleType("torch.nn.utils.rnn")

    def _pad_sequence(sequences, batch_first=True, padding_value=0):
        lengths = []
        for seq in sequences:
            shape = getattr(seq, "shape", None)
            lengths.append(shape[0] if shape is not None and len(shape) > 0 else len(seq))
        max_len = max(lengths) if lengths else 0
        padded = []
        for seq in sequences:
            arr = np.array(seq)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            pad_width = max_len - (arr.shape[0] if arr.ndim > 0 else 0)
            if pad_width > 0:
                pad_arr = np.full((pad_width, *arr.shape[1:]), padding_value)
                arr = np.concatenate([arr, pad_arr], axis=0)
            padded.append(arr)
        return _Tensor(np.stack(padded, axis=0))

    rnn_pkg.pad_sequence = _pad_sequence
    sys.modules["torch.nn"] = nn_pkg
    sys.modules["torch.nn.utils"] = utils_pkg
    sys.modules["torch.nn.utils.rnn"] = rnn_pkg

    # transformers stubs
    transformers_pkg = types.ModuleType("transformers")
    transformers_pkg.__path__ = []
    transformers_pkg.PreTrainedTokenizerBase = object
    transformers_pkg.Trainer = object
    transformers_pkg.TrainerCallback = type("TrainerCallback", (), {})
    transformers_pkg.GenerationMixin = object
    transformers_pkg.StoppingCriteria = type("StoppingCriteria", (), {})
    transformers_pkg.StoppingCriteriaList = list
    transformers_pkg.AutoConfig = type("AutoConfig", (), {})
    transformers_pkg.AutoTokenizer = type("AutoTokenizer", (), {})
    transformers_pkg.AutoModelForCausalLM = type("AutoModelForCausalLM", (), {})
    transformers_pkg.utils = types.SimpleNamespace(is_flash_attn_2_available=lambda: False)
    tok_base = types.ModuleType("transformers.tokenization_utils_base")
    tok_base.PaddingStrategy = object
    gen_utils = types.ModuleType("transformers.generation.utils")
    gen_utils.GenerationMixin = object
    gen_utils.StoppingCriteriaList = list
    transformers_utils = types.ModuleType("transformers.utils")
    transformers_utils.is_flash_attn_2_available = lambda: False
    sys.modules["transformers"] = transformers_pkg
    sys.modules["transformers.tokenization_utils_base"] = tok_base
    sys.modules["transformers.generation.utils"] = gen_utils
    sys.modules["transformers.utils"] = transformers_utils

    # deepspeed stubs (enough for math_llama_core imports)
    deepspeed_pkg = types.ModuleType("deepspeed")
    sys.modules["deepspeed"] = deepspeed_pkg
    zero_config_pkg = types.ModuleType("deepspeed.runtime.zero.config")
    zero_config_pkg.ZeroStageEnum = type("ZeroStageEnum", (), {})
    sys.modules["deepspeed.runtime.zero.config"] = zero_config_pkg
    loss_scaler_pkg = types.ModuleType("deepspeed.runtime.fp16.loss_scaler")
    loss_scaler_pkg.LossScaler = type("LossScaler", (), {})
    sys.modules["deepspeed.runtime.fp16.loss_scaler"] = loss_scaler_pkg

# Ensure a minimal transformers module exists even when torch is available.
if "transformers" not in sys.modules:
    transformers_pkg = types.ModuleType("transformers")
    transformers_pkg.__path__ = []
    transformers_pkg.PreTrainedTokenizerBase = object
    transformers_pkg.Trainer = object
    transformers_pkg.TrainerCallback = type("TrainerCallback", (), {})
    transformers_pkg.GenerationMixin = object
    transformers_pkg.StoppingCriteria = type("StoppingCriteria", (), {})
    transformers_pkg.StoppingCriteriaList = type("StoppingCriteriaList", (list,), {})
    transformers_pkg.AutoConfig = type("AutoConfig", (), {})
    transformers_pkg.AutoTokenizer = type("AutoTokenizer", (), {})
    transformers_pkg.AutoModelForCausalLM = type("AutoModelForCausalLM", (), {})
    transformers_pkg.utils = types.SimpleNamespace(is_flash_attn_2_available=lambda: False)
    tok_base = types.ModuleType("transformers.tokenization_utils_base")
    tok_base.PaddingStrategy = object
    gen_utils = types.ModuleType("transformers.generation.utils")
    gen_utils.GenerationMixin = object
    transformers_utils = types.ModuleType("transformers.utils")
    transformers_utils.is_flash_attn_2_available = lambda: False
    sys.modules["transformers"] = transformers_pkg
    sys.modules["transformers.tokenization_utils_base"] = tok_base
    sys.modules["transformers.generation.utils"] = gen_utils
    sys.modules["transformers.utils"] = transformers_utils
else:
    _tf = sys.modules["transformers"]
    _tf.PreTrainedTokenizerBase = getattr(_tf, "PreTrainedTokenizerBase", object)
    _tf.Trainer = getattr(_tf, "Trainer", object)
    _tf.TrainerCallback = getattr(_tf, "TrainerCallback", type("TrainerCallback", (), {}))
    _tf.GenerationMixin = getattr(_tf, "GenerationMixin", object)
    if not hasattr(_tf, "StoppingCriteria"):
        _tf.StoppingCriteria = type("StoppingCriteria", (), {})
    if not hasattr(_tf, "StoppingCriteriaList"):
        _tf.StoppingCriteriaList = type("StoppingCriteriaList", (list,), {})
    _tf.AutoConfig = getattr(_tf, "AutoConfig", type("AutoConfig", (), {}))
    _tf.AutoTokenizer = getattr(_tf, "AutoTokenizer", type("AutoTokenizer", (), {}))
    _tf.AutoModelForCausalLM = getattr(_tf, "AutoModelForCausalLM", type("AutoModelForCausalLM", (), {}))


class _FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)
        self.features = None
        self.saved_to = None

    @classmethod
    def from_list(cls, rows):
        return cls(rows)

    @classmethod
    def from_dict(cls, data, features=None):
        # Reconstruct row dictionaries from column-wise data.
        keys = list(data.keys())
        row_count = len(data[keys[0]]) if keys else 0
        rows = [{key: data[key][idx] for key in keys} for idx in range(row_count)]
        ds = cls(rows)
        ds.features = features
        return ds

    def select(self, indices):
        return _FakeDataset([self.rows[i] for i in indices])

    def save_to_disk(self, path):
        self.saved_to = Path(path)
        return self

    def push_to_hub(self, repo_id, private=True):
        # Record the push target so tests can assert on it if needed.
        self.pushed_to = (repo_id, private)
        return self

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class _FakeDatasetDict(dict):
    def save_to_disk(self, path):
        self.saved_to = Path(path)
        return self

    def push_to_hub(self, repo_id, private=True):
        self.pushed_to = (repo_id, private)
        return self


class _FakeValue:
    def __init__(self, dtype):
        self.dtype = dtype


class _FakeSequence:
    def __init__(self, feature):
        self.feature = feature


class _FakeFeatures(dict):
    pass


def _build_fake_datasets_module():
    module = types.SimpleNamespace()
    module.Dataset = _FakeDataset
    module.DatasetDict = _FakeDatasetDict
    module.Features = _FakeFeatures
    module.Value = _FakeValue
    module.Sequence = _FakeSequence
    module.load_dataset = lambda repo: {"loaded": repo}

    def _load_from_disk(path):
        if isinstance(path, Path) and not path.exists():
            raise FileNotFoundError(path)
        if isinstance(path, str) and path.startswith("missing"):
            raise FileNotFoundError(path)
        return {"disk": path}

    module.load_from_disk = _load_from_disk
    return module


def _build_fake_hf_module():
    class _FakeHfFolder:
        @staticmethod
        def get_token():
            return None

    class _FakeHfApi:
        def upload_file(self, **kwargs):  # pragma: no cover - behavior recorded via kwargs
            self.last_upload = kwargs

    return types.SimpleNamespace(HfApi=_FakeHfApi, HfFolder=_FakeHfFolder)


@pytest.fixture(autouse=True)
def stub_external_modules(monkeypatch):
    """
    Provide lightweight stand-ins for optional heavy dependencies so the data
    helpers can be imported without installing datasets/huggingface_hub.
    """
    fake_datasets = _build_fake_datasets_module()
    fake_hf = _build_fake_hf_module()
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    yield


@pytest.fixture(autouse=True)
def ensure_torch_stub(monkeypatch):
    """
    Ensure a minimal torch stub exists (or is patched) so tests can run in
    environments without the real torch package. When real torch is available,
    this fixture is a no-op aside from filling in missing attrs like SymBool.
    """
    try:
        import torch as torch_mod  # type: ignore
    except ImportError:  # pragma: no cover - exercised when torch absent
        torch_mod = sys.modules.get("torch")
        created = torch_mod is None

        class _Tensor:
            def __init__(self, data=None, dtype=None, device=None):
                arr = np.array([]) if data is None else np.array(data)
                self.data = arr
                self.dtype = dtype or arr.dtype
                self.device = device or "cpu"

            @property
            def shape(self):
                return getattr(self.data, "shape", ())

            def size(self, dim=None):
                if dim is None:
                    return self.data.size
                return self.data.shape[dim]

            def detach(self):
                return self

            def cpu(self):
                return self

            def float(self):
                return _Tensor(self.data.astype(float), dtype=float, device=self.device)

            def int(self):
                return _Tensor(self.data.astype(int), dtype=int, device=self.device)

            def __int__(self):
                return int(self.data)

            def tolist(self):
                return self.data.tolist()

            def __iter__(self):
                for item in self.data:
                    yield _Tensor(item, dtype=self.dtype, device=self.device)

            def __array__(self, dtype=None):
                return np.array(self.data, dtype=dtype)

            def __bool__(self):
                return bool(self.data.any() if hasattr(self.data, "any") else self.data)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, item):
                return _Tensor(self.data[item], dtype=self.dtype, device=self.device)

            def __setitem__(self, key, value):
                self.data[key] = value.data if isinstance(value, _Tensor) else value

            def __add__(self, other):
                return _Tensor(
                    self.data + (other.data if isinstance(other, _Tensor) else other),
                    dtype=self.dtype,
                    device=self.device,
                )

            def __sub__(self, other):
                return _Tensor(
                    self.data - (other.data if isinstance(other, _Tensor) else other),
                    dtype=self.dtype,
                    device=self.device,
                )

            def __mul__(self, other):
                return _Tensor(
                    self.data * (other.data if isinstance(other, _Tensor) else other),
                    dtype=self.dtype,
                    device=self.device,
                )

            def __truediv__(self, other):
                return _Tensor(
                    self.data / (other.data if isinstance(other, _Tensor) else other),
                    dtype=self.dtype,
                    device=self.device,
                )

            def __le__(self, other):
                other_data = other.data if isinstance(other, _Tensor) else other
                return _Tensor(self.data <= other_data, dtype=bool, device=self.device)

            def __invert__(self):
                return _Tensor(~self.data, dtype=self.dtype, device=self.device)

            def __eq__(self, other):
                return self.eq(other)

            def eq(self, other):
                other_data = other.data if isinstance(other, _Tensor) else other
                return _Tensor(self.data == other_data, dtype=bool, device=self.device)

            def all(self):
                return bool(self.data.all())

            def any(self, dim=None):
                return _Tensor(self.data.any(axis=dim), dtype=bool, device=self.device)

            def argmax(self, dim=None):
                return _Tensor(self.data.argmax(axis=dim), dtype=int, device=self.device)

            def sum(self, dim=None):
                return _Tensor(self.data.sum(axis=dim), dtype=self.dtype, device=self.device)

            def nansum(self, dim=None):
                return _Tensor(np.nansum(self.data, axis=dim), dtype=self.dtype, device=self.device)

            def mean(self, dim=None):
                return _Tensor(self.data.mean(axis=dim), dtype=float, device=self.device)

            def std(self, dim=None, ddof=0):
                return _Tensor(self.data.std(axis=dim, ddof=ddof), dtype=float, device=self.device)

            def unsqueeze(self, dim):
                return _Tensor(np.expand_dims(self.data, axis=dim), dtype=self.dtype, device=self.device)

            def expand_as(self, other):
                target_shape = getattr(other, "shape", None)
                return _Tensor(np.broadcast_to(self.data, target_shape), dtype=self.dtype, device=self.device)

            def view(self, *shape):
                return _Tensor(self.data.reshape(*shape), dtype=self.dtype, device=self.device)

            def repeat_interleave(self, repeats, dim=0):
                return _Tensor(np.repeat(self.data, repeats, axis=dim), dtype=self.dtype, device=self.device)

            def clone(self):
                return _Tensor(self.data.copy(), dtype=self.dtype, device=self.device)

            def item(self):
                return self.data.item()

        torch_mod = torch_mod or types.ModuleType("torch")
        torch_mod.__spec__ = getattr(torch_mod, "__spec__", importlib.machinery.ModuleSpec("torch", None))
        monkeypatch.setattr(torch_mod, "Tensor", getattr(torch_mod, "Tensor", _Tensor), raising=False)
        monkeypatch.setattr(
            torch_mod,
            "tensor",
            getattr(
                torch_mod,
                "tensor",
                lambda data=None, dtype=None, device=None: _Tensor(data, dtype=dtype, device=device),
            ),
            raising=False,
        )
        monkeypatch.setattr(
            torch_mod,
            "zeros_like",
            getattr(
                torch_mod,
                "zeros_like",
                lambda t, dtype=None, device=None: _Tensor(
                    np.zeros_like(np.array(t)), dtype=dtype or getattr(t, "dtype", np.int64), device=device
                ),
            ),
            raising=False,
        )
        monkeypatch.setattr(
            torch_mod,
            "ones_like",
            getattr(
                torch_mod,
                "ones_like",
                lambda t, dtype=None, device=None: _Tensor(
                    np.ones_like(np.array(t)), dtype=dtype or getattr(t, "dtype", np.int64), device=device
                ),
            ),
            raising=False,
        )
        monkeypatch.setattr(
            torch_mod,
            "full",
            getattr(
                torch_mod,
                "full",
                lambda shape, fill_value=0, dtype=None, device=None: _Tensor(
                    np.full(shape, fill_value, dtype=dtype or np.int64), dtype=dtype, device=device
                ),
            ),
            raising=False,
        )
        monkeypatch.setattr(
            torch_mod,
            "zeros",
            getattr(
                torch_mod,
                "zeros",
                lambda shape, dtype=None, device=None: _Tensor(
                    np.zeros(shape, dtype=dtype or np.int64), dtype=dtype, device=device
                ),
            ),
            raising=False,
        )
        monkeypatch.setattr(
            torch_mod,
            "ones",
            getattr(
                torch_mod,
                "ones",
                lambda shape, dtype=None, device=None: _Tensor(
                    np.ones(shape, dtype=dtype or np.int64), dtype=dtype, device=device
                ),
            ),
            raising=False,
        )
        monkeypatch.setattr(
            torch_mod,
            "arange",
            getattr(
                torch_mod,
                "arange",
                lambda n, device=None, dtype=None: _Tensor(np.arange(n, dtype=dtype), dtype=dtype, device=device),
            ),
            raising=False,
        )
        monkeypatch.setattr(
            torch_mod,
            "cat",
            getattr(
                torch_mod,
                "cat",
                lambda tensors, dim=0: _Tensor(np.concatenate([np.array(t) for t in tensors], axis=dim)),
            ),
            raising=False,
        )
        monkeypatch.setattr(
            torch_mod,
            "isclose",
            getattr(
                torch_mod,
                "isclose",
                lambda a, b, rtol=1e-05, atol=1e-08: _Tensor(
                    np.isclose(np.array(a), np.array(b), rtol=rtol, atol=atol)
                ),
            ),
            raising=False,
        )
        monkeypatch.setattr(torch_mod, "long", getattr(torch_mod, "long", np.int64), raising=False)
        monkeypatch.setattr(torch_mod, "bool", getattr(torch_mod, "bool", np.bool_), raising=False)
        monkeypatch.setattr(torch_mod, "SymBool", getattr(torch_mod, "SymBool", _Tensor), raising=False)
        monkeypatch.setattr(torch_mod, "SymFloat", getattr(torch_mod, "SymFloat", _Tensor), raising=False)

        @contextmanager
        def _no_grad():
            yield

        monkeypatch.setattr(torch_mod, "no_grad", getattr(torch_mod, "no_grad", _no_grad), raising=False)
        monkeypatch.setattr(torch_mod, "inference_mode", getattr(torch_mod, "inference_mode", _no_grad), raising=False)
        monkeypatch.setattr(
            torch_mod,
            "device",
            getattr(torch_mod, "device", lambda name=None: types.SimpleNamespace(type=str(name or "cpu"))),
            raising=False,
        )
        monkeypatch.setattr(
            torch_mod,
            "cuda",
            getattr(torch_mod, "cuda", types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)),
            raising=False,
        )

        if created:
            sys.modules["torch"] = torch_mod
            # basic nn.utils.rnn container to satisfy pad_sequence imports when torch absent
            nn_pkg = types.ModuleType("torch.nn")
            utils_pkg = types.ModuleType("torch.nn.utils")
            rnn_pkg = types.ModuleType("torch.nn.utils.rnn")

            def pad_sequence(sequences, batch_first=True, padding_value=0):
                lengths = []
                for seq in sequences:
                    shape = getattr(seq, "shape", None)
                    if shape is not None and len(shape) > 0:
                        lengths.append(shape[0])
                    else:
                        lengths.append(len(seq))
                max_len = max(lengths) if lengths else 0
                padded = []
                for seq in sequences:
                    arr = np.array(seq)
                    pad_width = max_len - (arr.shape[0] if arr.ndim > 0 else 0)
                    if pad_width > 0:
                        pad_arr = np.full((pad_width, *arr.shape[1:]), padding_value)
                        arr = np.concatenate([arr, pad_arr], axis=0)
                    padded.append(arr)
                return _Tensor(np.stack(padded, axis=0))

            rnn_pkg.pad_sequence = pad_sequence
            sys.modules["torch.nn"] = nn_pkg
            sys.modules["torch.nn.utils"] = utils_pkg
            sys.modules["torch.nn.utils.rnn"] = rnn_pkg
            # accelerate + distributed stubs
            acc_pkg = types.ModuleType("accelerate")
            acc_pkg.__path__ = []  # mark as package
            acc_utils = types.ModuleType("accelerate.utils")
            acc_utils.broadcast_object_list = lambda objs, *_, **__: objs  # passthrough
            acc_utils.gather_object = lambda obj, *_, **__: obj
            sys.modules["accelerate"] = acc_pkg
            sys.modules["accelerate.utils"] = acc_utils

            dist_pkg = types.ModuleType("torch.distributed")
            fsdp_pkg = types.ModuleType("torch.distributed.fsdp")
            fsdp_pkg.FullyShardedDataParallel = type("FSDP", (), {})
            sys.modules["torch.distributed"] = dist_pkg
            sys.modules["torch.distributed.fsdp"] = fsdp_pkg
            dist_pkg.fsdp = fsdp_pkg

            # transformers stubs sufficient for hierarchical_rollout import
            transformers_pkg = types.ModuleType("transformers")
            transformers_pkg.__path__ = []  # mark as package
            transformers_pkg.PreTrainedTokenizerBase = object
            transformers_pkg.Trainer = object
            transformers_pkg.TrainerCallback = type("TrainerCallback", (), {})
            transformers_pkg.GenerationMixin = object
            transformers_pkg.utils = types.SimpleNamespace(is_flash_attn_2_available=lambda: False)

            tok_base = types.ModuleType("transformers.tokenization_utils_base")
            tok_base.PaddingStrategy = object
            gen_utils = types.ModuleType("transformers.generation.utils")
            gen_utils.GenerationMixin = object
            transformers_utils = types.ModuleType("transformers.utils")
            transformers_utils.is_flash_attn_2_available = lambda: False
            sys.modules["transformers"] = transformers_pkg
            sys.modules["transformers.tokenization_utils_base"] = tok_base
            sys.modules["transformers.generation.utils"] = gen_utils
            sys.modules["transformers.utils"] = transformers_utils
    else:
        # Real torch is available; ensure optional symbols used in tests exist.
        if not hasattr(torch_mod, "SymBool"):
            monkeypatch.setattr(torch_mod, "SymBool", getattr(torch_mod, "Tensor", object), raising=False)
        if not hasattr(torch_mod, "SymFloat"):
            monkeypatch.setattr(torch_mod, "SymFloat", getattr(torch_mod, "Tensor", object), raising=False)

    # Backfill missing context managers/helpers even when torch was already present.
    @contextmanager
    def _fallback_no_grad():
        yield

    no_grad_obj = getattr(torch_mod, "no_grad", None)
    if no_grad_obj is None or not hasattr(no_grad_obj, "__enter__"):
        monkeypatch.setattr(torch_mod, "no_grad", _fallback_no_grad, raising=False)
    if not hasattr(torch_mod, "inference_mode") or not hasattr(getattr(torch_mod, "inference_mode"), "__enter__"):
        monkeypatch.setattr(torch_mod, "inference_mode", getattr(torch_mod, "no_grad"), raising=False)
    if not hasattr(torch_mod, "tensor"):
        monkeypatch.setattr(
            torch_mod,
            "tensor",
            lambda data=None, dtype=None, device=None: getattr(torch_mod, "Tensor", lambda x=None: x)(data),
            raising=False,
        )
    if not hasattr(torch_mod, "full"):

        def _full(shape, fill_value=0, dtype=None, device=None):
            arr = np.full(shape, fill_value, dtype=dtype or np.int64)
            tensor_cls = getattr(torch_mod, "Tensor", None)
            try:
                return tensor_cls(arr) if callable(tensor_cls) else arr
            except Exception:
                return arr

        monkeypatch.setattr(
            torch_mod,
            "full",
            _full,
            raising=False,
        )
    if not hasattr(torch_mod, "cat"):
        monkeypatch.setattr(
            torch_mod,
            "cat",
            lambda tensors, dim=0: getattr(
                torch_mod,
                "Tensor",
                lambda data: data,
            )(
                np.concatenate(
                    [np.array(getattr(t, "data", t)) for t in tensors],
                    axis=dim,
                ),
            ),
            raising=False,
        )

    dist_mod = sys.modules.get("torch.distributed")
    if dist_mod is None:
        dist_mod = types.SimpleNamespace()
        sys.modules["torch.distributed"] = dist_mod
    if dist_mod is not None:
        monkeypatch.setattr(
            dist_mod, "is_initialized", getattr(dist_mod, "is_initialized", lambda: False), raising=False
        )
        monkeypatch.setattr(dist_mod, "get_rank", getattr(dist_mod, "get_rank", lambda: 0), raising=False)
        monkeypatch.setattr(dist_mod, "get_world_size", getattr(dist_mod, "get_world_size", lambda: 1), raising=False)
        monkeypatch.setattr(
            dist_mod,
            "all_gather_object",
            getattr(
                dist_mod, "all_gather_object", lambda gathered, winners_local: gathered.__setitem__(0, winners_local)
            ),
            raising=False,
        )

    # Normalize pad_sequence for stub torch installs so sequences containing
    # _Tensor wrappers don't produce object-dtype arrays.
    def _safe_pad_sequence(sequences, batch_first=True, padding_value=0):
        lengths = []
        normalized = []
        for seq in sequences:
            data = getattr(seq, "data", seq)
            arr = np.array(data)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            lengths.append(arr.shape[0])
            normalized.append(arr)
        max_len = max(lengths) if lengths else 0
        padded_rows = []
        for arr in normalized:
            pad_width = max_len - arr.shape[0]
            if pad_width > 0:
                pad_arr = np.full((pad_width, *arr.shape[1:]), padding_value)
                arr = np.concatenate([arr, pad_arr], axis=0)
            padded_rows.append(arr)
        try:
            stacked = np.stack(padded_rows, axis=0)
        except Exception:
            stacked = padded_rows
        tensor_cls = getattr(torch_mod, "Tensor", None)
        try:
            return tensor_cls(stacked) if callable(tensor_cls) else stacked
        except Exception:
            return stacked

    if getattr(torch_mod, "__file__", None) is None:
        rnn_mod = sys.modules.get("torch.nn.utils.rnn")
        if rnn_mod is None:
            nn_pkg = sys.modules.get("torch.nn") or types.ModuleType("torch.nn")
            utils_pkg = sys.modules.get("torch.nn.utils") or types.ModuleType("torch.nn.utils")
            rnn_mod = types.ModuleType("torch.nn.utils.rnn")
            sys.modules["torch.nn"] = nn_pkg
            sys.modules["torch.nn.utils"] = utils_pkg
            sys.modules["torch.nn.utils.rnn"] = rnn_mod
        monkeypatch.setattr(rnn_mod, "pad_sequence", _safe_pad_sequence, raising=False)
    yield


@pytest.fixture(autouse=True)
def ensure_transformers_stub(monkeypatch):
    """
    Provide a minimal transformers stub when the real package is absent, or
    backfill key attributes when partial stubs are present.
    """
    try:
        import transformers as transformers_mod  # type: ignore
    except ImportError:
        transformers_mod = sys.modules.get("transformers")
        if transformers_mod is None:
            transformers_mod = types.ModuleType("transformers")
            transformers_mod.__path__ = []
            monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
        transformers_mod.PreTrainedTokenizerBase = getattr(transformers_mod, "PreTrainedTokenizerBase", object)
        transformers_mod.Trainer = getattr(transformers_mod, "Trainer", object)
        transformers_mod.TrainerCallback = getattr(
            transformers_mod,
            "TrainerCallback",
            type("TrainerCallback", (), {}),
        )
        transformers_mod.GenerationMixin = getattr(transformers_mod, "GenerationMixin", object)
        transformers_mod.StoppingCriteria = getattr(
            transformers_mod, "StoppingCriteria", type("StoppingCriteria", (), {})
        )
        transformers_mod.StoppingCriteriaList = getattr(
            transformers_mod,
            "StoppingCriteriaList",
            type("StoppingCriteriaList", (list,), {}),
        )
        transformers_mod.AutoConfig = getattr(transformers_mod, "AutoConfig", type("AutoConfig", (), {}))
        transformers_mod.AutoTokenizer = getattr(transformers_mod, "AutoTokenizer", type("AutoTokenizer", (), {}))
        transformers_mod.AutoModelForCausalLM = getattr(
            transformers_mod, "AutoModelForCausalLM", type("AutoModelForCausalLM", (), {})
        )
        transformers_mod.utils = getattr(
            transformers_mod,
            "utils",
            types.SimpleNamespace(is_flash_attn_2_available=lambda: False),
        )
        tok_base = types.ModuleType("transformers.tokenization_utils_base")
        tok_base.PaddingStrategy = object
        gen_utils = types.ModuleType("transformers.generation.utils")
        gen_utils.GenerationMixin = getattr(transformers_mod, "GenerationMixin", object)
        transformers_utils = getattr(transformers_mod, "utils", types.SimpleNamespace())
        transformers_utils.is_flash_attn_2_available = getattr(
            transformers_utils,
            "is_flash_attn_2_available",
            lambda: False,
        )
        monkeypatch.setitem(sys.modules, "transformers.tokenization_utils_base", tok_base)
        monkeypatch.setitem(sys.modules, "transformers.generation.utils", gen_utils)
        monkeypatch.setitem(sys.modules, "transformers.utils", transformers_utils)
    else:
        for attr, default in [
            ("StoppingCriteria", type("StoppingCriteria", (), {})),
            ("StoppingCriteriaList", type("StoppingCriteriaList", (list,), {})),
            ("AutoConfig", type("AutoConfig", (), {})),
            ("AutoTokenizer", type("AutoTokenizer", (), {})),
            ("AutoModelForCausalLM", type("AutoModelForCausalLM", (), {})),
        ]:
            if not hasattr(transformers_mod, attr):
                try:
                    monkeypatch.setattr(transformers_mod, attr, default, raising=False)
                except AttributeError:
                    setattr(transformers_mod, attr, default)


@pytest.fixture(autouse=True)
def ensure_deepspeed_stub(monkeypatch):
    """Stub deepspeed modules when not installed."""
    try:
        import deepspeed  # type: ignore  # noqa: F401
    except ImportError:
        ds_pkg = sys.modules.get("deepspeed") or types.ModuleType("deepspeed")
        monkeypatch.setitem(sys.modules, "deepspeed", ds_pkg)
        zero_config_pkg = types.ModuleType("deepspeed.runtime.zero.config")
        zero_config_pkg.ZeroStageEnum = type("ZeroStageEnum", (), {})
        monkeypatch.setitem(sys.modules, "deepspeed.runtime.zero.config", zero_config_pkg)
        loss_scaler_pkg = types.ModuleType("deepspeed.runtime.fp16.loss_scaler")
        loss_scaler_pkg.LossScaler = type("LossScaler", (), {})
        monkeypatch.setitem(sys.modules, "deepspeed.runtime.fp16.loss_scaler", loss_scaler_pkg)


@pytest.fixture(autouse=True)
def _clear_grpo_replay_stub(request):
    """
    Some training tests install placeholder GRPO replay modules; clear them
    before GRPO-specific unit tests so the real implementation can load.
    """
    path = str(getattr(request, "fspath", ""))
    if "grpo_trainer_replay_impl" in path:
        sys.modules.pop("src.training.grpo_trainer_replay_impl", None)
        sys.modules.pop("src.training.grpo_rewards_router", None)
        sys.modules.pop("src.training.grpo_trainer_replay_support", None)
        sys.modules.pop("src.training.grpo_dataset", None)
